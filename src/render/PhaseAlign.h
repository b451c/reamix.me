#pragma once

#include <cstddef>

namespace reamix::render {

// PhaseAlign — time-domain brute-force normalized cross-correlation.
// Port of `references/python-source/remix/crossfade.py::_phase_align`
// (L86-153). Per ADR-031 SB-4 this is a NEW port, NOT an alias of
// `src/dsp/WaveformXcorr` (which is FFT-based and has different callers).
//
// Consumer: `src/render/Crossfade.cpp::adaptive_crossfade` only
// (crossfade.py:299). `splice.py` uses `max_shift_samples` as a penalty
// normalizer in `_score_splice_pair`, does NOT call `_phase_align`.
//
// Algorithm (verbatim from Python):
//   n = min(nOut, nIn)
//   if n < 32 or maxShift <= 0:  return (incoming, 0)   [passthrough]
//   minOverlap = max(16, n / 2)
//   bestCorr   = -1.0
//   bestShift  = 0
//   for shift in [-maxShift, maxShift]:                 [ascending]
//     (o, i) = sliced views of length n - |shift|
//       shift >= 0:  o = out[shift:],  i = in[:n-shift]
//       shift <  0:  o = out[:n+shift], i = in[-shift:]
//     if len(o) < minOverlap: continue                  [overlap gate]
//     oN = ||o||_2; iN = ||i||_2
//     if oN < 1e-8 or iN < 1e-8: continue               [zero-signal guard]
//     corr = dot(o, i) / (oN * iN)
//     if corr > bestCorr: bestCorr = corr; bestShift = shift
//   if bestShift == 0: return (incoming, 0)             [fast passthrough]
//   // Shift-apply (zeroed non-shifted region):
//   result = zeros_like(incoming)
//   if bestShift > 0: result[:-bestShift] = incoming[bestShift:]
//   else:             result[-bestShift:] = incoming[:bestShift]
//   return (result, bestShift)
//
// Tie-breaking: strictly `corr > bestCorr` + initial sentinel -1.0 +
// ascending iteration order ⇒ FIRST-occurrence wins on equal correlation
// (smaller shift magnitude, leaning negative). This matches `np.argmax`
// semantics on `np.array([corr(-m), ..., corr(+m)])` (first maximum).
//
// Dtype discipline:
//   - Correlation math in f64 (Python casts via `.astype(np.float64)`).
//   - Output dtype follows `incoming`. Production caller (adaptive_crossfade
//     crossfade.py:275-276, 300-301) casts both inputs to f64 before passing,
//     so in production the full path is f64. This port exposes f64-only I/O.
//
// Stereo:
//   - Python uses first-channel for NCC lookup, applies shift to all channels.
//   - This C++ API is MONO. Caller handles stereo (first-channel alignMono
//     lookup + broadcast shift to all channels via applyShift).
//
// max_shift in production: `adaptive_crossfade` passes
// `int(max_phase_shift_ms * sr / 1000.0)` with default 3 ms → 132 @ sr=44100,
// 66 @ sr=22050 (crossfade.py:226, 249). The `_phase_align` API-default 64
// is never reached from the production caller; kept as a free parameter here.
class PhaseAlign
{
public:
    // Compute best shift + aligned-incoming output via time-domain NCC.
    //
    //   out          pointer to outgoing signal (f64, nOut samples)
    //   nOut         length of outgoing
    //   in_          pointer to incoming signal (f64, nIn samples)
    //   nIn          length of incoming
    //   maxShift     maximum |shift| searched; must be >= 0
    //   shiftOut     [out] best_shift ∈ [-maxShift, maxShift], or 0 if
    //                any of the early-return gates fire.
    //   alignedOut   [out] incoming-shifted buffer of EXACTLY nIn samples
    //                (caller allocates). Non-shifted region zeroed.
    //                If shiftOut == 0, alignedOut is a bit-exact copy of
    //                incoming (mirroring Python's `return (incoming, 0)`).
    //
    // No allocations. Allowed aliasing: `alignedOut` may NOT alias `in_`
    // for non-zero shifts (shift-apply reads and writes positions that
    // overlap). For the zero-shift early-return branch, aliasing is safe
    // (the copy is trivial and in-direction).
    static void alignMono(const double* out,
                          std::size_t   nOut,
                          const double* in_,
                          std::size_t   nIn,
                          int           maxShift,
                          int*          shiftOut,
                          double*       alignedOut);
};

} // namespace reamix::render
