#pragma once

#include <cstddef>

namespace reamix::render {

// Crossfade — multi-band adaptive crossfade for remix splices.
//
// Port of `references/python-source/remix/crossfade.py::adaptive_crossfade`
// (L219-332) with one documented deviation per ADR-032 Option B.myfix:
// phase-align integer shift is applied to the FULL `in_band` buffer (length
// max_xf_samples) instead of to the `[:xf_samples]` slice with a subsequent
// pad-back at `[0:xf_samples]`. The Python pad-back has a placement bug
// (envelope non-zero region `[pad_before : pad_before + xf_samples]`
// disjoint from data region `[0 : xf_samples]` for short-crossfade bands
// mid + treble), muting the post-fade incoming content by 21-34 dB on real
// music. The deviation recovers the intended behavior while keeping the
// phase-align shift semantic (3 ms at sr=44100) at the fade boundary.
// See ADR-032 for the root-cause analysis + empirical data.
//
// Composed primitives:
//   - reamix::dsp::Butterworth::sosFiltFilt         (SOS + zero-phase wrapper, phase-5 s30)
//   - reamix::render::PhaseAlign::alignMono         (time-domain NCC, phase-5 s31)
//   - internal _equal_power_fade                     (cos/sin over [0, pi/2], session 32)
//
// Audio buffer layout convention: channel-major row-major, (nChannels, nSamples).
// Mono works fine with nChannels = 1.
//
// Dtype discipline: f64 end-to-end at the API boundary. Production callers
// today pass f32 through Python; the C++ API is f64 to match the internal
// f64 cast that Python does at L275-276 and to match the parity goldens.
// An f32 façade can be layered later without affecting correctness.
class Crossfade
{
public:
    // Band spec (port of Python BANDS tuples from crossfade.py:24-28).
    //   lowHz    low cutoff in Hz; 0 means "lowpass below highHz".
    //   highHz   high cutoff in Hz; -1 means "highpass above lowHz" (Python None).
    //   xfMs     crossfade duration in milliseconds for this band.
    struct Band
    {
        int    lowHz;
        int    highHz;
        double xfMs;
    };

    // Default BANDS per crossfade.py:24-28 — bass LP@200, mid BP[200,4000], treble HP@4000.
    static constexpr Band kDefaultBands[3] = {
        {    0,  200, 200.0},
        {  200, 4000, 100.0},
        { 4000,   -1,  40.0},
    };

    // Default crossfade duration for simple_crossfade (crossfade.py:31).
    static constexpr double kDefaultCrossfadeMs = 75.0;

    // Computes the exact output length per-channel that adaptiveCrossfade
    // will write:  resultLen = outLen + inLen - max_xf_samples
    //   max_xf_samples = min(max_band_xf_samples, outLen, inLen)
    //   max_band_xf_samples = max_b(int(b.xfMs * sr / 1000))
    //
    // Pass `bands == nullptr` + `nBands == 0` to use kDefaultBands.
    static std::size_t computeResultLen(
        std::size_t outLen,
        std::size_t inLen,
        int         sr,
        const Band* bands,
        std::size_t nBands);

    // Computes the exact output length per-channel that simpleCrossfade
    // will write:  resultLen = outLen + inLen - n_overlap
    //   n_overlap = min(int(crossfadeMs * sr / 1000), outLen, inLen)
    static std::size_t computeSimpleResultLen(
        std::size_t outLen,
        std::size_t inLen,
        int         sr,
        double      crossfadeMs);

    // Port of adaptive_crossfade with ADR-032 fix applied.
    //
    //   outgoing    (nChannels × outLen) f64 row-major.
    //   incoming    (nChannels × inLen)  f64 row-major.
    //   sr          sample rate, Hz.
    //   bands       band configuration; pass nullptr+0 for kDefaultBands.
    //   nBands      number of bands.
    //   energyCompensate  match incoming RMS to outgoing per band, clipped to
    //                     ±9 dB (0.35..2.83). Default true in Python.
    //   phaseAlign  align incoming to outgoing phase per band via NCC.
    //               Default true in Python.
    //   maxPhaseShiftMs  max allowed phase shift in ms (3.0 default in Python).
    //                    Production passes 3.0 → 132 samples @ sr=44100.
    //   resultOut   (nChannels × resultCapacityPerChannel) f64 row-major.
    //               Caller must allocate at least `computeResultLen(...)` samples
    //               per channel. Extra capacity is ignored.
    //   resultCapacityPerChannel  asserted to be >= computeResultLen(...).
    //
    // Returns the number of samples-per-channel written (= computeResultLen).
    //
    // Exceptions: may propagate std::invalid_argument from Butterworth::sosFiltFilt
    // if per-band overlap buffer is shorter than the filter padlen (edge = 3*ntaps
    // = 15 for LP/HP order 4, 27 for BP order 4). Caller is responsible for
    // passing overlap windows long enough for the bandpass pass.
    static std::size_t adaptiveCrossfade(
        const double* outgoing,  std::size_t nChannels, std::size_t outLen,
        const double* incoming,  std::size_t inLen,
        int           sr,
        const Band*   bands,     std::size_t nBands,
        bool          energyCompensate,
        bool          phaseAlign,
        double        maxPhaseShiftMs,
        double*       resultOut,
        std::size_t   resultCapacityPerChannel);

    // Port of `simple_crossfade` (crossfade.py:335-366) — single-band equal-power
    // crossfade with an optional pre-fade gain ramp that smooths the clamped RMS
    // compensation factor from `gain` back to 1.0 across `min(n_overlap*4, inLen)`
    // samples of the incoming buffer. Intended as a faster / simpler fallback
    // path to adaptiveCrossfade — no bandpass cascade, no phase-align, no
    // per-band processing.
    //
    //   outgoing       (nChannels × outLen) f64 row-major.
    //   incoming       (nChannels × inLen)  f64 row-major.
    //   sr             sample rate, Hz.
    //   crossfadeMs    overlap duration in milliseconds. Python default 75.0
    //                  (kDefaultCrossfadeMs). `n_overlap = min(int(crossfadeMs *
    //                   sr / 1000), outLen, inLen)`.
    //   energyCompensate  when true AND n_overlap > 0: compute RMS over
    //                  outgoing[-n_overlap:] and incoming[:n_overlap], clip
    //                  rms_out/rms_in to [0.35, 2.83] (±9 dB); skip the
    //                  multiply when either RMS is below 1e-6 (gain threshold).
    //                  When active: multiply incoming[:ramp_len] by a linear
    //                  ramp `linspace(gain, 1.0, ramp_len)` with
    //                  `ramp_len = min(n_overlap*4, inLen)`.
    //   resultOut      (nChannels × resultCapacityPerChannel) f64 row-major.
    //                  Caller allocates at least `computeSimpleResultLen(...)`
    //                  samples per channel.
    //   resultCapacityPerChannel  asserted to be >= computeSimpleResultLen(...).
    //
    // Returns the number of samples-per-channel written (= computeSimpleResultLen).
    //
    // No exceptions — no Butterworth usage, pure scalar arithmetic.
    static std::size_t simpleCrossfade(
        const double* outgoing,  std::size_t nChannels, std::size_t outLen,
        const double* incoming,  std::size_t inLen,
        int           sr,
        double        crossfadeMs,
        bool          energyCompensate,
        double*       resultOut,
        std::size_t   resultCapacityPerChannel);
};

} // namespace reamix::render
