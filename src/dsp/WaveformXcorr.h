#pragma once

#include <cstddef>
#include <utility>

namespace reamix::dsp {

// FFT-based normalized cross-correlation of two equal-budget mono snippets.
// Port of `waveform_xcorr` (python-source/remix/waveform_utils.py L14-53).
// Used by:
//   - RepetitionMap (phase-3 step 8) — bar-boundary transition verification.
//   - Phase-4 remix transition cost scoring (future; same function).
//
// Algorithm in one sentence: window both snippets around their common center,
// compute xcorr(src, tgt) via `irfft(rfft(src) * conj(rfft(tgt)))`, restrict
// the result to `[-max_lag, +max_lag]` taps, normalize by the product of
// the raw L2 norms, and report the peak value + its lag.
//
// PARITY notes (f64 throughout matches numpy rfft/irfft on float64 inputs):
//
//   1. Center crop:   center_n = min(n, max(64, max_lag * 4)),
//                     offset   = (n - center_n) / 2  (integer division).
//      `n = min(nSrc, nTgt)`. Short inputs (n < 16) or `max_lag < 0`
//      short-circuit to (0.0, 0) matching Python L24.
//
//   2. Silence gate: if either post-crop L2 norm < 1e-8, return (0.0, 0).
//      Computed via `std::hypot`-free sum-of-squares then sqrt (numpy
//      `np.linalg.norm` semantics). Norm is f64 scalar.
//
//   3. FFT size:    smallest power of two `>= 2 * center_n`. `fft_size >= 2`
//                   always (center_n >= 16 given n >= 16 gate). numpy rfft
//                   zero-pads/truncates to `n=fft_size` internally — we pad
//                   explicitly at the real buffer.
//
//   4. Transform:   PocketFFT `r2c(fct=1.0)` on src_padded and tgt_padded.
//                   Multiply `src_fft[k] * conj(tgt_fft[k])` element-wise.
//                   `c2r(fct=1.0 / fft_size)` gives the length-fft_size real
//                   ifft with numpy's default 1/N normalization.
//
//   5. Window:      `pos = xcorr[0 .. max_lag]` (length max_lag+1),
//                   `neg = xcorr[fft_size - max_lag .. fft_size]`
//                                                  (length max_lag, empty if
//                                                  max_lag == 0).
//                   Concatenate `[neg, pos]` (length 2*max_lag + 1 when
//                   max_lag > 0, length 1 when max_lag == 0).
//
//   6. Normalize:   divide the windowed array by `src_norm * tgt_norm` (f64).
//                   Argmax over the normalized array.
//
//   7. Return:      `(max(0.0, normalized[best_idx]), best_idx - len(neg))`.
//                   Python casts similarity to float; C++ returns double.
//                   Lag sign matches Python: positive when src leads tgt.
//
// Numerical edge cases (mirroring numpy):
//   - irfft's default is "inverse by 1/N"; PocketFFT fct-parameter must match.
//   - `neg_lags = xcorr_full[-max_lag:] if max_lag > 0 else np.array([])` —
//     zero-length `neg` when max_lag == 0 (single-sample windowed output).
//   - `np.argmax` returns the FIRST occurrence of the maximum (important when
//     multiple lags tie). std::max_element has the same behavior.
//   - Power-of-two `fft_size` grows monotonically; for n=center_n = 16 this
//     yields fft_size = 32.
class WaveformXcorr
{
public:
    // Compute normalized xcorr. Inputs are float32 (matches the callers'
    // boundary_waveforms storage; production snippets are f32). Returns
    // (similarity in [0, 1], best_lag_samples). Lag is zero-centered:
    // positive → src best-aligned `lag` samples after tgt.
    //
    // `nSrc` and `nTgt` may differ; `n = min(nSrc, nTgt)` is used. Signatures
    // that give n < 16 or max_lag < 0 short-circuit per parity note 1.
    static std::pair<double, int>
    compute(const float* source,
            const float* target,
            std::size_t nSrc,
            std::size_t nTgt,
            int maxLag);
};

} // namespace reamix::dsp
