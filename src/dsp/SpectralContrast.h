#pragma once

#include <cstddef>
#include <vector>

namespace reamix::dsp {

// Spectral contrast from an amplitude spectrogram, matching
// `librosa.feature.spectral_contrast(y=y, sr=22050, n_fft=2048,
//  hop_length=512, fmin=200.0, n_bands=6, quantile=0.02, linear=False)`
// on librosa 0.11.0.
//
// Pipeline (mirrors librosa/feature/spectral.py spectral_contrast body):
//   S_amp  = _spectrogram(..., power=1)           // amplitude |STFT|¹
//   freq   = fft_frequencies(sr, n_fft)           // linspace(0, sr/2, nBins)
//   octa   = [0, fmin, 2·fmin, ..., 2^n_bands·fmin] = [0, 200, ..., 12800]
//   for k in 0..n_bands:
//       band = (freq >= octa[k]) & (freq <= octa[k+1])
//       if k > 0        : extend band by one bin below (seam sharing)
//       if k == n_bands : extend band to all bins above (Nyquist tail)
//       sub = S_amp[band, :]; if k < n_bands: drop last row (seam ownership)
//       n  = max(1, round-to-even(quantile · sum(band)))
//       sort(sub, axis=-2); valley[k,:] = mean(sub[:n]); peak[k,:] = mean(sub[-n:])
//   contrast = power_to_db(peak) - power_to_db(valley)
//
// power_to_db defaults: ref=1.0, amin=1e-10, top_db=80.0. Formula per array:
//   log_spec = 10·log10(max(amin, M))
//   log_spec = max(log_spec, log_spec.max() - top_db)
// Applied to peak and valley independently, then subtracted.
//
// PARITY: librosa/feature/spectral.py::spectral_contrast (L895–1006),
//         librosa/core/spectrum.py::power_to_db (L1556–1634),
//         librosa/core/convert.py::fft_frequencies (L1030).
//
// Input:
//   S[frame][bin] = |STFT|¹ (amplitude, float32). nBins must equal
//   1 + n_fft/2; n_fft is inferred as 2·(nBins − 1). Note: amplitude,
//   NOT power — differs from ChromaSTFT (power=2) and Mfcc (power=2 via
//   mel). _spectrogram's default is power=1; spectral_contrast accepts it.
//
// Output:
//   contrast[frame][n_bands+1] float32 (7 rows per frame). Shape
//   (n_frames, 7). Python dumps shape (7, n_frames) are transposed at
//   the parity test.
//
// Pass threshold: L∞ ≤ 1e-3 vs the Python dump (VALIDATION.md phase-2).
class SpectralContrast
{
public:
    // PARITY: librosa.feature.spectral_contrast defaults.
    static constexpr int    kNBands   = 6;            // produces 7 output rows
    static constexpr int    kNRows    = kNBands + 1;  // n_bands + 1 = 7
    static constexpr double kFmin     = 200.0;        // Hz; first octave edge
    // kQuantile MUST be double. Python literal `0.02` is a float64; the
    // float32 representation of 0.02 is 0.019999999552... which, when
    // multiplied by sum(band) and passed to nearbyint, round-flips the
    // half-integer value 1.5 down to 1.0 instead of banker's-rounding up to
    // 2.0 (1.4999...664 < 1.5 exactly). Same trap class as PitchTrack's
    // `1.0/0.01f` (session 6). See spec.md § "Known traps" → Step 5.
    static constexpr double kQuantile = 0.02;         // top/bottom fraction
    // power_to_db defaults pinned identically for peak/valley post-processing.
    static constexpr double kAmin     = 1.0e-10;
    static constexpr double kTopDb    = 80.0;

    SpectralContrast() = default;

    // S[frame][bin] amplitude spectrogram → contrast[frame][kNRows].
    // `sr` defaults to phase-2's 22050 per AnalysisConfig.
    std::vector<std::vector<float>>
    compute(const std::vector<std::vector<float>>& S,
            float sr = 22050.0f) const;
};

} // namespace reamix::dsp
