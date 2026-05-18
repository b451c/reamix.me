#pragma once

#include <vector>

namespace reamix::dsp {

// Spectral centroid per STFT frame, matching
// `librosa.feature.spectral_centroid(y=y, sr=22050, hop_length=512)` with all
// other defaults (n_fft=2048, win_length=None=n_fft, window='hann',
// center=True, pad_mode='constant') on librosa 0.11.0.
//
// We take |STFT| directly (amplitude, NOT power — same input class as
// SpectralContrast), isolating centroid drift from upstream STFT drift.
// Formula (librosa/feature/spectral.py::spectral_centroid):
//
//   S_norm[k, t] = S[k, t] / length[t]                  (L1 column-norm)
//   length[t]    = sum_k |S[k, t]|   in float64
//   if length[t] < tiny(float32) = 2^-126 ≈ 1.175e-38:
//       length[t] = 1.0             (fill=None → un-normalize silent frames)
//   centroid[t]  = sum_k ( freq[k] · S_norm[k, t] )     in float64
//   freq[k]      = k · sr / n_fft                        (k=0..n_fft/2)
//
// PARITY:
//   librosa/feature/spectral.py::spectral_centroid (y-branch via _spectrogram)
//   librosa/util/utils.py::normalize (L1 axis=-2, default threshold=tiny(S))
//   librosa/core/convert.py::fft_frequencies
//
// Precision dance (mirrors librosa):
//   S input is float32 (|STFT| on float32 audio). librosa's normalize()
//   casts magnitude to float64 ( `mag = np.abs(S).astype(float)` ) so
//   `length` is float64. Then `Snorm[:] = S / length` broadcasts to float64
//   and downcasts back to float32 via `np.empty_like(S)`. The freq-weighted
//   sum then upcasts float32 × float64 → float64. We replicate by:
//     - accumulate length in double,
//     - downcast normalized sample to float,
//     - accumulate centroid in double,
//     - return float.
//
// Input:  sAmp[frame][bin] — float32, amplitude |STFT|, shape
//         (n_frames, n_fft/2 + 1). Same convention as SpectralContrast.
//
// Output: centroid[frame] — float, length n_frames. Units: Hz.
//
// Pass threshold: L∞ ≤ 1e-3 (VALIDATION.md, phase-2). In practice expect
// ~chroma-class accumulation floor (≈ 2e-6 · peak ≈ 2e-2 for centroid peak
// ~10 kHz) — well within 1e-3 after normalization? No — threshold applies to
// absolute diff in Hz. For centroid values ~5 kHz with float64 accumulation,
// diff should stay below 1e-3 Hz.
//
// Cross-cutting rule (codified de18701, codeword "0.02f != 0.02"):
//   no decimal class constants in this header. sr is passed at call time.
class SpectralCentroid
{
public:
    // PARITY: librosa.feature.spectral_centroid defaults.
    static constexpr int    kNfft      = 2048;
    static constexpr int    kHopLength = 512;  // unused at compute() time
                                               // (test feeds |STFT|), kept for
                                               // completeness at orchestrator
                                               // call site.
    // Denominator guard for near-silent columns. PARITY:
    //   librosa.util.normalize default threshold=tiny(S). For float32 input
    //   tiny = 2^-126. Columns with L1-sum below this threshold are left
    //   un-normalized (length := 1.0 under fill=None). Declared as `double`
    //   per cross-cutting rule.
    static constexpr double kTinyFloat32 = 1.1754943508222875e-38;

    SpectralCentroid() = default;

    // |STFT| amplitude + sample rate → centroid[frame] in Hz.
    std::vector<float>
    compute(const std::vector<std::vector<float>>& sAmp, float sr) const;
};

} // namespace reamix::dsp
