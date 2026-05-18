#pragma once

#include <cstddef>
#include <vector>

namespace reamix::dsp {

// Chromagram from a power spectrogram, matching
// `librosa.feature.chroma_stft(S=|STFT|**2, sr=22050, n_fft=2048,
//  hop_length=512, tuning=<scalar>, n_chroma=12)` on librosa 0.11.0.
//
// Pipeline (mirrors librosa/feature/spectral.py L1262-1286):
//   chromafb = librosa.filters.chroma(sr, n_fft, tuning, n_chroma=12,
//                                     ctroct=5.0, octwidth=2, norm=2,
//                                     base_c=True, dtype=np.float32)
//   raw      = einsum('cf,ft->ct', chromafb, S)          // float32 matmul
//   chroma   = util.normalize(raw, norm=np.inf, axis=-2)  // L∞ per frame
//
// PARITY: librosa/filters.py::chroma (L266-407), librosa/util/utils.py::
// normalize (L792-...), librosa/core/convert.py::hz_to_octs (L1336-1372).
//
// Input:
//   S[frame][bin] = |STFT|² (power spectrogram, float32). nBins must equal
//   1 + n_fft/2; n_fft is inferred as 2·(nBins - 1). Matches the float32-
//   internal path `_spectrogram(..., power=2)` takes inside chroma_stft
//   (complex64 → float32 abs → float32 square).
//
// Tuning:
//   Fractional-bin deviation from A440, typically produced by
//   `PitchTrack::estimateTuning(S)` (ADR-010). `chroma_stft(tuning=None)`
//   would compute this internally; passing it explicitly lets parity
//   isolate chroma drift from PitchTrack drift.
//
// Output:
//   chroma[frame][n_chroma] float32, L∞-normalized per frame. Shape
//   (n_frames, 12). Python dumps shape (12, n_frames) are transposed at
//   the parity test.
class ChromaSTFT
{
public:
    // PARITY: librosa.feature.chroma_stft + librosa.filters.chroma defaults.
    static constexpr int    kNChroma  = 12;
    static constexpr float  kCtroct   = 5.0f;  // Gaussian center in octs (A0=27.5Hz)
    static constexpr float  kOctwidth = 2.0f;  // Gaussian half-width in octs
    // Filter-bank norm is L2 (per column), frame norm is L∞ (per time slice).
    // Both hard-coded to match `librosa.feature.chroma_stft(norm=np.inf,
    // **kwargs)` with default filters.chroma(norm=2).

    ChromaSTFT() = default;

    // S[frame][bin] power spectrogram → chroma[frame][kNChroma].
    // `tuning` is fractional-bin deviation from A440; `sr` defaults to
    // phase-2's 22050 per AnalysisConfig.
    std::vector<std::vector<float>>
    compute(const std::vector<std::vector<float>>& S,
            float tuning,
            float sr = 22050.0f) const;
};

} // namespace reamix::dsp
