#pragma once

#include <vector>

namespace reamix::dsp {

// Pitch-based tuning estimator matching `librosa.estimate_tuning` on the
// S-path (power spectrogram input) as called internally by
// `librosa.feature.chroma_stft(tuning=None)` on librosa 0.11.0:
//
//   tuning = estimate_tuning(S=|STFT|**2, sr=sr, bins_per_octave=n_chroma)
//
// Ported here to make the handcrafted 59-dim feature vector data-independent
// of Python: without this, ChromaSTFT parity would require either pinning
// tuning=0.0 (which empirically breaks chroma parity by 180-660× over the
// 1e-3 threshold on the phase-2 goldens — see ADR-010) or passing a
// pre-computed scalar from Python (silent divergence risk).
//
// Pipeline (matches librosa/core/pitch.py estimate_tuning → piptrack →
// pitch_tuning):
//   1. piptrack(|STFT|²) → per-bin pitches & magnitudes with defaults
//      fmin=150, fmax=4000, threshold=0.1 (× max per frame).
//   2. threshold = median(mag[pitch > 0])  (or 0 if none).
//   3. freqs = pitches[(mag >= threshold) & (pitch > 0)].
//   4. pitch_tuning(freqs, resolution=0.01, bins_per_octave=12):
//        residual = fmod(bpo · log2(f / 27.5), 1.0)   // 27.5 = 440/16
//        residual[residual >= 0.5] -= 1.0
//        bins = linspace(-0.5, 0.5, 101)              // 100-bin histogram
//        return bins[argmax(histogram)]               // left edge of max
//
// PARITY: librosa/core/pitch.py L26-177 (estimate_tuning, pitch_tuning),
//         L180-360 (piptrack), L416-473 (_parabolic_interpolation).
//         librosa/util/utils.py L1023-1127 (localmax stencil).
//         librosa/core/convert.py L1336-1372 (hz_to_octs) + L1586-1607
//         (fft_frequencies).
//
// Implementation notes:
//   - Input S[frame][bin] = |STFT|² (float32, non-negative). n_bins-1 must
//     equal n_fft/2, i.e. n_fft = 2 · (n_bins - 1) — inferred from S.
//   - Internal piptrack math runs in float32 to mirror librosa's complex64
//     → float32 abs → float32 square path. pitch_tuning switches to
//     float64 for the log2/fmod/histogram step (matches numpy / scipy).
//   - Output is the left edge of the winning histogram bin, per librosa's
//     `tuning[np.argmax(counts)]` semantics (NOT the bin center).
class PitchTrack
{
public:
    // PARITY: librosa/core/pitch.py piptrack defaults (fmin=150, fmax=4000,
    // threshold=0.1) and estimate_tuning (resolution=0.01, bpo=12).
    static constexpr float kFmin          = 150.0f;
    static constexpr float kFmax          = 4000.0f;
    static constexpr float  kRefThreshold  = 0.1f;
    static constexpr int    kBinsPerOctave = 12;
    // Resolution is double (not float) because the histogram bin count is
    // `int(ceil(1.0 / resolution)) + 1` — a float32 0.01 widens to a double
    // slightly > 0.01, which shifts the count from 101 to 102. Python uses
    // a float64 literal `resolution=0.01`, so we mirror that.
    static constexpr double kResolution    = 0.01;

    PitchTrack() = default;

    // Estimate tuning deviation in [-0.5, 0.5), in fractions of a bin.
    //
    // S[frame][bin] = |STFT|² (power spectrogram). n_fft is inferred as
    // 2·(n_bins - 1); sr defaults to phase-2's 22050 per AnalysisConfig.
    // On empty input (or no pitches above threshold) returns 0.0 and emits
    // no warning — matches librosa's `warnings.warn(...)` + `return 0.0`
    // branch minus the warning (C++ callers are expected to check inputs).
    float estimateTuning(const std::vector<std::vector<float>>& S,
                         float sr = 22050.0f,
                         int binsPerOctave = kBinsPerOctave,
                         double resolution = kResolution) const;
};

} // namespace reamix::dsp
