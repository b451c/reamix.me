#pragma once

#include "dsp/STFT.h"

#include <vector>

namespace reamix::dsp {

// Mel-spectrogram matching `librosa.feature.melspectrogram` on librosa 0.11.0
// defaults (when invoked from phase-2 dump script):
//   sr=22050, n_fft=2048, hop_length=512, n_mels=128,
//   fmin=0, fmax=sr/2, power=2.0, window="hann", center=True,
//   pad_mode="constant", htk=False, norm="slaney".
//
// Pipeline: STFT (float32 complex) → |·|² (float32 power) →
//           mel_basis @ power (float32 matmul, naive) → mel_power.
//           log_mel = power_to_db(mel_power, ref=np.max, amin=1e-10, top_db=80).
//
// Distinct from `reamix::MelSpectrogram` (phase-1, beat-this). Do NOT share
// state. See meta/RESEARCH.md 2026-04-20 phase-2 entry + ADR-009.
//
// Mel basis follows librosa `filters.mel(dtype=np.float32)`: triangular
// weights computed in float64, stored float32, then in-place Slaney norm
// scaled float32 *= float64 (same round-down sequence as numpy).
//
// Output layout: [frame][mel]. Python dumps shape (n_mels, n_frames) are
// transposed at the parity test.
class MelSpectrogramLibrosa
{
public:
    // PARITY: python-source/config.py:42-46 (sr, n_fft, hop, n_mels implicit
    // via librosa default).
    static constexpr int kSampleRate = 22050;
    static constexpr int kNfft = STFT::kNfft;
    static constexpr int kHopLength = STFT::kHopLength;
    static constexpr int kNumBins = STFT::kNumBins;          // 1025
    static constexpr int kNMels = 128;                        // librosa default

    // librosa.power_to_db defaults: amin=1e-10, top_db=80.
    static constexpr float kAmin = 1.0e-10f;
    static constexpr float kTopDbFloor = 80.0f;

    MelSpectrogramLibrosa();

    // Power mel-spectrogram. Output[frame][mel].
    // Matches `librosa.feature.melspectrogram(y, sr=22050, n_fft=2048,
    // hop_length=512, n_mels=128, power=2.0)` element-wise up to the float32
    // ULP floor (~peak × 2^-22 absolute; see phases/phase-2-features/validation.md).
    std::vector<std::vector<float>> power(const std::vector<float>& y) const;

    // log-mel in dB via `librosa.power_to_db(mel_power, ref=np.max, amin=1e-10,
    // top_db=80)`. Output range [-80, 0]. [frame][mel].
    std::vector<std::vector<float>> logMelDb(const std::vector<float>& y) const;

    // Apply power_to_db directly to an existing mel_power grid (used by MFCC
    // wiring downstream without recomputing STFT).
    //
    // `ref` selects the librosa.power_to_db reference:
    //   ref ≤ 0  →  use melPower.max()   (librosa `ref=np.max`, default here
    //                                     — matches log_mel.npy dumps).
    //   ref > 0  →  use the literal value (e.g. ref=1.0 as librosa.feature.mfcc
    //                                     does internally — matches log_mel_ref1.npy).
    // Uniform shifts of log-mel only move DCT coefficient 0; MFCC[k>0] is
    // invariant to the choice.
    std::vector<std::vector<float>>
    powerToDb(const std::vector<std::vector<float>>& melPower, float ref = 0.0f) const;

    // Mel filterbank, [mel][bin], float32. Exposed for parity debugging
    // (matches `librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128,
    // fmin=0, fmax=11025, htk=False, norm='slaney', dtype=np.float32)`).
    const std::vector<std::vector<float>>& melBasis() const { return melBasis_; }

private:
    STFT stft_;
    std::vector<std::vector<float>> melBasis_;  // [n_mels][n_bins]
};

} // namespace reamix::dsp
