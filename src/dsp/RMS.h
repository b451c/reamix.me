#pragma once

#include <vector>

namespace reamix::dsp {

// Root-mean-square energy per STFT-aligned frame, matching
// `librosa.feature.rms(y=y, hop_length=512)` with all other defaults
// (frame_length=2048, center=True, pad_mode='constant', dtype=np.float32)
// on librosa 0.11.0.
//
// Pipeline (mirrors librosa/feature/spectral.py::rms L? y-branch):
//
//   1. y_padded = np.pad(y, [(pad, pad)], mode='constant')   pad = frame_length/2
//   2. For each frame t in 0..n_frames-1:
//        start = t · hop_length
//        frame = y_padded[start : start + frame_length]
//        power = mean_over_samples( frame² )   (accumulated as float32;
//                                              np.square(..., dtype=float32) →
//                                              np.mean with default axis dtype)
//        rms[t] = sqrt(power)
//
//   n_frames = (len(y) // hop_length) + 1
//   (derived: padded_len = len(y) + 2·pad = len(y) + frame_length;
//            frames = (padded_len - frame_length) / hop_length + 1
//                   = len(y) / hop_length + 1.  For 60 s × 22050 Hz /
//            hop 512 this matches STFT's n_frames = 2584 exactly.)
//
// PARITY: librosa/feature/spectral.py::rms (y-branch) +
//         librosa/util/utils.py::frame (stride view → direct indexing in C++) +
//         librosa/util/utils.py::abs2 (→ np.square with dtype=float32).
//
// Input:  y — float32 PCM, mono, length sr·clip_seconds. Dump emits as
//         float64 (astype on read-back), so test loader casts back to f32.
//
// Output: rms[frame] — float32, length n_frames (= y.size()/hop_length + 1).
//
// Pass threshold: L∞ ≤ 1e-3 (VALIDATION.md, phase-2).
//
// Cross-cutting rule (codified de18701, codeword "0.02f != 0.02"):
//   there are no decimal class constants in RMS — just ints. The mean
//   division uses 1.0f/kFrameLength computed as float32 at call time
//   (frame_length is a power of two → reciprocal is exact in float32).
class RMS
{
public:
    // PARITY: librosa.feature.rms defaults.
    static constexpr int kFrameLength = 2048;
    static constexpr int kHopLength   = 512;

    RMS() = default;

    // y (float32 time-domain audio) → rms[frame] of length
    // (y.size() / kHopLength) + 1.
    std::vector<float> compute(const std::vector<float>& y) const;
};

} // namespace reamix::dsp
