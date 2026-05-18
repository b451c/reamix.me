#pragma once

#include <cstddef>
#include <vector>

namespace reamix::dsp {

// Onset-strength envelope from a mel-power spectrogram, matching
// `librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)` on
// librosa 0.11.0 (with all internal defaults: lag=1, max_size=1,
// ref=None, detrend=False, center=True, feature=melspectrogram,
// aggregate=np.mean).
//
// Pipeline (mirrors librosa/onset.py::onset_strength_multi body L580-639,
// onset_strength wrapper L351-366; CORRECTED from session-9 trap-scan
// which mis-described the pad — see spec.md § "Step 6 OnsetStrength"):
//
//   1. S       = mel_power                       (input, [frame][mel])
//   2. S_db    = power_to_db(S, ref=1.0,
//                            amin=1e-10,
//                            top_db=80.0)         (clip on GLOBAL max - 80)
//   3. ref     = S_db                             (max_size==1 → skip filter)
//   4. diff    = S_db[1:, :] - ref[:-1, :]        (adjacent-frame diff;
//                                                  shape (n_frames-1, n_mels))
//   5. diff    = max(0, diff)                     (half-wave rectify)
//   6. agg[k]  = mean_mel( diff[k, :] )           (axis=mel, k=0..n_frames-2)
//   7. left-pad with pad_width = lag + n_fft/(2·hop) = 1 + 2 = 3 zeros
//      (mode='constant'); then trim back to n_frames.
//   ⇒ out[0..2] == 0; out[i] = agg[i - pad_width] for i >= pad_width.
//
// PARITY: librosa/onset.py::onset_strength (L216-366),
//         librosa/onset.py::onset_strength_multi (L445-641),
//         librosa/core/spectrum.py::power_to_db  (L1556-1634).
//
// Input:  melPower[frame][mel] — float32, output of
//         reamix::dsp::MelSpectrogramLibrosa::power. Shape (n_frames, n_mels).
//         Non-negative; no np.abs needed (librosa applies one but it is a
//         no-op on this path — see L581 of onset.py).
//
// Output: onset[frame] — float32, length n_frames. First kCenterPad (=3 for
//         default n_fft=2048, hop_length=512) frames are always zero by
//         construction; the librosa center-shift compensation lives there.
//
// Pass threshold: L∞ ≤ 1e-3 vs the Python dump (VALIDATION.md phase-2).
//
// Cross-cutting rule (codified de18701, codeword "0.02f != 0.02"):
//   every decimal class constant is `double`, never `Xf`. log10 / nearbyint
//   downstream is sensitive to float32-representation drift on these
//   reciprocals/quantiles (PitchTrack session-6 + SpectralContrast session-9
//   precedents; see spec.md § "Cross-cutting rule").
class OnsetStrength
{
public:
    // PARITY: librosa.onset.onset_strength_multi defaults.
    static constexpr int kLag       = 1;     // 1-frame diff
    static constexpr int kMaxSize   = 1;     // no maximum_filter1d
    static constexpr int kNfft      = 2048;  // STFT n_fft
    static constexpr int kHopLength = 512;   // STFT hop
    // power_to_db defaults pinned identically.
    static constexpr double kAmin   = 1.0e-10;
    static constexpr double kTopDb  = 80.0;
    // Librosa center-mode framing compensation: pad_width = lag + n_fft/(2·hop).
    // For the default (lag=1, n_fft=2048, hop=512) this evaluates to 3.
    // PARITY: librosa/onset.py::onset_strength_multi L624-627.
    static constexpr int kCenterPad = kLag + kNfft / (2 * kHopLength);

    OnsetStrength() = default;

    // melPower[frame][mel] (output of MelSpectrogramLibrosa::power) →
    // onset[frame] of length melPower.size().
    std::vector<float>
    compute(const std::vector<std::vector<float>>& melPower) const;
};

} // namespace reamix::dsp
