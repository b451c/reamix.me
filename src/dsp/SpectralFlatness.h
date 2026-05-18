#pragma once

#include <vector>

namespace reamix::dsp {

// Spectral flatness per `librosa.feature.spectral_flatness(y, n_fft=2048,
// hop_length=512, power=2.0, amin=1e-10)` on librosa 0.11.0.
//
// Source audit (Hard Rule #8 — `inspect.getsource(librosa.feature.spectral_flatness)`
// 2026-04-20 session 4):
//
//   S, _ = _spectrogram(y=y, n_fft=2048, hop_length=512, power=1.0, ...)
//   # _spectrogram(..., power=1.0) returns MAGNITUDE |STFT|, NOT power |STFT|^2.
//   S_thresh = np.maximum(amin, S**power)
//   # power=2.0 applied AFTER magnitude → max(amin, |S|^2). amin after exponentiation.
//   gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))
//   amean = np.mean(S_thresh, axis=-2, keepdims=True)
//   flatness = gmean / amean                 # shape (1, t), dtype float32
//
// numpy dtype trace (1.26.4): S is f32 (librosa STFT on f32 input). `S**power`,
// `np.maximum(1e-10, ...)`, `np.log`, `np.mean`, `np.exp`, division — all stay
// in f32. 1e-10 is a Python scalar (f64) but numpy does not promote the array
// dtype on scalar-op-array with matching kind.
//
// API takes a PRE-COMPUTED magnitude matrix (matches SpectralCentroid
// convention; phase-2 precedent). Isolates parity tests from STFT ULP drift
// and lets the VocalFeatures orchestrator compute `|STFT(y_harmonic)|` once
// and reuse for the voice_band_ratio computation (Python wastefully computes
// STFT twice internally).
//
// C++ choice: accumulate log-sum and arithmetic-mean in f64 to keep drift at
// the f32 output ULP floor (~1.2e-7 × max|flatness|). Cast to f32 at the end.
// Target L∞ ≤ 1e-6 on pre-computed magnitude input (phase-2b/spec.md
// § "Parity targets per module").
class SpectralFlatness
{
public:
    // PARITY: librosa.feature.spectral_flatness defaults used throughout
    // phase-2 / 2b — hop_length=512, n_fft=2048.
    static constexpr int kNfft       = 2048;
    static constexpr int kHopLength  = 512;
    static constexpr int kNumBins    = kNfft / 2 + 1;  // 1025

    // power=2.0 and amin=1e-10 are librosa.feature.spectral_flatness defaults.
    // Kept as f32 literals so C++ arithmetic matches numpy's f32-kind path
    // (Python scalar 1e-10 → rounded to nearest f32 when broadcast against
    // f32 array via np.maximum). Effectively f32(1e-10) = 1.0000000133514e-10.
    static constexpr float kAmin     = 1e-10f;

    SpectralFlatness() = default;

    // mag: [frame][bin], float32 — `|STFT(y)|`. Shape from STFT (nFrames, 1025).
    // Returns flatness[frame], float32, length = mag.size().
    std::vector<float>
    compute(const std::vector<std::vector<float>>& mag) const;
};

} // namespace reamix::dsp
