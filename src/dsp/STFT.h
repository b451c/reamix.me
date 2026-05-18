#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace reamix::dsp {

// Short-Time Fourier Transform matching `librosa.stft` defaults on librosa 0.11.0:
//   n_fft=2048, hop_length=512, win_length=n_fft,
//   window="hann" (periodic / fftbins=True), center=True, pad_mode="constant" (zero-pad),
//   no output normalization, single-precision internals (matches librosa.stft
//   dtype inference when y is float32, which is librosa.load's default).
//
// Distinct from `reamix::MelSpectrogram` (phase-1, beat-this): that path uses
// n_fft=1024, hop=441, reflect padding, `1/sqrt(n_fft)` normalization. Do not
// share state between the two — see ADR-009 + meta/RESEARCH.md 2026-04-20.
//
// Output layout: [frame][bin], bins = 1 + n_fft/2. Column-major numpy dumps
// of shape (bins, frames) are transposed by the parity test at read time.
class STFT
{
public:
    // PARITY: python-source/config.py:43-44 (AnalysisConfig.hop_length, AnalysisConfig.n_fft)
    static constexpr int kNfft = 2048;
    static constexpr int kHopLength = 512;
    static constexpr int kNumBins = kNfft / 2 + 1;  // 1025

    STFT();

    // Compute complex STFT. Output[frame][bin], shape (numFrames(y.size()), 1025).
    std::vector<std::vector<std::complex<float>>>
    stft(const std::vector<float>& y) const;

    // Magnitude = |stft|. Same shape. Matches `np.abs(librosa.stft(y))` before
    // the dump script's `.astype(np.float64)` upcast.
    std::vector<std::vector<float>>
    magnitude(const std::vector<float>& y) const;

    // Number of output frames for a given raw (un-padded) input length, under
    // librosa's center=True convention:
    //   len(y_padded) = len(y) + n_fft     (n_fft/2 pad on each side)
    //   n_frames     = 1 + (len(y_padded) - n_fft) / hop_length
    //                = 1 + len(y) / hop_length
    int numFrames(std::size_t nSamples) const noexcept;

    // Access to the precomputed Hann window — needed by ISTFT (librosa reuses
    // the same window for forward and inverse when win_length == n_fft).
    const std::vector<float>& windowRef() const noexcept { return window_; }

private:
    // Periodic Hann window of length n_fft (fftbins=True convention).
    std::vector<float> window_;
};

} // namespace reamix::dsp
