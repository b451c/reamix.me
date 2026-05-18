#include "dsp/STFT.h"

#include "pocketfft_hdronly.h"

#include <cmath>
#include <numbers>

namespace reamix::dsp {

STFT::STFT()
{
    window_.resize(kNfft);
    // Periodic (fftbins=True) Hann: w[i] = 0.5 * (1 - cos(2*pi*i/N)), i=0..N-1.
    // Matches scipy.signal.get_window("hann", kNfft, fftbins=True) which
    // librosa.stft calls with default arguments.
    // PARITY: librosa.filters.get_window default + fftbins=True convention.
    constexpr float kTwoPi = 2.0f * static_cast<float>(std::numbers::pi);
    for (int i = 0; i < kNfft; ++i)
        window_[i] = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(i) / static_cast<float>(kNfft)));
}

int STFT::numFrames(std::size_t nSamples) const noexcept
{
    // center=True pads n_fft/2 on each side → padded length = nSamples + n_fft.
    // Frames = 1 + (padded - n_fft) / hop = 1 + nSamples / hop.
    return 1 + static_cast<int>(nSamples) / kHopLength;
}

std::vector<std::vector<std::complex<float>>>
STFT::stft(const std::vector<float>& y) const
{
    // Zero-pad n_fft/2 on each side (librosa center=True, pad_mode="constant").
    // Diverges from phase-1 MelSpectrogram's reflect padding — intentional.
    constexpr int kPad = kNfft / 2;
    const int nSamples = static_cast<int>(y.size());
    const int paddedLen = nSamples + 2 * kPad;

    std::vector<float> padded(paddedLen, 0.0f);
    for (int i = 0; i < nSamples; ++i)
        padded[kPad + i] = y[i];

    const int nFrames = numFrames(y.size());
    std::vector<std::vector<std::complex<float>>> out(
        nFrames, std::vector<std::complex<float>>(kNumBins));

    // PocketFFT r2c: real float in, complex float out (matches scipy.fft.rfft on float32).
    // fct=1.0f → no normalization (librosa default).
    pocketfft::shape_t shape = {static_cast<std::size_t>(kNfft)};
    pocketfft::stride_t strideIn = {sizeof(float)};
    pocketfft::stride_t strideOut = {sizeof(std::complex<float>)};

    std::vector<float> frame(kNfft);

    for (int f = 0; f < nFrames; ++f)
    {
        const int start = f * kHopLength;
        for (int n = 0; n < kNfft; ++n)
            frame[n] = padded[start + n] * window_[n];

        pocketfft::r2c(shape, strideIn, strideOut, /*axis=*/0, pocketfft::FORWARD,
                       frame.data(), out[f].data(), /*fct=*/1.0f);
    }

    return out;
}

std::vector<std::vector<float>>
STFT::magnitude(const std::vector<float>& y) const
{
    // Matches numpy's np.abs(complex64_array) semantics: hypot of real/imag in
    // single precision. Writing it explicitly (rather than std::abs on complex)
    // pins the order of operations to what numpy does.
    auto spec = stft(y);
    std::vector<std::vector<float>> mag(spec.size(), std::vector<float>(kNumBins));
    for (std::size_t f = 0; f < spec.size(); ++f)
    {
        for (int k = 0; k < kNumBins; ++k)
        {
            const float re = spec[f][k].real();
            const float im = spec[f][k].imag();
            mag[f][k] = std::hypot(re, im);
        }
    }
    return mag;
}

} // namespace reamix::dsp
