#include "dsp/HPSS.h"

#include "dsp/STFT.h"
#include "util/MedianFilter.h"
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace reamix::dsp {

namespace {

// PARITY: librosa.util.softmask (librosa/util/utils.py). Per-cell:
//   Z = max(X, Xref) cast to f32
//   bad = Z < tiny(f32) = 2^-126
//   where bad: Z=1, mask = split_zeros ? 0.5 : 0.0
//   else:     mask = X^p / (X^p + Xref^p)   with X'=X/Z, Xref'=Xref/Z (re-scaled)
// Inputs and output are f32, same dtype as librosa's magnitude spectrogram.
// power is a hardcoded compile-time 2 here (librosa.effects.hpss default).
// split_zeros is True because decompose.hpss computes it as
// `margin_harm == 1 AND margin_perc == 1`, and effects.hpss passes
// margin=1.0 by default — hence mask=0.5 on double-zero cells.
inline void softmaskPower2(const float* X, const float* Xref, std::size_t n,
                           bool splitZeros, float* mask)
{
    constexpr float kTiny = std::numeric_limits<float>::min(); // 2^-126 for f32
    const float zeroMask = splitZeros ? 0.5f : 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        const float x = X[i];
        const float r = Xref[i];
        const float z = std::max(x, r);
        if (z < kTiny) {
            mask[i] = zeroMask;
            continue;
        }
        // Relative re-scale avoids f32 overflow in X*X / R*R for loud inputs.
        const float xs = x / z;
        const float rs = r / z;
        const float xp = xs * xs;
        const float rp = rs * rs;
        mask[i] = xp / (xp + rp);
    }
}

// PARITY: librosa.istft with n_fft=2048, hop=512, win_length=2048,
// window=hann (periodic), center=True, length=nOut. Complex STFT input
// is stored as [frame][bin] (same layout as reamix::dsp::STFT::stft output).
//
// Algorithm (equivalent to librosa's head-buffer layout, written as a
// straight overlap-add for clarity):
//   padded_y  [n_fft/2 + nOut + n_fft/2] = 0
//   padded_ws [same size]                = 0
//   for each frame f in [0, nFrames):
//       time = window * irfft(stft[f], n=n_fft)
//       padded_y [f*hop .. f*hop+n_fft] += time
//       padded_ws[f*hop .. f*hop+n_fft] += window^2
//   y = padded_y[n_fft/2 : n_fft/2 + nOut]
//   ws = padded_ws[n_fft/2 : n_fft/2 + nOut]
//   y[ws > tiny] /= ws[ws > tiny]
//
// Note: the irfft has numpy's default `1/n_fft` normalization (matches
// `np.fft.irfft`). pocketfft's c2r with fct=1/n_fft produces the same.
std::vector<float> istftCentered(
    const std::vector<std::vector<std::complex<float>>>& stft,
    const std::vector<float>& window,
    std::size_t nOut)
{
    constexpr int kNfft = STFT::kNfft;
    constexpr int kHop  = STFT::kHopLength;
    constexpr int kBins = STFT::kNumBins;
    constexpr int kPad  = kNfft / 2;

    const int nFrames = static_cast<int>(stft.size());
    // Trim to what's needed for exact output length (matches librosa's
    // `n_frames = min(input_n_frames, ceil((length + 2*n_fft/2)/hop))`).
    // For our canonical forward STFT with n_frames = 1 + N/hop, padded_need
    // = ceil((N+n_fft)/hop) = ceil(N/hop) + n_fft/hop ≥ 1 + N/hop — so the
    // min is always the full set. We still cap defensively.
    const std::size_t paddedLen = static_cast<std::size_t>(2 * kPad) + nOut;
    const int nFramesNeeded =
        static_cast<int>((paddedLen + static_cast<std::size_t>(kHop) - 1u) /
                         static_cast<std::size_t>(kHop));
    const int nFramesUsed = std::min(nFrames, nFramesNeeded);

    const std::size_t bufSize =
        static_cast<std::size_t>(kNfft) +
        static_cast<std::size_t>(kHop) * static_cast<std::size_t>(std::max(nFramesUsed - 1, 0));
    const std::size_t bigSize = std::max(bufSize, paddedLen);

    std::vector<float> paddedY(bigSize, 0.0f);
    std::vector<float> paddedWs(bigSize, 0.0f);

    pocketfft::shape_t shape = {static_cast<std::size_t>(kNfft)};
    pocketfft::stride_t strideIn  = {sizeof(std::complex<float>)};
    pocketfft::stride_t strideOut = {sizeof(float)};

    std::vector<float> time(kNfft);

    // Precompute window^2 (reused per frame).
    std::vector<float> winSq(kNfft);
    for (int i = 0; i < kNfft; ++i)
        winSq[i] = window[i] * window[i];

    const float kInvNfft = 1.0f / static_cast<float>(kNfft);

    for (int f = 0; f < nFramesUsed; ++f) {
        // c2r with fct=1/n_fft matches np.fft.irfft(X, n=n_fft).
        pocketfft::c2r(shape, strideIn, strideOut, /*axis=*/0,
                       pocketfft::BACKWARD,
                       stft[f].data(), time.data(), kInvNfft);

        const std::size_t off = static_cast<std::size_t>(f) * static_cast<std::size_t>(kHop);
        for (int n = 0; n < kNfft; ++n) {
            const float t = time[n] * window[n];
            paddedY [off + static_cast<std::size_t>(n)] += t;
            paddedWs[off + static_cast<std::size_t>(n)] += winSq[n];
        }
        (void)kBins;
    }

    // Trim the center-pad and normalize by window_sumsquare.
    // PARITY: librosa trims n_fft/2 off the front of the window-sumsquare
    // before dividing; `util.fix_length(…, size=y.shape[-1])` truncates or
    // zero-extends so ws has length nOut.
    const float tinyWs = std::numeric_limits<float>::min(); // tiny(f32) = 2^-126

    std::vector<float> y(nOut, 0.0f);
    const std::size_t start = static_cast<std::size_t>(kPad);
    for (std::size_t n = 0; n < nOut; ++n) {
        const std::size_t k = start + n;
        if (k >= paddedY.size())
            break;
        const float ws = paddedWs[k];
        if (ws > tinyWs)
            y[n] = paddedY[k] / ws;
        else
            y[n] = paddedY[k];
    }
    return y;
}

} // namespace

std::vector<float> HPSS::harmonic(const std::vector<float>& y)
{
    if (y.empty())
        return {};

    // 1. Forward STFT (complex) and magnitude in (bins, frames) layout.
    STFT fwd;
    auto spec = fwd.stft(y);                 // [nFrames][1025]
    const std::size_t nFrames = spec.size();
    constexpr std::size_t kBins = STFT::kNumBins;

    // magnitude stored row-major (kBins rows, nFrames cols) — matches
    // numpy shape (bins, frames) that median_filter operates on.
    std::vector<float> mag(kBins * nFrames, 0.0f);
    // Phase unit vector stored as complex with the same [frame][bin] layout
    // as `spec`. PARITY: librosa.magphase returns phase=1+0j for |D|==0.
    std::vector<std::complex<float>> phase(nFrames * kBins, {0.0f, 0.0f});

    for (std::size_t f = 0; f < nFrames; ++f) {
        for (std::size_t k = 0; k < kBins; ++k) {
            const std::complex<float>& c = spec[f][k];
            const float m = std::hypot(c.real(), c.imag());
            mag[k * nFrames + f] = m;
            if (m == 0.0f) {
                phase[f * kBins + k] = {1.0f, 0.0f};
            } else {
                phase[f * kBins + k] = {c.real() / m, c.imag() / m};
            }
        }
    }

    // 2. Median filters on magnitude.
    //   harm kernel (1, 31) — smooth along time (cols).
    //   perc kernel (31, 1) — smooth along freq (rows).
    std::vector<float> harm(kBins * nFrames);
    std::vector<float> perc(kBins * nFrames);
    const std::size_t kKer = static_cast<std::size_t>(kKernelSize);
    reamix::util::medianFilter2DReflect<float>(
        mag.data(), kBins, nFrames, 1u,  kKer, harm.data());
    reamix::util::medianFilter2DReflect<float>(
        mag.data(), kBins, nFrames, kKer, 1u,  perc.data());

    // 3. Soft mask: mask_harm = harm^2 / (harm^2 + perc^2), split_zeros=True
    //    (margin_harm == margin_perc == 1.0, per decompose.hpss).
    std::vector<float> maskHarm(kBins * nFrames);
    softmaskPower2(harm.data(), perc.data(), harm.size(),
                   /*splitZeros=*/true, maskHarm.data());

    // 4. Apply mask to complex STFT: stft_harm[f][k] = D[f][k] * mask_harm[k,f].
    //    Equivalent to librosa's `(S * mask) * phase` because S is |D| from
    //    magphase and phase is D / |D| → S * phase = D on non-zero cells,
    //    and on zero cells phase=1 so S*phase = 0 = D.
    std::vector<std::vector<std::complex<float>>> stftHarm = spec;  // copy shape
    for (std::size_t f = 0; f < nFrames; ++f) {
        for (std::size_t k = 0; k < kBins; ++k) {
            stftHarm[f][k] *= maskHarm[k * nFrames + f];
        }
    }

    // 5. Inverse STFT to y_harmonic of length y.size().
    return istftCentered(stftHarm, fwd.windowRef(), y.size());
}

} // namespace reamix::dsp
