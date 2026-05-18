#include "dsp/WaveformXcorr.h"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace reamix::dsp {

std::pair<double, int>
WaveformXcorr::compute(const float* source,
                       const float* target,
                       std::size_t nSrc,
                       std::size_t nTgt,
                       int maxLag)
{
    // Parity note 1 — short-circuit gates. Python L23-25:
    //   n = min(len(source), len(target))
    //   if n < 16 or max_lag < 0: return 0.0, 0
    const std::size_t n = std::min(nSrc, nTgt);
    if (n < 16 || maxLag < 0)
        return {0.0, 0};

    // Parity note 1 (cont.) — center crop. Python L27-30:
    //   center_n = min(n, max(64, max_lag * 4))
    //   offset   = (n - center_n) // 2
    //   src = source[offset:offset+center_n].astype(np.float64)
    const std::size_t minWindow = static_cast<std::size_t>(std::max(64, maxLag * 4));
    const std::size_t centerN   = std::min(n, minWindow);
    const std::size_t offset    = (n - centerN) / 2;

    std::vector<double> src(centerN), tgt(centerN);
    for (std::size_t i = 0; i < centerN; ++i)
    {
        src[i] = static_cast<double>(source[offset + i]);
        tgt[i] = static_cast<double>(target[offset + i]);
    }

    // Parity note 2 — silence gate on L2 norms. numpy `np.linalg.norm(x)`
    // for 1-D x is `sqrt(sum(x**2))` in f64.
    double srcSumSq = 0.0, tgtSumSq = 0.0;
    for (std::size_t i = 0; i < centerN; ++i)
    {
        srcSumSq += src[i] * src[i];
        tgtSumSq += tgt[i] * tgt[i];
    }
    const double srcNorm = std::sqrt(srcSumSq);
    const double tgtNorm = std::sqrt(tgtSumSq);
    if (srcNorm < 1e-8 || tgtNorm < 1e-8)
        return {0.0, 0};

    // Parity note 3 — fft_size = smallest power of two >= 2 * center_n.
    std::size_t fftSize = 1;
    while (fftSize < 2 * centerN)
        fftSize *= 2;

    // Parity note 4 — forward rfft on zero-padded buffers. numpy
    // `rfft(x, n=fft_size)` is internally "truncate-or-zero-pad to fft_size
    // then fft". We zero-pad explicitly (centerN <= fftSize by construction).
    std::vector<double> srcPadded(fftSize, 0.0);
    std::vector<double> tgtPadded(fftSize, 0.0);
    std::copy(src.begin(), src.end(), srcPadded.begin());
    std::copy(tgt.begin(), tgt.end(), tgtPadded.begin());

    const std::size_t nBins = fftSize / 2 + 1;
    std::vector<std::complex<double>> srcFft(nBins), tgtFft(nBins);

    pocketfft::shape_t   shape        = {fftSize};
    pocketfft::stride_t  strideReal   = {sizeof(double)};
    pocketfft::stride_t  strideCplx   = {sizeof(std::complex<double>)};
    pocketfft::shape_t   axes         = {0};

    pocketfft::r2c(shape, strideReal, strideCplx, axes, pocketfft::FORWARD,
                   srcPadded.data(), srcFft.data(), /*fct=*/1.0);
    pocketfft::r2c(shape, strideReal, strideCplx, axes, pocketfft::FORWARD,
                   tgtPadded.data(), tgtFft.data(), /*fct=*/1.0);

    // Parity note 4 (cont.) — cross-spectrum `src * conj(tgt)`.
    std::vector<std::complex<double>> crossSpec(nBins);
    for (std::size_t k = 0; k < nBins; ++k)
        crossSpec[k] = srcFft[k] * std::conj(tgtFft[k]);

    // Parity note 4 (cont.) — irfft with default 1/N normalization.
    std::vector<double> xcorrFull(fftSize, 0.0);
    pocketfft::c2r(shape, strideCplx, strideReal, axes, pocketfft::BACKWARD,
                   crossSpec.data(), xcorrFull.data(),
                   /*fct=*/1.0 / static_cast<double>(fftSize));

    // Parity note 5 — window lag range. Python L46-48:
    //   pos_lags = xcorr_full[:max_lag + 1]
    //   neg_lags = xcorr_full[-max_lag:] if max_lag > 0 else np.array([])
    //   xcorr_window = np.concatenate([neg_lags, pos_lags])
    // When max_lag == 0, xcorrWindow has length 1 (just zero-lag pos); best
    // lag is forced to 0.
    const std::size_t posLen = static_cast<std::size_t>(maxLag) + 1;
    const std::size_t negLen = (maxLag > 0) ? static_cast<std::size_t>(maxLag) : 0;

    // Parity note 6 — normalize by product of raw norms and argmax.
    // numpy argmax returns first-occurrence on ties; std::max_element matches.
    const double denom = srcNorm * tgtNorm;
    double bestVal = -std::numeric_limits<double>::infinity();
    std::size_t bestIdx = 0;
    // neg_lags portion (i=0..negLen): xcorrFull[fft_size - negLen + i]
    for (std::size_t i = 0; i < negLen; ++i)
    {
        const double v = xcorrFull[fftSize - negLen + i] / denom;
        if (v > bestVal)
        {
            bestVal = v;
            bestIdx = i;
        }
    }
    // pos_lags portion (i=0..posLen): xcorrFull[i]
    for (std::size_t i = 0; i < posLen; ++i)
    {
        const double v = xcorrFull[i] / denom;
        if (v > bestVal)
        {
            bestVal = v;
            bestIdx = negLen + i;
        }
    }

    // Parity note 7 — clamp negative peaks to 0 (Python L53:
    // `max(0.0, float(normalized[best_idx]))`), report lag in samples
    // relative to zero-lag centre: `best_idx - len(neg)`.
    const double similarity = std::max(0.0, bestVal);
    const int    lagSamples = static_cast<int>(bestIdx) - static_cast<int>(negLen);
    return {similarity, lagSamples};
}

} // namespace reamix::dsp
