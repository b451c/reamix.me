#include "dsp/SpectralCentroid.h"

#include <cmath>

namespace reamix::dsp {

std::vector<float>
SpectralCentroid::compute(const std::vector<std::vector<float>>& sAmp,
                          float sr) const
{
    const int nFrames = static_cast<int>(sAmp.size());
    if (nFrames == 0) return {};
    const int nBins = static_cast<int>(sAmp[0].size());

    // freq[k] = k · sr / n_fft. PARITY: librosa.core.convert.fft_frequencies
    // (computed in float64 internally). n_fft derived from n_bins = nFft/2 + 1.
    const int    nFft    = (nBins - 1) * 2;
    const double freqHz0 = static_cast<double>(sr) / static_cast<double>(nFft);

    std::vector<float> out(static_cast<std::size_t>(nFrames), 0.0f);

    for (int t = 0; t < nFrames; ++t) {
        const auto& col = sAmp[t];

        // L1 column sum in float64 (PARITY: librosa.util.normalize
        // `mag = np.abs(S).astype(float)` casts to float64 before summation).
        double length = 0.0;
        for (int k = 0; k < nBins; ++k) {
            length += static_cast<double>(std::fabs(col[k]));
        }

        // PARITY: default threshold = tiny(float32); fill=None leaves silent
        // columns un-normalized by assigning length := 1.0.
        if (length < kTinyFloat32) length = 1.0;

        // Weighted sum. PARITY: librosa does
        //     Snorm = np.empty_like(S)          # float32
        //     Snorm[:] = S / length             # float32 assignment from
        //                                        # float64 broadcast
        //     centroid = np.sum(freq * Snorm)   # float64 (freq is float64)
        // Replicate by downcasting the normalized sample to float32 before
        // the freq-weighted accumulation.
        double centroid = 0.0;
        for (int k = 0; k < nBins; ++k) {
            const float  normSample = static_cast<float>(
                static_cast<double>(col[k]) / length);
            const double freqK      = freqHz0 * static_cast<double>(k);
            centroid += freqK * static_cast<double>(normSample);
        }
        out[t] = static_cast<float>(centroid);
    }

    return out;
}

} // namespace reamix::dsp
