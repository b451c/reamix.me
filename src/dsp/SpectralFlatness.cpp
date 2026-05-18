#include "dsp/SpectralFlatness.h"

#include <cmath>

namespace reamix::dsp {

std::vector<float>
SpectralFlatness::compute(const std::vector<std::vector<float>>& mag) const
{
    const int nFrames = static_cast<int>(mag.size());
    if (nFrames == 0) return {};
    const int nBins = static_cast<int>(mag[0].size());

    std::vector<float> out(static_cast<std::size_t>(nFrames), 0.0f);

    const double invN = 1.0 / static_cast<double>(nBins);

    for (int t = 0; t < nFrames; ++t) {
        const auto& col = mag[t];

        // PARITY: S_thresh = max(amin, |S|^power) with power=2.0, amin=1e-10.
        // |S|^2 in f32 (matches numpy np.maximum + S**power dtype path); log
        // and mean accumulated in f64 so end-to-end flatness drift from
        // numpy's all-f32 path stays at ~f32 output ULP (< 1e-6).
        double sumLog = 0.0;
        double sumLin = 0.0;
        for (int k = 0; k < nBins; ++k) {
            const float m      = col[k];
            const float sq     = m * m;                    // |S|^2 in f32
            const float thresh = sq > kAmin ? sq : kAmin;
            sumLog += std::log(static_cast<double>(thresh));
            sumLin += static_cast<double>(thresh);
        }

        const double gmean    = std::exp(sumLog * invN);
        const double amean    = sumLin * invN;
        const double flatness = gmean / amean;

        out[t] = static_cast<float>(flatness);
    }

    return out;
}

} // namespace reamix::dsp
