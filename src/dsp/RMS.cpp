#include "dsp/RMS.h"

#include <cmath>

namespace reamix::dsp {

std::vector<float>
RMS::compute(const std::vector<float>& y) const
{
    const int n = static_cast<int>(y.size());
    if (n == 0) return {};

    const int pad      = kFrameLength / 2;
    const int nFrames  = n / kHopLength + 1;
    const float invLen = 1.0f / static_cast<float>(kFrameLength);

    std::vector<float> out(static_cast<std::size_t>(nFrames), 0.0f);

    // Samples outside [0, n) are zero (pad_mode='constant' default value 0).
    // Implicit pad: pad samples live at indices [-pad, n + pad), where the
    // unpadded input occupies [0, n). We read y[idx] only when 0 ≤ idx < n.
    for (int t = 0; t < nFrames; ++t) {
        const int start = t * kHopLength - pad;  // absolute sample index in y
        // power = mean(frame²) accumulated in float32, matching
        // np.square(x, dtype=float32) + np.mean(..., axis=-2). Power-of-two
        // frame_length keeps invLen = 1/2048 exact in float32.
        float sumSq = 0.0f;
        for (int k = 0; k < kFrameLength; ++k) {
            const int idx = start + k;
            if (idx >= 0 && idx < n) {
                const float s = y[idx];
                sumSq += s * s;
            }
        }
        const float power = sumSq * invLen;
        out[t] = std::sqrt(power);
    }

    return out;
}

} // namespace reamix::dsp
