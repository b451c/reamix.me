#include "dsp/OnsetStrength.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace reamix::dsp {

std::vector<float>
OnsetStrength::compute(const std::vector<std::vector<float>>& melPower) const
{
    const int nFrames = static_cast<int>(melPower.size());
    if (nFrames == 0) return {};
    const int nMels = static_cast<int>(melPower[0].size());

    // ---- Step 1: power_to_db with ref=1.0, amin=1e-10, top_db=80.0 -----------
    // PARITY: librosa/core/spectrum.py::power_to_db (L1556-1634).
    //   log_spec = 10·log10(max(amin, S))  - 10·log10(max(amin, ref))
    //   log_spec = max(log_spec, log_spec.max() - top_db)   (clip on GLOBAL max)
    // For ref=1.0: log10(max(1e-10, 1.0)) = log10(1.0) = 0 → no subtraction.
    std::vector<std::vector<float>> sDb(nFrames, std::vector<float>(nMels));
    float globalMaxDb = -std::numeric_limits<float>::infinity();
    const float aminF = static_cast<float>(kAmin);
    for (int t = 0; t < nFrames; ++t) {
        for (int m = 0; m < nMels; ++m) {
            float v = melPower[t][m];
            if (v < aminF) v = aminF;
            // PARITY: 10·log10 (NOT 20·log10) — librosa's power_to_db is a
            //         power-domain conversion regardless of input amplitude
            //         vs power semantics (cf. SpectralContrast which feeds
            //         amplitude through the same 10·log10 path; same quirk).
            const float dB = 10.0f * std::log10(v);
            sDb[t][m] = dB;
            if (dB > globalMaxDb) globalMaxDb = dB;
        }
    }
    const float dbFloor = globalMaxDb - static_cast<float>(kTopDb);
    for (int t = 0; t < nFrames; ++t) {
        for (int m = 0; m < nMels; ++m) {
            if (sDb[t][m] < dbFloor) sDb[t][m] = dbFloor;
        }
    }

    // ---- Steps 2-6: diff(lag=1) → max(0, ·) → mean(axis=mel) -----------------
    // PARITY: librosa/onset.py::onset_strength_multi L606-609 (diff + clip),
    //         L613-621 (single-slice util.sync collapses to mean over mel axis).
    // Note: NO pre-pad on the diff input. The librosa pad happens AFTER
    // aggregation (step 7 below), with mode='constant', NOT 'edge'. The
    // session-9 trap-scan was wrong about this; corrected in spec.md
    // § "Step 6 OnsetStrength" by session 10.
    const int aLen = std::max(0, nFrames - kLag);
    std::vector<float> agg(aLen, 0.0f);
    const float invNMels = 1.0f / static_cast<float>(nMels);
    for (int k = 0; k < aLen; ++k) {
        // diff[k] = S_db[k+lag] - S_db[k]; rectify; mean across mel.
        float sum = 0.0f;
        const auto& rowNext = sDb[k + kLag];
        const auto& rowPrev = sDb[k];
        for (int m = 0; m < nMels; ++m) {
            const float d = rowNext[m] - rowPrev[m];
            if (d > 0.0f) sum += d;  // half-wave rectify (= max(0, d))
        }
        agg[k] = sum * invNMels;  // float32 mean (matches numpy on f32 input)
    }

    // ---- Step 7: left-pad kCenterPad zeros, trim back to nFrames -------------
    // PARITY: librosa/onset.py::onset_strength_multi L624-639.
    //   pad_width  = lag + n_fft // (2 * hop_length)
    //   onset_env  = np.pad(onset_env, [(0,0), (pad_width, 0)], mode='constant')
    //   onset_env  = onset_env[..., :n_frames]
    // ⇒ first kCenterPad frames are zero; output frame i ≥ kCenterPad
    // pulls from agg[i - kCenterPad].
    std::vector<float> out(nFrames, 0.0f);
    for (int k = 0; k < aLen; ++k) {
        const int outIdx = kCenterPad + k;
        if (outIdx >= nFrames) break;  // trim to original n_frames
        out[outIdx] = agg[k];
    }
    return out;
}

} // namespace reamix::dsp
