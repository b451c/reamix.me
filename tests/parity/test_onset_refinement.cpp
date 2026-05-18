// Parity test: reamix::OnsetRefinement must produce byte-identical output
// to REABeat's reference OnsetRefinement for the same input.
//
// Success criterion: max |Δ| == 0.0 on the refined-position vector for every
// case. Both ports share the same pocketfft instructions on the same input
// buffer, so bitwise parity is structurally guaranteed (same pattern as
// MelSpectrogram).
//
// Five synthetic cases exercise the distinct code paths:
//   1. empty_positions — positions.empty() early return
//   2. short_audio     — audioLen < kNfft short-circuit in detectOnsets
//   3. silence         — detectOnsets returns empty (no peaks clear threshold)
//   4. click_snap      — click train at 22050 Hz; positions shifted by ±10 ms
//                        snap back to transients
//   5. click_out_of_window — positions >30 ms from any click stay unchanged

#include "analysis/OnsetRefinement.h"
#include "OnsetRefinementReabeat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}

// Deterministic click train: impulses with short exponential decay at given
// sample offsets. Designed to produce clear spectral-flux peaks that clear
// the adaptive threshold.
std::vector<float> makeClickTrain(int sampleRate,
                                   float durationSec,
                                   const std::vector<float>& clickTimes)
{
    int n = static_cast<int>(sampleRate * durationSec);
    std::vector<float> audio(n, 0.0f);
    const int decaySamples = 256; // ~12 ms at 22050
    for (float t : clickTimes)
    {
        int idx = static_cast<int>(t * sampleRate);
        if (idx < 0 || idx >= n) continue;
        for (int i = 0; i < decaySamples && idx + i < n; ++i)
        {
            float env = std::exp(-5.0f * static_cast<float>(i) / decaySamples);
            audio[idx + i] += env;
        }
    }
    return audio;
}

struct Case {
    const char* name;
    std::vector<float> audio;
    std::vector<float> positions;
    int sampleRate;
};

bool runCase(const Case& c)
{
    auto ours = reamix::OnsetRefinement::refine(c.audio, c.sampleRate, c.positions);
    auto ref  = OnsetRefinementReabeat::refine(c.audio, c.sampleRate, c.positions);

    double md = maxAbsDiff(ours, ref);
    std::printf("  case=%-24s  audio=%6zu pos=%zu -> ours=%zu ref=%zu  max_abs_diff=%.17g\n",
                c.name, c.audio.size(), c.positions.size(),
                ours.size(), ref.size(), md);

    if (ours.size() != ref.size())
    {
        std::fprintf(stderr, "FAIL[%s]: size mismatch (ours=%zu, ref=%zu)\n",
                     c.name, ours.size(), ref.size());
        return false;
    }
    if (md != 0.0)
    {
        std::fprintf(stderr, "FAIL[%s]: bitwise divergence (max_abs_diff=%.17g)\n",
                     c.name, md);
        return false;
    }
    return true;
}

} // namespace

int main()
{
    std::printf("OnsetRefinement parity: reamix vs REABeat reference\n");

    const int sr = 22050;
    bool allPass = true;

    // 1. empty_positions — positions.empty() early return.
    {
        Case c{"empty_positions",
               makeClickTrain(sr, 1.0f, {0.25f, 0.5f, 0.75f}),
               {},
               sr};
        allPass &= runCase(c);
    }

    // 2. short_audio — audioLen < kNfft (1024). 512 samples forces the
    //    short-circuit inside detectOnsets; refine falls back to input positions.
    {
        Case c{"short_audio",
               std::vector<float>(512, 0.0f),
               {0.01f, 0.015f},
               sr};
        allPass &= runCase(c);
    }

    // 3. silence — 2 s of zeros → flux all-zero → no peaks → empty onsets
    //    → refine returns positions unchanged.
    {
        Case c{"silence",
               std::vector<float>(sr * 2, 0.0f),
               {0.25f, 0.5f, 0.75f, 1.0f},
               sr};
        allPass &= runCase(c);
    }

    // 4. click_snap — clicks at 0.25 / 0.5 / 0.75 / 1.0 s; positions shifted
    //    by +8 ms should snap to the nearest click (within 30 ms window).
    {
        std::vector<float> clicks{0.25f, 0.50f, 0.75f, 1.00f};
        std::vector<float> positions;
        for (float t : clicks) positions.push_back(t + 0.008f);
        Case c{"click_snap",
               makeClickTrain(sr, 1.5f, clicks),
               positions,
               sr};
        allPass &= runCase(c);
    }

    // 5. click_out_of_window — positions >50 ms from any click stay unchanged
    //    (refined[i] == positions[i]) while detectOnsets still fires.
    {
        std::vector<float> clicks{0.25f, 0.75f};
        std::vector<float> positions{0.50f, 1.00f}; // 250 ms from nearest click
        Case c{"click_out_of_window",
               makeClickTrain(sr, 1.5f, clicks),
               positions,
               sr};
        allPass &= runCase(c);
    }

    if (!allPass)
    {
        std::fprintf(stderr, "FAIL: one or more cases diverged\n");
        return 1;
    }

    std::printf("PASS: all cases byte-identical to REABeat reference\n");
    return 0;
}
