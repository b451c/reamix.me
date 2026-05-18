// Parity test: reamix::BeatInterpolator must produce byte-identical output
// to REABeat's reference BeatInterpolator for the same input.
//
// Success criterion: max |Δ| == 0.0 on the returned beat-time vector, and
// sizes must match exactly.
//
// Synthetic cases exercise each code path in BeatInterpolator::interpolate:
//   1) short_input    — beats.size() < 3 short-circuit
//   2) pass_through   — all gaps ≤ 1.35× median (no insertions)
//   3) even_fallback  — large gap, no beatLogits → evenly interpolated fill
//   4) hint_based     — large gap with sub-threshold logit peaks at expected
//                       positions → hint-based fill
//   5) hints_insufficient — large gap with logits present but hints not at
//                           expected positions → falls back to even spacing
//   6) degenerate_median  — duplicate beats produce median == 0 short-circuit
//
// BeatInterpolator is pure stdlib: no transcendentals, no randomness, no
// threading. Both ports consume the same inputs through the same arithmetic,
// so bitwise parity is structurally guaranteed. The cases below exist to
// cover branches, not to stress numerics.

#include "analysis/BeatInterpolator.h"
#include "BeatInterpolatorReabeat.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
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

struct Case
{
    std::string name;
    std::vector<float> beats;
    std::vector<float> beatLogits;  // may be empty
    float fps = 50.0f;
};

// --- Cases --------------------------------------------------------------

Case shortInput()
{
    Case c;
    c.name = "short_input";
    // size() < 3 → returns beats unchanged.
    c.beats = {1.0f, 2.0f};
    return c;
}

Case passThrough()
{
    Case c;
    c.name = "pass_through";
    // All gaps == 0.5s, median == 0.5s, ratio == 1.0 for every gap.
    // Every beat kept as-is; no interpolation.
    c.beats = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    return c;
}

Case evenFallback()
{
    Case c;
    c.name = "even_fallback";
    // Median = 0.5s. One missing region: gap 1.0→2.5 == 1.5s, ratio == 3.0,
    // nExpected == 3. No logits → fallback fills at 1.5, 2.0.
    c.beats = {0.0f, 0.5f, 1.0f, 2.5f, 3.0f, 3.5f};
    return c;
}

Case hintBased()
{
    Case c;
    c.name = "hint_based";
    c.beats = {0.0f, 0.5f, 1.0f, 2.5f, 3.0f, 3.5f};
    // 50 fps → 4 s of audio covered by ~200 frames.
    // Gap is beats[2]=1.0 → beats[3]=2.5.
    //   frameStart = int(1.0 * 50) + 1 = 51
    //   frameEnd   = int(2.5 * 50)     = 125
    //   nExpected  = round(1.5/0.5)    = 3
    // Expected insertion frames at rel 1/3, 2/3 → frames 75, 100 → times 1.5, 2.0.
    // Seed sub-threshold peaks at those frames; everything else far below
    // kSubThresholdMin so peak-picking around 75 and 100 is unambiguous.
    c.beatLogits.assign(200, -5.0f);
    // Peak at frame 75, with lower neighbours inside the (-2, 0) window.
    c.beatLogits[72] = -1.6f;
    c.beatLogits[73] = -1.4f;
    c.beatLogits[74] = -1.2f;
    c.beatLogits[75] = -0.3f;   // peak
    c.beatLogits[76] = -1.2f;
    c.beatLogits[77] = -1.4f;
    c.beatLogits[78] = -1.6f;
    // Peak at frame 100.
    c.beatLogits[97]  = -1.6f;
    c.beatLogits[98]  = -1.4f;
    c.beatLogits[99]  = -1.2f;
    c.beatLogits[100] = -0.4f;  // peak
    c.beatLogits[101] = -1.2f;
    c.beatLogits[102] = -1.4f;
    c.beatLogits[103] = -1.6f;
    return c;
}

Case hintsInsufficient()
{
    Case c;
    c.name = "hints_insufficient";
    c.beats = {0.0f, 0.5f, 1.0f, 2.5f, 3.0f, 3.5f};
    // Same gap as hint_based but the only in-window peak sits far from the
    // expected 1/3, 2/3 positions (frame 60 ≈ rel 0.133) so it never matches
    // → used < nExpected-1 → fallback to even spacing.
    c.beatLogits.assign(200, -5.0f);
    c.beatLogits[58] = -1.6f;
    c.beatLogits[59] = -1.4f;
    c.beatLogits[60] = -0.5f;  // single off-position peak
    c.beatLogits[61] = -1.4f;
    c.beatLogits[62] = -1.6f;
    return c;
}

Case degenerateMedian()
{
    Case c;
    c.name = "degenerate_median";
    // Duplicate timestamps → intervals {0, 0, 1.0, 0.5}, sorted {0, 0, 0.5, 1.0},
    // median = sorted[2] = 0.5. (Even-length median picks upper-middle.)
    // Actually want to trip median <= 0 branch: make it {1.0, 1.0, 1.0, 1.0, 2.0}
    // → intervals {0, 0, 0, 1.0}, sorted {0, 0, 0, 1.0}, median=sorted[2]=0 → short-circuit.
    c.beats = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f};
    return c;
}

// --- Runner -------------------------------------------------------------

int runCase(const Case& c)
{
    auto ours = reamix::BeatInterpolator::interpolate(c.beats, c.beatLogits, c.fps);
    auto ref  = BeatInterpolatorReabeat::interpolate(c.beats, c.beatLogits, c.fps);

    double d = maxAbsDiff(ours, ref);
    std::printf("[%s] in=%zu ours=%zu ref=%zu | max_abs_diff=%.17g\n",
                c.name.c_str(), c.beats.size(), ours.size(), ref.size(), d);

    if (ours.size() != ref.size() || d != 0.0)
    {
        std::fprintf(stderr, "FAIL [%s]\n", c.name.c_str());
        return 1;
    }
    return 0;
}

} // namespace

int main()
{
    int rc = 0;
    rc |= runCase(shortInput());
    rc |= runCase(passThrough());
    rc |= runCase(evenFallback());
    rc |= runCase(hintBased());
    rc |= runCase(hintsInsufficient());
    rc |= runCase(degenerateMedian());

    if (rc != 0)
    {
        std::fprintf(stderr, "FAIL: at least one case diverged from REABeat reference\n");
        return 1;
    }

    std::printf("PASS: byte-identical to REABeat reference on all 6 cases\n");
    return 0;
}
