// Parity test: reamix::DownbeatCleaner must produce byte-identical output
// to REABeat's reference DownbeatCleaner for the same input.
//
// Success criterion: max |Δ| == 0.0 on the returned downbeat-time vector,
// and sizes must match exactly. Per-element compare is also bit-equality
// via uint32_t memcpy (catches -0.0 vs +0.0 / NaN payload edge cases that
// arithmetic compare might miss, though DownbeatCleaner has no paths that
// emit those).
//
// DownbeatCleaner is pure stdlib: <algorithm>, <cmath>. Both ports consume
// identical inputs through identical arithmetic, so bitwise parity is
// structurally guaranteed. The cases below exist to cover branches.
//
// Cases (1:1 branch coverage of DownbeatCleaner::clean):
//   1) too_short             — rawDownbeats.size() < 2 → short-circuit copy
//   2) clean_4_4             — all gaps ≈ refInterval, no skip / no fill
//   3) too_close_skip        — gap < 0.6×ref → erroneous extra skipped
//   4) too_far_fill_one      — gap ≈ 2×ref → round(2)-1 = 1 missing filled
//   5) too_far_fill_two      — gap ≈ 3×ref → round(3)-1 = 2 missing filled
//   6) median_far_from_exp   — |median - expectedBar|/expectedBar > 0.15
//                              → ref = medianInterval branch; chosen so the
//                              outcome differs from what expectedBar would
//                              have produced (proves the branch fired)
//   7) waltz_3_4             — 3/4 at 150 BPM, expectedBar = 1.2 s, clean
//                              → ref = expectedBar branch exercised

#include "analysis/DownbeatCleaner.h"
#include "DownbeatCleanerReabeat.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
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

bool bitEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        uint32_t ua = 0, ub = 0;
        std::memcpy(&ua, &a[i], sizeof(uint32_t));
        std::memcpy(&ub, &b[i], sizeof(uint32_t));
        if (ua != ub) return false;
    }
    return true;
}

struct Case
{
    std::string name;
    std::vector<float> rawDownbeats;
    float tempo = 120.0f;
    int timeSigNum = 4;
};

// --- Cases --------------------------------------------------------------

Case tooShort()
{
    Case c;
    c.name = "too_short";
    // size() < 2 short-circuit: returns copy of input.
    c.rawDownbeats = {5.0f};
    c.tempo = 120.0f;
    c.timeSigNum = 4;
    return c;
}

Case clean4_4()
{
    Case c;
    c.name = "clean_4_4";
    // 5 downbeats spaced exactly 2.0 s at 120 BPM 4/4.
    // expectedBar = 60*4/120 = 2.0. intervals = {2,2,2,2}, median = 2.0.
    // |2 - 2|/2 = 0 < 0.15 → refInterval = expectedBar = 2.0.
    // All ratios == 1.0 → every downbeat kept, no skip, no fill.
    c.rawDownbeats = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    c.tempo = 120.0f;
    c.timeSigNum = 4;
    return c;
}

Case tooCloseSkip()
{
    Case c;
    c.name = "too_close_skip";
    // Bogus extra at 0.4 s between 0 and 2.0. Others clean at 2 s spacing.
    // intervals = {0.4, 1.6, 2.0, 2.0}, sorted = {0.4, 1.6, 2.0, 2.0},
    // median = sorted[2] = 2.0. ref = expectedBar = 2.0.
    //   gap 0→0.4 = 0.4, ratio 0.2 < 0.6 → skip 0.4.
    //   gap 0→2.0 = 2.0, ratio 1.0 → keep.
    //   gap 2.0→4.0 = 2.0 → keep.
    //   gap 4.0→6.0 = 2.0 → keep.
    // Expected result: {0, 2.0, 4.0, 6.0}.
    c.rawDownbeats = {0.0f, 0.4f, 2.0f, 4.0f, 6.0f};
    c.tempo = 120.0f;
    c.timeSigNum = 4;
    return c;
}

Case tooFarFillOne()
{
    Case c;
    c.name = "too_far_fill_one";
    // Gap of 2× expected at position 2.0→6.0 (ratio 2.0 > 1.35).
    // expectedBar = 2.0. intervals = {2, 4, 2}, sorted = {2, 2, 4},
    // median = sorted[1] = 2.0. ref = expectedBar = 2.0.
    //   gap 0→2.0 = 2.0, ratio 1.0 → keep.
    //   gap 2.0→6.0 = 4.0, ratio 2.0 → nMissing = round(2) - 1 = 1.
    //     insert 2.0 + 2.0*1 = 4.0. push 6.0.
    //   gap 6.0→8.0 = 2.0 → keep.
    // Expected: {0, 2.0, 4.0, 6.0, 8.0}.
    c.rawDownbeats = {0.0f, 2.0f, 6.0f, 8.0f};
    c.tempo = 120.0f;
    c.timeSigNum = 4;
    return c;
}

Case tooFarFillTwo()
{
    Case c;
    c.name = "too_far_fill_two";
    // Gap of 3× expected at position 2.0→8.0 (ratio 3.0 > 1.35).
    // intervals = {2, 6, 2}, median = sorted[1] = 2.0. ref = 2.0.
    //   gap 0→2.0 = 2.0 → keep.
    //   gap 2.0→8.0 = 6.0, ratio 3.0 → nMissing = round(3) - 1 = 2.
    //     insert 2.0 + 2.0*1 = 4.0, 2.0 + 2.0*2 = 6.0. push 8.0.
    //   gap 8.0→10.0 = 2.0 → keep.
    // Expected: {0, 2.0, 4.0, 6.0, 8.0, 10.0}.
    c.rawDownbeats = {0.0f, 2.0f, 8.0f, 10.0f};
    c.tempo = 120.0f;
    c.timeSigNum = 4;
    return c;
}

Case medianFarFromExpected()
{
    Case c;
    c.name = "median_far_from_expected";
    // Measured spacing 2.8 s, expectedBar = 2.0 (120 BPM 4/4).
    // intervals = {2.8, 2.8, 2.8}, sorted same, median = sorted[1] = 2.8.
    // |2.8 - 2.0| / 2.0 = 0.4 > 0.15 → branch fires: refInterval = medianInterval = 2.8.
    //   All gaps 2.8, ratio = 2.8/2.8 = 1.0 → every downbeat kept, no fill.
    // Expected: {0, 2.8, 5.6, 8.4}.
    //
    // If ref had (wrongly) stayed at expectedBar = 2.0, ratio = 2.8/2.0 = 1.4 > 1.35 →
    // algorithm would fill one extra at every gap, producing 7 elements instead of 4.
    // A size mismatch vs reference would immediately expose a broken branch on our side.
    c.rawDownbeats = {0.0f, 2.8f, 5.6f, 8.4f};
    c.tempo = 120.0f;
    c.timeSigNum = 4;
    return c;
}

Case waltz3_4()
{
    Case c;
    c.name = "waltz_3_4";
    // 3/4 at 150 BPM → expectedBar = 60*3/150 = 1.2 s.
    // 5 clean downbeats spaced 1.2 s.
    // intervals = {1.2, 1.2, 1.2, 1.2}, median = sorted[2] = 1.2.
    // |1.2 - 1.2| / 1.2 = 0 < 0.15 → ref = expectedBar = 1.2.
    // All gaps ratio = 1.0 → kept, no skip, no fill.
    // Expected: {0, 1.2, 2.4, 3.6, 4.8}.
    c.rawDownbeats = {0.0f, 1.2f, 2.4f, 3.6f, 4.8f};
    c.tempo = 150.0f;
    c.timeSigNum = 3;
    return c;
}

// --- Runner -------------------------------------------------------------

int runCase(const Case& c)
{
    auto ours = reamix::DownbeatCleaner::clean(c.rawDownbeats, c.tempo, c.timeSigNum);
    auto ref  = DownbeatCleanerReabeat::clean(c.rawDownbeats, c.tempo, c.timeSigNum);

    double d = maxAbsDiff(ours, ref);
    bool bit = bitEqual(ours, ref);
    std::printf("[%s] in=%zu ours=%zu ref=%zu | max_abs_diff=%.17g | bit_equal=%d\n",
                c.name.c_str(), c.rawDownbeats.size(), ours.size(), ref.size(),
                d, bit ? 1 : 0);

    if (ours.size() != ref.size() || d != 0.0 || !bit)
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
    int total = 0;
    int passed = 0;

    auto runAll = [&](const Case& c) {
        ++total;
        int r = runCase(c);
        if (r == 0) ++passed;
        rc |= r;
    };

    runAll(tooShort());
    runAll(clean4_4());
    runAll(tooCloseSkip());
    runAll(tooFarFillOne());
    runAll(tooFarFillTwo());
    runAll(medianFarFromExpected());
    runAll(waltz3_4());

    if (rc != 0)
    {
        std::fprintf(stderr, "FAIL: %d/%d cases diverged from REABeat reference\n",
                     total - passed, total);
        return 1;
    }

    std::printf("PASS: %d/%d cases bit-identical to REABeat reference\n", passed, total);
    return 0;
}
