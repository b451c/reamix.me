// Parity test: reamix::Postprocessor must produce byte-identical output
// to REABeat's reference Postprocessor for the same input.
//
// Success criterion (phase 1 spec): max |Δ| == 0.0 on all four output
// vectors (beatTimes, downbeatTimes, beatLogits, downbeatLogits), and
// sizes must match exactly.
//
// Four synthetic deterministic inputs exercise:
//   1) sparse peaks (including at boundaries)
//   2) dense/tied peaks (exercises dedup running-mean)
//   3) downbeats that must snap to beats + collapse duplicates
//   4) all-negative logits (empty output path)

#include "analysis/Postprocessor.h"
#include "PostprocessorReabeat.h"

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
    std::vector<float> beat;
    std::vector<float> downbeat;
};

Case sparsePeaks()
{
    // Length 100. Positive peaks at 0 (boundary), 25, 50, 75, 99 (boundary),
    // zeros elsewhere. Tiny negative dip at 80 to confirm the >0 threshold.
    Case c;
    c.name = "sparse_peaks";
    c.beat.assign(100, 0.0f);
    c.beat[0] = 2.0f;
    c.beat[25] = 3.0f;
    c.beat[50] = 1.5f;
    c.beat[75] = 4.0f;
    c.beat[99] = 2.5f;
    c.beat[80] = -1.0f;

    // Downbeats every 4th beat position.
    c.downbeat.assign(100, 0.0f);
    c.downbeat[0] = 1.0f;
    c.downbeat[50] = 2.0f;
    c.downbeat[99] = 0.75f;
    return c;
}

Case densePeaks()
{
    // Length 80. Several clusters of adjacent-equal positive values surrounded
    // by zeros, to exercise dedup running-mean across widths.
    Case c;
    c.name = "dense_peaks";
    c.beat.assign(80, 0.0f);
    // two adjacent equal peaks at 10,11 → merged to ~10.5 → round 11
    c.beat[10] = 2.0f;
    c.beat[11] = 2.0f;
    // three adjacent peaks at 30,31,32 → running mean → 31
    c.beat[30] = 3.0f;
    c.beat[31] = 3.0f;
    c.beat[32] = 3.0f;
    // isolated at 60
    c.beat[60] = 1.5f;

    c.downbeat.assign(80, 0.0f);
    c.downbeat[31] = 1.0f;
    c.downbeat[32] = 1.0f;
    return c;
}

Case downbeatSnap()
{
    // Length 150. Beats at 20, 40, 60, 80, 100, 120. Downbeats slightly
    // off-grid (19, 61) to force snap to nearest, plus two downbeats (79, 81)
    // that both snap to beat at 80 to exercise sort+unique collapse.
    Case c;
    c.name = "downbeat_snap";
    c.beat.assign(150, 0.0f);
    for (int f : {20, 40, 60, 80, 100, 120})
        c.beat[f] = 2.0f;

    c.downbeat.assign(150, 0.0f);
    c.downbeat[19] = 1.0f;
    c.downbeat[61] = 1.2f;
    c.downbeat[79] = 0.9f;
    c.downbeat[81] = 1.1f;
    return c;
}

Case allNegative()
{
    Case c;
    c.name = "all_negative";
    c.beat.assign(50, -1.0f);
    c.downbeat.assign(50, -1.0f);
    return c;
}

int runCase(const Case& c)
{
    reamix::Postprocessor ours(50.0f);
    PostprocessorReabeat ref(50.0f);

    auto r1 = ours.process(c.beat, c.downbeat);
    auto r2 = ref.process(c.beat, c.downbeat);

    double dBeat  = maxAbsDiff(r1.beatTimes,     r2.beatTimes);
    double dDown  = maxAbsDiff(r1.downbeatTimes, r2.downbeatTimes);
    double dBLog  = maxAbsDiff(r1.beatLogits,    r2.beatLogits);
    double dDLog  = maxAbsDiff(r1.downbeatLogits,r2.downbeatLogits);

    std::printf("[%s] beats ours=%zu ref=%zu | downbeats ours=%zu ref=%zu\n",
                c.name.c_str(),
                r1.beatTimes.size(),     r2.beatTimes.size(),
                r1.downbeatTimes.size(), r2.downbeatTimes.size());
    std::printf("  max_abs_diff beatTimes=%.17g downbeatTimes=%.17g beatLogits=%.17g downbeatLogits=%.17g\n",
                dBeat, dDown, dBLog, dDLog);

    bool sizesOk = (r1.beatTimes.size()     == r2.beatTimes.size())
                && (r1.downbeatTimes.size() == r2.downbeatTimes.size())
                && (r1.beatLogits.size()    == r2.beatLogits.size())
                && (r1.downbeatLogits.size()== r2.downbeatLogits.size());

    if (!sizesOk || dBeat != 0.0 || dDown != 0.0 || dBLog != 0.0 || dDLog != 0.0)
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
    rc |= runCase(sparsePeaks());
    rc |= runCase(densePeaks());
    rc |= runCase(downbeatSnap());
    rc |= runCase(allNegative());

    if (rc != 0)
    {
        std::fprintf(stderr, "FAIL: at least one case diverged from REABeat reference\n");
        return 1;
    }

    std::printf("PASS: byte-identical to REABeat reference on all 4 cases\n");
    return 0;
}
