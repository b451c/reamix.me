// test_region_dp_v2 — sesja 116 (ADR-115 E8, DEV-090) self-validation.
//
// Validates the v2 Region path search in RegionOptimizer (quality band +
// repetition-penalty DP with cooldown = one measured bar) on a synthetic
// 40-beat region (10 bars of 4) with hand-placed backward pairs:
//
//   D = (11 ->  8)  cost 0.28  1-bar loop, best pair but i - j = 3 < bar
//                              cooldown -> must never be used
//   A = (23 ->  8)  cost 0.30  4-bar loop
//   B = (31 -> 16)  cost 0.32  4-bar loop, one bar away from A on both axes
//   C = (31 ->  8)  cost 0.45  6-bar loop, outside the 0.08 band -> unused
//
// Extending 40 -> 88 beats (+48 = three 4-bar loops). Invariants:
//   1. v2 path hits the target within the tolerance and every jump is a
//      band pair (never C, never D);
//   2. the sequential run after every jump is at least one bar (cooldown);
//   3. with A and B equally long the DP alternates instead of repeating A
//      three times (repetition penalty 2 x 0.15 > the 0.02 cost difference
//      + tax difference of using B once);
//   4. blocking A (k-best augmentation) removes it from the path;
//   5. v2_scoring = false on the same inputs leaves the sesja-94 synthesizer
//      untouched (one (i, j) x N).
//
// Per ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`:
// hand-computed invariants, no Python ground truth.

#include "remix/RegionOptimizer.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace reamix::remix;

namespace {

struct Fixture {
    std::vector<double> rW;
    std::vector<double> beat_times;
    std::map<std::pair<int, int>, TransitionCandidate> region_candidates;
    int    n_region      = 40;
    int    n_total       = 41;
    double avg_beat      = 0.5;
    double region_start  = 0.0;
    double region_end    = 20.0;
    double tolerance_sec = 2.0;   // 4 beats
};

constexpr std::pair<int, int> kA {23, 8};
constexpr std::pair<int, int> kB {31, 16};
constexpr std::pair<int, int> kC {31, 8};
constexpr std::pair<int, int> kD {11, 8};

Fixture buildFixture(bool with_b)
{
    Fixture f;
    const int n = f.n_region;
    f.rW.assign(static_cast<std::size_t>(n) * n, INF);
    for (int i = 0; i + 1 < n; ++i) f.rW[(std::size_t) i * n + (i + 1)] = 0.02;
    f.rW[(std::size_t) kD.first * n + kD.second] = 0.28;
    f.rW[(std::size_t) kA.first * n + kA.second] = 0.30;
    if (with_b) f.rW[(std::size_t) kB.first * n + kB.second] = 0.32;
    f.rW[(std::size_t) kC.first * n + kC.second] = 0.45;
    f.beat_times.resize((std::size_t) f.n_total);
    for (int i = 0; i < f.n_total; ++i) f.beat_times[(std::size_t) i] = i * f.avg_beat;
    return f;
}

RemixPath run(const Fixture& fix, bool v2, double target,
              const std::set<std::pair<int, int>>* blocked = nullptr)
{
    RegionOptimizerInputs roin{};
    roin.n_beats                = fix.n_total;
    roin.beat_times             = fix.beat_times.data();
    roin.avg_beat_duration      = fix.avg_beat;
    roin.duration_tolerance_sec = fix.tolerance_sec;
    roin.candidates             = nullptr;
    roin.region_beta            = true;    // production Region flag
    roin.entry_beat_override    = 0;
    roin.exit_beat_override     = fix.n_region;
    roin.v2_scoring             = v2;
    roin.bar_beats              = 4;
    RegionOptimizer ropt(roin);
    return ropt.remix(target, fix.region_start, fix.region_end,
                      fix.rW.data(), fix.n_region, &fix.region_candidates, blocked);
}

int fails = 0;
void check(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (! ok) ++fails;
}

// Sequential run length after each jump (beats until the next jump or end).
bool cooldownRespected(const RemixPath& p, int bar)
{
    const auto& b = p.beat_indices;
    for (std::size_t k = 0; k + 1 < b.size(); ++k) {
        if (b[k + 1] == b[k] + 1) continue;
        int run = 0;
        for (std::size_t m = k + 1; m + 1 < b.size() && b[m + 1] == b[m] + 1; ++m) ++run;
        if (run < bar && k + 1 + (std::size_t) run + 1 < b.size()) return false;
    }
    return true;
}

} // namespace

int main()
{
    const double target = 44.0;   // 88 beats
    {
        Fixture fix = buildFixture(/*with_b=*/true);
        RemixPath p = run(fix, /*v2=*/true, target);
        check(! p.beat_indices.empty(), "v2: path produced");
        const int len = (int) p.beat_indices.size();
        check(std::abs(len - 88) <= 4, "v2: length within tolerance of the target");
        bool only_band = true, used_a = false, used_b = false;
        for (const auto& t : p.transitions) {
            if (t == kC || t == kD) only_band = false;
            if (t == kA) used_a = true;
            if (t == kB) used_b = true;
        }
        check(only_band, "v2: no jump outside the quality band (C) and no 1-bar loop (D)");
        check(cooldownRespected(p, 4), "v2: at least one bar of sequential playback after every jump");
        check(used_a && used_b, "v2: A and B alternate instead of A x 3 (repetition penalty)");
        bool consecutive_repeat = false;
        for (std::size_t k = 0; k + 1 < p.transitions.size(); ++k)
            if (p.transitions[k] == p.transitions[k + 1]) consecutive_repeat = true;
        check(! consecutive_repeat, "v2: no immediately repeated pair when an equal one exists");

        std::set<std::pair<int, int>> blocked { kA };
        RemixPath q = run(fix, true, target, &blocked);
        bool has_a = false;
        for (const auto& t : q.transitions) if (t == kA) has_a = true;
        check(! q.beat_indices.empty() && ! has_a, "v2: blocked pair A never used (k-best augmentation)");
    }
    {
        Fixture fix = buildFixture(/*with_b=*/false);
        RemixPath p = run(fix, true, target);
        int n_a = 0;
        for (const auto& t : p.transitions) if (t == kA) ++n_a;
        check(n_a == 3 && (int) p.transitions.size() == 3,
              "v2: with A alone the loop is A x 3 (penalty never adds a cut)");
    }
    {
        Fixture fix = buildFixture(true);
        RemixPath p = run(fix, /*v2=*/false, target);
        bool single_pair = ! p.transitions.empty();
        for (const auto& t : p.transitions) if (t != p.transitions.front()) single_pair = false;
        check(single_pair, "legacy: v2_scoring=false keeps the one-loop x N synthesizer");
    }
    std::printf("%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails == 0 ? 0 : 1;
}
