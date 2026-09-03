// test_grid_consistency — sesja 124 (DEV-088) self-validation of the
// analysis-stage grid: phase-consistent gap fill, holes kept, downbeats on
// the grid, measured bar, grid tempo. No Python / REABeat reference
// (C++-canonical per ADR-065).
//
// Fixture: 120 BPM (period 0.5 s), 121 beats 0..60 s, downbeats every 4.
// Asserts:
//   1. Clean grid: beats unchanged, bar 4, bpm 120, no fills, no drops.
//   2. A 6-period gap (5 beats removed) is filled back within 1e-9, with
//      the downbeat inside it kept (it lies on a filled beat).
//   3. A gap of 6.4 periods (the right side shifted by 0.4 period) stays a
//      hole; the downbeats adjacent to it are dropped.
//   4. A 30-period gap stays a hole even though it is phase-consistent.
//   5. An off-grid downbeat (0.4 period) is dropped; bpm follows the grid
//      (a 100 BPM grid reports 100, never an octave away).

#include "analysis/GridConsistency.h"

#include <cmath>
#include <cstdio>
#include <vector>

using reamix::analysis::makeConsistentGrid;

namespace {

constexpr double kPeriod = 0.5;

std::vector<double> cleanBeats(int n = 121)
{
    std::vector<double> b;
    for (int i = 0; i < n; ++i) b.push_back(i * kPeriod);
    return b;
}

std::vector<double> downbeatsEvery4(int n = 121)
{
    std::vector<double> d;
    for (int i = 0; i < n; i += 4) d.push_back(i * kPeriod);
    return d;
}

bool test_clean()
{
    const auto g = makeConsistentGrid(cleanBeats(), downbeatsEvery4(), 4);
    const bool ok = g.beats.size() == 121 && g.bar_beats == 4 && std::fabs(g.bpm - 120.0) < 1e-9
                 && g.n_filled_gaps == 0 && g.n_dropped_downbeats == 0 && g.downbeats.size() == 31
                 && g.beatIsDownbeat[0] && g.beatIsDownbeat[4] && !g.beatIsDownbeat[1];
    std::fprintf(stderr, "[%s] clean grid: %zu beats, bar %d, %.1f BPM, %zu downbeats\n",
                 ok ? "PASS" : "FAIL", g.beats.size(), g.bar_beats, g.bpm, g.downbeats.size());
    return ok;
}

bool test_consistent_gap_filled()
{
    auto b = cleanBeats();
    b.erase(b.begin() + 41, b.begin() + 46);   // beats 41..45 removed: gap 40 -> 46 = 6 periods
    const auto g = makeConsistentGrid(b, downbeatsEvery4(), 4);
    bool ok = g.n_filled_gaps == 1 && g.n_filled_beats == 5 && g.beats.size() == 121;
    for (std::size_t i = 0; ok && i < g.beats.size(); ++i)
        if (std::fabs(g.beats[i] - i * kPeriod) > 1e-9) ok = false;
    ok = ok && g.beatIsDownbeat[44] && g.n_dropped_downbeats == 0 && g.n_holes == 0;
    std::fprintf(stderr, "[%s] 6-period gap filled: %d beats back, downbeat 44 kept, %d dropped\n",
                 ok ? "PASS" : "FAIL", g.n_filled_beats, g.n_dropped_downbeats);
    return ok;
}

bool test_inconsistent_gap_kept()
{
    auto b = cleanBeats();
    b.erase(b.begin() + 41, b.begin() + 46);
    for (std::size_t i = 41; i < b.size(); ++i) b[i] += 0.4 * kPeriod;   // right side shifted
    auto d = downbeatsEvery4();
    for (auto& t : d) if (t > 40 * kPeriod) t += 0.4 * kPeriod;
    const auto g = makeConsistentGrid(b, d, 4);
    const bool ok = g.n_filled_gaps == 0 && g.n_holes == 1 && g.beats.size() == b.size()
                 && g.n_dropped_downbeats >= 1 && g.bar_beats == 4;
    std::fprintf(stderr, "[%s] 6.4-period gap kept as a hole (%d holes, %d downbeats dropped)\n",
                 ok ? "PASS" : "FAIL", g.n_holes, g.n_dropped_downbeats);
    return ok;
}

bool test_long_gap_kept()
{
    auto b = cleanBeats();
    b.erase(b.begin() + 41, b.begin() + 70);   // 29 beats removed: gap = 30 periods
    const auto g = makeConsistentGrid(b, downbeatsEvery4(), 4);
    const bool ok = g.n_filled_gaps == 0 && g.n_holes == 1 && g.beats.size() == b.size();
    std::fprintf(stderr, "[%s] 30-period gap kept as a hole\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool test_offgrid_downbeat_and_bpm()
{
    std::vector<double> b;
    for (int i = 0; i < 121; ++i) b.push_back(i * 0.6);   // 100 BPM
    std::vector<double> d;
    for (int i = 0; i < 121; i += 4) d.push_back(i * 0.6);
    d[3] += 0.4 * 0.6;   // one downbeat off the grid
    const auto g = makeConsistentGrid(b, d, 4);
    const bool ok = std::fabs(g.bpm - 100.0) < 1e-9 && g.n_dropped_downbeats == 1
                 && g.downbeats.size() == d.size() - 1 && g.bar_beats == 4;
    std::fprintf(stderr, "[%s] off-grid downbeat dropped (%d), bpm %.1f from the grid\n",
                 ok ? "PASS" : "FAIL", g.n_dropped_downbeats, g.bpm);
    return ok;
}

} // namespace

int main()
{
    bool ok = true;
    ok = test_clean()                   && ok;
    ok = test_consistent_gap_filled()   && ok;
    ok = test_inconsistent_gap_kept()   && ok;
    ok = test_long_gap_kept()           && ok;
    ok = test_offgrid_downbeat_and_bpm() && ok;
    std::fprintf(stderr, ok ? "test_grid_consistency: ALL PASS\n" : "test_grid_consistency: FAIL\n");
    return ok ? 0 : 1;
}
