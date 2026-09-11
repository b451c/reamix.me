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
//   3. A gap of 6.4 periods (the right side shifted by 0.4 period) is not
//      filled; it gets a phase-free lattice (sesja 135) and the downbeats
//      adjacent to it are dropped.
//   4. A 30-period gap is not filled (beyond max_fill_periods); its lattice
//      reproduces the removed beats, flagged synthetic.
//   6. Un-beated head / tail lattice when the file length is known.
//   5. An off-grid downbeat (0.4 period) is dropped; bpm follows the grid
//      (a 100 BPM grid reports 100, never an octave away).

#include "analysis/GridConsistency.h"

#include <algorithm>
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
    // Sesja 135 (DEV-122): the phase-inconsistent gap is not FILLED and, at
    // 5 lattice beats (6.4 - k >= 0.5 -> k <= 5), too short for a lattice
    // zone (min 8): it stays a hole, the adjacent downbeats are dropped.
    bool ok = g.n_filled_gaps == 0 && g.n_holes == 1 && g.n_lattice_zones == 0 && g.n_lattice_beats == 0
           && g.beats.size() == b.size() && g.n_dropped_downbeats >= 1 && g.bar_beats == 4
           && std::none_of(g.beatIsSynthetic.begin(), g.beatIsSynthetic.end(), [](bool x) { return x; });
    // With min_lattice_beats 4 the same gap gets its 5-beat lattice.
    const auto g4 = makeConsistentGrid(b, d, 4, 0.0, /*min_lattice_beats*/ 4);
    ok = ok && g4.n_lattice_zones == 1 && g4.n_lattice_beats == 5 && g4.beats.size() == b.size() + 5;
    for (int k = 1; ok && k <= 5; ++k) {
        const std::size_t idx = 40 + static_cast<std::size_t>(k);
        ok = std::fabs(g4.beats[idx] - (40 * kPeriod + k * kPeriod)) < 1e-9 && g4.beatIsSynthetic[idx] && !g4.beatIsDownbeat[idx];
    }
    ok = ok && !g4.beatIsSynthetic[40] && !g4.beatIsSynthetic[46] && !g4.beatIsDownbeat[40]   // 40: real downbeat next to the zone -> dropped
            && g4.beatIsDownbeat[36];
    std::fprintf(stderr, "[%s] 6.4-period gap: hole at min 8 (%d lattice beats), lattice of %d at min 4, %d downbeats dropped\n",
                 ok ? "PASS" : "FAIL", g.n_lattice_beats, g4.n_lattice_beats, g4.n_dropped_downbeats);
    return ok;
}

bool test_long_gap_kept()
{
    auto b = cleanBeats();
    b.erase(b.begin() + 41, b.begin() + 70);   // 29 beats removed: gap = 30 periods
    const auto g = makeConsistentGrid(b, downbeatsEvery4(), 4);
    // Sesja 135 (DEV-122): a 30-period gap is beyond max_fill_periods, so it
    // is a lattice zone: 29 beats from the left edge at the period - on an
    // exactly consistent gap they coincide with the removed beats - all
    // synthetic; the detector downbeats inside (44 .. 68) snap onto lattice
    // beats and are dropped, so is the real downbeat 40 next to the zone.
    bool ok = g.n_filled_gaps == 0 && g.n_holes == 1 && g.n_lattice_zones == 1 && g.n_lattice_beats == 29
           && g.beats.size() == 121 && g.n_dropped_downbeats == 8;
    for (std::size_t i = 0; ok && i < g.beats.size(); ++i)
        ok = std::fabs(g.beats[i] - i * kPeriod) < 1e-9 && g.beatIsSynthetic[i] == (i >= 41 && i <= 69);
    ok = ok && !g.beatIsDownbeat[40] && !g.beatIsDownbeat[44] && !g.beatIsDownbeat[68] && g.beatIsDownbeat[72] && g.beatIsDownbeat[36];
    std::fprintf(stderr, "[%s] 30-period gap: lattice of %d beats, %d downbeats dropped\n",
                 ok ? "PASS" : "FAIL", g.n_lattice_beats, g.n_dropped_downbeats);
    return ok;
}

// Sesja 135 (DEV-122): un-beated head and tail get a lattice when the file
// length is known; the first / last real downbeats next to a zone are dropped.
bool test_head_tail_lattice()
{
    std::vector<double> b, d;
    for (int i = 40; i <= 120; ++i) b.push_back(i * kPeriod);      // real beats 20 .. 60 s
    for (int i = 40; i <= 120; i += 4) d.push_back(i * kPeriod);
    const auto g = makeConsistentGrid(b, d, 4, /*duration*/ 70.0);
    // head: 19.5, 19.0, .., 0.5 (39 beats); tail: 60.5 .. 69.5 (19 beats: t <= 69.75)
    bool ok = g.n_lattice_zones == 2 && g.n_lattice_beats == 58 && g.beats.size() == 81 + 58
           && std::fabs(g.beats.front() - 0.5) < 1e-9 && std::fabs(g.beats.back() - 69.5) < 1e-9
           && std::fabs(g.bpm - 120.0) < 1e-9 && g.bar_beats == 4;
    for (std::size_t i = 0; ok && i < g.beats.size(); ++i)
        ok = std::fabs(g.beats[i] - (i + 1) * kPeriod) < 1e-9 && g.beatIsSynthetic[i] == (i < 39 || i >= 120);
    // real downbeats at 20 s (idx 39, next to the head zone) and 60 s (idx 119, next to the tail zone) dropped
    ok = ok && !g.beatIsDownbeat[39] && g.beatIsDownbeat[43] && g.beatIsDownbeat[115] && !g.beatIsDownbeat[119]
            && g.n_dropped_downbeats == 2 && g.downbeats.size() == d.size() - 2;
    // no lattice without a known duration: only the head zone
    const auto h = makeConsistentGrid(b, d, 4);
    ok = ok && h.n_lattice_zones == 1 && h.beats.size() == 81 + 39;
    std::fprintf(stderr, "[%s] head + tail lattice: %d beats in %d zones, %zu beats, %d downbeats dropped\n",
                 ok ? "PASS" : "FAIL", g.n_lattice_beats, g.n_lattice_zones, g.beats.size(), g.n_dropped_downbeats);
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
    ok = test_head_tail_lattice()       && ok;
    std::fprintf(stderr, ok ? "test_grid_consistency: ALL PASS\n" : "test_grid_consistency: FAIL\n");
    return ok ? 0 : 1;
}
