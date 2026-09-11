// test_beat_grid - ADR-115 E5 (sesja 115) self-validation of the beat /
// downbeat grid clean-up. C++-canonical, hand-computed expectations.
#include "remix/BeatGrid.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace reamix::remix;

static int g_fail = 0;
static void expectTrue(const char* what, bool v)
{
    if (! v) { std::printf("FAIL %s\n", what); ++g_fail; }
    else       std::printf("ok   %s\n", what);
}
static void expectEq(const char* what, int got, int want)
{
    if (got != want) { std::printf("FAIL %s: got %d want %d\n", what, got, want); ++g_fail; }
    else               std::printf("ok   %s = %d\n", what, got);
}

int main()
{
    // 0.5 s grid, 40 beats, with a 3 s hole after beat 19 (beats 20.. shifted).
    std::vector<double> bt;
    for (int i = 0; i < 40; ++i) bt.push_back(0.5 * i + (i >= 20 ? 3.0 : 0.0));

    // Downbeats every 2 grid beats (half-tempo grid: detector said TS = 4),
    // plus one off-grid downbeat (0.3 beat away), one in the hole, and one
    // duplicate of an existing downbeat.
    std::vector<double> db;
    for (int i = 0; i < 40; i += 2) db.push_back(bt[static_cast<std::size_t>(i)]);
    db.push_back(bt[7] + 0.15);        // 0.3 beat off -> dropped
    db.push_back(bt[19] + 1.5);        // inside the hole -> no beat within tol -> dropped
    db.push_back(bt[10]);              // duplicate -> de-duplicated

    const auto g = cleanBeatGrid(bt.data(), 40, db.data(), (int) db.size(), /*TS hint*/ 4);
    expectTrue("period = 0.5 s", std::abs(g.period_sec - 0.5) < 1e-9);
    expectEq("holes detected", g.n_holes, 1);
    expectTrue("hole flagged after beat 19", g.hole_after[19] == 1 && g.hole_after[18] == 0);
    expectEq("off-grid downbeats dropped (0.3-beat one + the one in the hole)", g.n_dropped_offgrid, 2);
    // Downbeat at beat 20 sits right after the hole -> dropped; at beat 18
    // the next gap (18 -> 19) is normal, but hole_after[19] means the
    // downbeat at 19 would be before-hole; 19 is not a downbeat here.
    expectEq("hole-adjacent downbeats dropped (beat 20)", g.n_dropped_hole, 1);
    expectEq("kept downbeats: 20 - 1 (beat 20) = 19", (int) g.downbeats.size(), 19);
    expectTrue("kept downbeats lie exactly on beats", g.downbeats[3] == bt[6]);
    expectEq("bar length measured = 2 grid beats (not the TS hint 4)", g.bar_beats, 2);
    expectTrue("bar_from_downbeats flag", g.bar_from_downbeats);

    // Thin grid: fewer than 4 spacings -> hint kept.
    const auto g2 = cleanBeatGrid(bt.data(), 40, db.data(), 3, 3);
    expectEq("thin downbeat list keeps the TS hint", g2.bar_beats, 3);
    expectTrue("thin grid flag off", ! g2.bar_from_downbeats);

    // No downbeats at all -> empty list, hint kept, still valid.
    const auto g3 = cleanBeatGrid(bt.data(), 40, nullptr, 0, 4);
    expectEq("no downbeats -> hint", g3.bar_beats, 4);
    expectEq("no downbeats -> empty", (int) g3.downbeats.size(), 0);

    // Downbeat on every beat (Meshuggah class): bar would measure 1 ->
    // downbeats replaced by a synthetic grid every max(2, hint) beats.
    std::vector<double> bt4;
    for (int i = 0; i < 40; ++i) bt4.push_back(0.5 * i);
    const auto g4 = cleanBeatGrid(bt4.data(), 40, bt4.data(), 40, 2);
    expectTrue("downbeat-on-every-beat -> synthetic grid", g4.synthetic_downbeats);
    expectEq("synthetic bar = max(2, hint 2)", g4.bar_beats, 2);
    expectEq("synthetic downbeats every 2 beats", (int) g4.downbeats.size(), 20);
    const auto g5 = cleanBeatGrid(bt4.data(), 40, bt4.data(), 40, 4);
    expectEq("synthetic bar = hint 4", g5.bar_beats, 4);
    expectEq("synthetic downbeats every 4 beats", (int) g5.downbeats.size(), 10);

    // Sesja 135 (DEV-122): lattice bar starts for the planner. 139 beats:
    // head zone 0..38 synthetic, real 39..119 with real downbeats at
    // 43, 47, .., 115 (39 and 119 dropped as zone-adjacent), tail zone
    // 120..138 synthetic. Head takes the phase of 43 -> 39 is outside the
    // zone, so 35, 31, .., 3 (9); tail takes 115 -> 123, 127, 131, 135 (4).
    {
        std::vector<bool> synth(139, false);
        for (int i = 0; i < 39; ++i) synth[static_cast<std::size_t>(i)] = true;
        for (int i = 120; i < 139; ++i) synth[static_cast<std::size_t>(i)] = true;
        std::vector<int> real;
        for (int i = 43; i <= 115; i += 4) real.push_back(i);
        const auto lat = latticeDownbeatIdx(synth, real, 4);
        expectEq("lattice downbeats: 9 head + 4 tail", (int) lat.size(), 13);
        expectTrue("head lattice 3, 7, .., 35", lat.size() >= 9 && lat[0] == 3 && lat[8] == 35);
        expectTrue("tail lattice 123 .. 135", lat.size() == 13 && lat[9] == 123 && lat[12] == 135);
        expectTrue("every lattice downbeat lies on a synthetic beat",
                   [&] { for (int i : lat) if (! synth[static_cast<std::size_t>(i)]) return false; return true; }());
        // A hole zone in the middle takes the phase of the downbeat before it.
        std::vector<bool> hole(60, false);
        for (int i = 21; i < 41; ++i) hole[static_cast<std::size_t>(i)] = true;   // zone 21..40
        std::vector<int> realh { 0, 4, 8, 12, 16, 44, 48, 52, 56 };                 // 20 / 40 dropped as adjacent
        const auto lh = latticeDownbeatIdx(hole, realh, 4);
        expectEq("hole lattice: 24, 28, 32, 36, 40", (int) lh.size(), 5);
        expectTrue("hole lattice phase from 16", lh.size() == 5 && lh[0] == 24 && lh[4] == 40);
        expectEq("no mask -> none", (int) latticeDownbeatIdx({}, real, 4).size(), 0);
    }

    std::printf(g_fail == 0 ? "test_beat_grid PASS\n" : "test_beat_grid FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
