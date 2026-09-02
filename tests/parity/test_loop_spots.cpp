// test_loop_spots — sesja 117 (ADR-115 E11) self-validation.
//
// Validates the loop-spot map built from a Region candidate pool on a
// synthetic 64-beat track (16 bars of 4, 0.5 s per beat) with hand-placed
// candidates (absolute indices, from -> to):
//
//   A = (15 ->  0)  q 0.82   bars 0-3   (4 bars, 8 s)     best
//   B = (31 -> 16)  q 0.75   bars 4-7   (4 bars, 8 s)     adjacent to A
//   C = (23 ->  8)  q 0.78   bars 2-5   (4 bars) overlaps A and B
//   D = (11 ->  8)  q 0.90   bar 2      (1 bar, 2 s) too short for a region
//   E = (63 ->  0)  q 0.71   whole song (16 bars)         over max_bars
//   F = (47 -> 40)  q 0.44   bars 10-11 (2 bars, 4 s)     below the floor
//   G = ( 7 -> 24)  q 0.95   FORWARD skip                 never a loop
//   H = (55 -> 40)  q 0.66   bars 10-13 (4 bars)          medium
//
// Invariants:
//   1. extract: G dropped; 7 spots best-first; span and bars hand-checked;
//   2. suggest (whole track, max_bars 8): D (span 2 s < 6 s), E (16 bars),
//      F (q < 0.5) are filtered; greedy non-overlap picks A, then C is
//      rejected (overlaps A), B accepted (adjacent, end == start), H accepted
//      -> [A, B, H];
//   3. max_count 1 -> [A] only;
//   4. window = bars 4-13 (8 s .. 28 s) -> A and C excluded (start before
//      the window), picks [B, H]; window = bars 8-9 -> nothing;
//   5. a window that cuts a spot by less than eps still contains it.
//
// Per ADR-065: hand-computed invariants, no Python ground truth.

#include "remix/LoopSpots.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

using namespace reamix::remix;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
            ++g_failures;                                                       \
        } else {                                                                \
            std::printf("  ok:   %s\n", msg);                                   \
        }                                                                       \
    } while (0)

TransitionCandidate cand(int from, int to, double q)
{
    TransitionCandidate c{};
    c.from_beat     = from;
    c.to_beat       = to;
    c.quality_score = q;
    c.total_cost    = 1.0 - q;
    return c;
}

bool isPair(const LoopSpot& s, int from, int to)
{
    return s.from_beat == from && s.to_beat == to;
}

} // namespace

int main()
{
    constexpr int    kBeats    = 64;
    constexpr int    kBarBeats = 4;
    constexpr double kBeatSec  = 0.5;
    std::vector<double> bt(kBeats);
    for (int i = 0; i < kBeats; ++i) bt[(std::size_t) i] = i * kBeatSec;

    std::map<std::pair<int, int>, TransitionCandidate> pool;
    auto add = [&](int from, int to, double q) { pool[{from, to}] = cand(from, to, q); };
    add(15,  0, 0.82);   // A
    add(31, 16, 0.75);   // B
    add(23,  8, 0.78);   // C
    add(11,  8, 0.90);   // D
    add(63,  0, 0.71);   // E
    add(47, 40, 0.44);   // F
    add( 7, 24, 0.95);   // G (forward)
    add(55, 40, 0.66);   // H

    std::printf("[1] extract\n");
    const auto all = extractLoopSpots(pool, bt.data(), kBeats, kBarBeats);
    CHECK(all.size() == 7, "forward pair dropped, 7 loop spots");
    CHECK(!all.empty() && isPair(all[0], 11, 8), "best first = D (q 0.90)");
    CHECK(all.size() >= 2 && isPair(all[1], 15, 0), "second = A (q 0.82)");
    CHECK(all.size() == 7 && isPair(all[6], 47, 40), "last = F (q 0.44)");
    for (const auto& s : all) {
        if (isPair(s, 15, 0)) {
            CHECK(std::abs(s.start_sec - 0.0) < 1e-9 && std::abs(s.end_sec - 8.0) < 1e-9,
                  "A span = [0, 8) s (beat 0 .. beat 16)");
            CHECK(s.bars == 4, "A = 4 bars");
        }
        if (isPair(s, 11, 8)) {
            CHECK(std::abs(s.start_sec - 4.0) < 1e-9 && std::abs(s.end_sec - 6.0) < 1e-9,
                  "D span = [4, 6) s");
            CHECK(s.bars == 1, "D = 1 bar");
        }
        if (isPair(s, 63, 0)) {
            CHECK(std::abs(s.end_sec - 32.0) < 1e-9, "E end extrapolated one beat past the last beat");
            CHECK(s.bars == 16, "E = 16 bars");
        }
    }

    std::printf("[2] suggest, whole track\n");
    LoopSpotFilter f;
    f.max_bars = 8;
    auto picks = suggestLoopSpots(all, f);
    CHECK(picks.size() == 3, "three non-overlapping spots");
    CHECK(picks.size() == 3 && isPair(picks[0], 15, 0),  "pick 1 = A");
    CHECK(picks.size() == 3 && isPair(picks[1], 31, 16), "pick 2 = B (adjacent to A allowed, C overlaps A)");
    CHECK(picks.size() == 3 && isPair(picks[2], 55, 40), "pick 3 = H (medium, F below floor)");
    for (const auto& p : picks) {
        CHECK(!isPair(p, 11, 8),  "D (2 s) filtered by min_span");
        CHECK(!isPair(p, 63, 0),  "E (16 bars) filtered by max_bars 8");
        CHECK(!isPair(p, 47, 40), "F (q 0.44) filtered by min_quality");
        CHECK(!isPair(p, 23, 8),  "C rejected: overlaps A");
    }

    std::printf("[3] max_count\n");
    f.max_count = 1;
    picks = suggestLoopSpots(all, f);
    CHECK(picks.size() == 1 && isPair(picks[0], 15, 0), "max_count 1 -> [A]");
    f.max_count = 5;

    std::printf("[4] window\n");
    f.window_start_sec = 8.0;    // bar 4
    f.window_end_sec   = 28.0;   // end of bar 13
    picks = suggestLoopSpots(all, f);
    CHECK(picks.size() == 2, "window bars 4-13 -> two spots");
    CHECK(picks.size() == 2 && isPair(picks[0], 31, 16) && isPair(picks[1], 55, 40),
          "window picks = [B, H]");
    f.window_start_sec = 16.0;   // bars 8-9: no candidate inside
    f.window_end_sec   = 20.0;
    picks = suggestLoopSpots(all, f);
    CHECK(picks.empty(), "window without a loop -> empty (\"no clean loop in this selection\")");

    std::printf("[5] window eps\n");
    f.window_start_sec = 8.0 + 0.0005;    // B starts at 8.0: inside by eps
    f.window_end_sec   = 16.0 - 0.0005;   // B ends at 16.0
    picks = suggestLoopSpots(all, f);
    CHECK(picks.size() == 1 && isPair(picks[0], 31, 16), "sub-eps cut still contains B");

    std::printf("[7] section spans (sesja 120, Blocks suggestions)\n");
    // Every pair becomes the span between its two cut points, forward pairs
    // included: G (7 -> 24) skips [8, 24) = 8 s .. 12 s, 4 bars.
    const auto spans = extractSectionSpans(pool, bt.data(), kBeats, kBarBeats);
    CHECK(spans.size() == 8, "all 8 pairs become spans (forward kept)");
    CHECK(!spans.empty() && isPair(spans[0], 7, 24), "best first = G (q 0.95)");
    for (const auto& s : spans) {
        if (isPair(s, 7, 24)) {
            CHECK(std::abs(s.start_sec - 4.0) < 1e-9 && std::abs(s.end_sec - 12.0) < 1e-9,
                  "G span = [4, 12) s (beats 8 .. 24)");
            CHECK(s.bars == 4, "G = 4 bars");
        }
        if (isPair(s, 15, 0))
            CHECK(std::abs(s.start_sec - 0.0) < 1e-9 && std::abs(s.end_sec - 8.0) < 1e-9,
                  "A span unchanged as a section span");
    }
    LoopSpotFilter sf;
    sf.min_bars  = 4;     // whole sections only
    sf.max_bars  = 32;
    sf.max_count = 8;
    auto sections = suggestLoopSpots(spans, sf);
    CHECK(sections.size() == 2, "min_bars 4 + greedy non-overlap -> 2 sections");
    CHECK(sections.size() == 2 && isPair(sections[0], 7, 24), "section 1 = G [4, 12)");
    CHECK(sections.size() == 2 && isPair(sections[1], 55, 40), "section 2 = H [20, 28) (A, B, C, E overlap G)");
    for (const auto& p : sections) {
        CHECK(!isPair(p, 11, 8),  "D (1 bar) filtered by min_bars");
        CHECK(!isPair(p, 15, 0),  "A overlaps G -> rejected");
        CHECK(!isPair(p, 63, 0),  "E (16 bars) overlaps everything after G");
    }
    { std::map<std::pair<int, int>, TransitionCandidate> adj;
      adj[{7, 8}] = cand(7, 8, 0.9);
      CHECK(extractSectionSpans(adj, bt.data(), kBeats, kBarBeats).empty(), "adjacent pair -> no span"); }

    std::printf("[6] empty inputs\n");
    CHECK(extractLoopSpots({}, bt.data(), kBeats, kBarBeats).empty(), "no candidates -> no spots");
    CHECK(extractLoopSpots(pool, nullptr, 0, kBarBeats).empty(), "no beats -> no spots");
    CHECK(suggestLoopSpots({}, LoopSpotFilter{}).empty(), "no spots -> no picks");

    if (g_failures == 0) {
        std::printf("test_loop_spots: PASS\n");
        return 0;
    }
    std::printf("test_loop_spots: %d FAILURE(S)\n", g_failures);
    return 1;
}
