#pragma once

#include <map>
#include <utility>
#include <vector>

#include "remix/TransitionCost.h"  // TransitionCandidate

namespace reamix::remix {

// ADR-115 E11 (sesja 117) - loop-spot suggestions for Region mode.
//
// A "loop spot" is a backward, bar-aligned jump pair (from_beat -> to_beat,
// to_beat < from_beat) taken from the Region candidate pool computed over the
// WHOLE track (`computeRegionCosts` with entry_beat = 0, exit_beat = n_beats;
// same gates, prior and composite the Region remix itself uses, so a spot the
// map shows is a pair the Region path search can actually take). Its span
// [start_sec, end_sec) = [beat_times[to_beat], beat_times[from_beat + 1]) is
// the material that repeats when the loop is taken.
//
// C++-canonical per ADR-065: no Python reference, hand-computed invariants in
// tests/unit/test_loop_spots.cpp.
struct LoopSpot
{
    int    from_beat = 0;   // pre-downbeat source beat (loop jumps out after it)
    int    to_beat   = 0;   // downbeat target (loop jumps back to it)
    double quality   = 0.0; // composite quality [0, 1], higher = cleaner splice
    double start_sec = 0.0; // beat_times[to_beat]
    double end_sec   = 0.0; // beat_times[from_beat + 1]
    int    bars      = 0;   // round((from_beat + 1 - to_beat) / bar_beats)
};

// Every backward candidate of `candidates` (absolute beat indices) as a
// LoopSpot, sorted best quality first (ties: earlier start, then shorter).
// Forward pairs (skips) are not loops and are dropped. `bar_beats` is the
// measured bar (BeatGrid) used to label the span length in bars.
std::vector<LoopSpot> extractLoopSpots(
    const std::map<std::pair<int, int>, TransitionCandidate>& candidates,
    const double* beat_times, int n_beats, int bar_beats);

struct LoopSpotFilter
{
    // Inclusive time window the whole span must lie in (item-relative
    // seconds). Defaults = no window (whole track).
    double window_start_sec = -1e300;
    double window_end_sec   =  1e300;
    double window_eps_sec   = 1e-3;

    double min_quality  = 0.5;   // WaveformView Medium bucket floor
    double min_span_sec = 6.0;   // Region minimum (ReaperBridge getItemRegion)
    int    max_bars     = 16;    // longer repeats are sections, not spots
    int    max_count    = 5;     // chips shown
};

// Best-first greedy pick of non-overlapping spots that pass the filter.
// Adjacent spans (end == start) are allowed; overlapping ones are not, so the
// UI never stacks chips. Result keeps the quality order.
std::vector<LoopSpot> suggestLoopSpots(const std::vector<LoopSpot>& all,
                                       const LoopSpotFilter&        filter);

} // namespace reamix::remix
