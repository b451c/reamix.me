#include "remix/LoopSpots.h"

#include <algorithm>
#include <cmath>

namespace reamix::remix {

std::vector<LoopSpot> extractLoopSpots(
    const std::map<std::pair<int, int>, TransitionCandidate>& candidates,
    const double* beat_times, int n_beats, int bar_beats)
{
    std::vector<LoopSpot> out;
    if (beat_times == nullptr || n_beats < 2) return out;
    const int bb = std::max(1, bar_beats);
    // Track-end extrapolation (a pre-downbeat source can be the last beat only
    // on a degenerate grid; keep the span finite anyway).
    const double last_step = beat_times[n_beats - 1] - beat_times[n_beats - 2];

    for (const auto& kv : candidates) {
        const int from = kv.first.first;
        const int to   = kv.first.second;
        if (to >= from) continue;                       // forward skip, not a loop
        if (to < 0 || from < 0 || from >= n_beats) continue;
        LoopSpot s;
        s.from_beat = from;
        s.to_beat   = to;
        s.quality   = kv.second.quality_score;
        s.start_sec = beat_times[to];
        s.end_sec   = (from + 1 < n_beats) ? beat_times[from + 1]
                                           : beat_times[n_beats - 1] + last_step;
        s.bars      = std::max(1, (int) std::lround((double) (from + 1 - to) / (double) bb));
        out.push_back(s);
    }
    std::sort(out.begin(), out.end(), [](const LoopSpot& a, const LoopSpot& b) {
        if (a.quality != b.quality) return a.quality > b.quality;
        if (a.start_sec != b.start_sec) return a.start_sec < b.start_sec;
        return a.end_sec < b.end_sec;
    });
    return out;
}

std::vector<LoopSpot> suggestLoopSpots(const std::vector<LoopSpot>& all,
                                       const LoopSpotFilter&        f)
{
    std::vector<LoopSpot> picked;
    if (f.max_count <= 0) return picked;
    for (const LoopSpot& s : all) {                    // `all` is best-first
        if (s.quality < f.min_quality) continue;
        if (s.end_sec - s.start_sec < f.min_span_sec) continue;
        if (s.bars > f.max_bars) continue;
        if (s.start_sec < f.window_start_sec - f.window_eps_sec) continue;
        if (s.end_sec   > f.window_end_sec   + f.window_eps_sec) continue;
        bool overlaps = false;
        for (const LoopSpot& p : picked)
            if (s.start_sec < p.end_sec - f.window_eps_sec
                && p.start_sec < s.end_sec - f.window_eps_sec) { overlaps = true; break; }
        if (overlaps) continue;
        picked.push_back(s);
        if ((int) picked.size() >= f.max_count) break;
    }
    return picked;
}

} // namespace reamix::remix
