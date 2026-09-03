#include "analysis/GridConsistency.h"
#include "remix/BeatGrid.h"

#include <algorithm>
#include <cmath>

namespace reamix::analysis {

namespace {

double medianOf(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double medianPeriod(const std::vector<double>& beats)
{
    std::vector<double> gaps;
    gaps.reserve(beats.size());
    for (std::size_t i = 1; i < beats.size(); ++i) gaps.push_back(beats[i] - beats[i - 1]);
    return medianOf(std::move(gaps));
}

} // namespace

ConsistentGrid makeConsistentGrid(const std::vector<double>& beats,
                                  const std::vector<double>& downbeats,
                                  int    time_signature_hint,
                                  int    max_fill_periods,
                                  double phase_tol,
                                  double gap_ratio)
{
    ConsistentGrid out;
    if (beats.size() < 2) {
        out.beats     = beats;
        out.downbeats = downbeats;
        out.beatIsDownbeat.assign(beats.size(), false);
        out.bar_beats = std::max(1, time_signature_hint);
        return out;
    }

    // 1. Phase-consistent gap fill.
    const double period = medianPeriod(beats);
    out.beats.reserve(beats.size());
    out.beats.push_back(beats[0]);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        const double gap = beats[i] - beats[i - 1];
        if (period > 0.0 && gap > gap_ratio * period) {
            const double r = gap / period;
            const int    n = static_cast<int>(std::lround(r));
            if (n >= 2 && n <= max_fill_periods && std::fabs(r - n) <= phase_tol) {
                const double spacing = gap / n;
                for (int k = 1; k < n; ++k) out.beats.push_back(beats[i - 1] + spacing * k);
                ++out.n_filled_gaps;
                out.n_filled_beats += n - 1;
            } else {
                ++out.n_holes;
            }
        }
        out.beats.push_back(beats[i]);
    }

    // 2. Downbeats on the grid, measured bar.
    const auto grid = remix::cleanBeatGrid(out.beats.data(), static_cast<int>(out.beats.size()),
                                           downbeats.data(), static_cast<int>(downbeats.size()),
                                           time_signature_hint);
    out.downbeats           = grid.downbeats;
    out.bar_beats           = grid.bar_beats;
    out.synthetic_downbeats = grid.synthetic_downbeats;
    out.n_dropped_downbeats = grid.n_dropped_offgrid + grid.n_dropped_hole;
    out.beatIsDownbeat.assign(out.beats.size(), false);
    for (int idx : grid.downbeat_idx)
        if (idx >= 0 && idx < static_cast<int>(out.beats.size()))
            out.beatIsDownbeat[static_cast<std::size_t>(idx)] = true;

    // 3. The grid's own tempo.
    const double p = medianPeriod(out.beats);
    out.bpm = p > 0.0 ? 60.0 / p : 0.0;
    return out;
}

} // namespace reamix::analysis
