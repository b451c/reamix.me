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
                                  double duration_sec,
                                  int    min_lattice_beats,
                                  int    max_fill_periods,
                                  double phase_tol,
                                  double gap_ratio)
{
    ConsistentGrid out;
    if (beats.size() < 2) {
        out.beats     = beats;
        out.downbeats = downbeats;
        out.beatIsDownbeat.assign(beats.size(), false);
        out.beatIsSynthetic.assign(beats.size(), false);
        out.bar_beats = std::max(1, time_signature_hint);
        return out;
    }

    // 1. Phase-consistent gap fill; every other gap gets a phase-free
    //    lattice from its left edge (sesja 135, DEV-122): beats every period
    //    while the far edge stays at least half a period away.
    const double period = medianPeriod(beats);
    std::vector<bool> synthetic;
    auto push = [&](double t, bool synth) { out.beats.push_back(t); synthetic.push_back(synth); };
    const int min_zone = std::max(1, min_lattice_beats);
    // Head lattice: from the first beat back towards the file start.
    if (period > 0.0 && beats[0] - period >= 0.5 * period) {
        std::vector<double> head;
        for (double t = beats[0] - period; t >= 0.5 * period; t -= period) head.push_back(t);
        if (static_cast<int>(head.size()) >= min_zone) {
            for (auto it = head.rbegin(); it != head.rend(); ++it) push(*it, true);
            ++out.n_lattice_zones;
            out.n_lattice_beats += static_cast<int>(head.size());
        }
    }
    push(beats[0], false);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        const double gap = beats[i] - beats[i - 1];
        if (period > 0.0 && gap > gap_ratio * period) {
            const double r = gap / period;
            const int    n = static_cast<int>(std::lround(r));
            if (n >= 2 && n <= max_fill_periods && std::fabs(r - n) <= phase_tol) {
                const double spacing = gap / n;
                for (int k = 1; k < n; ++k) push(beats[i - 1] + spacing * k, false);
                ++out.n_filled_gaps;
                out.n_filled_beats += n - 1;
            } else {
                ++out.n_holes;
                std::vector<double> lat;
                for (double t = beats[i - 1] + period; beats[i] - t >= 0.5 * period; t += period) lat.push_back(t);
                if (static_cast<int>(lat.size()) >= min_zone) {
                    for (double t : lat) push(t, true);
                    ++out.n_lattice_zones;
                    out.n_lattice_beats += static_cast<int>(lat.size());
                }
            }
        }
        push(beats[i], false);
    }
    // Tail lattice: from the last beat towards the file end.
    if (period > 0.0 && duration_sec > 0.0 && beats.back() + period <= duration_sec - 0.5 * period) {
        std::vector<double> lat;
        for (double t = beats.back() + period; t <= duration_sec - 0.5 * period; t += period) lat.push_back(t);
        if (static_cast<int>(lat.size()) >= min_zone) {
            for (double t : lat) push(t, true);
            ++out.n_lattice_zones;
            out.n_lattice_beats += static_cast<int>(lat.size());
        }
    }
    out.beatIsSynthetic = synthetic;

    // 2. Downbeats on the grid, measured bar. A downbeat on a lattice beat
    //    or next to one is dropped (the engines' rule for hole edges).
    const auto grid = remix::cleanBeatGrid(out.beats.data(), static_cast<int>(out.beats.size()),
                                           downbeats.data(), static_cast<int>(downbeats.size()),
                                           time_signature_hint);
    out.bar_beats           = grid.bar_beats;
    out.synthetic_downbeats = grid.synthetic_downbeats;
    out.n_dropped_downbeats = grid.n_dropped_offgrid + grid.n_dropped_hole;
    out.beatIsDownbeat.assign(out.beats.size(), false);
    const int nb = static_cast<int>(out.beats.size());
    auto synthAt = [&](int k) { return k >= 0 && k < nb && synthetic[static_cast<std::size_t>(k)]; };
    for (int idx : grid.downbeat_idx) {
        if (idx < 0 || idx >= nb) continue;
        if (synthAt(idx) || synthAt(idx - 1) || synthAt(idx + 1)) { ++out.n_dropped_downbeats; continue; }
        out.beatIsDownbeat[static_cast<std::size_t>(idx)] = true;
        out.downbeats.push_back(out.beats[static_cast<std::size_t>(idx)]);
    }

    // 3. The grid's own tempo.
    const double p = medianPeriod(out.beats);
    out.bpm = p > 0.0 ? 60.0 / p : 0.0;
    return out;
}

} // namespace reamix::analysis
