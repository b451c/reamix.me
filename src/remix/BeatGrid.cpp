#include "BeatGrid.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace reamix::remix {

namespace {

double medianOf(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    return v[m];
}

} // namespace

BeatGridResult cleanBeatGrid(const double* beat_times, int n_beats,
                             const double* downbeats, int n_downbeats,
                             int time_signature_hint,
                             double snap_tol_beats,
                             double hole_ratio)
{
    BeatGridResult out;
    out.bar_beats = std::max(1, time_signature_hint);
    if (beat_times == nullptr || n_beats < 2) return out;

    // Median beat period (robust to holes and spurious extras).
    std::vector<double> gaps;
    gaps.reserve(static_cast<std::size_t>(n_beats - 1));
    for (int i = 0; i + 1 < n_beats; ++i) gaps.push_back(beat_times[i + 1] - beat_times[i]);
    out.period_sec = medianOf(gaps);
    if (! (out.period_sec > 0.0)) return out;

    // Holes: gap after beat i larger than hole_ratio x period.
    out.hole_after.assign(static_cast<std::size_t>(n_beats), 0);
    for (int i = 0; i + 1 < n_beats; ++i) {
        if (gaps[static_cast<std::size_t>(i)] > hole_ratio * out.period_sec) {
            out.hole_after[static_cast<std::size_t>(i)] = 1;
            ++out.n_holes;
        }
    }

    // Snap downbeats to the nearest beat with tolerance; drop hole-adjacent.
    std::set<int> kept;
    const double tol = snap_tol_beats * out.period_sec;
    for (int d = 0; d < n_downbeats; ++d) {
        const double t = downbeats[d];
        if (! std::isfinite(t)) { ++out.n_dropped_offgrid; continue; }
        const double* hi = std::lower_bound(beat_times, beat_times + n_beats, t);
        int idx = static_cast<int>(hi - beat_times);
        if (idx >= n_beats) idx = n_beats - 1;
        if (idx > 0 && std::abs(beat_times[idx - 1] - t) <= std::abs(beat_times[idx] - t)) --idx;
        if (std::abs(beat_times[idx] - t) > tol) { ++out.n_dropped_offgrid; continue; }
        // A downbeat right after a hole has no usable pre-downbeat source;
        // a downbeat right before a hole starts a bar that stops at once.
        const bool after_hole  = idx > 0 && out.hole_after[static_cast<std::size_t>(idx - 1)] != 0;
        const bool before_hole = out.hole_after[static_cast<std::size_t>(idx)] != 0;
        if (after_hole || before_hole) { ++out.n_dropped_hole; continue; }
        kept.insert(idx);
    }

    for (int idx : kept) {
        out.downbeat_idx.push_back(idx);
        out.downbeats.push_back(beat_times[idx]);
    }

    // Measured bar length in grid beats: median spacing of consecutive kept
    // downbeats, ignoring spacings that straddle a hole.
    std::vector<double> spacings;
    for (std::size_t k = 0; k + 1 < out.downbeat_idx.size(); ++k) {
        const int a = out.downbeat_idx[k], b = out.downbeat_idx[k + 1];
        bool crosses_hole = false;
        for (int i = a; i < b; ++i)
            if (out.hole_after[static_cast<std::size_t>(i)] != 0) { crosses_hole = true; break; }
        if (! crosses_hole) spacings.push_back(static_cast<double>(b - a));
    }
    if (spacings.size() >= 4) {
        const int bar = static_cast<int>(std::lround(medianOf(spacings)));
        if (bar >= 2) {
            out.bar_beats = bar;
            out.bar_from_downbeats = true;
        } else if (bar >= 1) {
            // Downbeat on (almost) every beat: the detector's downbeats carry
            // no bar information. Keep the hint (never below 2) and lay a
            // synthetic grid, skipping hole edges like the real one.
            out.bar_beats = std::max(2, time_signature_hint);
            out.synthetic_downbeats = true;
            out.downbeat_idx.clear();
            out.downbeats.clear();
            for (int idx = 0; idx < n_beats; idx += out.bar_beats) {
                const bool after_hole  = idx > 0 && out.hole_after[static_cast<std::size_t>(idx - 1)] != 0;
                const bool before_hole = out.hole_after[static_cast<std::size_t>(idx)] != 0;
                if (after_hole || before_hole) continue;
                out.downbeat_idx.push_back(idx);
                out.downbeats.push_back(beat_times[idx]);
            }
        }
    }
    return out;
}

} // namespace reamix::remix
