// BeatGrid - consistency clean-up of the beat / downbeat grid before the
// engine consumes it (ADR-115 E5, sesja 115).
//
// Sesja-115 corpus audit (17 tracks) found the analysis grid inconsistent on
// about a third of the tracks: half-tempo beat grids with a detected time
// signature that does not match the downbeat spacing (Drake: 2 grid beats
// per bar with TS = 4), beat-tracker holes of up to 30 s bridged by
// DownbeatCleaner downbeats that lie on no beat (Chick Corea 74 / 572,
// Meshuggah 158 / 840 off-grid), and TS = 2 with a downbeat on almost every
// beat (Meshuggah). `resolveDbIndices` then snapped every downbeat to the
// nearest beat without tolerance, and every "bar" quantity (micro-skip,
// cooldown, the repetition prior's measure) used the detected TS. The
// listening panel's non-wins were exactly the shaky-grid tracks.
//
// This helper produces, for the v2 engine path only:
//   * downbeats snapped onto the beat grid, dropped when farther than
//     `snap_tol_beats` from any beat or when adjacent to a beat hole
//     (gap > `hole_ratio` x median period), de-duplicated;
//   * `bar_beats` = median spacing of the kept downbeats in grid beats
//     (the measured bar length), falling back to the detected TS when the
//     grid is too thin; a measured bar shorter than 2 beats (Meshuggah:
//     detector downbeat on nearly every beat) marks the downbeat list as
//     unusable and the grid is re-synthesised every max(2, TS) beats.
// Callers pass `downbeats` + `bar_beats` where they used to pass the raw
// downbeat times + `timeSigNum`. Legacy path untouched (parity).
//
// C++-canonical (ADR-065); validated by tests/parity/test_beat_grid.cpp.
#pragma once

#include <cstdint>
#include <vector>

namespace reamix::remix {

struct BeatGridResult
{
    std::vector<double>        downbeats;   // on-grid downbeat times (subset of beat_times)
    std::vector<int>           downbeat_idx;
    int                        bar_beats { 4 };
    int                        n_dropped_offgrid { 0 };
    int                        n_dropped_hole    { 0 };
    int                        n_holes           { 0 };
    std::vector<std::uint8_t>  hole_after;  // [n] 1 = gap after beat i exceeds hole_ratio x period
    double                     period_sec { 0.0 };
    bool                       bar_from_downbeats { false };
    // Detector downbeats judged unusable (measured bar < 2 grid beats, e.g.
    // a downbeat on almost every beat): downbeats were re-synthesised every
    // `bar_beats` from the first beat, skipping hole edges.
    bool                       synthetic_downbeats { false };
};

BeatGridResult cleanBeatGrid(const double* beat_times, int n_beats,
                             const double* downbeats, int n_downbeats,
                             int time_signature_hint,
                             double snap_tol_beats = 0.25,
                             double hole_ratio = 1.5);

} // namespace reamix::remix
