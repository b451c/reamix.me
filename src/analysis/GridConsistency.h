// GridConsistency - the analysis-stage beat / downbeat grid the whole product
// consumes (DEV-088 detector-side fix, sesja 124; ADR-115 E5 follow-up).
//
// The sesja-115 corpus audit found three grid inconsistencies coming out of
// BeatDetector: (a) the reported tempo is octave-corrected into 78-185 BPM
// while the beat list stays at the model's rate (Drake: 135 reported, 67
// grid; Bob Dylan: 156 reported, 78 grid); (b) beat-tracker holes - the
// model sees no beat in a quiet or rubato stretch, BeatInterpolator places
// beats on sub-threshold noise and the 25 % consistency pass removes them
// all again - bridged by DownbeatCleaner downbeats that lie on no beat;
// (c) a detected time signature that does not match the downbeat spacing.
// Sesja 115 guarded the v2 engine path downstream (`remix::cleanBeatGrid`);
// the UI mask, the reported BPM, the legacy path and the harness still saw
// the raw grid. This helper makes the bundle itself consistent:
//
//   1. Phase-consistent gaps are filled: a gap of r median periods with
//      |r - round(r)| <= `phase_tol` and round(r) <= `max_fill_periods` gets
//      round(r) - 1 evenly spaced beats (the tempo held through the gap, so
//      the fill is within phase_tol / 2 of the true beats). Longer or
//      phase-inconsistent holes (Alice in Chains 60 periods at -0.39,
//      Meshuggah polymeter) stay holes: a straight fill there would drift
//      up to half a beat and every splice on it would click.
//   2. Downbeats are snapped onto the (filled) grid, off-grid and
//      hole-adjacent ones dropped, the bar length measured, and a downbeat
//      list with a bar shorter than 2 beats re-synthesised
//      (`remix::cleanBeatGrid`, unchanged).
//   3. The reported tempo is the grid's own (60 / median period), never an
//      octave away from the beats the engine uses. Half-tempo grids
//      (Drake) are reported as such; subdividing them is a separate
//      decision (needs the model's sub-threshold logits at the midpoints).
//   4. Sesja 135 (DEV-122): the zones the detector left beatless - the
//      un-beated head (file start .. first beat), the tail (last beat ..
//      file end, when `duration_sec` is known) and every gap step 1 did not
//      fill - get a phase-free LATTICE at the grid period: beats continued
//      from the zone's real edge, the last interval before the far edge left
//      partial (0.5 .. 1.5 periods). They are flagged in `beatIsSynthetic`.
//      Features are extracted for them like for any beat, so the shape
//      planner may trim the first / last piece or land an ending inside
//      such a zone the way Adobe Audition's rated-clean plans do (Alice in
//      Chains: 33 s fingerpicked intro + 23 s tail + 29 s hole; Drake: 53 s
//      beatless outro). Only a zone of at least `min_lattice_beats` (8 =
//      the planner's shortest piece) gets a lattice: a 1-7 beat tail or
//      hole stays as it was, so a track without a real beatless zone keeps
//      a bit-identical grid (the sesja-135 measurement: lattices in every
//      tail changed 20 of 51 normal-ratio remixes for nothing). The
//      continuation engines never splice on them:
//      no engine downbeat lies on a lattice beat or next to one (the same
//      rule that dropped hole-adjacent downbeats before), so bar-aligned
//      candidate generation cannot start or land there; the planner gets
//      lattice bars from remix::latticeDownbeatIdx. The UI draws no tick
//      for a lattice beat.
//
// C++-canonical (ADR-065); validated by tests/parity/test_grid_consistency.cpp.
#pragma once

#include <vector>

namespace reamix::analysis {

struct ConsistentGrid
{
    std::vector<double> beats;
    std::vector<double> downbeats;       // subset of beats (real, never on / next to a lattice beat)
    std::vector<bool>   beatIsDownbeat;  // length == beats.size()
    std::vector<bool>   beatIsSynthetic; // length == beats.size(); sesja 135 lattice beats
    int                 bar_beats { 4 };
    double              bpm { 0.0 };     // 60 / median period of `beats`
    int                 n_filled_gaps { 0 };
    int                 n_filled_beats { 0 };
    int                 n_dropped_downbeats { 0 };
    int                 n_holes { 0 };   // gaps step 1 left unfilled (now lattice zones)
    int                 n_lattice_zones { 0 };   // head + tail + holes that received a lattice
    int                 n_lattice_beats { 0 };
    bool                synthetic_downbeats { false };
};

// `duration_sec` = the file length (0 = unknown: no tail lattice).
ConsistentGrid makeConsistentGrid(const std::vector<double>& beats,
                                  const std::vector<double>& downbeats,
                                  int    time_signature_hint,
                                  double duration_sec     = 0.0,
                                  int    min_lattice_beats = 8,
                                  int    max_fill_periods = 16,
                                  double phase_tol        = 0.15,
                                  double gap_ratio        = 1.5);

} // namespace reamix::analysis
