#pragma once

#include <cstdint>
#include <set>
#include <vector>

#include "analysis/StructureResult.h"

namespace reamix::remix {

// Port of `references/python-source/remix/viterbi_dp.py::_compute_segment_data`
// (L387-462, 2026-04-21). Originally ported inline in `TransitionCost.cpp`
// anonymous namespace (session 18); extracted here session 19 to share the
// same impl between TransitionCost and ViterbiDP (session 20 target). Zero
// semantic change at extraction — `test_transition_cost` stays bit-exact
// across the refactor (16-track corpus PASS, session-19 VALIDATION row).
//
// Outputs:
//   beat_to_segment: (n_beats,) int64. Segment index per beat; 0 when no
//     segments or beat not contained by any segment.
//   seg_sim: (n_segs, n_segs) f64 row-major. Segment similarity:
//     - diagonal 1.0
//     - same lowercase-label pair → 0.9
//     - different labels → 0.2
//     - refined with 0.6 × label_sim + 0.4 × max(0, cosine of mean feature
//       vectors) when features are available AND both segment means have
//       norm ≥ COSINE_DEGENERATE_FLOOR.
//   n_segs: size of seg_sim square; 1 on empty-segments fallback.
//   boundary_beats: set<int>. Beat b flagged if beat_to_segment[b] !=
//     beat_to_segment[b-1] for b ≥ 1. Consumed only by ViterbiDP's boundary
//     bonus; not used by TransitionCost itself.
//
// Degenerate case: segments empty → returns (zeros(n_beats), ones(1,1),
// empty set). Matches Python L400-405.
struct SegmentData
{
    std::vector<std::int64_t> beat_to_segment;
    std::vector<double>       seg_sim;   // flat (n_segs × n_segs), row-major
    int                       n_segs = 0;
    std::set<int>             boundary_beats;
};

// Degenerate-feature-vector guard: below this norm, the segment's feature
// mean is treated as zero and does not contribute to the cosine blend. Same
// value / same citation as `transition_cost.py::_cosine_sim` (L133); kept in
// this header so ViterbiDP and other future phase-4 consumers can reuse it
// without pulling TransitionCost.h.
// Source-of-truth: `transition_cost.py:133 (2026-04-21)`.
inline constexpr double COSINE_DEGENERATE_FLOOR = 1e-8;

// Features pointer may be null (label-only similarity path). When non-null
// it must be row-major f32 of shape (n_beats, n_features) — same layout as
// the numpy `features` array Python consumes.
SegmentData computeSegmentData(int                      n_beats,
                               const analysis::Segment* segments,
                               int                      n_segments,
                               const double*            beat_times,
                               const float*             features,
                               int                      n_features);

} // namespace reamix::remix
