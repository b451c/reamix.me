#pragma once

#include <string>
#include <vector>

#include "StructureResult.h"

namespace reamix::analysis {

// Segment-level consolidation for structure analysis. Port of
// references/python-source/analysis/segment_consolidation.py L22-193
// (`SegmentConsolidationMixin._consolidate_segments` + its helpers). The
// `refine_boundaries` / `_snap_to_onsets` / `_update_segment_boundaries`
// methods are out of scope — the `StructureAnalyzer._analyze` dispatcher
// never calls them.
//
// Dispatcher integration (structure_analyzer.py L151 + L172): consolidation
// runs on whichever path (CBM or novelty) produced the segment list BEFORE
// the boundaries-rebuild step. Session-9 port exposes it as a pure static
// function taking a segment list + duration; the session-10 dispatcher port
// will invoke it inside its own branch logic.
//
// Precision policy: the algorithm is deterministic sequential list
// processing + integer indexing + plain f64 arithmetic. No RNG, no spectral
// decomposition. Strict bitwise parity is achievable under IEEE 754 f64
// with identical predicate order and identical short-circuit semantics.
//
// Segment type: session-9 uses a module-local struct to keep the port
// surgical (no refactor of CBMSegmenter / NoveltySegmenter). Session 10
// may introduce `src/analysis/StructureResult.h` with a unified shared
// `Segment` type; at that point the three local structs become aliases
// into one canonical definition.
class SegmentConsolidation
{
public:
    // PARITY: references/python-source/analysis/structure_analyzer.py L34-42
    // Segment dataclass. Session 10: aliased to the unified
    // `reamix::analysis::Segment` in StructureResult.h (previously a local
    // 5-field struct identical to this alias).
    using Segment = reamix::analysis::Segment;

    // PARITY: references/python-source/config.py:57-59. Inlined here as
    // constexpr so the port's parity contract is "these three numbers match
    // Python source on both sides" — if Python flips one, C++ and its
    // goldens both need to track the change.
    static constexpr double kMacroMinDurationSec  = 10.0;
    static constexpr double kBridgeMergeMaxSec    =  8.0;
    static constexpr int    kMaxSegments          =   12;

    // Consolidate a segment list. Mirrors
    // `SegmentConsolidationMixin._consolidate_segments(segments, duration)`.
    //
    // Returns an empty vector on empty input (Python L87-88 short-circuit).
    // Otherwise applies (in order):
    //   1. Initial same-label merge pass (L90).
    //   2. `while changed` predicate loop (L95-144):
    //      a. short "bridge" flanked by same-label non-intro/outro peers
    //         → triplet-merge (L106-117).
    //      b. short middle (< 0.75 × min_duration) between same-label
    //         neighbours → triplet-merge (L119-128).
    //      c. short chorus/verse/bridge (< 0.6 × min_duration) with
    //         same-label predecessor → pair-merge (L130-139).
    //      Re-runs same-label merge after each full sweep.
    //   3. `while len > max_segments` cap loop (L146-174): greedily fuse
    //      the shortest interior segment.
    //   4. Position-based intro/outro overrides + boundary clamp to
    //      [0, duration] (L176-191).
    //
    // `duration` is the clip duration in seconds (dispatcher passes
    // `float(len(y)) / float(sr)` per Python L119).
    static std::vector<Segment>
    consolidate(const std::vector<Segment>& segments, double duration);
};

} // namespace reamix::analysis
