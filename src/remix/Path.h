#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace reamix::remix {

// Unified data contract for phase-4 remix-engine outputs.
//
// Port of references/python-source/remix/path.py (25 LOC dataclass).
// Introduced at session-16 phase-4 kickoff; consumed by all downstream
// phase-4 modules (TransitionCost, ViterbiDP, Optimizer, RegionOptimizer,
// BlockAssembly) as the canonical remix result type.
//
// Field conventions diff vs Python:
//   * Python `beat_indices: np.ndarray` (typically int64). C++ uses
//     std::vector<int> matching phases 1-3 beat-index convention. If
//     parity ever requires int64 semantics at serialization boundary, the
//     dump tool can promote at load/save time — same as session-10
//     StructureResult cluster_id int vs int64 handling.
//   * Python `transitions: List[Tuple[int, int]]` (from_beat, to_beat).
//     C++ uses std::vector<std::pair<int, int>>.
//   * Python `transition_metadata: Dict[Tuple[int, int], Dict[str, float]]`.
//     C++ uses std::map<std::pair<int, int>, std::map<std::string, double>> —
//     std::map (not unordered_map) because std::pair has no default hash,
//     and because ordered iteration matches Python 3.7+ dict insertion-order
//     semantics close enough for our dump + parity-compare purposes (keys
//     are (from, to) pairs which sort lexicographically in a stable way).
//
// Semantics:
//   * beat_indices: sequence of beat indices to play; output duration in
//     beats = beat_indices.size().
//   * total_cost: aggregate path cost from ViterbiDP (lower = better).
//     Parity acceptance ≤ 5 % delta vs Python per phase-4 spec.
//   * duration_beats: beat_indices.size() — redundant with the vector size
//     but kept for Python-parity symmetry in serialization and for cheap
//     consumer access.
//   * transitions: list of (from, to) pairs where `to != from + 1` — i.e.
//     the non-sequential jumps, the actual splice points. Sequential
//     steps are omitted from this list but still present in beat_indices.
//   * transition_metadata: per-transition diagnostic breakdown of the 10
//     cost signals (pre-redistribution) + final composite quality score.
//     Populated by TransitionCost, consumed by ViterbiDP for path cost
//     accumulation, by phase-4 parity tests for per-signal sub-gates
//     (ADR-026), and by the optional perceptual harness. Keys inside the
//     per-transition dict follow quality.py signal names: "waveform",
//     "successor", "edge_splice", "context", "label", "bar_align",
//     "section", "energy", "edge_energy", "centroid", plus "quality_score"
//     (composite) and any vocal/onset penalties applied.
struct RemixPath
{
    std::vector<int>                                              beat_indices;
    double                                                        total_cost     = 0.0;
    int                                                           duration_beats = 0;
    std::vector<std::pair<int, int>>                              transitions;
    std::map<std::pair<int, int>, std::map<std::string, double>>  transition_metadata;
};

} // namespace reamix::remix
