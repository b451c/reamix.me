#pragma once

#include <optional>
#include <cstdint>
#include <string>
#include <vector>

namespace reamix::analysis {

// Unified data contract for phase-3 structure segmentation.
//
// Port of references/python-source/analysis/structure_analyzer.py L34-53
// (`Segment` + `StructureResult` dataclasses). Introduced in session 10
// when the dispatcher port required composing CBMSegmenter + NoveltySegmenter
// + SegmentConsolidation outputs through one type — the three modules
// previously carried field-near-identical local `Segment` structs, now
// aliased to this canonical definition via `using Segment = ...;` at class
// scope.
//
// Field conventions diff vs Python:
//   * C++ field order is primitives first, string last (idiomatic); Python
//     uses declaration order (start, end, label, confidence, cluster_id).
//     Field *set* + *semantics* are identical.
//   * Python's `cluster_id: int = 0` (CPython unbounded int) maps to C++
//     `int` — values are tiny (≤ 12 on the 10-track corpus, hard cap via
//     `kMaxSegments=12` in SegmentConsolidation). NoveltySegmenter's
//     `ClusterResult::clusterIds` vector stays `std::int64_t` (raw sklearn
//     output); the cast to `int` happens at Segment construction site.
//   * `confidence` default is 0.0 (matches Python's required-field semantics
//     + NoveltySegmenter + SegmentConsolidation). CBMSegmenter's 0.8
//     placeholder is set explicitly at construction (CBMSegmenter.cpp:599).
//
// StructureResult adds one C++-only field: `DispatchPath path`. Python
// does not surface which branch fired (CBM vs novelty) — production code
// infers it from `embeddings is None`. The explicit enum makes the
// session-10 parity test a single equality check instead of a null-vs-not
// heuristic.
struct Segment
{
    double      start      = 0.0;
    double      end        = 0.0;
    double      confidence = 0.0;
    int         cluster_id = 0;
    std::string label;
};

enum class DispatchPath
{
    CBM,
    Novelty,
};

struct StructureResult
{
    std::vector<Segment>  segments;
    std::vector<double>   boundaries;
    std::optional<double> bpm;
    DispatchPath          path = DispatchPath::Novelty;
};

} // namespace reamix::analysis
