#pragma once

#include <optional>
#include <cstdint>
#include <string>
#include <vector>

namespace reamix::analysis {

// Section list of a track: the output type of `SectionClassifier::decode`
// (sesja 121, DEV-098) and the `segments` input of TransitionCost / Region /
// Block Assembly.
//
// Originally the port of references/python-source/analysis/structure_analyzer.py
// L34-53 (`Segment` + `StructureResult` dataclasses) shared by the CBM /
// novelty / consolidation stack; that stack was skipped since ADR-044 and
// deleted in sesja 122 (ADR-115 E6), the type stays as the contract.
//
// Field conventions (kept from the port so the disk cache format and the
// remix consumers are unchanged):
//   * `cluster_id` is `int` (values are tiny); `confidence` defaults to 0.0.
//   * `label` is a free string; the UI maps it through SegmentKind
//     (AnalysisBundle::mapLabel) and `kindAbbreviation`.
//   * `DispatchPath` is a legacy byte kept only for the disk-cache layout
//     (always Novelty since the stack removal).
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
