#include "StructureAnalyzer.h"

#include <utility>
#include <vector>

#include "CBMSegmenter.h"
#include "NoveltyFeatures.h"
#include "NoveltySegmenter.h"
#include "SegmentConsolidation.h"

namespace reamix::analysis {

namespace {

// Rebuild boundaries per structure_analyzer.py L152-155 / L174-177:
//   `[segments[0].start] + [seg.end for seg in segments]` as float64.
// Empty input handled CBM-style (`[0.0, duration]`, Python L155). Novelty
// branch's empty case is handled at the call site (keeps pre-consolidation
// boundaries from `_find_boundaries`); not exercised on the 10-track corpus.
std::vector<double> rebuildBoundaries(const std::vector<Segment>& segments,
                                      double                       duration)
{
    if (segments.empty()) {
        return {0.0, duration};
    }
    std::vector<double> out;
    out.reserve(segments.size() + 1);
    out.push_back(segments.front().start);
    for (const auto& s : segments) {
        out.push_back(s.end);
    }
    return out;
}

} // namespace

StructureResult
StructureAnalyzer::analyze(const float*  y,
                           std::size_t   nSamples,
                           int           sr,
                           double        bpm,
                           const double* beatTimes,
                           int           nBeats,
                           const double* downbeats,
                           int           nDownbeats,
                           const float*  beatFeatures,
                           int           nFeat)
{
    // PARITY: structure_analyzer.py L119 `duration = len(y) / sr`. CPython's
    // `int / int` returns f64; `static_cast<double>(nSamples) / sr` produces
    // the bit-identical value (verified by session-9 `existing=890 drift=0`
    // on all consolidation goldens, which depend on this exact value).
    const double duration = static_cast<double>(nSamples) /
                            static_cast<double>(sr);

    // --- CBM predicate + attempt -----------------------------------------
    // Python L127-133:
    //   if (not fast_mode AND beat_times is not None AND downbeats is not None
    //       AND len(downbeats) >= 4 AND beat_features is not None):
    //     cbm_segments = cbm_analyze(...)
    const bool cbmEligible = (nBeats       >  0
                              && nDownbeats   >= kCbmMinDownbeats
                              && beatFeatures != nullptr);

    if (cbmEligible) {
        auto cbm = CBMSegmenter::cbmAnalyze(
            beatFeatures, nBeats, nFeat,
            beatTimes, nBeats,
            downbeats, nDownbeats,
            y, nSamples, sr);

        // Python L142: `if cbm_segments and len(cbm_segments) >= 5`. C++
        // `cbm.success` mirrors Python's truthiness on Optional[List[dict]]
        // (None or [] → False; non-empty → True). The size check is the
        // separate output-count gate that splits the corpus 3/7 (CBM/novelty).
        if (cbm.success &&
            cbm.segments.size() >= static_cast<std::size_t>(kCbmMinSegments))
        {
            auto consolidated = SegmentConsolidation::consolidate(
                cbm.segments, duration);
            auto boundaries   = rebuildBoundaries(consolidated, duration);
            return StructureResult{
                std::move(consolidated),
                std::move(boundaries),
                bpm,
                DispatchPath::CBM,
            };
        }
    }

    // --- Novelty fallback ------------------------------------------------
    // Python L161-182. Pipeline: NoveltyFeatures::extract → computeSsm →
    // computeNovelty → findBoundaries → computeSegmentEmbeddings →
    // clusterSegments → createSegments → SegmentConsolidation::consolidate.
    const auto features = NoveltyFeatures::extract(y, nSamples, sr);

    int nTds = 0;
    const auto ssm     = NoveltySegmenter::computeSsm(features, kStandardStride, nTds);
    const auto novelty = NoveltySegmenter::computeNovelty(ssm, nTds);
    const auto bnd     = NoveltySegmenter::findBoundaries(
        novelty, duration, kMinSegmentDuration);

    int nSegs = 0;
    int nFt   = 0;
    const auto embeddings = NoveltySegmenter::computeSegmentEmbeddings(
        features, bnd.boundaries, sr, kHopLength, nSegs, nFt);
    const auto cluster = NoveltySegmenter::clusterSegments(
        embeddings, nSegs, nFt);

    // createSegments takes `const std::vector<float>& audio`. Wrap once;
    // the copy is O(nSamples) but happens only on the novelty branch.
    const std::vector<float> audioVec(y, y + nSamples);
    auto segments = NoveltySegmenter::createSegments(
        bnd.boundaries, cluster.clusterIds, duration, novelty, audioVec, sr);

    auto consolidated = SegmentConsolidation::consolidate(segments, duration);

    // Python L173-177 novelty branch: rebuild boundaries ONLY if segments
    // non-empty after consolidation; otherwise leave the pre-consolidation
    // boundaries returned by _find_boundaries. The 10-track corpus never
    // produces empty consolidated segments, but we mirror Python's logic
    // for forward compatibility.
    std::vector<double> boundaries;
    if (!consolidated.empty()) {
        boundaries = rebuildBoundaries(consolidated, duration);
    } else {
        boundaries = bnd.boundaries;
    }

    return StructureResult{
        std::move(consolidated),
        std::move(boundaries),
        bpm,
        DispatchPath::Novelty,
    };
}

} // namespace reamix::analysis
