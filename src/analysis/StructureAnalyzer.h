#pragma once

#include <cstddef>

#include "StructureResult.h"

namespace reamix::analysis {

// Structure-segmentation dispatcher. Port of
// references/python-source/analysis/structure_analyzer.py L83-191
// (`StructureAnalyzer._analyze`). Chooses between CBM bar-level DP
// (standard mode, quality path) and novelty peak-picking + spectral
// clustering (fast mode OR CBM prerequisites unmet). Both branches run
// `SegmentConsolidation::consolidate` before building the final boundaries
// array and returning a `StructureResult`.
//
// Session-10 decisions (ported from HANDOVER):
//   1. Unified Segment/StructureResult types live in StructureResult.h.
//      CBMSegmenter / NoveltySegmenter / SegmentConsolidation::Segment are
//      now `using` aliases to `reamix::analysis::Segment`.
//   2. `bpm` is a REQUIRED caller input. Python falls back to
//      `librosa.beat.beat_track(y)` when bpm is None; we don't port librosa
//      tempo estimation. Production path always carries bpm from phase-1
//      BeatDetector. The value is passed through to `StructureResult::bpm`
//      unmodified — no tempo estimation happens inside the dispatcher.
class StructureAnalyzer
{
public:
    // PARITY: references/python-source/analysis/structure_analyzer.py L71,
    // L66, L217.
    static constexpr int    kSampleRate         = 22050;
    static constexpr int    kHopLength          = 512;
    static constexpr int    kStandardStride     = 4;     // L217 non-fast path
    static constexpr double kMinSegmentDuration = 8.0;   // __init__ L66 default
    static constexpr int    kCbmMinDownbeats    = 4;     // L131 `>= 4`
    static constexpr int    kCbmMinSegments     = 5;     // L142 `>= 5`

    // Full dispatcher. Returns StructureResult{segments, boundaries, bpm, path}.
    //
    // Inputs:
    //   y / nSamples / sr  — mono float32 audio (sr normally == kSampleRate).
    //   bpm                — required; passed through to StructureResult::bpm.
    //   beatTimes / nBeats — beat times (seconds, f64) from phase-1 BeatDetector.
    //   downbeats / nDownbeats — downbeat times aligned to the same beat source.
    //   beatFeatures / nFeat   — row-major (nBeats × nFeat) f32 beat-synced
    //                            features from FeatureExtractor / orchestrator.
    //                            Pass nullptr to force novelty fallback.
    //
    // Dispatcher predicate (Python L127-142):
    //   CBM fires iff nBeats > 0 AND nDownbeats ≥ 4 AND
    //                 beatFeatures != nullptr AND CBM output ≥ 5 segments.
    //   Else: novelty fallback.
    //
    // Both branches run SegmentConsolidation::consolidate before building
    // the final boundaries array.
    static StructureResult
    analyze(const float*  y,
            std::size_t   nSamples,
            int           sr,
            double        bpm,
            const double* beatTimes,
            int           nBeats,
            const double* downbeats,
            int           nDownbeats,
            const float*  beatFeatures,
            int           nFeat);
};

} // namespace reamix::analysis
