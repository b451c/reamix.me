#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "Theme.h"
#include "analysis/FeatureExtractor.h"
#include "analysis/StructureResult.h"
#include "remix/TransitionCost.h"

// AnalysisBundle — output of AnalyzePipeline (stages 1-5 of the original
// AnalyzeWorker). Holds everything RemixPipeline (stages 6-7-8) needs to
// produce a remix without re-running the expensive analysis.
//
// Lifetime: owned by `MainComponent::analysisBundles_` map (one entry per
// source path). Shared via `std::shared_ptr` so a RemixPipeline thread can
// keep its input alive even if the cache evicts the entry mid-render.
//
// Memory: dominant cost is `stereoNative` (channel-major raw audio at the
// file's native sample rate), ~5 MB / minute / channel for 16-bit-equivalent
// f32. The 22050 Hz mono buffer used by stages 2-3 (BeatDetector +
// FeatureExtractor) is NOT carried — it is discarded after AnalyzePipeline
// finishes.
//
// Reference: per ADR-047, this is the AnalysisBundle data contract that
// closes DEV-025 (analyze/remix split missing). Lua parity:
// `references/lua-source/socket_commands.lua:10` analyze command + Python
// backend cache_helper.

namespace reamix::ui
{

struct AnalysisSegment
{
    double                     startSec { 0.0 };
    double                     endSec   { 0.0 };
    reamix::theme::SegmentKind kind     { reamix::theme::SegmentKind::Verse };
};

struct AnalysisBundle
{
    juce::String sourcePath;

    // ── Audio (stage 1) ──────────────────────────────────────────────
    // Native-rate channel-major buffer fed straight into Renderer at
    // RemixPipeline time. Stages 2-5 use a 22050 Hz mono downmix that is
    // discarded post-analysis (we don't keep two copies of the audio).
    std::vector<float> stereoNative;
    int                nChannels     { 0 };
    int                nativeSr      { 0 };
    std::size_t        nativeSamples { 0 };

    // ── BeatDetector (stage 2) ───────────────────────────────────────
    double              bpm        { 0.0 };
    int                 timeSigNum { 4 };
    std::vector<double> beatTimes;
    std::vector<double> downbeatTimes;
    std::vector<bool>   beatIsDownbeat; // length == beatTimes.size()

    // ── FeatureExtractor (stage 3) — moved in, not copied ────────────
    reamix::analysis::FeatureExtractor::Result feat;

    // ── StructureAnalyzer (stage 4) ──────────────────────────────────
    // ADR-044: empty default on auto path. Block Assembly mode (step 8,
    // session 61) will populate user labels here.
    reamix::analysis::StructureResult structure;

    // ── TransitionCost (stage 5) ─────────────────────────────────────
    reamix::remix::TransitionCostResult tc;

    // ── UI-friendly segment view ─────────────────────────────────────
    // Computed once at AnalyzePipeline completion from `structure.segments`,
    // mapping label strings to SegmentKind enum (mapLabel helper). Empty in
    // auto mode per ADR-044; populated when Block Assembly user labels land.
    std::vector<AnalysisSegment> uiSegments;
};

using AnalysisBundlePtr = std::shared_ptr<AnalysisBundle>;

} // namespace reamix::ui
