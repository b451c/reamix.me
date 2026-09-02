#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <set>
#include <utility>
#include <vector>

#include "render/Renderer.h"

// RemixOutput — output of RemixPipeline (stages 6-7-8 of the original
// AnalyzeWorker: CleanOptimizer + Renderer + WAV write).
//
// Per-key tmp WAV: each cached RemixOutput owns its own tmp WAV file at
// `juce::File::tempDirectory / "reamix_<keyHash>.wav"` so concurrent cache
// entries don't collide. Plugin destruction deletes all of them.
//
// Reference: per ADR-047, this is the RemixOutput data contract that lives
// in `MainComponent::remixCache_` (LRU 20 per source). Lua parity:
// `references/lua-source/socket_commands.lua:40` remix command output.

namespace reamix::ui
{

struct RemixOutput
{
    bool         ok           { false };
    juce::String errorMessage;
    // Sesja 119 (DEV-095): non-fatal note shown next to "Remix ready"
    // (e.g. queued blocks shorter than 2 beats were skipped).
    juce::String warningMessage;

    // Inputs that produced this output (echoed for cache-key construction
    // and Insert-time staleness detection).
    juce::String sourcePath;
    // ADR-056 (sesja 66) — REAPER MediaItem GUID at the time work was
    // dispatched. Echoed so handleRemixComplete can rebuild the same
    // composite cache key kickRemixPipeline used for the lookup, even if
    // the user has since switched to a different item with the same
    // sourcePath. Empty for synthetic / non-REAPER call paths (parity tests).
    juce::String itemGuid;
    double       targetSec      { 0.0 };
    double       regionStartSec { 0.0 }; // 0.0 + 0.0 means no region
    double       regionEndSec   { 0.0 };
    int          variation      { 0 };

    // DEV-033 / ADR-054 — actual region bounds the algorithm chose after
    // soft-boundary search (downbeat-preferred snap within ±splice_flex_beats
    // of user's regionStartSec/EndSec). Used by Insert side to split the
    // timeline at these positions instead of user's region bounds —
    // eliminates content-jump at pre/post-region item boundaries by making
    // the splits land where the algorithm actually starts/ends the loop.
    // Equal to regionStartSec/EndSec when no region (Duration mode) or when
    // legacy closest-beat snap was used.
    double       actualRegionStartSec { 0.0 };
    double       actualRegionEndSec   { 0.0 };
    // ADR-057 sesja 68 step 2e — boundary fade overlap. RemixPipeline extended
    // first/last EditClip source ranges by these amounts so Insert can overlap
    // pre/post-region with first/last WAV clip and equal-power crossfade
    // sample-exact content. 0.0 when not Region mode.
    double       boundaryLeadInSec    { 0.0 };
    double       boundaryLeadOutSec   { 0.0 };
    // Echoed for cache-key reconstruction in handleRemixComplete (so the
    // RemixCacheLRU::insert key matches the kickRemixPipeline lookup key).
    std::set<std::pair<int,int>> blockedTransitions;

    // ADR-115 P3 (sesja 123) — Edit density hash echoed from
    // RemixPipeline::Input (hashEditDensity). handleRemixComplete uses it to
    // match the kickRemixPipeline lookup key. 0 = mode default.
    juce::uint64 editDensityHash { 0 };
    // DEV-101 (sesja 123) — identity of the kick: the UI mode the remix was
    // computed for (ModeTabs::Mode as int, -1 = harness / parity) and the
    // two cache-key hashes taken at kick time. handleRemixComplete stores
    // under exactly this key and applies the result to the UI only when
    // the mode still matches, so an auto Duration remix landing after a
    // tab switch never paints the Blocks / Region tab.
    int          uiMode             { -1 };
    juce::uint64 blocksHash         { 0 };
    juce::uint64 qualityWeightsHash { 0 };

    // Remix metadata (stage 6+7).
    int                 nTransitions    { 0 };
    std::vector<double> transitionTimesSec;
    double              remixDurationSec { 0.0 };

    // Per-transition diagnostic data, parallel to transitionTimesSec.
    // Populated in RemixPipeline::run from RemixPath::transition_metadata
    // (key "quality_score" + "energy_diff_db") and bundle.structure.segments
    // lookup at beat-time. Empty labels in auto modes (ADR-044 skips
    // structure analysis); populated when Block Assembly user labels land.
    // Lua parity: server/handlers/_remix.py:154-185 transition_details.
    std::vector<float>        transitionQualities;     // 0..1
    std::vector<int>          transitionFromBeats;     // path-source beat idx
    std::vector<int>          transitionToBeats;       // path-dest beat idx
    std::vector<float>        transitionEnergyDiffsDb; // dB
    // DEV-087 (sesja 122): the overlap the edit plan applies at this splice
    // ("resolved_overlap_sec") and whether the onset-anchor geometry was
    // accepted ("anchor_overlap_samples" present). Harness CSV columns.
    std::vector<float>        transitionOverlapSec;
    std::vector<int>          transitionAnchorAccepted;
    std::vector<juce::String> transitionFromLabels;
    std::vector<juce::String> transitionToLabels;
    // Sesja 120 (DEV-099): 1 when a Blocks junction found no clean cut and
    // was spliced at the authored boundary ("junction_fallback" metadata,
    // sesja 119 DEV-094); 0 otherwise. Drives the red marker / pill.
    std::vector<int>          transitionFallbacks;
    // Sesja 120: Blocks junction index of each transition ("junction_idx"
    // metadata; -1 in Duration / Region) so the seam pills map by junction
    // and not by transition order (a continuation junction has no cut).
    std::vector<int>          transitionJunctions;

    // Per-key tmp WAV file path. Empty when not yet rendered or render
    // failed. RemixCache eviction deletes the file at this path.
    juce::String tmpWavPath;

    // Edit plan (stage 7) — consumed by step-9 Insert per ADR-046. Each
    // clip becomes a MediaItem on the source track with its take pointing
    // at the original audio file (D_STARTOFFS = sourceStartSec), overlapping
    // the next clip by overlapAfterSec for REAPER's native equal-power
    // crossfade. NOT consumed by Renderer — only by ReaperBridge.
    reamix::render::EditPlan editPlan;
};

} // namespace reamix::ui
