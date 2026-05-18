#pragma once

#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <utility>

#include "AnalysisBundle.h"
#include "RemixOutput.h"
#include "remix/Quality.h"  // QualityWeights for calibration override (ADR-058)
#include "UserBlock.h"

#include <map>
#include <vector>

// RemixPipeline — runs phase-4-second-half + phase-5 + WAV write on a
// background juce::Thread. Reads an AnalysisBundle (produced by
// AnalyzePipeline) and emits a RemixOutput.
//
// Stages owned by this worker (per ADR-047 § 2):
//   6. CleanOptimizer(bundle.tc + bundle.feat).remix(target, region, blocked, variation)
//   7. Renderer(bundle.stereoNative @ nativeSr, beatTimes, cfg).render(path)
//   8. juce::WavAudioFormat writer → tmp WAV at per-key path.
//
// Triggers (per ADR-047 § 3):
//   - Slider stop debounced 100 ms.
//   - Update button (force fresh variation on next session — session 56
//     scope keeps Update mapped to "kick RemixPipeline at current target").
//   - Insert button when `currentRemix_` is missing or stale (target
//     changed since last render).
//
// Cancellation: threadShouldExit() polled between stages and inside the
// renderer's clip-segment loop where possible. ~200-500 ms of wasted work
// possible if cancelled mid-Optimizer; negligible vs UX gain.

namespace reamix::ui
{

class RemixPipeline : public juce::Thread
{
public:
    struct Input
    {
        AnalysisBundlePtr     bundle;            // shared with cache; outlives worker.
        // ADR-056 (sesja 66) — REAPER MediaItem GUID at dispatch time. Echoed
        // into RemixOutput.itemGuid so handleRemixComplete builds the same
        // composite cache key kickRemixPipeline used for the lookup.
        juce::String          itemGuid;
        double                targetDurationSec  { 0.0 };
        std::optional<double> regionStartSec;    // session 59 wires RegionPanel
        std::optional<double> regionEndSec;
        std::set<std::pair<int,int>> blockedTransitions; // session 57 wires SpliceMarker
        int                   variation { 0 };           // session 57 wires Update / Try-different

        // ADR-051 (sesja 61) — Block Assembly mode. Active when
        // userBlocks non-empty AND queue.size() ≥ 2. Mutually exclusive
        // with regionMode (asserted at run() entry).
        std::vector<reamix::ui::UserBlock> userBlocks;
        std::vector<int>                   userBlocksQueue;
        std::map<int,int>                  junctionVariations;
        // ADR-057 (sesja 68) — Splice flexibility (Tight/Medium/Loose) UI
        // removed; default W=8 kept as interim hardcoded for Blocks branch
        // until DEV-040 β-model supersedes junction ±W search entirely.
        // Region branch ignores this field (uses splice_flex_beats=0 → legacy
        // argminAbsDiff path on user-selection edges).
        int                                spliceFlexBeats     { 8 };
        double                             driftPenaltyWeight  { 0.10 };

        // Sesja 71 (ADR-058 + ADR-059) — calibration override.
        // std::nullopt = use kDefaultQualityWeights (production + parity tests
        // path). Calibration harness (tools/calibration_harness) sets this to
        // a custom QualityWeights vector to sweep weight space during DEV-028
        // Bayesian optimization.
        //
        // SCOPE: this field affects ONLY Block Assembly + Region modes —
        // they build cost inputs at REMIX-time inside RemixPipeline::run,
        // so the override propagates cleanly into bin.quality_weights /
        // rcin.quality_weights (null = default). Duration mode is DIFFERENT:
        // the TC W-matrix is computed ONCE at ANALYZE-time inside
        // AnalyzePipeline.cpp:272 and cached in bundle.tc — this override
        // has NO EFFECT on cached W. Duration calibration requires a separate
        // code path: harness must call computeTransitionCosts directly with
        // custom weights and substitute the result for bundle.tc.
        // See phases/phase-6-ui/dev-028-lessons.md (sesja-71a [ARCHITECTURE]
        // entry) for full context.
        std::optional<reamix::remix::QualityWeights> qualityWeightsOverride;

        // ADR-080 RESCOPE + ADR-083 (sesja 92) + ADR-084 (sesja 93) —
        // AuditionBar 4-slider params. Defaults bit-exact replicate current
        // production behavior:
        //   - harmonic_vs_timbre  = 0.0  → Tone slider OFF → blend bypassed.
        //   - edit_length_jump_scale = 1.0 → Edit Length neutral (per
        //     ADR-084 multiplicative scale; supersedes sesja-92 additive).
        //   - edit_length_slider = 50   → cache-hash slider int [0..100].
        //   - allow_pm_seconds   = 5.0  → Allow ± slider default.
        //   - min_cut_beats      = 0    → Min cut OFF → legacy COOLDOWN_BARS × TS.
        //
        // SCOPE: same scope semantics as `qualityWeightsOverride`. Block
        // Assembly + Region modes consume directly from these fields at
        // REMIX-time. Duration mode: edit_length / allow_pm / min_cut are
        // RUN-time DP params (no W-matrix re-build needed) — Tone slider
        // requires W-matrix re-build (Path A sesja 93 re-computes TC
        // locally in RemixPipeline Duration branch when
        // qualityWeightsOverride.harmonic_vs_timbre > 0).
        double harmonic_vs_timbre     = 0.0;
        double edit_length_jump_scale = 1.0;   // ADR-084: multiplicative on jump cost
        int    edit_length_slider     = 50;    // [0..100] for cache hash; 50 = bit-exact
        double allow_pm_seconds       = 5.0;
        int    min_cut_beats          = 0;     // 0 = legacy compute COOLDOWN_BARS × TS

        // ADR-081 (sesja 96) — β-model candidate-space expansion for Block
        // Assembly junctions. Default flipped to true sesja 96 close per
        // user-confirmed listening test (smoke-in-REAPER vs prior version
        // memory). Legacy ±W=8 path remains accessible via flag=false (kept
        // for parity tests + rollback safety). No effect on Duration /
        // Region modes (Block branch only).
        bool   block_assembly_beta    = true;
    };

    using ProgressCb = std::function<void (juce::String label, double p01)>;
    using CompleteCb = std::function<void (RemixOutput out)>;

    RemixPipeline (Input        in,
                   ProgressCb   onProgress,
                   CompleteCb   onComplete,
                   juce::String tmpWavPath);  // per-key file owned by RemixCache

    ~RemixPipeline() override;

    void run() override;

    std::shared_ptr<std::atomic<bool>> aliveFlag() { return alive_; }

private:
    void postProgress (const juce::String& step, double p01);
    void postCompletion (RemixOutput out);

    Input        in_;
    ProgressCb   onProgress_;
    CompleteCb   onComplete_;
    juce::String tmpWavPath_;
    std::shared_ptr<std::atomic<bool>> alive_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RemixPipeline)
};

} // namespace reamix::ui
