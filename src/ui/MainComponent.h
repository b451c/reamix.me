#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "AnalysisBundle.h"
#include "AnalyzePipeline.h"
#include "BlockAssemblyPanel.h"
#include "DurationPanel.h"
#include "FilePeaksProvider.h"
#include "HeaderBar.h"
#include "LookAndFeelReamix.h"
#include "ModeTabs.h"
#include "PreviewController.h"
#include "RemixCache.h"
#include "RemixOutput.h"
#include "AuditionBar.h"

#include "AdvancedWeightsPanel.h"
#include "AdvancedWeightsWindow.h"
#include "RemixPipeline.h"
#include "SettingsPopover.h"
#include "SupportPopover.h"
#include "SourcePanel.h"
#include "StatusBar.h"
#include "Tooltip.h"
#include "TransportBar.h"
#include "UserBlock.h"
#include "CustomKindRegistry.h"
#include "BlockKindPickerPopover.h"  // for CustomKindAction enum
#include "WaveformView.h"

#include "analysis/BeatDetector.h"
#include "reaper/ReaperBridge.h"

// MainComponent — phase-6 root panel.
//
// Pipeline architecture (ADR-047, session 56): two background workers
// mirror Lua's analyze/remix command split.
//
//   AnalyzePipeline (juce::Thread) — stages 1-5 (load + beats + features +
//     structure-skip + cost matrix). Cached per source path in
//     `analysisBundles_` (shared_ptr so RemixPipeline can outlive cache
//     replacement).
//
//   RemixPipeline (juce::Thread) — stages 6-7-8 (Optimizer + Renderer +
//     WAV write). Cached per (source, target_q50ms, region, blockedHash,
//     variation) tuple in `remixCache_` (LRU 20 entries per source). Each
//     cached entry owns its own tmp WAV file.
//
// Triggers:
//   - Analyze button → AnalyzePipeline + auto-chain RemixPipeline at current
//     target via `followupAfterAnalysis_` queue.
//   - Slider stop (debounced 100 ms) → RemixPipeline at current target,
//     OR queue followup if AnalyzePipeline still in flight.
//   - Update button → RemixPipeline (variation upgrade lands session 57).
//   - Insert button → uses currentRemix_ if fresh; else kicks RemixPipeline
//     + queues Insert followup (Lua start_export pattern).
//
// State machine mirrors Lua `state.analyzing / state.analysis_complete /
// state.pending_operation / state.followup_after_analysis` 1:1.

class MainComponent : public juce::Component,
                      private juce::Timer
{
public:
    MainComponent();
    ~MainComponent() override;

    std::function<void()> onToggleDock;
    std::function<bool()> onIsDocked;

    std::function<void(bool)> onShowBeatsToggled;
    std::function<bool()>     onIsShowBeats;

    void paint   (juce::Graphics&) override;
    void resized() override;

    // Sesja 64 BUG-8 — invoked from main.cpp accelerator hook to dispatch
    // keyboard shortcuts that REAPER would otherwise eat in docked mode.
    // Returns true if the key was handled (= main.cpp returns 1 to REAPER).
    bool handleAccelKey (int virtualKeyCode, bool ctrl, bool shift);

private:
    // ── 100 ms selected-item poll ────────────────────────────────────
    void timerCallback() override;
    void applySelectedItem (const juce::String& name,
                            double itemDurationSec,
                            double sourceFileDurationSec,
                            const juce::String& sourcePath,
                            void*  itemPtr);
    void applyEmptyState();

    // ── Analysis state machine (Lua parity) ──────────────────────────
    enum class AnalysisState { Idle, Analyzing, Complete };

    struct PendingRemixOp
    {
        double                       targetSec      { 0.0 };
        std::optional<double>        regionStartSec;
        std::optional<double>        regionEndSec;
        std::set<std::pair<int,int>> blockedTransitions;
        int                          variation      { 0 };
    };

    struct PendingFollowup
    {
        enum class Kind { Remix, Insert };
        Kind            kind;
        PendingRemixOp  op; // populated for both Remix and Insert (Insert
                            // kicks RemixPipeline first if currentRemix_ stale)
    };

    AnalysisState                    analysisState_ { AnalysisState::Idle };
    std::optional<PendingFollowup>   followupAfterAnalysis_;

    // ── Analyze pipeline (stages 1-5) ────────────────────────────────
    void startAnalyze();      // user click on Analyze
    void handleAnalysisComplete (juce::String workerPath,
                                 reamix::ui::AnalysisBundlePtr bundle,
                                 juce::String error);

    // ── Remix pipeline (stages 6-7-8) ────────────────────────────────
    void kickRemixPipeline (const PendingRemixOp& op);
    void handleRemixComplete (reamix::ui::RemixOutput out);

    // ── Mode/region resolution (sesja 60) ────────────────────────────
    // Centralised: reads appMode_ + selectedRange_ + auto-flag state, writes
    // currentRegion_ + pushes to ModeTabs/DurationPanel, optionally kicks a
    // RemixPipeline run when the region changed. Idempotent — safe to call
    // from any of the inputs (mode click, drag-select, timerCallback poll).
    void recomputeRegionState();

    // ADR-050 Filozofia A — every mode switch (manual click OR auto-flip)
    // resets the waveform back to Source variant + clears selection + resets
    // slider target to origSec. Each mode operates on the original audio,
    // independent of any pending un-inserted edits from another mode.
    void resetMaterialView();

    // Sesja 65 — single-call clear for currentRemix_ + currentRemixMode_.
    // Use everywhere instead of bare currentRemix_.reset() so the mode
    // tag never desyncs from the optional payload.
    void clearCurrentRemix();

    // Sesja 65 — try to restore the entering mode's last computed remix
    // by looking up remixCache_ with the per-mode cache key. Used by
    // modeTabs_.onChange so each tab behaves like an independent
    // workspace. Returns true on cache hit (UI updated), false otherwise.
    bool tryRestoreModeRemix (reamix::ui::ModeTabs::Mode m);

    // Sesja 65 — extracted from kickRemixPipeline so both the pipeline
    // kick site and the modeTabs_.onChange restore site compute the same
    // blocksHash. Returns 0 if Blocks state is too small for a valid
    // arrangement (fewer than 1 block or fewer than 2 queue entries).
    juce::uint64 computeBlocksHash() const;

    // ── Slider debounce ──────────────────────────────────────────────
    class SliderDebounceTimer;
    std::unique_ptr<SliderDebounceTimer> sliderDebounceTimer_;
    PendingRemixOp                       pendingSliderOp_;
    void sliderDebounceFired();

    // ── UI rendering helpers ─────────────────────────────────────────
    // Apply analysis-side data to UI (BPM, beats, segments, peak provider
    // re-source). Called both on fresh AnalyzePipeline completion AND on
    // cache restore from `analysisBundles_` when user returns to a source.
    void applyAnalysisToUi (const reamix::ui::AnalysisBundle& bundle,
                            const juce::String& itemName,
                            bool fromCache);
    // Apply remix-side data to UI (transitionTimes for splice markers in
    // session 57; transport state Ready; tmpWavPath_ for Preview/Insert).
    // Called on RemixPipeline completion AND on remixCache_ hit.
    void applyRemixToUi (const reamix::ui::RemixOutput& remix);

    // ── Lazy ONNX init ───────────────────────────────────────────────
    bool ensureBeatDetectorReady (juce::String& outErrorMessage);

    // ── Components (unchanged by ADR-047) ────────────────────────────
    reamix::ui::LookAndFeelReamix lookAndFeel_;
    reamix::ui::HeaderBar         headerBar_;

    reamix::ui::SourcePanel       sourcePanel_;

    reamix::BeatDetector          beatDetector_;
    std::mutex                    beatDetectorLoadMutex_;
    bool                          beatDetectorLoaded_ { false };
    juce::String                  beatDetectorLoadError_;

    std::unique_ptr<juce::Thread> preWarmThread_;
    std::atomic<bool>             preWarmAborted_ { false };

    // ── Workers ──────────────────────────────────────────────────────
    std::unique_ptr<reamix::ui::AnalyzePipeline> analyzePipeline_;
    std::unique_ptr<reamix::ui::RemixPipeline>   remixPipeline_;

    // Zombie-reaping vectors per worker type. Cancelled-but-still-running
    // pipelines move here; timerCallback erases ones whose threads have
    // exited. Pattern mirrors session-44 fix for AnalyzeWorker UAF (DEV-015).
    std::vector<std::unique_ptr<reamix::ui::AnalyzePipeline>> stoppingAnalyze_;
    std::vector<std::unique_ptr<reamix::ui::RemixPipeline>>   stoppingRemix_;
    void reapStoppedWorkers();

    juce::String activeAnalyzePath_; // path the current AnalyzePipeline owns
    std::map<juce::String, double>      lastAnalyzeProgress_; // resume on item return
    std::map<juce::String, juce::String> lastAnalyzeStage_;    // sesja 64 — stage label per path

    // ── Cache ────────────────────────────────────────────────────────
    // Per-source AnalysisBundle (shared with in-flight RemixPipeline workers).
    std::map<juce::String, reamix::ui::AnalysisBundlePtr> analysisBundles_;

    // Bounded-LRU RemixOutput per (source, target, region, blocked, variation).
    // Each entry owns its tmp WAV file at `entry.output.tmpWavPath`.
    reamix::ui::RemixCacheLRU remixCache_ { 20 }; // ADR-047 § 1: LRU 20 / source

    // Latest remix shown in UI / used by Insert. Mirrors Lua
    // `state.remix_audio` + `state.transition_times` aggregated. None until
    // first RemixPipeline completes for current source.
    std::optional<reamix::ui::RemixOutput> currentRemix_;

    // Sesja 65 — mode that produced currentRemix_. Drives symmetric peek-
    // and-restore in modeTabs_.onChange: re-entering a tab whose mode
    // matches restores the remix view instead of resetting per ADR-050.
    // Reset alongside currentRemix_ via clearCurrentRemix() helper.
    std::optional<reamix::ui::ModeTabs::Mode> currentRemixMode_;

    // ── Item snapshot ────────────────────────────────────────────────
    juce::String currentSourcePath_;
    double       currentSourceDurationSec_ { 0.0 };

    // Tmp WAV path currently feeding PreviewController + ReaperBridge.
    // Always equals `currentRemix_->tmpWavPath` when present.
    juce::String tmpWavPath_;

    // ADR-056 (sesja 66) — composite identity for per-item state maps.
    // Pre-fix (sourcePath only) leaked state across distinct REAPER items
    // sharing the same audio file: original whole item vs post-Region-Insert
    // pre/post-region pieces vs duplicated copies all collided. Composite
    // (sourcePath, itemGuid) gives each REAPER MediaItem its own state slot.
    // itemGuid stable across REAPER undo (verified pre-flight check #1).
    struct ItemKey
    {
        juce::String sourcePath;
        juce::String itemGuid;

        bool operator< (const ItemKey& o) const
        {
            if (sourcePath != o.sourcePath) return sourcePath < o.sourcePath;
            return itemGuid < o.itemGuid;
        }
        bool operator== (const ItemKey& o) const
        {
            return sourcePath == o.sourcePath && itemGuid == o.itemGuid;
        }
    };

    // Helper: ItemKey for the currently-attached MediaItem. Empty when no
    // item attached (currentSourcePath_ + currentItemGuid_ both empty).
    ItemKey currentItemKey() const { return { currentSourcePath_, currentItemGuid_ }; }

    // Per-(source, item) target memory (preserves slider position across
    // item switches). ADR-056 — keyed by ItemKey so original item, fragment
    // pieces, and duplicate copies of the same audio file each get their
    // own slider state instead of cross-contaminating.
    std::map<ItemKey, double> targetByPath_;

    // ── Layout primitives ────────────────────────────────────────────
    class Placeholder : public juce::Component
    {
    public:
        Placeholder (juce::Colour background, juce::Colour bottomBorderColour,
                     juce::String label);
        void paint (juce::Graphics&) override;
    private:
        juce::Colour background_;
        juce::Colour bottomBorder_;
        juce::String label_;
    };

    reamix::ui::ModeTabs            modeTabs_;
    reamix::ui::DurationPanel       durationPanel_;
    reamix::ui::BlockAssemblyPanel  blockAssemblyPanel_;
    reamix::ui::AuditionBar         auditionBar_;       // ADR-080 RESCOPE + ADR-083 sesja 92
    // ADR-097 sesja 107 — advanced weights panel + window are lazy-created
    // on first SettingsPopover "Advanced..." click. Panel is NOT a child of
    // MainComponent; AdvancedWeightsWindow hosts it as setContentNonOwned
    // content. Panel outlives window across open/close cycles so slider
    // state persists in-session even when window is hidden.
    std::unique_ptr<reamix::ui::AdvancedWeightsPanel>  advancedPanel_;
    std::unique_ptr<reamix::ui::AdvancedWeightsWindow> advancedWindow_;
    reamix::ui::TransportBar        transportBar_;
    reamix::ui::StatusBar           statusBar_;

    // Sesja 64 — deferred-busy helper for compute paths. startBusy(label)
    // schedules statusBar_.setBusy(label) after 300 ms threshold; if the
    // compute finishes within the threshold, stopBusy() cancels the timer
    // before the spinner is ever shown (no flicker for short computes).
    class BusyDefer : public juce::Timer
    {
    public:
        BusyDefer (reamix::ui::StatusBar* sb) : sb_ (sb) {}
        void startBusy (juce::String label)
        {
            pending_ = std::move (label);
            startTimer (300);
        }
        void stopBusy ()
        {
            stopTimer();
            if (sb_ && sb_->isBusy()) sb_->clearBusy();
        }
        void timerCallback() override
        {
            if (sb_) sb_->setBusy (pending_);
            stopTimer();
        }
    private:
        reamix::ui::StatusBar* sb_;
        juce::String           pending_;
    };

    BusyDefer busyDeferRemix_   { &statusBar_ };
    BusyDefer busyDeferInsert_  { &statusBar_ };

    std::unique_ptr<reamix::ui::FilePeaksProvider> peaksProvider_;
    // Second peaks provider — points at currentRemix_->tmpWavPath while
    // showing Remix variant. Same FilePeaksProvider type; switch via
    // waveformView_.setPeaksProvider(...) per variant. Avoids duplicate
    // type per memory feedback_think_and_verify_before_writing.md.
    std::unique_ptr<reamix::ui::FilePeaksProvider> remixPeaksProvider_;
    reamix::ui::WaveformView                       waveformView_ { reamix::ui::WaveformView::Variant::Source };

    // Splice rich tooltip (session 57). MainComponent-level child so it can
    // overlay the WaveformView without being clipped.
    reamix::ui::Tooltip                            tooltip_;

    // ADR-084 sesja 93 — JUCE-native tooltip popup for AuditionBar slider
    // hover hints. Without a TooltipWindow instance in the hierarchy,
    // `juce::Slider::setTooltip()` calls store text but never render. This
    // window auto-binds to any setTooltip call in the descendant tree.
    reamix::ui::TransparentTooltipWindow           sliderTooltipWindow_ { this, 600 /* ms */ };

    // Per-source blocked-transition set. Lua state.blocked_transitions
    // persists across slider changes (HANDOVER gotcha "blocking persists
    // across slider changes — Lua wins").
    std::map<juce::String, std::set<std::pair<int,int>>> blockedBySource_;
    // Per-source variation index for ContextMenu "Try different splice".
    // Reset to 0 by ResetToBest. Lua start_remix(variation) parity.
    std::map<juce::String, int>                          variationBySource_;

    reamix::ui::PreviewController previewController_;
    bool swsModalShown_ { false };

    std::optional<reamix::ui::WaveformView::SelectionRange> selectedRange_;

    // Sesja 64 BUG-2 — last waveform click-to-seek position. Space starts
    // preview from this offset when no drag-select range is active. Reset
    // on source change / variant flip / empty state.
    std::optional<double> lastSeekSec_;

    // Sesja 64 BUG-3 — snapshot of editPlan clips taken right before a
    // "Try different splice" kick. handleRemixComplete compares the new
    // clips against this snapshot; if identical → variation alternatives
    // are exhausted, status emits "No more alternatives" and we revert
    // the variation counter.
    std::optional<std::vector<reamix::render::EditClip>> tryDifferentSnapshot_;

    reamix::ui::SettingsPopover   settingsPopover_;
    reamix::ui::SupportPopover    supportPopover_;   // sesja 108 — donation links

    juce::String prevSourcePath_;
    // ADR-056 (sesja 66) — also track previous item GUID so polling
    // detects MediaItem identity changes when sourcePath is unchanged
    // (post-Region-Insert clip click, post-Cmd+Z original restore, etc.).
    // Without this trigger applySelectedItem skips the fresh attach and
    // plugin keeps stale tmpWavPath_ from prior item — BUG-22 root cause.
    juce::String prevItemGuid_;
    bool         hasSelection_ { false };

    // DEV-050 (sesja 100b) — first inserted clip GUID after the most recent
    // successful Insert / Update. Polled in timerCallback by findItemByGuid;
    // null result implies user invoked REAPER undo (Cmd+Z) on the Insert,
    // which fires a one-shot StatusBar notice "Insert undone — Edit to start
    // over". Cleared after the notice fires, on next Insert (overwrites with
    // new GUID), and on applyEmptyState (deselect). Independent of the
    // GroupTracker ExtState entries — those persist across REAPER undo, so
    // they cannot be used as a Cmd+Z signal directly. Empty string = no
    // pending Insert to track.
    juce::String lastInsertedClipGuid_;

    // DEV-052 (sesja 100b) — discoverability hint fired once per
    // MainComponent lifetime when the user enters Region mode with a valid
    // (≥6 s) time-selection. Educates the user that drag-selecting in the
    // waveform refines the Region range. Subsequent Region-mode entries
    // skip the hint to avoid noise. Reset only when MainComponent is
    // re-created (plugin reload).
    bool regionDragSelectHintShown_ { false };

    // DEV-049 (sesja 100b) — Insert-time decoration toggles persisted via
    // ExtState ("reamix.me" SECTION). Both default ON per user smoke
    // choice. Splice markers = REAPER project markers at every splice
    // position (navigation aid, Shift+Arrow). Render region = REAPER
    // region spanning all new clips (quick render via Region Render
    // Matrix without manual time selection). Cleanup of stale reamix
    // markers happens in ReaperBridge::clearReamixMarkersInRange before
    // each Insert / Update so re-running the same Insert doesn't
    // accumulate.
    bool insertSpliceMarkersEnabled_ { true };
    bool insertRenderRegionEnabled_  { true };

    // ── Mode + region state (sesja 60, step 6) ───────────────────
    // Three-mode UX per user decision: user picks the mode by clicking a tab
    // (Duration / Region / Blocks). REAPER time-selection auto-flips the
    // plugin INTO Region mode (with AUTO flag on the tab) but never out — the
    // user's manual choice is sticky until either (a) they click a different
    // tab, or (b) REAPER time-selection changes to a *new* value.
    reamix::ui::ModeTabs::Mode appMode_ { reamix::ui::ModeTabs::Mode::Duration };

    // True when the current Region mode was triggered by REAPER time-selection
    // (vs. user clicking the Region tab). Drives ModeTabs::setAutoFlag.
    bool regionFromAuto_ { false };

    // Snapshot of the last REAPER time-selection the plugin already reacted
    // to. Re-acting requires a *change* — prevents the auto-flip from
    // immediately undoing the user's manual switch back to Duration.
    std::optional<reamix::reaper::TimeSelection> lastRespectedTimeSelection_;

    // Effective region currently driving the panel + pipeline + insert.
    // Resolved by recomputeRegionState() from appMode_ + selectedRange_ +
    // (when auto) lastRespectedTimeSelection_. nullopt when not in Region
    // mode OR Region mode but no valid selection.
    std::optional<reamix::reaper::ItemRegion> currentRegion_;

    // Snapshot of selectedRange_ at the moment a Region remix completed —
    // restored when the user clicks the "↺ Edit region" overlay to return to
    // editing mode (Source variant + scrim re-shown). Plan A — sesja 60.
    std::optional<reamix::ui::WaveformView::SelectionRange> lastRegionSelection_;

    // ── Block Assembly state (sesja 61, ADR-051) ──────────────────────
    // Opaque MediaItem* of the currently-attached source item. Cached at
    // applySelectedItem() time. ADR-052 (sesja 63 BUG-9 fix): NEVER pass
    // this raw pointer to a REAPER API call directly — it can be stale if
    // REAPER invalidated/reused the item between selection and async user
    // gesture (CallOutBox popover, undo, item delete). Always re-resolve
    // through findItemByGuid(currentItemGuid_) immediately before any
    // GetSetMediaItemInfo call. Kept here only for fast-path identity
    // checks (e.g. "did selection change since last tick").
    void* currentItemPtr_ { nullptr };

    // ADR-052 (sesja 63) — stable identity key for the currently-attached
    // item. Captured from getItemGuid(itemPtr) in applySelectedItem.
    // findItemByGuid(currentItemGuid_) re-resolves to a live MediaItem*
    // (or nullptr if REAPER deleted/replaced the item). All P_EXT load/
    // save round-trips MUST go through this re-resolve, not through
    // currentItemPtr_, to prevent SIGSEGV on stale-pointer dereference
    // (BUG-9 root cause: kind picker async callback fires after REAPER
    // freed the item).
    juce::String currentItemGuid_;

    // User-marked sections of the currently-attached source item. Loaded
    // from P_EXT on item attach, persisted on every mutation. Empty when
    // no item is selected. Sorted ascending by startSec (stability for
    // paint, palette ordering, and JSON round-trip determinism).
    std::vector<reamix::ui::UserBlock> userBlocks_;

    // ADR-092 sesja 100c — global per-user custom kind definitions. Loaded
    // from REAPER ExtState "reamix.custom_kinds" at construction; saved on
    // every mutation. Survives REAPER restart, project switch, machine
    // session. Per-user, not per-project.
    reamix::ui::CustomKindRegistry customKindRegistry_;

    // Block Assembly arrangement queue — vector of indices into userBlocks_,
    // in user-chosen order. May contain repeats (same block reused). Cleared
    // on item switch (queue is per-session; not persisted in v1 — could be
    // promoted to P_EXT in a follow-up if user demand surfaces).
    std::vector<int> userBlocksQueue_;

    // ADR-051 phase E — algorithm-resolved splice points for current
    // arrangement. Mirror of WaveformView's userBlockSplices_ for tooltip
    // hover composition (WaveformView's vector is write-only from outside).
    // Populated by phase J on RemixPipeline Blocks-mode completion.
    std::vector<reamix::ui::WaveformView::UserBlockSplice> lastBlockSplices_;

    // ADR-051 phase G — junction variation index per junction (size =
    // userBlocksQueue_.size() - 1). 0 = primary top-K splice, 1+ = re-rolled
    // alternative. Resized on queue mutation; mutated on seam right-click
    // "Try different splice". Phase J passes through to assembleBlocks().
    std::vector<int> junctionVariations_;

    // ADR-052 (sesja 63 BUG-17/18 fix) — per-source-path in-memory snapshot
    // of Block Assembly working state. Survives REAPER deselect (BUG-17:
    // user clicks empty area, plugin should retain the session). Survives
    // Insert (BUG-18: original item destroyed, inserted clip has same
    // sourcePath but different MediaItem* — restoring by sourcePath gets
    // the user back to their work). Out of scope: cross-process persistence
    // (that lives in P_EXT, see persistUserBlocks). 32-entry soft LRU not
    // implemented in v1 — typical session has 1-3 sources, map stays small.
    struct BlocksSessionState
    {
        std::vector<reamix::ui::UserBlock> userBlocks;
        std::vector<int>                   userBlocksQueue;
        std::vector<int>                   junctionVariations;
    };
    std::map<juce::String, BlocksSessionState> blocksSessionByPath_;

    // DEV-034 (sesja 63b) — per-source-path in-memory snapshot of the last
    // Region remix shown in the UI. Stores the region bounds AND the target
    // duration that produced the remix (needed because the auto-fired
    // kickRemixPipeline cache key includes targetSec — without restoring
    // both, the lookup misses and produces a different remix at fresh-
    // default target). Snapshotted in applyRemixToUi when a Region remix
    // lands. Restored in applySelectedItem cache-hit branch and in
    // modeTabs_.onChange Region tab re-entry.
    struct RegionSnapshot
    {
        reamix::reaper::ItemRegion region;
        double                     targetSec { 0.0 };
    };
    // ADR-056 (sesja 66) — keyed by ItemKey: distinct REAPER items sharing
    // the same audio file each get their own region snapshot.
    std::map<ItemKey, RegionSnapshot> lastRegionByPath_;

    // Sesja 65 — per-item "last active mode" tracker. Saved on every user-
    // intent mode change (manual tab click, auto-flip Region/Duration,
    // Region cache-hit restore). Read in applySelectedItem to gate the
    // Region auto-restore: previously the lastRegionByPath_ snapshot alone
    // forced Region tab on every re-attach, even if user was last in
    // Blocks for that item (test 59 user report).
    // ADR-056 (sesja 66) — keyed by ItemKey.
    std::map<ItemKey, reamix::ui::ModeTabs::Mode> lastModeByPath_;

    // Sesja 65 — per-source per-mode slider target. Each mode behaves like
    // an independent workspace: switching tabs restores the slider value +
    // looks up a matching entry in remixCache_ so the previously-computed
    // remix for that mode comes back instantly. Without this, calculating
    // in mode B overwrites currentRemix_ (single optional) and the user
    // loses their mode-A remix on the next tab switch.
    // ADR-056 (sesja 66) — keyed by ItemKey.
    std::map<ItemKey, std::map<reamix::ui::ModeTabs::Mode, double>>
        targetByPathMode_;

    // ADR-080 RESCOPE + ADR-083 (sesja 92) — per-(item, mode) AuditionBar
    // 4-slider params snapshot. DEV-079 sesja 101: keyed by (ItemKey, Mode)
    // so each tab is its own workspace (each mode does something different,
    // sliders in Duration mustn't bleed into Region/Blocks cache key).
    // Populated on user slider drag for the currently-active mode; restored
    // by snapping AuditionBar UI on tab switch. Defaults bit-exact baseline
    // (tone=0.0, editLength=50, allowPm=5, minCut=16).
    struct AuditionParams {
        double tone           = 0.0;
        int    editLength     = 50;
        int    allowPmSeconds = 5;
        int    minCutBeats    = 16;
    };
    std::map<ItemKey, std::map<reamix::ui::ModeTabs::Mode, AuditionParams>>
        auditionParamsByPathMode_;

    // Helper accessor — returns persisted params for (currentItemKey(), appMode_)
    // or default-constructed AuditionParams (= bit-exact baseline) when no
    // entry exists. Used by kickRemixPipeline to populate RemixPipeline::Input
    // and by tryRestoreModeRemix to compute the same auditionHash that was
    // stored under at the mode-specific compute time.
    AuditionParams currentAuditionParams() const;

    // ADR-087 STATUS UPDATE 1 D9 (sesja 98) + DEV-079 sesja 101 + ADR-097
    // sesja 107 — per-(item, mode) advanced weights QualityWeights snapshot.
    // Mirrors auditionParamsByPathMode_ per-mode pattern so each tab is its
    // own workspace (slider movement in mode B does not corrupt cache lookup
    // for mode A's compute). Schema_version=2 JSON Save format records
    // `mode_evaluated` per ADR-087 + ADR-098. Defaults bit-exact baseline
    // (kDefaultQualityWeights). Per ADR-097 (sesja 107), the panel + map ship
    // in the production dylib as opt-in opened from gear → Advanced... button.
    std::map<ItemKey, std::map<reamix::ui::ModeTabs::Mode, reamix::remix::QualityWeights>>
        qualityWeightsByPathMode_;

    // DEV-080 sesja 108 — host-side mirror of the user-set defaults that the
    // Advanced panel persists via ExtState "reamix.me" / "advanced_defaults"
    // (iter-10 "Set as defaults"). Loaded once at ctor via
    // loadAdvancedDefaultsFromExtState(); fall-back source for
    // currentQualityWeights() when no per-(item, mode) override exists. Pre-
    // fix this slot read compile-time reamix::remix::kDefaultQualityWeights,
    // which decoupled tick markers (panel's currentDefaultRaw_, restored from
    // ExtState on lazy-create) from slider thumbs (which followed the host's
    // compile-time fallback) on every REAPER restart.
    reamix::remix::QualityWeights currentDefaultWeights_ {
        reamix::remix::kDefaultQualityWeights };
    reamix::ui::BlockAssemblyBeta currentDefaultBeta_    {};

    // Helper accessor — returns persisted weights for (currentItemKey(),
    // appMode_) or currentDefaultWeights_ when no entry exists (DEV-080).
    reamix::remix::QualityWeights currentQualityWeights() const;

    // DEV-080 sesja 108 — read ExtState "reamix.me" / "advanced_defaults"
    // into currentDefaultWeights_ + currentDefaultBeta_. Called once from
    // MainComponent ctor (NOT lazy on first Advanced open) so the user-set
    // defaults reach the compute path even when the user never opens the
    // Advanced window.
    void loadAdvancedDefaultsFromExtState();

    // ADR-097 sesja 107 — SettingsPopover "Advanced..." callback. Lazy-creates
    // panel + window on first invocation, toggles visibility on subsequent
    // clicks. Persists open/closed state + window position via ExtState.
    void onAdvancedToggled();

    // Helper — wires AdvancedWeightsPanel callbacks (slider change, restore,
    // Save/Load) the first time the panel is created. Called from
    // onAdvancedToggled() lazy-create branch.
    void setupAdvancedPanel();

    // Helper — pushes current per-(item, mode) raw weights + track context
    // into the advanced panel. Called when the panel is created/opened or
    // when the item/mode changes. No-op when panel is null.
    void syncAdvancedPanelFromState();

    // Persist/restore Advanced window state via ExtState (section "reamix.me",
    // keys "advanced_open" + "advanced_pos"). Called on window close + plugin
    // launch respectively.
    void persistAdvancedWindowState();
    void restoreAdvancedWindowOnLaunch();

    // Sesja 65 — sources where user explicitly dismissed the assembled
    // Blocks remix view via "↺ Edit arrangement". Symmetric to Region's
    // lastRegionByPath_ erase on "↺ Edit region": suppresses the auto-
    // restore of the assembled view on cross-item attach + on Blocks tab
    // re-entry, so the dismiss intent survives navigation. Cleared when
    // user re-engages by clicking Assemble (new remix produced).
    // ADR-056 (sesja 66) — keyed by ItemKey.
    std::set<ItemKey> blocksDismissedSources_;

    // Sesja 66 (ADR-056 follow-up) — items where user explicitly dismissed
    // the Region remix view via "↺ Edit region". Region group entry stays
    // alive in P_EXT (other future sessions / project reopen still see it),
    // but for the current in-memory session this flag suppresses the Region
    // group dispatch in applySelectedItem so the user's "I want fresh edit
    // state, not the saved view" intent survives navigation. Cleared on
    // Region Insert (new group recreated → user re-engaged with Region
    // remix workflow).
    std::set<ItemKey> regionGroupDismissedSources_;

    // Snapshot current userBlocks_/Queue/Variations into the session map
    // keyed by currentSourcePath_. Idempotent. Called from persistUserBlocks
    // (so every mutation that hits P_EXT also hits the in-memory cache) and
    // from applyEmptyState BEFORE clearing local state (so a deselect does
    // not lose user work).
    void cacheBlocksSession();

    // Restore userBlocks_/Queue/Variations from blocksSessionByPath_[path].
    // Returns true when a hit was restored. Used by applySelectedItem to
    // recover a session when re-attaching to a previously-seen sourcePath
    // (same item, different item with same source, or post-Insert clip).
    bool restoreBlocksSession (const juce::String& sourcePath);

    // Persist userBlocks_ → P_EXT for the currently-attached item AND
    // snapshot to in-memory session cache. ADR-052 — no-op when item GUID
    // is empty OR findItemByGuid returns nullptr. Call after every
    // userBlocks_ mutation (mark / kind change / boundary edit / split /
    // merge / delete).
    void persistUserBlocks();

    // ADR-092 / DEV-060 + DEV-076 sesja 100c — shared handler for the
    // seam-pill context menu and the Blocks-mode splice-marker context
    // menu (both refer to the same junction). chosen: 1 = Try different,
    // 2 = Reset to best, 3 = Audition junction, 4 = Show details (Blocks
    // splice marker only — generic menu doesn't have Audition).
    //
    // DEV-076: when assembly is currently clean (currentRemix_.has_value()
    // before mutation), the variation re-roll triggers an immediate
    // kickRemixPipeline so user sees the new splice without clicking
    // Assemble. When dirty (queue mutation pending), behaviour stays
    // deferred-until-Assemble (existing semantic).
    void handleBlocksJunctionAction (int junctionIdx, int chosen);

    // ADR-092 sesja 100c — load + save customKindRegistry_ from REAPER
    // ExtState "reamix.custom_kinds" (single global key, JSON array).
    // load called once at ctor; save called after every mutation in the
    // picker handlers (Add / Rename / Recolor / Delete).
    void loadCustomKindRegistry();
    void saveCustomKindRegistry();

    // ADR-092 sesja 100c — cascade after deleting a custom kind id:
    // iterate userBlocks_ (current item only — other items keep their
    // customKindId references; on next attach kindDisplay falls back to
    // builtin if registry-miss). Returns count of cascaded blocks.
    int  cascadeCustomKindRemoval (const juce::String& removedId);

    // ADR-092 sesja 100c — modal flows for custom kind add / rename /
    // recolor / delete. Each opens a juce::AlertWindow modal with the
    // appropriate fields and persists registry changes on confirm.
    void showAddCustomKindModal (double clampedStart, double clampedEnd,
                                  juce::Point<int> anchorScreenPos,
                                  juce::Component::SafePointer<juce::Component> pickerSafe = {});

    // ADR-092 sesja 100c — edit-context Add modal. Adds to registry + repaints
    // popover. NO region commit (block being edited keeps its current kind
    // until user explicitly picks the new tile). Used by BlockEditPopover.
    void showAddCustomKindModalEditContext (
        juce::Component::SafePointer<juce::Component> popoverSafe);

    // ADR-092 sesja 100c iter 2 — center juce::AlertWindow over the plugin
    // window's screen rather than the system default screen. Multi-monitor
    // setups otherwise spawn modal on a different display than the plugin.
    void centerAlertWindowOverPlugin (juce::AlertWindow* aw);
    void showEditCustomKindModal (const juce::String& id,
                                   reamix::ui::CustomKindAction action,
                                   juce::Component::SafePointer<juce::Component> pickerSafe = {});

    // DEV-058 (a) sesja 100d — paleta card right-click context menu actions.
    // Each opens a modal / popover and applies the result to userBlocks_,
    // persists, and refreshes the panel. Multi-select aware: when
    // paletteSelected_ contains > 1 ids, batch versions act on all selected;
    // single-tile right-click acts on the right-clicked block.
    void showRenameBlockDialog       (int blockIdx);
    void showChangeKindMiniPicker    (int blockIdx, juce::Point<int> screenPos);
    void showDeleteBlockConfirm      (int blockIdx);
    // DEV-058 (b) sesja 100d — batch operations on multi-selected blocks.
    void showDeleteBlocksBatchConfirm (std::vector<int> blockIndices);
    void showChangeKindBatchPicker    (std::vector<int> blockIndices,
                                        juce::Point<int> screenPos);

    // DEV-058 (b/c) sesja 100d — selection helpers + reorder commit.
    void clearPaletteSelection();
    void commitPaletteReorder (int fromBlockIdx, int toBlockIdx);
    // Helper: remove block at idx from userBlocks_ + cascade userBlocksQueue_
    // (drop matching, shift > idx by -1). Caller persists + refreshes.
    void removeUserBlockAndCascade (int idx);

    // ADR-052 (sesja 63 BUG-11/13/14/15/16) — Option B "Assemble = Analyze
    // Again minus DSP" cleanup. Every Blocks-state mutation (queue reorder,
    // try-different / reset variation, splice-flex change) leaves the prior
    // Assemble output stale: lastBlockSplices_ still references old junction
    // positions, waveform variant still shows the old remix WAV, splice
    // markers point to old source-time positions. Audition / playhead /
    // canvas markers all read this stale data → wrong audition position
    // (BUG-11), missing splice lines (BUG-13), playhead off-canvas (BUG-14),
    // splice/queue desync (BUG-15). User diagnostic "Analyze Again fixes
    // everything" confirmed the gap: full pipeline restart rebuilds all
    // dependent state, Assemble alone doesn't. This helper performs the
    // same UI/state cleanup that the analyze path triggers transitively,
    // minus the DSP work — caller still needs to click Assemble to repopulate.
    void invalidateBlocksAssembledOutput();

    // Push current userBlocks_ + queue into BlockAssemblyPanel and WaveformView.
    // Called after any mutation (drag-mark create, edit, delete, queue add/
    // remove/reorder). Idempotent — safe to call from any mutation site.
    void refreshBlockAssemblyUi();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
