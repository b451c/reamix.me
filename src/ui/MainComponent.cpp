#include "MainComponent.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>

#include "Theme.h"
#include "BlockEditPopover.h"
#include "BlockKindPickerPopover.h"
#include "KindDisplay.h"
#include "ContextMenu.h"
#include "analysis/ModelManager.h"
#include "reaper/ReaperBridge.h"
#include "reaper/GroupTracker.h"
#include "reaper/RegionGroupTracker.h"
#include "reaper/BlocksGroupTracker.h"
#include "remix/BlockAssembly.h"
#include "AnalysisDiskCache.h"

#if REAMIX_WITH_REAPER_IO
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"
#endif

namespace
{
    // DEV-021 — one-shot background thread that pre-warms the BeatDetector
    // ONNX session.
    class BeatDetectorPreWarmThread : public juce::Thread
    {
    public:
        explicit BeatDetectorPreWarmThread (std::function<void()> work)
            : juce::Thread ("reamix-prewarm"), work_ (std::move (work)) {}
        void run() override { if (work_) work_(); }
    private:
        std::function<void()> work_;
    };

    // ADR-047 § 4 — slider debounce: 100 ms idle before triggering remix.
    constexpr int kSliderDebounceMs = 100;

    // ADR-049 — selected item that belongs to a remix group is a remix-clip
    // pointer to the original audio file; the remix space is the WHOLE source
    // file, not the clip's D_LENGTH. Returns sourceFileDuration for grouped
    // items, durationSec otherwise. Falls back to durationSec on any failure.
    double effectiveSourceDurationFor (const reamix::reaper::SelectedItem& it)
    {
        if (it.itemPtr == nullptr || it.sourceFileDuration <= 0.0)
            return it.durationSec;
        const juce::String guid = reamix::reaper::getItemGuid (it.itemPtr);
        if (guid.isEmpty())
            return it.durationSec;
        const auto group = reamix::reaper::groupTracker::findGroupForSelectedItem (guid);
        if (! group.has_value())
            return it.durationSec;
        return it.sourceFileDuration;
    }

    // Sesja 100 iter 3 (DEV-018) — VolumeSliderPopover removed. Replaced by
    // inline `volumePopupSlider_` member in WaveformView (no CallOutBox)
    // per user directive: "wystarczyłby sam suwak głośności bez tej ramki".
}

// ── SliderDebounceTimer ───────────────────────────────────────────
//
// One-shot trailing-edge debounce. Each `restart()` resets the count;
// after `kSliderDebounceMs` of idleness the timer fires and calls back.

class MainComponent::SliderDebounceTimer : public juce::Timer
{
public:
    explicit SliderDebounceTimer (std::function<void()> onFire)
        : onFire_ (std::move (onFire)) {}

    void restart()
    {
        startTimer (kSliderDebounceMs);
    }

    void cancel()  // sesja 64 BUG-1 — abort pending compute on reset-to-orig
    {
        stopTimer();
    }

    void timerCallback() override
    {
        stopTimer();
        if (onFire_) onFire_();
    }

private:
    std::function<void()> onFire_;
};

// ── Placeholder ─────────────────────────────────────────────────────
MainComponent::Placeholder::Placeholder (juce::Colour background,
                                         juce::Colour bottomBorderColour,
                                         juce::String label)
    : background_ (background),
      bottomBorder_ (bottomBorderColour),
      label_ (std::move (label))
{
}

void MainComponent::Placeholder::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    g.fillAll (background_);

    if (! bottomBorder_.isTransparent())
    {
        g.setColour (bottomBorder_);
        g.fillRect (getLocalBounds().withHeight (1).withY (getHeight() - 1));
    }

    if (label_.isNotEmpty())
    {
        g.setColour (Fg4);
        g.setFont (uiFont (fs::Xs, 500));
        g.drawText (label_, getLocalBounds(), juce::Justification::centred, false);
    }
}

namespace
{
    constexpr double kDurationMinSec        = 5.0;
    constexpr double kDurationMaxMultiplier = 2.5;
    constexpr int    kSelectedItemPollMs    = 100;

    juce::String rxFmt (double seconds)
    {
        const int s = juce::jmax (0, (int) std::round (seconds));
        const int m = s / 60;
        const int ss = s % 60;
        return juce::String (m) + ":" + (ss < 10 ? "0" : "") + juce::String (ss);
    }

    // Compare two RemixOutputs for "is the cached one still valid for the
    // current target" — used by Insert to decide whether currentRemix_ is
    // fresh enough or needs a re-render. 50 ms tolerance matches the cache
    // key quantum in RemixCache.h.
    bool remixMatchesTarget (const reamix::ui::RemixOutput& r, double target)
    {
        return std::abs (r.targetSec - target) < 0.025; // half-quantum
    }

    // Sesja 60 — fresh-vs-stale check for region mode. RemixOutput stores
    // region bounds even for non-region runs (default 0.0); we must verify
    // the cached output's bounds match the currently-active region (or that
    // both are inactive).
    bool remixMatchesRegion (const reamix::ui::RemixOutput& r,
                              const std::optional<reamix::reaper::ItemRegion>& region)
    {
        const bool cachedHasRegion = (r.regionEndSec - r.regionStartSec) > 0.001;
        if (cachedHasRegion != region.has_value())
            return false;
        if (! region.has_value())
            return true; // both inactive
        return std::abs (r.regionStartSec - region->startSec) < 0.025
            && std::abs (r.regionEndSec   - region->endSec)   < 0.025;
    }

    // BUG-4 family fix sesja 63 — derive the "effective region" for any
    // operation that retargets the current remix (slider drag, splice
    // ContextMenu Try-different / Reset / Block, Insert). When a Region
    // remix was just shown, `selectedRange_` is reset by applyRemixToUi
    // (the source-time scrim coords don't match the Remix variant timeline),
    // and the next 100 ms timer tick zeroes `currentRegion_`. Without this
    // helper, every retarget falls back to Duration mode silently. The
    // cached `currentRemix_` still carries the bounds it was produced with
    // — those are the authoritative "what region this remix represents".
    std::optional<reamix::reaper::ItemRegion> deriveEffectiveRegion (
        const std::optional<reamix::reaper::ItemRegion>& currentRegion,
        const std::optional<reamix::ui::RemixOutput>&    currentRemix)
    {
        if (currentRegion.has_value()) return currentRegion;
        if (! currentRemix.has_value()) return std::nullopt;
        if ((currentRemix->regionEndSec - currentRemix->regionStartSec) <= 0.001)
            return std::nullopt;
        reamix::reaper::ItemRegion r;
        r.startSec = currentRemix->regionStartSec;
        r.endSec   = currentRemix->regionEndSec;
        return r;
    }
}

// ── ctor ────────────────────────────────────────────────────────────

MainComponent::MainComponent()
    : peaksProvider_      (std::make_unique<reamix::ui::FilePeaksProvider>()),
      remixPeaksProvider_ (std::make_unique<reamix::ui::FilePeaksProvider>())
{
    setLookAndFeel (&lookAndFeel_);
    setOpaque (true);
    setWantsKeyboardFocus (true);

    // DEV-024 (sesja 95 ADR-085) — wire peaks-ready callback so background
    // peaks worker triggers WaveformView repaint when the silhouette becomes
    // ready (50-200 ms after item switch, off the message thread). Without
    // this, the waveform stays empty until the next user-driven repaint.
    // Both providers share the same callback (only one is active at a time;
    // the other's repaint is a no-op when its peaks are not the current view).
    auto peaksReadyCallback = [this] { waveformView_.repaint(); };
    peaksProvider_->setOnPeaksReady (peaksReadyCallback);
    remixPeaksProvider_->setOnPeaksReady (peaksReadyCallback);

    addAndMakeVisible (headerBar_);
    addAndMakeVisible (sourcePanel_);
    addAndMakeVisible (modeTabs_);
    addAndMakeVisible (durationPanel_);
    addChildComponent (blockAssemblyPanel_); // ADR-051 — visible only in Blocks mode
    addAndMakeVisible (waveformView_);
    addAndMakeVisible (auditionBar_);    // ADR-080 RESCOPE + ADR-083 sesja 92
    // ADR-097 sesja 107 — AdvancedWeightsPanel is no longer a child of the
    // main shell. It is lazy-created inside AdvancedWeightsWindow on user
    // click of SettingsPopover "Advanced..." button (see onAdvancedToggled).
    addAndMakeVisible (transportBar_);
    addAndMakeVisible (statusBar_);
    addChildComponent (settingsPopover_);
    addChildComponent (supportPopover_);   // sesja 108 — donation links popover
    addChildComponent (tooltip_); // hidden until first hover

    // ADR-084 sesja 93 — TooltipWindow color override moved to
    // LookAndFeelReamix constructor (LAF::findColour rules drawTooltip,
    // not Component::findColour). Initial sesja-93 attempt setColour on
    // TooltipWindow instance was no-op.

    // ADR-080 RESCOPE + ADR-083 (sesja 92) — AuditionBar slider callbacks.
    // On user drag/click: persist params per source path + trigger Viterbi
    // DP-only re-run via a fresh kickRemixPipeline at current target.
    // Per-track persistence mirrors `lastModeByPath_` pattern (in-memory map
    // keyed by ItemKey). Bit-exact baseline preserved when slider untouched
    // (defaults match current production behavior across all 3 modes).
    auto kickAuditionRerun = [this]
    {
        // DEV-079 sesja 101 — persist per-(item, mode) so each tab is its
        // own audition workspace. Mirrors targetByPathMode_ pattern.
        auditionParamsByPathMode_[currentItemKey()][appMode_] = AuditionParams{
            auditionBar_.toneValue(),
            auditionBar_.editLengthValue(),
            auditionBar_.allowPmSecondsValue(),
            auditionBar_.minCutBeatsValue()
        };
        // Trigger fresh kickRemixPipeline at current target if analysis
        // bundle is available; otherwise wait for user explicit action.
        if (analysisState_ != AnalysisState::Complete) return;
        if (currentSourcePath_.isEmpty()) return;

        PendingRemixOp op;
        op.targetSec = durationPanel_.getTarget();
        if (auto bIt = blockedBySource_.find (currentSourcePath_); bIt != blockedBySource_.end())
            op.blockedTransitions = bIt->second;
        if (currentRegion_.has_value())
        {
            op.regionStartSec = currentRegion_->startSec;
            op.regionEndSec   = currentRegion_->endSec;
        }
        op.variation = 0;
        kickRemixPipeline (op);
    };
    auditionBar_.onToneChanged           = [kickAuditionRerun](double)  { kickAuditionRerun(); };
    auditionBar_.onEditLengthChanged     = [kickAuditionRerun](int)     { kickAuditionRerun(); };
    auditionBar_.onAllowPmSecondsChanged = [kickAuditionRerun](int)     { kickAuditionRerun(); };
    auditionBar_.onMinCutBeatsChanged    = [kickAuditionRerun](int)     { kickAuditionRerun(); };

    // Sesja 108 — discrete quick-Advanced icon in AuditionBar top-right.
    // Routes to the same handler the gear → SettingsPopover → "Advanced..."
    // path uses; 1 click vs 3.
    auditionBar_.onAdvancedClicked = [this] { onAdvancedToggled(); };

    // Sesja 108 — AuditionBar collapse/expand. Persist to ExtState +
    // re-run resized() so WaveformView reservation grows/shrinks.
    auditionBar_.onCollapseToggled = [this]
    {
        if (SetExtState)
            SetExtState ("reamix.me", "audition_collapsed",
                         auditionBar_.isCollapsed() ? "1" : "0", true);
        resized();
    };
    // Apply persisted state at construction.
    if (GetExtState)
    {
        const char* raw = GetExtState ("reamix.me", "audition_collapsed");
        if (raw != nullptr && raw[0] == '1')
            auditionBar_.setCollapsed (true);
    }

    // ADR-097 sesja 107 — AdvancedWeightsPanel callbacks are wired in
    // setupAdvancedPanel() once the panel is lazy-created. SettingsPopover
    // "Advanced..." button routes here.
    settingsPopover_.onAdvancedToggled = [this] { onAdvancedToggled(); };

    // Sesja 108 — heart icon in HeaderBar opens the support popover.
    headerBar_.onHeartClicked = [this] { supportPopover_.show(); };

    waveformView_.setPeaksProvider (peaksProvider_.get());

    // ADR-057 (sesja 68) — Splice flexibility removed; ExtState key
    // "splice_flexibility" deprecated, no migration (any prior value ignored).

#if REAMIX_WITH_REAPER_IO
    // Sesja 65 — load persisted Snap region (default Off when unset).
    if (GetExtState)
    {
        const char* raw = GetExtState ("reamix.me", "snap_region");
        const juce::String s = (raw != nullptr) ? juce::String::fromUTF8 (raw) : juce::String();
        using SnapMode = reamix::ui::WaveformView::SnapMode;
        if      (s == "beats")     waveformView_.setSnapMode (SnapMode::Beats);
        else if (s == "downbeats") waveformView_.setSnapMode (SnapMode::Downbeats);
        // else: leave default (Off)
    }

    // Sesja 100 (DEV-018) — load persisted preview volume. Default 1.0 when
    // unset (matches pre-sesja-100 behaviour). Stored as decimal string e.g.
    // "0.85" for 85% gain.
    if (GetExtState)
    {
        const char* raw = GetExtState ("reamix.me", "preview_volume");
        if (raw != nullptr)
        {
            const juce::String s = juce::String::fromUTF8 (raw);
            const double v = juce::jlimit (0.0, 1.0, s.getDoubleValue());
            previewController_.setVolume (v);
            waveformView_.setPreviewVolume (v);
        }
    }

    // Sesja 100b (DEV-049) — load persisted Insert toggles. Default ON
    // for both per user smoke choice (feature visible from first Insert
    // without hunting the Settings popover). Stored as "1" / "0".
    if (GetExtState)
    {
        const char* rawSplice = GetExtState ("reamix.me", "insert_splice_markers");
        if (rawSplice != nullptr)
        {
            const juce::String s = juce::String::fromUTF8 (rawSplice);
            insertSpliceMarkersEnabled_ = (s == "1");
        }
        const char* rawRegion = GetExtState ("reamix.me", "insert_render_region");
        if (rawRegion != nullptr)
        {
            const juce::String s = juce::String::fromUTF8 (rawRegion);
            insertRenderRegionEnabled_ = (s == "1");
        }
    }
#endif

    // ADR-092 sesja 100c — load custom kind registry from REAPER ExtState.
    // Empty / missing key = empty registry (user has not added any customs
    // yet). Wires registry pointer to view components AFTER load so first
    // paint sees populated registry. setCustomKindRegistry repaints; cheap.
    loadCustomKindRegistry();
    waveformView_.setCustomKindRegistry (&customKindRegistry_);
    blockAssemblyPanel_.setCustomKindRegistry (&customKindRegistry_);

    applyEmptyState();

    // ── Slider debounce ──────────────────────────────────────────────
    sliderDebounceTimer_ = std::make_unique<SliderDebounceTimer> (
        [this] { sliderDebounceFired(); });

    // ── Wiring ───────────────────────────────────────────────────────

    sourcePanel_.onAnalyze = [this] { startAnalyze(); };

    durationPanel_.onTargetChanged = [this] (double sec)
    {
        statusBar_.setText (juce::String::fromUTF8 ("Target \xc2\xb7 ") + rxFmt (sec));

        // Sesja 65 — record per-mode target. Tab switching reads this back
        // to restore the slider + look up remixCache_ for that mode's last
        // computed remix. Saved unconditionally (covers BUG-1 reset path
        // below where target==orig should also persist as the Duration
        // baseline so a later tab switch back doesn't surprise the user).
        if (currentSourcePath_.isNotEmpty())
            targetByPathMode_[currentItemKey()][appMode_] = sec;

        // Sesja 64 BUG-1 — slider double-click reset semantic. When target
        // equals the panel's current ORIG (Duration mode = file length;
        // Region mode = region length), user wants "no retarget" — the
        // remix should disappear and the waveform return to Source variant.
        // Without this, kickRemixPipeline runs with target==orig and
        // produces a zero-splice "remix" that visually leaves the Remix
        // variant active (regression vs sesja 41 ADR-046 expectations).
        const double orig = durationPanel_.getOriginalDuration();
        if (std::abs (sec - orig) < 0.01)
        {
            sliderDebounceTimer_->cancel();
            previewController_.stop();
            waveformView_.setPlayhead (std::nullopt);
            waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
            waveformView_.setPeaksProvider (peaksProvider_.get());
            waveformView_.setSpliceMarkers ({});
            waveformView_.setEditPlan ({});
            clearCurrentRemix();
            tmpWavPath_ = currentSourcePath_;
            transportBar_.setState (analysisState_ == AnalysisState::Complete
                                      ? reamix::ui::TransportState::Ready
                                      : reamix::ui::TransportState::Idle);
            statusBar_.setNotice ("Reset to original");
            // Edit-region overlay (Region mode) — caller-managed: leaving
            // it in current state is fine, the next region click cycles it.
            return;
        }

        // ADR-047 § 4 — debounced slider: store latest target, restart timer.
        // The timerCallback drains pendingSliderOp_ when 100 ms idle elapses.
        // Blocked + variation persist per source across slider changes
        // (Lua waveform_remix.lua behaviour, HANDOVER session-56 gotcha).
        pendingSliderOp_ = {};
        pendingSliderOp_.targetSec = sec;
        if (currentSourcePath_.isNotEmpty())
        {
            auto bIt = blockedBySource_.find (currentSourcePath_);
            if (bIt != blockedBySource_.end()) pendingSliderOp_.blockedTransitions = bIt->second;
            auto vIt = variationBySource_.find (currentSourcePath_);
            if (vIt != variationBySource_.end()) pendingSliderOp_.variation = vIt->second;
        }
        // Sesja 60 — propagate region bounds when in Region mode so the
        // RemixPipeline routes through RegionOptimizer. BUG-4 family fix
        // sesja 63 — fall back to currentRemix_'s region bounds when
        // currentRegion_ has been transiently nulled by Remix-variant
        // selectedRange_ reset (otherwise slider silently produces Duration
        // remix while the user is on the Region tab). Sesja 65 — gate the
        // fallback on appMode_==Region: when user manually switched to
        // Duration tab, currentRemix_ may still carry stale region bounds
        // (resetMaterialView keeps currentRemix_ per ADR-050) and would
        // leak into the Duration slider's pipeline run.
        if (appMode_ == reamix::ui::ModeTabs::Mode::Region)
        {
            if (auto eff = deriveEffectiveRegion (currentRegion_, currentRemix_))
            {
                pendingSliderOp_.regionStartSec = eff->startSec;
                pendingSliderOp_.regionEndSec   = eff->endSec;
            }
        }
        sliderDebounceTimer_->restart();
    };

    transportBar_.onPlayStop = [this]
    {
        if (previewController_.isPlaying())
        {
            // Sesja 64 BUG-2 enhancement — keep playhead visible after Stop
            // so user sees where the next Space will resume from. Also store
            // that position into lastSeekSec_ so Space picks it up.
            const double stopPos = previewController_.getPositionSec();
            previewController_.stop();
            waveformView_.setPlayhead (stopPos);
            lastSeekSec_ = stopPos;
            transportBar_.setState (reamix::ui::TransportState::Ready);
            statusBar_.setNotice (juce::String::fromUTF8 (
                "Stopped \xe2\x80\x94 Space resumes from ") + rxFmt (stopPos));
            return;
        }

        if (tmpWavPath_.isEmpty())
        {
            statusBar_.setError ("Analyze a source before previewing");
            return;
        }

        // Sesja 64 BUG-2 — Space priority: drag-select range > last click-to-seek > 0:00.
        // selectedRange wins because it's an explicit drag intent (range to play);
        // lastSeekSec_ is single-click intent (start from this point); 0.0 fallback.
        const double startSec = selectedRange_.has_value()
            ? selectedRange_->startSec
            : lastSeekSec_.value_or (0.0);
        const double endSec   = selectedRange_.has_value() ? selectedRange_->endSec   : -1.0;
        const auto result = previewController_.play (tmpWavPath_, startSec, endSec);
        using PR = reamix::ui::PreviewController::PlayResult;
        switch (result)
        {
            case PR::OK:
                transportBar_.setState (reamix::ui::TransportState::Playing);
                statusBar_.setText (selectedRange_.has_value() ? "Playing selection"
                                                                : "Playing preview");
                break;

            case PR::NoSws:
                statusBar_.setError (
                    "Preview unavailable \xe2\x80\x94 install SWS Extension (sws-extension.org)");
                if (! swsModalShown_)
                {
                    swsModalShown_ = true;
                    juce::AlertWindow::showMessageBoxAsync (
                        juce::MessageBoxIconType::InfoIcon,
                        "reamix.me \xe2\x80\x94 SWS Extension required",
                        "Preview playback requires the SWS Extension.\n\n"
                        "Install it from sws-extension.org, then restart REAPER.\n\n"
                        "Analyze and Insert work without SWS \xe2\x80\x94 only preview is disabled.",
                        "OK",
                        this);
                }
                break;

            case PR::NoSource:
                statusBar_.setError ("Preview failed: could not load analysis output");
                break;

            case PR::InternalError:
                statusBar_.setError ("Preview failed: CF_CreatePreview returned null");
                break;
        }
    };

    transportBar_.onInsert = [this]
    {
        // ADR-046 — Insert remix clips on REAPER timeline.
        // ADR-047 § 3 rule 4 — auto-kick RemixPipeline if currentRemix_ is
        // missing or stale (target changed since last render). Lua parity:
        // remix_operations.lua:352-365 start_export auto-chains analyze/remix.
        // Sesja 60 — Region branch routes through insertRegionRemixClips
        // (split + delete + insert + shift) instead of insertRemixClips.

        auto picked = reamix::reaper::getSelectedItem();
        if (! picked.has_value())
        {
            statusBar_.setError ("No item selected");
            return;
        }
        if (picked->sourcePath != currentSourcePath_)
        {
            statusBar_.setText ("Source changed \xe2\x80\x94 re-analyze required");
            return;
        }

        const double currentTarget = durationPanel_.getTarget();

        // BUG-4 family fix sesja 63 — derive effective region for slider /
        // ContextMenu / Insert paths uniformly. See deriveEffectiveRegion()
        // doc comment for the reset/timer interaction this works around.
        const auto effectiveRegion =
            deriveEffectiveRegion (currentRegion_, currentRemix_);

        // Build a PendingRemixOp that mirrors the current UI state. Region
        // branch carries regionStartSec/EndSec so the pipeline routes through
        // RegionOptimizer (sesja 60).
        auto buildPendingOp = [this, currentTarget, effectiveRegion]
        {
            PendingRemixOp op;
            op.targetSec = currentTarget;
            if (auto bIt = blockedBySource_.find (currentSourcePath_); bIt != blockedBySource_.end())
                op.blockedTransitions = bIt->second;
            if (auto vIt = variationBySource_.find (currentSourcePath_); vIt != variationBySource_.end())
                op.variation = vIt->second;
            if (effectiveRegion.has_value())
            {
                op.regionStartSec = effectiveRegion->startSec;
                op.regionEndSec   = effectiveRegion->endSec;
            }
            return op;
        };

        // No analysis yet → queue Insert followup, kick analyze + auto-remix.
        if (analysisState_ != AnalysisState::Complete)
        {
            PendingFollowup f;
            f.kind = PendingFollowup::Kind::Insert;
            f.op   = buildPendingOp();
            followupAfterAnalysis_ = f;
            statusBar_.setText ("Will Insert after analyze completes");
            if (analysisState_ == AnalysisState::Idle) startAnalyze();
            return;
        }

        // Analysis complete but no remix yet, remix is for a different target,
        // or remix region bounds don't match current selection → kick
        // RemixPipeline + queue Insert followup.
        if (! currentRemix_.has_value()
            || ! remixMatchesTarget (*currentRemix_, currentTarget)
            || ! remixMatchesRegion (*currentRemix_, effectiveRegion))
        {
            PendingRemixOp op = buildPendingOp();
            kickRemixPipeline (op);
            PendingFollowup f;
            f.kind = PendingFollowup::Kind::Insert;
            f.op   = op;
            followupAfterAnalysis_ = f;
            statusBar_.setText ("Computing remix \xe2\x80\xa6 will Insert when ready");
            return;
        }

        // currentRemix_ is fresh → run Insert path.
        const auto& remix = *currentRemix_;
        if (remix.editPlan.clips.empty())
        {
            statusBar_.setError ("Remix has no clips");
            return;
        }

        // ── Region branch (sesja 60, BUG-4 fix sesja 63) ──────────
        // Read region bounds from the remix output itself, NOT from
        // currentRegion_/selectedRange_ in the UI state. Reason: applyRemixToUi
        // resets selectedRange_ after a Region remix lands (the source-time
        // scrim coords don't match the Remix variant timeline), and the next
        // 100 ms timer tick rewrites currentRegion_ to nullopt via
        // recomputeRegionState(). Insert previously fell through to the
        // Duration branch in that window. RemixOutput.regionStartSec/EndSec
        // are always populated by RemixPipeline (0.0/0.0 for non-region) so
        // they're the authoritative "what produced this remix" identity.
        const bool remixIsRegional = (remix.regionEndSec - remix.regionStartSec) > 0.001;
        if (remixIsRegional)
        {
            // DEV-033 / ADR-054 — split timeline at the algorithm's *actual*
            // entry/exit beat positions, not user's region bounds. This
            // eliminates the source-content jump at pre-region→remix and
            // remix→post-region boundaries: pre-region item ends exactly
            // where first remix clip plays from, post-region resumes
            // exactly where last remix clip ends. Falls back to user
            // bounds if the optimizer hasn't populated actuals (cache
            // entry from before this fix landed; defensive only).
            const double remixActualStart =
                (remix.actualRegionEndSec > remix.actualRegionStartSec)
                    ? remix.actualRegionStartSec
                    : remix.regionStartSec;
            const double remixActualEnd =
                (remix.actualRegionEndSec > remix.actualRegionStartSec)
                    ? remix.actualRegionEndSec
                    : remix.regionEndSec;

            // ADR-056 (sesja 66) — capture clicked-item GUID BEFORE Insert
            // so we record the right identity in the Region group. For a
            // clicked inserted clip the GUID is the clip's own; for a
            // pre-region piece (or the un-split original) it is the
            // canonical source GUID. Both forms resolve to the same group
            // entry via direct (#1) or indirect (#2) lookup paths.
            const juce::String pickedItemGuid =
                reamix::reaper::getItemGuid (picked->itemPtr);

            // DEV-038 sesja 67 — Region Update detection. If a Region group
            // already exists for this item (direct OR indirect-by-clipped-
            // clip-GUID), promote this Insert into Update: caller resolves
            // the canonical pre-region piece and populates the spec fields
            // the bridge expects to delete prior clips + restore D_LENGTH
            // before the standard split. Without this, a second Insert
            // splits AGAIN inside the prior pre-region piece and adds a
            // fresh remix alongside the existing one (Bug D1) or fails
            // when new bounds lie outside the shrunk pre-region (Bug D2).
            // Mode parity with Duration's groupTracker Update flow per
            // ADR-046. Always promotes when group is found — dismissed-flag
            // (Edit region) does NOT suppress Update because the user's
            // explicit Insert click is itself re-engagement.
            auto existingRegionGroup =
                reamix::reaper::regionGroupTracker::findGroupForSelectedItem (pickedItemGuid);

            const bool isRegionUpdate = existingRegionGroup.has_value()
                                         && existingRegionGroup->sourceGuid.isNotEmpty();

            // Resolve the source item to split. For Update path this is the
            // canonical pre-region piece (= group->sourceGuid item, which
            // keeps the original GUID across split per ADR-056 D2). For
            // fresh Insert it is whatever REAPER currently has selected.
            void* sourceItemForSplit = picked->itemPtr;
            if (isRegionUpdate)
            {
                void* preRegionPiece =
                    reamix::reaper::findItemByGuid (existingRegionGroup->sourceGuid);
                if (preRegionPiece == nullptr)
                {
                    // Pre-region piece is gone (user manually deleted the
                    // item, or project state diverged across reopen). Bail
                    // before SplitMediaItem on a possibly-dangling ptr —
                    // user can Cmd+Z to restore and try again.
                    statusBar_.setError (juce::String::fromUTF8 (
                        "Pre-region piece not found \xE2\x80\x94 press Cmd+Z and try again"));
                    return;
                }
                sourceItemForSplit = preRegionPiece;
            }

            // DEV-038 sesja 67 fix-1 — base position for absRegion bounds.
            // For fresh Insert, picked->positionSec IS the original item
            // position (user clicked the un-split source). For Update,
            // picked->positionSec is the position of the *clicked clip*
            // (post-dispatch picked = inserted clip, e.g. clip 1 at
            // absRegionStart of the prior Insert). Using picked->positionSec
            // for Update would offset every new clip by the clicked clip's
            // position relative to the original — causing the "items shift
            // forward into the song" symptom user reported (initial Step 3
            // smoke). Pre-region piece keeps the original D_POSITION across
            // split, so reading it from preRegionPiece via getItemPosition
            // gives the correct base for either fresh Insert or Update.
            const double sourceBasePosition = isRegionUpdate
                ? reamix::reaper::getItemPosition (sourceItemForSplit)
                : picked->positionSec;

            reamix::reaper::RegionInsertSpec spec;
            spec.trackPtr            = picked->trackPtr;
            spec.sourceItemToSplit   = sourceItemForSplit;
            spec.absRegionStartSec   = sourceBasePosition + remixActualStart;
            spec.absRegionEndSec     = sourceBasePosition + remixActualEnd;
            spec.originalSourcePath  = currentSourcePath_;
            spec.clips               = &remix.editPlan.clips;
            // ADR-057 sesja 68 step 2e — boundary fade overlap from RemixPipeline.
            spec.boundaryLeadInSec   = remix.boundaryLeadInSec;
            spec.boundaryLeadOutSec  = remix.boundaryLeadOutSec;
            spec.undoLabel = isRegionUpdate
                ? juce::String ("Region remix: update")
                : juce::String ("Region remix: splice into track");

            if (isRegionUpdate)
            {
                spec.existingItemGuidsToDelete = existingRegionGroup->itemGuids;
                spec.preRegionGuid              = existingRegionGroup->sourceGuid;
                spec.originalItemDurationSec    = existingRegionGroup->originalItemDurationSec;
            }

            // Sesja 100b (DEV-049) — splice markers + render region.
            // Plumbed through the spec so ReaperBridge::insertRegionRemixClips
            // adds them INSIDE its Undo_BeginBlock/EndBlock pair — Cmd+Z
            // then rolls back the clips and the markers atomically.
            //
            // originalItem* fields drive the render region's WHOLE-SONG
            // semantic: span pre-region + remix + post-region, not just
            // the spliced middle. Set unconditionally for both Insert
            // and Update paths (existing originalItemDurationSec set
            // above for Update only — extended here for Insert too).
            // sourceBasePosition was computed earlier as the original
            // item D_POSITION (pre-region piece on Update; picked->positionSec
            // on fresh Insert), exactly the geometry we need.
            spec.addSpliceMarkers          = insertSpliceMarkersEnabled_;
            spec.addRenderRegion           = insertRenderRegionEnabled_;
            spec.spliceTimesRel            = remix.transitionTimesSec;
            spec.renderRegionName          =
                juce::File (currentSourcePath_).getFileNameWithoutExtension();
            spec.originalItemPositionSec   = sourceBasePosition;
            if (! isRegionUpdate)
                spec.originalItemDurationSec = picked->durationSec;

            // The canonical group identity is the pre-region piece's GUID
            // (= group->sourceGuid for Update; = pickedItemGuid for fresh
            // Insert when user clicked an un-split original item).
            const juce::String regionGroupSourceGuid = isRegionUpdate
                ? existingRegionGroup->sourceGuid
                : pickedItemGuid;

            // DEV-038 sesja 67 — capture pre-Insert item-on-timeline length
            // for the group payload. On fresh Insert, picked->durationSec
            // is the original item duration. On Update, the existing group
            // already carries the correct value (preserving across multiple
            // Updates) so we keep it. Legacy entries (originalItemDurationSec
            // == 0) inherit 0.0 — Update path skips D_LENGTH restore for
            // those, matching pre-DEV-038 behavior.
            const double originalItemDur = isRegionUpdate
                ? existingRegionGroup->originalItemDurationSec
                : picked->durationSec;

            // Sesja 64 — busy spinner during REAPER mutations (split + delete +
            // shift + N inserts). StatusBar spinner deferred 300 ms (no flicker
            // for short ops); button busy is instant so user sees the click
            // landed.
            const juce::String busyText = isRegionUpdate
                ? juce::String::fromUTF8 ("Updating in REAPER\xE2\x80\xA6")
                : juce::String::fromUTF8 ("Inserting into REAPER\xE2\x80\xA6");
            const juce::String btnBusyText = isRegionUpdate
                ? juce::String::fromUTF8 ("Updating\xE2\x80\xA6")
                : juce::String::fromUTF8 ("Inserting\xE2\x80\xA6");
            busyDeferInsert_.startBusy (busyText);
            transportBar_.setInsertBusy (true, btnBusyText);
            headerBar_.setStatusKind (reamix::ui::HeaderStatus::Analyzing);
            const auto result = reamix::reaper::insertRegionRemixClips (spec);
            busyDeferInsert_.stopBusy();
            transportBar_.setInsertBusy (false);
            headerBar_.setStatusKind (result.ok
                                       ? reamix::ui::HeaderStatus::Ready
                                       : reamix::ui::HeaderStatus::Error);
            if (! result.ok)
            {
                statusBar_.setError ((isRegionUpdate ? juce::String ("Region update failed: ")
                                                     : juce::String ("Region insert failed: "))
                                     + result.errorMessage);
                return;
            }

            // ADR-056 (sesja 66) — persist Region group so applySelectedItem
            // can detect inserted-clip clicks + restore the original-item
            // remix view (UX parity with Duration mode's groupTracker per
            // ADR-046). Closes DEV-035 candidate from sesja 60 comment.
            // Captures region bounds + target + tmpWavPath so re-attach
            // can rebuild the same RemixCache key the previous run used.
            // DEV-038 sesja 67 — Update overwrites the prior entry with
            // fresh GUIDs + new bounds; originalItemDurationSec preserved.
            if (! regionGroupSourceGuid.isEmpty())
            {
                reamix::reaper::regionGroupTracker::RegionGroup g;
                g.sourceGuid              = regionGroupSourceGuid;
                g.trackGuid               = reamix::reaper::getTrackGuid (spec.trackPtr);
                g.basePosition            = spec.absRegionStartSec;
                g.regionStartSec          = remix.regionStartSec;
                g.regionEndSec            = remix.regionEndSec;
                g.targetSec               = remix.targetSec;
                g.tmpWavPath              = remix.tmpWavPath;
                g.itemGuids               = result.insertedItemGuids;
                g.originalItemDurationSec = originalItemDur;
                reamix::reaper::regionGroupTracker::saveRegionGroup (g);

                // ADR-056 (sesja 66 fix A) — Insert/Update is re-engagement:
                // user intentionally re-created a Region remix here, so any
                // prior "Edit region" dismiss intent for this item is now
                // stale. Erase the dismissed flag so the next attach + click
                // on inserted clip restores the new remix view (UX consistency
                // with Duration mode where Insert is also re-engagement).
                regionGroupDismissedSources_.erase (
                    ItemKey { currentSourcePath_, regionGroupSourceGuid });
            }

            // DEV-034 (sesja 63b) — keep lastRegionByPath_ entry alive after
            // Insert so the Cmd+Z (undo) workflow restores Region state on
            // re-selecting the now-rejoined whole item. Small pre/post-region
            // pieces are caught by the size guard in applySelectedItem
            // (snap.endSec > itemDurationSec → skip restore + erase entry).
            // ADR-056 (sesja 66) — clip clicks now also restore via the
            // saved Region group (closes DEV-035 candidate noted previously).

            // DEV-050 (sesja 100b) — capture first inserted clip GUID for
            // Cmd+Z detection in timerCallback. When user invokes REAPER
            // undo on the Insert, the inserted items are removed (their
            // GUIDs stop resolving via findItemByGuid) while the
            // GroupTracker ExtState entry survives — only the clip-GUID
            // resolution catches the undo signal.
            lastInsertedClipGuid_ = result.insertedItemGuids.empty()
                                  ? juce::String()
                                  : result.insertedItemGuids.front();

            transportBar_.setState (reamix::ui::TransportState::Inserted);
            statusBar_.setText ((isRegionUpdate ? juce::String ("Updated ")
                                                : juce::String ("Inserted "))
                                + juce::String (result.clipCount)
                                + " region clips");
            return;
        }

        // ── Non-region branch (ADR-046 verbatim) ───────────────────
        const juce::String selectedGuid = reamix::reaper::getItemGuid (picked->itemPtr);
        auto existingGroup = reamix::reaper::groupTracker::findGroupForSelectedItem (selectedGuid);

        reamix::reaper::InsertSpec spec;
        spec.trackPtr           = picked->trackPtr;
        spec.basePositionSec    = picked->positionSec;
        spec.originalSourcePath = currentSourcePath_;
        spec.clips              = &remix.editPlan.clips;

        bool         isUpdate           = false;
        juce::String resolvedSourceGuid = selectedGuid;

        if (existingGroup.has_value())
        {
            resolvedSourceGuid = existingGroup->sourceGuid;
            spec.existingItemGuidsToDelete = existingGroup->itemGuids;
            spec.basePositionSec           = existingGroup->basePosition;

            if (existingGroup->trackGuid.isNotEmpty())
            {
                if (void* groupTrack = reamix::reaper::findTrackByGuid (existingGroup->trackGuid))
                    spec.trackPtr = groupTrack;
            }

            // Post-Cmd+Z recovery: if the selected item IS the original
            // source (`selectedGuid == sourceGuid`), the previously inserted
            // clips were rolled back by REAPER undo — but the ExtState group
            // is still alive (ExtState writes don't participate in REAPER
            // undo). Treat as a fresh Insert that also wipes the source
            // (otherwise we'd leave the source on the timeline next to a
            // freshly-inserted clip set, the bug user reported).
            const bool postUndoRecover = (selectedGuid == existingGroup->sourceGuid);
            if (postUndoRecover)
            {
                spec.sourceItemToDelete = picked->itemPtr;
                spec.undoLabel          = "Insert remix clips";
            }
            else
            {
                isUpdate       = true;
                spec.undoLabel = "Update remix clips";
            }
        }
        else
        {
            spec.sourceItemToDelete = picked->itemPtr;
            spec.undoLabel          = "Insert remix clips";
        }

        // Sesja 100b (DEV-049) — splice markers + render region. Plumbed
        // through the spec so ReaperBridge::insertRemixClips adds them
        // INSIDE its Undo_BeginBlock/EndBlock pair → Cmd+Z atomically
        // rolls back clips + markers + region. Render-region span is
        // derived from clips.back() inside the bridge — no length field
        // to keep in sync.
        spec.addSpliceMarkers = insertSpliceMarkersEnabled_;
        spec.addRenderRegion  = insertRenderRegionEnabled_;
        spec.spliceTimesRel   = remix.transitionTimesSec;
        spec.renderRegionName =
            juce::File (currentSourcePath_).getFileNameWithoutExtension();

        // Sesja 64 — busy spinner during REAPER mutations.
        busyDeferInsert_.startBusy (juce::String::fromUTF8 ("Inserting into REAPER\xe2\x80\xa6"));
        transportBar_.setInsertBusy (true, juce::String::fromUTF8 ("Inserting\xe2\x80\xa6"));
        headerBar_.setStatusKind (reamix::ui::HeaderStatus::Analyzing);
        const auto result = reamix::reaper::insertRemixClips (spec);
        busyDeferInsert_.stopBusy();
        transportBar_.setInsertBusy (false);
        headerBar_.setStatusKind (result.ok
                                   ? reamix::ui::HeaderStatus::Ready
                                   : reamix::ui::HeaderStatus::Error);
        if (! result.ok)
        {
            statusBar_.setError ("Insert failed: " + result.errorMessage);
            return;
        }

        reamix::reaper::groupTracker::RemixGroup g;
        g.sourceGuid    = resolvedSourceGuid;
        g.trackGuid     = reamix::reaper::getTrackGuid (spec.trackPtr);
        g.basePosition  = spec.basePositionSec;
        g.itemGuids     = result.insertedItemGuids;
        // Sesja 100b — capture the target the remix was rendered at so a
        // post-Insert clip click (the selection REAPER auto-makes after
        // SplitMediaItem) lands on the same RemixCache key + restores the
        // Remix variant immediately. Without this, applySelectedItem fell
        // back to floor(durationSec) for resolvedTarget → cache miss →
        // variant stayed Source until kickRemixPipeline finished its async
        // recompute. UX parity with Region / Blocks group dispatch.
        g.targetSec     = remix.targetSec;
        reamix::reaper::groupTracker::saveRemixGroup (g);

        // DEV-041 (sesja 95 ADR-085) — Block Assembly Insert also writes a
        // BlocksGroup tracker entry so applySelectedItem can canonicalize a
        // post-Insert clip click back to the original source identity AND
        // pre-prime lastModeByPath_[newKey] = Blocks. Mirrors RegionGroupTracker
        // (sesja 66 ADR-056) for Region mode. Conditional on appMode_ == Blocks
        // to avoid polluting the Blocks namespace with Duration-mode Inserts.
        // Re-engagement: erase any prior dismiss flag for this source so the
        // next attach + clip click restores the new Blocks view (UX consistency
        // with Region's regionGroupDismissedSources_ erase on Insert).
        if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks
            && ! resolvedSourceGuid.isEmpty())
        {
            reamix::reaper::blocksGroupTracker::BlocksGroup bg;
            bg.sourceGuid              = resolvedSourceGuid;
            bg.trackGuid               = g.trackGuid;
            bg.basePosition            = spec.basePositionSec;
            bg.targetSec               = currentTarget;
            bg.tmpWavPath              = remix.tmpWavPath;
            bg.itemGuids               = result.insertedItemGuids;
            bg.originalItemDurationSec = picked->durationSec;
            reamix::reaper::blocksGroupTracker::saveBlocksGroup (bg);

            blocksDismissedSources_.erase (
                ItemKey { currentSourcePath_, resolvedSourceGuid });
        }

        // DEV-050 (sesja 100b) — capture first inserted clip GUID for
        // Cmd+Z detection in timerCallback. See Region-branch site above
        // for the rationale: clip GUID resolution is the only reliable
        // signal because GroupTracker ExtState survives REAPER undo.
        lastInsertedClipGuid_ = result.insertedItemGuids.empty()
                              ? juce::String()
                              : result.insertedItemGuids.front();

        transportBar_.setState (reamix::ui::TransportState::Inserted);
        statusBar_.setText ((isUpdate ? juce::String ("Updated ")
                                       : juce::String ("Inserted "))
                            + juce::String (result.clipCount) + " clips");
    };

    // ADR-039 — WaveformView wiring.
    waveformView_.onSeek = [this] (double sec)
    {
        // Sesja 64 BUG-2 — remember last click-to-seek so Space resumes
        // from this point instead of always 0:00.
        lastSeekSec_ = sec;

        if (previewController_.isPlaying())
        {
            previewController_.seek (sec);
            waveformView_.setPlayhead (sec);
            return;
        }
        if (tmpWavPath_.isEmpty()) return;

        const double endSec = selectedRange_.has_value() ? selectedRange_->endSec : -1.0;
        const auto r = previewController_.play (tmpWavPath_, sec, endSec);
        if (r == reamix::ui::PreviewController::PlayResult::OK)
        {
            transportBar_.setState (reamix::ui::TransportState::Playing);
            statusBar_.setText (selectedRange_.has_value() ? "Playing selection"
                                                             : "Playing preview");
        }
    };
    waveformView_.onSelectionChanged = [this] (std::optional<reamix::ui::WaveformView::SelectionRange> r)
    {
        selectedRange_ = r;

        // Sesja 60 — in Region mode, the waveform selection IS the region.
        // Re-resolve and propagate to panel + pipeline. In Duration mode
        // selection stays preview-only (existing Space-plays-selection
        // behavior from sesja 47/48).
        if (appMode_ == reamix::ui::ModeTabs::Mode::Region)
        {
            recomputeRegionState();
            return;
        }

        if (r.has_value())
        {
            if (previewController_.isPlaying())
            {
                previewController_.updateRange (r->startSec, r->endSec);
                waveformView_.setPlayhead (r->startSec);
                statusBar_.setText ("Playing selection");
            }
            else
            {
                statusBar_.setText ("Selection ready \xe2\x80\x94 Space to play, Esc to clear");
            }
        }
        else
        {
            if (previewController_.isPlaying())
                previewController_.clearRange();
            statusBar_.setText ("Selection cleared");
        }
    };

    // ── ModeTabs onChange (sesja 60) ─────────────────────────────────
    // User clicked a tab. Update appMode_, drop the AUTO flag (this is now
    // a manual choice), snapshot the current REAPER time-selection so the
    // auto-detector won't immediately re-flip back to Region.
    // ADR-050 Filozofia A: reset material view (waveform → Source variant +
    // clear selection + slider → original).
    modeTabs_.onChange = [this] (reamix::ui::ModeTabs::Mode m)
    {
        if (m == appMode_) return; // no-op click on already-active tab

        // Sesja 65 — capture the outgoing mode's slider target so a future
        // switch back can restore it. Save unconditionally — covers the
        // case where onTargetChanged didn't fire (programmatic setTarget
        // on item attach, no manual slider movement before this click).
        const reamix::ui::ModeTabs::Mode oldMode = appMode_;
        if (currentSourcePath_.isNotEmpty())
            targetByPathMode_[currentItemKey()][oldMode] = durationPanel_.getTarget();

        appMode_ = m;
        regionFromAuto_ = false;
        lastRespectedTimeSelection_ = reamix::reaper::getTimeSelection();

        // DEV-079 sesja 101 — snap AuditionBar UI thumbs to the new mode's
        // persisted slider positions (or bit-exact defaults if no entry yet).
        // Setters do NOT fire onChanged callbacks (per AuditionBar.h:54), so
        // this does not re-trigger kickAuditionRerun. The audition hash that
        // currentAuditionParams() now returns matches what was stored when
        // mode m's last remix was computed → tryRestoreModeRemix cache HIT.
        {
            const auto ap = currentAuditionParams();
            auditionBar_.setTone           (ap.tone);
            auditionBar_.setEditLength     (ap.editLength);
            auditionBar_.setAllowPmSeconds (ap.allowPmSeconds);
            auditionBar_.setMinCutBeats    (ap.minCutBeats);
        }

        // ADR-097 sesja 107 — sync advanced panel state (when open) on mode
        // switch. DEV-079 sesja 101: each tab is its own workspace; panel UI
        // reflects mode's persisted weights or kDefaultQualityWeights.
        // setRawWeights does NOT fire onAnyChanged (suppress flag inside
        // panel) so this does not re-trigger kickDevCalibrationRerun.
        if (advancedPanel_)
        {
            advancedPanel_->setRawWeights (currentQualityWeights());
            bool itemHasOverride = false;
            if (auto pmIt = qualityWeightsByPathMode_.find (currentItemKey());
                pmIt != qualityWeightsByPathMode_.end())
                itemHasOverride = ! pmIt->second.empty();
            advancedPanel_->setBadgeState (
                itemHasOverride
                  ? reamix::ui::AdvancedWeightsPanel::BadgeState::Modified
                  : reamix::ui::AdvancedWeightsPanel::BadgeState::NoSave);

            // ADR-087 STATUS UPDATE 1 D15 (sesja 98) — mode-aware β-section.
            switch (m)
            {
                case reamix::ui::ModeTabs::Mode::Duration:
                    advancedPanel_->setAppMode (reamix::ui::AdvancedWeightsPanel::AppMode::Duration);
                    break;
                case reamix::ui::ModeTabs::Mode::Region:
                    advancedPanel_->setAppMode (reamix::ui::AdvancedWeightsPanel::AppMode::Region);
                    break;
                case reamix::ui::ModeTabs::Mode::Blocks:
                    advancedPanel_->setAppMode (reamix::ui::AdvancedWeightsPanel::AppMode::Blocks);
                    break;
            }
        }

        // Sesja 65 — record user intent so cross-item attach (test 59)
        // and post-deselect re-attach respect the mode user was last in
        // for this source. See lastModeByPath_ comment in header.
        if (currentSourcePath_.isNotEmpty())
            lastModeByPath_[currentItemKey()] = m;

        // Sesja 65 — per-mode RemixCache lookup. Each tab is its own
        // workspace: Duration / Region / Blocks each persists its own
        // slider target + cached remix output across tab switches. User
        // reported the previous symmetric peek-and-restore (which read
        // currentRemix_/Mode_) only preserved the LAST calculated mode —
        // calculating in mode B overwrote currentRemix_ and the mode-A
        // remix was lost on the next switch back.
        //
        // Lookup logic mirrors kickRemixPipeline's cache-hit path:
        //   - Duration: key(source, T_D, 0, 0, blocked, variation)
        //   - Region:   key(source, T_R, regionStart, regionEnd, blocked, variation)
        //   - Blocks:   key(source, T_B, 0, 0, blocked, variation) + blocksHash
        // Cache hit → applyRemixToUi instantly + slider/scrim restored.
        // Miss → resetMaterialView, user re-computes.
        const bool restored = tryRestoreModeRemix (m);

        if (! restored)
            resetMaterialView();

        recomputeRegionState();

        // DEV-052 (sesja 100b) — discoverability hint when user enters
        // Region mode with a valid (≥ 6 s) time-selection. Drag-select on
        // the waveform refines the Region range; the affordance is not
        // obvious from the static tab + range readout alone. One-shot per
        // plugin lifetime — sufficient as an educational nudge without
        // becoming noise on every tab click.
        if (m == reamix::ui::ModeTabs::Mode::Region
            && ! regionDragSelectHintShown_
            && currentRegion_.has_value()
            && (currentRegion_->endSec - currentRegion_->startSec) >= 6.0)
        {
            statusBar_.setNotice ("Drag in waveform to refine selection", 4000);
            regionDragSelectHintShown_ = true;
        }
    };

    // ── SpliceMarker wiring (Remix variant, session 57) ──────────────
    waveformView_.onSpliceClick = [this] (int /*idx*/, double tSec)
    {
        // Audition seam — ±1 s window around the marker.
        if (tmpWavPath_.isEmpty()) return;
        const double startSec = std::max (0.0, tSec - 1.0);
        const double endSec   = tSec + 1.0;
        const auto r = previewController_.play (tmpWavPath_, startSec, endSec);
        if (r == reamix::ui::PreviewController::PlayResult::OK)
        {
            transportBar_.setState (reamix::ui::TransportState::Playing);
            statusBar_.setText ("Auditioning splice");
        }
    };

    waveformView_.onSpliceContextMenu = [this] (int idx, juce::Point<int> screenPos)
    {
        if (! currentRemix_.has_value()) return;
        if (idx < 0 || idx >= (int) currentRemix_->transitionFromBeats.size()) return;

        // DEV-060 sesja 100c — Blocks-mode splice marker shows seam-pill
        // semantyka (Re-roll splice / Reset to best / Audition junction)
        // instead of the generic Duration/Region menu. Splice marker idx in
        // Blocks Remix variant maps 1:1 to junctionIdx (each junction
        // produces exactly one transition entry).
        if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks
            && idx < (int) userBlocksQueue_.size() - 1)
        {
            const int junctionIdx = idx;
            junctionVariations_.resize (std::max ((int) junctionVariations_.size(),
                                                  (int) userBlocksQueue_.size() - 1), 0);
            const int currentVariation = junctionVariations_[(std::size_t) junctionIdx];

            juce::PopupMenu m;
            m.addItem (1, "Try different splice");
            m.addItem (2, "Reset to best", currentVariation > 0);
            m.addSeparator();
            m.addItem (3, "Audition junction");

            juce::PopupMenu::Options opts;
            opts = opts.withTargetScreenArea ({ screenPos.x, screenPos.y, 1, 1 });
            m.showMenuAsync (opts, [this, junctionIdx] (int chosen)
            {
                handleBlocksJunctionAction (junctionIdx, chosen);
            });
            return;
        }

        const int fromBeat = currentRemix_->transitionFromBeats[(std::size_t) idx];
        const int toBeat   = currentRemix_->transitionToBeats  [(std::size_t) idx];
        const double markerTime = (idx < (int) currentRemix_->transitionTimesSec.size())
                                  ? currentRemix_->transitionTimesSec[(std::size_t) idx]
                                  : 0.0;
        const juce::String sourcePath = currentSourcePath_;

        reamix::ui::ContextMenu::showSpliceMenuAsync (
            waveformView_, screenPos,
            [this, fromBeat, toBeat, markerTime, sourcePath]
            (reamix::ui::ContextMenu::SpliceResult res)
        {
            using SR = reamix::ui::ContextMenu::SpliceResult;
            if (sourcePath != currentSourcePath_) return; // user switched item

            PendingRemixOp op;
            op.targetSec           = durationPanel_.getTarget();
            op.blockedTransitions  = blockedBySource_[currentSourcePath_];
            // Sesja 60 — preserve current region when retrying / resetting /
            // blocking from the splice context menu. BUG-4 family fix sesja 63
            // — same effectiveRegion derivation as the slider / Insert paths;
            // currentRegion_ may be nullopt while currentRemix_ still carries
            // the originating region bounds.
            if (auto eff = deriveEffectiveRegion (currentRegion_, currentRemix_))
            {
                op.regionStartSec = eff->startSec;
                op.regionEndSec   = eff->endSec;
            }

            switch (res)
            {
                case SR::TryDifferent:
                {
                    int& v = variationBySource_[currentSourcePath_];
                    v += 1;
                    op.variation = v;
                    // Sesja 64 BUG-3 — snapshot clips for "exhausted" check
                    // in handleRemixComplete. If new clips match this, plugin
                    // emits "No more alternatives" and reverts v.
                    if (currentRemix_.has_value())
                        tryDifferentSnapshot_ = currentRemix_->editPlan.clips;
                    else
                        tryDifferentSnapshot_.reset();
                    statusBar_.setText ("Trying different splice");
                    kickRemixPipeline (op);
                    break;
                }
                case SR::ResetToBest:
                {
                    blockedBySource_.erase (currentSourcePath_);
                    variationBySource_[currentSourcePath_] = 0;
                    op.blockedTransitions.clear();
                    op.variation = 0;
                    statusBar_.setText ("Reset to best splices");
                    kickRemixPipeline (op);
                    break;
                }
                case SR::Block:
                {
                    auto& set = blockedBySource_[currentSourcePath_];
                    set.insert ({fromBeat, toBeat});
                    op.blockedTransitions = set;
                    op.variation          = variationBySource_[currentSourcePath_];
                    statusBar_.setText ("Splice blocked \xe2\x80\x94 re-routing");
                    kickRemixPipeline (op);
                    break;
                }
                case SR::Seek:
                {
                    if (waveformView_.onSeek) waveformView_.onSeek (markerTime);
                    break;
                }
                case SR::None:
                    break;
            }
        });
    };

    waveformView_.onSpliceHoverChanged = [this] (int idx, juce::Point<int> screenPos)
    {
        if (idx < 0 || ! currentRemix_.has_value())
        {
            tooltip_.hide();
            return;
        }
        const auto& cr = *currentRemix_;
        if (idx >= (int) cr.transitionFromBeats.size())
        {
            tooltip_.hide();
            return;
        }

        reamix::ui::Tooltip::Data d;
        const auto& fromL = cr.transitionFromLabels[(std::size_t) idx];
        const auto& toL   = cr.transitionToLabels  [(std::size_t) idx];
        if (fromL.isNotEmpty() && toL.isNotEmpty())
            d.title = "Splice \xc2\xb7 " + fromL + " \xe2\x86\x92 " + toL;
        else
            d.title = juce::String::fromUTF8 ("Splice");

        const float quality = cr.transitionQualities[(std::size_t) idx];
        const int   pct     = (int) std::round (quality * 100.0f);
        d.rows.push_back ({"quality", juce::String (pct) + "%"});

        const float dE = cr.transitionEnergyDiffsDb[(std::size_t) idx];
        const juce::String dEStr = (dE > 0 ? "+" : "") + juce::String (dE, 1) + " dB";
        d.rows.push_back ({juce::String::fromUTF8 ("\xce\x94" "Energy"), dEStr});

        d.rows.push_back ({"beat", juce::String (cr.transitionFromBeats[(std::size_t) idx])
                                 + juce::String::fromUTF8 (" \xe2\x86\x92 ")
                                 + juce::String (cr.transitionToBeats  [(std::size_t) idx])});

        d.qbarPct = quality;
        if (quality > 0.7f)      d.qbarColour = reamix::theme::Good;
        else if (quality >= 0.5f) d.qbarColour = reamix::theme::Warn;
        else                      d.qbarColour = reamix::theme::Bad;

        d.hint = juce::String::fromUTF8 ("Click to audition \xc2\xb7 Right-click for options");
        tooltip_.setData (std::move (d));

        // Position above the marker, in MainComponent-local coords. Tooltip
        // clamps itself to parent bounds in showAt — lift off the marker by
        // ~140 px so it doesn't sit under the cursor.
        const auto local = getLocalPoint (nullptr, screenPos);
        tooltip_.showAt ({ local.x - 90, std::max (4, local.y - 140) });
    };

    addKeyListener (&transportBar_);

    onShowBeatsToggled = [this] (bool yes) { waveformView_.setShowBeats (yes); };
    onIsShowBeats      = [this]            { return waveformView_.getShowBeats(); };

    headerBar_.onGearClicked = [this]
    {
        if (settingsPopover_.isOpen())
        {
            settingsPopover_.hideMe();
        }
        else
        {
            settingsPopover_.setDocked    (onIsDocked     ? onIsDocked()     : false);
            settingsPopover_.setShowBeats (onIsShowBeats  ? onIsShowBeats()  : true);
            // Sesja 60 — propagate current snap mode from WaveformView.
            using SnapMode   = reamix::ui::WaveformView::SnapMode;
            using SnapRegion = reamix::ui::SettingsPopover::SnapRegion;
            const SnapMode m = waveformView_.getSnapMode();
            settingsPopover_.setSnapRegion (
                m == SnapMode::Beats     ? SnapRegion::Beats
                : m == SnapMode::Downbeats ? SnapRegion::Downbeats
                                           : SnapRegion::Off);
            // Sesja 100b (DEV-049) — propagate Insert toggles from
            // persisted state so the popover paints the current values.
            settingsPopover_.setInsertSpliceMarkers (insertSpliceMarkersEnabled_);
            settingsPopover_.setInsertRenderRegion  (insertRenderRegionEnabled_);
            // ADR-053 — refresh cache stats on every show (changes since
            // last open: new analyses cached, user cleared elsewhere).
            settingsPopover_.setCacheStats (
                reamix::ui::AnalysisDiskCache::totalSizeBytes(),
                reamix::ui::AnalysisDiskCache::countEntries());
            settingsPopover_.show();
        }
    };

    // ADR-053 — cache action handlers. Clear deletes every .bundle file
    // and re-displays the now-empty stats so the user sees the action took
    // effect without re-opening the popover. Reveal opens Finder at the
    // cache directory so the user can poke around / verify size manually.
    settingsPopover_.onClearCache = [this]
    {
        const int n = reamix::ui::AnalysisDiskCache::clearAll();
        settingsPopover_.setCacheStats (
            reamix::ui::AnalysisDiskCache::totalSizeBytes(),
            reamix::ui::AnalysisDiskCache::countEntries());

        // Sesja 65 — Clear cache must be an honest reset. Disk wipe alone
        // leaves in-memory bundles + currentRemix_ alive, so SourcePanel
        // keeps showing "Analyze Again" and the waveform keeps showing
        // splice markers from a phantom analysis. User reported this in
        // test 57. Drop all session-scoped analysis state too.
        analysisBundles_.clear();
        blockedBySource_.clear();
        variationBySource_.clear();
        clearCurrentRemix();
        resetMaterialView();
        sourcePanel_.setHasAnalysis (false);
        sourcePanel_.setAnalyzing (false, 0.0);
        analysisState_ = AnalysisState::Idle;
        headerBar_.setStatusKind (reamix::ui::HeaderStatus::Ready);
        transportBar_.setState (reamix::ui::TransportState::Idle);
        if (currentSourcePath_.isNotEmpty())
            tmpWavPath_ = currentSourcePath_;

        statusBar_.setNotice (juce::String::fromUTF8 ("Cleared ")
            + juce::String (n)
            + juce::String (n == 1 ? " cached analysis" : " cached analyses"));
    };
    settingsPopover_.onRevealCache = []
    {
        reamix::ui::AnalysisDiskCache::revealInFinder();
    };

    settingsPopover_.onDockToggled = [this]
    {
        if (onToggleDock) onToggleDock();
    };
    settingsPopover_.onShowBeatsToggled = [this] (bool yes)
    {
        if (onShowBeatsToggled) onShowBeatsToggled (yes);
    };

    // Sesja 100b (DEV-049) — Insert decoration toggles. Persist to
    // ExtState immediately so toggle state survives plugin reload.
    settingsPopover_.onInsertSpliceMarkersToggled = [this] (bool yes)
    {
        insertSpliceMarkersEnabled_ = yes;
#if REAMIX_WITH_REAPER_IO
        if (SetExtState)
            SetExtState ("reamix.me", "insert_splice_markers",
                          yes ? "1" : "0", true);
#endif
    };
    settingsPopover_.onInsertRenderRegionToggled = [this] (bool yes)
    {
        insertRenderRegionEnabled_ = yes;
#if REAMIX_WITH_REAPER_IO
        if (SetExtState)
            SetExtState ("reamix.me", "insert_render_region",
                          yes ? "1" : "0", true);
#endif
    };

    // ADR-057 (sesja 68) — Splice flexibility cycler removed; β-model in
    // DEV-040 supersedes the junction ±W search this used to control.

    // Sesja 60 Plan A — Edit region overlay button handler. Flip waveform back
    // to Source variant, restore the selection that produced the current remix,
    // hide the button. The slider stays as-is (target unchanged); user can
    // re-drag in the waveform to update the region (triggers a new remix +
    // flips back to Remix + button reappears).
    waveformView_.onEditRegionClicked = [this]
    {
        // DEV-036 (sesja 63b) — three pre-existing cleanups missing from the
        // sesja-60 onEditRegionClicked introduction:
        //   1. Stop in-flight preview — was playing the remix WAV that we are
        //      about to stop showing; user clicking Edit means "back to the
        //      source view" so the audio should reset too.
        //   2. Hide the splice tooltip — it referenced a marker on the Remix
        //      variant we are flipping away from; sticks visible otherwise
        //      (user-reported "stuck tooltip in the middle" sesja 63b).
        //   3. Swap preview source from remix WAV to currentSourcePath_ — the
        //      next click in the waveform should preview the original audio
        //      under the scrim, not the remix output (user-reported "I see
        //      source waveform but preview keeps playing the first remix").
        previewController_.stop();
        tooltip_.hide();
        tmpWavPath_ = currentSourcePath_;

        waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
        if (peaksProvider_)
            waveformView_.setPeaksProvider (peaksProvider_.get());
        waveformView_.setSpliceMarkers ({});
        waveformView_.setEditPlan ({});
        if (lastRegionSelection_.has_value())
        {
            waveformView_.setSelection (*lastRegionSelection_);
            selectedRange_ = *lastRegionSelection_;
        }
        waveformView_.setShowEditRegionButton (false);

        // Sesja 65 — clearing the cached remix is the only way the
        // sesja-65 symmetric peek-and-restore (modeTabs_.onChange) does
        // NOT bring this view back when user tab-peeks Blocks → Region.
        // Edit region is an explicit "abandon this remix view" action.
        clearCurrentRemix();

        // Sesja 65 follow-up — also erase the per-source Region snapshot
        // so cross-item attach (applySelectedItem) does NOT auto-restore
        // the abandoned remix when the user comes back to this source.
        // Without this the lastRegionByPath_ entry survives Edit region
        // and re-fires the full restore on the next attach (user-reported
        // test 59 follow-up: Region remix → Edit region → switch item B →
        // back to A → splice view re-appears, undoing the explicit dismiss).
        if (currentSourcePath_.isNotEmpty())
            lastRegionByPath_.erase (currentItemKey());

        // ADR-056 (sesja 66 fix A) — also dismiss any saved Region group
        // entry for this item so the step-6 dispatch detection does not
        // re-prime lastRegionByPath_ from P_EXT on the next attach. Region
        // group entry stays alive in P_EXT (other future sessions still see
        // it); only the current in-memory dismiss intent is recorded so
        // user's "I clicked Edit region" survives navigation. Cleared on
        // Region Insert (group recreated → re-engagement).
        if (currentSourcePath_.isNotEmpty())
            regionGroupDismissedSources_.insert (currentItemKey());

        statusBar_.setText (juce::String::fromUTF8 (
            "Drag in waveform to change region selection"));
    };

    // ADR-051 sesja-61 hot-fix — Edit arrangement overlay flips waveform back
    // to Source variant so user can re-edit blocks/queue post-Insert / post-
    // Re-analyze. State (userBlocks_/queue/junctionVariations_) is preserved
    // throughout — variant flip is purely visual.
    waveformView_.onEditArrangementClicked = [this]
    {
        waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
        if (peaksProvider_)
            waveformView_.setPeaksProvider (peaksProvider_.get());
        waveformView_.setSpliceMarkers ({});
        waveformView_.setEditPlan ({});
        waveformView_.setShowEditArrangementButton (false);

        // Sesja 65 — abandon the assembled remix so the symmetric peek-
        // and-restore in modeTabs_.onChange doesn't bring it back when
        // user tab-peeks Region/Duration → Blocks. Symmetric to the
        // onEditRegionClicked clearCurrentRemix call.
        clearCurrentRemix();

        // Sesja 65 follow-up — also flag the source as "Blocks dismissed"
        // so cross-item attach (applySelectedItem) does NOT auto-restore
        // the assembled view via the kickRemixPipeline cache hit. Without
        // this, the userBlocks/queue/blocksHash combo on the next attach
        // hits the cached entry and restores the dismissed view (test 59
        // user follow-up: Edit arrangement → switch item → back → again
        // assembly view). Cleared on next Assemble click.
        if (currentSourcePath_.isNotEmpty())
            blocksDismissedSources_.insert (currentItemKey());

        // segBar paint resumes user-block tiles + algorithm splice lines;
        // userBlocks_ already pushed via refreshBlockAssemblyUi.
        statusBar_.setText (juce::String::fromUTF8 (
            "Editing arrangement \xE2\x80\x94 marked blocks + queue preserved"));
    };

    // ADR-051 — Block Assembly drag-mark on segBar surfaces a kind picker.
    // Append a UserBlock at confirmed kind, persist to P_EXT, repaint.
    waveformView_.onMarkBlock = [this] (double startSec, double endSec,
                                         juce::Point<int> screenPos)
    {
        if (currentSourceDurationSec_ <= 0.0) return;

        // Sesja 64 BUG-5 — clamp new block to nearest existing-block edges
        // so blocks never overlap. userBlocks_ is sorted by startSec; iterate
        // and clip the new range whenever it crosses an existing edge.
        double cs = startSec;
        double ce = endSec;
        for (const auto& blk : userBlocks_)
        {
            if (ce <= blk.startSec || cs >= blk.endSec) continue; // no overlap
            if (cs >= blk.startSec && ce <= blk.endSec)
            {
                statusBar_.setNotice (juce::String::fromUTF8 (
                    "New block overlaps existing \xe2\x80\x94 cannot create"));
                return;
            }
            if (cs < blk.startSec && ce > blk.endSec)
                ce = blk.startSec;       // existing inside new → take left half
            else if (cs < blk.startSec)
                ce = blk.startSec;       // partial right
            else
                cs = blk.endSec;         // partial left
        }
        if (ce - cs < 0.5)
        {
            statusBar_.setNotice (juce::String::fromUTF8 (
                "Block too small after clamp \xe2\x80\x94 try a larger range"));
            return;
        }
        const double clampedStart = cs;
        const double clampedEnd   = ce;

        const auto smartDefault =
            reamix::ui::smartKindForPosition (clampedStart, currentSourceDurationSec_);

        auto picker = std::make_unique<reamix::ui::BlockKindPickerPopover> (
            smartDefault, clampedStart, clampedEnd, &customKindRegistry_);

        // ADR-092 sesja 100c — onPicked carries optional customKindId.
        // Built-in pick: customKindId = nullopt; userBlock.kind = picked enum.
        // Custom pick: customKindId = registry id; userBlock.kind stays Verse
        // fallback (so cross-machine display has something sensible if the
        // user opens project on another machine without the registry entry).
        picker->onPicked = [this, clampedStart, clampedEnd]
                           (reamix::theme::SegmentKind kind,
                            std::optional<juce::String> customKindId)
        {
            reamix::ui::UserBlock b;
            b.startSec     = clampedStart;
            b.endSec       = clampedEnd;
            b.kind         = kind;
            b.customKindId = customKindId;
            userBlocks_.push_back (b);
            std::sort (userBlocks_.begin(), userBlocks_.end(),
                       [](const auto& a, const auto& z)
                       { return a.startSec < z.startSec; });
            persistUserBlocks();
            refreshBlockAssemblyUi();
            clearCurrentRemix(); // userBlocks changed → existing remix stale
            if (userBlocksQueue_.size() >= 2)
                blockAssemblyPanel_.setDirty (true);

            statusBar_.setText (juce::String::fromUTF8 (
                "Block added \xe2\x80\x94 ") + juce::String ((int) userBlocks_.size())
                + (userBlocks_.size() == 1 ? " block" : " blocks"));
        };

        // ADR-092 sesja 100c — "+ Add custom" tile clicked. Picker stays
        // open behind the Add modal (sesja 100c iter 3 UX); on confirm we
        // create registry entry, commit marked region, dismiss picker. On
        // cancel picker remains so user can pick a different kind.
        picker->onAddCustomRequested =
            [this, clampedStart, clampedEnd, screenPos]
            (juce::Component::SafePointer<juce::Component> pickerSafe)
        {
            showAddCustomKindModal (clampedStart, clampedEnd, screenPos, pickerSafe);
        };

        // ADR-092 sesja 100c — right-click on custom tile → user picked an
        // edit action. Picker stays open (sesja 100c iter 2 — user can chain
        // edits without re-drag-marking); pickerSafe forwarded so the modal
        // callback can repaint the picker after registry mutation.
        picker->onEditCustomAction =
            [this] (juce::String id, reamix::ui::CustomKindAction action,
                    juce::Component::SafePointer<juce::Component> pickerSafe)
        {
            showEditCustomKindModal (id, action, pickerSafe);
        };

        // CallOutBox::launchAsynchronously takes ownership of `picker`. The
        // anchor is a 4×4 rect at the mouseUp screen position so the popover
        // points back to where the user finished marking.
        const juce::Rectangle<int> anchor { screenPos.x - 2, screenPos.y - 2, 4, 4 };
        juce::CallOutBox::launchAsynchronously (std::move (picker), anchor, nullptr);
    };

    // Sesja 100 iter 3 (DEV-018) — Preview volume changed via inline floating
    // slider on WaveformView (no CallOutBox; bypasses bubble background per
    // user directive). Live updates PreviewController so user hears the
    // change while dragging + persists to ExtState.
    waveformView_.onPreviewVolumeChanged = [this] (double linear01)
    {
        previewController_.setVolume (linear01);
#if REAMIX_WITH_REAPER_IO
        if (SetExtState)
        {
            SetExtState ("reamix.me", "preview_volume",
                         juce::String (linear01, 4).toRawUTF8(),
                         true);
        }
#endif
    };

    // DEV-029 (d) sesja 100b — direct boundary edit on segBar (drag the
    // edge with cursor). Mirrors the BlockEditPopover.onBoundariesChanged
    // mutation pattern: clamp to neighbour bounds, persist, refresh,
    // invalidate the assembled remix output. WaveformView already
    // performed the clamp + min-block-size guard before firing.
    waveformView_.onUserBlockBoundariesChanged =
        [this] (int idx, double newStart, double newEnd)
    {
        if (idx < 0 || idx >= (int) userBlocks_.size()) return;
        auto& b = userBlocks_[(std::size_t) idx];
        if (std::abs (b.startSec - newStart) < 1e-6
            && std::abs (b.endSec   - newEnd)   < 1e-6) return;
        b.startSec = newStart;
        b.endSec   = newEnd;
        // Sort-stable since we clamped to neighbour bounds; no resort
        // required. Persist + UI refresh + remix invalidate per the
        // existing nudger handler shape.
        persistUserBlocks();
        refreshBlockAssemblyUi();
        invalidateBlocksAssembledOutput();
        if (userBlocksQueue_.size() >= 2)
            blockAssemblyPanel_.setDirty (true);
    };

    waveformView_.onUserBlockClicked = [this] (int idx)
    {
        // ADR-051 phase H — open BlockEditPopover at the clicked block.
        if (idx < 0 || idx >= (int) userBlocks_.size()) return;

        const auto& b = userBlocks_[(std::size_t) idx];
        std::optional<reamix::ui::UserBlock> prevBlock;
        std::optional<reamix::ui::UserBlock> nextBlock;
        if (idx > 0) prevBlock = userBlocks_[(std::size_t) (idx - 1)];
        if (idx + 1 < (int) userBlocks_.size())
            nextBlock = userBlocks_[(std::size_t) (idx + 1)];

        // BPM from current bundle (drives ±1-beat fallback when no
        // beatTimes available). Sesja 64 BUG-7 — pass real beatTimes for
        // boundary-snapping nudge; fallback to bpm-shift if bundle missing.
        double bpm = 120.0;
        std::vector<double> beatTimes;
        auto bIt = analysisBundles_.find (currentSourcePath_);
        if (bIt != analysisBundles_.end() && bIt->second != nullptr)
        {
            bpm        = bIt->second->bpm;
            beatTimes  = bIt->second->beatTimes;
        }

        auto popover = std::make_unique<reamix::ui::BlockEditPopover> (
            b, currentSourceDurationSec_, bpm, prevBlock, nextBlock,
            std::move (beatTimes), &customKindRegistry_);

        popover->onKindChanged = [this, idx]
            (reamix::theme::SegmentKind k, std::optional<juce::String> customKindId)
        {
            if (idx < 0 || idx >= (int) userBlocks_.size()) return;
            userBlocks_[(std::size_t) idx].kind         = k;
            userBlocks_[(std::size_t) idx].customKindId = customKindId;
            persistUserBlocks();
            refreshBlockAssemblyUi();
            clearCurrentRemix();
            blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
            statusBar_.setText (juce::String::fromUTF8 ("Block re-labeled"));
        };

        // ADR-092 sesja 100c — Add custom + edit-action handlers (BlockEdit
        // parity with BlockKindPickerPopover). Add only mutates registry +
        // repaints popover; user must click the new tile to apply it to
        // this block. Edit actions reuse the picker's modal flow.
        popover->onAddCustomRequested =
            [this] (juce::Component::SafePointer<juce::Component> popoverSafe)
        {
            // Reuse showAddCustomKindModal; in edit context we don't have
            // a freshly-marked region, so pass dummy clamped range. Modal
            // confirm path adds to registry + repaints popover; no region
            // commit (existing block stays as-is until user picks tile).
            //
            // Note: showAddCustomKindModal's lambda body conditionally
            // commits a new UserBlock. Using a registry-only variant would
            // be cleaner but requires duplicating modal scaffolding. As
            // mitigation, we route through a dedicated edit-context path.
            showAddCustomKindModalEditContext (popoverSafe);
        };
        popover->onEditCustomAction =
            [this] (juce::String id, reamix::ui::CustomKindAction action,
                    juce::Component::SafePointer<juce::Component> popoverSafe)
        {
            showEditCustomKindModal (id, action, popoverSafe);
        };
        popover->onBoundariesChanged = [this, idx] (double newStart, double newEnd)
        {
            if (idx < 0 || idx >= (int) userBlocks_.size()) return;
            userBlocks_[(std::size_t) idx].startSec = newStart;
            userBlocks_[(std::size_t) idx].endSec   = newEnd;
            persistUserBlocks();
            refreshBlockAssemblyUi();
            clearCurrentRemix();
            blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
        };
        popover->onDelete = [this, idx]
        {
            if (idx < 0 || idx >= (int) userBlocks_.size()) return;
            const std::size_t deletedIdx = (std::size_t) idx;
            userBlocks_.erase (userBlocks_.begin() + (long) deletedIdx);
            persistUserBlocks();
            // Drop matching queue entries and shift indices > deletedIdx.
            auto& q = userBlocksQueue_;
            q.erase (std::remove_if (q.begin(), q.end(),
                                      [deletedIdx] (int qi) { return qi == (int) deletedIdx; }),
                     q.end());
            for (auto& qi : q) if (qi > (int) deletedIdx) --qi;
            refreshBlockAssemblyUi();
            clearCurrentRemix();
            blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
            statusBar_.setText (juce::String::fromUTF8 (
                "Block deleted \xE2\x80\x94 ") + juce::String ((int) userBlocks_.size())
                + (userBlocks_.size() == 1 ? " block remaining" : " blocks remaining"));
        };

        // Sesja 100 (DEV-029) — Split / Merge Left / Merge Right.
        popover->onSplit = [this, idx]
        {
            if (idx < 0 || idx >= (int) userBlocks_.size()) return;
            const std::size_t splitIdx = (std::size_t) idx;
            const auto orig = userBlocks_[splitIdx];
            const double midSec = 0.5 * (orig.startSec + orig.endSec);
            // Block A: [orig.startSec, midSec]; Block B: [midSec, orig.endSec].
            // Both keep orig.kind. Mutate in-place + insert second after.
            reamix::ui::UserBlock blockA = orig;
            blockA.endSec = midSec;
            reamix::ui::UserBlock blockB = orig;
            blockB.startSec = midSec;
            userBlocks_[splitIdx] = blockA;
            userBlocks_.insert (userBlocks_.begin() + (long) (splitIdx + 1), blockB);
            // Queue rewrite: every queue index > splitIdx shifts +1 (we inserted
            // one new block immediately after splitIdx; existing splitIdx still
            // refers to blockA).
            for (auto& qi : userBlocksQueue_)
                if (qi > (int) splitIdx) ++qi;
            persistUserBlocks();
            refreshBlockAssemblyUi();
            clearCurrentRemix();
            blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
            statusBar_.setText (juce::String::fromUTF8 (
                "Block split \xe2\x80\x94 ") + juce::String ((int) userBlocks_.size())
                + " blocks");
        };

        popover->onMergeLeft = [this, idx]
        {
            if (idx < 1 || idx >= (int) userBlocks_.size()) return;
            const std::size_t curIdx  = (std::size_t) idx;
            const std::size_t prevIdx = curIdx - 1;
            // Resulting block spans [prev.startSec, cur.endSec], keeps cur.kind.
            userBlocks_[curIdx].startSec = userBlocks_[prevIdx].startSec;
            userBlocks_.erase (userBlocks_.begin() + (long) prevIdx);
            // Queue: drop all entries == prevIdx, shift entries > prevIdx by -1.
            // Resulting block (formerly curIdx) is now at prevIdx.
            auto& q = userBlocksQueue_;
            q.erase (std::remove_if (q.begin(), q.end(),
                                      [prevIdx] (int qi) { return qi == (int) prevIdx; }),
                     q.end());
            for (auto& qi : q) if (qi > (int) prevIdx) --qi;
            persistUserBlocks();
            refreshBlockAssemblyUi();
            clearCurrentRemix();
            blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
            statusBar_.setText (juce::String::fromUTF8 (
                "Blocks merged \xe2\x80\x94 ") + juce::String ((int) userBlocks_.size())
                + " blocks");
        };

        popover->onMergeRight = [this, idx]
        {
            if (idx < 0 || idx + 1 >= (int) userBlocks_.size()) return;
            const std::size_t curIdx  = (std::size_t) idx;
            const std::size_t nextIdx = curIdx + 1;
            // Resulting block spans [cur.startSec, next.endSec], keeps cur.kind.
            userBlocks_[curIdx].endSec = userBlocks_[nextIdx].endSec;
            userBlocks_.erase (userBlocks_.begin() + (long) nextIdx);
            // Queue: drop entries == nextIdx, shift entries > nextIdx by -1.
            auto& q = userBlocksQueue_;
            q.erase (std::remove_if (q.begin(), q.end(),
                                      [nextIdx] (int qi) { return qi == (int) nextIdx; }),
                     q.end());
            for (auto& qi : q) if (qi > (int) nextIdx) --qi;
            persistUserBlocks();
            refreshBlockAssemblyUi();
            clearCurrentRemix();
            blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
            statusBar_.setText (juce::String::fromUTF8 (
                "Blocks merged \xe2\x80\x94 ") + juce::String ((int) userBlocks_.size())
                + " blocks");
        };

        const juce::Rectangle<int> anchor { 200, 200, 4, 4 }; // overridden below
        // Anchor at the screen position of the segBar block; approximate via
        // global mouse pos at click time.
        const auto screenAnchor = juce::Desktop::getInstance().getMainMouseSource().getScreenPosition();
        const juce::Rectangle<int> anchorRect {
            (int) screenAnchor.x - 2, (int) screenAnchor.y - 2, 4, 4 };
        juce::CallOutBox::launchAsynchronously (std::move (popover), anchorRect, nullptr);
        juce::ignoreUnused (anchor);
    };

    // ADR-051 phase E — algorithm-resolved splice hover → rich tooltip.
    waveformView_.onUserBlockSpliceHover =
        [this] (int spliceIdx, juce::Point<int> screenPos)
    {
        if (spliceIdx < 0)
        {
            tooltip_.hide();
            return;
        }
        // Splice data lives in the MainComponent shadow (lastBlockSplices_) —
        // WaveformView's vector is intentionally write-only from outside.
        // Phase J populates this on RemixPipeline Blocks-mode completion.
        if (spliceIdx >= (int) lastBlockSplices_.size())
        {
            tooltip_.hide();
            return;
        }
        const auto& sp = lastBlockSplices_[(std::size_t) spliceIdx];

        reamix::ui::Tooltip::Data d;
        d.title = juce::String::fromUTF8 ("SPLICE \xC2\xB7 ALGORITHM RESOLVED");
        d.rows.push_back ({ "quality",
                            juce::String ((int) std::round (sp.qualityScore * 100.0f)) + "%" });
        d.rows.push_back ({ "drift",
                            juce::String (sp.driftBeats, 1) + " beats" });
        d.qbarPct    = juce::jlimit (0.0f, 1.0f, sp.qualityScore);
        d.qbarColour = (sp.qualityScore >= 0.7f) ? juce::Colour (0xFF44CC44)
                       : (sp.qualityScore >= 0.5f) ? juce::Colour (0xFFCCCC44)
                                                    : juce::Colour (0xFFCC4444);
        d.hint       = juce::String::fromUTF8 (
            "Splice picked by algorithm within your block's flexibility window");
        tooltip_.setData (d);

        // Anchor at hover screen position, converted to tooltip parent-local.
        const auto parentLocal = getLocalPoint (nullptr, screenPos);
        tooltip_.showAt (parentLocal + juce::Point<int> (12, 12));
    };

    // ADR-051 phase C — paleta card → append index to queue, refresh UI.
    blockAssemblyPanel_.onCardClicked = [this] (int blockIdx)
    {
        if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
        userBlocksQueue_.push_back (blockIdx);
        refreshBlockAssemblyUi();
        if (userBlocksQueue_.size() >= 2)
        {
            blockAssemblyPanel_.setDirty (true);
            clearCurrentRemix(); // queue changed → existing remix stale
        }
        statusBar_.setText (juce::String::fromUTF8 (
            "Block added to queue \xe2\x80\x94 ")
            + juce::String ((int) userBlocksQueue_.size())
            + (userBlocksQueue_.size() == 1 ? " block" : " blocks"));
    };

    // DEV-058 (a) sesja 100d — paleta card right-click context menu.
    // Multi-select aware (DEV-058 (b)): when paletteSelected_.size() > 1
    // and the right-clicked card is in the selection, build a batch menu
    // (Delete N blocks / Change kind for N selected). Otherwise build the
    // single-tile menu (Rename / Change kind / Delete) and the click also
    // already replaced the selection to {idx} via mouseDown.
    blockAssemblyPanel_.onCardContextMenu = [this] (int idx, juce::Point<int> screenPos)
    {
        if (idx < 0 || idx >= (int) userBlocks_.size()) return;

        const auto& sel = blockAssemblyPanel_.getPaletteSelection();
        const bool isBatch = sel.size() > 1 && sel.find (idx) != sel.end();

        juce::PopupMenu m;
        if (isBatch)
        {
            const int n = (int) sel.size();
            m.addItem (10, juce::String ("Change kind for ") + juce::String (n) + " blocks");
            m.addSeparator();
            m.addItem (11, juce::String ("Delete ") + juce::String (n) + " blocks");
        }
        else
        {
            m.addItem (1, "Rename block");
            m.addItem (2, "Change kind");
            m.addSeparator();
            m.addItem (3, "Delete block");
        }

        const juce::Rectangle<int> anchor { screenPos.x, screenPos.y, 1, 1 };
        m.showMenuAsync (juce::PopupMenu::Options{}.withTargetScreenArea (anchor),
            [this, idx, screenPos, isBatch] (int picked)
            {
                if (picked == 0) return;
                if (isBatch)
                {
                    std::vector<int> indices (
                        blockAssemblyPanel_.getPaletteSelection().begin(),
                        blockAssemblyPanel_.getPaletteSelection().end());
                    if (picked == 10) showChangeKindBatchPicker  (std::move (indices), screenPos);
                    else if (picked == 11) showDeleteBlocksBatchConfirm (std::move (indices));
                    return;
                }
                if (picked == 1)      showRenameBlockDialog    (idx);
                else if (picked == 2) showChangeKindMiniPicker (idx, screenPos);
                else if (picked == 3) showDeleteBlockConfirm   (idx);
            });
    };

    // DEV-058 (b) sesja 100d — multi-select status bar feedback.
    blockAssemblyPanel_.onPaletteSelectionChanged =
        [this] (const std::set<int>& sel)
    {
        if (sel.size() <= 1) return; // single-select doesn't need callout
        statusBar_.setText (juce::String ((int) sel.size()) + " blocks selected");
    };

    // DEV-058 (c) sesja 100d — paleta drag-reorder commit.
    blockAssemblyPanel_.onPaletteReorder =
        [this] (int fromBlockIdx, int toBlockIdx)
    {
        commitPaletteReorder (fromBlockIdx, toBlockIdx);
    };

    // DEV-077 (NEW sesja 100d) — cross-section drag from paleta into queue.
    // Insert userBlocksQueue_ entry at queuePos referencing blockIdx;
    // analogous to onCardClicked (which appends to end) but with precise
    // gap-targeting per drop position. Existing dirty / clear-current-remix
    // semantics preserved.
    blockAssemblyPanel_.onPaletteToQueueInsert =
        [this] (int blockIdx, int queuePos)
    {
        if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
        const int n = (int) userBlocksQueue_.size();
        const int pos = juce::jlimit (0, n, queuePos);
        userBlocksQueue_.insert (userBlocksQueue_.begin() + pos, blockIdx);
        persistUserBlocks();
        refreshBlockAssemblyUi();
        clearCurrentRemix();
        blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
        statusBar_.setText (juce::String::fromUTF8 (
            "Block inserted into queue \xe2\x80\x94 ")
            + juce::String ((int) userBlocksQueue_.size())
            + (userBlocksQueue_.size() == 1 ? " block" : " blocks"));
    };

    // ADR-051 phase F — queue tile context menu (Remove / Move Up / Move Down).
    blockAssemblyPanel_.onQueueTileContextMenu =
        [this] (int queuePos, juce::Point<int> screenPos)
    {
        if (queuePos < 0 || queuePos >= (int) userBlocksQueue_.size()) return;

        juce::PopupMenu m;
        m.addItem (1, "Remove from queue");
        m.addSeparator();
        m.addItem (2, "Move up",   queuePos > 0);
        m.addItem (3, "Move down", queuePos + 1 < (int) userBlocksQueue_.size());

        juce::PopupMenu::Options opts;
        opts = opts.withTargetScreenArea ({ screenPos.x, screenPos.y, 1, 1 });
        m.showMenuAsync (opts, [this, queuePos] (int chosen)
        {
            if (queuePos < 0 || queuePos >= (int) userBlocksQueue_.size()) return;
            switch (chosen)
            {
                case 1: // Remove
                    userBlocksQueue_.erase (userBlocksQueue_.begin() + queuePos);
                    break;
                case 2: // Move up
                    if (queuePos > 0)
                        std::swap (userBlocksQueue_[(std::size_t) queuePos],
                                   userBlocksQueue_[(std::size_t) (queuePos - 1)]);
                    break;
                case 3: // Move down
                    if (queuePos + 1 < (int) userBlocksQueue_.size())
                        std::swap (userBlocksQueue_[(std::size_t) queuePos],
                                   userBlocksQueue_[(std::size_t) (queuePos + 1)]);
                    break;
                default: return;
            }
            refreshBlockAssemblyUi();
            // ADR-052 sesja 63 BUG-11/13/14/15 — Option B Assemble cleanup.
            // Queue mutation invalidates all per-junction visualization state;
            // user must click Assemble to repopulate. Without this, audition
            // / playhead / canvas markers fire on stale lastBlockSplices_.
            invalidateBlocksAssembledOutput();
            if (userBlocksQueue_.size() >= 2)
                blockAssemblyPanel_.setDirty (true);
            statusBar_.setText (juce::String::fromUTF8 (
                "Queue updated \xE2\x80\x94 ")
                + juce::String ((int) userBlocksQueue_.size())
                + (userBlocksQueue_.size() == 1 ? " block" : " blocks"));
        });
    };

    // DEV-030 (sesja 100b) — queue drag-drop reorder. Drop into gap `to`
    // means: take the tile at `from`, remove it, insert it at index
    // adjusted for the removal. mouseUp filters trivial drops (same
    // slot), so we always perform the move here.
    blockAssemblyPanel_.onQueueReorder = [this] (int from, int to)
    {
        const int n = (int) userBlocksQueue_.size();
        if (from < 0 || from >= n) return;
        if (to < 0 || to > n)      return;

        const int blockIdx = userBlocksQueue_[(std::size_t) from];
        userBlocksQueue_.erase (userBlocksQueue_.begin() + from);

        // After erase, indices >= from shifted left by 1. Adjust the
        // target gap when the dragged tile was BEFORE the drop site so
        // the insertion lands in the visually correct slot.
        const int adjustedTo = (to > from) ? (to - 1) : to;
        userBlocksQueue_.insert (userBlocksQueue_.begin() + adjustedTo, blockIdx);

        // Variations follow the same reorder so per-junction state lines
        // up with the new tile sequence. Resize-on-demand mirrors the
        // pattern used by onQueueTileContextMenu / onSeamContextMenu.
        if ((int) junctionVariations_.size() < n - 1)
            junctionVariations_.resize ((std::size_t) (n - 1), 0);

        refreshBlockAssemblyUi();
        invalidateBlocksAssembledOutput();
        if (userBlocksQueue_.size() >= 2)
            blockAssemblyPanel_.setDirty (true);
        persistUserBlocks();

        statusBar_.setText (juce::String::fromUTF8 (
            "Queue reordered \xE2\x80\x94 ")
            + juce::String ((int) userBlocksQueue_.size())
            + (userBlocksQueue_.size() == 1 ? " block" : " blocks"));
    };

    // ADR-051 phase G — seam context menu. ADR-092 / DEV-060 + DEV-076 sesja
    // 100c — action handling extracted to handleBlocksJunctionAction so the
    // Blocks-mode splice-marker context menu (DEV-060) shares the same
    // semantic + the same real-time-on-clean gate (DEV-076).
    blockAssemblyPanel_.onSeamContextMenu = [this] (int junctionIdx, juce::Point<int> screenPos)
    {
        if (junctionIdx < 0 || junctionIdx >= (int) userBlocksQueue_.size() - 1) return;
        junctionVariations_.resize (std::max ((int) junctionVariations_.size(),
                                              (int) userBlocksQueue_.size() - 1), 0);
        const int currentVariation = junctionVariations_[(std::size_t) junctionIdx];

        juce::PopupMenu m;
        m.addItem (1, "Try different splice");
        m.addItem (2, "Reset to best", currentVariation > 0);
        m.addSeparator();
        m.addItem (3, "Audition junction");

        juce::PopupMenu::Options opts;
        opts = opts.withTargetScreenArea ({ screenPos.x, screenPos.y, 1, 1 });
        m.showMenuAsync (opts, [this, junctionIdx] (int chosen)
        {
            handleBlocksJunctionAction (junctionIdx, chosen);
        });
    };

    // ADR-051 phase G — queue tile audition (left-click): play block in
    // isolation from the source WAV using PreviewController.
    // Sesja 100b — fractionInTile (0..1) maps the click X within the tile
    // rect to a source-time offset inside the block, so the user can
    // seek-on-click anywhere along the tile per user smoke verbatim
    // *"kiedy klikne w dane miejsce danego bloku to playhead odtwarza z
    // tego miejsca"*. Play continues to b.endSec — the click sets the
    // entry point, not the playback length.
    blockAssemblyPanel_.onQueueTileAudition =
        [this] (int queuePos, double fractionInTile)
    {
        if (queuePos < 0 || queuePos >= (int) userBlocksQueue_.size()) return;
        const int blockIdx = userBlocksQueue_[(std::size_t) queuePos];
        if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
        if (currentSourcePath_.isEmpty()) return;

        const auto& b = userBlocks_[(std::size_t) blockIdx];
        const double frac = juce::jlimit (0.0, 0.99, fractionInTile);
        const double startSec = b.startSec + (b.endSec - b.startSec) * frac;
        previewController_.play (currentSourcePath_, startSec, b.endSec);
        statusBar_.setText (juce::String::fromUTF8 ("Auditioning block \xE2\x80\x94 ")
            + juce::String (blockIdx + 1) + " of "
            + juce::String ((int) userBlocks_.size()));
    };

    // ADR-051 phase G — seam audition (left-click on pill): play ~4 beats
    // around the algorithm-resolved splice from source WAV. Lua pattern from
    // remix_blocks.lua:200-242 (CONTEXT = 2 beats either side).
    blockAssemblyPanel_.onSeamAudition = [this] (int junctionIdx)
    {
        if (junctionIdx < 0 || junctionIdx >= (int) lastBlockSplices_.size()) return;
        if (currentSourcePath_.isEmpty()) return;

        const auto& sp = lastBlockSplices_[(std::size_t) junctionIdx];

        // ±2 seconds default if we don't have beat-precise context. When
        // bundle is loaded we could refine to ±2 beats; for phase G the
        // ±2 sec window is robust (Lua CONTEXT = 2 beats ≈ 1 sec @ 120 BPM).
        const double startSec = std::max (0.0, sp.sourceTimeSec - 2.0);
        const double endSec   = sp.sourceTimeSec + 2.0;
        previewController_.play (currentSourcePath_, startSec, endSec);
        statusBar_.setText (juce::String::fromUTF8 ("Auditioning junction ")
            + juce::String (junctionIdx + 1));
    };

    // ADR-051 sesja-61 hot-fix — paleta wraps to multiple rows when narrow;
    // panel preferred height changes → MainComponent layout recomputes.
    blockAssemblyPanel_.onPreferredHeightChanged = [this]
    {
        if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks) resized();
    };

    // ADR-051 phase J — Re-analyze CTA kicks the Block Assembly pipeline.
    blockAssemblyPanel_.onAssembleClicked = [this]
    {
        if (userBlocksQueue_.size() < 2)
        {
            statusBar_.setText (juce::String::fromUTF8 (
                "Add at least 2 blocks to the queue before remixing"));
            return;
        }
        if (analysisState_ != AnalysisState::Complete)
        {
            statusBar_.setText (juce::String::fromUTF8 (
                "Analyze the source first, then arrange blocks"));
            return;
        }
        blockAssemblyPanel_.setDirty (false);

        // Sesja 65 — re-engagement: user clicking Assemble cancels any
        // prior "Edit arrangement" dismissal so the new assembled view
        // can be restored on future attach / tab switch.
        if (currentSourcePath_.isNotEmpty())
            blocksDismissedSources_.erase (currentItemKey());

        // Sesja 64 — Assemble button busy insta-on (immediate click feedback).
        // StatusBar spinner armed by kickRemixPipeline → 300 ms threshold.
        blockAssemblyPanel_.setAssembleBusy (true,
            juce::String::fromUTF8 ("Computing\xe2\x80\xa6"));

        PendingRemixOp op;
        op.targetSec = durationPanel_.getTarget();
        // Region fields stay nullopt — Blocks pipeline branch picks
        // userBlocks/queue/variations from MainComponent state.
        kickRemixPipeline (op);
    };

    settingsPopover_.onSnapRegionChanged =
        [this] (reamix::ui::SettingsPopover::SnapRegion r)
    {
        using SnapMode   = reamix::ui::WaveformView::SnapMode;
        using SnapRegion = reamix::ui::SettingsPopover::SnapRegion;
        const SnapMode m =
            r == SnapRegion::Beats     ? SnapMode::Beats
          : r == SnapRegion::Downbeats ? SnapMode::Downbeats
                                       : SnapMode::Off;
        waveformView_.setSnapMode (m);

        // Sesja 65 — persist across REAPER restart.
    #if REAMIX_WITH_REAPER_IO
        const char* key = (r == SnapRegion::Beats)     ? "beats"
                          : (r == SnapRegion::Downbeats) ? "downbeats"
                                                         : "off";
        if (SetExtState) SetExtState ("reamix.me", "snap_region", key, true);
    #endif
    };

    // Sesja 108 — initial window size. Layout sum of fixed-height rows
    // (header 40 + source 68 + tabs 36 + duration 78 + audition 226 +
    // transport 54 + status 22 = 524) leaves no room for WaveformView at
    // 520. Set to 760 height so the waveform gets ~236 px (reasonable for
    // peaks + minimap + splice markers); width 720 keeps ruler tick labels
    // legible. User can still resize freely after launch.
    setSize (720, 760);

    startTimer (kSelectedItemPollMs);

    // DEV-021 ONNX pre-warm.
    preWarmThread_ = std::make_unique<BeatDetectorPreWarmThread> ([this]
    {
        if (preWarmAborted_.load()) return;

        // Sesja 111 — visible download progress while the pre-warm thread
        // pulls the ~80 MB beat-this ONNX. Without this the plugin window
        // appears frozen on first launch (no item selected yet, SourcePanel
        // shows empty state). StatusBar.setBusy already supplies an accent
        // spinner + 60 Hz loading-line animation — premium feel for free.
        auto onDownloadProgress = [this] (juce::int64 downloaded, juce::int64 total)
        {
            if (preWarmAborted_.load()) return;
            const auto mbDone  = static_cast<int> ((downloaded + 500'000) / 1'000'000);
            const auto mbTotal = total > 0 ? static_cast<int> ((total + 500'000) / 1'000'000) : 0;
            // When the final byte arrives, switch the label to "Loading model" so
            // the user sees forward motion through the ~1-3 s SHA-256 verify +
            // ONNX session load that follows. Without this, the bar lingers at
            // "Downloading · 83 / 83 MB" with a spinning indicator, which reads
            // as "stuck" rather than "validating".
            juce::String label;
            if (total > 0 && downloaded >= total)
                label = juce::String::fromUTF8 ("Loading AI model\xe2\x80\xa6");
            else if (mbTotal > 0)
                label = juce::String::fromUTF8 ("Downloading AI model \xc2\xb7 ")
                      + juce::String (mbDone) + " / " + juce::String (mbTotal) + " MB";
            else
                label = juce::String::fromUTF8 ("Downloading AI model \xc2\xb7 ")
                      + juce::String (mbDone) + " MB";
            juce::MessageManager::callAsync ([this, label]
            {
                if (preWarmAborted_.load()) return;
                statusBar_.setBusy (label);
            });
        };

        juce::String err;
        const bool ok = ensureBeatDetectorReady (err, onDownloadProgress);

        juce::MessageManager::callAsync ([this, ok, err]
        {
            if (preWarmAborted_.load()) return;
            statusBar_.clearBusy();
            if (! ok && err.isNotEmpty())
            {
                statusBar_.setError (err);
                showStartupErrorWithPlatformHint (err);
            }
        });
    });
    preWarmThread_->startThread();

    // Sesja 111 (KROK 3) — first-launch welcome modal. Deferred via
    // MessageManager::callAsync so the constructor finishes (and the plugin
    // window is actually visible) before the modal grabs focus.
    juce::MessageManager::callAsync ([this] { showWelcomeIfFirstLaunch(); });

    // Sesja 111 v1.0.2 — update check on session start. Detached background
    // thread queries GitHub Releases API; if a newer tag is found, surfaces
    // an AlertWindow with "Open Releases" / "Later" buttons. Safe to abort
    // mid-flight via updateCheckAborted_ flag (set in destructor).
    checkForUpdatesAsync();

    // DEV-080 sesja 108 — pull user-set defaults from ExtState into host-side
    // mirrors BEFORE any compute path can read currentQualityWeights().
    // Synchronous on the message thread; cheap (one GetExtState + one JSON
    // parse on a short single-line record). Pre-fix the read lived inside
    // setupAdvancedPanel() which only ran on first Advanced open, so the
    // compute path silently used compile-time defaults until the user opened
    // the panel — and then still used compile-time defaults until they moved
    // a slider, because currentQualityWeights() fallback was unchanged.
    loadAdvancedDefaultsFromExtState();

    // ADR-097 sesja 107 — restore Advanced window if it was open last
    // REAPER session. Deferred to async so the main window peer is created
    // first; juce::DocumentWindow::setVisible(true) needs a live message loop.
    juce::MessageManager::callAsync ([this]
    {
        restoreAdvancedWindowOnLaunch();
    });
}

MainComponent::~MainComponent()
{
    stopTimer();

    // ADR-097 sesja 107 — persist Advanced window state + destroy window/panel.
    // Done before any other teardown so window's onCloseRequested doesn't fire
    // mid-destruction. Drop LAF references on window + panel before our LAF
    // member dies (each setLookAndFeel(nullptr) must precede the LAF owner's
    // destruction to avoid dangling pointer in drawTooltip et al.).
    if (advancedWindow_)
    {
        persistAdvancedWindowState();
        advancedWindow_->setLookAndFeel (nullptr);
        advancedWindow_->setVisible (false);
        advancedWindow_.reset();
    }
    if (advancedPanel_)
        advancedPanel_->setLookAndFeel (nullptr);
    advancedPanel_.reset();

    if (sliderDebounceTimer_) sliderDebounceTimer_->stopTimer();

    preWarmAborted_.store (true);
    if (preWarmThread_ != nullptr)
    {
        preWarmThread_->stopThread (30000);
        preWarmThread_.reset();
    }

    // Sesja 111 v1.0.2 — flag update-check thread to skip any pending
    // network read / message-thread callback. Thread is detached so we
    // don't join here; the flag check inside each step prevents UAF.
    updateCheckAborted_.store (true);

    // Pipeline workers — wait for any in-flight thread before our members
    // (beatDetector_, analysisBundles_, etc.) get destroyed. Alive flags
    // were set by each worker's dtor via `aliveFlag()->store(false)`, so
    // any queued message-thread callbacks are already inert.
    if (analyzePipeline_ != nullptr)
    {
        analyzePipeline_->stopThread (30000);
        analyzePipeline_.reset();
    }
    for (auto& w : stoppingAnalyze_)
        if (w != nullptr) w->stopThread (30000);
    stoppingAnalyze_.clear();

    if (remixPipeline_ != nullptr)
    {
        remixPipeline_->stopThread (30000);
        remixPipeline_.reset();
    }
    for (auto& w : stoppingRemix_)
        if (w != nullptr) w->stopThread (30000);
    stoppingRemix_.clear();

    // ADR-047 § 1 — clean every cached tmp WAV file. Eviction-on-destroy.
    for (const auto& path : remixCache_.evictAll())
    {
        if (path.isNotEmpty()) juce::File (path).deleteFile();
    }
    if (tmpWavPath_.isNotEmpty())
    {
        juce::File f (tmpWavPath_);
        if (f.existsAsFile()) f.deleteFile();
    }

    removeKeyListener (&transportBar_);
    setLookAndFeel (nullptr);
}

void MainComponent::reapStoppedWorkers()
{
    stoppingAnalyze_.erase (
        std::remove_if (
            stoppingAnalyze_.begin(), stoppingAnalyze_.end(),
            [] (const std::unique_ptr<reamix::ui::AnalyzePipeline>& w)
            {
                return w == nullptr || ! w->isThreadRunning();
            }),
        stoppingAnalyze_.end());

    stoppingRemix_.erase (
        std::remove_if (
            stoppingRemix_.begin(), stoppingRemix_.end(),
            [] (const std::unique_ptr<reamix::ui::RemixPipeline>& w)
            {
                return w == nullptr || ! w->isThreadRunning();
            }),
        stoppingRemix_.end());
}

bool MainComponent::ensureBeatDetectorReady (juce::String& outErrorMessage,
                                              std::function<void(juce::int64, juce::int64)> downloadProgressCb)
{
    std::lock_guard<std::mutex> lock (beatDetectorLoadMutex_);

    if (beatDetectorLoaded_) return true;
    if (beatDetectorLoadError_.isNotEmpty())
    {
        outErrorMessage = beatDetectorLoadError_;
        return false;
    }

    std::string downloadError;
    if (! reamix::ModelManager::ensureDownloaded (std::move (downloadProgressCb), &downloadError))
    {
        beatDetectorLoadError_ = juce::String ("Model download failed: ")
                               + juce::String (downloadError);
        outErrorMessage = beatDetectorLoadError_;
        return false;
    }

    const auto modelPath = reamix::ModelManager::modelPath().getFullPathName().toStdString();
    if (! beatDetector_.loadModel (modelPath))
    {
        beatDetectorLoadError_ = "Unable to load beat-detection model";
        outErrorMessage = beatDetectorLoadError_;
        return false;
    }

    beatDetectorLoaded_ = true;
    return true;
}

// ── AnalyzePipeline (stages 1-5) ─────────────────────────────────────

void MainComponent::startAnalyze()
{
    if (currentSourcePath_.isEmpty())
    {
        statusBar_.setError ("Select an item first");
        return;
    }

    if (analyzePipeline_ != nullptr && analyzePipeline_->isThreadRunning())
    {
        // Lua parity: start_analysis (remix_operations.lua:246) ignores
        // duplicate calls while another analysis runs.
        statusBar_.setText ("Analysis already running on "
                            + juce::File (activeAnalyzePath_).getFileName());
        return;
    }

    sourcePanel_.setAnalyzing (true, 0.0);
    resized();
    statusBar_.clearError();
    statusBar_.setText ("Starting analysis...");

    previewController_.stop();
    waveformView_.setPlayhead (std::nullopt);
    selectedRange_.reset();
    lastSeekSec_.reset();
    waveformView_.clearSelection();

    // Sesja 111 — if click-Analyze races the pre-warm download, this callback
    // surfaces progress in SourcePanel (already showing the analyzing layout)
    // as the "Downloading model · X / Y MB" stage label. Once ensureDownloaded
    // returns, the analyze pipeline progress callback below takes over.
    const juce::String downloadingForPath = currentSourcePath_;
    auto onDownloadProgress = [this, downloadingForPath] (juce::int64 downloaded, juce::int64 total)
    {
        const auto mbDone  = static_cast<int> ((downloaded + 500'000) / 1'000'000);
        const auto mbTotal = total > 0 ? static_cast<int> ((total + 500'000) / 1'000'000) : 0;
        const double frac  = total > 0 ? static_cast<double> (downloaded) / static_cast<double> (total) : 0.0;
        juce::String stage;
        if (total > 0 && downloaded >= total)
            stage = juce::String::fromUTF8 ("Loading model\xe2\x80\xa6");
        else if (mbTotal > 0)
            stage = juce::String::fromUTF8 ("Downloading model \xc2\xb7 ")
                  + juce::String (mbDone) + " / " + juce::String (mbTotal) + " MB";
        else
            stage = juce::String::fromUTF8 ("Downloading model \xc2\xb7 ")
                  + juce::String (mbDone) + " MB";
        juce::MessageManager::callAsync ([this, downloadingForPath, frac, stage]
        {
            if (currentSourcePath_ != downloadingForPath) return;
            sourcePanel_.setAnalyzing (true, frac, stage);
            statusBar_.setText (stage);
        });
    };

    juce::String err;
    if (! ensureBeatDetectorReady (err, onDownloadProgress))
    {
        sourcePanel_.setAnalyzing (false, 0.0);
        resized();
        statusBar_.setError (err);
        return;
    }

    // ADR-047 § 3 rule 1 — auto-chain a Remix at current target on Analyze
    // completion. Lua start_analysis pattern: analyze → cache → remix runs
    // automatically afterwards (followup_after_analysis="remix").
    PendingFollowup f;
    f.kind = PendingFollowup::Kind::Remix;
    f.op.targetSec = durationPanel_.getTarget();
    followupAfterAnalysis_ = f;

    analysisState_ = AnalysisState::Analyzing;
    headerBar_.setStatusKind (reamix::ui::HeaderStatus::Analyzing);

    reamix::ui::AnalyzePipeline::Input in;
    in.sourcePath = currentSourcePath_;

    const juce::String workerPath = currentSourcePath_;
    activeAnalyzePath_ = workerPath;

    auto progressCb = [this, workerPath] (juce::String step, double p01)
    {
        lastAnalyzeProgress_[workerPath] = p01;
        lastAnalyzeStage_[workerPath]    = step;
        if (currentSourcePath_ != workerPath) return;
        // Sesja 64 — pass stage label to SourcePanel so progress row shows
        // "Detecting beats" etc. instead of just %; addresses NOTE-1
        // (progress feels static / not real-time).
        sourcePanel_.setAnalyzing (true, p01, step);
        statusBar_.setText (step + juce::String::fromUTF8 (" \xe2\x80\xa6 ")
                            + juce::String ((int) std::round (p01 * 100.0)) + "%");
    };
    auto completeCb = [this, workerPath] (reamix::ui::AnalysisBundlePtr bundle, juce::String error)
    {
        handleAnalysisComplete (workerPath, std::move (bundle), std::move (error));
    };

    analyzePipeline_ = std::make_unique<reamix::ui::AnalyzePipeline> (
        std::move (in), beatDetector_, std::move (progressCb), std::move (completeCb));
    analyzePipeline_->startThread();
}

void MainComponent::handleAnalysisComplete (juce::String workerPath,
                                            reamix::ui::AnalysisBundlePtr bundle,
                                            juce::String error)
{
    lastAnalyzeProgress_.erase (workerPath);
    lastAnalyzeStage_.erase (workerPath);
    if (activeAnalyzePath_ == workerPath) activeAnalyzePath_.clear();
    if (analyzePipeline_ != nullptr)
        stoppingAnalyze_.push_back (std::move (analyzePipeline_));

    const bool showing = (currentSourcePath_ == workerPath);

    if (showing)
    {
        sourcePanel_.setAnalyzing (false, 0.0);
        resized();
    }

    if (bundle == nullptr || ! error.isEmpty())
    {
        if (showing)
        {
            statusBar_.setError (error.isEmpty() ? "Analysis failed" : error);
            transportBar_.setState (reamix::ui::TransportState::Idle);
            headerBar_.setStatusKind (reamix::ui::HeaderStatus::Error);
        }
        else
        {
            headerBar_.setStatusKind (reamix::ui::HeaderStatus::Ready);
        }
        analysisState_ = AnalysisState::Idle;
        followupAfterAnalysis_.reset();
        return;
    }

    // Cache bundle by worker path (resilient to mid-analysis item switch).
    analysisBundles_[workerPath] = bundle;

    // ADR-053 (sesja 63 BUG-19 follow-up) — persist bundle to disk so the
    // next plugin session restores it without re-running DSP. Synchronous
    // write on message thread; bundle ~5-20 MB → 50-200 ms typical. Async
    // promotion deferred unless this surfaces as a UX issue. save() is
    // best-effort: failure (disk full / permission) leaves user with the
    // in-memory bundle as before — next session falls back to fresh
    // Analyze, no functional regression.
    if (bundle != nullptr)
        reamix::ui::AnalysisDiskCache::save (*bundle);

    // Sesja 60 — flip Analyze button label to "Analyze Again" when the bundle
    // belongs to the currently-shown source. (When workerPath != currentSourcePath_
    // we don't touch label state; SourcePanel is showing a different item.)
    if (workerPath == currentSourcePath_)
        sourcePanel_.setHasAnalysis (true);

    if (! showing)
    {
        // Quietly cached. analysisState_ stays whatever it was (probably
        // Idle for currently-shown item). When user returns to workerPath,
        // applySelectedItem detects bundle hit and flips state appropriately.
        return;
    }

    analysisState_ = AnalysisState::Complete;
    headerBar_.setStatusKind (reamix::ui::HeaderStatus::Ready);
    statusBar_.clearError();

    juce::String itemName;
    if (auto picked = reamix::reaper::getSelectedItem())
        itemName = picked->name;

    applyAnalysisToUi (*bundle, itemName, /*fromCache=*/false);

    // Drain followup queue per Lua start_analysis (remix_operations.lua:286
    // followup_after_analysis dispatch).
    if (followupAfterAnalysis_.has_value())
    {
        const auto f = *followupAfterAnalysis_;
        followupAfterAnalysis_.reset();

        if (f.kind == PendingFollowup::Kind::Remix
            || f.kind == PendingFollowup::Kind::Insert)
        {
            // Both remix and insert require RemixPipeline to run first.
            // Insert followup is preserved for handleRemixComplete to dispatch.
            if (f.kind == PendingFollowup::Kind::Insert)
                followupAfterAnalysis_ = f;
            kickRemixPipeline (f.op);
        }
    }
}

// ── RemixPipeline (stages 6-7-8) ─────────────────────────────────────

void MainComponent::kickRemixPipeline (const PendingRemixOp& op)
{
    if (currentSourcePath_.isEmpty()) return;
    auto bundleIt = analysisBundles_.find (currentSourcePath_);
    if (bundleIt == analysisBundles_.end() || bundleIt->second == nullptr) return;

    // ADR-047 § 1 — RemixCache hit → use cached output instantly, no
    // pipeline run. Quality-first per memory feedback_quality_first_priority.md.
    auto cacheKey = reamix::ui::makeRemixCacheKey (
        currentSourcePath_,
        currentItemGuid_,                       // ADR-056 (sesja 66)
        op.targetSec,
        op.regionStartSec.value_or (0.0),
        op.regionEndSec.value_or (0.0),
        op.blockedTransitions,
        op.variation);

    // ADR-080 RESCOPE + ADR-083 (sesja 92) — AuditionBar 4-slider identity
    // hash. Default sliders → hash 0 → cache lookup compatible with
    // pre-sesja-92 entries. User drag → non-zero → forces fresh remix render
    // (the actual reason audition slider drags previously did nothing — they
    // hit cached old WAV under identical pre-sesja-92 cache key).
    {
        const auto ap = currentAuditionParams();
        cacheKey.auditionHash = reamix::ui::hashAuditionParams (
            ap.tone, ap.editLength, ap.allowPmSeconds, ap.minCutBeats);
    }

    // ADR-087 STATUS UPDATE 1 (sesja 98) + ADR-097 sesja 107 — QualityWeights
    // identity hash. Returns 0 when weights == kDefaultQualityWeights so
    // pre-sesja-98 cache entries (and untouched sliders) compare equal in
    // cache lookups → preserves existing remix outputs.
    cacheKey.qualityWeightsHash = reamix::ui::hashQualityWeights (currentQualityWeights());

    // ADR-051 — Block Assembly mode contributes a separate identity hash
    // (queue + boundaries + kinds + variations + flexibility) so cache lookup
    // distinguishes between arrangements while staying compatible with the
    // existing keyspace.
    if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks
        && userBlocks_.size() >= 1 && userBlocksQueue_.size() >= 2)
    {
        juce::uint64 h = 1469598103934665603ull; // FNV-1a 64-bit basis
        const auto mix = [&] (juce::uint64 v)
        {
            h ^= v;
            h *= 1099511628211ull;
        };
        for (const auto& b : userBlocks_)
        {
            mix ((juce::uint64) std::lround (b.startSec * 1000.0));
            mix ((juce::uint64) std::lround (b.endSec   * 1000.0));
            mix ((juce::uint64) (int) b.kind);
        }
        for (int q : userBlocksQueue_) mix ((juce::uint64) q);
        for (int v : junctionVariations_) mix ((juce::uint64) v);
        // Splice flexibility removed sesja 68 ADR-057 — no longer part of
        // cache key (constant W=8 hardcoded in pipeline until DEV-040 ships).
        cacheKey.blocksHash = h;
    }

    const auto* hit = remixCache_.find (cacheKey);
    if (hit != nullptr)
    {
        // If the WAV file was deleted out from under us (rare; tmp dir
        // cleanup), fall through to re-render.
        if (juce::File (hit->tmpWavPath).existsAsFile())
        {
            applyRemixToUi (*hit);
            currentRemix_     = *hit;
            currentRemixMode_ = appMode_;

            // Drain Insert followup against this cache hit.
            if (followupAfterAnalysis_.has_value()
                && followupAfterAnalysis_->kind == PendingFollowup::Kind::Insert)
            {
                followupAfterAnalysis_.reset();
                if (transportBar_.onInsert) transportBar_.onInsert();
            }
            return;
        }
    }

    // Cancel any in-flight RemixPipeline — latest target wins (ADR-047 § 4).
    if (remixPipeline_ != nullptr)
    {
        remixPipeline_->signalThreadShouldExit();
        stoppingRemix_.push_back (std::move (remixPipeline_));
    }

    // Sesja 64 — deferred busy: shows spinner + accent text + loading line
    // after 300 ms threshold. Short cache-hit paths above never trigger this.
    // Per-mode label so user sees what kind of work is happening.
    juce::String busyLabel;
    if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks
        && userBlocks_.size() >= 1 && userBlocksQueue_.size() >= 2)
        busyLabel = juce::String::fromUTF8 ("Computing arrangement\xe2\x80\xa6");
    else if (op.regionStartSec.has_value() && op.regionEndSec.has_value())
        busyLabel = juce::String::fromUTF8 ("Computing region remix\xe2\x80\xa6");
    else
        busyLabel = juce::String::fromUTF8 ("Computing remix\xe2\x80\xa6");
    busyDeferRemix_.startBusy (busyLabel);
    // Sesja 108 — pulse HeaderBar dot for the whole remix compute, mirroring
    // analyze. User feedback: pulsowanie spojnie podczas kazdej heavy operacji.
    headerBar_.setStatusKind (reamix::ui::HeaderStatus::Analyzing);

    reamix::ui::RemixPipeline::Input in;
    in.bundle              = bundleIt->second;
    in.itemGuid            = currentItemGuid_;  // ADR-056 (sesja 66)
    in.targetDurationSec   = op.targetSec;
    in.regionStartSec      = op.regionStartSec;
    in.regionEndSec        = op.regionEndSec;
    in.blockedTransitions  = op.blockedTransitions;
    in.variation           = op.variation;

    // ADR-080 RESCOPE + ADR-083 (sesja 92) — AuditionBar 4-slider params.
    // Populated from auditionBar_ (always reflects current visible state).
    // Defaults bit-exact baseline when slider untouched.
    {
        const auto ap = currentAuditionParams();
        // Tone slider — propagated via `qualityWeightsOverride` so the blend
        // logic in computeQualityScore sees the user value. When tone == 0.0
        // the override sets harmonic_vs_timbre = 0.0 → blend bypassed →
        // bit-exact baseline (matches kDefaultQualityWeights).
        //
        // Sesja 98 ADR-087 + sesja 107 ADR-097: AdvancedWeightsPanel may tune
        // the 7-component simplex via per-(item, mode) weights map; merge Tone
        // slider on top so both can coexist (Tone is a separate blend
        // coefficient, NOT part of the simplex sum=1.0 invariant).
        {
            reamix::remix::QualityWeights w = currentQualityWeights();
            if (ap.tone > 0.0)
                w.harmonic_vs_timbre = ap.tone;
            if (! reamix::ui::qualityWeightsAtDefault (w))
                in.qualityWeightsOverride = w;
        }
        // Edit Length / Allow ± / Min cut — fields on RemixPipeline::Input
        // forwarded directly to TC/RC/BA Inputs in RemixPipeline::run().
        in.harmonic_vs_timbre     = ap.tone;
        // ADR-084 sesja 93 — Edit Length MULTIPLICATIVE jump-cost scale
        // (supersedes sesja-92 ADR-083 additive penalty). Map slider
        // [0..100] → 2^((slider-50)/25) ∈ [0.25, 4.0]. Slider=50 → 1.0
        // bit-exact baseline. Slider value also passed separately for
        // cache hash (avoids log2 reverse-conversion).
        in.edit_length_slider     = ap.editLength;
        in.edit_length_jump_scale = std::pow(2.0,
                                             ((double) ap.editLength - 50.0) / 25.0);
        in.allow_pm_seconds       = (double) ap.allowPmSeconds;
        // ADR-084 sesja 93 — Min cut: always pass user value (no sentinel
        // remap at default 16). Optimizer uses min_seq_after_jump_user_override
        // flag to bypass adaptive cooldown. Removes sesja-92 discontinuity
        // at slider=16 where adaptive scaling re-engaged silently.
        in.min_cut_beats          = ap.minCutBeats;
    }

    // ADR-051 — Block Assembly mode bypasses Region/Auto branches when
    // userBlocks are populated AND queue ≥ 2. Splice flexibility maps to
    // search_window_beats; drift penalty kept at default ADR-051 weight.
    if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks
        && userBlocks_.size() >= 1 && userBlocksQueue_.size() >= 2)
    {
        in.userBlocks       = userBlocks_;
        in.userBlocksQueue  = userBlocksQueue_;
        for (std::size_t j = 0; j < junctionVariations_.size(); ++j)
        {
            if (junctionVariations_[j] > 0)
                in.junctionVariations[(int) j] = junctionVariations_[j];
        }
        // ADR-057 (sesja 68) — Splice flexibility removed; RemixPipeline now
        // reads its own hardcoded interim default (W=8, Medium-equivalent)
        // until DEV-040 β-model ships.
        in.driftPenaltyWeight  = reamix::remix::BLOCK_DRIFT_PENALTY_WEIGHT;
        // Region fields irrelevant in Blocks mode; clear so RemixPipeline's
        // mode-resolution picks Blocks branch.
        in.regionStartSec.reset();
        in.regionEndSec.reset();
    }

    const juce::String wavPath = reamix::ui::tmpWavPathFor (cacheKey);

    auto progressCb = [this] (juce::String step, double p01)
    {
        // Sesja 64 — feed progress into busy state if active so spinner
        // label updates with stage + %. Falls back to setText if compute
        // finished within threshold (busy was never armed visually).
        const juce::String label = step
            + juce::String::fromUTF8 (" \xe2\x80\xa6 ")
            + juce::String ((int) std::round (p01 * 100.0)) + "%";
        if (statusBar_.isBusy()) statusBar_.setBusy (label);
        else                     statusBar_.setText (label);
    };
    auto completeCb = [this] (reamix::ui::RemixOutput out)
    {
        handleRemixComplete (std::move (out));
    };

    remixPipeline_ = std::make_unique<reamix::ui::RemixPipeline> (
        std::move (in), std::move (progressCb), std::move (completeCb), wavPath);
    remixPipeline_->startThread();
}

void MainComponent::handleRemixComplete (reamix::ui::RemixOutput out)
{
    // Sesja 64 — clear deferred busy state (spinner + loading line) at entry,
    // before stale-source guard or error branch can short-circuit. setError
    // wins over busy in StatusBar paint so the error path below still renders
    // correctly even if stopBusy ran first. Also clears Assemble button busy
    // (no-op if not active).
    busyDeferRemix_.stopBusy();
    blockAssemblyPanel_.setAssembleBusy (false);
    // Sesja 108 — return HeaderBar dot to Ready (pulse off). Error paths
    // below override to Error when out.ok == false.
    headerBar_.setStatusKind (reamix::ui::HeaderStatus::Ready);

    if (remixPipeline_ != nullptr)
        stoppingRemix_.push_back (std::move (remixPipeline_));

    if (! out.ok)
    {
        statusBar_.setError (out.errorMessage.isEmpty() ? "Remix failed" : out.errorMessage);
        followupAfterAnalysis_.reset();
        return;
    }

    // Stale-source guard: user may have switched items mid-render. If so,
    // we still cache the output (it's a valid (path, target) tuple) but
    // don't update UI.
    const bool showing = (out.sourcePath == currentSourcePath_);

    // Insert into RemixCache; evict oldest entry from this source if at
    // bound. Returned path is the evicted WAV, delete it now. Cache key
    // MUST match the one kickRemixPipeline used for the look-up (full
    // blocked set, not empty) — session 57 introduces non-empty blocked
    // sets via SpliceMarker context-menu Block action.
    auto cacheKey = reamix::ui::makeRemixCacheKey (
        out.sourcePath, out.itemGuid,           // ADR-056 (sesja 66)
        out.targetSec, out.regionStartSec, out.regionEndSec,
        out.blockedTransitions, out.variation);
    cacheKey.auditionHash = out.auditionHash;  // ADR-083 sesja 92

    // DEV-079 sesja 101 + ADR-097 sesja 107 — close storage/lookup desync.
    // Lookup paths set qualityWeightsHash from currentQualityWeights(); the
    // storage path mirrors the same assignment so storage key matches lookup
    // key when user has moved any advanced weight slider.
    cacheKey.qualityWeightsHash = reamix::ui::hashQualityWeights (currentQualityWeights());

    // Sesja 65 — Blocks-mode storage MUST also include blocksHash so the
    // key matches kickRemixPipeline's lookup key. Without this, every
    // Assemble result was stored under blocksHash=0 while lookups used
    // blocksHash=H → cache always missed for Blocks. Pre-existing bug
    // surfaced by sesja-65 cross-mode tab restore (where Blocks tab
    // re-entry after a Duration peek silently lost the assembled remix).
    if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks)
    {
        const juce::uint64 h = computeBlocksHash();
        if (h != 0) cacheKey.blocksHash = h;
    }

    const juce::String evicted = remixCache_.insert (cacheKey, out);
    if (evicted.isNotEmpty()) juce::File (evicted).deleteFile();

    if (! showing) return;

    // Sesja 64 BUG-3 — Try different splice "exhausted" detection. Compare
    // clips against pre-kick snapshot; if identical, variation cycle has
    // wrapped back to the same path → revert variation counter + emit
    // explicit feedback so user knows it's not a broken feature.
    if (tryDifferentSnapshot_.has_value())
    {
        const auto& prev = *tryDifferentSnapshot_;
        const auto& curr = out.editPlan.clips;
        bool same = (prev.size() == curr.size());
        for (std::size_t i = 0; same && i < prev.size(); ++i)
        {
            if (prev[i].startBeat != curr[i].startBeat ||
                prev[i].endBeat   != curr[i].endBeat   ||
                std::abs (prev[i].sourceStartSec - curr[i].sourceStartSec) > 0.001 ||
                std::abs (prev[i].sourceEndSec   - curr[i].sourceEndSec)   > 0.001)
                same = false;
        }
        tryDifferentSnapshot_.reset();
        if (same)
        {
            // Revert variation increment so user can still ResetToBest cleanly.
            auto vIt = variationBySource_.find (out.sourcePath);
            if (vIt != variationBySource_.end() && vIt->second > 0)
                vIt->second -= 1;
            statusBar_.setNotice (juce::String::fromUTF8 (
                "No more alternative splices \xe2\x80\x94 already at best variant"));
            // Don't apply the (identical) remix to UI — leaves the existing
            // currentRemix_ valid + spares an unnecessary visual repaint.
            return;
        }
    }

    currentRemix_     = out;
    currentRemixMode_ = appMode_;
    applyRemixToUi (*currentRemix_);

    // Drain Insert followup if the user clicked Insert during RemixPipeline.
    if (followupAfterAnalysis_.has_value()
        && followupAfterAnalysis_->kind == PendingFollowup::Kind::Insert)
    {
        followupAfterAnalysis_.reset();
        if (transportBar_.onInsert) transportBar_.onInsert();
    }
}

// ── Slider debounce fire (ADR-047 § 4) ───────────────────────────────

void MainComponent::sliderDebounceFired()
{
    if (analysisState_ == AnalysisState::Idle) return; // no bundle to remix

    if (analysisState_ == AnalysisState::Analyzing)
    {
        // Overwrite followup with latest target — newest slider value wins.
        PendingFollowup f;
        f.kind = PendingFollowup::Kind::Remix;
        f.op   = pendingSliderOp_;
        followupAfterAnalysis_ = f;
        return;
    }

    // Complete → kick fresh remix.
    kickRemixPipeline (pendingSliderOp_);
}

// ── UI rendering helpers ─────────────────────────────────────────────

void MainComponent::applyAnalysisToUi (const reamix::ui::AnalysisBundle& bundle,
                                       const juce::String& itemName,
                                       bool fromCache)
{
    sourcePanel_.setAnalyzing (false, 0.0);
    resized();

    reamix::ui::SourceInfo si;
    si.name     = itemName;
    si.duration = currentSourceDurationSec_;
    si.bpm      = bundle.bpm;
    si.beats    = (int) bundle.beatTimes.size();
    si.empty    = false;
    sourcePanel_.setSource (si);

    std::vector<reamix::ui::WaveformView::Segment> wvSegs;
    wvSegs.reserve (bundle.uiSegments.size());
    for (const auto& s : bundle.uiSegments)
        wvSegs.push_back ({ s.startSec, s.endSec, s.kind });
    waveformView_.setSegments (std::move (wvSegs));

    std::vector<reamix::ui::WaveformView::Beat> wvBeats;
    wvBeats.reserve (bundle.beatTimes.size());
    for (size_t i = 0; i < bundle.beatTimes.size(); ++i)
        wvBeats.push_back ({ bundle.beatTimes[i],
                              i < bundle.beatIsDownbeat.size() ? bundle.beatIsDownbeat[i] : false });
    waveformView_.setBeats (std::move (wvBeats));

    waveformView_.setAnalyzed (true);

    // ADR-051 § Consequence #5 — post-Analyze with features unlocks Blocks
    // tab. (Was previously gated on segments which are empty post-ADR-044
    // and never re-populate on auto path; ADR-051 relaxes the condition.)
    modeTabs_.setBlocksEnabled (true);

    transportBar_.setState (reamix::ui::TransportState::Idle);

    const juce::String prefix = fromCache ? "Analysis cached: " : "Analysis complete: ";
    const juce::String msg = prefix + juce::String ((int) bundle.beatTimes.size()) + " beats, "
                           + juce::String (bundle.bpm, 1) + " BPM";
    statusBar_.setText (msg);
}

void MainComponent::applyRemixToUi (const reamix::ui::RemixOutput& remix)
{
    // Sesja 65 BUG-21 — stop in-flight preview when the underlying WAV is
    // about to change. Without this, slider-driven Region (or Duration /
    // Blocks) recompute leaves the previous tmp WAV playing in PreviewController
    // even after tmpWavPath_ swaps to the new remix output → user clicks
    // waveform expecting fresh audio but hears the stale preview. Pre-existing
    // bug; surfaced after sesja 64 made Region mode usable end-to-end.
    // Conditional on actual path change so cache-hit re-applications (Test 52
    // tab peek-and-restore returning the same cached remix) don't interrupt
    // a perfectly valid in-flight preview.
    if (! tmpWavPath_.isEmpty() && tmpWavPath_ != remix.tmpWavPath)
    {
        previewController_.stop();
        waveformView_.setPlayhead (std::nullopt);
    }

    tmpWavPath_ = remix.tmpWavPath;
    transportBar_.setState (reamix::ui::TransportState::Ready);

    // Session 57 — Remix variant + SpliceMarker overlay + ADR-045 (c) tile-list.
    // Hide any tooltip from a previous render before swapping data.
    tooltip_.hide();

    // Swap peaks provider to the remix tmp WAV; flip variant; provide
    // splice markers + edit plan to drive new paint stages.
    remixPeaksProvider_->setSourcePath (remix.tmpWavPath);
    waveformView_.setPeaksProvider (remixPeaksProvider_.get());
    waveformView_.setVariant (reamix::ui::WaveformView::Variant::Remix);

    // DEV-042 Path R upgrade (sesja 99) — remap source-time beats to
    // remix-time beats per edit plan clip mapping. Sesja 95 shipped Path Q
    // (clear beats in Remix variant) as quick fix to the source-time-on-
    // remix-canvas mismatch user saw sesja 69 as "marker nie na beat".
    // Sesja 99 user smoke pushback: "zniknęły mi oznaczenia beatów na
    // waveform mimo włączonego show beats w ustawieniach" — beats are
    // useful for time orientation; clear was too aggressive.
    //
    // Remap rule: for each source-time beat T, find clip(s) with
    // [sourceStartSec, sourceEndSec] containing T; emit one remix-time
    // beat at (clip.timelineStartSec + (T - clip.sourceStartSec)) per
    // matching clip. A beat may appear 0 times (excised), 1 time (most
    // cases), or N times (replayed section under Block Assembly).
    {
        std::vector<reamix::ui::WaveformView::Beat> remixBeats;
        auto bundleIt = analysisBundles_.find (currentSourcePath_);
        if (bundleIt != analysisBundles_.end()
            && bundleIt->second != nullptr
            && ! remix.editPlan.clips.empty())
        {
            const auto& bundle = *bundleIt->second;
            remixBeats.reserve (bundle.beatTimes.size());
            constexpr double kEps = 1e-6;
            for (std::size_t i = 0; i < bundle.beatTimes.size(); ++i)
            {
                const double srcT = bundle.beatTimes[i];
                const bool   isDb = (i < bundle.beatIsDownbeat.size())
                                    ? bundle.beatIsDownbeat[i] : false;
                for (const auto& c : remix.editPlan.clips)
                {
                    if (srcT >= c.sourceStartSec - kEps
                        && srcT <= c.sourceEndSec + kEps)
                    {
                        const double remixT = c.timelineStartSec
                                            + (srcT - c.sourceStartSec);
                        remixBeats.push_back ({ remixT, isDb });
                    }
                }
            }
        }
        waveformView_.setBeats (std::move (remixBeats));
    }

    std::vector<reamix::ui::WaveformView::SpliceMarker> markers;
    markers.reserve (remix.transitionTimesSec.size());
    for (std::size_t i = 0; i < remix.transitionTimesSec.size(); ++i)
    {
        const float q = (i < remix.transitionQualities.size())
                        ? remix.transitionQualities[i] : 0.0f;
        reamix::ui::WaveformView::SpliceMarker m;
        m.timeSec      = remix.transitionTimesSec[i];
        m.qualityScore = q;
        m.quality      = (q > 0.7f) ? reamix::ui::WaveformView::SpliceQuality::Good
                       : (q >= 0.5f) ? reamix::ui::WaveformView::SpliceQuality::Medium
                                     : reamix::ui::WaveformView::SpliceQuality::Bad;
        m.fromBeat     = (i < remix.transitionFromBeats.size()) ? remix.transitionFromBeats[i] : -1;
        m.toBeat       = (i < remix.transitionToBeats.size())   ? remix.transitionToBeats[i]   : -1;
        m.fromLabel    = (i < remix.transitionFromLabels.size()) ? remix.transitionFromLabels[i] : juce::String();
        m.toLabel      = (i < remix.transitionToLabels.size())   ? remix.transitionToLabels[i]   : juce::String();
        m.energyDiffDb = (i < remix.transitionEnergyDiffsDb.size()) ? remix.transitionEnergyDiffsDb[i] : 0.0f;
        markers.push_back (std::move (m));
    }
    waveformView_.setSpliceMarkers (std::move (markers));
    waveformView_.setEditPlan (remix.editPlan);

    // Sesja 60 Plan A — in Region mode, snapshot the selection that produced
    // this remix, clear the on-canvas scrim (it referenced original-time
    // coords which don't match the Remix variant timeline), and show the
    // "↺ Edit region" overlay button so the user can return to edit mode.
    const bool isRegionRemix =
        (remix.regionEndSec - remix.regionStartSec) > 0.001;
    if (isRegionRemix && appMode_ == reamix::ui::ModeTabs::Mode::Region)
    {
        if (selectedRange_.has_value())
            lastRegionSelection_ = *selectedRange_;
        selectedRange_.reset();
        lastSeekSec_.reset();
        waveformView_.clearSelection();
        waveformView_.setShowEditRegionButton (true);
        waveformView_.setShowEditArrangementButton (false);

        // DEV-034 — record bounds + target that produced this remix so a
        // later deselect→reselect of the same source can restore Region
        // mode AND the slider value (mirror of cacheBlocksSession() for
        // Block Assembly state). Both fields are needed: the cache-key
        // lookup in kickRemixPipeline includes targetSec, so without it
        // the auto-fired remix produces a different output.
        if (currentSourcePath_.isNotEmpty())
        {
            RegionSnapshot snap;
            snap.region.startSec = remix.regionStartSec;
            snap.region.endSec   = remix.regionEndSec;
            snap.targetSec       = remix.targetSec;
            lastRegionByPath_[currentItemKey()] = snap;
        }
    }
    else if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks)
    {
        // Sesja 61 hot-fix — Blocks-mode remix flips waveform to Remix variant;
        // overlay lets user return to Source-variant editing without losing
        // userBlocks_/queue/junctionVariations_ state.
        waveformView_.setShowEditRegionButton (false);
        waveformView_.setShowEditArrangementButton (true);
    }
    else
    {
        waveformView_.setShowEditRegionButton (false);
        waveformView_.setShowEditArrangementButton (false);
    }

    // ADR-051 — Block Assembly mode: populate algorithm-resolved splice
    // points on the source segBar (dual-layer visualization). Splice times
    // here are SOURCE-time (sourceEndSec of each clip = where the cut
    // happens on the source).
    if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks
        && ! remix.editPlan.clips.empty()
        && remix.editPlan.clips.size() >= 2)
    {
        std::vector<reamix::ui::WaveformView::UserBlockSplice> splices;
        const auto& clips = remix.editPlan.clips;
        splices.reserve (clips.size() - 1);
        const std::size_t expectedJunctions = std::min (
            (std::size_t) std::max (0, (int) userBlocksQueue_.size() - 1),
            clips.size() - 1);
        for (std::size_t i = 0; i < expectedJunctions; ++i)
        {
            reamix::ui::WaveformView::UserBlockSplice sp;
            sp.sourceTimeSec = clips[i].sourceEndSec;
            sp.qualityScore  = (i < remix.transitionQualities.size())
                ? remix.transitionQualities[i] : 0.7f;
            sp.driftBeats = 0.0; // approx-only; quality-tuning may refine
            sp.leftBlockIdx  = (int) i < (int) userBlocksQueue_.size()
                ? userBlocksQueue_[i] : -1;
            sp.rightBlockIdx = (int) (i + 1) < (int) userBlocksQueue_.size()
                ? userBlocksQueue_[i + 1] : -1;
            splices.push_back (sp);
        }
        lastBlockSplices_ = splices;
        waveformView_.setUserBlockSplices (std::move (splices));

        // Drive seam-pill colors from the same quality scores.
        std::vector<float> seamQ;
        seamQ.reserve (remix.transitionQualities.size());
        for (float q : remix.transitionQualities) seamQ.push_back (q);
        blockAssemblyPanel_.setSeamQualities (seamQ);
    }
    else
    {
        lastBlockSplices_.clear();
        waveformView_.setUserBlockSplices ({});
    }

    const juce::String msg = "Remix ready: "
                           + juce::String (remix.nTransitions) + " splices, "
                           + rxFmt (remix.remixDurationSec);
    statusBar_.setText (msg);
}

// ── 100 ms timer poll ────────────────────────────────────────────────

void MainComponent::timerCallback()
{
    reapStoppedWorkers();

    if (previewController_.isPlaying())
    {
        waveformView_.setPlayhead (previewController_.getPositionSec());
    }
    else if (transportBar_.getState() == reamix::ui::TransportState::Playing)
    {
        // Sesja 64 BUG-2 enhancement — keep playhead on natural-end position
        // so user sees where preview stopped + Space resumes from there.
        const double endPos = previewController_.getPositionSec();
        previewController_.stop();
        waveformView_.setPlayhead (endPos);
        lastSeekSec_ = endPos;
        transportBar_.setState (reamix::ui::TransportState::Ready);
        statusBar_.setNotice (juce::String::fromUTF8 (
            "Preview finished \xe2\x80\x94 Space resumes from ") + rxFmt (endPos));
    }

    const auto picked = reamix::reaper::getSelectedItem();

    if (! picked.has_value())
    {
        if (hasSelection_)
        {
            applyEmptyState();
            hasSelection_   = false;
            prevSourcePath_ = {};
            prevItemGuid_   = {};   // ADR-056 (sesja 66)
        }
        // No item → no region. Force Duration mode + clear auto-flag state.
        if (currentRegion_.has_value() || appMode_ != reamix::ui::ModeTabs::Mode::Duration)
        {
            appMode_ = reamix::ui::ModeTabs::Mode::Duration;
            regionFromAuto_ = false;
            recomputeRegionState();
        }
        return;
    }

    const auto& it = *picked;

    // ADR-056 (sesja 66) — fetch itemGuid eagerly for polling-decision logic
    // below. REAPER's getItemGuid is a fast string-keyed lookup, safe to
    // call at 10 Hz polling rate.
    const juce::String itGuid = reamix::reaper::getItemGuid (it.itemPtr);

    const bool firstSelection = ! hasSelection_;
    const bool pathChanged    = prevSourcePath_.isNotEmpty()
                                && prevSourcePath_ != it.sourcePath;
    // ADR-056 (sesja 66) — itemGuid identity change. Triggers fresh attach
    // even when sourcePath unchanged (post-Region-Insert clip click,
    // post-Cmd+Z original restore). Without this, polling skipped
    // applySelectedItem and plugin kept stale tmpWavPath_ / state from
    // prior MediaItem — direct cause of BUG-22 and BUG-20.
    const bool itemChanged    = prevItemGuid_.isNotEmpty()
                                && itGuid.isNotEmpty()
                                && prevItemGuid_ != itGuid;

    if (firstSelection || it.sourcePath != prevSourcePath_ || itemChanged)
    {
        applySelectedItem (it.name, it.durationSec, it.sourceFileDuration,
                           it.sourcePath, it.itemPtr);

        if (pathChanged)
        {
            const bool hasBundle      = analysisBundles_.count (it.sourcePath) > 0;
            const bool workerRunsHere = (activeAnalyzePath_ == it.sourcePath)
                                         && analyzePipeline_ != nullptr;
            if (! hasBundle && ! workerRunsHere)
            {
                transportBar_.setState (reamix::ui::TransportState::Idle);
                statusBar_.setText (juce::String::fromUTF8 (
                    "Source changed \xe2\x80\x94 re-analyze required"));
            }
        }

        hasSelection_   = true;
        prevSourcePath_ = it.sourcePath;
        prevItemGuid_   = itGuid;     // ADR-056 (sesja 66)
    }
    else
    {
        const auto bundleHit = analysisBundles_.find (it.sourcePath);
        const bool hasAnalysis = (bundleHit != analysisBundles_.end()
                                   && bundleHit->second != nullptr);

        // ADR-056 (sesja 66) — same-item polling refresh; if this is a
        // Region group click, keep the canonical view (full audio length
        // + source filename) instead of overwriting back to the inserted
        // clip's fragment duration.
        const auto sameTickGroup =
            reamix::reaper::regionGroupTracker::findGroupForSelectedItem (itGuid);
        const bool sameTickIsRegionGroupClick = sameTickGroup.has_value();
        const double effectiveDurSi = sameTickIsRegionGroupClick
                                        ? it.sourceFileDuration
                                        : it.durationSec;

        reamix::ui::SourceInfo si;
        si.name     = sameTickIsRegionGroupClick
                        ? it.sourcePath.fromLastOccurrenceOf ("/", false, false)
                        : it.name;
        si.duration = effectiveDurSi;
        si.bpm      = hasAnalysis ? bundleHit->second->bpm : 0.0;
        si.beats    = hasAnalysis ? (int) bundleHit->second->beatTimes.size() : 0;
        si.empty    = false;
        sourcePanel_.setSource (si);

        // ADR-049 — when item is a remix clip belonging to a group, the slider
        // range covers the whole original audio file, not the clip length.
        // ADR-056 (sesja 66) — Region group click: use sourceFileDuration
        // directly (effectiveSourceDurationFor only handles Duration's
        // groupTracker, not Region's).
        const double effectiveDur = sameTickIsRegionGroupClick
                                      ? it.sourceFileDuration
                                      : effectiveSourceDurationFor (it);
        const double maxSec = std::round (effectiveDur * kDurationMaxMultiplier);
        durationPanel_.setRange (kDurationMinSec, std::max (kDurationMinSec + 1.0, maxSec));
        durationPanel_.setOriginalDuration (effectiveDur);
    }

    // ── Auto-detect REAPER time-selection (sesja 60) ─────────────────
    // Plugin auto-flips into Region mode when REAPER time-selection becomes a
    // *new* value (not previously respected). Manual switch back to Duration
    // is sticky — auto-detector won't re-flip until time-selection changes.
    auto curSel = reamix::reaper::getTimeSelection();

    auto sameSelection = [] (const std::optional<reamix::reaper::TimeSelection>& a,
                              const std::optional<reamix::reaper::TimeSelection>& b)
    {
        if (a.has_value() != b.has_value()) return false;
        if (! a.has_value()) return true;
        return std::abs (a->startSec - b->startSec) < 0.001
            && std::abs (a->endSec   - b->endSec)   < 0.001;
    };

    if (! sameSelection (curSel, lastRespectedTimeSelection_))
    {
        // Time-selection changed (or appeared/disappeared). Decide whether
        // to auto-flip mode.
        if (curSel.has_value()
            && appMode_ != reamix::ui::ModeTabs::Mode::Region)
        {
            // Verify the selection overlaps the current item by ≥ 6 s.
            auto rg = reamix::reaper::getItemRegion (it.durationSec, it.positionSec);
            if (rg.has_value())
            {
                appMode_ = reamix::ui::ModeTabs::Mode::Region;
                regionFromAuto_ = true;
                modeTabs_.setMode (appMode_);
                // Sesja 65 — record auto-flip as user intent (REAPER time-
                // selection IS the user's signal). See lastModeByPath_.
                if (currentSourcePath_.isNotEmpty())
                    lastModeByPath_[currentItemKey()] = appMode_;
                // ADR-050 Filozofia A — auto-flip resets the visible material
                // exactly like a manual click does.
                resetMaterialView();
            }
        }
        else if (! curSel.has_value()
                 && appMode_ == reamix::ui::ModeTabs::Mode::Region
                 && regionFromAuto_)
        {
            // Time-selection was cleared and we were in auto-Region. Drop
            // back to Duration. (User-manual Region stays.)
            appMode_ = reamix::ui::ModeTabs::Mode::Duration;
            regionFromAuto_ = false;
            modeTabs_.setMode (appMode_);
            if (currentSourcePath_.isNotEmpty())
                lastModeByPath_[currentItemKey()] = appMode_;
            resetMaterialView();
        }

        lastRespectedTimeSelection_ = curSel;
    }

    recomputeRegionState();

    // DEV-050 (sesja 100b) — Cmd+Z detection on the most recent Insert /
    // Update. We track the first inserted clip's GUID; if findItemByGuid
    // stops resolving it, REAPER undo removed the inserted items.
    // ExtState entries written by GroupTracker / RegionGroupTracker /
    // BlocksGroupTracker do NOT participate in REAPER undo, so polling
    // those would always succeed and miss the signal — clip-GUID
    // resolution is the only reliable trigger. One-shot notice + clear
    // so subsequent ticks don't re-fire. Placed at the END of the tick
    // (after applySelectedItem) so the cache-restored notice that fires
    // when REAPER auto-selects the now-restored source item lands first
    // and our more specific "Insert undone" notice paints over it.
    if (lastInsertedClipGuid_.isNotEmpty()
        && reamix::reaper::findItemByGuid (lastInsertedClipGuid_) == nullptr)
    {
        statusBar_.setNotice ("Insert undone \xe2\x80\x94 Edit to start over", 3500);
        lastInsertedClipGuid_.clear();
    }
}

void MainComponent::clearCurrentRemix()
{
    currentRemix_.reset();
    currentRemixMode_.reset();
}

juce::uint64 MainComponent::computeBlocksHash() const
{
    if (userBlocks_.size() < 1 || userBlocksQueue_.size() < 2) return 0;
    juce::uint64 h = 1469598103934665603ull; // FNV-1a 64-bit basis
    const auto mix = [&] (juce::uint64 v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    for (const auto& b : userBlocks_)
    {
        mix ((juce::uint64) std::lround (b.startSec * 1000.0));
        mix ((juce::uint64) std::lround (b.endSec   * 1000.0));
        mix ((juce::uint64) (int) b.kind);
    }
    for (int q : userBlocksQueue_) mix ((juce::uint64) q);
    for (int v : junctionVariations_) mix ((juce::uint64) v);
    // Splice flexibility removed sesja 68 ADR-057 — no longer part of hash.
    return h;
}

bool MainComponent::tryRestoreModeRemix (reamix::ui::ModeTabs::Mode m)
{
    using Mode = reamix::ui::ModeTabs::Mode;
    if (currentSourcePath_.isEmpty()) return false;
    if (analysisBundles_.count (currentSourcePath_) == 0) return false;

    std::set<std::pair<int,int>> blocked;
    if (auto bIt = blockedBySource_.find (currentSourcePath_);
        bIt != blockedBySource_.end())
        blocked = bIt->second;

    int variation = 0;
    if (auto vIt = variationBySource_.find (currentSourcePath_);
        vIt != variationBySource_.end())
        variation = vIt->second;

    auto pmIt = targetByPathMode_.find (currentItemKey());
    auto findTarget = [&] (Mode mode) -> std::optional<double>
    {
        if (pmIt == targetByPathMode_.end()) return std::nullopt;
        auto tIt = pmIt->second.find (mode);
        if (tIt == pmIt->second.end()) return std::nullopt;
        return tIt->second;
    };

    if (m == Mode::Duration)
    {
        const auto target = findTarget (m);
        if (! target.has_value()) return false;

        auto key = reamix::ui::makeRemixCacheKey (
            currentSourcePath_, currentItemGuid_,   // ADR-056 (sesja 66)
            *target, 0.0, 0.0, blocked, variation);
        {
            const auto ap = currentAuditionParams();
            key.auditionHash = reamix::ui::hashAuditionParams (
                ap.tone, ap.editLength, ap.allowPmSeconds, ap.minCutBeats);
            key.qualityWeightsHash = reamix::ui::hashQualityWeights (currentQualityWeights());
        }
        const auto* hit = remixCache_.find (key);
        if (hit == nullptr) return false;
        if (! juce::File (hit->tmpWavPath).existsAsFile()) return false;

        durationPanel_.setTarget (*target);
        applyRemixToUi (*hit);
        currentRemix_     = *hit;
        currentRemixMode_ = m;
        return true;
    }

    if (m == Mode::Region)
    {
        auto rIt = lastRegionByPath_.find (currentItemKey());
        if (rIt == lastRegionByPath_.end()) return false;

        const double target = findTarget (m).value_or (rIt->second.targetSec);
        auto key = reamix::ui::makeRemixCacheKey (
            currentSourcePath_, currentItemGuid_,   // ADR-056 (sesja 66)
            target,
            rIt->second.region.startSec, rIt->second.region.endSec,
            blocked, variation);
        {
            const auto ap = currentAuditionParams();
            key.auditionHash = reamix::ui::hashAuditionParams (
                ap.tone, ap.editLength, ap.allowPmSeconds, ap.minCutBeats);
            key.qualityWeightsHash = reamix::ui::hashQualityWeights (currentQualityWeights());
        }
        const auto* hit = remixCache_.find (key);
        if (hit == nullptr) return false;
        if (! juce::File (hit->tmpWavPath).existsAsFile()) return false;

        reamix::ui::WaveformView::SelectionRange sel;
        sel.startSec = rIt->second.region.startSec;
        sel.endSec   = rIt->second.region.endSec;
        selectedRange_ = sel;
        waveformView_.setSelection (sel);
        durationPanel_.setTarget (target);
        applyRemixToUi (*hit);
        currentRemix_     = *hit;
        currentRemixMode_ = m;
        return true;
    }

    if (m == Mode::Blocks)
    {
        // Sesja 65 — respect Edit-arrangement dismissal also on tab
        // re-entry (manual click). User who explicitly abandoned the
        // assembled view via "↺ Edit arrangement" should not see it
        // restored on the next Blocks tab click — they want to keep
        // editing the queue, not re-view the dismissed remix.
        if (blocksDismissedSources_.count (currentItemKey()) > 0)
            return false;

        const juce::uint64 blocksHash = computeBlocksHash();
        if (blocksHash == 0) return false;

        const double target = findTarget (m).value_or (durationPanel_.getTarget());
        auto key = reamix::ui::makeRemixCacheKey (
            currentSourcePath_, currentItemGuid_,   // ADR-056 (sesja 66)
            target, 0.0, 0.0, blocked, variation);
        key.blocksHash = blocksHash;
        {
            const auto ap = currentAuditionParams();
            key.auditionHash = reamix::ui::hashAuditionParams (
                ap.tone, ap.editLength, ap.allowPmSeconds, ap.minCutBeats);
            key.qualityWeightsHash = reamix::ui::hashQualityWeights (currentQualityWeights());
        }
        const auto* hit = remixCache_.find (key);
        if (hit == nullptr) return false;
        if (! juce::File (hit->tmpWavPath).existsAsFile()) return false;

        durationPanel_.setTarget (target);
        applyRemixToUi (*hit);
        currentRemix_     = *hit;
        currentRemixMode_ = m;
        return true;
    }

    return false;
}

void MainComponent::resetMaterialView()
{
    // ADR-050 Filozofia A — switching modes resets the user-visible material
    // back to "the original audio". Concretely:
    //   - waveform variant → Source (peaks of bundle.sourcePath, not of any
    //     pending Remix output);
    //   - drag-select cleared (mode-specific selection semantics);
    //   - slider target → original duration baseline so a fresh Region pick
    //     or fresh Duration retarget starts from a neutral position.
    //
    // currentRemix_ is NOT reset — the cached remix output stays available
    // (e.g. for Insert if user hits it without re-targeting). Only the
    // *visible* state is reverted.
    waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
    waveformView_.setSpliceMarkers ({});
    waveformView_.setEditPlan ({});
    waveformView_.clearSelection();
    waveformView_.setShowEditRegionButton (false);
    selectedRange_.reset();
    lastSeekSec_.reset();
    lastRegionSelection_.reset();
    if (peaksProvider_)
        waveformView_.setPeaksProvider (peaksProvider_.get());
    if (currentSourceDurationSec_ > 0.0)
        durationPanel_.setTarget (currentSourceDurationSec_);
    tooltip_.hide();
}

void MainComponent::recomputeRegionState()
{
    using Mode = reamix::ui::ModeTabs::Mode;

    // Resolve the effective region from current state.
    std::optional<reamix::reaper::ItemRegion> region;

    if (appMode_ == Mode::Region)
    {
        if (regionFromAuto_)
        {
            // Auto-detected: REAPER time-selection drives the region. Convert
            // back to item-relative via getItemRegion (also re-validates
            // ≥ 6 s + not-entire-item gates).
            auto picked = reamix::reaper::getSelectedItem();
            if (picked.has_value())
                region = reamix::reaper::getItemRegion (picked->durationSec,
                                                         picked->positionSec);
        }
        else
        {
            // Manual mode: plugin's waveform selection IS the region. Apply
            // the same ≥ 6 s minimum (sub-6s = treat as preview-only, no
            // region — Insert disabled).
            if (selectedRange_.has_value()
                && (selectedRange_->endSec - selectedRange_->startSec) >= 6.0)
            {
                reamix::reaper::ItemRegion r;
                r.startSec = selectedRange_->startSec;
                r.endSec   = selectedRange_->endSec;
                region = r;
            }
        }

        // Sesja 65 BUG-21 follow-up — when a Region remix is already showing
        // (currentRemix_ holds the region bounds it was produced for), keep
        // the slider active for dynamic re-target without forcing the user
        // through "↺ Edit region" first. applyRemixToUi resets selectedRange_
        // (line 2173 — Remix variant timeline doesn't match the source-time
        // scrim), so without this fallback the slider goes inactive on the
        // next recomputeRegionState tick. Mirror of deriveEffectiveRegion.
        if (! region.has_value()
            && currentRemix_.has_value()
            && (currentRemix_->regionEndSec - currentRemix_->regionStartSec) > 0.001)
        {
            reamix::reaper::ItemRegion r;
            r.startSec = currentRemix_->regionStartSec;
            r.endSec   = currentRemix_->regionEndSec;
            region = r;
        }
    }

    currentRegion_ = region;

    // Push to ModeTabs — keep mode + AUTO flag in sync.
    modeTabs_.setMode (appMode_);
    modeTabs_.setAutoFlag (appMode_ == Mode::Region && regionFromAuto_);

    // ADR-051 — Block Assembly mode is the only place segBar drag-mark
    // is active and where user blocks render. Other modes hide the tiles
    // (the segBar reverts to anchor-only base layer).
    const bool inBlocks = (appMode_ == Mode::Blocks);
    waveformView_.setBlockMarkingEnabled (inBlocks);
    waveformView_.setUserBlocks (inBlocks ? userBlocks_
                                          : std::vector<reamix::ui::UserBlock>{});
    blockAssemblyPanel_.setUserBlocks (inBlocks ? userBlocks_
                                                : std::vector<reamix::ui::UserBlock>{});
    blockAssemblyPanel_.setQueue      (inBlocks ? userBlocksQueue_
                                                : std::vector<int>{});
    if (! inBlocks)
        waveformView_.setShowEditArrangementButton (false);

    // Mode-panel slot swap (DurationPanel ↔ BlockAssemblyPanel).
    resized();

    // Push to DurationPanel — label + tickrow + slider semantics flip.
    if (region.has_value())
    {
        durationPanel_.setRegion (
            reamix::ui::DurationPanel::RegionInfo { region->startSec, region->endSec });
        durationPanel_.setOriginalDuration (region->endSec - region->startSec);
        durationPanel_.setActive (true);
    }
    else
    {
        durationPanel_.setRegion (std::nullopt);
        if (currentSourceDurationSec_ > 0.0)
            durationPanel_.setOriginalDuration (currentSourceDurationSec_);
        // Region mode without a selection — slider has no defined target, so
        // disable it. Hint via status bar so user knows what to do next.
        if (appMode_ == Mode::Region)
        {
            durationPanel_.setActive (false);
            statusBar_.setText (juce::String::fromUTF8 (
                "Region mode \xe2\x80\x94 drag in waveform to select fragment"));
        }
        else
        {
            durationPanel_.setActive (true);
        }
    }
}

void MainComponent::applySelectedItem (const juce::String& name,
                                       double itemDurationSec,
                                       double sourceFileDurationSec,
                                       const juce::String& sourcePath,
                                       void*  itemPtr)
{
    // ADR-056 (sesja 66) — Region group dispatch detection. Clicking an
    // inserted Region clip (or the post-Insert pre-region piece sharing the
    // original's GUID) re-attaches plugin state to the canonical original-
    // item identity (sourceGuid + full audio duration) so user sees the
    // full remix view (waveform + Region tab + scrim + slider + Remix
    // variant w/ splice markers) instead of a fragment view. UX parity
    // with Duration mode's groupTracker (ADR-046). Closes DEV-035 candidate.
    const juce::String incomingGuid = reamix::reaper::getItemGuid (itemPtr);
    // Sesja 100b — Duration tracker dispatch (UX parity with Region +
    // Blocks). Without this, post-Insert clip clicks routed through the
    // generic flow with `newKey = (sourcePath, clipGuid)` — the new
    // clip GUID had no prior targetByPath_ entry so resolvedTarget fell
    // back to floor(durationSec), which missed the RemixCache (rendered
    // at the user's actual target). Cache miss left the waveform on
    // Source until kickRemixPipeline finished its async recompute. With
    // this dispatch, effectiveItemGuid canonicalizes to the source GUID
    // and the targetSec the group was inserted at pre-primes targetByPath_,
    // so the cache hit fires immediately and the Remix variant flips.
    auto durationGroup = reamix::reaper::groupTracker::findGroupForSelectedItem (incomingGuid);
    auto regionGroup = reamix::reaper::regionGroupTracker::findGroupForSelectedItem (incomingGuid);
    // DEV-041 (sesja 95 ADR-085) — Blocks group dispatch detection. Mirrors
    // Region's tracker query above. Block Assembly Insert clips share Duration's
    // groupTracker namespace BUT applySelectedItem doesn't query that for
    // dispatch (Duration is the default mode). Without a Blocks-specific
    // tracker, post-Insert clip clicks fell through to lastModeByPath_ check
    // which missed because the inserted clip's NEW GUID doesn't match the
    // pre-Insert key (saved under original item's GUID).
    auto blocksGroup = reamix::reaper::blocksGroupTracker::findGroupForSelectedItem (incomingGuid);
    // ADR-056 (sesja 66 fix A) — dismissed flag overrides group dispatch.
    // Composed from raw incomingGuid (source-of-truth in REAPER) AND from
    // group->sourceGuid (canonical identity user dismissed via Edit region):
    // either form must suppress the dispatch.
    const ItemKey rawKey { sourcePath, incomingGuid };
    const ItemKey canonicalKey { sourcePath,
                                 regionGroup.has_value() ? regionGroup->sourceGuid
                                                         : incomingGuid };
    const bool regionDispatchDismissed =
           regionGroupDismissedSources_.count (rawKey)       > 0
        || regionGroupDismissedSources_.count (canonicalKey) > 0;
    const bool isRegionGroupClick = regionGroup.has_value() && ! regionDispatchDismissed;

    // DEV-041 (sesja 95) — Blocks dispatch dismiss check. Region wins if both
    // present (extremely rare — a single GUID can only belong to one tracker
    // because each Insert produces fresh GUIDs; Region+Blocks for same source
    // would mean user inserted Region first then re-inserted same source as
    // Blocks). Reuse blocksDismissedSources_ (sesja 65 ADR-055) for "Edit
    // arrangement" dismiss intent — same in-memory map already used to suppress
    // post-Edit-arrangement re-prime.
    const ItemKey blocksCanonicalKey { sourcePath,
                                       blocksGroup.has_value() ? blocksGroup->sourceGuid
                                                               : incomingGuid };
    const bool blocksDispatchDismissed =
           blocksDismissedSources_.count (rawKey)              > 0
        || blocksDismissedSources_.count (blocksCanonicalKey) > 0;
    const bool isBlocksGroupClick = ! isRegionGroupClick
                                     && blocksGroup.has_value()
                                     && ! blocksDispatchDismissed;

    // Sesja 100b — Duration takes the lowest priority of the three (Region
    // wins; Blocks wins next). isDurationGroupClick fires only on indirect
    // clip lookups (Direct = clicking the source item with same GUID
    // doesn't change anything since effectiveItemGuid resolves to the same
    // value). No dismissed-flag analog: Duration mode has no "Edit
    // arrangement" / "Edit region" parallel where the user explicitly opts
    // out of group dispatch — they just click another item.
    const bool isDurationGroupClick = ! isRegionGroupClick
                                       && ! isBlocksGroupClick
                                       && durationGroup.has_value();

    const juce::String effectiveItemGuid =
          isRegionGroupClick   ? regionGroup->sourceGuid
        : isBlocksGroupClick   ? blocksGroup->sourceGuid
        : isDurationGroupClick ? durationGroup->sourceGuid
        :                        incomingGuid;
    const double effectiveItemDurationSec =
        (isRegionGroupClick || isBlocksGroupClick || isDurationGroupClick)
            ? sourceFileDurationSec
            : itemDurationSec;

    // Lua parity (remix_operations.lua:178-217 update_selected_item): does
    // NOT cancel running analysis — analysis runs to completion regardless
    // of current selection. Stop preview + clear visual transients only.
    previewController_.stop();
    waveformView_.setPlayhead (std::nullopt);
    selectedRange_.reset();
    lastSeekSec_.reset();
    waveformView_.clearSelection();

    // DEV-034 (sesja 63b) — reset all Region-mode UI artifacts before
    // attaching to the new item. Without this, the Region tab + Edit-region
    // overlay + lastRegionSelection_ from a previous item leak onto the
    // newly-selected one (visible on items with no analysis: Region tab
    // stays highlighted, "↺ Edit region" button paints, clicking it
    // restores the previous item's selection scrim). The conditional
    // lastRegionByPath_ restore further down overrides these defaults
    // when the new source has its own snapshot.
    appMode_                    = reamix::ui::ModeTabs::Mode::Duration;
    regionFromAuto_             = false;
    currentRegion_.reset();
    lastRegionSelection_.reset();
    lastRespectedTimeSelection_.reset();
    modeTabs_.setMode (reamix::ui::ModeTabs::Mode::Duration);
    modeTabs_.setAutoFlag (false);
    waveformView_.setShowEditRegionButton (false);
    waveformView_.setShowEditArrangementButton (false);
    durationPanel_.setRegion (std::nullopt);

    // DEV-034 follow-up — only persist the slider value when it actually
    // represents a Duration-mode target. The Region-mode slider carries a
    // different semantic (output length for a region, not the whole track),
    // and saving it here would leak into Duration on a later attach to the
    // same source.
    // ADR-056 — write under PRIOR item key (currentItemKey() reflects
    // pre-update currentSourcePath_/currentItemGuid_). currentItemPtr_/
    // currentItemGuid_ are updated to the NEW item below.
    if (currentSourcePath_.isNotEmpty()
        && appMode_ == reamix::ui::ModeTabs::Mode::Duration)
        targetByPath_[currentItemKey()] = durationPanel_.getTarget();

    // ADR-049 — for remix-grouped items the duration baseline is the source
    // file, not the clip's D_LENGTH. Lua bug-parity skipped (Lua copies the
    // same anti-pattern from remix_operations.lua:186,201-204).
    // ADR-056 (sesja 66) — when this is a Region-group click (inserted clip
    // or pre-region piece), use full sourceFileDuration even before the
    // ADR-049 group lookup runs — it has not been updated for Region groups
    // yet (Region uses its own tracker, not Duration's groupTracker).
    reamix::reaper::SelectedItem probe;
    probe.itemPtr            = itemPtr;
    probe.durationSec        = effectiveItemDurationSec;
    probe.sourceFileDuration = sourceFileDurationSec;
    const double effectiveDur = effectiveSourceDurationFor (probe);

    // ADR-056 (sesja 66) — compose the NEW item's ItemKey using the EFFECTIVE
    // identity. For inserted Region clips effectiveItemGuid == regionGroup
    // sourceGuid, so all subsequent map lookups attach to the canonical
    // original identity. For normal clicks effectiveItemGuid == raw itemGuid.
    const ItemKey newKey { sourcePath, effectiveItemGuid };

    // ADR-056 (sesja 66) — pre-prime per-item state maps from the saved
    // Region group BEFORE bundle resolution + auto-restore branches run.
    // This causes the existing DEV-034 + sesja 65 ADR-055 cache-hit + Region
    // restore code paths (~150 LOC further down) to fire automatically with
    // group's region bounds + target — no special-case branch needed.
    if (isRegionGroupClick)
    {
        lastModeByPath_[newKey] = reamix::ui::ModeTabs::Mode::Region;
        lastRegionByPath_[newKey] = RegionSnapshot {
            reamix::reaper::ItemRegion {
                regionGroup->regionStartSec, regionGroup->regionEndSec
            },
            regionGroup->targetSec
        };
        targetByPathMode_[newKey][reamix::ui::ModeTabs::Mode::Region] =
            regionGroup->targetSec;
        // Hint user that we just restored their Region group view (vs
        // attaching to a fresh fragment item).
        statusBar_.setNotice (juce::String::fromUTF8 (
            "Restored Region remix \xe2\x80\x94 ")
            + juce::String (regionGroup->regionStartSec, 1) + "s\xe2\x80\x93"
            + juce::String (regionGroup->regionEndSec, 1) + "s");
    }
    else if (isBlocksGroupClick)
    {
        // DEV-041 (sesja 95 ADR-085) — pre-prime lastModeByPath_ so the
        // wasBlocks check ~250 lines below fires for this newKey. Without
        // this pre-prime, post-Insert clip clicks landed on Duration default
        // (the canonical user-flagged sesja-66-close regression). queue +
        // userBlocks restoration is handled by restoreBlocksSession (called
        // further down) keyed by sourcePath — independent of GUID identity.
        lastModeByPath_[newKey] = reamix::ui::ModeTabs::Mode::Blocks;
        targetByPathMode_[newKey][reamix::ui::ModeTabs::Mode::Blocks] =
            blocksGroup->targetSec;
        statusBar_.setNotice (juce::String::fromUTF8 (
            "Restored Block Assembly view"));
    }
    else if (isDurationGroupClick)
    {
        // Sesja 100b — post-Insert clip click in Duration mode. Pre-prime
        // lastModeByPath_ so a stray Region auto-flip later doesn't pull
        // the user out of Duration; pre-prime targetByPath_ so
        // kickRemixPipeline (~120 lines below) gets a RemixCache hit at
        // the inserted target → applyRemixToUi flips to Remix variant
        // immediately. targetSec == 0 is a sentinel for legacy entries
        // (pre-sesja-100b GroupTracker schema); skip pre-prime in that
        // case so we don't clobber a real prior session value.
        lastModeByPath_[newKey] = reamix::ui::ModeTabs::Mode::Duration;
        if (durationGroup->targetSec > 0.0)
        {
            targetByPath_[newKey] = durationGroup->targetSec;
            targetByPathMode_[newKey][reamix::ui::ModeTabs::Mode::Duration] =
                durationGroup->targetSec;
        }
        // No notice — Duration is the default mode, the user just saw
        // the Insert succeed; surfacing "Restored Duration remix" would
        // be redundant noise after every Insert.
    }

    currentSourcePath_        = sourcePath;
    currentSourceDurationSec_ = effectiveDur;

    // ADR-051 (sesja 61) — user-marked blocks live per-item in P_EXT.
    // ADR-052 (sesja 63) — also capture the item GUID so async writes
    // can re-resolve through findItemByGuid (raw itemPtr can become
    // stale before a CallOutBox callback fires; see BUG-9).
    // ADR-056 (sesja 66) — currentItemGuid_ holds the EFFECTIVE GUID
    // (canonical original for Region group click) so currentItemKey()
    // helper returns the right composite identity for state lookups.
    currentItemPtr_  = itemPtr;
    currentItemGuid_ = effectiveItemGuid;

    // ADR-087 STATUS UPDATE 1 D9 (sesja 98) + ADR-097 sesja 107 — when the
    // advanced panel window is open, push persisted weights back into its
    // sliders (suppressCallback so no re-render fires on item-switch). Track
    // context (path + duration) feeds JSONL Save flow.
    if (advancedPanel_)
    {
        const auto w = currentQualityWeights();
        advancedPanel_->setRawWeights (w);
        advancedPanel_->setTrackContext (sourcePath, juce::String(),
                                          itemDurationSec, 0.0);
        // DEV-079 sesja 101 — badge "Modified" iff any per-mode override
        // exists for this item (any one of the 3 modes touched).
        bool itemHasOverride = false;
        if (auto pmIt = qualityWeightsByPathMode_.find (currentItemKey());
            pmIt != qualityWeightsByPathMode_.end())
            itemHasOverride = ! pmIt->second.empty();
        advancedPanel_->setBadgeState (
            itemHasOverride
              ? reamix::ui::AdvancedWeightsPanel::BadgeState::Modified
              : reamix::ui::AdvancedWeightsPanel::BadgeState::NoSave);
    }

    // ADR-052 (sesja 63 BUG-17/18) — try the in-memory session cache
    // first (keyed by sourcePath, not item identity). Hit means we are
    // re-attaching to a sourcePath we have worked on this plugin session
    // (deselect → re-select, or post-Insert clip with same source). Use
    // the cached state — it is at least as fresh as P_EXT (every mutation
    // writes to both). Miss falls back to P_EXT load + fresh queue/
    // variations (the existing sesja-61 behavior).
    if (restoreBlocksSession (sourcePath))
    {
        // Re-persist immediately so the now-current item's P_EXT mirrors
        // the restored state (covers BUG-18: post-Insert clip P_EXT is
        // empty even though the source has prior work cached).
        if (! currentItemGuid_.isEmpty())
        {
            void* live = reamix::reaper::findItemByGuid (currentItemGuid_);
            if (live != nullptr)
                reamix::reaper::saveUserBlocks (live, userBlocks_);
        }
    }
    else
    {
        userBlocks_      = reamix::reaper::loadUserBlocks (itemPtr);
        userBlocksQueue_.clear(); // queue is per-session, not persisted v1.
        junctionVariations_.clear();
    }
    blockAssemblyPanel_.setDirty (false);
    refreshBlockAssemblyUi();

    // ADR-051 — Blocks tab original gating: only enabled post-Analyze
    // because compatibility-matrix features (Try different splice / seam
    // re-roll) need the bundle. ADR-052 (sesja 63 BUG-19) decouples user-
    // mark visibility from analysis: if userBlocks_ already exist (loaded
    // from P_EXT or restored from session cache), Blocks tab MUST be
    // accessible so user can see + edit their persisted work without
    // re-running DSP. Compatibility-dependent UI elements gate themselves
    // separately on bundle presence.
    const bool hasBundle    = analysisBundles_.count (sourcePath) > 0;
    const bool hasUserMarks = ! userBlocks_.empty();
    modeTabs_.setBlocksEnabled (hasBundle || hasUserMarks);

    // ADR-052 — confirm to user that persisted state was actually
    // restored. Without this, post-reopen looks identical to first-time
    // attach — user has no signal that their previous work survived. Only
    // emitted when there IS state to report (P_EXT load returned blocks
    // OR session cache restored). Omitted when nothing to restore so the
    // standard "Ready to analyze" status stays visible.
    if (hasUserMarks)
    {
        statusBar_.setText (juce::String::fromUTF8 ("Restored ")
            + juce::String ((int) userBlocks_.size())
            + juce::String::fromUTF8 (" user block")
            + juce::String (userBlocks_.size() == 1 ? "" : "s")
            + juce::String::fromUTF8 (" from project"));
    }

    // currentRemix_ is per-source; clear on switch (will repopulate from
    // RemixCache hit or fresh RemixPipeline run for the new source).
    clearCurrentRemix();
    tmpWavPath_.clear();

    // ADR-056 (sesja 66) — use effectiveItemDurationSec so Region group
    // clicks (inserted clip / pre-region piece) display the FULL audio file
    // length, not the fragment's tiny duration. Plus override the displayed
    // name to the canonical source filename for inserted clips so the user
    // sees the original track name rather than "Remix M:SS-M:SS".
    reamix::ui::SourceInfo si;
    si.name     = isRegionGroupClick
                    ? sourcePath.fromLastOccurrenceOf ("/", false, false)
                    : name;
    si.duration = effectiveItemDurationSec;
    si.bpm      = 0.0;
    si.beats    = 0;
    si.empty    = false;
    sourcePanel_.setSource (si);

    // Sesja 60 — Analyze label reflects whether this source has a cached
    // bundle. After plugin restart the in-memory map is empty → label
    // says "Analyze" (not "Analyze Again", which would be misleading).
    sourcePanel_.setHasAnalysis (analysisBundles_.count (sourcePath) > 0);

    const double maxSec = std::round (effectiveDur * kDurationMaxMultiplier);
    durationPanel_.setRange (kDurationMinSec, std::max (kDurationMinSec + 1.0, maxSec));
    durationPanel_.setOriginalDuration (effectiveDur);
    durationPanel_.setActive (true);

    const double ceiling = effectiveDur * kDurationMaxMultiplier;
    const auto   remembered = targetByPath_.find (newKey);
    const double freshDefault = std::floor (effectiveDur);
    double       resolvedTarget = freshDefault;
    if (remembered != targetByPath_.end()
        && remembered->second >= kDurationMinSec
        && remembered->second <= ceiling)
    {
        resolvedTarget = remembered->second;
    }
    durationPanel_.setTarget (resolvedTarget);

    peaksProvider_->setSourcePath (sourcePath);
    // Restore Source variant + Source peaks provider on item switch.
    waveformView_.setPeaksProvider (peaksProvider_.get());
    waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
    waveformView_.setSpliceMarkers ({});
    waveformView_.setEditPlan ({});
    tooltip_.hide();
    waveformView_.setSegments ({});
    waveformView_.setBeats    ({});
    waveformView_.setHasSource (true);
    waveformView_.setAnalyzed (false);
    waveformView_.repaint();

    // Decide UI state in priority order:
    //   1) Worker is currently analyzing THIS item → restore progress.
    //   2) Bundle cached → restore Complete state, kick auto-remix at target.
    //   3) Fresh item → Idle state.
    if (activeAnalyzePath_ == sourcePath && analyzePipeline_ != nullptr)
    {
        const auto lp = lastAnalyzeProgress_.find (sourcePath);
        const double p01 = (lp != lastAnalyzeProgress_.end()) ? lp->second : 0.0;
        const auto ls = lastAnalyzeStage_.find (sourcePath);
        const juce::String stage = (ls != lastAnalyzeStage_.end()) ? ls->second : juce::String();
        sourcePanel_.setAnalyzing (true, p01, stage);
        resized();
        transportBar_.setState (reamix::ui::TransportState::Idle);
        statusBar_.setText ("Analyzing \xe2\x80\xa6 "
                            + juce::String ((int) std::round (p01 * 100.0)) + "%");
        analysisState_ = AnalysisState::Analyzing;
        headerBar_.setStatusKind (reamix::ui::HeaderStatus::Analyzing);
        return;
    }

    const auto bundleHit = analysisBundles_.find (sourcePath);
    reamix::ui::AnalysisBundlePtr resolvedBundle;
    bool fromDiskCache = false;
    if (bundleHit != analysisBundles_.end() && bundleHit->second != nullptr)
    {
        resolvedBundle = bundleHit->second;
    }
    else
    {
        // ADR-053 (sesja 63 BUG-19 follow-up) — try the persistent disk
        // cache before falling through to "Ready to analyze". Cross-project
        // sticky: a track analyzed in any prior plugin session is restored
        // here without DSP. tryLoad re-decodes audio from the source file
        // (~1-2 s for FLAC) and returns nullptr on missing / version-
        // mismatch / source-file-changed → user just re-clicks Analyze.
        if (auto disk = reamix::ui::AnalysisDiskCache::tryLoad (sourcePath))
        {
            analysisBundles_[sourcePath] = disk;
            resolvedBundle               = std::move (disk);
            fromDiskCache                = true;
        }
    }

    if (resolvedBundle != nullptr)
    {
        analysisState_ = AnalysisState::Complete;
        headerBar_.setStatusKind (reamix::ui::HeaderStatus::Ready);
        applyAnalysisToUi (*resolvedBundle, name, /*fromCache=*/true);

        // Sesja 65 — surface cache-hit feedback for BOTH in-memory and disk
        // cache. Previously only fired for disk; in-memory cross-item
        // returns silently restored. Notice paint priority survives the
        // immediate applyRemixToUi setText call from the kickRemixPipeline
        // cache hit below (auto-clears after 2.5 s).
        statusBar_.setNotice (juce::String::fromUTF8 (
            fromDiskCache
                ? "Analysis restored from disk cache \xe2\x80\x94 "
                : "Analysis restored from cache \xe2\x80\x94 ")
            + juce::String ((int) resolvedBundle->beatTimes.size())
            + " beats, "
            + juce::String (resolvedBundle->bpm, 1) + " BPM");

        // ADR-047 § 3 rule 5 — auto-trigger Remix at the now-restored
        // target, so Preview/Insert have something to play. Cache hit on
        // RemixPipeline will be instant if user revisits a target.
        // Per-source blocked + variation persist across item switches
        // (Lua state.blocked_transitions per-source semantics, session 57).
        PendingRemixOp op;
        op.targetSec = resolvedTarget;
        if (auto bIt = blockedBySource_.find (sourcePath); bIt != blockedBySource_.end())
            op.blockedTransitions = bIt->second;
        if (auto vIt = variationBySource_.find (sourcePath); vIt != variationBySource_.end())
            op.variation = vIt->second;

        // DEV-034 (sesja 63b) — if the last interaction with this source
        // produced a Region remix, restore that mode + bounds before kicking
        // the pipeline. The propagated regionStart/EndSec make the cache
        // key match the previous run, so RemixCache returns the cached
        // RemixOutput instantly and applyRemixToUi flips back to the Remix
        // variant + Edit-region overlay. Without this, a user clicking an
        // empty REAPER timeline area then re-selecting the item silently
        // dropped to Duration mode for the whole track.
        //
        // Defensive: skip the restore when the cached region falls outside
        // the current item's duration. This catches the post-Region-Insert
        // case where the original item has been split into pre/post-region
        // pieces — those pieces are smaller than the cached region, so
        // restoring would route Insert through coordinates that don't exist
        // in the current item. Insert that completes successfully clears
        // the entry below; this guard handles items that escape that path.
        //
        // Sesja 65 — gate on lastModeByPath_[sourcePath] == Region. Without
        // this gate the restore fires for items where user was last in
        // Blocks mode (test 59 user report: Item A in Blocks → Item B →
        // back to Item A unconditionally jumped to Region). The Region
        // snapshot remains in lastRegionByPath_ for the case where user
        // re-enters Region tab manually — modeTabs_.onChange peek path
        // would still find currentRemix_ if the in-session Region remix
        // is still around.
        const auto modeIt = lastModeByPath_.find (newKey);
        const bool wasRegion =
               modeIt != lastModeByPath_.end()
            && modeIt->second == reamix::ui::ModeTabs::Mode::Region;

        // Sesja 65 — set true by the wasRegion+no-snapshot branch (post
        // Edit-region attach) to suppress the unconditional Duration-mode
        // kickRemixPipeline below. Without this the consolation Duration
        // remix overrides the Region "no selection" state.
        bool skipAttachRemixKick = false;
        if (auto rIt = lastRegionByPath_.find (newKey);
            wasRegion
            && rIt != lastRegionByPath_.end()
            && rIt->second.region.endSec <= effectiveItemDurationSec + 0.001)
        {
            const auto& snap = rIt->second;
            appMode_        = reamix::ui::ModeTabs::Mode::Region;
            regionFromAuto_ = false;
            modeTabs_.setMode (appMode_);
            modeTabs_.setAutoFlag (false);

            reamix::ui::WaveformView::SelectionRange sel;
            sel.startSec = snap.region.startSec;
            sel.endSec   = snap.region.endSec;
            selectedRange_ = sel;
            waveformView_.setSelection (sel);

            // Restore the slider to the target that produced the cached
            // remix — without this the cache-key lookup below misses
            // (target differs from cached) and produces a fresh remix at
            // the freshDefault target instead of the one user actually had.
            durationPanel_.setTarget (snap.targetSec);

            // Suppress the next-tick auto-flip detector: whatever REAPER's
            // current time-selection is, treat it as already-respected so it
            // doesn't immediately overwrite the restored Region state.
            lastRespectedTimeSelection_ = reamix::reaper::getTimeSelection();

            op.regionStartSec = snap.region.startSec;
            op.regionEndSec   = snap.region.endSec;
            op.targetSec      = snap.targetSec;

            // Sesja 65 — setNotice instead of setText so the message
            // survives the immediate applyRemixToUi setText call from the
            // kickRemixPipeline cache hit below. Notice paint priority is
            // higher than text and auto-clears after 2.5 s.
            statusBar_.setNotice (juce::String::fromUTF8 ("Restored region ")
                + juce::String (snap.region.startSec, 1) + "s\xe2\x80\x93"
                + juce::String (snap.region.endSec, 1) + "s");
        }
        else if (rIt != lastRegionByPath_.end()
                 && rIt->second.region.endSec > effectiveItemDurationSec + 0.001)
        {
            // Item geometry no longer fits the snapshot (post-Insert split,
            // user re-imported a shorter version). Drop the stale entry so
            // we don't keep retrying. Sesja 65 — narrowed the condition to
            // geometry-mismatch only; previously this branch also erased
            // when wasRegion=false (mode-was-not-Region) which would lose
            // the snapshot for future Region tab re-entry.
            // ADR-056 (sesja 66) — uses effectiveItemDurationSec so Region
            // group clicks (where raw itemDurationSec is the small inserted
            // clip's length) don't false-positive erase the just-pre-primed
            // snapshot.
            lastRegionByPath_.erase (rIt);
        }
        else if (wasRegion)
        {
            // Sesja 65 — wasRegion + no snapshot path. Happens when user
            // clicked Edit region (which erases lastRegionByPath_) and
            // then switched items. Coming back: keep the user in Region
            // tab so they see "Region mode — drag in waveform to select
            // fragment" instead of being thrown back to Duration. No
            // remix to restore (cleared explicitly), no snapshot to drive
            // a kickRemixPipeline → recompute on the user's next region
            // pick. resetMaterialView clears any waveform splice markers
            // / edit plan that would otherwise leak from the prior remix
            // visual state. recomputeRegionState pushes the panel + tabs
            // to the Region semantic. Suppress the trailing kickRemixPipeline
            // — without that suppression we silently produce a Duration
            // remix (op has no region bounds) which puts splice markers
            // back on the waveform and overrides the "no selection" state
            // the user expects after Edit region.
            appMode_        = reamix::ui::ModeTabs::Mode::Region;
            regionFromAuto_ = false;
            resetMaterialView();
            recomputeRegionState();
            skipAttachRemixKick = true;
        }

        // Sesja 65 — Blocks symmetric auto-restore: if user was last in
        // Blocks for this source and Block work survived (userBlocks_
        // populated by restoreBlocksSession or P_EXT load), come back to
        // the Blocks tab. Don't auto-fire kickRemixPipeline — Blocks
        // assemble is heavier than a Region remix; user re-clicks Assemble
        // if they want the assembled output back. The queue/userBlocks UI
        // stays populated either way.
        const bool wasBlocks =
               modeIt != lastModeByPath_.end()
            && modeIt->second == reamix::ui::ModeTabs::Mode::Blocks;
        const bool blocksDismissed =
            blocksDismissedSources_.count (newKey) > 0;
        // ADR-056 (sesja 66 fix C) — restore Blocks tab purely on user-intent
        // (lastModeByPath_=Blocks). Pre-fix required ! userBlocks_.empty()
        // — but tab IS user intent regardless of whether they marked any
        // blocks. Without this fix: user clicks Blocks tab on item B (no
        // marks), navigates to item A, comes back to B → Duration default
        // (test 5 user repro sesja 66). Pre-existing gap from sesja 65,
        // surfaced + fixed in sesja 66.
        if (! wasRegion && wasBlocks)
        {
            appMode_ = reamix::ui::ModeTabs::Mode::Blocks;
            modeTabs_.setMode (appMode_);
            recomputeRegionState();
            // Sesja 65 — when user clicked Edit arrangement before leaving
            // this source, suppress the kickRemixPipeline auto-restore so
            // the assembled view does NOT come back (mirror of wasRegion+
            // no-snapshot path). User sees Blocks tab with queue/userBlocks
            // restored but Source variant — they can re-edit or re-Assemble.
            // ADR-056 — also suppress kick when userBlocks_.empty() (no
            // arrangement to compute) so we don't fire a no-op pipeline.
            if (blocksDismissed || userBlocks_.empty())
                skipAttachRemixKick = true;
        }

        if (! skipAttachRemixKick)
            kickRemixPipeline (op);
        else
        {
            // Sesja 65 — wasRegion+no-snapshot path or wasBlocks+dismissed
            // path landed. Align transport state + Source-variant preview
            // wiring so user can click the waveform to seek/preview the
            // original audio without first triggering a remix recompute.
            // tmpWavPath_ was cleared by the applySelectedItem reset
            // block; without re-pointing it at currentSourcePath_, every
            // waveform click would call previewController_.play("") and
            // silently fail (user-reported test 59 follow-up: "preview
            // jakby nie reaguje" until first remix calc fires).
            analysisState_ = AnalysisState::Complete;
            transportBar_.setState (reamix::ui::TransportState::Ready);
            if (currentSourcePath_.isNotEmpty())
                tmpWavPath_ = currentSourcePath_;
        }
    }
    else
    {
        sourcePanel_.setAnalyzing (false, 0.0);
        resized();
        analysisState_ = AnalysisState::Idle;
        transportBar_.setState (reamix::ui::TransportState::Idle);
        // BUG-19 status note: only emit "Ready to analyze" when there are
        // no user blocks to surface (otherwise the BUG-19 fix path already
        // wrote the more specific "Restored N user blocks" message).
        if (userBlocks_.empty())
            statusBar_.setText ("Ready to analyze");
    }
}

void MainComponent::applyEmptyState()
{
    previewController_.stop();
    waveformView_.setPlayhead (std::nullopt);
    selectedRange_.reset();
    lastSeekSec_.reset();
    waveformView_.clearSelection();

    // DEV-034 follow-up — only persist the slider value when it actually
    // represents a Duration-mode target. The Region-mode slider carries a
    // different semantic (output length for a region, not the whole track),
    // and saving it here would leak into Duration on a later attach to the
    // same source.
    if (currentSourcePath_.isNotEmpty()
        && appMode_ == reamix::ui::ModeTabs::Mode::Duration)
        targetByPath_[currentItemKey()] = durationPanel_.getTarget();

    sourcePanel_.setAnalyzing (false, 0.0);

    currentSourcePath_        = {};
    currentSourceDurationSec_ = 0.0;
    clearCurrentRemix();
    tmpWavPath_.clear();
    analysisState_ = AnalysisState::Idle;

    // DEV-050 (sesja 100b) — drop the pending undo-tracker on deselect so
    // a stale GUID can't fire a notice for a Cmd+Z that happened earlier
    // on a different item.
    lastInsertedClipGuid_.clear();

    // ADR-051 — empty state clears block memory in the UI.
    // ADR-052 (sesja 63 BUG-17) — snapshot to in-memory session cache
    // BEFORE clearing local state, so re-selecting the same source brings
    // the work back. Cache survives until plugin destruction. No P_EXT
    // write here — every mutation already persisted as it happened.
    cacheBlocksSession();
    currentItemPtr_  = nullptr;
    currentItemGuid_ = {};
    userBlocks_.clear();
    userBlocksQueue_.clear();
    junctionVariations_.clear();
    blockAssemblyPanel_.setUserBlocks ({});
    blockAssemblyPanel_.setQueue ({});
    blockAssemblyPanel_.setDirty (false);

    // Sesja 60 — empty state forces Duration + clears region/auto state.
    appMode_ = reamix::ui::ModeTabs::Mode::Duration;
    regionFromAuto_ = false;
    currentRegion_.reset();
    lastRespectedTimeSelection_.reset();
    lastRegionSelection_.reset();
    modeTabs_.setMode (reamix::ui::ModeTabs::Mode::Duration);
    modeTabs_.setAutoFlag (false);
    modeTabs_.setBlocksEnabled (false); // ADR-051 — re-disable until Analyze
    waveformView_.setBlockMarkingEnabled (false);
    waveformView_.setUserBlocks ({});
    waveformView_.setShowEditArrangementButton (false);
    durationPanel_.setRegion (std::nullopt);
    waveformView_.setShowEditRegionButton (false);

    reamix::ui::SourceInfo si;
    si.empty = true;
    sourcePanel_.setSource (si);
    sourcePanel_.setHasAnalysis (false);

    durationPanel_.setActive (false);

    peaksProvider_->setSourcePath ({});
    waveformView_.setPeaksProvider (peaksProvider_.get());
    waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
    waveformView_.setSpliceMarkers ({});
    waveformView_.setEditPlan ({});
    tooltip_.hide();
    waveformView_.setSegments ({});
    waveformView_.setBeats    ({});
    waveformView_.setHasSource (false);
    waveformView_.setAnalyzed (false);
    waveformView_.repaint();

    transportBar_.setState (reamix::ui::TransportState::Idle);
    statusBar_.setText ("No item selected");
}

void MainComponent::handleBlocksJunctionAction (int junctionIdx, int chosen)
{
    if (junctionIdx < 0 || junctionIdx >= (int) userBlocksQueue_.size() - 1) return;

    // DEV-076 sesja 100c — capture cleanliness BEFORE mutation. When clean
    // (assembly was up-to-date), variation re-roll triggers immediate
    // kickRemixPipeline so user sees the new splice without clicking
    // Assemble. When dirty (queue mutation pending), behaviour stays
    // deferred-until-Assemble.
    //
    // Sesja 100c iter 2 — DON'T invalidateBlocksAssembledOutput in the
    // real-time path: that flips waveform variant back to Source +
    // resets peaks provider → visible flash + brief marker disappearance.
    // Pipeline cache key includes junctionVariations_ (computeBlocksHash
    // line 2400) so the bumped variation guarantees a cache miss and
    // recompute; handleRemixComplete swaps state on completion. We only
    // need to mark the panel busy so the user has a visible spinner.
    const bool wasClean = currentRemix_.has_value()
                       && currentRemixMode_.has_value()
                       && *currentRemixMode_ == reamix::ui::ModeTabs::Mode::Blocks
                       && ! blockAssemblyPanel_.isAssembleBusy();

    auto kickClean = [this]
    {
        if (analysisState_ != AnalysisState::Complete) return;
        blockAssemblyPanel_.setAssembleBusy (true,
            juce::String::fromUTF8 ("Computing\xe2\x80\xa6"));

        // Snapshot current clips so handleRemixComplete can detect "exhausted"
        // (new variation produced same clips → revert variation + emit
        // "no more alternatives"). Mirrors Duration TryDifferent path.
        if (currentRemix_.has_value())
            tryDifferentSnapshot_ = currentRemix_->editPlan.clips;
        else
            tryDifferentSnapshot_.reset();

        PendingRemixOp op;
        op.targetSec = durationPanel_.getTarget();
        kickRemixPipeline (op);
    };

    switch (chosen)
    {
        case 1:
        {
            if ((int) junctionVariations_.size() <= junctionIdx)
                junctionVariations_.resize ((std::size_t) junctionIdx + 1, 0);
            junctionVariations_[(std::size_t) junctionIdx] =
                (junctionVariations_[(std::size_t) junctionIdx] + 1) % 5; // BLOCK_TOP_K = 5
            blockAssemblyPanel_.setJunctionVariations (junctionVariations_);
            cacheBlocksSession();

            if (wasClean)
            {
                statusBar_.setText (juce::String::fromUTF8 (
                    "Trying different splice"));
                kickClean();
            }
            else
            {
                // Deferred path — assembly is already dirty (queue mutation
                // pending). Standard invalidate so audition / playhead /
                // markers don't reference stale lastBlockSplices_ until
                // user clicks Assemble.
                invalidateBlocksAssembledOutput();
                blockAssemblyPanel_.setDirty (true);
                statusBar_.setText (juce::String::fromUTF8 (
                    "Splice variation #") + juce::String (
                        junctionVariations_[(std::size_t) junctionIdx] + 1)
                    + juce::String::fromUTF8 (" \xE2\x80\x94 click Assemble to apply"));
            }
            break;
        }
        case 2:
            if ((int) junctionVariations_.size() > junctionIdx
                && junctionVariations_[(std::size_t) junctionIdx] != 0)
            {
                junctionVariations_[(std::size_t) junctionIdx] = 0;
                blockAssemblyPanel_.setJunctionVariations (junctionVariations_);
                cacheBlocksSession();

                if (wasClean)
                {
                    statusBar_.setText (juce::String::fromUTF8 (
                        "Resetting splice to best"));
                    kickClean();
                }
                else
                {
                    invalidateBlocksAssembledOutput();
                    blockAssemblyPanel_.setDirty (true);
                    statusBar_.setText (juce::String::fromUTF8 (
                        "Splice reset to best \xE2\x80\x94 click Assemble to apply"));
                }
            }
            break;
        case 3:
            if (blockAssemblyPanel_.onSeamAudition)
                blockAssemblyPanel_.onSeamAudition (junctionIdx);
            break;
        default:
            break;
    }
}

void MainComponent::invalidateBlocksAssembledOutput()
{
    // Stop any audition/preview that might be reading from the about-to-be-
    // invalidated tmp WAV. Playhead clears too — its position is meaningful
    // only against a live preview source.
    previewController_.stop();
    waveformView_.setPlayhead (std::nullopt);

    // Clear all per-junction visualization state. lastBlockSplices_ is the
    // single source of truth for seam-pill audition positions (BUG-11) and
    // canvas-marker hover tooltips. waveformView splice markers + edit plan
    // drive canvas paint (BUG-13/15). UserBlockSplices drive segBar splice
    // lines (BUG-15). All four come from the most recent Assemble output;
    // any queue mutation makes them lies until the next Assemble runs.
    lastBlockSplices_.clear();
    waveformView_.setSpliceMarkers ({});
    waveformView_.setUserBlockSplices ({});
    waveformView_.setEditPlan ({});

    // Flip waveform back to Source variant if it was showing the previous
    // Assemble's remix peaks. Hide the Edit-arrangement overlay (it's
    // meaningful only when there IS an assembled output to edit back from).
    // Sesja 100b iter 2 — preserve the user's zoom/pan across this swap;
    // every Blocks-mode mutation (boundary drag, kind change, queue
    // reorder, ...) routes through here and a view reset every time
    // would force the user to re-zoom for every micro-edit.
    waveformView_.setVariant (reamix::ui::WaveformView::Variant::Source);
    waveformView_.setShowEditArrangementButton (false);
    waveformView_.setPeaksProviderPreserveView (peaksProvider_.get());

    // Drop the cached remix output + tmp WAV pointer. Insert / Preview
    // call sites guard against nullopt currentRemix_; transport flips back
    // to Idle to reflect "nothing playable until next Assemble".
    clearCurrentRemix();
    tmpWavPath_.clear();
    transportBar_.setState (reamix::ui::TransportState::Idle);
}

void MainComponent::cacheBlocksSession()
{
    // ADR-052 (sesja 63 BUG-17/18) — snapshot working state into in-memory
    // session map so a subsequent deselect / Insert / re-select can restore
    // it. No-op when no source is attached (would key on an empty path).
    if (currentSourcePath_.isEmpty()) return;
    BlocksSessionState s;
    s.userBlocks         = userBlocks_;
    s.userBlocksQueue    = userBlocksQueue_;
    s.junctionVariations = junctionVariations_;
    blocksSessionByPath_[currentSourcePath_] = std::move (s);
}

bool MainComponent::restoreBlocksSession (const juce::String& sourcePath)
{
    auto it = blocksSessionByPath_.find (sourcePath);
    if (it == blocksSessionByPath_.end()) return false;
    userBlocks_         = it->second.userBlocks;
    userBlocksQueue_    = it->second.userBlocksQueue;
    junctionVariations_ = it->second.junctionVariations;
    return true;
}

void MainComponent::persistUserBlocks()
{
    // ADR-051 (sesja 61) — write current userBlocks_ to P_EXT:reamix_blocks
    // for the currently-attached item. Empty vector clears the entry.
    // Caller is any mutation site (mark / kind change / boundary edit /
    // split / merge / delete). Cheap (<1 ms even for 100 blocks); safe
    // to call on every mutation without batching.
    //
    // ADR-052 (sesja 63 BUG-9 fix) — re-resolve the live MediaItem* from
    // the cached GUID right before the REAPER call. The cached
    // currentItemPtr_ may be stale (REAPER freed/replaced the item between
    // applySelectedItem and this async user gesture; symptom = SIGSEGV in
    // GetSetMediaItemInfo_String with PC=0). findItemByGuid returns
    // nullptr if the item is gone → silent no-op (state survives in the
    // in-memory session cache; will re-persist on next live item attach).
    //
    // ADR-052 (sesja 63 BUG-17/18) — also snapshot to the in-memory
    // session cache so deselect / Insert / re-select preserves the work.
    cacheBlocksSession();

    if (currentItemGuid_.isEmpty()) return;
    void* live = reamix::reaper::findItemByGuid (currentItemGuid_);
    if (live == nullptr) return;
    reamix::reaper::saveUserBlocks (live, userBlocks_);
}

void MainComponent::centerAlertWindowOverPlugin (juce::AlertWindow* aw)
{
    if (aw == nullptr) return;
    // Resolve the plugin window's screen bounds via this top-level component.
    auto* top = getTopLevelComponent();
    if (top == nullptr) return;
    const auto pluginScreenBounds = top->getScreenBounds();
    aw->setCentrePosition (pluginScreenBounds.getCentreX(),
                           pluginScreenBounds.getCentreY());
}

void MainComponent::loadCustomKindRegistry()
{
    // ADR-092 sesja 100c — single global key "reamix.custom_kinds" holds
    // JSON array of {id, name, color}. Survives REAPER restart, project
    // switch, machine session. Per-user, not per-project.
#if REAMIX_WITH_REAPER_IO
    if (! GetExtState) return;
    const char* raw = GetExtState ("reamix.me", "custom_kinds");
    if (raw == nullptr) return;
    const juce::String json = juce::String::fromUTF8 (raw);
    customKindRegistry_.deserialize (json);
#endif
}

void MainComponent::saveCustomKindRegistry()
{
#if REAMIX_WITH_REAPER_IO
    if (! SetExtState) return;
    const juce::String json = customKindRegistry_.serialize();
    SetExtState ("reamix.me", "custom_kinds", json.toRawUTF8(), true);
#endif
}

int MainComponent::cascadeCustomKindRemoval (const juce::String& removedId)
{
    // ADR-092 sesja 100c — when a custom kind is deleted, all UserBlocks
    // referencing the removed id (current attached item only) lose their
    // customKindId override; they fall back to b.kind (built-in). Other
    // items not currently loaded are unaffected here — kindDisplay()
    // gracefully handles registry-miss by falling back to b.kind.
    int n = 0;
    for (auto& b : userBlocks_)
    {
        if (b.customKindId.has_value() && *b.customKindId == removedId)
        {
            b.customKindId.reset();
            ++n;
        }
    }
    if (n > 0)
    {
        persistUserBlocks();
        refreshBlockAssemblyUi();
        clearCurrentRemix();
    }
    return n;
}

void MainComponent::showAddCustomKindModal (double clampedStart, double clampedEnd,
                                             juce::Point<int> /*anchorScreenPos*/,
                                             juce::Component::SafePointer<juce::Component> pickerSafe)
{
    // ADR-092 sesja 100c — Add custom kind modal. Two fields:
    //   • Name (text input)
    //   • Color (12 preset palette swatches; user picks one)
    //
    // On confirm: registry.add(name, color) → returns new id → save registry
    // → create UserBlock with customKindId = new id → push + sort + persist.
    // On cancel: just-marked region is forfeit (no UserBlock created).
    auto* aw = new juce::AlertWindow ("Add custom kind",
                                       "Name your section type and pick a color.",
                                       juce::AlertWindow::NoIcon);
    aw->addTextEditor ("name", "", "Name");

    // 12 preset colors — built-in segment palette.
    using namespace reamix::theme;
    const std::vector<std::pair<juce::String, juce::Colour>> presets = {
        { "Sage",     SegIntro },
        { "Teal",     SegVerse },
        { "Steel",    SegPreChorus },
        { "Copper",   SegChorus },
        { "Mauve",    SegPostChorus },
        { "Rose",     SegBridge },
        { "Ochre",    SegBuildup },
        { "Rust",     SegDrop },
        { "Indigo",   SegBreakdown },
        { "Gold",     SegSolo },
        { "Jade",     SegInstrumental },
        { "Gray",     SegOutro },
    };
    juce::StringArray colorNames;
    for (const auto& p : presets) colorNames.add (p.first);
    aw->addComboBox ("color", colorNames, "Color");

    aw->addButton ("Add",    1, juce::KeyPress (juce::KeyPress::returnKey));
    aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
    centerAlertWindowOverPlugin (aw);

    aw->enterModalState (true,
        juce::ModalCallbackFunction::create (
            [this, aw, presets, pickerSafe] (int result)
        {
            std::unique_ptr<juce::AlertWindow> own (aw);
            if (result != 1)
            {
                // Cancel — picker stays; user can pick another kind or Esc.
                return;
            }

            const juce::String name = aw->getTextEditorContents ("name").trim();
            if (name.isEmpty())
            {
                statusBar_.setNotice ("Name required");
                return;
            }
            const int colorIdx = aw->getComboBoxComponent ("color")->getSelectedItemIndex();
            const juce::Colour color = (colorIdx >= 0 && colorIdx < (int) presets.size())
                ? presets[(std::size_t) colorIdx].second
                : reamix::theme::SegVerse;

            // Sesja 100c iter 4 — Add only mutates the registry. Picker
            // stays open so user can: (a) click the new tile to commit the
            // marked region with this kind, (b) Add another custom in
            // sequence, (c) pick an existing tile, or (d) Esc to cancel.
            // No region commit here — user makes the explicit pick after.
            customKindRegistry_.add (name, color);
            saveCustomKindRegistry();
            if (pickerSafe != nullptr) pickerSafe->repaint();

            statusBar_.setText ("Custom kind added \xe2\x80\x94 " + name
                + " \xe2\x80\x94 click it to apply");
        }), false);
}

void MainComponent::showAddCustomKindModalEditContext (
    juce::Component::SafePointer<juce::Component> popoverSafe)
{
    // ADR-092 sesja 100c — edit-context Add. Variant of showAddCustomKindModal
    // without the marked-region commit. Block being edited stays unchanged;
    // user clicks the new tile in the popover to apply it explicitly.
    auto* aw = new juce::AlertWindow ("Add custom kind",
                                       "Name your section type and pick a color.",
                                       juce::AlertWindow::NoIcon);
    aw->addTextEditor ("name", "", "Name");

    using namespace reamix::theme;
    const std::vector<std::pair<juce::String, juce::Colour>> presets = {
        { "Sage",     SegIntro },
        { "Teal",     SegVerse },
        { "Steel",    SegPreChorus },
        { "Copper",   SegChorus },
        { "Mauve",    SegPostChorus },
        { "Rose",     SegBridge },
        { "Ochre",    SegBuildup },
        { "Rust",     SegDrop },
        { "Indigo",   SegBreakdown },
        { "Gold",     SegSolo },
        { "Jade",     SegInstrumental },
        { "Gray",     SegOutro },
    };
    juce::StringArray colorNames;
    for (const auto& p : presets) colorNames.add (p.first);
    aw->addComboBox ("color", colorNames, "Color");

    aw->addButton ("Add",    1, juce::KeyPress (juce::KeyPress::returnKey));
    aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
    centerAlertWindowOverPlugin (aw);

    aw->enterModalState (true,
        juce::ModalCallbackFunction::create (
            [this, aw, presets, popoverSafe] (int result)
        {
            std::unique_ptr<juce::AlertWindow> own (aw);
            if (result != 1) return;

            const juce::String name = aw->getTextEditorContents ("name").trim();
            if (name.isEmpty()) { statusBar_.setNotice ("Name required"); return; }
            const int colorIdx = aw->getComboBoxComponent ("color")->getSelectedItemIndex();
            const juce::Colour color = (colorIdx >= 0 && colorIdx < (int) presets.size())
                ? presets[(std::size_t) colorIdx].second
                : reamix::theme::SegVerse;

            customKindRegistry_.add (name, color);
            saveCustomKindRegistry();
            if (popoverSafe != nullptr) popoverSafe->repaint();

            statusBar_.setText ("Custom kind added \xe2\x80\x94 " + name
                + " \xe2\x80\x94 click it to apply");
        }), false);
}

void MainComponent::showEditCustomKindModal (const juce::String& id,
                                              reamix::ui::CustomKindAction action,
                                              juce::Component::SafePointer<juce::Component> pickerSafe)
{
    // ADR-092 sesja 100c — open action-specific modal. After mutation,
    // saveCustomKindRegistry + repaint. For Delete: cascade userBlocks_.
    auto entry = customKindRegistry_.lookup (id);
    if (! entry.has_value())
    {
        statusBar_.setNotice ("Custom kind not found");
        return;
    }

    using Action = reamix::ui::CustomKindAction;

    if (action == Action::Rename)
    {
        auto* aw = new juce::AlertWindow ("Rename custom kind",
                                           "Enter new name for \"" + entry->name + "\".",
                                           juce::AlertWindow::NoIcon);
        aw->addTextEditor ("name", entry->name, "Name");
        aw->addButton ("Save",   1, juce::KeyPress (juce::KeyPress::returnKey));
        aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
        centerAlertWindowOverPlugin (aw);
        aw->enterModalState (true,
            juce::ModalCallbackFunction::create ([this, aw, id, pickerSafe] (int r)
        {
            std::unique_ptr<juce::AlertWindow> own (aw);
            if (r != 1) return;
            const juce::String newName = aw->getTextEditorContents ("name").trim();
            if (newName.isEmpty()) { statusBar_.setNotice ("Name required"); return; }
            customKindRegistry_.rename (id, newName);
            saveCustomKindRegistry();
            refreshBlockAssemblyUi();
            waveformView_.repaint();
            if (pickerSafe != nullptr) pickerSafe->repaint();
            statusBar_.setText ("Custom kind renamed \xe2\x80\x94 " + newName);
        }), false);
        return;
    }

    if (action == Action::Recolor)
    {
        auto* aw = new juce::AlertWindow ("Change color",
                                           "Pick a new color for \"" + entry->name + "\".",
                                           juce::AlertWindow::NoIcon);
        using namespace reamix::theme;
        const std::vector<std::pair<juce::String, juce::Colour>> presets = {
            { "Sage",     SegIntro },
            { "Teal",     SegVerse },
            { "Steel",    SegPreChorus },
            { "Copper",   SegChorus },
            { "Mauve",    SegPostChorus },
            { "Rose",     SegBridge },
            { "Ochre",    SegBuildup },
            { "Rust",     SegDrop },
            { "Indigo",   SegBreakdown },
            { "Gold",     SegSolo },
            { "Jade",     SegInstrumental },
            { "Gray",     SegOutro },
        };
        juce::StringArray colorNames;
        for (const auto& p : presets) colorNames.add (p.first);
        aw->addComboBox ("color", colorNames, "Color");
        aw->addButton ("Save",   1, juce::KeyPress (juce::KeyPress::returnKey));
        aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
        centerAlertWindowOverPlugin (aw);
        aw->enterModalState (true,
            juce::ModalCallbackFunction::create ([this, aw, id, presets, pickerSafe] (int r)
        {
            std::unique_ptr<juce::AlertWindow> own (aw);
            if (r != 1) return;
            const int idx = aw->getComboBoxComponent ("color")->getSelectedItemIndex();
            if (idx < 0 || idx >= (int) presets.size()) return;
            customKindRegistry_.recolor (id, presets[(std::size_t) idx].second);
            saveCustomKindRegistry();
            refreshBlockAssemblyUi();
            waveformView_.repaint();
            if (pickerSafe != nullptr) pickerSafe->repaint();
            statusBar_.setText ("Custom kind color updated");
        }), false);
        return;
    }

    if (action == Action::Delete)
    {
        // Count blocks that will cascade for confirm dialog text.
        int wouldCascade = 0;
        for (const auto& b : userBlocks_)
            if (b.customKindId.has_value() && *b.customKindId == id) ++wouldCascade;

        const juce::String warn = wouldCascade > 0
            ? "Delete \"" + entry->name + "\"?\n\n"
              + juce::String (wouldCascade) + (wouldCascade == 1 ? " block uses" : " blocks use")
              + " this kind. They will revert to their built-in fallback."
            : "Delete \"" + entry->name + "\"?";

        auto* aw = new juce::AlertWindow ("Delete custom kind", warn,
                                           juce::AlertWindow::NoIcon);
        aw->addButton ("Delete", 1, juce::KeyPress (juce::KeyPress::returnKey));
        aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
        centerAlertWindowOverPlugin (aw);
        aw->enterModalState (true,
            juce::ModalCallbackFunction::create ([this, aw, id, pickerSafe] (int r)
        {
            std::unique_ptr<juce::AlertWindow> own (aw);
            if (r != 1) return;
            const int n = cascadeCustomKindRemoval (id);
            customKindRegistry_.remove (id);
            saveCustomKindRegistry();
            refreshBlockAssemblyUi();
            waveformView_.repaint();
            if (pickerSafe != nullptr) pickerSafe->repaint();
            if (n > 0)
                statusBar_.setText ("Custom kind deleted \xe2\x80\x94 "
                    + juce::String (n) + (n == 1 ? " block reverted" : " blocks reverted"));
            else
                statusBar_.setText ("Custom kind deleted");
        }), false);
        return;
    }
}

void MainComponent::refreshBlockAssemblyUi()
{
    // ADR-051 — sync userBlocks_ + queue → BlockAssemblyPanel + WaveformView
    // segBar (the latter only paints when in Blocks mode per
    // setBlockMarkingEnabled gating in recomputeRegionState).
    blockAssemblyPanel_.setUserBlocks (userBlocks_);
    blockAssemblyPanel_.setQueue      (userBlocksQueue_);

    // Resize junctionVariations_ to match queue (zero-fill new entries).
    const int needed = std::max (0, (int) userBlocksQueue_.size() - 1);
    if ((int) junctionVariations_.size() != needed)
        junctionVariations_.resize ((std::size_t) needed, 0);
    blockAssemblyPanel_.setJunctionVariations (junctionVariations_);

    if (appMode_ == reamix::ui::ModeTabs::Mode::Blocks)
        waveformView_.setUserBlocks (userBlocks_);

    // ADR-052 (sesja 63 BUG-17/18) — every UI refresh that follows a
    // mutation snapshots state into the in-memory session cache. Cheap
    // (3 vector copies) and idempotent — caching the just-restored
    // state on item-attach is a no-op identity write.
    cacheBlocksSession();
}

// ─────────────────────────────────────────────────────────────────────
// DEV-058 sesja 100d — paleta context menu actions + selection helpers.
// ─────────────────────────────────────────────────────────────────────

void MainComponent::clearPaletteSelection()
{
    blockAssemblyPanel_.clearPaletteSelection();
}

void MainComponent::removeUserBlockAndCascade (int idx)
{
    if (idx < 0 || idx >= (int) userBlocks_.size()) return;
    userBlocks_.erase (userBlocks_.begin() + idx);
    auto& q = userBlocksQueue_;
    q.erase (std::remove_if (q.begin(), q.end(),
                              [idx] (int qi) { return qi == idx; }),
             q.end());
    for (auto& qi : q) if (qi > idx) --qi;
}

// DEV-058 (a) — Rename block. AlertWindow with TextEditor; on OK update
// userBlocks_[idx].labelOverride. Empty input clears the override (block
// reverts to kind label).
void MainComponent::showRenameBlockDialog (int blockIdx)
{
    if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
    const auto& b = userBlocks_[(std::size_t) blockIdx];
    const juce::String currentLabel = b.labelOverride.has_value() ? *b.labelOverride
                                                                   : juce::String();

    auto* aw = new juce::AlertWindow ("Rename block",
                                       "Enter a custom name for this block (leave blank to revert to kind label).",
                                       juce::MessageBoxIconType::NoIcon);
    aw->addTextEditor ("name", currentLabel, "Name:");
    aw->addButton ("OK",     1, juce::KeyPress (juce::KeyPress::returnKey));
    aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
    centerAlertWindowOverPlugin (aw);
    aw->enterModalState (true,
        juce::ModalCallbackFunction::create (
            [this, aw, blockIdx] (int result)
            {
                std::unique_ptr<juce::AlertWindow> own (aw);
                if (result != 1) return;
                if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
                const juce::String name = aw->getTextEditor ("name")->getText().trim();
                auto& b = userBlocks_[(std::size_t) blockIdx];
                if (name.isEmpty()) b.labelOverride.reset();
                else                b.labelOverride = name;
                persistUserBlocks();
                refreshBlockAssemblyUi();
                waveformView_.repaint();
                statusBar_.setText (juce::String ("Block renamed"));
            }));
}

// DEV-058 (a) — Change kind. Re-uses BlockKindPickerPopover anchored at
// click position. On pick: update userBlocks_[idx].kind + customKindId.
// Block-level mutation invalidates the assembled remix.
void MainComponent::showChangeKindMiniPicker (int blockIdx, juce::Point<int> screenPos)
{
    if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
    const auto& b = userBlocks_[(std::size_t) blockIdx];

    auto popover = std::make_unique<reamix::ui::BlockKindPickerPopover>(
        b.kind, b.startSec, b.endSec, &customKindRegistry_);
    const int rows = reamix::ui::BlockKindPickerPopover::rowsFor (
        reamix::ui::BlockKindPickerPopover::totalPillCount (&customKindRegistry_));
    popover->setSize (
        reamix::ui::BlockKindPickerPopover::kContentWidth,
        reamix::ui::BlockKindPickerPopover::contentHeightFor (rows));

    popover->onPicked =
        [this, blockIdx] (reamix::theme::SegmentKind k,
                           std::optional<juce::String> customKindId)
        {
            if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
            auto& b2 = userBlocks_[(std::size_t) blockIdx];
            b2.kind = k;
            b2.customKindId = customKindId;
            persistUserBlocks();
            refreshBlockAssemblyUi();
            waveformView_.repaint();
            invalidateBlocksAssembledOutput();
            statusBar_.setText (juce::String ("Block kind changed"));
        };

    popover->onAddCustomRequested =
        [this] (juce::Component::SafePointer<juce::Component> pickerSafe)
        {
            showAddCustomKindModalEditContext (pickerSafe);
        };

    popover->onEditCustomAction =
        [this] (juce::String id, reamix::ui::CustomKindAction action,
                 juce::Component::SafePointer<juce::Component> pickerSafe)
        {
            showEditCustomKindModal (id, action, pickerSafe);
        };

    const juce::Rectangle<int> anchor { screenPos.x, screenPos.y, 1, 1 };
    juce::CallOutBox::launchAsynchronously (std::move (popover), anchor, nullptr);
}

// DEV-058 (a) — Delete block. Confirm dialog + cascade queue. References
// to the block in userBlocksQueue_ are dropped; entries past idx shifted.
void MainComponent::showDeleteBlockConfirm (int blockIdx)
{
    if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return;
    const auto& b = userBlocks_[(std::size_t) blockIdx];
    const juce::String name = b.labelOverride.has_value() && b.labelOverride->isNotEmpty()
        ? *b.labelOverride
        : reamix::ui::builtinKindLabel (b.kind);

    auto* aw = new juce::AlertWindow ("Delete block",
        juce::String ("Delete block \"") + name + "\"?\n"
        "Any queue entries referencing this block will also be removed.",
        juce::MessageBoxIconType::WarningIcon);
    aw->addButton ("Delete", 1, juce::KeyPress (juce::KeyPress::returnKey));
    aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
    centerAlertWindowOverPlugin (aw);
    aw->enterModalState (true,
        juce::ModalCallbackFunction::create (
            [this, aw, blockIdx] (int result)
            {
                std::unique_ptr<juce::AlertWindow> own (aw);
                if (result != 1) return;
                removeUserBlockAndCascade (blockIdx);
                clearPaletteSelection();
                persistUserBlocks();
                refreshBlockAssemblyUi();
                waveformView_.setUserBlocks (userBlocks_);
                clearCurrentRemix();
                blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
                statusBar_.setText (juce::String ("Block deleted"));
            }));
}

// DEV-058 (b) — Batch delete. Confirm count + cascade. Sort indices
// descending so erasure shifts don't invalidate later indices.
void MainComponent::showDeleteBlocksBatchConfirm (std::vector<int> blockIndices)
{
    if (blockIndices.empty()) return;
    std::sort (blockIndices.begin(), blockIndices.end(), std::greater<int>());

    auto* aw = new juce::AlertWindow ("Delete blocks",
        juce::String ("Delete ") + juce::String ((int) blockIndices.size())
            + " blocks?\nQueue entries referencing them will also be removed.",
        juce::MessageBoxIconType::WarningIcon);
    aw->addButton ("Delete", 1, juce::KeyPress (juce::KeyPress::returnKey));
    aw->addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));
    centerAlertWindowOverPlugin (aw);
    aw->enterModalState (true,
        juce::ModalCallbackFunction::create (
            [this, aw, blockIndices = std::move (blockIndices)] (int result) mutable
            {
                std::unique_ptr<juce::AlertWindow> own (aw);
                if (result != 1) return;
                for (int idx : blockIndices) removeUserBlockAndCascade (idx);
                clearPaletteSelection();
                persistUserBlocks();
                refreshBlockAssemblyUi();
                waveformView_.setUserBlocks (userBlocks_);
                clearCurrentRemix();
                blockAssemblyPanel_.setDirty (userBlocksQueue_.size() >= 2);
                statusBar_.setText (juce::String ((int) blockIndices.size())
                    + " blocks deleted");
            }));
}

// DEV-058 (b) — Batch change kind. Apply user pick to all selected.
void MainComponent::showChangeKindBatchPicker (std::vector<int> blockIndices,
                                                  juce::Point<int> screenPos)
{
    if (blockIndices.empty()) return;
    const int firstIdx = blockIndices.front();
    if (firstIdx < 0 || firstIdx >= (int) userBlocks_.size()) return;

    const auto& b = userBlocks_[(std::size_t) firstIdx];
    auto popover = std::make_unique<reamix::ui::BlockKindPickerPopover>(
        b.kind, b.startSec, b.endSec, &customKindRegistry_);
    const int rows = reamix::ui::BlockKindPickerPopover::rowsFor (
        reamix::ui::BlockKindPickerPopover::totalPillCount (&customKindRegistry_));
    popover->setSize (
        reamix::ui::BlockKindPickerPopover::kContentWidth,
        reamix::ui::BlockKindPickerPopover::contentHeightFor (rows));

    popover->onPicked =
        [this, indices = std::move (blockIndices)] (
            reamix::theme::SegmentKind k,
            std::optional<juce::String> customKindId)
        {
            for (int idx : indices)
            {
                if (idx < 0 || idx >= (int) userBlocks_.size()) continue;
                userBlocks_[(std::size_t) idx].kind = k;
                userBlocks_[(std::size_t) idx].customKindId = customKindId;
            }
            persistUserBlocks();
            refreshBlockAssemblyUi();
            waveformView_.repaint();
            invalidateBlocksAssembledOutput();
            statusBar_.setText (juce::String ((int) indices.size())
                + " blocks updated");
        };

    popover->onAddCustomRequested =
        [this] (juce::Component::SafePointer<juce::Component> pickerSafe)
        {
            showAddCustomKindModalEditContext (pickerSafe);
        };

    popover->onEditCustomAction =
        [this] (juce::String id, reamix::ui::CustomKindAction action,
                 juce::Component::SafePointer<juce::Component> pickerSafe)
        {
            showEditCustomKindModal (id, action, pickerSafe);
        };

    const juce::Rectangle<int> anchor { screenPos.x, screenPos.y, 1, 1 };
    juce::CallOutBox::launchAsynchronously (std::move (popover), anchor, nullptr);
}

// DEV-058 (c) — paleta drag-reorder commit. Move userBlocks_[from] to
// position toBlockIdx (post-removal index). Remap userBlocksQueue_ so
// existing queue entries follow their block to the new position.
void MainComponent::commitPaletteReorder (int fromBlockIdx, int toBlockIdx)
{
    const int n = (int) userBlocks_.size();
    if (fromBlockIdx < 0 || fromBlockIdx >= n) return;
    if (toBlockIdx   < 0 || toBlockIdx   >= n) return;
    if (fromBlockIdx == toBlockIdx) return;

    // Build index permutation: oldIdx → newIdx after the move.
    // Move-element-in-vector semantic: extract block at fromBlockIdx,
    // insert at toBlockIdx. Track old→new for queue remap.
    std::vector<int> oldToNew ((std::size_t) n, -1);

    auto moved = userBlocks_[(std::size_t) fromBlockIdx];
    userBlocks_.erase (userBlocks_.begin() + fromBlockIdx);

    // After erase: indices > fromBlockIdx shifted down by 1.
    // After insert: indices >= toBlockIdx shifted up by 1.
    // For queue remap we want oldIdx → newIdx.
    if (toBlockIdx > fromBlockIdx)
    {
        // After erase fromBlockIdx, the target slot is toBlockIdx - 1
        // already (post-erase numbering). No additional shift needed
        // when inserting at toBlockIdx in post-erase space because
        // the caller passed insertion-style toBlockIdx that already
        // accounts for this (BlockAssemblyPanel::mouseUp logic).
        userBlocks_.insert (userBlocks_.begin() + toBlockIdx, moved);
    }
    else
    {
        userBlocks_.insert (userBlocks_.begin() + toBlockIdx, moved);
    }

    // Build forward map for queue remap.
    // In the OLD numbering, before any move:
    //   - fromBlockIdx maps to its new final position = toBlockIdx
    //   - other indices shift around the move
    if (fromBlockIdx < toBlockIdx)
    {
        for (int i = 0; i < fromBlockIdx; ++i) oldToNew[(std::size_t) i] = i;
        for (int i = fromBlockIdx + 1; i <= toBlockIdx; ++i) oldToNew[(std::size_t) i] = i - 1;
        oldToNew[(std::size_t) fromBlockIdx] = toBlockIdx;
        for (int i = toBlockIdx + 1; i < n; ++i) oldToNew[(std::size_t) i] = i;
    }
    else // toBlockIdx < fromBlockIdx
    {
        for (int i = 0; i < toBlockIdx; ++i) oldToNew[(std::size_t) i] = i;
        oldToNew[(std::size_t) fromBlockIdx] = toBlockIdx;
        for (int i = toBlockIdx; i < fromBlockIdx; ++i) oldToNew[(std::size_t) i] = i + 1;
        for (int i = fromBlockIdx + 1; i < n; ++i) oldToNew[(std::size_t) i] = i;
    }

    for (auto& qi : userBlocksQueue_)
    {
        if (qi >= 0 && qi < n) qi = oldToNew[(std::size_t) qi];
    }

    persistUserBlocks();
    refreshBlockAssemblyUi();
    waveformView_.setUserBlocks (userBlocks_);
    clearPaletteSelection();
    statusBar_.setText (juce::String ("Block reordered"));
}

void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll (reamix::theme::Bg1);
}

void MainComponent::resized()
{
    auto r = getLocalBounds();

    headerBar_    .setBounds (r.removeFromTop (40));
    sourcePanel_  .setBounds (r.removeFromTop (sourcePanel_.getPreferredHeight()));
    modeTabs_     .setBounds (r.removeFromTop (36));

    // ADR-051 — mode-aware mode-panel slot. Duration / Region modes show the
    // 78 h DurationPanel; Blocks mode swaps in BlockAssemblyPanel
    // (~170 h taller — paleta + queue + dirty-banner-reserved row).
    const bool inBlocks = (appMode_ == reamix::ui::ModeTabs::Mode::Blocks);
    durationPanel_      .setVisible (! inBlocks);
    blockAssemblyPanel_ .setVisible (inBlocks);
    if (inBlocks)
        blockAssemblyPanel_.setBounds (r.removeFromTop (blockAssemblyPanel_.getPreferredHeight()));
    else
        durationPanel_.setBounds (r.removeFromTop (78));

    statusBar_    .setBounds (r.removeFromBottom (22));
    transportBar_ .setBounds (r.removeFromBottom (54));
    // ADR-080 RESCOPE + ADR-083 sesja 92 + ADR-084 sesja 93 — AuditionBar
    // 4-slider panel. Bumped 124h → 156h (4 × 36h slider rows + 3 × 4h gaps)
    // for endpoint scale labels under each slider track per sesja 93 UX
    // pushback ("slidery sa nieintuicyjne, brak miarek"). Sits between
    // WaveformView (above) and TransportBar (below).
    // Iter-9.d — audition allocation 210 px (= kBodyHeight 186 + 24 internal
    // top/bot padding inside AuditionBar).
    // Sesja 108 — bumped 210 → 226 px to give the quick-Advanced icon (16×16
    // in the top padding gap) vertical breathing room. Top padding 12 → 28,
    // bottom stays 12. AuditionBar.resized + paint both removeFromTop(28) /
    // removeFromBottom(12) so Tone (first row) sits 28 px below top divider.
    auditionBar_.setBounds (r.removeFromBottom (auditionBar_.getPreferredHeight()));

    // ADR-097 sesja 107 — AdvancedWeightsPanel is hosted by
    // AdvancedWeightsWindow (separate juce::DocumentWindow), not embedded in
    // the main shell. The shell layout no longer reserves space for it.

    waveformView_ .setBounds (r);

    settingsPopover_.setBounds (getLocalBounds());
    supportPopover_ .setBounds (getLocalBounds());  // sesja 108
}

// Sesja 64 BUG-8 — accelerator hook entry point. Called from main.cpp's
// translateAccel callback when the user has clicked into our window (focused
// or mouse-over). Dispatches Space / Esc / Enter to the same handlers
// TransportBar uses; returns true to signal the key was consumed.
//
// Only handles the global transport shortcuts here. Component-local arrows
// (WaveformView seek) are dispatched through JUCE's normal focus path because
// the user must explicitly click the waveform to focus it (no docked-focus
// interception needed for them).
bool MainComponent::handleAccelKey (int vk, bool /*ctrl*/, bool /*shift*/)
{
    // OS virtual-key codes (macOS / Windows match for these): SPACE=0x20,
    // ESCAPE=0x1B, RETURN=0x0D.
    constexpr int kSpace  = 0x20;
    constexpr int kEscape = 0x1B;
    constexpr int kReturn = 0x0D;

    if (vk == kSpace || vk == kEscape)
    {
        if (transportBar_.onPlayStop) { transportBar_.onPlayStop(); return true; }
    }
    else if (vk == kReturn)
    {
        if (transportBar_.onInsert)   { transportBar_.onInsert();   return true; }
    }
    return false;
}

// ADR-080 RESCOPE + ADR-083 (sesja 92) + DEV-079 sesja 101 — return persisted
// AuditionBar params for the (currently-attached item, currently-active mode).
// Default-constructed AuditionParams (all fields = bit-exact baseline) when no
// entry exists — preserves current production behavior for first-touch
// interaction. Per-mode keying (DEV-079) prevents slider movement in mode B
// from corrupting cache lookups for mode A's previously-computed remix.
MainComponent::AuditionParams MainComponent::currentAuditionParams() const
{
    const auto key  = currentItemKey();
    const auto pmIt = auditionParamsByPathMode_.find (key);
    if (pmIt == auditionParamsByPathMode_.end()) return AuditionParams{};
    const auto mIt = pmIt->second.find (appMode_);
    if (mIt == pmIt->second.end()) return AuditionParams{};
    return mIt->second;
}

// ADR-087 STATUS UPDATE 1 D9 (sesja 98) + ADR-097 sesja 107 — return
// persisted advanced-weight QualityWeights for currently-attached item.
// DEV-080 sesja 108 — fallback flipped from compile-time kDefaultQualityWeights
// to host-side currentDefaultWeights_ so the user's "Set as defaults" choice
// reaches the compute path symmetrically with the panel's tick markers across
// REAPER restarts. currentDefaultWeights_ is seeded with kDefaultQualityWeights
// at construction and overwritten by loadAdvancedDefaultsFromExtState() when
// ExtState "advanced_defaults" carries a persisted record.
reamix::remix::QualityWeights MainComponent::currentQualityWeights() const
{
    // DEV-079 sesja 101 — per-(item, mode) lookup. Mirrors currentAuditionParams().
    const auto key  = currentItemKey();
    const auto pmIt = qualityWeightsByPathMode_.find (key);
    if (pmIt == qualityWeightsByPathMode_.end()) return currentDefaultWeights_;
    const auto mIt = pmIt->second.find (appMode_);
    if (mIt == pmIt->second.end()) return currentDefaultWeights_;
    return mIt->second;
}

// DEV-080 sesja 108 — read ExtState "reamix.me" / "advanced_defaults" into
// host-side currentDefaultWeights_ + currentDefaultBeta_. Called once from
// MainComponent ctor so the compute path (currentQualityWeights() → cache key
// + kickRemixPipeline) observes user-set defaults from first compute, even
// when the user never opens the Advanced window in this REAPER session. Pre-
// fix this read lived inside setupAdvancedPanel() which only ran on first
// Advanced open — leaving every pre-open compute on compile-time defaults.
void MainComponent::loadAdvancedDefaultsFromExtState()
{
    if (! GetExtState) return;
    const char* raw = GetExtState ("reamix.me", "advanced_defaults");
    if (raw == nullptr || raw[0] == '\0') return;

    reamix::ui::DevCalibrationRecord rec;
    if (! reamix::ui::DevCalibrationStorage::fromJsonLine (
            juce::String::fromUTF8 (raw), rec)) return;

    currentDefaultWeights_ = rec.weights_raw;
    currentDefaultBeta_    = rec.block_assembly_beta;
}

// ───────────────────────────────────────────────────────────────────────────
// ADR-097 sesja 107 — AdvancedWeightsWindow lifecycle.
// ───────────────────────────────────────────────────────────────────────────

void MainComponent::setupAdvancedPanel()
{
    // Wire callbacks ONCE per panel lifetime. Called from the lazy-create
    // branch of onAdvancedToggled().
    if (! advancedPanel_) return;

    // ADR-087 STATUS UPDATE 1 (sesja 98) + ADR-097 sesja 107 — panel slider
    // change kicks a fresh remix at the current target so user hears the new
    // cost-weight tuning in real time. Mirrors kickAuditionRerun pattern.
    advancedPanel_->onAnyChanged = [this]
    {
        qualityWeightsByPathMode_[currentItemKey()][appMode_] = advancedPanel_->weights();
        if (analysisState_ != AnalysisState::Complete) return;
        if (currentSourcePath_.isEmpty()) return;

        PendingRemixOp op;
        op.targetSec = durationPanel_.getTarget();
        if (auto bIt = blockedBySource_.find (currentSourcePath_); bIt != blockedBySource_.end())
            op.blockedTransitions = bIt->second;
        if (currentRegion_.has_value())
        {
            op.regionStartSec = currentRegion_->startSec;
            op.regionEndSec   = currentRegion_->endSec;
        }
        op.variation = 0;
        kickRemixPipeline (op);
    };

    advancedPanel_->onRestoreDefault = [this]
    {
        // DEV-079 sesja 101 — Restore default applies only to currently-active
        // mode. Other modes' calibration retained (each tab is its own
        // workspace). If the (item, mode) entry exists, drop it; if the item
        // map is now empty, drop the item entry too.
        auto pmIt = qualityWeightsByPathMode_.find (currentItemKey());
        if (pmIt != qualityWeightsByPathMode_.end())
        {
            pmIt->second.erase (appMode_);
            if (pmIt->second.empty())
                qualityWeightsByPathMode_.erase (pmIt);
        }
    };

    advancedPanel_->onPreferredHeightChanged = [this]
    {
        if (advancedWindow_) advancedWindow_->refitToPanel();
    };

    advancedPanel_->getCurrentMode = [this] () -> juce::String
    {
        switch (appMode_)
        {
            case reamix::ui::ModeTabs::Mode::Duration: return "duration";
            case reamix::ui::ModeTabs::Mode::Region:   return "region";
            case reamix::ui::ModeTabs::Mode::Blocks:   return "blocks";
        }
        return "unknown";
    };

    // ADR-097 sesja 107 iter-10 — "Set as defaults" persistence. Panel fires
    // the callback after user confirms; we serialize via the same JSON shape
    // DevCalibrationStorage uses (weights_raw + block_assembly_beta fields)
    // and store under ExtState "reamix.me" / "advanced_defaults".
    // DEV-080 sesja 108 — also mirror into host-side currentDefaultWeights_ +
    // currentDefaultBeta_ so the compute path picks up the new defaults
    // immediately (no need to wait for next REAPER restart + ctor reload).
    advancedPanel_->onSetAsDefaultsConfirmed =
        [this] (reamix::remix::QualityWeights w, reamix::ui::BlockAssemblyBeta b)
    {
        currentDefaultWeights_ = w;
        currentDefaultBeta_    = b;
        if (! SetExtState) return;
        reamix::ui::DevCalibrationRecord rec;
        rec.weights_raw         = w;
        rec.block_assembly_beta = b;
        rec.timestamp           = reamix::ui::DevCalibrationStorage::nowIsoUtc();
        rec.plugin_version      = REAMIX_VERSION;
        const auto line = reamix::ui::DevCalibrationStorage::toJsonLine (rec);
        SetExtState ("reamix.me", "advanced_defaults", line.toRawUTF8(), true);
    };

    // DEV-080 sesja 108 — push user-set defaults (loaded at ctor via
    // loadAdvancedDefaultsFromExtState) into the panel so its tick markers,
    // double-click-return values, and "Restore defaults" target reflect the
    // user's persisted choice. Pre-fix this read lived inline here and ran
    // only on first Advanced open; the ctor-time load is now the canonical
    // source of truth and the panel just mirrors it.
    advancedPanel_->setDefaultsForTicks (currentDefaultWeights_, currentDefaultBeta_);

    advancedPanel_->setPluginVersion (REAMIX_VERSION);
}

void MainComponent::syncAdvancedPanelFromState()
{
    if (! advancedPanel_) return;
    advancedPanel_->setRawWeights (currentQualityWeights());

    bool itemHasOverride = false;
    if (auto pmIt = qualityWeightsByPathMode_.find (currentItemKey());
        pmIt != qualityWeightsByPathMode_.end())
        itemHasOverride = ! pmIt->second.empty();
    advancedPanel_->setBadgeState (
        itemHasOverride
          ? reamix::ui::AdvancedWeightsPanel::BadgeState::Modified
          : reamix::ui::AdvancedWeightsPanel::BadgeState::NoSave);

    switch (appMode_)
    {
        case reamix::ui::ModeTabs::Mode::Duration:
            advancedPanel_->setAppMode (reamix::ui::AdvancedWeightsPanel::AppMode::Duration);
            break;
        case reamix::ui::ModeTabs::Mode::Region:
            advancedPanel_->setAppMode (reamix::ui::AdvancedWeightsPanel::AppMode::Region);
            break;
        case reamix::ui::ModeTabs::Mode::Blocks:
            advancedPanel_->setAppMode (reamix::ui::AdvancedWeightsPanel::AppMode::Blocks);
            break;
    }

    advancedPanel_->setTrackContext (currentSourcePath_, juce::String(),
                                      0.0, 0.0);
}

void MainComponent::onAdvancedToggled()
{
    if (advancedWindow_ && advancedWindow_->isVisible())
    {
        // Hide + persist + destroy. Panel state preserved (will be re-shown
        // with same slider positions if user reopens).
        persistAdvancedWindowState();
        advancedWindow_->setLookAndFeel (nullptr);
        advancedWindow_->setVisible (false);
        advancedWindow_.reset();
        return;
    }

    // Lazy-create panel on first open.
    if (! advancedPanel_)
    {
        advancedPanel_ = std::make_unique<reamix::ui::AdvancedWeightsPanel>();
        // Inherit the same custom LAF the main shell uses — without this the
        // panel renders with JUCE's default LAF (sliders without proper track
        // + default button skin) since the host DocumentWindow lives outside
        // MainComponent's child hierarchy.
        advancedPanel_->setLookAndFeel (&lookAndFeel_);
        setupAdvancedPanel();
    }
    syncAdvancedPanelFromState();

    // Create window hosting the panel.
    advancedWindow_ = std::make_unique<reamix::ui::AdvancedWeightsWindow> (advancedPanel_.get());
    // Iter-5 — apply reamix LAF to the window itself so its child
    // TransparentTooltipWindow inherits our drawTooltip override (with the
    // alpha-blended Bg1). Without this the tooltip resolves LAF via
    // DocumentWindow's default-V4 chain and renders opaque.
    advancedWindow_->setLookAndFeel (&lookAndFeel_);
    advancedWindow_->onCloseRequested = [this]
    {
        // User closed via X / Esc. Persist state, hide, destroy. Defer the
        // destruction one cycle so we don't unwind from inside DocumentWindow's
        // own close handler.
        juce::MessageManager::callAsync ([this]
        {
            if (! advancedWindow_) return;
            persistAdvancedWindowState();
            advancedWindow_->setLookAndFeel (nullptr);
            advancedWindow_->setVisible (false);
            advancedWindow_.reset();
        });
    };

    // Sesja 108 — Advanced window position FOLLOWS the main reamix.me window
    // on every open. User feedback: "okno advanced powinno wiedziec gdzie
    // jest okno main, niezaleznie czy jest na innym ekranie czy floating
    // musi sie otwierac tuz obok main okna by nie zaslaniac waveform."
    //
    // Strategy: pick side (right / left of main) with more room on the same
    // display main currently lives on; align top with main. Persisted size
    // is unused because AdvancedWeightsWindow setResizable(false) keeps
    // dimensions constant (default width + panel preferred height).
    const auto mainScreen   = getScreenBounds();
    const auto& displays    = juce::Desktop::getInstance().getDisplays();
    const auto* display     = displays.getDisplayForRect (mainScreen);
    const auto displayArea  = display != nullptr
                              ? display->userArea
                              : displays.getPrimaryDisplay()->userArea;

    const int aw  = advancedWindow_->getWidth();
    const int ah  = advancedWindow_->getHeight();
    const int gap = 10;

    const int xRight = mainScreen.getRight() + gap;
    const int xLeft  = mainScreen.getX() - gap - aw;

    int finalX;
    if (xRight + aw <= displayArea.getRight())
        finalX = xRight;                                  // room on right
    else if (xLeft >= displayArea.getX())
        finalX = xLeft;                                   // room on left
    else                                                  // neither side fits — clamp
        finalX = juce::jmax (displayArea.getX(),
                             displayArea.getRight() - aw);

    int finalY = mainScreen.getY();
    finalY = juce::jmax (displayArea.getY(),
                         juce::jmin (finalY, displayArea.getBottom() - ah));

    advancedWindow_->setBounds (finalX, finalY, aw, ah);

    advancedWindow_->setVisible (true);
    advancedWindow_->toFront (true);

    if (SetExtState)
        SetExtState ("reamix.me", "advanced_open", "1", true);
}

void MainComponent::persistAdvancedWindowState()
{
    if (! SetExtState) return;
    // Sesja 108 — position is no longer persisted because the Advanced window
    // re-computes placement relative to main on every open (follows main
    // across displays / floating / docked). Only the open/closed flag is
    // persisted so restoreAdvancedWindowOnLaunch picks up the state across
    // REAPER restart.
    SetExtState ("reamix.me", "advanced_open", "0", true);
}

void MainComponent::restoreAdvancedWindowOnLaunch()
{
    if (! GetExtState) return;
    const char* raw = GetExtState ("reamix.me", "advanced_open");
    if (raw == nullptr) return;
    if (juce::String::fromUTF8 (raw) != "1") return;

    // Same code path as user click — lazy-create panel + window, restore
    // persisted bounds, show.
    onAdvancedToggled();
}

// ── Sesja 111 KROK 3+4 — startup-time UX modals ─────────────────────────

void MainComponent::showWelcomeIfFirstLaunch()
{
    if (welcomeShown_) return;
    welcomeShown_ = true;

    // Skip if user has dismissed before (persisted in REAPER ExtState).
    if (GetExtState)
    {
        const char* raw = GetExtState ("reamix.me", "welcome_shown");
        if (raw != nullptr && raw[0] == '1')
            return;
    }

    // Mark seen immediately so an abrupt REAPER close still suppresses
    // re-show on the next launch.
    if (SetExtState)
        SetExtState ("reamix.me", "welcome_shown", "1", true);

    juce::String body = juce::String::fromUTF8 (
        "Three remix modes:\n"
        "  \xe2\x80\xa2 Duration \xe2\x80\x94 retarget any track to any length.\n"
        "  \xe2\x80\xa2 Region \xe2\x80\x94 retarget only the selected time range.\n"
        "  \xe2\x80\xa2 Blocks \xe2\x80\x94 arrange sections manually like a remix DJ.\n\n"
        "First Analyze on each track downloads a one-time 80 MB AI model.\n"
        "Subsequent analyses on the same track are instant.\n\n");

   #if JUCE_MAC
    body += juce::String::fromUTF8 (
        "macOS \xe2\x80\x94 if the plugin is blocked on first load:\n"
        "  1. Open System Settings \xe2\x86\x92 Privacy & Security.\n"
        "  2. Scroll to the Security section.\n"
        "  3. Click \xe2\x80\x9cOpen Anyway\xe2\x80\x9d next to reamix.me.");
   #elif JUCE_LINUX
    body += juce::String::fromUTF8 (
        "Linux \xe2\x80\x94 if model download fails, install:\n"
        "  sudo apt install libcurl4    (Debian/Ubuntu)\n"
        "  sudo dnf install libcurl     (Fedora)");
   #elif JUCE_WINDOWS
    body += juce::String::fromUTF8 (
        "Windows \xe2\x80\x94 no additional setup required.\n"
        "Click Analyze to begin.");
   #endif

    juce::AlertWindow::showMessageBoxAsync (
        juce::MessageBoxIconType::InfoIcon,
        juce::String::fromUTF8 ("Welcome to reamix.me"),
        body,
        "OK",
        this);
}

namespace
{
    struct VersionTriplet { int major{0}, minor{0}, patch{0}; };

    std::optional<VersionTriplet> parseVersionTag (const juce::String& tag)
    {
        // Accept "v1.0.2" or "1.0.2".
        auto trimmed = tag.startsWithIgnoreCase ("v") ? tag.substring (1) : tag;
        auto parts = juce::StringArray::fromTokens (trimmed, ".", "");
        if (parts.size() < 3) return std::nullopt;
        VersionTriplet v;
        v.major = parts[0].getIntValue();
        v.minor = parts[1].getIntValue();
        v.patch = parts[2].getIntValue();
        return v;
    }

    bool isStrictlyNewer (const VersionTriplet& latest, const VersionTriplet& current)
    {
        if (latest.major != current.major) return latest.major > current.major;
        if (latest.minor != current.minor) return latest.minor > current.minor;
        return latest.patch > current.patch;
    }
}

void MainComponent::checkForUpdatesAsync()
{
    // Capture current version literal at compile time so a future bump in
    // StatusBar version_ + a future tag both flow through one comparison.
    // Keep this string in lockstep with StatusBar.h::version_ — see memory
    // feedback_release_tag_bump_ui_version_lockstep.md.
    constexpr const char* kCurrentVersion = "v1.0.2";

    std::thread ([this]
    {
        if (updateCheckAborted_.load()) return;

        juce::URL url ("https://api.github.com/repos/b451c/reamix.me/releases/latest");
        auto stream = url.createInputStream (
            juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                .withConnectionTimeoutMs (10000));

        if (! stream || updateCheckAborted_.load()) return;

        const auto json = stream->readEntireStreamAsString();
        if (updateCheckAborted_.load() || json.isEmpty()) return;

        const auto parsed = juce::JSON::parse (json);
        if (! parsed.isObject()) return;

        const auto tagName = parsed.getProperty ("tag_name", {}).toString();
        if (tagName.isEmpty()) return;

        const auto latest  = parseVersionTag (tagName);
        const auto current = parseVersionTag (kCurrentVersion);
        if (! latest || ! current) return;

        if (! isStrictlyNewer (*latest, *current)) return;

        if (updateCheckAborted_.load()) return;

        juce::MessageManager::callAsync ([this, tagName]
        {
            if (updateCheckAborted_.load()) return;
            showUpdateAvailableModal (tagName);
        });
    }).detach();
}

void MainComponent::showUpdateAvailableModal (const juce::String& latestTag)
{
    if (updateAvailableShown_) return;
    updateAvailableShown_ = true;

    const juce::String body = juce::String::fromUTF8 (
        "A newer version of reamix.me is available.\n\n"
        "  Current: v1.0.2\n"
        "  Latest:  ") + latestTag + juce::String::fromUTF8 (
        "\n\n"
        "Update via:\n"
        "  \xe2\x80\xa2 ReaPack \xe2\x80\x94 Extensions \xe2\x80\x92 ReaPack \xe2\x80\x92 Synchronize packages\n"
        "  \xe2\x80\xa2 Manual download from the Releases page");

    juce::AlertWindow::showOkCancelBox (
        juce::AlertWindow::InfoIcon,
        juce::String::fromUTF8 ("reamix.me \xe2\x80\x94 Update available"),
        body,
        "Open Releases",
        "Later",
        this,
        juce::ModalCallbackFunction::create ([] (int result)
        {
            if (result == 1) // "Open Releases" button (index 1, "OK"-slot)
                juce::URL ("https://github.com/b451c/reamix.me/releases/latest")
                    .launchInDefaultBrowser();
        }));
}

void MainComponent::showStartupErrorWithPlatformHint (const juce::String& err)
{
    if (startupErrorShown_) return;
    startupErrorShown_ = true;

    juce::String hint;
   #if JUCE_MAC
    hint = juce::String::fromUTF8 (
        "\n\nCheck:\n"
        "  \xe2\x80\xa2 Network reachable (model downloaded from GitHub).\n"
        "  \xe2\x80\xa2 macOS Privacy & Security has not blocked reamix.me\n"
        "    (System Settings \xe2\x86\x92 Privacy & Security \xe2\x86\x92 \xe2\x80\x9cOpen Anyway\xe2\x80\x9d).");
   #elif JUCE_LINUX
    hint = juce::String::fromUTF8 (
        "\n\nCheck:\n"
        "  \xe2\x80\xa2 libcurl installed: sudo apt install libcurl4\n"
        "  \xe2\x80\xa2 Network reachable (model downloaded from GitHub).");
   #elif JUCE_WINDOWS
    hint = juce::String::fromUTF8 (
        "\n\nCheck:\n"
        "  \xe2\x80\xa2 Network reachable (model downloaded from GitHub).\n"
        "  \xe2\x80\xa2 Windows Defender / antivirus not blocking REAPER.");
   #endif

    juce::AlertWindow::showMessageBoxAsync (
        juce::MessageBoxIconType::WarningIcon,
        juce::String::fromUTF8 ("reamix.me \xe2\x80\x94 Model setup failed"),
        err + hint,
        "OK",
        this);
}
