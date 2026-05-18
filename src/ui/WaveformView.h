#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "Theme.h"
#include "UserBlock.h"
#include "CustomKindRegistry.h"
#include "render/Renderer.h" // EditPlan for Remix-variant tile-list (ADR-045 (c))

// WaveformView — phase-6 step 2 (ADR-036 D3 row 8 / D7 step 2).
//
// Renders the source-variant waveform well: 18h ruler + flex canvas + 20h
// bottom band. The component pulls peaks through a PeaksProvider it does
// not own (REAPER PCM_source arrives in step 9; step 2 uses a synthetic
// provider fed by MainComponent).
//
// ADR-045 (session 51 design / session 54 implementation): mockup deviation —
//   - 22h toolbar removed entirely (chip BEATS moved to SettingsPopover
//     DISPLAY section; chip SEGMENTS deleted, ADR-044 made it a no-op).
//   - Source variant gains the 18h top ruler (mockup had it remix-only at
//     waveform-blocks.jsx:91).
//   - Source bottom 20h band: empty in auto modes with subtle 1 px Fg3 25 %
//     downbeat anchor ticks (forward-looking manual-edit affordance);
//     colored cells only when segments_ is non-empty (Block Assembly mode,
//     phase-6 step 8).
//
// Layout + paint-layer definitions map to the mockup:
//   - plugin.css:287-364 (.rx-wave-well / .rx-wave-ruler / .rx-wave-canvas /
//     .rx-wave-beatline / .rx-wave-hover / .rx-seg-bar / .rx-seg-cell).
//   - primitives.jsx:117-197 (RxWave + RxRuler + RxSegmentBar).
//   - primitives.jsx:107-115 (rxPeakAt — used only by the synthetic
//     provider, not by this class).
//
// Mechanics ported (ADR-036 D1 JUCE-mechanics reference):
//   - Peak cache-key pattern (startSec / endSec / widthPx / revision)
//     from EditView/waveform_rendering.cpp:25-130.
//   - timeToX / xToTime helper pattern from REABeat WaveformView.cpp:127-137.
//   - Adaptive "nice interval" ruler ticks, 80-px target — REABeat
//     WaveformView.cpp:879 and EditView/rendering.cpp:432 both use this.
//   - Hover scrubber: single-pixel drawVerticalLine + partial-rect repaint
//     (one divergence from REABeat which repaints the whole component).
//
// Step 2 does NOT render: splice markers, playhead, remix variant, region
// overlay, zoom/pan. Those land at step 7 (remix + playhead + markers) and
// step 6 (region overlay). The juce::Timer base is declared but dormant
// here — activated for the playhead at step 7.

namespace reamix::ui
{

// Peaks provider — caller-owned, nullable.
class PeaksProvider
{
public:
    virtual ~PeaksProvider() = default;

    virtual double      getTotalDurationSeconds() const = 0;

    // Fill nPx bins from [startSec, endSec] with min/max sample amplitude
    // per bin (mono-summed, range [-1, 1]). Implementations must treat
    // out-of-range requests gracefully (empty bins = 0, 0).
    virtual void        getPeakBlock (double startSec, double endSec, int nPx,
                                      float* minOut, float* maxOut) = 0;

    // Bumped when the underlying source changes; WaveformView re-reads
    // peaks on revision change even if (startSec, endSec, widthPx) match.
    virtual std::int64_t getRevision() const { return 0; }
};

class WaveformView : public juce::Component,
                     private juce::Timer
{
public:
    enum class Variant { Source, Remix };

    // Quality color buckets per Lua waveform_remix.lua:283-292:
    //   q > 0.7 → Good (green); q ≥ 0.5 → Medium (warn yellow); q < 0.5 → Bad (red).
    enum class SpliceQuality { Good, Medium, Bad };

    // Sesja 60 — drag-select snap target (region mode UX). Off = swobodny
    // drag (default, also for Duration preview-selection); Beats / Downbeats
    // = mouseUp endpoints clamped to nearest beat or downbeat in the analysis.
    enum class SnapMode { Off, Beats, Downbeats };

    struct Segment        { double start; double end; reamix::theme::SegmentKind kind; };
    struct Beat           { double time;  bool   isDownbeat; };
    struct SelectionRange { double startSec; double endSec; };

    // SpliceMarker — one entry per transition in the remix path (session 57).
    // Visual: 1 px quality-colored line full-canvas-height + 9×7 downward
    // triangle at canvas top (mockup plugin.css:400-416 .rx-splice).
    // Tooltip data: title "from → to", quality %, ΔEnergy dB.
    struct SpliceMarker
    {
        double         timeSec;          // position on remix timeline
        SpliceQuality  quality;          // bucket for color
        float          qualityScore;     // 0..1 for tooltip percent
        int            fromBeat;
        int            toBeat;
        juce::String   fromLabel;        // "" in auto modes (ADR-044)
        juce::String   toLabel;
        float          energyDiffDb;
    };

    // Stacked region heights (plugin.css + waveform-blocks.jsx:57-90;
    // ADR-045 — 22h toolbar removed, ruler now applies to both variants).
    //
    //   Source:  18h ruler + flex canvas + 20h bottom band
    //   Remix:   18h ruler + flex canvas (+ 20h tile-list — step 7)
    //
    // Source bottom band paints downbeat anchor ticks when segments_ is
    // empty (auto modes, post-ADR-044) and colored cells when segments_ is
    // non-empty (Block Assembly mode, phase-6 step 8).
    static constexpr int kRulerHeight   = 18; // .rx-wave-ruler (both variants)
    static constexpr int kSegBarHeight  = 20; // .rx-seg-bar    (source only)
    static constexpr int kMinCanvasHeight = 80;
    static constexpr int kMinHeight = kRulerHeight + kMinCanvasHeight + kSegBarHeight;

    explicit WaveformView (Variant = Variant::Source);
    ~WaveformView() override;

    // Mutable variant — flipped to Remix from MainComponent::applyRemixToUi
    // post-RemixPipeline completion, back to Source on item switch / empty.
    // Resets cache + hover/drag state and triggers a full repaint.
    void setVariant         (Variant);
    Variant getVariant      () const noexcept { return variant_; }

    // Setters — caller keeps ownership of provider; vectors are copied.
    void setPeaksProvider   (PeaksProvider*);

    // Sesja 100b iter 2 — variant that preserves the current zoom / pan
    // state when swapping providers. Use this on Blocks-mode mutations
    // that invalidate the assembled remix (kind change, queue reorder,
    // boundary drag, etc.) — the user is still iterating on the same
    // source visually and expects their zoom to stay put. Default
    // setPeaksProvider above keeps its view-reset semantic for the
    // item-switch path which legitimately needs a fresh view.
    void setPeaksProviderPreserveView (PeaksProvider*);
    void setSegments        (std::vector<Segment>);
    void setBeats           (std::vector<Beat>);

    // Remix-variant data (session 57). Empty vector / empty plan = no draw.
    void setSpliceMarkers   (std::vector<SpliceMarker>);
    void setEditPlan        (reamix::render::EditPlan);
    // ADR-045 — sole UI control is the SettingsPopover DISPLAY "Show beats"
    // toggle; default ON (changed from session-50 default OFF).
    void setShowBeats       (bool);
    bool getShowBeats       () const noexcept { return showBeats_; }

    // DEV-022 / ADR-041 — gates the canvas between "loaded but not yet
    // analyzed" (faint silhouette + overlay text) and "analyzed"
    // (full-opacity peaks, no overlay). MainComponent flips true from
    // applyAnalysisToUi (fresh + cache-hit) and false from applySelectedItem
    // cache-miss / applyEmptyState / onFastToggled invalidate.
    void setAnalyzed        (bool);

    // DEV-022 / ADR-041 (session 49 extension) — three-state overlay system:
    //   hasSource_ = false                         → empty-state brand splash
    //                                                (big dot + reamix.me + headline)
    //   hasSource_ = true, hasAnalysis_ = false    → faint silhouette + pill CTA
    //                                                ("Click Analyze to remix")
    //   hasSource_ = true, hasAnalysis_ = true     → full-opacity peaks, no overlay
    // MainComponent flips hasSource_ true from applySelectedItem / false from
    // applyEmptyState.
    void setHasSource       (bool);

    // Step-4 PreviewController wires playhead here (ADR-036 D7 step 4).
    // std::nullopt hides the line; any value renders a 1 px accent-color
    // vertical across the canvas region at timeToX(sec). Called from
    // MainComponent::timerCallback on 100 ms tick with the preview
    // controller's current position, or std::nullopt when playback stops.
    void setPlayhead        (std::optional<double> seconds);

    // Flash ring feedback at a time position (250 ms expanding ellipse,
    // fade-out). Triggered internally on click-to-seek; MainComponent may
    // also call after an external seek. REABeat pattern, WaveformView.cpp:935.
    void triggerFlash       (double seconds);

    // Emitted on mouseUp when the gesture was a click (drag distance
    // < kDragThresholdPx) inside canvas / segment bar. Receiver maps to
    // seek-source or seek-remix based on variant (step 7).
    std::function<void (double seconds)> onSeek;

    // Splice-marker callbacks (Remix variant). Left-click on marker fires
    // onSpliceClick (auditions ±1 s around the seam in MainComponent).
    // Right-click fires onSpliceContextMenu (4-action ContextMenu).
    // Hover transitions (-1 ↔ idx) fire onSpliceHoverChanged so MainComponent
    // can show/hide the rich Tooltip. screenPos is in screen coordinates
    // (already converted via localPointToScreen) for direct showAt /
    // PopupMenu::showMenuAsync placement.
    std::function<void (int idx, double remixTimeSec)>            onSpliceClick;
    std::function<void (int idx, juce::Point<int> screenPos)>     onSpliceContextMenu;
    std::function<void (int idx, juce::Point<int> screenPos)>     onSpliceHoverChanged; // idx == -1 → hide tooltip

    // Emitted on mouseUp when the gesture was a drag-select, and on
    // clearSelection() — with std::nullopt in the second case. Receiver
    // caches the range for play-selection dispatch in TransportBar
    // onPlayStop (ADR-039 MVP).
    std::function<void (std::optional<SelectionRange>)> onSelectionChanged;

    // Selection accessors. Manually-set ranges do NOT fire
    // onSelectionChanged — the callback is drag-initiated only, to avoid
    // feedback loops when MainComponent writes state back.
    void                          setSelection   (std::optional<SelectionRange>);

    void     setSnapMode (SnapMode m);
    SnapMode getSnapMode() const noexcept { return snapMode_; }

    // Sesja 60 — overlay button "↺ Edit region" shown on the canvas top-left
    // corner when set. MainComponent toggles ON after Region remix completes
    // (variant flips to Remix, selection cleared) and OFF when the user clicks
    // the button (which triggers `onEditRegionClicked`).
    void setShowEditRegionButton (bool);
    std::function<void()> onEditRegionClicked;

    // Sesja 61 hot-fix — analogous overlay button "↺ Edit arrangement" shown
    // on the canvas in Blocks mode + Remix variant. Click → MainComponent
    // flips back to Source variant so user can re-edit blocks/queue without
    // losing the arrangement state. Premium-feel: same visual language as
    // Region's Edit-region button, different label / callback.
    void setShowEditArrangementButton (bool);
    std::function<void()> onEditArrangementClicked;

    // Sesja 100 (DEV-018, ADR-091 agent-side design) — Preview volume
    // overlay icon in the bottom-RIGHT corner of canvas (sesja 100 iter 2 —
    // moved from top-right so the expanded slider doesn't overlay the ruler).
    // Always visible; clicking toggles a floating vertical slider hosted as
    // a direct child component (no CallOutBox bubble per sesja 100 iter 3
    // user directive: "wystarczyłby sam suwak głośności bez tej ramki").
    //
    // Discreet but functional per user directive sesja 100: same Bg1.alpha
    // 0.90 backdrop + Sm/500 font as Edit overlay, smaller width (icon +
    // 3-digit %); clicking opens the slider rather than dragging in-place.
    //
    // onPreviewVolumeChanged fires every time the user drags the inline
    // slider; MainComponent persists via SetExtState + applies to
    // PreviewController. State source-of-truth lives in MainComponent;
    // WaveformView mirrors only the displayed percentage via setPreviewVolume.
    void setPreviewVolume (double linear01);
    double getPreviewVolume() const noexcept { return previewVolume_; }
    std::function<void(double)> onPreviewVolumeChanged;

    void                          clearSelection ();

    // ── Block Assembly mode (sesja 61, ADR-051) ───────────────────────
    // Enables drag-on-segBar marking. When true:
    //   - drag on segBar empty area → new block mark; emits onMarkBlock(start,
    //     end, screenAnchor) on mouseUp (snapped by snapMode_).
    //   - click on existing user block in segBar → emits onUserBlockClicked(idx).
    //   - segBar paints userBlocks_ as kind-colored tiles (Source variant only;
    //     downbeat anchors still paint when blocks list is empty).
    // When false: segBar mouse handling is unchanged (no marking, no clicks).
    void setBlockMarkingEnabled (bool);
    void setUserBlocks (std::vector<reamix::ui::UserBlock>);

    // ADR-092 sesja 100c — registry consulted at paint time for custom kindy
    // (name + color overrides). Stored as raw pointer so the picker / paint
    // sites read latest state when the user mutates registry mid-session.
    // Caller (MainComponent) owns lifetime; pointer must outlive WaveformView.
    void setCustomKindRegistry (const reamix::ui::CustomKindRegistry* registry) noexcept
    {
        customKindRegistry_ = registry;
        repaint();
    }

    // Emitted on segBar drag-up in Block Assembly mode. (start,end) are in
    // item-relative seconds, snapped per snapMode_. screenAnchorPos is the
    // mouse-up position in screen coords (for CallOutBox anchor).
    std::function<void (double startSec, double endSec,
                        juce::Point<int> screenAnchorPos)> onMarkBlock;

    // Emitted on segBar single-click in Block Assembly mode when the click
    // hits an existing user block (otherwise no-op). idx indexes userBlocks_
    // in display order (sorted by startSec). Phase H wires this to PaletteSourceEdit.
    std::function<void (int blockIdx)> onUserBlockClicked;

    // DEV-029 (d) sesja 100b — boundary-drag finished. Emitted on mouseUp
    // after the user dragged a block boundary on the segBar. (newStartSec,
    // newEndSec) are the resulting bounds; the host is expected to mutate
    // userBlocks_[blockIdx], persist, and refresh the UI. Boundary-drag
    // is enabled only when the cursor lands within kBoundaryHitPx of an
    // existing block edge on mouseDown.
    std::function<void (int blockIdx, double newStartSec, double newEndSec)>
        onUserBlockBoundariesChanged;

    // ── Soft-boundary visualization (sesja 61, ADR-051 § D3) ──────────
    // Algorithm-resolved splice points. Each entry is one junction between
    // adjacent queue tiles, expressed as the source-side time where the
    // splice OUT happens (algorithm chose to leave the source at this time).
    // MainComponent populates after RemixPipeline completes for Blocks mode.
    struct UserBlockSplice
    {
        double sourceTimeSec;     // where on the source the splice happens
        double driftBeats;        // |splice - user-authored boundary| in beats
        float  qualityScore;      // 0..1 — used for tooltip percent + line tint
        int    leftBlockIdx;      // userBlocks_ index of the block exiting
        int    rightBlockIdx;     // userBlocks_ index of the block entering
    };
    void setUserBlockSplices (std::vector<UserBlockSplice>);

    // Hover tooltip data callback (idx into userBlockSplices_; -1 = hide).
    // MainComponent owns the Tooltip component; this callback hands off the
    // hovered splice for tooltip composition.
    std::function<void (int spliceIdx, juce::Point<int> screenPos)> onUserBlockSpliceHover;
    std::optional<SelectionRange> getSelection   () const { return selection_; }

    void paint          (juce::Graphics&)         override;
    void resized        ()                        override;
    void mouseMove      (const juce::MouseEvent&) override;
    void mouseExit      (const juce::MouseEvent&) override;
    void mouseDown      (const juce::MouseEvent&) override;
    void mouseDrag      (const juce::MouseEvent&) override;
    void mouseUp        (const juce::MouseEvent&) override;
    void mouseWheelMove (const juce::MouseEvent&,
                         const juce::MouseWheelDetails&) override;
    void mouseMagnify   (const juce::MouseEvent&, float scaleFactor) override;
    bool keyPressed     (const juce::KeyPress&)   override;

private:
    // Timer drives the 250 ms flash ring animation (~60 Hz ticks). Step-7
    // playhead animation may also hook in here. Otherwise dormant.
    void timerCallback() override;

    // Region rects (local coords). Empty rect for disabled region.
    //   segBar  — Source variant only.
    //   tileList — Remix variant only.
    juce::Rectangle<int> rulerArea()    const;
    juce::Rectangle<int> canvasArea()   const;
    juce::Rectangle<int> segBarArea()   const;
    juce::Rectangle<int> tileListArea() const;

    // Full-width waveform coord transforms. Valid once viewDurationSec_ > 0.
    float  timeToX (double seconds, int regionX, int regionWidth) const;
    double xToTime (float px,       int regionX, int regionWidth) const;

    // Paint stages (order: bg → canvas layers → selection → segbar → ruler
    // → playhead → flash → hover). ADR-045: paintToolbar removed.
    void paintRuler         (juce::Graphics&);
    void paintCanvas        (juce::Graphics&);
    void paintSelection     (juce::Graphics&);
    void paintSegBar        (juce::Graphics&);
    // Remix variant: 1 px quality-color line full-canvas-height + 9×7
    // downward triangle per marker (mockup .rx-splice; plugin.css:400-416).
    // Hovered marker brightens by ×1.25 (CSS :hover filter).
    void paintSpliceMarkers (juce::Graphics&);
    // Remix variant: per-clip duration tiles in 20h bottom band, width ∝
    // clip.durationSec (ADR-045 § Decision (c)). Alternating gray fill matches
    // ADR-046 Insert color scheme for visual continuity with REAPER timeline.
    void paintTileList      (juce::Graphics&);
    // ADR-045 (a) A1+ — Source bottom 20h band when no user labels: 1 px
    // Fg3 25 % alpha vertical ticks at downbeat x-positions, hinting where
    // future "snap to downbeat" interactions will land.
    void paintDownbeatAnchors (juce::Graphics&, juce::Rectangle<int> bar);
    // ADR-051 — Block Assembly mode segBar layers. paintUserBlocks renders
    // existing user-marked sections as kind-colored tiles. paintMarkPreview
    // renders the provisional mark while the user is dragging on segBar
    // (real-time feedback before kind picker opens). paintUserBlockSplices
    // renders algorithm-resolved splice points as 2 px Accent vertical lines
    // (post-preview, sesja 61 phase E dual-layer).
    void paintUserBlocks        (juce::Graphics&, juce::Rectangle<int> bar);
    void paintMarkPreview       (juce::Graphics&, juce::Rectangle<int> bar);
    void paintUserBlockSplices  (juce::Graphics&, juce::Rectangle<int> bar);
    void paintPlayhead      (juce::Graphics&);
    void paintFlash         (juce::Graphics&);
    void paintHover         (juce::Graphics&);
    // DEV-022 / ADR-041 — unified overlay for empty-state AND pre-analyze
    // state. Big Accent dot + "reamix" / ".me" brand mark + headline. The
    // headline differs per state (empty = product tagline; pre-analyze =
    // CTA "Click Analyze…"). Called from paintCanvas when hasAnalysis_ or
    // hasSource_ is false — fully replaces the waveform visual (no peaks,
    // no beat grid, no segment tint, no centerline) because showing any of
    // those pre-analysis is misleading per user session 49 feedback.
    void paintSplashOverlay (juce::Graphics&, juce::Rectangle<int> canvas,
                              bool isEmpty);

    // Rebuilds peak cache if any key component changed. Called at the head
    // of paintCanvas().
    void ensurePeaks();

    // Dirty rect covering the hover scrubber line + label pill for a given
    // cursor x. Spans ruler-bottom → segbar-bottom vertically (ruler is
    // excluded — the scrubber does not cross it per plugin.css:326).
    juce::Rectangle<int> hoverDirtyRect (int cursorX) const;

    // View navigation (ADR-039 full-scope session 48). EditView pattern:
    //   waveform_view.cpp:677-708 — ZoomHorizontal + ScrollH with clamping.
    //   edit_view.cpp:1244-1270    — WM_MOUSEHWHEEL pan + WM_GESTURE pinch.
    // factor > 1 zooms IN (shorter view); centerTime stays under the cursor.
    // deltaTime shifts viewStartSec_ along the timeline (positive = right).
    void zoomHorizontal (double factor, double centerTime);
    void scrollH        (double deltaSec);

    // Ruler helpers.
    static double pickTickInterval (double pxPerSec);
    static juce::String formatSeconds       (double s); // "M:SS"
    static juce::String formatHoverSeconds  (double s); // "M:SS.T"

    // Splice-marker hit-test. Returns the index of the closest marker within
    // kSpliceHitToleranceX pixels of cursorX, or -1 if none.
    int  spliceHitTest    (int cursorX) const;
    void updateHoveredSplice (int newIdx, juce::Point<int> screenPos);

    // ADR-051 — snap a time t to nearest beat / downbeat per snapMode_.
    // Returns t unchanged when SnapMode::Off or beats_ empty. Pure-function
    // helper extracted from mouseUp drag-select flow (sesja 60) for reuse
    // by segBar marking flow.
    double snapTimeToBeats (double t) const;

    // ADR-051 — segBar hit-test for click-on-existing-user-block in Blocks
    // mode. Returns userBlocks_ index when (cursorX, cursorY) is inside any
    // block's painted rect, -1 otherwise. Used in mouseDown to disambiguate
    // "click existing block to edit" vs "drag to mark new block".
    int userBlockHitTest (juce::Point<int> pos) const;

    // DEV-029 (d) sesja 100b — boundary hit-test on segBar. Returns
    // userBlocks_ index when (cursorX, cursorY) is within kBoundaryHitPx
    // of any block's left or right edge; outIsStart receives true for
    // left edge, false for right edge. Returns -1 when no boundary in
    // range (priority for resize over click-to-edit).
    int userBlockBoundaryHitTest (juce::Point<int> pos, bool* outIsStart) const;

    // ADR-051 — splice-line hit-test for hover tooltip. Returns the index
    // into userBlockSplices_ when cursorX is within ±kSpliceHitToleranceX
    // of a splice line's painted x. -1 otherwise.
    int userBlockSpliceHitTest (int cursorX) const;

    struct PeaksCache
    {
        double startSec = 0.0;
        double endSec   = 0.0;
        int    widthBins = 0;
        std::int64_t revision = -1;
        std::vector<float> minBuf;
        std::vector<float> maxBuf;
        bool valid = false;
    };

    Variant              variant_; // mutable since session 57 (setVariant)

    PeaksProvider*       peaks_ = nullptr;
    std::vector<Segment> segments_;
    std::vector<Beat>    beats_;

    SnapMode             snapMode_ { SnapMode::Off };

    // Edit-region overlay button state (sesja 60 Plan A).
    bool                 showEditRegionButton_ { false };
    bool                 editRegionHover_      { false };
    juce::Rectangle<int> editRegionBtnBounds_  {};

    // Edit-arrangement overlay button state (sesja 61 hot-fix). Mutually
    // exclusive with the region overlay (different modes).
    bool                 showEditArrangementButton_ { false };
    bool                 editArrangementHover_      { false };
    juce::Rectangle<int> editArrangementBtnBounds_  {};

    // Preview volume overlay (sesja 100, DEV-018). Always visible on canvas
    // bottom-right corner; click toggles inline slider popup (no CallOutBox).
    // Discreet but functional.
    double               previewVolume_       { 1.0 };
    bool                 volumeHover_         { false };
    juce::Rectangle<int> volumeBtnBounds_     {};

    // Sesja 100 iter 3 (DEV-018) — inline vertical slider, child component
    // of WaveformView. Hidden until user clicks volumeBtnBounds_; clicking
    // anywhere outside the slider bounds dismisses it. Bypasses CallOutBox
    // entirely so there's no bubble/arrow background — just the bare track
    // + thumb floating over the waveform.
    juce::Slider         volumePopupSlider_;
    bool                 volumePopupVisible_  { false };

    void positionVolumePopup();
    void showVolumePopup();
    void hideVolumePopup();

    // Remix-variant data — empty in Source variant. Splice markers track
    // current path's transitions; editPlan_ feeds tile-list. Both reset
    // on setVariant(Source) and on setSpliceMarkers({}) / setEditPlan({}).
    std::vector<SpliceMarker>      spliceMarkers_;
    reamix::render::EditPlan       editPlan_;
    int                            hoveredSpliceIdx_ { -1 };
    static constexpr int kSpliceHitToleranceX = 5;
    std::optional<double> playheadSec_; // step-4 preview playhead (nullopt = hidden)
    // ADR-045 — default ON (was session-50 default OFF per mockup
    // plugin.jsx:22-23). Settings popover DISPLAY section is the sole UI
    // control. Segment tint is now data-driven (active iff segments_ is
    // non-empty, i.e. Block Assembly mode); no toggle.
    bool showBeats_ = true;

    // DEV-022 / ADR-041 — false until analysis completes. When false:
    //   canvas peaks render at kBarOpacityPreAnalyze (0.25) instead of the
    //   full 0.78; segment tint + beat grid are skipped (neither is
    //   meaningful without analysis results); a centred "Click Analyze
    //   to remix" label renders on top.
    bool hasAnalysis_     = false;
    // DEV-022 / ADR-041 (session 49 extension) — false when no REAPER item
    // is selected. Renders an empty-state brand splash (big Accent dot +
    // "reamix" / ".me" + headline) instead of the faint-silhouette pill.
    bool hasSource_       = false;

    double viewStartSec_    = 0.0;
    double viewDurationSec_ = 0.0;

    PeaksCache cache_;
    int cursorX_ = -1; // -1 = no hover

    // Drag-select state (ADR-039 MVP). dragging_ = true between a mouseDown
    // inside canvas/segbar and the matching mouseUp. dragStartSec_ is the
    // pinned anchor; dragCurrentSec_ tracks the moving edge. dragStartPos_
    // is in local pixel coords, used for the click-vs-drag threshold check
    // on mouseUp.
    bool dragging_           = false;
    juce::Point<int> dragStartPos_;
    double dragStartSec_     = 0.0;
    double dragCurrentSec_   = 0.0;
    static constexpr int kDragThresholdPx = 5;

    // Zoom / pan tuning (ADR-039 session 48). Min view duration 0.1 s is a
    // practical floor for selection work — FilePeaksProvider does not expose
    // the item sample rate so the EditView `32/sr` formula is not available
    // (~0.7 ms @ 44.1k); 100 ms is more than fine for our use case (user
    // wants to mark splice regions, not sample-edit). kZoomFactor matches
    // EditView ZOOM_FACTOR. kZoomDamping matches EditView pinch damping.
    // kPanPct matches EditView WM_MOUSEHWHEEL step (10% of view per delta).
    static constexpr double kMinViewDurationSec = 0.1;
    static constexpr double kZoomFactor         = 1.25;
    static constexpr double kZoomDamping        = 0.15;
    static constexpr double kPanPct             = 0.10;

    // Finalized selection (nullopt = no range). Painted as Accent-tinted
    // scrim across the canvas region.
    std::optional<SelectionRange> selection_;

    // DEV-031 sesja 100b — canvas selection edge-drag state. Mirrors
    // segBarBoundaryDragging_ but for the canvas selection_ scrim
    // (Duration play-selection / Region scrim). Started when mouseDown
    // lands within kSelectionEdgeHitPx of selection_->startSec or
    // selection_->endSec; updates only the dragged edge until mouseUp.
    bool   selectionEdgeDragging_  { false };
    bool   selectionEdgeIsStart_   { false };
    static constexpr int kSelectionEdgeHitPx = 5;

    // DEV-072 sesja 100b — auto-pan thresholds + rate. While drag is
    // active (any of dragging_ / segBarDragging_ / segBarBoundaryDragging_
    // / selectionEdgeDragging_), if the cursor sits within
    // kEdgePanThresholdPx of the canvas left/right edge, viewStartSec_
    // pans toward the cursor at a rate proportional to (threshold -
    // distance) / threshold. Component::beginDragAutoRepeat keeps
    // mouseDrag firing at ~25 Hz even when the cursor is held still so
    // the pan continues without manual cursor wobble.
    static constexpr int    kEdgePanThresholdPx = 20;
    static constexpr int    kAutoRepeatMs       = 40;
    static constexpr double kEdgePanViewFrac    = 1.5; // view fractions per second at edge

    // ── Block Assembly mode (sesja 61, ADR-051) ───────────────────────
    // Activated by setBlockMarkingEnabled(true) when MainComponent enters
    // appMode_::Blocks. Drives:
    //   - segBar mouse handling (drag-mark vs click-existing).
    //   - paintSegBar layering (user blocks tiles + provisional drag preview).
    bool                                blockMarkingEnabled_ { false };
    std::vector<reamix::ui::UserBlock>  userBlocks_;

    // ADR-092 sesja 100c — non-owning. Caller (MainComponent) owns lifetime.
    const reamix::ui::CustomKindRegistry* customKindRegistry_ { nullptr };

    // segBar drag state. Independent of dragging_ (canvas drag-select) so
    // the two gestures cannot interfere when blockMarkingEnabled_ is on.
    bool                segBarDragging_         { false };
    juce::Point<int>    segBarDragStartPos_;
    double              segBarDragStartSec_     { 0.0 };
    double              segBarDragCurrentSec_   { 0.0 };
    // True when the segBar drag started inside an existing user block —
    // suppresses onMarkBlock emission on mouseUp so we don't create a new
    // block overlapping an old one. Click-on-existing fires onUserBlockClicked
    // instead (handled in mouseUp click branch). Mid-drag-of-block fires
    // onUserBlockBoundariesChanged with translated bounds.
    bool                segBarDragStartedOnBlock_ { false };
    int                 segBarDragStartBlockIdx_  { -1 };

    // DEV-057 sesja 100d — mid-tile drag state. Activated when a drag started
    // on an existing block (segBarDragStartedOnBlock_) crosses the click-vs-
    // drag pixel threshold. Translates the entire block in time (preserving
    // duration); clamped by neighbour boundaries (or 0 / item duration on
    // edges). Commits via onUserBlockBoundariesChanged on mouseUp so the
    // host re-uses the existing sesja-100b boundary-drag persistence path.
    bool                segBarMidDragActive_       { false };
    int                 segBarMidDragBlockIdx_     { -1 };
    double              segBarMidDragOffsetSec_    { 0.0 };  // cursor - block.startSec at mouseDown
    double              segBarMidDragLiveStart_    { 0.0 };  // in-progress new startSec
    double              segBarMidDragOriginalDur_  { 0.0 };  // preserved across drag
    double              segBarMidDragClampMinStart_{ 0.0 };  // prev neighbour end (or 0)
    double              segBarMidDragClampMaxStart_{ 0.0 };  // (next neighbour start | itemDur) - dur

    // DEV-029 (d) sesja 100b — segBar boundary-drag state. Independent of
    // the new-block-mark flow above. boundaryDragging_ → mouseDown
    // detected the cursor within kBoundaryHitPx of an existing block edge;
    // mouseDrag updates the live preview boundary; mouseUp commits via
    // onUserBlockBoundariesChanged. boundaryIsStartEdge_ disambiguates
    // left edge (startSec) vs right edge (endSec) being dragged.
    // boundaryDragLiveSec_ holds the in-progress new boundary value.
    bool                segBarBoundaryDragging_   { false };
    int                 segBarBoundaryBlockIdx_   { -1 };
    bool                segBarBoundaryIsStart_    { false };
    double              segBarBoundaryLiveSec_    { 0.0 };
    double              segBarBoundaryFixedSec_   { 0.0 };  // the OTHER edge (clamp anchor)
    double              segBarBoundaryClampMin_   { 0.0 };  // prev neighbour's end (or 0)
    double              segBarBoundaryClampMax_   { 0.0 };  // next neighbour's start (or item dur)
    // Sesja 100b iter 2 — shared boundary drag. When two user blocks are
    // touching (A.endSec == B.startSec within kSharedBoundaryEps), grabbing
    // the join drags BOTH boundaries in lockstep — the join moves and the
    // block on each side resizes accordingly. Without this, user couldn't
    // pull block B back without first shrinking block A. -1 = single-edge
    // drag (the segBarBoundaryBlockIdx_ block alone).
    int                 segBarBoundarySharedBlockIdx_ { -1 };
    bool                segBarBoundarySharedIsStart_  { false };
    double              segBarBoundarySharedFixedSec_ { 0.0 };
    static constexpr double kSharedBoundaryEps = 1e-3;
    // Sesja 100b iter 2 — hovered user block (segBar). Drives a faint
    // brightness boost on the tile beneath the cursor so the user sees
    // which block they're about to manipulate (matters most when blocks
    // touch and the boundary is ambiguous to the eye).
    int                 hoveredUserBlockIdx_      { -1 };

    static constexpr int    kBoundaryHitPx        = 6;
    static constexpr double kMinBlockSec          = 0.5;

    // Algorithm-resolved splice points for current arrangement (sesja 61,
    // phase E). Empty in non-Blocks modes or before RemixPipeline runs.
    std::vector<UserBlockSplice>  userBlockSplices_;
    int                           hoveredSpliceUserIdx_ { -1 };

    // Flash ring state (ADR-039 full-scope session 48). Expanding 4→20 px
    // Accent ellipse, 250 ms fade. flashPosSec_ < 0 = inactive.
    double       flashPosSec_    = -1.0;
    std::uint32_t flashStartMs_  = 0;
    static constexpr int kFlashDurationMs = 250;

    // DEV-055 sesja 100c — pulsing alpha for the empty-segBar drag-mark hint
    // when blockMarkingEnabled_ && userBlocks_.empty(). Timer runs at ~40 Hz
    // while hint is visible; stops when first block marked.
    bool         segBarHintActive_ = false;
    static constexpr float kFlashStartR   = 4.0f;
    static constexpr float kFlashEndR     = 20.0f;

    // 2 px per silhouette bar, minus 0.5 gap — primitives.jsx:168-170.
    static constexpr int   kSampleEvery = 2;
    static constexpr float kBarOpacity  = 0.78f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (WaveformView)
};

} // namespace reamix::ui
