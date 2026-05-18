#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

// SettingsPopover — phase-6 step 5 (ADR-036 D3 row 13).
//
// Gear-anchored popover per shell.jsx:197-216 styling (bg-1 + LineStrong
// border + R3 radius + ElevPop shadow + 260w).
//
// Sections (top → bottom):
//   DISPLAY  (ADR-045)  — single "Show beats" toggle, default ON.
//   WINDOW   (ADR-043)  — single dock-toggle button (REABeat parity).
//
// Both sections expose their state via setters called from MainComponent at
// show() time, so the popover does not maintain its own truth — it reflects
// WaveformView's showBeats_ + the host window's dock state.
//
// Why no other sections (ADR-043 § Consequences gate):
//   - "Show segments" toggle: ADR-044 makes auto-mode segments empty (no
//     labels), and Block Assembly (phase-6 step 8) renders user labels by
//     mode — a toggle would have no useful values either way.
//   - Shortcuts list: TransportBar already renders Space/Enter/Esc as kbd
//     chips on the buttons themselves.
//   - Model-manager: deferred to phase-7 per ADR-043.
//   Any further section addition still requires its own ADR per ADR-043
//   § Consequences.
//
// Dismiss behaviour: enters modal state on show() so click-outside-body
// triggers inputAttemptWhenModal() → hide. Esc in keyPressed() also
// dismisses. Modal is non-blocking for paint/layout; only input is captured.

namespace reamix::ui
{

class SettingsPopover : public juce::Component
{
public:
    SettingsPopover();
    ~SettingsPopover() override;

    // Current dock state — paint flips the button label accordingly.
    // Caller (MainComponent) sets this at show() time from onIsDocked().
    void setDocked (bool);

    // ADR-045 — DISPLAY "Show beats" toggle state. Caller (MainComponent)
    // sets this at show() time from onIsShowBeats().
    void setShowBeats (bool);

    // Sesja 100b — DEV-049 INSERT section state. Both toggles default ON
    // per user smoke; MainComponent overrides at show() from persisted
    // ExtState. setInsertSpliceMarkers controls "Insert splice markers"
    // (project markers at every splice on Insert); setInsertRenderRegion
    // controls "Insert render region" (region spanning all new clips for
    // Region Render Matrix usage).
    void setInsertSpliceMarkers (bool);
    void setInsertRenderRegion  (bool);

    // Sesja 60 — DISPLAY "Snap region to" cycler. Click row cycles through
    // Off → Beats → Downbeats → Off. State 1:1 maps to WaveformView::SnapMode.
    enum class SnapRegion { Off, Beats, Downbeats };
    void setSnapRegion (SnapRegion);

    // Splice flexibility (Tight/Medium/Loose) cycler removed sesja 68
    // ADR-057. Geometric audit + Test 1+2 listening confirmed the
    // junction-anchored ±W search is parameterizing a wrong model;
    // β-model in DEV-040 supersedes.

    // Fired on dock-button click. SettingsPopover hides itself first,
    // then invokes the callback so the enclosing DockableWindow rebuild
    // does not run while a modal child component is live.
    std::function<void()> onDockToggled;

    // ADR-097 (sesja 107) — Advanced weights window. Popover hides itself
    // first, then invokes the callback so the host can open / toggle the
    // separate juce::DocumentWindow without modal interference. Default
    // hidden — only power users opening this window see the 8 raw weight
    // sliders + 4 β-controls.
    std::function<void()> onAdvancedToggled;

    // ADR-045 — fired on "Show beats" toggle click. Popover stays open so
    // the user can see the immediate effect on the waveform; one tap can
    // be undone with another tap without re-opening the gear.
    std::function<void(bool)> onShowBeatsToggled;

    // Sesja 60 — fired when user cycles the Snap region row. Receives the
    // new SnapRegion value.
    std::function<void(SnapRegion)> onSnapRegionChanged;

    // Sesja 100b — DEV-049 INSERT toggles. Popover stays open so the
    // user can flip both before closing (settings are usually paired).
    std::function<void(bool)> onInsertSpliceMarkersToggled;
    std::function<void(bool)> onInsertRenderRegionToggled;

    // ADR-053 (sesja 63 BUG-19 follow-up) — analysis disk-cache controls.
    // Caller (MainComponent) refreshes stats + wires the action handlers.
    void setCacheStats (juce::int64 totalBytes, int entryCount);
    std::function<void()> onClearCache;
    std::function<void()> onRevealCache;

    void show();
    void hideMe();
    bool isOpen() const noexcept { return isVisible(); }

    // juce::Component
    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;
    void inputAttemptWhenModal() override;

private:
    void layOutHitRegions();
    static int computeBodyHeight();

    bool docked_        = false;
    bool dockBtnHover_  = false;
    bool dockBtnPressed_= false;

    // ADR-045 — DISPLAY section state. Default ON matches WaveformView's
    // showBeats_ default; MainComponent overwrites at show() if the user
    // had toggled it earlier.
    bool showBeats_       = true;
    bool beatsRowHover_   = false;
    bool beatsRowPressed_ = false;

    // Sesja 60 — Snap region cycler state. Default Off (no snap).
    SnapRegion snapRegion_       = SnapRegion::Off;
    bool       snapRowHover_     = false;
    bool       snapRowPressed_   = false;

    // Sesja 100b — DEV-049 INSERT section state. Both default ON
    // (matches user smoke choice — feature should be visible from
    // first Insert without hunting for the toggle).
    bool insertSpliceMarkers_         = true;
    bool insertRenderRegion_          = true;
    bool insertSpliceRowHover_        = false;
    bool insertSpliceRowPressed_      = false;
    bool insertRegionRowHover_        = false;
    bool insertRegionRowPressed_      = false;

    juce::Rectangle<int> dockBtnRect_;
    juce::Rectangle<int> advancedBtnRect_;     // ADR-097 sesja 107
    bool                 advancedBtnHover_   = false;
    bool                 advancedBtnPressed_ = false;
    juce::Rectangle<int> beatsRowRect_;
    juce::Rectangle<int> snapRowRect_;
    juce::Rectangle<int> insertSpliceRowRect_;
    juce::Rectangle<int> insertRegionRowRect_;

    // ADR-053 — CACHE section state. Clear + Reveal share a row, side-by-
    // side. Stats string is rebuilt from setCacheStats inputs and rendered
    // above the buttons (read-only).
    juce::String         cacheStatsText_  = "Empty";
    juce::Rectangle<int> cacheStatsRect_;
    juce::Rectangle<int> cacheClearRect_;
    juce::Rectangle<int> cacheRevealRect_;
    bool                 cacheClearHover_   = false;
    bool                 cacheClearPressed_ = false;
    bool                 cacheRevealHover_   = false;
    bool                 cacheRevealPressed_ = false;

    // Body height = padding-top + (DISPLAY title + margin + row) +
    //               sectionGap + (WINDOW title + margin + button) + padding-bot.
    static constexpr int kBodyWidth   = 260; // plugin.css:610
    static constexpr int kPadding     = 12;  // plugin.css:609
    static constexpr int kAnchorRight = 8;   // shell.jsx:201
    static constexpr int kAnchorTop   = 38;  // shell.jsx:201
    static constexpr int kTitleH      = 9;   // Xs
    static constexpr int kTitleMargB  = 8;   // plugin.css:617 margin-bottom
    static constexpr int kBtnH        = 30;
    static constexpr int kBeatsRowH   = 30;  // matches kBtnH so row metrics align
    static constexpr int kSnapRowH    = 30;  // sesja 60 — Snap region cycler row
    static constexpr int kInsertRowH  = 30;  // sesja 100b — DEV-049 toggles share row metric
    static constexpr int kRowGap      = 6;   // gap between adjacent rows in same section
    static constexpr int kSectionGap  = 12;  // visual separation DISPLAY ↔ WINDOW
    // ADR-053 — CACHE section sizes.
    static constexpr int kCacheStatsH = 20;  // single-line readout
    static constexpr int kCacheBtnH   = 28;  // Clear + Reveal share a row
    static constexpr int kCacheBtnGap = 8;   // horizontal gap between buttons

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SettingsPopover)
};

} // namespace reamix::ui
