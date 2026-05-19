#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

// StatusBar — 22h bottom strip: left text + right-aligned version chip.
// Ported from phases/phase-6-ui/reamix.me JUCE/plugin.css § .rx-statusbar
// (L430-441) + shell.jsx § RxStatusBar (L188-194).
//
// Step-3 scope (ADR-036 D7):
// - setText() for normal messages (Fg3 mono xs).
// - setError() switches the whole bar to Bad colour per .rx-statusbar.error
//   (plugin.css:441). Clearing restores normal text.
// - Right-aligned "v0.3.0" pinned in Fg4 regardless of state (shell.jsx:192).
//
// Sesja 64 — busy state (spinner-family fix for NOTE-9 + cross-refs).
// setBusy(label) renders accent-colored bold label + small spinning arc on
// the left + 2px accent loading line ślidząca above the bar. juce::Timer at
// 60 Hz drives the loading-line slide + spinner rotation while busy.
// clearBusy() restores normal text. Used by 4 compute paths: Slider remix
// recompute, Block Assembly Assemble, Insert into REAPER, plus secondary
// paths (preview rebuild). Threshold (300 ms) lives in MainComponent — by
// the time setBusy is called, busy IS visible. Error state takes priority
// over busy if both fire.

namespace reamix::ui
{

class StatusBar : public juce::Component, private juce::Timer
{
public:
    StatusBar();
    ~StatusBar() override = default;

    void setText    (juce::String);  // normal message; clears error on call
    void setError   (juce::String);  // non-empty → renders in Bad, overrides text
    void clearError();               // restores normal text

    // Sesja 64 — busy state.
    void setBusy   (juce::String label);  // accent text + spinner + loading line
    void clearBusy ();                    // restore normal text
    bool isBusy    () const noexcept { return busy_; }

    // Sesja 64 BUG-3 — short-lived "notice" message: accent-bold text auto-
    // clears after durationMs (default 2500). For user-visible feedback that
    // setText alone is too quiet for ("No more alternatives", "Reset to
    // original", "Stopped — Space resumes from M:SS"). Error overrides
    // notice; busy overrides notice; clearing returns to last text_.
    void setNotice (juce::String label, int durationMs = 2500);
    void clearNotice ();

    void paint (juce::Graphics&) override;

private:
    void timerCallback() override;

    juce::String text_    { "Ready" };
    juce::String error_;               // non-empty → error state
    juce::String busyLabel_;           // non-empty → busy state
    juce::String notice_;              // non-empty → short-lived accent notice
    bool         busy_    { false };
    float        animPhase_ { 0.0f };  // [0..1) drives loading line + spinner rotation
    juce::String version_ { "v1.0.3" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StatusBar)
};

} // namespace reamix::ui
