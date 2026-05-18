#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

// TransportBar — 54h bottom-docked transport: 2 buttons in a grid
// (auto, 1fr) with 8px gap. PlayStop toggle (morphs Preview.good ↔ Stop.bad
// based on TransportState; Space/Esc chip), Insert-or-Update (.rx-btn.primary
// full-width + Enter chip).
//
// Ported from phases/phase-6-ui/reamix.me JUCE/plugin.css § .rx-transport /
// .rx-btn.play / .rx-btn.stop / .rx-btn.primary / .rx-kbd (L418-183) +
// shell.jsx § RxTransport (L171-186).
//
// DEVIATION FROM MOCKUP (ADR-038, session 46): mockup shell.jsx:171-186
// shows two adjacent buttons (Preview=good + Stop=bad). Session-46 user
// feedback: "Po co to rozdzielac na dwa przyciski preview zamienia sie w
// stop i odwrotnie" — one toggle button that swaps label/color/icon is
// clearer + saves horizontal real estate. Space AND Esc both invoke the
// toggle; Enter still invokes Insert.
//
// Step-3/4 scope (ADR-036 D7 + ADR-038):
// - Two std::function slots: onPlayStop / onInsert.
// - Button state per TransportState (Idle / Ready / Playing / Inserted):
//   * Idle    → all disabled (empty / analyzing state).
//   * Ready   → PlayStop shows "Preview" (good + play-icon); Insert enabled.
//   * Playing → PlayStop shows "Stop"    (bad  + stop-icon); Insert disabled.
//   * Inserted → Insert label becomes "Update"; PlayStop back to "Preview".
// - Implements juce::KeyListener: Space AND Esc both invoke onPlayStop;
//   Enter invokes onInsert. MainComponent registers this as a key listener
//   so shortcuts fire from anywhere in the window (⚑ P5 foundation).
// - No PreviewController wiring here; slot is invoked stubless (step 4
//   ties into PreviewController, step 9 ties Insert to ReaperBridge).

namespace reamix::ui
{

enum class TransportState
{
    Idle,       // no source / analyzing — all buttons disabled
    Ready,      // analyzed, not playing — Preview + Insert active
    Playing,    // preview in progress — only Stop active
    Inserted    // result inserted — Insert labelled "Update", all enabled
};

class TransportBar : public juce::Component, public juce::KeyListener
{
public:
    TransportBar();
    ~TransportBar() override = default;

    void setState (TransportState);
    TransportState getState() const noexcept { return state_; }

    // Sesja 64 — busy state for Insert button during REAPER timeline mutations.
    // When busy: button background dimmed, label replaced with the busy label
    // (e.g. "Inserting…"), button ignores clicks. Cleared by setBusy(false).
    void setInsertBusy (bool busy, juce::String label = {});
    bool isInsertBusy () const noexcept { return insertBusy_; }

    // Single toggle slot — ADR-038. Invoked on click OR Space OR Esc.
    // MainComponent's handler queries PreviewController::isPlaying() and
    // dispatches to play or stop accordingly.
    std::function<void()> onPlayStop;
    std::function<void()> onInsert;

    void paint     (juce::Graphics&) override;
    void resized   () override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;

    // KeyListener's keyPressed(KeyPress, Component*) shadows Component's
    // keyPressed(KeyPress) — re-expose the base for focused callers
    // (TransportBar itself never takes focus, but the using declaration
    // silences -Woverloaded-virtual).
    using juce::Component::keyPressed;

    bool keyPressed (const juce::KeyPress&,
                     juce::Component* originatingComponent) override;

private:
    // ADR-038 — single PlayStop slot instead of separate Preview/Stop.
    enum class Button { None, PlayStop, Insert };

    Button buttonAt     (juce::Point<int>) const;
    bool   isEnabledFor (Button) const noexcept;
    void   invoke       (Button);
    void   setHover     (Button);
    void   setPressed   (Button);

    TransportState state_ { TransportState::Idle };

    bool         insertBusy_  { false };
    juce::String insertBusyLabel_;

    juce::Rectangle<int> playStopBounds_;
    juce::Rectangle<int> insertBounds_;

    Button hover_   { Button::None };
    Button pressed_ { Button::None };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TransportBar)
};

} // namespace reamix::ui
