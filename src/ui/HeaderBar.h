#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

// HeaderBar — 40h top strip: mark dot + "reamix" / ".me" + flex status strip +
// gear icon button. Ported from phases/phase-6-ui/reamix.me JUCE/plugin.css
// § .rx-header / .rx-mark / .rx-status-strip / .rx-icon-btn (L22-79).
//
// Step-1 behavior (ADR-036 D8):
// - Mark + name + empty status strip + gear are fully styled.
// - Status strip content is fixed "Ready" placeholder (dynamic content + error
//   style land step 3 with StatusBar wiring).
// - Gear is hit-testable with hover/pressed states but the onGearClicked
//   callback is a no-op placeholder; SettingsPopover lands step 5.
// - Status-dot pulse animation (⚑ P4) is deferred to step 3.

namespace reamix::ui
{

enum class HeaderStatus
{
    Ready,     // Fg2 text + Fg3 dot
    Analyzing, // Fg2 text + accent pulsing dot
    Good,      // Fg2 text + Good dot
    Error      // Bad text + Bad dot, strip bg tinted
};

class HeaderBar : public juce::Component,
                  private juce::Timer
{
public:
    HeaderBar();
    ~HeaderBar() override = default;

    void setStatusKind (HeaderStatus);

    // Invoked when the gear button is clicked. No-op until step 5 wires
    // SettingsPopover. Safe to leave unset.
    std::function<void()> onGearClicked;

    // Sesja 108 — heart icon shortcut for SupportPopover (donation links).
    // Positioned 6 px to the left of the gear icon. Same hover/pressed
    // styling for visual consistency. Host wires this to
    // MainComponent::onSupportToggled().
    std::function<void()> onHeartClicked;

    void paint  (juce::Graphics&) override;
    void resized() override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;

private:
    // Sesja 108 — LED-style breathing pulse on the status dot while
    // statusKind_ == Analyzing. Cosine modulates alpha 0.4..1.0 over a
    // 1.2 s period at 30 fps. Idle/Good/Error states keep the dot static.
    void timerCallback() override;

    HeaderStatus   statusKind_  { HeaderStatus::Ready };

    juce::Rectangle<int> gearBounds_;
    bool gearHover_   = false;
    bool gearPressed_ = false;

    // Sesja 108 — heart icon (donation shortcut) sits to the left of gear.
    juce::Rectangle<int> heartBounds_;
    bool heartHover_   = false;
    bool heartPressed_ = false;

    float pulsePhase_ = 0.0f;  // 0..1; advances kPulseHz × kPulseCycleSec per tick
    static constexpr float kPulseHz       = 30.0f;
    static constexpr float kPulseCycleSec = 1.2f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HeaderBar)
};

} // namespace reamix::ui
