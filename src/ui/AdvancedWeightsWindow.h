#pragma once

// Sesja 107 — ADR-097 implementation.
//
// AdvancedWeightsWindow — juce::DocumentWindow host for AdvancedWeightsPanel
// (the 8 raw cost weights + 4 Block Assembly β-controls per ADR-098 post-
// PerceptualMapping cleanup). Opened from SettingsPopover "Advanced..." button
// per ADR-097 sub-decision Path A.
//
// Ownership: window holds a non-owning pointer to the panel. MainComponent
// owns the panel (std::unique_ptr) so its state survives across window
// open/close cycles. Window destruction does NOT destroy the panel.
//
// Lifecycle:
//   - Lazy-created in MainComponent on first onAdvancedToggled invocation.
//   - On close button / Esc: window invokes onCloseRequested callback;
//     MainComponent persists window bounds + sets advanced_open ExtState to 0.
//   - On REAPER restart: MainComponent reads advanced_open ExtState; if 1,
//     re-creates window + restores bounds from advanced_pos ExtState.
//
// Modeless: window does not block the main reamix plugin window. User can
// drag a slider and hear the audition feedback in the main window
// simultaneously.

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <memory>

#include "LookAndFeelReamix.h"

namespace reamix::ui
{

class AdvancedWeightsPanel;

class AdvancedWeightsWindow : public juce::DocumentWindow
{
public:
    // The panel pointer must outlive this window. MainComponent owns the
    // panel via std::unique_ptr; window references it.
    explicit AdvancedWeightsWindow (AdvancedWeightsPanel* panel);
    ~AdvancedWeightsWindow() override;

    // Fired when the user clicks the close button or presses Esc. Host
    // (MainComponent) persists window bounds + flips advanced_open ExtState
    // to "0" + deletes the window.
    std::function<void()> onCloseRequested;

    // Re-reads panel->getPreferredHeight() and resizes the window content
    // area accordingly. Called by host when the panel's preferred height
    // changes (e.g. on mode-tab switch that toggles β-section visibility).
    void refitToPanel();

    // juce::DocumentWindow
    void closeButtonPressed() override;

private:
    AdvancedWeightsPanel* panel_ = nullptr;

    // ADR-097 sesja 107 iter-4 — local TooltipWindow so slider/label tooltips
    // surface inside the advanced window. The main shell's TooltipWindow is
    // bound to a different top-level NSWindow and positions popups in its
    // own coordinate space — leaving advanced-window hovers without tips.
    // Iter-5 — TransparentTooltipWindow flips setOpaque(false) so the LAF's
    // alpha-blended drawTooltip renders as actual translucent pill.
    std::unique_ptr<TransparentTooltipWindow> tooltipWindow_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedWeightsWindow)
};

} // namespace reamix::ui
