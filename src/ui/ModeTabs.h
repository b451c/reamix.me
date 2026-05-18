#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

// ModeTabs — 36h tab strip Duration / Blocks, with auto-active Region pill.
// Ported from `phases/phase-6-ui/reamix.me JUCE/shell.jsx:68-83 RxModeTabs` +
// `plugin.css:185-214` § .rx-mode-tabs.
//
// Behavior (sesja 60 — three-mode model per user decision):
// - Three tabs always visible: Duration / Region / Blocks. User picks the
//   mode by clicking a tab.
// - AUTO badge on the Region tab is INDEPENDENT of which tab is active —
//   driven by `setAutoFlag(true)` from MainComponent when REAPER time-selection
//   was the trigger that brought us into Region mode (vs. user click). The
//   badge tells the user "plugin noticed the time-selection, you didn't pick
//   this manually".
// - Blocks tab is rendered but disabled until `setBlocksEnabled(true)`
//   (sesja 62-63).
//
// Active tab uses Accent text + Accent 2px bottom-border indicator. AUTO badge
// uses AccentDim bg + Accent text in small caps per CSS plugin.css:207-213.

namespace reamix::ui
{

class ModeTabs : public juce::Component
{
public:
    enum class Mode
    {
        Duration,
        Region,
        Blocks
    };

    ModeTabs();
    ~ModeTabs() override = default;

    void setMode (Mode);
    Mode getMode() const noexcept { return mode_; }

    // AUTO badge on Region tab. Independent of mode — driven by MainComponent
    // when REAPER time-selection (not user click) was the trigger.
    void setAutoFlag (bool);
    bool isAutoFlagged() const noexcept { return autoFlag_; }

    // Blocks tab enabled state — disabled until user has analyzed an item
    // (Lua parity: shell.jsx:77 `disabled={!analyzed}`).
    void setBlocksEnabled (bool);

    // Invoked when the user clicks any tab.
    std::function<void(Mode)> onChange;

    void paint     (juce::Graphics&) override;
    void resized   () override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;

private:
    void recomputeLayout();

    Mode mode_           { Mode::Duration };
    bool autoFlag_       { false };
    bool blocksEnabled_  { false };

    juce::Rectangle<int> durationBounds_ {};
    juce::Rectangle<int> regionBounds_   {};
    juce::Rectangle<int> blocksBounds_   {};
    int                  hoverIdx_       { -1 }; // 0=Duration, 1=Region, 2=Blocks

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ModeTabs)
};

} // namespace reamix::ui
