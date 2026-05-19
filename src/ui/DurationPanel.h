#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <optional>

// DurationPanel — 78h mode panel: 156px readout column (label "Target
// duration" + large mono `rxFmt(target)` + signed delta chip) on the left,
// slider column on the right (track + ORIG dashed thumb + target solid thumb
// + tickrow min/ORIG/max).
//
// Ported from phases/phase-6-ui/reamix.me JUCE/plugin.css § .rx-mode-panel /
// .rx-duration-row / .rx-readout / .rx-slider / .rx-tickrow (L216-285) +
// shell.jsx § RxDurationPanel (L86-139).
//
// Step-3 scope (ADR-036 D7):
// - Custom slider implementation (not juce::Slider) — the 2-thumb layout
//   (ORIG decoration + target) + ORIG dashed style are simpler expressed
//   directly in paint + mouseDrag than via LookAndFeel::drawLinearSlider.
// - onTargetChanged fires on every drag step (integer seconds per
//   shell.jsx:97 Math.round(v)).
// - setEnabled(false) renders the slider in disabled style (Fg4 fill +
//   Fg3 thumb) — consumed at step 6 by Region mode.

namespace reamix::ui
{

class DurationPanel : public juce::Component
{
public:
    DurationPanel();
    ~DurationPanel() override = default;

    void setRange             (double minSec, double maxSec);
    void setOriginalDuration  (double origSec);
    void setTarget            (double sec);   // programmatic set; no callback
    double getTarget()              const noexcept { return target_;   }
    double getOriginalDuration()    const noexcept { return origSec_;  }  // sesja 64 BUG-1

    // Sesja 60, step 6 — Region mode display.
    // When set, the readout label flips from "TARGET DURATION" to
    // "REGION · LIVE", the readout value stays target_ (slider is ACTIVE per
    // Lua remix_ui.lua:343-372 — Region mode core feature is retargeting the
    // region), and the tickrow center label switches to "REGION M:SS-M:SS".
    // ORIG marker on the slider tracks `origSec_` as before; caller (MainComponent)
    // sets origSec_ = region duration when entering region, restores to file
    // duration on exit.
    struct RegionInfo
    {
        double startSec {0.0};
        double endSec   {0.0};
    };
    void setRegion (std::optional<RegionInfo>);

    // Avoid clashing with juce::Component::setEnabled / isEnabled — "active"
    // here means the slider is live (targets mode-panel Region vs Duration).
    void setActive (bool) noexcept;
    bool isActive() const noexcept { return enabled_; }

    // DEV-081 sesja 112 — "Replace original item" checkbox state. Controls
    // whether Insert overwrites the source clip (true, current behaviour)
    // or leaves the source intact and appends remix clips right after it
    // on the same track (false). Default true preserves the pre-DEV-081
    // workflow. Caller persists state via ExtState; programmatic set on
    // startup uses setShouldReplaceOriginal.
    void setShouldReplaceOriginal (bool) noexcept;
    bool shouldReplaceOriginal()   const noexcept;

    // DEV-081 sesja 112 — checkbox visibility tracks the Region tab, NOT
    // the regionInfo_ payload (so the user sees the toggle as soon as they
    // switch to the Region tab, even before a time selection is created).
    // MainComponent calls this from recomputeRegionState whenever appMode_
    // changes.
    void setRegionTabActive (bool) noexcept;

    static constexpr int kPanelHeight = 100; // DEV-081: 78 → 100 to host the checkbox row.

    std::function<void(double)> onTargetChanged;
    std::function<void(bool)>   onShouldReplaceOriginalToggled;

    void paint            (juce::Graphics&) override;
    void resized          () override;
    void mouseDown        (const juce::MouseEvent&) override;
    void mouseDrag        (const juce::MouseEvent&) override;
    void mouseUp          (const juce::MouseEvent&) override;
    void mouseDoubleClick (const juce::MouseEvent&) override;
    void mouseMove        (const juce::MouseEvent&) override;
    void mouseExit        (const juce::MouseEvent&) override;

private:
    double targetFromMouseX (int x) const;
    int    xFromTarget      (double sec) const;

    // Canonical ranges — plugin.jsx:118-119 min=120, max=round(origDur*2.5).
    double minSec_  { 120.0 };
    double maxSec_  { 480.0 };
    double origSec_ { 192.0 };
    // Sesja 64 BUG-1 — hover state on ORIG label + marker (clickable hint).
    bool hoveringOrigLabel_  { false };
    bool hoveringOrigMarker_ { false };
    double target_  { 150.0 };
    bool   enabled_ { true };
    bool   dragging_ { false };
    std::optional<RegionInfo> regionInfo_;

    // Layout bounds (computed in resized()).
    juce::Rectangle<int> readoutCol_;
    juce::Rectangle<int> sliderCol_;
    juce::Rectangle<int> trackBounds_;     // 4px-thick track line rect
    juce::Rectangle<int> trackHitArea_;    // 22h tall interaction strip
    juce::Rectangle<int> tickrowBounds_;

    // DEV-081 sesja 112 — checkbox host; positioned in the bottom 22 px row
    // by resized(). Listener is wired in ctor → invokes onShouldReplaceOriginalToggled.
    juce::ToggleButton replaceOriginalToggle_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DurationPanel)
};

} // namespace reamix::ui
