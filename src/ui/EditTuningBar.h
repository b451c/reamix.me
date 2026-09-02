#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

#include "Theme.h"

// EditTuningBar — ADR-115 P3 (sesja 123). The one user control besides the
// target duration: "Edit density", five detents mapped to the minimum run
// between two cuts in bars (16 / 8 / 4 / 2 / 1). Left = fewer cuts, right =
// more cuts. Replaces the AuditionBar (Tone / Edit length / Allow +- / Min
// segment, ADR-080 / 083 / 084) and the Advanced weights window (ADR-097):
// the weights are calibrated by the blinded panels, the density is the axis
// the literature and every shipped remix tool agree on.
//
// The default detent is per mode (MainComponent::defaultDensityBars): the
// engine's own default cooldown (Duration 4 bars, Region 1 bar), so the
// default position is bit-exact with the production behaviour. A tick under
// the track marks it; double-click returns to it.
//
// Collapsible header ("Edit tuning") kept from the AuditionBar so the
// waveform can take the space; the host persists the state.

namespace reamix::ui
{

class EditTuningBar : public juce::Component,
                      private juce::Slider::Listener
{
public:
    EditTuningBar();
    ~EditTuningBar() override;

    static constexpr int kDetentBars[5] = { 16, 8, 4, 2, 1 };
    static int detentForBars (int bars) noexcept;
    static int barsForDetent (int detent) noexcept;

    // Programmatic setters — no callback.
    void setBars        (int bars);        // one of kDetentBars
    void setDefaultBars (int bars);        // the mode's default (tick + double-click)
    int  barsValue()    const noexcept { return bars_; }
    int  defaultBars()  const noexcept { return defaultBars_; }

    std::function<void(int)> onBarsChanged;   // user gesture only

    void setCollapsed (bool collapsed);
    bool isCollapsed() const noexcept { return collapsed_; }
    int  getPreferredHeight() const noexcept;
    std::function<void()> onCollapseToggled;

    // juce::Component
    void paint     (juce::Graphics&) override;
    void resized   () override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;

private:
    void sliderValueChanged (juce::Slider* s) override;
    juce::String readout() const;

    static constexpr int kRowHeight     = 36;   // 22 slider + 14 endpoint labels
    static constexpr int kBodyHeight    = kRowHeight;
    static constexpr int kLabelWidth    = 88;
    static constexpr int kReadoutWidth  = 110;
    static constexpr int kPadding       = 12;
    static constexpr int kTopPadding    = 28;   // header strip
    static constexpr int kBottomPadding = 12;

    juce::Slider slider_;                       // 0..4 detents
    juce::Label  label_;                        // "Edit density" + tooltip

    int  bars_        = 4;
    int  defaultBars_ = 4;
    bool suppressCallbacks_ = false;

    bool collapsed_     = false;
    bool headerHover_   = false;
    bool headerPressed_ = false;
    juce::Rectangle<int> headerHitBounds_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EditTuningBar)
};

} // namespace reamix::ui
