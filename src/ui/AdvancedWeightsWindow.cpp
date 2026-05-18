#include "AdvancedWeightsWindow.h"

#include "AdvancedWeightsPanel.h"
#include "Theme.h"

namespace reamix::ui
{

namespace
{
    namespace th = reamix::theme;

    // Width derived from AdvancedWeightsPanel::layoutColumns: kLabelColW=220 +
    // kReadoutColW=110 + min slider track ~160 + 2×kPanelPadding=24 = 514 →
    // round to 540 for headroom on action row buttons (3 × ~170 px).
    constexpr int kDefaultWidth  = 540;
    constexpr int kFallbackHeight = 480;
} // namespace

AdvancedWeightsWindow::AdvancedWeightsWindow (AdvancedWeightsPanel* panel)
    : juce::DocumentWindow ("Advanced Weights",
                            th::Bg1,
                            juce::DocumentWindow::closeButton)
    , panel_ (panel)
{
    setUsingNativeTitleBar (true);
    setResizable (false, false);  // panel layout is fixed (sliders+actions); width/height computed
    // Stay above the main reamix plugin window even when REAPER is in floating
    // (always-on-top) mode. Without this the advanced window slips behind
    // every time the main window regains focus.
    setAlwaysOnTop (true);

    if (panel_ != nullptr)
    {
        setContentNonOwned (panel_, true);
        const int h = panel_->getPreferredHeight();
        setSize (kDefaultWidth, h > 0 ? h : kFallbackHeight);
    }
    else
    {
        setSize (kDefaultWidth, kFallbackHeight);
    }

    centreWithSize (getWidth(), getHeight());

    // Slider/label tooltips inside the advanced window. 600 ms delay matches
    // the main shell's sliderTooltipWindow_ (MainComponent.h sesja 93).
    // Iter-5 — TransparentTooltipWindow enables alpha blending.
    tooltipWindow_ = std::make_unique<TransparentTooltipWindow> (this, 600);
}

AdvancedWeightsWindow::~AdvancedWeightsWindow()
{
    // Drop content reference so DocumentWindow destruction doesn't touch
    // panel_ (panel lifetime is owned by host MainComponent).
    clearContentComponent();
}

void AdvancedWeightsWindow::refitToPanel()
{
    if (panel_ == nullptr) return;
    const int h = panel_->getPreferredHeight();
    if (h > 0)
        setSize (getWidth(), h);
}

void AdvancedWeightsWindow::closeButtonPressed()
{
    if (onCloseRequested) onCloseRequested();
    // Caller decides when to actually destroy the window. Default JUCE
    // behaviour (delete-on-close) is suppressed by not calling setVisible(false)
    // here — host invokes that via its onCloseRequested handler.
}

} // namespace reamix::ui
