#include "EditTuningBar.h"
#include "EditDensity.h"

namespace reamix::ui
{

namespace th = reamix::theme;

int EditTuningBar::detentForBars (int bars) noexcept
{
    for (int d = 0; d < 5; ++d)
        if (kDetentBars[d] == bars) return d;
    return 2;   // 4 bars
}

int EditTuningBar::barsForDetent (int detent) noexcept
{
    return kDetentBars[juce::jlimit (0, 4, detent)];
}

EditTuningBar::EditTuningBar()
{
    label_.setText ("Edit density", juce::dontSendNotification);
    setCutLabels (false);
    label_.setFont (th::uiFont (th::fs::Md, 500));
    label_.setColour (juce::Label::textColourId, th::Fg1);
    label_.setColour (juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    label_.setJustificationType (juce::Justification::centredLeft);
    label_.setInterceptsMouseClicks (true, false);
    addAndMakeVisible (label_);

    slider_.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
    slider_.setRange (0.0, 4.0, 1.0);
    slider_.setValue ((double) detentForBars (bars_), juce::dontSendNotification);
    slider_.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider_.setColour (juce::Slider::trackColourId,      th::Bg4);
    slider_.setColour (juce::Slider::backgroundColourId, th::Bg3);
    slider_.setColour (juce::Slider::thumbColourId,      th::Accent);
    slider_.setDoubleClickReturnValue (true, (double) detentForBars (defaultBars_));
    slider_.addListener (this);
    addAndMakeVisible (slider_);
}

EditTuningBar::~EditTuningBar() = default;

void EditTuningBar::setBars (int bars)
{
    suppressCallbacks_ = true;
    bars_ = barsForDetent (detentForBars (bars));
    slider_.setValue ((double) detentForBars (bars_), juce::dontSendNotification);
    suppressCallbacks_ = false;
    repaint();
}

void EditTuningBar::setDefaultBars (int bars)
{
    defaultBars_ = barsForDetent (detentForBars (bars));
    slider_.setDoubleClickReturnValue (true, (double) detentForBars (defaultBars_));
    repaint();
}

void EditTuningBar::setCollapsed (bool c)
{
    if (c == collapsed_) return;
    collapsed_ = c;
    slider_.setVisible (! collapsed_);
    label_.setVisible (! collapsed_);
    resized();
    repaint();
}

int EditTuningBar::getPreferredHeight() const noexcept
{
    return collapsed_ ? kTopPadding : (kTopPadding + kBodyHeight + kBottomPadding);
}

void EditTuningBar::sliderValueChanged (juce::Slider* s)
{
    if (suppressCallbacks_ || s != &slider_) return;
    const int bars = barsForDetent ((int) std::lround (slider_.getValue()));
    if (bars == bars_) return;
    bars_ = bars;
    repaint();
    if (onBarsChanged) onBarsChanged (bars_);
}

void EditTuningBar::setCutLabels (bool cutLabels)
{
    cutLabels_ = cutLabels;
    label_.setTooltip (cutLabels
        ? juce::String ("How many cuts the edit may use.\n"
                        "Fewer cuts: long untouched stretches, the edit stays close to the original.\n"
                        "More cuts: at least 2 or 4 cuts - a shorter version removes several phrases "
                        "instead of one, a longer one repeats several; fewer when the track has no clean way.\n"
                        "Double-click the slider to return to the default.")
        : juce::String ("Length of the loop between two cuts, in bars.\n"
                        "Fewer cuts: long untouched stretches, the edit stays close to the original.\n"
                        "More cuts: shorter phrases may be repeated or skipped.\n"
                        "Double-click the slider to return to the default."));
    repaint();
}

juce::String EditTuningBar::readout() const
{
    if (cutLabels_) return durationDensityLabel (bars_);
    const juce::String n = juce::String (bars_) + (bars_ == 1 ? " bar" : " bars");
    return bars_ == defaultBars_ ? juce::String::fromUTF8 ("Default \xc2\xb7 ") + n : n;
}

void EditTuningBar::resized()
{
    headerHitBounds_ = { 0, 0, getWidth(), kTopPadding };
    if (collapsed_) return;

    auto area = getLocalBounds();
    area.removeFromTop    (kTopPadding);
    area.removeFromBottom (kBottomPadding);
    area = area.reduced (kPadding, 0);

    constexpr int kSliderHeight = 22;
    auto row       = area.removeFromTop (kRowHeight);
    auto sliderRow = row.removeFromTop (kSliderHeight);
    auto labelArea = sliderRow.removeFromLeft (kLabelWidth);
    sliderRow.removeFromRight (kReadoutWidth);
    slider_.setBounds (sliderRow);
    label_.setBounds  (labelArea);
}

void EditTuningBar::paint (juce::Graphics& g)
{
    g.fillAll (th::Bg2);
    g.setColour (th::LineStrong);
    g.fillRect (0, 0, getWidth(), 1);
    g.fillRect (0, getHeight() - 1, getWidth(), 1);

    // Header strip: "Edit tuning" + chevron (collapse / expand).
    {
        const auto hb = headerHitBounds_.toFloat();
        if (headerPressed_)      { g.setColour (th::Bg4); g.fillRect (hb); }
        else if (headerHover_)   { g.setColour (th::Bg3); g.fillRect (hb); }

        const juce::Colour labelCol = (headerHover_ || headerPressed_) ? th::Fg1 : th::Fg2;
        const juce::Font labelFont = th::uiFont (th::fs::Sm, 500);
        const juce::String labelText { "Edit tuning" };
        const float labelW = labelFont.getStringWidthFloat (labelText);
        g.setColour (labelCol);
        g.setFont (labelFont);
        g.drawText (labelText, juce::Rectangle<float> ((float) kPadding, 0.0f, labelW, (float) kTopPadding),
                    juce::Justification::centredLeft, false);

        const float chevSize = 8.0f;
        const float chevCx   = (float) kPadding + labelW + 8.0f + chevSize * 0.5f;
        const float chevCy   = (float) kTopPadding * 0.5f;
        juce::Path chev;
        if (collapsed_)
            chev.addTriangle (chevCx - chevSize * 0.5f, chevCy - chevSize * 0.5f,
                              chevCx - chevSize * 0.5f, chevCy + chevSize * 0.5f,
                              chevCx + chevSize * 0.5f, chevCy);
        else
            chev.addTriangle (chevCx - chevSize * 0.5f, chevCy - chevSize * 0.5f,
                              chevCx + chevSize * 0.5f, chevCy - chevSize * 0.5f,
                              chevCx,                   chevCy + chevSize * 0.5f);
        g.fillPath (chev);
    }

    if (collapsed_) return;

    auto area = getLocalBounds();
    area.removeFromTop    (kTopPadding);
    area.removeFromBottom (kBottomPadding);
    area = area.reduced (kPadding, 0);

    constexpr int kSliderHeight   = 22;
    constexpr int kEndpointHeight = 14;

    auto row    = area.removeFromTop (kRowHeight);
    auto topRow = row.removeFromTop (kSliderHeight);
    topRow.removeFromLeft (kLabelWidth);
    auto readoutArea = topRow.removeFromRight (kReadoutWidth);

    g.setFont (th::monoFont (th::fs::Sm, 400));
    g.setColour (th::Fg2);
    g.drawText (readout(), readoutArea.toFloat(), juce::Justification::centredRight, true);

    // Endpoint labels + detent ticks under the track; the default detent's
    // tick is brighter so the neutral position is visible at a glance.
    auto endpointRow = row.removeFromTop (kEndpointHeight);
    endpointRow.removeFromLeft (kLabelWidth);
    endpointRow.removeFromRight (kReadoutWidth);

    const auto sb = slider_.getBounds();
    const float trackX0 = (float) sb.getX() + 8.0f;          // JUCE linear slider thumb inset
    const float trackX1 = (float) sb.getRight() - 8.0f;
    for (int d = 0; d < 5; ++d)
    {
        const float x = trackX0 + (trackX1 - trackX0) * (float) d / 4.0f;
        const bool isDefault = kDetentBars[d] == defaultBars_;
        g.setColour (isDefault ? th::Accent.withAlpha (0.9f) : th::Fg3.withAlpha (0.6f));
        g.fillRect (juce::Rectangle<float> (x - 0.5f, (float) endpointRow.getY(), 1.0f, isDefault ? 5.0f : 3.0f));
    }

    g.setFont (th::uiFont (th::fs::Xs, 400));
    g.setColour (th::Fg3);
    const auto textRow = endpointRow.withTrimmedTop (4);
    g.drawText ("Fewer cuts", textRow.toFloat(), juce::Justification::topLeft,  true);
    g.drawText ("More cuts",  textRow.toFloat(), juce::Justification::topRight, true);
}

void EditTuningBar::mouseMove (const juce::MouseEvent& e)
{
    const bool overHdr = headerHitBounds_.contains (e.getPosition());
    if (overHdr != headerHover_)
    {
        headerHover_ = overHdr;
        repaint (headerHitBounds_);
    }
    setMouseCursor (overHdr ? juce::MouseCursor::PointingHandCursor : juce::MouseCursor::NormalCursor);
}

void EditTuningBar::mouseExit (const juce::MouseEvent&)
{
    if (headerHover_ || headerPressed_)
    {
        headerHover_   = false;
        headerPressed_ = false;
        repaint (headerHitBounds_);
    }
}

void EditTuningBar::mouseDown (const juce::MouseEvent& e)
{
    if (headerHitBounds_.contains (e.getPosition()))
    {
        headerPressed_ = true;
        repaint (headerHitBounds_);
    }
}

void EditTuningBar::mouseUp (const juce::MouseEvent& e)
{
    const bool wasPressed = headerPressed_;
    headerPressed_ = false;
    repaint (headerHitBounds_);
    if (wasPressed && headerHitBounds_.contains (e.getPosition()))
    {
        setCollapsed (! collapsed_);
        if (onCollapseToggled) onCollapseToggled();
    }
}

} // namespace reamix::ui
