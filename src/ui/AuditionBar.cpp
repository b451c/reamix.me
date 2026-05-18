#include "AuditionBar.h"

namespace reamix::ui
{

namespace th = reamix::theme;

AuditionBar::AuditionBar()
{
    // ADR-084 sesja 93 — helper to configure each title label with text +
    // tooltip + visual style (uiFont Md 500 / Fg1 / transparent background).
    // Tooltips are attached HERE (not on sliders) so hover-on-track during
    // drag does not spam tooltip popups.
    auto configureLabel = [this](juce::Label& l,
                                 const juce::String& text,
                                 const juce::String& tooltip)
    {
        l.setText (text, juce::dontSendNotification);
        l.setTooltip (tooltip);
        l.setFont (th::uiFont (th::fs::Md, 500));
        l.setColour (juce::Label::textColourId, th::Fg1);
        l.setColour (juce::Label::backgroundColourId, juce::Colours::transparentBlack);
        l.setJustificationType (juce::Justification::centredLeft);
        l.setInterceptsMouseClicks (true, false);  // intercept hover, no click
        addAndMakeVisible (l);
    };

    configureLabel (toneLabel_, "Tone",
        "Spectral matching axis for transition cost.\n"
        "Timbre: maximize instrumentation and texture similarity.\n"
        "Harmonic: maximize chord and key continuity.");

    configureLabel (editLengthLabel_, "Edit length",
        "Splice density preference.\n"
        "Short: more splices, denser cuts.\n"
        "Long: fewer splices, longer untouched segments.\n"
        "Center: no preference (optimizer chooses).");

    // U+00B1 PLUS-MINUS SIGN built explicitly to avoid UTF-8 mojibake on
    // JUCE Inter face rendering (same pattern as paint() endpoint labels).
    const juce::String plusMinus = juce::String::charToString (juce::juce_wchar (0x00B1));
    configureLabel (allowPmLabel_, "Allow " + plusMinus,
        "Permitted duration drift from target, in seconds.\n"
        "0 s: exact target match (smaller search space).\n"
        "15 s: tolerate \xc2\xb1""15 s for better-sounding splice paths.");

    configureLabel (minCutLabel_, "Min segment",
        "Minimum length of any segment between two splices, in beats.\n"
        "4 beats: dense splices allowed (rapid-cut character).\n"
        "32 beats: sparse splices, long contiguous regions.");

    // Tone slider — continuous 0.0-1.0 (interval 0 = no snap).
    toneSlider_.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
    toneSlider_.setRange (0.0, 1.0, 0.0);
    toneSlider_.setValue (tone_, juce::dontSendNotification);
    toneSlider_.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    toneSlider_.setColour (juce::Slider::trackColourId,         th::Bg4);
    toneSlider_.setColour (juce::Slider::backgroundColourId,    th::Bg3);
    toneSlider_.setColour (juce::Slider::thumbColourId,         th::Accent);
    // Sesja 107 iter-4 — double-click → reset to default.
    toneSlider_.setDoubleClickReturnValue (true, 0.5);  // Balanced (mid)
    toneSlider_.addListener (this);
    addAndMakeVisible (toneSlider_);

    // Edit Length slider — discrete 0-100 snap 1.
    editLengthSlider_.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
    editLengthSlider_.setRange (0.0, 100.0, 1.0);
    editLengthSlider_.setValue ((double) editLength_, juce::dontSendNotification);
    editLengthSlider_.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    editLengthSlider_.setColour (juce::Slider::trackColourId,      th::Bg4);
    editLengthSlider_.setColour (juce::Slider::backgroundColourId, th::Bg3);
    editLengthSlider_.setColour (juce::Slider::thumbColourId,      th::Accent);
    editLengthSlider_.setDoubleClickReturnValue (true, 50.0);  // ADR-084 default 50
    editLengthSlider_.addListener (this);
    addAndMakeVisible (editLengthSlider_);

    // Allow ± slider — discrete 0-15 snap 1.
    allowPmSlider_.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
    allowPmSlider_.setRange (0.0, 15.0, 1.0);
    allowPmSlider_.setValue ((double) allowPmSeconds_, juce::dontSendNotification);
    allowPmSlider_.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    allowPmSlider_.setColour (juce::Slider::trackColourId,      th::Bg4);
    allowPmSlider_.setColour (juce::Slider::backgroundColourId, th::Bg3);
    allowPmSlider_.setColour (juce::Slider::thumbColourId,      th::Accent);
    allowPmSlider_.setDoubleClickReturnValue (true, 6.0);  // ADR-084 default 6 s
    allowPmSlider_.addListener (this);
    addAndMakeVisible (allowPmSlider_);

    // Min cut slider — discrete 4-32 snap 1.
    minCutSlider_.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
    minCutSlider_.setRange (4.0, 32.0, 1.0);
    minCutSlider_.setValue ((double) minCutBeats_, juce::dontSendNotification);
    minCutSlider_.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    minCutSlider_.setColour (juce::Slider::trackColourId,      th::Bg4);
    minCutSlider_.setColour (juce::Slider::backgroundColourId, th::Bg3);
    minCutSlider_.setColour (juce::Slider::thumbColourId,      th::Accent);
    minCutSlider_.setDoubleClickReturnValue (true, 16.0);  // ADR-084 default 16 beats
    minCutSlider_.addListener (this);
    addAndMakeVisible (minCutSlider_);
}

AuditionBar::~AuditionBar() = default;

void AuditionBar::setTone (double v)
{
    suppressCallbacks_ = true;
    tone_ = juce::jlimit (0.0, 1.0, v);
    toneSlider_.setValue (tone_, juce::dontSendNotification);
    suppressCallbacks_ = false;
    repaint();
}

void AuditionBar::setEditLength (int v)
{
    suppressCallbacks_ = true;
    editLength_ = juce::jlimit (0, 100, v);
    editLengthSlider_.setValue ((double) editLength_, juce::dontSendNotification);
    suppressCallbacks_ = false;
    repaint();
}

void AuditionBar::setAllowPmSeconds (int v)
{
    suppressCallbacks_ = true;
    allowPmSeconds_ = juce::jlimit (0, 15, v);
    allowPmSlider_.setValue ((double) allowPmSeconds_, juce::dontSendNotification);
    suppressCallbacks_ = false;
    repaint();
}

void AuditionBar::setMinCutBeats (int v)
{
    suppressCallbacks_ = true;
    minCutBeats_ = juce::jlimit (4, 32, v);
    minCutSlider_.setValue ((double) minCutBeats_, juce::dontSendNotification);
    suppressCallbacks_ = false;
    repaint();
}

void AuditionBar::setCollapsed (bool c)
{
    if (c == collapsed_) return;
    collapsed_ = c;

    // Hide/show all slider Components so they don't intercept mouse clicks
    // while collapsed (and so JUCE doesn't waste paint cycles on them).
    toneSlider_      .setVisible (! collapsed_);
    editLengthSlider_.setVisible (! collapsed_);
    allowPmSlider_   .setVisible (! collapsed_);
    minCutSlider_    .setVisible (! collapsed_);
    toneLabel_       .setVisible (! collapsed_);
    editLengthLabel_ .setVisible (! collapsed_);
    allowPmLabel_    .setVisible (! collapsed_);
    minCutLabel_     .setVisible (! collapsed_);

    resized();
    repaint();
}

int AuditionBar::getPreferredHeight() const noexcept
{
    return collapsed_ ? kTopPadding : (kTopPadding + kBodyHeight + kBottomPadding);
}

void AuditionBar::sliderValueChanged (juce::Slider* s)
{
    if (suppressCallbacks_) return;

    if (s == &toneSlider_) {
        tone_ = toneSlider_.getValue();
        repaint();
        if (onToneChanged) onToneChanged (tone_);
    } else if (s == &editLengthSlider_) {
        editLength_ = (int) editLengthSlider_.getValue();
        repaint();
        if (onEditLengthChanged) onEditLengthChanged (editLength_);
    } else if (s == &allowPmSlider_) {
        allowPmSeconds_ = (int) allowPmSlider_.getValue();
        repaint();
        if (onAllowPmSecondsChanged) onAllowPmSecondsChanged (allowPmSeconds_);
    } else if (s == &minCutSlider_) {
        minCutBeats_ = (int) minCutSlider_.getValue();
        repaint();
        if (onMinCutBeatsChanged) onMinCutBeatsChanged (minCutBeats_);
    }
}

juce::String AuditionBar::toneReadout() const
{
    // 0.0 → "Timbre 100%", 0.5 → "Balanced", 1.0 → "Harmonic 100%".
    if (tone_ < 0.01)        return "Timbre 100%";
    if (tone_ > 0.99)        return "Harmonic 100%";
    if (std::abs (tone_ - 0.5) < 0.01) return "Balanced";
    if (tone_ < 0.5) {
        const int pct = (int) std::round ((1.0 - tone_) * 100.0);
        return "Timbre " + juce::String (pct) + "%";
    }
    const int pct = (int) std::round (tone_ * 100.0);
    return "Harmonic " + juce::String (pct) + "%";
}

juce::String AuditionBar::editLengthReadout() const
{
    // ADR-084 sesja 93 — Edit length now meaningful readout instead of
    // raw int. Center 50 = Default; left = Short %; right = Long %.
    if (editLength_ == 50) return "Default";
    if (editLength_ < 50) {
        const int pct = (int) std::round ((50.0 - editLength_) * 2.0);  // 0→100% / 50→0%
        return "Short " + juce::String (pct) + "%";
    }
    const int pct = (int) std::round ((editLength_ - 50.0) * 2.0);      // 50→0% / 100→100%
    return "Long " + juce::String (pct) + "%";
}

void AuditionBar::resized()
{
    // Sesja 108 — header hit-test region: from left edge to start of the
    // quick-Advanced icon area, full top-padding height. Click anywhere in
    // here toggles the collapsed state. Advanced icon (right edge) keeps
    // its own click handler.
    headerHitBounds_ = { 0, 0,
                         getWidth() - kPadding - 16 - 6, // 6 px gap before icon
                         kTopPadding };

    // Quick-Advanced icon 16×16 centred in the 28 px top gap.
    constexpr int kAdvancedIconSize = 16;
    advancedBounds_ = { getWidth() - kPadding - kAdvancedIconSize,
                        (kTopPadding - kAdvancedIconSize) / 2,
                        kAdvancedIconSize, kAdvancedIconSize };

    if (collapsed_)
        return; // no slider rows to lay out

    auto area = getLocalBounds();
    area.removeFromTop    (kTopPadding);
    area.removeFromBottom (kBottomPadding);
    area = area.reduced (kPadding, 0);

    // ADR-084 sesja 93 — row layout split into 2 sub-rows:
    //   top 22h: [label kLabelWidth] [slider FILL] [readout kReadoutWidth]
    //   bottom 14h:                  [endpointL] FLEX [endpointR]   (paint only)
    // Slider thumb rests in top 22h area. Endpoint labels below for scale clarity.
    // Title labels (juce::Label children) host tooltip on hover.
    constexpr int kSliderHeight = 22;

    auto layoutRow = [&](juce::Slider& slider, juce::Label& label) -> void {
        auto row = area.removeFromTop (kRowHeight);
        auto sliderRow = row.removeFromTop (kSliderHeight);
        auto labelArea = sliderRow.removeFromLeft (kLabelWidth);
        sliderRow.removeFromRight (kReadoutWidth);
        slider.setBounds (sliderRow);
        label.setBounds  (labelArea);
        // Bottom 14h reserved for endpoint scale labels (rendered in paint()).
        area.removeFromTop (kRowGap);
    };

    layoutRow (toneSlider_,        toneLabel_);
    layoutRow (editLengthSlider_,  editLengthLabel_);
    layoutRow (allowPmSlider_,     allowPmLabel_);
    layoutRow (minCutSlider_,      minCutLabel_);
}

void AuditionBar::paint (juce::Graphics& g)
{
    g.fillAll (th::Bg2);

    // Top + bottom 1px dividers (LineStrong).
    g.setColour (th::LineStrong);
    g.fillRect (0, 0, getWidth(), 1);
    g.fillRect (0, getHeight() - 1, getWidth(), 1);

    // Sesja 108 — header strip: "Audition" label + chevron (collapse/expand
    // affordance). Hover background hint covers the entire hit region except
    // the right-edge Advanced icon. Drawn before the slider rows so its
    // hover background sits below the icon overlay.
    {
        const auto hb = headerHitBounds_.toFloat();
        if (headerPressed_)
        {
            g.setColour (th::Bg4);
            g.fillRect (hb);
        }
        else if (headerHover_)
        {
            g.setColour (th::Bg3);
            g.fillRect (hb);
        }

        // "Edit tuning" label.
        const juce::Colour labelCol = (headerHover_ || headerPressed_) ? th::Fg1 : th::Fg2;
        const juce::Font labelFont = th::uiFont (th::fs::Sm, 500);
        const juce::String labelText { "Edit tuning" };
        const float labelW = labelFont.getStringWidthFloat (labelText);
        g.setColour (labelCol);
        g.setFont (labelFont);
        const juce::Rectangle<float> labelRect ((float) kPadding,
                                                 0.0f,
                                                 labelW,
                                                 (float) kTopPadding);
        g.drawText (labelText, labelRect, juce::Justification::centredLeft, false);

        // Chevron triangle next to label. Down-pointing when expanded,
        // right-pointing when collapsed. Positioned 8 px after the actual
        // text end (label width measured dynamically so the gap is constant
        // regardless of label-string changes).
        const float chevSize = 8.0f;
        const float chevCx   = (float) kPadding + labelW + 8.0f + chevSize * 0.5f;
        const float chevCy   = (float) kTopPadding * 0.5f;
        juce::Path chev;
        if (collapsed_)
        {
            // Right-pointing (▶)
            chev.addTriangle (chevCx - chevSize * 0.5f, chevCy - chevSize * 0.5f,
                              chevCx - chevSize * 0.5f, chevCy + chevSize * 0.5f,
                              chevCx + chevSize * 0.5f, chevCy);
        }
        else
        {
            // Down-pointing (▼)
            chev.addTriangle (chevCx - chevSize * 0.5f, chevCy - chevSize * 0.5f,
                              chevCx + chevSize * 0.5f, chevCy - chevSize * 0.5f,
                              chevCx,                   chevCy + chevSize * 0.5f);
        }
        g.setColour (labelCol);
        g.fillPath (chev);
    }

    // Per-row labels (left) + dynamic readout (right) + endpoint scale labels.
    // Sesja 108 — mirror resized()'s asymmetric top/bottom padding (28 top
    // for quick-Advanced icon, 12 bottom) so paint coords match slider
    // rectangles. Top/bottom 1 px dividers above stay at the outermost edge
    // (drawn before this inset). Skipped when collapsed (slider Components
    // hidden).
    if (collapsed_)
    {
        // Still draw the Advanced icon below.
    }
    else
    {
    auto area = getLocalBounds();
    area.removeFromTop    (kTopPadding);
    area.removeFromBottom (kBottomPadding);
    area = area.reduced (kPadding, 0);

    constexpr int kSliderHeight = 22;
    constexpr int kEndpointHeight = 14;

    auto paintRow = [&](const juce::String& readout,
                        const juce::String& endpointLeft,
                        const juce::String& endpointRight) {
        auto row = area.removeFromTop (kRowHeight);

        // Top 22h — slider area + readout (title label drawn by Label child).
        auto topRow = row.removeFromTop (kSliderHeight);
        topRow.removeFromLeft (kLabelWidth);
        auto readoutArea = topRow.removeFromRight (kReadoutWidth);

        g.setFont (th::monoFont (th::fs::Sm, 400));
        g.setColour (th::Fg2);
        g.drawText (readout, readoutArea.reduced (0, 0).toFloat(),
                    juce::Justification::centredRight, true);

        // Bottom 14h — endpoint scale labels under slider track.
        auto endpointRow = row.removeFromTop (kEndpointHeight);
        endpointRow.removeFromLeft (kLabelWidth);    // align with slider area
        endpointRow.removeFromRight (kReadoutWidth);

        g.setFont (th::uiFont (th::fs::Xs, 400));
        g.setColour (th::Fg3);
        g.drawText (endpointLeft,  endpointRow.toFloat(),
                    juce::Justification::topLeft,  true);
        g.drawText (endpointRight, endpointRow.toFloat(),
                    juce::Justification::topRight, true);

        area.removeFromTop (kRowGap);
    };

    paintRow (toneReadout(),                                "Timbre",  "Harmonic");
    paintRow (editLengthReadout(),                          "Short",   "Long");
    paintRow (juce::String (allowPmSeconds_) + " s",        "0 s",     "15 s");
    paintRow (juce::String (minCutBeats_) + " beats",       "4 beats", "32 beats");
    }  // end if (collapsed_) else { ... } — sesja 108

    // Sesja 108 — quick-Advanced icon (16×16) in the top-right padding gap.
    // Three short tracks with offset thumbs — visually mirrors the slider
    // rows above, hinting at the cost-weight sliders inside Advanced. Sized
    // and coloured discreetly: Fg3 default, Fg1 on hover, Bg3/Bg4 fill on
    // hover/pressed for affordance feedback.
    if (advancedBounds_.getWidth() > 0)
    {
        const auto b = advancedBounds_.toFloat();

        if (advancedPressed_)
        {
            g.setColour (th::Bg4);
            g.fillRoundedRectangle (b.expanded (2.0f), th::r::R2);
        }
        else if (advancedHover_)
        {
            g.setColour (th::Bg3);
            g.fillRoundedRectangle (b.expanded (2.0f), th::r::R2);
        }

        const juce::Colour iconCol = (advancedHover_ || advancedPressed_) ? th::Fg1 : th::Fg3;
        g.setColour (iconCol);

        // 3 horizontal tracks of length 12 at y = 4, 8, 12 within 16×16 frame,
        // with thumb circles at 1/4, 3/4, 1/2 positions (visually varied to
        // suggest "tunable sliders" not "menu lines").
        const float ox = b.getX();
        const float oy = b.getY();
        const float trackW = 12.0f;
        const float trackThickness = 1.2f;
        const float thumbR = 1.6f;

        auto drawTrack = [&] (float y, float thumbFrac)
        {
            // Track line.
            g.fillRoundedRectangle (ox + 2.0f, oy + y - trackThickness * 0.5f,
                                    trackW, trackThickness, trackThickness * 0.5f);
            // Thumb dot.
            const float tx = ox + 2.0f + trackW * thumbFrac;
            g.fillEllipse (tx - thumbR, oy + y - thumbR, thumbR * 2.0f, thumbR * 2.0f);
        };

        drawTrack (4.0f,  0.25f);
        drawTrack (8.0f,  0.70f);
        drawTrack (12.0f, 0.45f);
    }
}

void AuditionBar::mouseMove (const juce::MouseEvent& e)
{
    const auto pos     = e.getPosition();
    const bool overAdv = advancedBounds_.contains (pos);
    // Header click area covers the strip MINUS the advanced icon (advanced
    // icon has its own hover/click).
    const bool overHdr = ! overAdv && headerHitBounds_.contains (pos);

    if (overAdv != advancedHover_)
    {
        advancedHover_ = overAdv;
        repaint (advancedBounds_.expanded (4));
    }
    if (overHdr != headerHover_)
    {
        headerHover_ = overHdr;
        repaint (headerHitBounds_);
    }

    setMouseCursor ((overAdv || overHdr) ? juce::MouseCursor::PointingHandCursor
                                         : juce::MouseCursor::NormalCursor);
}

void AuditionBar::mouseExit (const juce::MouseEvent&)
{
    if (headerHover_ || headerPressed_)
    {
        headerHover_   = false;
        headerPressed_ = false;
        repaint (headerHitBounds_);
    }
    if (advancedHover_ || advancedPressed_)
    {
        advancedHover_   = false;
        advancedPressed_ = false;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint (advancedBounds_.expanded (4));
    }
}

void AuditionBar::mouseDown (const juce::MouseEvent& e)
{
    const auto pos = e.getPosition();
    if (advancedBounds_.contains (pos))
    {
        advancedPressed_ = true;
        repaint (advancedBounds_.expanded (4));
    }
    else if (headerHitBounds_.contains (pos))
    {
        headerPressed_ = true;
        repaint (headerHitBounds_);
    }
}

void AuditionBar::mouseUp (const juce::MouseEvent& e)
{
    const auto pos        = e.getPosition();
    const bool wasAdvPress = advancedPressed_;
    const bool wasHdrPress = headerPressed_;
    advancedPressed_ = false;
    headerPressed_   = false;
    repaint (advancedBounds_.expanded (4));
    repaint (headerHitBounds_);
    if (wasAdvPress && advancedBounds_.contains (pos) && onAdvancedClicked)
        onAdvancedClicked();
    else if (wasHdrPress && headerHitBounds_.contains (pos))
    {
        setCollapsed (! collapsed_);
        if (onCollapseToggled) onCollapseToggled();
    }
}

} // namespace reamix::ui
