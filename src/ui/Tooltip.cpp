#include "Tooltip.h"

#include "Theme.h"

namespace reamix::ui
{

namespace
{
    int textWidth (const juce::Font& font, const juce::String& s)
    {
        if (s.isEmpty()) return 0;
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, s, 0.0f, 0.0f);
        return (int) std::ceil (ga.getBoundingBox (0, -1, false).getWidth());
    }
}

Tooltip::Tooltip()
{
    setInterceptsMouseClicks (false, false);
    setWantsKeyboardFocus (false);
    setVisible (false);
}

void Tooltip::setData (Data d)
{
    data_ = std::move (d);
    repaint();
}

void Tooltip::showAt (juce::Point<int> anchor)
{
    auto bounds = computePreferredBounds();
    bounds.setPosition (anchor);

    if (auto* parent = getParentComponent())
    {
        const auto pb = parent->getLocalBounds().reduced (4);
        if (bounds.getRight()  > pb.getRight())  bounds.setX (pb.getRight()  - bounds.getWidth());
        if (bounds.getBottom() > pb.getBottom()) bounds.setY (pb.getBottom() - bounds.getHeight());
        bounds.setX (juce::jmax (pb.getX(), bounds.getX()));
        bounds.setY (juce::jmax (pb.getY(), bounds.getY()));
    }

    setBounds (bounds);
    setVisible (true);
    toFront (false);
    repaint();
}

void Tooltip::hide()
{
    setVisible (false);
}

juce::Rectangle<int> Tooltip::computePreferredBounds() const
{
    using namespace reamix::theme;

    const auto titleFont = uiFont (fs::Xs, 400);
    const auto rowFont   = uiFont (fs::Sm, 400);
    const auto rowFontV  = monoFont (fs::Sm, 400);
    const auto hintFont  = uiFont (10.0f, 400);

    int contentW = kMinWidth - 2 * kPadX;
    if (data_.title.isNotEmpty())
        contentW = juce::jmax (contentW, textWidth (titleFont, data_.title.toUpperCase()));
    for (const auto& rr : data_.rows)
        contentW = juce::jmax (contentW,
                                textWidth (rowFont, rr.k) + kRowGap + textWidth (rowFontV, rr.v));
    if (data_.hint.isNotEmpty())
        contentW = juce::jmax (contentW, textWidth (hintFont, data_.hint));

    int h = kPadY;
    if (data_.title.isNotEmpty()) h += kTitleH + kTitleMargBot;
    h += (int) data_.rows.size() * kRowHeight;
    if (data_.qbarPct > 0.0f) h += 6 /*margin-top*/ + 3 /*bar*/;
    if (data_.hint.isNotEmpty()) h += 6 /*margin-top*/ + 11;
    h += kPadY;

    return juce::Rectangle<int> (0, 0, contentW + 2 * kPadX, h);
}

void Tooltip::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds().toFloat();

    // Shadow (ElevPop: offset Y 4, blur 4, alpha 0.55) — plugin.css:80
    juce::DropShadow sh (juce::Colours::black.withAlpha (0.55f),
                          ElevPop.blur, { 0, ElevPop.offsetY });
    sh.drawForRectangle (g, getLocalBounds());

    // Bg-0 + LineStrong border + R2 radius — plugin.css:555-560
    g.setColour (Bg0);
    g.fillRoundedRectangle (bounds, r::R2);
    g.setColour (LineStrong);
    g.drawRoundedRectangle (bounds.reduced (0.5f), r::R2, 1.0f);

    auto inner = getLocalBounds().reduced (kPadX, kPadY);

    // Title — plugin.css:568-571
    if (data_.title.isNotEmpty())
    {
        g.setColour (Fg3);
        g.setFont (uiFont (fs::Xs, 400));
        auto row = inner.removeFromTop (kTitleH);
        g.drawText (data_.title.toUpperCase(), row,
                    juce::Justification::centredLeft, true);
        inner.removeFromTop (kTitleMargBot);
    }

    // Rows — plugin.css:572-574
    for (const auto& rr : data_.rows)
    {
        auto row = inner.removeFromTop (kRowHeight);
        g.setColour (Fg2);
        g.setFont (uiFont (fs::Sm, 400));
        g.drawText (rr.k, row, juce::Justification::centredLeft, true);
        g.setColour (Fg0);
        g.setFont (monoFont (fs::Sm, 400));
        g.drawText (rr.v, row, juce::Justification::centredRight, true);
    }

    // Quality bar — plugin.css:575-579
    if (data_.qbarPct > 0.0f)
    {
        inner.removeFromTop (6);
        auto bar = inner.removeFromTop (3);
        g.setColour (Bg3);
        g.fillRoundedRectangle (bar.toFloat(), 1.5f);
        const float fillW = bar.toFloat().getWidth()
                          * juce::jlimit (0.0f, 1.0f, data_.qbarPct);
        if (data_.qbarColour.getAlpha() > 0)
        {
            g.setColour (data_.qbarColour);
            g.fillRoundedRectangle (bar.toFloat().withWidth (fillW), 1.5f);
        }
    }

    // Hint — plugin.css:580
    if (data_.hint.isNotEmpty())
    {
        inner.removeFromTop (6);
        auto row = inner.removeFromTop (11);
        g.setColour (Fg3);
        g.setFont (uiFont (10.0f, 400));
        g.drawText (data_.hint, row, juce::Justification::centredLeft, true);
    }
}

} // namespace reamix::ui
