#include "SupportPopover.h"

#include "Theme.h"

namespace reamix::ui
{

namespace th = reamix::theme;

SupportPopover::SupportPopover()
{
    setWantsKeyboardFocus (true);
    setInterceptsMouseClicks (true, false);
    setVisible (false);
}

SupportPopover::~SupportPopover() = default;

void SupportPopover::show()
{
    if (auto* parent = getParentComponent())
        setBounds (parent->getLocalBounds());

    layOutHitRegions();
    setVisible (true);
    toFront (true);
    grabKeyboardFocus();
    if (! isCurrentlyModal())
        enterModalState (false, nullptr, false);
    repaint();
}

void SupportPopover::hideMe()
{
    if (isCurrentlyModal())
        exitModalState (0);
    setVisible (false);
    hover_   = HitRow::None;
    pressed_ = HitRow::None;
}

void SupportPopover::inputAttemptWhenModal()
{
    hideMe();
}

void SupportPopover::resized()
{
    layOutHitRegions();
}

void SupportPopover::layOutHitRegions()
{
    const int bodyX = getLocalBounds().getRight() - kAnchorRight - kBodyWidth;
    const int bodyY = kAnchorTop;
    const int bodyH = kPadding
                    + (kTitleH + kTitleMargB)
                    + 4 * kRowH + 3 * kRowGap
                    + kPadding;

    bodyRect_ = { bodyX, bodyY, kBodyWidth, bodyH };

    auto inner = bodyRect_.reduced (kPadding);
    inner.removeFromTop (kTitleH + kTitleMargB);

    rowGitHub_ = inner.removeFromTop (kRowH);
    inner.removeFromTop (kRowGap);
    rowKoFi_   = inner.removeFromTop (kRowH);
    inner.removeFromTop (kRowGap);
    rowBuyMe_  = inner.removeFromTop (kRowH);
    inner.removeFromTop (kRowGap);
    rowPayPal_ = inner.removeFromTop (kRowH);
}

SupportPopover::HitRow SupportPopover::rowAt (juce::Point<int> p) const
{
    if (rowGitHub_.contains (p)) return HitRow::GitHubSponsors;
    if (rowKoFi_  .contains (p)) return HitRow::KoFi;
    if (rowBuyMe_ .contains (p)) return HitRow::BuyMeCoffee;
    if (rowPayPal_.contains (p)) return HitRow::PayPal;
    return HitRow::None;
}

void SupportPopover::paint (juce::Graphics& g)
{
    // Soft scrim — dims everything outside the body so the popover reads as
    // "modal layer above the main canvas" without obstructing visibility.
    g.fillAll (juce::Colour::fromFloatRGBA (0.0f, 0.0f, 0.0f, 0.35f));

    // Drop shadow below the body.
    {
        juce::DropShadow shadow (juce::Colours::black.withAlpha (0.4f), 18, { 0, 6 });
        juce::Path bodyPath;
        bodyPath.addRoundedRectangle (bodyRect_.toFloat(), th::r::R3);
        shadow.drawForPath (g, bodyPath);
    }

    // Body background + border.
    g.setColour (th::Bg2);
    g.fillRoundedRectangle (bodyRect_.toFloat(), th::r::R3);
    g.setColour (th::Line);
    g.drawRoundedRectangle (bodyRect_.toFloat(), th::r::R3, 1.0f);

    // Header title.
    auto inner = bodyRect_.reduced (kPadding);
    auto titleRect = inner.removeFromTop (kTitleH + kTitleMargB);
    g.setColour (th::Fg3);
    g.setFont (th::uiFont (th::fs::Xs, 600));
    g.drawText ("SUPPORT REAMIX.ME",
                titleRect.toFloat(),
                juce::Justification::topLeft, false);

    // Row painter: filled rounded background on hover/pressed, label + sub.
    auto paintRow = [&] (const juce::Rectangle<int>& rect,
                         HitRow row,
                         const char* primary,
                         const char* subtitle)
    {
        const bool isHover   = (hover_   == row);
        const bool isPressed = (pressed_ == row);

        if (isPressed)
        {
            g.setColour (th::Bg4);
            g.fillRoundedRectangle (rect.toFloat(), th::r::R2);
        }
        else if (isHover)
        {
            g.setColour (th::Bg3);
            g.fillRoundedRectangle (rect.toFloat(), th::r::R2);
        }

        auto rowInner = rect.reduced (10, 0);

        // Right-side chevron hint (▸) — affordance that row opens external link.
        const auto chevRect = rowInner.removeFromRight (16);
        g.setColour (isHover || isPressed ? th::Fg1 : th::Fg3);
        {
            const float cx = (float) chevRect.getCentreX();
            const float cy = (float) chevRect.getCentreY();
            const float s  = 5.0f;
            juce::Path chev;
            chev.addTriangle (cx - s * 0.4f, cy - s,
                              cx - s * 0.4f, cy + s,
                              cx + s * 0.6f, cy);
            g.fillPath (chev);
        }

        // Two-line label with explicit Y bands to prevent visual overlap.
        // Primary (Md 500 Fg1) occupies top 20 px (text baseline ~16);
        // subtitle (Sm 400 Fg3) occupies bottom 18 px with a 2 px clear gap.
        const int rowH       = rowInner.getHeight();
        const int primaryH   = 20;
        const int gapBetween = 2;
        const int subtitleH  = 16;
        const int blockH     = primaryH + gapBetween + subtitleH;
        const int topPad     = juce::jmax (0, (rowH - blockH) / 2);

        const auto primaryRect = rowInner.withY (rowInner.getY() + topPad)
                                         .withHeight (primaryH);
        const auto subRect     = rowInner.withY (rowInner.getY() + topPad
                                                  + primaryH + gapBetween)
                                         .withHeight (subtitleH);

        g.setColour (isHover || isPressed ? th::Fg0 : th::Fg1);
        g.setFont (th::uiFont (th::fs::Md, 500));
        g.drawText (primary, primaryRect.toFloat(),
                    juce::Justification::centredLeft, false);

        g.setColour (th::Fg3);
        g.setFont (th::uiFont (th::fs::Sm, 400));
        g.drawText (subtitle, subRect.toFloat(),
                    juce::Justification::centredLeft, false);
    };

    paintRow (rowGitHub_, HitRow::GitHubSponsors, "GitHub Sponsors",
                                                  "github.com/sponsors/b451c");
    paintRow (rowKoFi_,   HitRow::KoFi,           "Ko-fi",
                                                  "ko-fi.com/quickmd");
    paintRow (rowBuyMe_,  HitRow::BuyMeCoffee,    "Buy Me a Coffee",
                                                  "buymeacoffee.com/bsroczynskh");
    paintRow (rowPayPal_, HitRow::PayPal,         "PayPal",
                                                  "paypal.me/b451c");
}

void SupportPopover::mouseDown (const juce::MouseEvent& e)
{
    const auto p   = e.getPosition();
    const auto row = rowAt (p);
    if (row == HitRow::None && ! bodyRect_.contains (p))
    {
        hideMe();
        return;
    }
    pressed_ = row;
    repaint();
}

void SupportPopover::mouseUp (const juce::MouseEvent& e)
{
    const auto p   = e.getPosition();
    const auto row = rowAt (p);
    const auto was = pressed_;
    pressed_ = HitRow::None;
    repaint();

    if (was == HitRow::None || row != was) return;

    juce::URL target;
    switch (row)
    {
        case HitRow::GitHubSponsors:
            target = juce::URL ("https://github.com/sponsors/b451c"); break;
        case HitRow::KoFi:
            target = juce::URL ("https://ko-fi.com/quickmd"); break;
        case HitRow::BuyMeCoffee:
            target = juce::URL ("https://buymeacoffee.com/bsroczynskh"); break;
        case HitRow::PayPal:
            target = juce::URL ("https://www.paypal.com/paypalme/b451c"); break;
        default: return;
    }
    target.launchInDefaultBrowser();
    hideMe();
}

void SupportPopover::mouseMove (const juce::MouseEvent& e)
{
    const auto row = rowAt (e.getPosition());
    if (row != hover_)
    {
        hover_ = row;
        setMouseCursor (row != HitRow::None
                        ? juce::MouseCursor::PointingHandCursor
                        : juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void SupportPopover::mouseExit (const juce::MouseEvent&)
{
    if (hover_ != HitRow::None || pressed_ != HitRow::None)
    {
        hover_   = HitRow::None;
        pressed_ = HitRow::None;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint();
    }
}

bool SupportPopover::keyPressed (const juce::KeyPress& k)
{
    if (k == juce::KeyPress::escapeKey)
    {
        hideMe();
        return true;
    }
    return false;
}

} // namespace reamix::ui
