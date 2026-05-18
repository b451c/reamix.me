#include "ModeTabs.h"
#include "Theme.h"

namespace reamix::ui
{

namespace
{
    // plugin.css:186-192 .rx-mode-tabs — padding 0 12px, gap 2px, bottom border
    constexpr int kPadX        = 12;
    constexpr int kTabGap      = 2;

    // plugin.css:193-202 .rx-mode-tab — padding 8 12, font Md, indicator 2px.
    // Vertical padding is implicit in component height (36h - 2px indicator);
    // we don't use a separate kTabPadY constant.
    constexpr int kTabPadX     = 12;
    constexpr int kIndicatorH  = 2;

    int textWidthPx (const juce::Font& font, const juce::String& text)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, text, 0.0f, 0.0f);
        return (int) std::ceil (ga.getBoundingBox (0, -1, false).getWidth());
    }
}

ModeTabs::ModeTabs()
{
    setWantsKeyboardFocus (false);
    setInterceptsMouseClicks (true, false);
}

void ModeTabs::setMode (Mode m)
{
    if (m == mode_)
        return;
    mode_ = m;
    repaint();
}

void ModeTabs::setAutoFlag (bool flag)
{
    if (flag == autoFlag_)
        return;
    autoFlag_ = flag;
    // Region tab needs more width when AUTO badge is on — re-layout so
    // Blocks tab is pushed right correspondingly (sesja 60 smoke fix:
    // without this the badge overlapped Blocks).
    recomputeLayout();
    repaint();
}

void ModeTabs::setBlocksEnabled (bool enabled)
{
    if (enabled == blocksEnabled_)
        return;
    blocksEnabled_ = enabled;
    if (! enabled && hoverIdx_ == 1)
    {
        hoverIdx_ = -1;
        setMouseCursor (juce::MouseCursor::NormalCursor);
    }
    repaint();
}

void ModeTabs::resized()
{
    recomputeLayout();
}

void ModeTabs::recomputeLayout()
{
    using namespace reamix::theme;

    durationBounds_ = {};
    regionBounds_   = {};
    blocksBounds_   = {};

    const int h          = getHeight();
    const auto labelFont = uiFont (fs::Md, 400);
    const int tabH       = h - kIndicatorH;

    const juce::String durationLabel { "Duration" };
    const juce::String regionLabel   { "Region" };
    const juce::String blocksLabel   { "Blocks" };

    const int durationW = textWidthPx (labelFont, durationLabel) + kTabPadX * 2;

    // Region tab gets extra width for the AUTO badge when flagged.
    // Conservative measurement: 8px font "AUTO" + padding ≈ 30 px + 6 px gap.
    constexpr int kBadgeW   = 30;
    constexpr int kBadgeGap = 6;
    int regionW = textWidthPx (labelFont, regionLabel) + kTabPadX * 2;
    if (autoFlag_)
        regionW += kBadgeGap + kBadgeW;

    const int blocksW = textWidthPx (labelFont, blocksLabel) + kTabPadX * 2;

    durationBounds_ = juce::Rectangle<int> (
        kPadX, 0, durationW, tabH);
    regionBounds_ = juce::Rectangle<int> (
        durationBounds_.getRight() + kTabGap, 0, regionW, tabH);
    blocksBounds_ = juce::Rectangle<int> (
        regionBounds_.getRight() + kTabGap, 0, blocksW, tabH);
}

void ModeTabs::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds();

    // plugin.css:190 background Bg1 + L191 1px bottom Line
    g.fillAll (Bg1);
    g.setColour (Line);
    g.fillRect (bounds.withHeight (1).withY (bounds.getBottom() - 1));

    const auto labelFont = uiFont (fs::Md, 400);

    auto paintTab = [&] (const juce::Rectangle<int>& r,
                          const juce::String& label,
                          bool isActive, bool isHover, bool isDisabled,
                          int textPadLeft = 0)
    {
        juce::Colour textCol = Fg2;
        if (isDisabled)        textCol = Fg4;
        else if (isActive)     textCol = Accent;
        else if (isHover)      textCol = Fg1;

        g.setColour (textCol);
        g.setFont (labelFont);
        const juce::Rectangle<int> textArea (
            r.getX() + textPadLeft, r.getY(),
            r.getWidth() - textPadLeft, r.getHeight());
        g.drawText (label, textArea,
                    textPadLeft > 0 ? juce::Justification::centredLeft
                                     : juce::Justification::centred, false);

        if (isActive && ! isDisabled)
        {
            g.setColour (Accent);
            g.fillRect (juce::Rectangle<int> (
                r.getX(), r.getBottom(), r.getWidth(), kIndicatorH));
        }
    };

    // ── Duration tab ────────────────────────────────────────────────
    paintTab (durationBounds_, "Duration",
              mode_ == Mode::Duration, hoverIdx_ == 0, false);

    // ── Region tab (with optional AUTO badge) ──────────────────────
    {
        const bool isActive = (mode_ == Mode::Region);
        // Region label is left-aligned with kTabPadX padding so the badge
        // can sit immediately to the right with a small gap.
        paintTab (regionBounds_, "Region", isActive, hoverIdx_ == 1, false,
                  autoFlag_ ? kTabPadX : 0);

        if (autoFlag_)
        {
            // AUTO badge — plugin.css:207-213. AccentDim bg + Accent text in
            // small caps, padding 1px 4px, letter-spacing 0.1em, radius 2px.
            const auto badgeFont = uiFont (fs::Xs - 1.0f, 700);
            const juce::String badgeText { "AUTO" };
            const int regionLabelW = textWidthPx (labelFont, juce::String ("Region"));
            const int badgeTextW = textWidthPx (badgeFont, badgeText);
            constexpr int kBadgePadX = 4;
            constexpr int kBadgePadY = 1;
            const int badgeW = badgeTextW + kBadgePadX * 2 + 4;
            const int badgeH = (int) badgeFont.getHeight() + kBadgePadY * 2;
            constexpr int kBadgeGap = 6;
            const int badgeX = regionBounds_.getX() + kTabPadX + regionLabelW + kBadgeGap;
            const int badgeY = regionBounds_.getCentreY() - badgeH / 2;

            const juce::Rectangle<float> badgeRect (
                (float) badgeX, (float) badgeY, (float) badgeW, (float) badgeH);
            g.setColour (AccentDim);
            g.fillRoundedRectangle (badgeRect, 2.0f);
            g.setColour (Accent);
            g.setFont (badgeFont);
            g.drawText (badgeText,
                        juce::Rectangle<int> (badgeX, badgeY, badgeW, badgeH),
                        juce::Justification::centred, false);
        }
    }

    // ── Blocks tab ──────────────────────────────────────────────────
    paintTab (blocksBounds_, "Blocks",
              mode_ == Mode::Blocks, hoverIdx_ == 2, ! blocksEnabled_);
}

void ModeTabs::mouseMove (const juce::MouseEvent& e)
{
    int idx = -1;
    if (durationBounds_.contains (e.getPosition()))
        idx = 0;
    else if (regionBounds_.contains (e.getPosition()))
        idx = 1;
    else if (blocksBounds_.contains (e.getPosition()) && blocksEnabled_)
        idx = 2;

    if (idx != hoverIdx_)
    {
        hoverIdx_ = idx;
        setMouseCursor (idx >= 0 ? juce::MouseCursor::PointingHandCursor
                                 : juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void ModeTabs::mouseExit (const juce::MouseEvent&)
{
    if (hoverIdx_ != -1)
    {
        hoverIdx_ = -1;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void ModeTabs::mouseDown (const juce::MouseEvent& e)
{
    auto pick = [this] (Mode m)
    {
        if (mode_ != m)
        {
            mode_ = m;
            repaint();
            if (onChange) onChange (m);
        }
    };

    if (durationBounds_.contains (e.getPosition()))           { pick (Mode::Duration); return; }
    if (regionBounds_.contains (e.getPosition()))              { pick (Mode::Region);   return; }
    if (blocksEnabled_ && blocksBounds_.contains (e.getPosition()))
                                                                { pick (Mode::Blocks);   return; }
}

} // namespace reamix::ui
