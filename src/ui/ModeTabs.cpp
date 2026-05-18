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

    // BETA badge — static, always shown on Region + Blocks. Signals "less mature
    // mode than Duration" via neutral Fg3/Bg3 colours (cf. AUTO badge below which
    // uses Accent for dynamic plugin-state). 30 px conservative width matches
    // AUTO badge constants for visual rhythm.
    constexpr int kBetaW   = 30;
    constexpr int kBetaGap = 6;

    // AUTO badge — dynamic, only when autoFlag_ true (REAPER time-selection triggered).
    constexpr int kAutoW   = 30;
    constexpr int kAutoGap = 6;

    int regionW = textWidthPx (labelFont, regionLabel) + kTabPadX * 2
                  + kBetaGap + kBetaW;
    if (autoFlag_)
        regionW += kAutoGap + kAutoW;

    const int blocksW = textWidthPx (labelFont, blocksLabel) + kTabPadX * 2
                        + kBetaGap + kBetaW;

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

    // Shared helper to draw a small badge (BETA / AUTO style) at a given x/y
    // inside a tab. Returns the rect drawn so caller can chain badges.
    auto drawBadge = [&] (int x, int centreY,
                          const juce::String& text,
                          juce::Colour bg, juce::Colour fg)
        -> juce::Rectangle<int>
    {
        const auto badgeFont = uiFont (fs::Xs - 1.0f, 700);
        const int badgeTextW = textWidthPx (badgeFont, text);
        constexpr int kBadgePadX = 4;
        constexpr int kBadgePadY = 1;
        const int badgeW = badgeTextW + kBadgePadX * 2 + 4;
        const int badgeH = (int) badgeFont.getHeight() + kBadgePadY * 2;
        const int badgeY = centreY - badgeH / 2;

        const juce::Rectangle<float> badgeRect (
            (float) x, (float) badgeY, (float) badgeW, (float) badgeH);
        g.setColour (bg);
        g.fillRoundedRectangle (badgeRect, 2.0f);
        g.setColour (fg);
        g.setFont (badgeFont);
        g.drawText (text,
                    juce::Rectangle<int> (x, badgeY, badgeW, badgeH),
                    juce::Justification::centred, false);
        return { x, badgeY, badgeW, badgeH };
    };

    constexpr int kBadgeGap = 6;

    // ── Region tab (BETA always + AUTO when flagged) ───────────────
    {
        const bool isActive = (mode_ == Mode::Region);
        // Region label is left-aligned with kTabPadX padding so badges
        // sit immediately to the right with a small gap.
        paintTab (regionBounds_, "Region", isActive, hoverIdx_ == 1, false,
                  kTabPadX);

        const int regionLabelW = textWidthPx (labelFont, juce::String ("Region"));
        int badgeX = regionBounds_.getX() + kTabPadX + regionLabelW + kBadgeGap;

        // BETA badge first (always) — neutral Bg3/Fg3 normally; brightens to
        // Bg4/Fg2 ONLY on hover (not on active/click state).
        const bool regionHover = (hoverIdx_ == 1);
        const juce::Colour regionBetaBg = regionHover ? Bg4 : Bg3;
        const juce::Colour regionBetaFg = regionHover ? Fg2 : Fg3;
        const auto betaRect = drawBadge (badgeX, regionBounds_.getCentreY(),
                                          "BETA", regionBetaBg, regionBetaFg);
        badgeX = betaRect.getRight() + kBadgeGap;

        // AUTO badge after BETA (when flagged) — Accent colours.
        if (autoFlag_)
            drawBadge (badgeX, regionBounds_.getCentreY(),
                        "AUTO", AccentDim, Accent);
    }

    // ── Blocks tab (with BETA badge) ────────────────────────────────
    {
        const bool isActive   = (mode_ == Mode::Blocks);
        const bool isDisabled = ! blocksEnabled_;
        paintTab (blocksBounds_, "Blocks", isActive, hoverIdx_ == 2,
                  isDisabled, kTabPadX);

        const int blocksLabelW = textWidthPx (labelFont, juce::String ("Blocks"));
        const int badgeX = blocksBounds_.getX() + kTabPadX + blocksLabelW
                           + kBadgeGap;

        // BETA badge dims when tab is disabled so it doesn't draw focus.
        // Brightens on hover only (not active/click). When disabled, no hover
        // brightness (hoverIdx_ already -1 for disabled tab per mouseMove).
        const bool blocksHover = (hoverIdx_ == 2);
        juce::Colour betaBg, betaFg;
        if (isDisabled)        { betaBg = Bg2; betaFg = Fg4; }
        else if (blocksHover)  { betaBg = Bg4; betaFg = Fg2; }
        else                   { betaBg = Bg3; betaFg = Fg3; }
        drawBadge (badgeX, blocksBounds_.getCentreY(), "BETA", betaBg, betaFg);
    }
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
