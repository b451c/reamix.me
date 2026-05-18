#include "SettingsPopover.h"

#include "Theme.h"

namespace reamix::ui
{

SettingsPopover::SettingsPopover()
{
    setWantsKeyboardFocus (true);
    setInterceptsMouseClicks (true, false);
    setVisible (false);
}

SettingsPopover::~SettingsPopover() = default;

void SettingsPopover::setDocked (bool d)
{
    if (docked_ == d) return;
    docked_ = d;
    repaint();
}

void SettingsPopover::setShowBeats (bool s)
{
    if (showBeats_ == s) return;
    showBeats_ = s;
    repaint();
}

void SettingsPopover::setSnapRegion (SnapRegion r)
{
    if (snapRegion_ == r) return;
    snapRegion_ = r;
    repaint();
}

void SettingsPopover::setInsertSpliceMarkers (bool s)
{
    if (insertSpliceMarkers_ == s) return;
    insertSpliceMarkers_ = s;
    repaint();
}

void SettingsPopover::setInsertRenderRegion (bool s)
{
    if (insertRenderRegion_ == s) return;
    insertRenderRegion_ = s;
    repaint();
}

void SettingsPopover::setCacheStats (juce::int64 totalBytes, int entryCount)
{
    if (entryCount <= 0 || totalBytes <= 0)
    {
        cacheStatsText_ = "Empty";
    }
    else
    {
        // Pretty-print bytes → MB (or KB for tiny caches). Single-decimal
        // for MB so stat row stays compact at 260 px.
        const double mb = (double) totalBytes / (1024.0 * 1024.0);
        const juce::String sizeText = (mb >= 1.0)
            ? juce::String (mb, 1) + " MB"
            : juce::String ((double) totalBytes / 1024.0, 0) + " KB";
        cacheStatsText_ = sizeText
            + juce::String::fromUTF8 (" \xC2\xB7 ")
            + juce::String (entryCount)
            + juce::String (entryCount == 1 ? " track" : " tracks");
    }
    repaint();
}

void SettingsPopover::show()
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

void SettingsPopover::hideMe()
{
    if (isCurrentlyModal())
        exitModalState (0);
    setVisible (false);
    dockBtnHover_           = false;
    dockBtnPressed_         = false;
    advancedBtnHover_       = false;
    advancedBtnPressed_     = false;
    beatsRowHover_          = false;
    beatsRowPressed_        = false;
    snapRowHover_           = false;
    snapRowPressed_         = false;
    insertSpliceRowHover_   = false;
    insertSpliceRowPressed_ = false;
    insertRegionRowHover_   = false;
    insertRegionRowPressed_ = false;
    cacheClearHover_        = false;
    cacheClearPressed_      = false;
    cacheRevealHover_       = false;
    cacheRevealPressed_     = false;
}

void SettingsPopover::inputAttemptWhenModal()
{
    hideMe();
}

void SettingsPopover::resized()
{
    layOutHitRegions();
}

int SettingsPopover::computeBodyHeight()
{
    // DISPLAY section: Show beats + Snap region.
    // INSERT section (sesja 100b DEV-049): Splice markers + Render region.
    // WINDOW section: Dock toggle + Advanced... (ADR-097 sesja 107).
    // CACHE section (ADR-053): Stats row + Clear/Reveal button row.
    return kPadding
         + (kTitleH + kTitleMargB)
         + kBeatsRowH + kRowGap + kSnapRowH
         + kSectionGap
         + (kTitleH + kTitleMargB) + kInsertRowH + kRowGap + kInsertRowH
         + kSectionGap
         + (kTitleH + kTitleMargB) + kBtnH + kRowGap + kBtnH
         + kSectionGap
         + (kTitleH + kTitleMargB) + kCacheStatsH + kRowGap + kCacheBtnH
         + kPadding;
}

void SettingsPopover::layOutHitRegions()
{
    const int bodyX = getLocalBounds().getRight() - kAnchorRight - kBodyWidth;
    const int bodyY = kAnchorTop;
    const int bodyH = computeBodyHeight();

    auto body = juce::Rectangle<int> (bodyX, bodyY, kBodyWidth, bodyH);

    auto inner = body.reduced (kPadding);

    // ── DISPLAY section ──
    inner.removeFromTop (kTitleH + kTitleMargB);
    beatsRowRect_ = inner.removeFromTop (kBeatsRowH);
    inner.removeFromTop (kRowGap);
    snapRowRect_  = inner.removeFromTop (kSnapRowH);

    inner.removeFromTop (kSectionGap);

    // ── INSERT section (sesja 100b — DEV-049) ──
    inner.removeFromTop (kTitleH + kTitleMargB);
    insertSpliceRowRect_ = inner.removeFromTop (kInsertRowH);
    inner.removeFromTop (kRowGap);
    insertRegionRowRect_ = inner.removeFromTop (kInsertRowH);

    inner.removeFromTop (kSectionGap);

    // ── WINDOW section ──
    inner.removeFromTop (kTitleH + kTitleMargB);
    dockBtnRect_     = inner.removeFromTop (kBtnH);
    inner.removeFromTop (kRowGap);
    advancedBtnRect_ = inner.removeFromTop (kBtnH);

    inner.removeFromTop (kSectionGap);

    // ── CACHE section (ADR-053) ──
    inner.removeFromTop (kTitleH + kTitleMargB);
    cacheStatsRect_ = inner.removeFromTop (kCacheStatsH);
    inner.removeFromTop (kRowGap);
    auto btnRow = inner.removeFromTop (kCacheBtnH);
    const int btnW = (btnRow.getWidth() - kCacheBtnGap) / 2;
    cacheClearRect_  = btnRow.removeFromLeft (btnW);
    btnRow.removeFromLeft (kCacheBtnGap);
    cacheRevealRect_ = btnRow;
}

// ── Paint ───────────────────────────────────────────────────────────

void SettingsPopover::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    if (! isVisible()) return;

    const int bodyX = getLocalBounds().getRight() - kAnchorRight - kBodyWidth;
    const int bodyY = kAnchorTop;
    const int bodyH = computeBodyHeight();

    auto body = juce::Rectangle<int> (bodyX, bodyY, kBodyWidth, bodyH);

    // Shadow — plugin.css:611 ElevPop.
    juce::DropShadow sh (juce::Colours::black.withAlpha (0.55f),
                          ElevPop.blur, { 0, ElevPop.offsetY });
    sh.drawForRectangle (g, body);

    // Bg-1 + LineStrong border + R3 — plugin.css:604-613.
    g.setColour (Bg1);
    g.fillRoundedRectangle (body.toFloat(), r::R3);
    g.setColour (LineStrong);
    g.drawRoundedRectangle (body.toFloat().reduced (0.5f), r::R3, 1.0f);

    auto inner = body.reduced (kPadding);

    const auto drawSectionTitle = [&] (const juce::String& title)
    {
        g.setColour (Fg3);
        g.setFont (uiFont (fs::Xs, 400));
        auto row = inner.removeFromTop (kTitleH);
        g.drawText (title, row, juce::Justification::centredLeft, true);
        inner.removeFromTop (kTitleMargB);
    };

    // ── DISPLAY section ──
    drawSectionTitle ("DISPLAY");
    {
        // Reserve the row that layOutHitRegions captured into beatsRowRect_.
        // We don't recompute geometry here — paint trusts the laid-out rect.
        inner.removeFromTop (kBeatsRowH);

        // Row background — mirrors the dock-button rest/hover/press states
        // for visual consistency. The whole row is the click target.
        juce::Colour bg     = Bg4;
        juce::Colour bgEdge = Line;
        if (beatsRowPressed_) { bg = Bg3; bgEdge = LineStrong; }
        else if (beatsRowHover_) { bg = Bg5; bgEdge = LineStrong; }

        const auto rowf = beatsRowRect_.toFloat();
        g.setColour (bg);
        g.fillRoundedRectangle (rowf, r::R3);
        g.setColour (bgEdge);
        g.drawRoundedRectangle (rowf.reduced (0.5f), r::R3, 1.0f);

        // Label left-aligned with 10 px gutter; toggle indicator right-aligned.
        constexpr int kGutter   = 10;
        constexpr int kSwatchW  = 28;
        constexpr int kSwatchH  = 14;

        g.setColour (Fg1);
        g.setFont (uiFont (fs::Md, 500));
        const auto labelArea = beatsRowRect_.withTrimmedLeft (kGutter)
                                            .withTrimmedRight (kGutter + kSwatchW + 6);
        g.drawText ("Show beats", labelArea, juce::Justification::centredLeft, true);

        // Toggle pill — Accent fill when ON, Bg5 + Line border when OFF.
        const int swatchX = beatsRowRect_.getRight() - kGutter - kSwatchW;
        const int swatchY = beatsRowRect_.getY() + (beatsRowRect_.getHeight() - kSwatchH) / 2;
        const juce::Rectangle<int> swatch (swatchX, swatchY, kSwatchW, kSwatchH);
        const auto swatchf = swatch.toFloat();
        const float radius = kSwatchH * 0.5f;

        g.setColour (showBeats_ ? Accent : Bg2);
        g.fillRoundedRectangle (swatchf, radius);
        g.setColour (showBeats_ ? AccentLo : LineStrong);
        g.drawRoundedRectangle (swatchf.reduced (0.5f), radius, 1.0f);

        // Knob — circle inside the pill, slides left↔right with state.
        const float knobR = (float) (kSwatchH - 4) * 0.5f;
        const float knobCY = (float) swatch.getCentreY();
        const float knobCX = showBeats_
            ? (float) swatch.getRight() - 2.0f - knobR
            : (float) swatch.getX()     + 2.0f + knobR;
        g.setColour (showBeats_ ? Bg1 : Fg2);
        g.fillEllipse (knobCX - knobR, knobCY - knobR, knobR * 2.0f, knobR * 2.0f);
    }

    // ── Snap region row (sesja 60) ──
    inner.removeFromTop (kRowGap);
    inner.removeFromTop (kSnapRowH);
    {
        juce::Colour bg     = Bg4;
        juce::Colour bgEdge = Line;
        if (snapRowPressed_)   { bg = Bg3; bgEdge = LineStrong; }
        else if (snapRowHover_) { bg = Bg5; bgEdge = LineStrong; }

        const auto rowf = snapRowRect_.toFloat();
        g.setColour (bg);
        g.fillRoundedRectangle (rowf, r::R3);
        g.setColour (bgEdge);
        g.drawRoundedRectangle (rowf.reduced (0.5f), r::R3, 1.0f);

        constexpr int kGutter = 10;
        const auto labelArea = snapRowRect_.withTrimmedLeft (kGutter)
                                            .withTrimmedRight (kGutter + 80);
        g.setColour (Fg1);
        g.setFont (uiFont (fs::Md, 500));
        g.drawText ("Snap region to", labelArea,
                    juce::Justification::centredLeft, true);

        const juce::String stateText = (snapRegion_ == SnapRegion::Beats)
            ? juce::String ("Beats")
            : (snapRegion_ == SnapRegion::Downbeats)
                ? juce::String ("Downbeats")
                : juce::String ("Off");
        const auto valueArea = snapRowRect_.withTrimmedRight (kGutter)
                                            .withTrimmedLeft (snapRowRect_.getWidth() - 90);
        g.setColour (snapRegion_ == SnapRegion::Off ? Fg2 : Accent);
        g.setFont (uiFont (fs::Md, 500));
        g.drawText (stateText, valueArea,
                    juce::Justification::centredRight, true);
    }

    // Splice flexibility row removed sesja 68 ADR-057 — see DEV-040 for the
    // β-model that supersedes the junction ±W search this row used to control.

    inner.removeFromTop (kSectionGap);

    // ── INSERT section (sesja 100b — DEV-049) ──
    drawSectionTitle ("INSERT");

    // Helper: paint an Accent-pill toggle row matching the "Show beats"
    // visual pattern. on=current state; hover/pressed flip the row bg.
    const auto drawToggleRow = [&] (const juce::Rectangle<int>& rect,
                                     const juce::String& label,
                                     bool on, bool hover, bool pressed)
    {
        juce::Colour bg     = Bg4;
        juce::Colour bgEdge = Line;
        if (pressed)    { bg = Bg3; bgEdge = LineStrong; }
        else if (hover) { bg = Bg5; bgEdge = LineStrong; }

        const auto rowf = rect.toFloat();
        g.setColour (bg);
        g.fillRoundedRectangle (rowf, r::R3);
        g.setColour (bgEdge);
        g.drawRoundedRectangle (rowf.reduced (0.5f), r::R3, 1.0f);

        constexpr int kGutter   = 10;
        constexpr int kSwatchW  = 28;
        constexpr int kSwatchH  = 14;

        g.setColour (Fg1);
        g.setFont (uiFont (fs::Md, 500));
        const auto labelArea = rect.withTrimmedLeft (kGutter)
                                    .withTrimmedRight (kGutter + kSwatchW + 6);
        g.drawText (label, labelArea, juce::Justification::centredLeft, true);

        const int swatchX = rect.getRight() - kGutter - kSwatchW;
        const int swatchY = rect.getY() + (rect.getHeight() - kSwatchH) / 2;
        const juce::Rectangle<int> swatch (swatchX, swatchY, kSwatchW, kSwatchH);
        const auto swatchf = swatch.toFloat();
        const float radius = kSwatchH * 0.5f;

        g.setColour (on ? Accent : Bg2);
        g.fillRoundedRectangle (swatchf, radius);
        g.setColour (on ? AccentLo : LineStrong);
        g.drawRoundedRectangle (swatchf.reduced (0.5f), radius, 1.0f);

        const float knobR = (float) (kSwatchH - 4) * 0.5f;
        const float knobCY = (float) swatch.getCentreY();
        const float knobCX = on
            ? (float) swatch.getRight() - 2.0f - knobR
            : (float) swatch.getX()     + 2.0f + knobR;
        g.setColour (on ? Bg1 : Fg2);
        g.fillEllipse (knobCX - knobR, knobCY - knobR, knobR * 2.0f, knobR * 2.0f);
    };

    inner.removeFromTop (kInsertRowH);
    drawToggleRow (insertSpliceRowRect_, "Splice markers",
                    insertSpliceMarkers_,
                    insertSpliceRowHover_, insertSpliceRowPressed_);

    inner.removeFromTop (kRowGap);
    inner.removeFromTop (kInsertRowH);
    drawToggleRow (insertRegionRowRect_, "Render region",
                    insertRenderRegion_,
                    insertRegionRowHover_, insertRegionRowPressed_);

    inner.removeFromTop (kSectionGap);

    // ── WINDOW section ──
    drawSectionTitle ("WINDOW");

    const auto drawWindowBtn = [&] (const juce::Rectangle<int>& rect,
                                     const juce::String& label,
                                     bool hover, bool pressed)
    {
        // Default button styling matches LookAndFeelReamix::drawButtonBackground
        // rest state: Bg4 + Line border + R3 radius + Fg1 text. Hover bumps
        // to Bg5 + LineStrong; pressed darkens to Bg3.
        juce::Colour bg      = Bg4;
        juce::Colour bgEdge  = Line;
        if (pressed)    { bg = Bg3; bgEdge = LineStrong; }
        else if (hover) { bg = Bg5; bgEdge = LineStrong; }

        g.setColour (bg);
        g.fillRoundedRectangle (rect.toFloat(), r::R3);
        g.setColour (bgEdge);
        g.drawRoundedRectangle (rect.toFloat().reduced (0.5f), r::R3, 1.0f);

        g.setColour (Fg1);
        g.setFont (uiFont (fs::Md, 500));
        g.drawText (label, rect, juce::Justification::centred, true);
    };

    {
        inner.removeFromTop (kBtnH);
        const juce::String dockLabel = docked_ ? juce::String ("Undock window")
                                                : juce::String ("Dock to REAPER");
        drawWindowBtn (dockBtnRect_, dockLabel,
                       dockBtnHover_, dockBtnPressed_);
    }
    inner.removeFromTop (kRowGap);
    {
        inner.removeFromTop (kBtnH);
        drawWindowBtn (advancedBtnRect_, "Advanced...",
                       advancedBtnHover_, advancedBtnPressed_);
    }

    inner.removeFromTop (kSectionGap);

    // ── CACHE section (ADR-053) ──
    drawSectionTitle ("CACHE");
    {
        // Stats row (read-only).
        inner.removeFromTop (kCacheStatsH);
        g.setColour (Fg2);
        g.setFont (uiFont (fs::Sm, 400));
        g.drawText (cacheStatsText_, cacheStatsRect_,
                    juce::Justification::centredLeft, true);

        // Button row.
        inner.removeFromTop (kRowGap);
        inner.removeFromTop (kCacheBtnH);

        const auto drawCacheBtn = [&] (const juce::Rectangle<int>& rect,
                                        const juce::String& label,
                                        bool hover, bool pressed)
        {
            juce::Colour bg     = Bg4;
            juce::Colour bgEdge = Line;
            if (pressed)    { bg = Bg3; bgEdge = LineStrong; }
            else if (hover) { bg = Bg5; bgEdge = LineStrong; }

            g.setColour (bg);
            g.fillRoundedRectangle (rect.toFloat(), r::R3);
            g.setColour (bgEdge);
            g.drawRoundedRectangle (rect.toFloat().reduced (0.5f), r::R3, 1.0f);

            g.setColour (Fg1);
            g.setFont (uiFont (fs::Sm, 500));
            g.drawText (label, rect, juce::Justification::centred, true);
        };

        drawCacheBtn (cacheClearRect_,  "Clear",         cacheClearHover_,  cacheClearPressed_);
        drawCacheBtn (cacheRevealRect_, "Show in Finder", cacheRevealHover_, cacheRevealPressed_);
    }
}

// ── Input ──────────────────────────────────────────────────────────

void SettingsPopover::mouseDown (const juce::MouseEvent& e)
{
    if (! isVisible()) return;

    const auto p = e.getPosition();

    if (beatsRowRect_.contains (p))
    {
        beatsRowPressed_ = true;
        repaint (beatsRowRect_);
        return;
    }
    if (snapRowRect_.contains (p))
    {
        snapRowPressed_ = true;
        repaint (snapRowRect_);
        return;
    }
    if (insertSpliceRowRect_.contains (p))
    {
        insertSpliceRowPressed_ = true;
        repaint (insertSpliceRowRect_);
        return;
    }
    if (insertRegionRowRect_.contains (p))
    {
        insertRegionRowPressed_ = true;
        repaint (insertRegionRowRect_);
        return;
    }
    if (dockBtnRect_.contains (p))
    {
        dockBtnPressed_ = true;
        repaint (dockBtnRect_);
        return;
    }
    if (advancedBtnRect_.contains (p))
    {
        advancedBtnPressed_ = true;
        repaint (advancedBtnRect_);
        return;
    }
    if (cacheClearRect_.contains (p))
    {
        cacheClearPressed_ = true;
        repaint (cacheClearRect_);
        return;
    }
    if (cacheRevealRect_.contains (p))
    {
        cacheRevealPressed_ = true;
        repaint (cacheRevealRect_);
        return;
    }

    // Body rect rebuilt for click-outside-body test (mirrors paint()).
    const int bodyX = getLocalBounds().getRight() - kAnchorRight - kBodyWidth;
    const int bodyY = kAnchorTop;
    const int bodyH = computeBodyHeight();
    const juce::Rectangle<int> body (bodyX, bodyY, kBodyWidth, bodyH);

    if (! body.contains (p))
        hideMe();
}

void SettingsPopover::mouseUp (const juce::MouseEvent& e)
{
    const auto p = e.getPosition();

    if (beatsRowPressed_)
    {
        const bool fire = beatsRowRect_.contains (p);
        beatsRowPressed_ = false;
        if (fire)
        {
            // Toggle locally first so paint reflects new state immediately,
            // then notify. Popover stays open so user can see effect.
            showBeats_ = ! showBeats_;
            repaint();
            if (onShowBeatsToggled) onShowBeatsToggled (showBeats_);
        }
        else
        {
            repaint (beatsRowRect_);
        }
        return;
    }
    if (snapRowPressed_)
    {
        const bool fire = snapRowRect_.contains (p);
        snapRowPressed_ = false;
        if (fire)
        {
            // Cycle Off → Beats → Downbeats → Off.
            switch (snapRegion_)
            {
                case SnapRegion::Off:       snapRegion_ = SnapRegion::Beats;     break;
                case SnapRegion::Beats:     snapRegion_ = SnapRegion::Downbeats; break;
                case SnapRegion::Downbeats: snapRegion_ = SnapRegion::Off;       break;
            }
            repaint();
            if (onSnapRegionChanged) onSnapRegionChanged (snapRegion_);
        }
        else
        {
            repaint (snapRowRect_);
        }
        return;
    }
    if (insertSpliceRowPressed_)
    {
        const bool fire = insertSpliceRowRect_.contains (p);
        insertSpliceRowPressed_ = false;
        if (fire)
        {
            insertSpliceMarkers_ = ! insertSpliceMarkers_;
            repaint();
            if (onInsertSpliceMarkersToggled)
                onInsertSpliceMarkersToggled (insertSpliceMarkers_);
        }
        else
        {
            repaint (insertSpliceRowRect_);
        }
        return;
    }
    if (insertRegionRowPressed_)
    {
        const bool fire = insertRegionRowRect_.contains (p);
        insertRegionRowPressed_ = false;
        if (fire)
        {
            insertRenderRegion_ = ! insertRenderRegion_;
            repaint();
            if (onInsertRenderRegionToggled)
                onInsertRenderRegionToggled (insertRenderRegion_);
        }
        else
        {
            repaint (insertRegionRowRect_);
        }
        return;
    }
    if (cacheClearPressed_)
    {
        const bool fire = cacheClearRect_.contains (p);
        cacheClearPressed_ = false;
        repaint (cacheClearRect_);
        if (fire && onClearCache) onClearCache();
        return;
    }
    if (cacheRevealPressed_)
    {
        const bool fire = cacheRevealRect_.contains (p);
        cacheRevealPressed_ = false;
        repaint (cacheRevealRect_);
        if (fire && onRevealCache) onRevealCache();
        return;
    }

    if (advancedBtnPressed_)
    {
        const bool fire = advancedBtnRect_.contains (p);
        advancedBtnPressed_ = false;
        repaint (advancedBtnRect_);
        if (fire)
        {
            // Hide FIRST so modal state exits before the host opens / focuses
            // the AdvancedWeightsWindow. Same rationale as the dock-toggle
            // sequencing below.
            hideMe();
            if (onAdvancedToggled) onAdvancedToggled();
        }
        return;
    }

    const bool fireDock = dockBtnPressed_ && dockBtnRect_.contains (p);
    dockBtnPressed_ = false;
    repaint (dockBtnRect_);
    if (fireDock)
    {
        // Hide FIRST so modal state exits and the window-rebuild that
        // onDockToggled() triggers does not run while we hold keyboard
        // focus / modal state. REABeat's toggleDock tears down and
        // rebuilds the HWND; holding modal is unsafe during that.
        hideMe();
        if (onDockToggled) onDockToggled();
    }
}

void SettingsPopover::mouseMove (const juce::MouseEvent& e)
{
    if (! isVisible()) return;

    const auto p = e.getPosition();

    const bool overBeats = beatsRowRect_.contains (p);
    if (overBeats != beatsRowHover_)
    {
        beatsRowHover_ = overBeats;
        repaint (beatsRowRect_);
    }

    const bool overSnap = snapRowRect_.contains (p);
    if (overSnap != snapRowHover_)
    {
        snapRowHover_ = overSnap;
        repaint (snapRowRect_);
    }

    const bool overSplice = insertSpliceRowRect_.contains (p);
    if (overSplice != insertSpliceRowHover_)
    {
        insertSpliceRowHover_ = overSplice;
        repaint (insertSpliceRowRect_);
    }

    const bool overRegion = insertRegionRowRect_.contains (p);
    if (overRegion != insertRegionRowHover_)
    {
        insertRegionRowHover_ = overRegion;
        repaint (insertRegionRowRect_);
    }

    const bool overDock = dockBtnRect_.contains (p);
    if (overDock != dockBtnHover_)
    {
        dockBtnHover_ = overDock;
        repaint (dockBtnRect_);
    }

    const bool overAdvanced = advancedBtnRect_.contains (p);
    if (overAdvanced != advancedBtnHover_)
    {
        advancedBtnHover_ = overAdvanced;
        repaint (advancedBtnRect_);
    }

    const bool overClear = cacheClearRect_.contains (p);
    if (overClear != cacheClearHover_)
    {
        cacheClearHover_ = overClear;
        repaint (cacheClearRect_);
    }

    const bool overReveal = cacheRevealRect_.contains (p);
    if (overReveal != cacheRevealHover_)
    {
        cacheRevealHover_ = overReveal;
        repaint (cacheRevealRect_);
    }

    setMouseCursor ((overBeats || overSnap || overDock || overAdvanced
                       || overClear || overReveal)
                        ? juce::MouseCursor::PointingHandCursor
                        : juce::MouseCursor::NormalCursor);
}

void SettingsPopover::mouseExit (const juce::MouseEvent&)
{
    bool dirty = false;
    if (dockBtnHover_ || dockBtnPressed_)
    {
        dockBtnHover_   = false;
        dockBtnPressed_ = false;
        repaint (dockBtnRect_);
        dirty = true;
    }
    if (advancedBtnHover_ || advancedBtnPressed_)
    {
        advancedBtnHover_   = false;
        advancedBtnPressed_ = false;
        repaint (advancedBtnRect_);
        dirty = true;
    }
    if (beatsRowHover_ || beatsRowPressed_)
    {
        beatsRowHover_   = false;
        beatsRowPressed_ = false;
        repaint (beatsRowRect_);
        dirty = true;
    }
    if (snapRowHover_ || snapRowPressed_)
    {
        snapRowHover_   = false;
        snapRowPressed_ = false;
        repaint (snapRowRect_);
        dirty = true;
    }
    if (insertSpliceRowHover_ || insertSpliceRowPressed_)
    {
        insertSpliceRowHover_   = false;
        insertSpliceRowPressed_ = false;
        repaint (insertSpliceRowRect_);
        dirty = true;
    }
    if (insertRegionRowHover_ || insertRegionRowPressed_)
    {
        insertRegionRowHover_   = false;
        insertRegionRowPressed_ = false;
        repaint (insertRegionRowRect_);
        dirty = true;
    }
    if (dirty)
        setMouseCursor (juce::MouseCursor::NormalCursor);
}

bool SettingsPopover::keyPressed (const juce::KeyPress& k)
{
    if (k.isKeyCode (juce::KeyPress::escapeKey))
    {
        hideMe();
        return true;
    }
    return false;
}

} // namespace reamix::ui
