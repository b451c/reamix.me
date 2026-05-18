#include "SourcePanel.h"
#include "Theme.h"

#include <cmath>

namespace reamix::ui
{

namespace
{
    // primitives.jsx:45-49 rxFmt — "M:SS" floor format. Local copy; trivial
    // and several panels need it.
    juce::String rxFmt (double seconds)
    {
        const int s = juce::jmax (0, (int) std::round (seconds));
        const int m = s / 60;
        const int ss = s % 60;
        return juce::String (m) + ":" + (ss < 10 ? "0" : "") + juce::String (ss);
    }

    float measureText (const juce::Font& font, const juce::String& text)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, text, 0.0f, 0.0f);
        return ga.getBoundingBox (0, -1, false).getWidth();
    }

    // primitives.jsx § RxIcon 'refresh' — circular arrow. 300° clockwise arc
    // open at the top-right, with a small arrowhead at the starting end of
    // the arc (~1 o'clock) pointing downward-right tangent to the motion.
    //
    // Coordinate convention: juce::Path::addCentredArc uses angle 0 = 12
    // o'clock, increasing clockwise. Point on circle at angle θ:
    //   (cx + r·sin(θ), cy − r·cos(θ)).
    void drawRefreshIcon (juce::Graphics& g, juce::Rectangle<float> area,
                          const juce::Colour& colour)
    {
        const float cx = area.getCentreX();
        const float cy = area.getCentreY();
        const float r  = std::min (area.getWidth(), area.getHeight()) * 0.40f;

        // Arc: ~30° CW from top (1 o'clock) → ~330° (11 o'clock) = 300° sweep.
        juce::Path arc;
        arc.addCentredArc (cx, cy, r, r, 0.0f,
                            juce::MathConstants<float>::pi * 0.17f,
                            juce::MathConstants<float>::pi * 1.83f,
                            true);
        g.setColour (colour);
        g.strokePath (arc, juce::PathStrokeType (1.25f,
                                                   juce::PathStrokeType::curved,
                                                   juce::PathStrokeType::butt));

        // Arrowhead at the start angle (1 o'clock). Triangle with tip pointing
        // outward-right (away from circle), backing toward the arc.
        const float a  = juce::MathConstants<float>::pi * 0.17f;
        const float ax = cx + r * std::sin (a);
        const float ay = cy - r * std::cos (a);
        const float sz = r * 0.55f;

        juce::Path head;
        head.addTriangle (ax - sz * 0.4f, ay - sz * 0.9f,
                          ax + sz * 0.9f, ay + sz * 0.0f,
                          ax - sz * 0.4f, ay + sz * 0.7f);
        g.fillPath (head);
    }
}

SourcePanel::SourcePanel()
{
    setInterceptsMouseClicks (true, false);
    setOpaque (true);
    info_.empty = true;
    info_.name  = "No item selected";
}

void SourcePanel::setSource (SourceInfo si)
{
    info_ = std::move (si);
    repaint();
}

void SourcePanel::setAnalyzing (bool a, double p01, juce::String stageLabel)
{
    const double clamped = juce::jlimit (0.0, 1.0, p01);
    if (a == analyzing_
        && std::abs (clamped - progress_) < 1.0e-4
        && stageLabel == stageLabel_)
        return;
    const bool wasAnalyzing = analyzing_;
    analyzing_  = a;
    progress_   = clamped;
    stageLabel_ = std::move (stageLabel);

    // Sesja 108 — real-time elapsed-time counter. Start the 10 Hz Timer +
    // anchor the steady-clock start point on the false → true edge; stop
    // and reset on the true → false edge. Per-tick repaints re-render the
    // elapsed string in paint() so the user sees a smooth ticker rather
    // than a step-wise update that only refreshed when pipeline progress
    // callbacks arrived.
    if (a && ! wasAnalyzing)
    {
        analyzeStart_      = std::chrono::steady_clock::now();
        currentElapsedSec_ = 0.0;
        startTimerHz (kElapsedTimerHz);
    }
    else if (! a && wasAnalyzing)
    {
        stopTimer();
        currentElapsedSec_ = 0.0;
    }

    resized();
    repaint();
}

void SourcePanel::timerCallback()
{
    const auto now = std::chrono::steady_clock::now();
    currentElapsedSec_ = std::chrono::duration<double> (now - analyzeStart_).count();
    repaint();
}

void SourcePanel::setHasAnalysis (bool yes)
{
    if (yes == hasAnalysis_)
        return;
    hasAnalysis_ = yes;
    // Width changes between "Analyze" and "Analyze Again" — full re-layout.
    resized();
    repaint();
}

int SourcePanel::getPreferredHeight() const noexcept
{
    // 68 compact, 86 analyzing (18h progress row added under content).
    return analyzing_ ? 86 : 68;
}

void SourcePanel::resized()
{
    using namespace reamix::theme;

    const int padX = 12;
    const int padY = 10;
    const int w    = getWidth();
    const int h    = getHeight();

    // Right column: Analyze button (sm, 22h, variable width).
    const auto btnFont  = uiFont (fs::Sm, 500);
    const juce::String btnLabel = analyzing_
        ? juce::String::fromUTF8 ("Analyzing\xe2\x80\xa6")
        : (hasAnalysis_ ? juce::String ("Analyze Again") : juce::String ("Analyze"));
    const int btnTextW  = (int) std::ceil (measureText (btnFont, btnLabel));
    const int btnIconW  = 12;
    const int btnIconGap = 6;
    const int btnPadX   = 9;   // .rx-btn.sm padding 0 9
    const int btnW      = btnPadX + btnIconW + btnIconGap + btnTextW + btnPadX;
    const int btnH      = 22;

    // Right block height spans content area; vertically centered on the
    // compact content zone (exclude progress row).
    const int compactBody = 68 - 2 * padY;  // fixed compact content height = 48
    const int rightCY = padY + compactBody / 2;

    analyzeButtonBounds_ = juce::Rectangle<int> (w - padX - btnW,
                                                   rightCY - btnH / 2,
                                                   btnW, btnH);

    // Progress row — only meaningful when analyzing. Position below the
    // compact block with 6px gap.
    const int progY = padY + compactBody + 6;
    progressRowBounds_ = juce::Rectangle<int> (padX, progY,
                                                 w - 2 * padX, 18);
    (void) h; // silence warning; preferredHeight drives layout from parent
}

void SourcePanel::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds();

    // plugin.css:83-84 bg Bg1 + bottom 1px Line
    g.fillAll (Bg1);
    g.setColour (Line);
    g.fillRect (bounds.withHeight (1).withY (bounds.getBottom() - 1));

    const int padX = 12;
    const int padY = 10;
    const int compactBody = 68 - 2 * padY;  // 48 px

    // ── Left column: name (row 1) + meta (row 2) ──
    const int leftX = padX;
    const int leftRight = analyzeButtonBounds_.getX() - 10;
    const int leftW = juce::jmax (0, leftRight - leftX);

    // plugin.css:92-96 name: Fg0 / 500 / Md (13px), nowrap+ellipsis
    const auto nameFont = uiFont (fs::Md, 500);
    const int nameH = 16;
    const int metaH = 14;
    const int rowGap = 6; // plugin.css:89 row-gap

    const int contentH = nameH + rowGap + metaH; // 36
    const int contentY = padY + (compactBody - contentH) / 2;

    g.setFont (nameFont);
    g.setColour (info_.empty ? Fg3 : Fg0);
    g.drawText (info_.name,
                juce::Rectangle<int> (leftX, contentY, leftW, nameH),
                juce::Justification::centredLeft, true);

    // plugin.css:98-104 meta: Fg2 / Sm (11px), flex gap 8, sep Fg4
    const auto metaFont = uiFont (fs::Sm, 400);
    const auto metaMonoFont = monoFont (fs::Sm, 400);
    const int metaY = contentY + nameH + rowGap;

    if (info_.empty)
    {
        g.setFont (metaFont);
        g.setColour (Fg2);
        g.drawText ("Select an item in REAPER to begin",
                    juce::Rectangle<int> (leftX, metaY, leftW, metaH),
                    juce::Justification::centredLeft, true);
    }
    else
    {
        // Compose as: "{dur} · {bpm} BPM · {beats} beats". The " · " separator
        // is drawn as a 2 px filled circle between chunks rather than as a
        // U+00B7 text glyph — the text path was silently dropping the glyph
        // at 11px (neither Inter nor JetBrainsMono's middle-dot rendered
        // reliably here, possibly a font-subsetting issue). The circle is
        // font-independent and pixel-precise. Colour is Fg3 (mockup says
        // Fg4 = #3E3B37 on Bg1, contrast ~1.6:1 invisible; Fg3 = #5F5A53 is
        // readable at ~3:1). Minor visual deviation vs plugin.css:104 —
        // logged in phases/phase-6-ui/log.md § session-41.
        const juce::String dur  = rxFmt (info_.duration);
        // Pre-analysis state uses em-dash for BPM / beats (Lua shows only the
        // duration line on a fresh item; we keep the 3-chunk layout but mute
        // unanalyzed values — mockup shows fully-populated values because
        // it's static; port matches Lua pre-analysis UX).
        const juce::String bpm  = info_.bpm > 0.0
            ? juce::String (info_.bpm, 1) + " BPM"
            : juce::String::fromUTF8 ("\xe2\x80\x94 BPM");
        const juce::String bts  = info_.beats > 0
            ? juce::String (info_.beats) + " beats"
            : juce::String::fromUTF8 ("\xe2\x80\x94 beats");

        int x = leftX;
        const int sepGap = 10; // total width reserved for sep (4 + dot + 4 ~= 10)
        const float dotSize = 2.0f;

        auto drawTextChunk = [&] (const juce::String& s, const juce::Colour& c, const juce::Font& f)
        {
            const int wChunk = (int) std::ceil (measureText (f, s));
            g.setFont (f);
            g.setColour (c);
            g.drawText (s,
                        juce::Rectangle<int> (x, metaY, wChunk + 2, metaH),
                        juce::Justification::centredLeft, false);
            x += wChunk;
        };

        auto drawDotSep = [&]
        {
            const float cx = (float) x + sepGap * 0.5f;
            const float cy = (float) metaY + metaH * 0.5f;
            g.setColour (Fg3);
            g.fillEllipse (cx - dotSize * 0.5f,
                            cy - dotSize * 0.5f,
                            dotSize, dotSize);
            x += sepGap;
        };

        drawTextChunk (dur, Fg2, metaMonoFont);
        drawDotSep();
        drawTextChunk (bpm, Fg2, metaMonoFont);
        drawDotSep();
        drawTextChunk (bts, Fg2, metaMonoFont);
    }

    // ── Analyze button ──
    // plugin.css:130-151 .rx-btn + .rx-btn.sm
    {
        const auto btnArea = analyzeButtonBounds_.toFloat();
        // Sesja 99 — DEV-051 (sesja 62 NOTE-6): when no item selected,
        // Analyze button looked active even though click is no-op (gated
        // by `! info_.empty` at handler line ~419). Visual disabled state
        // now matches click semantic.
        const bool disabled = analyzing_ || info_.empty;
        const bool hover   = (hover_ == HitRegion::Analyze);
        const bool pressed = (pressed_ == HitRegion::Analyze);

        juce::Colour bg   = Bg4;
        juce::Colour fg   = Fg1;
        juce::Colour bord = LineStrong;
        if (disabled) { bg = Bg3; fg = Fg3; bord = Line; }
        else if (pressed) bg = Bg3;
        else if (hover)   { bg = Bg5; fg = Fg0; }

        g.setColour (bg);
        g.fillRoundedRectangle (btnArea, r::R3);
        g.setColour (bord);
        g.drawRoundedRectangle (btnArea.reduced (0.5f), r::R3, 1.0f);

        // Icon + text
        const int btnPadX = 9;
        const int iconW = 12;
        const int iconGap = 6;
        const auto iconArea = juce::Rectangle<float> (
            btnArea.getX() + btnPadX,
            btnArea.getCentreY() - iconW * 0.5f,
            (float) iconW, (float) iconW);
        drawRefreshIcon (g, iconArea, fg);

        const auto btnFont = uiFont (fs::Sm, 500);
        g.setFont (btnFont);
        g.setColour (fg);
        const juce::String btnLabel = analyzing_
            ? juce::String::fromUTF8 ("Analyzing\xe2\x80\xa6")
            : (hasAnalysis_ ? juce::String ("Analyze Again") : juce::String ("Analyze"));
        const int textX = (int) iconArea.getRight() + iconGap;
        g.drawText (btnLabel,
                    juce::Rectangle<int> (textX, (int) btnArea.getY(),
                                            (int) btnArea.getRight() - textX - btnPadX,
                                            (int) btnArea.getHeight()),
                    juce::Justification::centred, false);
    }

    // ── Progress row (only while analyzing) ──
    // plugin.css:107-127 .rx-analyze-row + .rx-progress + .rx-progress-label
    //
    // Sesja 64 layout (stage-aware): [stage label] [progress bar fill] [% · eta]
    // Stage label in Accent + 500 weight; "%" + eta still mono Sm 400 in Fg2.
    if (analyzing_)
    {
        const auto row = progressRowBounds_;
        const int barH = 4;
        const int rightLabelW = 74;
        const int gap = 10;

        // Stage label measured + rendered on the left when present.
        const auto stageFont = uiFont (fs::Sm, 500);
        int stageLabelW = 0;
        const bool hasStage = stageLabel_.isNotEmpty();
        if (hasStage)
        {
            stageLabelW = (int) std::ceil (measureText (stageFont, stageLabel_));
            stageLabelW = juce::jmin (stageLabelW, 160); // cap so bar doesn't shrink to nothing
            g.setFont (stageFont);
            g.setColour (Accent);
            g.drawText (stageLabel_,
                        juce::Rectangle<int> (row.getX(), row.getY(),
                                                stageLabelW, row.getHeight()),
                        juce::Justification::centredLeft, true);
        }

        const int barLeft = row.getX() + (hasStage ? stageLabelW + gap : 0);
        const int barRight = row.getRight() - rightLabelW - gap;
        const auto barArea = juce::Rectangle<int> (barLeft,
                                                     row.getCentreY() - barH / 2,
                                                     juce::jmax (0, barRight - barLeft),
                                                     barH).toFloat();
        g.setColour (Bg3);
        g.fillRoundedRectangle (barArea, 2.0f);

        const float fillW = (float) barArea.getWidth() * (float) progress_;
        if (fillW > 0.0f)
        {
            g.setColour (Accent);
            g.fillRoundedRectangle (barArea.withWidth (fillW), 2.0f);
        }

        g.setColour (Fg2);
        g.setFont (monoFont (fs::Sm, 400));
        const int pct = (int) std::round (progress_ * 100.0);
        // Sesja 108 — show real-time elapsed seconds (driven by 10 Hz Timer)
        // instead of the synthetic ETA estimate. Decoupled from progress%
        // so user sees a smooth ticker even when pipeline stages don't fire.
        const juce::String lbl = juce::String (pct)
                                  + juce::String::fromUTF8 ("% \xc2\xb7 ")
                                  + juce::String (currentElapsedSec_, 1) + "s";
        g.drawText (lbl,
                    juce::Rectangle<int> (barRight + gap, row.getY(),
                                            rightLabelW, row.getHeight()),
                    juce::Justification::centredRight, false);
    }
}

// ── Hit testing ──────────────────────────────────────────────────────

SourcePanel::HitRegion SourcePanel::regionAt (juce::Point<int> p) const
{
    if (info_.empty || analyzing_)
    {
        if (! info_.empty && analyzeButtonBounds_.contains (p))
            return HitRegion::Analyze; // actually disabled when analyzing, but
                                        // allow hover state consistency
        return HitRegion::None;
    }
    if (analyzeButtonBounds_.contains (p))  return HitRegion::Analyze;
    return HitRegion::None;
}

void SourcePanel::setHover (HitRegion h)
{
    if (h == hover_) return;
    hover_ = h;
    setMouseCursor (hover_ != HitRegion::None
                     ? juce::MouseCursor::PointingHandCursor
                     : juce::MouseCursor::NormalCursor);
    repaint();
}

void SourcePanel::setPressed (HitRegion h)
{
    if (h == pressed_) return;
    pressed_ = h;
    repaint();
}

void SourcePanel::mouseMove (const juce::MouseEvent& e)
{
    setHover (regionAt (e.getPosition()));
}

void SourcePanel::mouseExit (const juce::MouseEvent&)
{
    setHover (HitRegion::None);
    setPressed (HitRegion::None);
}

void SourcePanel::mouseDown (const juce::MouseEvent& e)
{
    const auto h = regionAt (e.getPosition());
    if (h == HitRegion::Analyze && analyzing_)
        return; // disabled during analyze
    setPressed (h);
}

void SourcePanel::mouseUp (const juce::MouseEvent& e)
{
    const auto h = regionAt (e.getPosition());
    const auto wasPressed = pressed_;
    setPressed (HitRegion::None);

    if (h != wasPressed || h == HitRegion::None)
        return;

    switch (h)
    {
        case HitRegion::Analyze:
            if (! analyzing_ && ! info_.empty && onAnalyze) onAnalyze();
            break;
        case HitRegion::None:
            break;
    }
}

} // namespace reamix::ui
