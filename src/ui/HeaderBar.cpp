#include "HeaderBar.h"
#include "LookAndFeelReamix.h"
#include "Theme.h"

namespace reamix::ui
{

// Gear icon path — 16×16 logical viewBox. Proper 8-tooth cog: closed
// outline tracing around the rim with flat-topped trapezoidal teeth
// (alternating outer / inner radius vertices) plus a center bushing
// circle. Replaces the prior 8-radial-line + center-circle drawing
// which user (sesja 62 NOTE-11) reported as looking like a sun /
// theme-toggle, not a gear.
static juce::Path buildGearPath()
{
    juce::Path p;
    constexpr float cx = 8.0f, cy = 8.0f;
    constexpr int   N        = 8;       // tooth count
    constexpr float outerR   = 6.5f;    // tooth-tip radius
    constexpr float innerR   = 5.0f;    // tooth-root (valley) radius
    constexpr float hubR     = 2.4f;    // center bushing radius
    constexpr float toothTop = 0.55f;   // tooth top fraction of pitch

    const float pitch   = juce::MathConstants<float>::twoPi / (float) N;
    const float halfTop = pitch * 0.5f * toothTop;

    auto pt = [&] (float r, float a) -> juce::Point<float>
    {
        return { cx + r * std::cos (a), cy + r * std::sin (a) };
    };

    for (int i = 0; i < N; ++i)
    {
        // Center the path on top (-π/2) so first tooth points up.
        const float c    = (float) i * pitch - juce::MathConstants<float>::halfPi;
        const float topL = c - halfTop;                       // tooth top-left edge
        const float topR = c + halfTop;                       // tooth top-right edge
        const float nextL = c + pitch - halfTop;              // next tooth top-left edge

        const auto v1 = pt (outerR, topL);
        const auto v2 = pt (outerR, topR);
        const auto v3 = pt (innerR, topR);
        const auto v4 = pt (innerR, nextL);

        if (i == 0) p.startNewSubPath (v1);
        else        p.lineTo (v1);
        p.lineTo (v2);
        p.lineTo (v3);
        p.lineTo (v4);
    }
    p.closeSubPath();

    // Center bushing — separate sub-path stroked alongside the rim.
    p.addEllipse (cx - hubR, cy - hubR, 2.0f * hubR, 2.0f * hubR);

    return p;
}

HeaderBar::HeaderBar()
{
    setWantsKeyboardFocus (false);
    setInterceptsMouseClicks (true, false);
}

void HeaderBar::setStatusKind (HeaderStatus kind)
{
    if (kind == statusKind_)
        return;
    const bool wasAnalyzing = (statusKind_ == HeaderStatus::Analyzing);
    statusKind_ = kind;

    // Sesja 108 — start/stop the LED breathing timer on entering/leaving
    // the Analyzing state. Reset phase on entry so the pulse always begins
    // at full-on for a clean visual cue.
    if (kind == HeaderStatus::Analyzing && ! wasAnalyzing)
    {
        pulsePhase_ = 0.0f;
        startTimerHz ((int) kPulseHz);
    }
    else if (kind != HeaderStatus::Analyzing && wasAnalyzing)
    {
        stopTimer();
        pulsePhase_ = 0.0f;
    }

    repaint();
}

void HeaderBar::timerCallback()
{
    pulsePhase_ += 1.0f / (kPulseHz * kPulseCycleSec);
    if (pulsePhase_ >= 1.0f) pulsePhase_ -= 1.0f;
    repaint();
}

void HeaderBar::resized()
{
    // plugin.css:26 .rx-header padding 0 10px 0 12px → gear sits at right edge
    // with 10px right padding; gear is 24×24.
    const int gearSize = 24;
    gearBounds_ = juce::Rectangle<int> (getWidth() - gearSize - 10,
                                         (getHeight() - gearSize) / 2,
                                         gearSize, gearSize);

    // Sesja 108 — heart icon 24×24 sits 6 px to the left of the gear.
    const int heartSize = 24;
    heartBounds_ = juce::Rectangle<int> (gearBounds_.getX() - 6 - heartSize,
                                          (getHeight() - heartSize) / 2,
                                          heartSize, heartSize);
}

void HeaderBar::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds();

    // plugin.css:28 bg + L29 bottom 1px Line
    g.fillAll (Bg1);
    g.setColour (Line);
    g.fillRect (bounds.withHeight (1).withY (bounds.getBottom() - 1));

    // Layout constants — plugin.css:26-30 (padding 0 10 0 12, gap 10).
    // Right-side padding is applied to gearBounds_ in resized(); do not
    // re-use it here.
    const int padLeft  = 12;
    const int gap      = 10;

    // ── Mark (logo + reamix / .me) ──
    const int markGap = 7; // plugin.css:31 gap 7px within .rx-mark
    const juce::Rectangle<int> markArea {
        padLeft,
        0,
        bounds.getHeight(),
        bounds.getHeight()
    };

    // Session 49 — brand mark is now the bundled sun-burst logo PNG rather
    // than the simple rounded-square Accent dot the mockup shipped. Scaled
    // up to 18×18 in the 40h header (was 10×10 as a pure marker dot) so the
    // detail is legible; any smaller and the radial rays muddle. If the logo
    // failed to decode (unlikely once bundled), fall back to the original
    // Accent dot so the header still renders.
    const int markSize = 18;
    const juce::Rectangle<float> markRect (
        (float) markArea.getX(),
        (float) markArea.getCentreY() - (float) markSize * 0.5f,
        (float) markSize, (float) markSize);

    const auto& logo = LookAndFeelReamix::brandLogo();
    if (logo.isValid())
    {
        // Sesja 111 v1.0.3 — source PNG is 128×124, target render here is 18×18
        // (≈7x downscale). JUCE's default mediumResamplingQuality produces
        // jagged sun-burst rays on Windows (Direct2D/GDI less forgiving than
        // macOS Quartz). highResamplingQuality uses bicubic interpolation —
        // smooth rays at small sizes.
        g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);
        g.drawImageWithin (logo,
                           (int) markRect.getX(), (int) markRect.getY(),
                           (int) markRect.getWidth(), (int) markRect.getHeight(),
                           juce::RectanglePlacement::centred
                           | juce::RectanglePlacement::onlyReduceInSize,
                           false);
    }
    else
    {
        g.setColour (Accent);
        g.fillRoundedRectangle (markRect, r::R1);
        g.setColour (juce::Colour (0x40000000));
        g.drawRoundedRectangle (markRect.reduced (0.5f), r::R1, 1.0f);
    }

    // Name "reamix" — plugin.css:37-42 (13/600 Fg0, 0.01em letter-spacing)
    auto textWidth = [] (const juce::Font& font, const juce::String& text)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, text, 0.0f, 0.0f);
        return ga.getBoundingBox (0, -1, false).getWidth();
    };

    const int nameX = (int) markRect.getRight() + markGap;
    const juce::String reamixText { "reamix" };
    const auto reamixFont = uiFont (13.0f, 600);
    const int reamixW = (int) std::ceil (textWidth (reamixFont, reamixText));
    g.setColour (Fg0);
    g.setFont (reamixFont);
    g.drawText (reamixText,
                juce::Rectangle<int> (nameX, 0, reamixW, bounds.getHeight()),
                juce::Justification::centredLeft, false);

    // ".me" — plugin.css:43 Fg3 500
    const int tldX = nameX + reamixW;
    const juce::String tldText { ".me" };
    const auto tldFont = uiFont (13.0f, 500);
    const int tldW = (int) std::ceil (textWidth (tldFont, tldText));
    g.setColour (Fg3);
    g.setFont (tldFont);
    g.drawText (tldText,
                juce::Rectangle<int> (tldX, 0, tldW, bounds.getHeight()),
                juce::Justification::centredLeft, false);

    // ── Status dot ──
    // Sesja 108 — dropped the "Ready" label + Bg2 pill background per user
    // pushback "po co ten napis ready?" — the dot alone carries the state
    // signal (idle / working / error). Dot position preserves sesja-92
    // baseline (just inside the would-be strip's left padding).
    const int stripX     = tldX + tldW + gap;
    const int dotDiam    = 6;
    const int stripPadX  = 10;
    juce::Rectangle<float> stripDot (
        (float) (stripX + stripPadX),
        (float) bounds.getCentreY() - dotDiam * 0.5f,
        (float) dotDiam, (float) dotDiam);
    juce::Colour dotColour = Fg3;
    switch (statusKind_)
    {
        case HeaderStatus::Ready:     dotColour = Fg3;    break;
        case HeaderStatus::Analyzing:
        {
            // LED breathing pulse. Cosine over phase∈[0,1) maps to alpha
            // 0.4..1.0 with 1.0 at phase=0 (full-on start) and 0.4 at
            // phase=0.5 (dim mid-cycle). Driven by 30 Hz Timer in
            // timerCallback while statusKind_ == Analyzing.
            const float a = 0.4f + 0.6f * 0.5f *
                (1.0f + std::cos (pulsePhase_
                                   * juce::MathConstants<float>::twoPi));
            dotColour = Accent.withAlpha (a);
            break;
        }
        case HeaderStatus::Good:      dotColour = Good;   break;
        case HeaderStatus::Error:     dotColour = Bad;    break;
    }
    g.setColour (dotColour);
    g.fillEllipse (stripDot);

    // ── Gear icon button ──
    // plugin.css:71-78. 24×24, transparent default, hover bg Bg3 + Fg0,
    // active bg Bg4.
    juce::Colour gearBg = juce::Colours::transparentBlack;
    juce::Colour gearFg = Fg2;
    if (gearPressed_)       { gearBg = Bg4; gearFg = Fg0; }
    else if (gearHover_)    { gearBg = Bg3; gearFg = Fg0; }

    if (! gearBg.isTransparent())
    {
        g.setColour (gearBg);
        g.fillRoundedRectangle (gearBounds_.toFloat(), r::R2);
    }

    g.setColour (gearFg);
    auto gearPath = buildGearPath();
    // Scale 16-unit path into gear bounds with 4px inset (visual weight).
    auto targetArea = gearBounds_.toFloat().reduced (4.0f);
    auto transform = gearPath.getTransformToScaleToFit (targetArea, true);
    g.strokePath (gearPath, juce::PathStrokeType (1.25f,
                                                    juce::PathStrokeType::mitered,
                                                    juce::PathStrokeType::butt),
                  transform);

    // ── Heart icon (donation shortcut) ──
    // Sesja 108. Same hover/pressed treatment as gear for visual consistency.
    juce::Colour heartBg = juce::Colours::transparentBlack;
    juce::Colour heartFg = Fg2;
    if (heartPressed_)    { heartBg = Bg4; heartFg = Fg0; }
    else if (heartHover_) { heartBg = Bg3; heartFg = Fg0; }

    if (! heartBg.isTransparent())
    {
        g.setColour (heartBg);
        g.fillRoundedRectangle (heartBounds_.toFloat(), r::R2);
    }

    g.setColour (heartFg);
    {
        // 16-unit viewBox heart path, filled. Two top lobes via cubic curves,
        // bottom apex at (8, 14). Inset 4 px in 24×24 bounds matches gear.
        juce::Path heart;
        heart.startNewSubPath (8.0f, 14.0f);
        heart.cubicTo (0.0f,  9.0f,  0.0f, 5.0f,  0.0f,  5.0f);
        heart.cubicTo (0.0f,  2.0f,  2.0f, 0.5f,  4.5f,  0.5f);
        heart.cubicTo (6.5f,  0.5f,  7.5f, 2.5f,  8.0f,  4.5f);
        heart.cubicTo (8.5f,  2.5f,  9.5f, 0.5f,  11.5f, 0.5f);
        heart.cubicTo (14.0f, 0.5f,  16.0f, 2.0f, 16.0f, 5.0f);
        heart.cubicTo (16.0f, 5.0f,  16.0f, 9.0f, 8.0f,  14.0f);
        heart.closeSubPath();
        const auto heartTarget = heartBounds_.toFloat().reduced (4.0f);
        const auto heartXform  = heart.getTransformToScaleToFit (heartTarget, true);
        g.fillPath (heart, heartXform);
    }
}

// ── Hit-testing & hover state ────────────────────────────────────────

void HeaderBar::mouseMove (const juce::MouseEvent& e)
{
    const auto pos       = e.getPosition();
    const bool overGear  = gearBounds_ .contains (pos);
    const bool overHeart = heartBounds_.contains (pos);

    if (overGear != gearHover_)
    {
        gearHover_ = overGear;
        repaint (gearBounds_);
    }
    if (overHeart != heartHover_)
    {
        heartHover_ = overHeart;
        repaint (heartBounds_);
    }
    setMouseCursor ((overGear || overHeart) ? juce::MouseCursor::PointingHandCursor
                                            : juce::MouseCursor::NormalCursor);
}

void HeaderBar::mouseExit (const juce::MouseEvent&)
{
    if (gearHover_ || gearPressed_ || heartHover_ || heartPressed_)
    {
        gearHover_    = false;
        gearPressed_  = false;
        heartHover_   = false;
        heartPressed_ = false;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint (gearBounds_);
        repaint (heartBounds_);
    }
}

void HeaderBar::mouseDown (const juce::MouseEvent& e)
{
    const auto pos = e.getPosition();
    if (gearBounds_.contains (pos))
    {
        gearPressed_ = true;
        repaint (gearBounds_);
    }
    else if (heartBounds_.contains (pos))
    {
        heartPressed_ = true;
        repaint (heartBounds_);
    }
}

void HeaderBar::mouseUp (const juce::MouseEvent& e)
{
    const auto pos          = e.getPosition();
    const bool wasGearPress = gearPressed_;
    const bool wasHeartPress = heartPressed_;
    gearPressed_  = false;
    heartPressed_ = false;
    repaint (gearBounds_);
    repaint (heartBounds_);
    if (wasGearPress && gearBounds_.contains (pos) && onGearClicked)
        onGearClicked();
    else if (wasHeartPress && heartBounds_.contains (pos) && onHeartClicked)
        onHeartClicked();
}

} // namespace reamix::ui
