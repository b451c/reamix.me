#include "StatusBar.h"
#include "Theme.h"

#include <cmath>

namespace reamix::ui
{

StatusBar::StatusBar()
{
    setInterceptsMouseClicks (false, false); // purely passive strip
    setOpaque (true);
}

void StatusBar::setText (juce::String text)
{
    const bool errorWasSet = error_.isNotEmpty();
    error_.clear();
    if (text == text_ && ! errorWasSet)
        return;
    text_ = std::move (text);
    repaint();
}

void StatusBar::setError (juce::String err)
{
    if (err == error_)
        return;
    error_ = std::move (err);
    repaint();
}

void StatusBar::clearError()
{
    if (error_.isEmpty())
        return;
    error_.clear();
    repaint();
}

void StatusBar::setBusy (juce::String label)
{
    if (busy_ && label == busyLabel_)
        return;
    busyLabel_ = std::move (label);
    if (! busy_)
    {
        busy_ = true;
        animPhase_ = 0.0f;
        startTimerHz (60);
    }
    repaint();
}

void StatusBar::clearBusy ()
{
    if (! busy_)
        return;
    busy_ = false;
    busyLabel_.clear();
    stopTimer();
    repaint();
}

void StatusBar::setNotice (juce::String label, int durationMs)
{
    notice_ = std::move (label);
    repaint();
    juce::Component::SafePointer<StatusBar> self (this);
    juce::Timer::callAfterDelay (durationMs, [self]
    {
        if (self != nullptr) self->clearNotice();
    });
}

void StatusBar::clearNotice ()
{
    if (notice_.isEmpty()) return;
    notice_.clear();
    repaint();
}

void StatusBar::timerCallback()
{
    // 1.5 s loop: 60 Hz × 1.5 s = 90 ticks/cycle. Increment 1/90.
    animPhase_ += 1.0f / 90.0f;
    if (animPhase_ >= 1.0f) animPhase_ -= 1.0f;
    repaint();
}

void StatusBar::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds();
    const bool isError    = error_.isNotEmpty();
    const bool showBusy   = busy_ && ! isError;  // error wins over busy
    const bool showNotice = ! isError && ! showBusy && notice_.isNotEmpty();

    // plugin.css:435 bg Bg0 + L436 top 1px Line
    g.fillAll (Bg0);
    g.setColour (Line);
    g.fillRect (bounds.withHeight (1));

    // ── Loading line above the bar (busy state) ──
    // 2px accent line slides a 36px-wide segment from left to right across
    // the top edge of the status bar (just below the 1px Line border).
    if (showBusy)
    {
        const int lineY = 1;
        const int lineH = 2;
        const int barW = bounds.getWidth();
        const int segW = 36;
        // Position: -segW (off-screen left) → barW (off-screen right).
        const float xStart = -static_cast<float> (segW)
                             + animPhase_ * static_cast<float> (barW + segW);

        // Subtle dim accent track underneath (helps eye anchor).
        g.setColour (AccentDim);
        g.fillRect (juce::Rectangle<int> (0, lineY, barW, lineH));

        // Bright accent moving segment.
        g.setColour (Accent);
        g.fillRect (juce::Rectangle<float> (xStart, (float) lineY,
                                              (float) segW, (float) lineH));
    }

    // plugin.css:433 padding 0 12px
    const int padX = 12;
    const int textY = 0;
    const int textH = bounds.getHeight();

    // Right-aligned version chip — shell.jsx:192 (color Fg4, always).
    auto textWidth = [] (const juce::Font& font, const juce::String& text)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, text, 0.0f, 0.0f);
        return ga.getBoundingBox (0, -1, false).getWidth();
    };

    // Font for left text — Sm (11px) + 500 weight when busy/notice for visibility,
    // Xs (9px) + 400 when normal/error per plugin.css:437-439.
    const auto normalFont = monoFont (fs::Xs, 400);
    const auto busyFont   = monoFont (fs::Sm, 500);
    const auto& leftFont  = (showBusy || showNotice) ? busyFont : normalFont;

    const int versionW = (int) std::ceil (textWidth (normalFont, version_));

    // Left status text region
    const int leftX = padX;
    int       leftStartX = leftX;
    int       leftW   = juce::jmax (0, bounds.getWidth() - padX - versionW - padX - 8);

    // ── Spinner (busy state) ──
    // 12 px circle at the left, before the label. ~270° arc rotating with
    // animPhase_. Sits vertically centered.
    if (showBusy)
    {
        const float spinSize = 12.0f;
        const float cx = (float) leftX + spinSize * 0.5f;
        const float cy = (float) textY + (float) textH * 0.5f;
        const float r  = spinSize * 0.42f;

        // Faint full-circle backing for visual mass.
        g.setColour (AccentDim);
        g.drawEllipse (cx - r, cy - r, r * 2.0f, r * 2.0f, 1.0f);

        // Bright rotating arc (~270°).
        const float startAngle = animPhase_ * juce::MathConstants<float>::twoPi;
        const float arcSweep   = juce::MathConstants<float>::twoPi * 0.75f;
        juce::Path arc;
        arc.addCentredArc (cx, cy, r, r, 0.0f,
                            startAngle, startAngle + arcSweep, true);
        g.setColour (Accent);
        g.strokePath (arc, juce::PathStrokeType (1.5f,
                                                   juce::PathStrokeType::curved,
                                                   juce::PathStrokeType::butt));

        leftStartX = leftX + (int) spinSize + 8;  // 8 px gap between spinner and label
        leftW      = juce::jmax (0, bounds.getWidth() - leftStartX - padX - versionW - 8);
    }

    // Notice: subtle accent-tinted background pill behind the text + bright
    // accent text for high visibility without becoming alarming. Painted
    // before the spinner-row check (no spinner during notice — short event).
    if (showNotice)
    {
        const float padX  = 6.0f;
        const float padY  = 2.0f;
        const float textW = (float) std::ceil (textWidth (busyFont, notice_));
        juce::Rectangle<float> pill (
            (float) leftStartX - padX,
            (float) textY + (float) textH * 0.5f - (float) busyFont.getHeight() * 0.5f - padY,
            textW + padX * 2.0f,
            (float) busyFont.getHeight() + padY * 2.0f);
        // Sesja 65 — bumped pill background from AccentDim (alpha 0.18) to
        // AccentGlow (alpha 0.35) for legibility. User flagged the notice
        // in test 57 as too subtle to read at a glance.
        g.setColour (AccentGlow);
        g.fillRoundedRectangle (pill, 3.0f);
        g.setColour (Accent);
        g.drawRoundedRectangle (pill.reduced (0.5f), 3.0f, 1.0f);
    }

    g.setFont (leftFont);
    juce::String displayText;
    if      (isError)    { g.setColour (Bad);     displayText = error_; }
    else if (showBusy)   { g.setColour (Accent);  displayText = busyLabel_; }
    else if (showNotice) { g.setColour (AccentHi); displayText = notice_; }
    else                 { g.setColour (Fg3);     displayText = text_; }

    g.drawText (displayText,
                juce::Rectangle<int> (leftStartX, textY, leftW, textH),
                juce::Justification::centredLeft, true);

    // Right version chip — Fg4, always normal font.
    g.setFont (normalFont);
    g.setColour (Fg4);
    const int rightX = bounds.getWidth() - padX - versionW;
    g.drawText (version_,
                juce::Rectangle<int> (rightX, textY, versionW, textH),
                juce::Justification::centredLeft, false);
}

} // namespace reamix::ui
