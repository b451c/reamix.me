#include "TransportBar.h"
#include "LookAndFeelReamix.h"
#include "Theme.h"

#include <cmath>

namespace reamix::ui
{

namespace
{
    float measureText (const juce::Font& font, const juce::String& text)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, text, 0.0f, 0.0f);
        return ga.getBoundingBox (0, -1, false).getWidth();
    }

    // primitives.jsx § RxIcon 'play' — right-pointing triangle.
    void drawPlayIcon (juce::Graphics& g, juce::Rectangle<float> area,
                       const juce::Colour& colour)
    {
        juce::Path p;
        const float cx = area.getCentreX();
        const float cy = area.getCentreY();
        const float w  = area.getWidth();
        const float h  = area.getHeight();
        p.addTriangle (cx - w * 0.3f, cy - h * 0.42f,
                       cx - w * 0.3f, cy + h * 0.42f,
                       cx + w * 0.42f, cy);
        g.setColour (colour);
        g.fillPath (p);
    }

    // primitives.jsx § RxIcon 'stop' — filled square.
    void drawStopIcon (juce::Graphics& g, juce::Rectangle<float> area,
                       const juce::Colour& colour)
    {
        g.setColour (colour);
        g.fillRect (area.reduced (area.getWidth() * 0.22f));
    }

    // primitives.jsx § RxIcon 'insert' — down-arrow into a horizontal bar.
    void drawInsertIcon (juce::Graphics& g, juce::Rectangle<float> area,
                         const juce::Colour& colour)
    {
        const float cx = area.getCentreX();
        const float top = area.getY() + area.getHeight() * 0.18f;
        const float mid = area.getY() + area.getHeight() * 0.58f;
        const float bot = area.getY() + area.getHeight() * 0.82f;

        juce::Path arrow;
        arrow.startNewSubPath (cx, top);
        arrow.lineTo          (cx, mid);
        // Arrowhead
        juce::Path head;
        head.addTriangle (cx - area.getWidth() * 0.28f, mid - area.getWidth() * 0.08f,
                          cx + area.getWidth() * 0.28f, mid - area.getWidth() * 0.08f,
                          cx, mid + area.getWidth() * 0.28f);

        g.setColour (colour);
        g.strokePath (arrow, juce::PathStrokeType (1.25f,
                                                     juce::PathStrokeType::mitered,
                                                     juce::PathStrokeType::butt));
        g.fillPath (head);
        // Horizontal bar (insertion target)
        g.fillRect (juce::Rectangle<float> (area.getX() + area.getWidth() * 0.18f,
                                              bot - 1.0f,
                                              area.getWidth() * 0.64f,
                                              1.5f));
    }
}

TransportBar::TransportBar()
{
    setInterceptsMouseClicks (true, false);
    setOpaque (true);
    setWantsKeyboardFocus (false); // installed as a KeyListener on parent
}

void TransportBar::setState (TransportState s)
{
    if (s == state_)
        return;
    state_ = s;
    // Reset any hover/pressed state that refers to a now-disabled button.
    if (! isEnabledFor (hover_))   hover_   = Button::None;
    if (! isEnabledFor (pressed_)) pressed_ = Button::None;
    repaint();
}

void TransportBar::setInsertBusy (bool busy, juce::String label)
{
    if (busy == insertBusy_ && label == insertBusyLabel_)
        return;
    insertBusy_      = busy;
    insertBusyLabel_ = std::move (label);
    if (insertBusy_)
    {
        if (hover_   == Button::Insert) hover_   = Button::None;
        if (pressed_ == Button::Insert) pressed_ = Button::None;
    }
    repaint();
}

bool TransportBar::isEnabledFor (Button b) const noexcept
{
    switch (b)
    {
        case Button::PlayStop:
            // Any non-Idle state (Ready / Playing / Inserted) — user can
            // always start preview when something is analyzed, and can
            // always stop when playing.
            return state_ != TransportState::Idle;
        case Button::Insert:
            // Sesja 64 — disabled while busy (Inserting…) so user can't
            // click again mid-mutation.
            if (insertBusy_) return false;
            return state_ == TransportState::Ready
                || state_ == TransportState::Inserted;
        case Button::None:
            return false;
    }
    return false;
}

void TransportBar::resized()
{
    // plugin.css:418-428 .rx-transport: padding 10/12, gap 8, 34h buttons.
    // ADR-038: two-button grid (auto, 1fr) — single PlayStop + full-width
    // Insert instead of mockup's three-button (Preview + Stop + Insert).
    const int padX = 12;
    const int padY = 10;
    const int gap  = 8;
    const int btnH = 34;
    const int w = getWidth();
    const int h = getHeight();

    const int y = padY + (h - padY * 2 - btnH) / 2;

    const auto btnFont = reamix::theme::uiFont (14.0f, 500);
    const auto kbdFont = reamix::theme::monoFont (9.0f, 500);
    const int iconW   = 12;
    const int iconGap = 6;
    const int padBtn  = 12;
    const int kbdPadL = 6;
    const int kbdPadX = 4;

    auto measureBtn = [&] (const juce::String& label, const juce::String& kbd)
    {
        const int textW = (int) std::ceil (measureText (btnFont, label));
        const int kbdW  = (int) std::ceil (measureText (kbdFont, kbd))
                          + kbdPadX * 2;
        return padBtn + iconW + iconGap + textW + kbdPadL + kbdW + padBtn;
    };

    // Size for the widest label ("Preview" — "Stop" is shorter) so the
    // button does not reflow when the user toggles.
    const int playStopW = measureBtn ("Preview", "Space");

    playStopBounds_ = juce::Rectangle<int> (padX, y, playStopW, btnH);

    // DEV-020 (session 49 v2) — cap Insert at ≈2× PlayStop, flush RIGHT edge.
    // Earlier v1 centered Insert in the leftover column but user review:
    // "może powinien być bardziej przy prawej krawędzi a nie po środku".
    // Right-align reads more "primary action" without rozpycha-ing: the gap
    // sits between the buttons, not outside Insert. Mockup spec `grid-
    // template-columns: auto auto 1fr` + `.rx-btn.full { width: 100% }` was
    // 3-button (Preview | Stop | Insert) — ADR-038 collapsed to 2, so we
    // manually place Insert at the column's right edge with a natural
    // upper-bound width.
    const int insertCol     = playStopBounds_.getRight() + gap;
    const int availableColW = juce::jmax (80, w - padX - insertCol);
    const int insertNatW    = measureBtn ("Insert", "Enter");
    const int insertTarget  = juce::jmax (insertNatW, 2 * playStopW);
    const int insertW       = juce::jmin (insertTarget, availableColW);
    const int insertX       = w - padX - insertW; // flush to right edge
    insertBounds_  = juce::Rectangle<int> (insertX, y, insertW, btnH);
}

void TransportBar::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds();

    // plugin.css:424 bg Bg1 + L425 top 1px Line
    g.fillAll (Bg1);
    g.setColour (Line);
    g.fillRect (bounds.withHeight (1));

    const auto paintBtn = [&] (Button which,
                               juce::Rectangle<int> area,
                               const juce::String& label,
                               const juce::String& kbdText,
                               const juce::Colour& baseBg,
                               const juce::Colour& hoverBg,
                               const juce::Colour& borderColour,
                               const juce::Colour& textColour,
                               const juce::Colour& disabledBg,
                               const juce::Colour& disabledText,
                               auto iconFn)
    {
        const bool enabled  = isEnabledFor (which);
        const bool isHover  = enabled && hover_   == which;
        const bool isDown   = enabled && pressed_ == which;

        juce::Colour bg   = baseBg;
        juce::Colour fg   = textColour;
        juce::Colour bord = borderColour;
        if (! enabled) { bg = disabledBg; fg = disabledText; bord = juce::Colours::transparentBlack; }
        else if (isDown) bg = baseBg.darker (0.10f);
        else if (isHover) bg = hoverBg;

        const auto areaF = area.toFloat();
        g.setColour (bg);
        g.fillRoundedRectangle (areaF, r::R3);
        if (! bord.isTransparent())
        {
            g.setColour (bord);
            g.drawRoundedRectangle (areaF.reduced (0.5f), r::R3, 1.0f);
        }

        // Icon + label + kbd chip layout.
        const int iconW = 12;
        const int iconGap = 6;
        const int kbdPadL = 6;
        const int kbdPadX = 4;

        const auto btnFont = reamix::theme::uiFont (14.0f, 500);
        const auto kbdFont = reamix::theme::monoFont (9.0f, 500);
        const int labelW = (int) std::ceil (measureText (btnFont, label));
        const int kbdTextW = (int) std::ceil (measureText (kbdFont, kbdText));
        const int kbdW = kbdTextW + kbdPadX * 2;

        const int totalW = iconW + iconGap + labelW + kbdPadL + kbdW;
        const int startX = area.getX() + (area.getWidth() - totalW) / 2;

        const auto iconArea = juce::Rectangle<float> (
            (float) startX,
            (float) area.getCentreY() - iconW * 0.5f,
            (float) iconW, (float) iconW);
        iconFn (g, iconArea, fg);

        g.setFont (btnFont);
        g.setColour (fg);
        g.drawText (label,
                    juce::Rectangle<int> (startX + iconW + iconGap,
                                            area.getY(),
                                            labelW + 2,
                                            area.getHeight()),
                    juce::Justification::centredLeft, false);

        if (enabled)
        {
            // plugin.css:173-183 kbd chip: mono 9, rgba black-ish bg per button
            // variant. For primary/play/stop (colorful bg) use rgba(0,0,0,0.15);
            // for default buttons Fg3/Bg2 (but we don't render default buttons
            // on this bar — always one of the 3 colored variants).
            const juce::Colour chipBg  { 0x26000000 };
            const juce::Colour chipFg  { 0x80000000 };
            const auto chipArea = juce::Rectangle<float> (
                (float) (startX + iconW + iconGap + labelW + kbdPadL),
                (float) area.getCentreY() - 7.0f,
                (float) kbdW,
                14.0f);
            LookAndFeelReamix::drawKbdChip (g, chipArea, kbdText, chipFg, chipBg);
        }
    };

    // ADR-038 single PlayStop toggle — swap label/colors/icon per state_.
    const bool showingStop = (state_ == TransportState::Playing);
    if (showingStop)
    {
        paintBtn (Button::PlayStop, playStopBounds_, "Stop", "Space",
                  Bad,
                  juce::Colour (0xFFE07A60),              // plugin.css:170 hover
                  juce::Colour (0xFFA84D37),              // plugin.css:169 border
                  juce::Colour (0xFF1A0A08),              // plugin.css:169 text
                  juce::Colour::fromFloatRGBA (213/255.0f, 101/255.0f,  76/255.0f, 0.18f),
                  juce::Colour::fromFloatRGBA ( 26/255.0f,  10/255.0f,   8/255.0f, 0.40f),
                  drawStopIcon);
    }
    else
    {
        paintBtn (Button::PlayStop, playStopBounds_, "Preview", "Space",
                  Good,
                  juce::Colour (0xFF7FD79C),              // plugin.css:166 hover
                  juce::Colour (0xFF4A9E68),              // plugin.css:165 border
                  juce::Colour (0xFF0A1A10),              // plugin.css:165 text
                  juce::Colour::fromFloatRGBA (108/255.0f, 194/255.0f, 138/255.0f, 0.18f),
                  juce::Colour::fromFloatRGBA ( 10/255.0f,  26/255.0f,  16/255.0f, 0.40f),
                  drawPlayIcon);
    }

    // plugin.css:156-163 primary (Insert/Update)
    // Sesja 64 — busy state: label swapped to busy label, button disabled
    // (paintBtn disables visually via isEnabledFor → already returns false
    // for Insert when insertBusy_).
    juce::String insertLabel;
    if (insertBusy_ && insertBusyLabel_.isNotEmpty())
        insertLabel = insertBusyLabel_;
    else
        insertLabel = (state_ == TransportState::Inserted)
                        ? juce::String ("Update")
                        : juce::String ("Insert");
    paintBtn (Button::Insert, insertBounds_, insertLabel, "Enter",
              Accent,
              AccentHi,                               // plugin.css:162 hover
              AccentLo,                               // plugin.css:159 border
              juce::Colour (0xFF1B120A),              // plugin.css:158 text
              juce::Colour::fromFloatRGBA (232/255.0f, 161/255.0f,  90/255.0f, 0.25f),
              juce::Colour::fromFloatRGBA ( 27/255.0f,  18/255.0f,  10/255.0f, 0.45f),
              drawInsertIcon);
}

// ── Hit testing ──────────────────────────────────────────────────────

TransportBar::Button TransportBar::buttonAt (juce::Point<int> p) const
{
    if (playStopBounds_.contains (p) && isEnabledFor (Button::PlayStop)) return Button::PlayStop;
    if (insertBounds_  .contains (p) && isEnabledFor (Button::Insert))   return Button::Insert;
    return Button::None;
}

void TransportBar::setHover (Button b)
{
    if (b == hover_) return;
    hover_ = b;
    setMouseCursor (b != Button::None
                     ? juce::MouseCursor::PointingHandCursor
                     : juce::MouseCursor::NormalCursor);
    repaint();
}

void TransportBar::setPressed (Button b)
{
    if (b == pressed_) return;
    pressed_ = b;
    repaint();
}

void TransportBar::invoke (Button b)
{
    if (! isEnabledFor (b)) return;
    switch (b)
    {
        case Button::PlayStop: if (onPlayStop) onPlayStop(); break;
        case Button::Insert:   if (onInsert)   onInsert();   break;
        case Button::None:     break;
    }
}

void TransportBar::mouseMove (const juce::MouseEvent& e)
{
    setHover (buttonAt (e.getPosition()));
}

void TransportBar::mouseExit (const juce::MouseEvent&)
{
    setHover (Button::None);
    setPressed (Button::None);
}

void TransportBar::mouseDown (const juce::MouseEvent& e)
{
    setPressed (buttonAt (e.getPosition()));
}

void TransportBar::mouseUp (const juce::MouseEvent& e)
{
    const auto b = buttonAt (e.getPosition());
    const auto wasPressed = pressed_;
    setPressed (Button::None);
    if (b != Button::None && b == wasPressed)
        invoke (b);
}

bool TransportBar::keyPressed (const juce::KeyPress& key, juce::Component*)
{
    // ADR-038: Space AND Esc both invoke PlayStop (toggle). Esc-as-stop is
    // a redundant shortcut for users who reach for Escape on reflex.
    if (key == juce::KeyPress::spaceKey)   { invoke (Button::PlayStop); return true; }
    if (key == juce::KeyPress::escapeKey)  { invoke (Button::PlayStop); return true; }
    if (key == juce::KeyPress::returnKey)  { invoke (Button::Insert);   return true; }
    return false;
}

} // namespace reamix::ui
