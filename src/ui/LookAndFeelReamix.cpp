#include "LookAndFeelReamix.h"
#include "Theme.h"

#include "BinaryData.h"

namespace reamix::ui
{

namespace
{
    // Lookup table: FaceSlot → {data, size}. Bound to the BinaryData
    // symbols produced by juce_add_binary_data() for the
    // reamix_font_binary target (see CMakeLists.txt).
    struct FontBlob { const char* data; int size; };

    FontBlob fontBlobFor (LookAndFeelReamix::FaceSlot) noexcept;
}

// ── Construction ─────────────────────────────────────────────────────

LookAndFeelReamix::LookAndFeelReamix()
{
    using namespace reamix::theme;

    // Default widget colours — overrides inherit LookAndFeel_V4 defaults
    // then remap the ones we care about to design tokens.
    setColour (juce::ResizableWindow::backgroundColourId, Bg1);
    setColour (juce::Label::textColourId,                 Fg1);
    setColour (juce::Label::backgroundColourId,           juce::Colours::transparentBlack);
    setColour (juce::TextButton::buttonColourId,          Bg4);
    setColour (juce::TextButton::buttonOnColourId,        Accent);
    setColour (juce::TextButton::textColourOnId,          juce::Colour (0xFF1B120A));
    setColour (juce::TextButton::textColourOffId,         Fg1);
    setColour (juce::Slider::backgroundColourId,          Bg3);
    setColour (juce::Slider::trackColourId,               Accent);
    setColour (juce::Slider::thumbColourId,               Fg0);
    setColour (juce::Slider::rotarySliderFillColourId,    Accent);
    setColour (juce::PopupMenu::backgroundColourId,       Bg0);
    setColour (juce::PopupMenu::textColourId,             Fg1);
    setColour (juce::PopupMenu::highlightedBackgroundColourId, Bg3);
    setColour (juce::PopupMenu::highlightedTextColourId,  Fg0);
    setColour (juce::ToggleButton::textColourId,          Fg2);
    setColour (juce::ToggleButton::tickColourId,          Accent);
    setColour (juce::ToggleButton::tickDisabledColourId,  Fg4);
    setColour (juce::ScrollBar::backgroundColourId,       Bg2);
    setColour (juce::ScrollBar::thumbColourId,            Bg4);
    setColour (juce::ScrollBar::trackColourId,            Bg2);

    // ADR-084 sesja 93 — TooltipWindow subtle dark theme. JUCE LookAndFeel_V4
    // ::drawTooltip() reads colors via findColour() on the LAF (NOT on the
    // TooltipWindow component instance), so these setColour calls are the
    // only effective override path. setColour on TooltipWindow instance does
    // nothing for these IDs.
    setColour (juce::TooltipWindow::backgroundColourId,
               juce::Colour (0xF0131210));   // Bg1 z 94% alpha
    setColour (juce::TooltipWindow::textColourId,         Fg1);
    setColour (juce::TooltipWindow::outlineColourId,      LineStrong);
}

// ── Typeface dispatch ────────────────────────────────────────────────

juce::Typeface::Ptr LookAndFeelReamix::getTypefaceForFont (const juce::Font& font)
{
    // Any font whose family resolves to one of our bundled faces
    // returns a cached Typeface::Ptr; everything else falls through
    // to the default (LookAndFeel_V4 base will use platform defaults).
    const auto slot = resolveSlot (font);
    if (slot == FaceSlot::NumSlots)
        return LookAndFeel_V4::getTypefaceForFont (font);
    return typefaceFor (slot);
}

LookAndFeelReamix::FaceSlot LookAndFeelReamix::resolveSlot (const juce::Font& font)
{
    const auto family = font.getTypefaceName();
    const auto style  = font.getTypefaceStyle();

    const bool isMono = (family == reamix::theme::font::MonoFamily);
    const bool isUi   = (family == reamix::theme::font::UiFamily);

    if (! isMono && ! isUi)
        return FaceSlot::NumSlots;

    if (isMono)
    {
        if (style.containsIgnoreCase ("SemiBold")) return FaceSlot::JbmSemiBold;
        if (style.containsIgnoreCase ("Medium"))   return FaceSlot::JbmMedium;
        return FaceSlot::JbmRegular;
    }

    // isUi — Inter
    if (style.containsIgnoreCase ("Bold") && ! style.containsIgnoreCase ("SemiBold"))
        return FaceSlot::InterBold;
    if (style.containsIgnoreCase ("SemiBold"))     return FaceSlot::InterSemiBold;
    if (style.containsIgnoreCase ("Medium"))       return FaceSlot::InterMedium;
    return FaceSlot::InterRegular;
}

juce::Typeface::Ptr LookAndFeelReamix::typefaceFor (FaceSlot slot)
{
    const auto index = static_cast<std::size_t> (slot);
    if (typefaces_[index] == nullptr)
    {
        const auto blob = fontBlobFor (slot);
        if (blob.data != nullptr && blob.size > 0)
            typefaces_[index] = juce::Typeface::createSystemTypefaceFor (blob.data, blob.size);
    }
    return typefaces_[index];
}

// ── Buttons ──────────────────────────────────────────────────────────

void LookAndFeelReamix::drawButtonBackground (juce::Graphics& g, juce::Button& button,
                                              const juce::Colour& /*backgroundColour*/,
                                              bool isMouseOverButton, bool isButtonDown)
{
    using namespace reamix::theme;

    const auto bounds = button.getLocalBounds().toFloat().reduced (0.5f);
    const float radius = r::R3; // plugin.css:137 .rx-btn border-radius: var(--rx-r-3)
    const bool isPrimary = button.getProperties()["rx-variant"].toString() == "primary";
    const bool isPlay    = button.getProperties()["rx-variant"].toString() == "play";
    const bool isStop    = button.getProperties()["rx-variant"].toString() == "stop";
    const bool isGhost   = button.getProperties()["rx-variant"].toString() == "ghost";

    juce::Colour fill = Bg4;             // plugin.css:134 rest
    juce::Colour border = LineStrong;    // plugin.css:136 rest
    if (isPrimary) { fill = Accent;    border = AccentLo; }
    if (isPlay)    { fill = Good;      border = juce::Colour (0xFF4A9E68); } // plugin.css:165
    if (isStop)    { fill = Bad;       border = juce::Colour (0xFFA84D37); } // plugin.css:169
    if (isGhost)   { fill = juce::Colours::transparentBlack; border = fill; }

    if (isMouseOverButton)
    {
        if (isPrimary)      fill = AccentHi;
        else if (isPlay)    fill = juce::Colour (0xFF7FD79C); // plugin.css:166
        else if (isStop)    fill = juce::Colour (0xFFE07A60); // plugin.css:170
        else if (isGhost)   fill = Bg3;
        else                fill = Bg5;                        // plugin.css:144
    }
    if (isButtonDown && ! isPrimary && ! isPlay && ! isStop)
        fill = Bg3;                                            // plugin.css:145

    if (! button.isEnabled())
    {
        // plugin.css:146-149 disabled rest
        fill = Bg3;
        border = Line;
    }

    g.setColour (fill);
    g.fillRoundedRectangle (bounds, radius);
    if (! border.isTransparent())
    {
        g.setColour (border);
        g.drawRoundedRectangle (bounds, radius, 1.0f);
    }
}

void LookAndFeelReamix::drawButtonText (juce::Graphics& g, juce::TextButton& button,
                                        bool /*isMouseOverButton*/, bool /*isButtonDown*/)
{
    using namespace reamix::theme;

    const bool isPrimary = button.getProperties()["rx-variant"].toString() == "primary";
    const bool isPlay    = button.getProperties()["rx-variant"].toString() == "play";
    const bool isStop    = button.getProperties()["rx-variant"].toString() == "stop";

    // plugin.css:139 font-family var(--rx-font-ui), plugin.css:139 13px / 500 default
    int   weight = 500;
    float size   = fs::Md;
    if (isPrimary || isPlay || isStop) weight = 600; // plugin.css:161/165/169
    g.setFont (uiFont (size, weight));

    juce::Colour textColour = Fg1; // plugin.css:135
    if (isPrimary) textColour = juce::Colour (0xFF1B120A); // plugin.css:158
    if (isPlay)    textColour = juce::Colour (0xFF0A1A10); // plugin.css:165
    if (isStop)    textColour = juce::Colour (0xFF1A0A08); // plugin.css:169
    if (! button.isEnabled())
        textColour = Fg3; // plugin.css:147

    g.setColour (textColour);
    const int yIndent = juce::jmin (4, button.proportionOfHeight (0.3f));
    const int cornerSize = juce::jmin (button.getHeight(), button.getWidth()) / 2;
    const int fontHeight = juce::roundToInt (g.getCurrentFont().getHeight() * 0.6f);
    const int leftIndent  = juce::jmin (fontHeight, 2 + cornerSize / (button.isConnectedOnLeft()  ? 4 : 2));
    const int rightIndent = juce::jmin (fontHeight, 2 + cornerSize / (button.isConnectedOnRight() ? 4 : 2));
    const int textWidth = button.getWidth() - leftIndent - rightIndent;
    if (textWidth > 0)
        g.drawFittedText (button.getButtonText(),
                          leftIndent, yIndent,
                          textWidth, button.getHeight() - yIndent * 2,
                          juce::Justification::centred, 2);
}

// ── Linear slider (simplified) ───────────────────────────────────────

void LookAndFeelReamix::drawLinearSlider (juce::Graphics& g,
                                          int x, int y, int width, int height,
                                          float sliderPos, float /*minSliderPos*/,
                                          float /*maxSliderPos*/,
                                          juce::Slider::SliderStyle,
                                          juce::Slider& slider)
{
    using namespace reamix::theme;

    // plugin.css:253-262 track + fill, 4px thickness; thumb 14×14 circle
    // with 2px accent border per plugin.css:264-271. Sesja 100 iter 4 —
    // added vertical-orientation branch (was hardcoded horizontal pre-iter-4,
    // which broke DEV-018 inline volume popup that uses LinearVertical style).
    // Sesja 107 iter-4 — thumb tightened 14 → 10 px on user feedback.
    // Track stays 4 px (same accent fill + Bg3 rail metric).
    const float trackThick = 4.0f;
    const float thumbDiameter = 10.0f;
    const auto bounds = juce::Rectangle<float> ((float) x, (float) y, (float) width, (float) height);
    const bool disabled = ! slider.isEnabled();

    if (slider.isVertical())
    {
        // Vertical track: centred horizontally, full bounds height. JUCE
        // passes sliderPos as the thumb's Y coordinate (low Y = high value
        // because top is max for LinearVertical).
        auto track = juce::Rectangle<float> (bounds.getCentreX() - trackThick * 0.5f,
                                              bounds.getY(),
                                              trackThick,
                                              bounds.getHeight());
        g.setColour (Bg3);
        g.fillRoundedRectangle (track, 2.0f);

        // Fill: from thumb Y down to bottom of track (high value = full fill
        // from bottom-to-top below thumb). Mirror of horizontal "fill from
        // left to thumb X".
        auto fill = juce::Rectangle<float> (track.getX(),
                                             sliderPos,
                                             track.getWidth(),
                                             track.getBottom() - sliderPos);
        g.setColour (disabled ? Fg4 : Accent);
        g.fillRoundedRectangle (fill, 2.0f);

        juce::Rectangle<float> thumb (bounds.getCentreX() - thumbDiameter * 0.5f,
                                       sliderPos - thumbDiameter * 0.5f,
                                       thumbDiameter, thumbDiameter);
        g.setColour (disabled ? Fg3 : Fg0);
        g.fillEllipse (thumb);
        g.setColour (disabled ? Fg4 : Accent);
        g.drawEllipse (thumb, 1.0f);
        return;
    }

    // Horizontal (default existing path).
    auto track = juce::Rectangle<float> (bounds.getX(),
                                          bounds.getCentreY() - trackThick * 0.5f,
                                          bounds.getWidth(), trackThick);

    g.setColour (Bg3);
    g.fillRoundedRectangle (track, 2.0f);

    auto fill = track.withWidth (juce::jlimit (0.0f, track.getWidth(), sliderPos - track.getX()));
    g.setColour (disabled ? Fg4 : Accent);
    g.fillRoundedRectangle (fill, 2.0f);

    juce::Rectangle<float> thumb (sliderPos - thumbDiameter * 0.5f,
                                   bounds.getCentreY() - thumbDiameter * 0.5f,
                                   thumbDiameter, thumbDiameter);
    g.setColour (disabled ? Fg3 : Fg0);
    g.fillEllipse (thumb);
    g.setColour (disabled ? Fg4 : Accent);
    g.drawEllipse (thumb, 2.0f);
}

// ── Popup menu ──────────────────────────────────────────────────────

void LookAndFeelReamix::drawPopupMenuBackground (juce::Graphics& g, int width, int height)
{
    using namespace reamix::theme;

    // plugin.css:583-591 bg-0, 1px LineStrong border, r2 radius, elev-pop shadow
    auto bounds = juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height).reduced (0.5f);
    g.setColour (Bg0);
    g.fillRoundedRectangle (bounds, r::R2);
    g.setColour (LineStrong);
    g.drawRoundedRectangle (bounds, r::R2, 1.0f);
}

void LookAndFeelReamix::drawPopupMenuItem (juce::Graphics& g,
                                           const juce::Rectangle<int>& area,
                                           bool isSeparator, bool isActive, bool isHighlighted,
                                           bool /*isTicked*/, bool /*hasSubMenu*/,
                                           const juce::String& text,
                                           const juce::String& shortcutKeyText,
                                           const juce::Drawable* /*icon*/,
                                           const juce::Colour* textColour)
{
    using namespace reamix::theme;

    if (isSeparator)
    {
        // plugin.css:601 1px Line, 4px horizontal margin
        auto sep = area.toFloat().withSizeKeepingCentre ((float) area.getWidth() - 4.0f, 1.0f);
        g.setColour (Line);
        g.fillRect (sep);
        return;
    }

    auto bounds = area.reduced (4, 2).toFloat(); // plugin.css:594 padding 6px 8px — JUCE pads via area
    if (isHighlighted && isActive)
    {
        g.setColour (Bg3);
        g.fillRoundedRectangle (bounds, r::R1);
    }

    const auto fg = isHighlighted ? Fg0 : (textColour != nullptr ? *textColour : Fg1);
    g.setColour (isActive ? fg : Fg3);
    g.setFont (uiFont (fs::Sm, 500));
    g.drawText (text,
                bounds.reduced (6.0f, 0.0f),
                juce::Justification::centredLeft, true);

    if (shortcutKeyText.isNotEmpty())
    {
        g.setColour (Fg3);
        g.setFont (monoFont (10.0f, 500));
        g.drawText (shortcutKeyText,
                    bounds.reduced (6.0f, 0.0f),
                    juce::Justification::centredRight, true);
    }
}

// ── Label ────────────────────────────────────────────────────────────

void LookAndFeelReamix::drawLabel (juce::Graphics& g, juce::Label& label)
{
    using namespace reamix::theme;

    g.fillAll (label.findColour (juce::Label::backgroundColourId));

    if (! label.isBeingEdited())
    {
        const auto alpha = label.isEnabled() ? 1.0f : 0.5f;
        auto font = label.getFont();
        if (font.getTypefaceName() == "<Sans-Serif>")
            font = uiFont (font.getHeight(), 400);
        g.setColour (label.findColour (juce::Label::textColourId).withMultipliedAlpha (alpha));
        g.setFont (font);

        auto textArea = label.getBorderSize().subtractedFrom (label.getLocalBounds());
        g.drawFittedText (label.getText(), textArea, label.getJustificationType(),
                          juce::jmax (1, (int) ((float) textArea.getHeight() / font.getHeight())),
                          label.getMinimumHorizontalScale());

        g.setColour (label.findColour (juce::Label::outlineColourId).withMultipliedAlpha (alpha));
    }
    else if (label.isEnabled())
    {
        g.setColour (label.findColour (juce::Label::outlineColourId));
    }

    g.drawRect (label.getLocalBounds());
}

// ── Tooltip ──────────────────────────────────────────────────────────
//
// Sesja 107 iter-4 — redesigned for premium feel per user feedback "Tooltipy
// powinny byc glenboko przemyslane by ladnie wygaldaly i by byly na
// przezrovzystym tle". Default JUCE_V4 tooltip is a flat opaque rectangle.
//
// Design tokens:
//   - Background: Bg1 @ 92% alpha (~0xEB) — semi-transparent so the layer
//     beneath bleeds through and the popup feels light, not stamped on top.
//   - Border: LineStrong, 1px, R2 corners (matches RxPanel pillshape).
//   - Shadow: 8 px blur Black @ 35% alpha, offset (0, 4) — same elevation
//     curve as the gear popover (ElevPop) but slightly softer.
//   - Padding: 10×6 (horiz × vert).
//   - Font: Inter Regular Sm (~12 px) in Fg0 with 1.35 line height.
//   - Max width 360 px; multi-line wrap above that.
//
// Tooltip text is rendered with line-breaks honoured (\n in setTooltip text
// translates to actual line breaks in the rendered popup).

namespace
{
    constexpr int   kTooltipPaddingX = 10;
    constexpr int   kTooltipPaddingY = 6;
    constexpr int   kTooltipMaxWidth = 360;
    constexpr float kTooltipFontSize = 12.0f;
    constexpr float kTooltipLineH    = 1.35f;
    constexpr int   kTooltipShadowMargin = 8;

    juce::TextLayout buildTooltipLayout (const juce::String& text, int wrapWidth)
    {
        using namespace reamix::theme;
        juce::AttributedString s;
        s.setText (text);
        s.setJustification (juce::Justification::topLeft);
        s.setWordWrap (juce::AttributedString::byWord);
        s.setLineSpacing (kTooltipLineH);
        s.setColour (Fg0);
        s.setFont (uiFont (kTooltipFontSize, 400));
        juce::TextLayout layout;
        layout.createLayout (s, (float) wrapWidth);
        return layout;
    }
}

juce::Rectangle<int> LookAndFeelReamix::getTooltipBounds (const juce::String& tipText,
                                                          juce::Point<int> screenPos,
                                                          juce::Rectangle<int> parentArea)
{
    const int wrap = juce::jmin (kTooltipMaxWidth, parentArea.getWidth() - 2 * kTooltipShadowMargin);
    const auto layout = buildTooltipLayout (tipText, wrap);

    const int textW = (int) std::ceil (layout.getWidth());
    const int textH = (int) std::ceil (layout.getHeight());

    const int w = textW + 2 * kTooltipPaddingX;
    const int h = textH + 2 * kTooltipPaddingY;

    // Anchor below + right of the cursor by 14 px so the tip does not sit
    // under the pointer hot spot. Clamp inside parent area minus shadow.
    int x = screenPos.x + 14;
    int y = screenPos.y + 18;
    if (x + w > parentArea.getRight() - kTooltipShadowMargin)
        x = screenPos.x - w - 6;
    if (y + h > parentArea.getBottom() - kTooltipShadowMargin)
        y = screenPos.y - h - 6;
    x = juce::jmax (parentArea.getX() + kTooltipShadowMargin, x);
    y = juce::jmax (parentArea.getY() + kTooltipShadowMargin, y);

    return { x, y, w, h };
}

void LookAndFeelReamix::drawTooltip (juce::Graphics& g, const juce::String& text,
                                     int width, int height)
{
    using namespace reamix::theme;

    const auto bounds = juce::Rectangle<int> (0, 0, width, height).toFloat();

    // Soft drop shadow under the pill.
    juce::DropShadow sh (juce::Colours::black.withAlpha (0.35f),
                          8, { 0, 4 });
    sh.drawForRectangle (g, bounds.toNearestInt());

    // Semi-transparent body — Bg1 @ ~80% alpha so the underlying canvas
    // tints through visibly (iter-7 user feedback: "wiecej przezroczystosci").
    const auto body = bounds;
    g.setColour (Bg1.withAlpha (0.80f));
    g.fillRoundedRectangle (body, r::R2);

    // Subtle outline — 0.6 px LineStrong (avoids harsh 1 px on retina).
    g.setColour (LineStrong.withAlpha (0.85f));
    g.drawRoundedRectangle (body.reduced (0.5f), r::R2, 0.6f);

    // Inner text — re-layout against actual width since drawTooltip is
    // called from a separate render path than getTooltipBounds.
    const int textW = width - 2 * kTooltipPaddingX;
    const auto layout = buildTooltipLayout (text, textW);
    layout.draw (g, body.reduced ((float) kTooltipPaddingX,
                                  (float) kTooltipPaddingY));
}

// ── Toggle button ────────────────────────────────────────────────────

void LookAndFeelReamix::drawToggleButton (juce::Graphics& g, juce::ToggleButton& button,
                                          bool isMouseOverButton, bool /*isButtonDown*/)
{
    using namespace reamix::theme;

    // plugin.css:451-458 14×14 box, 2px radius, checkmark on active
    const float boxSize = 14.0f;
    const auto bounds = button.getLocalBounds().toFloat();
    juce::Rectangle<float> box (bounds.getX(),
                                 bounds.getCentreY() - boxSize * 0.5f,
                                 boxSize, boxSize);

    const bool on = button.getToggleState();
    g.setColour (on ? Accent : Bg3);
    g.fillRoundedRectangle (box, r::R1);
    g.setColour (on ? AccentLo : LineStrong);
    g.drawRoundedRectangle (box, r::R1, 1.0f);

    if (on)
    {
        // Checkmark: simple polyline, plugin.css:458 rotate(-45) translate(1,-1)
        juce::Path check;
        const float cx = box.getX() + 4.0f;
        const float cy = box.getCentreY();
        check.startNewSubPath (cx,          cy + 1.0f);
        check.lineTo          (cx + 3.0f,   cy + 3.5f);
        check.lineTo          (cx + 7.5f,   cy - 2.0f);
        g.setColour (juce::Colour (0xFF1B120A));
        g.strokePath (check, juce::PathStrokeType (1.5f));
    }

    // Label text
    const auto text = button.getButtonText();
    if (text.isNotEmpty())
    {
        g.setColour (isMouseOverButton ? Fg1 : Fg2);
        g.setFont (uiFont (fs::Sm, 400));
        auto textArea = bounds.withLeft (box.getRight() + 7.0f); // plugin.css:446 gap 7px
        g.drawText (text, textArea, juce::Justification::centredLeft, true);
    }
}

// ── Custom reamix primitives (placeholder impls) ─────────────────────

void LookAndFeelReamix::drawSpliceMarker (juce::Graphics& g, juce::Rectangle<float> area,
                                          const juce::Colour& qualityColour)
{
    // plugin.css:400-416 — 1px line + 9×7 triangle cap at top.
    // Step-1 placeholder. Full hit-testing + hover behavior lands step 7.
    g.setColour (qualityColour);
    g.fillRect (juce::Rectangle<float> (area.getCentreX() - 0.5f, area.getY(),
                                         1.0f, area.getHeight()));
    juce::Path tri;
    tri.startNewSubPath (area.getCentreX() - 4.5f, area.getY() - 1.0f);
    tri.lineTo          (area.getCentreX() + 4.5f, area.getY() - 1.0f);
    tri.lineTo          (area.getCentreX(),         area.getY() + 6.0f);
    tri.closeSubPath();
    g.fillPath (tri);
}

void LookAndFeelReamix::drawPillBadge (juce::Graphics& g, juce::Rectangle<float> area,
                                       const juce::String& text,
                                       const juce::Colour& textColour,
                                       const juce::Colour& backgroundColour)
{
    // plugin.css:462-473 — full-radius pill, uppercase xs text, 2px/8px padding.
    using namespace reamix::theme;
    g.setColour (backgroundColour);
    g.fillRoundedRectangle (area, area.getHeight() * 0.5f);
    g.setColour (textColour);
    g.setFont (uiFont (fs::Xs, 600));
    g.drawText (text.toUpperCase(), area, juce::Justification::centred, true);
}

void LookAndFeelReamix::drawKbdChip (juce::Graphics& g, juce::Rectangle<float> area,
                                     const juce::String& text,
                                     const juce::Colour& textColour,
                                     const juce::Colour& backgroundColour)
{
    // plugin.css:173-180 — 14h, 4px H-padding, 2px radius, mono 9px.
    using namespace reamix::theme;
    g.setColour (backgroundColour);
    g.fillRoundedRectangle (area, 2.0f);
    g.setColour (textColour);
    g.setFont (monoFont (9.0f, 500));
    g.drawText (text, area, juce::Justification::centred, true);
}

void LookAndFeelReamix::drawBrandLogo (juce::Graphics& g,
                                        juce::Rectangle<float> bounds)
{
    // DEV-082 sesja 112 — parse the bundled SVG once, cache the resulting
    // Drawable as a function-local static, then draw within whatever rect
    // the caller hands us. SVG mask was baked into ray geometry pre-bundle
    // so JUCE's SVG parser (which does not honour `<mask>`) renders the
    // hollow ring correctly out of the box.
    static const std::unique_ptr<juce::Drawable> drawable =
        juce::Drawable::createFromImageData (
            BinaryData::reamixlogo_svg,
            (std::size_t) BinaryData::reamixlogo_svgSize);
    if (drawable == nullptr)
        return;
    drawable->drawWithin (g, bounds,
                          juce::RectanglePlacement::centred,
                          1.0f);
}

// ── Font blob lookup — binds FaceSlot to BinaryData symbols ──────────

namespace
{
    FontBlob fontBlobFor (LookAndFeelReamix::FaceSlot slot) noexcept
    {
        using Slot = LookAndFeelReamix::FaceSlot;
        switch (slot)
        {
            case Slot::InterRegular:
                return { BinaryData::InterRegular_ttf,         BinaryData::InterRegular_ttfSize };
            case Slot::InterMedium:
                return { BinaryData::InterMedium_ttf,          BinaryData::InterMedium_ttfSize };
            case Slot::InterSemiBold:
                return { BinaryData::InterSemiBold_ttf,        BinaryData::InterSemiBold_ttfSize };
            case Slot::InterBold:
                return { BinaryData::InterBold_ttf,            BinaryData::InterBold_ttfSize };
            case Slot::JbmRegular:
                return { BinaryData::JetBrainsMonoRegular_ttf,  BinaryData::JetBrainsMonoRegular_ttfSize };
            case Slot::JbmMedium:
                return { BinaryData::JetBrainsMonoMedium_ttf,   BinaryData::JetBrainsMonoMedium_ttfSize };
            case Slot::JbmSemiBold:
                return { BinaryData::JetBrainsMonoSemiBold_ttf, BinaryData::JetBrainsMonoSemiBold_ttfSize };
            case Slot::NumSlots:
                break;
        }
        return { nullptr, 0 };
    }
}

} // namespace reamix::ui
