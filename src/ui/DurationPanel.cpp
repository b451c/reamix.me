#include "DurationPanel.h"
#include "Theme.h"

#include <cmath>

namespace reamix::ui
{

namespace
{
    juce::String rxFmt (double seconds)
    {
        const int s = juce::jmax (0, (int) std::round (seconds));
        const int m = s / 60;
        const int ss = s % 60;
        return juce::String (m) + ":" + (ss < 10 ? "0" : "") + juce::String (ss);
    }

    juce::String rxFmtSigned (double deltaSec)
    {
        // primitives.jsx:50-53 — '−' (minus) or '+' prefix + rxFmt(|delta|).
        const juce::String sign = deltaSec < 0.0
            ? juce::String::fromUTF8 ("\xe2\x88\x92") // U+2212
            : juce::String ("+");
        return sign + rxFmt (std::abs (deltaSec));
    }

    float measureText (const juce::Font& font, const juce::String& text)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (font, text, 0.0f, 0.0f);
        return ga.getBoundingBox (0, -1, false).getWidth();
    }
}

DurationPanel::DurationPanel()
{
    setInterceptsMouseClicks (true, false);
    setOpaque (true);
}

void DurationPanel::setRange (double minSec, double maxSec)
{
    minSec_ = minSec;
    maxSec_ = juce::jmax (minSec + 1.0, maxSec);
    target_ = juce::jlimit (minSec_, maxSec_, target_);
    repaint();
}

void DurationPanel::setOriginalDuration (double origSec)
{
    origSec_ = origSec;
    repaint();
}

void DurationPanel::setTarget (double sec)
{
    const double clamped = juce::jlimit (minSec_, maxSec_, sec);
    if (std::abs (clamped - target_) < 1.0e-6)
        return;
    target_ = clamped;
    repaint();
}

void DurationPanel::setActive (bool on) noexcept
{
    if (on == enabled_)
        return;
    enabled_ = on;
    repaint();
}

void DurationPanel::setRegion (std::optional<RegionInfo> info)
{
    // Detect material change (start/end shift or activate/deactivate).
    const bool wasActive = regionInfo_.has_value();
    const bool isActive  = info.has_value();
    const bool changed   = (wasActive != isActive)
        || (isActive && wasActive
            && (regionInfo_->startSec != info->startSec
                || regionInfo_->endSec != info->endSec));
    if (! changed) return;
    regionInfo_ = info;
    repaint();
}

int DurationPanel::xFromTarget (double sec) const
{
    const double range = juce::jmax (1.0, maxSec_ - minSec_);
    const double p = (sec - minSec_) / range;
    return trackBounds_.getX()
         + (int) std::round (p * (double) trackBounds_.getWidth());
}

double DurationPanel::targetFromMouseX (int x) const
{
    const int x0 = trackBounds_.getX();
    const int w  = juce::jmax (1, trackBounds_.getWidth());
    const double p = juce::jlimit (0.0, 1.0, (double) (x - x0) / (double) w);
    const double v = minSec_ + p * (maxSec_ - minSec_);
    return std::round (v); // shell.jsx:97 integer seconds
}

void DurationPanel::resized()
{
    const int padX = 12;
    const int padTop = 14;
    const int padBot = 12;
    const int w = getWidth();
    const int h = getHeight();

    const int readoutColW = 156; // plugin.css § .rx-duration-row min-width
    const int colGap = 14;       // plugin.css:223 .rx-duration-row gap

    readoutCol_ = juce::Rectangle<int> (padX, padTop,
                                         readoutColW,
                                         h - padTop - padBot);

    const int sliderX = padX + readoutColW + colGap;
    const int sliderW = juce::jmax (40, w - sliderX - padX);
    sliderCol_ = juce::Rectangle<int> (sliderX, padTop, sliderW,
                                         h - padTop - padBot);

    // Slider stack inside sliderCol_: 22h interaction strip + 8px gap + ~14h tickrow.
    const int trackH = 4;
    const int hitH   = 22;
    const int gap    = 8;
    const int tickH  = 14;
    const int stackH = hitH + gap + tickH;
    const int stackY = sliderCol_.getY() + (sliderCol_.getHeight() - stackH) / 2;

    trackHitArea_ = juce::Rectangle<int> (sliderCol_.getX(), stackY,
                                            sliderCol_.getWidth(), hitH);
    trackBounds_  = juce::Rectangle<int> (sliderCol_.getX(),
                                            trackHitArea_.getCentreY() - trackH / 2,
                                            sliderCol_.getWidth(), trackH);
    tickrowBounds_ = juce::Rectangle<int> (sliderCol_.getX(),
                                             trackHitArea_.getBottom() + gap,
                                             sliderCol_.getWidth(), tickH);
}

void DurationPanel::paint (juce::Graphics& g)
{
    using namespace reamix::theme;

    const auto bounds = getLocalBounds();

    // plugin.css:218-221 bg Bg1 + bottom 1px Line
    g.fillAll (Bg1);
    g.setColour (Line);
    g.fillRect (bounds.withHeight (1).withY (bounds.getBottom() - 1));

    // ── Readout column ──
    // plugin.css:225-245
    //   .rx-readout-label: Xs (9px), 0.12em caps, Fg3
    //   .rx-readout: Display (28px), Mono 500, Fg0, line-height 1
    //   .delta: Accent 16px margin-left 8
    {
        const int rx = readoutCol_.getX();
        const int ry = readoutCol_.getY();
        const int rw = readoutCol_.getWidth();

        // Label — "Target duration" uppercase with 0.12em caps. JUCE Font has
        // no letter-spacing API; render uppercase for visual parity.
        // Region mode (sesja 60): label flips to "REGION · LIVE" with Accent
        // color (mockup shell.jsx:147 .rx-readout-label.live).
        const auto labelFont = uiFont (fs::Xs, 500);
        g.setFont (labelFont);
        const bool isRegion = regionInfo_.has_value();
        g.setColour (isRegion ? Accent : Fg3);
        const juce::String labelText = isRegion
            ? juce::String::fromUTF8 ("REGION \xc2\xb7 LIVE")
            : juce::String ("TARGET DURATION");
        g.drawText (labelText,
                    juce::Rectangle<int> (rx, ry, rw, 12),
                    juce::Justification::centredLeft, false);

        // Readout value — Display mono 500. Fg3 if disabled.
        const auto valFont = monoFont (fs::Display, 500);
        const juce::String valText = rxFmt (target_);
        const float valW = measureText (valFont, valText);
        g.setFont (valFont);
        g.setColour (enabled_ ? Fg0 : Fg3);
        const int valY = ry + 12 + 4;
        g.drawText (valText,
                    juce::Rectangle<int> (rx, valY, (int) std::ceil (valW) + 4, 30),
                    juce::Justification::centredLeft, false);

        // Delta chip — Accent 16px to the right of the readout, only when
        // non-zero and enabled.
        const double delta = target_ - origSec_;
        if (enabled_ && std::abs (delta) >= 0.5)
        {
            const auto deltaFont = monoFont (16.0f, 500);
            const juce::String deltaText = rxFmtSigned (delta);
            g.setFont (deltaFont);
            g.setColour (Accent);
            const int deltaX = rx + (int) std::ceil (valW) + 8;
            g.drawText (deltaText,
                        juce::Rectangle<int> (deltaX, valY + 12,
                                                rw - (deltaX - rx),
                                                20),
                        juce::Justification::centredLeft, false);
        }
    }

    // ── Slider column ──
    // plugin.css:249-277
    //   .rx-slider-track: 4h, Bg3, r2
    //   .rx-slider-fill: Accent, r2 (disabled: Fg4)
    //   .rx-slider-thumb: 14×14, Fg0 bg, 2px Accent border (disabled: Fg4 border + Fg3 bg)
    //   .rx-slider-thumb.orig: 10×10, transparent, dashed Fg3 border
    {
        const auto track = trackBounds_.toFloat();
        g.setColour (Bg3);
        g.fillRoundedRectangle (track, 2.0f);

        const int targetX = xFromTarget (target_);
        const float fillW = (float) (targetX - trackBounds_.getX());
        if (fillW > 0.0f)
        {
            g.setColour (enabled_ ? Accent : Fg4);
            g.fillRoundedRectangle (track.withWidth (fillW), 2.0f);
        }

        // ORIG thumb — dashed 10×10 square (mockup shows circle with dashed
        // border; JUCE's drawRoundedRectangle with dashed stroke matches the
        // visual intent closely).
        const int origX = xFromTarget (origSec_);
        const juce::Rectangle<float> origThumb (
            origX - 5.0f, track.getCentreY() - 5.0f, 10.0f, 10.0f);

        // Sesja 64 BUG-1 — hover halo on ORIG marker (clickable hint).
        if (hoveringOrigMarker_)
        {
            const juce::Rectangle<float> halo = origThumb.expanded (4.0f);
            g.setColour (AccentDim);
            g.fillEllipse (halo);
        }
        g.setColour (juce::Colours::transparentBlack);
        g.fillRoundedRectangle (origThumb, 5.0f);
        g.setColour (hoveringOrigMarker_ ? AccentHi : Fg3);
        const float dashLen[] = { 2.0f, 2.0f };
        juce::Path origRing;
        origRing.addEllipse (origThumb);
        juce::PathStrokeType stroke (hoveringOrigMarker_ ? 1.5f : 1.0f);
        juce::Path dashed;
        stroke.createDashedStroke (dashed, origRing, dashLen, 2);
        g.fillPath (dashed);

        // Target thumb — 14×14 circle, Fg0 fill + 2px Accent border.
        const juce::Rectangle<float> thumb (
            targetX - 7.0f, track.getCentreY() - 7.0f, 14.0f, 14.0f);
        g.setColour (enabled_ ? Fg0 : Fg3);
        g.fillEllipse (thumb);
        g.setColour (enabled_ ? Accent : Fg4);
        g.drawEllipse (thumb.reduced (1.0f), 2.0f);
    }

    // ── Tickrow ──
    // plugin.css:279-285 — justify-between, mono 10px, Fg3, "ORIG" bold Fg3 + value
    // Region mode: center label switches to "REGION M:SS-M:SS" so the user
    // sees the live REAPER selection bounds without a separate panel.
    {
        const auto font = monoFont (10.0f, 400);
        g.setFont (font);
        g.setColour (Fg3);

        const juce::String minStr  = rxFmt (minSec_);
        const juce::String maxStr  = rxFmt (maxSec_);
        const juce::String centerStr = regionInfo_.has_value()
            ? juce::String ("REGION ")
                + rxFmt (regionInfo_->startSec)
                + juce::String::fromUTF8 ("\xe2\x80\x93") // U+2013 en-dash
                + rxFmt (regionInfo_->endSec)
            : juce::String ("ORIG ") + rxFmt (origSec_);

        g.drawText (minStr, tickrowBounds_,
                    juce::Justification::centredLeft, false);
        g.drawText (maxStr, tickrowBounds_,
                    juce::Justification::centredRight, false);

        // Sesja 64 BUG-1 — hover state on ORIG label: subtle accent-tinted
        // pill behind text + accent-colored text, signals clickable.
        if (hoveringOrigLabel_ && ! regionInfo_.has_value())
        {
            const float labelW = measureText (font, centerStr);
            const float padH   = 6.0f;
            const float padV   = 2.0f;
            juce::Rectangle<float> pill (
                (float) tickrowBounds_.getCentreX() - labelW * 0.5f - padH,
                (float) tickrowBounds_.getY() + (float) tickrowBounds_.getHeight() * 0.5f
                    - (float) font.getHeight() * 0.5f - padV,
                labelW + padH * 2.0f,
                (float) font.getHeight() + padV * 2.0f);
            g.setColour (AccentDim);
            g.fillRoundedRectangle (pill, 3.0f);
            g.setColour (AccentLo);
            g.drawRoundedRectangle (pill.reduced (0.5f), 3.0f, 1.0f);
            g.setColour (AccentHi);
        }
        else
        {
            g.setColour (Fg3);
        }
        g.drawText (centerStr, tickrowBounds_,
                    juce::Justification::centred, false);
    }
}

// ── Mouse ─────────────────────────────────────────────────────────────

void DurationPanel::mouseDown (const juce::MouseEvent& e)
{
    if (! enabled_) return;
    const auto p = e.getPosition();

    // Sesja 64 BUG-1 enhancement #2 — clicking the "ORIG M:SS" label below
    // the slider also resets target to ORIG. Label sits in tickrowBounds_,
    // OUTSIDE trackHitArea_, so without this branch the click is ignored.
    // Approximate label as middle 1/3 of tickrow (label is centered, width
    // varies with rxFmt). Hidden when in Region mode (label flips to
    // "REGION M:SS-M:SS"); user must use double-click in Region mode.
    if (! regionInfo_.has_value() && tickrowBounds_.contains (p))
    {
        const int third     = tickrowBounds_.getWidth() / 3;
        const int leftEdge  = tickrowBounds_.getX() + third;
        const int rightEdge = tickrowBounds_.getRight() - third;
        if (p.getX() >= leftEdge && p.getX() <= rightEdge)
        {
            if (std::abs (origSec_ - target_) > 1.0e-6)
            {
                target_ = juce::jlimit (minSec_, maxSec_, origSec_);
                repaint();
                if (onTargetChanged) onTargetChanged (target_);
            }
            return;
        }
    }

    if (! trackHitArea_.contains (p)) return;
    dragging_ = true;

    // Sesja 64 BUG-1 enhancement #1 — snap to ORIG when click lands within
    // ±8 px of the ORIG marker on the slider track. Single click anywhere
    // near ORIG resets (analogous to double-click on the track). Prevents
    // rapid clicks accidentally drifting the target by 1-2 seconds.
    const int  origX  = xFromTarget (origSec_);
    constexpr int kOrigSnapPx = 8;
    const double v = (std::abs (p.getX() - origX) <= kOrigSnapPx)
        ? origSec_
        : targetFromMouseX (p.getX());

    if (std::abs (v - target_) > 1.0e-6)
    {
        target_ = v;
        repaint();
        if (onTargetChanged) onTargetChanged (target_);
    }
}

void DurationPanel::mouseDrag (const juce::MouseEvent& e)
{
    if (! dragging_ || ! enabled_) return;
    const double v = targetFromMouseX (e.getPosition().getX());
    if (std::abs (v - target_) > 1.0e-6)
    {
        target_ = v;
        repaint();
        if (onTargetChanged) onTargetChanged (target_);
    }
}

void DurationPanel::mouseUp (const juce::MouseEvent&)
{
    dragging_ = false;
}

// Sesja 64 BUG-1 — hover hit-test for ORIG label + marker. Returns the
// pair (hoveringLabel, hoveringMarker) for the current cursor position.
// Both false in Region mode (label flips to "REGION ..." which is read-only).
namespace {
struct OrigHoverHit { bool label = false; bool marker = false; };
}

void DurationPanel::mouseMove (const juce::MouseEvent& e)
{
    if (! enabled_)
    {
        if (hoveringOrigLabel_ || hoveringOrigMarker_)
        {
            hoveringOrigLabel_ = hoveringOrigMarker_ = false;
            setMouseCursor (juce::MouseCursor::NormalCursor);
            repaint();
        }
        return;
    }

    const auto p = e.getPosition();
    bool overLabel  = false;
    bool overMarker = false;

    if (! regionInfo_.has_value())
    {
        if (tickrowBounds_.contains (p))
        {
            const int third     = tickrowBounds_.getWidth() / 3;
            const int leftEdge  = tickrowBounds_.getX() + third;
            const int rightEdge = tickrowBounds_.getRight() - third;
            overLabel = (p.getX() >= leftEdge && p.getX() <= rightEdge);
        }
        if (trackHitArea_.contains (p))
        {
            const int origX = xFromTarget (origSec_);
            constexpr int kOrigSnapPx = 8;
            overMarker = (std::abs (p.getX() - origX) <= kOrigSnapPx);
        }
    }

    if (overLabel != hoveringOrigLabel_ || overMarker != hoveringOrigMarker_)
    {
        hoveringOrigLabel_  = overLabel;
        hoveringOrigMarker_ = overMarker;
        setMouseCursor ((overLabel || overMarker)
                          ? juce::MouseCursor::PointingHandCursor
                          : juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void DurationPanel::mouseExit (const juce::MouseEvent&)
{
    if (hoveringOrigLabel_ || hoveringOrigMarker_)
    {
        hoveringOrigLabel_ = hoveringOrigMarker_ = false;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void DurationPanel::mouseDoubleClick (const juce::MouseEvent& e)
{
    // ADR-050 — double-click anywhere on the slider track resets target to
    // the original duration baseline (origSec_). In Duration mode that's the
    // file length; in Region mode setOriginalDuration is region duration, so
    // double-click here equals "no retarget, keep the selected fragment as-is".
    if (! enabled_) return;
    if (! trackHitArea_.contains (e.getPosition())) return;
    if (std::abs (origSec_ - target_) < 1.0e-6) return;
    target_ = juce::jlimit (minSec_, maxSec_, origSec_);
    repaint();
    if (onTargetChanged) onTargetChanged (target_);
}

} // namespace reamix::ui
