#include "WaveformView.h"
#include <cstdint>
#include "LookAndFeelReamix.h"
#include "KindDisplay.h"

#include <algorithm>
#include <cmath>

namespace reamix::ui
{

using namespace reamix::theme;

// ── Construction ────────────────────────────────────────────────────

WaveformView::WaveformView (Variant v)
    : variant_ (v)
{
    setOpaque (true);
    // ⚑ P5 keyboard-navigation (full Tab chain lands step 3+) — the
    // component is reachable; focus-outline painting arrives later.
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::IBeamCursor);

    // Sesja 100 iter 3 (DEV-018) — inline volume slider, hidden by default.
    // Bare LinearVertical, no text box, no track frame beyond JUCE default
    // thumb + fill. Bounds set by positionVolumePopup() when shown.
    volumePopupSlider_.setSliderStyle (juce::Slider::LinearVertical);
    volumePopupSlider_.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    volumePopupSlider_.setRange (0.0, 100.0, 1.0);
    volumePopupSlider_.setValue (100.0, juce::dontSendNotification);
    volumePopupSlider_.onValueChange = [this]
    {
        const double linear01 = volumePopupSlider_.getValue() / 100.0;
        previewVolume_ = juce::jlimit (0.0, 1.0, linear01);
        repaint (volumeBtnBounds_);
        if (onPreviewVolumeChanged) onPreviewVolumeChanged (previewVolume_);
    };
    volumePopupSlider_.setVisible (false);
    addChildComponent (volumePopupSlider_);
}

WaveformView::~WaveformView() = default;

// ── Setters ─────────────────────────────────────────────────────────

void WaveformView::setVariant (Variant v)
{
    if (variant_ == v) return;
    variant_ = v;

    // Variant-specific data is irrelevant the moment the variant flips —
    // splice markers + tile list belong to the Remix view; segments_ +
    // beats_ stay (both are conceptually per-source data, valid for both
    // variants — beats grid, segment tint persist).
    if (variant_ == Variant::Source)
    {
        spliceMarkers_.clear();
        editPlan_.clips.clear();
        hoveredSpliceIdx_ = -1;
    }

    // Drag/selection state is variant-agnostic; cache must rebuild because
    // bottom-band geometry differs.
    cache_.valid = false;
    cursorX_     = -1;
    repaint();
}

void WaveformView::setPeaksProvider (PeaksProvider* p)
{
    peaks_ = p;
    cache_.valid = false;

    if (peaks_ != nullptr)
    {
        viewStartSec_    = 0.0;
        viewDurationSec_ = std::max (peaks_->getTotalDurationSeconds(), 0.001);
    }
    else
    {
        viewStartSec_    = 0.0;
        viewDurationSec_ = 0.0;
    }
    repaint();
}

void WaveformView::setPeaksProviderPreserveView (PeaksProvider* p)
{
    peaks_ = p;
    cache_.valid = false;
    // Sesja 100b iter 2 — clamp the existing view against the new
    // provider's duration in case the user-zoomed range now sits past
    // the end of a (potentially shorter) source. Otherwise leave
    // viewStartSec_ / viewDurationSec_ alone so the user's zoom + pan
    // survives the variant flip.
    if (peaks_ != nullptr)
    {
        const double total = std::max (peaks_->getTotalDurationSeconds(), 0.001);
        if (viewDurationSec_ <= 0.0 || viewDurationSec_ > total)
            viewDurationSec_ = total;
        const double maxStart = std::max (0.0, total - viewDurationSec_);
        viewStartSec_ = juce::jlimit (0.0, maxStart, viewStartSec_);
    }
    repaint();
}

void WaveformView::setSegments (std::vector<Segment> s)
{
    segments_ = std::move (s);
    repaint (canvasArea().getUnion (segBarArea()));
}

void WaveformView::setBeats (std::vector<Beat> b)
{
    beats_ = std::move (b);
    repaint (canvasArea());
}

void WaveformView::setSpliceMarkers (std::vector<SpliceMarker> m)
{
    spliceMarkers_   = std::move (m);
    hoveredSpliceIdx_ = -1; // markers may have moved; clear hover
    repaint();
}

void WaveformView::setEditPlan (reamix::render::EditPlan plan)
{
    editPlan_ = std::move (plan);
    repaint (tileListArea());
}

void WaveformView::setPlayhead (std::optional<double> sec)
{
    // Idempotent for identical state — skip repaint when nothing changes.
    if (sec.has_value() == playheadSec_.has_value()
        && (! sec.has_value() || std::abs (*sec - *playheadSec_) < 1e-6))
        return;
    playheadSec_ = sec;
    repaint (canvasArea());
}

void WaveformView::setSelection (std::optional<SelectionRange> r)
{
    if (selection_.has_value() == r.has_value()
        && (! r.has_value()
            || (std::abs (r->startSec - selection_->startSec) < 1e-6
                && std::abs (r->endSec   - selection_->endSec)   < 1e-6)))
        return;
    selection_ = r;
    repaint (canvasArea());
}

void WaveformView::clearSelection()
{
    setSelection (std::nullopt);
}

void WaveformView::setSnapMode (SnapMode m)
{
    snapMode_ = m;
}

void WaveformView::setShowEditRegionButton (bool yes)
{
    if (yes == showEditRegionButton_) return;
    showEditRegionButton_ = yes;
    if (! yes) editRegionHover_ = false;
    repaint();
}

void WaveformView::setShowEditArrangementButton (bool yes)
{
    if (yes == showEditArrangementButton_) return;
    showEditArrangementButton_ = yes;
    if (! yes) editArrangementHover_ = false;
    repaint();
}

void WaveformView::setPreviewVolume (double linear01)
{
    const double clamped = juce::jlimit (0.0, 1.0, linear01);
    if (std::abs (clamped - previewVolume_) < 1e-6) return;
    previewVolume_ = clamped;
    // Sync inline popup slider so the next show() reflects the current state
    // (don't fire onValueChange — caller already knows the new value).
    volumePopupSlider_.setValue (clamped * 100.0, juce::dontSendNotification);
    if (! volumeBtnBounds_.isEmpty())
        repaint (volumeBtnBounds_);
}

void WaveformView::positionVolumePopup()
{
    // Position the inline slider just ABOVE the volume overlay icon (which
    // sits in canvas bottom-right). Slider expands upward into the waveform
    // area; bare track + thumb so the waveform underneath stays visible.
    constexpr int kPopupW = 28;
    constexpr int kPopupH = 130;
    constexpr int kGap    = 4;

    const int rightAligned =
        volumeBtnBounds_.getRight() - kPopupW;
    const int aboveOverlay =
        volumeBtnBounds_.getY() - kPopupH - kGap;

    volumePopupSlider_.setBounds (rightAligned, aboveOverlay, kPopupW, kPopupH);
}

void WaveformView::showVolumePopup()
{
    positionVolumePopup();
    volumePopupSlider_.setVisible (true);
    volumePopupSlider_.toFront (true);
    volumePopupSlider_.grabKeyboardFocus();
    volumePopupVisible_ = true;
}

void WaveformView::hideVolumePopup()
{
    volumePopupSlider_.setVisible (false);
    volumePopupVisible_ = false;
}

void WaveformView::setBlockMarkingEnabled (bool yes)
{
    if (blockMarkingEnabled_ == yes) return;
    blockMarkingEnabled_ = yes;
    if (! yes)
    {
        segBarDragging_ = false;
        segBarDragStartedOnBlock_ = false;
        segBarDragStartBlockIdx_  = -1;
    }
    // DEV-055 sesja 100c — re-arm hint pulse timer state.
    const bool wantHint = blockMarkingEnabled_ && userBlocks_.empty();
    if (wantHint != segBarHintActive_)
    {
        segBarHintActive_ = wantHint;
        if (wantHint && ! isTimerRunning()) startTimer (40);
    }
    repaint (segBarArea());
}

void WaveformView::setUserBlocks (std::vector<reamix::ui::UserBlock> blocks)
{
    userBlocks_ = std::move (blocks);
    // DEV-055 sesja 100c — re-arm hint pulse on (empty/non-empty) flip.
    const bool wantHint = blockMarkingEnabled_ && userBlocks_.empty();
    if (wantHint != segBarHintActive_)
    {
        segBarHintActive_ = wantHint;
        if (wantHint && ! isTimerRunning()) startTimer (40);
    }
    repaint (segBarArea());
}

void WaveformView::setUserBlockSplices (std::vector<UserBlockSplice> splices)
{
    userBlockSplices_ = std::move (splices);
    if (hoveredSpliceUserIdx_ >= (int) userBlockSplices_.size())
        hoveredSpliceUserIdx_ = -1;
    repaint (segBarArea());
}

void WaveformView::setShowBeats (bool yes)
{
    if (showBeats_ == yes) return;
    showBeats_ = yes;
    repaint (canvasArea());
}

void WaveformView::setAnalyzed (bool yes)
{
    if (hasAnalysis_ == yes) return;
    hasAnalysis_ = yes;
    // Opacity of existing peaks + visibility of overlay label both change;
    // repaint the whole canvas (segment tint decisions also flip).
    repaint (canvasArea());
}

void WaveformView::setHasSource (bool yes)
{
    if (hasSource_ == yes) return;
    hasSource_ = yes;
    repaint (canvasArea());
}

// ── Geometry ────────────────────────────────────────────────────────

juce::Rectangle<int> WaveformView::rulerArea() const
{
    // ADR-045 — ruler now applies to both variants (mockup deviation;
    // mockup waveform-blocks.jsx:91 had it remix-only).
    return getLocalBounds().removeFromTop (kRulerHeight);
}

juce::Rectangle<int> WaveformView::segBarArea() const
{
    // SegBar is source-only per waveform-blocks.jsx:153. Bottom band
    // semantics post-ADR-045: downbeat anchor ticks when no labels;
    // colored cells when Block Assembly populated segments_.
    if (variant_ != Variant::Source) return {};
    return getLocalBounds().removeFromBottom (kSegBarHeight);
}

juce::Rectangle<int> WaveformView::tileListArea() const
{
    // Remix-only bottom 20h band: per-clip duration tiles per ADR-045 (c).
    if (variant_ != Variant::Remix) return {};
    return getLocalBounds().removeFromBottom (kSegBarHeight);
}

juce::Rectangle<int> WaveformView::canvasArea() const
{
    auto r = getLocalBounds();
    r.removeFromTop (kRulerHeight);
    // Bottom 20h band reserved for either segBar (Source) or tileList (Remix).
    r.removeFromBottom (kSegBarHeight);
    return r;
}

float WaveformView::timeToX (double seconds, int regionX, int regionWidth) const
{
    if (viewDurationSec_ <= 0.0 || regionWidth <= 0)
        return static_cast<float> (regionX);
    const double frac = (seconds - viewStartSec_) / viewDurationSec_;
    return static_cast<float> (regionX) + static_cast<float> (frac * regionWidth);
}

double WaveformView::xToTime (float px, int regionX, int regionWidth) const
{
    if (regionWidth <= 0) return viewStartSec_;
    const double frac = (px - regionX) / static_cast<double> (regionWidth);
    return viewStartSec_ + frac * viewDurationSec_;
}

// ── Peak cache ──────────────────────────────────────────────────────

void WaveformView::ensurePeaks()
{
    if (peaks_ == nullptr)
    {
        cache_.valid = false;
        return;
    }

    // Re-sync view duration from the provider. `setPeaksProvider` captured the
    // initial duration at construction, but FilePeaksProvider starts empty and
    // learns its duration only when `setSourcePath` runs later (user picks an
    // item in REAPER). Without this refresh the waveform window stays pinned
    // at the 0-or-microscopic duration from plugin init and no silhouette
    // shows. Revision changes on every setSourcePath so we gate the refresh
    // on revision delta — cheap, avoids churn.
    const auto  revision = peaks_->getRevision();
    if (revision != cache_.revision)
    {
        const double newDur = std::max (peaks_->getTotalDurationSeconds(), 0.001);
        if (std::abs (newDur - viewDurationSec_) > 1.0e-6)
        {
            viewStartSec_    = 0.0;
            viewDurationSec_ = newDur;
        }
    }

    const auto  canvas  = canvasArea();
    const int   width   = canvas.getWidth();
    const int   nBins   = std::max (1, width / kSampleEvery);
    const double endSec = viewStartSec_ + viewDurationSec_;

    if (cache_.valid
        && cache_.widthBins == nBins
        && std::abs (cache_.startSec - viewStartSec_) < 1.0e-6
        && std::abs (cache_.endSec   - endSec) < 1.0e-6
        && cache_.revision == revision)
    {
        return;
    }

    cache_.minBuf.assign (static_cast<std::size_t> (nBins), 0.0f);
    cache_.maxBuf.assign (static_cast<std::size_t> (nBins), 0.0f);

    peaks_->getPeakBlock (viewStartSec_, endSec, nBins,
                          cache_.minBuf.data(), cache_.maxBuf.data());

    cache_.startSec  = viewStartSec_;
    cache_.endSec    = endSec;
    cache_.widthBins = nBins;
    cache_.revision  = revision;
    cache_.valid     = true;
}

// ── Paint ───────────────────────────────────────────────────────────

void WaveformView::paint (juce::Graphics& g)
{
    // plugin.css:287-291 .rx-wave-well { background: var(--rx-bg-2); border-radius: 4px }.
    // MainComponent (parent) paints Bg1 behind us (MainComponent.cpp:756), so
    // filling Bg1 under the rounded Bg2 rect lets the 4 px corner radius show
    // through to match the parent — no visible seam at the corners.
    g.fillAll (Bg1);
    g.setColour (Bg2);
    g.fillRoundedRectangle (getLocalBounds().toFloat(), 4.0f);

    paintCanvas        (g);
    paintSelection     (g);
    if (variant_ == Variant::Source) paintSegBar   (g);
    if (variant_ == Variant::Remix)  paintTileList (g);
    paintRuler         (g);
    if (variant_ == Variant::Remix) paintSpliceMarkers (g);
    paintPlayhead      (g);
    paintFlash         (g);
    paintHover         (g);

    // Sesja 60 Plan A — Edit region overlay button on canvas top-left.
    if (showEditRegionButton_)
    {
        const auto canvas = canvasArea();
        // Compose label "↺ Edit region" using a U+21BA (anticlockwise open
        // circle arrow) followed by the text. JUCE drawing the rune through
        // the regular UI font works on macOS bundled fonts.
        const auto labelFont = uiFont (fs::Sm, 500);
        const juce::String label = juce::String::fromUTF8 ("\xe2\x86\xba  Edit region");
        juce::GlyphArrangement ga;
        ga.addLineOfText (labelFont, label, 0.0f, 0.0f);
        const int textW = (int) std::ceil (ga.getBoundingBox (0, -1, false).getWidth());
        const int padX  = 10;
        const int btnW  = textW + padX * 2;
        const int btnH  = 24;
        const int margin = 8;

        editRegionBtnBounds_ = juce::Rectangle<int> (
            canvas.getX() + margin,
            canvas.getY() + margin,
            btnW, btnH);

        const auto rect = editRegionBtnBounds_.toFloat();
        // Sesja 99 — DEV-053 (sesja 62 NOTE-10): backdrop alpha 0.92 →
        // 0.65 so the waveform under the overlay button stays visible.
        // Mirrored on Edit-arrangement overlay below.
        g.setColour (Bg1.withAlpha (0.55f));
        g.fillRoundedRectangle (rect, 3.0f);
        g.setColour (editRegionHover_ ? Accent : AccentLo);
        g.drawRoundedRectangle (rect.reduced (0.5f), 3.0f, 1.0f);

        g.setColour (editRegionHover_ ? AccentHi : Accent);
        g.setFont (labelFont);
        g.drawText (label, editRegionBtnBounds_,
                    juce::Justification::centred, false);
    }

    // Sesja 61 hot-fix — Edit arrangement overlay button (Blocks mode parallel
    // to Region mode's Edit region overlay; mutually exclusive in practice).
    if (showEditArrangementButton_)
    {
        const auto canvas = canvasArea();
        const auto labelFont = uiFont (fs::Sm, 500);
        const juce::String label = juce::String::fromUTF8 ("\xe2\x86\xba  Edit arrangement");
        juce::GlyphArrangement ga;
        ga.addLineOfText (labelFont, label, 0.0f, 0.0f);
        const int textW = (int) std::ceil (ga.getBoundingBox (0, -1, false).getWidth());
        const int padX  = 10;
        const int btnW  = textW + padX * 2;
        const int btnH  = 24;
        const int margin = 8;

        editArrangementBtnBounds_ = juce::Rectangle<int> (
            canvas.getX() + margin,
            canvas.getY() + margin,
            btnW, btnH);

        const auto rect = editArrangementBtnBounds_.toFloat();
        // Sesja 99 — DEV-053: matches Edit-region overlay backdrop alpha.
        g.setColour (Bg1.withAlpha (0.55f));
        g.fillRoundedRectangle (rect, 3.0f);
        g.setColour (editArrangementHover_ ? Accent : AccentLo);
        g.drawRoundedRectangle (rect.reduced (0.5f), 3.0f, 1.0f);

        g.setColour (editArrangementHover_ ? AccentHi : Accent);
        g.setFont (labelFont);
        g.drawText (label, editArrangementBtnBounds_,
                    juce::Justification::centred, false);
    }

    // Sesja 100 (DEV-018, ADR-091) — Preview volume overlay on canvas
    // bottom-right (sesja 100 user smoke iter 2 — top-right zasłaniał ruler
    // and the expanded slider popup overlay collided with adaptive ruler).
    // Always rendered (no toggle); user directive: dyskretny ale funkcjonalny.
    // Same Bg1.alpha 0.85 + Sm/500 font as Edit overlays.
    {
        const auto canvas = canvasArea();
        const auto labelFont = uiFont (fs::Sm, 500);

        // Path-drawn speaker icon + percentage text. Speaker stays
        // monochrome via Accent stroke; percentage carries the live value
        // so user reads volume at a glance.
        const int pct = (int) std::round (previewVolume_ * 100.0);
        const juce::String pctLabel = juce::String (pct) + "%";

        juce::GlyphArrangement gaPct;
        gaPct.addLineOfText (labelFont, pctLabel, 0.0f, 0.0f);
        const int pctW = (int) std::ceil (gaPct.getBoundingBox (0, -1, false).getWidth());

        constexpr int speakerW = 14;   // path-drawn speaker glyph width
        constexpr int gap      = 6;    // between speaker and percentage
        const int padX  = 10;
        const int btnW  = speakerW + gap + pctW + padX * 2;
        const int btnH  = 24;
        const int margin = 8;

        volumeBtnBounds_ = juce::Rectangle<int> (
            canvas.getRight() - margin - btnW,
            canvas.getBottom() - margin - btnH,
            btnW, btnH);

        const auto rect = volumeBtnBounds_.toFloat();
        g.setColour (Bg1.withAlpha (0.55f));
        g.fillRoundedRectangle (rect, 3.0f);
        g.setColour (volumeHover_ ? Accent : AccentLo);
        g.drawRoundedRectangle (rect.reduced (0.5f), 3.0f, 1.0f);

        const auto iconColour = volumeHover_ ? AccentHi : Accent;
        g.setColour (iconColour);

        // Path-drawn speaker icon — small trapezoidal cone with two
        // sound waves when volume > 0, just cone when muted. Drawn
        // inside an 14×12 box centered vertically in the button.
        const float iconX = rect.getX() + (float) padX;
        const float iconCY = rect.getCentreY();
        const float iconH = 12.0f;
        const float coneX0 = iconX;
        const float coneY  = iconCY - iconH * 0.5f;

        juce::Path speaker;
        // Cone body: small square 4×6 + trapezoid 4×12 to the right
        speaker.startNewSubPath (coneX0,        iconCY - 3.0f);
        speaker.lineTo         (coneX0 + 3.5f,  iconCY - 3.0f);
        speaker.lineTo         (coneX0 + 7.0f,  coneY);
        speaker.lineTo         (coneX0 + 7.0f,  coneY + iconH);
        speaker.lineTo         (coneX0 + 3.5f,  iconCY + 3.0f);
        speaker.lineTo         (coneX0,         iconCY + 3.0f);
        speaker.closeSubPath();
        g.fillPath (speaker);

        if (previewVolume_ > 0.0)
        {
            // One arc when volume <= 50%, two arcs when > 50%
            const float arcX = coneX0 + 9.0f;
            juce::Path waves;
            waves.startNewSubPath (arcX, iconCY - 4.0f);
            waves.quadraticTo (arcX + 2.5f, iconCY, arcX, iconCY + 4.0f);
            if (previewVolume_ > 0.5)
            {
                waves.startNewSubPath (arcX + 3.0f, iconCY - 6.0f);
                waves.quadraticTo (arcX + 6.5f, iconCY, arcX + 3.0f, iconCY + 6.0f);
            }
            g.strokePath (waves, juce::PathStrokeType (1.5f));
        }

        // Percentage text aligned to the right of the icon
        g.setFont (labelFont);
        const juce::Rectangle<int> pctBounds (
            (int) (rect.getX() + padX + speakerW + gap),
            (int) rect.getY(),
            pctW + 2,
            (int) rect.getHeight());
        g.drawText (pctLabel, pctBounds,
                    juce::Justification::centredLeft, false);
    }
}

void WaveformView::paintRuler (juce::Graphics& g)
{
    const auto r = rulerArea();
    if (r.isEmpty()) return;

    // plugin.css:293-297 — bg-1 + bottom 1px line.
    g.setColour (Bg1);
    g.fillRect (r);
    g.setColour (Line);
    g.fillRect (r.withHeight (1).withY (r.getBottom() - 1));

    if (viewDurationSec_ <= 0.0) return;

    const double pxPerSec = r.getWidth() / viewDurationSec_;
    const double interval = pickTickInterval (pxPerSec);
    const double firstTick = std::floor (viewStartSec_ / interval) * interval;
    const double endSec    = viewStartSec_ + viewDurationSec_;

    // Mono 9, Fg3 — plugin.css:303-304.
    const auto tickFont = monoFont (fs::Xs, 400);

    for (double t = firstTick; t <= endSec + 0.5 * interval; t += interval)
    {
        if (t < viewStartSec_ - 0.5 * interval) continue;
        const float x = timeToX (t, r.getX(), r.getWidth());
        if (x < r.getX() - 1 || x > r.getRight() + 1) continue;

        // plugin.css:301 — 1px LineStrong tick.
        g.setColour (LineStrong);
        g.drawVerticalLine (juce::roundToInt (x),
                            static_cast<float> (r.getY()),
                            static_cast<float> (r.getBottom()));

        // plugin.css:302-305 — 3px left padding, mono 9 fg-3, line-height 18.
        g.setColour (Fg3);
        g.setFont (tickFont);
        const int labelX = juce::roundToInt (x) + 3;
        const int labelWidth = std::min (r.getRight() - labelX, 50);
        if (labelWidth > 6)
        {
            g.drawText (formatSeconds (t),
                        labelX, r.getY(), labelWidth, r.getHeight(),
                        juce::Justification::centredLeft, false);
        }
    }
}

void WaveformView::paintCanvas (juce::Graphics& g)
{
    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    // DIAG-WAVE-49 REMOVED post-verification (session 49). The probe proved
    // the splash code path was correct; the actual bug was a stale legacy
    // `reaper_reamix.dylib` filename (without -arm64 suffix) left over in
    // UserPlugins from an earlier build config, which REAPER preferred over
    // the -arm64 file at load time. Fix was to overwrite the legacy file
    // with identical new-build content. See DEV-023 in DEVIATIONS.md +
    // session-49 log entry for prevention (install script / CMake rename).

    // DEV-022 / ADR-041 (session 49) — three-state canvas:
    //   hasSource_=false              → brand splash, empty-state headline
    //   hasSource_=true, !hasAnalysis_ → brand splash, CTA headline
    //   hasSource_=true,  hasAnalysis_ → full waveform (peaks/beats/segments)
    // User feedback session 49: pre-analyze faint silhouette was useless
    // because Preview / Insert are still gated — replacing with an explicit
    // CTA splash removes the misleading "almost usable" visual state.
    if (! hasSource_ || ! hasAnalysis_)
    {
        paintSplashOverlay (g, canvas, /*isEmpty=*/! hasSource_);
        return;
    }

    // plugin.css:307-311 — .rx-wave-canvas { background: var(--rx-bg-2) }.
    // Component-level rounded Bg2 fill in paint() already covers this region;
    // no per-canvas fill needed (was redundant).

    // Beat grid (under the peaks per mockup stacking) — plugin.css:313-314.
    // Alpha adapts to zoom: at low zoom (many beats in view, ≤5 px per beat)
    // alpha stays at mockup default so beats don't crowd the silhouette; at
    // high zoom (≥30 px per beat) alpha scales up ~3× so sparse beats stay
    // readable. User session-48 feedback: "im dalej tym powinny być mniej
    // widoczne, a jak się zoomuję to powinny być lepiej widoczne".
    // Post-analysis state guaranteed by the early return above; no gate
    // needed on hasAnalysis_ here.
    if (showBeats_ && viewDurationSec_ > 0.0 && ! beats_.empty())
    {
        const double viewEnd = viewStartSec_ + viewDurationSec_;

        int visibleBeats = 0;
        for (const auto& b : beats_)
            if (b.time >= viewStartSec_ - 0.01 && b.time <= viewEnd + 0.01)
                ++visibleBeats;
        const double pxPerBeat = visibleBeats > 0
                                  ? (double) canvas.getWidth() / (double) visibleBeats
                                  : (double) canvas.getWidth();
        const float t = (float) juce::jlimit (0.0, 1.0, (pxPerBeat - 5.0) / 25.0);

        constexpr float kPlainMin     = 0.04f;
        constexpr float kPlainMax     = 0.22f;
        constexpr float kDownbeatMin  = 0.11f;
        constexpr float kDownbeatMax  = 0.50f;
        const float plainA    = kPlainMin    + t * (kPlainMax    - kPlainMin);
        const float downbeatA = kDownbeatMin + t * (kDownbeatMax - kDownbeatMin);
        const juce::Colour plain    { juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, plainA) };
        const juce::Colour downbeat { juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, downbeatA) };
        for (const auto& b : beats_)
        {
            if (b.time < viewStartSec_ - 0.01 || b.time > viewEnd + 0.01) continue;
            const float x = timeToX (b.time, canvas.getX(), canvas.getWidth());
            g.setColour (b.isDownbeat ? downbeat : plain);
            g.drawVerticalLine (juce::roundToInt (x),
                                static_cast<float> (canvas.getY()),
                                static_cast<float> (canvas.getBottom()));
        }
    }

    // Mirrored silhouette — primitives.jsx:131-171. Per-bar segment tint at
    // 0.78 alpha; bars stepped by kSampleEvery (2 px), 1.5 px rendered width.
    ensurePeaks();
    if (cache_.valid && ! cache_.minBuf.empty())
    {
        const float centerY  = canvas.getY() + canvas.getHeight() * 0.5f;
        const float maxBarH  = std::max (6.0f, canvas.getHeight() - 6.0f);
        const int   nBins    = cache_.widthBins;
        const float barWidth = static_cast<float> (kSampleEvery) - 0.5f;
        const auto  defaultColour = Fg2;

        for (int bin = 0; bin < nBins; ++bin)
        {
            const float pk = std::max (std::abs (cache_.minBuf[static_cast<std::size_t> (bin)]),
                                       std::abs (cache_.maxBuf[static_cast<std::size_t> (bin)]));
            const float h = std::max (1.2f, std::min (pk, 1.0f) * maxBarH);

            const int xPx = canvas.getX() + bin * kSampleEvery;
            if (xPx >= canvas.getRight()) break;

            juce::Colour c = defaultColour;
            // ADR-045 — segment tint is data-driven; segments_ is non-empty
            // only in Block Assembly mode (phase-6 step 8). Auto modes
            // (post-ADR-044) supply zero segments → no tint, no toggle.
            if (! segments_.empty())
            {
                const double tSrc = viewStartSec_
                                    + (bin + 0.5) * (viewDurationSec_ / std::max (nBins, 1));
                for (const auto& s : segments_)
                {
                    if (tSrc >= s.start && tSrc < s.end)
                    {
                        c = segmentColour (s.kind);
                        break;
                    }
                }
            }

            g.setColour (c.withAlpha (kBarOpacity));
            g.fillRect (static_cast<float> (xPx),
                        centerY - h * 0.5f,
                        barWidth,
                        h);
        }
    }

    // Centerline — primitives.jsx:173 stroke rgba(0,0,0,0.35) 0.5px.
    g.setColour (juce::Colour::fromFloatRGBA (0.0f, 0.0f, 0.0f, 0.35f));
    g.fillRect (juce::Rectangle<float> (static_cast<float> (canvas.getX()),
                                        canvas.getY() + canvas.getHeight() * 0.5f - 0.25f,
                                        static_cast<float> (canvas.getWidth()),
                                        0.5f));
}

void WaveformView::paintSplashOverlay (juce::Graphics& g,
                                       juce::Rectangle<int> canvas,
                                       bool isEmpty)
{
    // DEV-022 / ADR-041 — unified brand splash. Session 49 logo-swap: the
    // big marker dot is replaced by the bundled sun-burst PNG (provided by
    // user this session). Two-row layout: logo + "reamix" / ".me" mark in
    // a row, headline underneath. Headline copy swaps per state so the
    // visual template is shared between empty and pre-analyze.
    using namespace reamix::theme;

    const int logoSize = 40;      // larger-than-header hero for the splash
    const int markGap  = 14;      // scaled gap — matches 20pt Font vertical rhythm
    const auto reamixFont   = uiFont (28.0f, 600);
    const auto tldFont      = uiFont (28.0f, 500);
    const auto headlineFont = uiFont (14.0f, 400);

    const juce::String nameText { "reamix" };
    const juce::String tldText  { ".me" };
    const juce::String headlineText = isEmpty
        ? juce::String ("Retarget any track to any duration.")
        : juce::String ("Click Analyze to prepare the remix.");

    const auto widthOf = [] (const juce::Font& f, const juce::String& t)
    {
        juce::GlyphArrangement ga;
        ga.addLineOfText (f, t, 0.0f, 0.0f);
        return ga.getBoundingBox (0, -1, false).getWidth();
    };

    const float nameW     = widthOf (reamixFont, nameText);
    const float tldW      = widthOf (tldFont,    tldText);
    const float headlineW = widthOf (headlineFont, headlineText);

    const float markW  = (float) logoSize + (float) markGap + nameW + tldW;
    const float markH  = std::max ((float) logoSize, (float) reamixFont.getHeight());
    const float headH  = (float) headlineFont.getHeight();
    const float rowGap = 14.0f;
    const float groupH = markH + rowGap + headH;

    const float groupTop = canvas.getCentreY() - groupH * 0.5f;
    const float markX    = canvas.getCentreX() - markW * 0.5f;

    // Brand logo — bundled PNG via LookAndFeelReamix::brandLogo(). Falls
    // back to the original Accent rounded-square if decode fails.
    const float logoY = groupTop + (markH - (float) logoSize) * 0.5f;
    const juce::Rectangle<int> logoRect ((int) markX, (int) logoY,
                                          logoSize, logoSize);
    const auto& logo = LookAndFeelReamix::brandLogo();
    if (logo.isValid())
    {
        // Sesja 111 v1.0.3 — match HeaderBar logo rendering: bicubic
        // interpolation for clean sun-burst rays under aggressive
        // downscale (source 128×124 → target logoSize). See HeaderBar
        // cpp:153 for the Windows VM jagged-edges precedent.
        g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);
        g.drawImageWithin (logo,
                           logoRect.getX(), logoRect.getY(),
                           logoRect.getWidth(), logoRect.getHeight(),
                           juce::RectanglePlacement::centred
                           | juce::RectanglePlacement::onlyReduceInSize,
                           false);
    }
    else
    {
        g.setColour (Accent);
        g.fillRoundedRectangle (logoRect.toFloat(), r::R2);
        g.setColour (juce::Colour (0x40000000));
        g.drawRoundedRectangle (logoRect.toFloat().reduced (0.5f), r::R2, 1.0f);
    }

    // Row: "reamix" + ".me" vertically centered against the logo height.
    const float textRowTop = groupTop + (markH - (float) reamixFont.getHeight()) * 0.5f;

    // "reamix" Fg0 600
    const float nameX = markX + (float) logoSize + (float) markGap;
    g.setColour (Fg0);
    g.setFont (reamixFont);
    g.drawText (nameText,
                juce::Rectangle<float> (nameX, textRowTop,
                                        nameW + 4.0f,
                                        (float) reamixFont.getHeight()),
                juce::Justification::centredLeft, false);

    // ".me" Fg3 500
    g.setColour (Fg3);
    g.setFont (tldFont);
    g.drawText (tldText,
                juce::Rectangle<float> (nameX + nameW, textRowTop,
                                        tldW + 4.0f,
                                        (float) tldFont.getHeight()),
                juce::Justification::centredLeft, false);

    // Headline — Fg2, centered under the mark.
    const float headX = canvas.getCentreX() - headlineW * 0.5f;
    const float headY = groupTop + markH + rowGap;
    g.setColour (Fg2);
    g.setFont (headlineFont);
    g.drawText (headlineText,
                juce::Rectangle<float> (headX, headY, headlineW + 4.0f, headH),
                juce::Justification::centredLeft, false);
}

void WaveformView::paintSegBar (juce::Graphics& g)
{
    const auto bar = segBarArea();
    if (bar.isEmpty()) return;

    // plugin.css:343-348 — bg-1 background + top 1px line.
    g.setColour (Bg1);
    g.fillRect (bar);
    g.setColour (Line);
    g.fillRect (bar.withHeight (1));

    if (viewDurationSec_ <= 0.0) return;

    // ADR-045 (a) A1+ — when no other content, the band shows subtle
    // downbeat anchor ticks. ADR-051 stacks user-block tiles + provisional
    // drag preview ON TOP of the anchors when in Blocks mode; both paint
    // even alongside the anchor base layer.
    //
    // DEV-055 sesja 100c — Blocks-mode coach mark renders even when
    // segments_ AND userBlocks_ are both empty (was being hidden by the
    // early return below). Empty-state IS the most-needs-hint scenario.
    if (segments_.empty() && userBlocks_.empty() && ! segBarDragging_)
    {
        paintDownbeatAnchors (g, bar);
        if (blockMarkingEnabled_)
        {
            const double tMs   = (double) juce::Time::getMillisecondCounter();
            const double phase = std::fmod (tMs, 1200.0) / 1200.0;
            const float  pulse = 0.5f + 0.5f * (float) std::sin (phase * juce::MathConstants<double>::twoPi);
            const float  alpha = 0.45f + 0.40f * pulse;
            g.setColour (Accent.withAlpha (alpha));
            g.setFont (monoFont (fs::Sm, 700).withExtraKerningFactor (0.10f));
            g.drawText (juce::String::fromUTF8 (
                "\xE2\x86\x94  DRAG HERE TO MARK YOUR FIRST BLOCK"),
                bar, juce::Justification::centred, false);
        }
        return;
    }
    // Anchor base layer remains visible when user blocks paint (subtle
    // downbeat hints between/around tiles).
    paintDownbeatAnchors (g, bar);

    // plugin.css:354 — Xs 9px, 0.06em letter-spacing, 600, uppercase.
    const float kSegLabelKerning = 0.06f; // em units — matches CSS 0.06em.
    const juce::Colour labelColour { juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.85f) };
    const juce::Colour cellDivider { juce::Colour::fromFloatRGBA (0.0f, 0.0f, 0.0f, 0.35f) };
    const auto cellFont = uiFont (fs::Xs, 600).withExtraKerningFactor (kSegLabelKerning);

    const double viewEnd = viewStartSec_ + viewDurationSec_;

    for (std::size_t i = 0; i < segments_.size() && ! segments_.empty(); ++i)
    {
        const auto& s = segments_[i];
        if (s.end <= viewStartSec_ || s.start >= viewEnd) continue;

        const double clippedStart = std::max (s.start, viewStartSec_);
        const double clippedEnd   = std::min (s.end,   viewEnd);

        const float xStart = timeToX (clippedStart, bar.getX(), bar.getWidth());
        const float xEnd   = timeToX (clippedEnd,   bar.getX(), bar.getWidth());
        const int   cellX  = juce::roundToInt (xStart);
        const int   cellW  = juce::roundToInt (xEnd) - cellX;
        if (cellW <= 0) continue;

        g.setColour (segmentColour (s.kind));
        g.fillRect (cellX, bar.getY(), cellW, bar.getHeight());

        // plugin.css:358 — right border rgba(0,0,0,0.35); skip last cell.
        if (i + 1 < segments_.size() && cellX + cellW < bar.getRight())
        {
            g.setColour (cellDivider);
            g.fillRect (cellX + cellW - 1, bar.getY(), 1, bar.getHeight());
        }

        // plugin.css:359 — label only when cell is wide enough to read.
        if (cellW > 40)
        {
            // Normalize "pre-chorus" → "PRE CHORUS" like primitives.jsx:184.
            juce::String label;
            switch (s.kind)
            {
                case SegmentKind::Intro:        label = "INTRO";        break;
                case SegmentKind::Verse:        label = "VERSE";        break;
                case SegmentKind::PreChorus:    label = "PRE CHORUS";   break;
                case SegmentKind::Chorus:       label = "CHORUS";       break;
                case SegmentKind::PostChorus:   label = "POST CHORUS";  break;
                case SegmentKind::Bridge:       label = "BRIDGE";       break;
                case SegmentKind::Buildup:      label = "BUILDUP";      break;
                case SegmentKind::Drop:         label = "DROP";         break;
                case SegmentKind::Breakdown:    label = "BREAKDOWN";    break;
                case SegmentKind::Solo:         label = "SOLO";         break;
                case SegmentKind::Instrumental: label = "INSTRUMENTAL"; break;
                case SegmentKind::Outro:        label = "OUTRO";        break;
                case SegmentKind::NumKinds:     break;
            }
            g.setColour (labelColour);
            g.setFont (cellFont);
            // plugin.css:353 — 0 5px horizontal padding inside the cell.
            g.drawText (label,
                        cellX + 5, bar.getY(),
                        std::max (0, cellW - 10), bar.getHeight(),
                        juce::Justification::centredLeft, false);
        }
    }

    // ADR-051 — Block Assembly user-marked tiles paint over anchors / segments_.
    // Provisional drag preview paints last so it sits ON TOP of the placed tiles.
    // Splice lines paint OVER tiles (algorithm's resolved cuts visible above
    // user-authored intent) so user can see drift at a glance.
    if (! userBlocks_.empty())          paintUserBlocks       (g, bar);
    if (! userBlockSplices_.empty())    paintUserBlockSplices (g, bar);
    if (segBarDragging_)                paintMarkPreview      (g, bar);

    // DEV-029 (d) sesja 100b — boundary-drag preview line. Vertical
    // accent stripe across the segBar at the live edge position so the
    // user sees the boundary moving in real time. Block content beneath
    // is the original (pre-drag) tile; commit happens on mouseUp.
    if (segBarBoundaryDragging_)
    {
        const int x = (int) timeToX (segBarBoundaryLiveSec_,
                                       bar.getX(), bar.getWidth());
        g.setColour (Accent);
        g.fillRect (x - 1, bar.getY(), 2, bar.getHeight());
    }

    // ADR-051 phase I — coach mark when in Blocks mode + no user blocks yet.
    // Premium empty-state UX: ghost text guiding first action. Auto-hides
    // once the user marks anything (userBlocks_ non-empty branch above).
    //
    // DEV-055 sesja 100c — alpha pulse so the hint catches a first-time user's
    // eye. Period 1.2 s; alpha 0.45–0.85 envelope. Timer-driven repaint via
    // segBarHintActive_ flag (re-armed in setBlockMarkingEnabled / setUserBlocks).
    if (blockMarkingEnabled_ && userBlocks_.empty() && ! segBarDragging_)
    {
        const double tMs    = (double) juce::Time::getMillisecondCounter();
        const double phase  = std::fmod (tMs, 1200.0) / 1200.0;     // 0..1
        const float  pulse  = 0.5f + 0.5f * (float) std::sin (phase * juce::MathConstants<double>::twoPi);
        const float  alpha  = 0.45f + 0.40f * pulse;                 // 0.45..0.85

        g.setColour (Accent.withAlpha (alpha));
        g.setFont (monoFont (fs::Sm, 700).withExtraKerningFactor (0.10f));
        g.drawText (juce::String::fromUTF8 (
            "\xE2\x86\x94  DRAG HERE TO MARK YOUR FIRST BLOCK"),
            bar, juce::Justification::centred, false);
    }
}

void WaveformView::paintUserBlocks (juce::Graphics& g, juce::Rectangle<int> bar)
{
    // ADR-051 — kind-colored solid tile per UserBlock with kind label inside
    // (when wide enough). Visual continuity with the legacy segments_ cell
    // rendering (same colors via segmentColour, same plugin.css:354 9 px Xs
    // 0.06em uppercase label, same divider). Differs from segments_:
    //   - thin Accent border at top so user perceives "I made these" vs.
    //     "the algorithm made these".
    //   - no inter-tile right border (gaps are valid; tiles can be sparse).
    using namespace reamix::theme;
    const float kSegLabelKerning = 0.06f;
    const auto  cellFont       = uiFont (fs::Xs, 600).withExtraKerningFactor (kSegLabelKerning);
    const juce::Colour labelColour { juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.9f) };

    const double viewEnd = viewStartSec_ + viewDurationSec_;
    for (std::size_t i = 0; i < userBlocks_.size(); ++i)
    {
        const auto& b = userBlocks_[i];

        // DEV-057 sesja 100d — mid-drag block renders at translated
        // position. Other blocks render normally so the user perceives
        // the dragged tile gliding past stationary neighbours.
        const bool isMidDragged =
               segBarMidDragActive_
            && (int) i == segBarMidDragBlockIdx_;
        const double effStart = isMidDragged
            ? segBarMidDragLiveStart_ : b.startSec;
        const double effEnd   = isMidDragged
            ? (segBarMidDragLiveStart_ + segBarMidDragOriginalDur_)
            : b.endSec;

        if (effEnd <= viewStartSec_ || effStart >= viewEnd) continue;

        const double clippedStart = std::max (effStart, viewStartSec_);
        const double clippedEnd   = std::min (effEnd,   viewEnd);

        const float xStart = timeToX (clippedStart, bar.getX(), bar.getWidth());
        const float xEnd   = timeToX (clippedEnd,   bar.getX(), bar.getWidth());
        const int   cellX  = juce::roundToInt (xStart);
        const int   cellW  = juce::roundToInt (xEnd) - cellX;
        if (cellW <= 0) continue;

        // ADR-092 sesja 100c — kindDisplay() consults customKindRegistry_
        // when block.customKindId is set; falls back to built-in
        // segmentColour / labelFor on registry-miss or no customKindId.
        // Local fallback when registry pointer is null (defensive — no
        // crash even before MainComponent has wired the registry).
        const KindDisplay disp = (customKindRegistry_ != nullptr)
            ? kindDisplay (b, *customKindRegistry_)
            : KindDisplay { builtinKindLabel (b.kind), segmentColour (b.kind) };
        const auto kindColour = disp.color;

        // Sesja 100b iter 2 — hover highlight. DEV-057 — mid-dragged tile
        // gets a stronger lift (brighter + accent outline) to read as
        // "you're holding this".
        const bool isHovered =
               (int) i == hoveredUserBlockIdx_
            || (int) i == segBarBoundarySharedBlockIdx_
            || (segBarBoundaryDragging_ && (int) i == segBarBoundaryBlockIdx_);
        g.setColour (isMidDragged ? kindColour.brighter (0.30f)
                                   : (isHovered ? kindColour.brighter (0.18f) : kindColour));
        g.fillRect (cellX, bar.getY(), cellW, bar.getHeight());

        // Top accent stripe — 1 px brighter version of the kind color so the
        // tile reads as user-authored. Maps tile to "your authored block".
        g.setColour (kindColour.brighter (0.4f));
        g.fillRect (cellX, bar.getY(), cellW, 1);

        // DEV-057 — Accent border around mid-dragged tile (signals "lifted").
        if (isMidDragged)
        {
            g.setColour (Accent);
            g.drawRect (cellX, bar.getY(), cellW, bar.getHeight(), 1);
        }

        if (cellW > 40)
        {
            const juce::String label = disp.name.toUpperCase();
            g.setColour (labelColour);
            g.setFont (cellFont);
            g.drawText (label,
                        cellX + 5, bar.getY(),
                        std::max (0, cellW - 10), bar.getHeight(),
                        juce::Justification::centredLeft, false);
        }
    }

    // DEV-056 sesja 100c — visible divider between touching tiles. After all
    // tiles are painted, draw a thin Bg0 line at every shared boundary
    // (b[i+1].startSec ≈ b[i].endSec within 5 ms tolerance) so the user
    // perceives two adjacent blocks as separate even when their colours
    // are similar. Bg0 reads as "panel background" → "rozdzielenie", not
    // accent (which would read as a marker).
    if (userBlocks_.size() >= 2)
    {
        const double viewEnd2 = viewStartSec_ + viewDurationSec_;
        for (std::size_t i = 0; i + 1 < userBlocks_.size(); ++i)
        {
            const auto& a = userBlocks_[i];
            const auto& bn = userBlocks_[i + 1];
            if (std::abs (bn.startSec - a.endSec) > 0.005) continue;
            if (a.endSec <= viewStartSec_ || a.endSec >= viewEnd2) continue;

            const float xJoin = timeToX (a.endSec, bar.getX(), bar.getWidth());
            const int xJoinI  = juce::roundToInt (xJoin);
            g.setColour (Bg0);
            g.fillRect (xJoinI - 1, bar.getY() + 1, 2, bar.getHeight() - 2);
        }
    }
}

void WaveformView::paintUserBlockSplices (juce::Graphics& g, juce::Rectangle<int> bar)
{
    // ADR-051 § D3 — algorithm-resolved splice points painted as 2 px Accent
    // vertical lines spanning the segBar. Hovered splice brightens by ×1.25
    // (matches splice marker hover convention from session 57).
    using namespace reamix::theme;
    const double viewEnd = viewStartSec_ + viewDurationSec_;
    const auto baseColour = Accent;

    for (std::size_t i = 0; i < userBlockSplices_.size(); ++i)
    {
        const auto& sp = userBlockSplices_[i];
        if (sp.sourceTimeSec < viewStartSec_ - 1e-3 || sp.sourceTimeSec > viewEnd + 1e-3) continue;
        const float x = timeToX (sp.sourceTimeSec, bar.getX(), bar.getWidth());
        const bool hovered = ((int) i == hoveredSpliceUserIdx_);
        const auto col = hovered ? baseColour.brighter (0.25f) : baseColour;
        g.setColour (col);
        g.fillRect (juce::roundToInt (x) - 1, bar.getY(),
                    2, bar.getHeight());
    }
}

void WaveformView::paintMarkPreview (juce::Graphics& g, juce::Rectangle<int> bar)
{
    // ADR-051 — provisional mark while user is dragging on segBar in Blocks
    // mode. Renders smart-default kind color at 60 % opacity (signals
    // "tentative") with dashed Accent edges and a snap pulse at endpoints
    // when snapMode_ != Off and the cursor is near a beat (~< 4 px).
    // DEV-057 sesja 100d — suppressed during mid-tile drag (different
    // visual language: the dragged tile slides at full opacity in
    // paintUserBlocks; a provisional scrim would compete visually).
    if (! segBarDragging_ || segBarMidDragActive_) return;
    // Suppressed when drag started ON an existing block — that gesture
    // becomes either a click (no provisional scrim needed) or escalates
    // to mid-drag (handled above). Provisional scrim is for empty-segBar
    // marking only.
    if (segBarDragStartedOnBlock_) return;

    double a = std::min (segBarDragStartSec_, segBarDragCurrentSec_);
    double b = std::max (segBarDragStartSec_, segBarDragCurrentSec_);
    if (snapMode_ != SnapMode::Off)
    {
        const double sa = snapTimeToBeats (a);
        const double sb = snapTimeToBeats (b);
        if (sa < sb) { a = sa; b = sb; }
    }

    const float xStart = timeToX (a, bar.getX(), bar.getWidth());
    const float xEnd   = timeToX (b, bar.getX(), bar.getWidth());
    const int   x0 = juce::roundToInt (xStart);
    const int   x1 = juce::roundToInt (xEnd);
    const int   w  = std::max (1, x1 - x0);

    // Smart-default kind for live color preview (Intro near 0, Outro near end,
    // else Verse). Item duration unknown here; we approximate using
    // viewStartSec_ + viewDurationSec_ as a proxy when the view spans the
    // whole item (the common case during marking). Edge cases will land at
    // the popover anyway, where MainComponent has the precise duration.
    const double itemDurEstimate = viewStartSec_ + viewDurationSec_;
    const auto provisionalKind = reamix::ui::smartKindForPosition (a, itemDurEstimate);
    const auto fill = reamix::theme::segmentColour (provisionalKind).withAlpha (0.55f);
    g.setColour (fill);
    g.fillRect (x0, bar.getY(), w, bar.getHeight());

    // Dashed Accent edge on each side — signals "tentative".
    using reamix::theme::Accent;
    g.setColour (Accent);
    const float dashLen[] { 3.0f, 2.0f };
    juce::Path edge;
    edge.startNewSubPath ((float) x0 + 0.5f, (float) bar.getY());
    edge.lineTo          ((float) x0 + 0.5f, (float) bar.getBottom());
    juce::PathStrokeType (1.0f).createDashedStroke (edge, edge, dashLen, 2);
    g.strokePath (edge, juce::PathStrokeType (1.0f));

    juce::Path edge2;
    edge2.startNewSubPath ((float) x1 + 0.5f, (float) bar.getY());
    edge2.lineTo          ((float) x1 + 0.5f, (float) bar.getBottom());
    juce::PathStrokeType (1.0f).createDashedStroke (edge2, edge2, dashLen, 2);
    g.strokePath (edge2, juce::PathStrokeType (1.0f));
}

void WaveformView::paintDownbeatAnchors (juce::Graphics& g, juce::Rectangle<int> bar)
{
    // ADR-045 (a) A1+ — 1 px Fg3 25 % alpha vertical ticks at downbeat
    // x-positions inside the Source bottom 20h band. Reserved spatial
    // affordance for future "snap to downbeat" interaction; deliberately
    // very subtle (barely visible) per design.
    if (beats_.empty() || viewDurationSec_ <= 0.0) return;

    const double viewEnd = viewStartSec_ + viewDurationSec_;
    g.setColour (Fg3.withAlpha (0.25f));
    for (const auto& b : beats_)
    {
        if (! b.isDownbeat) continue;
        if (b.time < viewStartSec_ - 0.01 || b.time > viewEnd + 0.01) continue;
        const float x = timeToX (b.time, bar.getX(), bar.getWidth());
        g.drawVerticalLine (juce::roundToInt (x),
                            static_cast<float> (bar.getY()),
                            static_cast<float> (bar.getBottom()));
    }
}

void WaveformView::paintTileList (juce::Graphics& g)
{
    // ADR-045 § Decision (c) — Remix variant 20h bottom band: per-clip
    // duration tiles, width ∝ clip.durationSec. Alternating subtle gray
    // fill matches ADR-046 Insert color scheme (#3a3a3a / #4a4a4a) so the
    // visual continuity between plugin preview and REAPER timeline is
    // immediately readable. Empty clips → just background + top line.
    const auto bar = tileListArea();
    if (bar.isEmpty()) return;

    g.setColour (Bg1);
    g.fillRect (bar);
    g.setColour (Line);
    g.fillRect (bar.withHeight (1));

    if (editPlan_.clips.empty()) return;
    if (viewDurationSec_ <= 0.0) return;

    // Sesja 64 — cells delimited by the same splice marker positions painted
    // on the waveform above so cell edges line up 1:1 with the orange ticks.
    // In Remix variant (where this bar lives) the waveform is the *remix*
    // audio, so spliceMarkers_.timeSec (remix-time) is the right axis. Fall
    // back to userBlockSplices_ (source-time, Block Assembly diagnostic) if
    // spliceMarkers_ is empty, then to legacy clip-range mapping.
    const bool useRemixSplices  = ! spliceMarkers_.empty();
    const bool useSourceSplices = ! useRemixSplices && ! userBlockSplices_.empty();

    const juce::Colour grayA = juce::Colour::fromRGB (0x3a, 0x3a, 0x3a);
    const juce::Colour grayB = juce::Colour::fromRGB (0x4a, 0x4a, 0x4a);
    const juce::Colour cellDivider { juce::Colour::fromFloatRGBA (0.0f, 0.0f, 0.0f, 0.35f) };
    const juce::Colour labelColour { juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.85f) };
    const auto labelFont = monoFont (fs::Xs, 600);

    const int barLeft  = bar.getX();
    const int barRight = bar.getRight();

    // Build delimiter list (continuous; cell N spans delims[N] .. delims[N+1]).
    std::vector<double> delims;
    const double total = peaks_ ? peaks_->getTotalDurationSeconds() : 0.0;
    const double endBound = total > 0.0 ? total : (viewStartSec_ + viewDurationSec_);
    delims.push_back (0.0);
    if (useRemixSplices)
    {
        for (const auto& sm : spliceMarkers_) delims.push_back (sm.timeSec);
    }
    else if (useSourceSplices)
    {
        for (const auto& sp : userBlockSplices_) delims.push_back (sp.sourceTimeSec);
    }
    else
    {
        // Legacy editPlan delimiters for Duration / Region mode.
        for (const auto& c : editPlan_.clips) delims.push_back (c.sourceStartSec);
        if (! editPlan_.clips.empty())
            delims.push_back (editPlan_.clips.back().sourceEndSec);
    }
    delims.push_back (endBound);
    std::sort (delims.begin(), delims.end());
    delims.erase (std::unique (delims.begin(), delims.end(),
        [] (double a, double b) { return std::abs (a - b) < 1e-3; }), delims.end());

    for (std::size_t i = 0; i + 1 < delims.size(); ++i)
    {
        const double t0 = delims[i];
        const double t1 = delims[i + 1];
        if (t1 <= t0) continue;

        const float xStart = timeToX (t0, barLeft, bar.getWidth());
        const float xEnd   = timeToX (t1, barLeft, bar.getWidth());

        if (xEnd < (float) barLeft || xStart > (float) barRight) continue;

        const int cellX = juce::jmax (barLeft,  juce::roundToInt (xStart));
        const int cellR = juce::jmin (barRight, juce::roundToInt (xEnd));
        const int cellW = cellR - cellX;
        if (cellW <= 0) continue;

        g.setColour ((i % 2 == 0) ? grayA : grayB);
        g.fillRect (cellX, bar.getY() + 1, cellW, bar.getHeight() - 1);

        const bool rightOnScreen = juce::roundToInt (xEnd) <= barRight;
        if (rightOnScreen && i + 2 < delims.size())
        {
            g.setColour (cellDivider);
            g.fillRect (cellX + cellW - 1, bar.getY() + 1, 1, bar.getHeight() - 1);
        }

        if (cellW > 28)
        {
            const int seconds = juce::jmax (1, (int) std::lround (t1 - t0));
            juce::String label = juce::String (seconds) + "s";
            g.setColour (labelColour);
            g.setFont (labelFont);
            g.drawText (label,
                        cellX + 4, bar.getY(),
                        std::max (0, cellW - 8), bar.getHeight(),
                        juce::Justification::centredLeft, false);
        }
    }
}

void WaveformView::paintSpliceMarkers (juce::Graphics& g)
{
    // mockup .rx-splice + plugin.css:400-416. 1 px Q-color full-canvas line +
    // 9×7 downward triangle at canvas top. q-good/warn/bad map to theme
    // tokens defined in tokens.css (--rx-good / --rx-warn / --rx-bad).
    if (spliceMarkers_.empty() || viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    const double viewEnd = viewStartSec_ + viewDurationSec_;
    const float yT = (float) canvas.getY();
    const float yB = (float) canvas.getBottom();

    for (std::size_t i = 0; i < spliceMarkers_.size(); ++i)
    {
        const auto& m = spliceMarkers_[i];
        if (m.timeSec < viewStartSec_ - 1e-3 || m.timeSec > viewEnd + 1e-3) continue;

        juce::Colour c;
        switch (m.quality)
        {
            case SpliceQuality::Good:   c = Good;  break;
            case SpliceQuality::Medium: c = Warn;  break;
            case SpliceQuality::Bad:    c = Bad;   break;
        }
        // Hover brighten — plugin.css:416 filter: brightness(1.25).
        if ((int) i == hoveredSpliceIdx_)
            c = c.brighter (0.25f);

        const int ix = (int) timeToX (m.timeSec, canvas.getX(), canvas.getWidth());
        g.setColour (c);
        g.drawVerticalLine (ix, yT, yB);

        // 9×7 downward triangle at top — clip-path polygon(0 0, 100% 0, 50% 100%).
        juce::Path tri;
        const float triW = 9.0f;
        const float triH = 7.0f;
        const float triX = (float) ix - triW * 0.5f;
        tri.startNewSubPath (triX,              yT - 1.0f);
        tri.lineTo          (triX + triW,       yT - 1.0f);
        tri.lineTo          (triX + triW * 0.5f, yT - 1.0f + triH);
        tri.closeSubPath();
        g.fillPath (tri);
    }
}

int WaveformView::spliceHitTest (int cursorX) const
{
    if (spliceMarkers_.empty() || viewDurationSec_ <= 0.0) return -1;
    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return -1;
    const double viewEnd = viewStartSec_ + viewDurationSec_;

    int    bestIdx = -1;
    int    bestDx  = kSpliceHitToleranceX + 1;
    for (std::size_t i = 0; i < spliceMarkers_.size(); ++i)
    {
        const auto& m = spliceMarkers_[i];
        if (m.timeSec < viewStartSec_ - 1e-3 || m.timeSec > viewEnd + 1e-3) continue;
        const int ix = (int) timeToX (m.timeSec, canvas.getX(), canvas.getWidth());
        const int dx = std::abs (cursorX - ix);
        if (dx <= kSpliceHitToleranceX && dx < bestDx)
        {
            bestDx  = dx;
            bestIdx = (int) i;
        }
    }
    return bestIdx;
}

void WaveformView::updateHoveredSplice (int newIdx, juce::Point<int> screenPos)
{
    if (newIdx == hoveredSpliceIdx_) return;
    hoveredSpliceIdx_ = newIdx;
    repaint (canvasArea());
    if (onSpliceHoverChanged) onSpliceHoverChanged (newIdx, screenPos);
}

double WaveformView::snapTimeToBeats (double t) const
{
    if (snapMode_ == SnapMode::Off || beats_.empty()) return t;

    const bool downbeatsOnly = (snapMode_ == SnapMode::Downbeats);
    double bestT = t;
    double bestD = std::numeric_limits<double>::infinity();
    for (const auto& bt : beats_)
    {
        if (downbeatsOnly && ! bt.isDownbeat) continue;
        const double d = std::abs (bt.time - t);
        if (d < bestD) { bestD = d; bestT = bt.time; }
    }
    return bestT;
}

int WaveformView::userBlockHitTest (juce::Point<int> pos) const
{
    if (userBlocks_.empty() || viewDurationSec_ <= 0.0) return -1;
    const auto bar = segBarArea();
    if (! bar.contains (pos)) return -1;

    const double viewEnd = viewStartSec_ + viewDurationSec_;
    for (std::size_t i = 0; i < userBlocks_.size(); ++i)
    {
        const auto& b = userBlocks_[i];
        if (b.endSec < viewStartSec_ - 1e-6 || b.startSec > viewEnd + 1e-6) continue;
        const int x0 = (int) timeToX (b.startSec, bar.getX(), bar.getWidth());
        const int x1 = (int) timeToX (b.endSec,   bar.getX(), bar.getWidth());
        if (pos.x >= x0 && pos.x <= x1) return (int) i;
    }
    return -1;
}

// DEV-029 (d) sesja 100b — boundary hit test on segBar. Iterates user
// blocks in display order, picks the closest edge within kBoundaryHitPx
// pixels. Returns block idx + side (start vs end) via out param.
int WaveformView::userBlockBoundaryHitTest (juce::Point<int> pos,
                                              bool* outIsStart) const
{
    if (outIsStart) *outIsStart = false;
    if (userBlocks_.empty() || viewDurationSec_ <= 0.0) return -1;
    const auto bar = segBarArea();
    if (! bar.contains (pos)) return -1;

    const double viewEnd = viewStartSec_ + viewDurationSec_;
    int  bestIdx = -1;
    int  bestDx  = kBoundaryHitPx + 1;
    bool bestIsStart = false;
    for (std::size_t i = 0; i < userBlocks_.size(); ++i)
    {
        const auto& b = userBlocks_[i];
        if (b.endSec < viewStartSec_ - 1e-6 || b.startSec > viewEnd + 1e-6) continue;
        const int x0 = (int) timeToX (b.startSec, bar.getX(), bar.getWidth());
        const int x1 = (int) timeToX (b.endSec,   bar.getX(), bar.getWidth());
        const int dx0 = std::abs (pos.x - x0);
        const int dx1 = std::abs (pos.x - x1);
        if (dx0 <= kBoundaryHitPx && dx0 < bestDx)
        {
            bestIdx = (int) i; bestDx = dx0; bestIsStart = true;
        }
        if (dx1 <= kBoundaryHitPx && dx1 < bestDx)
        {
            bestIdx = (int) i; bestDx = dx1; bestIsStart = false;
        }
    }
    if (bestIdx >= 0 && outIsStart) *outIsStart = bestIsStart;
    return bestIdx;
}

int WaveformView::userBlockSpliceHitTest (int cursorX) const
{
    if (userBlockSplices_.empty() || viewDurationSec_ <= 0.0) return -1;
    const auto bar = segBarArea();
    if (bar.isEmpty()) return -1;
    const double viewEnd = viewStartSec_ + viewDurationSec_;

    int    bestIdx = -1;
    int    bestDx  = kSpliceHitToleranceX + 1;
    for (std::size_t i = 0; i < userBlockSplices_.size(); ++i)
    {
        const auto& sp = userBlockSplices_[i];
        if (sp.sourceTimeSec < viewStartSec_ - 1e-3 || sp.sourceTimeSec > viewEnd + 1e-3) continue;
        const int ix = (int) timeToX (sp.sourceTimeSec, bar.getX(), bar.getWidth());
        const int dx = std::abs (cursorX - ix);
        if (dx <= kSpliceHitToleranceX && dx < bestDx)
        {
            bestDx  = dx;
            bestIdx = (int) i;
        }
    }
    return bestIdx;
}

void WaveformView::paintSelection (juce::Graphics& g)
{
    // ADR-039 MVP — Accent-tinted scrim + 1 px edges spanning the canvas
    // region only (not segbar / ruler / toolbar). Visual reference:
    // mockup waveform-blocks.jsx:99-119 (region overlay, accent 18% fill +
    // accent border) and EditView DrawSelection pattern.
    if (viewDurationSec_ <= 0.0) return;

    double a = 0.0, b = 0.0;
    if (dragging_)
    {
        a = std::min (dragStartSec_, dragCurrentSec_);
        b = std::max (dragStartSec_, dragCurrentSec_);
    }
    else if (selection_.has_value())
    {
        a = selection_->startSec;
        b = selection_->endSec;
    }
    else
    {
        return;
    }
    if (b - a < 1e-6) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    const float x0 = timeToX (a, canvas.getX(), canvas.getWidth());
    const float x1 = timeToX (b, canvas.getX(), canvas.getWidth());
    if (x1 <= x0) return;

    const float y0 = static_cast<float> (canvas.getY());
    const float yH = static_cast<float> (canvas.getHeight());

    // Sesja 100 iter 4 (DEV-070) — selection scrim + edges colour shifted
    // from Accent (orange) to Info (cool blue) so they don't visually
    // conflict with orange medium-quality splice markers. Same palette as
    // playhead; selection + playhead form a "navigation" colour family
    // distinct from the splice-quality semantic family (Good/Warn/Bad).
    const juce::Colour fill { Info.withAlpha (0.18f) };
    g.setColour (fill);
    g.fillRect (juce::Rectangle<float> (x0, y0, x1 - x0, yH));

    g.setColour (Info);
    g.drawVerticalLine (juce::roundToInt (x0), y0, y0 + yH);
    g.drawVerticalLine (juce::roundToInt (x1), y0, y0 + yH);

    // Sesja 100 iter 3 (DEV-070) — live selection-length readout during
    // active drag-select. User reports: "kiedy zaznaczam to powinienm
    // widziec dlugosc ktora zaznacza". Renders a small pill centred over
    // the selection at canvas top, showing formatted Δ length. Hidden once
    // selection is committed (mouseUp). Sesja 100 iter 4 — pill colour
    // also shifted to Info to match scrim/edges.
    if (dragging_)
    {
        const double length = b - a;
        const juce::String text = "\xCE\x94 " + formatHoverSeconds (length);
        const auto pillFont = monoFont (fs::Xs, 400);

        juce::GlyphArrangement arr;
        arr.addLineOfText (pillFont, text, 0.0f, 0.0f);
        const float textW = arr.getBoundingBox (0, -1, false).getWidth();

        const float pillW = textW + 8.0f;
        const float pillH = 11.0f;
        const float midX  = (x0 + x1) * 0.5f;
        float pillX = midX - pillW * 0.5f;
        pillX = juce::jlimit (static_cast<float> (canvas.getX()),
                              canvas.getRight() - pillW,
                              pillX);
        const float pillY = y0 + 2.0f;

        g.setColour (Info);
        g.fillRoundedRectangle (pillX, pillY, pillW, pillH, r::R1);

        g.setColour (Bg0);
        g.setFont (pillFont);
        g.drawText (text,
                    juce::roundToInt (pillX), juce::roundToInt (pillY),
                    juce::roundToInt (pillW), juce::roundToInt (pillH),
                    juce::Justification::centred, false);
    }
}

void WaveformView::paintPlayhead (juce::Graphics& g)
{
    // plugin.css:315-325 — .rx-wave-playhead { 1px accent line + 0.5px accent
    // glow halo at 50% α + 7×6 downward-pointing triangle at top }. Both glow
    // and triangle are additions to session-47's plain-line playhead.
    if (! playheadSec_.has_value() || viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    const double t   = *playheadSec_;
    const double rel = viewStartSec_ + viewDurationSec_ <= 0.0
                     ? 0.0
                     : (t - viewStartSec_) / viewDurationSec_;
    if (rel < 0.0 || rel > 1.0) return;

    const int   ix = (int) timeToX (t, canvas.getX(), canvas.getWidth());
    const float yT = (float) canvas.getY();
    const float yB = (float) canvas.getBottom();

    // Sesja 100 (DEV-068) — playhead colour shifted from Accent (orange) to
    // Info (#6A9BC4 cool blue) so it doesn't visually conflict with orange
    // medium-quality splice markers. Info palette token already in tokens.css
    // (line 36) — fits the existing design without introducing a new colour.
    // 0.5 px glow halo — drawn as two adjacent vertical lines at 50 % α
    // flanking the core line.
    g.setColour (Info.withAlpha (0.5f));
    g.drawVerticalLine (ix - 1, yT, yB);
    g.drawVerticalLine (ix + 1, yT, yB);

    // Core 1 px Info line.
    g.setColour (Info);
    g.drawVerticalLine (ix, yT, yB);

    // 7 × 6 downward-pointing triangle at canvas top — centred on the line.
    // plugin.css:321-325 clip-path: polygon(0 0, 100% 0, 50% 100%).
    juce::Path tri;
    const float triW = 7.0f;
    const float triH = 6.0f;
    const float triX = (float) ix - triW * 0.5f;
    tri.startNewSubPath (triX,              yT);
    tri.lineTo          (triX + triW,       yT);
    tri.lineTo          (triX + triW * 0.5f, yT + triH);
    tri.closeSubPath();
    g.setColour (Info);  // sesja 100 DEV-068 — match repositioned playhead line
    g.fillPath (tri);
}

void WaveformView::paintFlash (juce::Graphics& g)
{
    // REABeat WaveformView.cpp:935-955 pattern. Triggered on click-to-seek;
    // 250 ms expanding Accent ellipse, fade alpha from 0.6 → 0 linear.
    if (flashPosSec_ < 0.0 || viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    const std::uint32_t elapsed = juce::Time::getMillisecondCounter() - flashStartMs_;
    if (elapsed >= (std::uint32_t) kFlashDurationMs) return;

    const double rel = (flashPosSec_ - viewStartSec_) / viewDurationSec_;
    if (rel < 0.0 || rel > 1.0) return;

    const float progress = (float) elapsed / (float) kFlashDurationMs;
    const float radius   = kFlashStartR + progress * (kFlashEndR - kFlashStartR);
    const float alpha    = 1.0f - progress;
    const float fx       = timeToX (flashPosSec_, canvas.getX(), canvas.getWidth());
    const float cy       = canvas.getY() + canvas.getHeight() * 0.5f;

    g.setColour (Accent.withAlpha (alpha * 0.6f));
    g.drawEllipse (fx - radius, cy - radius, radius * 2.0f, radius * 2.0f, 1.5f);
}

void WaveformView::triggerFlash (double seconds)
{
    flashPosSec_  = seconds;
    flashStartMs_ = juce::Time::getMillisecondCounter();
    if (! isTimerRunning())
        startTimerHz (60);
    repaint (canvasArea());
}

void WaveformView::timerCallback()
{
    bool keepAlive = false;

    if (flashPosSec_ >= 0.0)
    {
        const std::uint32_t elapsed = juce::Time::getMillisecondCounter() - flashStartMs_;
        if (elapsed >= (std::uint32_t) kFlashDurationMs)
            flashPosSec_ = -1.0;
        else
            keepAlive = true;
        repaint (canvasArea());
    }

    // DEV-055 sesja 100c — pulse the empty-segBar hint while it's visible.
    // Re-checks predicate so segBarHintActive_ stays in sync if the user
    // marks/unmarks blocks while paint hasn't run.
    const bool wantHint = blockMarkingEnabled_ && userBlocks_.empty() && ! segBarDragging_;
    segBarHintActive_ = wantHint;
    if (wantHint)
    {
        keepAlive = true;
        repaint (segBarArea());
    }

    if (! keepAlive) stopTimer();
}

void WaveformView::paintHover (juce::Graphics& g)
{
    if (cursorX_ < 0 || viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    const auto bar    = segBarArea();
    if (canvas.isEmpty()) return;

    const int cx = cursorX_;
    if (cx < canvas.getX() || cx > canvas.getRight()) return;

    // plugin.css:326-330 — 1px white 0.25α from canvas top to segbar bottom.
    const juce::Colour line { juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.25f) };
    g.setColour (line);
    g.drawVerticalLine (cx,
                        static_cast<float> (canvas.getY()),
                        static_cast<float> (bar.getBottom()));

    // plugin.css:331-341 — pill at top:2, padding 1px 4px, Mono 9,
    // bg-4 background, fg-0 text, 2px radius, translateX(-50%).
    const juce::String text = formatHoverSeconds (xToTime (static_cast<float> (cx),
                                                            canvas.getX(), canvas.getWidth()));
    const auto pillFont = monoFont (fs::Xs, 400);

    juce::GlyphArrangement arr;
    arr.addLineOfText (pillFont, text, 0.0f, 0.0f);
    const float textW = arr.getBoundingBox (0, -1, false).getWidth();

    const float pillW = textW + 8.0f; // 4px horizontal padding each side.
    const float pillH = 11.0f;        // ~9px line-height + 1px top/bottom pad.
    float pillX = static_cast<float> (cx) - pillW * 0.5f;
    pillX = juce::jlimit (static_cast<float> (canvas.getX()),
                          canvas.getRight() - pillW,
                          pillX);
    const float pillY = canvas.getY() + 2.0f;

    g.setColour (Bg4);
    g.fillRoundedRectangle (pillX, pillY, pillW, pillH, r::R1);

    g.setColour (Fg0);
    g.setFont (pillFont);
    g.drawText (text,
                juce::roundToInt (pillX), juce::roundToInt (pillY),
                juce::roundToInt (pillW), juce::roundToInt (pillH),
                juce::Justification::centred, false);
}

// ── Layout + invalidation ───────────────────────────────────────────

void WaveformView::resized()
{
    cache_.valid = false;
}

// ── Mouse ───────────────────────────────────────────────────────────

void WaveformView::mouseMove (const juce::MouseEvent& e)
{
    // ADR-045 — toolbar removed; mouseMove only manages the IBeam scrubber
    // over the canvas/segbar region. Ruler hover area is excluded by the
    // hoverDirtyRect span (canvas top → segbar bottom).
    const auto pos = e.getPosition();

    // Sesja 60 — Edit region overlay button hover takes priority over the
    // splice/IBeam scrubber so the user gets pointer + button highlight.
    if (showEditRegionButton_)
    {
        const bool overBtn = editRegionBtnBounds_.contains (pos);
        if (overBtn != editRegionHover_)
        {
            editRegionHover_ = overBtn;
            repaint (editRegionBtnBounds_);
        }
        if (overBtn)
        {
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
            return;
        }
    }

    if (showEditArrangementButton_)
    {
        const bool overBtn = editArrangementBtnBounds_.contains (pos);
        if (overBtn != editArrangementHover_)
        {
            editArrangementHover_ = overBtn;
            repaint (editArrangementBtnBounds_);
        }
        if (overBtn)
        {
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
            return;
        }
    }

    // Sesja 100 (DEV-018) — volume overlay hover takes priority over the
    // canvas IBeam scrubber. Always rendered → always hit-tested.
    {
        const bool overBtn = volumeBtnBounds_.contains (pos);
        if (overBtn != volumeHover_)
        {
            volumeHover_ = overBtn;
            repaint (volumeBtnBounds_);
        }
        if (overBtn)
        {
            setMouseCursor (juce::MouseCursor::PointingHandCursor);
            return;
        }
    }

    // DEV-029 (d) sesja 100b — boundary hover on segBar takes priority
    // over splice / IBeam cursors. Cursor flips to LeftRightResize when
    // within kBoundaryHitPx of an existing block edge so the user knows
    // a drag will resize, not create a new block.
    if (blockMarkingEnabled_ && segBarArea().contains (pos))
    {
        bool isStart = false;
        if (userBlockBoundaryHitTest (pos, &isStart) >= 0)
        {
            setMouseCursor (juce::MouseCursor::LeftRightResizeCursor);
            // Hovered-block tracking still updates so the underlying
            // tile (or both tiles when boundaries are shared) glows.
            const int hbIdx = userBlockHitTest (pos);
            if (hbIdx != hoveredUserBlockIdx_)
            {
                hoveredUserBlockIdx_ = hbIdx;
                repaint (segBarArea());
            }
            return;
        }
        // Sesja 100b iter 2 — block-tile hover (no-edge hit). Subtle
        // brightness boost on the tile beneath the cursor so the user
        // can see which block they're about to manipulate; matters most
        // when adjacent tiles touch and the boundary visual is ambiguous.
        // DEV-057 sesja 100d — also flip cursor to DraggingHand when the
        // hover lands inside an existing tile body so the user knows
        // a drag will translate the block (not create a new one).
        const int hbIdx = userBlockHitTest (pos);
        if (hbIdx != hoveredUserBlockIdx_)
        {
            hoveredUserBlockIdx_ = hbIdx;
            repaint (segBarArea());
        }
        if (hbIdx >= 0)
        {
            setMouseCursor (juce::MouseCursor::DraggingHandCursor);
            return;
        }
    }
    else if (hoveredUserBlockIdx_ != -1)
    {
        hoveredUserBlockIdx_ = -1;
        repaint (segBarArea());
    }

    // DEV-031 sesja 100b — canvas selection edge hover. Same affordance
    // logic as the segBar boundary above: resize cursor on either edge
    // signals that a drag will move that edge, not start a fresh
    // selection.
    if (selection_.has_value() && canvasArea().contains (pos))
    {
        const auto canvas = canvasArea();
        const int xStart = (int) timeToX (selection_->startSec,
                                            canvas.getX(), canvas.getWidth());
        const int xEnd   = (int) timeToX (selection_->endSec,
                                            canvas.getX(), canvas.getWidth());
        if (std::abs (pos.x - xStart) <= kSelectionEdgeHitPx
            || std::abs (pos.x - xEnd) <= kSelectionEdgeHitPx)
        {
            setMouseCursor (juce::MouseCursor::LeftRightResizeCursor);
            return;
        }
    }

    // Splice hit-test (Remix variant). Pointing cursor when over a marker.
    int newSplice = -1;
    if (variant_ == Variant::Remix && canvasArea().contains (pos))
        newSplice = spliceHitTest (pos.x);

    // ADR-051 — user-block splice hit-test (Source variant in Blocks mode).
    int newUserBlockSplice = -1;
    if (variant_ == Variant::Source && segBarArea().contains (pos))
        newUserBlockSplice = userBlockSpliceHitTest (pos.x);

    const bool overSomething = (newSplice >= 0) || (newUserBlockSplice >= 0);
    setMouseCursor (overSomething
                    ? juce::MouseCursor::PointingHandCursor
                    : juce::MouseCursor::IBeamCursor);

    updateHoveredSplice (newSplice, e.getScreenPosition());

    if (newUserBlockSplice != hoveredSpliceUserIdx_)
    {
        hoveredSpliceUserIdx_ = newUserBlockSplice;
        repaint (segBarArea());
        if (onUserBlockSpliceHover)
            onUserBlockSpliceHover (newUserBlockSplice, e.getScreenPosition());
    }

    const int prev = cursorX_;
    cursorX_ = pos.x;

    if (prev != cursorX_)
    {
        if (prev >= 0) repaint (hoverDirtyRect (prev));
        repaint (hoverDirtyRect (cursorX_));
    }
}

void WaveformView::mouseExit (const juce::MouseEvent&)
{
    if (cursorX_ >= 0)
    {
        repaint (hoverDirtyRect (cursorX_));
        cursorX_ = -1;
    }
    if (hoveredSpliceIdx_ != -1)
        updateHoveredSplice (-1, {});
    if (hoveredUserBlockIdx_ != -1)
    {
        hoveredUserBlockIdx_ = -1;
        repaint (segBarArea());
    }
}

void WaveformView::mouseDown (const juce::MouseEvent& e)
{
    const auto pos = e.getPosition();

    // Sesja 60 — Edit region overlay button intercepts before any other
    // canvas handling (splice / drag-select / seek).
    if (showEditRegionButton_ && editRegionBtnBounds_.contains (pos))
    {
        if (onEditRegionClicked) onEditRegionClicked();
        return;
    }

    // Sesja 61 hot-fix — Edit arrangement overlay (Blocks mode parallel).
    if (showEditArrangementButton_ && editArrangementBtnBounds_.contains (pos))
    {
        if (onEditArrangementClicked) onEditArrangementClicked();
        return;
    }

    // Sesja 100 iter 3 (DEV-018) — volume overlay click toggles inline
    // slider popup (no CallOutBox; bypasses the bubble/arrow background).
    if (volumeBtnBounds_.contains (pos))
    {
        if (volumePopupVisible_) hideVolumePopup();
        else                     showVolumePopup();
        return;
    }

    // Sesja 100 iter 3 (DEV-018) — clicking anywhere outside the popup
    // dismisses it (the click STILL falls through to canvas / segbar
    // handling below; popup hide is best-effort dismiss like macOS modals).
    if (volumePopupVisible_
        && ! volumePopupSlider_.getBounds().contains (pos))
    {
        hideVolumePopup();
        // intentionally fall through — user's click on the canvas should
        // still scrub / drag-select normally.
    }

    if (viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    const auto bar    = segBarArea();
    const bool insideCanvas = canvas.contains (pos);
    const bool insideSegBar = bar.contains (pos);
    if (! (insideCanvas || insideSegBar)) return;

    // Splice marker handling — Remix variant only. Right-click → context
    // menu (4 actions: Try different / Reset / Block / Seek). Left-click →
    // audition seam (MainComponent plays ±1 s around the marker time).
    // Either click consumes the event so drag-select / seek do not also fire.
    if (variant_ == Variant::Remix && insideCanvas)
    {
        const int spliceIdx = spliceHitTest (pos.x);
        if (spliceIdx >= 0 && spliceIdx < (int) spliceMarkers_.size())
        {
            grabKeyboardFocus();
            if (e.mods.isPopupMenu())
            {
                if (onSpliceContextMenu) onSpliceContextMenu (spliceIdx, e.getScreenPosition());
            }
            else
            {
                if (onSpliceClick) onSpliceClick (spliceIdx, spliceMarkers_[(std::size_t) spliceIdx].timeSec);
            }
            return;
        }
    }

    // Grab keyboard focus so keyPressed overrides fire (Cmd+A / Home / End /
    // Shift-extend). Grabbing on every mouseDown matches DAW convention —
    // clicking the waveform makes it the focus target for subsequent keys.
    grabKeyboardFocus();

    // ADR-051 — Block Assembly mode: segBar is the marking surface. Hits in
    // segBar take a different gesture path from canvas drag-select. We start
    // tracking even on existing blocks (so click vs drag is decided in mouseUp);
    // segBarDragStartedOnBlock_ suppresses new-block creation in the drag case.
    if (insideSegBar && blockMarkingEnabled_)
    {
        // DEV-029 (d) sesja 100b — boundary-drag has priority over the
        // generic segBar gestures: when cursor lands within kBoundaryHitPx
        // of an existing block edge, we resize that edge instead of
        // creating a new block (drag) or opening edit popover (click).
        bool isStart = false;
        const int boundaryIdx = userBlockBoundaryHitTest (pos, &isStart);
        if (boundaryIdx >= 0)
        {
            const auto& b = userBlocks_[(std::size_t) boundaryIdx];
            // Clamp range: previous neighbour's end (or 0) ↔ next
            // neighbour's start (or item duration). kMinBlockSec guards
            // against collapsing the dragged block; the OPPOSITE edge
            // also acts as a clamp so the block preserves a minimum width.
            const double prevEnd  = (boundaryIdx > 0)
                ? userBlocks_[(std::size_t) (boundaryIdx - 1)].endSec : 0.0;
            const double nextStart = (boundaryIdx + 1 < (int) userBlocks_.size())
                ? userBlocks_[(std::size_t) (boundaryIdx + 1)].startSec
                : viewStartSec_ + viewDurationSec_;

            // Sesja 100b iter 2 — shared boundary detection. If we grabbed
            // block A's right edge AND block B's left edge sits at the same
            // time (within kSharedBoundaryEps), drag updates both block A
            // and block B. Symmetric for grabbing B's left edge while A's
            // right edge touches. Clamp range widens correspondingly: we
            // can move the join up to the far neighbour's edge of either
            // side (minus minBlockSec for the side being squeezed).
            int    sharedIdx     = -1;
            bool   sharedIsStart = false;
            double sharedFixed   = 0.0;
            double extendedClampMin = 0.0;
            double extendedClampMax = 0.0;
            if (! isStart && boundaryIdx + 1 < (int) userBlocks_.size())
            {
                const auto& nb = userBlocks_[(std::size_t) (boundaryIdx + 1)];
                if (std::abs (b.endSec - nb.startSec) < kSharedBoundaryEps)
                {
                    sharedIdx     = boundaryIdx + 1;
                    sharedIsStart = true;             // we move B's start
                    sharedFixed   = nb.endSec;        // B's right edge stays
                    const double nbNextStart = (sharedIdx + 1 < (int) userBlocks_.size())
                        ? userBlocks_[(std::size_t) (sharedIdx + 1)].startSec
                        : viewStartSec_ + viewDurationSec_;
                    extendedClampMin = b.startSec    + kMinBlockSec;
                    extendedClampMax = nb.endSec     - kMinBlockSec;
                    // Also clamp against B's next neighbour (no overlap).
                    extendedClampMax = std::min (extendedClampMax, nbNextStart);
                }
            }
            else if (isStart && boundaryIdx > 0)
            {
                const auto& pb = userBlocks_[(std::size_t) (boundaryIdx - 1)];
                if (std::abs (pb.endSec - b.startSec) < kSharedBoundaryEps)
                {
                    sharedIdx     = boundaryIdx - 1;
                    sharedIsStart = false;            // we move A's end
                    sharedFixed   = pb.startSec;      // A's left edge stays
                    const double pbPrevEnd = (sharedIdx > 0)
                        ? userBlocks_[(std::size_t) (sharedIdx - 1)].endSec
                        : 0.0;
                    extendedClampMin = pb.startSec   + kMinBlockSec;
                    extendedClampMax = b.endSec      - kMinBlockSec;
                    extendedClampMin = std::max (extendedClampMin, pbPrevEnd);
                }
            }

            segBarBoundaryDragging_ = true;
            segBarBoundaryBlockIdx_ = boundaryIdx;
            segBarBoundaryIsStart_  = isStart;
            segBarBoundaryFixedSec_ = isStart ? b.endSec : b.startSec;
            segBarBoundaryLiveSec_  = isStart ? b.startSec : b.endSec;
            if (sharedIdx >= 0)
            {
                // Shared-boundary drag: clamp range spans both blocks.
                segBarBoundarySharedBlockIdx_ = sharedIdx;
                segBarBoundarySharedIsStart_  = sharedIsStart;
                segBarBoundarySharedFixedSec_ = sharedFixed;
                segBarBoundaryClampMin_       = extendedClampMin;
                segBarBoundaryClampMax_       = extendedClampMax;
            }
            else
            {
                segBarBoundarySharedBlockIdx_ = -1;
                segBarBoundaryClampMin_ = isStart ? prevEnd
                                                    : (b.startSec + kMinBlockSec);
                segBarBoundaryClampMax_ = isStart ? (b.endSec - kMinBlockSec)
                                                    : nextStart;
            }
            // Keep ew-resize cursor active during drag.
            setMouseCursor (juce::MouseCursor::LeftRightResizeCursor);
            return;
        }

        const int hitIdx = userBlockHitTest (pos);
        const double tBar = juce::jlimit (viewStartSec_,
                                           viewStartSec_ + viewDurationSec_,
                                           xToTime (static_cast<float> (pos.x),
                                                    bar.getX(), bar.getWidth()));
        segBarDragging_           = true;
        segBarDragStartPos_       = pos;
        segBarDragStartSec_       = tBar;
        segBarDragCurrentSec_     = tBar;
        segBarDragStartedOnBlock_ = (hitIdx >= 0);
        segBarDragStartBlockIdx_  = hitIdx;

        // DEV-057 sesja 100d — pre-compute mid-drag clamps if we landed on
        // an existing block. mouseDrag will activate segBarMidDragActive_
        // once the cursor moves past the click-vs-drag threshold. The
        // clamps mirror the boundary-drag math (prev neighbour end /
        // next neighbour start) but for translation, not resize: we cap
        // the new startSec so endSec = startSec + duration never collides
        // with the next neighbour, and startSec never crosses the prev.
        if (hitIdx >= 0)
        {
            const auto& b = userBlocks_[(std::size_t) hitIdx];
            segBarMidDragBlockIdx_     = hitIdx;
            segBarMidDragOriginalDur_  = b.endSec - b.startSec;
            segBarMidDragOffsetSec_    = tBar - b.startSec;
            segBarMidDragLiveStart_    = b.startSec;
            const double prevEnd  = (hitIdx > 0)
                ? userBlocks_[(std::size_t) (hitIdx - 1)].endSec : 0.0;
            const double nextStart = (hitIdx + 1 < (int) userBlocks_.size())
                ? userBlocks_[(std::size_t) (hitIdx + 1)].startSec
                : viewStartSec_ + viewDurationSec_;
            segBarMidDragClampMinStart_ = prevEnd;
            segBarMidDragClampMaxStart_ = nextStart - segBarMidDragOriginalDur_;
        }
        return;
    }

    const double t = juce::jlimit (viewStartSec_,
                                    viewStartSec_ + viewDurationSec_,
                                    xToTime (static_cast<float> (pos.x),
                                             canvas.getX(), canvas.getWidth()));

    // DEV-031 sesja 100b — selection-edge drag. When a finalized
    // selection_ exists, clicking within kSelectionEdgeHitPx of either
    // edge enters edge-drag mode (rather than starting a fresh drag-
    // select). Drag adjusts only that edge; the opposite edge stays
    // pinned. Live audio scrub via PreviewController::seek when preview
    // is already playing — see mouseDrag.
    if (selection_.has_value() && ! e.mods.isShiftDown())
    {
        const int xStart = (int) timeToX (selection_->startSec,
                                            canvas.getX(), canvas.getWidth());
        const int xEnd   = (int) timeToX (selection_->endSec,
                                            canvas.getX(), canvas.getWidth());
        const int dxStart = std::abs (pos.x - xStart);
        const int dxEnd   = std::abs (pos.x - xEnd);
        if (dxStart <= kSelectionEdgeHitPx && dxStart < dxEnd)
        {
            dragging_                = true;
            selectionEdgeDragging_   = true;
            selectionEdgeIsStart_    = true;
            dragStartPos_            = pos;
            dragStartSec_            = selection_->endSec;   // pinned edge
            dragCurrentSec_          = selection_->startSec; // moving edge
            beginDragAutoRepeat (kAutoRepeatMs);
            setMouseCursor (juce::MouseCursor::LeftRightResizeCursor);
            return;
        }
        if (dxEnd <= kSelectionEdgeHitPx)
        {
            dragging_                = true;
            selectionEdgeDragging_   = true;
            selectionEdgeIsStart_    = false;
            dragStartPos_            = pos;
            dragStartSec_            = selection_->startSec; // pinned edge
            dragCurrentSec_          = selection_->endSec;   // moving edge
            beginDragAutoRepeat (kAutoRepeatMs);
            setMouseCursor (juce::MouseCursor::LeftRightResizeCursor);
            return;
        }
    }

    // ADR-039 full — Shift+click extends the existing selection. Anchor the
    // drag-start at the FAR edge of the current selection so mouseDrag/Up
    // produces a range from that edge to the cursor. If no selection exists
    // yet, Shift behaves like a normal click (falls through to the default
    // drag-select path below).
    if (e.mods.isShiftDown() && selection_.has_value())
    {
        const double mid = 0.5 * (selection_->startSec + selection_->endSec);
        const double farEdge = (t < mid) ? selection_->endSec : selection_->startSec;
        dragging_        = true;
        dragStartPos_    = pos;
        dragStartSec_    = farEdge;
        dragCurrentSec_  = t;
        beginDragAutoRepeat (kAutoRepeatMs);
        repaint (canvas);
        return;
    }

    // ADR-039 MVP — mouseDown starts a drag-select. onSeek / selection
    // decision is deferred to mouseUp based on the drag-distance threshold.
    // Gesture is cancelled entirely if mouseUp fires outside the waveform.
    dragging_        = true;
    dragStartPos_    = pos;
    dragStartSec_    = t;
    dragCurrentSec_  = t;
    beginDragAutoRepeat (kAutoRepeatMs);
}

void WaveformView::mouseDrag (const juce::MouseEvent& e)
{
    if (viewDurationSec_ <= 0.0) return;

    // DEV-029 (d) sesja 100b — segBar boundary-drag: live-update the
    // dragged edge clamped to neighbour bounds + min block size. Block
    // mutation is deferred to mouseUp; meanwhile paintSegBar reads
    // segBarBoundaryLiveSec_ to render the preview tile.
    if (segBarBoundaryDragging_)
    {
        const auto bar = segBarArea();
        if (bar.isEmpty()) return;
        const double t = juce::jlimit (segBarBoundaryClampMin_,
                                        segBarBoundaryClampMax_,
                                        xToTime (static_cast<float> (e.x),
                                                 bar.getX(), bar.getWidth()));
        if (std::abs (t - segBarBoundaryLiveSec_) < 1e-6) return;
        segBarBoundaryLiveSec_ = t;
        repaint (bar);
        return;
    }

    // ADR-051 — Block Assembly segBar drag: extend the provisional mark.
    if (segBarDragging_)
    {
        const auto bar = segBarArea();
        if (bar.isEmpty()) return;

        // DEV-057 sesja 100d — escalate to mid-tile drag once the cursor
        // crosses the click-vs-drag threshold AND the gesture started
        // inside an existing block. Translates the entire block by
        // preserving duration; clamped against neighbour bounds. Once
        // active, we suppress the new-mark drag preview (different visual
        // language: existing tile visibly slides instead of a provisional
        // mark scrim).
        if (segBarDragStartedOnBlock_
            && segBarMidDragBlockIdx_ >= 0
            && ! segBarMidDragActive_)
        {
            const int dx = e.getPosition().x - segBarDragStartPos_.x;
            const int dy = e.getPosition().y - segBarDragStartPos_.y;
            if (dx * dx + dy * dy >= kDragThresholdPx * kDragThresholdPx)
            {
                segBarMidDragActive_ = true;
                setMouseCursor (juce::MouseCursor::DraggingHandCursor);
            }
        }

        if (segBarMidDragActive_)
        {
            const double tCursor = xToTime (static_cast<float> (e.x),
                                             bar.getX(), bar.getWidth());
            const double rawStart = tCursor - segBarMidDragOffsetSec_;
            const double clamped  = juce::jlimit (segBarMidDragClampMinStart_,
                                                   segBarMidDragClampMaxStart_,
                                                   rawStart);
            if (std::abs (clamped - segBarMidDragLiveStart_) >= 1e-6)
            {
                segBarMidDragLiveStart_ = clamped;
                repaint (bar);
            }
            return;
        }

        const double t = juce::jlimit (viewStartSec_,
                                        viewStartSec_ + viewDurationSec_,
                                        xToTime (static_cast<float> (e.x),
                                                 bar.getX(), bar.getWidth()));
        if (std::abs (t - segBarDragCurrentSec_) < 1e-6) return;
        segBarDragCurrentSec_ = t;
        repaint (bar);
        return;
    }

    if (! dragging_) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    // DEV-072 sesja 100b — auto-pan when cursor is inside the edge zone.
    // beginDragAutoRepeat keeps mouseDrag firing at ~25 Hz so the pan
    // continues without manual cursor movement. Pan rate is proportional
    // to (threshold - distance) / threshold so the closer to the edge
    // the faster the scroll. Only kicks in when the view is zoomed
    // (viewDurationSec_ < total content) and there's room to pan that
    // direction.
    {
        const int dxLeft  = e.x - canvas.getX();
        const int dxRight = canvas.getRight() - e.x;
        double panSecPerTick = 0.0;
        if (dxLeft < kEdgePanThresholdPx && viewStartSec_ > 0.0)
        {
            const double frac = double (kEdgePanThresholdPx - dxLeft)
                              / double (kEdgePanThresholdPx);
            panSecPerTick = -viewDurationSec_ * kEdgePanViewFrac
                            * (kAutoRepeatMs / 1000.0) * frac;
        }
        else if (dxRight < kEdgePanThresholdPx)
        {
            const double frac = double (kEdgePanThresholdPx - dxRight)
                              / double (kEdgePanThresholdPx);
            panSecPerTick = +viewDurationSec_ * kEdgePanViewFrac
                            * (kAutoRepeatMs / 1000.0) * frac;
        }
        if (panSecPerTick != 0.0)
        {
            // Use scrollH for clamp-aware viewStart bumping (same path
            // wheel-pan uses); stops at content edges so the user can
            // never scroll past the start / end.
            scrollH (panSecPerTick);
        }
    }

    const double t = juce::jlimit (viewStartSec_,
                                    viewStartSec_ + viewDurationSec_,
                                    xToTime (static_cast<float> (e.x),
                                             canvas.getX(), canvas.getWidth()));
    if (std::abs (t - dragCurrentSec_) < 1e-6) return;
    dragCurrentSec_ = t;

    // DEV-031 sesja 100b — selection-edge live commit. We re-emit the
    // selection_ snapshot every mouseDrag tick so the host (MainComponent)
    // can update PreviewController range mid-drag (existing
    // updateRange path already handles this) and the scrim repaints
    // continuously. Live audio scrub via PreviewController::seek when
    // preview is playing — host wires this through onSelectionChanged.
    if (selectionEdgeDragging_ && selection_.has_value())
    {
        if (selectionEdgeIsStart_)
        {
            selection_->startSec = std::min (t, dragStartSec_ - 0.05);
        }
        else
        {
            selection_->endSec   = std::max (t, dragStartSec_ + 0.05);
        }
        if (onSelectionChanged) onSelectionChanged (selection_);
    }

    repaint (canvas);
}

void WaveformView::mouseUp (const juce::MouseEvent& e)
{
    // DEV-029 (d) sesja 100b — segBar boundary-drag commit. Snap-to-beat
    // is respected per snapMode_ EXCEPT when the snapped value would land
    // on the neighbour block's boundary (= clampMin / clampMax). Sesja
    // 100b user smoke iter 2 verbatim *"w ustawieniach mam funckje snak
    // region to i tutaj tez to powinno byc respektowane"* — snap-to-beat
    // setting is binding. Iter 1 bug *"przeciaga automatycznie do
    // najblizszego bloku"* came from snap rounding to the neighbour's
    // beat (user blocks tend to land on beat grid → nearest beat to a
    // mid-gap drop is often the neighbour edge); the exclusion below
    // preserves a real gap when user dropped between blocks. Touching
    // the neighbour edge is still reachable: clampMin/clampMax docking
    // happens when liveSec already equals the clamp edge, in which
    // case we commit it verbatim.
    if (segBarBoundaryDragging_)
    {
        const int    idx     = segBarBoundaryBlockIdx_;
        const bool   isStart = segBarBoundaryIsStart_;
        double       newEdge = segBarBoundaryLiveSec_;

        constexpr double kTouchEps = 1e-9;
        const bool touchingNeighbour =
               std::abs (newEdge - segBarBoundaryClampMin_) < kTouchEps
            || std::abs (newEdge - segBarBoundaryClampMax_) < kTouchEps;

        if (! touchingNeighbour && snapMode_ != SnapMode::Off)
        {
            const double snapped = snapTimeToBeats (newEdge);
            // Only accept snap targets STRICTLY inside the gap. If the
            // nearest beat sits on the neighbour's boundary (or beyond),
            // keep the raw cursor position so user-released-mid-gap
            // doesn't auto-stick to the neighbour.
            if (snapped > segBarBoundaryClampMin_
                && snapped < segBarBoundaryClampMax_)
                newEdge = snapped;
        }

        const double newStart = isStart ? newEdge : segBarBoundaryFixedSec_;
        const double newEnd   = isStart ? segBarBoundaryFixedSec_ : newEdge;

        // Sesja 100b iter 2 — shared boundary commit. The second block's
        // far edge is segBarBoundarySharedFixedSec_; its near edge moves
        // to newEdge in lockstep. Emit a second callback so the host
        // mutates both userBlocks_ entries in the same tick.
        const int    sharedIdx     = segBarBoundarySharedBlockIdx_;
        const bool   sharedIsStart = segBarBoundarySharedIsStart_;
        const double sharedFixed   = segBarBoundarySharedFixedSec_;

        segBarBoundaryDragging_       = false;
        segBarBoundaryBlockIdx_       = -1;
        segBarBoundarySharedBlockIdx_ = -1;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint (segBarArea());

        if (idx >= 0
            && std::abs (newEnd - newStart) >= kMinBlockSec
            && onUserBlockBoundariesChanged)
        {
            onUserBlockBoundariesChanged (idx, newStart, newEnd);
        }
        if (sharedIdx >= 0 && onUserBlockBoundariesChanged)
        {
            const double sharedStart = sharedIsStart ? newEdge : sharedFixed;
            const double sharedEnd   = sharedIsStart ? sharedFixed : newEdge;
            if (std::abs (sharedEnd - sharedStart) >= kMinBlockSec)
                onUserBlockBoundariesChanged (sharedIdx, sharedStart, sharedEnd);
        }
        return;
    }

    // DEV-057 sesja 100d — mid-tile drag commit. Snap-to-beat respected
    // on the new startSec when snapMode_ != Off, EXCEPT when the snap
    // target lands on a neighbour boundary (mirror sesja-100b boundary-
    // drag fix where snap to neighbour-edge auto-merged blocks). Commit
    // via onUserBlockBoundariesChanged — host re-uses sesja-100b
    // persistence path. Suppresses click-vs-drag branches below since
    // mid-drag has its own commit semantic (translation, not click).
    if (segBarMidDragActive_)
    {
        const int    idx        = segBarMidDragBlockIdx_;
        const double duration   = segBarMidDragOriginalDur_;
        double       newStart   = segBarMidDragLiveStart_;

        constexpr double kTouchEps = 1e-9;
        if (snapMode_ != SnapMode::Off)
        {
            const double snapped = snapTimeToBeats (newStart);
            const double snappedEnd = snapped + duration;
            const bool touchingNeighbour =
                   std::abs (snapped     - segBarMidDragClampMinStart_) < kTouchEps
                || std::abs (snappedEnd  - (segBarMidDragClampMaxStart_ + duration)) < kTouchEps;
            if (! touchingNeighbour
                && snapped >= segBarMidDragClampMinStart_
                && snapped <= segBarMidDragClampMaxStart_)
                newStart = snapped;
        }
        const double newEnd = newStart + duration;

        segBarMidDragActive_       = false;
        segBarMidDragBlockIdx_     = -1;
        segBarDragging_            = false;
        segBarDragStartedOnBlock_  = false;
        segBarDragStartBlockIdx_   = -1;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint (segBarArea());

        if (idx >= 0 && duration >= kMinBlockSec && onUserBlockBoundariesChanged)
            onUserBlockBoundariesChanged (idx, newStart, newEnd);
        return;
    }

    // ADR-051 — Block Assembly segBar drag-up. Click vs drag distinguished
    // by pixel threshold; drag → mark new block (unless started on existing),
    // click → edit existing block when present.
    if (segBarDragging_)
    {
        segBarDragging_ = false;
        const int dx = e.getPosition().x - segBarDragStartPos_.x;
        const int dy = e.getPosition().y - segBarDragStartPos_.y;
        const bool wasDrag = (dx * dx + dy * dy) >= (kDragThresholdPx * kDragThresholdPx);

        if (wasDrag && ! segBarDragStartedOnBlock_)
        {
            double a = std::min (segBarDragStartSec_, segBarDragCurrentSec_);
            double b = std::max (segBarDragStartSec_, segBarDragCurrentSec_);
            if (snapMode_ != SnapMode::Off)
            {
                const double sa = snapTimeToBeats (a);
                const double sb = snapTimeToBeats (b);
                if (sa < sb) { a = sa; b = sb; }
            }
            // Drop micro-marks (< 1 beat ≈ 0.5 s) — likely accidental flicks.
            if (b - a >= 0.5 && onMarkBlock)
                onMarkBlock (a, b, e.getScreenPosition());
        }
        else if (! wasDrag && segBarDragStartedOnBlock_ && segBarDragStartBlockIdx_ >= 0)
        {
            if (onUserBlockClicked) onUserBlockClicked (segBarDragStartBlockIdx_);
        }

        segBarDragStartedOnBlock_ = false;
        segBarDragStartBlockIdx_  = -1;
        segBarMidDragBlockIdx_    = -1;
        repaint (segBarArea());
        return;
    }

    if (! dragging_) return;
    dragging_ = false;

    // DEV-031 sesja 100b — selection edge-drag commit. selection_ was
    // already mutated live during mouseDrag; here we apply optional
    // snap and emit a final onSelectionChanged so any debounced host
    // path (Region pipeline kick) sees the resting value. Skip the
    // click-vs-drag branch below — edge-drag with cursor stationary
    // is still a valid commit.
    if (selectionEdgeDragging_)
    {
        selectionEdgeDragging_ = false;
        setMouseCursor (juce::MouseCursor::NormalCursor);
        if (selection_.has_value() && snapMode_ != SnapMode::Off)
        {
            const double sa = snapTimeToBeats (selection_->startSec);
            const double sb = snapTimeToBeats (selection_->endSec);
            if (sa < sb)
            {
                selection_->startSec = sa;
                selection_->endSec   = sb;
            }
        }
        repaint (canvasArea());
        if (onSelectionChanged) onSelectionChanged (selection_);
        return;
    }

    // Click vs drag: 5 px pixel threshold matches EditView precedent.
    const int dx = e.getPosition().x - dragStartPos_.x;
    const int dy = e.getPosition().y - dragStartPos_.y;
    const bool wasDrag = (dx * dx + dy * dy) >= (kDragThresholdPx * kDragThresholdPx);

    if (! wasDrag)
    {
        // ADR-039 — click-outside-selection clears (user session 47 replaced
        // Esc-clear with click-outside-clear because Esc dispatch proved
        // unreliable). Click inside an active selection preserves it so
        // users can fine-tune seek position mid-playback. Click-to-seek
        // itself is Lua remix_ui.lua:264-275 / mockup plugin.jsx:185.
        const bool clickedOutside = selection_.has_value()
            && (dragStartSec_ < selection_->startSec
                || dragStartSec_ > selection_->endSec);
        if (clickedOutside)
        {
            selection_.reset();
            if (onSelectionChanged)
                onSelectionChanged (std::nullopt);
        }
        if (onSeek)
            onSeek (dragStartSec_);
        // Flash ring feedback at the click position (REABeat precedent). Acts
        // as a visual anchor the eye can follow through the rapid seek.
        triggerFlash (dragStartSec_);
        // Repaint in case a prior drag-in-progress selection was being
        // rendered; selection_ is the source of truth after mouseUp.
        repaint (canvasArea());
        return;
    }

    double a = std::min (dragStartSec_, dragCurrentSec_);
    double b = std::max (dragStartSec_, dragCurrentSec_);

    // Sesja 60 — apply snap to nearest beat/downbeat when enabled. Tries to
    // preserve a < b ordering even if both endpoints snap to the same beat
    // (rare with very narrow drags); when collapsed, snap is bypassed for
    // that pair so the selection is at least 1 beat-gap wide.
    if (snapMode_ != SnapMode::Off)
    {
        const double sa = snapTimeToBeats (a);
        const double sb = snapTimeToBeats (b);
        if (sa < sb)
        {
            a = sa;
            b = sb;
        }
        // Else: snap collapsed both ends to the same beat — keep raw values.
    }

    SelectionRange r { a, b };
    selection_ = r;
    repaint (canvasArea());
    if (onSelectionChanged)
        onSelectionChanged (selection_);
}

bool WaveformView::keyPressed (const juce::KeyPress& key)
{
    // Session 48 verified keyPressed fires reliably when the waveform has
    // focus (grabbed on mouseDown). Session-47 Esc-clear attempts failed
    // because Esc was dispatched to MainComponent / TransportBar first —
    // Esc was never handled at the WaveformView level the way Cmd+A / Home
    // / End are here. See ADR-039 Follow-up.
    if (peaks_ == nullptr || viewDurationSec_ <= 0.0)
        return false;

    const double itemDuration = peaks_->getTotalDurationSeconds();

    // Cmd+A — select all (entire item, not just the current viewport).
    if (key == juce::KeyPress ('a', juce::ModifierKeys::commandModifier, 0)
     || key == juce::KeyPress ('A', juce::ModifierKeys::commandModifier, 0))
    {
        if (itemDuration <= 0.0) return true;
        SelectionRange r { 0.0, itemDuration };
        selection_ = r;
        if (onSelectionChanged) onSelectionChanged (selection_);
        repaint (canvasArea());
        return true;
    }

    // Home / End — seek to start / end of the item. Matches EditView
    // input_handling convention (jump to extremes).
    if (key == juce::KeyPress::homeKey)
    {
        if (onSeek) onSeek (0.0);
        triggerFlash (0.0);
        return true;
    }
    if (key == juce::KeyPress::endKey)
    {
        if (itemDuration <= 0.0) return true;
        if (onSeek) onSeek (itemDuration);
        triggerFlash (itemDuration);
        return true;
    }

    return false;
}

void WaveformView::mouseWheelMove (const juce::MouseEvent& e,
                                   const juce::MouseWheelDetails& wheel)
{
    if (viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    // deltaX → horizontal pan (2-finger trackpad swipe, or Shift+wheel on
    // some systems). Positive deltaX = swipe right → view moves right.
    // EditView edit_view.cpp:1244-1252 divides by 60 to scale SWELL delta;
    // JUCE normalizes deltaX to [-1, 1] already so we just apply kPanPct.
    if (std::abs (wheel.deltaX) > 1.0e-4f)
    {
        const double sign = wheel.isReversed ? 1.0 : -1.0;
        scrollH (sign * wheel.deltaX * viewDurationSec_ * kPanPct);
    }

    // deltaY → zoom, origin under cursor. EditView waveform_view.cpp:677-695
    // pattern; pow(kZoomFactor, n) where n is wheel.deltaY (JUCE already
    // integrates the delta so small trackpad scrolls yield small factors).
    if (std::abs (wheel.deltaY) > 1.0e-4f)
    {
        const double factor = std::pow (kZoomFactor, (double) wheel.deltaY);
        const double centerTime = xToTime (static_cast<float> (e.x),
                                            canvas.getX(), canvas.getWidth());
        zoomHorizontal (factor, centerTime);
    }
}

void WaveformView::mouseMagnify (const juce::MouseEvent& e, float scaleFactor)
{
    if (viewDurationSec_ <= 0.0) return;

    const auto canvas = canvasArea();
    if (canvas.isEmpty()) return;

    // Pinch gesture — dampen so pinch feels smooth rather than jumpy.
    // EditView edit_view.cpp:1255-1269 uses 0.15 damping on WM_GESTURE.
    const double raw    = (double) scaleFactor;
    const double factor = 1.0 + (raw - 1.0) * kZoomDamping;
    const double centerTime = xToTime (static_cast<float> (e.x),
                                        canvas.getX(), canvas.getWidth());
    zoomHorizontal (factor, centerTime);
}

void WaveformView::zoomHorizontal (double factor, double centerTime)
{
    if (viewDurationSec_ <= 0.0 || factor <= 0.0) return;
    if (peaks_ == nullptr) return;

    const double itemDuration = std::max (peaks_->getTotalDurationSeconds(),
                                           kMinViewDurationSec);

    double newDuration = viewDurationSec_ / factor;
    newDuration = std::max (kMinViewDurationSec, std::min (itemDuration, newDuration));
    if (std::abs (newDuration - viewDurationSec_) < 1.0e-6) return;

    // EditView waveform_view.cpp:685-687 — keep centerTime under cursor.
    const double ratio = (centerTime - viewStartSec_) / viewDurationSec_;
    double newStart    = centerTime - ratio * newDuration;
    // Clamp to [0, itemDuration - newDuration].
    newStart = std::max (0.0, std::min (itemDuration - newDuration, newStart));

    viewStartSec_    = newStart;
    viewDurationSec_ = newDuration;
    cache_.valid     = false;
    repaint();
}

void WaveformView::scrollH (double deltaSec)
{
    if (viewDurationSec_ <= 0.0) return;
    if (peaks_ == nullptr) return;

    const double itemDuration = std::max (peaks_->getTotalDurationSeconds(),
                                           kMinViewDurationSec);
    const double maxStart = std::max (0.0, itemDuration - viewDurationSec_);

    double newStart = viewStartSec_ + deltaSec;
    newStart = std::max (0.0, std::min (maxStart, newStart));
    if (std::abs (newStart - viewStartSec_) < 1.0e-6) return;

    viewStartSec_ = newStart;
    cache_.valid  = false;
    repaint();
}

juce::Rectangle<int> WaveformView::hoverDirtyRect (int cursorX) const
{
    // Wide enough to cover a ~80 px pill centered on cursor plus the 1 px
    // line; tall enough to span canvas + segbar (ruler excluded).
    const auto canvas = canvasArea();
    const auto bar    = segBarArea();
    const int halfW = 48;
    const int x = cursorX - halfW;
    const int w = halfW * 2;
    const int y = canvas.getY();
    const int h = bar.getBottom() - y;
    return juce::Rectangle<int> (x, y, w, h)
              .getIntersection (getLocalBounds());
}

// ── Ruler helpers ───────────────────────────────────────────────────

double WaveformView::pickTickInterval (double pxPerSec)
{
    // EditView's "nice intervals" table — rendering.cpp:432. Target = 80 px
    // per tick (research-scout findings). Sub-second intervals kept in case
    // the Remix variant / zoom ever drives the view to ms-scale.
    static constexpr double kNice[] = {
        0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0
    };
    constexpr double kTargetPxPerTick = 80.0;

    for (double iv : kNice)
        if (iv * pxPerSec >= kTargetPxPerTick) return iv;

    return kNice[std::size (kNice) - 1];
}

juce::String WaveformView::formatSeconds (double s)
{
    // primitives.jsx:45-49 — rxFmt: clamp ≥ 0, round, "M:SS".
    const int total = std::max (0, static_cast<int> (std::round (s)));
    const int m = total / 60;
    const int sec = total % 60;
    return juce::String (m) + ":" + juce::String (sec).paddedLeft ('0', 2);
}

juce::String WaveformView::formatHoverSeconds (double s)
{
    // "M:SS.T" tenth-of-second precision for the hover pill.
    s = std::max (0.0, s);
    const int m = static_cast<int> (std::floor (s / 60.0));
    const double rem = s - m * 60.0;
    const int sec = static_cast<int> (std::floor (rem));
    const int tenth = std::min (9, static_cast<int> (std::floor ((rem - sec) * 10.0)));
    return juce::String (m) + ":" + juce::String (sec).paddedLeft ('0', 2)
           + "." + juce::String (tenth);
}

} // namespace reamix::ui
