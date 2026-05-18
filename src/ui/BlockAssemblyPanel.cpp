#include "BlockAssemblyPanel.h"
#include "KindDisplay.h"

#include <algorithm>
#include <limits>

namespace reamix::ui
{

using namespace reamix::theme;

namespace
{
    juce::String formatMSS (double sec)
    {
        const int totalSec = (int) std::floor (sec);
        const int m = totalSec / 60;
        const int s = totalSec % 60;
        return juce::String (m) + ":" + juce::String (s).paddedLeft ('0', 2);
    }
} // namespace

BlockAssemblyPanel::BlockAssemblyPanel()
{
    setOpaque (true);
    setMouseCursor (juce::MouseCursor::NormalCursor);
}

void BlockAssemblyPanel::setUserBlocks (std::vector<reamix::ui::UserBlock> b)
{
    const int prevRows = cardRowsNeeded();
    userBlocks_ = std::move (b);
    if (hoveredCardIdx_ >= (int) userBlocks_.size())
        hoveredCardIdx_ = -1;
    if (cardRowsNeeded() != prevRows && onPreferredHeightChanged)
        onPreferredHeightChanged();
    repaint();
}

void BlockAssemblyPanel::setQueue (std::vector<int> q)
{
    queue_ = std::move (q);
    if (hoveredQueueTileIdx_ >= (int) queue_.size()) hoveredQueueTileIdx_ = -1;
    if (hoveredSeamIdx_ >= (int) queue_.size() - 1)  hoveredSeamIdx_      = -1;
    repaint();
}

void BlockAssemblyPanel::setSeamQualities (std::vector<float> q)
{
    seamQualities_ = std::move (q);
    repaint (queueRowArea());
}

void BlockAssemblyPanel::setJunctionVariations (std::vector<int> v)
{
    junctionVariations_ = std::move (v);
    repaint (queueRowArea());
}

double BlockAssemblyPanel::queueTotalDuration() const
{
    double total = 0.0;
    for (int idx : queue_)
    {
        if (idx < 0 || idx >= (int) userBlocks_.size()) continue;
        const auto& b = userBlocks_[(std::size_t) idx];
        total += std::max (0.0, b.endSec - b.startSec);
    }
    return total;
}

juce::Rectangle<int> BlockAssemblyPanel::queueTileRect (int queuePos) const
{
    if (queuePos < 0 || queuePos >= (int) queue_.size()) return {};
    const auto row = queueRowArea();
    if (row.isEmpty()) return {};

    constexpr int kRulerH = 14;
    const auto tileBand = row.withTrimmedTop (kRulerH).reduced (1, 0);

    const double total = queueTotalDuration();
    if (total <= 0.0) return {};

    double accum = 0.0;
    for (int p = 0; p <= queuePos; ++p)
    {
        const int idx = queue_[(std::size_t) p];
        if (idx < 0 || idx >= (int) userBlocks_.size()) return {};
        const auto& b = userBlocks_[(std::size_t) idx];
        const double dur = std::max (0.0, b.endSec - b.startSec);
        if (p == queuePos)
        {
            const double frac0 = accum / total;
            const double frac1 = (accum + dur) / total;
            const int x0 = tileBand.getX() + (int) std::round (frac0 * tileBand.getWidth());
            const int x1 = tileBand.getX() + (int) std::round (frac1 * tileBand.getWidth());
            return { x0, tileBand.getY(),
                     std::max (4, x1 - x0), tileBand.getHeight() };
        }
        accum += dur;
    }
    return {};
}

int BlockAssemblyPanel::queueTileHitTest (juce::Point<int> pos) const
{
    for (int p = 0; p < (int) queue_.size(); ++p)
    {
        const auto r = queueTileRect (p);
        if (! r.isEmpty() && r.contains (pos)) return p;
    }
    return -1;
}

juce::Rectangle<int> BlockAssemblyPanel::seamPillRect (int junctionIdx) const
{
    if (junctionIdx < 0 || junctionIdx >= (int) queue_.size() - 1) return {};
    const auto leftTile  = queueTileRect (junctionIdx);
    if (leftTile.isEmpty()) return {};
    constexpr int kPillW = 30;
    constexpr int kPillH = 16;
    const int cx = leftTile.getRight();
    // Pill straddles the seam — half on each side, vertically centered on
    // the ruler/tile boundary line for visibility.
    const auto row = queueRowArea();
    const int cy = row.getY() + 14 - kPillH / 2; // 14h ruler boundary
    // Sesja 64 BUG-10 — clamp pill horizontally inside row area. Last pill
    // (between penultimate and last tile) was clipped off the right edge
    // because cx fell at the tileBand.right and pill extends kPillW/2 past.
    int x = cx - kPillW / 2;
    if (x + kPillW > row.getRight())  x = row.getRight()  - kPillW;
    if (x < row.getX())               x = row.getX();
    return { x, cy, kPillW, kPillH };
}

int BlockAssemblyPanel::seamPillHitTest (juce::Point<int> pos) const
{
    for (int j = 0; j < (int) queue_.size() - 1; ++j)
    {
        const auto r = seamPillRect (j);
        if (! r.isEmpty() && r.contains (pos)) return j;
    }
    return -1;
}

void BlockAssemblyPanel::setDirty (bool yes)
{
    if (dirty_ == yes) return;
    dirty_ = yes;
    repaint (bannerArea());
}

// ── Geometry ────────────────────────────────────────────────────────

int BlockAssemblyPanel::cardsPerRow() const
{
    const int avail = std::max (0, getWidth() - 2 * kHorizontalPadding);
    // Row capacity = floor((avail + gap) / (cardW + gap))
    const int n = (avail + kCardGap) / (kCardW + kCardGap);
    return std::max (1, n);
}

int BlockAssemblyPanel::cardRowsNeeded() const
{
    if (userBlocks_.empty()) return 1; // empty-state coach mark still needs 1 row
    const int per = cardsPerRow();
    const int rows = ((int) userBlocks_.size() + per - 1) / per;
    return std::min (kMaxPaletteRows, std::max (1, rows));
}

int BlockAssemblyPanel::paletteHeight() const
{
    const int rows = cardRowsNeeded();
    return rows * kCardRowH + std::max (0, rows - 1) * kCardRowGap;
}

int BlockAssemblyPanel::getPreferredHeight() const
{
    return kHeaderRowH + paletteHeight() + kQueueAndBannerH;
}

juce::Rectangle<int> BlockAssemblyPanel::paletteHeaderArea() const
{
    return { kHorizontalPadding, 0,
             getWidth() - 2 * kHorizontalPadding, kHeaderRowH };
}

juce::Rectangle<int> BlockAssemblyPanel::paletteRowArea() const
{
    return { kHorizontalPadding, kHeaderRowH,
             getWidth() - 2 * kHorizontalPadding, paletteHeight() };
}

juce::Rectangle<int> BlockAssemblyPanel::queueHeaderArea() const
{
    const int y = kHeaderRowH + paletteHeight() + kSectionGap;
    return { kHorizontalPadding, y,
             getWidth() - 2 * kHorizontalPadding, kHeaderRowH };
}

juce::Rectangle<int> BlockAssemblyPanel::queueRowArea() const
{
    const int y = kHeaderRowH + paletteHeight() + kSectionGap + kHeaderRowH;
    return { kHorizontalPadding, y,
             getWidth() - 2 * kHorizontalPadding, kQueueRowH };
}

juce::Rectangle<int> BlockAssemblyPanel::bannerArea() const
{
    const int y = kHeaderRowH + paletteHeight() + kSectionGap +
                  kHeaderRowH + kQueueRowH + kSectionGap;
    return { kHorizontalPadding, y,
             getWidth() - 2 * kHorizontalPadding, kBannerH };
}

juce::Rectangle<int> BlockAssemblyPanel::assembleBtnRect() const
{
    // Sesja 64 — keep button rect visible while busy (label swap renders
    // "Computing…"). Hide only when queue is too short to have a button.
    if (queue_.size() < 2) return {};
    const auto qHeader = queueHeaderArea();
    constexpr int kBtnW = 90;
    constexpr int kBtnH = 18;
    return { qHeader.getRight() - kBtnW,
             qHeader.getY() + (qHeader.getHeight() - kBtnH) / 2 - 1,
             kBtnW, kBtnH };
}

bool BlockAssemblyPanel::assembleBtnEnabled() const noexcept
{
    // Sesja 64 — busy state forces disabled so user can't trigger a second
    // arrangement compute while the first is in flight.
    if (assembleBusy_) return false;
    return queue_.size() >= 2;
}

void BlockAssemblyPanel::setAssembleBusy (bool busy, juce::String label)
{
    if (busy == assembleBusy_ && label == assembleBusyLabel_) return;
    assembleBusy_      = busy;
    assembleBusyLabel_ = std::move (label);
    if (assembleBusy_) hoveredAssembleBtn_ = false;
    repaint();
}

juce::Rectangle<int> BlockAssemblyPanel::cardRect (int blockIdx) const
{
    if (blockIdx < 0 || blockIdx >= (int) userBlocks_.size()) return {};
    const int per = cardsPerRow();
    const int row = blockIdx / per;
    const int col = blockIdx % per;
    if (row >= kMaxPaletteRows) return {}; // off the visible grid; rare with 4 rows
    const auto rowArea = paletteRowArea();
    const int x = rowArea.getX() + col * (kCardW + kCardGap);
    const int y = rowArea.getY() + row * (kCardRowH + kCardRowGap);
    if (x + kCardW > rowArea.getRight()) return {}; // safety; cardsPerRow should prevent
    return { x, y + 4, kCardW, kCardRowH - 8 };
}

int BlockAssemblyPanel::cardHitTest (juce::Point<int> pos) const
{
    for (int i = 0; i < (int) userBlocks_.size(); ++i)
    {
        const auto r = cardRect (i);
        if (! r.isEmpty() && r.contains (pos)) return i;
    }
    return -1;
}

std::optional<juce::Colour> BlockAssemblyPanel::compatColour (int /*blockIdx*/) const
{
    // Phase F wires real lookup against BlockCompatResult (sourced from
    // post-Analyze MainComponent state). For phase C the indicator reads
    // "neutral" — no compatibility hint.
    return std::nullopt;
}

// ── Paint ───────────────────────────────────────────────────────────

void BlockAssemblyPanel::paint (juce::Graphics& g)
{
    g.fillAll (Bg2);

    // Top + bottom 1 px Line dividers (matches DurationPanel visual chrome).
    g.setColour (Line);
    g.fillRect (0, 0, getWidth(), 1);
    g.fillRect (0, getHeight() - 1, getWidth(), 1);

    paintHeader (g, paletteHeaderArea(),
                 juce::String::fromUTF8 ("AVAILABLE BLOCKS"),
                 juce::String ((int) userBlocks_.size())
                     + (userBlocks_.size() == 1 ? " marked" : " marked"));

    paintPaletteRow (g, paletteRowArea());

    double totalDur = 0.0;
    for (int idx : queue_)
        if (idx >= 0 && idx < (int) userBlocks_.size())
            totalDur += userBlocks_[(std::size_t) idx].endSec
                       - userBlocks_[(std::size_t) idx].startSec;

    // Sesja-61 close UX fix — reserve right-side space in queue header for
    // the dedicated "Assemble" button so meta text doesn't collide with it.
    const auto qHeaderArea = queueHeaderArea();
    const auto assembleBtn = assembleBtnRect();
    const int  metaReserve = assembleBtn.isEmpty() ? 0 : (assembleBtn.getWidth() + 6);
    const auto qHeaderForMeta = qHeaderArea.withTrimmedRight (metaReserve);

    paintHeader (g, qHeaderForMeta,
                 juce::String::fromUTF8 ("ARRANGEMENT TIMELINE"),
                 queue_.empty()
                     ? juce::String::fromUTF8 ("\xE2\x80\x94")  // em-dash
                     : juce::String ((int) queue_.size())
                       + (queue_.size() == 1 ? " block \xC2\xB7 " : " blocks \xC2\xB7 ")
                       + formatMSS (totalDur));

    // Sesja-61 close UX fix — Assemble button (premium pill, Accent-tinted
    // when ready, brighter on hover, pulses subtly when dirty so user knows
    // arrangement state is stale). Banner below stays purely informational.
    if (! assembleBtn.isEmpty())
    {
        const bool hovered = (hoveredAssembleBtn_ && ! assembleBusy_);
        // Sesja 64 — busy state visually dims the button + swaps label so user
        // sees that arrangement compute is in flight.
        juce::Colour fill;
        juce::Colour edge;
        juce::Colour labelCol;
        if (assembleBusy_)
        {
            fill     = Bg4.withAlpha (0.55f);
            edge     = AccentLo;
            labelCol = Accent;
        }
        else if (dirty_)
        {
            fill     = Accent.withAlpha (hovered ? 0.85f : 0.65f);
            edge     = Accent;
            labelCol = juce::Colours::white.withAlpha (0.96f);
        }
        else
        {
            fill     = Bg4.withAlpha (hovered ? 1.00f : 0.85f);
            edge     = hovered ? LineStrong : Line;
            labelCol = Fg2;
        }

        g.setColour (fill);
        g.fillRoundedRectangle (assembleBtn.toFloat(), 3.0f);
        g.setColour (edge);
        g.drawRoundedRectangle (assembleBtn.toFloat().reduced (0.5f), 3.0f, 1.0f);
        g.setColour (labelCol);
        g.setFont (uiFont (fs::Sm, 700).withExtraKerningFactor (0.06f));
        const juce::String btnLabel = (assembleBusy_ && assembleBusyLabel_.isNotEmpty())
            ? assembleBusyLabel_
            : juce::String ("Assemble");
        g.drawText (btnLabel, assembleBtn,
                    juce::Justification::centred, false);
    }

    paintQueueRow (g, queueRowArea());

    if (dirty_) paintBanner (g, bannerArea());

    // DEV-077 sesja 100d — drag overlay LAST so ghost + insertion lines
    // sit above every panel section (was: ghost in paintPaletteRow got
    // occluded by paintQueueRow when cursor crossed into queue area).
    paintDragGhost (g);
}

void BlockAssemblyPanel::paintHeader (juce::Graphics& g,
                                       juce::Rectangle<int> area,
                                       const juce::String& title,
                                       const juce::String& meta)
{
    g.setColour (Fg3);
    g.setFont (monoFont (fs::Xs, 600).withExtraKerningFactor (0.08f));
    g.drawText (title, area, juce::Justification::centredLeft, false);

    if (meta.isNotEmpty())
    {
        g.setColour (Fg4);
        g.setFont (monoFont (fs::Xs, 500));
        g.drawText (meta, area, juce::Justification::centredRight, false);
    }
}

void BlockAssemblyPanel::paintPaletteRow (juce::Graphics& g,
                                           juce::Rectangle<int> area)
{
    if (userBlocks_.empty())
    {
        // Empty-state hint (full coach-mark visual lands phase I).
        g.setColour (Bg3);
        g.fillRect (area);
        g.setColour (Fg4);
        g.setFont (uiFont (fs::Sm, 400));
        g.drawText (juce::String::fromUTF8 (
            "Drag on the section bar below to mark your first block"),
            area, juce::Justification::centred, false);
        return;
    }

    for (int i = 0; i < (int) userBlocks_.size(); ++i)
    {
        const auto rect = cardRect (i);
        if (rect.isEmpty()) break; // off-screen, phase F adds horizontal scroll

        // DEV-058 (c) sesja 100d — skip drawing the dragged tile in its
        // home slot during a drag-reorder (it's drawn at cursor position
        // below as a ghost). Other tiles render normally so the user
        // perceives the gap left behind by the dragged tile.
        if (dragPaletteActive_ && i == dragPaletteBlockIdx_) continue;

        const auto& b = userBlocks_[(std::size_t) i];
        // ADR-092 sesja 100c — registry-aware kind display.
        const KindDisplay disp = (customKindRegistry_ != nullptr)
            ? kindDisplay (b, *customKindRegistry_)
            : KindDisplay { builtinKindLabel (b.kind), segmentColour (b.kind) };
        const auto kindCol = disp.color;
        const bool isHovered  = (i == hoveredCardIdx_);
        const bool isSelected = paletteSelected_.find (i) != paletteSelected_.end();

        // Body — kind colour at 30 % (idle) / 50 % (hover) / 60 % (selected).
        const float bodyAlpha = isSelected ? 0.60f : (isHovered ? 0.50f : 0.30f);
        g.setColour (kindCol.withAlpha (bodyAlpha));
        g.fillRoundedRectangle (rect.toFloat(), 3.0f);

        // Border — kind colour or compat-color hint when present. DEV-058 (b):
        // selected tiles get a 2 px Accent border (overrides compat hint).
        if (isSelected)
        {
            g.setColour (Accent);
            g.drawRoundedRectangle (rect.toFloat().reduced (1.0f), 3.0f, 2.0f);
        }
        else
        {
            const auto compat = compatColour (i);
            const auto borderCol = compat.has_value() ? *compat : kindCol.withAlpha (0.85f);
            g.setColour (borderCol);
            g.drawRoundedRectangle (rect.toFloat().reduced (0.5f), 3.0f, 1.0f);
        }

        // Top accent stripe (kind brighter; matches segBar tile painting).
        g.setColour (kindCol.brighter (0.4f));
        g.fillRect (rect.getX(), rect.getY(), rect.getWidth(), 1);

        // Kind label (uppercase, mono Xs) — disp.name already honors
        // labelOverride (kindDisplay applies it as the final step).
        const juce::String label = disp.name;

        g.setColour (juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.92f));
        g.setFont (monoFont (fs::Xs, 700).withExtraKerningFactor (0.10f));
        g.drawText (label.toUpperCase(),
                    rect.reduced (8, 6).removeFromTop (rect.getHeight() / 2),
                    juce::Justification::topLeft, false);

        // Duration (mono Xs)
        const double dur = std::max (0.0, b.endSec - b.startSec);
        g.setColour (Fg2);
        g.setFont (monoFont (fs::Xs, 500));
        g.drawText (formatMSS (dur),
                    rect.reduced (8, 6).removeFromBottom (rect.getHeight() / 2 - 2),
                    juce::Justification::bottomLeft, false);

        // DEV-078 (NEW sesja 100d) — usage badge in top-right corner.
        // Count of queue entries referencing this block. Zero use → no
        // badge. ≥1 → small Accent circle + white count digit so the
        // user can see at a glance which blocks are already in the
        // arrangement (and how many times they're used). Pattern: DAW
        // clip reference indicators (Ableton, Reason).
        const int usedCount = (int) std::count (queue_.begin(), queue_.end(), i);
        if (usedCount > 0)
        {
            constexpr int kBadgeR  = 14;
            constexpr int kBadgePad = 4;
            const juce::Rectangle<int> badge (
                rect.getRight() - kBadgeR - kBadgePad,
                rect.getY()     + kBadgePad,
                kBadgeR, kBadgeR);
            g.setColour (Accent);
            g.fillEllipse (badge.toFloat());
            g.setColour (juce::Colours::white);
            g.setFont (monoFont (fs::Xs, 800));
            g.drawText (juce::String (usedCount), badge,
                        juce::Justification::centred, false);
        }
    }

    // DEV-058 (c) + DEV-077 sesja 100d — ghost tile + insertion lines moved
    // to paintDragGhost(), called LAST in paint(), so the overlay always
    // sits above queue tiles + headers + banner.
}

void BlockAssemblyPanel::paintQueueRow (juce::Graphics& g,
                                         juce::Rectangle<int> area)
{
    // Background panel
    g.setColour (Bg2);
    g.fillRect (area);
    g.setColour (Line);
    g.drawRect (area, 1);

    if (queue_.empty())
    {
        g.setColour (Fg4);
        g.setFont (uiFont (fs::Sm, 400));
        g.drawText (juce::String::fromUTF8 (
            "Empty queue \xC2\xB7 click a block above to begin an arrangement"),
            area, juce::Justification::centred, false);
        return;
    }

    constexpr int kRulerH = 14;
    const auto rulerBand = area.removeFromTop (kRulerH);
    // tileBand layout is recomputed inside queueTileRect() per p; no local ref needed.

    // Ruler — Bg1 strip with vertical tick lines + M:SS labels every 10/20/30s.
    g.setColour (Bg1);
    g.fillRect (rulerBand);
    g.setColour (Line);
    g.fillRect (rulerBand.getX(), rulerBand.getBottom() - 1,
                rulerBand.getWidth(), 1);

    const double total = queueTotalDuration();
    const double tickInterval = total < 60.0 ? 10.0 : total < 180.0 ? 20.0 : 30.0;
    const int rulerLeft  = rulerBand.getX() + 2;
    const int rulerRight = rulerBand.getRight() - 2;
    const int rulerWidth = std::max (4, rulerRight - rulerLeft);

    g.setColour (Fg3);
    g.setFont (monoFont (fs::Xs, 500));
    if (total > 0.0)
    {
        for (double t = 0.0; t <= total; t += tickInterval)
        {
            const int x = rulerLeft + (int) std::round ((t / total) * rulerWidth);
            g.setColour (LineStrong);
            g.drawVerticalLine (x, (float) rulerBand.getY(), (float) rulerBand.getBottom());
            g.setColour (Fg3);
            g.drawText (formatMSS (t),
                        x + 3, rulerBand.getY(),
                        40, rulerBand.getHeight(),
                        juce::Justification::centredLeft, false);
        }
    }

    // Tiles — proportional widths, kind colors, labels.
    for (int p = 0; p < (int) queue_.size(); ++p)
    {
        const int idx = queue_[(std::size_t) p];
        if (idx < 0 || idx >= (int) userBlocks_.size()) continue;
        const auto& b = userBlocks_[(std::size_t) idx];
        const auto rect = queueTileRect (p);
        if (rect.isEmpty()) continue;

        // ADR-092 sesja 100c — registry-aware kind display.
        const KindDisplay disp = (customKindRegistry_ != nullptr)
            ? kindDisplay (b, *customKindRegistry_)
            : KindDisplay { builtinKindLabel (b.kind), segmentColour (b.kind) };
        const auto kindCol = disp.color;
        const bool isHovered = (p == hoveredQueueTileIdx_);

        g.setColour (kindCol.withAlpha (isHovered ? 0.85f : 0.70f));
        g.fillRect (rect);

        // Top accent stripe (matches segBar tile aesthetic).
        g.setColour (kindCol.brighter (0.5f));
        g.fillRect (rect.getX(), rect.getY(), rect.getWidth(), 2);

        // Right divider — except last tile.
        if (p + 1 < (int) queue_.size())
        {
            g.setColour (juce::Colour::fromFloatRGBA (0.0f, 0.0f, 0.0f, 0.45f));
            g.fillRect (rect.getRight() - 1, rect.getY(), 1, rect.getHeight());
        }

        // Label — kind + duration. Skip if tile too narrow.
        if (rect.getWidth() > 36)
        {
            g.setColour (juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.95f));
            g.setFont (monoFont (fs::Xs, 700).withExtraKerningFactor (0.10f));
            const juce::String label = disp.name;
            g.drawText (label.toUpperCase(),
                        rect.reduced (4, 4).removeFromTop (rect.getHeight() / 2),
                        juce::Justification::topLeft, true);

            const double dur = std::max (0.0, b.endSec - b.startSec);
            g.setColour (juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.8f));
            g.setFont (monoFont (fs::Xs, 500));
            g.drawText (formatMSS (dur),
                        rect.reduced (4, 4).removeFromBottom (rect.getHeight() / 2 - 2),
                        juce::Justification::bottomLeft, true);
        }
    }

    // Seam pills — over the ruler/tile boundary, between adjacent tiles.
    for (int j = 0; j < (int) queue_.size() - 1; ++j)
    {
        const auto pill = seamPillRect (j);
        if (pill.isEmpty()) continue;

        const float quality = (j < (int) seamQualities_.size())
            ? seamQualities_[(std::size_t) j] : -1.0f;

        // Quality color: green > 0.65, yellow >= 0.4, red < 0.4. Neutral
        // gray when quality is unknown (pre-analysis).
        juce::Colour pillColour;
        if (quality < 0.0f)        pillColour = Fg3;
        else if (quality > 0.65f)  pillColour = juce::Colour (0xFF44CC44);
        else if (quality >= 0.40f) pillColour = juce::Colour (0xFFCCCC44);
        else                       pillColour = juce::Colour (0xFFCC4444);

        const bool isHovered = (j == hoveredSeamIdx_);
        const auto bgColour = isHovered ? pillColour.brighter (0.30f) : pillColour;

        g.setColour (Bg1);
        g.fillRoundedRectangle (pill.toFloat().expanded (1.0f, 1.0f), 4.0f);
        g.setColour (bgColour);
        g.fillRoundedRectangle (pill.toFloat(), 3.0f);

        // Quality % label or "—" when unknown.
        const juce::String label = (quality < 0.0f)
            ? juce::String::fromUTF8 ("\xE2\x80\x94")
            : juce::String ((int) std::round (quality * 100.0f)) + "%";

        g.setColour (juce::Colours::white.withAlpha (quality < 0.0f ? 0.6f : 0.95f));
        g.setFont (monoFont (fs::Xs, 700));
        g.drawText (label, pill, juce::Justification::centred, false);

        // ² re-roll badge (Phase G — junctionVariations_[j] > 0)
        if (j < (int) junctionVariations_.size() && junctionVariations_[(std::size_t) j] > 0)
        {
            const int badgeR = 6;
            const int bx = pill.getRight() - badgeR / 2;
            const int by = pill.getY() - badgeR / 2;
            g.setColour (Accent);
            g.fillEllipse ((float) bx, (float) by, (float) badgeR, (float) badgeR);
            g.setColour (juce::Colours::white);
            g.setFont (monoFont (8.0f, 800));
            g.drawText (juce::String (junctionVariations_[(std::size_t) j] + 1),
                        bx - 2, by - 2, badgeR + 4, badgeR + 4,
                        juce::Justification::centred, false);
        }
    }

    // DEV-030 (sesja 100b) — drag-drop reorder overlay. Ghost tile drawn
    // semi-transparent at the cursor X (centred under the cursor),
    // insertion line drawn at the gap where the tile would land if
    // released. Only rendered when the gesture has crossed the click-vs-
    // drag threshold (dragQueueActive_ true).
    if (dragQueueActive_
        && dragQueueIdx_ >= 0
        && dragQueueIdx_ < (int) queue_.size())
    {
        const int idx = queue_[(std::size_t) dragQueueIdx_];
        if (idx >= 0 && idx < (int) userBlocks_.size())
        {
            const auto& b = userBlocks_[(std::size_t) idx];
            const auto baseRect = queueTileRect (dragQueueIdx_);
            if (! baseRect.isEmpty())
            {
                // Ghost tile follows cursor X; vertical position pinned to
                // the queue row.
                const int ghostW = baseRect.getWidth();
                const int ghostX = juce::jlimit (area.getX(),
                                                  area.getRight() - ghostW,
                                                  dragQueueCurrentPos_.x - ghostW / 2);
                const juce::Rectangle<int> ghost { ghostX, baseRect.getY(),
                                                    ghostW, baseRect.getHeight() };

                // ADR-092 sesja 100c — registry-aware ghost color so it
                // matches the real tile being dragged.
                const KindDisplay ghostDisp = (customKindRegistry_ != nullptr)
                    ? kindDisplay (b, *customKindRegistry_)
                    : KindDisplay { builtinKindLabel (b.kind), segmentColour (b.kind) };
                const auto kindCol = ghostDisp.color;
                g.setColour (kindCol.withAlpha (0.55f));
                g.fillRect (ghost);
                g.setColour (kindCol.brighter (0.5f));
                g.fillRect (ghost.getX(), ghost.getY(), ghost.getWidth(), 2);

                // Insertion line — 2 px accent at the gap edge. Snap to
                // the right edge of the previous tile (or left edge of
                // next) so the line aligns with existing tile boundaries.
                if (dragQueueInsertionPos_ >= 0)
                {
                    int lineX = area.getX();
                    if (dragQueueInsertionPos_ == 0)
                    {
                        const auto firstR = queueTileRect (0);
                        if (! firstR.isEmpty()) lineX = firstR.getX();
                    }
                    else if (dragQueueInsertionPos_ >= (int) queue_.size())
                    {
                        const auto lastR = queueTileRect ((int) queue_.size() - 1);
                        if (! lastR.isEmpty()) lineX = lastR.getRight();
                    }
                    else
                    {
                        const auto leftR = queueTileRect (dragQueueInsertionPos_ - 1);
                        if (! leftR.isEmpty()) lineX = leftR.getRight();
                    }
                    g.setColour (Accent);
                    g.fillRect (lineX - 1, baseRect.getY() - 2, 2,
                                 baseRect.getHeight() + 4);
                }
            }
        }
    }

    // DEV-077 sesja 100d — queue-side insertion line moved to
    // paintDragGhost (always-on-top final pass).
}

// DEV-058 (c) + DEV-077 sesja 100d — drag overlay rendered LAST in paint().
// Renders the insertion line (paleta-side OR queue-side based on cursor
// position) and the ghost tile at the cursor. Always-on-top so the queue
// tiles / queue header / banner do NOT occlude the drag visual feedback
// (sesja 100d iter 1 user smoke: "blok chowa sie pod arrange przed
// puszczeniem" — was rendering the ghost in paintPaletteRow at the
// beginning of paint(), so subsequent paintQueueRow drew over it).
void BlockAssemblyPanel::paintDragGhost (juce::Graphics& g)
{
    if (! dragPaletteActive_) return;
    if (dragPaletteBlockIdx_ < 0
        || dragPaletteBlockIdx_ >= (int) userBlocks_.size()) return;

    const auto& b = userBlocks_[(std::size_t) dragPaletteBlockIdx_];
    const KindDisplay disp = (customKindRegistry_ != nullptr)
        ? kindDisplay (b, *customKindRegistry_)
        : KindDisplay { builtinKindLabel (b.kind), segmentColour (b.kind) };

    // ── Insertion line ───────────────────────────────────────────────
    if (dragPaletteCrossingToQueue_
        && dragPaletteQueueInsertPos_ >= 0)
    {
        // Queue-side gap — Accent line at the target queue position.
        const auto qRow = queueRowArea();
        int lineX = qRow.getX() + 4;
        if (queue_.empty())
        {
            lineX = qRow.getX() + 8;
        }
        else if (dragPaletteQueueInsertPos_ <= 0)
        {
            const auto r0 = queueTileRect (0);
            if (! r0.isEmpty()) lineX = r0.getX() - 1;
        }
        else if (dragPaletteQueueInsertPos_ >= (int) queue_.size())
        {
            const auto rL = queueTileRect ((int) queue_.size() - 1);
            if (! rL.isEmpty()) lineX = rL.getRight() + 1;
        }
        else
        {
            const auto rL = queueTileRect (dragPaletteQueueInsertPos_ - 1);
            if (! rL.isEmpty()) lineX = rL.getRight();
        }

        int lineY = qRow.getY() + 14 + 2;
        int lineH = qRow.getHeight() - 14 - 4;
        if (! queue_.empty())
        {
            const auto rRef = queueTileRect (
                std::min (dragPaletteQueueInsertPos_, (int) queue_.size() - 1));
            if (! rRef.isEmpty())
            {
                lineY = rRef.getY() - 2;
                lineH = rRef.getHeight() + 4;
            }
        }
        g.setColour (Accent);
        g.fillRect (lineX - 1, lineY, 2, lineH);
    }
    else if (dragPaletteInsertTargetIdx_ >= 0
             && dragPaletteInsertTargetIdx_ <= (int) userBlocks_.size())
    {
        // Paleta-side gap — Accent line between cards.
        int lineX = 0, lineY = 0, lineH = kCardRowH;
        if (dragPaletteInsertTargetIdx_ == (int) userBlocks_.size())
        {
            const int lastIdx = (int) userBlocks_.size() - 1;
            const auto lr = cardRect (lastIdx);
            if (! lr.isEmpty())
            {
                lineX = lr.getRight() + 2;
                lineY = lr.getY();
                lineH = lr.getHeight();
            }
        }
        else
        {
            const auto tr = cardRect (dragPaletteInsertTargetIdx_);
            if (! tr.isEmpty())
            {
                lineX = tr.getX() - 3;
                lineY = tr.getY();
                lineH = tr.getHeight();
            }
        }
        g.setColour (Accent);
        g.fillRect (lineX, lineY, 2, lineH);
    }

    // ── Ghost tile (paleta-card-shape, follows cursor) ───────────────
    const int ghostX = dragPaletteCurrentPos_.x - kCardW / 2;
    const int ghostY = dragPaletteCurrentPos_.y - kCardRowH / 2;
    const juce::Rectangle<int> ghostRect (ghostX, ghostY, kCardW, kCardRowH);

    g.setColour (disp.color.withAlpha (0.55f));
    g.fillRoundedRectangle (ghostRect.toFloat(), 3.0f);
    g.setColour (Accent);
    g.drawRoundedRectangle (ghostRect.toFloat().reduced (1.0f), 3.0f, 2.0f);

    g.setColour (juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, 0.92f));
    g.setFont (monoFont (fs::Xs, 700).withExtraKerningFactor (0.10f));
    g.drawText (disp.name.toUpperCase(),
                ghostRect.reduced (8, 6).removeFromTop (kCardRowH / 2),
                juce::Justification::topLeft, false);
}

void BlockAssemblyPanel::paintBanner (juce::Graphics& g,
                                       juce::Rectangle<int> area)
{
    // Sesja-61 close UX fix — banner is now purely informational; the action
    // verb lives in the dedicated "Assemble" button in queue header. Banner
    // explains *why* the button glows (dirty state).
    g.setColour (Accent.withAlpha (0.18f));
    g.fillRect (area);

    g.setColour (Accent);
    g.setFont (monoFont (fs::Xs, 600).withExtraKerningFactor (0.08f));
    g.drawText (juce::String::fromUTF8 (
        "EDITS PENDING \xC2\xB7 SEAM SCORES STALE"),
        area.reduced (8, 0), juce::Justification::centredLeft, false);
}

void BlockAssemblyPanel::resized()
{
    // No child components yet — geometry is paint-time.
}

void BlockAssemblyPanel::mouseMove (const juce::MouseEvent& e)
{
    const auto pos = e.getPosition();
    const int  cardIdx     = cardHitTest      (pos);
    const int  tileIdx     = queueTileHitTest (pos);
    const int  seamIdx     = seamPillHitTest  (pos);
    const bool overAssemble =
        assembleBtnEnabled() && assembleBtnRect().contains (pos);

    bool dirty = false;
    if (cardIdx != hoveredCardIdx_)         { hoveredCardIdx_      = cardIdx;       dirty = true; }
    if (tileIdx != hoveredQueueTileIdx_)    { hoveredQueueTileIdx_ = tileIdx;       dirty = true; }
    if (seamIdx != hoveredSeamIdx_)         { hoveredSeamIdx_      = seamIdx;       dirty = true; }
    if (overAssemble != hoveredAssembleBtn_) { hoveredAssembleBtn_ = overAssemble;  dirty = true; }

    setMouseCursor ((cardIdx >= 0 || tileIdx >= 0 || seamIdx >= 0 || overAssemble)
                        ? juce::MouseCursor::PointingHandCursor
                        : juce::MouseCursor::NormalCursor);
    if (dirty) repaint();
}

void BlockAssemblyPanel::mouseExit (const juce::MouseEvent&)
{
    bool dirty = false;
    if (hoveredCardIdx_      != -1)   { hoveredCardIdx_      = -1;    dirty = true; }
    if (hoveredQueueTileIdx_ != -1)   { hoveredQueueTileIdx_ = -1;    dirty = true; }
    if (hoveredSeamIdx_      != -1)   { hoveredSeamIdx_      = -1;    dirty = true; }
    if (hoveredAssembleBtn_)          { hoveredAssembleBtn_  = false; dirty = true; }
    if (dirty)
    {
        setMouseCursor (juce::MouseCursor::NormalCursor);
        repaint();
    }
}

void BlockAssemblyPanel::mouseDown (const juce::MouseEvent& e)
{
    const auto pos = e.getPosition();
    const auto screenPos = e.getScreenPosition();

    // Sesja-61 close UX fix — Assemble button is the action verb. Banner
    // below is purely informational (no longer clickable).
    if (assembleBtnEnabled() && assembleBtnRect().contains (pos))
    {
        if (onAssembleClicked) onAssembleClicked();
        return;
    }

    // Seam pill — left-click auditions the splice, right-click opens menu.
    const int seamIdx = seamPillHitTest (pos);
    if (seamIdx >= 0)
    {
        if (e.mods.isPopupMenu())
        {
            if (onSeamContextMenu) onSeamContextMenu (seamIdx, screenPos);
        }
        else
        {
            if (onSeamAudition) onSeamAudition (seamIdx);
        }
        return;
    }

    // Queue tile — right-click opens menu; left-click is deferred to
    // mouseUp so we can distinguish click (audition) from drag-drop
    // (reorder, DEV-030 sesja 100b). dragQueueIdx_ stays valid through
    // the gesture; mouseUp decides which callback fires based on the
    // cursor travel distance.
    const int tileIdx = queueTileHitTest (pos);
    if (tileIdx >= 0)
    {
        if (e.mods.isPopupMenu())
        {
            if (onQueueTileContextMenu) onQueueTileContextMenu (tileIdx, screenPos);
            return;
        }
        dragQueueIdx_         = tileIdx;
        dragQueueStartPos_    = pos;
        dragQueueCurrentPos_  = pos;
        dragQueueInsertionPos_= -1;
        dragQueueActive_      = false;
        return;
    }

    // Paleta card.
    const int cardIdx = cardHitTest (pos);
    if (cardIdx < 0)
    {
        // DEV-058 (b) sesja 100d — click on empty paleta area clears
        // multi-selection. Esc handler does the same; click-elsewhere
        // matches Finder / VS Code conventions.
        if (! paletteSelected_.empty())
        {
            paletteSelected_.clear();
            paletteSelectionAnchor_ = -1;
            if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
            repaint();
        }
        return;
    }

    if (e.mods.isPopupMenu())
    {
        // Right-click. If the right-clicked card is NOT in the current
        // multi-select set, replace selection with {cardIdx} so the menu
        // acts on the explicitly-clicked block (matches macOS Finder).
        // Existing multi-select preserved when right-click hits a member.
        if (paletteSelected_.find (cardIdx) == paletteSelected_.end())
        {
            paletteSelected_ = { cardIdx };
            paletteSelectionAnchor_ = cardIdx;
            if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
            repaint();
        }
        if (onCardContextMenu) onCardContextMenu (cardIdx, screenPos);
        return;
    }

    // DEV-058 (b) sesja 100d — selection modifiers.
    //   Cmd / Ctrl click → toggle in selection (no queue append, no drag).
    //   Shift click → range from anchor (no queue append, no drag).
    //   Plain click  → defer to mouseUp; mouseUp decides click (append +
    //                  single-select replace) vs drag (reorder).
    if (e.mods.isCommandDown())
    {
        if (paletteSelected_.find (cardIdx) != paletteSelected_.end())
            paletteSelected_.erase (cardIdx);
        else
            paletteSelected_.insert (cardIdx);
        paletteSelectionAnchor_ = cardIdx;
        if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
        dragPaletteClickModified_ = true;
        repaint();
        return;
    }
    if (e.mods.isShiftDown() && paletteSelectionAnchor_ >= 0
        && paletteSelectionAnchor_ < (int) userBlocks_.size())
    {
        const int lo = std::min (paletteSelectionAnchor_, cardIdx);
        const int hi = std::max (paletteSelectionAnchor_, cardIdx);
        for (int i = lo; i <= hi; ++i) paletteSelected_.insert (i);
        if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
        dragPaletteClickModified_ = true;
        repaint();
        return;
    }

    // Plain click — defer click-vs-drag to mouseUp (DEV-058 (c) drag-
    // reorder support). Capture mouseDown state for both code paths.
    dragPaletteBlockIdx_         = cardIdx;
    dragPaletteStartPos_         = pos;
    dragPaletteCurrentPos_       = pos;
    dragPaletteInsertTargetIdx_  = -1;
    dragPaletteActive_           = false;
    dragPaletteClickModified_    = false;
}

// DEV-030 (sesja 100b) — queue drag-drop reorder. mouseDown captures the
// dragged tile + start position; this method tracks the cursor and
// recomputes the insertion gap. Once the cursor travels past the click-
// vs-drag threshold the gesture is committed as a drag (dragQueueActive_
// = true) which causes paint to render the ghost tile + insertion line.
void BlockAssemblyPanel::mouseDrag (const juce::MouseEvent& e)
{
    // DEV-058 (c) + DEV-077 sesja 100d — paleta drag tracking. Cursor
    // path decides what mouseUp does:
    //   - inside paletteRowArea() → onPaletteReorder (re-arrange paleta)
    //   - inside queueRowArea()   → onPaletteToQueueInsert (cross-section)
    //   - elsewhere              → no commit on mouseUp.
    if (dragPaletteBlockIdx_ >= 0 && ! dragPaletteClickModified_)
    {
        dragPaletteCurrentPos_ = e.getPosition();

        if (! dragPaletteActive_)
        {
            const int dx = std::abs (dragPaletteCurrentPos_.x - dragPaletteStartPos_.x);
            const int dy = std::abs (dragPaletteCurrentPos_.y - dragPaletteStartPos_.y);
            if (dx + dy < kPaletteDragThresholdPx) return;
            dragPaletteActive_ = true;
            // Clear multi-selection when drag starts — v1 does single-tile
            // reorder only. Selection visual could mislead user into
            // thinking the drag will move all selected.
            if (paletteSelected_.size() > 1)
            {
                paletteSelected_ = { dragPaletteBlockIdx_ };
                paletteSelectionAnchor_ = dragPaletteBlockIdx_;
                if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
            }
            // Hide native cursor — ghost tile follows pointer instead.
            setMouseCursor (juce::MouseCursor::DraggingHandCursor);
        }

        const auto qArea = queueRowArea();
        const bool nowCrossing = qArea.contains (dragPaletteCurrentPos_);
        if (nowCrossing != dragPaletteCrossingToQueue_)
        {
            dragPaletteCrossingToQueue_ = nowCrossing;
        }

        if (dragPaletteCrossingToQueue_)
        {
            // Compute queue insertion gap from cursor X (existing helper
            // walks queue tile midpoints). Empty queue → gap 0.
            const int newQueuePos = computeInsertionPos (dragPaletteCurrentPos_.x);
            if (newQueuePos != dragPaletteQueueInsertPos_)
                dragPaletteQueueInsertPos_ = newQueuePos;
        }
        else
        {
            const int newTarget = computePaletteInsertTarget (dragPaletteCurrentPos_);
            if (newTarget != dragPaletteInsertTargetIdx_)
                dragPaletteInsertTargetIdx_ = newTarget;
        }
        repaint(); // ghost + insertion line update every tick
        return;
    }

    if (dragQueueIdx_ < 0) return;

    dragQueueCurrentPos_ = e.getPosition();

    if (! dragQueueActive_)
    {
        const int dx = std::abs (dragQueueCurrentPos_.x - dragQueueStartPos_.x);
        const int dy = std::abs (dragQueueCurrentPos_.y - dragQueueStartPos_.y);
        if (dx + dy < kQueueDragThresholdPx) return;
        dragQueueActive_ = true;
    }

    const int newPos = computeInsertionPos (dragQueueCurrentPos_.x);
    if (newPos != dragQueueInsertionPos_)
    {
        dragQueueInsertionPos_ = newPos;
        repaint();
    }
}

void BlockAssemblyPanel::mouseUp (const juce::MouseEvent& /*e*/)
{
    // DEV-058 (c) + DEV-077 sesja 100d — paleta drag commit. Branch on
    // dragPaletteCrossingToQueue_ → onPaletteToQueueInsert; otherwise
    // onPaletteReorder; click fallback stays existing onCardClicked.
    if (dragPaletteBlockIdx_ >= 0)
    {
        const int  from           = dragPaletteBlockIdx_;
        const int  paletteTarget  = dragPaletteInsertTargetIdx_;
        const int  queueTarget    = dragPaletteQueueInsertPos_;
        const bool wasActive      = dragPaletteActive_;
        const bool wasModified    = dragPaletteClickModified_;
        const bool wasCrossing    = dragPaletteCrossingToQueue_;

        dragPaletteBlockIdx_           = -1;
        dragPaletteInsertTargetIdx_    = -1;
        dragPaletteQueueInsertPos_     = -1;
        dragPaletteActive_             = false;
        dragPaletteClickModified_      = false;
        dragPaletteCrossingToQueue_    = false;
        setMouseCursor (juce::MouseCursor::NormalCursor);

        if (wasModified)
        {
            // Cmd / Shift click already handled in mouseDown — no additional
            // mouseUp action.
            return;
        }

        if (! wasActive)
        {
            // Plain click — single-select replace + append to queue
            // (existing onCardClicked behavior preserved).
            paletteSelected_ = { from };
            paletteSelectionAnchor_ = from;
            if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
            if (onCardClicked) onCardClicked (from);
            repaint();
            return;
        }

        if (wasCrossing && queueTarget >= 0)
        {
            // DEV-077 — cross-section drag. Insert at queue gap; precise
            // positioning vs onCardClicked which always appends to end.
            if (onPaletteToQueueInsert) onPaletteToQueueInsert (from, queueTarget);
            repaint();
            return;
        }

        // Drag past threshold — commit reorder if target valid + non-self.
        // target == from OR target == from + 1 → drop on self / immediate
        // right of self → no-op.
        if (paletteTarget >= 0 && paletteTarget != from && paletteTarget != from + 1)
        {
            // Translate insertion-style target (0..userBlocks_.size()) into
            // the block-index that should land at the from position. For
            // moveBlock semantic in MainComponent we pass userBlocks_ index
            // directly: when target > from we want to insert AFTER target-1
            // → final index = target-1; when target < from we want to
            // insert BEFORE target → final index = target.
            const int finalIdx = (paletteTarget > from) ? paletteTarget - 1 : paletteTarget;
            if (onPaletteReorder) onPaletteReorder (from, finalIdx);
        }
        repaint();
        return;
    }

    if (dragQueueIdx_ < 0) return;

    const int from = dragQueueIdx_;
    const int to   = dragQueueInsertionPos_;
    const bool wasActive = dragQueueActive_;

    dragQueueIdx_          = -1;
    dragQueueInsertionPos_ = -1;
    dragQueueActive_       = false;

    if (! wasActive)
    {
        // Click (no significant travel) → audition the tile we picked up
        // from the position the user clicked. fractionInTile maps the
        // local click X to [0, 1] across the tile rect; host then seeks
        // into the block at the equivalent source-time offset.
        double fraction = 0.0;
        const auto rect = queueTileRect (from);
        if (! rect.isEmpty() && rect.getWidth() > 0)
        {
            fraction = juce::jlimit (0.0, 1.0,
                double (dragQueueStartPos_.x - rect.getX())
                / double (rect.getWidth()));
        }
        if (onQueueTileAudition) onQueueTileAudition (from, fraction);
        return;
    }

    // Drop into the same slot is a no-op (insertion at `from` or
    // `from + 1` lands the tile back where it started).
    if (to < 0 || to == from || to == from + 1)
    {
        repaint();
        return;
    }

    if (onQueueReorder) onQueueReorder (from, to);
    repaint();
}

// Map a cursor X (local to panel) to a queue insertion gap index.
// Returns 0..queue_.size(). Gap 0 = before tile 0; gap N = after last
// tile. Hit-tests tile midpoints: cursor left of tile i's midpoint →
// gap i; cursor right of last tile's midpoint → gap N.
int BlockAssemblyPanel::computeInsertionPos (int xLocal) const
{
    const int n = (int) queue_.size();
    if (n == 0) return 0;

    for (int i = 0; i < n; ++i)
    {
        const auto r = queueTileRect (i);
        if (r.isEmpty()) continue;
        const int mid = r.getX() + r.getWidth() / 2;
        if (xLocal < mid) return i;
    }
    return n;
}

// DEV-058 (c) sesja 100d — paleta drag insertion target. Returns a
// userBlocks_ index in [0, userBlocks_.size()] where the dragged tile
// will land on drop. The paleta wraps to N rows; we hit-test cards in
// row-major order using their painted rect. Closest card on the same
// row gets priority (pick left or right side of card based on cursor X
// vs midpoint); if cursor is below the last visible row, target is end-
// of-list.
int BlockAssemblyPanel::computePaletteInsertTarget (juce::Point<int> pos) const
{
    const int n = (int) userBlocks_.size();
    if (n == 0) return 0;

    // Walk all visible cards; pick the closest by manhattan distance.
    int bestIdx = -1;
    int bestDist = std::numeric_limits<int>::max();
    bool bestSideLeft = true;

    for (int i = 0; i < n; ++i)
    {
        const auto r = cardRect (i);
        if (r.isEmpty()) continue;
        const int rowYMid = r.getY() + r.getHeight() / 2;
        const int rowDelta = std::abs (pos.y - rowYMid);
        const int xMid     = r.getX() + r.getWidth() / 2;
        const int colDelta = std::abs (pos.x - xMid);
        const int dist = rowDelta * 2 + colDelta; // bias rows over cols

        if (dist < bestDist)
        {
            bestDist     = dist;
            bestIdx      = i;
            bestSideLeft = (pos.x < xMid);
        }
    }

    if (bestIdx < 0) return n;
    return bestSideLeft ? bestIdx : (bestIdx + 1);
}

// DEV-058 (b) sesja 100d — selection setters.
void BlockAssemblyPanel::setPaletteSelection (std::set<int> ids)
{
    if (ids == paletteSelected_) return;
    paletteSelected_ = std::move (ids);
    if (! paletteSelected_.empty())
        paletteSelectionAnchor_ = *paletteSelected_.begin();
    else
        paletteSelectionAnchor_ = -1;
    if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
    repaint();
}

void BlockAssemblyPanel::clearPaletteSelection()
{
    if (paletteSelected_.empty()) return;
    paletteSelected_.clear();
    paletteSelectionAnchor_ = -1;
    if (onPaletteSelectionChanged) onPaletteSelectionChanged (paletteSelected_);
    repaint();
}

juce::String BlockAssemblyPanel::kindLabel (SegmentKind k)
{
    switch (k)
    {
        case SegmentKind::Intro:        return "Intro";
        case SegmentKind::Verse:        return "Verse";
        case SegmentKind::PreChorus:    return "Pre-Chorus";
        case SegmentKind::Chorus:       return "Chorus";
        case SegmentKind::PostChorus:   return "Post-Chorus";
        case SegmentKind::Bridge:       return "Bridge";
        case SegmentKind::Buildup:      return "Buildup";
        case SegmentKind::Drop:         return "Drop";
        case SegmentKind::Breakdown:    return "Breakdown";
        case SegmentKind::Solo:         return "Solo";
        case SegmentKind::Instrumental: return "Instrumental";
        case SegmentKind::Outro:        return "Outro";
        case SegmentKind::NumKinds:     break;
    }
    return {};
}

} // namespace reamix::ui
