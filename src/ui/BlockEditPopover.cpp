#include "BlockEditPopover.h"

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

BlockEditPopover::BlockEditPopover (const reamix::ui::UserBlock& block,
                                     double itemDurationSec,
                                     double bpm,
                                     std::optional<reamix::ui::UserBlock> prevBlock,
                                     std::optional<reamix::ui::UserBlock> nextBlock,
                                     std::vector<double> beatTimes,
                                     const reamix::ui::CustomKindRegistry* registry)
    : block_           (block),
      itemDurationSec_ (itemDurationSec),
      beatSec_         (bpm > 1.0 ? 60.0 / bpm : 0.5),
      beatTimes_       (std::move (beatTimes)),
      prev_            (std::move (prevBlock)),
      next_            (std::move (nextBlock)),
      registry_        (registry),
      focusedPillIdx_  ((int) block.kind)
{
    if (focusedPillIdx_ < 0 || focusedPillIdx_ >= (int) SegmentKind::NumKinds)
        focusedPillIdx_ = (int) SegmentKind::Verse;

    // ADR-092 sesja 100c — focus the custom tile if block uses one.
    if (block.customKindId.has_value() && registry_ != nullptr)
    {
        const auto entries = registry_->all();
        for (int i = 0; i < (int) entries.size(); ++i)
            if (entries[(std::size_t) i].first == *block.customKindId)
            {
                focusedPillIdx_ = (int) SegmentKind::NumKinds + i;
                break;
            }
    }

    setSize (kContentWidth, computeContentHeight (registry_));
    setWantsKeyboardFocus (true);
    setMouseClickGrabsKeyboardFocus (true);
}

int BlockEditPopover::computeContentHeight (const reamix::ui::CustomKindRegistry* registry)
{
    // Grid dynamically grows: 12 builtin + N custom + 1 Add tile.
    const int customCount = registry != nullptr ? registry->size() : 0;
    const int total = (int) reamix::theme::SegmentKind::NumKinds + customCount + 1;
    const int rows  = (total + kCols - 1) / kCols;
    // Sesja-61 baseline: kRows=3 ⇒ 340 px. Each extra row adds (kPillH + kPillGap) = 38.
    if (rows <= 3) return 340;
    return 340 + (rows - 3) * (kPillH + kPillGap);
}

int BlockEditPopover::pillCount() const
{
    const int customCount = registry_ != nullptr ? registry_->size() : 0;
    return (int) SegmentKind::NumKinds + customCount + 1;
}

int BlockEditPopover::rows() const
{
    const int total = pillCount();
    return (total + kCols - 1) / kCols;
}

bool BlockEditPopover::isBuiltinSlot (int idx) const
{
    return idx >= 0 && idx < (int) SegmentKind::NumKinds;
}

bool BlockEditPopover::isCustomSlot (int idx) const
{
    if (registry_ == nullptr) return false;
    return idx >= (int) SegmentKind::NumKinds
        && idx < (int) SegmentKind::NumKinds + registry_->size();
}

bool BlockEditPopover::isAddSlot (int idx) const
{
    return idx == pillCount() - 1;
}

juce::String BlockEditPopover::customIdAtSlot (int idx) const
{
    if (registry_ == nullptr || ! isCustomSlot (idx)) return {};
    const int customIdx = idx - (int) SegmentKind::NumKinds;
    auto entries = registry_->all();
    if (customIdx < 0 || customIdx >= (int) entries.size()) return {};
    return entries[(std::size_t) customIdx].first;
}

// ── Geometry ────────────────────────────────────────────────────────

juce::Rectangle<int> BlockEditPopover::pillRect (int kindIdx) const
{
    const int row = kindIdx / kCols;
    const int col = kindIdx % kCols;
    const int gridY = kPadding + kHeaderH + 8;
    const int x = kPadding + col * (kPillW + kPillGap);
    const int y = gridY + row * (kPillH + kPillGap);
    return { x, y, kPillW, kPillH };
}

juce::Rectangle<int> BlockEditPopover::startNudgerArea() const
{
    // Sesja 100c — gridBottom dynamic w.r.t. registry.
    const int gridBottom = kPadding + kHeaderH + 8 + rows() * (kPillH + kPillGap);
    const int y = gridBottom + kSectionGap;
    return { kPadding, y, kContentWidth - 2 * kPadding, kNudgerH };
}

juce::Rectangle<int> BlockEditPopover::endNudgerArea() const
{
    const auto sa = startNudgerArea();
    return sa.translated (0, sa.getHeight() + 6);
}

// Sesja 100 (DEV-029) — action row containing Split / Merge Left / Merge Right.
// Sits between the End nudger and the Delete button. Three equal-width
// buttons with kPillGap spacing.
juce::Rectangle<int> BlockEditPopover::actionRowArea() const
{
    const auto en = endNudgerArea();
    const int y = en.getBottom() + kSectionGap;
    return { kPadding, y, kContentWidth - 2 * kPadding, kDeleteH };
}

juce::Rectangle<int> BlockEditPopover::splitBtnRect() const
{
    const auto a = actionRowArea();
    const int btnW = (a.getWidth() - 2 * kPillGap) / 3;
    return { a.getX(), a.getY(), btnW, a.getHeight() };
}

juce::Rectangle<int> BlockEditPopover::mergeLeftBtnRect() const
{
    const auto a = actionRowArea();
    const int btnW = (a.getWidth() - 2 * kPillGap) / 3;
    return { a.getX() + btnW + kPillGap, a.getY(), btnW, a.getHeight() };
}

juce::Rectangle<int> BlockEditPopover::mergeRightBtnRect() const
{
    const auto a = actionRowArea();
    const int btnW = (a.getWidth() - 2 * kPillGap) / 3;
    return { a.getX() + 2 * (btnW + kPillGap), a.getY(),
             a.getRight() - (a.getX() + 2 * (btnW + kPillGap)), a.getHeight() };
}

juce::Rectangle<int> BlockEditPopover::deleteBtnRect() const
{
    const auto ar = actionRowArea();
    const int y = ar.getBottom() + kSectionGap;
    return { kPadding, y, kContentWidth - 2 * kPadding, kDeleteH };
}

juce::Rectangle<int> BlockEditPopover::nudgerBtnRect (juce::Rectangle<int> area,
                                                       int slot) const
{
    // Layout: [<<-beat] [<-fine] [time-readout 110w] [+fine->] [+beat->>]
    const int btnW = kNudgeBtnW;
    const int spacing = 4;
    const int totalBtns = 4;
    const int readoutW  = 110;
    const int totalW    = totalBtns * btnW + 4 * spacing + readoutW;
    const int leftLabel = 70; // "Start" / "End"
    const int x0 = area.getX() + leftLabel;

    if (slot < 2)
    {
        // before readout
        return { x0 + slot * (btnW + spacing), area.getY() + 1, btnW, area.getHeight() - 2 };
    }
    else
    {
        // after readout
        const int afterX = x0 + 2 * (btnW + spacing) + readoutW + spacing;
        return { afterX + (slot - 2) * (btnW + spacing), area.getY() + 1, btnW, area.getHeight() - 2 };
    }
    juce::ignoreUnused (totalW);
}

// ── Hit test ────────────────────────────────────────────────────────

BlockEditPopover::HotElement BlockEditPopover::elementAt (juce::Point<int> pos) const
{
    for (int i = 0; i < pillCount(); ++i)
    {
        if (pillRect (i).contains (pos)) return HotElement::Pill;
    }
    const auto sn = startNudgerArea();
    if (sn.contains (pos))
    {
        if (nudgerBtnRect (sn, 0).contains (pos)) return HotElement::StartMinusBeat;
        if (nudgerBtnRect (sn, 1).contains (pos)) return HotElement::StartMinusFine;
        if (nudgerBtnRect (sn, 2).contains (pos)) return HotElement::StartPlusFine;
        if (nudgerBtnRect (sn, 3).contains (pos)) return HotElement::StartPlusBeat;
    }
    const auto en = endNudgerArea();
    if (en.contains (pos))
    {
        if (nudgerBtnRect (en, 0).contains (pos)) return HotElement::EndMinusBeat;
        if (nudgerBtnRect (en, 1).contains (pos)) return HotElement::EndMinusFine;
        if (nudgerBtnRect (en, 2).contains (pos)) return HotElement::EndPlusFine;
        if (nudgerBtnRect (en, 3).contains (pos)) return HotElement::EndPlusBeat;
    }
    // Sesja 100 (DEV-029) — action row hit-test (only respond when enabled).
    if (canSplit()      && splitBtnRect().contains (pos))      return HotElement::Split;
    if (canMergeLeft()  && mergeLeftBtnRect().contains (pos))  return HotElement::MergeLeft;
    if (canMergeRight() && mergeRightBtnRect().contains (pos)) return HotElement::MergeRight;
    if (deleteBtnRect().contains (pos)) return HotElement::Delete;
    return HotElement::None;
}

// ── Paint ───────────────────────────────────────────────────────────

void BlockEditPopover::paint (juce::Graphics& g)
{
    g.fillAll (Bg2);
    g.setColour (Line);
    g.drawRect (getLocalBounds(), 1);

    // Header
    g.setColour (Fg3);
    g.setFont (monoFont (fs::Xs, 600).withExtraKerningFactor (0.08f));
    g.drawText (formatHeader(),
                juce::Rectangle<int> (kPadding, kPadding,
                                       kContentWidth - 2 * kPadding, kHeaderH),
                juce::Justification::centredLeft, false);

    // ADR-092 sesja 100c — dynamic grid: 12 builtin + N custom + Add tile.
    const int total = pillCount();
    const auto customs = registry_ != nullptr ? registry_->all()
                                              : std::vector<std::pair<juce::String, CustomKindEntry>>{};
    for (int idx = 0; idx < total; ++idx)
    {
        const auto rect = pillRect (idx);
        const bool isFocused = (idx == focusedPillIdx_);
        const bool isHovered = (hovered_ == HotElement::Pill && rect.contains (
                                  getMouseXYRelative()));

        if (isAddSlot (idx))
        {
            const float alpha = isHovered ? 0.20f : 0.12f;
            g.setColour (Accent.withAlpha (alpha));
            g.fillRoundedRectangle (rect.toFloat(), 4.0f);

            juce::Path p;
            p.addRoundedRectangle (rect.toFloat().reduced (0.5f), 4.0f);
            const float dashes[] = { 4.0f, 3.0f };
            juce::PathStrokeType stroke (isFocused ? 2.0f : 1.2f);
            stroke.createDashedStroke (p, p, dashes, 2);
            g.setColour (isFocused ? Accent : Accent.withAlpha (0.85f));
            g.fillPath (p);

            g.setColour (Accent);
            g.setFont (uiFont (fs::Sm, 600));
            g.drawText (juce::String::fromUTF8 ("+ Add custom"),
                        rect, juce::Justification::centred, false);
            continue;
        }

        juce::Colour kindColour;
        juce::String label;
        if (isBuiltinSlot (idx))
        {
            const auto kind = (SegmentKind) idx;
            kindColour = segmentColour (kind);
            label      = labelFor (kind);
        }
        else
        {
            const int customIdx = idx - (int) SegmentKind::NumKinds;
            if (customIdx < 0 || customIdx >= (int) customs.size()) continue;
            kindColour = customs[(std::size_t) customIdx].second.color;
            label      = customs[(std::size_t) customIdx].second.name;
        }

        float alpha = 0.28f;
        if (isHovered) alpha = 0.44f;
        if (isFocused) alpha = 0.60f;
        g.setColour (kindColour.withAlpha (alpha));
        g.fillRoundedRectangle (rect.toFloat(), 4.0f);

        if (isFocused)
        {
            g.setColour (Accent);
            g.drawRoundedRectangle (rect.toFloat().reduced (1.0f), 4.0f, 2.0f);
        }
        else
        {
            g.setColour (kindColour.withAlpha (0.85f));
            g.drawRoundedRectangle (rect.toFloat().reduced (0.5f), 4.0f, 1.0f);
        }

        g.setColour (juce::Colour::fromFloatRGBA (1.0f, 1.0f, 1.0f, isFocused ? 0.97f : 0.85f));
        g.setFont (uiFont (fs::Sm, 600));
        g.drawText (label, rect, juce::Justification::centred, false);
    }

    // Nudger rows — start
    const auto drawNudgerRow = [&] (juce::Rectangle<int> area,
                                     const juce::String& label,
                                     double timeSec,
                                     HotElement minusBeat, HotElement minusFine,
                                     HotElement plusFine, HotElement plusBeat)
    {
        // Label on left
        g.setColour (Fg2);
        g.setFont (uiFont (fs::Sm, 500));
        g.drawText (label, area.withWidth (60).translated (4, 0),
                    juce::Justification::centredLeft, true);

        const auto drawBtn = [&] (juce::Rectangle<int> r, const juce::String& s, HotElement el)
        {
            const bool h = (hovered_ == el);
            g.setColour (h ? Bg5 : Bg4);
            g.fillRoundedRectangle (r.toFloat(), 3.0f);
            g.setColour (h ? LineStrong : Line);
            g.drawRoundedRectangle (r.toFloat().reduced (0.5f), 3.0f, 1.0f);
            g.setColour (Fg1);
            g.setFont (monoFont (fs::Sm, 700));
            g.drawText (s, r, juce::Justification::centred, false);
        };
        // Sesja 99 — DEV-062 (sesja 62 NOTE-17): replaced bare arrow
        // glyphs «/‹/›/» with explicit unit-bearing labels. Standard
        // DAW shorthand: "1b" = 1 beat, ".1s" = 0.1 second, U+2212 minus
        // sign for typographic consistency.
        drawBtn (nudgerBtnRect (area, 0),
                  juce::String::fromUTF8 ("\xE2\x88\x92" "1b"),  minusBeat);
        drawBtn (nudgerBtnRect (area, 1),
                  juce::String::fromUTF8 ("\xE2\x88\x92" ".1s"), minusFine);

        // Time readout (between fine buttons)
        const int btnW = kNudgeBtnW, spacing = 4;
        const int readoutX = area.getX() + 70 + 2 * (btnW + spacing);
        const int readoutW = 110;
        const auto readoutR = juce::Rectangle<int> (
            readoutX, area.getY() + 1, readoutW, area.getHeight() - 2);
        g.setColour (Bg1);
        g.fillRoundedRectangle (readoutR.toFloat(), 3.0f);
        g.setColour (Accent);
        g.setFont (monoFont (fs::Md, 700));
        g.drawText (formatMSS (timeSec), readoutR, juce::Justification::centred, false);

        drawBtn (nudgerBtnRect (area, 2), juce::String ("+.1s"), plusFine);
        drawBtn (nudgerBtnRect (area, 3), juce::String ("+1b"),  plusBeat);
    };

    drawNudgerRow (startNudgerArea(),  "Start", block_.startSec,
                   HotElement::StartMinusBeat, HotElement::StartMinusFine,
                   HotElement::StartPlusFine,  HotElement::StartPlusBeat);
    drawNudgerRow (endNudgerArea(),    "End",   block_.endSec,
                   HotElement::EndMinusBeat,   HotElement::EndMinusFine,
                   HotElement::EndPlusFine,    HotElement::EndPlusBeat);

    // Sesja 100 (DEV-029) — action row: Split / Merge Left / Merge Right.
    // Neutral Accent-tinted (non-destructive); disabled state when predicate
    // returns false (block too short / first block / last block).
    auto drawActionBtn = [&] (juce::Rectangle<int> r,
                              const juce::String& label,
                              HotElement el,
                              bool enabled)
    {
        const bool h = enabled && (hovered_ == el);
        if (! enabled)
        {
            g.setColour (Bg3);
            g.fillRoundedRectangle (r.toFloat(), 3.0f);
            g.setColour (Line);
            g.drawRoundedRectangle (r.toFloat().reduced (0.5f), 3.0f, 1.0f);
            g.setColour (Fg4);
        }
        else
        {
            g.setColour (h ? Accent.withAlpha (0.20f) : Bg1);
            g.fillRoundedRectangle (r.toFloat(), 3.0f);
            g.setColour (h ? Accent : Line);
            g.drawRoundedRectangle (r.toFloat().reduced (0.5f), 3.0f, 1.0f);
            g.setColour (h ? AccentHi : Accent);
        }
        g.setFont (uiFont (fs::Sm, 600));
        g.drawText (label, r, juce::Justification::centred, false);
    };

    drawActionBtn (splitBtnRect(),      "Split",       HotElement::Split,      canSplit());
    drawActionBtn (mergeLeftBtnRect(),  "Merge Left",  HotElement::MergeLeft,  canMergeLeft());
    drawActionBtn (mergeRightBtnRect(), "Merge Right", HotElement::MergeRight, canMergeRight());

    // Delete button — destructive Bad-tinted
    const auto delR = deleteBtnRect();
    const bool delH = (hovered_ == HotElement::Delete);
    const juce::Colour bad { 0xFFCC4444 };
    g.setColour (bad.withAlpha (delH ? 0.40f : 0.20f));
    g.fillRoundedRectangle (delR.toFloat(), 3.0f);
    g.setColour (bad);
    g.drawRoundedRectangle (delR.toFloat().reduced (0.5f), 3.0f, 1.0f);
    g.setFont (uiFont (fs::Sm, 600));
    g.drawText ("Delete this block", delR, juce::Justification::centred, false);
}

void BlockEditPopover::resized() {}

void BlockEditPopover::mouseMove (const juce::MouseEvent& e)
{
    const auto h = elementAt (e.getPosition());
    if (h == hovered_) return;
    hovered_ = h;
    setMouseCursor (h != HotElement::None
                    ? juce::MouseCursor::PointingHandCursor
                    : juce::MouseCursor::NormalCursor);
    repaint();
}

void BlockEditPopover::mouseExit (const juce::MouseEvent&)
{
    if (hovered_ == HotElement::None) return;
    hovered_ = HotElement::None;
    setMouseCursor (juce::MouseCursor::NormalCursor);
    repaint();
}

void BlockEditPopover::mouseDown (const juce::MouseEvent& e)
{
    const auto pos = e.getPosition();
    // ADR-092 sesja 100c — pill-grid hit-test (builtin / custom / Add tile).
    for (int i = 0; i < pillCount(); ++i)
    {
        if (! pillRect (i).contains (pos)) continue;

        // Right-click on custom tile → Rename/Recolor/Delete menu.
        if (e.mods.isPopupMenu() && isCustomSlot (i))
        {
            const juce::String id = customIdAtSlot (i);
            if (id.isEmpty()) return;

            juce::PopupMenu m;
            m.addItem (1, "Rename");
            m.addItem (2, "Change color");
            m.addSeparator();
            m.addItem (3, "Delete");
            const auto screenPos = e.getScreenPosition();
            juce::PopupMenu::Options opts;
            opts = opts.withTargetScreenArea ({ screenPos.x, screenPos.y, 1, 1 });
            juce::Component::SafePointer<BlockEditPopover> safe (this);
            m.showMenuAsync (opts, [safe, id] (int chosen)
            {
                if (safe == nullptr) return;
                CustomKindAction action;
                switch (chosen)
                {
                    case 1: action = CustomKindAction::Rename;  break;
                    case 2: action = CustomKindAction::Recolor; break;
                    case 3: action = CustomKindAction::Delete;  break;
                    default: return;
                }
                safe->requestEdit (id, action);
            });
            return;
        }

        if (isAddSlot (i))     { requestAdd();                            return; }
        if (isCustomSlot (i))  { confirmKindCustom (customIdAtSlot (i));  return; }
        if (isBuiltinSlot (i)) { confirmKindBuiltin ((SegmentKind) i);    return; }
    }
    const auto h = elementAt (pos);
    switch (h)
    {
        // Sesja 64 BUG-7 — beat nudges snap to beatTimes_ boundary (not
        // shift by 60/bpm). Fine nudges (±0.1 s) keep precise behavior.
        case HotElement::StartMinusBeat: applyStartBeatStep (-1);     return;
        case HotElement::StartMinusFine: applyStartNudge    (-0.1);   return;
        case HotElement::StartPlusFine:  applyStartNudge    (+0.1);   return;
        case HotElement::StartPlusBeat:  applyStartBeatStep (+1);     return;
        case HotElement::EndMinusBeat:   applyEndBeatStep   (-1);     return;
        case HotElement::EndMinusFine:   applyEndNudge      (-0.1);   return;
        case HotElement::EndPlusFine:    applyEndNudge      (+0.1);   return;
        case HotElement::EndPlusBeat:    applyEndBeatStep   (+1);     return;
        case HotElement::Split:          confirmSplit();              return;
        case HotElement::MergeLeft:      confirmMergeLeft();          return;
        case HotElement::MergeRight:     confirmMergeRight();         return;
        case HotElement::Delete:         confirmDelete();             return;
        default: return;
    }
}

bool BlockEditPopover::keyPressed (const juce::KeyPress& key)
{
    if (key == juce::KeyPress::escapeKey)
    {
        closeHost();
        return true;
    }
    return false;
}

void BlockEditPopover::applyStartNudge (double delta)
{
    double v = block_.startSec + delta;
    // Sesja 64 BUG-6 — allow touching the previous block's end exactly
    // (zero-gap adjacent boundary). Previous +0.5 reservation forced a
    // minimum 0.5 s gap which broke "fully consecutive blocks" use case.
    const double minBound = prev_.has_value() ? prev_->endSec : 0.0;
    const double maxBound = block_.endSec - 0.5;  // keep block min size
    v = juce::jlimit (minBound, maxBound, v);
    if (std::abs (v - block_.startSec) < 1e-6) return;
    block_.startSec = v;
    repaint (startNudgerArea());
    if (onBoundariesChanged) onBoundariesChanged (block_.startSec, block_.endSec);
}

void BlockEditPopover::applyEndNudge (double delta)
{
    double v = block_.endSec + delta;
    const double minBound = block_.startSec + 0.5;  // keep block min size
    // Sesja 64 BUG-6 — allow touching next block's start exactly.
    const double maxBound = next_.has_value() ? next_->startSec : itemDurationSec_;
    v = juce::jlimit (minBound, maxBound, v);
    if (std::abs (v - block_.endSec) < 1e-6) return;
    block_.endSec = v;
    repaint (endNudgerArea());
    if (onBoundariesChanged) onBoundariesChanged (block_.startSec, block_.endSec);
}

// Sesja 64 BUG-7 — snap-to-beat-boundary nudge. Replaces "shift by beat
// duration" semantics. Uses bundle.beatTimes (passed via constructor); falls
// back to legacy beatSec_ shift when beatTimes_ is empty (defensive).
void BlockEditPopover::applyStartBeatStep (int direction)
{
    if (beatTimes_.empty())
    {
        applyStartNudge (direction * beatSec_);  // fallback
        return;
    }
    const double cur = block_.startSec;
    double target = cur;
    if (direction > 0)
    {
        auto it = std::upper_bound (beatTimes_.begin(), beatTimes_.end(), cur + 1e-3);
        if (it == beatTimes_.end()) return;  // already past last beat
        target = *it;
    }
    else
    {
        auto it = std::lower_bound (beatTimes_.begin(), beatTimes_.end(), cur - 1e-3);
        if (it == beatTimes_.begin()) return;  // already at first beat
        --it;
        target = *it;
    }
    applyStartNudge (target - cur);  // reuses clamp + callback path
}

void BlockEditPopover::applyEndBeatStep (int direction)
{
    if (beatTimes_.empty())
    {
        applyEndNudge (direction * beatSec_);  // fallback
        return;
    }
    const double cur = block_.endSec;
    double target = cur;
    if (direction > 0)
    {
        auto it = std::upper_bound (beatTimes_.begin(), beatTimes_.end(), cur + 1e-3);
        if (it == beatTimes_.end()) return;
        target = *it;
    }
    else
    {
        auto it = std::lower_bound (beatTimes_.begin(), beatTimes_.end(), cur - 1e-3);
        if (it == beatTimes_.begin()) return;
        --it;
        target = *it;
    }
    applyEndNudge (target - cur);
}

void BlockEditPopover::confirmKindBuiltin (SegmentKind kind)
{
    block_.kind = kind;
    block_.customKindId.reset();   // builtin pick clears any custom assignment
    focusedPillIdx_ = (int) kind;
    repaint();
    if (onKindChanged) onKindChanged (kind, std::nullopt);
}

void BlockEditPopover::confirmKindCustom (const juce::String& id)
{
    if (id.isEmpty()) return;
    block_.customKindId = id;
    // Keep block_.kind as fallback (used when registry-miss elsewhere).
    if (registry_ != nullptr)
    {
        auto entries = registry_->all();
        for (int i = 0; i < (int) entries.size(); ++i)
            if (entries[(std::size_t) i].first == id)
            {
                focusedPillIdx_ = (int) SegmentKind::NumKinds + i;
                break;
            }
    }
    repaint();
    if (onKindChanged) onKindChanged (block_.kind, id);
}

void BlockEditPopover::requestAdd()
{
    if (onAddCustomRequested)
    {
        juce::Component::SafePointer<juce::Component> safe (this);
        onAddCustomRequested (safe);
    }
}

void BlockEditPopover::requestEdit (const juce::String& id, CustomKindAction action)
{
    if (onEditCustomAction)
    {
        juce::Component::SafePointer<juce::Component> safe (this);
        onEditCustomAction (id, action, safe);
    }
}

void BlockEditPopover::confirmDelete()
{
    auto cb = onDelete;
    closeHost();
    if (cb) cb();
}

// Sesja 100 (DEV-029) — Split / Merge Left / Merge Right confirm handlers.
// Each fires the callback before dismissing so the popover stays alive long
// enough to read its members; MainComponent owns the userBlocks_ mutation.

void BlockEditPopover::confirmSplit()
{
    if (! canSplit()) return;
    auto cb = onSplit;
    closeHost();
    if (cb) cb();
}

void BlockEditPopover::confirmMergeLeft()
{
    if (! canMergeLeft()) return;
    auto cb = onMergeLeft;
    closeHost();
    if (cb) cb();
}

void BlockEditPopover::confirmMergeRight()
{
    if (! canMergeRight()) return;
    auto cb = onMergeRight;
    closeHost();
    if (cb) cb();
}

void BlockEditPopover::closeHost()
{
    if (auto* host = findParentComponentOfClass<juce::CallOutBox>())
        host->dismiss();
}

juce::String BlockEditPopover::formatHeader() const
{
    return juce::String::fromUTF8 ("EDIT BLOCK  ")
         + formatMSS (block_.startSec)
         + juce::String::fromUTF8 (" \xE2\x80\x93 ")
         + formatMSS (block_.endSec);
}

juce::String BlockEditPopover::labelFor (SegmentKind k)
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
