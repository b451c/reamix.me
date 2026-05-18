#include "BlockKindPickerPopover.h"

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

int BlockKindPickerPopover::totalPillCount (const CustomKindRegistry* registry)
{
    const int customCount = registry != nullptr ? registry->size() : 0;
    return (int) SegmentKind::NumKinds + customCount + 1; // + Add tile
}

int BlockKindPickerPopover::rowsFor (int pillCount)
{
    if (pillCount <= 0) return 1;
    return (pillCount + kCols - 1) / kCols;
}

int BlockKindPickerPopover::contentHeightFor (int rows)
{
    // Sesja 100c — fixed geometry. Sesja-61 had hardcoded 168 px which left
    // footer ~6 px shy of the last row; tolerable when footer text was short
    // and rows fixed at 3, but adding a 4th row (12 + 1 customs + Add tile)
    // and longer footer ("right-click custom to edit") made the bottom-row
    // labels collide with the footer text. Proper formula:
    //   top pad + header + 8 px gap + grid + 8 px gap + footer + bottom pad.
    if (rows <= 0) rows = 1;
    const int gridH = rows * kPillH + (rows - 1) * kPillGap;
    return kPadding + kHeaderH + 8 + gridH + 8 + kFooterH + kPadding;
}

BlockKindPickerPopover::BlockKindPickerPopover (SegmentKind smartDefault,
                                                double rangeStartSec,
                                                double rangeEndSec,
                                                const CustomKindRegistry* registry)
    : focusedIdx_      ((int) smartDefault),
      rangeStartSec_   (rangeStartSec),
      rangeEndSec_     (rangeEndSec),
      registry_        (registry)
{
    if (focusedIdx_ < 0 || focusedIdx_ >= (int) SegmentKind::NumKinds)
        focusedIdx_ = (int) SegmentKind::Verse;

    setSize (kContentWidth, contentHeightFor (rowsFor (pillCount())));
    setWantsKeyboardFocus (true);
    setMouseClickGrabsKeyboardFocus (true);
}

int BlockKindPickerPopover::pillCount() const
{
    return totalPillCount (registry_);
}

int BlockKindPickerPopover::rows() const
{
    return rowsFor (pillCount());
}

bool BlockKindPickerPopover::isBuiltinSlot (int idx) const
{
    return idx >= 0 && idx < (int) SegmentKind::NumKinds;
}

bool BlockKindPickerPopover::isCustomSlot (int idx) const
{
    if (registry_ == nullptr) return false;
    return idx >= (int) SegmentKind::NumKinds
        && idx < (int) SegmentKind::NumKinds + registry_->size();
}

bool BlockKindPickerPopover::isAddSlot (int idx) const
{
    return idx == pillCount() - 1;
}

juce::String BlockKindPickerPopover::customIdAtSlot (int idx) const
{
    if (registry_ == nullptr || ! isCustomSlot (idx)) return {};
    const int customIdx = idx - (int) SegmentKind::NumKinds;
    auto entries = registry_->all();
    if (customIdx < 0 || customIdx >= (int) entries.size()) return {};
    return entries[(std::size_t) customIdx].first;
}

void BlockKindPickerPopover::paint (juce::Graphics& g)
{
    g.fillAll (Bg2);
    g.setColour (Line);
    g.drawRect (getLocalBounds(), 1);

    const juce::Rectangle<int> headerArea {
        kPadding, kPadding, getWidth() - 2 * kPadding, kHeaderH
    };
    g.setColour (Fg3);
    g.setFont (monoFont (fs::Xs, 600).withExtraKerningFactor (0.08f));
    g.drawText (formatHeader(), headerArea, juce::Justification::centredLeft, false);

    const int total = pillCount();
    const auto customs = registry_ != nullptr ? registry_->all()
                                              : std::vector<std::pair<juce::String, CustomKindEntry>>{};

    for (int idx = 0; idx < total; ++idx)
    {
        const auto rect = pillRect (idx);
        const bool isFocused = (idx == focusedIdx_);
        const bool isHovered = (idx == hoveredIdx_);

        if (isAddSlot (idx))
        {
            // "+ Add custom" tile — dashed Accent border, faint Bg3 fill.
            const float alpha = isHovered ? 0.20f : 0.12f;
            g.setColour (Accent.withAlpha (alpha));
            g.fillRoundedRectangle (rect.toFloat(), 4.0f);

            // Dashed border via path strokeType
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
            if (customIdx >= 0 && customIdx < (int) customs.size())
            {
                kindColour = customs[(std::size_t) customIdx].second.color;
                label      = customs[(std::size_t) customIdx].second.name;
            }
            else
            {
                continue; // safety
            }
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

    // Footer
    const juce::Rectangle<int> footerArea {
        kPadding, getHeight() - kFooterH - kPadding,
        getWidth() - 2 * kPadding, kFooterH
    };
    g.setColour (Fg4);
    g.setFont (monoFont (fs::Xs, 500));
    g.drawText (juce::String::fromUTF8 ("\xE2\x86\xA9 confirm  \xC2\xB7  esc cancel  \xC2\xB7  arrows navigate  \xC2\xB7  right-click custom to edit"),
                footerArea, juce::Justification::centredLeft, false);
}

void BlockKindPickerPopover::resized() {}

juce::Rectangle<int> BlockKindPickerPopover::pillRect (int idx) const
{
    const int row = idx / kCols;
    const int col = idx % kCols;
    const int gridY = kPadding + kHeaderH + 8;
    const int x = kPadding + col * (kPillW + kPillGap);
    const int y = gridY + row * (kPillH + kPillGap);
    return { x, y, kPillW, kPillH };
}

int BlockKindPickerPopover::pillHitTest (juce::Point<int> pos) const
{
    const int total = pillCount();
    for (int idx = 0; idx < total; ++idx)
        if (pillRect (idx).contains (pos)) return idx;
    return -1;
}

void BlockKindPickerPopover::mouseMove (const juce::MouseEvent& e)
{
    const int idx = pillHitTest (e.getPosition());
    if (idx == hoveredIdx_) return;
    hoveredIdx_ = idx;
    setMouseCursor (idx >= 0 ? juce::MouseCursor::PointingHandCursor
                              : juce::MouseCursor::NormalCursor);
    repaint();
}

void BlockKindPickerPopover::mouseExit (const juce::MouseEvent&)
{
    if (hoveredIdx_ < 0) return;
    hoveredIdx_ = -1;
    setMouseCursor (juce::MouseCursor::NormalCursor);
    repaint();
}

void BlockKindPickerPopover::mouseDown (const juce::MouseEvent& e)
{
    const int idx = pillHitTest (e.getPosition());
    if (idx < 0) return;

    // Right-click on custom tile opens edit menu (Rename / Recolor / Delete).
    if (e.mods.isPopupMenu() && isCustomSlot (idx))
    {
        const juce::String id = customIdAtSlot (idx);
        if (id.isEmpty()) return;

        juce::PopupMenu m;
        m.addItem (1, "Rename");
        m.addItem (2, "Change color");
        m.addSeparator();
        m.addItem (3, "Delete");

        const auto screenPos = e.getScreenPosition();
        juce::PopupMenu::Options opts;
        opts = opts.withTargetScreenArea ({ screenPos.x, screenPos.y, 1, 1 });

        // Self-pointer dance — picker may dismiss before async menu fires.
        // Capture id by value; main component holds registry, so action is
        // safe to dispatch even if picker is gone by the time menu closes.
        // BUT we must guard against accessing `this` after dismiss.
        juce::Component::SafePointer<BlockKindPickerPopover> safe (this);
        m.showMenuAsync (opts, [safe, id] (int chosen)
        {
            if (safe == nullptr) return;
            CustomKindAction action;
            switch (chosen)
            {
                case 1: action = CustomKindAction::Rename;   break;
                case 2: action = CustomKindAction::Recolor;  break;
                case 3: action = CustomKindAction::Delete;   break;
                default: return;
            }
            safe->requestEdit (id, action);
        });
        return;
    }

    // Left-click → confirm.
    if (isAddSlot (idx))
    {
        requestAdd();
        return;
    }
    if (isCustomSlot (idx))
    {
        confirmCustom (customIdAtSlot (idx));
        return;
    }
    if (isBuiltinSlot (idx))
    {
        confirmBuiltin ((SegmentKind) idx);
        return;
    }
}

bool BlockKindPickerPopover::keyPressed (const juce::KeyPress& key)
{
    const int total = pillCount();

    if (key == juce::KeyPress::escapeKey) { cancel(); return true; }
    if (key == juce::KeyPress::returnKey || key == juce::KeyPress::spaceKey)
    {
        if (isAddSlot (focusedIdx_))           { requestAdd(); return true; }
        if (isCustomSlot (focusedIdx_))        { confirmCustom (customIdAtSlot (focusedIdx_)); return true; }
        if (isBuiltinSlot (focusedIdx_))       { confirmBuiltin ((SegmentKind) focusedIdx_); return true; }
    }
    if (key == juce::KeyPress::leftKey)
    {
        focusedIdx_ = (focusedIdx_ - 1 + total) % total;
        repaint();
        return true;
    }
    if (key == juce::KeyPress::rightKey)
    {
        focusedIdx_ = (focusedIdx_ + 1) % total;
        repaint();
        return true;
    }
    if (key == juce::KeyPress::upKey)
    {
        const int row = focusedIdx_ / kCols;
        const int col = focusedIdx_ % kCols;
        const int totalRows = rows();
        int newRow = (row - 1 + totalRows) % totalRows;
        int candidate = newRow * kCols + col;
        // Wrap if out of bounds (last row may not be full).
        while (candidate >= total) { newRow = (newRow - 1 + totalRows) % totalRows; candidate = newRow * kCols + col; }
        focusedIdx_ = candidate;
        repaint();
        return true;
    }
    if (key == juce::KeyPress::downKey)
    {
        const int row = focusedIdx_ / kCols;
        const int col = focusedIdx_ % kCols;
        const int totalRows = rows();
        int newRow = (row + 1) % totalRows;
        int candidate = newRow * kCols + col;
        while (candidate >= total) { newRow = (newRow + 1) % totalRows; candidate = newRow * kCols + col; }
        focusedIdx_ = candidate;
        repaint();
        return true;
    }
    return false;
}

void BlockKindPickerPopover::confirmBuiltin (SegmentKind kind)
{
    auto cb = onPicked;
    dismissHost();
    if (cb) cb (kind, std::nullopt);
}

void BlockKindPickerPopover::confirmCustom (const juce::String& id)
{
    auto cb = onPicked;
    dismissHost();
    // Verse fallback so legacy code paths reading `kind` still get a sensible
    // built-in display when the registry can't resolve `id` later.
    if (cb) cb (SegmentKind::Verse, id);
}

void BlockKindPickerPopover::requestAdd()
{
    // Sesja 100c iter 3 — picker stays open during Add modal. Caller
    // dismisses via pickerSafe on Add-confirm; on Cancel picker remains
    // so user can pick a different kind without losing the drag-mark.
    if (onAddCustomRequested)
    {
        juce::Component::SafePointer<juce::Component> safe (this);
        onAddCustomRequested (safe);
    }
}

void BlockKindPickerPopover::requestEdit (const juce::String& id, CustomKindAction action)
{
    // Sesja 100c iter 2 — DO NOT dismiss host. Picker stays alive while
    // edit modal is up; modal cascade blocks picker input but its paint
    // remains visible behind the modal. After modal closes, caller invokes
    // pickerSafe->repaint() to re-render with the mutated registry.
    if (onEditCustomAction)
    {
        juce::Component::SafePointer<juce::Component> safe (this);
        onEditCustomAction (id, action, safe);
    }
}

void BlockKindPickerPopover::cancel()
{
    auto cb = onCancel;
    dismissHost();
    if (cb) cb();
}

void BlockKindPickerPopover::dismissHost()
{
    if (auto* host = findParentComponentOfClass<juce::CallOutBox>())
        host->dismiss();
}

juce::String BlockKindPickerPopover::formatHeader() const
{
    return juce::String::fromUTF8 ("MARK SECTION  ")
         + formatMSS (rangeStartSec_)
         + juce::String::fromUTF8 (" \xE2\x80\x93 ")
         + formatMSS (rangeEndSec_);
}

juce::String BlockKindPickerPopover::labelFor (SegmentKind k)
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
