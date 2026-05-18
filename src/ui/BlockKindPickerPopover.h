#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "Theme.h"
#include "CustomKindRegistry.h"

#include <functional>
#include <optional>

// BlockKindPickerPopover — premium pill-grid kind chooser shown after a
// drag-mark gesture in Block Assembly mode (ADR-051 § D2 + ADR-092 sesja 100c).
//
// Designed to be hosted in juce::CallOutBox::launchAsynchronously anchored at
// the drag-release point. The component owns its own keyboard focus + arrow-key
// navigation; Enter confirms the focused kind, Esc / click-away cancels.
//
// Visual:
//   • header line "MARK SECTION 0:30 – 1:15" (Mono Xs, uppercase, Fg3) so the
//     user can verify what range they're labeling before kind picks.
//   • dynamic 4 cols × N rows pill grid: 12 built-in kindy (fixed slots 0-11) +
//     M custom kindy from CustomKindRegistry (slots 12..12+M-1) +
//     "+ Add custom" tile at the last slot.
//   • Smart-default (per ADR-051 § Premium-feel #3) gets keyboard focus + 2 px
//     Accent inner ring on first paint.
//   • footer line "↩ confirm · esc cancel" (Mono Xs, Fg4).
//
// Smart-default rule (sesja 61): caller resolves kind via
// reamix::ui::smartKindForPosition (UserBlock.h) before constructing.
//
// Custom kind interactions (ADR-092 sesja 100c):
//   • Left-click custom tile → onPicked(SegmentKind::Verse fallback, customKindId).
//   • Left-click "+ Add custom" tile → onAddCustomRequested(); picker dismisses;
//     caller shows Add modal (name + color) and creates registry entry.
//   • Right-click custom tile → in-picker PopupMenu {Rename / Change color /
//     Delete} → onEditCustomAction(id, action); picker dismisses; caller shows
//     action-specific modal.

namespace reamix::ui
{

enum class CustomKindAction
{
    Rename,
    Recolor,
    Delete,
};

class BlockKindPickerPopover : public juce::Component
{
public:
    BlockKindPickerPopover (reamix::theme::SegmentKind smartDefault,
                            double rangeStartSec,
                            double rangeEndSec,
                            const CustomKindRegistry* registry);
    ~BlockKindPickerPopover() override = default;

    // Picked: built-in kind OR custom (when customKindId.has_value(), kind is
    // the fallback for legacy / cross-machine display).
    std::function<void (reamix::theme::SegmentKind kind,
                        std::optional<juce::String> customKindId)> onPicked;

    // "+ Add custom" tile clicked. Caller responsibility: show name + color
    // modal. Picker stays open behind modal (sesja 100c iter 3); on confirm
    // caller creates registry entry, commits the just-marked region, and
    // dismisses picker via pickerSafe. On cancel picker remains so user can
    // pick another kind or Esc out without losing the drag-mark.
    std::function<void (juce::Component::SafePointer<juce::Component> pickerSafe)>
        onAddCustomRequested;

    // Right-click custom tile → user picked an action. Caller responsibility:
    // show action-specific modal (rename text field / color picker /
    // delete confirm with cascade count) and mutate registry. Picker stays
    // open (sesja 100c iter 2 UX — user can edit several kindy in a row
    // without re-drag-marking). pickerSafe is a SafePointer to the picker
    // for caller to invoke repaint() after registry mutation; null when
    // picker is destroyed before modal callback.
    std::function<void (juce::String customKindId, CustomKindAction action,
                        juce::Component::SafePointer<juce::Component> pickerSafe)> onEditCustomAction;

    // Cancel (Esc / click-away).
    std::function<void()> onCancel;

    // Geometry — dynamic with custom count + 1 add tile.
    static constexpr int kCols      = 4;
    static constexpr int kPillW     = 120;
    static constexpr int kPillH     = 36;
    static constexpr int kPillGap   = 8;
    static constexpr int kHeaderH   = 18;
    static constexpr int kFooterH   = 22;
    static constexpr int kPadding   = 12;
    static constexpr int kContentWidth = 528;

    // Pre-construction sizing helper for callers (CallOutBox launch site needs
    // setSize before construct). Returns content size for the supplied
    // registry — registry may be null at that call site for "Verse-only"
    // fallback (12 + 1 = 13 → 4 rows).
    static int totalPillCount (const CustomKindRegistry* registry);
    static int rowsFor (int pillCount);
    static int contentHeightFor (int rows);

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;

private:
    juce::Rectangle<int> pillRect (int idx) const;
    int                  pillHitTest (juce::Point<int> pos) const;
    int                  pillCount() const;
    int                  rows() const;

    // Slot semantics: 0..11 = built-in SegmentKind, 12..12+customCount-1 =
    // custom (registry.all() insertion order), last slot = add tile.
    bool                 isBuiltinSlot (int idx) const;
    bool                 isCustomSlot  (int idx) const;
    bool                 isAddSlot     (int idx) const;
    juce::String         customIdAtSlot (int idx) const;

    void confirmBuiltin (reamix::theme::SegmentKind kind);
    void confirmCustom  (const juce::String& id);
    void requestAdd();
    void requestEdit    (const juce::String& id, CustomKindAction action);
    void cancel();
    void dismissHost();

    juce::String formatHeader() const;
    static juce::String labelFor (reamix::theme::SegmentKind k);

    int     focusedIdx_  { 0 };
    int     hoveredIdx_  { -1 };
    double  rangeStartSec_;
    double  rangeEndSec_;
    const CustomKindRegistry* registry_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BlockKindPickerPopover)
};

} // namespace reamix::ui
