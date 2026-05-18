#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "Theme.h"
#include "UserBlock.h"
#include "CustomKindRegistry.h"
#include "BlockKindPickerPopover.h" // CustomKindAction enum

#include <functional>
#include <optional>

// BlockEditPopover — phase-6 step 8 EDIT-mode popover (ADR-051 phase H, sesja 61).
//
// Opens via WaveformView::onUserBlockClicked when user clicks an existing
// user-marked block on the segBar. Anchored at the click point as a JUCE
// CallOutBox.
//
// Phase H MVP scope (sesja 61):
//   • Re-label via 4×3 kind pill grid (same visual language as
//     BlockKindPickerPopover so the create vs edit story is consistent).
//   • Start / end nudgers ±0.1 s / ±1 beat with monotonic-neighbour clamping
//     (mockup palette-source-edit.jsx:73-89 logic; clamp to (prev->endSec
//     + 0.5, next->startSec - 0.5) so blocks don't overlap and don't collapse).
//   • Delete button — removes the block, dismisses popover.
//
// Deferred to sesja 62 (per ADR-051 § Implementation plan H scope cut):
//   • Split (mid-point), Merge Left, Merge Right.
//   • Drag boundary handles directly on the segBar (ew-resize cursor, accent
//     line during drag).
//   • Audition Section / Start edge / End edge with range scrim + playhead.
// MVP without these still gives the user re-label + nudge + delete, which
// covers the most common edit cases. Power-user features for sesja 62.
//
// All callbacks are emitted before the popover dismisses itself; recipient
// (MainComponent) is responsible for state mutation (userBlocks_, persist).

namespace reamix::ui
{

class BlockEditPopover : public juce::Component
{
public:
    // `block` is a snapshot of the user block being edited; nudgers shift
    // within the bounds of `prevBlock` / `nextBlock` (when present, nullopt
    // otherwise — first / last block in the sorted list).
    BlockEditPopover (const reamix::ui::UserBlock& block,
                      double itemDurationSec,
                      double bpm,
                      std::optional<reamix::ui::UserBlock> prevBlock,
                      std::optional<reamix::ui::UserBlock> nextBlock,
                      // Sesja 64 BUG-7 — beat-boundary times from
                      // AnalysisBundle. ±1 beat nudge snaps to nearest
                      // boundary instead of shifting by 60/bpm seconds.
                      // Empty → falls back to legacy beat-duration shift.
                      std::vector<double> beatTimes = {},
                      // ADR-092 sesja 100c — registry pointer for custom
                      // kindy in pill grid. Null → builtin-only grid (legacy).
                      const reamix::ui::CustomKindRegistry* registry = nullptr);
    ~BlockEditPopover() override = default;

    // Callbacks (one of these fires per user action; popover dismisses).
    // ADR-092 sesja 100c — onKindChanged extended with optional customKindId.
    // Built-in pick: customKindId = nullopt → block reverts to built-in.
    // Custom pick: customKindId = id → block uses custom; kind acts as fallback.
    std::function<void (reamix::theme::SegmentKind kind,
                        std::optional<juce::String> customKindId)> onKindChanged;
    std::function<void (double newStartSec, double newEndSec)> onBoundariesChanged;
    std::function<void()>                                  onDelete;

    // ADR-092 sesja 100c — "+ Add custom" tile clicked. Caller opens Add
    // modal (name + color), on confirm adds to registry + repaints popover
    // via pickerSafe. Popover stays open. User then clicks the new tile to
    // apply it to this block.
    std::function<void (juce::Component::SafePointer<juce::Component>)>
        onAddCustomRequested;

    // ADR-092 sesja 100c — right-click custom tile → Rename/Recolor/Delete.
    // Same callback pattern as BlockKindPickerPopover.
    std::function<void (juce::String customKindId,
                        reamix::ui::CustomKindAction action,
                        juce::Component::SafePointer<juce::Component>)>
        onEditCustomAction;

    // Sesja 100 (DEV-029) — full edit-mode actions per ADR-091 + mockup
    // `palette-source-edit.jsx` EDIT mode. Each fires before the popover
    // dismisses; MainComponent owns the userBlocks_ mutation + state sync.
    //
    // onSplit — split current block at its midpoint into two blocks of the
    //   same kind. Disabled when block is too short (< 1.0 s) to avoid
    //   creating sub-second sub-blocks.
    // onMergeLeft — absorb prev_ into this block (resulting span is
    //   [prev.startSec, block.endSec]; resulting kind is the current block's).
    //   Disabled when prev_ is nullopt (this is the first block).
    // onMergeRight — symmetric: absorb next_ into this block ([block.startSec,
    //   next.endSec]; kind = current). Disabled when next_ is nullopt.
    std::function<void()>                                  onSplit;
    std::function<void()>                                  onMergeLeft;
    std::function<void()>                                  onMergeRight;

    static constexpr int kContentWidth  = 528;
    static int computeContentHeight (const reamix::ui::CustomKindRegistry* registry);

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;

private:
    // Geometry
    static constexpr int kCols      = 4;
    static constexpr int kRows      = 3;
    static constexpr int kPillW     = 120;
    static constexpr int kPillH     = 30;
    static constexpr int kPillGap   = 8;
    static constexpr int kPadding   = 12;
    static constexpr int kHeaderH   = 18;
    static constexpr int kSectionGap = 14;
    static constexpr int kNudgerH   = 28;
    static constexpr int kNudgeBtnW = 36;
    static constexpr int kDeleteH   = 30;

    enum class HotElement
    {
        None,
        Pill,           // index = 0..(builtinCount + customCount + 1) - 1
        StartMinusBeat,
        StartMinusFine, // -0.1 s
        StartPlusFine,  // +0.1 s
        StartPlusBeat,
        EndMinusBeat,
        EndMinusFine,
        EndPlusFine,
        EndPlusBeat,
        Split,          // sesja 100 DEV-029
        MergeLeft,      // sesja 100 DEV-029
        MergeRight,     // sesja 100 DEV-029
        Delete,
    };

    juce::Rectangle<int> pillRect (int kindIdx) const;
    int                  pillCount() const;
    int                  rows()      const;
    bool                 isBuiltinSlot (int idx) const;
    bool                 isCustomSlot  (int idx) const;
    bool                 isAddSlot     (int idx) const;
    juce::String         customIdAtSlot (int idx) const;
    juce::Rectangle<int> startNudgerArea() const;
    juce::Rectangle<int> endNudgerArea()   const;
    juce::Rectangle<int> actionRowArea()   const;  // sesja 100 DEV-029
    juce::Rectangle<int> splitBtnRect()    const;  // sesja 100 DEV-029
    juce::Rectangle<int> mergeLeftBtnRect() const; // sesja 100 DEV-029
    juce::Rectangle<int> mergeRightBtnRect() const;// sesja 100 DEV-029
    juce::Rectangle<int> deleteBtnRect()   const;
    juce::Rectangle<int> nudgerBtnRect (juce::Rectangle<int> area, int slot) const; // 0..3

    HotElement elementAt (juce::Point<int> pos) const;

    bool canSplit()      const noexcept { return (block_.endSec - block_.startSec) >= 1.0; }
    bool canMergeLeft()  const noexcept { return prev_.has_value(); }
    bool canMergeRight() const noexcept { return next_.has_value(); }

    void applyStartNudge (double delta);
    void applyEndNudge   (double delta);
    // Sesja 64 BUG-7 — direction = -1 / +1 step in the beat grid.
    void applyStartBeatStep (int direction);
    void applyEndBeatStep   (int direction);
    void confirmKindBuiltin (reamix::theme::SegmentKind);
    void confirmKindCustom  (const juce::String& id);
    void requestAdd();
    void requestEdit        (const juce::String& id, reamix::ui::CustomKindAction action);
    void confirmDelete();
    // Sesja 100 (DEV-029) — fire callback if predicate-enabled, then dismiss.
    void confirmSplit();
    void confirmMergeLeft();
    void confirmMergeRight();

    juce::String formatHeader() const;
    static juce::String labelFor (reamix::theme::SegmentKind k);

    void closeHost();

    // State
    reamix::ui::UserBlock                  block_;
    double                                  itemDurationSec_ {0.0};
    double                                  beatSec_         {0.5};   // 60/bpm
    std::vector<double>                     beatTimes_;                // sesja 64 BUG-7
    std::optional<reamix::ui::UserBlock>    prev_;
    std::optional<reamix::ui::UserBlock>    next_;

    // ADR-092 sesja 100c — non-owning. Caller (MainComponent) owns lifetime.
    const reamix::ui::CustomKindRegistry*   registry_ { nullptr };

    int                                     focusedPillIdx_  {0};
    HotElement                              hovered_         {HotElement::None};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BlockEditPopover)
};

} // namespace reamix::ui
