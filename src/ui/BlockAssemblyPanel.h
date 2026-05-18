#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "Theme.h"
#include "UserBlock.h"
#include "CustomKindRegistry.h"

#include <functional>
#include <set>
#include <vector>

// BlockAssemblyPanel — phase-6 step 8 panel (ADR-051) replacing DurationPanel
// in the layout when MainComponent.appMode_ == ModeTabs::Mode::Blocks.
//
// Layout (sesja 61 phase C — skeleton):
//   ┌────────────────────────────────────────────────────────────────┐
//   │ AVAILABLE BLOCKS ·  N marked                              16h  │
//   │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ...                  46h  │
//   │ │INTRO │ │VERSE │ │CHORUS│ │BRIDGE│  scrollable horizontal     │
//   │ └──────┘ └──────┘ └──────┘ └──────┘                            │
//   │   gap                                                      6h  │
//   │ ARRANGEMENT TIMELINE  ·  N blocks · M:SS                  16h  │
//   │ ┌────────────────────────────────────────────────────────┐    │
//   │ │ queue placeholder (mockup v2-alternates 14h ruler        56h│
//   │ │ + 42h tile row — full UI in phase F)                    │    │
//   │ └────────────────────────────────────────────────────────┘    │
//   │   gap                                                      6h  │
//   │ Re-analyze CTA banner (hidden when clean — phase I)       18h  │
//   └────────────────────────────────────────────────────────────────┘
//   Total height target: ~170 px.
//
// Phase C deliverables (this file):
//   • paleta cards rendered from userBlocks_ (click → onCardClick(idx)).
//   • queue placeholder + index list (paint stub) so MainComponent can route
//     onCardClick to a queue mutation in phase F.
//   • dirty banner reserved height — phase I wires the show/hide + CTA.
//
// Phase F wires real queue UI; phase G wires re-roll + audition; phase H
// wires Edit-mode launch from card right-click; phase I wires dirty banner.

namespace reamix::ui
{

class BlockAssemblyPanel : public juce::Component
{
public:
    BlockAssemblyPanel();
    ~BlockAssemblyPanel() override = default;

    // Set the current user-marked blocks (drives paleta cards rendering +
    // compatibility hint colour vs last-queued).
    void setUserBlocks (std::vector<reamix::ui::UserBlock>);

    // ADR-092 sesja 100c — registry consulted at paint time (paleta cards +
    // queue tiles use kindDisplay() for custom kindy name + color overrides).
    // Caller (MainComponent) owns lifetime; pointer must outlive panel.
    void setCustomKindRegistry (const reamix::ui::CustomKindRegistry* reg) noexcept
    {
        customKindRegistry_ = reg;
        repaint();
    }

    // Set the current queue (vector of userBlocks_ indices, in display order).
    // Phase C uses this for the queue text-summary; phase F renders timeline.
    void setQueue (std::vector<int> queueIndices);

    // ADR-051 phase F — quality score per junction (0..1). Size must equal
    // queue.size() - 1. Empty → seams render in neutral gray (pre-analysis
    // state). Phase J populates from BlockAssembly::computeBlockCompatibility
    // post-Analyze.
    void setSeamQualities (std::vector<float> qualities);

    // ADR-051 phase G — junction variation index per junction (0 = primary
    // top-1 splice, 1+ = re-rolled to k-th alternative). Size = queue.size()
    // - 1. Drives the "²" badge on re-rolled seam pills.
    void setJunctionVariations (std::vector<int> variations);

    // Set the dirty / re-analyze banner state. Phase I wires this from
    // MainComponent on any user-block mutation; today the setter just
    // re-paints (the full banner UI lands phase I).
    void setDirty (bool yes);

    // Sesja 64 — Assemble button busy state during arrangement compute.
    // When busy: button label replaced (e.g. "Computing…"), button ignores
    // clicks. Cleared by setAssembleBusy(false).
    void setAssembleBusy (bool busy, juce::String label = {});
    bool isAssembleBusy  () const noexcept { return assembleBusy_; }

    // Fired when user clicks a paleta card. idx indexes userBlocks_.
    std::function<void (int blockIdx)> onCardClicked;

    // Fired when user right-clicks a paleta card. MainComponent inspects
    // current paletteSelected_ via getPaletteSelection() to decide
    // single-tile vs batch (multi-select) context menu (DEV-058 (a) + (b)
    // sesja 100d). When the right-clicked card IS in the selection AND
    // selection.size() > 1 → batch menu (Delete N blocks / Change kind for N
    // / etc); otherwise single-tile menu (Rename / Change kind / Delete)
    // and the click also resets the selection to {idx}.
    std::function<void (int blockIdx, juce::Point<int> screenPos)> onCardContextMenu;

    // DEV-058 (b) sesja 100d — paleta multi-select. Plain click selects
    // single block + appends to queue (existing onCardClicked behavior).
    // Cmd+click toggles a block in/out of selection without queue append.
    // Shift+click extends from anchor (last selected). Selection drives
    // visual highlight (Accent border) + status bar count + right-click
    // batch menu. Cleared by clearPaletteSelection() (Esc / click empty
    // area / mode change).
    void setPaletteSelection      (std::set<int> ids);
    const std::set<int>& getPaletteSelection() const noexcept { return paletteSelected_; }
    void clearPaletteSelection    ();
    std::function<void (const std::set<int>& blockIndices)> onPaletteSelectionChanged;

    // DEV-058 (c) sesja 100d — fired on mouseUp after a paleta drag-drop
    // gesture exceeded the click-vs-drag threshold. fromBlockIdx is the
    // userBlocks_ index dragged; toBlockIdx is the userBlocks_ index of
    // the target slot (drop position). MainComponent reorders userBlocks_
    // (insertion-style) and remaps userBlocksQueue_ indices accordingly.
    // No-op when fromBlockIdx == toBlockIdx (drop on self). Multi-select
    // with drag is currently single-tile-only — selection is cleared on
    // drag start so the user gets predictable single-tile reorder UX.
    std::function<void (int fromBlockIdx, int toBlockIdx)> onPaletteReorder;

    // DEV-077 (NEW sesja 100d) — fired on mouseUp when a paleta drag
    // ended inside the queue row area. blockIdx is the dragged block;
    // queuePos is the insertion gap (0..queue_.size()). Distinct from
    // onPaletteReorder (paleta→paleta) and onCardClicked (append-to-end
    // via plain click) so MainComponent can choose precise insertion
    // semantic vs append.
    std::function<void (int blockIdx, int queuePos)> onPaletteToQueueInsert;

    // Fired when user right-clicks a queue tile. Surfaces the Remove /
    // Move Up / Move Down menu (MainComponent owns the menu).
    std::function<void (int queuePos, juce::Point<int> screenPos)> onQueueTileContextMenu;

    // Fired when user left-clicks a queue tile. Phase G: auditions the block
    // from the click position. fractionInTile is in [0, 1]: 0.0 = play from
    // the block's start, 1.0 = play from the very end (effectively no
    // audio). Sesja 100b extended the signature with fractionInTile per
    // user smoke verbatim *"kiedy klikne w dane miejsce danego bloku to
    // playhead odtwarza z tego miejsca"* — clicking at 50% of the tile
    // width plays from the midpoint of the block, etc.
    std::function<void (int queuePos, double fractionInTile)> onQueueTileAudition;

    // DEV-030 (sesja 100b) — fired on mouseUp after a drag-drop gesture
    // exceeded the click-vs-drag threshold. `fromPos` is the queue index
    // of the dragged tile; `toPos` is the gap index where it should land
    // (0..queue.size(); inserting at toPos shifts existing tiles right).
    // No-op when fromPos == toPos OR fromPos == toPos - 1 (drop into the
    // same slot). MainComponent owns queue_ + persistence + remix
    // invalidation.
    std::function<void (int fromPos, int toPos)> onQueueReorder;

    // Fired when user right-clicks a seam pill. Phase G: opens the Try
    // different splice / Reset to best / Audition junction menu.
    std::function<void (int junctionIdx, juce::Point<int> screenPos)> onSeamContextMenu;

    // Fired when user left-clicks a seam pill. Phase G: auditions ~4 beats
    // around the algorithm-resolved splice.
    std::function<void (int junctionIdx)> onSeamAudition;

    // Fired when user clicks the dedicated "Assemble" button in queue
    // header (sesja-61 close UX fix — was previously the dirty-banner CTA;
    // user reported "Re-analyze" naming collided conceptually with the
    // source-Analyze button at top of plugin). Banner is now purely
    // informational; this is the explicit action verb.
    std::function<void()> onAssembleClicked;

    // Fired when the panel's preferred height changes (paleta wrapped to a
    // different row count). MainComponent re-fires its own resized() so the
    // mode-panel slot grows / shrinks. Sesja 61 hot-fix: paleta originally
    // overflowed narrow windows; wrap-to-rows is the Lua remix_blocks.lua
    // pattern (L142-147 auto-wrap).
    std::function<void()> onPreferredHeightChanged;

    // Layout helper for MainComponent::resized. Dynamic — depends on paleta
    // row count which depends on getWidth() and userBlocks_.size().
    int getPreferredHeight() const;

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;

private:
    // Geometry constants
    static constexpr int kHeaderRowH        = 16;
    static constexpr int kCardRowH          = 46;
    static constexpr int kCardRowGap        = 4;   // vertical gap between wrapped rows
    static constexpr int kCardW             = 110;
    static constexpr int kCardGap           = 6;
    static constexpr int kQueueRowH         = 56;
    static constexpr int kBannerH           = 18;
    static constexpr int kSectionGap        = 6;
    static constexpr int kHorizontalPadding = 12;
    static constexpr int kPaletteMinH       =
        kHeaderRowH + kCardRowH;
    static constexpr int kQueueAndBannerH   =
        kSectionGap + kHeaderRowH + kQueueRowH + kSectionGap + kBannerH;
    // Paleta height = kHeaderRowH + N rows × (kCardRowH + kCardRowGap) - kCardRowGap;
    // grows with cardRowsNeeded(). Cap at kMaxRows to prevent the panel from
    // dominating the window when user has many blocks (rare, but possible).
    static constexpr int kMaxPaletteRows    = 4;

    // Region helpers
    juce::Rectangle<int> paletteHeaderArea() const;
    juce::Rectangle<int> paletteRowArea()    const; // multi-row region
    juce::Rectangle<int> queueHeaderArea()   const;
    juce::Rectangle<int> queueRowArea()      const;
    juce::Rectangle<int> bannerArea()        const;

    // Number of paleta rows needed to fit userBlocks_ at current width.
    // Min 1, max kMaxPaletteRows. 0 returned only when userBlocks_ empty.
    int cardRowsNeeded() const;
    int cardsPerRow()    const;
    int paletteHeight()  const; // computed from rows used

    // Sesja-61 close UX fix — dedicated Assemble button rect inside the queue
    // header row. Right-aligned. Empty rect when queue.size() < 2 (button
    // disabled / hidden — no point assembling 0-1 blocks).
    juce::Rectangle<int> assembleBtnRect() const;
    bool                 assembleBtnEnabled() const noexcept;

    // Paleta card rect for a given userBlocks_ index. Returns empty rect when
    // off-screen (horizontal scroll TBD; phase C clips to right edge).
    juce::Rectangle<int> cardRect (int blockIdx) const;
    int                  cardHitTest (juce::Point<int> pos) const;

    // Queue tile rect for a given queue position. Width proportional to
    // userBlocks_[queue_[pos]].duration_sec / total. Empty when out of range.
    juce::Rectangle<int> queueTileRect (int queuePos) const;
    int                  queueTileHitTest (juce::Point<int> pos) const;

    // Seam pill rect for junction j (between queue tiles j and j+1).
    juce::Rectangle<int> seamPillRect (int junctionIdx) const;
    int                  seamPillHitTest (juce::Point<int> pos) const;

    // Total queue duration in seconds (for proportional tile widths).
    double               queueTotalDuration() const;

    // Compatibility colour vs last queued — used for card border tint.
    // Phase C placeholder: returns std::nullopt (real lookup phase F).
    std::optional<juce::Colour> compatColour (int blockIdx) const;

    // Paint helpers
    void paintHeader      (juce::Graphics&, juce::Rectangle<int> area,
                           const juce::String& title, const juce::String& meta);
    void paintPaletteRow  (juce::Graphics&, juce::Rectangle<int> area);
    void paintQueueRow    (juce::Graphics&, juce::Rectangle<int> area);
    void paintBanner      (juce::Graphics&, juce::Rectangle<int> area);

    // DEV-077 sesja 100d — drag overlay rendered LAST in paint() so the
    // ghost tile + insertion line are always on top of every panel
    // section (paleta tiles, queue tiles, queue header, banner). Without
    // this dedicated final pass the ghost was occluded by paintQueueRow
    // when the cursor crossed into queueRowArea.
    void paintDragGhost   (juce::Graphics&);

    static juce::String   kindLabel (reamix::theme::SegmentKind k);

    // State
    std::vector<reamix::ui::UserBlock> userBlocks_;

    // ADR-092 sesja 100c — non-owning. Caller (MainComponent) owns lifetime.
    const reamix::ui::CustomKindRegistry* customKindRegistry_ { nullptr };
    std::vector<int>                   queue_;
    std::vector<float>                 seamQualities_;        // size = queue_.size() - 1; empty pre-analysis
    std::vector<int>                   junctionVariations_;   // size = queue_.size() - 1; phase G
    bool                               dirty_       { false };
    bool                               assembleBusy_ { false };  // sesja 64
    juce::String                       assembleBusyLabel_;       // sesja 64
    int                                hoveredCardIdx_       { -1 };
    int                                hoveredQueueTileIdx_  { -1 };
    int                                hoveredSeamIdx_       { -1 };
    bool                               hoveredAssembleBtn_   { false };

    // DEV-030 (sesja 100b) — queue drag-drop state. dragQueueIdx_ = the
    // tile picked up on mouseDown; -1 = no active gesture. Audition vs
    // reorder is decided on mouseUp by comparing the cursor distance from
    // the mouseDown position against kQueueDragThresholdPx — under the
    // threshold and we treat it as a click (audition); over, we fire
    // onQueueReorder(dragQueueIdx_, dragInsertionPos_). The insertion
    // position is recomputed on every mouseDrag tick from the cursor X
    // relative to existing tile rects.
    int                 dragQueueIdx_         { -1 };
    juce::Point<int>    dragQueueStartPos_;
    juce::Point<int>    dragQueueCurrentPos_;
    int                 dragQueueInsertionPos_{ -1 };
    bool                dragQueueActive_      { false }; // crossed threshold
    static constexpr int kQueueDragThresholdPx = 5;

    int  computeInsertionPos (int xLocal) const;

    // DEV-058 (b) sesja 100d — paleta multi-select state. Selection drives
    // paint highlight + right-click batch menu. paletteSelectionAnchor_
    // is the last single-clicked / Cmd-clicked block; Shift+click ranges
    // from anchor to the new clicked block.
    std::set<int>       paletteSelected_;
    int                 paletteSelectionAnchor_ { -1 };

    // DEV-058 (c) sesja 100d — paleta drag-reorder state. Mirror of queue
    // drag state for the paleta row(s). Click vs drag decided in mouseUp
    // by kPaletteDragThresholdPx; under threshold = onCardClicked
    // (existing append-to-queue), over = onPaletteReorder OR (DEV-077)
    // onPaletteToQueueInsert when cursor crossed into queueRowArea. Only
    // single-tile reorder for v1 — selection cleared on drag start.
    int                 dragPaletteBlockIdx_   { -1 };
    juce::Point<int>    dragPaletteStartPos_;
    juce::Point<int>    dragPaletteCurrentPos_;
    int                 dragPaletteInsertTargetIdx_ { -1 }; // userBlocks_ index of drop slot
    bool                dragPaletteActive_     { false };
    bool                dragPaletteClickModified_ { false }; // Cmd / Shift on mouseDown — pure selection, no reorder
    // DEV-077 (NEW sesja 100d) — set when the live drag cursor sits
    // inside queueRowArea(). dragPaletteQueueInsertPos_ is the gap index
    // 0..queue_.size() computed from cursor X. mouseUp branches on this.
    bool                dragPaletteCrossingToQueue_  { false };
    int                 dragPaletteQueueInsertPos_   { -1 };
    static constexpr int kPaletteDragThresholdPx = 5;

    int  computePaletteInsertTarget (juce::Point<int> pos) const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BlockAssemblyPanel)
};

} // namespace reamix::ui
