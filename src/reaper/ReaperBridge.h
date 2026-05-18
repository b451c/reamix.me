#pragma once

#include "render/Renderer.h"

#include <juce_core/juce_core.h>

#include <optional>
#include <vector>

namespace reamix::ui { struct UserBlock; }

// ReaperBridge — query + mutation layer between JUCE UI and REAPER host state.
//
// ADR-036 D4 + D7 + ADR-037 + ADR-046: this file owns the subset of REAPER API
// calls used to (a) discover the currently-selected media item and (b) insert
// remix clips on the timeline. Time-selection polling is in step 6 (separate
// future addition).
//
// Step-3b scope (ADR-037):
//   - getSelectedItem() — 1:1 port of Lua remix_operations.lua:178-217
//     (update_selected_item).
//
// Step-9 scope (ADR-046, this session):
//   - insertRemixClips() — port of Lua remix_insert.lua:127-167 create_remix_clip
//     (per-clip MediaItem construction) + :286-339 insert_remix_clips_to_project
//     (Undo block + delete-existing + insert + UpdateArrange) non-region branch.
//   - getItemGuid() / getTrackGuid() / findItemByGuid() — used by GroupTracker
//     for Update detection (existing remix lookup by GUID).
//
// Not in scope here: time-selection read (step 6), region-remix insert path
// (step 6 — splits source item, replaces middle, shifts post-region), block-
// assembly mode insert (step 8 — uses same insertRemixClips with user-labeled
// clips for color/name override).

namespace reamix::reaper
{

struct SelectedItem
{
    juce::String name;           // GetTakeName result
    double       durationSec        {0.0};   // GetMediaItemInfo_Value D_LENGTH (length of the clip on the timeline)
    double       sourceFileDuration {0.0};   // GetMediaSourceLength (length of the underlying audio file; ≥ durationSec for trimmed/remix clips)
    double       positionSec        {0.0};   // GetMediaItemInfo_Value D_POSITION
    juce::String sourcePath;     // GetMediaSourceFileName result
    void*        itemPtr  {nullptr};  // opaque MediaItem*
    void*        trackPtr {nullptr};  // opaque MediaTrack*
};

// Returns the first selected media item's metadata + source path, or
// std::nullopt when:
//   - no item is selected in the active project,
//   - the selected item has no active take,
//   - the active take has no PCM source (e.g. empty take).
//
// Safe to call from the message thread; does not touch JUCE or REAPER
// timers itself. Intended to be called at ~100 ms from a juce::Timer.
std::optional<SelectedItem> getSelectedItem();

// ─────────────────────────────────────────────────────────────────────
// Step 6 — Time-selection / region detection (sesja 60)
// ─────────────────────────────────────────────────────────────────────

// REAPER time-selection (loop range). Both endpoints are absolute project
// seconds. Returns std::nullopt when no selection is active (start == end).
struct TimeSelection
{
    double startSec   {0.0};
    double endSec     {0.0};
    double durationSec {0.0};
};

// Port of Lua remix_insert.lua:62-68 get_time_selection. Calls
// `GetSet_LoopTimeRange(false, false, &start, &end, false)`. Returns nullopt
// when end <= start (no time-selection).
std::optional<TimeSelection> getTimeSelection();

// Item-relative region for a given selected item. The region is the overlap
// of the REAPER time-selection with the item, expressed in item-relative
// seconds (0 = item start). Returns nullopt when:
//   - itemDurationSec <= 0,
//   - no time-selection exists,
//   - overlap < 6.0 s (Lua min ~4 bars guard at remix_insert.lua:85),
//   - selection covers entire item (within 0.1 s slack — Lua remix_insert.lua:86-88
//     "use full remix"; entire-item selection falls through to Duration mode).
//
// Port of Lua remix_insert.lua:72-91 get_item_region. Item pointer is not
// required by the Lua semantic — only durationSec + positionSec are used.
struct ItemRegion
{
    double startSec {0.0};   // item-relative (0 .. itemDurationSec)
    double endSec   {0.0};   // item-relative (0 .. itemDurationSec)
};

std::optional<ItemRegion> getItemRegion (double itemDurationSec,
                                          double itemPositionSec);

// ─────────────────────────────────────────────────────────────────────
// Step 9 — Insert/Update primitives (ADR-046)
// ─────────────────────────────────────────────────────────────────────

// GUID query helpers. Return empty juce::String on failure (no SDK symbol,
// null pointer, REAPER returned empty string).
juce::String getItemGuid (void* itemPtr);
juce::String getTrackGuid (void* trackPtr);

// Linear scan of all tracks in the active project for a MediaItem with the
// given GUID. Returns nullptr if not found. O(N tracks × M items).
void* findItemByGuid (const juce::String& itemGuid);

// Read D_POSITION (project-absolute timeline seconds) of an item. Returns
// 0.0 if itemPtr is null or the SDK symbol is unavailable. Used by the
// Region Update path to base absRegionStart calculations on the canonical
// pre-region piece's position rather than the clicked clip's position
// (DEV-038 sesja 67 fix follow-up).
double getItemPosition (void* itemPtr);

// Resolve a MediaTrack* for a track GUID. Returns nullptr if not found.
void* findTrackByGuid (const juce::String& trackGuid);

// Specification for an insert/update operation. Caller computes everything;
// insertRemixClips wraps the mutation in Undo_BeginBlock + PreventUIRefresh.
struct InsertSpec
{
    void*        trackPtr           {nullptr};   // target track (where clips go)
    double       basePositionSec    {0.0};       // timeline base for clips[i].timelineStartSec
    juce::String originalSourcePath;             // forwarded to PCM_Source_CreateFromFile
    const std::vector<reamix::render::EditClip>* clips {nullptr};

    // Items to delete BEFORE inserting (Update mode + replace-source mode):
    // - For Update: list of existing remix item GUIDs from a prior insert.
    // - For replace-source: the user-selected source item (when it IS the source).
    // Both can be present simultaneously (rare but valid).
    std::vector<juce::String> existingItemGuidsToDelete;
    void*                     sourceItemToDelete {nullptr};

    // Sesja 100b — DEV-049 Insert-time decoration knobs. Splice point
    // markers + render region are added inside the same Undo block as
    // the items themselves so a single Cmd+Z removes both clips and
    // markers atomically. Caller passes splice timestamps relative to
    // basePositionSec (matching `clips[i].timelineStartSec` semantics);
    // the bridge derives the render-region end from clips.back() so
    // there's no separate length field to keep in sync.
    bool                      addSpliceMarkers {false};
    bool                      addRenderRegion  {false};
    std::vector<double>       spliceTimesRel;
    juce::String              renderRegionName;

    // Undo-block label shown in REAPER's Undo history.
    juce::String undoLabel { "Insert remix clips" };
};

struct InsertResult
{
    bool                     ok            {false};
    juce::String             errorMessage;
    int                      clipCount     {0};
    std::vector<juce::String> insertedItemGuids;
};

// Performs the insert/update batch. Side effects:
//   - Reads/mutates REAPER MediaItem state (creates new items, deletes specified existing items).
//   - Wraps everything in Undo_BeginBlock + PreventUIRefresh + UpdateArrange + Undo_EndBlock.
//   - Sets per-item D_POSITION/D_LENGTH/D_FADEINLEN/D_FADEOUTLEN/D_FADEINSHAPE/D_FADEOUTSHAPE/
//     D_STARTOFFS/B_LOOPSRC/I_CUSTOMCOLOR/P_NAME per ADR-046 § Decision Q1.
//
// Color scheme (ADR-046): alternating gray (#3a3a3a even, #4a4a4a odd) per
// clip index. Block Assembly mode (step 8) will override via clip.kind once
// EditClip carries that field — for now, all clips get gray colors.
//
// Take name format: "Remix M:SS-M:SS" using clip.sourceStartSec / sourceEndSec.
//
// Safe to call from the message thread. Synchronous; returns when REAPER has
// finished the batch.
InsertResult insertRemixClips (const InsertSpec& spec);

// ─────────────────────────────────────────────────────────────────────
// Step 6 — Region remix insert (splice INTO original track) — sesja 60
// ─────────────────────────────────────────────────────────────────────
//
// Region remix is mechanically different from the non-region path: instead of
// replacing the source item with remix clips, it splits the source item at
// region boundaries, deletes only the middle (region) portion, inserts remix
// clips in its place, and shifts the post-region tail by the duration delta.
// Pre-region audio is left untouched.
//
// Lua port: remix_insert.lua:176-236 insert_region_remix.
//
// Region branch supports both Insert (fresh splice) and Update (replace
// existing Region remix in place):
//   - Insert path (DEV-038 N/A): existingItemGuidsToDelete empty,
//     preRegionGuid empty → standard split + delete + insert + shift.
//   - Update path (DEV-038 sesja 67): existingItemGuidsToDelete carries the
//     prior inserted-clip GUIDs + post-region GUID; preRegionGuid is the
//     original sourceItem GUID (= regionGroup->sourceGuid, which the pre-
//     region piece keeps post-split per ADR-056 D2); originalItemDurationSec
//     is the pre-Insert item-on-timeline D_LENGTH (= picked->durationSec at
//     first Insert time, captured by caller into RegionGroup payload).
//     Update path: (a) delete existingItemGuidsToDelete + post-region GUID;
//     (b) findItemByGuid(preRegionGuid) → restore D_LENGTH to
//     originalItemDurationSec; (c) sourceItemToSplit set to the restored
//     pre-region piece; (d) standard split + insert + shift on NEW absRegion
//     bounds. Caller (transportBar_.onInsert) then overwrites the
//     RegionGroup entry with new GUIDs.
struct RegionInsertSpec
{
    void*        trackPtr            {nullptr};
    void*        sourceItemToSplit   {nullptr};   // the item being spliced
    double       absRegionStartSec   {0.0};       // project-absolute time
    double       absRegionEndSec     {0.0};
    juce::String originalSourcePath;
    const std::vector<reamix::render::EditClip>* clips {nullptr};
    juce::String undoLabel { "Region remix: splice into track" };

    // DEV-038 sesja 67 — Update flow inputs. All optional (default values
    // skip Update path → Insert behaves as before). Caller populates these
    // when an existing RegionGroup is found for the selected item.
    std::vector<juce::String> existingItemGuidsToDelete;
    juce::String              preRegionGuid;            // = group->sourceGuid
    double                    originalItemDurationSec {0.0};

    // ADR-057 sesja 68 step 2e — boundary fade overlap. RemixPipeline extended
    // first/last EditClip source ranges by these amounts (each side). Insert
    // pipeline overlaps pre-region/first-WAV by leadInSec and last-WAV/post-
    // region by leadOutSec, applies equal-power crossfade. With sample-exact
    // content match (RemixPipeline override) the crossfade preserves full vol.
    double                    boundaryLeadInSec  {0.0};
    double                    boundaryLeadOutSec {0.0};

    // Sesja 100b — DEV-049 Insert-time decoration knobs. Mirror of
    // InsertSpec fields. Splice times here are relative to
    // absRegionStartSec (= start of the inserted region on the timeline).
    //
    // originalItemPositionSec / originalItemDurationSec describe the
    // pre-split source item bounds (= picked.positionSec / .durationSec
    // for fresh Insert; = pre-region piece position / original duration
    // for Update). The render region uses these to span the WHOLE
    // post-remix track including unchanged pre-region + post-region
    // pieces — the user's render output covers the entire new song,
    // not just the spliced middle. Sesja 100b user smoke verbatim:
    // *"region nadal zaznacza tylko wstawione regiony a nie caly nowy
    // utwor"*. Splice point markers stay confined to the spliced region
    // (their semantics is per-cut).
    bool                      addSpliceMarkers {false};
    bool                      addRenderRegion  {false};
    std::vector<double>       spliceTimesRel;
    juce::String              renderRegionName;
    double                    originalItemPositionSec {-1.0};  // -1 = unset → fall back to absRegion span
};

struct RegionInsertResult
{
    bool         ok          {false};
    juce::String errorMessage;
    int          clipCount   {0};
    // ADR-056 (sesja 66) — REAPER GUIDs of every inserted clip, in order.
    // Captured for regionGroupTracker::saveRegionGroup so applySelectedItem
    // can later detect inserted-clip clicks and dispatch to the original-
    // item view (UX parity with Duration mode's groupTracker).
    std::vector<juce::String> insertedItemGuids;
};

RegionInsertResult insertRegionRemixClips (const RegionInsertSpec& spec);

// ─────────────────────────────────────────────────────────────────────
// Step 8 — UserBlock persistence (ADR-051, sesja 61)
// ─────────────────────────────────────────────────────────────────────
//
// User-marked sections live per-item in P_EXT:reamix_blocks (JSON encoded).
// Pattern reused from ADR-036 ⚑ P2 GroupTracker which uses GetProjExtState
// (project-level). UserBlocks are per-item so use GetSetMediaItemInfo_String
// "P_EXT:reamix_blocks" instead. Survives save/reopen.
// Note: key uses underscore not dot — matches SneakPeak (P_EXT:SneakPeak_Dynamics)
// and Lua RemixTool (P_EXT:RemixTool); a sesja-61 dot variant crashed REAPER.

std::vector<reamix::ui::UserBlock> loadUserBlocks (void* itemPtr);

void saveUserBlocks (void* itemPtr,
                     const std::vector<reamix::ui::UserBlock>& blocks);

// Sesja 100b — DEV-049 splice markers + render region.
//
// Optional REAPER project markers/regions added after Insert so the user
// can navigate splice points (Shift+arrow → next marker) and quickly
// render the entire new clip-set via File → Render → Source: Region
// without manual time-selection. Both surfaces are user-toggleable in
// SettingsPopover.
//
// `clearReamixMarkersInRange` removes ALL prior reamix-tagged markers +
// regions whose position lands inside [startSec, endSec]. Used on
// Insert/Update prior to adding the new ones so re-running an Insert in
// the same area doesn't accumulate stale markers from prior tries.
//
// Naming convention: every reamix-managed marker/region carries the
// "reamix:" prefix (case-sensitive) so the cleanup pass can identify
// them without colliding with user-authored markers in the same range.
void clearReamixMarkersInRange (double startSec, double endSec);

// Add a point marker at every (basePositionSec + spliceTimeRel) for each
// entry in `spliceTimesRel`. No-op when vector empty. Markers carry the
// "reamix:splice" name + Info-blue colour so they're distinguishable
// from user markers without being too loud.
void addSpliceMarkers (double basePositionSec,
                       const std::vector<double>& spliceTimesRel);

// Add a region spanning [startSec, endSec] named "reamix:render <suffix>"
// (suffix typically = source file name) so the user can identify which
// Insert produced it via the Region Render Matrix.
void addRenderRegion (double startSec, double endSec,
                      const juce::String& nameSuffix);

} // namespace reamix::reaper
