#pragma once

#include <juce_core/juce_core.h>

#include <optional>
#include <vector>

// BlocksGroupTracker — third member of the trio (GroupTracker for Duration,
// RegionGroupTracker for Region, BlocksGroupTracker for Block Assembly).
// Tracks the source-item identity behind each Block Assembly Insert so the
// user can click any inserted Blocks clip and have the plugin restore the
// Blocks tab + queue + userBlocks view (UX parity with Region mode's
// regionGroupTracker per ADR-056).
//
// DEV-041 (sesja 95) — closes user-flagged sesja-66-close gap: after Block
// Assembly + Insert + click empty area + re-click any inserted clip, plugin
// silently fell through to Duration tab because applySelectedItem only
// canonicalized via Region's tracker. lastModeByPath_ keyed on the inserted
// clip's NEW GUID (not the canonical original) so the wasBlocks check missed.
// Pattern mirrors RegionGroupTracker exactly.
//
// Schema (sesja 95 ADR-085 implementation; sister keyspace to ADR-056 +
// ADR-046 Q2):
//   - SECTION="reamix.me" (shared with Duration groupTracker + Region group +
//     DockableWindow)
//   - INDEX_KEY="blocks_group_index" — newline-separated source GUIDs.
//   - "blocks_group:<source_guid>" — payload pipe-separated:
//     "{trackGuid}|{basePosition:.12f}|{targetSec:.12f}|{tmpWavPath}|
//      {itemGuid_1,itemGuid_2,...}|{originalItemDurationSec:.12f}"
//
// Distinct INDEX_KEY + "blocks_group:" prefix (vs Duration's "group_index" +
// "group:" and Region's "region_group_index" + "region_group:") prevents
// Block groups from polluting the other two lookup namespaces.
//
// Note: queue + userBlocks are NOT stored in this tracker — those live in
// the in-memory session cache (`blocksSessionByPath_` per ADR-052) keyed by
// sourcePath, which already survives item deselect → reselect within the
// same plugin session. This tracker only carries the GUID identity the
// applySelectedItem dispatch needs to canonicalize a clip click back to the
// original source identity.
//
// All API calls are no-ops when REAMIX_WITH_REAPER_IO is undefined (host-
// less build for unit tests).

namespace reamix::reaper::blocksGroupTracker
{

struct BlocksGroup
{
    juce::String              sourceGuid;
    juce::String              trackGuid;
    double                    basePosition  {0.0};

    // Block Assembly context — captured at Insert time for symmetry with
    // RegionGroupTracker. targetSec + tmpWavPath enable RemixCache lookup
    // parity if a future Update flow is added (Blocks currently does not
    // support post-Insert Update — clicking inserted clip restores view
    // but does not promote next Insert into Update; this matches Lua
    // behavior remix_blocks.lua).
    double                    targetSec     {0.0};
    juce::String              tmpWavPath;

    std::vector<juce::String> itemGuids;

    // Pre-Insert item-on-timeline length, captured from picked->durationSec
    // before the first split. 0.0 for legacy entries (none currently — this
    // is a sesja-95 NEW tracker so all entries written by this code path
    // carry the field). Reserved for future Update flow symmetry with
    // RegionGroupTracker (sesja-67 ADR DEV-038 precedent).
    double                    originalItemDurationSec {0.0};
};

// Look up a Blocks group by item GUID. Two paths:
//   1. Direct: itemGuid IS a source-guid → returns its group.
//   2. Indirect: itemGuid appears in some group's itemGuids → returns that group.
// Returns std::nullopt if no Blocks group is found in the active project.
std::optional<BlocksGroup> findGroupForSelectedItem (const juce::String& itemGuid);

// Persist a Blocks group. Idempotent: re-saving with the same sourceGuid
// overwrites the per-group entry and dedups the index.
void saveBlocksGroup (const BlocksGroup& group);

// Remove a Blocks group entry (clears per-group payload + filters source
// GUID out of the index). Used when user explicitly dismisses Block view
// or when Insert places zero clips.
void removeBlocksGroup (const juce::String& sourceGuid);

// Returns the current Blocks-group source-GUID index. Empty list if no
// groups stored or REAPER IO disabled.
std::vector<juce::String> loadBlocksGroupIndex();

// Internal helper exposed for testing / diagnostics. Returns the raw
// payload for a given source GUID (or empty string if missing).
juce::String loadBlocksGroupRaw (const juce::String& sourceGuid);

} // namespace reamix::reaper::blocksGroupTracker
