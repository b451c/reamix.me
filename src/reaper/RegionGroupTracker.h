#pragma once

#include <juce_core/juce_core.h>

#include <optional>
#include <vector>

// RegionGroupTracker — sister module to GroupTracker (ADR-046, Duration mode).
// Tracks the source-item identity + remix context behind each Region Insert
// so the user can click any inserted Region clip and have the plugin restore
// the full remix view (Region tab + scrim + slider target + Remix variant
// with splice markers + cached tmp WAV path) — UX parity with Duration mode.
//
// Pre-fix UX audit (sesja 66): closes DEV-035 candidate from sesja 60 comment
// in MainComponent.cpp:564-566 ("Recovery on inserted remix clips is
// intentionally NOT supported in this iteration — it requires Region-Insert
// group tracking analogous to Duration mode's groupTracker"). Plan B+
// scope expansion driven by user pushback that single-clip view (per-item
// state without group tracker) loses remix context.
//
// Schema (ADR-056 sesja 66; sister keyspace to ADR-046 Q2):
//   - SECTION="reamix.me" (shared with Duration groupTracker + DockableWindow)
//   - INDEX_KEY="region_group_index" — newline-separated source GUIDs.
//   - "region_group:<source_guid>" — payload pipe-separated:
//     "{trackGuid}|{basePosition:.12f}|{regionStartSec:.12f}|
//      {regionEndSec:.12f}|{targetSec:.12f}|{tmpWavPath}|
//      {itemGuid_1,itemGuid_2,...}|{originalItemDurationSec:.12f}"
//
// DEV-038 sesja 67 — added trailing originalItemDurationSec field to support
// Region Update flow: restoring pre-region piece's D_LENGTH on Update requires
// knowing the pre-Insert item-on-timeline length (which can differ from
// sourceFileDuration when the source item was trimmed before Insert). Parser
// is defensive — legacy 7-field entries return 0.0 here, and Update path
// skips D_LENGTH restore when originalItemDurationSec <= 0 (falls back to
// fresh-Insert behavior for legacy entries; next Insert rewrites with new
// schema).
//
// Distinct INDEX_KEY + "region_group:" prefix (vs Duration's "group_index" +
// "group:") prevents Region groups from polluting Duration's lookup namespace.
//
// All API calls are no-ops when REAMIX_WITH_REAPER_IO is undefined (host-
// less build for unit tests).

namespace reamix::reaper::regionGroupTracker
{

struct RegionGroup
{
    juce::String              sourceGuid;
    juce::String              trackGuid;
    double                    basePosition  {0.0};

    // Region context — captured at Insert time so re-attach can restore
    // identical UI state (slider target + selection scrim + RemixCache key).
    double                    regionStartSec {0.0};
    double                    regionEndSec   {0.0};
    double                    targetSec      {0.0};
    juce::String              tmpWavPath;     // For RemixCache lookup parity

    std::vector<juce::String> itemGuids;

    // DEV-038 sesja 67 — pre-Insert item-on-timeline length, used by Update
    // path to restore pre-region piece's D_LENGTH after deleting the prior
    // inserted clips + post-region piece. Captured in transportBar_.onInsert
    // Region branch from picked->durationSec BEFORE the first split. 0.0 for
    // legacy entries written before sesja 67 (Update path treats as "fresh
    // Insert" fallback).
    double                    originalItemDurationSec {0.0};
};

// Look up a Region group by item GUID. Two paths:
//   1. Direct: itemGuid IS a source-guid → returns its group.
//   2. Indirect: itemGuid appears in some group's itemGuids → returns that group.
// Returns std::nullopt if no Region group is found in the active project.
std::optional<RegionGroup> findGroupForSelectedItem (const juce::String& itemGuid);

// Persist a Region group. Idempotent: re-saving with the same sourceGuid
// overwrites the per-group entry and dedups the index.
void saveRegionGroup (const RegionGroup& group);

// Remove a Region group entry (clears per-group payload + filters source
// GUID out of the index). Used when user explicitly dismisses a Region
// remix or when Insert places zero clips.
void removeRegionGroup (const juce::String& sourceGuid);

// Returns the current Region-group source-GUID index. Empty list if no
// groups stored or REAPER IO disabled.
std::vector<juce::String> loadRegionGroupIndex();

// Internal helper exposed for testing / diagnostics. Returns the raw
// payload for a given source GUID (or empty string if missing).
juce::String loadRegionGroupRaw (const juce::String& sourceGuid);

} // namespace reamix::reaper::regionGroupTracker
