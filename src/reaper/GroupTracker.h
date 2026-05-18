#pragma once

#include <juce_core/juce_core.h>

#include <optional>
#include <vector>

// GroupTracker — REAPER project-level mapping between source items and
// their remix clip groups. Persistence via GetProjExtState/SetProjExtState
// (SECTION = "reamix.me", per ADR-046 § Decision Q2; matches DockableWindow
// precedent in src/ui/DockableWindow.h:43,108,150,275).
//
// Schema (originally 1:1 Lua port, references/lua-source/remix_groups.lua:140-150;
// extended sesja 100b with optional 4th field for Duration-mode RemixCache hit
// after Cmd+Z / item-switch — mirror of RegionGroupTracker + BlocksGroupTracker):
//   - SECTION="reamix.me"
//   - INDEX_KEY="group_index" — newline-separated list of source GUIDs.
//   - "group:<source_guid>" — payload "{track_guid}|{base_position:.12f}|
//     {item_guid_1,item_guid_2,...}|{target_sec:.12f}" pipe-separated.
//     The 4th field is OPTIONAL for backwards compatibility — entries
//     written before sesja 100b have only 3 fields; loader treats missing
//     target_sec as 0.0 (sentinel = unknown, falls back to legacy lookup).
//
// Why "reamix.me" instead of Lua's "RemixToolTimeline": Lua's value is a
// legacy from the TCP-server architecture; new C++ port has no migration
// concern (different runtime, different file layout). "reamix.me" matches the
// existing DockableWindow ExtState SECTION used for dock-state persistence.
//
// All API calls are no-ops when REAMIX_WITH_REAPER_IO is undefined (host-
// less build for unit tests).

namespace reamix::reaper::groupTracker
{

struct RemixGroup
{
    juce::String              sourceGuid;
    juce::String              trackGuid;
    double                    basePosition {0.0};
    std::vector<juce::String> itemGuids;
    // Sesja 100b — Duration target the inserted clips were rendered at.
    // 0.0 = sentinel for legacy entries (pre-sesja-100b). Used by
    // applySelectedItem to pre-prime targetByPath_ on post-Insert clip
    // clicks so RemixCache returns the cached output immediately and
    // the waveform flips to Remix variant — UX parity with Region /
    // Blocks group dispatch (which already carry their target).
    double                    targetSec    {0.0};
};

// Look up a group by item GUID. Two paths:
//   1. Direct: itemGuid IS a source-guid → returns its group.
//   2. Indirect: itemGuid appears in some group's itemGuids → returns that group.
// Returns std::nullopt if no group is found in the active project.
std::optional<RemixGroup> findGroupForSelectedItem (const juce::String& itemGuid);

// Persist a group. Idempotent: re-saving with the same sourceGuid overwrites
// the per-group entry and dedups the index.
void saveRemixGroup (const RemixGroup& group);

// Remove a group entry (clears per-group payload + filters source GUID out
// of the index). Used when an Insert places zero clips (degenerate case).
void removeRemixGroup (const juce::String& sourceGuid);

// Returns the current source-GUID index. Empty list if no groups stored
// or REAPER IO disabled.
std::vector<juce::String> loadRemixGroupIndex();

// Internal helper exposed for testing/diagnostics. Returns the raw payload
// for a given source GUID (or empty string if missing).
juce::String loadRemixGroupRaw (const juce::String& sourceGuid);

} // namespace reamix::reaper::groupTracker
