#include "reaper/ReaperBridge.h"
#include "ui/UserBlock.h"

#if REAMIX_WITH_REAPER_IO
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"
#endif

#include <cmath>
#include <cstring>
#include <optional>

namespace reamix::reaper
{

std::optional<SelectedItem> getSelectedItem()
{
#if REAMIX_WITH_REAPER_IO
    // 1:1 port of Lua remix_operations.lua:178-217 update_selected_item.
    if (! GetSelectedMediaItem)
        return std::nullopt;

    MediaItem* item = GetSelectedMediaItem (nullptr, 0);
    if (item == nullptr)
        return std::nullopt;

    MediaItem_Take* take = GetActiveTake (item);
    if (take == nullptr)
        return std::nullopt;

    PCM_source* source = GetMediaItemTake_Source (take);
    if (source == nullptr)
        return std::nullopt;

    SelectedItem out;
    out.itemPtr           = item;
    out.trackPtr          = GetMediaItem_Track (item);
    out.durationSec       = GetMediaItemInfo_Value (item, "D_LENGTH");
    out.positionSec       = GetMediaItemInfo_Value (item, "D_POSITION");
    out.sourceFileDuration = GetMediaSourceLength
                                ? GetMediaSourceLength (source, nullptr)
                                : out.durationSec;
    if (out.sourceFileDuration <= 0.0)
        out.sourceFileDuration = out.durationSec;

    if (const char* tn = GetTakeName (take))
        out.name = juce::String::fromUTF8 (tn);

    char pathBuf [4096] = {0};
    GetMediaSourceFileName (source, pathBuf, sizeof (pathBuf));
    out.sourcePath = juce::String::fromUTF8 (pathBuf);

    if (out.sourcePath.isEmpty())
        return std::nullopt;

    return out;
#else
    return std::nullopt;
#endif
}

// ─────────────────────────────────────────────────────────────────────
// Step 6 — Time-selection / region detection (sesja 60)
// ─────────────────────────────────────────────────────────────────────

std::optional<TimeSelection> getTimeSelection()
{
#if REAMIX_WITH_REAPER_IO
    // 1:1 port of Lua remix_insert.lua:62-68 get_time_selection.
    if (! GetSet_LoopTimeRange)
        return std::nullopt;

    double startSec = 0.0;
    double endSec   = 0.0;
    GetSet_LoopTimeRange (false, false, &startSec, &endSec, false);

    if (endSec <= startSec)
        return std::nullopt;

    TimeSelection out;
    out.startSec    = startSec;
    out.endSec      = endSec;
    out.durationSec = endSec - startSec;
    return out;
#else
    return std::nullopt;
#endif
}

std::optional<ItemRegion> getItemRegion (double itemDurationSec,
                                          double itemPositionSec)
{
    // 1:1 port of Lua remix_insert.lua:72-91 get_item_region.
    // Note: Lua takes an `item` parameter and asserts non-nil; the C++ port
    // drops it because the function never dereferences the pointer — only
    // duration/position from the SelectedItem struct are used.

    if (itemDurationSec <= 0.0)
        return std::nullopt;

    auto sel = getTimeSelection();
    if (! sel.has_value())
        return std::nullopt;

    const double itemStart = itemPositionSec;

    // Item-relative overlap (Lua L82-83). Lua keeps a local `item_end` for
    // readability but never reads it — port follows.
    const double regionStart = std::max (0.0, sel->startSec - itemStart);
    const double regionEnd   = std::min (itemDurationSec,    sel->endSec   - itemStart);

    // Min ~4 bars (Lua L85 — "min ~4 bars needed for remix").
    if (regionEnd - regionStart < 6.0)
        return std::nullopt;

    // Entire-item slack (Lua L86-88 — "covers entire item — use full remix").
    if (regionStart <= 0.1 && regionEnd >= itemDurationSec - 0.1)
        return std::nullopt;

    ItemRegion out;
    out.startSec = regionStart;
    out.endSec   = regionEnd;
    return out;
}

// ─────────────────────────────────────────────────────────────────────
// Step 9 — Insert/Update (ADR-046)
// ─────────────────────────────────────────────────────────────────────

#if REAMIX_WITH_REAPER_IO

namespace
{
    // Format seconds → "M:SS" (Lua remix_insert.lua:152-155 pattern).
    juce::String formatMSS (double sec)
    {
        const int totalSec = (int) std::floor (sec);
        const int m = totalSec / 60;
        const int s = totalSec % 60;
        return juce::String (m) + ":" + juce::String (s).paddedLeft ('0', 2);
    }

    // Alternating mid-grays per ADR-046 § Decision Q1 color scheme.
    // Sesja 100b iter 3 — brightened from 0x3a / 0x4a (user smoke verbatim
    // *"sa za ciemne — fajnie ze sie roznia co drugi blok ale sa za
    // ciemne"*); the original values blended into REAPER's default dark
    // track background so the splice boundaries weren't visible. New
    // values keep the same Δ16 contrast between odd/even tiles but sit
    // mid-tone so they read clearly on either dark- or light-themed
    // tracks. 0x01000000 sets the REAPER "custom color" flag (else
    // REAPER ignores).
    int alternatingGrayColor (int clipIndex)
    {
        const bool even = (clipIndex % 2) == 0;
        const int r = even ? 0x6a : 0x80;
        const int g = even ? 0x6a : 0x80;
        const int b = even ? 0x6a : 0x80;
        return ColorToNative (r, g, b) | 0x01000000;
    }

    // Sesja 100b — DEV-049. Reamix-managed markers / regions carry the
    // "reamix:" prefix so the cleanup pass can identify them without
    // colliding with user-authored markers in the same range.
    constexpr const char* kSpliceMarkerPrefix = "reamix:splice";
    constexpr const char* kRenderRegionPrefix = "reamix:render";

    // Info-blue (#6A9BC4) matches the in-plugin "navigation colour" family
    // used for playhead / selection scrim / selection edges (ADR-091 era,
    // sesja 100). Subtle on REAPER's dark ruler without screaming for
    // attention.
    int reamixMarkerColor()
    {
        return ColorToNative (0x6A, 0x9B, 0xC4) | 0x01000000;
    }

    // Slight green-cyan tint for render regions so they read distinct
    // from splice point markers at a glance — same blue family but
    // lifted toward turquoise.
    int reamixRegionColor()
    {
        return ColorToNative (0x70, 0xB0, 0x9C) | 0x01000000;
    }

    bool nameHasReamixPrefix (const char* name)
    {
        if (name == nullptr) return false;
        return std::strncmp (name, "reamix:", 7) == 0;
    }
} // namespace

juce::String getItemGuid (void* itemPtr)
{
    if (itemPtr == nullptr || ! GetSetMediaItemInfo_String)
        return {};

    char buf [128] = {0};
    if (GetSetMediaItemInfo_String ((MediaItem*) itemPtr, "GUID", buf, false))
        return juce::String::fromUTF8 (buf);

    return {};
}

juce::String getTrackGuid (void* trackPtr)
{
    if (trackPtr == nullptr || ! GetSetMediaTrackInfo_String)
        return {};

    char buf [128] = {0};
    if (GetSetMediaTrackInfo_String ((MediaTrack*) trackPtr, "GUID", buf, false))
        return juce::String::fromUTF8 (buf);

    return {};
}

double getItemPosition (void* itemPtr)
{
    if (itemPtr == nullptr || ! GetMediaItemInfo_Value) return 0.0;
    return GetMediaItemInfo_Value ((MediaItem*) itemPtr, "D_POSITION");
}

void* findItemByGuid (const juce::String& itemGuid)
{
    if (itemGuid.isEmpty() || ! CountTracks || ! GetTrack
        || ! CountTrackMediaItems || ! GetTrackMediaItem)
        return nullptr;

    const int trackCount = CountTracks (nullptr);
    for (int ti = 0; ti < trackCount; ++ti)
    {
        MediaTrack* tr = GetTrack (nullptr, ti);
        if (tr == nullptr) continue;

        const int itemCount = CountTrackMediaItems (tr);
        for (int ii = 0; ii < itemCount; ++ii)
        {
            MediaItem* it = GetTrackMediaItem (tr, ii);
            if (it == nullptr) continue;

            if (getItemGuid (it) == itemGuid)
                return it;
        }
    }

    return nullptr;
}

void* findTrackByGuid (const juce::String& trackGuid)
{
    if (trackGuid.isEmpty() || ! CountTracks || ! GetTrack)
        return nullptr;

    const int trackCount = CountTracks (nullptr);
    for (int ti = 0; ti < trackCount; ++ti)
    {
        MediaTrack* tr = GetTrack (nullptr, ti);
        if (tr == nullptr) continue;

        if (getTrackGuid (tr) == trackGuid)
            return tr;
    }

    return nullptr;
}

InsertResult insertRemixClips (const InsertSpec& spec)
{
    InsertResult result;

    if (spec.trackPtr == nullptr || spec.clips == nullptr || spec.clips->empty())
    {
        result.errorMessage = "Insert called with empty spec";
        return result;
    }

    if (! AddMediaItemToTrack || ! AddTakeToMediaItem
        || ! PCM_Source_CreateFromFile || ! SetMediaItemTake_Source
        || ! SetMediaItemInfo_Value || ! SetMediaItemTakeInfo_Value
        || ! Undo_BeginBlock || ! Undo_EndBlock || ! PreventUIRefresh
        || ! UpdateArrange || ! SelectAllMediaItems || ! SetMediaItemSelected
        || ! UpdateItemInProject)
    {
        result.errorMessage = "REAPER SDK symbols missing";
        return result;
    }

    MediaTrack* track = (MediaTrack*) spec.trackPtr;

    Undo_BeginBlock();
    PreventUIRefresh (1);
    SelectAllMediaItems (nullptr, false);

    // Step 1: delete existing remix items by GUID — runs in BOTH destructive
    // and insert-as-new modes. For insert-as-new (DEV-081 sesja 112) this is
    // the Update flow: previous insert-as-new clips for the same source are
    // removed before the new ones land at the same basePosition. Caller
    // populates spec.existingItemGuidsToDelete only when an Update is
    // intended; left empty otherwise the loop is a no-op.
    if (DeleteTrackMediaItem)
    {
        for (const auto& guid : spec.existingItemGuidsToDelete)
        {
            void* it = findItemByGuid (guid);
            if (it == nullptr) continue;

            MediaTrack* itTrack = GetMediaItem_Track ? GetMediaItem_Track ((MediaItem*) it) : nullptr;
            if (itTrack != nullptr)
                DeleteTrackMediaItem (itTrack, (MediaItem*) it);
        }

        // Step 2: delete the source item ONLY when not in insert-as-new mode.
        // insert-as-new explicitly preserves the source clip on the timeline.
        if (! spec.insertAsNewItem && spec.sourceItemToDelete != nullptr)
            DeleteTrackMediaItem (track, (MediaItem*) spec.sourceItemToDelete);
    }

    // Step 3: create one MediaItem per clip.
    int clipIndex = 0;
    for (const auto& clip : *spec.clips)
    {
        // Per Lua remix_insert.lua:127-167 create_remix_clip.
        const std::string srcPath = clip.sourcePath.empty()
            ? spec.originalSourcePath.toStdString()
            : clip.sourcePath;

        PCM_source* source = PCM_Source_CreateFromFile (srcPath.c_str());
        if (source == nullptr)
        {
            ++clipIndex;
            continue;
        }

        MediaItem* item = AddMediaItemToTrack (track);
        MediaItem_Take* take = (item != nullptr) ? AddTakeToMediaItem (item) : nullptr;
        const double duration = clip.durationSec > 0.0
            ? clip.durationSec
            : std::max (0.0, clip.sourceEndSec - clip.sourceStartSec);

        if (item == nullptr || take == nullptr || duration <= 0.0)
        {
            ++clipIndex;
            continue;
        }

        SetMediaItemTake_Source (take, source);
        SetMediaItemInfo_Value (item, "D_POSITION", spec.basePositionSec + clip.timelineStartSec);
        SetMediaItemInfo_Value (item, "D_LENGTH", duration);
        SetMediaItemInfo_Value (item, "B_LOOPSRC", 0.0);
        SetMediaItemTakeInfo_Value (take, "D_STARTOFFS", clip.sourceStartSec);
        SetMediaItemInfo_Value (item, "D_FADEINLEN", clip.fadeInSec);
        SetMediaItemInfo_Value (item, "D_FADEOUTLEN", clip.fadeOutSec);

        // ADR-046 § Decision Q1: equal-power (cosine) fade shapes — closest
        // REAPER-native match to Crossfade._equal_power_crossfade fallback in
        // renderer.py:341+. Shape 7 = REAPER's equal-power.
        SetMediaItemInfo_Value (item, "C_FADEINSHAPE", 7.0);
        SetMediaItemInfo_Value (item, "C_FADEOUTSHAPE", 7.0);

        // Take name: "Remix M:SS-M:SS" per ADR-046; auto-mode without segment
        // labels gets "Remix" prefix (Block Assembly will override later).
        const juce::String takeName = juce::String ("Remix ")
            + formatMSS (clip.sourceStartSec) + "-" + formatMSS (clip.sourceEndSec);
        if (GetSetMediaItemTakeInfo_String)
        {
            char buf [256] = {0};
            takeName.copyToUTF8 (buf, sizeof (buf));
            GetSetMediaItemTakeInfo_String (take, "P_NAME", buf, true);
        }

        // Alternating gray colors per ADR-046.
        SetMediaItemInfo_Value (item, "I_CUSTOMCOLOR", (double) alternatingGrayColor (clipIndex));

        SetMediaItemSelected (item, true);
        UpdateItemInProject (item);

        const juce::String guid = getItemGuid (item);
        if (guid.isNotEmpty())
            result.insertedItemGuids.push_back (guid);
        ++result.clipCount;

        ++clipIndex;
    }

    // Sesja 100b — DEV-049. Splice markers + render region added INSIDE
    // the Undo block so REAPER's undo state captures them along with the
    // items. A single Cmd+Z then rolls back clips AND markers atomically.
    //
    // Range computed from the actual clip span (last clip's
    // timelineStartSec + durationSec) rather than spec.totalRangeSec —
    // sesja 100b user smoke verbatim *"w duration nadal tez nie
    // zaznaczala calego"* showed that remix.targetSec from the renderer
    // can lag behind the materialised clip end (clip durations carry
    // crossfade tails the target metric doesn't reflect). Reading the
    // last clip is foolproof.
    if (result.clipCount > 0 && spec.clips != nullptr && ! spec.clips->empty())
    {
        const auto&  lastClip = spec.clips->back();
        const double actualLen = lastClip.timelineStartSec + lastClip.durationSec;
        const double rangeEnd  = spec.basePositionSec + actualLen;
        clearReamixMarkersInRange (spec.basePositionSec, rangeEnd);
        if (spec.addSpliceMarkers)
            addSpliceMarkers (spec.basePositionSec, spec.spliceTimesRel);
        if (spec.addRenderRegion)
            addRenderRegion (spec.basePositionSec, rangeEnd, spec.renderRegionName);
    }

    // DEV-081 sesja 112 — when inserting as new item (Region mode
    // non-destructive path), restore selection to the original source
    // BEFORE UpdateArrange so the MainComponent polling timer keeps the
    // same selected-item identity it had before Insert. Without this, the
    // post-loop SetMediaItemSelected calls above left every NEW clip
    // selected, polling detected an itemChanged signal on the next tick,
    // and applySelectedItem force-flipped appMode_ to Duration — the
    // regression user reported in sesja 112 ("przeskakuje tryb na
    // duration"). Bridge honours this field only when insertAsNewItem
    // is true; legacy destructive paths leave selection on inserted clips
    // because the source was deleted anyway.
    if (spec.insertAsNewItem && spec.preserveSelectionItemPtr != nullptr)
    {
        SelectAllMediaItems (nullptr, false);
        SetMediaItemSelected ((MediaItem*) spec.preserveSelectionItemPtr, true);
    }

    PreventUIRefresh (-1);
    UpdateArrange();

    const std::string label = spec.undoLabel.toStdString();
    Undo_EndBlock (label.c_str(), -1);

    result.ok = result.clipCount > 0;
    if (! result.ok)
        result.errorMessage = "No clips were created";

    return result;
}

// ─────────────────────────────────────────────────────────────────────
// Step 6 — Region remix insert (splice INTO original track) — sesja 60
// 1:1 port of Lua remix_insert.lua:176-236 insert_region_remix.
// ─────────────────────────────────────────────────────────────────────

namespace
{
    // Per-clip MediaItem creation — extracted from insertRemixClips loop body
    // for reuse in region branch. Creates one MediaItem on `track`, configures
    // it per ADR-046 § Decision Q1 (source / position / fades / take name /
    // gray color), returns the created MediaItem* or nullptr on any failure.
    MediaItem* createSingleRemixClip (MediaTrack* track,
                                       const reamix::render::EditClip& clip,
                                       double basePositionSec,
                                       const std::string& originalSourcePath,
                                       int clipIndex)
    {
        const std::string srcPath = clip.sourcePath.empty()
            ? originalSourcePath
            : clip.sourcePath;

        PCM_source* source = PCM_Source_CreateFromFile (srcPath.c_str());
        if (source == nullptr) return nullptr;

        MediaItem* item = AddMediaItemToTrack (track);
        MediaItem_Take* take = (item != nullptr) ? AddTakeToMediaItem (item) : nullptr;
        const double duration = clip.durationSec > 0.0
            ? clip.durationSec
            : std::max (0.0, clip.sourceEndSec - clip.sourceStartSec);

        if (item == nullptr || take == nullptr || duration <= 0.0)
            return nullptr;

        SetMediaItemTake_Source (take, source);
        SetMediaItemInfo_Value (item, "D_POSITION", basePositionSec + clip.timelineStartSec);
        SetMediaItemInfo_Value (item, "D_LENGTH", duration);
        SetMediaItemInfo_Value (item, "B_LOOPSRC", 0.0);
        SetMediaItemTakeInfo_Value (take, "D_STARTOFFS", clip.sourceStartSec);
        SetMediaItemInfo_Value (item, "D_FADEINLEN", clip.fadeInSec);
        SetMediaItemInfo_Value (item, "D_FADEOUTLEN", clip.fadeOutSec);
        SetMediaItemInfo_Value (item, "C_FADEINSHAPE", 7.0);
        SetMediaItemInfo_Value (item, "C_FADEOUTSHAPE", 7.0);

        const juce::String takeName = juce::String ("Remix ")
            + formatMSS (clip.sourceStartSec) + "-" + formatMSS (clip.sourceEndSec);
        if (GetSetMediaItemTakeInfo_String)
        {
            char buf [256] = {0};
            takeName.copyToUTF8 (buf, sizeof (buf));
            GetSetMediaItemTakeInfo_String (take, "P_NAME", buf, true);
        }

        SetMediaItemInfo_Value (item, "I_CUSTOMCOLOR", (double) alternatingGrayColor (clipIndex));
        SetMediaItemSelected (item, true);
        UpdateItemInProject (item);
        return item;
    }
}

RegionInsertResult insertRegionRemixClips (const RegionInsertSpec& spec)
{
    RegionInsertResult result;

    if (spec.trackPtr == nullptr || spec.sourceItemToSplit == nullptr
        || spec.clips == nullptr || spec.clips->empty()
        || spec.absRegionEndSec <= spec.absRegionStartSec)
    {
        result.errorMessage = "Region insert called with invalid spec";
        return result;
    }

    if (! AddMediaItemToTrack || ! AddTakeToMediaItem
        || ! PCM_Source_CreateFromFile || ! SetMediaItemTake_Source
        || ! SetMediaItemInfo_Value || ! SetMediaItemTakeInfo_Value
        || ! Undo_BeginBlock || ! Undo_EndBlock || ! PreventUIRefresh
        || ! UpdateArrange || ! SelectAllMediaItems || ! SetMediaItemSelected
        || ! UpdateItemInProject || ! SplitMediaItem || ! DeleteTrackMediaItem
        || ! GetMediaItemInfo_Value)
    {
        result.errorMessage = "REAPER SDK symbols missing";
        return result;
    }

    MediaTrack* track          = (MediaTrack*) spec.trackPtr;
    MediaItem*  sourceItem     = (MediaItem*) spec.sourceItemToSplit;
    const double absRegionStart = spec.absRegionStartSec;
    const double absRegionEnd   = spec.absRegionEndSec;
    const double originalRegionDur = absRegionEnd - absRegionStart;

    Undo_BeginBlock();
    PreventUIRefresh (1);
    SelectAllMediaItems (nullptr, false);

    // ── DEV-038 sesja 67 — Update path (Region replace-in-place) ────
    // When existingItemGuidsToDelete is non-empty, this is an Update of an
    // existing Region remix (user already Inserted once, now changes
    // target/region and clicks Insert again). Without Update path,
    // a second Insert would split AGAIN inside the prior pre-region piece
    // and add a fresh remix alongside the existing one (Bug D1) or fail
    // silently when new bounds lie outside the shrunk pre-region (Bug D2).
    //
    // Update steps (all inside the same Undo block as the standard split
    // + insert that follows, so a single Cmd+Z reverts the whole Update):
    //   (a) Delete every prior inserted clip + post-region GUID. Defensive:
    //       findItemByGuid(empty/missing) → nullptr, skip (user may have
    //       manually deleted some clips between Insert and Update).
    //   (b) findItemByGuid(preRegionGuid) → restore D_LENGTH to
    //       originalItemDurationSec. Pre-region piece keeps the original
    //       sourceItem GUID across split (ADR-056 D2). After this restore,
    //       the pre-region piece IS the original item again, ready for
    //       fresh split at NEW absRegionStart / absRegionEnd.
    //   (c) sourceItem reassigned to the restored pre-region piece — caller
    //       may have passed `picked->itemPtr` which is the clicked clip in
    //       a Region-group dispatch case (NOT the pre-region piece); the
    //       restore-and-reassign makes the standard split path below work
    //       uniformly for Insert and Update.
    //
    // Legacy entries (originalItemDurationSec == 0) skip the D_LENGTH
    // restore but still process deletes — falls through to the standard
    // split which may fail gracefully if pre-region piece's D_LENGTH is
    // too small for the new bounds. The next successful Insert overwrites
    // the group entry with fresh schema.
    const bool isUpdate = ! spec.existingItemGuidsToDelete.empty();
    if (isUpdate && DeleteTrackMediaItem)
    {
        for (const auto& guid : spec.existingItemGuidsToDelete)
        {
            if (guid.isEmpty()) continue;
            void* it = findItemByGuid (guid);
            if (it == nullptr) continue; // user manually deleted — OK, skip.

            MediaTrack* itTrack = GetMediaItem_Track
                ? GetMediaItem_Track ((MediaItem*) it)
                : nullptr;
            if (itTrack != nullptr)
                DeleteTrackMediaItem (itTrack, (MediaItem*) it);
        }

        if (spec.preRegionGuid.isNotEmpty()
            && spec.originalItemDurationSec > 0.0)
        {
            void* preRegion = findItemByGuid (spec.preRegionGuid);
            if (preRegion != nullptr)
            {
                SetMediaItemInfo_Value ((MediaItem*) preRegion,
                                         "D_LENGTH",
                                         spec.originalItemDurationSec);
                sourceItem = (MediaItem*) preRegion;
            }
        }
    }

    // Step 1 — split at region start → pre-region (left, original ptr) + rest (right).
    MediaItem* restItem = SplitMediaItem (sourceItem, absRegionStart);
    if (restItem == nullptr)
    {
        PreventUIRefresh (-1);
        Undo_EndBlock ("Region remix (split failed)", -1);
        result.errorMessage = "SplitMediaItem at region start failed";
        return result;
    }

    // Step 2 — split rest at region end → region-only (left) + post (right).
    // post may be nullptr when region extends to end of source item.
    MediaItem* postItem = SplitMediaItem (restItem, absRegionEnd);

    // Step 3 — delete the region-only portion (the middle slice we just isolated).
    DeleteTrackMediaItem (track, restItem);

    // Step 4 — insert remix clips at base position = absRegionStart.
    // Track the first/last inserted items so step 6 can apply edge fades
    // for a "delicate mix" with the surrounding pre-region / post-region
    // items (BUG-B sesja 63 — Renderer's clip plan defaults edge fades to
    // 0.0, leaving a perceptible click at the splice boundary).
    int clipIndex = 0;
    MediaItem* firstInserted = nullptr;
    MediaItem* lastInserted  = nullptr;
    for (const auto& clip : *spec.clips)
    {
        MediaItem* item = createSingleRemixClip (track, clip, absRegionStart,
                                                  spec.originalSourcePath.toStdString(),
                                                  clipIndex);
        if (item != nullptr)
        {
            ++result.clipCount;
            if (firstInserted == nullptr) firstInserted = item;
            lastInserted = item;
            // ADR-056 (sesja 66) — collect GUID for regionGroupTracker.
            result.insertedItemGuids.push_back (getItemGuid ((void*) item));
        }
        ++clipIndex;
    }

    // Step 5 — shift post-region item by remix-vs-region duration delta.
    // remix extent = the position of the last clip's end on the timeline.
    if (postItem != nullptr && ! spec.clips->empty())
    {
        const auto& last = spec.clips->back();
        const double remixExtent = last.timelineStartSec
            + (last.durationSec > 0.0
                ? last.durationSec
                : std::max (0.0, last.sourceEndSec - last.sourceStartSec));
        const double delta = remixExtent - originalRegionDur;
        if (std::fabs (delta) > 0.001)
        {
            const double oldPos = GetMediaItemInfo_Value (postItem, "D_POSITION");
            SetMediaItemInfo_Value (postItem, "D_POSITION", oldPos + delta);
        }
    }

    // ADR-056 (sesja 66 fix B) — capture post-region piece GUID into group
    // memberships. SplitMediaItem assigns a NEW GUID to the post-region
    // piece (verified via diag log: pre-region keeps original GUID, post-
    // region gets fresh one), so without this capture the post-region piece
    // falls outside Region group dispatch detection and clicking it shows
    // a fragment Duration view instead of the saved remix view (Test 2
    // user repro sesja 66). Pre-region piece keeps the original sourceItem
    // GUID which IS group->sourceGuid → already covered by dispatch direct
    // path #1, no need to capture separately.
    if (postItem != nullptr)
    {
        const juce::String postGuid = getItemGuid ((void*) postItem);
        if (postGuid.isNotEmpty())
            result.insertedItemGuids.push_back (postGuid);
    }

    // ADR-057 (sesja 68 step 2e) — overlap-crossfade boundary. RemixPipeline
    // extended first/last EditClip source ranges by boundaryLeadIn/OutSec
    // beyond user-selection edges (regStart - leadIn / regEnd + leadOut).
    // Insert pipeline mirrors the overlap on the timeline:
    //   - Pre-region D_LENGTH extended by leadIn past user-edge
    //   - First WAV clip D_POSITION shifted earlier by leadIn
    //   - Pre-region D_FADEOUTLEN = first WAV D_FADEINLEN = 2 × leadIn (full
    //     overlap window)
    //   - Mirror for last-WAV / post-region with leadOut
    // In overlap region pre-region and WAV play SAME source content (sample-
    // exact by construction in RemixPipeline override) so equal-power
    // crossfade preserves full vol → no click + no doubled-syllable.
    const double leadIn  = std::max (0.0, spec.boundaryLeadInSec);
    const double leadOut = std::max (0.0, spec.boundaryLeadOutSec);
    const double fadeIn  = 2.0 * leadIn;   // total overlap window on entry
    const double fadeOut = 2.0 * leadOut;  // total overlap window on exit

    if (leadIn > 0.0)
    {
        if (sourceItem != nullptr)
        {
            const double curLen = GetMediaItemInfo_Value ((MediaItem*) sourceItem, "D_LENGTH");
            SetMediaItemInfo_Value ((MediaItem*) sourceItem, "D_LENGTH", curLen + leadIn);
            SetMediaItemInfo_Value ((MediaItem*) sourceItem, "D_FADEOUTLEN",   fadeIn);
            SetMediaItemInfo_Value ((MediaItem*) sourceItem, "C_FADEOUTSHAPE", 7.0);
        }
        if (firstInserted != nullptr)
        {
            const double curPos = GetMediaItemInfo_Value (firstInserted, "D_POSITION");
            SetMediaItemInfo_Value (firstInserted, "D_POSITION",   curPos - leadIn);
            SetMediaItemInfo_Value (firstInserted, "D_FADEINLEN",  fadeIn);
            SetMediaItemInfo_Value (firstInserted, "C_FADEINSHAPE", 7.0);
        }
    }

    if (leadOut > 0.0)
    {
        if (lastInserted != nullptr)
        {
            // Last WAV's D_LENGTH already extended via clip.durationSec set in
            // createSingleRemixClip (RemixPipeline added leadOut to last clip's
            // durationSec). Item naturally ends `leadOut` past user-edge. Just
            // apply fade-out.
            SetMediaItemInfo_Value (lastInserted, "D_FADEOUTLEN",   fadeOut);
            SetMediaItemInfo_Value (lastInserted, "C_FADEOUTSHAPE", 7.0);
        }
        if (postItem != nullptr)
        {
            const double curPos = GetMediaItemInfo_Value (postItem, "D_POSITION");
            SetMediaItemInfo_Value (postItem, "D_POSITION",  curPos - leadOut);
            SetMediaItemInfo_Value (postItem, "D_FADEINLEN",  fadeOut);
            SetMediaItemInfo_Value (postItem, "C_FADEINSHAPE", 7.0);
        }
    }

    // Sesja 100b — DEV-049. Splice markers + render region added INSIDE
    // the Undo block so a single Cmd+Z atomically rolls back clips +
    // markers + region.
    //
    // Render region semantic for Region mode: span the WHOLE post-remix
    // track, not just the replaced middle. The source item is split into
    // pre-region + remix-clips + post-region; render region must cover
    // all three so File → Render → Region produces the full song.
    // User smoke verbatim sesja 100b: *"region nadal zaznacza tylko
    // wstawione regiony a nie caly nowy utwor"*.
    //
    // Geometry:
    //   regionStart = originalItemPositionSec   (pre-region start)
    //   regionEnd   = originalItemPositionSec + originalItemDurationSec
    //                 + (actualLen - originalRegionLen)
    //               where the delta accounts for the remix changing
    //               the spliced segment's length, which post-region
    //               absorbs by shifting forward / backward.
    //   actualLen   = clips.back().timelineStartSec + .durationSec
    //                 (foolproof — covers boundary fade tails baked into
    //                 the last clip's durationSec).
    //
    // Splice point markers still anchor on absRegionStartSec — their
    // semantic is per-cut WITHIN the spliced region, not the whole song.
    if (result.clipCount > 0 && spec.clips != nullptr && ! spec.clips->empty())
    {
        const auto&  lastClip   = spec.clips->back();
        const double actualLen  = lastClip.timelineStartSec + lastClip.durationSec;
        const double originalRegionLen = spec.absRegionEndSec - spec.absRegionStartSec;
        const double remixDelta        = actualLen - originalRegionLen;

        double regionStart;
        double regionEnd;
        if (spec.originalItemPositionSec >= 0.0
            && spec.originalItemDurationSec > 0.0)
        {
            regionStart = spec.originalItemPositionSec;
            regionEnd   = spec.originalItemPositionSec
                        + spec.originalItemDurationSec
                        + remixDelta;
        }
        else
        {
            // Fallback: cover the spliced region only when the caller
            // hasn't supplied original-item bounds.
            regionStart = spec.absRegionStartSec - spec.boundaryLeadInSec;
            regionEnd   = spec.absRegionStartSec + actualLen;
        }

        clearReamixMarkersInRange (regionStart, regionEnd);
        if (spec.addSpliceMarkers)
            addSpliceMarkers (spec.absRegionStartSec, spec.spliceTimesRel);
        if (spec.addRenderRegion)
            addRenderRegion (regionStart, regionEnd, spec.renderRegionName);
    }

    PreventUIRefresh (-1);
    UpdateArrange();

    const std::string label = spec.undoLabel.toStdString();
    Undo_EndBlock (label.c_str(), -1);

    result.ok = result.clipCount > 0;
    if (! result.ok)
        result.errorMessage = "No clips were created";

    return result;
}

#endif // REAMIX_WITH_REAPER_IO (Step 9 block)

// ─────────────────────────────────────────────────────────────────────
// Step 8 — UserBlock persistence (ADR-051, sesja 61)
// ─────────────────────────────────────────────────────────────────────

// ADR-051 sesja 61 hot-fix — P_EXT key follows SneakPeak / Lua convention
// (underscores, no dots). The earlier "P_EXT:reamix.blocks" key with a dot
// was the only such usage in the codebase + references; reverted to the
// established `_` style after a one-shot REAPER crash in
// GetSetMediaItemInfo_String (REAPER-2026-04-27-115301.ips, sesja 61).
constexpr const char* kPExtBlocksKey = "P_EXT:reamix_blocks";

std::vector<reamix::ui::UserBlock> loadUserBlocks (void* itemPtr)
{
#if REAMIX_WITH_REAPER_IO
    if (itemPtr == nullptr || ! GetSetMediaItemInfo_String)
        return {};

    // 4 KB initial buffer (matches SneakPeak char buf[256]-class hygiene
    // scaled up for JSON; typical 4-12 user blocks ≈ 300-1000 bytes).
    char buf [4 * 1024] = {0};
    if (! GetSetMediaItemInfo_String ((MediaItem*) itemPtr,
                                       kPExtBlocksKey,
                                       buf, false))
        return {};

    return reamix::ui::deserializeUserBlocks (juce::String::fromUTF8 (buf));
#else
    (void) itemPtr;
    return {};
#endif
}

void saveUserBlocks (void* itemPtr,
                     const std::vector<reamix::ui::UserBlock>& blocks)
{
#if REAMIX_WITH_REAPER_IO
    if (itemPtr == nullptr || ! GetSetMediaItemInfo_String)
        return;

    // Empty vector → write empty string so a previously-stored value is cleared.
    const juce::String json = blocks.empty()
        ? juce::String()
        : reamix::ui::serializeUserBlocks (blocks);

    // GetSetMediaItemInfo_String with setNewValue=true reads the buffer; size
    // it dynamically to (jsonSize+1, min 1 KB) so REAPER's internal copy
    // doesn't touch ~16 KB of zero-padding (sesja 61 crash guard).
    const std::string utf8  = json.toStdString();
    const std::size_t need  = utf8.size() + 1;
    const std::size_t alloc = std::max<std::size_t> (1024, need);
    std::vector<char> buf (alloc, 0);
    std::strncpy (buf.data(), utf8.c_str(), alloc - 1);
    GetSetMediaItemInfo_String ((MediaItem*) itemPtr,
                                kPExtBlocksKey,
                                buf.data(), true);
#else
    (void) itemPtr;
    (void) blocks;
#endif
}

// ── Sesja 100b — DEV-049 splice markers + render region ───────────────

#if REAMIX_WITH_REAPER_IO

void clearReamixMarkersInRange (double startSec, double endSec)
{
    if (! EnumProjectMarkers || ! DeleteProjectMarker) return;
    if (endSec <= startSec) return;

    // Tolerance to absorb FP drift between the position we wrote on
    // Insert vs. what REAPER stored back. 1 ms is well below any audible
    // splice resolution and won't skip a marker placed exactly on the
    // edge of the cleanup range.
    constexpr double kEps = 1e-3;

    // EnumProjectMarkers reorders/recompacts on delete, so we collect
    // (markrgnindexnumber, isrgn) pairs first, then DeleteProjectMarker
    // by *displayed* index number (stable across the project lifetime).
    struct Hit { int idxNumber; bool isrgn; };
    std::vector<Hit> hits;

    for (int i = 0; ; ++i)
    {
        bool        isrgn   = false;
        double      pos     = 0.0;
        double      rgnend  = 0.0;
        const char* name    = nullptr;
        int         idxNum  = 0;
        const int next = EnumProjectMarkers (i, &isrgn, &pos, &rgnend, &name, &idxNum);
        if (next == 0) break;  // no more markers at this index
        if (! nameHasReamixPrefix (name)) continue;

        // For point markers (rgnend = 0), match position against range.
        // For regions, match if either endpoint or the entire span sits
        // inside [startSec, endSec] — covers the typical Update scenario
        // where the new region replaces an overlapping prior one.
        const double markerStart = pos;
        const double markerEnd   = isrgn ? rgnend : pos;
        const bool overlaps =
               (markerStart >= startSec - kEps && markerStart <= endSec + kEps)
            || (markerEnd   >= startSec - kEps && markerEnd   <= endSec + kEps)
            || (markerStart <= startSec + kEps && markerEnd   >= endSec - kEps);

        if (overlaps)
            hits.push_back ({ idxNum, isrgn });
    }

    for (const auto& h : hits)
        DeleteProjectMarker (nullptr, h.idxNumber, h.isrgn);
}

void addSpliceMarkers (double basePositionSec,
                       const std::vector<double>& spliceTimesRel)
{
    if (! AddProjectMarker2) return;
    if (spliceTimesRel.empty()) return;

    const int colour = reamixMarkerColor();
    for (std::size_t i = 0; i < spliceTimesRel.size(); ++i)
    {
        const double pos = basePositionSec + spliceTimesRel[i];
        // wantidx = -1 → REAPER auto-assigns the next available marker
        // ID. rgnend = 0 + isrgn = false → point marker.
        AddProjectMarker2 (nullptr, false, pos, 0.0,
                            kSpliceMarkerPrefix, -1, colour);
    }
}

void addRenderRegion (double startSec, double endSec,
                      const juce::String& nameSuffix)
{
    if (! AddProjectMarker2) return;
    if (endSec <= startSec) return;

    juce::String name = kRenderRegionPrefix;
    if (nameSuffix.isNotEmpty())
        name += " " + nameSuffix;
    const std::string utf8 = name.toStdString();

    AddProjectMarker2 (nullptr, true, startSec, endSec,
                        utf8.c_str(), -1, reamixRegionColor());
}

#else

void clearReamixMarkersInRange (double, double)                    {}
void addSpliceMarkers (double, const std::vector<double>&)         {}
void addRenderRegion  (double, double, const juce::String&)        {}

#endif

#if ! REAMIX_WITH_REAPER_IO

juce::String getItemGuid (void*)              { return {}; }
juce::String getTrackGuid (void*)             { return {}; }
void*        findItemByGuid (const juce::String&)  { return nullptr; }
void*        findTrackByGuid (const juce::String&) { return nullptr; }
double       getItemPosition (void*)          { return 0.0; }
InsertResult insertRemixClips (const InsertSpec&)
{
    InsertResult r;
    r.errorMessage = "REAPER IO disabled at build time";
    return r;
}
RegionInsertResult insertRegionRemixClips (const RegionInsertSpec&)
{
    RegionInsertResult r;
    r.errorMessage = "REAPER IO disabled at build time";
    return r;
}

#endif

} // namespace reamix::reaper
