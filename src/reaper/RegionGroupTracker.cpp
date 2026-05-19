#include "reaper/RegionGroupTracker.h"

#if REAMIX_WITH_REAPER_IO
#include "reaper_plugin_functions.h"
#endif

#include <algorithm>
#include <cstdio>
#include <optional>

namespace reamix::reaper::regionGroupTracker
{

#if REAMIX_WITH_REAPER_IO

namespace
{
    // ADR-056 sesja 66: shares SECTION="reamix.me" with Duration groupTracker
    // (ADR-046 Q2) + DockableWindow ExtState. Distinct INDEX_KEY +
    // "region_group:" prefix isolates Region keyspace from Duration's.
    constexpr const char* kSection  = "reamix.me";
    constexpr const char* kIndexKey = "region_group_index";

    juce::String perGroupKey (const juce::String& sourceGuid)
    {
        return "region_group:" + sourceGuid;
    }

    juce::String readExtState (const char* key)
    {
        if (! GetProjExtState) return {};
        std::vector<char> buf (16 * 1024, 0);
        const int n = GetProjExtState (nullptr, kSection, key, buf.data(), (int) buf.size());
        if (n <= 0) return {};
        return juce::String::fromUTF8 (buf.data());
    }

    void writeExtState (const char* key, const juce::String& value)
    {
        if (! SetProjExtState) return;
        const std::string utf8 = value.toStdString();
        SetProjExtState (nullptr, kSection, key, utf8.c_str());
    }

    std::vector<juce::String> splitLines (const juce::String& s)
    {
        std::vector<juce::String> out;
        if (s.isEmpty()) return out;
        juce::StringArray arr;
        arr.addLines (s);
        for (const auto& line : arr)
            if (line.isNotEmpty()) out.push_back (line);
        return out;
    }

    std::vector<juce::String> splitByPipe (const juce::String& s)
    {
        std::vector<juce::String> out;
        int from = 0;
        for (int i = 0; i < s.length(); ++i)
        {
            if (s[i] == '|')
            {
                out.push_back (s.substring (from, i));
                from = i + 1;
            }
        }
        out.push_back (s.substring (from));
        return out;
    }

    std::vector<juce::String> splitByComma (const juce::String& s)
    {
        std::vector<juce::String> out;
        if (s.isEmpty()) return out;
        int from = 0;
        for (int i = 0; i < s.length(); ++i)
        {
            if (s[i] == ',')
            {
                if (i > from) out.push_back (s.substring (from, i));
                from = i + 1;
            }
        }
        if (from < s.length()) out.push_back (s.substring (from));
        return out;
    }

    juce::String joinIndex (const std::vector<juce::String>& guids)
    {
        juce::String out;
        bool first = true;
        for (const auto& g : guids)
        {
            if (g.isEmpty()) continue;
            if (! first) out += "\n";
            out += g;
            first = false;
        }
        return out;
    }

    juce::String formatDouble (double v)
    {
        // Match Duration groupTracker's %.12f precision so cumulative
        // save/reopen drift is impossible.
        char buf [64] = {0};
        std::snprintf (buf, sizeof (buf), "%.12f", v);
        return juce::String::fromUTF8 (buf);
    }
} // namespace

std::vector<juce::String> loadRegionGroupIndex()
{
    return splitLines (readExtState (kIndexKey));
}

juce::String loadRegionGroupRaw (const juce::String& sourceGuid)
{
    if (sourceGuid.isEmpty()) return {};
    return readExtState (perGroupKey (sourceGuid).toRawUTF8());
}

namespace
{
    std::optional<RegionGroup> loadRegionGroup (const juce::String& sourceGuid)
    {
        if (sourceGuid.isEmpty()) return std::nullopt;

        const juce::String raw = readExtState (perGroupKey (sourceGuid).toRawUTF8());
        if (raw.isEmpty()) return std::nullopt;

        const auto parts = splitByPipe (raw);
        // parts[0] = trackGuid, parts[1] = basePosition, parts[2] = regionStart,
        // parts[3] = regionEnd, parts[4] = targetSec, parts[5] = tmpWavPath,
        // parts[6] = itemGuids (csv),
        // parts[7] = originalItemDurationSec (DEV-038 sesja 67; legacy entries
        // miss this field — parser returns 0.0 default).

        RegionGroup g;
        g.sourceGuid     = sourceGuid;
        g.trackGuid      = parts.size() > 0 ? parts[0] : juce::String{};
        g.basePosition   = parts.size() > 1 ? parts[1].getDoubleValue() : 0.0;
        g.regionStartSec = parts.size() > 2 ? parts[2].getDoubleValue() : 0.0;
        g.regionEndSec   = parts.size() > 3 ? parts[3].getDoubleValue() : 0.0;
        g.targetSec      = parts.size() > 4 ? parts[4].getDoubleValue() : 0.0;
        g.tmpWavPath     = parts.size() > 5 ? parts[5] : juce::String{};
        if (parts.size() > 6)
            g.itemGuids = splitByComma (parts[6]);
        g.originalItemDurationSec = parts.size() > 7 ? parts[7].getDoubleValue() : 0.0;

        return g;
    }
} // namespace

std::optional<RegionGroup> findGroupForSelectedItem (const juce::String& itemGuid)
{
    if (itemGuid.isEmpty()) return std::nullopt;

    // Direct: the item GUID itself is a source-guid (rare but possible —
    // user may click on the original source item that was used as the
    // Region Insert anchor; the group entry is keyed by that GUID).
    if (auto direct = loadRegionGroup (itemGuid))
        return direct;

    // Indirect: scan all groups for a matching item GUID inside their lists.
    for (const auto& sourceGuid : loadRegionGroupIndex())
    {
        auto group = loadRegionGroup (sourceGuid);
        if (! group) continue;

        for (const auto& g : group->itemGuids)
            if (g == itemGuid) return group;
    }

    return std::nullopt;
}

void saveRegionGroup (const RegionGroup& group)
{
    if (group.sourceGuid.isEmpty()) return;

    juce::String itemsCsv;
    bool first = true;
    for (const auto& g : group.itemGuids)
    {
        if (g.isEmpty()) continue;
        if (! first) itemsCsv += ",";
        itemsCsv += g;
        first = false;
    }

    const juce::String payload =
          group.trackGuid + "|"
        + formatDouble (group.basePosition)   + "|"
        + formatDouble (group.regionStartSec) + "|"
        + formatDouble (group.regionEndSec)   + "|"
        + formatDouble (group.targetSec)      + "|"
        + group.tmpWavPath                    + "|"
        + itemsCsv                            + "|"
        + formatDouble (group.originalItemDurationSec);

    writeExtState (perGroupKey (group.sourceGuid).toRawUTF8(), payload);

    // Update index with dedup (same pattern as Duration groupTracker).
    auto index = loadRegionGroupIndex();
    if (std::find (index.begin(), index.end(), group.sourceGuid) == index.end())
        index.push_back (group.sourceGuid);

    std::vector<juce::String> seen;
    std::vector<juce::String> dedup;
    for (const auto& g : index)
    {
        if (g.isEmpty()) continue;
        if (std::find (seen.begin(), seen.end(), g) != seen.end()) continue;
        seen.push_back (g);
        dedup.push_back (g);
    }

    writeExtState (kIndexKey, joinIndex (dedup));
}

void removeRegionGroup (const juce::String& sourceGuid)
{
    if (sourceGuid.isEmpty()) return;

    writeExtState (perGroupKey (sourceGuid).toRawUTF8(), {});

    auto index = loadRegionGroupIndex();
    std::vector<juce::String> filtered;
    for (const auto& g : index)
        if (g != sourceGuid && g.isNotEmpty()) filtered.push_back (g);
    writeExtState (kIndexKey, joinIndex (filtered));
}

#else // ! REAMIX_WITH_REAPER_IO

std::vector<juce::String>   loadRegionGroupIndex()                                  { return {}; }
juce::String                loadRegionGroupRaw (const juce::String&)                { return {}; }
std::optional<RegionGroup>  findGroupForSelectedItem (const juce::String&)          { return std::nullopt; }
void                        saveRegionGroup (const RegionGroup&)                    {}
void                        removeRegionGroup (const juce::String&)                 {}

#endif

} // namespace reamix::reaper::regionGroupTracker
