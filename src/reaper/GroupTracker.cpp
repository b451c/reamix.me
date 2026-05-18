#include "reaper/GroupTracker.h"

#if REAMIX_WITH_REAPER_IO
#include "reaper_plugin_functions.h"
#endif

#include <algorithm>
#include <cstdio>

namespace reamix::reaper::groupTracker
{

#if REAMIX_WITH_REAPER_IO

namespace
{
    // ADR-046 § Decision Q2: SECTION = "reamix.me" (matches DockableWindow
    // precedent + ADR-037 wording). INDEX_KEY = "group_index" (1:1 Lua port).
    constexpr const char* kSection  = "reamix.me";
    constexpr const char* kIndexKey = "group_index";

    juce::String perGroupKey (const juce::String& sourceGuid)
    {
        return "group:" + sourceGuid;
    }

    juce::String readExtState (const char* key)
    {
        if (! GetProjExtState) return {};

        // GetProjExtState requires a fixed-size out buffer; use a generous one.
        // 16 KB covers ~500 GUIDs in the index (each ~38 chars + delim) — well
        // beyond any realistic remix group count per project.
        std::vector<char> buf (16 * 1024, 0);
        const int n = GetProjExtState (nullptr, kSection, key, buf.data(), (int) buf.size());
        if (n <= 0)
            return {};

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
        {
            if (line.isNotEmpty())
                out.push_back (line);
        }
        return out;
    }

    std::vector<juce::String> splitByPipe (const juce::String& s)
    {
        // Mirrors Lua remix_utils.split_string(s, "|") used in remix_groups.lua:126.
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
                if (i > from)
                    out.push_back (s.substring (from, i));
                from = i + 1;
            }
        }
        if (from < s.length())
            out.push_back (s.substring (from));
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

    juce::String formatBasePosition (double v)
    {
        // Lua remix_groups.lua:142: string.format("%.12f", base_position).
        // Match exactly — full f64 precision avoids any cumulative drift if a
        // project is saved/reopened many times.
        char buf [64] = {0};
        std::snprintf (buf, sizeof (buf), "%.12f", v);
        return juce::String::fromUTF8 (buf);
    }
} // namespace

std::vector<juce::String> loadRemixGroupIndex()
{
    return splitLines (readExtState (kIndexKey));
}

juce::String loadRemixGroupRaw (const juce::String& sourceGuid)
{
    if (sourceGuid.isEmpty()) return {};
    return readExtState (perGroupKey (sourceGuid).toRawUTF8());
}

namespace
{
    std::optional<RemixGroup> loadRemixGroup (const juce::String& sourceGuid)
    {
        if (sourceGuid.isEmpty()) return std::nullopt;

        const juce::String raw = readExtState (perGroupKey (sourceGuid).toRawUTF8());
        if (raw.isEmpty()) return std::nullopt;

        const auto parts = splitByPipe (raw);
        // parts[0] = track_guid, parts[1] = base_position, parts[2] = item_guids (csv).
        // parts[3] = target_sec (sesja 100b extension — 0.0 sentinel when missing).

        RemixGroup g;
        g.sourceGuid = sourceGuid;
        g.trackGuid    = parts.size() > 0 ? parts[0] : juce::String{};
        g.basePosition = parts.size() > 1 ? parts[1].getDoubleValue() : 0.0;
        if (parts.size() > 2)
            g.itemGuids = splitByComma (parts[2]);
        g.targetSec    = parts.size() > 3 ? parts[3].getDoubleValue() : 0.0;

        return g;
    }
} // namespace

std::optional<RemixGroup> findGroupForSelectedItem (const juce::String& itemGuid)
{
    if (itemGuid.isEmpty()) return std::nullopt;

    // Direct: the item GUID itself is a source-guid.
    if (auto direct = loadRemixGroup (itemGuid))
        return direct;

    // Indirect: scan all groups for a matching item GUID inside their lists.
    for (const auto& sourceGuid : loadRemixGroupIndex())
    {
        auto group = loadRemixGroup (sourceGuid);
        if (! group) continue;

        for (const auto& g : group->itemGuids)
        {
            if (g == itemGuid)
                return group;
        }
    }

    return std::nullopt;
}

void saveRemixGroup (const RemixGroup& group)
{
    if (group.sourceGuid.isEmpty()) return;

    // Build payload: "{track_guid}|{base_position:.12f}|{item_guid_1,item_guid_2,...}"
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
        group.trackGuid + "|" + formatBasePosition (group.basePosition) + "|"
        + itemsCsv + "|" + formatBasePosition (group.targetSec);

    writeExtState (perGroupKey (group.sourceGuid).toRawUTF8(), payload);

    // Update index with dedup (mirrors Lua remix_groups.lua:152-154 +
    // save_remix_group_index seen-table dedup).
    auto index = loadRemixGroupIndex();
    if (std::find (index.begin(), index.end(), group.sourceGuid) == index.end())
        index.push_back (group.sourceGuid);

    // Dedup pass (idempotency): keep first occurrence of each guid.
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

void removeRemixGroup (const juce::String& sourceGuid)
{
    if (sourceGuid.isEmpty()) return;

    // Lua remix_groups.lua:160: writes empty string (REAPER treats as "clear").
    writeExtState (perGroupKey (sourceGuid).toRawUTF8(), {});

    auto index = loadRemixGroupIndex();
    std::vector<juce::String> filtered;
    for (const auto& g : index)
    {
        if (g != sourceGuid && g.isNotEmpty())
            filtered.push_back (g);
    }
    writeExtState (kIndexKey, joinIndex (filtered));
}

#else // ! REAMIX_WITH_REAPER_IO

std::vector<juce::String>  loadRemixGroupIndex()                                  { return {}; }
juce::String               loadRemixGroupRaw (const juce::String&)                { return {}; }
std::optional<RemixGroup>  findGroupForSelectedItem (const juce::String&)         { return std::nullopt; }
void                       saveRemixGroup (const RemixGroup&)                     {}
void                       removeRemixGroup (const juce::String&)                 {}

#endif

} // namespace reamix::reaper::groupTracker
