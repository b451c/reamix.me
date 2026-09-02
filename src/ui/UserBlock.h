#pragma once

#include "ui/Theme.h"

#include <juce_core/juce_core.h>

#include <algorithm>
#include <optional>
#include <vector>

// UserBlock — user-marked section of a single source audio item.
//
// ADR-051 (sesja 61): Block Assembly mode is the sole label source post-ADR-044.
// Auto-segmentation was disabled because Python's CBM produced systematically
// wrong labels (Dua Lipa label_match 49.5 % < 64.2 % no-labels). User now
// marks sections directly on the source waveform via drag-on-bar gesture.
//
// Persistence: per-item via P_EXT:reamix_blocks (REAPER GetSetMediaItemInfo_String;
// underscore key style matches SneakPeak + Lua RemixTool convention).
// JSON encoding with short keys for compactness:
//   [{"s":12.5,"e":34.7,"k":3},{"s":34.7,"e":67.1,"k":1,"l":"Verse 2"},
//    {"s":67.1,"e":89.0,"k":1,"ck":"ck_1a2b3c4d"}]
//
//   s  = startSec       (item-relative seconds, 0.0 = item start)
//   e  = endSec         (item-relative, > startSec)
//   k  = kind           (reamix::theme::SegmentKind enum int 0..11)
//   l  = labelOverride  (optional — when user re-types display name)
//   ck = customKindId   (optional — ADR-092 sesja 100c, references global
//                        per-user CustomKindRegistry. When present, overrides
//                        kind for display; when registry-miss on this machine,
//                        kind is the fallback. Backwards-compatible: legacy
//                        4-field rows {s,e,k,l} parse with customKindId nullopt.)
//
// Survives REAPER project save/reopen (P_EXT pattern reused from ADR-036 ⚑ P2
// GroupTracker, validated through phase-6 sessions 55-60).

namespace reamix::ui
{

struct UserBlock
{
    double      startSec {0.0};
    double      endSec   {0.0};
    reamix::theme::SegmentKind kind     {reamix::theme::SegmentKind::Verse};
    std::optional<juce::String> labelOverride;

    // ADR-092 sesja 100c — when set, this block uses a custom kind defined
    // in the global per-user CustomKindRegistry. `kind` remains valid as a
    // fallback colour/label when the registry doesn't know this id (e.g.
    // user opened the project on a machine without their custom kindy).
    std::optional<juce::String> customKindId;
};

inline juce::var userBlockToVar (const UserBlock& b)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("s", b.startSec);
    obj->setProperty ("e", b.endSec);
    obj->setProperty ("k", (int) b.kind);
    if (b.labelOverride.has_value() && b.labelOverride->isNotEmpty())
        obj->setProperty ("l", *b.labelOverride);
    if (b.customKindId.has_value() && b.customKindId->isNotEmpty())
        obj->setProperty ("ck", *b.customKindId);
    return juce::var (obj);
}

inline std::optional<UserBlock> userBlockFromVar (const juce::var& v)
{
    if (! v.isObject()) return std::nullopt;

    UserBlock b;
    b.startSec = (double) v.getProperty ("s", 0.0);
    b.endSec   = (double) v.getProperty ("e", 0.0);

    if (b.endSec <= b.startSec) return std::nullopt;

    const int k = (int) v.getProperty ("k", (int) reamix::theme::SegmentKind::Verse);
    if (k < 0 || k >= (int) reamix::theme::SegmentKind::NumKinds) return std::nullopt;
    b.kind = (reamix::theme::SegmentKind) k;

    if (v.hasProperty ("l"))
    {
        const juce::String s = v.getProperty ("l", juce::String()).toString();
        if (s.isNotEmpty()) b.labelOverride = s;
    }
    // ADR-092 sesja 100c — optional customKindId; legacy rows without "ck"
    // parse with customKindId nullopt (backwards-compatible).
    if (v.hasProperty ("ck"))
    {
        const juce::String ck = v.getProperty ("ck", juce::String()).toString();
        // Format gate: must start with "ck_" prefix; malformed silently
        // ignored so the block falls back to its built-in `kind`.
        if (ck.startsWith ("ck_")) b.customKindId = ck;
    }
    return b;
}

inline juce::String serializeUserBlocks (const std::vector<UserBlock>& blocks)
{
    juce::Array<juce::var> arr;
    for (const auto& b : blocks)
        arr.add (userBlockToVar (b));
    // Compact JSON (no pretty-print) — P_EXT has a soft size limit, and the
    // string round-trips through REAPER's project file as a single line.
    return juce::JSON::toString (juce::var (arr), true);
}

inline std::vector<UserBlock> deserializeUserBlocks (const juce::String& json)
{
    std::vector<UserBlock> out;
    if (json.isEmpty()) return out;

    const juce::var v = juce::JSON::parse (json);
    if (! v.isArray()) return out;

    for (int i = 0; i < v.size(); ++i)
    {
        if (auto b = userBlockFromVar (v[i]); b.has_value())
            out.push_back (*b);
    }
    // Sort by start so paint / palette / persistence are order-stable.
    std::sort (out.begin(), out.end(),
               [](const UserBlock& a, const UserBlock& b) { return a.startSec < b.startSec; });
    return out;
}

// Sesja 120 (DEV-097) — the whole Block Assembly state of an item: blocks +
// arrangement queue + per-junction variations, so a project reopen brings
// the arrangement back, not only the marks. P_EXT payload v2 is an object
//   {"v":2,"blocks":[...],"queue":[0,2,1],"var":[0,1]}
// and the reader still accepts the v1 bare array (blocks only, queue empty).
// Queue entries index `blocks`; entries out of range are dropped on read.
struct UserBlocksState
{
    std::vector<UserBlock> blocks;
    std::vector<int>       queue;
    std::vector<int>       variations;   // size = max(0, queue.size() - 1)
};

inline juce::String serializeUserBlocksState (const UserBlocksState& st)
{
    if (st.blocks.empty()) return {};
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("v", 2);
    juce::Array<juce::var> blocks;
    for (const auto& b : st.blocks) blocks.add (userBlockToVar (b));
    obj->setProperty ("blocks", juce::var (blocks));
    juce::Array<juce::var> queue;
    for (int q : st.queue) queue.add (q);
    obj->setProperty ("queue", juce::var (queue));
    juce::Array<juce::var> vars;
    for (int v : st.variations) vars.add (v);
    obj->setProperty ("var", juce::var (vars));
    return juce::JSON::toString (juce::var (obj), true);
}

inline UserBlocksState deserializeUserBlocksState (const juce::String& json)
{
    UserBlocksState st;
    if (json.isEmpty()) return st;
    const juce::var v = juce::JSON::parse (json);
    if (v.isArray())                       // v1 payload: blocks only
    {
        st.blocks = deserializeUserBlocks (json);
        return st;
    }
    if (! v.isObject()) return st;

    const juce::var blocks = v.getProperty ("blocks", juce::var());
    if (blocks.isArray())
    {
        for (int i = 0; i < blocks.size(); ++i)
            if (auto b = userBlockFromVar (blocks[i]); b.has_value())
                st.blocks.push_back (*b);
        // The queue indexes the SORTED block list, so sort exactly as the
        // writer (userBlocks_ is kept sorted by start) before validating.
        std::stable_sort (st.blocks.begin(), st.blocks.end(),
                          [](const UserBlock& a, const UserBlock& b) { return a.startSec < b.startSec; });
    }
    const juce::var queue = v.getProperty ("queue", juce::var());
    if (queue.isArray())
        for (int i = 0; i < queue.size(); ++i)
        {
            const int q = (int) queue[i];
            if (q >= 0 && q < (int) st.blocks.size()) st.queue.push_back (q);
        }
    const juce::var vars = v.getProperty ("var", juce::var());
    const int nJunctions = std::max (0, (int) st.queue.size() - 1);
    st.variations.assign ((std::size_t) nJunctions, 0);
    if (vars.isArray())
        for (int i = 0; i < vars.size() && i < nJunctions; ++i)
            st.variations[(std::size_t) i] = std::max (0, (int) vars[i]);
    return st;
}

// Smart-default kind by position within an item: Intro at first 15 %,
// Outro at last 15 %, Verse otherwise. ADR-051 § Premium-feel #3.
inline reamix::theme::SegmentKind smartKindForPosition (double startSec, double itemDurationSec) noexcept
{
    if (itemDurationSec <= 0.0) return reamix::theme::SegmentKind::Verse;
    const double rel = startSec / itemDurationSec;
    if (rel < 0.15) return reamix::theme::SegmentKind::Intro;
    if (rel > 0.85) return reamix::theme::SegmentKind::Outro;
    return reamix::theme::SegmentKind::Verse;
}

} // namespace reamix::ui
