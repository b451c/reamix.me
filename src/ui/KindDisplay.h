#pragma once

#include "Theme.h"
#include "UserBlock.h"
#include "CustomKindRegistry.h"

#include <juce_core/juce_core.h>
#include <juce_graphics/juce_graphics.h>

// kindDisplay — single resolution helper used by every paint / picker /
// dropdown site that needs the user-visible name + color for a UserBlock.
//
// ADR-092 sesja 100c — Option B Registry-only design. When `customKindId`
// is set on the UserBlock and the registry knows the id, custom kind name
// + color win. Otherwise (no customKindId, OR registry-miss) the built-in
// Theme::SegmentKind name + color are used.
//
// Header-only because only the helper itself + label-from-enum lookup are
// inline; real paint code consumes the returned struct.
//
// Built-in label lookup mirrors existing duplicated switches in
// BlockEditPopover / BlockKindPickerPopover / BlockAssemblyPanel — we
// keep it single-source here and call from those sites in a follow-up.

namespace reamix::ui
{

struct KindDisplay
{
    juce::String name;
    juce::Colour color;
};

inline juce::String builtinKindLabel (reamix::theme::SegmentKind k)
{
    using SK = reamix::theme::SegmentKind;
    switch (k)
    {
        case SK::Intro:        return "Intro";
        case SK::Verse:        return "Verse";
        case SK::PreChorus:    return "Pre-Chorus";
        case SK::Chorus:       return "Chorus";
        case SK::PostChorus:   return "Post-Chorus";
        case SK::Bridge:       return "Bridge";
        case SK::Buildup:      return "Buildup";
        case SK::Drop:         return "Drop";
        case SK::Breakdown:    return "Breakdown";
        case SK::Solo:         return "Solo";
        case SK::Instrumental: return "Instrumental";
        case SK::Outro:        return "Outro";
        case SK::NumKinds:     break;
    }
    return {};
}

// Resolve a UserBlock to its display attributes.
//   1. labelOverride (per-block name override) wins for the name field.
//   2. customKindId + registry hit → registry name + color.
//   3. registry miss OR no customKindId → built-in kind name + color.
inline KindDisplay kindDisplay (const UserBlock& b, const CustomKindRegistry& reg)
{
    KindDisplay out;

    if (b.customKindId.has_value())
    {
        if (auto entry = reg.lookup (*b.customKindId); entry.has_value())
        {
            out.name  = entry->name;
            out.color = entry->color;
        }
        else
        {
            // Registry-miss — fall back to built-in.
            out.name  = builtinKindLabel (b.kind);
            out.color = reamix::theme::segmentColour (b.kind);
        }
    }
    else
    {
        out.name  = builtinKindLabel (b.kind);
        out.color = reamix::theme::segmentColour (b.kind);
    }

    if (b.labelOverride.has_value() && b.labelOverride->isNotEmpty())
        out.name = *b.labelOverride;

    return out;
}

// Resolve a SegmentKind / customKindId pair when no UserBlock is in hand
// (e.g. picker draws built-in 12 then iterates registry; this helper covers
// the picker built-in path without faking a UserBlock).
inline KindDisplay kindDisplayBuiltin (reamix::theme::SegmentKind k)
{
    return { builtinKindLabel (k), reamix::theme::segmentColour (k) };
}

inline KindDisplay kindDisplayCustom (const juce::String& id,
                                      const CustomKindRegistry& reg)
{
    if (auto entry = reg.lookup (id); entry.has_value())
        return { entry->name, entry->color };
    return { {}, reamix::theme::Fg3 };  // miss → caller treats as orphaned id
}

} // namespace reamix::ui
