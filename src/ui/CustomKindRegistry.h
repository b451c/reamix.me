#pragma once

#include <juce_core/juce_core.h>
#include <juce_graphics/juce_graphics.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// CustomKindRegistry — global per-user store of custom Block Assembly kindy
// beyond the 12 built-in SegmentKind values (Theme.h). Each entry has a
// stable string ID generated at add time; user-visible attributes (name,
// color) can be renamed/recolored without changing ID, so all UserBlocks
// referencing that ID stay valid through edits.
//
// Persistence: serialized to JSON, persisted via REAPER ExtState
// "reamix.custom_kinds" (global per-user, NOT per-project — sesja 100a Q4).
//
// Per ADR-092 (sesja 100c) — Option B Registry-only design.
//
// Stable ID format: "ck_" + 8 hex chars (juce::Random); collision check on add.
//
// Lookup-miss semantics: when a UserBlock references an ID that no longer
// exists in the registry (e.g. user opened project on machine without the
// custom kindy), kindDisplay() falls back to UserBlock.kind (built-in).

namespace reamix::ui
{

struct CustomKindEntry
{
    juce::String  name;
    juce::Colour  color;
};

class CustomKindRegistry
{
public:
    CustomKindRegistry() = default;
    ~CustomKindRegistry() = default;

    // Add a new custom kind. Returns the generated stable ID.
    // Caller responsible for triggering save() afterwards.
    juce::String add (const juce::String& name, juce::Colour color);

    // Mutate an existing entry. Silently no-op if id not found.
    void rename  (const juce::String& id, const juce::String& newName);
    void recolor (const juce::String& id, juce::Colour newColor);

    // Remove. Returns true if entry was present (caller may need to know
    // for cascade UI confirmation). Caller responsible for cascading on
    // any UserBlocks referencing the deleted ID.
    bool remove (const juce::String& id);

    // Lookup. Returns std::nullopt on miss (acceptable — caller falls back
    // to UserBlock.kind built-in).
    std::optional<CustomKindEntry> lookup (const juce::String& id) const;

    // Ordered iteration (insertion order via insertionOrder_) — picker
    // shows custom kindy in the order user added them.
    std::vector<std::pair<juce::String, CustomKindEntry>> all() const;

    int size() const noexcept { return (int) insertionOrder_.size(); }
    bool empty() const noexcept { return insertionOrder_.empty(); }

    // Persistence — JSON array of {id, name, color_hex}.
    juce::String serialize() const;
    void         deserialize (const juce::String& json);

    // Test-only deterministic ID seed (for unit tests). Production uses
    // juce::Random::getSystemRandom().
    void setIdSeedForTesting (juce::int64 seed);

private:
    std::unordered_map<std::string, CustomKindEntry> entries_;
    std::vector<juce::String>                        insertionOrder_;

    // ID generation state — nullopt means use system random (production).
    std::optional<juce::int64> testSeed_;

    juce::String generateUniqueId();
};

} // namespace reamix::ui
