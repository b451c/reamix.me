#include "CustomKindRegistry.h"

#include <juce_core/juce_core.h>

namespace reamix::ui
{

namespace
{
juce::String hexFromInt64 (juce::int64 v)
{
    juce::String hex = juce::String::toHexString ((juce::int64) (v & 0xFFFFFFFFLL));
    while (hex.length() < 8) hex = "0" + hex;
    return hex.substring (0, 8);
}
} // namespace

juce::String CustomKindRegistry::generateUniqueId()
{
    // Try up to 8 attempts before giving up — collision space 2^32 makes
    // duplicate generation extremely improbable; loop is defensive.
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        juce::int64 raw;
        if (testSeed_.has_value())
        {
            // Deterministic for unit tests: hash seed + insertion count + attempt.
            raw = (*testSeed_) ^ ((juce::int64) insertionOrder_.size() << 16)
                  ^ ((juce::int64) attempt);
        }
        else
        {
            raw = juce::Random::getSystemRandom().nextInt64();
        }

        const juce::String candidate = "ck_" + hexFromInt64 (raw);
        if (entries_.find (candidate.toStdString()) == entries_.end())
            return candidate;
    }
    // Fallback — append insertion-count suffix (still unique by construction).
    return "ck_fallback_" + juce::String ((int) insertionOrder_.size());
}

juce::String CustomKindRegistry::add (const juce::String& name, juce::Colour color)
{
    const juce::String id = generateUniqueId();
    entries_[id.toStdString()] = { name, color };
    insertionOrder_.push_back (id);
    return id;
}

void CustomKindRegistry::rename (const juce::String& id, const juce::String& newName)
{
    auto it = entries_.find (id.toStdString());
    if (it != entries_.end()) it->second.name = newName;
}

void CustomKindRegistry::recolor (const juce::String& id, juce::Colour newColor)
{
    auto it = entries_.find (id.toStdString());
    if (it != entries_.end()) it->second.color = newColor;
}

bool CustomKindRegistry::remove (const juce::String& id)
{
    auto it = entries_.find (id.toStdString());
    if (it == entries_.end()) return false;

    entries_.erase (it);
    for (auto orderIt = insertionOrder_.begin(); orderIt != insertionOrder_.end(); ++orderIt)
    {
        if (*orderIt == id) { insertionOrder_.erase (orderIt); break; }
    }
    return true;
}

std::optional<CustomKindEntry> CustomKindRegistry::lookup (const juce::String& id) const
{
    auto it = entries_.find (id.toStdString());
    if (it == entries_.end()) return std::nullopt;
    return it->second;
}

std::vector<std::pair<juce::String, CustomKindEntry>> CustomKindRegistry::all() const
{
    std::vector<std::pair<juce::String, CustomKindEntry>> out;
    out.reserve (insertionOrder_.size());
    for (const auto& id : insertionOrder_)
    {
        auto it = entries_.find (id.toStdString());
        if (it != entries_.end())
            out.push_back ({ id, it->second });
    }
    return out;
}

juce::String CustomKindRegistry::serialize() const
{
    juce::Array<juce::var> arr;
    for (const auto& id : insertionOrder_)
    {
        auto it = entries_.find (id.toStdString());
        if (it == entries_.end()) continue;

        auto* obj = new juce::DynamicObject();
        obj->setProperty ("id",    id);
        obj->setProperty ("name",  it->second.name);
        // Color encoded as ARGB hex string for human-readable JSON.
        obj->setProperty ("color", it->second.color.toDisplayString (true));
        arr.add (juce::var (obj));
    }
    return juce::JSON::toString (juce::var (arr), true);
}

void CustomKindRegistry::deserialize (const juce::String& json)
{
    entries_.clear();
    insertionOrder_.clear();

    if (json.isEmpty()) return;

    const juce::var v = juce::JSON::parse (json);
    if (! v.isArray()) return;

    for (int i = 0; i < v.size(); ++i)
    {
        const juce::var& e = v[i];
        if (! e.isObject()) continue;

        const juce::String id    = e.getProperty ("id",    juce::String()).toString();
        const juce::String name  = e.getProperty ("name",  juce::String()).toString();
        const juce::String hex   = e.getProperty ("color", juce::String()).toString();

        if (id.isEmpty() || name.isEmpty() || hex.isEmpty()) continue;
        // Validate ID format ("ck_" prefix) — silently skip malformed entries.
        if (! id.startsWith ("ck_")) continue;
        // Reject duplicates from corrupt JSON.
        if (entries_.find (id.toStdString()) != entries_.end()) continue;

        const juce::Colour color = juce::Colour::fromString (hex);

        entries_[id.toStdString()] = { name, color };
        insertionOrder_.push_back (id);
    }
}

void CustomKindRegistry::setIdSeedForTesting (juce::int64 seed)
{
    testSeed_ = seed;
}

} // namespace reamix::ui
