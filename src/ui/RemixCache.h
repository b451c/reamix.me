#pragma once

#include <juce_core/juce_core.h>

#include <cmath>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "RemixOutput.h"
#include "../remix/Quality.h"

// RemixCache — bounded LRU keyed by (sourcePath, targetSec_q50ms,
// region, blockedSet, variation). Per-source bound 20 entries.
//
// Per-key tmp WAV files: each cached RemixOutput owns a unique
// `reamix_<keyHash>.wav` in juce tempDirectory. On eviction or destroy
// the caller deletes the file (this class returns evicted paths via
// `evictedWavPaths`).
//
// Reference: ADR-047 § 1 — quality-first cache (user upgrade from
// initial PROPOSED no-cache plan).

namespace reamix::ui
{

struct RemixCacheKey
{
    juce::String sourcePath;
    // ADR-056 (sesja 66) — composite identity (sourcePath, itemGuid).
    // Disambiguates between multiple MediaItems referencing the same audio
    // file (e.g. original whole item vs post-Region-Insert pre/post-region
    // pieces vs duplicated copies). Empty itemGuid = legacy / synthetic
    // entries (parity tests, source-only previews); non-empty = REAPER item
    // identity bound. Closes BUG-20 + BUG-22 root-cause: per-source state
    // leaks between distinct REAPER items sharing the same audio file.
    juce::String itemGuid;
    int          targetMs50    { 0 }; // round(targetSec * 20.0) — 50 ms quantum
    int          regionStartMs50 { 0 };
    int          regionEndMs50   { 0 };
    juce::uint64 blockedHash   { 0 };
    int          variation     { 0 };
    // ADR-051 (sesja 61) — Block Assembly identity hash (queue + per-block
    // boundaries + kinds + junctionVariations + spliceFlexBeats). Zero in
    // non-Blocks mode (auto / Region cache continues to work unchanged).
    juce::uint64 blocksHash    { 0 };
    // ADR-080 RESCOPE + ADR-083 (sesja 92) — AuditionBar 4-slider identity
    // hash (tone + edit_length + allow_pm_seconds + min_cut_beats). Zero
    // when all four sliders at default (bit-exact baseline) — preserves
    // pre-sesja-92 cache entries lookup compatibility. Non-zero invalidates
    // old cache hits when user drags any slider off default → forces fresh
    // remix render.
    juce::uint64 auditionHash  { 0 };

    // ADR-087 STATUS UPDATE 1 (sesja 98) + ADR-097 sesja 107 — advanced
    // weights QualityWeights identity hash. Zero when weights ==
    // kDefaultQualityWeights (bit-exact baseline) — preserves pre-sesja-98
    // cache entries lookup compatibility. Non-zero invalidates old cache
    // hits when the user tweaks any cost weight via the Advanced window.
    // Users who never open the window never write a non-zero value here.
    juce::uint64 qualityWeightsHash { 0 };

    bool operator< (const RemixCacheKey& o) const
    {
        if (sourcePath     != o.sourcePath)     return sourcePath     < o.sourcePath;
        if (itemGuid       != o.itemGuid)       return itemGuid       < o.itemGuid;
        if (targetMs50     != o.targetMs50)     return targetMs50     < o.targetMs50;
        if (regionStartMs50!= o.regionStartMs50)return regionStartMs50< o.regionStartMs50;
        if (regionEndMs50  != o.regionEndMs50)  return regionEndMs50  < o.regionEndMs50;
        if (blockedHash    != o.blockedHash)    return blockedHash    < o.blockedHash;
        if (variation      != o.variation)      return variation      < o.variation;
        if (blocksHash     != o.blocksHash)     return blocksHash     < o.blocksHash;
        if (auditionHash   != o.auditionHash)   return auditionHash   < o.auditionHash;
        return qualityWeightsHash < o.qualityWeightsHash;
    }

    bool operator== (const RemixCacheKey& o) const
    {
        return sourcePath == o.sourcePath
            && itemGuid   == o.itemGuid
            && targetMs50 == o.targetMs50
            && regionStartMs50 == o.regionStartMs50
            && regionEndMs50 == o.regionEndMs50
            && blockedHash == o.blockedHash
            && variation == o.variation
            && blocksHash == o.blocksHash
            && auditionHash == o.auditionHash
            && qualityWeightsHash == o.qualityWeightsHash;
    }
};

inline juce::uint64 hashBlockedTransitions (const std::set<std::pair<int,int>>& s)
{
    juce::uint64 h = 1469598103934665603ull; // FNV-1a 64-bit offset basis
    for (const auto& p : s)
    {
        h ^= (juce::uint64) p.first;
        h *= 1099511628211ull;
        h ^= (juce::uint64) p.second;
        h *= 1099511628211ull;
    }
    return h;
}

// ADR-080 RESCOPE + ADR-083 (sesja 92) — hash of AuditionBar 4-slider state.
// Returns 0 when all 4 sliders at bit-exact baseline defaults (tone=0.0,
// editLength=50, allowPm=5, minCut=16) so cache entries created pre-sesja-92
// or with sliders untouched compare equal in cache lookups → preserves
// existing remix outputs across plugin upgrade. Non-zero hash invalidates
// stale cache hits the moment user drags any slider off default.
inline juce::uint64 hashAuditionParams (double tone, int editLength,
                                        int allowPmSeconds, int minCutBeats)
{
    const bool atDefault = (tone == 0.0)
                        && (editLength     == 50)
                        && (allowPmSeconds == 5)
                        && (minCutBeats    == 16);
    if (atDefault) return 0;

    juce::uint64 h = 1469598103934665603ull; // FNV-1a 64-bit offset basis
    h ^= (juce::uint64) std::lround (tone * 10000.0);
    h *= 1099511628211ull;
    h ^= (juce::uint64) editLength;
    h *= 1099511628211ull;
    h ^= (juce::uint64) allowPmSeconds;
    h *= 1099511628211ull;
    h ^= (juce::uint64) minCutBeats;
    h *= 1099511628211ull;
    return h;
}

// ADR-087 STATUS UPDATE 1 D2 (sesja 98) — true when QualityWeights match
// kDefaultQualityWeights bit-for-bit on the 7 active fields + flag fields.
// Used by hashQualityWeights atDefault sentinel + by RemixPipeline.cpp
// Path A guard (Duration mode re-compute trigger).
//
// Compares only the production-active fields (waveform, sequential_continuity,
// transient_continuity, energy, edge_energy, bar_align, centroid) plus the
// non-simplex flags (use_harmonic_mean, harmonic_vs_timbre). Inactive fields
// (successor, edge_splice, context, label, section, mfcc_continuity, extra1)
// are forced to 0.0 by the perceptual mapping function and by kDefaultQualityWeights,
// so equality on them is implicit.
inline bool qualityWeightsAtDefault (const reamix::remix::QualityWeights& w) noexcept
{
    const auto& d = reamix::remix::kDefaultQualityWeights;
    return w.waveform              == d.waveform
        && w.sequential_continuity == d.sequential_continuity
        && w.transient_continuity  == d.transient_continuity
        && w.energy                == d.energy
        && w.edge_energy           == d.edge_energy
        && w.bar_align             == d.bar_align
        && w.centroid              == d.centroid
        && w.vocal_continuity      == d.vocal_continuity   // ADR-088 sesja 98
        && w.use_harmonic_mean     == d.use_harmonic_mean
        && w.harmonic_vs_timbre    == d.harmonic_vs_timbre;
}

// ADR-087 STATUS UPDATE 1 (sesja 98) — FNV-1a 64-bit hash of QualityWeights.
// Returns 0 when QualityWeights == kDefaultQualityWeights so cache entries
// created pre-sesja-98 (or with dev sliders untouched) compare equal in cache
// lookups → preserves existing remix outputs. Non-zero hash invalidates
// stale cache hits the moment developer tweaks any cost weight.
inline juce::uint64 hashQualityWeights (const reamix::remix::QualityWeights& w)
{
    if (qualityWeightsAtDefault (w)) return 0;

    juce::uint64 h = 1469598103934665603ull; // FNV-1a 64-bit offset basis
    const auto mix = [&h] (double v)
    {
        h ^= (juce::uint64) std::llround (v * 1.0e9);
        h *= 1099511628211ull;
    };
    mix (w.waveform);
    mix (w.sequential_continuity);
    mix (w.transient_continuity);
    mix (w.energy);
    mix (w.edge_energy);
    mix (w.bar_align);
    mix (w.centroid);
    mix (w.vocal_continuity);  // ADR-088 sesja 98
    mix (w.harmonic_vs_timbre);
    h ^= (juce::uint64) (w.use_harmonic_mean ? 1ull : 0ull);
    h *= 1099511628211ull;
    return h;
}

inline RemixCacheKey makeRemixCacheKey (const juce::String& sourcePath,
                                        const juce::String& itemGuid,
                                        double               targetSec,
                                        double               regionStartSec,
                                        double               regionEndSec,
                                        const std::set<std::pair<int,int>>& blocked,
                                        int                  variation)
{
    RemixCacheKey k;
    k.sourcePath      = sourcePath;
    k.itemGuid        = itemGuid;
    k.targetMs50      = (int) std::lround (targetSec * 20.0);       // 50 ms quantum
    k.regionStartMs50 = (int) std::lround (regionStartSec * 20.0);
    k.regionEndMs50   = (int) std::lround (regionEndSec * 20.0);
    k.blockedHash     = hashBlockedTransitions (blocked);
    k.variation       = variation;
    return k;
}

inline juce::String tmpWavPathFor (const RemixCacheKey& k)
{
    // Stable file name from key fields. Hash for compactness; collision
    // would just mean overwrite of one cached entry by another, but the
    // 53-bit hash space + bounded 20-per-source LRU makes that astronomical.
    juce::uint64 h = (juce::uint64) k.sourcePath.hashCode64();
    h ^= ((juce::uint64) k.itemGuid.hashCode64()) * 0xCBF29CE484222325ull;
    h ^= ((juce::uint64) k.targetMs50) * 0x9E3779B97F4A7C15ull;
    h ^= ((juce::uint64) k.regionStartMs50) * 0xBF58476D1CE4E5B9ull;
    h ^= ((juce::uint64) k.regionEndMs50) * 0x94D049BB133111EBull;
    h ^= k.blockedHash;
    h ^= ((juce::uint64) k.variation) * 0xD1B54A32D192ED03ull;
    h ^= k.blocksHash * 0xC2B2AE3D27D4EB4Full;
    h ^= k.auditionHash * 0x14057B7EF767814Full;  // ADR-083 sesja 92
    juce::String name = "reamix_" + juce::String::toHexString ((juce::int64) h) + ".wav";
    return juce::File::getSpecialLocation (juce::File::tempDirectory)
                .getChildFile (name).getFullPathName();
}

class RemixCacheLRU
{
public:
    explicit RemixCacheLRU (int perSourceBound) : perSourceBound_ (perSourceBound) {}

    // Returns nullptr if not found. Bumps the entry to most-recently-used.
    const RemixOutput* find (const RemixCacheKey& key)
    {
        auto it = index_.find (key);
        if (it == index_.end()) return nullptr;
        entries_.splice (entries_.begin(), entries_, it->second);
        return &(it->second->second);
    }

    // Insert (or replace existing). Evicts oldest entry from same source if
    // per-source count is at bound. Returns the WAV path of any evicted entry
    // so the caller can delete the file (empty if no eviction).
    juce::String insert (const RemixCacheKey& key, RemixOutput out)
    {
        juce::String evictedPath;

        auto existing = index_.find (key);
        if (existing != index_.end())
        {
            existing->second->second = std::move (out);
            entries_.splice (entries_.begin(), entries_, existing->second);
            return evictedPath; // no eviction; replaced in place
        }

        // Count entries for this source; evict oldest if at bound.
        int countSrc = 0;
        for (const auto& e : entries_)
            if (e.first.sourcePath == key.sourcePath) ++countSrc;

        if (countSrc >= perSourceBound_)
        {
            // Walk from oldest end backwards to find the LRU entry for this
            // source.
            for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
            {
                if (it->first.sourcePath == key.sourcePath)
                {
                    evictedPath = it->second.tmpWavPath;
                    auto fwdIt = std::next (it).base(); // reverse → forward
                    index_.erase (fwdIt->first);
                    entries_.erase (fwdIt);
                    break;
                }
            }
        }

        entries_.emplace_front (key, std::move (out));
        index_[key] = entries_.begin();
        return evictedPath;
    }

    // Drop all entries matching this source. Returns evicted WAV paths.
    std::vector<juce::String> evictSource (const juce::String& sourcePath)
    {
        std::vector<juce::String> paths;
        for (auto it = entries_.begin(); it != entries_.end(); )
        {
            if (it->first.sourcePath == sourcePath)
            {
                paths.push_back (it->second.tmpWavPath);
                index_.erase (it->first);
                it = entries_.erase (it);
            }
            else
            {
                ++it;
            }
        }
        return paths;
    }

    // Plugin destruction — return all WAV paths so caller can delete files.
    std::vector<juce::String> evictAll()
    {
        std::vector<juce::String> paths;
        paths.reserve (entries_.size());
        for (auto& e : entries_) paths.push_back (e.second.tmpWavPath);
        entries_.clear();
        index_.clear();
        return paths;
    }

    int size() const { return (int) entries_.size(); }

private:
    using EntryList = std::list<std::pair<RemixCacheKey, RemixOutput>>;
    int        perSourceBound_;
    EntryList  entries_;                                   // front = most recent
    std::map<RemixCacheKey, EntryList::iterator> index_;
};

} // namespace reamix::ui
