#pragma once
// ---------------------------------------------------------------------------
// SpliceAcceptance — the Duration acceptance vocabulary in one place
// (ADR-116 A4, started sesja 130 with the boundary family; Region / Blocks
// still carry their own copies until step 4 collapses them here).
//
// A remix is accepted at a TIER: the DP runs over the pool masked to the
// candidates that pass the tier, and the result holds when it ends at the
// song's ending, every cut is at or above kAcceptMinQ and the render length
// is inside the cap (RemixPipeline Duration v2). The tier means one thing
// per family:
//   continuation (family 0): waveform xcorr >= tier (DEV-116: rated-bad
//       cuts after the phrase gate had 0.64 / 0.77, every ok cut >= 0.84);
//   boundary (family 1): normalised edge distance d / scale <= cap(tier)
//       (sesja-129 probe: every ok boundary cut <= 0.62 of the track's
//       phrase-lag scale, every bad one >= 0.82) - the waveform xcorr does
//       not judge a section change (RC-2).
// ---------------------------------------------------------------------------

#include "remix/TransitionCost.h"

namespace reamix::remix {

// Every cut of an accepted path is at or above this composite quality.
inline constexpr double kAcceptMinQ = 0.45;

// Waveform-similarity tiers tried in order; 0.0 = the unfiltered pool.
inline constexpr double kAcceptTiers[] = { 0.80, 0.70, 0.60, 0.0 };
inline constexpr int    kNumAcceptTiers = 4;

// Boundary-family cap on d / scale at a waveform tier.
inline double boundaryEdgeCap(double waveform_tier) noexcept
{
    if (waveform_tier >= 0.80) return 0.60;
    if (waveform_tier >= 0.70) return 0.80;
    if (waveform_tier >= 0.60) return 1.00;
    return 1e300;   // unfiltered
}

// True when a candidate of `family` with this waveform xcorr / normalised
// edge distance is masked out of the pool at this tier (Duration pool and
// Blocks junction pools alike).
inline bool maskedAtTierRaw(int family, double waveform_similarity, double edge_distance, double tier) noexcept
{
    if (tier <= 0.0) return false;
    if (family == TransitionCandidate::kFamilyBoundary) {
        if (edge_distance < 0.0) return false;   // no edge scale on this track: never masked
        return edge_distance > boundaryEdgeCap(tier);
    }
    return waveform_similarity < tier;
}

inline bool maskedAtTier(const TransitionCandidate& c, double tier) noexcept
{
    return maskedAtTierRaw(c.family, c.waveform_similarity, c.edge_distance, tier);
}

} // namespace reamix::remix
