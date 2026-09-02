#pragma once

#include <optional>

#include "remix/Quality.h"
#include "remix/SignalNorm.h"

namespace reamix::remix {

// Shared per-pair splice scorer (sesja 119, DEV-096).
//
// One scoring body for Region and Block Assembly so the two modes cannot
// drift apart again: hard gates (legacy absolute 8 dB or the v2 successor
// view, ADR-115 E8 / DEV-091), the ADR-115 E2 p98 loudness reject, the
// signal qualities (legacy formulas or the E1 sequential-baseline mappings),
// the composite (`computeQualityScore`) and the vocal / onset penalties.
//
// Everything is addressed by ABSOLUTE beat index over the whole track. The
// caller owns candidate generation (bar alignment, micro-skip, chroma
// prefilter, repetition prior, block windows) and any mode-specific term
// applied after the composite (Region span penalty, Block drift / fragment
// penalties).
//
// Region legacy parity: the successor and edge-splice similarities may be
// passed in from the caller's precomputed f32 matrices (`region_cost.py`
// bit-exact path); when absent they are computed inline in f64, which is
// the Block Assembly formula. The context similarity is the mean-feature
// cosine over [max(ctx_lo, i - 2), i + 1) x [j, min(ctx_hi, j + 3)); Region
// passes its region bounds, Blocks the whole track.
//
// C++-canonical (ADR-065): no Python source for the shared body; the two
// legacy paths it replaces stay bit-exact through the existing parity tests.

enum class PairGate
{
    None,             // no edge-energy hard gate (Block Assembly legacy: graduated penalty instead)
    LegacyAbsolute,   // |end(i) - start(j)| > ENERGY_HARD_BLOCK_DB rejects (Python parity)
    SuccessorView     // ADR-115 E8: |start(j) - start(i+1)| or |end(i) - end(j-1)| > 8 dB rejects
};

// Whole-track scoring context. All pointers are caller-owned and optional
// unless stated; a null pointer drops the matching signal exactly as the
// mode-specific scorers did.
struct PairScorerTrack
{
    int          n_total    = 0;      // required
    int          n_features = 0;      // required
    const float* features   = nullptr; // (n_total, n_features) row-major, required

    // Waveform xcorr (boundary snippets per beat).
    const float* boundary_waveforms = nullptr;
    int          n_samples_per_bnd  = 0;
    int          max_lag            = 0;
    bool         has_waveforms      = false;

    // Edge-energy dB arrays (whole track) - both or neither.
    const double* edge_db_end   = nullptr;
    const double* edge_db_start = nullptr;

    // Edge features for the inline edge-splice cosine (used only when the
    // request carries no precomputed edge_splice_sim).
    const float* edge_features_start = nullptr;
    const float* edge_features_end   = nullptr;
    int          n_edge_features     = 0;

    // Per-beat scalars.
    const double* rms_energy               = nullptr;
    const double* spectral_centroid        = nullptr;
    const double* onset_strength           = nullptr;
    const double* vocal_activity           = nullptr;
    const double* edge_vocal_activity_start = nullptr;
    const double* edge_vocal_activity_end   = nullptr;
    const double* edge_vocal_onset_start   = nullptr;
    const double* edge_vocal_release_end   = nullptr;

    // Precomputed whole-track helpers (empty / null = signal absent).
    const double* onset_norm               = nullptr;  // (n_total)
    int           onset_norm_n             = 0;
    const double* mfcc_continuity_matrix   = nullptr;  // (n_total x n_total)
    const double* chroma_continuity_matrix = nullptr;  // (n_total x n_total)

    // Context-window clip bounds [ctx_lo, ctx_hi) in absolute beats.
    int ctx_lo = 0;
    int ctx_hi = 0;

    // Scoring policy.
    bool                   v2               = false;
    const SignalBaselines* baselines        = nullptr;   // required when v2
    const QualityWeights*  weights          = nullptr;   // required (resolved by the caller)
    bool                   track_has_vocals = false;     // Region TRACK_VOCAL_THRESHOLD gate
    PairGate               gate             = PairGate::None;
    // Block Assembly legacy soft term: energy penalty 0..0.30 above 6 dB on
    // the legacy edge view (block_assembly.py:310-316). Off on the v2 path.
    bool                   graduated_energy_penalty = false;
};

struct PairScorerRequest
{
    int abs_i = 0;   // source beat (splice leaves after beat i)
    int abs_j = 0;   // destination beat (splice lands on beat j)

    std::optional<double> successor_sim;    // caller-precomputed (Region f32); nullopt = inline f64
    std::optional<double> edge_splice_sim;  // same

    double label_match = 0.0;
    double section_sim = 0.0;
    double bar_aligned = 0.0;
};

struct PairScore
{
    bool   rejected        = false;   // a hard gate fired; quality is 0
    int    gate            = 0;       // 1 = edge-energy gate, 2 = loudness reject
    double quality         = 0.0;     // composite minus penalties, clamped >= 0
    double energy_diff_db  = 0.0;     // |end(i) - start(j)| (legacy edge view)
    bool   has_waveform    = false;
    double waveform_sim    = 0.0;
    int    lag             = 0;
    double successor_sim   = 0.0;
    double edge_splice_sim = 0.0;
    double context_sim     = 0.0;
};

PairScore scorePair(const PairScorerTrack& track, const PairScorerRequest& req);

} // namespace reamix::remix
