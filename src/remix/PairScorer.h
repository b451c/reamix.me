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
    // Sesja 129 (ADR-116 step 2) — voice-band log-mel END edges (n_total x n_edge_mel).
    const float*  edge_mel_end             = nullptr;
    int           n_edge_mel               = 0;

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
    // ADR-116 step 3 (sesja 130): score the pair as a BOUNDARY cut (leaves
    // at a phrase end, lands on a phrase start - src/remix/BoundaryFamily.h)
    // in the substitution view: beat i against beat j-1, no waveform term,
    // kV2BoundaryQualityWeights, loudness gates on end(i) vs end(j-1) and
    // rms(i) vs rms(j-1), no continuation penalties. v2 only (the legacy
    // path ignores it); the caller decides the family by geometry.
    bool   boundary    = false;
    // ADR-117 (sesja 131): the shape planner's relaxed tiers judge a
    // section-to-section seam without the per-track p98 loudness reject
    // (gate 2) - the natural section changes ARE the tail of that
    // distribution, so the reject removes every seam as large as the song's
    // own biggest step. The 8 dB hard block stays. Default off = every other
    // caller unchanged.
    bool   skip_loudness_reject = false;
    // Sesja 134 (ADR-117 step 3, DEV-121): also skip the 8 dB edge-energy
    // hard block (gate 1). The Audition round rated intro -> ending cut-ins
    // with +10 dB steps clean when the landing is a section start; the
    // planner asks for this only on such landings. Default off.
    bool   skip_energy_gate = false;
};

// Sesja 134 open-seam caps (edge view: the beat the ear leaves vs the beat
// it lands on). On the 27 rated seams of the two sesja-134 rounds (16
// Audition, 11 ours) the substitution-view gates and the composite do not
// separate "dynamika" from clean, the edge view does: every clean seam has
// |end(i) - start(j)| <= 11.3 dB and rms(j) / rms(i) <= 11.7 dB; the five
// cut-ins rated bad for their jump sit at 13.2 / 25 dB and 16 / 27 dB. The
// three remaining bad seams (Daft Punk verse -> outro) are not separable by
// any current signal (DEV-121). References/listening/2026-09-11-sesja134-
// audition-path/rated_seams_s134.json is the table.
inline constexpr double kOpenSeamMaxEdgeStepDb = 12.0;
inline constexpr double kOpenSeamMaxRmsJumpDb  = 14.0;
// Sesja 135 (DEV-121) open-seam gate 4: brightness collapse. The remaining
// "dynamika" family after the caps was Daft Punk's verse -> filtered-outro
// cut-ins (63 / 95 / 109 / 127 -> 379 / 393 / 411): a bright riff cut into
// the low-passed version of itself. centroidCollapseV2 (SignalNorm.h: the
// 8-beat context step in log centroid minus the song's own drop into the
// landing, in p90 consecutive-step units) puts them at -1.95 .. -2.52 and
// every rated-clean open seam at >= -1.00 (the two Alice seams from the
// un-beated head; real contexts >= -0.44); on the 51-case extreme corpus
// only those three planner seams and Woodkid x0.33 (-1.14, unrated) are
// below -1.0. Table: tools/dev/shape_eval/cutin_probe.py on
// references/listening/2026-09-11-sesja134-audition-path/rated_seams_s134.json.
inline constexpr int    kOpenSeamContextBeats        = 8;
inline constexpr double kOpenSeamMaxCentroidCollapse = 1.5;

struct PairScore
{
    bool   rejected        = false;   // a hard gate fired; quality is 0
    int    gate            = 0;       // 1 = edge-energy gate, 2 = loudness reject, 3 = open-seam edge cap (sesja 134), 4 = open-seam brightness collapse (sesja 135)
    double quality         = 0.0;     // composite minus penalties, clamped >= 0
    double energy_diff_db  = 0.0;     // |end(i) - start(j)| (legacy edge view)
    bool   has_waveform    = false;
    double waveform_sim    = 0.0;
    int    lag             = 0;
    double successor_sim   = 0.0;
    double edge_splice_sim = 0.0;
    double context_sim     = 0.0;
    int    family          = 0;       // sesja 130: TransitionCandidate::kFamily* (1 when scored as a boundary cut)
    double edge_distance   = -1.0;    // sesja 130: normalised edge distance d / scale (-1 = not available)
    // Sesja 134 diagnostics (harness --judge-seams): the composite's inputs
    // on the boundary path, so a gated or low-q seam can be read out.
    double tail_step_db    = 0.0;     // |end(i) - end(j-1)| (boundary view) or |end(i) - start(j)|
    double energy_match    = 0.0;
    double edge_energy_match = 0.0;
    double centroid_match  = 0.0;
    double transient_continuity = -1.0;   // -1 = not available
    double mfcc_continuity = -1.0;
    double edge_continuity = -1.0;
    // Sesja 135: centroidCollapseV2 of the pair on an open seam (0 when not
    // computed: gated earlier, not an open seam, or no centroid baseline).
    double centroid_collapse = 0.0;
};

PairScore scorePair(const PairScorerTrack& track, const PairScorerRequest& req);

} // namespace reamix::remix
