#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "analysis/StructureResult.h"
#include "remix/Path.h"
#include "remix/Quality.h"  // QualityWeights override (ADR-058)

namespace reamix::remix {

// Block assembly: user-driven section arrangement.
//
// Port of `references/python-source/remix/block_assembly.py` (523 LOC,
// 2026-04-22). Session-24 target per HANDOVER-23.
//
// After structure analysis the user can manually order song sections
// (Intro → Verse → Chorus → Verse → Chorus → Outro); this module:
//
//   1. Enriches raw segments into `BlockInfo` — beat-range bounds + display
//      names with disambiguation (Verse 1 / Verse 2 / etc.).
//   2. Pre-computes a block compatibility matrix (quality score + optimal
//      splice points per block pair) so the UI can render green/yellow/red
//      junction indicators instantly.
//   3. Assembles a user-chosen block sequence into a `RemixPath` that the
//      existing renderer / preview pipeline consumes.
//
// The heavy lift is `computeBlockCompatibility`: for every block pair (i, j)
// it searches a ±8-beat window around the boundary exit(i) → entry(j),
// scoring each candidate with the same 10-signal `computeQualityScore` (from
// Quality.cpp) that TransitionCost + RegionCost use, PLUS a graduated energy
// penalty (not a hard gate — user explicitly chose this arrangement so the
// energy mismatch is softened rather than blocked). The top-K alternatives
// per pair are kept so the UI can offer "Try Another" per-junction re-roll.
//
// --- Citation discipline (ADR-026) ----------------------------------------
// Every hand-calibrated number in this file carries a
// `block_assembly.py:LINE (2026-04-22)` source citation. Session-24 pre-port
// audit classified 23 main-path constants (B1-B23) in `weights-audit.md` —
// see session-24 log entry for the full taxonomy breakdown.
//
// --- VALUE-DRIFT: 3rd instance of the 12.0 dB pattern (ADR-NNN) -----------
// `block_assembly.py:368` reuses the same `edge_energy_match = max(0, 1 -
// min(diff, 12.0) / 12.0)` formula as transition_cost.py:477 (session 18)
// and region_cost.py:360 (session 23). 3rd instance TRIGGERS the
// HANDOVER-19 L114 consolidation-ADR threshold. BlockAssembly.cpp consumes
// the canonical `EDGE_ENERGY_SATURATION_DB` constant from TransitionCost.h
// (promoted session 18) — Hard Rule #1 single-source-of-truth; 3 consumers
// now (TransitionCost + RegionCost via session-24 retrofit + BlockAssembly).
//
// --- FMA contraction (ADR-028) --------------------------------------------
// `_score_junction` composes weighted sums through `computeQualityScore`;
// parity against CPython requires `-ffp-contract=off` on the test target.
// 7th reuse per ADR-028.

// --- Module constants (ADR-026 audit, session 24) -----------------------

// Number of alternative splice candidates per block pair. UI affordance:
// variation=0 returns the best splice, variation=1..4 return the 2nd..5th
// best, enabling "Try Another" and per-junction re-roll.
// UNJUSTIFIED per session-24 audit (B1) — HIGH priority (UX variation).
// Source-of-truth: block_assembly.py:31 (2026-04-22).
inline constexpr int BLOCK_TOP_K = 5;

// Crossfade duration table — applied to each block transition based on
// transition quality. Graduated: better quality → shorter crossfade.
// All three UNJUSTIFIED per session-24 audit (B2/B3/B4).
//   B2 STANDARD=75.0 matches RemixConfig.crossfade_ms (duplication).
//   B4 LONG=300.0 matches RemixConfig.vocal_crossfade_ms (duplication).
// Source-of-truth: block_assembly.py:34-36 (2026-04-22).
inline constexpr double BLOCK_CROSSFADE_STANDARD_MS = 75.0;
inline constexpr double BLOCK_CROSSFADE_EXTENDED_MS = 150.0;
inline constexpr double BLOCK_CROSSFADE_LONG_MS     = 300.0;

// Quality color thresholds for UI green/yellow/red indicator and crossfade
// graduation. Both UNJUSTIFIED per session-24 audit (B5/B6).
// Source-of-truth: block_assembly.py:37-38 (2026-04-22).
inline constexpr double BLOCK_QUALITY_GREEN  = 0.65;
inline constexpr double BLOCK_QUALITY_YELLOW = 0.40;

// Search window ±SEARCH_WINDOW beats around each boundary for the best
// splice point. "~2.7s at 176 BPM" per Python comment; hardcoded flat 8
// beats regardless of BPM. UNJUSTIFIED per session-24 audit (B7); HIGH
// priority — scoring cost is O(n² × (2W+1)²), so a change ripples.
// Source-of-truth: block_assembly.py:186 (2026-04-22).
//
// ADR-051 (sesja 61): default kept for parity with Python golden goldens;
// runtime callers (UI Block Assembly mode) override via
// BlockCompatInputs::search_window_beats — driven by the user's "Splice
// flexibility" Settings cycler (Tight=4 / Medium=8 / Loose=16).
inline constexpr int BLOCK_SEARCH_WINDOW_BEATS = 8;

// ADR-051 (sesja 61) — drift penalty weight. Penalises splice points that
// drift from the user's authored block boundary; cost units (added to
// transition_cost). Higher = stricter "respect user intent"; lower =
// algorithm prefers cleaner splices over user-faithful boundaries.
//
// Initial guess 0.10. Quality-tuning sesja 65 calibrates against perceptual
// 16-track A/B (ADR-035 + ADR-051 § Open follow-up #1).
inline constexpr double BLOCK_DRIFT_PENALTY_WEIGHT = 0.10;

// Graduated energy penalty: 0 below threshold, ramps up with slope up to
// cap. Python comment: "0 at 6 dB, 0.15 at 12 dB, 0.30 at 18 dB".
// UNJUSTIFIED per session-24 audit (B10/B11/B12).
// Source-of-truth: block_assembly.py:314-316 (2026-04-22).
inline constexpr double BLOCK_ENERGY_PENALTY_THRESHOLD_DB = 6.0;
inline constexpr double BLOCK_ENERGY_PENALTY_CAP          = 0.30;
inline constexpr double BLOCK_ENERGY_PENALTY_SLOPE        = 0.025;

// Context-window radii (asymmetric — session-23 C11 pattern). Session-24
// audit class B14 (UNJUSTIFIED-DRIFT). Hardcoded despite DEAD-CONFIG
// `config.context_window_beats=2`.
// Source-of-truth: block_assembly.py:348-349 (2026-04-22).
inline constexpr int BLOCK_CONTEXT_BEFORE_LO_OFFSET = -2;  // lo = bi + offset
inline constexpr int BLOCK_CONTEXT_AFTER_HI_OFFSET  =  3;  // hi = bj + offset

// Label match sentinel — "unknown" labels do NOT count as matching.
// MARKER per session-24 audit (B15).
// Source-of-truth: block_assembly.py:358 (2026-04-22).
inline constexpr const char* BLOCK_LABEL_UNKNOWN = "unknown";

// Section similarity scaler: label_match * 0.9 + 0.1 (so unknown-unknown
// pairs still contribute 0.1 baseline). UNJUSTIFIED per session-24 audit
// (B16).
// Source-of-truth: block_assembly.py:360 (2026-04-22).
inline constexpr double BLOCK_SECTION_SIM_SCALE = 0.9;
inline constexpr double BLOCK_SECTION_SIM_BIAS  = 0.1;

// Scalar-diff scales for energy/centroid soft matches — UNJUSTIFIED per
// session-24 audit (B17/B19). Numerically match TransitionCost::RMS_DIFF_SCALE
// and CENTROID_DIFF_SCALE (both 5.0) but defined locally because Python
// block_assembly.py hardcodes them (no import).
// Source-of-truth: block_assembly.py:364 + :372 (2026-04-22).
inline constexpr double BLOCK_RMS_DIFF_SCALE      = 5.0;
inline constexpr double BLOCK_CENTROID_DIFF_SCALE = 5.0;

// Silence / degeneracy floors.
inline constexpr double BLOCK_DB_FLOOR       = 1e-6;  // block_assembly.py:219-220 (B8/B9)
inline constexpr double BLOCK_NORM_FLOOR     = 1e-8;  // block_assembly.py:344+352 (B13)

// Block transition marker sentinel (session-22 F2 sub-class reuse).
// MARKER per session-24 audit (B21).
// Source-of-truth: block_assembly.py:500 (2026-04-22).
inline constexpr double BLOCK_TRANSITION_MARKER = 1.0;

// --- Data ----------------------------------------------------------------

// PARITY: BlockInfo dataclass at block_assembly.py:44-58.
struct BlockInfo
{
    int         segment_idx;
    std::string label;
    std::string display_name;  // "Verse 1", "Chorus 2" (disambiguated)
    int         start_beat;
    int         end_beat;      // exclusive
    double      start_sec;
    double      end_sec;
    int         n_beats;
    double      duration_sec;
    int         cluster_id;
};

// PARITY: BlockCompatResult dataclass at block_assembly.py:136-150. Flat
// vectors replace numpy ndarrays: quality/splice_from/splice_to are (n, n)
// primary arrays; top_k_* are (n, n, K) alternative slots.
//
// Indexing conventions:
//   - Primary[i, j] = primary[i * n + j].
//   - TopK[i, j, k]  = top_k[(i * n + j) * K + k].
//
// Zero-initialization on unpopulated slots matches Python `np.zeros` default.
struct BlockCompatResult
{
    int n = 0;   // = number of blocks

    std::vector<double>  quality;       // (n, n) best quality score per pair
    std::vector<int64_t> splice_from;   // (n, n) best exit beat per pair
    std::vector<int64_t> splice_to;     // (n, n) best entry beat per pair

    std::vector<double>  top_k_quality; // (n, n, BLOCK_TOP_K)
    std::vector<int64_t> top_k_from;
    std::vector<int64_t> top_k_to;
};

// Optional inputs for the compatibility scoring.
// Null pointers map to Python's `None` — the corresponding signal drops out
// and its weight is redistributed by `computeQualityScore`.
struct BlockCompatInputs
{
    // REQUIRED --------------------------------------------------------------
    const BlockInfo* blocks;      // (n_blocks,)
    int              n_blocks;

    const double*    beat_times;  // (n_beats,) f64 seconds
    int              n_beats;

    const float*     features;    // (n_beats, n_features) row-major f32
    int              n_features;

    // OPTIONAL WAVEFORM SCORING --------------------------------------------
    const float* boundary_waveforms;    // (n_bnd, n_samples) row-major f32
    int          n_boundary_waveforms;
    int          n_samples_per_bnd;
    int          waveform_sample_rate;

    // OPTIONAL EDGE FEATURES -----------------------------------------------
    const float* edge_features_start;   // (n_beats, n_edge_features) f32
    const float* edge_features_end;
    int          n_edge_features;

    // OPTIONAL EDGE RMS (linear). Both required or both null → edge-dB signal
    // drops out AND graduated energy penalty = 0.
    const double* edge_rms_start;       // (n_beats,) linear rms
    const double* edge_rms_end;

    // OPTIONAL PER-BEAT SCALARS --------------------------------------------
    const double* rms_energy;           // (n_beats,) → energy_match
    const double* spectral_centroid;    // (n_beats,) → centroid_match
    const double* vocal_activity;       // (n_beats,) → vocal penalty

    // FIX-IN-PORT (sesja 71, ADR-059): onset-sustain penalty for splice
    // landing on sustain. Python `block_assembly.py::_score_junction` does
    // NOT call `compute_onset_penalty` (parity gap; sesja-69 listening
    // confirmed audibility). When this pointer is null, BA matches legacy
    // Python behavior (parity tests use null). When non-null, onset penalty
    // is applied — production callers (RemixPipeline Block branch) pass it.
    const double* onset_strength;       // (n_beats,) → onset-sustain penalty

    // ADR-088 sesja 98 — vocal phrase boundary signals (see TransitionCost.h).
    // Per-pair (i→j) quality = 1 - max(release_end[i], onset_start[j]); fed
    // to qi.vocal_continuity. nullptr default preserves bit-exact parity.
    const double* edge_vocal_onset_start;
    const double* edge_vocal_release_end;

    // OPTIONAL DOWNBEATS (seconds) — used for bar-aligned scoring.
    const double* downbeats;            // (n_db,) f64 seconds
    int           n_downbeats;

    // PARAMETERS -----------------------------------------------------------
    int time_signature = 4;

    // ADR-051 (sesja 61) — junction search-window radius (beats either side
    // of the user-authored boundary). Default = BLOCK_SEARCH_WINDOW_BEATS to
    // preserve Python parity for legacy callers. Block Assembly UI overrides
    // from the "Splice flexibility" setting (Tight=4 / Medium=8 / Loose=16).
    int search_window_beats = BLOCK_SEARCH_WINDOW_BEATS;

    // ADR-051 — drift penalty weight applied at junction scoring. Default
    // 0.0 preserves parity for legacy callers (Python golden tool does not
    // emit the drift term yet — § Consequence #7 schedules a regenerate).
    // Block Assembly UI sets BLOCK_DRIFT_PENALTY_WEIGHT explicitly.
    double drift_penalty_weight = 0.0;

    // QualityWeights override (sesja 71, ADR-058 calibration). nullptr →
    // use kDefaultQualityWeights (preserves parity tests). Quality.h is
    // already included via the existing chain.
    const QualityWeights* quality_weights = nullptr;

    // ───────────────────────────────────────────────────────────────────
    // ADR-081 (sesja 96) — β-model candidate-space expansion for Block
    // junctions. Default `block_assembly_beta = false` preserves the legacy
    // ±W candidate-search path verbatim (Python parity + bit-exact baseline).
    // RemixPipeline Block branch enables β-mode post listening-A/B verdict.
    //
    // Sesja 69 captured design (rolled back) + sesja 89 cost-architecture
    // freeze + ADR-081 PROPOSED reactivation roadmap. See
    // `phases/phase-6-ui/log.md:4438-4530+` for sesja-69 audit findings.
    // ───────────────────────────────────────────────────────────────────
    bool block_assembly_beta = false;

    // β-fields below have NO effect when block_assembly_beta == false.

    // Linear fragment-preservation soft cost. quality -=
    //   fragment_penalty_weight × ((1 - kept_i) + (1 - kept_j))
    // where kept_i = (block_i.end - bi) / block_i.length (clamped [0,1]),
    // kept_j = (bj - block_j.start) / block_j.length (clamped [0,1]).
    // Outside-block candidates are clamp-neutral (kept = 1, no penalty).
    // Default 0.03 per sesja 69 user-confirmed value (ADR-058 § Decision).
    double fragment_penalty_weight = 0.03;

    // Block whose n_beats ≤ this threshold bypasses fragment penalty AND
    // min_jump filter (treated as quality-only, no spatial constraints).
    // Default 4 per sesja 69 captured design.
    int short_block_threshold_beats = 4;

    // Top-K spatial diversity: greedy fill ensures selected candidates have
    // ≥ this many beats of separation between (bi, bj) tuples. Preserves
    // highest-quality candidate first; later candidates dropped if too close
    // to already-selected. Default 4 beats per sesja 69 captured design.
    int top_k_min_separation_beats = 4;

    // Extension of search beyond block boundary on each side:
    //   bi ∈ [block_i.start - W_out, min(block_i.end + W_out, n_beats - 1))
    //   bj ∈ [max(block_j.start - W_out, 0), min(block_j.end + W_out, n_beats))
    // Default 8 beats per sesja 69 captured design + Patch 1.
    int outside_window_beats = 8;

    // Minimum beats between bi and bj for non-adjacent block pairs. Filters
    // out adjacent-beat candidates that produce no actual cut (content
    // continues = numerically high quality but no audible splice). Bypassed
    // when block_i.end == block_j.start (legitimately adjacent block pairs).
    // Default 4 per sesja 69 captured design + Patch 2.
    int min_jump_beats = 4;

    // Filter bi ∈ pre_db_set (last beat of bar) AND bj ∈ db_set (downbeat
    // of next bar). When downbeats are unavailable (db_set empty), filter
    // is bypassed (all candidates pass). Default true per sesja 69 captured
    // design + Patch 3.
    bool downbeat_only_splices = true;

    // Lazy compute: when true AND block_sequence is non-null, only compute
    // (block_sequence[k], block_sequence[k+1]) junction pairs instead of
    // full n × n. ~70× speedup measured sesja 69 (17-block × 5-block-queue).
    // Off-sequence pairs land as zero-quality fallback (UI palette badges
    // show neutral state). Default true; turn off for full n×n if UI ever
    // needs interactive compatibility colors on every palette block (not
    // currently a feature).
    bool block_sequence_lazy = true;

    // Sesja-target queue (block indices). Required when block_sequence_lazy
    // == true; ignored otherwise. Sesja 96 RemixPipeline Block branch wires
    // this to in_.userBlocksQueue.
    const int* block_sequence  = nullptr;
    int        n_block_sequence = 0;
};

// --- Public API ----------------------------------------------------------

// Convert raw segments + beat-times into enriched BlockInfo with beat-level
// boundaries + disambiguated display names.
//
// Empty-in → empty-out (Python L89). n_beats < 2 also → empty.
//
// PARITY: block_assembly.py::build_blocks (L83-130).
std::vector<BlockInfo>
buildBlocks(const analysis::Segment* segments,
            int                       n_segments,
            const double*             beat_times,
            int                       n_beats);

// Pre-compute block compatibility matrix. See BlockCompatResult docstring.
//
// PARITY: block_assembly.py::compute_block_compatibility (L153-276).
BlockCompatResult
computeBlockCompatibility(const BlockCompatInputs& inputs);

// Assemble a user-ordered block sequence into a RemixPath.
//
// `block_sequence` is a vector of block indices in user-chosen order (may
// contain repeats). `variation` (0 = best, 1 = 2nd best, ...) applies to
// all junctions unless overridden by `junction_variations` map (per-junction
// rank by junction index, where junction `j` is between block_sequence[j]
// and block_sequence[j+1]).
//
// PARITY: block_assembly.py::assemble_blocks (L415-514).
//
// ADR-084 sesja 93 — `edit_length_jump_scale` parameter (default 1.0)
// MULTIPLIES the per-junction `1.0 - quality` cost (supersedes sesja-92
// ADR-083 additive penalty). Default 1.0 → bit-exact baseline.
// MainComponent maps slider [0..100] via 2^((slider-50)/25) ∈ [0.25, 4.0].
// Slider=50 → 1.0 (bit-exact). Slider=100 → 4× junction cost (DP biases
// toward higher-quality junctions). Slider=0 → 0.25× junction cost
// (algorithm accepts lower-quality junctions more easily).
// ADR-081 (sesja 96) — `allow_outside_window` parameter (default false)
// addresses DEV-043: when false, entry/exit beats are clamped to
// [block.start, block.end - 1] (legacy Python parity preserved). When
// true, outside-block-window candidates from β-model land in final audio
// (clamp loosened to global [0, n_beats - 1] with exit ≥ entry invariant).
// Block β-mode (RemixPipeline.cpp Block branch) sets this true when
// block_assembly_beta is enabled.
RemixPath
assembleBlocks(const std::vector<int>&              block_sequence,
               const std::vector<BlockInfo>&        blocks,
               const double*                        beat_times,
               int                                  n_beats,
               const BlockCompatResult&             compat,
               int                                  variation              = 0,
               const std::map<int, int>*            junction_variations    = nullptr,
               double                               edit_length_jump_scale = 1.0,
               bool                                 allow_outside_window   = false);

// Crossfade duration for a given quality score — UI helper. Useful for the
// dump tool and parity validation.
// PARITY: block_assembly.py::_crossfade_for_quality (L517-523).
double crossfadeForQuality(double quality);

} // namespace reamix::remix
