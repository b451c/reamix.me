#pragma once

#include <cstdint>
#include <set>
#include <vector>

#include "analysis/RepetitionMap.h"

namespace reamix::remix {

// Viterbi DP kernel + sparse neighbor list builder + downbeat arrays.
//
// Port of `references/python-source/remix/viterbi_dp.py` (507 LOC,
// 2026-04-21). Session-20 target per HANDOVER-19. Three free functions
// mirror the Python names one-to-one (`_build_neighbors`, `_viterbi_dp`,
// `_compute_downbeat_arrays`); the fourth Python helper
// `_compute_segment_data` already landed as a shared module at
// `src/remix/SegmentData.{h,cpp}` (extracted session 19 from
// TransitionCost.cpp's anonymous namespace).
//
// Consumers: `Optimizer::CleanOptimizer` facade (session 21+), which
// orchestrates TransitionCost → SegmentData → downbeat arrays → neighbors
// → Viterbi DP.
//
// PARITY notes ----------------------------------------------------------
//
// --- Constants provenance (ADR-026 + addendum) -------------------------
// All 12+ hand-calibrated numbers from `viterbi_dp.py` (L24-35 top-level
// + L178 function-scope LINF_PENALTY_WEIGHT + 9 inline magic literals in
// the DP kernel) are ported VERBATIM. Session-20 pre-port audit classified
// each in `phases/phase-4-remix-engine/weights-audit.md`:
//   - CLEAN (2): INF (mathematical sentinel), COOLDOWN_BARS (meter-aware).
//   - UNJUSTIFIED (most): BACKWARD, JUMP_BASE, BOUNDARY, SECTION, SPAN,
//     DURATION, LINF_PENALTY + inline literals.
//   - DECLARATION-DRIFT (DEAD): DURATION_TOLERANCE is declared + imported
//     by optimizer but NEVER referenced at runtime. Per audit, NOT ported.
//   - FALLBACK-DEFAULT: MIN_SEQ_AFTER_JUMP / MIN_FORWARD_JUMP_BEATS are
//     default args always overridden by optimizer.py L109-110 to
//     `COOLDOWN_BARS × time_signature`. Exposed as *_DEFAULT below.
// Kept as constexpr in the .cpp anonymous namespace — none of the
// calibration constants are part of the public API. The DEFAULT params
// exposed here are caller-facing function arguments only.
//
// --- FMA contraction (ADR-028) ----------------------------------------
// ViterbiDP composes `dp[t,i] + W[i,j]` accumulations + weighted penalty
// sums. Same FMA vulnerability as Quality / TransitionCost: `-ffp-contract=off`
// is required on `test_viterbi_dp` target. Third reuse of ADR-028.
//
// --- Why INF stays out of this header ---------------------------------
// `TransitionCost.h` already defines `inline constexpr double INF = 1e9;`
// in the same `reamix::remix` namespace. A `CleanOptimizer.cpp` that
// includes both headers simultaneously (session 21 target) would trigger
// a redefinition error. Python has the same duplicate: `transition_cost.py:44`
// AND `viterbi_dp.py:24` each declare `INF = 1e9` locally. In C++ we
// move the ViterbiDP-side INF into the .cpp anonymous namespace so the
// public headers do not collide. All public-API types use the f64 literal
// implicitly; no caller needs to reference the symbol.
//
// --- Iteration-order determinism --------------------------------------
// `_build_neighbors` uses `np.argpartition(w_row, k)[:k]` (unsorted top-K
// by cost) — same semantics as `std::nth_element` (session-19 validated
// at 13 254-pair corpus scale on TransitionCost). Sub-neighbor order is
// then sorted (`sorted(neighbors)` at Python L123); C++ matches via
// `std::set<int>` plus copy to vector.
//
// Sparse indices + offsets are int64 to match Python `np.int64` dtype.

// ---------------------------------------------------------------------------
// Defaults (see Python signatures)
// ---------------------------------------------------------------------------
//
// MIN_SEQ_AFTER_JUMP_DEFAULT / MIN_FORWARD_JUMP_DEFAULT: Python L27-28.
// FALLBACK-DEFAULT class: always overridden at production call sites by
// `optimizer.py:109-110` → `COOLDOWN_BARS × time_signature` (4/4 → 16,
// 3/4 → 12, 6/8 → 24). Exposed here so tests can invoke the low-level
// kernel matching the Python default-arg contract.
// Source-of-truth: `viterbi_dp.py:27 + :28 (2026-04-21)`.
inline constexpr int kMinSeqAfterJumpDefault   = 16;
inline constexpr int kMinForwardJumpDefault    = 16;

// MAX_NEIGHBORS_DEFAULT / MAX_TRANSITIONS_DEFAULT: Python L48 + L158.
// UNJUSTIFIED class per weights-audit — no citation, no inline rationale.
// Source-of-truth: `viterbi_dp.py:48 (_build_neighbors)` +
//                  `viterbi_dp.py:158 (_viterbi_dp)`.
inline constexpr int kMaxNeighborsDefault      = 32;
inline constexpr int kMaxTransitionsDefault    = 6;

// DEFAULT_CONFIG.remix.min_segment_beats: CLEAN class (config-level
// 4-bar minimum-loop-length, meter-agnostic constraint). Used at
// `viterbi_dp.py:237` as backward-jump cutoff. Exposed as part of
// `ViterbiDPInputs` so tests can vary it independently of `time_signature`.
// Source-of-truth: `config.py:105 (2026-04-21)`.
inline constexpr int kMinSegmentBeatsDefault   = 16;

// COOLDOWN_BARS: CLEAN class — genre-independent 4-bar phrase protection
// unit. Multiplied by `time_signature` at caller sites to yield per-meter
// cooldown in beats (4/4 → 16, 3/4 → 12, 6/8 → 24). Python declares this
// at `viterbi_dp.py:26` and imports it at `optimizer.py:37`. Exposed in
// this header (session 22) so `src/remix/Optimizer.cpp` can compute the
// same phrase-protection cooldowns without duplicating the literal —
// single-source-of-truth discipline per CLAUDE.md Hard Rule #1.
// Session-20 originally placed it in `ViterbiDP.cpp` anonymous namespace;
// session-22 promotes it to public constexpr because a second TU consumer
// (Optimizer.cpp) has landed. Value unchanged, zero semantic drift.
// Source-of-truth: `viterbi_dp.py:26 (2026-04-21)`.
inline constexpr int COOLDOWN_BARS             = 4;

// JUMP_PENALTY_BASE: UNJUSTIFIED (HIGH) per weights-audit session-20.
// Base coefficient for jump-cost scaling in both the full-track DP
// (`viterbi_dp.py:304`) and the region DP (`region_optimizer.py:260`).
// Session-20 originally placed it in `ViterbiDP.cpp` anonymous namespace
// (module-private — only `_viterbi_dp` was a consumer). Session-23
// promotes to public constexpr because `RegionOptimizer.cpp` is the
// second TU consumer — same Hard Rule #1 single-source-of-truth trigger
// as COOLDOWN_BARS in session-22. Value unchanged, zero semantic drift.
// Source-of-truth: `viterbi_dp.py:30 (2026-04-21)`.
inline constexpr double JUMP_PENALTY_BASE      = 0.8;

// DURATION_PENALTY_WEIGHT: UNJUSTIFIED (MEDIUM) per weights-audit
// session-20. Scales `|t - target_beats| / max(1, target_beats)` duration
// deviation in both the full-track DP (`viterbi_dp.py:624`) and the region
// DP (`region_optimizer.py:281`). Distinct from
// `config.duration_penalty_weight=10.0` which is never read by remix/* —
// the 3.0 here is the authoritative DP-inner-loop value. Session-23
// promotion rationale identical to JUMP_PENALTY_BASE above (2nd TU
// consumer lands in RegionOptimizer.cpp).
// Source-of-truth: `viterbi_dp.py:34 (2026-04-21)`.
inline constexpr double DURATION_PENALTY_WEIGHT = 3.0;

// ---------------------------------------------------------------------------
// Downbeat arrays
// ---------------------------------------------------------------------------
//
// Port of `_compute_downbeat_arrays` (viterbi_dp.py:468-507).
//
// Python returns `(pre_downbeat_arr, downbeat_arr, downbeat_indices)` or
// `(None, None, None)` when `downbeat_constraint=False` or no downbeats.
// C++ encodes this via the `valid` flag; when false, all three fields are
// empty/default. Production callers always pass `downbeat_constraint=True`
// per `config.py:108`; the flag is exposed to mirror the Python API
// contract end-to-end.
//
// Fallback behavior: when `downbeats == nullptr` or `n_downbeats == 0` AND
// `downbeat_constraint == true`, Python synthesizes via
// `set(range(0, n_beats, time_signature))` (L491). Mirrors in C++.
struct DownbeatArrays
{
    std::vector<std::int8_t> pre_downbeat_arr;  // (n_beats,) 0/1
    std::vector<std::int8_t> downbeat_arr;      // (n_beats,) 0/1
    std::set<int>            downbeat_indices;
    std::set<int>            pre_downbeat_indices;
    bool                     valid = false;     // false == Python (None, None, None)
};

// `beat_times` unused by the Python implementation when `downbeats` is
// None (beat-index fallback is pure arithmetic on n_beats + time_signature).
// When `downbeats != nullptr && n_downbeats > 0`, each downbeat time is
// mapped to the nearest beat index via `argmin(|beat_times - db_time|)`.
DownbeatArrays computeDownbeatArrays(const double* beat_times,
                                     int           n_beats,
                                     const double* downbeats,
                                     int           n_downbeats,
                                     int           time_signature,
                                     bool          downbeat_constraint = true);

// ---------------------------------------------------------------------------
// Neighbor CSR
// ---------------------------------------------------------------------------
//
// Port of `_build_neighbors` (viterbi_dp.py:41-138).
//
// CSR-style sparse representation matching Python's flat (indices, offsets)
// return. `indices` flat int64 of length `offsets[n_beats]`; each beat's
// neighbor list lives in `indices[offsets[i] : offsets[i+1]]`.
//
// Iteration order per beat: `sorted(neighbors)` at Python L123. C++ uses
// `std::set<int>` for O(log n) deduplication + sorted iteration, then
// copies into the flat `indices` array.
//
// `downbeats == nullptr` (DownbeatArrays::valid == false) disables the
// bar-alignment filter (jumps TO any non-self beat allowed; pre-downbeat
// can_jump predicate skipped — every beat is a potential jump origin).
// `jumps == nullptr / n_jumps == 0` disables repetition-map candidates
// (sequential + W top-K + segment-boundary candidates still apply).
struct NeighborCSR
{
    std::vector<std::int64_t> indices;
    std::vector<std::int64_t> offsets;  // size = n_beats + 1
};

NeighborCSR buildNeighbors(int                                n_beats,
                           const double*                      W,                 // (n_beats, n_beats) f64 row-major
                           const analysis::RepetitionJump*    jumps,             // nullable
                           int                                n_jumps,
                           const DownbeatArrays*              downbeat_arrays,   // nullable
                           const std::set<int>&               segment_boundaries,
                           int                                max_neighbors    = kMaxNeighborsDefault,
                           int                                min_forward_jump = kMinForwardJumpDefault);

// ---------------------------------------------------------------------------
// Viterbi DP
// ---------------------------------------------------------------------------
//
// Port of `_viterbi_dp` (viterbi_dp.py:144-381).
//
// DP tables kept as `std::vector<double>` row-major — `dp[t * n_beats + i]`.
// Same layout as Python numpy defaults. Column-major would break parity
// because accumulator order differs (numpy iterates row-major regardless
// of strides).
//
// Inputs bundle mirrors the Python keyword-argument signature, with
// additions for the `segment_boundaries` (moved from Python's implicit
// recomputation at L191-195 to explicit caller-provided set) and
// `min_segment_beats` (exposed instead of reading `DEFAULT_CONFIG.remix.*`
// at runtime — testable + parameterizable).
//
// `beat_to_segment` / `seg_sim_matrix` / `n_segs` — caller computes via
// `remix::SegmentData::computeSegmentData` and passes pointers.
//
// `pre_downbeat_arr` / `downbeat_arr` — from `DownbeatArrays`. Pass null
// pointers when `DownbeatArrays::valid == false` (mirrors Python
// `pre_downbeat_arr is not None and downbeat_arr is not None` check at
// L175).
struct ViterbiDPInputs
{
    // REQUIRED
    const double*       W;                        // (n_beats, n_beats) row-major f64
    int                 n_beats;
    int                 target_length;
    int                 min_target_length;
    int                 intro_beats;
    int                 outro_beats;
    bool                is_shortening;

    const std::int64_t* neighbor_indices;
    int                 n_neighbor_indices;       // = offsets[n_beats]
    const std::int64_t* neighbor_offsets;         // size n_beats + 1

    const std::int64_t* beat_to_segment;          // (n_beats,)
    const double*       seg_sim_matrix;           // (n_segs, n_segs) row-major f64, or nullptr when n_segs <= 1
    int                 n_segs;

    // NULLABLE — pair goes together. Null = no bar info (Python None).
    const std::int8_t*  pre_downbeat_arr;
    const std::int8_t*  downbeat_arr;

    // DP parameters (Python default args).
    int                 max_transitions    = kMaxTransitionsDefault;
    int                 min_jumps          = 0;
    int                 min_seq_after_jump = kMinSeqAfterJumpDefault;
    int                 min_forward_jump   = kMinForwardJumpDefault;
    int                 min_segment_beats  = kMinSegmentBeatsDefault;  // DEFAULT_CONFIG.remix.min_segment_beats

    // ---- ADR-083 Edit Length slider — per-edge global penalty (sesja 92) -
    // Per-jump-edge constant cost added to every non-sequential transition
    // (j != i + 1) BEFORE the existing JUMP_PENALTY_BASE × xcorr² composition.
    // Default 0.0 → bit-exact baseline preservation (current production
    // behavior unchanged when slider untouched).
    //
    // User-controlled via AuditionBar Edit Length slider (ADR-083 host
    // component). Slider range 0-100 snap 1, default 50 center-zero map:
    //   penalty = ((slider − 50) / 50) × kEditLengthPenaltyMax
    //   penalty ∈ [-1.0, +1.0]  with kEditLengthPenaltyMax = 1.0
    // Slider=50 → penalty=0.0 → bit-exact baseline.
    // ADR-084 sesja 93 SUPERSEDES sesja-92 additive penalty design:
    // Edit Length is now a MULTIPLICATIVE jump-cost scale that multiplies
    // the existing `JUMP_PENALTY_BASE × jump_penalty_scale × quality_factor`
    // per-jump term. Default 1.0 → bit-exact baseline (no scaling).
    // Slider=50 → 1.0 (center, baseline). Slider=100 → 4.0 (4× per-jump
    // cost → DP picks fewer cuts). Slider=0 → 0.25 (0.25× cost → DP picks
    // more cuts). Mapping at MainComponent: 2^((slider-50)/25) ∈ [0.25, 4.0].
    //
    // Why multiplicative not additive: research-scout sesja 93 verdict —
    // additive ±1.0 was too small relative to the existing scaled jump
    // penalty (max ~2.4 per jump on typical target_ratio). Multiplicative
    // 16× span actually shifts DP rank-order. Bit-exact baseline at
    // slider=50 (scale=1.0 → identity multiplication) preserved.
    //
    // DP-friendly per Bertsekas: cost(i,j) = function of (i,j) only,
    // optimal substructure preserved. Backtrace + Bellman unchanged.
    //
    // CANONICAL DEFINITION (no Python reference): per ADR-084 (sesja 93;
    // supersedes sesja-92 ADR-083 § Implementation step 4-6 additive design).
    // Self-validated by `tests/parity/test_edit_length.cpp`.
    double              edit_length_jump_scale = 1.0;
};

struct ViterbiPath
{
    std::vector<std::int64_t> path;         // beat indices; empty on "no valid endpoint" Python return
    double                    total_cost;   // INF (= 1e9) when path empty
};

ViterbiPath viterbiDP(const ViterbiDPInputs& inputs);

} // namespace reamix::remix
