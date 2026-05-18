#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "remix/Path.h"
#include "remix/TransitionCost.h"  // TransitionCandidate, INF
#include "remix/ViterbiDP.h"       // JUMP_PENALTY_BASE, DURATION_PENALTY_WEIGHT (session-23 promoted)

namespace reamix::remix {

// RegionOptimizer — region-constrained remix (time-selection based).
//
// Port of `references/python-source/remix/region_optimizer.py::RegionRemixMixin`
// (357 LOC, 2026-04-22). Session-23 target per HANDOVER-22.
//
// Python has this as a mixin trait inherited into `CleanOptimizer`. The C++
// port (Q1 rec A — user-approved session 23 kickoff) is a STANDALONE class
// that takes a `RegionOptimizerInputs` bundle carrying the upstream stages'
// outputs directly — no composition with `CleanOptimizer`. Rationale:
//   - Consistent with sessions 18/19/20/22 isolation pattern.
//   - C++ benefits from explicit composition over Python-idiomatic mixin.
//   - Session-22 CleanOptimizer already exposes its __init__ state via
//     diagnostic accessors, but that was for parity-test bisection; widening
//     the public surface further for RegionOptimizer would couple modules.
//
// Main entry: `remix(target_duration, region_start_sec, region_end_sec,
// region_W, region_candidates)`. The `region_W` + `region_candidates`
// come from a prior `computeRegionCosts` call (session-23 RegionCost
// module). Per Q2 rec B (user-approved), the Python `region_W=None`
// fallback branch (`_build_region_cost_matrix` extracting from global W
// at region_optimizer.py:160-178) is NOT ported — production caller
// (`server/handlers/_remix.py:114`) always passes pre-computed region_W.
// Recorded in `weights-audit.md` as ORPHAN-FALLBACK sub-class.
//
// PARITY notes:
//
// --- Citation discipline (ADR-026) ----------------------------------------
// 15 main-path constants classified in `weights-audit.md` session-23:
//   CLEAN (2): R14 j!=i+1 consecutive-beat (sesja-22 reuse), R15 sample_rate.
//   DEFENSIVE (6): R1 exit_beat floor, R2 min region/target/tolerance 2 guards
//                 (multiple occurrences), R11/R12 dist/target_beats denom floors,
//                 R13 fallback repeats guard.
//   UNJUSTIFIED (5): R3/R5 backward-jump penalty 0.5 (R5 BENIGN-DUPLICATION of R3),
//                    R4 region-fallback micro-skip 4, R6 neighbor top-k 8,
//                    R7 REGION_COOLDOWN 8, R9 jump-cost cap+scale, R10 terminal scale.
//   BENIGN-DUPLICATION (1): R5 — second `0.5` literal at region_optimizer.py:176
//                              duplicating L158; port verbatim, do NOT DRY.
//                              Same sub-class as session-22 D9.
//   MARKER (1): R8 `999` sentinel "allow initial jump" — session-22 F2 sub-class.
//
// --- Imports reused from ViterbiDP.h (session-23 promotion) ---------------
// `JUMP_PENALTY_BASE=0.8` and `DURATION_PENALTY_WEIGHT=3.0` are consumed by
// `_region_dp` at region_optimizer.py:260+281. Python imports from
// `viterbi_dp.py`; C++ reuses `reamix::remix::JUMP_PENALTY_BASE` +
// `DURATION_PENALTY_WEIGHT` promoted to public constexpr this session per
// Hard Rule #1 (2 TU consumers: ViterbiDP.cpp + RegionOptimizer.cpp).
//
// --- `is_extending` branch ------------------------------------------------
// When `target_duration > region_duration`, backward jumps get a +0.5
// penalty (region_optimizer.py:157-158 — applied to `rW[ri, rj]` entries).
// R3 (applied branch) and R5 (fallback branch — skipped per Q2). Session-23
// corpus includes 16 × r1.2 cases (extending mode) to exercise this.
//
// --- REGION_COOLDOWN = 8 vs COOLDOWN_BARS × time_signature ----------------
// Region DP uses a HARDCODED cooldown of 8 (R7) at
// region_optimizer.py:226. Distinct from global `COOLDOWN_BARS × TS = 16`
// for 4/4 — regions operate on shorter scales where 8-beat cooldown
// allows more viable paths. UNJUSTIFIED per audit — no empirical citation
// for the specific value 8.
//
// --- FMA contraction (ADR-028, 5th reuse) ---------------------------------
// RegionOptimizer composes penalty sums through W + jump + terminal +
// duration penalties. Same FMA vulnerability. `-ffp-contract=off` on
// `test_region_optimizer` target.

// ---------------------------------------------------------------------------
// Inputs bundle
// ---------------------------------------------------------------------------

struct RegionOptimizerInputs
{
    // Track-level state (from CleanOptimizer __init__ — caller provides).
    // All pointers caller-owned; must outlive RegionOptimizer.
    int           n_beats;
    const double* beat_times;              // (n_beats,) f64 seconds
    double        avg_beat_duration;       // CleanOptimizer._avg_beat_duration
    double        duration_tolerance_sec;  // CleanOptimizer._duration_tolerance_sec

    // Global candidates — merged with region_candidates at
    // `_region_transitions` (region_optimizer.py:328-330). Null → region-only.
    const std::map<std::pair<int, int>, TransitionCandidate>* candidates;

    // PARAMETERS.
    int sample_rate = 22050;  // R15 CLEAN — for alignment_offset_sec metadata

    // DEV-033 (sesja 63b, ADR-054) — soft-boundary entry/exit beat selection.
    // When splice_flex_beats > 0 AND downbeats != nullptr, the algorithm
    // searches a ±splice_flex_beats window around the user's region.startSec
    // / region.endSec for the closest DOWNBEAT (phrase boundary) and uses
    // that as entry_beat / exit_beat. Fallback to first-beat-at-or-after
    // (entry) / last-beat-at-or-before (exit) when no downbeat is in window.
    // Defaults (0 / nullptr) preserve legacy argminAbsDiff (closest beat)
    // behavior — used by the parity test against the Python port and
    // any caller that doesn't want the new soft-boundary semantics.
    const double* downbeats        = nullptr;  // (n_downbeats,) f64 seconds
    int           n_downbeats      = 0;
    int           splice_flex_beats = 0;       // 0 = legacy closest-beat snap

    // ADR-084 Edit Length slider — MULTIPLICATIVE jump-cost scale (sesja 93,
    // supersedes sesja-92 ADR-083 additive design). Default 1.0 → bit-exact
    // baseline. MainComponent maps slider [0..100] via 2^((slider-50)/25)
    // ∈ [0.25, 4.0]. Applied at jump-branch in regionDp() — multiplies the
    // existing JUMP_PENALTY_BASE × min(REGION_JUMP_COST_CAP, raw_w × scale)
    // term. Region path mirrors full-track ViterbiDP analog.
    double        edit_length_jump_scale = 1.0;

    // ADR-083 sesja 92 — UI Min cut slider override for region cooldown.
    // Default 0 → legacy REGION_COOLDOWN = 8 → bit-exact baseline. UI
    // passes user slider value 4-32 beats. Note Region uses different
    // baseline cooldown (8) than full-track (16) — baseline applies
    // uniformly to both modes; user-perceptual semantic is "minimum
    // segment length".
    int           min_seq_after_jump_override = 0;

    // ADR-081 STATUS UPDATE 1 sesja 94 — Region β-model "inner-loop
    // synthesizer". Default false → bit-exact baseline → parity test
    // 48/48 PASS preserved. When true (production flip in
    // RemixPipeline.cpp Region branch):
    //   - REGION_BACKWARD_JUMP_PENALTY_EXTEND reduced 0.5 → 0.05
    //     (in extending mode; cooldown still bounds runaway loops).
    //   - REGION_JUMP_COST_CAP raised 1.0 → 5.0 (preserves quality
    //     discrimination across raw_w range).
    //   - JUMP_PENALTY_BASE effective scale reduced 0.8 → 0.3 (multi-
    //     iteration paths competitive with single-jump alternatives).
    //   - Diagnostic log appended to ~/Desktop/reamix_diag.log per
    //     remix() call (sesja-94-only debug aid; removed at handover).
    //
    // Empirical evidence from existing test_region_optimizer corpus:
    // 16 tracks × {0.5, 0.8, 1.2} = 48 cases produce exactly
    // n_transitions=1 each, regardless of region scale. This matches
    // user-perceived "loop the whole region 1-2×" structural failure
    // mode (sesja 64 + sesja 93). β path addresses cost-function
    // bottleneck without expanding candidate-space (per ADR-081 §B
    // STATUS UPDATE 1 pre-code audit).
    bool          region_beta = false;

    // ADR-081 STATUS UPDATE 2 sesja 94 — caller-provided entry/exit beat
    // override. RegionOptimizer locally derives entry_beat / exit_beat
    // via argminAbsDiff (closest-beat snap) which can DIVERGE from
    // RemixPipeline's lower_bound/upper_bound convention by 1 beat for
    // user regions whose boundaries don't align with beat times.
    //
    // When that divergence happens, region_W (built by RegionCost using
    // RemixPipeline's entry/exit) has dimension N_caller × N_caller but
    // RegionOptimizer reads it with stride N_local where N_local !=
    // N_caller → SCRAMBLED rW values across the matrix. This bug
    // pre-dates sesja 94 (latent since session 23 RegionOptimizer port);
    // surfaced sesja 94 via loop synthesizer diagnostic showing total_cost
    // 5-10× higher than expected on user smoke regions.
    //
    // Fix: caller (RemixPipeline) passes the SAME entry/exit beats it
    // used to build region_W. When ≥ 0, RegionOptimizer skips local
    // argminAbsDiff and uses these directly. Default -1 = "no caller
    // override" → preserves parity test legacy behavior.
    int           entry_beat_override = -1;
    int           exit_beat_override  = -1;
};

// ---------------------------------------------------------------------------
// RegionOptimizer — stateless with respect to calls (inputs bundle immutable).
// ---------------------------------------------------------------------------
class RegionOptimizer
{
public:
    explicit RegionOptimizer(const RegionOptimizerInputs& inputs);

    // Main entry. Port of `remix_region` (region_optimizer.py:35-137).
    //
    // `region_W` + `region_candidates` MUST be non-null per Q2 (rec B,
    // user-approved). Session-23 omits the `region_W=None` fallback branch
    // that extracts from the global W matrix — production handler always
    // passes pre-computed inputs.
    //
    // `blocked_transitions` (sesja 100, DEV-032): set of (fromBeat, toBeat)
    // pairs in **global beat indices** that must NOT appear as jumps in the
    // resulting path. Used by remix_k_best for path diversification + by
    // ContextMenu "Avoid this transition" persisted blocks. Default nullptr
    // = no transitions forbidden (parity-preserving with pre-DEV-032 calls).
    RemixPath remix(double                                                        target_duration,
                    double                                                        region_start_sec,
                    double                                                        region_end_sec,
                    const double*                                                 region_W,
                    int                                                           n_region,  // = exit_beat - entry_beat
                    const std::map<std::pair<int, int>, TransitionCandidate>*     region_candidates,
                    const std::set<std::pair<int, int>>*                          blocked_transitions = nullptr);

    // Sesja 100 (DEV-032, ADR-091 — Region mirror of ADR-048 Duration K-best).
    //
    // Returns up to `k` distinct paths for the same region target, each
    // distinguished by forbidding one transition from the previous best
    // path's jump list. Same algorithm as CleanOptimizer::remix_k_best:
    //   1. best = remix(...) with caller's blocked_transitions
    //   2. for each (fb, tb) in best.transitions:
    //        augmented = blocked ∪ {(fb, tb)}
    //        alt = remix(..., augmented); dedup by beat_indices
    //   3. return paths[:k]
    //
    // For Region, both DP paths (regionDp + regionLoopSynthesize) honour
    // blocked_transitions. The β-path emits N-loop variants where the same
    // (i, j) pair repeats N times; blocking a pair removes the entire family.
    //
    // `k <= 1` short-circuits to `[best]`. Returns at least 1 path for any
    // non-degenerate region (caller may still receive an empty path on
    // n_beats < REGION_MIN_VIABLE_DP — same semantic as remix()).
    std::vector<RemixPath>
    remix_k_best(double                                                        target_duration,
                 double                                                        region_start_sec,
                 double                                                        region_end_sec,
                 const double*                                                 region_W,
                 int                                                           n_region,
                 const std::map<std::pair<int, int>, TransitionCandidate>*     region_candidates,
                 int                                                           k,
                 const std::set<std::pair<int, int>>*                          blocked_transitions = nullptr);

    // Convenience replicating `_remix.py:140-147` for Region mode. Mirrors
    // CleanOptimizer::remix_variation:
    //   k        = max(2, variation_idx + 1)
    //   path_idx = min(variation_idx, len(paths) - 1)
    // Used by RemixPipeline when `Input::variation > 0` in Region mode.
    RemixPath
    remix_variation(double                                                        target_duration,
                    double                                                        region_start_sec,
                    double                                                        region_end_sec,
                    const double*                                                 region_W,
                    int                                                           n_region,
                    const std::map<std::pair<int, int>, TransitionCandidate>*     region_candidates,
                    int                                                           variation_idx,
                    const std::set<std::pair<int, int>>*                          blocked_transitions = nullptr);

    // Diagnostic accessors for parity-test bisection. Populated after each
    // `remix()` call (so caller can dump intermediate state).
    int entryBeat()      const noexcept { return entry_beat_; }
    int exitBeat()       const noexcept { return exit_beat_; }
    int targetBeats()    const noexcept { return target_beats_; }
    int minTarget()      const noexcept { return min_target_; }
    int maxTarget()      const noexcept { return max_target_; }
    bool isExtending()   const noexcept { return is_extending_; }
    double regionDuration() const noexcept { return region_duration_; }

    const std::vector<std::int64_t>& neighborIndices() const noexcept { return nb_indices_; }
    const std::vector<std::int64_t>& neighborOffsets() const noexcept { return nb_offsets_; }
    const std::vector<double>&       regionCostMatrix() const noexcept { return rW_; }

private:
    // region_optimizer.py:143-179 — fallback branch (L160-178) NOT ported
    // per Q2 rec B. Input region_W is copied (extending mode adds +0.5 to
    // backward cells) then returned.
    void buildRegionCostMatrix(int            n_region,
                               const double*  region_W,
                               bool           is_extending);

    // region_optimizer.py:181-213. CSR neighbor list builder.
    void buildRegionNeighbors(int n_region);

    // region_optimizer.py:215-288. Returns (best_cost, best_t, best_ri).
    struct DpResult
    {
        double best_cost;
        int    best_t;
        int    best_ri;
        std::vector<std::int64_t> parent;  // flat (max_target+1) × n_region
    };
    DpResult regionDp(int n_region,
                      int min_target,
                      int max_target,
                      int target_beats) const;

    // ADR-081 STATUS UPDATE 2 sesja 94 — explicit loop synthesizer for
    // extending mode. Replaces free-form DP with structural (i, j, N) path
    // enumeration:
    //   path = [0..i sequential] + N × [jump i→j + (j+1)..i sequential]
    //                            + [i+1..n_region-1 sequential]
    //   total t = n_region + N × (i-j+1) (must land in [min_target, max_target])
    //
    // Each iteration's "loop length" L = i - j (must be ≥ effective_cooldown
    // to satisfy DP cooldown constraint when treated as standard path).
    //
    // Empirical motivation (sesja 94 diagnostic): standard regionDp produces
    // n_transitions=1 in 47/48 corpus cases despite 800-1700+ viable backward
    // candidates per case. DP correctly minimizes total body cost, and any
    // non-zero per-jump tax structurally biases toward fewer jumps. User
    // intent ("find ideal 8-sec inner loop and iterate") requires explicit
    // multi-iteration structure, not cost-function rebalancing alone.
    //
    // Returns empty `path` vector if no viable (i, j, N) found within
    // [min_target, max_target]; caller falls back to regionDp.
    struct LoopSynthResult
    {
        std::vector<std::int64_t> path;      // empty = no valid synth, use DP
        std::vector<std::pair<int, int>> transitions;  // absolute beat indices
        double total_cost = 0.0;
        int    chosen_i  = -1;  // region-relative
        int    chosen_j  = -1;  // region-relative
        int    chosen_N  = 0;
    };
    LoopSynthResult regionLoopSynthesize(int    n_region,
                                          int    min_target,
                                          int    max_target,
                                          int    target_beats,
                                          int    effective_cooldown,
                                          int    entry_beat) const;

    // region_optimizer.py:290-305. Repeated region fallback when DP fails.
    RemixPath regionFallback(int    entry_beat,
                             int    exit_beat,
                             int    target_beats,
                             double target_duration,
                             double region_duration) const;

    // region_optimizer.py:307-320. DP backtrace → absolute beat indices.
    std::vector<std::int64_t> regionBacktrace(int                             best_t,
                                              int                             best_ri,
                                              const std::vector<std::int64_t>& parent,
                                              int                             n_region,
                                              int                             max_target,
                                              int                             entry_beat) const;

    // region_optimizer.py:322-357. Extract transitions + metadata.
    void regionTransitions(const std::vector<std::int64_t>&                          path,
                           const std::map<std::pair<int, int>, TransitionCandidate>* region_candidates,
                           std::vector<std::pair<int, int>>&                         out_transitions,
                           std::map<std::pair<int, int>, std::map<std::string, double>>& out_metadata) const;

    // -- State from inputs ----------------------------------------------------
    int                                                         n_beats_;
    const double*                                               beat_times_;
    double                                                      avg_beat_duration_;
    double                                                      duration_tolerance_sec_;
    const std::map<std::pair<int, int>, TransitionCandidate>*   candidates_;
    int                                                         sample_rate_;
    const double*                                               downbeats_;
    int                                                         n_downbeats_;
    int                                                         splice_flex_beats_;
    double                                                      edit_length_jump_scale_ = 1.0;  // ADR-084 sesja 93
    int                                                         min_seq_after_jump_override_ = 0;  // ADR-083
    bool                                                        region_beta_ = false;  // ADR-081 STATUS UPDATE 1 sesja 94
    int                                                         entry_beat_override_ = -1;  // ADR-081 STATUS UPDATE 2 sesja 94
    int                                                         exit_beat_override_  = -1;

    // -- Per-call state (cleared on each `remix()` entry) ---------------------
    int    entry_beat_       = 0;
    int    exit_beat_        = 0;
    int    target_beats_     = 0;
    int    min_target_       = 0;
    int    max_target_       = 0;
    bool   is_extending_     = false;
    double region_duration_  = 0.0;

    // Sesja 100 (DEV-032) — blocked transitions in **global beat indices**.
    // Set at remix() entry, consumed by regionDp neighbor traversal +
    // regionLoopSynthesize (i, j) iteration to skip forbidden jumps. Reset
    // to nullptr at remix() exit. nullptr = no transitions forbidden
    // (parity-preserving with pre-DEV-032 callers).
    const std::set<std::pair<int, int>>* blocked_transitions_ = nullptr;

    std::vector<double>       rW_;           // (n_region, n_region) row-major
    std::vector<std::int64_t> nb_indices_;
    std::vector<std::int64_t> nb_offsets_;   // size n_region + 1
};

} // namespace reamix::remix
