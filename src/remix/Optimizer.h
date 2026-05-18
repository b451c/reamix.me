#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "analysis/RepetitionMap.h"
#include "analysis/StructureResult.h"
#include "remix/Path.h"
#include "remix/SegmentData.h"
#include "remix/TransitionCost.h"
#include "remix/ViterbiDP.h"

namespace reamix::remix {

// CleanOptimizer facade — orchestrates TransitionCost + SegmentData +
// DownbeatArrays + buildNeighbors + viterbiDP into the end-to-end RemixPath.
//
// Port of `references/python-source/remix/optimizer.py` (479 LOC,
// 2026-04-22). Session-22 target per HANDOVER-21.
//
// PORTED entries:
//   - `remix()` (main path, Python `remix()`) — session 22.
//   - `remix_k_best()` (K-best diversity for Duration mode "Try different
//     splice"; Python `optimizer.py:389-458`) — session 58 / ADR-048.
//   - `remix_variation()` (convenience replicating server `_remix.py:140-147`
//     `k=max(2,v+1)` + `idx=min(v,len-1)` semantic) — session 58 / ADR-048.
// Deferred:
//   - `remix_region` + mixin methods (RegionOptimizer, session 23 — separate
//     class without K-best; see `RegionOptimizer.h`).
//   - `remix_from_matrix` (legacy CLI wrapper, not ported).
//
// Consumers: phase-5 renderer (RemixPath → audio output). The 48-case corpus
// (16 tracks × {0.3, 0.5, 0.7}) layered on top of the session-21 ViterbiDP
// goldens closes the phase-4 spec L36-42 acceptance grid end-to-end.
//
// PARITY notes (read CAREFULLY before changing constants / literals):
//
// --- Citation discipline (ADR-026) ----------------------------------------
// Every numerical literal in Optimizer.cpp carries a
// `optimizer.py:LINE (2026-04-22)` comment. 30 new constants / literals
// classified in `phases/phase-4-remix-engine/weights-audit.md` session 22:
//   CLEAN (4):  A2 preserve_intro_beats=8, A3 preserve_outro_beats=4,
//               A5 duration_tolerance_sec=5.0, F3 sample_rate=22050.
//   FALLBACK-DEFAULT (3): A1 time_signature=4 (track-derived), A4
//               max_transitions=6 (UI-override default 12), A6
//               waveform_sample_rate=0.
//   DEFENSIVE (7): B1 max(1,time_signature), B3 cumulative fallback, D1
//               target_beats floor 2, D4 tolerance_beats floor 2, E3
//               cooldown floor, E4 empty-path fallback, F1 alignment_offset.
//   UNJUSTIFIED (14): B2 avg_beat_duration=0.5, C2 max_outro=3×TS, D2
//               adaptive_tolerance_sec_floor=2.0, D3 target_ratio_clip
//               [0.4,1.0], D5 max_beats cap ×3, D6 intro/outro scaling
//               threshold 0.8, D7 intro_scale floor 0.25 + min 4, D8
//               outro_scale floor 0.3 + min 2, E1 min_jumps threshold 0.45,
//               E2 cooldown coefficient ×2, E5 post-DP extension cap 0.2+4.
//   BENIGN-DUPLICATION (1): D9 two identical target_ratio computations at
//               optimizer.py:206 + :222. Port verbatim, do NOT DRY — session
//               -21 dump tool `compute_dp_params` validated bit-exact with
//               the duplicate.
//   MARKER (1): F2 is_repetition_jump=1.0 sentinel.
// No DEAD/ORPHAN-IMPORT found in optimizer.py main-path (vs session-20
// DURATION_TOLERANCE which IS dead in viterbi_dp.py, deferred to harness).
//
// --- Session-20 CRITICAL CALLER OVERRIDE re-confirmed ---------------------
// `_build_neighbors` receives `min_forward_jump = time_signature` (= 4 for
// 4/4), NOT the declared default 16. This was the session-20 pre-port catch
// that prevented a restricted neighbor set; session-21 48-case corpus
// validated the production call bit-exact. `_viterbi_dp` separately receives
// `min_forward_jump = adaptive_forward` computed per target_ratio branch
// (≥0.5 uses COOLDOWN_BARS×TS=16 for 4/4; <0.5 uses the adaptive formula).
// Crucial: TWO different min_forward_jump values flow into two different
// callees. Both ported verbatim in `runDpAndBuildPath()`.
//
// --- Adaptive-cooldown formula (session-21 validated on 16 r0.3 cases) ----
// optimizer.py:249-258: when target_ratio < 0.5, cooldown scales down to
// allow more cuts. Formula `max(1, int(COOLDOWN_BARS * target_ratio * 2))`
// ports verbatim with `std::max(1, static_cast<int>(COOLDOWN_BARS *
// target_ratio * 2.0))`. Python `int()` truncates toward zero for positive;
// std::static_cast<int> matches for positive values. At session-21 48-case
// scale, zero drift; ViterbiDP session-20 port already uses this cast.
//
// --- Post-DP outro extension — ZERO current parity coverage ---------------
// optimizer.py:289-301. Fires when DP's last beat lies in the outro region
// AND there's song remaining. Extends path to end-of-song OR 20% of path
// length (whichever smaller). ViterbiDP parity tests (sessions 20+21) check
// ONLY raw `dp_path` — NOT the post-extension `remix_path`. Session-22
// `test_optimizer` is the FIRST parity gate on this branch. Potential
// surprise zone — if smoke produces drift, bisect here first.
//
// --- FMA contraction (ADR-028) --------------------------------------------
// Optimizer composes ViterbiDP penalties through W accumulation, plus the
// post-DP extension + metadata write. Same FMA vulnerability as upstream.
// `-ffp-contract=off` on `test_optimizer` target = 4th reuse of ADR-028.
//
// --- transition_metadata ordering ------------------------------------------
// Python uses insertion-ordered dict (Python 3.7+), building via path-order
// for-loop at optimizer.py:312-341. C++ uses `std::map<pair,map<string,
// double>>` = lex-order by key. For parity dumps, serialize sorted-by-
// (from, to) flat arrays on BOTH sides to ensure deterministic compare.
// Same pattern as session-19 candidates dump. Do NOT attempt to match
// Python's insertion order in C++ — per-key compare is order-agnostic.
//
// --- `blocked_transitions` W-mutation contract ----------------------------
// Python `remix()` mutates `self._W` in-place (blocked → INF), then restores
// on exit via try/finally. C++ preserves this: `W_` stored as non-const
// pointer (caller-owned), `remix()` applies blocks, runs DP, restores on
// normal return or exception (RAII guard via scope-exit block). For session
// -22 smoke test, blocked_transitions is empty — this code path is dead
// unless phase-6 UI exercises it.

// ---------------------------------------------------------------------------
// Default constructor arguments (Python optimizer.py L82-98)
// ---------------------------------------------------------------------------
//
// Exposed as constexpr so parity tests and future RegionOptimizer can
// reference the same values without re-tracking citations.

// A2/A3: preserve_intro_beats / preserve_outro_beats.
// CLEAN — cited at config.py:106-107 AND optimizer.py:92-93 with agreement.
// Consumed by `findIntroEnd()` / `findOutroBeats()` as fallback when no
// "intro"/"outro" label matches.
// Source-of-truth: `optimizer.py:92 + :93` ↔ `config.py:106 + :107`.
inline constexpr int kPreserveIntroBeatsDefault = 8;
inline constexpr int kPreserveOutroBeatsDefault = 4;

// A4: max_transitions.
// FALLBACK-DEFAULT — optimizer.py:94 declares 6; server/handlers/_remix
// _options.py:57 overrides to 12 (`_normalize_remix_options` default).
// The 6 value is reachable only if a caller constructs CleanOptimizer
// without going through the UI flow (e.g. parity tests). Session-22
// smoke test + future corpus pass the production value (varies per
// options).
// Source-of-truth: `optimizer.py:94 (2026-04-22)`.
inline constexpr int kMaxTransitionsDefaultOpt = 6;

// A5: duration_tolerance_sec.
// CLEAN — facade contract in ABSOLUTE SECONDS. Not to be confused with
// `RemixConfig.duration_tolerance=0.15` (ratio) at config.py:112 which
// drives `_remix_options.py:55` `max_slack_sec` seed separately. Session
// -20 ADR-026 addendum documented this as the facade canonical default.
// Source-of-truth: `optimizer.py:95 (2026-04-22)`.
inline constexpr double kDurationToleranceSecDefault = 5.0;

// A1: time_signature default. FALLBACK-DEFAULT — production always passes
// the BeatDetector-derived value (typically 4 for 4/4; 3 for 3/4; 6 for
// 6/8). Ignored for smoke test (billie_jean is 4/4).
// Source-of-truth: `optimizer.py:88 (2026-04-22)`.
inline constexpr int kTimeSignatureDefault = 4;

// F3: sample_rate.
// CLEAN — `DEFAULT_CONFIG.analysis.sample_rate = 22050` at config.py:32.
// Used ONLY in `_extract_remix_path` alignment_offset_sec computation
// (`alignment_lag_samples / max(1, sample_rate)`). Optimizer does NOT
// need the track's actual SR for the main-path DP; the value is a
// metadata-layer divisor. Exposed here so the input bundle can override
// without recompiling.
// Source-of-truth: `optimizer.py:329` ↔ `config.py:32 (2026-04-22)`.
inline constexpr int kSampleRateForMetadata = 22050;

// ---------------------------------------------------------------------------
// Inputs bundle (API shape Q1 rec A — consistent with TransitionCostInputs +
// ViterbiDPInputs + SegmentData free-function patterns).
// ---------------------------------------------------------------------------

struct CleanOptimizerInputs
{
    // REQUIRED — W matrix + candidates come from TransitionCost session 18.
    // `W` is MUTABLE (non-const) because `remix()` applies blocked-
    // transitions in place then restores. Caller retains ownership; the
    // buffer must outlive CleanOptimizer.
    double*                                                     W;
    const std::map<std::pair<int, int>, TransitionCandidate>*   candidates;
    int                                                         n_beats;  // = rows/cols of W

    // REQUIRED — beat timing.
    const double*  beat_times;    // (n_beats,) f64 seconds

    // SEGMENT CONTEXT (optional but strongly recommended — label-aware
    // intro/outro, segment similarity, boundary bonuses).
    // `segments == nullptr` → treated as Python `segments=None` → degenerate
    // SegmentData (all labels "unknown"), `findIntroEnd/Outro` return
    // preserve_* defaults.
    const analysis::Segment* segments;
    int                      n_segments;

    // FEATURES for segment similarity refinement (see SegmentData.h).
    // `features == nullptr` → label-only segment similarity (0.9 same, 0.2
    // cross). Session-19 corpus dumps always pass features; skip for smoke.
    const float* features;
    int          n_features;

    // DOWNBEATS for bar-alignment gates + RepetitionMap.
    // `downbeats == nullptr || n_downbeats == 0` → `_compute_downbeat_arrays`
    // synthesizes `range(0, n_beats, time_signature)` (viterbi_dp.py:491).
    // `downbeat_constraint=false` would fully disable (not threaded here —
    // matches production config.py:108 `True`).
    const double* downbeats;
    int           n_downbeats;

    // REPETITION MAP jumps — optional.
    // `jumps == nullptr || n_jumps == 0` → no repetition-based candidates
    // in `buildNeighbors`; no `is_repetition_jump` metadata in output.
    const analysis::RepetitionJump* jumps;
    int                             n_jumps;

    // PARAMETERS (Python constructor defaults; UI overrides in production).
    int    time_signature         = kTimeSignatureDefault;       // A1
    int    preserve_intro_beats   = kPreserveIntroBeatsDefault;  // A2
    int    preserve_outro_beats   = kPreserveOutroBeatsDefault;  // A3
    int    max_transitions        = kMaxTransitionsDefaultOpt;   // A4
    double duration_tolerance_sec = kDurationToleranceSecDefault;// A5
    int    sample_rate            = kSampleRateForMetadata;      // F3

    // ADR-083 sesja 92 — UI Min cut slider override for min_seq_after_jump
    // (cooldown beats between non-sequential transitions). Default 0 →
    // legacy compute `COOLDOWN_BARS × time_signature` (= 16 for 4/4) →
    // bit-exact baseline preservation. UI passes user slider value 4-32
    // beats when active.
    int    min_seq_after_jump_override = 0;

    // ADR-084 sesja 93 — UI Edit Length slider MULTIPLICATIVE jump-cost
    // scale propagated to ViterbiDPInputs::edit_length_jump_scale.
    // Default 1.0 → bit-exact baseline (no scaling). MainComponent maps
    // slider [0..100] via `2^((slider-50)/25)` ∈ [0.25, 4.0]. Supersedes
    // sesja-92 ADR-083 additive penalty design (range too small to
    // dominate the existing ~2.4 per-jump penalty).
    double edit_length_jump_scale = 1.0;

    // A6/A7: waveform_sample_rate / boundary_waveforms — stored by Python
    // `__init__` (L101-102) but NOT consumed by `optimize()` main-path.
    // Used only by `remix_region` (mixin, session 23) + phase-5 renderer.
    // OMITTED from session-22 port per Principle 2 (Simplicity First).
};

// ---------------------------------------------------------------------------
// CleanOptimizer — stateful facade
// ---------------------------------------------------------------------------
//
// `__init__` (session-22 port):
//   1. Store inputs (pointers + owned Segment copy).
//   2. Guard n_beats > 1: compute avg_beat_duration + cumulative.
//      Else: avg=0.5, cumulative=[0.0, 0.5] (B2/B3 fallback).
//   3. Compute SegmentData via session-19 `computeSegmentData`.
//   4. Compute DownbeatArrays via session-20 `computeDownbeatArrays`.
//   5. Find effective intro/outro via label scan.
//   6. Build neighbor CSR with `min_forward_jump = time_signature`
//      (CRITICAL SESSION-20 CALLER OVERRIDE).
//
// `remix(target_duration, blocked_transitions)`:
//   a. `n_beats < 2` short-circuit → RemixPath with sequential indices.
//   b. Compute DpParams.
//   c. Apply blocked_transitions to W (save old values).
//   d. RAII guard: run DP + build path. On scope exit restore W.
//   e. Return RemixPath with beat_indices + transitions + metadata.
class CleanOptimizer
{
public:
    explicit CleanOptimizer(const CleanOptimizerInputs& inputs);

    // Main entry point. Matches Python `remix(target_duration,
    // blocked_transitions=None)`.
    //
    // `blocked_transitions == nullptr || empty` → no mutation of W.
    // Otherwise each `(from_b, to_b)` pair has its W entry set to INF
    // for the duration of this call, then restored before return (or
    // on exception via RAII).
    RemixPath remix(double                                        target_duration,
                    const std::set<std::pair<int, int>>*          blocked_transitions = nullptr);

    // Port of optimizer.py:389-458 (session 58, ADR-048 / DEV-027).
    //
    // Returns up to `k` deterministic alternative paths by blocking each
    // transition in the best path (one at a time) and re-running DP. paths[0]
    // = best (unblocked); paths[1..] = alternatives ordered by which
    // transition of `best` was blocked (i.e. by transition position in the
    // base path, not by alternative quality — Python production semantic).
    // Dedup via `beat_indices` equality drops re-discoveries of the base
    // path or earlier alternatives.
    //
    // `blocked_transitions`: persistent user-blocked set, applied to every
    // generated path (base AND each alternative).
    // `k`: max paths to return; `k <= 1` short-circuits to `[best]`.
    //
    // Implementation per ADR-048 Option B: re-uses `remix(target, augmented)`
    // per loop iteration; the existing RAII guard inside `remix()` handles W
    // mutate + restore safely. Parity strategy `max_diff <= 1e-9` vs Python
    // golden on 2-track in-session corpus (billie_jean + woodkid_iron_acoustic
    // × variation 0..3); full 16-track × 4-variation gate deferred to phase-6
    // close per ADR-034.
    //
    // PORT FIX vs Python (`optimizer.py:410-411`): Python `n_beats < 2`
    // short-circuit returns `[self.remix(target_duration)]` without passing
    // `blocked_transitions`; C++ port forwards `blocked_transitions` for
    // semantic consistency with the standard `remix()` call (parity-neutral
    // on our corpus where every track has hundreds of beats).
    std::vector<RemixPath>
    remix_k_best(double                                        target_duration,
                 int                                           k,
                 const std::set<std::pair<int, int>>*          blocked_transitions = nullptr);

    // Convenience replicating Python server `_remix.py:140-147` exactly:
    //
    //   k        = max(2, variation_idx + 1)         (Python `_remix.py:143`)
    //   path_idx = min(variation_idx, len(paths)-1)  (Python `_remix.py:146`)
    //
    // Used by RemixPipeline when `Input::variation > 0` (Duration mode "Try
    // different splice" UX). Caller passes `variation_idx == 0` is allowed
    // but should prefer `remix(target, blocked)` directly (faster — skips
    // k-best machinery for the same result).
    //
    // ADR-048 / DEV-027 (session 58).
    RemixPath
    remix_variation(double                                        target_duration,
                    int                                           variation_idx,
                    const std::set<std::pair<int, int>>*          blocked_transitions = nullptr);

    // Diagnostic accessors for parity tests. Not part of Python surface;
    // expose the __init__ outputs so per-stage bisection can check each
    // computation independently of DP.
    int effectiveIntro() const noexcept { return effective_intro_; }
    int effectiveOutro() const noexcept { return effective_outro_; }

    double                    avgBeatDuration() const noexcept { return avg_beat_duration_; }
    const std::vector<double>& cumulative()     const noexcept { return cumulative_; }
    const SegmentData&        segmentData()     const noexcept { return segment_data_; }
    const DownbeatArrays&     downbeatArrays()  const noexcept { return downbeat_arrays_; }
    const NeighborCSR&        neighbors()       const noexcept { return neighbors_; }

private:
    // _dp_params output (Python dict L229-236). Named struct instead of a
    // map for type safety + zero-allocation hot path.
    struct DpParams
    {
        int    target_beats;
        bool   is_shortening;
        int    intro_beats;
        int    outro_beats;
        int    effective_max;
        int    effective_min;
    };

    int      findIntroEnd() const;
    int      findOutroBeats() const;
    DpParams computeDpParams(double target_duration) const;

    // W taken as non-const pointer because caller-owned buffer may be the
    // one currently blocked by blocked_transitions (see `remix()`).
    RemixPath runDpAndBuildPath(double* W, const DpParams& params) const;

    RemixPath extractRemixPath(const std::vector<std::int64_t>& path,
                               double                            cost) const;

    // -- State from inputs ---------------------------------------------------
    double*                                                     W_;           // non-const, caller-owned
    const std::map<std::pair<int, int>, TransitionCandidate>*   candidates_;  // nullable lookup
    int                                                         n_beats_;
    std::vector<double>                                         beat_times_;  // f64 promoted copy (Python L104)

    std::vector<analysis::Segment>                              segments_;    // owned copy (Python `list(segments or [])`)

    const analysis::RepetitionJump*                             jumps_;
    int                                                         n_jumps_;

    int                                                         time_signature_;
    int                                                         min_seq_after_jump_;    // COOLDOWN_BARS × TS
    int                                                         min_forward_jump_;      // COOLDOWN_BARS × TS
    int                                                         preserve_intro_;
    int                                                         preserve_outro_;
    int                                                         max_transitions_;
    double                                                      duration_tolerance_sec_;
    double                                                      edit_length_jump_scale_ = 1.0;  // ADR-084 sesja 93
    // ADR-083 sesja 92 — true when UI Min cut slider was dragged off
    // default 16. When set, _run_dp_and_build_path uses min_seq_after_jump_
    // directly instead of adaptive cooldown scaling — user explicit override
    // takes precedence over algorithm's aggressive-shortening adaptive scale.
    bool                                                        min_seq_after_jump_user_override_ = false;
    int                                                         sample_rate_;

    double                                                      avg_beat_duration_;
    std::vector<double>                                         cumulative_;  // size n_beats + 1

    SegmentData                                                 segment_data_;
    DownbeatArrays                                              downbeat_arrays_;

    int                                                         effective_intro_;
    int                                                         effective_outro_;

    NeighborCSR                                                 neighbors_;
};

} // namespace reamix::remix
