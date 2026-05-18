#pragma once

#include <cstddef>
#include <vector>

#include "analysis/StructureResult.h"

namespace reamix::remix {

// Transition pre-screening via recurrence-matrix diagonal search.
//
// Port of `references/python-source/remix/transition_prescreen.py` (160 LOC,
// 2026-04-22). Session-24 target per HANDOVER-23.
//
// Pre-identifies structurally meaningful transition candidates BEFORE the
// brute-force TransitionCost scoring loop. Based on ICASSP 2023 approach:
//
//   1. Build recurrence matrix R via `reamix::analysis::Recurrence::build`
//      (phase-3 session 7) with k_neighbors=12 (CALLER OVERRIDE, not kDefaultK).
//   2. For each segment pair (α → β), search R for bar-aligned diagonals near
//      the boundary exit(α) → entry(β). Bar-aligned diagonals indicate
//      repeated musical patterns — natural splice points.
//   3. Verify top-3 diagonals per pair with WaveformXcorr cross-correlation
//      against a 3 ms lag window (strict — different purpose than
//      TransitionCost's 30 ms alignment budget).
//   4. Deduplicate by (from, to) keeping the highest recurrence_score.
//   5. Sort by `diagonal_length × recurrence_score × max(0.1, waveform_sim)`.
//
// Pre-screened transitions BYPASS the chroma prefilter in TransitionCost
// (still subject to energy + quality-floor gates). See TransitionCost::
// TransitionCostInputs::prescreened_pairs — the flat `[pi0, pj0, pi1, pj1,
// ...]` array built from `PrescreenedTransition::from_beat / to_beat`.
//
// --- Citation discipline (ADR-026) ----------------------------------------
// Every hand-calibrated number in this file carries a
// `transition_prescreen.py:LINE (2026-04-22)` or `recurrence.py:LINE` source
// citation. Session-24 pre-port audit classified 12 main-path constants
// (P1-P12) in `weights-audit.md` — see session-24 log entry for the full
// taxonomy breakdown.
//
// --- CRITICAL CALLER OVERRIDE (5th confirmation of the pattern) ---------
// `transition_prescreen.py:74` calls `build_recurrence_matrix(..., k_neighbors
// =12)` — declared default is 10 (`recurrence.py:35`) and C++ mirrors with
// `Recurrence::kDefaultK=10`. Port MUST pass 12 explicitly, NOT kDefaultK.
// Pattern confirmed 5 times now (sessions 20/21/22/23/24) as the dominant
// finding mechanism. If BlockAssembly or any future module reads a default
// arg, grep caller sites BEFORE port.
//
// --- FMA contraction (ADR-028) --------------------------------------------
// `findDiagonals` sums doubles for mean_sim; parity against CPython requires
// `-ffp-contract=off` on the test target. 6th reuse per ADR-028 (after
// Quality / TransitionCost / ViterbiDP / Optimizer / RegionOptimizer).

// --- Module constants (ADR-026 audit, session 24) -----------------------

// Recurrence k_neighbors for prescreen mode — CRITICAL CALLER OVERRIDE.
// Source-of-truth: transition_prescreen.py:74 (2026-04-22). (P1)
inline constexpr int PRESCREEN_K_NEIGHBORS = 12;

// Waveform verification lag window: 3 ms. 10× tighter than TransitionCost's
// 30 ms alignment shift — different purpose (structural verification vs.
// audio alignment). UNJUSTIFIED per session-24 audit (P2).
// Source-of-truth: transition_prescreen.py:84 (2026-04-22).
inline constexpr double PRESCREEN_MAX_LAG_MS = 3.0;

// Search window around segment boundaries: ±4 bars. UNJUSTIFIED per
// session-24 audit (P3).
// Source-of-truth: transition_prescreen.py:98 (2026-04-22).
inline constexpr int PRESCREEN_SEARCH_BARS = 4;

// Top-3 diagonals per segment pair kept for waveform verification.
// UNJUSTIFIED per session-24 audit (P6).
// Source-of-truth: transition_prescreen.py:123 (2026-04-22).
inline constexpr int PRESCREEN_TOP_DIAGONALS_PER_PAIR = 3;

// Minimum mean recurrence similarity for a diagonal to pass.
// UNJUSTIFIED per session-24 audit (P7).
// Source-of-truth: transition_prescreen.py:124 (2026-04-22).
inline constexpr double PRESCREEN_MIN_RECURRENCE_SIM = 0.05;

// Default minimum waveform similarity. Overridable via
// PrescreenInputs::min_waveform_similarity. FALLBACK-DEFAULT per
// session-24 audit (P8) — production caller never overrides.
// Source-of-truth: transition_prescreen.py:49 (2026-04-22).
inline constexpr double PRESCREEN_DEFAULT_MIN_WAVEFORM_SIM = 0.30;

// Sort-key waveform floor: results ranked by
//   length × rec_score × max(PRESCREEN_SORT_WF_FLOOR, waveform_sim).
// Prevents waveform_sim=0 cases (has_wf=false) from collapsing the sort key.
// UNJUSTIFIED per session-24 audit (P9).
// Source-of-truth: transition_prescreen.py:157 (2026-04-22).
inline constexpr double PRESCREEN_SORT_WF_FLOOR = 0.1;

// Gate below which prescreen returns an empty list without doing any work.
// PARITY: transition_prescreen.py:70 — `n < 8 OR n_segments < 2 → []`.
inline constexpr int PRESCREEN_MIN_BEATS = 8;

// --- Data ----------------------------------------------------------------

// A structurally pre-identified transition candidate.
// PARITY: PrescreenedTransition frozen dataclass at transition_prescreen.py:29-38.
struct PrescreenedTransition
{
    int    from_beat;
    int    to_beat;
    int    diagonal_length;      // length in beats of matched pattern
    double recurrence_score;     // mean R along diagonal
    double waveform_similarity;  // verification xcorr (0 if has_wf=false)
};

// Inputs bundle — raw pointers per session-22/23 pattern.
struct PrescreenInputs
{
    // Beat features: (n_beats, n_features) row-major f32.
    const float* features;
    int          n_beats;
    int          n_features;

    // Beat onset times: (n_beats,) f64 seconds. Currently unused internally
    // (mirror of recurrence.py vestigial arg pattern), but kept for Python-
    // API parity. Callers may pass nullptr if not available.
    const double* beat_times;

    // Structure segments (label + start/end in seconds). Required.
    const analysis::Segment* segments;
    int                       n_segments;

    // Optional downbeats (seconds). Currently unused by prescreen logic
    // (pattern-analogous to recurrence.py vestigial arg), but accepted for
    // caller-side API symmetry with _analyze.py. nullptr OK.
    const double* downbeats;
    int           n_downbeats;

    // Optional waveform verification. If boundary_waveforms == nullptr OR
    // waveform_sample_rate == 0 OR n_boundary_waveforms < n_beats, skip
    // waveform gate.
    const float* boundary_waveforms;    // (n_bnd, n_samples) row-major f32
    int          n_boundary_waveforms;
    int          n_samples_per_bnd;
    int          waveform_sample_rate;

    // Parameters (Python function defaults).
    int    time_signature         = 4;
    double min_waveform_similarity = PRESCREEN_DEFAULT_MIN_WAVEFORM_SIM;
};

// Main entry point. Returns list of PrescreenedTransition sorted DESC by
// `length × recurrence_score × max(PRESCREEN_SORT_WF_FLOOR, waveform_sim)`.
// Empty list if n_beats < PRESCREEN_MIN_BEATS OR n_segments < 2.
std::vector<PrescreenedTransition>
prescreenTransitions(const PrescreenInputs& inputs);

} // namespace reamix::remix
