#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "analysis/StructureResult.h"
#include "remix/TransitionCost.h"  // TransitionCandidate, ENERGY_HARD_BLOCK_DB, INF, TRACK_VOCAL_THRESHOLD, chromaRange

namespace reamix::remix {

// Region-specific transition cost computation.
//
// Port of `references/python-source/remix/region_cost.py` (431 LOC,
// 2026-04-21). Session-23 target per HANDOVER-22. Companion to
// `RegionOptimizer` — region remix mode needs its own W matrix with
// RELAXED hard gates because the global micro-skip (4 × time_signature)
// blocks all close-range jumps in a small region.
//
// Relaxations vs. TransitionCost (global `compute_transition_costs`):
//   - micro-skip: absolute 4 beats (flat, not `4 × time_signature`) — design
//     choice; regions operate on region-relative indices where a 4-beat
//     minimum prevents "every-bar" jitter while still allowing intra-section
//     splicing.
//   - chroma prefilter: 0.55 (vs. global 0.45) — more candidates survive
//     because region is small; tighter filter would empty the candidate pool.
//   - span penalty (cross-section): halved (vs. global full penalty) — regions
//     are usually single-section, so cross-section span is rare and expected
//     when it happens.
//   - same-section span: ZERO penalty (vs. global SPAN_PENALTY_SAME_SECTION) —
//     "that's the whole point of region remix" per region_cost.py:387.
//
// Shared with TransitionCost (via public includes):
//   - ENERGY_HARD_BLOCK_DB = 8.0 dB hard gate.
//   - TRACK_VOCAL_THRESHOLD = 0.75 track-level vocal detection.
//   - TransitionCandidate struct for per-pair diagnostic record.
//   - waveform_xcorr via dsp::WaveformXcorr.
//   - chromaRange helper (session-23 promotion from TransitionCost.cpp anon namespace).
//
// Unlike TransitionCost, this module does NOT:
//   - Compute vocal-band deceptive-splice detection (regions are short; the
//     full-mix xcorr carries the parity budget).
//   - Emit a global `importance` array (region DP uses terminal penalties, not
//     importance-weighted composition).
//   - Emit `segment_boundaries` (regions typically within one section).
//
// PARITY notes:
//
// --- Citation discipline (ADR-026) --------------------------------------------
// Every hand-calibrated number in this file carries a
// `// region_cost.py:LINE (2026-04-22)` source citation. Session-23 pre-port
// audit classified 20 main-path constants (C1-C20) in `weights-audit.md`:
//   - CLEAN (6): C6/C7/C8/C9/C10/C20 — inherited from session-18
//     TransitionCost + session-17 Quality patterns.
//   - FALLBACK-DEFAULT (2): C3/C4 — handler always overrides.
//   - UNJUSTIFIED (10): C1/C2/C5/C14/C16/C17/C18/C19 — region-specific
//     scales + relaxed thresholds without Python citation.
//   - UNJUSTIFIED-DRIFT (2): C11 (context window literals ignore
//     `config.context_window_beats=2` which is DEAD-CONFIG), C15 (edge_energy
//     cap 12.0 — same 8/12 dB pattern research-scout flagged session-16).
//   - MARKER (1): C13 — `bar_aligned = 1.0` sentinel.
//   - DEFENSIVE (1): C12 — norm numerical guard.
//
// --- FMA contraction (ADR-028) ------------------------------------------------
// RegionCost composes weighted sums through `Quality::computeQualityScore`
// plus direct dot products for context similarity. `-ffp-contract=off`
// required on the parity test target per ADR-028 (5th reuse).
//
// --- Iteration order (np.argpartition equivalent) -----------------------------
// RegionCost does NOT use top-K candidate selection — it evaluates every
// region-internal pair that passes hard gates. No `std::nth_element` needed.
// `candidates` map iteration is lexicographic by (from, to) per session-18
// TransitionCost convention.

// ---------------------------------------------------------------------------
// Region-specific hard gates (relaxed vs. global)
// ---------------------------------------------------------------------------

// Region-relative micro-skip: absolute 4 beats (not time-sig-scaled).
// Distinct from global `micro_skip = 4 × time_signature` — regions use
// region-relative indices where a flat 4-beat floor prevents per-bar jitter.
// Comment at `region_cost.py:4-9` justifies the relaxation vs. global.
// UNJUSTIFIED per session-23 audit (C1) — no per-meter scaling rationale.
// Source-of-truth: `region_cost.py:38 (2026-04-22)`.
inline constexpr int REGION_MICRO_SKIP_BEATS = 4;

// Region-relative chroma prefilter: 0.55 (vs. global CHROMA_PREFILTER_THRESHOLD
// = 0.45). More candidates survive in small region pools. UNJUSTIFIED per
// session-23 audit (C2) — no empirical evidence for the 0.55 value.
// Source-of-truth: `region_cost.py:39 (2026-04-22)`.
inline constexpr double REGION_CHROMA_PREFILTER = 0.55;

// Sequential-fill scale + cap (region-specific).
// region_W[ri, ri+1] = min(chroma_D[ri, ri+1] * 0.3, 0.12).
// UNJUSTIFIED per session-23 audit (C5). Matches global SEQUENTIAL_COEFF_NONBOUND
// numerically but region skips the boundary-aware branch.
// Source-of-truth: `region_cost.py:118 (2026-04-22)`.
inline constexpr double REGION_SEQUENTIAL_COEFF = 0.3;
inline constexpr double REGION_SEQUENTIAL_CAP   = 0.12;

// Context window (half-window literals hardcoded, NOT `config.context_window_beats=2`
// which is DEAD-CONFIG per session-23 audit). Asymmetric: 2 beats before + 3
// beats after the split point (region_cost.py:337-338).
// UNJUSTIFIED-DRIFT per session-23 audit (C11).
// Source-of-truth: `region_cost.py:337 + :338 (2026-04-22)`.
inline constexpr int REGION_CONTEXT_BEFORE_BEATS = 2;
inline constexpr int REGION_CONTEXT_AFTER_BEATS  = 3;

// Section-sim blend: `section_sim = label_match * 0.9 + 0.1` — 0.9 when labels
// match, 0.1 otherwise (saturation floor). UNJUSTIFIED per session-23 audit (C17).
// Source-of-truth: `region_cost.py:368 (2026-04-22)`.
inline constexpr double REGION_SECTION_SIM_SCALE = 0.9;
inline constexpr double REGION_SECTION_SIM_BIAS  = 0.1;

// Cross-section span penalty halving — "halved for cross-section (short jumps
// are expected in regions)". Applied when `jump_beats < SPAN_PENALTY_MAX_BEATS
// and label_match < 0.5`. UNJUSTIFIED per session-23 audit (C18, C19).
// Source-of-truth: `region_cost.py:385-386 (2026-04-22)`.
inline constexpr double REGION_LABEL_MATCH_THRESHOLD = 0.5;
inline constexpr double REGION_SPAN_PENALTY_HALVING  = 0.5;

// ---------------------------------------------------------------------------
// Inputs / outputs
// ---------------------------------------------------------------------------

// Inputs bundle — mirrors compute_region_costs kwargs at region_cost.py:42-62.
// Raw pointers + explicit sizes for zero-copy interop with NpyIO-loaded
// parity test buffers. All pointers caller-owned.
struct RegionCostInputs
{
    // REQUIRED ---------------------------------------------------------------
    int           entry_beat;               // region-start beat index (inclusive)
    int           exit_beat;                // region-end beat index (exclusive)
    const float*  features;                 // (n_total, n_features) row-major f32
    int           n_total;                  // total number of beats in track
    int           n_features;
    const double* beat_times;               // (n_total,) f64 seconds

    // SEGMENT CONTEXT (optional) --------------------------------------------
    const analysis::Segment* segments;
    int                       n_segments;

    // WAVEFORM SCORING (optional; nullptr → skip xcorr branch) -------------
    const float* boundary_waveforms;
    int          n_boundary_waveforms;
    int          n_samples_per_bnd;
    int          waveform_sample_rate;

    // EDGE ARRAYS (optional) -----------------------------------------------
    const double* edge_rms_start;
    const double* edge_rms_end;
    const float*  edge_features_start;      // (n_total, n_edge_features)
    const float*  edge_features_end;
    int           n_edge_features;

    // PER-BEAT SCALARS (all optional) --------------------------------------
    const double* rms_energy;
    const double* onset_strength;
    const double* spectral_centroid;
    const double* vocal_activity;
    const double* edge_vocal_activity_start;
    const double* edge_vocal_activity_end;
    // ADR-088 sesja 98 — vocal phrase boundary signals (see TransitionCost.h).
    const double* edge_vocal_onset_start;
    const double* edge_vocal_release_end;

    // DOWNBEAT INDICATOR ---------------------------------------------------
    const double* downbeats;
    int           n_downbeats;

    // PARAMETERS -----------------------------------------------------------
    int time_signature = 4;  // CALLER OVERRIDE per session-23 audit — handler
                             // at _remix.py:133 passes `time_signature or 4`.

    // QualityWeights override (sesja 71, ADR-058 calibration). nullptr →
    // use kDefaultQualityWeights (preserves parity tests). Quality.h is
    // already included via TransitionCost.h.
    const QualityWeights* quality_weights = nullptr;

};

// Outputs bundle.
struct RegionCostResult
{
    // region_W[ri * n_region + rj]: (n_region, n_region) row-major f64,
    // region-relative indices. n_region = exit_beat - entry_beat.
    std::vector<double> region_W;

    // Candidates keyed by ABSOLUTE beat indices (from_beat, to_beat).
    // Matches Python region_cost.py:146 — `candidates[(candidate.from_beat,
    // candidate.to_beat)] = candidate` where from_beat/to_beat are absolute.
    std::map<std::pair<int, int>, TransitionCandidate> candidates;

    int n_region = 0;
};

// Main entry point. Port of `compute_region_costs` (region_cost.py:42-148).
RegionCostResult computeRegionCosts(const RegionCostInputs& inputs);

} // namespace reamix::remix
