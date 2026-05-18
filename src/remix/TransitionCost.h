#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "analysis/StructureResult.h"
#include "remix/Quality.h"  // QualityWeights override (ADR-058)

namespace reamix::remix {

// Transition cost computation — 10-signal diagnostic-calibrated scorer.
//
// Port of `references/python-source/remix/transition_cost.py` (559 LOC,
// 2026-04-21). Session-18 target per HANDOVER. Consumes the weights and
// penalty constants from `Quality.{h,cpp}` (session 17) + waveform xcorr
// (`src/dsp/WaveformXcorr`, session 11) + optional vocal-band xcorr
// (`src/dsp/SosFilt`, session 18) for deceptive-splice detection.
//
// The module composes three layers:
//
//   1. Shared matrices (O(n_beats²) each): chroma distance, row-shifted
//      full-feature cosine similarity, edge-splice similarity, segment
//      similarity (via `_compute_segment_data` helper — Python has it in
//      `viterbi_dp.py:387-462`; we inline it here because TransitionCost
//      is the sole consumer until ViterbiDP lands session ~19-20).
//   2. Per-candidate scoring loop (O(n_beats × k)): for each source i, pick
//      top-k by chroma distance, apply hard gates (chroma, energy, vocal
//      runtime), compute the 10 similarity signals, apply span / vocal /
//      onset penalties from `Quality.cpp`, compose the composite quality
//      score, and write W[i, j] = 1 - quality.
//   3. Sequential fill: W[i, i+1] gets a reduced cost (no splice at all)
//      that scales with chroma distance + segment-boundary indicator.
//
// PARITY notes (read CAREFULLY before changing any constant below):
//
// --- Citation discipline (ADR-026 + addendum) ------------------------------
// Every hand-calibrated number in this file is a verbatim port of a
// corresponding constant in `transition_cost.py` (L44-58) or `quality.py`
// (L22-59 — cited via `Quality.h`). Each constant carries a
// `// transition_cost.py:LINE (2026-04-21)` source citation. Docstring
// drift at `transition_cost.py:1-22` (stale waveform=0.22, sum=1.04) is
// ignored per Hard Rule #6; canonical source is `quality.py L22-31`.
//
// --- The spec-bug catches (session 16 + session 18) -------------------------
// (1) ENERGY_HARD_BLOCK_DB = 8.0 (hard gate), NOT 12.0 (= saturation of
//     the `edge_energy_match` SOFT penalty at `transition_cost.py:477`).
//     Spec.md referenced 12 dB pre-session-16; research-scout caught it.
// (2) VOCAL_GATE_THRESHOLD = 0.50 is DECLARED at `transition_cost.py:53`
//     but NOT USED at runtime. The real runtime gate at
//     `transition_cost.py:426` is `min(va_i, va_j) > 0.35` — hardcoded.
//     We expose BOTH constants; runtime uses 0.35 per Python behavior.
//     Tracked in `weights-audit.md` as a declaration-vs-runtime drift.
//
// --- The hard-gate vs soft-penalty distinction -----------------------------
// A HARD GATE short-circuits the candidate with `continue` (no score, no
// W[i,j] write; W stays INF). A SOFT PENALTY is a subtractive component of
// the composite quality score. Confusing the two is the "weakest link"
// bug class the user flagged at phase-4 kickoff.
//
// Hard gates (transition_cost.py L390-399 + L534-535 + chroma prefilter L390):
//   - chroma_distance > 0.45     → skip (CHROMA_PREFILTER_THRESHOLD)
//   - |edge_dB_end - edge_dB_start| > 8 dB → skip (ENERGY_HARD_BLOCK_DB)
//   - composite quality < 0.45   → skip (QUALITY_HARD_FLOOR, in Quality.h)
//   - micro-skip: jump_beats < 4×time_signature → skip (MICRO_SKIP)
//
// Soft-penalty saturations (NOT gates):
//   - edge_energy_match = max(0, 1 - min(diff, 12) / 12)     L477
//   - energy_match = max(0, 1 - rms_diff * 5.0)              L472
//   - centroid_match = max(0, 1 - c_diff * 5.0)              L483
//
// --- FMA contraction (ADR-028) ---------------------------------------------
// TransitionCost composes weighted sums (via `Quality::computeQualityScore`)
// whose parity against CPython depends on `-ffp-contract=off` on the test
// target. The plugin target keeps default FMA (strictly more accurate,
// inaudible bit-level difference). CMakeLists.txt applies the flag to
// `test_transition_cost` per ADR-028 precedent.
//
// --- Iteration order and determinism ---------------------------------------
// `np.argpartition(chroma_row, k)[:k]` at transition_cost.py:380 returns
// the k smallest entries in the first k positions but NOT SORTED. Python
// iterates in this partition order; we use `std::nth_element` for the same
// behavior. The iteration order affects only which `(i, j)` pair's
// candidate record is written LAST if duplicates exist (prescreened +
// top-k overlap); the final value is identical (pure function of i, j).
// For the `candidates` map, C++ uses `std::map<pair, Candidate>` (sorted
// by key) — iteration-order in dump is lexicographic, while Python's dict
// preserves insertion-order. Dump-time comparison is per-key, so order
// mismatch does not affect parity.

// ---------------------------------------------------------------------------
// Hard-gate thresholds + structural constants
// ---------------------------------------------------------------------------

// Sentinel "blocked transition" cost. Python `INF = 1e9` at
// `transition_cost.py:44`. Used to init W before sequential fill + to mark
// chroma_row entries in the micro-skip block.
// Source-of-truth: `transition_cost.py:44 (2026-04-21)`.
inline constexpr double INF = 1e9;

// Hard gate: waveform xcorr floor below which we would block transitions.
// Python declares this at `transition_cost.py:49` as WAVEFORM_QUALITY_FLOOR.
// Note: in the RUNTIME CODE PATH of `compute_transition_costs`, this
// constant is NOT applied as a standalone gate — the QUALITY_HARD_FLOOR at
// 0.45 (Quality.h) subsumes it (a transition with waveform_sim < 0.35 and
// W_WAVEFORM=0.34 contributes at most 0.34 × 0.35 = 0.12 to composite, so
// the hard floor at 0.45 blocks it unless other signals compensate). Kept
// as declared constant for spec consistency and downstream module use
// (BlockAssembly compatibility matrix may apply it as gate).
// Source-of-truth: `transition_cost.py:49 (2026-04-21)`.
inline constexpr double WAVEFORM_QUALITY_FLOOR = 0.35;

// Hard gate: transitions whose row-shifted chroma distance exceeds this
// are skipped pre-score. This is the primary candidate filter: before
// computing any other signal, the source beat keeps only its k chroma-
// closest targets AND those whose chroma distance is within threshold.
// Source-of-truth: `transition_cost.py:50 (2026-04-21)`.
inline constexpr double CHROMA_PREFILTER_THRESHOLD = 0.45;

// Hard gate: absolute edge-RMS difference in dB. If the source's trailing
// edge is louder (or quieter) than the destination's leading edge by more
// than 8 dB, the splice is audibly unnatural.
//
// CRITICAL: this is 8.0 dB, NOT 12.0 dB. The 12.0 value at
// `transition_cost.py:477` is the saturation cap of the `edge_energy_match`
// SOFT penalty (= `1 - min(diff, 12)/12`), not a gate. Research-scout
// 2026-04-21 + session-16 kickoff caught the spec.md mismatch pre-port.
// Source-of-truth: `transition_cost.py:51 (2026-04-21)`.
inline constexpr double ENERGY_HARD_BLOCK_DB = 8.0;

// Documentation anchor for the 4-bar micro-skip in 4/4. Actual runtime
// value depends on time signature: micro_skip = 4 × time_signature
// (`transition_cost.py:361`). 4/4 → 16, 3/4 → 12, 6/8 → 24.
// Defined here so the header documents the "4 bars" intent; call sites
// in TransitionCost.cpp compute it per-track.
// Source-of-truth: `transition_cost.py:52 + L361 (2026-04-21)`.
inline constexpr int MICRO_SKIP_BEATS_DEFAULT_4_4 = 16;

// DECLARED-BUT-UNUSED at runtime. Python `transition_cost.py:53` defines
// `VOCAL_GATE_THRESHOLD = 0.50` but the runtime check for vocal-band
// deceptive-splice detection at L426 uses a HARDCODED 0.35 instead
// (`min(va_i, va_j) > 0.35`). This is a declaration-vs-runtime drift in
// the Python reference. Port BOTH per Principle 6 (document, don't fix
// reference): `VOCAL_GATE_THRESHOLD` matches the Python const (useful if
// future session harmonizes the code), `VOCAL_SPLICE_ACTIVITY_MIN`
// matches actual runtime behavior. Runtime uses the 0.35 value. Flagged
// in `weights-audit.md` session 18.
// Source-of-truth: `transition_cost.py:53` (const), `:426` (runtime).
inline constexpr double VOCAL_GATE_THRESHOLD     = 0.50;  // unused at runtime
inline constexpr double VOCAL_SPLICE_ACTIVITY_MIN = 0.35; // actual runtime gate

// Track-level vocal detection: if `max(vocal_activity) < 0.75`, the entire
// track is treated as instrumental (skips vocal penalty + deceptive-splice
// branch). Comment at `transition_cost.py:54-57` justifies the 0.75 value
// empirically against HPSS mis-classifying funk bass/guitar as vocals.
// Source-of-truth: `transition_cost.py:58 (2026-04-21)`.
inline constexpr double TRACK_VOCAL_THRESHOLD = 0.75;

// Sequential-transition floor: `W[i, i+1] = min(chroma_D[i, i+1] * 0.3,
// SEQUENTIAL_FLOOR)` on non-boundary pairs; segment boundaries get a
// harsher `chroma_D * 0.5 + 0.1`. UNJUSTIFIED per weights-audit.md —
// no citation, no comment, matches SPAN_PENALTY_SAME_SECTION numerically
// which may be accidental.
// Source-of-truth: `transition_cost.py:218 (2026-04-21)`.
inline constexpr double SEQUENTIAL_FLOOR            = 0.12;
inline constexpr double SEQUENTIAL_COEFF_NONBOUND   = 0.3;   // `transition_cost.py:227`
inline constexpr double SEQUENTIAL_COEFF_BOUNDARY   = 0.5;   // `transition_cost.py:229`
inline constexpr double SEQUENTIAL_BIAS_BOUNDARY    = 0.1;   // `transition_cost.py:229`

// Deceptive-splice detector: when full-mix waveform sim exceeds vocal-band
// sim by more than this, the instruments mask a vocal mismatch; we
// REPLACE full-mix with vocal-band as the reported waveform signal.
// Source-of-truth: `transition_cost.py:434 (2026-04-21)`.
inline constexpr double DECEPTIVE_SPLICE_GAP_THRESHOLD = 0.3;

// Soft-penalty saturations (NOT gates — see docstring above).
inline constexpr double EDGE_ENERGY_SATURATION_DB = 12.0; // `transition_cost.py:477`
inline constexpr double RMS_DIFF_SCALE            = 5.0;  // `transition_cost.py:472`
inline constexpr double CENTROID_DIFF_SCALE       = 5.0;  // `transition_cost.py:483`

// Context-window radius (asymmetric per Python):
//   context_before = features[i-3 : i+1]   (4 beats ending at source)
//   context_after  = features[j   : j+4]   (4 beats starting at dest)
// Source-of-truth: `transition_cost.py:450-451 (2026-04-21)`.
inline constexpr int CONTEXT_BEFORE_LO_OFFSET = -3;  // lo = i + offset
inline constexpr int CONTEXT_BEFORE_HI_OFFSET =  1;  // hi = i + offset
inline constexpr int CONTEXT_AFTER_LO_OFFSET  =  0;  // lo = j + offset
inline constexpr int CONTEXT_AFTER_HI_OFFSET  =  4;  // hi = j + offset

// Vocal-band bandpass for deceptive-splice detection.
// Python: `scipy.signal.butter(4, [250, 3400], btype="bandpass", fs=sr,
// output="sos")` at `transition_cost.py:351`. In C++, SOS coefficients are
// pre-computed in the Python dump tool (session 18) and loaded at runtime;
// future session ports `butter + zpk2sos` natively.
// Source-of-truth: `transition_cost.py:351 (2026-04-21)`.
inline constexpr double VOCAL_BAND_LO_HZ        = 250.0;
inline constexpr double VOCAL_BAND_HI_HZ        = 3400.0;
inline constexpr int    VOCAL_BAND_FILTER_ORDER = 4;
inline constexpr int    VOCAL_BAND_MIN_SR       = 8000;  // `transition_cost.py:348`

// Silence / degeneracy floors.
inline constexpr double EDGE_DB_FLOOR = 1e-6;  // `transition_cost.py:206-207`
// Note: `COSINE_DEGENERATE_FLOOR` (1e-8) moved to `src/remix/SegmentData.h`
// session 19 when `_compute_segment_data` was extracted out of this module.
// TransitionCost.cpp now consumes it via `#include "remix/SegmentData.h"`.

// Default parameters (Python function defaults). Callers may override.
inline constexpr int    MAX_CANDIDATES_PER_BEAT_DEFAULT     = 16;   // `transition_cost.py:278`
inline constexpr double WAVEFORM_ALIGN_MAX_SHIFT_MS_DEFAULT = 30.0; // `config.py:165`

// ---------------------------------------------------------------------------
// Importance from segment labels (transition_cost.py:93-103)
// ---------------------------------------------------------------------------

// Per-section baseline importance. Used by `computeImportance` to build a
// per-beat importance array; consumed by the ViterbiDP DP kernel
// (session 19+) as a weight on path-cost accumulation.
double sectionImportance(const std::string& label);

// ---------------------------------------------------------------------------
// chroma slice range (config.py::chroma_range)
// ---------------------------------------------------------------------------
//
// Port of `config.py::chroma_range` helper (L20-24). 59-dim full feature
// vector → (40, 52); 39-dim fast-mode → (20, 32). Session 23 promoted from
// TransitionCost.cpp anonymous namespace to public — RegionCost.cpp is the
// second TU consumer per Hard Rule #1 single-source-of-truth. Value
// unchanged, zero semantic drift.
inline constexpr int N_CHROMA_DIMS   = 12;   // config.py:16
inline constexpr int N_CONTRAST_DIMS = 7;    // config.py:17

std::pair<int, int> chromaRange(int n_features);

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Per-candidate diagnostic record. Mirrors Python `TransitionCandidate`
// (transition_cost.py:64-77) frozen dataclass.
struct TransitionCandidate
{
    int    from_beat;
    int    to_beat;
    double quality_score;          // composite [0, 1], higher = better
    double waveform_similarity;    // 0 if not computed
    double successor_similarity;
    double edge_splice_similarity; // 0 if not computed
    double chroma_distance;
    double energy_diff_db;         // 0 if edge_db not provided
    int    alignment_lag_samples;  // 0 if no xcorr
    double total_cost;             // = 1 - quality_score

    // DEV-028 sesja 80 — full per-pair component values for D1 correlation
    // matrix (ADR-068 § D1). Captures all 12 cost components on surviving
    // pairs so the Python correlation script can compute Pearson ρ over
    // 16-track corpus. Default 0.0 preserves backward compatibility for
    // callers that don't read these fields.
    double context_similarity      = 0.0;
    double label_match             = 0.0;
    double section_similarity      = 0.0;
    double bar_aligned             = 0.0;
    double energy_match            = 1.0;
    double edge_energy_match       = 1.0;
    double centroid_match          = 1.0;
    double transient_continuity    = 0.0;  // 0 if onset_strength absent
    double mfcc_continuity         = 0.0;  // 0 if features absent (matrix empty)
};

// Inputs bundle — raw pointers + explicit sizes for zero-copy interop with
// NpyIO-loaded parity test buffers. All pointers are caller-owned. A null
// optional signal skips the corresponding branch in `compute`.
struct TransitionCostInputs
{
    // REQUIRED ----------------------------------------------------------
    const float*  features;        // (n_beats, n_features) row-major f32
    int           n_beats;
    int           n_features;
    const double* beat_times;      // (n_beats,) f64, seconds

    // SEGMENT CONTEXT (optional, but matters for label/section/importance) -
    // nullptr → all labels are "unknown", importance = 0.5 flat,
    // seg_sim_matrix = ones(1,1).
    const analysis::Segment* segments;
    int                       n_segments;

    // Sequential-fill boundary indicator (optional).
    // segment_boundaries[k] in [0, n_beats) flags a beat as a boundary —
    // W[i, i+1] uses the harsher coeff when either i or i+1 is flagged.
    // nullptr → no boundaries.
    const int* segment_boundaries;
    int        n_segment_boundaries;

    // WAVEFORM SCORING (optional; nullptr → waveform_sim signal missing,
    // weight redistributed in Quality::computeQualityScore).
    const float* boundary_waveforms;  // (n_bnd, n_samples) row-major f32
    int          n_boundary_waveforms;
    int          n_samples_per_bnd;
    int          waveform_sample_rate;

    // Pre-filtered vocal-band waveforms (optional; nullptr → skip
    // deceptive-splice detection). Callers pre-compute via the Python
    // `scipy.signal.butter + sosfilt` pipeline (session 18 test path) or
    // via `dsp::SosFilt::apply` + SOS coefficients loaded from dump
    // (session 18+ production path). Shape matches boundary_waveforms.
    const float* vocal_band_waveforms;

    // EDGE RMS for energy gate + edge_energy_match soft penalty.
    // Both required or both null. nullptr → edge-dB signal missing.
    const double* edge_rms_start;     // (n_beats,)
    const double* edge_rms_end;

    // EDGE FEATURES for edge_splice_sim.
    // Both required or both null. nullptr → edge-splice signal missing.
    const float* edge_features_start; // (n_beats, n_edge_features)
    const float* edge_features_end;
    int          n_edge_features;

    // PER-BEAT SCALARS (all optional).
    const double* rms_energy;         // (n_beats,) → energy_match
    const double* onset_strength;     // (n_beats,) → compute_onset_penalty
    const double* spectral_centroid;  // (n_beats,) → centroid_match
    const double* vocal_activity;     // (n_beats,) → vocal penalty + gate
    const double* edge_vocal_activity_start;  // (n_beats,) optional
    const double* edge_vocal_activity_end;    // (n_beats,) optional
    // ADR-088 sesja 98 — vocal phrase boundary signals.
    // edge_vocal_release_end[i] = release pending at beat i (vocal cuts off).
    // edge_vocal_onset_start[j] = onset starting at beat j (entering mid-word).
    // Per-pair quality = 1 - max(release_end[i], onset_start[j]); fed to
    // qi.vocal_continuity. nullptr default preserves bit-exact parity.
    const double* edge_vocal_onset_start;     // (n_beats,) optional
    const double* edge_vocal_release_end;     // (n_beats,) optional

    // DOWNBEAT INDICATOR for bar-alignment scoring.
    // If downbeats is null OR n_downbeats == 0, falls back to
    // `range(0, n_beats, time_signature)` per Python L315.
    const double* downbeats;          // (n_db,) f64, seconds
    int           n_downbeats;

    // PARAMETERS (Python function defaults).
    int    time_signature             = 4;
    // Sesja 81 ADR-068 D3: default lowered 0.45 → 0.20. Phase-4 parity tests
    // pin to LEGACY_QUALITY_HARD_FLOOR = 0.45 explicitly to remain Python-
    // bit-exact past this default flip; production (AnalyzePipeline) picks
    // up 0.20 via this field default.
    double quality_floor              = QUALITY_HARD_FLOOR;
    double chroma_prefilter           = CHROMA_PREFILTER_THRESHOLD;
    int    max_candidates_per_beat    = MAX_CANDIDATES_PER_BEAT_DEFAULT;
    double waveform_align_max_shift_ms = WAVEFORM_ALIGN_MAX_SHIFT_MS_DEFAULT;

    // Prescreened pairs flat array: [pi0, pj0, pi1, pj1, ...]. Bypass the
    // chroma-prefilter gate (still subject to energy + quality-floor gates).
    // nullptr → no prescreen.
    const int* prescreened_pairs;
    int        n_prescreened_pairs;

    // QualityWeights override (sesja 71, ADR-058 calibration). nullptr →
    // use kDefaultQualityWeights (preserves parity tests + production
    // baseline). Calibration callers (tools/calibration_harness) pass a
    // non-default `QualityWeights*` to sweep weight space.
    const QualityWeights* quality_weights = nullptr;

    // ---- Reserved 11th cost-component slot, DEV-028 sesja 74 ----------
    // Optional flat (n_beats × n_beats) row-major double matrix indexed
    // [i*n + j]. nullptr → no extra1 contribution (production default).
    // Calibration harness allocates + fills this matrix Python-side per
    // signal source (synthetic / onset / ibi / mfcc_continuity / mert /
    // muq) and points TC at it for the expressivity sweep.
    const double* extra1_per_pair = nullptr;
    int           extra1_n        = 0;  // size of the square matrix dimension

};

// Outputs bundle. Caller-owned (not reused across calls).
struct TransitionCostResult
{
    // W[i * n_beats + j] = cost of transition i → j. INF = blocked.
    // Sequential W[i, i+1] filled with the reduced-cost rule.
    std::vector<double> W;

    // chroma_D[i * n_beats + j] = row-shifted cosine distance in [0, 2].
    // Row-shifted means D[i, j] = 1 - cos(chroma[i+1], chroma[j]) for
    // i < n_beats - 1; last row filled with 1.0.
    std::vector<double> chroma_D;

    // importance[i] = baseline weight per beat (0.3 outro / 1.0 chorus / 0.5 default).
    std::vector<double> importance;

    // Sparse candidate map. Iteration order is lexicographic by (from, to).
    std::map<std::pair<int, int>, TransitionCandidate> candidates;

    int n_beats = 0;
};

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
TransitionCostResult computeTransitionCosts(const TransitionCostInputs& inputs);

// ---------------------------------------------------------------------------
// Exposed helpers (for parity test bisection)
// ---------------------------------------------------------------------------

// Per-beat importance from segment labels + beat times. Returns size
// n_beats; defaults to 0.5 when no segment contains the beat.
// Port of `transition_cost.py::compute_importance` (L106-122).
std::vector<double> computeImportance(int                       n_beats,
                                      const analysis::Segment*  segments,
                                      int                       n_segments,
                                      const double*             beat_times);

// Row-shifted chroma distance matrix. chroma_D[i, j] = 1 - cos(normed
// chroma[i+1], normed chroma[j]) for i < n-1; last row = 1.0.
// Port of `transition_cost.py::compute_chroma_distance` (L147-160).
std::vector<double> computeChromaDistance(const float* features,
                                          int          n_beats,
                                          int          n_features);

} // namespace reamix::remix
