#pragma once

#include <optional>
#include <string_view>
#include <vector>

namespace reamix::remix {

// Quality weights, thresholds, and score computation for remix transitions.
//
// Port of references/python-source/remix/quality.py (201 LOC, 2026-04-21).
// Single source of truth for the 10-signal weighted quality model used by
// TransitionCost (clean method), the forthcoming ViterbiDP cost accumulator,
// and the phase-6 UI quality-gate coloring. No other module should define
// quality weights — per Python docstring L3-5.
//
// --- Sources of truth and audit status ------------------------------------
// Canonical weights block: `quality.py:22-31` (2026-04-21). The docstring in
// `transition_cost.py:1-22` lists a STALE copy of these weights (waveform=0.22,
// sum=1.04); per research-scout 2026-04-21 + Hard Rule #6 in CLAUDE.md, the
// drift is flagged (weights-audit.md) and NOT fixed in the Python reference.
// Port comments cite `quality.py L22-31 (2026-04-21)` as canonical. Never
// `transition_cost.py:1-22`.
//
// 21 hand-calibrated numbers in this file are ported VERBATIM per ADR-026.
// Audit classification (CLEAN / SUSPECT / UNJUSTIFIED) lives in
// `phases/phase-4-remix-engine/weights-audit.md` and is updated during
// `TransitionCost` parity investigation. Evidence chain feeds the optional
// perceptual listening harness + pre-ship weight calibration (ROADMAP L209+).
//
// W_SUCCESSOR (0.14) + W_CONTEXT (0.18) = 0.32 of simplex. Python's header
// comment cites "ICASSP 2023 beat-continuity finding" as grounding; that
// citation is NOT independently verifiable (research-scout 2026-04-21, dblp
// + targeted search). Flagged SUSPECT in weights-audit.md with HIGH priority
// for the perceptual harness. Port constants verbatim; DO NOT copy the
// phantom ICASSP attribution into the comment below.

// ---------------------------------------------------------------------------
// Quality weights (sum = 1.0 within floating-point tolerance, see static_assert)
// ---------------------------------------------------------------------------

// Audio waveform cross-correlation at splice boundary.
// Source-of-truth `quality.py:22`.
inline constexpr double W_WAVEFORM = 0.34;

// Row-shifted cosine: "does destination j sound like source i+1?"
// Perceptual motivation: beat-continuity dominates (internal listening
// tests per RemixTool diagnostics, citation TBD; flagged SUSPECT in
// weights-audit.md — research-scout 2026-04-21).
// Source-of-truth `quality.py:23`.
inline constexpr double W_SUCCESSOR = 0.14;

// Boundary-edge feature similarity at splice point.
// Source-of-truth `quality.py:24`.
inline constexpr double W_EDGE_SPLICE = 0.12;

// 2-beat context-window continuity (Wenner 2013 homogeneity analogue).
// Same SUSPECT flag as W_SUCCESSOR (see above block comment).
// Source-of-truth `quality.py:25`.
inline constexpr double W_CONTEXT = 0.18;

// Same segment-label bonus (e.g. chorus→chorus, verse→verse).
// Source-of-truth `quality.py:26`.
inline constexpr double W_LABEL = 0.08;

// Pre-downbeat → downbeat alignment bonus.
// Source-of-truth `quality.py:27`.
inline constexpr double W_BAR_ALIGN = 0.06;

// Segment-similarity-matrix cell at (from_section, to_section).
// Source-of-truth `quality.py:28`.
inline constexpr double W_SECTION = 0.04;

// Whole-beat RMS continuity.
// Source-of-truth `quality.py:29`.
inline constexpr double W_ENERGY = 0.02;

// Edge-RMS continuity at splice boundary.
// Source-of-truth `quality.py:30`.
inline constexpr double W_EDGE_ENERGY = 0.01;

// Spectral-centroid (brightness) continuity.
// Source-of-truth `quality.py:31`.
inline constexpr double W_CENTROID = 0.01;

// Compile-time sum check. IEEE-754 double summation of these specific
// constants is not bit-exactly 1.0 (accumulated rounding), so a tight
// tolerance is used rather than `== 1.0`. Deviation beyond 1e-12 indicates
// a transcription error during port maintenance.
inline constexpr double kQualityWeightsSum =
    W_WAVEFORM + W_SUCCESSOR + W_EDGE_SPLICE + W_CONTEXT + W_LABEL +
    W_BAR_ALIGN + W_SECTION + W_ENERGY + W_EDGE_ENERGY + W_CENTROID;
static_assert(kQualityWeightsSum > 1.0 - 1e-12 &&
              kQualityWeightsSum < 1.0 + 1e-12,
              "Quality weights must sum to 1.0 (see quality.py L17).");

// ---------------------------------------------------------------------------
// QualityWeights struct — runtime-overridable container of the 10 W_* weights.
//
// Added sesja 71 (DEV-028 calibration umbrella, ADR-058) to enable Bayesian
// optimization sweeps over weight space WITHOUT recompiling the plugin.
// `kDefaultQualityWeights` matches the verbatim Python values above bit-for-
// bit; `computeQualityScore(inputs)` defaults to it, preserving parity tests.
//
// Field order mirrors `quality.py:22-31` declaration order, NOT the
// accumulation order used inside `computeQualityScore` (the function uses
// fields by name, so struct order is for designated-init readability only).
//
// Calibration callers (tools/calibration_harness, tools/calibration_orchestrator)
// build a non-default `QualityWeights` and pass it via the Pipeline
// `qualityWeightsOverride` field, which propagates to TC / RC / BA Inputs
// `qualityWeights` pointer.
// ---------------------------------------------------------------------------
struct QualityWeights
{
    double waveform;     // W_WAVEFORM      0.34
    double successor;    // W_SUCCESSOR     0.14
    double edge_splice;  // W_EDGE_SPLICE   0.12
    double context;      // W_CONTEXT       0.18
    double label;        // W_LABEL         0.08
    double bar_align;    // W_BAR_ALIGN     0.06
    double section;      // W_SECTION       0.04
    double energy;       // W_ENERGY        0.02
    double edge_energy;  // W_EDGE_ENERGY   0.01
    double centroid;     // W_CENTROID      0.01

    // ---- 11th cost component: transient continuity (sesja 75, ADR-064) ----
    // Per-junction similarity of normalised onset strength: M[i,j] = 1 -
    // |onset_norm[i] - onset_norm[j]|. Default 0.0 → no contribution → bit-
    // exact parity preserved against pre-sesja-75 baseline. Calibration
    // (sesja 76+) sets non-zero; first ADR-063 candidate productionised.
    // Wired across all three modes (TransitionCost, RegionCost, BlockAssembly).
    double transient_continuity = 0.0;

    // ---- 12th cost component: MFCC spectral continuity (sesja 77, ADR-066) ----
    // Per-junction MFCC + delta-MFCC L2 similarity at splice boundary, per
    // Stylianou-Syrdal 2001 + Vepa-King 2002. Default 0.0 → no contribution →
    // bit-exact parity preserved. Sesja-77 expressivity check 5/5 PASS on
    // tiesto/meshuggah/dance_monkey/wardruna/alice (ADR-063 Candidate B).
    // Wired across all three modes (TransitionCost, RegionCost, BlockAssembly).
    double mfcc_continuity = 0.0;

    // ---- Generic scratch slot for future ADR-063 candidate testing -------
    // Used by calibration_harness for expressivity sweeps of C (MERT),
    // D (MuQ), F (LPC-formant) and other future candidates before
    // productionisation. When a candidate is productionised it gets a named
    // field (like `transient_continuity` and `mfcc_continuity` above) and
    // `extra1` is freed for the next test. Default 0.0 → no contribution.
    double extra1 = 0.0;

    // ---- 13th cost component: vocal continuity (sesja 98, ADR-088) ------
    // Per-junction continuity of vocal phrase boundaries:
    //   M[i,j] = 1 - max(edgeVocalReleaseEnd[i], edgeVocalOnsetStart[j])
    // High value = clean splice (no vocal phrase straddled). Low value =
    // splice cuts mid-word (source has vocal release pending OR destination
    // has vocal onset starting). Default 0.0 → no contribution → bit-exact
    // parity preserved against pre-sesja-98 baseline. Wired across all three
    // modes (TransitionCost, RegionCost, BlockAssembly).
    //
    // Surfaced sesja-98 mid-impl audit: edgeVocalOnsetStart + edgeVocalReleaseEnd
    // were produced + cached + Python-parity-tested but never consumed by
    // any cost component. ADR-088 closes the gap. Self-validated by
    // tests/parity/test_vocal_continuity.cpp (no Python ground truth, per
    // ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`).
    double vocal_continuity = 0.0;

    // ---- D4 collapse: sequential_continuity (sesja 81, ADR-068 D4) -------
    // Per-junction mean of the three sequential-cosine components measured
    // ρ=0.67-0.71 redundant in sesja-80 D1 correlation matrix:
    //   seq = mean( successor_sim, context_sim, mfcc_continuity? )
    // The mfcc_continuity term is excluded from the average when nullopt.
    // Productionisation per ADR-068 D4: this single weight replaces the
    // collapsed triplet (`.successor`, `.context`, `.mfcc_continuity` are
    // all set to 0.0 in the new production simplex; only this aggregate
    // weight contributes to the sequential dimension). Default 0.0 →
    // no contribution → bit-exact parity preserved against pre-sesja-81
    // baseline (kLegacyQualityWeights also has 0.0 here).
    double sequential_continuity = 0.0;

    // ---- D2 harmonic mean (sesja 81, ADR-068 D2) -------------------------
    // When true, computeQualityScore returns:
    //   Q = available_weight / Σ_i ( wᵢ / max(qᵢ, kHarmonicEpsilon) )
    // instead of the standard arithmetic weighted mean. The harmonic mean is
    // dominated by the WEAKEST cost component, so a single low-quality
    // signal punishes the composite proportionally more (vs arithmetic
    // where weak signals can be masked by strong ones). Per sesja-78 ADR-068
    // hypothesis: better robustness against single-axis splice weaknesses.
    // Default false → bit-exact parity preserved (kLegacyQualityWeights
    // keeps it false). Productionisation via kProductionQualityWeights or
    // the post-D3-flip kDefaultQualityWeights setting it true.
    bool use_harmonic_mean = false;

    // ---- ADR-080 Tone slider — blend coefficient (sesja 92) --------------
    // BLEND COEFFICIENT in [0.0, 1.0], NOT a weight. NOT summed in
    // `kDefaultQualityWeightsSum` invariant (which constrains the timbre
    // block weights to sum to 1.0). Blends timbre-block score (current
    // 7-component simplex) with full-mix chroma similarity per:
    //   final_score = (1 - tone) × timbre_score + tone × chroma_sim
    // Default 0.0 → blend bypassed → bit-exact baseline preservation
    // (current production behavior unchanged when slider untouched).
    // User-controlled via AuditionBar Tone slider (ADR-083 host component).
    // Slider range 0.0-1.0, default thumb at left (Timbre 100%); user
    // explicitly drags right to enable harmonic blend per ADR-080 RESCOPE
    // STATUS UPDATE 1 (sesja 92).
    //
    // Per-pair `qi.full_mix_chroma_continuity` (QualityInputs, std::optional)
    // is the chroma similarity value sliced from `feat.features[40..51]` and
    // computed via `computeChromaContinuityMatrix`. When nullopt → blend
    // bypassed (chroma input not provided by upstream).
    //
    // CANONICAL DEFINITION (no Python reference): per ADR-080 RESCOPE +
    // ADR-083 (sesja 92). Self-validated by
    // `tests/parity/test_harmonic_vs_timbre_blend.cpp`.
    double harmonic_vs_timbre = 0.0;
};

// Harmonic-mean denominator floor — guards `wᵢ / qᵢ` against q≈0 producing
// catastrophically large reciprocals. Tuned per ADR-068 § D2 to ε=1e-3 so a
// single-component q=0 contributes ~1000×wᵢ to the inverse-sum (effectively
// zeroing the composite without producing NaN/Inf).
inline constexpr double kHarmonicMeanEpsilon = 1e-3;

// Default weights — SESJA 81 production simplex (ADR-068 D1+D2+D4 integrated).
//
// Per sesja-80 D1 correlation matrix evidence:
//   - 3 components measured CONSTANT-ZERO across 31 562 surviving pairs:
//     edge_splice (W=0.12), label (W=0.08), section (W=0.04). Total 0.24
//     of legacy simplex was multiplying zero in current production
//     (cache-state issue: warm_cache populates edge_features=empty,
//     beat_labels=all-unknown, segments=empty). Their default weights
//     drop to 0.0; code paths preserved for future cache fix.
//   - Continuity triplet (successor, context, mfcc_continuity) measured
//     ρ=0.67-0.71 redundant. Collapsed via ADR-068 D4 into single
//     `sequential_continuity` weight. Legacy weights stay zero in default;
//     collapse path sums them and divides by available count.
//   - Remaining 6 orthogonal components (|ρ| < 0.4) get redistributed
//     weight share. Final 6-D production simplex sums to 1.0:
//     waveform 0.50 + sequential_continuity 0.20 + transient_continuity
//     0.15 + energy 0.07 + edge_energy 0.04 + centroid 0.02 +
//     bar_align 0.02 = 1.00.
//   - use_harmonic_mean = true per ADR-068 D2: weakest cost component
//     dominates composite, harmonising more punishing of single-axis
//     splice weakness.
//
// Listening A/B (sesja 81 close, references/listening/dev-028/sesja-81/)
// validates audible quality vs kLegacyQualityWeights baseline. Phase-4
// Python-parity tests pin to kLegacyQualityWeights explicitly to remain
// bit-exact past this default flip. New self-validation parity test
// `tests/parity/test_sequential_continuity.cpp` exercises the D4 collapse +
// D2 harmonic branches (no Python ground truth, per ADR-065).
inline constexpr QualityWeights kDefaultQualityWeights{
    /* waveform              */ 0.50,  // sesja-72 dominant audibility signal; redistributed up from 0.34
    /* successor             */ 0.0,   // collapsed into sequential_continuity (ADR-068 D4)
    /* edge_splice           */ 0.0,   // dead-zero in current cache (sesja-80 D1)
    /* context               */ 0.0,   // collapsed into sequential_continuity (ADR-068 D4)
    /* label                 */ 0.0,   // dead-zero in current cache (sesja-80 D1)
    /* bar_align             */ 0.02,  // reduced from 0.06 per sesja-80 redistribution
    /* section               */ 0.0,   // dead-zero in current cache (sesja-80 D1)
    /* energy                */ 0.07,  // increased from 0.02 per sesja-80 redistribution
    /* edge_energy           */ 0.04,  // increased from 0.01 per sesja-80 redistribution
    /* centroid              */ 0.02,  // increased from 0.01 per sesja-80 redistribution
    /* transient_continuity  */ 0.15,  // ADR-064 productionised at 0.15 (was 0 default)
    /* mfcc_continuity       */ 0.0,   // collapsed into sequential_continuity (ADR-068 D4)
    /* extra1                */ 0.0,   // sesja 74 scratch slot for future candidates
    /* vocal_continuity      */ 0.0,   // ADR-088 sesja 98 — default 0.0 preserves bit-exact baseline; calibration sets >0
    /* sequential_continuity */ 0.20,  // ADR-068 D4 collapse — replaces successor + context + mfcc_continuity triplet
    /* use_harmonic_mean     */ true,  // ADR-068 D2 — weakest-component-punishing composite
    /* harmonic_vs_timbre    */ 0.0    // ADR-080 RESCOPE + ADR-083 (sesja 92) — blend coefficient, default 0.0 = pure timbre = bit-exact baseline
};

// Compile-time sum check on the struct. New slots default to 0.0 → sum stays
// bit-exactly equal to the legacy 10-field sum. (use_harmonic_mean is a bool
// flag, not a weight, so it is excluded from the sum.)
inline constexpr double kDefaultQualityWeightsSum =
    kDefaultQualityWeights.waveform    + kDefaultQualityWeights.successor +
    kDefaultQualityWeights.edge_splice + kDefaultQualityWeights.context +
    kDefaultQualityWeights.label       + kDefaultQualityWeights.bar_align +
    kDefaultQualityWeights.section     + kDefaultQualityWeights.energy +
    kDefaultQualityWeights.edge_energy + kDefaultQualityWeights.centroid +
    kDefaultQualityWeights.transient_continuity +
    kDefaultQualityWeights.mfcc_continuity +
    kDefaultQualityWeights.extra1 +
    kDefaultQualityWeights.vocal_continuity +
    kDefaultQualityWeights.sequential_continuity;
static_assert(kDefaultQualityWeightsSum > 1.0 - 1e-12 &&
              kDefaultQualityWeightsSum < 1.0 + 1e-12,
              "kDefaultQualityWeights fields must sum to 1.0.");

// ---------------------------------------------------------------------------
// Legacy 10-component Python-bit-exact simplex (sesja 81, ADR-068).
//
// Frozen reference for phase-4 parity tests that validate the C++ port
// against Python ground truth. Identical to kDefaultQualityWeights at commit
// 1; will diverge in a later commit when kDefaultQualityWeights flips to the
// new 6-D production simplex (D1 redistribution + D4 collapse).
//
// Phase-4 parity tests that go through computeQualityScore (directly or via
// TC/RC/BA) MUST pass `kLegacyQualityWeights` explicitly to remain Python-
// bit-exact past the production-default flip. The new test_sequential_
// continuity.cpp (sesja 81) self-validates the D4 collapse formula on
// hand-computed fixtures (no Python ground truth) per ADR-065.
// ---------------------------------------------------------------------------
inline constexpr QualityWeights kLegacyQualityWeights{
    /* waveform              */ W_WAVEFORM,
    /* successor             */ W_SUCCESSOR,
    /* edge_splice           */ W_EDGE_SPLICE,
    /* context               */ W_CONTEXT,
    /* label                 */ W_LABEL,
    /* bar_align             */ W_BAR_ALIGN,
    /* section               */ W_SECTION,
    /* energy                */ W_ENERGY,
    /* edge_energy           */ W_EDGE_ENERGY,
    /* centroid              */ W_CENTROID,
    /* transient_continuity  */ 0.0,
    /* mfcc_continuity       */ 0.0,
    /* extra1                */ 0.0,
    /* vocal_continuity      */ 0.0,  // ADR-088 sesja 98 — legacy parity preserved (no contribution)
    /* sequential_continuity */ 0.0,
    /* use_harmonic_mean     */ false,
    /* harmonic_vs_timbre    */ 0.0  // ADR-080 RESCOPE — phase-4 parity tests pin to legacy explicitly; blend bypassed
};

inline constexpr double kLegacyQualityWeightsSum =
    kLegacyQualityWeights.waveform    + kLegacyQualityWeights.successor +
    kLegacyQualityWeights.edge_splice + kLegacyQualityWeights.context +
    kLegacyQualityWeights.label       + kLegacyQualityWeights.bar_align +
    kLegacyQualityWeights.section     + kLegacyQualityWeights.energy +
    kLegacyQualityWeights.edge_energy + kLegacyQualityWeights.centroid +
    kLegacyQualityWeights.transient_continuity +
    kLegacyQualityWeights.mfcc_continuity +
    kLegacyQualityWeights.extra1 +
    kLegacyQualityWeights.vocal_continuity +
    kLegacyQualityWeights.sequential_continuity;
static_assert(kLegacyQualityWeightsSum > 1.0 - 1e-12 &&
              kLegacyQualityWeightsSum < 1.0 + 1e-12,
              "kLegacyQualityWeights fields must sum to 1.0.");

// ---------------------------------------------------------------------------
// Display thresholds (UI quality-gate coloring, phase-6 consumer).
// Source-of-truth `quality.py:36-38`.
// ---------------------------------------------------------------------------
inline constexpr double QUALITY_GREAT      = 0.7;
inline constexpr double QUALITY_GOOD       = 0.5;

// Transitions with composite score < this are HARD-BLOCKED (INF cost).
// Also referenced by TransitionCost (QUALITY_HARD_FLOOR hard gate).
//
// SESJA 81 ADR-068 D3: lowered from 0.45 → 0.20 to soften the hard-floor gate.
// Sesja-80 D1 measurement showed all 11/11 corpus tracks bit-perfectly hit
// `min_quality = 0.450` — i.e. the 0.45 floor was the sole reason no lower-
// quality candidate survived. Combined with D2 harmonic mean (which already
// punishes weak signals composite-wide), the floor's job becomes "block
// catastrophic-only" rather than "block 50%-quality". 0.20 retains the gate
// against truly degenerate splices (< 1/5 composite quality) while letting
// the harmonic mean + sequential_continuity gating do the perceptual work.
inline constexpr double QUALITY_HARD_FLOOR = 0.20;

// Legacy 0.45 floor — frozen for phase-4 parity tests (test_transition_cost
// goldens were generated with the Python production floor 0.45). Pin to this
// constant explicitly in tests; production (RemixPipeline → AnalyzePipeline →
// computeTransitionCosts) picks up `QUALITY_HARD_FLOOR = 0.20` via the
// TransitionCostInputs::quality_floor field default initializer.
inline constexpr double LEGACY_QUALITY_HARD_FLOOR = 0.45;

// ---------------------------------------------------------------------------
// Span penalty constants (additive, not multiplicative).
// Applied by Optimizer / ViterbiDP against the composite score.
// Source-of-truth `quality.py:43-45`.
// ---------------------------------------------------------------------------
inline constexpr double SPAN_PENALTY_CROSS_SECTION = 0.25;
inline constexpr double SPAN_PENALTY_SAME_SECTION  = 0.12;
inline constexpr int    SPAN_PENALTY_MAX_BEATS     = 32;

// ---------------------------------------------------------------------------
// Vocal splice penalty — see compute_vocal_penalty below.
// Source-of-truth `quality.py:50-52`.
// ---------------------------------------------------------------------------
inline constexpr double VOCAL_CONTINUITY_PENALTY = 0.10;
inline constexpr double VOCAL_SPLICE_THRESHOLD   = 0.40;
inline constexpr double VOCAL_SPLICE_PENALTY     = 0.30;

// ---------------------------------------------------------------------------
// Onset-sustain penalty — see compute_onset_penalty below.
// Source-of-truth `quality.py:58-59`.
// ---------------------------------------------------------------------------
inline constexpr double ONSET_SUSTAIN_PENALTY   = 0.12;
inline constexpr double ONSET_SUSTAIN_THRESHOLD = 0.25;

// ---------------------------------------------------------------------------
// Quality score inputs — 10 signal similarities in [0, 1].
//
// Field order mirrors `quality.py:62-73` keyword-only signature. Waveform
// and edge_splice are optional (None in Python = not computed in "fast"
// analysis mode); missing signals have their weight redistributed
// proportionally across available signals, matching Python L84-120.
// ---------------------------------------------------------------------------
struct QualityInputs
{
    std::optional<double> waveform_sim;     // optional — null in fast mode
    double                successor_sim;     // required
    std::optional<double> edge_splice_sim;   // optional — null in fast mode
    double                context_sim;       // required
    double                label_match;       // required
    double                section_sim;       // required
    double                bar_aligned;       // required
    double                energy_match;      // required
    double                edge_energy_match; // required
    double                centroid_match;    // required

    // ---- 11th slot — transient continuity (sesja 75 ADR-064) -------------
    // Optional: nullopt when onset_strength was not provided to the upstream
    // mode (TransitionCost / RegionCost / BlockAssembly all populate this
    // when their respective onset_strength input is non-null). nullopt + the
    // matching `weights.transient_continuity` weight redistribute via the
    // standard missing-signal path in computeQualityScore.
    std::optional<double> transient_continuity;

    // ---- 12th slot — MFCC spectral continuity (sesja 77 ADR-066) ---------
    // Optional: nullopt when MFCC features were not provided. TC/RC/BA
    // populate this from the pre-computed n×n matrix produced by
    // computeMfccContinuityMatrix. nullopt + matching weight redistribute
    // via the standard missing-signal path.
    std::optional<double> mfcc_continuity;

    // ---- Generic scratch slot (sesja 74) ---------------------------------
    // Default nullopt → no contribution. Calibration harness fills via
    // TC/RC/BA `extra1_per_pair` matrix during expressivity sweeps.
    std::optional<double> extra1_value;

    // ---- 13th slot — vocal phrase continuity (sesja 98 ADR-088) ----------
    // Per-pair (i→j): quality = 1 - max(release_at_i, onset_at_j) where
    //   release_at_i = edgeVocalReleaseEnd[i]   (source has vocal release pending)
    //   onset_at_j   = edgeVocalOnsetStart[j]   (destination has vocal onset starting)
    // High = clean splice (no vocal phrase straddled). Low = mid-word cut.
    // nullopt when input pointers unavailable → missing-signal redistribute.
    std::optional<double> vocal_continuity;

    // ---- ADR-080 Tone slider — full-mix chroma similarity (sesja 92) -----
    // Per-pair full-mix chroma cosine similarity in [0, 1]. When provided,
    // computeQualityScore blends timbre block score with chroma per the
    // `weights.harmonic_vs_timbre` slider (ADR-080 RESCOPE + ADR-083).
    //
    // Populated by TC/RC/BA per-pair scoring loop from chroma slice
    // `feat.features[i*nFeat + 40..51]` via `computeChromaContinuityMatrix`
    // (Quality.h:467) called once per analyze.
    //
    // nullopt → blend bypassed (chroma input not available); computeQuality
    // returns timbre-only score (bit-exact baseline). Default 0.0 weight on
    // `harmonic_vs_timbre` ALSO bypasses the blend; both nullopt-on-input
    // AND zero-weight-on-blend short-circuit to bit-exact baseline.
    std::optional<double> full_mix_chroma_continuity;

};

// ---------------------------------------------------------------------------
// Onset normalisation helper (sesja 75 ADR-064).
//
// Min-max normalises onset_strength to [0, 1] over the full beat range.
// Returns an n-length vector; consumers compose per-pair transient
// continuity inline as:
//
//     qi.transient_continuity = 1.0 - std::abs(onset_norm[i] - onset_norm[j]);
//
// Edge case: when max == min (uniform onset, degenerate input), the
// normalisation denominator is clamped to 1e-9 to avoid divide-by-zero;
// the resulting onset_norm is uniformly close to 0, so transient_continuity
// reduces to ~1.0 across all pairs (no-information signal — Viterbi will
// not pick splices on this dimension for such tracks). Empty input returns
// an empty vector.
//
// Caller-owned input pointer (caller guarantees length >= n_beats).
//
// CANONICAL DEFINITION (no Python reference): formula is C++-native, port
// extension per ADR-064 (DEV-028 first cost-component productionised).
// Bit-exact tested by `tests/parity/test_transient_continuity.cpp`.
// ---------------------------------------------------------------------------
std::vector<double> computeOnsetNorm(const double* onset_strength, int n_beats);

// ---------------------------------------------------------------------------
// MFCC spectral-continuity matrix helper (sesja 77 ADR-066).
//
// Produces an n_beats × n_beats row-major similarity matrix for use as the
// `mfcc_continuity` cost component. Per Stylianou-Syrdal 2001 + Vepa-King
// 2002, MFCC L2 + delta-MFCC L2 outperform LPC-formant for human-perceived
// concatenation-discontinuity. Formula (canonical, C++-native; mirrors
// calibration_harness::computeExtra1Matrix signal=mfcc_continuity branch):
//
//   K        = min(13, n_features)             // first 13 dims = MFCC
//   normRef  = sqrt(K)
//   d_static = sqrt(sum_k (f[i,k] - f[j,k])^2) / normRef
//   ip       = max(0, i-1)
//   jn       = min(n-1, j+1)
//   d_dyn    = sqrt(sum_k ((f[i,k]-f[ip,k]) - (f[jn,k]-f[j,k]))^2) / normRef
//   raw      = clamp((0.7 * d_static + 0.3 * d_dyn) / 0.6, 0, 1)
//   M[i,j]   = 1 - raw                         // similarity in [0, 1]
//
// Edge cases:
//   - features == nullptr / n_beats <= 0 / n_features <= 0 → empty vector.
//   - K <= 0 (no MFCC bands available) → empty vector.
//
// CANONICAL DEFINITION (no Python reference): per ADR-066 (sesja 77).
// Full-band MFCC is L2-normalized by `FeatureExtractor.cpp:329-340` BEFORE
// this function sees it, so the formula's saturation 0.6 produces a useful
// spread on bounded inputs.
//
// Bit-exact tested by `tests/parity/test_mfcc_continuity.cpp`.
// ---------------------------------------------------------------------------
inline constexpr double MFCC_CONTINUITY_STATIC_RATIO  = 0.7;
inline constexpr double MFCC_CONTINUITY_DYNAMIC_RATIO = 0.3;
inline constexpr double MFCC_CONTINUITY_SATURATION    = 0.6;
inline constexpr int    MFCC_CONTINUITY_N_BANDS_MAX   = 13;

std::vector<double> computeMfccContinuityMatrix(const float* features,
                                                int          n_beats,
                                                int          n_features);

// ---------------------------------------------------------------------------
// Chroma continuity matrix helper.
//
// Produces an n_beats × n_beats row-major chroma cosine similarity matrix.
// Formula (canonical, C++-native):
//
//   c_i, c_j = chroma vectors at beats i, j (kNChroma = 12 dims).
//   |c_i| = sqrt(sum_k c_i[k]²)         // L2 norm of L∞-normalised chroma
//   cos_ij = sum_k(c_i[k] * c_j[k]) / max(|c_i| * |c_j|, ε)
//   M[i,j] = clamp(cos_ij, 0, 1)        // similarity in [0, 1]
//
// Edge cases:
//   - features == nullptr / n_beats <= 0 / n_chroma <= 0 → empty vector.
//   - zero-norm chroma vector (silent frame) → similarity 0 with anything.
//
// CANONICAL DEFINITION (no Python reference): introduced as the chroma
// kernel underlying ADR-080 PROPOSED Audition Timbre/Harmonic slider.
// ---------------------------------------------------------------------------
std::vector<double> computeChromaContinuityMatrix(const float* features,
                                                  int          n_beats,
                                                  int          n_chroma);

// Compute composite quality score in [0, 1], higher = better transition.
//
// Port of `quality.py::compute_quality_score` (L62-123). Implements dynamic
// weight redistribution: when optional signals are missing, their weight is
// re-allocated proportionally across available signals (prevents pessimistic
// NEUTRAL=0.5 default from the legacy pre-2026 path). Accumulation order
// matches Python exactly (waveform, edge_splice, then the 8 always-available
// signals in Python's tuple order) for bitwise parity.
//
// The `weights` parameter defaults to `kDefaultQualityWeights` (bit-exact
// match of the W_* constants), preserving parity tests + production behavior.
// Calibration callers pass a non-default `QualityWeights` to sweep the
// weight simplex during DEV-028 Bayesian optimization (ADR-058).
double computeQualityScore(const QualityInputs& inputs,
                           const QualityWeights& weights = kDefaultQualityWeights);

// Compute additive vocal penalty (subtract from composite score).
//
// Port of `quality.py::compute_vocal_penalty` (L126-159). Two components:
//   (1) continuity: penalize abrupt vocal-level changes (voice <-> silence);
//   (2) splice-through: penalize cutting through active vocals (both sides
//       high = mid-word risk). Uses edge-resolution vocal features when
//       available, falls back to beat-level vocal activity.
//
// edge_va_end / edge_va_start: optional (std::nullopt = beat-level fallback).
// Returns value >= 0 to SUBTRACT from quality score.
double computeVocalPenalty(double va_source,
                           double va_dest,
                           std::optional<double> edge_va_end   = std::nullopt,
                           std::optional<double> edge_va_start = std::nullopt);

// Compute additive onset-sustain penalty (subtract from composite score).
//
// Port of `quality.py::compute_onset_penalty` (L162-181). Only the
// destination onset is considered (where the listener "arrives").
// std::nullopt destination = no penalty (onset_dest unknown).
double computeOnsetPenalty(std::optional<double> onset_dest = std::nullopt);

// Human-readable label for a quality score. "Great" / "Good" / "Weak".
// Port of `quality.py::quality_label` (L184-190).
std::string_view qualityLabel(double score);

// Config color key for a quality score. "good" / "medium" / "bad" / "unknown".
// Port of `quality.py::quality_color_key` (L193-201).
std::string_view qualityColorKey(std::optional<double> score);

} // namespace reamix::remix
