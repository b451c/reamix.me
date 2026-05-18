// test_harmonic_vs_timbre_blend — sesja 92 (ADR-080 RESCOPE + ADR-083).
//
// Self-validation parity test for the Tone slider blend logic in
// `reamix::remix::computeQualityScore` plus the per-pair full-mix chroma
// continuity composition consumed by TransitionCost / RegionCost /
// BlockAssembly.
//
// CANONICAL SOURCE: Quality.cpp::computeQualityScore Tone blend block
// (post-timbre-score, pre-return). NO Python reference (per ADR-065 +
// memory `feedback_python_no_longer_source_of_truth.md`).
//
// FORMULA (ADR-084 sesja 93 — replaced sesja-92 convex blend to fix
// mid-α inertness on harmonic-monotone tracks):
//   timbre_coef  = 1 - 0.7·α
//   chroma_coef  = √α
//   final_score  = clamp(timbre_coef · timbre + chroma_coef · chroma, 0, 1)
//
// Test asserts:
//   1. Default weights.harmonic_vs_timbre = 0.0 → blend bypassed → bit-
//      exact baseline (regression guard against silent default change).
//   2. Pure chroma at slider=1.0: timbre_coef=0.3 + chroma_coef=1.0 →
//      0.3·0.7 + 1.0·0.85 = 1.06 → clamp = 1.0.
//   3. Balanced 0.5 blend: timbre_coef=0.65 + chroma_coef=√0.5 →
//      0.65·0.6 + √0.5·0.4 (~0.6728).
//   4. nullopt chroma input bypasses blend even with slider > 0.
//   5. Boundary clamping (out-of-range slider values).
//   6. computeChromaContinuityMatrix produces expected similarity matrix
//      on hand-computed chord-progression fixture.
//
// Per memory `feedback_python_no_longer_source_of_truth.md`: parity tests
// for new cost components validate C++ against itself, not vs Python.

#include "remix/Quality.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <optional>
#include <vector>

using reamix::remix::QualityInputs;
using reamix::remix::QualityWeights;
using reamix::remix::computeChromaContinuityMatrix;
using reamix::remix::computeQualityScore;
using reamix::remix::kDefaultQualityWeights;
using reamix::remix::kLegacyQualityWeights;

namespace {

bool nearly_equal(double a, double b, double eps = 1e-12) {
    return std::abs(a - b) <= eps;
}

// Helper to build minimal QualityInputs with all required fields zeroed.
QualityInputs makeMinimalInputs() {
    QualityInputs qi{};
    qi.successor_sim     = 0.0;
    qi.context_sim       = 0.0;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;
    return qi;
}

// ---------------------------------------------------------------------------
// Test 1 — Default weights.harmonic_vs_timbre = 0.0 → blend bypassed.
//
// Even when chroma input is provided, blend should NOT activate at the
// default 0.0 slider value. This is the bit-exact baseline regression
// guard — kDefaultQualityWeights MUST remain bit-exact equivalent to
// the pre-sesja-92 production simplex behavior.
// ---------------------------------------------------------------------------
bool test_default_bypass() {
    QualityInputs qi = makeMinimalInputs();
    qi.successor_sim          = 0.7;
    qi.full_mix_chroma_continuity = 0.4;  // Provide chroma; should be IGNORED.

    QualityWeights w = kLegacyQualityWeights;  // legacy: harmonic_vs_timbre = 0.0
    w.successor       = 1.0;  // single-signal isolation
    // (Other 9 weights stay 0 → only successor contributes; sum invariant
    //  of kLegacyQualityWeights breaks here but parity test path doesn't
    //  reuse the constant — we override fields explicitly.)
    w.waveform = 0.0; w.edge_splice = 0.0; w.context = 0.0;
    w.label = 0.0; w.bar_align = 0.0; w.section = 0.0;
    w.energy = 0.0; w.edge_energy = 0.0; w.centroid = 0.0;

    const double q = computeQualityScore(qi, w);
    // Expected: weighted_sum = 1.0 * 0.7 = 0.7. Blend bypassed
    // (harmonic_vs_timbre = 0.0). Result = 0.7 (NOT blended with chroma=0.4).
    if (! nearly_equal(q, 0.7)) {
        std::fprintf(stderr, "[FAIL] default bypass: q=%.20g, expected 0.7\n", q);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — Pure chroma at slider = 1.0 (post ADR-084 damped curve).
//
// With harmonic_vs_timbre = 1.0:
//   timbre_coef = 1 - 0.7·1 = 0.3
//   chroma_coef = √1 = 1.0
//   blend = 0.3 · timbre + 1.0 · chroma = 0.3·0.7 + 1.0·0.85 = 1.06
//   clamp(1.06, 0, 1) = 1.0
// Note: sesja-92 convex blend returned 0.85 here. Post-ADR-084 damped+sqrt
// curve clamps to 1.0 at α=1 because timbre never fully zeros (0.3 floor).
// ---------------------------------------------------------------------------
bool test_pure_chroma() {
    QualityInputs qi = makeMinimalInputs();
    qi.successor_sim                 = 0.7;
    qi.full_mix_chroma_continuity    = 0.85;

    QualityWeights w{};
    w.successor          = 1.0;
    w.harmonic_vs_timbre = 1.0;

    const double q = computeQualityScore(qi, w);
    // Expected: 0.3·0.7 + 1.0·0.85 = 1.06 → clamp = 1.0.
    if (! nearly_equal(q, 1.0)) {
        std::fprintf(stderr, "[FAIL] pure chroma (damped curve): q=%.20g, expected 1.0\n", q);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — Balanced 0.5 blend (post ADR-084 damped curve).
//
// With harmonic_vs_timbre = 0.5:
//   timbre_coef = 1 - 0.7·0.5 = 0.65
//   chroma_coef = √0.5 ≈ 0.7071067811865476
//   blend = 0.65·0.6 + √0.5·0.4 = 0.39 + 0.28284271247461903 = 0.67284271247461903
// Note: sesja-92 convex blend returned 0.5; post-ADR-084 ~0.673 because
// chroma already contributes √0.5 ≈ 0.71 instead of 0.5 at mid α (this
// is the formula change that fixes mid-α inertness on monotone-harmony tracks).
// ---------------------------------------------------------------------------
bool test_balanced_blend() {
    QualityInputs qi = makeMinimalInputs();
    qi.successor_sim                 = 0.6;
    qi.full_mix_chroma_continuity    = 0.4;

    QualityWeights w{};
    w.successor          = 1.0;
    w.harmonic_vs_timbre = 0.5;

    const double q = computeQualityScore(qi, w);
    const double expected = 0.65 * 0.6 + std::sqrt(0.5) * 0.4;
    if (! nearly_equal(q, expected)) {
        std::fprintf(stderr, "[FAIL] balanced blend (damped curve): q=%.20g, expected %.20g\n",
                     q, expected);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — nullopt chroma input bypasses blend.
//
// When upstream did NOT populate full_mix_chroma_continuity (chroma slice
// empty), blend MUST bypass even with slider > 0. Returns timbre score.
// ---------------------------------------------------------------------------
bool test_nullopt_chroma_bypass() {
    QualityInputs qi = makeMinimalInputs();
    qi.successor_sim = 0.7;
    // qi.full_mix_chroma_continuity stays nullopt by default.

    QualityWeights w{};
    w.successor          = 1.0;
    w.harmonic_vs_timbre = 0.8;  // slider engaged BUT no chroma input.

    const double q = computeQualityScore(qi, w);
    // Expected: blend bypassed (chroma nullopt), q = timbre = 0.7.
    if (! nearly_equal(q, 0.7)) {
        std::fprintf(stderr, "[FAIL] nullopt chroma bypass: q=%.20g, expected 0.7\n", q);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 5 — Boundary clamping on slider value.
//
// computeQualityScore clamps weights.harmonic_vs_timbre to [0, 1] internally
// via std::clamp. Out-of-range values must NOT produce undefined behavior.
// ---------------------------------------------------------------------------
bool test_boundary_clamping() {
    QualityInputs qi = makeMinimalInputs();
    qi.successor_sim                 = 0.6;
    qi.full_mix_chroma_continuity    = 0.4;

    QualityWeights w{};
    w.successor = 1.0;

    // Above-range slider: clamp to 1.0 → damped curve at α=1.
    // 0.3 · timbre + 1.0 · chroma = 0.3·0.6 + 1.0·0.4 = 0.18 + 0.4 = 0.58.
    // sesja-92 convex blend returned 0.4; post-ADR-084 returns 0.58.
    w.harmonic_vs_timbre = 1.5;
    double q1 = computeQualityScore(qi, w);
    if (! nearly_equal(q1, 0.58)) {
        std::fprintf(stderr, "[FAIL] above-range clamp (damped): q=%.20g, expected 0.58\n", q1);
        return false;
    }

    // Below-range slider — the bypass guard triggers FIRST (>= 0.0 check)
    // and returns timbre score directly without clamping. Negative slider
    // values therefore behave identically to slider = 0.0 → bypass.
    w.harmonic_vs_timbre = -0.5;
    double q2 = computeQualityScore(qi, w);
    if (! nearly_equal(q2, 0.6)) {
        std::fprintf(stderr, "[FAIL] below-range bypass: q=%.20g, expected 0.6 (= timbre)\n", q2);
        return false;
    }

    // Chroma value above 1.0: clamp to 1.0.
    w.harmonic_vs_timbre = 1.0;
    qi.full_mix_chroma_continuity = 1.5;
    double q3 = computeQualityScore(qi, w);
    if (! nearly_equal(q3, 1.0)) {
        std::fprintf(stderr, "[FAIL] chroma above-range clamp: q=%.20g, expected 1.0\n", q3);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 6 — computeChromaContinuityMatrix on hand-computed fixture.
//
// 4-beat synthetic chroma where beats 0, 1 share chord (C major: pitch
// classes [0, 4, 7]), beats 2, 3 share chord (G major: pitch classes
// [7, 11, 2]). Expected: high similarity within chord, low across.
// ---------------------------------------------------------------------------
bool test_chroma_continuity_matrix() {
    // 4 beats × 12 chroma dims tightly packed.
    const int n_beats  = 4;
    const int n_chroma = 12;
    std::vector<float> chroma(static_cast<std::size_t>(n_beats) * n_chroma, 0.0f);

    // Beat 0: C major triad — [0, 4, 7] active.
    chroma[0 * 12 + 0] = 1.0f; chroma[0 * 12 + 4] = 1.0f; chroma[0 * 12 + 7] = 1.0f;
    // Beat 1: also C major — same vector.
    chroma[1 * 12 + 0] = 1.0f; chroma[1 * 12 + 4] = 1.0f; chroma[1 * 12 + 7] = 1.0f;
    // Beat 2: G major triad — [7, 11, 2] active.
    chroma[2 * 12 + 7] = 1.0f; chroma[2 * 12 + 11] = 1.0f; chroma[2 * 12 + 2] = 1.0f;
    // Beat 3: also G major.
    chroma[3 * 12 + 7] = 1.0f; chroma[3 * 12 + 11] = 1.0f; chroma[3 * 12 + 2] = 1.0f;

    const auto m = computeChromaContinuityMatrix(chroma.data(), n_beats, n_chroma);
    if ((int) m.size() != n_beats * n_beats) {
        std::fprintf(stderr, "[FAIL] matrix size %d != %d\n", (int) m.size(), n_beats * n_beats);
        return false;
    }

    // Diagonal = 1.0 (cosine of identical vectors).
    for (int i = 0; i < n_beats; ++i) {
        const double d = m[(std::size_t) i * n_beats + i];
        if (! nearly_equal(d, 1.0)) {
            std::fprintf(stderr, "[FAIL] diagonal m[%d,%d]=%.20g\n", i, i, d);
            return false;
        }
    }

    // Within-chord pair (0,1) → 1.0 (identical chroma).
    const double m01 = m[(std::size_t) 0 * n_beats + 1];
    if (! nearly_equal(m01, 1.0)) {
        std::fprintf(stderr, "[FAIL] within-chord m[0,1]=%.20g, expected 1.0\n", m01);
        return false;
    }

    // Within-chord pair (2,3) → 1.0.
    const double m23 = m[(std::size_t) 2 * n_beats + 3];
    if (! nearly_equal(m23, 1.0)) {
        std::fprintf(stderr, "[FAIL] within-chord m[2,3]=%.20g, expected 1.0\n", m23);
        return false;
    }

    // Cross-chord pair (0,2): one shared pitch class (7) over 3 active each.
    // cos = 1 / 3 = 0.333... within numerical tolerance.
    const double m02 = m[(std::size_t) 0 * n_beats + 2];
    if (! nearly_equal(m02, 1.0 / 3.0, 1e-6)) {
        std::fprintf(stderr, "[FAIL] cross-chord m[0,2]=%.20g, expected 1/3\n", m02);
        return false;
    }

    // Symmetry: m[i,j] == m[j,i].
    for (int i = 0; i < n_beats; ++i) {
        for (int j = 0; j < n_beats; ++j) {
            const double a = m[(std::size_t) i * n_beats + j];
            const double b = m[(std::size_t) j * n_beats + i];
            if (! nearly_equal(a, b)) {
                std::fprintf(stderr, "[FAIL] asymmetry m[%d,%d]=%.20g vs m[%d,%d]=%.20g\n",
                             i, j, a, j, i, b);
                return false;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 7 — kDefaultQualityWeights bit-exact preservation.
//
// kDefaultQualityWeights.harmonic_vs_timbre MUST = 0.0 (production default
// preserves bit-exact baseline when slider untouched). Compile-time check
// would be cleanest, but constexpr float comparison is best done at runtime.
// ---------------------------------------------------------------------------
bool test_default_weights_invariant() {
    if (kDefaultQualityWeights.harmonic_vs_timbre != 0.0) {
        std::fprintf(stderr, "[FAIL] kDefaultQualityWeights.harmonic_vs_timbre=%.20g, expected 0.0\n",
                     kDefaultQualityWeights.harmonic_vs_timbre);
        return false;
    }
    if (kLegacyQualityWeights.harmonic_vs_timbre != 0.0) {
        std::fprintf(stderr, "[FAIL] kLegacyQualityWeights.harmonic_vs_timbre=%.20g, expected 0.0\n",
                     kLegacyQualityWeights.harmonic_vs_timbre);
        return false;
    }
    return true;
}

} // namespace

int main()
{
    std::printf("test_harmonic_vs_timbre_blend — sesja 92 ADR-080 RESCOPE + ADR-083 self-validation\n");

    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"default_bypass_at_slider_0",     test_default_bypass},
        {"pure_chroma_at_slider_1",        test_pure_chroma},
        {"balanced_blend_at_slider_0_5",   test_balanced_blend},
        {"nullopt_chroma_bypass",          test_nullopt_chroma_bypass},
        {"boundary_clamping",              test_boundary_clamping},
        {"chroma_continuity_matrix_chord", test_chroma_continuity_matrix},
        {"default_weights_invariant",      test_default_weights_invariant},
    };

    int pass = 0, fail = 0;
    for (const auto& c : cases) {
        const bool ok = c.fn();
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (ok) ++pass; else ++fail;
    }

    std::printf("\nSummary: %d/%d passed\n", pass, pass + fail);
    return fail == 0 ? 0 : 1;
}
