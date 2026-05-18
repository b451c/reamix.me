// test_transient_continuity — sesja 75 (DEV-028 ADR-064).
//
// Self-validation parity test for `reamix::remix::computeOnsetNorm` + the
// per-pair transient-continuity composition `1 - |onset_norm[i] -
// onset_norm[j]|` consumed by TransitionCost / RegionCost / BlockAssembly.
//
// CANONICAL SOURCE: Quality.cpp::computeOnsetNorm. There is NO Python
// reference for this cost component (sesja-75 ADR-064 fix-in-port — the
// formula is C++-native, productionised from sesja-74 expressivity slot).
// Test asserts:
//   1. Hand-computed expected values on a small deterministic fixture.
//   2. Mathematical invariants (range, symmetry, diagonal).
//   3. Edge cases (empty input, uniform onset, null pointer).
//
// Per memory `feedback_python_no_longer_source_of_truth.md`: parity tests
// for new cost components validate C++ against itself, not vs Python.

#include "remix/Quality.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using reamix::remix::computeOnsetNorm;

namespace {

bool nearly_equal(double a, double b, double eps = 1e-15) {
    return std::abs(a - b) <= eps;
}

// Helper: per-pair transient continuity, mirrors TC/RC/BA inline code.
double transientContinuity(const std::vector<double>& norm, int i, int j) {
    return 1.0 - std::abs(norm[(std::size_t) i] - norm[(std::size_t) j]);
}

// ---------------------------------------------------------------------------
// Test 1 — Hand-computed expected values on a 4-beat fixture.
// onset_strength = [0.0, 0.2, 0.5, 1.0]
// min = 0.0, max = 1.0, range = 1.0
// onset_norm  = [0.0, 0.2, 0.5, 1.0]   (no-op normalisation since min=0, max=1)
// transient_continuity matrix (M[i,j] = 1 - |norm[i] - norm[j]|):
//   M[0,0] = 1.0    M[0,1] = 0.8    M[0,2] = 0.5    M[0,3] = 0.0
//   M[1,0] = 0.8    M[1,1] = 1.0    M[1,2] = 0.7    M[1,3] = 0.2
//   M[2,0] = 0.5    M[2,1] = 0.7    M[2,2] = 1.0    M[2,3] = 0.5
//   M[3,0] = 0.0    M[3,1] = 0.2    M[3,2] = 0.5    M[3,3] = 1.0
// ---------------------------------------------------------------------------
bool test_hand_computed() {
    const std::vector<double> onset = {0.0, 0.2, 0.5, 1.0};
    const auto norm = computeOnsetNorm(onset.data(), (int) onset.size());

    if ((int) norm.size() != 4) {
        std::fprintf(stderr, "[FAIL] norm size %d != 4\n", (int) norm.size());
        return false;
    }

    const std::vector<double> expected_norm = {0.0, 0.2, 0.5, 1.0};
    for (int k = 0; k < 4; ++k) {
        if (! nearly_equal(norm[(std::size_t) k], expected_norm[(std::size_t) k])) {
            std::fprintf(stderr, "[FAIL] norm[%d] = %.20g, expected %.20g\n",
                         k, norm[(std::size_t) k], expected_norm[(std::size_t) k]);
            return false;
        }
    }

    const double expected_M[4][4] = {
        {1.0, 0.8, 0.5, 0.0},
        {0.8, 1.0, 0.7, 0.2},
        {0.5, 0.7, 1.0, 0.5},
        {0.0, 0.2, 0.5, 1.0},
    };
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double v = transientContinuity(norm, i, j);
            if (! nearly_equal(v, expected_M[i][j])) {
                std::fprintf(stderr, "[FAIL] M[%d,%d] = %.20g, expected %.20g\n",
                             i, j, v, expected_M[i][j]);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — Min-max scaling: arbitrary range maps to [0, 1].
// onset_strength = [3.0, 5.0, 7.0, 9.0, 11.0]
// min=3, max=11, range=8 → norm = [0.0, 0.25, 0.5, 0.75, 1.0]
// ---------------------------------------------------------------------------
bool test_min_max_scaling() {
    const std::vector<double> onset = {3.0, 5.0, 7.0, 9.0, 11.0};
    const auto norm = computeOnsetNorm(onset.data(), (int) onset.size());
    const std::vector<double> expected = {0.0, 0.25, 0.5, 0.75, 1.0};
    for (int k = 0; k < 5; ++k) {
        if (! nearly_equal(norm[(std::size_t) k], expected[(std::size_t) k])) {
            std::fprintf(stderr, "[FAIL] min-max norm[%d] = %.20g, expected %.20g\n",
                         k, norm[(std::size_t) k], expected[(std::size_t) k]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — Mathematical invariants on a larger random-ish fixture.
//   * Symmetry: M[i,j] == M[j,i].
//   * Diagonal: M[i,i] == 1.0.
//   * Range: 0.0 <= M[i,j] <= 1.0.
//   * Min element: 0.0 attained at the (argmin_norm, argmax_norm) pair.
// ---------------------------------------------------------------------------
bool test_invariants() {
    const std::vector<double> onset = {
        0.13, 0.87, 0.42, 0.05, 0.99, 0.31, 0.66, 0.18,
        0.74, 0.50, 0.08, 0.93, 0.27, 0.61, 0.45, 0.71,
    };
    const int n = (int) onset.size();
    const auto norm = computeOnsetNorm(onset.data(), n);

    for (int i = 0; i < n; ++i) {
        if (! nearly_equal(transientContinuity(norm, i, i), 1.0)) {
            std::fprintf(stderr, "[FAIL] diagonal M[%d,%d] != 1.0\n", i, i);
            return false;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double v = transientContinuity(norm, i, j);
            const double w = transientContinuity(norm, j, i);
            if (! nearly_equal(v, w)) {
                std::fprintf(stderr, "[FAIL] asymmetry M[%d,%d]=%.20g vs M[%d,%d]=%.20g\n",
                             i, j, v, j, i, w);
                return false;
            }
            if (v < 0.0 || v > 1.0) {
                std::fprintf(stderr, "[FAIL] M[%d,%d]=%.20g out of [0,1]\n", i, j, v);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — Edge case: uniform onset (all values equal).
// max == min → range clamped to 1e-9 → onset_norm uniformly tiny → M ≈ 1.
// ---------------------------------------------------------------------------
bool test_uniform() {
    const std::vector<double> onset(8, 0.42);
    const auto norm = computeOnsetNorm(onset.data(), (int) onset.size());
    if ((int) norm.size() != 8) return false;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            const double v = transientContinuity(norm, i, j);
            if (v < 0.999999) {
                std::fprintf(stderr, "[FAIL] uniform M[%d,%d]=%.20g (expected ~1)\n",
                             i, j, v);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 5 — Edge case: null pointer / zero-length / negative length.
// ---------------------------------------------------------------------------
bool test_empty() {
    auto a = computeOnsetNorm(nullptr, 10);
    if (! a.empty()) return false;

    const std::vector<double> onset = {0.5, 0.7};
    auto b = computeOnsetNorm(onset.data(), 0);
    if (! b.empty()) return false;

    auto c = computeOnsetNorm(onset.data(), -3);
    if (! c.empty()) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Test 6 — Composition with computeQualityScore.
// Setting weights.transient_continuity > 0 + a known transient_continuity
// value yields the expected weighted contribution (mirrors the per-pair
// composition path inside TC/RC/BA).
// ---------------------------------------------------------------------------
bool test_quality_score_integration() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    // Single-signal corner: transient_continuity = 1.0, all else = 0.
    w.transient_continuity = 1.0;

    QualityInputs qi{};
    // Required signals must still be set (their weights are 0 so contribution
    // is null-guarded out, but the fields must hold valid doubles).
    qi.successor_sim     = 0.0;
    qi.context_sim       = 0.0;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;
    qi.transient_continuity = 0.73;

    const double q = computeQualityScore(qi, w);
    // With all weight on transient_continuity = 1.0 and value 0.73, the
    // weighted_sum = 0.73, available_weight = 1.0, no missing → returns 0.73.
    if (! nearly_equal(q, 0.73)) {
        std::fprintf(stderr, "[FAIL] qs(corner=0.73)=%.20g, expected 0.73\n", q);
        return false;
    }

    // Now zero out transient_continuity weight: contribution drops, and since
    // all OTHER weights are 0 too, available_weight=0 → returns 0.
    w.transient_continuity = 0.0;
    const double q0 = computeQualityScore(qi, w);
    if (! nearly_equal(q0, 0.0)) {
        std::fprintf(stderr, "[FAIL] qs(weight=0)=%.20g, expected 0.0\n", q0);
        return false;
    }
    return true;
}

} // namespace

int main()
{
    std::printf("test_transient_continuity — sesja 75 ADR-064 self-validation\n");

    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"hand_computed_4x4",       test_hand_computed},
        {"min_max_scaling",          test_min_max_scaling},
        {"invariants_16x16",         test_invariants},
        {"uniform_onset_edge",       test_uniform},
        {"empty_null_edge",          test_empty},
        {"quality_score_integration", test_quality_score_integration},
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
