// test_mfcc_continuity — sesja 77 (DEV-028 ADR-066).
//
// Self-validation parity test for `reamix::remix::computeMfccContinuityMatrix`.
//
// CANONICAL SOURCE: Quality.cpp::computeMfccContinuityMatrix. There is NO
// Python reference for this cost component (sesja-77 ADR-066 — formula is
// C++-native, productionised from the sesja-74 expressivity slot after
// sesja-77 PASS evidence on 5 cells: tiesto, meshuggah, dance_monkey,
// wardruna, alice_in_chains).
//
// Test asserts:
//   1. Hand-computed expected values on a small 4-beat × K=2 fixture.
//   2. Uniform features → similarity ≡ 1.0 (no signal).
//   3. Range invariant: 0 ≤ M[i,j] ≤ 1 across a larger fixture.
//   4. Boundary behaviour at i=0 (ip clamps) and j=n-1 (jn clamps).
//   5. Edge cases: null pointer, n_beats ≤ 0, n_features ≤ 0.
//   6. Composition with computeQualityScore (corner weight = 1.0).
//
// Per memory `feedback_python_no_longer_source_of_truth.md`: parity tests
// for new cost components validate C++ against itself, not vs Python.

#include "remix/Quality.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using reamix::remix::computeMfccContinuityMatrix;
using reamix::remix::MFCC_CONTINUITY_DYNAMIC_RATIO;
using reamix::remix::MFCC_CONTINUITY_SATURATION;

namespace {

bool nearly_equal(double a, double b, double eps = 1e-12) {
    return std::abs(a - b) <= eps;
}

double mfccAt(const std::vector<double>& m, int n, int i, int j) {
    return m[(std::size_t) i * (std::size_t) n + (std::size_t) j];
}

// ---------------------------------------------------------------------------
// Test 1 — Hand-computed expected values on a 4-beat × K=2 fixture.
// features (row-major float32):
//   f[0] = [0, 0]
//   f[1] = [1, 0]
//   f[2] = [0, 1]
//   f[3] = [1, 1]
// normRef = sqrt(2). Saturation = 0.6. Ratios = 0.7 / 0.3.
//
// Verify M[0,1] (boundary i=0 → ip=0; j=1 → jn=2):
//   d_static = sqrt((0-1)^2 + (0-0)^2) / sqrt(2) = 1/sqrt(2)
//   dl = f[0] - f[0] = [0, 0]
//   dr = f[2] - f[1] = [-1, 1]
//   dd = dl - dr = [1, -1]
//   d_dyn = sqrt(1 + 1) / sqrt(2) = 1.0
//   raw = 0.7/sqrt(2) + 0.3 = 0.4949747... + 0.3 = 0.7949747...
//   raw / 0.6 = 1.32495... → clamped to 1.0
//   M[0,1] = 0.0
//
// Verify M[1,2] (interior i=1 → ip=0; j=2 → jn=3):
//   d_static = sqrt((1-0)^2 + (0-1)^2) / sqrt(2) = sqrt(2)/sqrt(2) = 1.0
//   dl = f[1] - f[0] = [1, 0]
//   dr = f[3] - f[2] = [1, 0]
//   dd = [0, 0]
//   d_dyn = 0
//   raw = 0.7. raw / 0.6 = 1.166... → clamped to 1.0. M[1,2] = 0.0
//
// Verify M[1,1] (diagonal interior; d_static=0, dl=[1,0], dr=f[2]-f[1]=[-1,1]):
//   dd = dl - dr = [2, -1]; d_dyn_sq = 5; d_dyn = sqrt(5)/sqrt(2) = sqrt(2.5)
//   raw = 0.3 * sqrt(2.5) = 0.4743... ; raw/0.6 = 0.7905... ;
//   M[1,1] = 1 - 0.7905... = 0.2094...
// ---------------------------------------------------------------------------
bool test_hand_computed() {
    const int n_beats    = 4;
    const int n_features = 2;
    const float feat[8] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };

    const auto m = computeMfccContinuityMatrix(feat, n_beats, n_features);
    if ((int) m.size() != n_beats * n_beats) {
        std::fprintf(stderr, "[FAIL] matrix size %d != %d\n",
                     (int) m.size(), n_beats * n_beats);
        return false;
    }

    // M[0,1] = 0.0 (saturated)
    {
        const double v = mfccAt(m, n_beats, 0, 1);
        if (! nearly_equal(v, 0.0)) {
            std::fprintf(stderr, "[FAIL] M[0,1]=%.20g, expected 0.0\n", v);
            return false;
        }
    }

    // M[1,2] = 0.0 (saturated)
    {
        const double v = mfccAt(m, n_beats, 1, 2);
        if (! nearly_equal(v, 0.0)) {
            std::fprintf(stderr, "[FAIL] M[1,2]=%.20g, expected 0.0\n", v);
            return false;
        }
    }

    // M[1,1]: hand-computed
    {
        const double sqrt2  = std::sqrt(2.0);
        const double sqrt5  = std::sqrt(5.0);
        const double d_dyn  = sqrt5 / sqrt2;            // = sqrt(2.5)
        const double raw    = MFCC_CONTINUITY_DYNAMIC_RATIO * d_dyn;
        const double scaled = std::min(1.0, raw / MFCC_CONTINUITY_SATURATION);
        const double expected = 1.0 - scaled;
        const double v = mfccAt(m, n_beats, 1, 1);
        if (! nearly_equal(v, expected, 1e-13)) {
            std::fprintf(stderr, "[FAIL] M[1,1]=%.20g, expected %.20g\n",
                         v, expected);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — Uniform features → all M[i,j] == 1.0 (no spectral discontinuity).
// ---------------------------------------------------------------------------
bool test_uniform_features() {
    const int n_beats    = 5;
    const int n_features = 4;
    std::vector<float> feat(n_beats * n_features, 0.5f);
    const auto m = computeMfccContinuityMatrix(feat.data(), n_beats, n_features);
    if ((int) m.size() != n_beats * n_beats) return false;
    for (int i = 0; i < n_beats; ++i) {
        for (int j = 0; j < n_beats; ++j) {
            const double v = mfccAt(m, n_beats, i, j);
            if (! nearly_equal(v, 1.0)) {
                std::fprintf(stderr, "[FAIL] uniform M[%d,%d]=%.20g, expected 1.0\n",
                             i, j, v);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — Range invariant on a larger 8-beat × K=13 fixture (exercises
// the n_features ≥ 13 path that production runs hit).
// ---------------------------------------------------------------------------
bool test_range_invariant() {
    const int n_beats    = 8;
    const int n_features = 16;       // ≥ MFCC_CONTINUITY_N_BANDS_MAX (13)
    std::vector<float> feat((std::size_t) n_beats * n_features);
    // Deterministic pseudo-random pattern.
    for (std::size_t k = 0; k < feat.size(); ++k) {
        feat[k] = static_cast<float>(std::sin(0.37 * (double) k)
                                   + std::cos(0.91 * (double) k));
    }
    const auto m = computeMfccContinuityMatrix(feat.data(), n_beats, n_features);
    for (int i = 0; i < n_beats; ++i) {
        for (int j = 0; j < n_beats; ++j) {
            const double v = mfccAt(m, n_beats, i, j);
            if (v < 0.0 || v > 1.0) {
                std::fprintf(stderr, "[FAIL] M[%d,%d]=%.20g out of [0,1]\n",
                             i, j, v);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — Boundary clamps: ip = max(0, i-1), jn = min(n-1, j+1).
// At i=0: ip=0 (no previous beat). At j=n-1: jn=n-1 (no next beat).
// Hand-verify M[0, n-1] uses ip=0 + jn=n-1.
// ---------------------------------------------------------------------------
bool test_boundary_clamps() {
    const int n_beats    = 3;
    const int n_features = 1;
    const float feat[3] = {0.0f, 1.0f, 0.0f};

    const auto m = computeMfccContinuityMatrix(feat, n_beats, n_features);

    // (i=0, j=2): ip=0, jn=2 (already at boundary).
    //   d_static = |0 - 0| / sqrt(1) = 0
    //   dl = f[0] - f[0] = 0
    //   dr = f[2] - f[2] = 0
    //   dd = 0; d_dyn = 0
    //   raw = 0; M[0,2] = 1.0
    {
        const double v = mfccAt(m, n_beats, 0, 2);
        if (! nearly_equal(v, 1.0)) {
            std::fprintf(stderr, "[FAIL] M[0,2]=%.20g, expected 1.0\n", v);
            return false;
        }
    }

    // (i=2, j=0): ip=1, jn=1.
    //   d_static = |0 - 0| / 1 = 0
    //   dl = f[2] - f[1] = -1
    //   dr = f[1] - f[0] = 1
    //   dd = -2; d_dyn = sqrt(4) / 1 = 2
    //   raw = 0.3 * 2 = 0.6; raw/0.6 = 1.0; M = 0.0
    {
        const double v = mfccAt(m, n_beats, 2, 0);
        if (! nearly_equal(v, 0.0)) {
            std::fprintf(stderr, "[FAIL] M[2,0]=%.20g, expected 0.0\n", v);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 5 — Edge cases: null pointer, n_beats ≤ 0, n_features ≤ 0.
// ---------------------------------------------------------------------------
bool test_empty_null_edge() {
    auto a = computeMfccContinuityMatrix(nullptr, 4, 13);
    if (! a.empty()) return false;

    const float feat[2] = {0.0f, 1.0f};
    auto b = computeMfccContinuityMatrix(feat, 0, 2);
    if (! b.empty()) return false;

    auto c = computeMfccContinuityMatrix(feat, 2, 0);
    if (! c.empty()) return false;

    auto d = computeMfccContinuityMatrix(feat, -3, 2);
    if (! d.empty()) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Test 6 — Composition with computeQualityScore.
// Single-signal corner: weights.mfcc_continuity = 1.0, mfcc_continuity = 0.42
// → weighted_sum = 0.42, available_weight = 1.0 → returns 0.42.
// ---------------------------------------------------------------------------
bool test_quality_score_integration() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.mfcc_continuity = 1.0;

    QualityInputs qi{};
    qi.successor_sim     = 0.0;
    qi.context_sim       = 0.0;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;
    qi.mfcc_continuity   = 0.42;

    const double q = computeQualityScore(qi, w);
    if (! nearly_equal(q, 0.42)) {
        std::fprintf(stderr, "[FAIL] qs(corner=0.42)=%.20g, expected 0.42\n", q);
        return false;
    }

    // Zero out weight: contribution drops; all other weights are 0 → returns 0.
    w.mfcc_continuity = 0.0;
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
    std::printf("test_mfcc_continuity — sesja 77 ADR-066 self-validation\n");

    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"hand_computed_4x4_K2",      test_hand_computed},
        {"uniform_features",          test_uniform_features},
        {"range_invariant_8x8_K13",   test_range_invariant},
        {"boundary_clamps",           test_boundary_clamps},
        {"empty_null_edge",           test_empty_null_edge},
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
