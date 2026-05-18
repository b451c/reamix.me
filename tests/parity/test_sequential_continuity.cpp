// test_sequential_continuity — sesja 81 (DEV-028 ADR-068 D4 collapse + D2 harmonic).
//
// Self-validation parity test for two new branches in
// `reamix::remix::computeQualityScore`:
//   (D4) sequential_continuity collapse — per sesja-80 D1 correlation matrix,
//        three components measured ρ=0.67-0.71 redundant; ADR-068 D4 collapses
//        them into a single weight = mean of available constituents.
//   (D2) harmonic-mean composition — gated by `weights.use_harmonic_mean`;
//        Q = available_weight / Σ(wᵢ / max(qᵢ, ε)). ADR-068 D2.
//
// CANONICAL SOURCE: Quality.cpp::computeQualityScore § sequential_continuity
// branch + harmonic-mean branch. There is NO Python reference for either
// formula. Per memory `feedback_python_no_longer_source_of_truth.md` and
// ADR-065, parity tests for new C++-canonical components self-validate
// against hand-computed expected values + mathematical invariants.
//
// Test asserts (D4):
//   1. Hand-computed mean over all three constituents (mfcc_continuity present).
//   2. Hand-computed mean over two constituents (mfcc_continuity nullopt).
//   3. Weight = 0 → no contribution (legacy bit-exact path).
//   4. Corner composition: only sequential_continuity weight = 1.0.
//   5. Mixed composition: sequential_continuity coexists with other weights.
//   6. Mathematical invariant: collapse value ≡ direct mean.
// Test asserts (D2):
//   7. Harmonic mean of two equal-weight signals matches hand-computed.
//   8. Harmonic punishes weak signals more than arithmetic (relative ordering).
//   9. Harmonic falls back to 0 when no signal contributes.
//  10. Epsilon floor prevents NaN on q=0 input.

#include "remix/Quality.h"

#include <cmath>
#include <cstdio>
#include <optional>

namespace {

bool nearly_equal(double a, double b, double eps = 1e-15) {
    return std::abs(a - b) <= eps;
}

// ---------------------------------------------------------------------------
// Test 1 — Hand-computed mean over all three constituents.
// successor=0.6, context=0.8, mfcc_continuity=0.4
// expected seq = (0.6 + 0.8 + 0.4) / 3 = 0.6
// With weight = 1.0 and all other weights 0 → q = 0.6.
// ---------------------------------------------------------------------------
bool test_three_constituents() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.sequential_continuity = 1.0;

    QualityInputs qi{};
    qi.successor_sim       = 0.6;
    qi.context_sim         = 0.8;
    qi.mfcc_continuity     = 0.4;
    qi.label_match         = 0.0;
    qi.section_sim         = 0.0;
    qi.bar_aligned         = 0.0;
    qi.energy_match        = 0.0;
    qi.edge_energy_match   = 0.0;
    qi.centroid_match      = 0.0;

    const double q   = computeQualityScore(qi, w);
    const double exp = (0.6 + 0.8 + 0.4) / 3.0;
    if (! nearly_equal(q, exp)) {
        std::fprintf(stderr, "[FAIL] three_constituents q=%.20g expected=%.20g\n", q, exp);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — Hand-computed mean over two constituents (mfcc_continuity nullopt).
// successor=0.5, context=0.9, mfcc_continuity=nullopt
// expected seq = (0.5 + 0.9) / 2 = 0.7
// ---------------------------------------------------------------------------
bool test_two_constituents() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.sequential_continuity = 1.0;

    QualityInputs qi{};
    qi.successor_sim       = 0.5;
    qi.context_sim         = 0.9;
    qi.mfcc_continuity     = std::nullopt;
    qi.label_match         = 0.0;
    qi.section_sim         = 0.0;
    qi.bar_aligned         = 0.0;
    qi.energy_match        = 0.0;
    qi.edge_energy_match   = 0.0;
    qi.centroid_match      = 0.0;

    const double q   = computeQualityScore(qi, w);
    const double exp = (0.5 + 0.9) / 2.0;
    if (! nearly_equal(q, exp)) {
        std::fprintf(stderr, "[FAIL] two_constituents q=%.20g expected=%.20g\n", q, exp);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — Weight = 0 → no contribution → legacy bit-exact path.
// With sequential_continuity weight = 0 and a non-zero successor weight,
// the result must equal what the per-component path would have produced.
// ---------------------------------------------------------------------------
bool test_weight_zero_no_contribution() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.sequential_continuity = 0.0;
    w.successor             = 1.0;  // single per-component weight

    QualityInputs qi{};
    qi.successor_sim       = 0.42;
    qi.context_sim         = 0.99;  // would shift seq mean; must NOT contribute
    qi.mfcc_continuity     = 0.99;
    qi.label_match         = 0.0;
    qi.section_sim         = 0.0;
    qi.bar_aligned         = 0.0;
    qi.energy_match        = 0.0;
    qi.edge_energy_match   = 0.0;
    qi.centroid_match      = 0.0;

    const double q = computeQualityScore(qi, w);
    if (! nearly_equal(q, 0.42)) {
        std::fprintf(stderr, "[FAIL] weight_zero q=%.20g expected=0.42 (collapse must NOT fire)\n", q);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — Mixed composition: sequential_continuity coexists with other weights.
// weights: waveform=0.5, sequential_continuity=0.5
// successor=0.4, context=0.6, mfcc_continuity=0.8 → seq = 0.6
// waveform_sim = 1.0
// q = 0.5 * 1.0 + 0.5 * 0.6 = 0.5 + 0.3 = 0.8
// ---------------------------------------------------------------------------
bool test_mixed_composition() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.waveform              = 0.5;
    w.sequential_continuity = 0.5;

    QualityInputs qi{};
    qi.waveform_sim        = 1.0;
    qi.successor_sim       = 0.4;
    qi.context_sim         = 0.6;
    qi.mfcc_continuity     = 0.8;
    qi.label_match         = 0.0;
    qi.section_sim         = 0.0;
    qi.bar_aligned         = 0.0;
    qi.energy_match        = 0.0;
    qi.edge_energy_match   = 0.0;
    qi.centroid_match      = 0.0;

    const double q   = computeQualityScore(qi, w);
    const double seq = (0.4 + 0.6 + 0.8) / 3.0;
    const double exp = 0.5 * 1.0 + 0.5 * seq;
    if (! nearly_equal(q, exp)) {
        std::fprintf(stderr, "[FAIL] mixed q=%.20g expected=%.20g\n", q, exp);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 5 — Mathematical invariant: identical constituents → mean = constituent.
// successor=context=mfcc=0.55 → seq = 0.55
// ---------------------------------------------------------------------------
bool test_identical_constituents() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.sequential_continuity = 1.0;

    QualityInputs qi{};
    qi.successor_sim       = 0.55;
    qi.context_sim         = 0.55;
    qi.mfcc_continuity     = 0.55;
    qi.label_match         = 0.0;
    qi.section_sim         = 0.0;
    qi.bar_aligned         = 0.0;
    qi.energy_match        = 0.0;
    qi.edge_energy_match   = 0.0;
    qi.centroid_match      = 0.0;

    const double q = computeQualityScore(qi, w);
    if (! nearly_equal(q, 0.55)) {
        std::fprintf(stderr, "[FAIL] identical q=%.20g expected=0.55\n", q);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 6 — Legacy weights → no sequential_continuity contribution → behavior
// matches kLegacyQualityWeights's per-component path. This guards against
// silent bit-exact regression (legacy parity tests rely on this property).
// ---------------------------------------------------------------------------
bool test_legacy_weights_unchanged() {
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;
    using reamix::remix::kLegacyQualityWeights;

    QualityInputs qi{};
    qi.waveform_sim        = 1.0;
    qi.successor_sim       = 0.0;  // would have contributed via collapse if seq weight > 0
    qi.context_sim         = 0.0;
    qi.mfcc_continuity     = 0.0;
    qi.edge_splice_sim     = 1.0;
    qi.label_match         = 1.0;
    qi.section_sim         = 1.0;
    qi.bar_aligned         = 1.0;
    qi.energy_match        = 1.0;
    qi.edge_energy_match   = 1.0;
    qi.centroid_match      = 1.0;

    const double q = computeQualityScore(qi, kLegacyQualityWeights);

    // Hand-computed: legacy weights = waveform 0.34 + successor 0.14 +
    // edge_splice 0.12 + context 0.18 + label 0.08 + bar_align 0.06 +
    // section 0.04 + energy 0.02 + edge_energy 0.01 + centroid 0.01.
    // Values: w(1.0) + s(0.0) + e(1.0) + c(0.0) + l(1.0) + b(1.0) +
    //         sec(1.0) + en(1.0) + ee(1.0) + cn(1.0)
    //       = 0.34 + 0.0 + 0.12 + 0.0 + 0.08 + 0.06 + 0.04 + 0.02 + 0.01 + 0.01
    //       = 0.68
    const double exp = 0.34 + 0.12 + 0.08 + 0.06 + 0.04 + 0.02 + 0.01 + 0.01;
    if (! nearly_equal(q, exp, 1e-12)) {
        std::fprintf(stderr, "[FAIL] legacy q=%.20g expected=%.20g\n", q, exp);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 7 — D2 harmonic mean of two equal-weight signals.
// weights: waveform=0.5, successor=0.5, use_harmonic_mean=true
// values:  waveform_sim=1.0, successor_sim=0.5
// expected: Q = (0.5+0.5) / (0.5/1.0 + 0.5/0.5) = 1.0 / (0.5 + 1.0) = 0.6667
// arithmetic counterpart: 0.5*1.0 + 0.5*0.5 = 0.75 (would be HIGHER).
// ---------------------------------------------------------------------------
bool test_harmonic_two_signals() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.waveform          = 0.5;
    w.successor         = 0.5;
    w.use_harmonic_mean = true;

    QualityInputs qi{};
    qi.waveform_sim      = 1.0;
    qi.successor_sim     = 0.5;
    qi.context_sim       = 0.0;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;

    const double q   = computeQualityScore(qi, w);
    const double exp = 1.0 / (0.5 / 1.0 + 0.5 / 0.5);
    if (! nearly_equal(q, exp, 1e-12)) {
        std::fprintf(stderr, "[FAIL] harmonic_two_signals q=%.20g expected=%.20g\n", q, exp);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 8 — D2 harmonic punishes weak signals more than arithmetic.
// Same weights/inputs, run both modes; harmonic must be < arithmetic when
// signals are unequal.
// ---------------------------------------------------------------------------
bool test_harmonic_punishes_weak() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityInputs qi{};
    qi.waveform_sim      = 1.0;   // strong
    qi.successor_sim     = 0.2;   // weak
    qi.context_sim       = 0.0;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;

    QualityWeights w_arith{};
    w_arith.waveform = 0.5;
    w_arith.successor = 0.5;
    w_arith.use_harmonic_mean = false;

    QualityWeights w_harmo = w_arith;
    w_harmo.use_harmonic_mean = true;

    const double q_arith = computeQualityScore(qi, w_arith);
    const double q_harmo = computeQualityScore(qi, w_harmo);

    // Arithmetic: 0.5*1.0 + 0.5*0.2 = 0.6
    // Harmonic:   1.0 / (0.5/1.0 + 0.5/0.2) = 1.0 / (0.5 + 2.5) = 0.3333
    if (! (q_harmo < q_arith)) {
        std::fprintf(stderr, "[FAIL] harmonic must punish weak signal more: q_harmo=%.20g q_arith=%.20g\n",
                     q_harmo, q_arith);
        return false;
    }
    if (! nearly_equal(q_arith, 0.6, 1e-12)) {
        std::fprintf(stderr, "[FAIL] q_arith=%.20g expected 0.6\n", q_arith);
        return false;
    }
    if (! nearly_equal(q_harmo, 1.0 / 3.0, 1e-12)) {
        std::fprintf(stderr, "[FAIL] q_harmo=%.20g expected 0.3333..\n", q_harmo);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 9 — D2 harmonic falls back to 0 when no signal contributes.
// All weights zero, harmonic flag set → return 0.0 (mirrors arithmetic edge).
// ---------------------------------------------------------------------------
bool test_harmonic_zero_weights() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;

    QualityWeights w{};
    w.use_harmonic_mean = true;  // all weights default 0.0

    QualityInputs qi{};
    qi.successor_sim     = 0.5;
    qi.context_sim       = 0.5;
    qi.label_match       = 0.5;
    qi.section_sim       = 0.5;
    qi.bar_aligned       = 0.5;
    qi.energy_match      = 0.5;
    qi.edge_energy_match = 0.5;
    qi.centroid_match    = 0.5;

    const double q = computeQualityScore(qi, w);
    if (! nearly_equal(q, 0.0)) {
        std::fprintf(stderr, "[FAIL] harmonic_zero_weights q=%.20g expected 0.0\n", q);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 10 — D2 harmonic ε floor prevents NaN/Inf on q=0 input.
// successor=0 with weight=1.0, harmonic=true → harmonic_inv_sum = 1.0/ε,
// Q = 1.0 / (1.0/ε) = ε ≈ 1e-3. Must not return NaN/Inf.
// ---------------------------------------------------------------------------
bool test_harmonic_epsilon_floor() {
    using reamix::remix::QualityWeights;
    using reamix::remix::QualityInputs;
    using reamix::remix::computeQualityScore;
    using reamix::remix::kHarmonicMeanEpsilon;

    QualityWeights w{};
    w.successor         = 1.0;
    w.use_harmonic_mean = true;

    QualityInputs qi{};
    qi.successor_sim     = 0.0;     // catastrophically weak
    qi.context_sim       = 0.0;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;

    const double q = computeQualityScore(qi, w);
    if (std::isnan(q) || std::isinf(q)) {
        std::fprintf(stderr, "[FAIL] harmonic q on q=0 input is NaN/Inf: %.20g\n", q);
        return false;
    }
    // Q = 1.0 / (1.0 / ε) = ε
    if (! nearly_equal(q, kHarmonicMeanEpsilon, 1e-15)) {
        std::fprintf(stderr, "[FAIL] harmonic q=%.20g expected ε=%.20g\n", q, kHarmonicMeanEpsilon);
        return false;
    }
    return true;
}

} // namespace

int main()
{
    std::printf("test_sequential_continuity — sesja 81 ADR-068 D4+D2 self-validation\n");

    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"three_constituents",         test_three_constituents},
        {"two_constituents_no_mfcc",   test_two_constituents},
        {"weight_zero_no_contrib",     test_weight_zero_no_contribution},
        {"mixed_composition",          test_mixed_composition},
        {"identical_constituents",     test_identical_constituents},
        {"legacy_weights_unchanged",   test_legacy_weights_unchanged},
        {"harmonic_two_signals",       test_harmonic_two_signals},
        {"harmonic_punishes_weak",     test_harmonic_punishes_weak},
        {"harmonic_zero_weights",      test_harmonic_zero_weights},
        {"harmonic_epsilon_floor",     test_harmonic_epsilon_floor},
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
