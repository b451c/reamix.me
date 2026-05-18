// test_vocal_continuity — sesja 98 (ADR-088) self-validation.
//
// Validates the vocal phrase continuity cost component formula on hand-
// computed fixtures. No Python ground truth (dev calibration is C++-canonical
// per ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`).
//
// Formula under test (STATUS UPDATE 1, sesja 98):
//   if vocal_density(i, j) < 0.1:
//       qi.vocal_continuity = 1.0                                  (silence)
//   else:
//       qi.vocal_continuity = 0.5 + 0.5 * max(release_end[i], onset_start[j])
//
// Where vocal_density(i, j) = max(vocal_activity[i], vocal_activity[j]).
//
// Initial PROPOSED formula `1 - max(...)` was inverted — punished phrase
// boundaries (best splice points). User-flagged sesja 98 smoke "brak
// połączeń" confirmed: composite Q crashed via harmonic mean × epsilon for
// vocal-dense tracks. Fixed formula:
//   - HIGH boundary signals → HIGH quality (reward phrase alignment).
//   - Silence (vocal_density < 0.1) → q = 1.0 (instrumental gap = clean).
//   - Soft floor 0.5: prevents harmonic_inv_sum term from blowing up via
//     epsilon-clamped denominator (kHarmonicMeanEpsilon = 1e-3 → q ≈ 0
//     produces 1000× weight amplification in composite).

#include "remix/Quality.h"
#include "ui/RemixCache.h"

#include <cmath>
#include <cstdio>

namespace
{

constexpr double kTol = 1e-9;

#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) { std::printf ("FAIL: %s — %s\n", #expr, msg); return false; }\
} while (0)

// Helper replicating production formula on per-pair basis.
double formula (double rel_i, double on_j, double va_i, double va_j)
{
    constexpr double kSilenceThreshold = 0.1;
    const double vocal_density = std::max (va_i, va_j);
    if (vocal_density < kSilenceThreshold) return 1.0;
    return 0.5 + 0.5 * std::max (rel_i, on_j);
}

bool testFormulaSilenceGapClean()
{
    // Both vocal_activity below 0.1 → q = 1.0 regardless of boundary signals.
    CHECK (std::abs (formula (0.0, 0.0, 0.0, 0.0) - 1.0) < kTol,
           "silence + zero boundaries → q=1.0");
    CHECK (std::abs (formula (0.8, 0.6, 0.05, 0.0) - 1.0) < kTol,
           "silence (va<0.1) overrides boundary signals → q=1.0");
    return true;
}

bool testFormulaBoundaryRewarded()
{
    // Vocal active, phrase boundary aligned (release_end OR onset_start HIGH)
    //  → q approaches 1.0.
    CHECK (std::abs (formula (1.0, 0.0, 0.5, 0.5) - 1.0) < kTol,
           "release_end=1 + vocal active → q=1.0 (boundary aligned)");
    CHECK (std::abs (formula (0.0, 1.0, 0.5, 0.5) - 1.0) < kTol,
           "onset_start=1 + vocal active → q=1.0 (boundary aligned)");
    CHECK (std::abs (formula (1.0, 1.0, 0.5, 0.5) - 1.0) < kTol,
           "both boundaries → q=1.0");
    return true;
}

bool testFormulaMidPhraseSoftFloor()
{
    // Vocal active, no boundary signal (mid-phrase steady vocal) → q=0.5.
    // Must NOT be near 0 (that would harmonic-mean-crash composite).
    CHECK (std::abs (formula (0.0, 0.0, 0.7, 0.7) - 0.5) < kTol,
           "mid-phrase steady vocal → q=0.5 (soft floor)");
    return true;
}

bool testFormulaPartialBoundary()
{
    // Vocal active + partial boundary signal → linear interpolation in [0.5, 1.0].
    CHECK (std::abs (formula (0.4, 0.0, 0.5, 0.5) - 0.7) < kTol,
           "release_end=0.4 + vocal active → q = 0.5 + 0.5*0.4 = 0.7");
    CHECK (std::abs (formula (0.0, 0.6, 0.5, 0.5) - 0.8) < kTol,
           "onset_start=0.6 + vocal active → q = 0.5 + 0.5*0.6 = 0.8");
    return true;
}

bool testFormulaNeverCrashes()
{
    // Sweep all combinations — q must always be in [0.5, 1.0] when vocal
    // active, or 1.0 when silence. Never below 0.5 → harmonic mean stays
    // manageable.
    const double samples [] = { 0.0, 0.25, 0.5, 0.75, 1.0 };
    for (double rel : samples)
        for (double on : samples)
            for (double vai : samples)
                for (double vaj : samples)
                {
                    const double q = formula (rel, on, vai, vaj);
                    if (q < 0.5 - 1e-9 || q > 1.0 + 1e-9)
                    {
                        std::printf (
                            "FAIL: q out of [0.5, 1.0]: rel=%.2f on=%.2f vai=%.2f vaj=%.2f → q=%.6f\n",
                            rel, on, vai, vaj, q);
                        return false;
                    }
                }
    return true;
}

bool testDefaultWeightZeroPreservesBaseline()
{
    const auto& d = reamix::remix::kDefaultQualityWeights;
    CHECK (d.vocal_continuity == 0.0,
           "kDefaultQualityWeights.vocal_continuity == 0.0");
    CHECK (reamix::ui::qualityWeightsAtDefault (d),
           "default weights match atDefault helper");
    // Modifying only vocal_continuity should NOT match atDefault.
    auto modified = d;
    modified.vocal_continuity = 0.10;
    CHECK (! reamix::ui::qualityWeightsAtDefault (modified),
           "non-zero vocal_continuity must trip atDefault");
    return true;
}

bool testHashSensitivity()
{
    const auto& d = reamix::remix::kDefaultQualityWeights;
    CHECK (reamix::ui::hashQualityWeights (d) == 0,
           "default weights hash to 0 (atDefault sentinel)");
    auto modified = d;
    modified.vocal_continuity = 0.05;
    const auto h = reamix::ui::hashQualityWeights (modified);
    CHECK (h != 0, "vocal_continuity != 0 produces non-zero hash");

    // Hash must change when vocal_continuity moves.
    auto modified2 = d;
    modified2.vocal_continuity = 0.15;
    const auto h2 = reamix::ui::hashQualityWeights (modified2);
    CHECK (h2 != h, "different vocal_continuity values hash differently");
    return true;
}

bool testQualityScoreContribution()
{
    // When vocal_continuity weight > 0 and qi.vocal_continuity provided,
    // computeQualityScore must include it in the weighted sum.
    auto weights = reamix::remix::kDefaultQualityWeights;
    // Boost vocal_continuity, zero out others to isolate.
    weights.waveform              = 0.0;
    weights.sequential_continuity = 0.0;
    weights.transient_continuity  = 0.0;
    weights.energy                = 0.0;
    weights.edge_energy           = 0.0;
    weights.bar_align             = 0.0;
    weights.centroid              = 0.0;
    weights.vocal_continuity      = 1.0;
    weights.use_harmonic_mean     = false;

    reamix::remix::QualityInputs qi {};
    qi.successor_sim     = 0.5;
    qi.context_sim       = 0.5;
    qi.label_match       = 0.0;
    qi.section_sim       = 0.0;
    qi.bar_aligned       = 0.0;
    qi.energy_match      = 0.0;
    qi.edge_energy_match = 0.0;
    qi.centroid_match    = 0.0;
    qi.vocal_continuity  = 0.75;  // simulated formula output (mid-phrase + half boundary)

    const double q = reamix::remix::computeQualityScore (qi, weights);
    CHECK (std::abs (q - 0.75) < 1e-6,
           "isolated vocal_continuity weight reproduces qi value");
    return true;
}

bool testNulloptRedistributesWeight()
{
    // When qi.vocal_continuity is nullopt, the missing-signal redistribution
    // path should redistribute its weight to provided signals — bit-exact
    // baseline behavior preserved.
    auto weights = reamix::remix::kDefaultQualityWeights;
    weights.vocal_continuity = 0.10;  // non-zero weight allocated

    reamix::remix::QualityInputs qi {};
    qi.successor_sim     = 1.0;
    qi.context_sim       = 1.0;
    qi.label_match       = 1.0;
    qi.section_sim       = 1.0;
    qi.bar_aligned       = 1.0;
    qi.energy_match      = 1.0;
    qi.edge_energy_match = 1.0;
    qi.centroid_match    = 1.0;
    qi.transient_continuity = 1.0;
    qi.mfcc_continuity      = 1.0;
    qi.full_mix_chroma_continuity.reset();
    qi.vocal_continuity.reset();  // missing signal

    // computeQualityScore should not crash + should produce a value in [0, 1].
    const double q = reamix::remix::computeQualityScore (qi, weights);
    CHECK (q >= 0.0 && q <= 1.0, "score in [0,1] when vocal_continuity missing");
    return true;
}

} // namespace

int main()
{
    bool ok = true;
    struct TC { const char* name; bool (*fn)(); };
    const TC cases [] = {
        { "formulaSilenceGapClean",       testFormulaSilenceGapClean },
        { "formulaBoundaryRewarded",      testFormulaBoundaryRewarded },
        { "formulaMidPhraseSoftFloor",    testFormulaMidPhraseSoftFloor },
        { "formulaPartialBoundary",       testFormulaPartialBoundary },
        { "formulaNeverCrashes",          testFormulaNeverCrashes },
        { "defaultWeightZeroPreservesBaseline", testDefaultWeightZeroPreservesBaseline },
        { "hashSensitivity",              testHashSensitivity },
        { "qualityScoreContribution",     testQualityScoreContribution },
        { "nulloptRedistributesWeight",   testNulloptRedistributesWeight },
    };
    for (auto& tc : cases)
    {
        const bool pass = tc.fn();
        std::printf ("%s %s\n", pass ? "PASS" : "FAIL", tc.name);
        if (! pass) ok = false;
    }
    std::printf ("\n%s — vocal_continuity (ADR-088 sesja 98)\n",
                 ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
