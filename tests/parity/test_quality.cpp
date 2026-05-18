// Parity: reamix::remix::{computeQualityScore, computeVocalPenalty,
// computeOnsetPenalty} vs references/python-source/remix/quality.py
// (session 17, 2026-04-21).
//
// Bitwise target. The Python reference is pure scalar arithmetic (sums of
// weight×value, max/min clamps, no library calls) and the C++ port mirrors
// the Python accumulation order exactly, so every input must map to an
// IEEE-754 bit-identical output — any non-bitwise result flags a
// transcription or ordering bug.
//
// Fixtures under tests/parity/reference/data/quality/ (emitted by
// tools/dump_quality_tests.py):
//   quality_score_inputs.npy    (N, 10) f64  — 10 signal values
//   quality_score_flags.npy     (N, 2)  i64  — [waveform_missing,
//                                               edge_splice_missing] booleans
//   quality_score_expected.npy  (N,)    f64  — compute_quality_score output
//
//   vocal_penalty_inputs.npy    (M, 4)  f64  — [va_source, va_dest,
//                                               edge_va_end, edge_va_start]
//   vocal_penalty_flags.npy     (M, 2)  i64  — [edge_end_missing,
//                                               edge_start_missing]
//   vocal_penalty_expected.npy  (M,)    f64
//
//   onset_penalty_inputs.npy    (K,)    f64  — destination onset values
//   onset_penalty_flags.npy     (K,)    i64  — dest_missing boolean
//   onset_penalty_expected.npy  (K,)    f64
//
// N=210 (10 handcrafted edge + 200 random, PCG64 seed=42).
// M=110 (10 edge + 100 random, seed=7).
// K=57  (7 edge + 50 random, seed=11).

#include "remix/Quality.h"

#include "NpyIO.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool bitwiseEqualF64(double a, double b)
{
    std::uint64_t ai, bi;
    std::memcpy(&ai, &a, sizeof(ai));
    std::memcpy(&bi, &b, sizeof(bi));
    return ai == bi;
}

struct FailCount { int n = 0; };

// --- compute_quality_score ---------------------------------------------------
void runQualityScoreCases(const fs::path& dir, FailCount& fc)
{
    auto inputsMat   = reamix::test::loadNpy2DFloat64((dir / "quality_score_inputs.npy").string());
    auto flagsMat    = reamix::test::loadNpy2DInt64  ((dir / "quality_score_flags.npy").string());
    auto expected    = reamix::test::loadNpy1DFloat64((dir / "quality_score_expected.npy").string());

    if (inputsMat.rows != flagsMat.rows || inputsMat.rows != expected.size()) {
        std::printf("[FAIL] quality_score: row-count mismatch inputs=%zu flags=%zu expected=%zu\n",
                    inputsMat.rows, flagsMat.rows, expected.size());
        ++fc.n;
        return;
    }
    if (inputsMat.cols != 10 || flagsMat.cols != 2) {
        std::printf("[FAIL] quality_score: shape mismatch inputs (_, %zu) expected (_, 10), "
                    "flags (_, %zu) expected (_, 2)\n",
                    inputsMat.cols, flagsMat.cols);
        ++fc.n;
        return;
    }

    const std::size_t N = inputsMat.rows;
    std::size_t mismatches = 0;
    double maxAbs = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        reamix::remix::QualityInputs in;
        const bool waveform_missing    = flagsMat.at(i, 0) != 0;
        const bool edge_splice_missing = flagsMat.at(i, 1) != 0;

        if (!waveform_missing)    in.waveform_sim    = inputsMat.at(i, 0);
        in.successor_sim     = inputsMat.at(i, 1);
        if (!edge_splice_missing) in.edge_splice_sim = inputsMat.at(i, 2);
        in.context_sim       = inputsMat.at(i, 3);
        in.label_match       = inputsMat.at(i, 4);
        in.section_sim       = inputsMat.at(i, 5);
        in.bar_aligned       = inputsMat.at(i, 6);
        in.energy_match      = inputsMat.at(i, 7);
        in.edge_energy_match = inputsMat.at(i, 8);
        in.centroid_match    = inputsMat.at(i, 9);

        // Sesja 81 ADR-068 D1+D4 default flip — pin to legacy 10-component
        // simplex (kLegacyQualityWeights) explicitly so the Python-parity
        // fixture comparison stays bit-exact past the production-default
        // change (kDefaultQualityWeights now ships the new 6-D simplex).
        const double got = reamix::remix::computeQualityScore(
            in, reamix::remix::kLegacyQualityWeights);
        const double want = expected[i];
        if (!bitwiseEqualF64(got, want)) {
            ++mismatches;
            const double d = std::abs(got - want);
            if (d > maxAbs) maxAbs = d;
            if (mismatches <= 3) {
                std::printf("  [DIFF] i=%zu  got=%.17g  want=%.17g  abs_diff=%.3e\n",
                            i, got, want, d);
            }
        }
    }

    const bool pass = (mismatches == 0);
    std::printf("%-22s  N=%zu  bitwise_mismatches=%zu  max_abs=%.3e  %s\n",
                "quality_score", N, mismatches, maxAbs, pass ? "PASS" : "FAIL");
    if (!pass) ++fc.n;
}

// --- compute_vocal_penalty ---------------------------------------------------
void runVocalPenaltyCases(const fs::path& dir, FailCount& fc)
{
    auto inputsMat   = reamix::test::loadNpy2DFloat64((dir / "vocal_penalty_inputs.npy").string());
    auto flagsMat    = reamix::test::loadNpy2DInt64  ((dir / "vocal_penalty_flags.npy").string());
    auto expected    = reamix::test::loadNpy1DFloat64((dir / "vocal_penalty_expected.npy").string());

    if (inputsMat.rows != flagsMat.rows || inputsMat.rows != expected.size()) {
        std::printf("[FAIL] vocal_penalty: row-count mismatch inputs=%zu flags=%zu expected=%zu\n",
                    inputsMat.rows, flagsMat.rows, expected.size());
        ++fc.n;
        return;
    }
    if (inputsMat.cols != 4 || flagsMat.cols != 2) {
        std::printf("[FAIL] vocal_penalty: shape mismatch inputs (_, %zu) expected (_, 4), "
                    "flags (_, %zu) expected (_, 2)\n",
                    inputsMat.cols, flagsMat.cols);
        ++fc.n;
        return;
    }

    const std::size_t M = inputsMat.rows;
    std::size_t mismatches = 0;
    double maxAbs = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
        const double va_source = inputsMat.at(i, 0);
        const double va_dest   = inputsMat.at(i, 1);
        std::optional<double> edge_end;
        std::optional<double> edge_start;
        if (flagsMat.at(i, 0) == 0) edge_end   = inputsMat.at(i, 2);
        if (flagsMat.at(i, 1) == 0) edge_start = inputsMat.at(i, 3);

        const double got  = reamix::remix::computeVocalPenalty(
            va_source, va_dest, edge_end, edge_start
        );
        const double want = expected[i];
        if (!bitwiseEqualF64(got, want)) {
            ++mismatches;
            const double d = std::abs(got - want);
            if (d > maxAbs) maxAbs = d;
            if (mismatches <= 3) {
                std::printf("  [DIFF] i=%zu  got=%.17g  want=%.17g  abs_diff=%.3e\n",
                            i, got, want, d);
            }
        }
    }

    const bool pass = (mismatches == 0);
    std::printf("%-22s  M=%zu  bitwise_mismatches=%zu  max_abs=%.3e  %s\n",
                "vocal_penalty", M, mismatches, maxAbs, pass ? "PASS" : "FAIL");
    if (!pass) ++fc.n;
}

// --- compute_onset_penalty ---------------------------------------------------
void runOnsetPenaltyCases(const fs::path& dir, FailCount& fc)
{
    auto inputs   = reamix::test::loadNpy1DFloat64((dir / "onset_penalty_inputs.npy").string());
    auto flags    = reamix::test::loadNpy1DInt64  ((dir / "onset_penalty_flags.npy").string());
    auto expected = reamix::test::loadNpy1DFloat64((dir / "onset_penalty_expected.npy").string());

    if (inputs.size() != flags.size() || inputs.size() != expected.size()) {
        std::printf("[FAIL] onset_penalty: size mismatch inputs=%zu flags=%zu expected=%zu\n",
                    inputs.size(), flags.size(), expected.size());
        ++fc.n;
        return;
    }

    const std::size_t K = inputs.size();
    std::size_t mismatches = 0;
    double maxAbs = 0.0;
    for (std::size_t i = 0; i < K; ++i) {
        std::optional<double> dest;
        if (flags[i] == 0) dest = inputs[i];

        const double got  = reamix::remix::computeOnsetPenalty(dest);
        const double want = expected[i];
        if (!bitwiseEqualF64(got, want)) {
            ++mismatches;
            const double d = std::abs(got - want);
            if (d > maxAbs) maxAbs = d;
            if (mismatches <= 3) {
                std::printf("  [DIFF] i=%zu  got=%.17g  want=%.17g  abs_diff=%.3e\n",
                            i, got, want, d);
            }
        }
    }

    const bool pass = (mismatches == 0);
    std::printf("%-22s  K=%zu  bitwise_mismatches=%zu  max_abs=%.3e  %s\n",
                "onset_penalty", K, mismatches, maxAbs, pass ? "PASS" : "FAIL");
    if (!pass) ++fc.n;
}

// --- labels / color keys (unit-style spot checks, no golden dump) -----------
void runLabelAndColorKeyChecks(FailCount& fc)
{
    struct LabelCase { double score; const char* want; };
    const LabelCase label_cases[] = {
        { 0.0,  "Weak"  },
        { 0.44, "Weak"  },
        { 0.45, "Weak"  },  // below QUALITY_GOOD=0.5
        { 0.50, "Good"  },  // boundary >=
        { 0.69, "Good"  },
        { 0.70, "Good"  },  // boundary: strict > for Great, so 0.7 is still Good
        { 0.71, "Great" },
        { 1.00, "Great" },
    };
    for (const auto& c : label_cases) {
        const auto got = reamix::remix::qualityLabel(c.score);
        if (got != c.want) {
            std::printf("  [DIFF] qualityLabel(%.3f) got=\"%s\" want=\"%s\"\n",
                        c.score, std::string(got).c_str(), c.want);
            ++fc.n;
        }
    }

    struct ColorCase { std::optional<double> score; const char* want; };
    const ColorCase color_cases[] = {
        { std::nullopt, "unknown" },
        { 0.0,          "bad"     },
        { 0.49,         "bad"     },
        { 0.50,         "medium"  },
        { 0.70,         "medium"  },  // strict > for "good"
        { 0.71,         "good"    },
        { 1.00,         "good"    },
    };
    for (const auto& c : color_cases) {
        const auto got = reamix::remix::qualityColorKey(c.score);
        if (got != c.want) {
            const double s = c.score.value_or(-999.0);
            std::printf("  [DIFF] qualityColorKey(%s=%.3f) got=\"%s\" want=\"%s\"\n",
                        c.score.has_value() ? "some" : "none",
                        s, std::string(got).c_str(), c.want);
            ++fc.n;
        }
    }

    std::printf("%-22s  label_cases=%zu  color_cases=%zu  PASS-unless-DIFFs-above\n",
                "label_and_color_key",
                sizeof(label_cases) / sizeof(label_cases[0]),
                sizeof(color_cases) / sizeof(color_cases[0]));
}

// --- weights sum sanity (compile-time already asserted; runtime echo) -------
void runWeightsSumEcho()
{
    const double sum = reamix::remix::kQualityWeightsSum;
    std::printf("%-22s  sum=%.17g  delta_from_1.0=%.3e\n",
                "weights_sum_check", sum, std::abs(sum - 1.0));
}

} // namespace

int main(int argc, char** argv)
{
    const fs::path root = (argc > 1)
        ? fs::path(argv[1])
        : fs::path("tests/parity/reference/data/quality");

    if (!fs::exists(root) || !fs::is_directory(root)) {
        std::fprintf(stderr, "data directory not found: %s\n"
                             "run `python tools/dump_quality_tests.py` first\n",
                     root.string().c_str());
        return 2;
    }

    FailCount fc;

    runWeightsSumEcho();
    runQualityScoreCases(root, fc);
    runVocalPenaltyCases(root, fc);
    runOnsetPenaltyCases(root, fc);
    runLabelAndColorKeyChecks(fc);

    std::printf("\nfailures: %d\n", fc.n);
    return fc.n == 0 ? 0 : 1;
}
