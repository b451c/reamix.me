// test_signal_norm — ADR-115 (sesja 114) self-validation of the v2 scoring
// primitives: DistanceBaseline (sequential-baseline percentile), the
// buildSignalBaselines helpers, and the geometric composite in
// computeQualityScore. C++-canonical, hand-computed expectations, no Python
// reference (per feedback_python_no_longer_source_of_truth.md).
#include "remix/Quality.h"
#include "remix/SignalNorm.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace reamix::remix;

static int g_fail = 0;

static void expectNear(const char* what, double got, double want, double tol = 1e-9)
{
    if (std::abs(got - want) > tol) {
        std::printf("FAIL %s: got %.9f want %.9f\n", what, got, want);
        ++g_fail;
    } else {
        std::printf("ok   %s = %.6f\n", what, got);
    }
}

static void expectTrue(const char* what, bool v)
{
    if (! v) { std::printf("FAIL %s\n", what); ++g_fail; }
    else       std::printf("ok   %s\n", what);
}

int main()
{
    // ---- DistanceBaseline: 40 reference steps 1..40 (sesja 115 exp mapping)
    {
        std::vector<double> ref;
        for (int k = 40; k >= 1; --k) ref.push_back(static_cast<double>(k));   // unsorted on purpose
        DistanceBaseline b(ref);
        expectTrue("baseline valid at 40 samples", b.valid());
        // nearest-rank p90 over 40 sorted values: idx = round(0.9 * 39) = 35 -> 36
        expectNear("scale = p90 = 36", b.scale(), 36.0);
        // nearest-rank p98: idx = round(0.98 * 39) = 38 -> 39
        expectNear("reject threshold = p98 = 39", b.rejectAbove(), 39.0);
        expectNear("q(0)     = 1", b.quality(0.0), 1.0);
        expectNear("q(scale) = e^-1", b.quality(36.0), std::exp(-1.0));
        expectNear("q(2 x scale) = e^-2 (no saturation in the tail)", b.quality(72.0), std::exp(-2.0));
        expectTrue("q keeps a gradient beyond the largest step", b.quality(80.0) < b.quality(41.0));
        expectNear("q(NaN) = 0", b.quality(std::nan("")), 0.0);
        expectTrue("reject(39)   = false (at p98)", ! b.reject(39.0));
        expectTrue("reject(39.5) = true", b.reject(39.5));
        expectTrue("reject(NaN)  = true", b.reject(std::nan("")));
        expectNear("p50 nearest-rank", b.percentile(0.5), 21.0);
        expectNear("p100", b.percentile(1.0), 40.0);
        DistanceBaseline small({1, 2, 3});
        expectTrue("baseline invalid below kMinSamples", ! small.valid());
        expectNear("invalid baseline never penalises", small.quality(50.0), 1.0);
        expectTrue("invalid baseline never rejects", ! small.reject(50.0));
    }

    // ---- sequentialAbsDiff / sequentialPairDiff ---------------------------
    {
        const double v[] = {1.0, 2.0, 4.0, 4.0, 8.0};
        auto lin = sequentialAbsDiff(v, 5, false);
        expectTrue("lin diff count", lin.size() == 4);
        expectNear("lin diff[0]", lin[0], 1.0);
        expectNear("lin diff[2]", lin[2], 0.0);
        auto lg = sequentialAbsDiff(v, 5, true);
        expectNear("log diff[0] = ln2", lg[0], std::log(2.0));
        expectNear("log diff[3] = ln2", lg[3], std::log(2.0));
        const double e[] = {0.0, -3.0, -6.0};
        const double s[] = {-1.0, -2.0, -12.0};
        auto pd = sequentialPairDiff(e, s, 3);
        expectTrue("pair diff count", pd.size() == 2);
        expectNear("pair diff[0] = |s1 - e0|", pd[0], 2.0);
        expectNear("pair diff[1] = |s2 - e1|", pd[1], 9.0);
    }

    // ---- buildSignalBaselines + V2 quality helpers -------------------------
    {
        // rms doubles every beat: all 39 sequential log-steps = ln 2 (n = 40)
        // -> scale = p90 = ln 2, reject above p98 = ln 2
        std::vector<double> rms;
        for (int i = 0; i < 40; ++i) rms.push_back(std::pow(2.0, i));
        SignalBaselines b = buildSignalBaselines(rms.data(), nullptr, nullptr, nullptr, nullptr, 40);
        expectTrue("energy baseline valid", b.energy.valid());
        expectTrue("centroid baseline invalid (null)", ! b.centroid.valid());
        const double ln2 = std::log(2.0);
        expectNear("energy scale = ln 2", b.energy.scale(), ln2, 1e-12);
        expectNear("dRMS = x1 -> q = 1", energyQualityV2(b, 1.0, 1.0, 0.123), 1.0);
        expectNear("dRMS = x1.5 -> exp(-ln1.5 / ln2)", energyQualityV2(b, 1.0, 1.5, 0.123),
                   std::exp(-std::log(1.5) / ln2), 1e-12);
        expectNear("dRMS = x2 (one scale) -> e^-1", energyQualityV2(b, 1.0, 2.0, 0.123), std::exp(-1.0), 1e-9);
        expectNear("dRMS = x4 (two scales) -> e^-2", energyQualityV2(b, 1.0, 4.0, 0.123), std::exp(-2.0), 1e-9);
        // ln(2^(i+1)) - ln(2^i) rounds a few ULP around ln 2, so probe just
        // below / above the p98 step instead of on the tie
        expectTrue("x1.999 (just below p98) not rejected", ! loudnessRejectV2(b, 1.0, 1.999, 0.0));
        expectTrue("x2.001 (just above p98) rejected", loudnessRejectV2(b, 1.0, 2.001, 0.0));
        expectTrue("edge baseline invalid -> edge never rejects", ! loudnessRejectV2(b, 1.0, 1.0, 99.0));
        expectNear("centroid falls back to legacy", centroidQualityV2(b, 1.0, 9.0, 0.321), 0.321);
    }

    // ---- geometric composite ------------------------------------------------
    {
        QualityInputs qi{};
        qi.waveform_sim      = 0.8;
        qi.successor_sim     = 0.9;   // sequential group: mean(successor, context) = 0.7
        qi.context_sim       = 0.5;
        qi.energy_match      = 0.0;   // clamp-to-zero signal
        qi.edge_energy_match = 1.0;
        qi.centroid_match    = 1.0;
        qi.transient_continuity = 0.5;
        QualityWeights w = kV2QualityWeights;
        const double got = computeQualityScore(qi, w);
        // exp( (0.45 ln0.8 + 0.25 ln0.7 + 0.08 ln0.05 + 0.05 ln1 + 0.05 ln1 + 0.12 ln0.5) / 1.0 )
        const double want = std::exp(0.45 * std::log(0.8) + 0.25 * std::log(0.7)
                                     + 0.08 * std::log(0.05) + 0.12 * std::log(0.5));
        expectNear("geometric composite, energy floored at 0.05", got, want, 1e-9);
        expectTrue("a zeroed signal no longer zeroes the composite", got > 0.5);
        // harmonic legacy on the same inputs collapses (documents the ADR-115 motivation)
        QualityWeights h = kDefaultQualityWeights;
        const double harm = computeQualityScore(qi, h);
        expectTrue("harmonic legacy composite < 0.2 on the same inputs", harm < 0.2);
        // missing optional signal is renormalised out
        QualityInputs qm = qi; qm.transient_continuity.reset();
        const double got2 = computeQualityScore(qm, w);
        const double want2 = std::exp((0.45 * std::log(0.8) + 0.25 * std::log(0.7)
                                       + 0.08 * std::log(0.05)) / 0.88);
        expectNear("geometric composite renormalises a missing signal", got2, want2, 1e-9);
        // all signals perfect -> 1.0
        QualityInputs q1{}; q1.waveform_sim = 1.0; q1.successor_sim = 1.0; q1.context_sim = 1.0;
        q1.energy_match = 1.0; q1.edge_energy_match = 1.0; q1.centroid_match = 1.0; q1.transient_continuity = 1.0;
        expectNear("all-ones -> 1.0", computeQualityScore(q1, w), 1.0, 1e-12);
    }

    std::printf(g_fail == 0 ? "test_signal_norm PASS\n" : "test_signal_norm FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
