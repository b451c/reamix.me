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
    // ---- DistanceBaseline: 10 reference steps 1..10 ----------------------
    {
        DistanceBaseline b({3, 1, 4, 1, 5, 9, 2, 6, 5, 3});   // sorted: 1 1 2 3 3 4 5 5 6 9
        expectTrue("baseline valid at 10 samples", b.valid());
        expectNear("q(0)   = 1 (better than every step)", b.quality(0.0), 1.0);
        expectNear("q(1)   = 8/10 (ties count as not-worse)", b.quality(1.0), 0.8);
        expectNear("q(3)   = 5/10", b.quality(3.0), 0.5);
        expectNear("q(4.5) = 4/10", b.quality(4.5), 0.4);
        expectNear("q(9)   = 0", b.quality(9.0), 0.0);
        expectNear("q(100) = 0", b.quality(100.0), 0.0);
        expectNear("q(NaN) = 0", b.quality(std::nan("")), 0.0);
        expectNear("p50 nearest-rank", b.percentile(0.5), 4.0);
        expectNear("p100", b.percentile(1.0), 9.0);
        DistanceBaseline small({1, 2, 3});
        expectTrue("baseline invalid below kMinSamples", ! small.valid());
        expectNear("invalid baseline never penalises", small.quality(50.0), 1.0);
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
        // rms doubles every beat: all sequential log-steps = ln 2 (n = 12)
        std::vector<double> rms;
        for (int i = 0; i < 12; ++i) rms.push_back(std::pow(2.0, i));
        SignalBaselines b = buildSignalBaselines(rms.data(), nullptr, nullptr, nullptr, nullptr, 12);
        expectTrue("energy baseline valid", b.energy.valid());
        expectTrue("centroid baseline invalid (null)", ! b.centroid.valid());
        expectNear("dRMS = x1.5 (< ln2 step) -> q = 1", energyQualityV2(b, 1.0, 1.5, 0.123), 1.0);
        // ln(2^(i+1)) - ln(2^i) rounds a few ULP around ln 2, so probe just
        // below / above the step instead of on the tie
        expectNear("dRMS = x1.999 (just below the step) -> q = 1", energyQualityV2(b, 1.0, 1.999, 0.123), 1.0);
        expectNear("dRMS = x2.001 (just above the step) -> q = 0", energyQualityV2(b, 1.0, 2.001, 0.123), 0.0);
        expectNear("dRMS = x4 -> q = 0", energyQualityV2(b, 1.0, 4.0, 0.123), 0.0);
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
        // exp( (0.45 ln0.8 + 0.25 ln0.7 + 0.08 ln0.1 + 0.05 ln1 + 0.05 ln1 + 0.12 ln0.5) / 1.0 )
        const double want = std::exp(0.45 * std::log(0.8) + 0.25 * std::log(0.7)
                                     + 0.08 * std::log(0.1) + 0.12 * std::log(0.5));
        expectNear("geometric composite, energy floored at 0.1", got, want, 1e-9);
        expectTrue("a zeroed signal no longer zeroes the composite", got > 0.5);
        // harmonic legacy on the same inputs collapses (documents the ADR-115 motivation)
        QualityWeights h = kDefaultQualityWeights;
        const double harm = computeQualityScore(qi, h);
        expectTrue("harmonic legacy composite < 0.2 on the same inputs", harm < 0.2);
        // missing optional signal is renormalised out
        QualityInputs qm = qi; qm.transient_continuity.reset();
        const double got2 = computeQualityScore(qm, w);
        const double want2 = std::exp((0.45 * std::log(0.8) + 0.25 * std::log(0.7)
                                       + 0.08 * std::log(0.1)) / 0.88);
        expectNear("geometric composite renormalises a missing signal", got2, want2, 1e-9);
        // all signals perfect -> 1.0
        QualityInputs q1{}; q1.waveform_sim = 1.0; q1.successor_sim = 1.0; q1.context_sim = 1.0;
        q1.energy_match = 1.0; q1.edge_energy_match = 1.0; q1.centroid_match = 1.0; q1.transient_continuity = 1.0;
        expectNear("all-ones -> 1.0", computeQualityScore(q1, w), 1.0, 1e-12);
    }

    std::printf(g_fail == 0 ? "test_signal_norm PASS\n" : "test_signal_norm FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
