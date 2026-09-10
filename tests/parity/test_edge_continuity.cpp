// test_edge_continuity — sesja 129 (ADR-116 step 2) self-validation.
//
// Validates the edge-continuity splice signal on hand-computed fixtures. No
// Python ground truth (C++-canonical per ADR-065); the perceptual evidence is
// meta/research/sesja-129-edge-continuity.md (sandbox prototype
// tools/dev/vocal_eval/edge_probe.py: probe cuts 8/8 ordered, rated Duration
// cuts AUC 0.91).
//
// Signal under test (SignalNorm.h):
//   d(i, j)   = RMS dB difference across mel bands between the END edge of
//               beat i and the END edge of beat j-1
//   scale     = median of d(b, b + 8 bars) over b (fallback one bar)
//   quality   = exp(-d / scale); 1.0 when the scale is invalid or j == 0
// and the composite slot `edge_continuity` in computeQualityScore.

#include "remix/Quality.h"
#include "remix/SignalNorm.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double kTol = 1e-9;

#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) { std::printf ("FAIL: %s — %s\n", #expr, msg); return false; }\
} while (0)

using namespace reamix::remix;

// n_beats rows of n_mel bands; row b = base + pattern(b).
std::vector<float> makeEdges (int n_beats, int n_mel, double (*rowValue) (int b, int m))
{
    std::vector<float> e ((std::size_t) n_beats * (std::size_t) n_mel);
    for (int b = 0; b < n_beats; ++b)
        for (int m = 0; m < n_mel; ++m)
            e[(std::size_t) b * (std::size_t) n_mel + (std::size_t) m] = (float) rowValue (b, m);
    return e;
}

bool testDistance()
{
    // Two rows: constant 0 dB and constant 3 dB -> RMS difference 3 dB.
    std::vector<float> e = { 0.f, 0.f, 0.f, 0.f,  3.f, 3.f, 3.f, 3.f,  0.f, 4.f, 0.f, 4.f };
    CHECK (std::abs (edgeMelDistance (e.data(), 4, 0, 1) - 3.0) < kTol, "constant offset = 3 dB");
    CHECK (std::abs (edgeMelDistance (e.data(), 4, 1, 0) - 3.0) < kTol, "symmetric");
    CHECK (std::abs (edgeMelDistance (e.data(), 4, 0, 0)) < kTol, "self distance 0");
    // Row 2 vs row 0: half the bands differ by 4 dB -> sqrt(mean(0,16,0,16)) = sqrt(8).
    CHECK (std::abs (edgeMelDistance (e.data(), 4, 0, 2) - std::sqrt (8.0)) < kTol, "mixed bands RMS");
    CHECK (edgeMelDistance (nullptr, 4, 0, 1) == 0.0, "null -> 0");
    return true;
}

bool testScaleMedianAndFallback()
{
    // A 4/4 track of 200 beats whose end edges follow a 4-beat (one bar)
    // sawtooth: rows one bar apart are identical, rows 8 bars apart too.
    // Add a slow ramp so the phrase-lag distance is a known constant:
    // row(b) = 0.1 * b  -> d(b, b + 32) = 3.2 dB for every b, d(b, b+4) = 0.4.
    auto ramp = [] (int b, int) { return 0.1 * b; };
    const auto e = makeEdges (200, 8, ramp);
    const auto s = buildEdgeContinuityScale (e.data(), 200, 8, 32, 4);
    CHECK (s.valid(), "phrase lag has >= 32 samples");
    CHECK (std::abs (s.scale - 3.2) < 1e-5, "median phrase-lag distance = 3.2 dB");
    CHECK (s.samples == 200 - 32, "samples = n - lag");

    // 40 beats: the phrase lag (32) leaves 8 samples < kMinSamples -> fallback one bar.
    const auto e2 = makeEdges (40, 8, ramp);
    const auto s2 = buildEdgeContinuityScale (e2.data(), 40, 8, 32, 4);
    CHECK (s2.valid(), "fallback lag valid");
    CHECK (std::abs (s2.scale - 0.4) < 1e-5, "fallback = one-bar median 0.4 dB");
    CHECK (s2.samples == 36, "fallback samples = n - bar");

    // 20 beats: neither lag reaches 32 samples -> invalid, quality neutral.
    const auto e3 = makeEdges (20, 8, ramp);
    const auto s3 = buildEdgeContinuityScale (e3.data(), 20, 8, 32, 4);
    CHECK (! s3.valid(), "too short -> invalid");
    CHECK (s3.quality (5.0) == 1.0, "invalid scale -> quality 1");
    CHECK (buildEdgeContinuityScale (nullptr, 200, 8, 32, 4).valid() == false, "null -> invalid");
    return true;
}

bool testQualityMapping()
{
    EdgeContinuityScale s; s.scale = 2.0; s.samples = 100;
    CHECK (s.valid(), "valid fixture");
    CHECK (std::abs (s.quality (0.0) - 1.0) < kTol, "d = 0 -> 1");
    CHECK (std::abs (s.quality (2.0) - std::exp (-1.0)) < kTol, "d = scale -> e^-1");
    CHECK (std::abs (s.quality (4.0) - std::exp (-2.0)) < kTol, "d = 2 scale -> e^-2");
    CHECK (s.quality (-1.0) == 1.0, "negative d clamps to 1");
    CHECK (s.quality (std::nan ("")) == 0.0, "NaN -> 0");
    return true;
}

bool testPairValueAndBaselines()
{
    // Ramp track; a consecutive pair (i -> i+1) compares row i with row i
    // -> d = 0 -> quality 1 (a natural transition is never penalised).
    auto ramp = [] (int b, int) { return 0.1 * b; };
    const auto e = makeEdges (200, 8, ramp);
    const SignalBaselines b = buildSignalBaselines (nullptr, nullptr, nullptr, nullptr, nullptr, 200,
                                                    e.data(), 8, 4);
    CHECK (b.edge_continuity.valid(), "baselines carry the edge scale");
    CHECK (std::abs (b.edge_continuity.scale - 3.2) < 1e-5, "lag = 8 x bar = 32 beats");

    const auto nat = edgeContinuityV2 (b, e.data(), 8, 200, 10, 11);
    CHECK (nat.available, "consecutive pair available");
    CHECK (std::abs (nat.distance) < kTol && std::abs (nat.quality - 1.0) < kTol, "consecutive pair d = 0, q = 1");

    // A jump 10 -> 43 compares row 10 with row 42: d = 3.2 = one scale -> e^-1.
    const auto jump = edgeContinuityV2 (b, e.data(), 8, 200, 10, 43);
    CHECK (std::abs (jump.distance - 1.0) < 1e-5, "d / scale = 1");
    CHECK (std::abs (jump.quality - std::exp (-1.0)) < 1e-5, "quality e^-1");

    // Landing on beat 0 has no original context -> not available, neutral.
    const auto j0 = edgeContinuityV2 (b, e.data(), 8, 200, 10, 0);
    CHECK (! j0.available && j0.quality == 1.0 && j0.distance == -1.0, "j = 0 neutral");
    // Missing arrays -> neutral; legacy baselines (no edge args) -> invalid scale.
    CHECK (! edgeContinuityV2 (b, nullptr, 8, 200, 10, 43).available, "null rows neutral");
    const SignalBaselines legacy = buildSignalBaselines (nullptr, nullptr, nullptr, nullptr, nullptr, 200);
    CHECK (! legacy.edge_continuity.valid(), "legacy baselines have no edge scale");
    CHECK (! edgeContinuityV2 (legacy, e.data(), 8, 200, 10, 43).available, "invalid scale neutral");
    return true;
}

bool testCompositeSlot()
{
    // Geometric composite with two active weights: waveform 0.5, edge 0.5.
    QualityWeights w{};
    w.use_geometric_mean = true; w.geometric_floor = 0.05;
    w.waveform = 0.5; w.edge_continuity = 0.5;
    QualityInputs q{};
    q.waveform_sim = 0.81; q.successor_sim = 0.0; q.context_sim = 0.0;
    q.label_match = 0.0; q.section_sim = 0.0; q.bar_aligned = 0.0;
    q.energy_match = 1.0; q.edge_energy_match = 1.0; q.centroid_match = 1.0;
    q.edge_continuity = 0.25;
    // exp((0.5 ln 0.81 + 0.5 ln 0.25) / 1.0) = sqrt(0.81 * 0.25) = 0.45
    CHECK (std::abs (computeQualityScore (q, w) - 0.45) < 1e-9, "geometric mean over the two slots");
    // nullopt input -> the weight is missing, the composite is waveform alone.
    q.edge_continuity.reset();
    CHECK (std::abs (computeQualityScore (q, w) - 0.81) < 1e-9, "missing slot redistributes");
    // kV2 carries the weight, legacy / default do not.
    CHECK (kV2QualityWeights.edge_continuity > 0.0, "v2 weight set");
    CHECK (kDefaultQualityWeights.edge_continuity == 0.0 && kLegacyQualityWeights.edge_continuity == 0.0,
           "legacy / default weight 0");
    return true;
}
} // namespace

int main()
{
    int fails = 0;
    fails += testDistance()              ? 0 : 1;
    fails += testScaleMedianAndFallback() ? 0 : 1;
    fails += testQualityMapping()        ? 0 : 1;
    fails += testPairValueAndBaselines() ? 0 : 1;
    fails += testCompositeSlot()         ? 0 : 1;
    if (fails == 0) std::printf ("test_edge_continuity: all 5 groups PASS\n");
    else            std::printf ("test_edge_continuity: %d group(s) FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
