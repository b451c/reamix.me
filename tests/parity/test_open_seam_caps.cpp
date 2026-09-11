// test_open_seam_caps — sesja 135 (DEV-121) self-validation of the open-seam
// gates in the shared pair scorer (src/remix/PairScorer.cpp, boundary path
// with skip_energy_gate): gate 3 = the sesja-134 edge-view caps, gate 4 = the
// brightness-collapse cap (SignalNorm.h centroidCollapseV2). C++-canonical
// (ADR-065): hand-computed values on a synthetic track, no Python ground
// truth; the perceptual gate is the blinded round.
//
// Fixture: 64 beats, spectral centroid bright (1.0 / 0.9 alternating) on beats
// 0-31 and dark (0.3 / 0.27) on 32-63, so every consecutive log step is
// log(1 / 0.9) = 0.10536 except the one drop at 31 -> 32 (p90 scale =
// 0.10536). Flat rms and edge dB (no other gate fires).
//
// Checks:
//   1. a bright -> dark cut-in the song does not make itself (i 31 -> j 40:
//      the landing's own predecessor is dark): collapse = log(0.285 / 0.95)
//      / 0.10536 = -11.427, gate 4.
//   2. the same drop where the song itself darkens (j = 32): excess 0, no gate.
//   3. a dark -> bright cut-in (rise): never charged.
//   4. gate 3 (edge cap) fires before gate 4 on the same pair.
//   5. without skip_energy_gate the collapse is not computed (0, gate != 4).
#include "remix/PairScorer.h"
#include "remix/Quality.h"
#include "remix/SignalNorm.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) { std::printf ("FAIL: %s — %s\n", #expr, msg); return false; }\
} while (0)

using namespace reamix::remix;

constexpr int kBeats = 64;
constexpr int kFeat  = 4;

struct Fixture
{
    std::vector<float>  features;
    std::vector<double> rms, cent, edge_end, edge_start;
    SignalBaselines     baselines;
    PairScorerTrack     track {};

    Fixture()
    {
        features.assign ((std::size_t) kBeats * kFeat, 0.5f);   // identical rows: cosine 1
        rms.assign ((std::size_t) kBeats, 0.5);
        edge_end.assign ((std::size_t) kBeats, -10.0);
        edge_start.assign ((std::size_t) kBeats, -10.0);
        for (int b = 0; b < kBeats; ++b)
        {
            const double base = b < 32 ? 1.0 : 0.3;
            cent.push_back (base * ((b % 2) == 0 ? 1.0 : 0.9));
        }
        baselines = buildSignalBaselines (rms.data(), cent.data(), nullptr, edge_end.data(), edge_start.data(), kBeats);
        track.n_total = kBeats; track.n_features = kFeat; track.features = features.data();
        track.edge_db_end = edge_end.data(); track.edge_db_start = edge_start.data();
        track.rms_energy = rms.data(); track.spectral_centroid = cent.data();
        track.ctx_lo = 0; track.ctx_hi = kBeats;
        track.v2 = true; track.baselines = &baselines; track.weights = &kV2QualityWeights;
        track.gate = PairGate::None;
    }

    PairScore open (int i, int j, bool skip_energy = true) const
    {
        PairScorerRequest req;
        req.abs_i = i; req.abs_j = j; req.bar_aligned = 1.0; req.boundary = true;
        req.skip_loudness_reject = true; req.skip_energy_gate = skip_energy;
        return scorePair (track, req);
    }
};

bool test1_collapse()
{
    Fixture f;
    CHECK (f.baselines.centroid.valid(), "63 consecutive steps build the centroid baseline");
    CHECK (std::fabs (f.baselines.centroid.scale() - std::log (1.0 / 0.9)) < 1e-9, "p90 scale = the alternation step");
    const PairScore s = f.open (31, 40);
    const double expected = std::log (0.285 / 0.95) / std::log (1.0 / 0.9);
    CHECK (std::fabs (s.centroid_collapse - expected) < 1e-9, "collapse = log(mean L / mean O) / scale with the song flat into the landing");
    CHECK (std::fabs (expected + 11.427) < 1e-3, "hand value -11.427");
    CHECK (s.rejected && s.gate == 4, "gate 4 fires below -kOpenSeamMaxCentroidCollapse");
    return true;
}

bool test2_song_darkens_itself()
{
    Fixture f;
    const PairScore s = f.open (31, 32);   // the landing's own predecessor is the bright part
    CHECK (! s.rejected, "the song's own drop is not charged");
    CHECK (std::fabs (s.centroid_collapse) < 1e-9, "excess 0: remix drop == natural drop");
    return true;
}

bool test3_rise_never_charged()
{
    Fixture f;
    const PairScore s = f.open (39, 8);
    CHECK (! s.rejected && s.gate == 0, "dark -> bright is a rise, no gate");
    CHECK (s.centroid_collapse > 11.0, "the value is positive (reported, not gated)");
    return true;
}

bool test4_edge_cap_first()
{
    Fixture f;
    f.edge_start[40] = 5.0;   // |end(31) - start(40)| = 15 dB > 12
    const PairScore s = f.open (31, 40);
    CHECK (s.rejected && s.gate == 3, "the edge cap (gate 3) precedes the collapse cap");
    CHECK (std::fabs (s.centroid_collapse) < 1e-12, "collapse not computed after an earlier gate");
    return true;
}

bool test5_not_open()
{
    Fixture f;
    const PairScore s = f.open (31, 40, /*skip_energy*/ false);
    CHECK (s.gate != 4 && std::fabs (s.centroid_collapse) < 1e-12, "the collapse cap is an open-seam gate only");
    return true;
}
} // namespace

int main()
{
    int failed = 0;
    auto run = [&] (const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::printf ("%s %s\n", ok ? "PASS" : "FAIL", name);
        if (! ok) ++failed;
    };
    run ("1 brightness collapse gate 4",       test1_collapse);
    run ("2 the song's own drop is free",      test2_song_darkens_itself);
    run ("3 a rise is never charged",          test3_rise_never_charged);
    run ("4 edge cap before collapse cap",     test4_edge_cap_first);
    run ("5 open-seam gate only",              test5_not_open);
    std::printf (failed == 0 ? "ALL PASS\n" : "%d FAILED\n", failed);
    return failed == 0 ? 0 : 1;
}
