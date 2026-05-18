// Parity: reamix::analysis::Recurrence::build vs
// python-source/analysis/recurrence.py::build_recurrence_matrix
// dumped as R_matrix.npy + 3 component aids (R_feat / R_chroma / R_homo)
// on the 10-track phase-2 corpus (real beats from librosa.beat.beat_track
// per ADR-018 consequences; billie_jean_real skipped — uses a distinct
// real-beats fixture that does NOT emit phase-3 keys).
//
// Inputs per track (under references/golden/phase-2/dumps/<track>/):
//   recurrence_feature_matrix.npy  f64, (n_beats, 59)   L2-normalized per row
//   R_matrix.npy                   f64, (n_beats, n_beats)  primary target
//   R_feat.npy                     f64, (n_beats, n_beats)  bisection
//   R_chroma.npy                   f64, (n_beats, n_beats)  bisection
//   R_homo.npy                     f64, (n_beats, n_beats)  bisection
//
// Gate: L∞ ≤ 1e-3 on R (phase-3 spec.md acceptance criterion).
// On failure, per-component diffs are printed so bisection is visible
// without re-running.

#include "analysis/Recurrence.h"
#include "NpyIO.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DiffStats
{
    double maxAbs  = 0.0;
    double meanAbs = 0.0;
    std::size_t count = 0;
};

DiffStats compareMatrix(const reamix::test::NpyMatrixF64& python,
                        const std::vector<double>& cpp)
{
    DiffStats s;
    double sum = 0.0;
    const std::size_t N = python.rows * python.cols;
    for (std::size_t i = 0; i < N; ++i) {
        const double d = std::abs(python.data[i] - cpp[i]);
        if (d > s.maxAbs) s.maxAbs = d;
        sum += d;
    }
    s.count = N;
    s.meanAbs = N ? sum / static_cast<double>(N) : 0.0;
    return s;
}

double matrixPeakAbs(const reamix::test::NpyMatrixF64& m)
{
    double peak = 0.0;
    for (double v : m.data) {
        const double a = std::abs(v);
        if (a > peak) peak = a;
    }
    return peak;
}

struct TrackResult
{
    std::string name;
    bool ran = false;
    int  nBeats = 0;
    double peakR = 0.0;
    DiffStats diffR;
    DiffStats diffFeat;
    DiffStats diffChroma;
    DiffStats diffHomo;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path featPath   = dumpDir / "recurrence_feature_matrix.npy";
    const fs::path rPath      = dumpDir / "R_matrix.npy";
    const fs::path rFeatPath  = dumpDir / "R_feat.npy";
    const fs::path rChromaPath= dumpDir / "R_chroma.npy";
    const fs::path rHomoPath  = dumpDir / "R_homo.npy";

    // billie_jean_real uses the --real-beats fixture and does NOT emit
    // phase-3 keys (compute_real_beats_features path, not compute_features).
    if (!fs::exists(featPath) || !fs::exists(rPath)) {
        std::printf("[skip] %s — no phase-3 recurrence dumps\n", r.name.c_str());
        return r;
    }

    const auto features = reamix::test::loadNpy2DFloat64(featPath.string());
    const auto pyR      = reamix::test::loadNpy2DFloat64(rPath.string());
    const auto pyFeat   = reamix::test::loadNpy2DFloat64(rFeatPath.string());
    const auto pyChroma = reamix::test::loadNpy2DFloat64(rChromaPath.string());
    const auto pyHomo   = reamix::test::loadNpy2DFloat64(rHomoPath.string());

    const int nBeats = static_cast<int>(features.rows);
    const int nFeat  = static_cast<int>(features.cols);
    if (pyR.rows != static_cast<std::size_t>(nBeats) ||
        pyR.cols != static_cast<std::size_t>(nBeats))
    {
        std::printf("[FAIL] %s — R shape mismatch (py %zux%zu, expected %dx%d)\n",
                    r.name.c_str(), pyR.rows, pyR.cols, nBeats, nBeats);
        return r;
    }

    // Downcast golden f64 feature matrix to f32 for the C++ API. The round-
    // trip is bit-exact (Python cast f32→f64 at dump time; reversing is
    // lossless because f64 exactly represents every f32 value).
    std::vector<float> featuresF32(features.data.size());
    for (std::size_t i = 0; i < features.data.size(); ++i)
        featuresF32[i] = static_cast<float>(features.data[i]);

    const auto res = reamix::analysis::Recurrence::build(
        featuresF32.data(), nBeats, nFeat,
        reamix::analysis::Recurrence::kDefaultK);

    r.nBeats     = res.nBeats;
    r.peakR      = matrixPeakAbs(pyR);
    r.diffR      = compareMatrix(pyR,      res.R);
    r.diffFeat   = compareMatrix(pyFeat,   res.rFeat);
    r.diffChroma = compareMatrix(pyChroma, res.rChroma);
    r.diffHomo   = compareMatrix(pyHomo,   res.rHomo);
    r.ran = true;
    return r;
}

} // namespace

int main(int argc, char** argv)
{
    const fs::path root = (argc > 1)
        ? fs::path(argv[1])
        : fs::path("references/golden/phase-2/dumps");

    if (!fs::exists(root) || !fs::is_directory(root)) {
        std::fprintf(stderr, "dumps directory not found: %s\n", root.string().c_str());
        return 2;
    }

    std::vector<fs::path> trackDirs;
    for (auto& entry : fs::directory_iterator(root))
        if (entry.is_directory()) trackDirs.push_back(entry.path());
    std::sort(trackDirs.begin(), trackDirs.end());

    if (trackDirs.empty()) {
        std::fprintf(stderr, "no tracks in %s\n", root.string().c_str());
        return 2;
    }

    // Primary gate on R. Components printed as diagnostics only — individual
    // component thresholds would only surface on bisection, which a spike
    // on R would trigger manually.
    constexpr double kThresholdAbs = 1.0e-3;

    std::printf("%-28s  %6s  %10s  %12s  %12s  %12s  %12s  %s\n",
                "track", "nBeats",
                "peak(R)", "diff(R)", "diff(feat)", "diff(chroma)", "diff(homo)",
                "status");
    std::printf("%-28s  %6s  %10s  %12s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------",
                "----------", "------------", "------------",
                "------------", "------------", "------");

    int ran = 0, failures = 0;
    double overallMaxR = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool pass = r.diffR.maxAbs <= kThresholdAbs;
        if (!pass) ++failures;
        if (r.diffR.maxAbs > overallMaxR) overallMaxR = r.diffR.maxAbs;

        std::printf("%-28s  %6d  %10.4e  %12.4e  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.nBeats,
                    r.peakR,
                    r.diffR.maxAbs,
                    r.diffFeat.maxAbs,
                    r.diffChroma.maxAbs,
                    r.diffHomo.maxAbs,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall R max_diff = %.6e  (abs gate %.0e)\n",
                overallMaxR, kThresholdAbs);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
