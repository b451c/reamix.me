// Parity: reamix::dsp::HPSS::harmonic vs `librosa.effects.hpss(y)[0]`
// from tools/dump_python_features.py (phase-2b session 2 extension).
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   y_audio.npy          float32, (nSamples,)
//   hpss_y_harmonic.npy  float32, (nSamples,)  — librosa.effects.hpss(y)[0]
//
// Formal pass threshold: L∞ ≤ 1e-4 on y_harmonic (phase-2b spec.md).
// Realized drift budget: STFT ~1e-5 |mag| drift propagates through
// softmask (pow 2, rescale) and then ~log₂(n_fft/hop)=2 overlap-add adds
// per output sample in ISTFT. Expected a few × 1e-5 on typical music.

#include "dsp/HPSS.h"
#include "NpyIO.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DiffStats {
    double maxAbs  = 0.0;
    double meanAbs = 0.0;
    std::size_t count = 0;
};

DiffStats compare(const std::vector<float>& python,
                  const std::vector<float>& cpp)
{
    DiffStats s;
    double sum = 0.0;
    const std::size_t n = std::min(python.size(), cpp.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(
            static_cast<double>(python[i]) - static_cast<double>(cpp[i]));
        if (d > s.maxAbs) s.maxAbs = d;
        sum += d;
        ++s.count;
    }
    s.meanAbs = s.count ? sum / static_cast<double>(s.count) : 0.0;
    return s;
}

struct TrackResult {
    std::string name;
    bool ran = false;
    std::size_t nSamples = 0;
    double peakAbs = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath    = dumpDir / "y_audio.npy";
    const fs::path harmPath = dumpDir / "hpss_y_harmonic.npy";
    if (!fs::exists(yPath) || !fs::exists(harmPath)) {
        std::printf("[skip] %s — missing y_audio / hpss_y_harmonic\n",
                    r.name.c_str());
        return r;
    }

    auto y      = reamix::test::loadNpy1DFloat32(yPath.string());
    auto pyHarm = reamix::test::loadNpy1DFloat32(harmPath.string());

    r.nSamples = y.size();
    if (pyHarm.size() != y.size()) {
        std::printf("[FAIL] %s — pyHarm length %zu != y length %zu\n",
                    r.name.c_str(), pyHarm.size(), y.size());
        return r;
    }

    auto cppHarm = reamix::dsp::HPSS::harmonic(y);
    if (cppHarm.size() != y.size()) {
        std::printf("[FAIL] %s — cpp harmonic length %zu != y length %zu\n",
                    r.name.c_str(), cppHarm.size(), y.size());
        return r;
    }

    double peak = 0.0;
    for (float v : pyHarm) {
        const double a = std::abs(static_cast<double>(v));
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(pyHarm, cppHarm);
    r.ran     = true;
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

    constexpr double kThreshold = 1e-4;

    std::printf("%-28s  %8s  %12s  %12s  %12s  %s\n",
                "track", "nSamples", "peak_abs", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %8s  %12s  %12s  %12s  %s\n",
                "----------------------------", "--------",
                "------------", "------------", "------------", "------");

    int ran = 0, failures = 0;
    double overallMax = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool pass = (r.diff.maxAbs <= kThreshold);
        if (!pass) ++failures;
        if (r.diff.maxAbs > overallMax) overallMax = r.diff.maxAbs;

        std::printf("%-28s  %8zu  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.nSamples,
                    r.peakAbs, r.diff.maxAbs, r.diff.meanAbs,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall max_diff = %.6e  (threshold %.0e)\n",
                overallMax, kThreshold);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
