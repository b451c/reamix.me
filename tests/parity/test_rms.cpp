// Parity: reamix::dsp::RMS vs `librosa.feature.rms(y=y, hop_length=512)`
// (frame_length=2048, center=True, pad_mode='constant', dtype=float32)
// from tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   y_audio.npy  float32, (n_samples,)
//   rms.npy      float64, (n_frames,)
//
// We feed y_audio.npy directly into RMS::compute. Length convention:
//   n_frames = n_samples / hop_length + 1. Matches STFT / mel / onset
//   pipelines at hop=512 (clip 60 s × 22050 Hz → 2584 frames).
//
// Formal pass threshold: L∞ ≤ 1e-3 (VALIDATION.md, phase-2).

#include "dsp/RMS.h"
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

DiffStats compare(const std::vector<double>& python,
                  const std::vector<float>& cpp)
{
    DiffStats s;
    double sum = 0.0;
    const std::size_t n = std::min(python.size(), cpp.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(python[i] - static_cast<double>(cpp[i]));
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
    std::size_t frames  = 0;
    std::size_t samples = 0;
    double peakAbs      = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath   = dumpDir / "y_audio.npy";
    const fs::path rmsPath = dumpDir / "rms.npy";
    if (!fs::exists(yPath) || !fs::exists(rmsPath)) {
        std::printf("[skip] %s — missing y_audio / rms\n", r.name.c_str());
        return r;
    }

    auto pyY   = reamix::test::loadNpy1DFloat32(yPath.string());    // (n_samples,)
    auto pyRms = reamix::test::loadNpy1DFloat64(rmsPath.string());  // (n_frames,)

    r.samples = pyY.size();
    r.frames  = pyRms.size();

    reamix::dsp::RMS rms;
    auto cppRms = rms.compute(pyY);

    if (cppRms.size() != r.frames) {
        std::printf("[FAIL] %s — cpp rms length %zu != py frames %zu\n",
                    r.name.c_str(), cppRms.size(), r.frames);
        return r;
    }

    double peak = 0.0;
    for (double v : pyRms) {
        const double a = std::abs(v);
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(pyRms, cppRms);
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

    constexpr double kThreshold = 1e-3;

    std::printf("%-28s  %8s  %6s  %12s  %12s  %12s  %s\n",
                "track", "samples", "frames",
                "peak_abs", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %8s  %6s  %12s  %12s  %12s  %s\n",
                "----------------------------", "--------", "------",
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

        std::printf("%-28s  %8zu  %6zu  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.samples, r.frames,
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
