// Parity: reamix::analysis::NoveltyFeatures (end-to-end y → 25-dim features)
// vs StructureAnalyzer._extract_features (python-source/analysis/
// structure_analyzer.py L193-210) dumped as `novelty_features_25d.npy` by
// tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   y_audio.npy                 float32, 1-D                (pre-resampled PCM)
//   novelty_features_25d.npy    float32, (25, frames)        (reference)
//
// Exercises the full pipeline (STFT → MFCC(13) via mel+log+DCT →
// PitchTrack::estimateTuning → ChromaSTFT(12) → per-row z-score → vstack
// [mfcc*1.5, chroma]). ADR-019 gate: 1e-3 absolute OR 5e-7 ratio
// (ADR-011 escape precedent).

#include "analysis/NoveltyFeatures.h"
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
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    std::size_t count = 0;
};

DiffStats compare(const reamix::test::NpyMatrixF32& python,
                  const std::vector<std::vector<float>>& cpp)
{
    // python (rows, cols) = (25, n_frames); cpp is [frame][25].
    DiffStats s;
    double sum = 0.0;
    for (std::size_t r = 0; r < python.rows; ++r) {
        for (std::size_t c = 0; c < python.cols; ++c) {
            const double py = static_cast<double>(python.at(r, c));
            const double cv = static_cast<double>(cpp[c][r]);
            const double d = std::abs(py - cv);
            if (d > s.maxAbs) s.maxAbs = d;
            sum += d;
            ++s.count;
        }
    }
    s.meanAbs = s.count ? sum / static_cast<double>(s.count) : 0.0;
    return s;
}

struct TrackResult {
    std::string name;
    bool ran = false;
    std::size_t frames = 0;
    double peakAbs     = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath        = dumpDir / "y_audio.npy";
    const fs::path featPath     = dumpDir / "novelty_features_25d.npy";
    if (!fs::exists(yPath) || !fs::exists(featPath)) {
        std::printf("[skip] %s — missing y_audio / novelty_features_25d\n",
                    r.name.c_str());
        return r;
    }

    auto y       = reamix::test::loadNpy1DFloat32(yPath.string());
    auto pyFeat  = reamix::test::loadNpy2DFloat32(featPath.string());  // (25, T)

    auto cppFeat = reamix::analysis::NoveltyFeatures::extract(
        y.data(), y.size(), 22050);                                    // [T][25]

    r.frames = cppFeat.size();

    if (pyFeat.rows != static_cast<std::size_t>(
            reamix::analysis::NoveltyFeatures::kNFeat) ||
        pyFeat.cols != r.frames) {
        std::printf("[FAIL] %s — shape mismatch (py %zux%zu, cpp %zu frames × 25)\n",
                    r.name.c_str(), pyFeat.rows, pyFeat.cols, r.frames);
        return r;
    }

    double peak = 0.0;
    for (float v : pyFeat.data) {
        const double a = std::abs(static_cast<double>(v));
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(pyFeat, cppFeat);
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

    // ADR-019 dual gate: absolute OR ratio (ADR-011 escape precedent).
    constexpr double kThresholdAbs   = 1.0e-3;
    constexpr double kThresholdRatio = 5.0e-7;

    std::printf("%-28s  %6s  %12s  %12s  %12s  %12s  %s\n",
                "track", "frames",
                "peak_abs", "max_diff", "mean_diff", "ratio", "status");
    std::printf("%-28s  %6s  %12s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------",
                "------------", "------------",
                "------------", "------------", "------");

    int ran = 0, failures = 0;
    double overallMaxDiff  = 0.0;
    double overallMaxRatio = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const double ratio = (r.peakAbs > 0.0)
            ? (r.diff.maxAbs / r.peakAbs)
            : r.diff.maxAbs;

        const bool passAbs   = r.diff.maxAbs <= kThresholdAbs;
        const bool passRatio = ratio         <= kThresholdRatio;
        const bool pass      = passAbs || passRatio;

        if (!pass) ++failures;
        if (r.diff.maxAbs > overallMaxDiff)  overallMaxDiff  = r.diff.maxAbs;
        if (ratio         > overallMaxRatio) overallMaxRatio = ratio;

        std::printf("%-28s  %6zu  %12.4e  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.frames,
                    r.peakAbs, r.diff.maxAbs, r.diff.meanAbs, ratio,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall max_diff = %.6e  (abs gate %.0e)\n",
                overallMaxDiff, kThresholdAbs);
    std::printf("overall max_ratio = %.6e (ratio gate %.0e, ADR-011 escape)\n",
                overallMaxRatio, kThresholdRatio);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
