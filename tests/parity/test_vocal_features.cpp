// Parity: reamix::analysis::VocalFeatures vs Python
// `_extract_vocal_features` (vocal_features.py L40-218) in default-mode
// (`vocal_use_pyin=False`, per ADR-014). 8 per-beat f64 output fields;
// 3 are constant-zero stubs that assert `== 0.0` bitwise.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   y_audio.npy                  f32 (n_samples,)
//   hpss_y_harmonic.npy          f32 (n_samples,)
//   stft_magnitude_harmonic.npy  f32 (1025, n_frames)
//   -- All three Python-side so parity isolates VocalFeatures orchestrator
//      drift from upstream HPSS + STFT drift.
//
//   vocal_activity.npy           f64 (n_beats,)
//   voiced_ratio.npy             f64 (n_beats,) — zeros (ADR-014 D4)
//   f0_hz.npy                    f64 (n_beats,) — zeros (ADR-014 D2)
//   f0_confidence.npy            f64 (n_beats,) — zeros (ADR-014 D3)
//   edge_vocal_activity_start.npy  f64 (n_beats,)
//   edge_vocal_activity_end.npy    f64 (n_beats,)
//   edge_vocal_onset_start.npy     f64 (n_beats,)
//   edge_vocal_release_end.npy     f64 (n_beats,)
//
// Synthetic beats per dump_python_features.py:768:
//   beat_frames = np.arange(10, T, 20)  where T = n_stft_frames (y/hop+1).
//
// Formal pass threshold: L∞ ≤ 1e-3 on each meaningful output;
// `== 0.0` bitwise on the three ADR-014 zero-stubs.

#include "analysis/VocalFeatures.h"
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

DiffStats compare(const std::vector<double>& py,
                  const std::vector<double>& cpp)
{
    DiffStats s;
    double sum = 0.0;
    const std::size_t n = std::min(py.size(), cpp.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(py[i] - cpp[i]);
        if (d > s.maxAbs) s.maxAbs = d;
        sum += d;
        ++s.count;
    }
    s.meanAbs = s.count ? sum / static_cast<double>(s.count) : 0.0;
    return s;
}

struct FieldReport {
    std::string name;
    DiffStats diff;
    double threshold;
    bool pass;
};

struct TrackResult {
    std::string name;
    bool ran = false;
    int nBeats = 0;
    int nFrames = 0;
    std::vector<FieldReport> fields;
};

// Transpose row-major (bins, frames) → [frame][bin] for VocalFeatures API.
std::vector<std::vector<float>>
transposeBinsFramesToFrameBin(const reamix::test::NpyMatrixF32& m)
{
    std::vector<std::vector<float>> out(m.cols, std::vector<float>(m.rows));
    for (std::size_t b = 0; b < m.rows; ++b) {
        for (std::size_t t = 0; t < m.cols; ++t) {
            out[t][b] = m.data[b * m.cols + t];
        }
    }
    return out;
}

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath     = dumpDir / "y_audio.npy";
    const fs::path yhPath    = dumpDir / "hpss_y_harmonic.npy";
    const fs::path magPath   = dumpDir / "stft_magnitude_harmonic.npy";
    const fs::path vaPath    = dumpDir / "vocal_activity.npy";
    if (!fs::exists(yPath) || !fs::exists(yhPath) || !fs::exists(magPath)
        || !fs::exists(vaPath))
    {
        std::printf("[skip] %s — missing vocal-features inputs\n", r.name.c_str());
        return r;
    }

    auto y    = reamix::test::loadNpy1DFloat32(yPath.string());
    auto yh   = reamix::test::loadNpy1DFloat32(yhPath.string());
    auto mag2 = reamix::test::loadNpy2DFloat32(magPath.string());  // (1025, T)
    auto mag  = transposeBinsFramesToFrameBin(mag2);

    // Synthetic beat frames: arange(10, T, 20), T = stft frame count.
    const int T = static_cast<int>(mag.size());
    std::vector<int> beatFrames;
    for (int v = 10; v < T; v += 20) beatFrames.push_back(v);
    const int nBeats = static_cast<int>(beatFrames.size());

    r.nFrames = T;
    r.nBeats  = nBeats;

    auto cpp = reamix::analysis::VocalFeatures::compute(
        y, yh, mag, /*sr=*/22050.0, beatFrames, nBeats);

    struct Spec {
        const char* name;
        const char* file;
        const std::vector<double>* cppVec;
        double threshold;
    };

    const std::vector<Spec> specs = {
        {"vocal_activity",            "vocal_activity.npy",
         &cpp.vocalActivity,          1e-3},
        {"voiced_ratio",              "voiced_ratio.npy",
         &cpp.voicedRatio,            0.0},   // ADR-014 D4: bitwise zero
        {"f0_hz",                     "f0_hz.npy",
         &cpp.f0Hz,                   0.0},   // ADR-014 D2: bitwise zero
        {"f0_confidence",             "f0_confidence.npy",
         &cpp.f0Confidence,           0.0},   // ADR-014 D3: bitwise zero
        {"edge_vocal_activity_start", "edge_vocal_activity_start.npy",
         &cpp.edgeVocalActivityStart, 1e-3},
        {"edge_vocal_activity_end",   "edge_vocal_activity_end.npy",
         &cpp.edgeVocalActivityEnd,   1e-3},
        {"edge_vocal_onset_start",    "edge_vocal_onset_start.npy",
         &cpp.edgeVocalOnsetStart,    1e-3},
        {"edge_vocal_release_end",    "edge_vocal_release_end.npy",
         &cpp.edgeVocalReleaseEnd,    1e-3},
    };

    for (const auto& s : specs) {
        const fs::path p = dumpDir / s.file;
        if (!fs::exists(p)) {
            std::printf("[skip field] %s / %s — missing dump\n",
                        r.name.c_str(), s.name);
            continue;
        }
        auto py = reamix::test::loadNpy1DFloat64(p.string());
        FieldReport fr;
        fr.name      = s.name;
        fr.diff      = compare(py, *s.cppVec);
        fr.threshold = s.threshold;
        fr.pass      = (fr.diff.maxAbs <= s.threshold);
        r.fields.push_back(fr);
    }
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
        std::fprintf(stderr, "dumps directory not found: %s\n",
                     root.string().c_str());
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

    std::printf("%-28s  %-28s  %6s  %12s  %12s  %12s  %s\n",
                "track", "field", "nBeats",
                "threshold", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %-28s  %6s  %12s  %12s  %12s  %s\n",
                "----------------------------", "----------------------------",
                "------", "------------", "------------", "------------", "------");

    int ran = 0, failures = 0;
    double overallMax = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        for (const auto& f : r.fields) {
            if (!f.pass) ++failures;
            if (f.diff.maxAbs > overallMax) overallMax = f.diff.maxAbs;
            std::printf("%-28s  %-28s  %6d  %12.2e  %12.4e  %12.4e  %s\n",
                        r.name.c_str(), f.name.c_str(), r.nBeats,
                        f.threshold, f.diff.maxAbs, f.diff.meanAbs,
                        f.pass ? "PASS" : "FAIL");
        }
    }

    std::printf("\noverall max_diff = %.6e\n", overallMax);
    std::printf("ran %d track(s), %d field failure(s)\n", ran, failures);
    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
