// Parity: reamix::dsp::ChromaSTFT vs
// `librosa.feature.chroma_stft(y=y, sr=22050, n_fft=2048, hop_length=512,
//  tuning=<python_estimate>)` from tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   stft_magnitude.npy  float64, (bins=1025, frames)   (|librosa.stft(y)|)
//   tuning_est.npy      float64, (1,)                  (Python tuning scalar)
//   chroma_stft.npy     float64, (12, frames)          (reference)
//
// The C++ side reconstructs |STFT|² in float32 from stft_magnitude.npy (cast
// down — lossless on |·| since the dump upcast float32→float64 after the abs)
// and squares in float32, matching librosa's internal complex64 → f32 abs →
// f32 square path (see tools/dump_python_features.py L158). This isolates
// ChromaSTFT from STFT drift, which has its own parity test.
//
// Tuning strategy: feed `tuning_est.npy[0]` (Python scalar, float64 → float
// at call) instead of rerunning `PitchTrack::estimateTuning` in-test. This
// gates step 4b on chroma alone — the full-stack
// y→STFT→|·|²→PitchTrack→ChromaSTFT chain is exercised later by the
// FeatureExtractor orchestrator (port-order step 10).
//
// Formal pass threshold: L∞ ≤ 1e-3 (VALIDATION.md, phase-2).

#include "dsp/ChromaSTFT.h"
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

DiffStats compare(const reamix::test::NpyMatrixF64& python,
                  const std::vector<std::vector<float>>& cpp)
{
    // python (rows, cols) = (n_chroma=12, n_frames); cpp is [frame][n_chroma].
    DiffStats s;
    double sum = 0.0;
    for (std::size_t r = 0; r < python.rows; ++r) {
        for (std::size_t c = 0; c < python.cols; ++c) {
            const double py = python.at(r, c);
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
    std::size_t bins   = 0;
    double pyTuning    = 0.0;
    double peakAbs     = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path stftPath   = dumpDir / "stft_magnitude.npy";
    const fs::path tuningPath = dumpDir / "tuning_est.npy";
    const fs::path chromaPath = dumpDir / "chroma_stft.npy";
    if (!fs::exists(stftPath) || !fs::exists(tuningPath) || !fs::exists(chromaPath)) {
        std::printf("[skip] %s — missing stft_magnitude / tuning_est / chroma_stft\n",
                    r.name.c_str());
        return r;
    }

    auto pyStft    = reamix::test::loadNpy2DFloat64(stftPath.string());   // (bins, frames)
    auto pyTuning  = reamix::test::loadNpy1DFloat64(tuningPath.string()); // (1,)
    auto pyChroma  = reamix::test::loadNpy2DFloat64(chromaPath.string()); // (12, frames)

    if (pyTuning.size() != 1) {
        std::printf("[FAIL] %s — tuning_est shape != (1,)\n", r.name.c_str());
        return r;
    }

    r.bins     = pyStft.rows;
    r.frames   = pyStft.cols;
    r.pyTuning = pyTuning[0];

    // S[frame][bin] = (|stft|[bin,frame] cast to f32)² in float32.
    std::vector<std::vector<float>> sPower(
        r.frames, std::vector<float>(r.bins, 0.0f));
    for (std::size_t b = 0; b < r.bins; ++b) {
        for (std::size_t t = 0; t < r.frames; ++t) {
            const float m = static_cast<float>(pyStft.at(b, t));
            sPower[t][b] = m * m;
        }
    }

    reamix::dsp::ChromaSTFT chroma;
    auto cppChroma = chroma.compute(
        sPower, static_cast<float>(r.pyTuning), 22050.0f);

    if (pyChroma.rows != 12 || pyChroma.cols != r.frames ||
        cppChroma.size() != r.frames) {
        std::printf("[FAIL] %s — shape mismatch (py %zux%zu, cpp %zu frames)\n",
                    r.name.c_str(), pyChroma.rows, pyChroma.cols, cppChroma.size());
        return r;
    }

    double peak = 0.0;
    for (double v : pyChroma.data) { const double a = std::abs(v); if (a > peak) peak = a; }

    r.peakAbs = peak;
    r.diff    = compare(pyChroma, cppChroma);
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

    std::printf("%-28s  %6s  %12s  %12s  %12s  %12s  %s\n",
                "track", "frames", "py_tuning",
                "peak_abs", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %6s  %12s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------",
                "------------", "------------",
                "------------", "------------", "------");

    int ran = 0, failures = 0;
    double overallMax = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool pass = (r.diff.maxAbs <= kThreshold);
        if (!pass) ++failures;
        if (r.diff.maxAbs > overallMax) overallMax = r.diff.maxAbs;

        std::printf("%-28s  %6zu  %+12.6f  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.frames, r.pyTuning,
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
