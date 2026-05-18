// Parity: reamix::dsp::STFT(y) vs librosa.stft(y) from tools/dump_python_features.py.
//
// Inputs (per track):
//   references/golden/phase-2/dumps/<track>/y_audio.npy        float32, 1-D
//   references/golden/phase-2/dumps/<track>/stft_magnitude.npy float64, (bins, frames)
//
// Success: L∞(|cpp_stft| - python_magnitude) on every golden case.
//   Formal threshold (VALIDATION.md): 1e-3.
//   Session target (realized): ≤ 1e-5. Python stored magnitude is effectively
//   float32 precision (np.abs(complex64)).astype(float64); pocketfft vs
//   scipy.fft on matching precision should hit ULP-level.

#include "dsp/STFT.h"
#include "NpyIO.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TrackResult {
    std::string name;
    std::size_t frames = 0;
    std::size_t bins = 0;
    double max_diff = 0.0;
    double mean_diff = 0.0;
    bool ran = false;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    fs::path yPath = dumpDir / "y_audio.npy";
    fs::path magPath = dumpDir / "stft_magnitude.npy";
    if (!fs::exists(yPath) || !fs::exists(magPath)) {
        std::printf("[skip] %s — missing y_audio.npy or stft_magnitude.npy\n", r.name.c_str());
        return r;
    }

    auto y = reamix::test::loadNpy1DFloat32(yPath.string());
    auto mag = reamix::test::loadNpy2DFloat64(magPath.string());

    reamix::dsp::STFT stft;
    auto cppMag = stft.magnitude(y);  // [frame][bin]

    const std::size_t nFrames = cppMag.size();
    const std::size_t nBins = nFrames ? cppMag[0].size() : 0;

    r.frames = nFrames;
    r.bins = nBins;

    if (mag.rows != nBins || mag.cols != nFrames) {
        std::printf("[FAIL] %s — shape mismatch: python (%zu, %zu) vs cpp (%zu, %zu)\n",
                    r.name.c_str(), mag.rows, mag.cols, nBins, nFrames);
        return r;
    }

    double maxD = 0.0;
    double sumD = 0.0;
    std::size_t count = 0;
    for (std::size_t bin = 0; bin < nBins; ++bin) {
        for (std::size_t f = 0; f < nFrames; ++f) {
            const double py = mag.at(bin, f);
            const double cpp = static_cast<double>(cppMag[f][bin]);
            const double d = std::abs(py - cpp);
            if (d > maxD) maxD = d;
            sumD += d;
            ++count;
        }
    }

    r.max_diff = maxD;
    r.mean_diff = count ? sumD / static_cast<double>(count) : 0.0;
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
    for (auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory())
            trackDirs.push_back(entry.path());
    }
    std::sort(trackDirs.begin(), trackDirs.end());

    if (trackDirs.empty()) {
        std::fprintf(stderr, "no tracks in %s\n", root.string().c_str());
        return 2;
    }

    // VALIDATION.md formal threshold; realized target is ≤ 1e-5.
    constexpr double kThreshold = 1e-3;

    int failures = 0;
    int ran = 0;
    double overallMax = 0.0;

    std::printf("%-30s  %6s  %5s  %14s  %14s  %s\n",
                "track", "frames", "bins", "max_diff", "mean_diff", "status");
    std::printf("%-30s  %6s  %5s  %14s  %14s  %s\n",
                "------------------------------", "------", "-----",
                "--------------", "--------------", "------");

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;
        overallMax = std::max(overallMax, r.max_diff);
        const bool pass = r.max_diff <= kThreshold;
        if (!pass) ++failures;
        std::printf("%-30s  %6zu  %5zu  %14.6e  %14.6e  %s\n",
                    r.name.c_str(), r.frames, r.bins, r.max_diff, r.mean_diff,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall max_diff = %.6e  (threshold %.0e)\n", overallMax, kThreshold);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
