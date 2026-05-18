// Parity: reamix::dsp::SpectralCentroid vs
// `librosa.feature.spectral_centroid(y=y, sr=22050, hop_length=512)`
// (n_fft=2048, window='hann', center=True, pad_mode='constant')
// from tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   stft_magnitude.npy    float64, (n_bins=1025, n_frames)
//   spectral_centroid.npy float64, (n_frames,)
//
// We feed Python's |STFT| magnitude directly (cast f64 → f32) into
// SpectralCentroid::compute, isolating centroid drift from upstream STFT
// drift (which has its own 1e-3 absolute-gate VALIDATION row). Same input
// class as test_spectral_contrast.
//
// Formal pass threshold: L∞ ≤ 1e-3 Hz (VALIDATION.md, phase-2).

#include "dsp/SpectralCentroid.h"
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
    std::size_t frames = 0;
    std::size_t bins   = 0;
    double peakAbs     = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path stftPath = dumpDir / "stft_magnitude.npy";
    const fs::path scPath   = dumpDir / "spectral_centroid.npy";
    if (!fs::exists(stftPath) || !fs::exists(scPath)) {
        std::printf("[skip] %s — missing stft_magnitude / spectral_centroid\n",
                    r.name.c_str());
        return r;
    }

    auto pyStft = reamix::test::loadNpy2DFloat64(stftPath.string()); // (bins, frames)
    auto pySc   = reamix::test::loadNpy1DFloat64(scPath.string());   // (frames,)

    r.bins   = pyStft.rows;
    r.frames = pyStft.cols;

    if (pySc.size() != r.frames) {
        std::printf("[FAIL] %s — centroid length %zu != stft frames %zu\n",
                    r.name.c_str(), pySc.size(), r.frames);
        return r;
    }

    // sAmp[frame][bin] = pyStft[bin, frame] cast to float32.
    // PARITY: dump emits float64 of an originally-float32 |STFT| output
    //         (librosa.abs(complex64) → float32) → cast is lossless.
    std::vector<std::vector<float>> sAmp(
        r.frames, std::vector<float>(r.bins, 0.0f));
    for (std::size_t k = 0; k < r.bins; ++k) {
        for (std::size_t t = 0; t < r.frames; ++t) {
            sAmp[t][k] = static_cast<float>(pyStft.at(k, t));
        }
    }

    reamix::dsp::SpectralCentroid sc;
    auto cppSc = sc.compute(sAmp, 22050.0f);

    if (cppSc.size() != r.frames) {
        std::printf("[FAIL] %s — cpp centroid length %zu != frames %zu\n",
                    r.name.c_str(), cppSc.size(), r.frames);
        return r;
    }

    double peak = 0.0;
    for (double v : pySc) {
        const double a = std::abs(v);
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(pySc, cppSc);
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

    std::printf("%-28s  %6s  %5s  %12s  %12s  %12s  %s\n",
                "track", "frames", "bins",
                "peak_hz", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %6s  %5s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------", "-----",
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

        std::printf("%-28s  %6zu  %5zu  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.frames, r.bins,
                    r.peakAbs, r.diff.maxAbs, r.diff.meanAbs,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall max_diff = %.6e Hz  (threshold %.0e)\n",
                overallMax, kThreshold);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
