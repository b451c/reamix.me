// Parity: reamix::dsp::SpectralFlatness vs
// `librosa.feature.spectral_flatness(y=y_harmonic, n_fft=2048, hop_length=512,
// power=2.0, amin=1e-10)[0]` on librosa 0.11.0.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   stft_magnitude_harmonic.npy   float32, (1025, n_frames)
//     -- `np.abs(librosa.stft(y_harmonic, n_fft=2048, hop=512))` where
//        y_harmonic is the Python HPSS output. Feeding Python's magnitude
//        (not C++ STFT of HPSS output) isolates SpectralFlatness drift from
//        upstream STFT/HPSS ULP drift, same pattern as test_spectral_centroid.
//   spectral_flatness.npy          float32, (n_frames,)
//
// Formal pass threshold: L∞ ≤ 1e-6 (phase-2b spec § Parity targets).

#include "dsp/SpectralFlatness.h"
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
        const double d = std::abs(static_cast<double>(python[i])
                                  - static_cast<double>(cpp[i]));
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

    const fs::path magPath = dumpDir / "stft_magnitude_harmonic.npy";
    const fs::path fPath   = dumpDir / "spectral_flatness.npy";
    if (!fs::exists(magPath) || !fs::exists(fPath)) {
        std::printf("[skip] %s — missing stft_magnitude_harmonic / spectral_flatness\n",
                    r.name.c_str());
        return r;
    }

    auto pyMag  = reamix::test::loadNpy2DFloat32(magPath.string());  // (bins, frames)
    auto pyFlat = reamix::test::loadNpy1DFloat32(fPath.string());

    r.bins   = pyMag.rows;
    r.frames = pyMag.cols;

    // Transpose (bins, frames) row-major → [frame][bin] to match SpectralFlatness API.
    std::vector<std::vector<float>> mag(r.frames, std::vector<float>(r.bins));
    for (std::size_t b = 0; b < r.bins; ++b) {
        for (std::size_t t = 0; t < r.frames; ++t) {
            mag[t][b] = pyMag.data[b * r.frames + t];
        }
    }

    reamix::dsp::SpectralFlatness flatness;
    auto cppFlat = flatness.compute(mag);

    if (cppFlat.size() != r.frames) {
        std::printf("[FAIL] %s — cpp flatness length %zu != py %zu\n",
                    r.name.c_str(), cppFlat.size(), r.frames);
        return r;
    }

    double peak = 0.0;
    for (float v : pyFlat) {
        const double a = std::abs(static_cast<double>(v));
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(pyFlat, cppFlat);
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

    constexpr double kThreshold = 1e-6;

    std::printf("%-28s  %6s  %6s  %12s  %12s  %12s  %s\n",
                "track", "bins", "frames",
                "peak_abs", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %6s  %6s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------", "------",
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

        std::printf("%-28s  %6zu  %6zu  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.bins, r.frames,
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
