// Parity: reamix::dsp::OnsetStrength vs
// `librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)`
// from tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   mel_power.npy      float64, (n_mels=128, n_frames)
//   onset_strength.npy float64, (n_frames,)
//
// We feed Python's mel_power.npy directly (cast f64 → f32) into
// OnsetStrength::compute, isolating step-6 drift from upstream
// MelSpectrogramLibrosa::power drift (which has its own ADR-011 hybrid
// gate). Same pattern as test_chroma_stft / test_spectral_contrast.
//
// Formal pass threshold: L∞ ≤ 1e-3 (VALIDATION.md, phase-2).

#include "dsp/OnsetStrength.h"
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
    std::size_t mels   = 0;
    double peakAbs     = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path melPath   = dumpDir / "mel_power.npy";
    const fs::path onsetPath = dumpDir / "onset_strength.npy";
    if (!fs::exists(melPath) || !fs::exists(onsetPath)) {
        std::printf("[skip] %s — missing mel_power / onset_strength\n",
                    r.name.c_str());
        return r;
    }

    auto pyMel   = reamix::test::loadNpy2DFloat64(melPath.string());     // (mels, frames)
    auto pyOnset = reamix::test::loadNpy1DFloat64(onsetPath.string());   // (frames,)

    r.mels   = pyMel.rows;
    r.frames = pyMel.cols;

    if (pyOnset.size() != r.frames) {
        std::printf("[FAIL] %s — onset length %zu != mel frames %zu\n",
                    r.name.c_str(), pyOnset.size(), r.frames);
        return r;
    }

    // melPower[frame][mel] = pyMel[mel, frame] cast to f32.
    // PARITY: dump_python_features.py emits mel_power as float64 of an
    //         originally-float32 librosa output → cast back is lossless.
    std::vector<std::vector<float>> melPower(
        r.frames, std::vector<float>(r.mels, 0.0f));
    for (std::size_t m = 0; m < r.mels; ++m) {
        for (std::size_t t = 0; t < r.frames; ++t) {
            melPower[t][m] = static_cast<float>(pyMel.at(m, t));
        }
    }

    reamix::dsp::OnsetStrength os;
    auto cppOnset = os.compute(melPower);

    if (cppOnset.size() != r.frames) {
        std::printf("[FAIL] %s — cpp onset length %zu != frames %zu\n",
                    r.name.c_str(), cppOnset.size(), r.frames);
        return r;
    }

    double peak = 0.0;
    for (double v : pyOnset) {
        const double a = std::abs(v);
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(pyOnset, cppOnset);
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

    std::printf("%-28s  %6s  %12s  %12s  %12s  %s\n",
                "track", "frames", "peak_abs", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %6s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------",
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

        std::printf("%-28s  %6zu  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.frames,
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
