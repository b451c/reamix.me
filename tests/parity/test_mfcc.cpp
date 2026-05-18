// Parity: reamix::analysis::Mfcc (end-to-end y → MFCC) vs
// `librosa.feature.mfcc(y=y, sr=22050, n_mfcc=N, n_fft=2048, hop_length=512)`
// from tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   y_audio.npy   float32, 1-D                (pre-resampled input PCM)
//   mfcc_40.npy   float64, (40, frames)       (reference — standard vector)
//   mfcc_20.npy   float64, (20, frames)       (reference — fast vector)
//
// Exercises the full pipeline (STFT → mel_basis → |·|² → power_to_db(ref=1.0)
// → DCT-II ortho → slice). Unit-level parity of MelSpectrogramLibrosa and
// DCT is covered by test_mel_librosa + test_dct; this test guards against
// chaining / ref-value / precision-dtype regressions in the MFCC wiring.
//
// Formal pass threshold: L∞ ≤ 1e-3 (VALIDATION.md).

#include "analysis/Mfcc.h"
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

DiffStats compare(const reamix::test::NpyMatrixF64& python,
                  const std::vector<std::vector<float>>& cpp)
{
    // python (rows, cols) = (n_mfcc, n_frames); cpp is [frame][mfcc].
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
    double peak40 = 0.0;
    double peak20 = 0.0;
    DiffStats mfcc40;
    DiffStats mfcc20;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath      = dumpDir / "y_audio.npy";
    const fs::path mfcc40Path = dumpDir / "mfcc_40.npy";
    const fs::path mfcc20Path = dumpDir / "mfcc_20.npy";
    if (!fs::exists(yPath) || !fs::exists(mfcc40Path) || !fs::exists(mfcc20Path)) {
        std::printf("[skip] %s — missing y_audio / mfcc_40 / mfcc_20\n", r.name.c_str());
        return r;
    }

    auto y        = reamix::test::loadNpy1DFloat32(yPath.string());
    auto pyMfcc40 = reamix::test::loadNpy2DFloat64(mfcc40Path.string());
    auto pyMfcc20 = reamix::test::loadNpy2DFloat64(mfcc20Path.string());

    reamix::analysis::Mfcc mfcc40(40);
    reamix::analysis::Mfcc mfcc20(20);
    auto cpp40 = mfcc40.compute(y);
    auto cpp20 = mfcc20.compute(y);

    r.frames = cpp40.size();

    if (pyMfcc40.rows != 40 || pyMfcc40.cols != r.frames ||
        pyMfcc20.rows != 20 || pyMfcc20.cols != r.frames) {
        std::printf("[FAIL] %s — shape mismatch (py40 %zux%zu, py20 %zux%zu, cpp40 %zu frames)\n",
                    r.name.c_str(), pyMfcc40.rows, pyMfcc40.cols,
                    pyMfcc20.rows, pyMfcc20.cols, r.frames);
        return r;
    }

    double p40 = 0.0, p20 = 0.0;
    for (double v : pyMfcc40.data) { double a = std::abs(v); if (a > p40) p40 = a; }
    for (double v : pyMfcc20.data) { double a = std::abs(v); if (a > p20) p20 = a; }

    r.peak40 = p40;
    r.peak20 = p20;
    r.mfcc40 = compare(pyMfcc40, cpp40);
    r.mfcc20 = compare(pyMfcc20, cpp20);
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

    constexpr double kThreshold = 1e-3;

    std::printf("%-28s  %6s  %12s  %12s  %12s  %12s  %s\n",
                "track", "frames",
                "m40_peak_abs", "m40_max_diff",
                "m20_peak_abs", "m20_max_diff", "status");
    std::printf("%-28s  %6s  %12s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------",
                "------------", "------------",
                "------------", "------------", "------");

    int ran = 0, failures = 0;
    double overallMax40 = 0.0, overallMax20 = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool pass = (r.mfcc40.maxAbs <= kThreshold)
                       && (r.mfcc20.maxAbs <= kThreshold);
        if (!pass) ++failures;
        if (r.mfcc40.maxAbs > overallMax40) overallMax40 = r.mfcc40.maxAbs;
        if (r.mfcc20.maxAbs > overallMax20) overallMax20 = r.mfcc20.maxAbs;

        std::printf("%-28s  %6zu  %12.4e  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.frames,
                    r.peak40, r.mfcc40.maxAbs,
                    r.peak20, r.mfcc20.maxAbs,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall mfcc40 max_diff = %.6e  mfcc20 max_diff = %.6e  (threshold %.0e)\n",
                overallMax40, overallMax20, kThreshold);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
