// Parity: reamix::dsp::DCT vs scipy.fft.dct(type=2, norm='ortho') as consumed
// by librosa.feature.mfcc. Isolates the DCT step: input is Python-computed
// log_mel (from tools/dump_python_features.py), output is the first M DCT-II
// coefficients which equal `mfcc_M.npy`.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   log_mel_ref1.npy  float64, (n_mels, frames) — power_to_db(mel, ref=1.0),
//                                                 matching librosa.feature.mfcc's
//                                                 internal call path (NOT log_mel.npy
//                                                 which uses ref=np.max for display).
//   mfcc_40.npy       float64, (40, frames)
//   mfcc_20.npy       float64, (20, frames)
//
// We load log_mel_ref1, cast down to float32 (lossless: dump was float32
// upcast), transpose to [frame][n_mels], run DCT(128, 40) and DCT(128, 20)
// per frame, and compare to the reference mfcc_* matrices.
//
// Why ref=1.0 variant: uniform shifts of log_mel only move DCT coefficient 0
// (cosine sums to 0 for k>0). Using log_mel (ref=np.max) would produce the
// same mfcc_{1..M-1} but mfcc_0 off by ~sqrt(N)·10·log10(mel_power.max()).
// Rather than special-casing k=0, we feed the variant that matches librosa's
// internal mfcc path exactly.
//
// Formal pass threshold: L∞ ≤ 1e-3 (VALIDATION.md).
// Realized target (per handover): ≤ 1e-6 on mfcc_* — log compresses range and
// DCT is a linear transform with small matrix in ortho normalization.

#include "dsp/DCT.h"
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

// python shape (rows, cols) = (n_mfcc, n_frames); cpp is [frame][mfcc].
DiffStats compare(const reamix::test::NpyMatrixF64& python,
                  const std::vector<std::vector<float>>& cpp)
{
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

    const fs::path logPath    = dumpDir / "log_mel_ref1.npy";
    const fs::path mfcc40Path = dumpDir / "mfcc_40.npy";
    const fs::path mfcc20Path = dumpDir / "mfcc_20.npy";
    if (!fs::exists(logPath) || !fs::exists(mfcc40Path) || !fs::exists(mfcc20Path)) {
        std::printf("[skip] %s — missing log_mel_ref1 / mfcc_40 / mfcc_20\n", r.name.c_str());
        return r;
    }

    auto pyLog    = reamix::test::loadNpy2DFloat64(logPath.string());
    auto pyMfcc40 = reamix::test::loadNpy2DFloat64(mfcc40Path.string());
    auto pyMfcc20 = reamix::test::loadNpy2DFloat64(mfcc20Path.string());

    const std::size_t nMels   = pyLog.rows;
    const std::size_t nFrames = pyLog.cols;

    if (pyMfcc40.rows != 40 || pyMfcc40.cols != nFrames ||
        pyMfcc20.rows != 20 || pyMfcc20.cols != nFrames) {
        std::printf("[FAIL] %s — shape mismatch (log %zux%zu, m40 %zux%zu, m20 %zux%zu)\n",
                    r.name.c_str(), pyLog.rows, pyLog.cols,
                    pyMfcc40.rows, pyMfcc40.cols, pyMfcc20.rows, pyMfcc20.cols);
        return r;
    }

    // Transpose (n_mels, n_frames) → [frame][n_mels] and cast float64→float32.
    // The dump's float64 is a lossless upcast of librosa's float32 log_mel,
    // so casting back recovers the exact float32 values seen internally by
    // scipy.fft.dct when it was called on S.
    std::vector<std::vector<float>> framedLog(
        nFrames, std::vector<float>(nMels, 0.0f));
    for (std::size_t f = 0; f < nFrames; ++f)
        for (std::size_t m = 0; m < nMels; ++m)
            framedLog[f][m] = static_cast<float>(pyLog.at(m, f));

    reamix::dsp::DCT dct40(static_cast<int>(nMels), 40);
    reamix::dsp::DCT dct20(static_cast<int>(nMels), 20);
    auto cpp40 = dct40.applyFrames(framedLog);
    auto cpp20 = dct20.applyFrames(framedLog);

    double peak40 = 0.0, peak20 = 0.0;
    for (double v : pyMfcc40.data) { double a = std::abs(v); if (a > peak40) peak40 = a; }
    for (double v : pyMfcc20.data) { double a = std::abs(v); if (a > peak20) peak20 = a; }

    r.frames = nFrames;
    r.peak40 = peak40;
    r.peak20 = peak20;
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
