// Parity: reamix::dsp::MelSpectrogramLibrosa vs librosa.feature.melspectrogram
// + librosa.power_to_db, from tools/dump_python_features.py.
//
// Inputs (per track):
//   references/golden/phase-2/dumps/<track>/y_audio.npy    float32, 1-D
//   references/golden/phase-2/dumps/<track>/mel_power.npy  float64, (n_mels, frames)
//   references/golden/phase-2/dumps/<track>/log_mel.npy    float64, (n_mels, frames)
//
// Success:
//   log_mel:   L∞ ≤ 1e-3 (absolute, formal VALIDATION.md threshold — log
//              domain compresses magnitude range so absolute gate holds).
//   mel_power: hybrid gate per ADR-011 — `max_diff ≤ 1e-3 OR
//              max_diff / mel_power_peak ≤ 5e-7`. The |STFT|² ULP floor is
//              `peak × 2 × 2^-23 ≈ peak × 2.4e-7`, so the absolute-1e-3
//              gate implicitly assumed peak < ~4200. Commercial masters
//              with peak > 7000 break that, without any port regression.
//              Ratio gate admits natural float32 behavior while still
//              rejecting real regressions (accidental float16 → ratio > 1e-4,
//              200× over). Headroom 2× vs observed max (2.67e-7 on Tiësto
//              9.2k peak) to absorb cross-track ULP-distribution variance.

#include "dsp/MelSpectrogramLibrosa.h"
#include "NpyIO.h"

#include <algorithm>
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

DiffStats compare2D(const reamix::test::NpyMatrixF64& python,
                    const std::vector<std::vector<float>>& cpp)
{
    // Python shape (rows, cols) = (n_mels, n_frames).
    // C++ layout [frame][mel]  → cpp[c][r] aligns with python.at(r, c).
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
    std::size_t mels = 0;
    double melPowerPeakPy = 0.0;
    DiffStats mel;
    DiffStats logMel;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath    = dumpDir / "y_audio.npy";
    const fs::path melPath  = dumpDir / "mel_power.npy";
    const fs::path logPath  = dumpDir / "log_mel.npy";
    if (!fs::exists(yPath) || !fs::exists(melPath) || !fs::exists(logPath)) {
        std::printf("[skip] %s — missing y_audio / mel_power / log_mel\n", r.name.c_str());
        return r;
    }

    auto y      = reamix::test::loadNpy1DFloat32(yPath.string());
    auto pyMel  = reamix::test::loadNpy2DFloat64(melPath.string());
    auto pyLog  = reamix::test::loadNpy2DFloat64(logPath.string());

    reamix::dsp::MelSpectrogramLibrosa mel;
    auto cppMel = mel.power(y);           // [frame][mel]
    auto cppLog = mel.powerToDb(cppMel);  // [frame][mel]

    r.frames = cppMel.size();
    r.mels = r.frames ? cppMel[0].size() : 0;

    if (pyMel.rows != r.mels || pyMel.cols != r.frames ||
        pyLog.rows != r.mels || pyLog.cols != r.frames) {
        std::printf("[FAIL] %s — shape mismatch (py mel %zux%zu, py log %zux%zu, cpp %zux%zu)\n",
                    r.name.c_str(), pyMel.rows, pyMel.cols,
                    pyLog.rows, pyLog.cols, r.mels, r.frames);
        return r;
    }

    double peak = 0.0;
    for (double v : pyMel.data) if (v > peak) peak = v;
    r.melPowerPeakPy = peak;
    r.mel = compare2D(pyMel, cppMel);
    r.logMel = compare2D(pyLog, cppLog);
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

    // Formal VALIDATION.md thresholds. ADR-011: mel_power gains a hybrid
    // ratio-based gate (float32 |STFT|² ULP floor admits peak-proportional
    // drift); log_mel stays absolute-bounded (log compresses magnitude).
    constexpr double kAbsThreshold   = 1e-3;
    constexpr double kMelRatioFloor  = 5e-7;

    std::printf("%-28s  %6s  %4s  %12s  %12s  %12s  %12s  %12s  %12s  %s\n",
                "track", "frames", "mels",
                "mel_peak_py", "mel_max_diff", "mel_ratio", "mel_mean_diff",
                "log_max_diff", "log_mean_diff", "status");
    std::printf("%-28s  %6s  %4s  %12s  %12s  %12s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------", "----",
                "------------", "------------", "------------", "-------------",
                "------------", "-------------", "------");

    int ran = 0;
    int failures = 0;
    double overallMelMax = 0.0;
    double overallMelRatio = 0.0;
    double overallLogMax = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const double melRatio = (r.melPowerPeakPy > 0.0)
            ? r.mel.maxAbs / r.melPowerPeakPy
            : 0.0;

        // ADR-011 hybrid gate for mel_power: either absolute or ratio PASSES.
        const bool passMel = (r.mel.maxAbs <= kAbsThreshold)
                          || (melRatio    <= kMelRatioFloor);
        const bool passLog = r.logMel.maxAbs <= kAbsThreshold;
        const bool pass = passMel && passLog;
        if (!pass) ++failures;

        if (r.mel.maxAbs > overallMelMax)    overallMelMax   = r.mel.maxAbs;
        if (melRatio    > overallMelRatio)   overallMelRatio = melRatio;
        if (r.logMel.maxAbs > overallLogMax) overallLogMax   = r.logMel.maxAbs;

        const char* tag = pass
            ? ((r.mel.maxAbs <= kAbsThreshold) ? "PASS" : "PASS (ratio)")
            : "FAIL";

        std::printf("%-28s  %6zu  %4zu  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.frames, r.mels,
                    r.melPowerPeakPy, r.mel.maxAbs, melRatio, r.mel.meanAbs,
                    r.logMel.maxAbs, r.logMel.meanAbs, tag);
    }

    std::printf("\noverall mel_power max_diff = %.6e  max_ratio = %.4e  (abs %.0e or ratio %.0e per ADR-011)\n",
                overallMelMax, overallMelRatio, kAbsThreshold, kMelRatioFloor);
    std::printf("         log_mel   max_diff = %.6e  (abs %.0e)\n",
                overallLogMax, kAbsThreshold);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
