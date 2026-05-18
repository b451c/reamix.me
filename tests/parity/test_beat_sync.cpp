// Parity: reamix::analysis::BeatSync::sync vs `librosa.util.sync(data,
// beat_frames, aggregate=np.mean, pad=True, axis=-1)` on librosa 0.11.0,
// using dumps from tools/dump_python_features.py.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   mel_power.npy            float64, (n_mels=128, T)        — aggregation input
//   beat_frames.npy          int64,   (n_beats,)             — synthetic beats
//   beat_sync_mel_power.npy  float64, (n_mels, n_beats + 1)  — reference result
//
// The beat_frames list is a synthetic stride-20 fixture
// (np.arange(10, T, 20)); it does NOT come from a real beat tracker. Its
// purpose is to exercise librosa's fix_frames pad/dedupe (prepends {0, T},
// takes np.unique.astype(int)) plus the slice-mean aggregation on ~20-
// frame windows. Real beats from phase-1 would be thinner (~130 BPM ≈ 20
// frames at hop=512) so this is a realistic density.
//
// Why mean parity is expected to be near-bitwise (and not just near-floor):
// np.mean uses pairwise summation on float64 arrays; this port does a
// naive sequential sum. For typical slice sizes (< ~30 elements), the ULP
// difference between pairwise and naive is within 1-2 ULP of the peak
// mel_power, i.e. ~1e-12 on values of order 1e4. Four-plus orders under
// the formal 1e-3 gate.
//
// Formal pass threshold: L∞ ≤ 1e-3 (VALIDATION.md, phase-2).

#include "analysis/BeatSync.h"
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
                  const std::vector<double>& cpp)
{
    DiffStats s;
    double sum = 0.0;
    const std::size_t n = std::min(python.size(), cpp.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(python[i] - cpp[i]);
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
    std::size_t nMels   = 0;
    std::size_t tFrames = 0;
    std::size_t nBeats  = 0;
    std::size_t nSlices = 0;
    double peakAbs      = 0.0;
    DiffStats diff;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path melPath    = dumpDir / "mel_power.npy";
    const fs::path beatPath   = dumpDir / "beat_frames.npy";
    const fs::path refPath    = dumpDir / "beat_sync_mel_power.npy";
    if (!fs::exists(melPath) || !fs::exists(beatPath) || !fs::exists(refPath)) {
        std::printf("[skip] %s — missing mel_power / beat_frames / beat_sync_mel_power\n",
                    r.name.c_str());
        return r;
    }

    const auto mel       = reamix::test::loadNpy2DFloat64(melPath.string());
    const auto beats64   = reamix::test::loadNpy1DInt64(beatPath.string());
    const auto ref       = reamix::test::loadNpy2DFloat64(refPath.string());

    r.nMels   = mel.rows;
    r.tFrames = mel.cols;
    r.nBeats  = beats64.size();
    r.nSlices = ref.cols;

    if (ref.rows != r.nMels) {
        std::printf("[FAIL] %s — ref rows %zu != mel rows %zu\n",
                    r.name.c_str(), ref.rows, r.nMels);
        return r;
    }

    std::vector<int> beats(beats64.size());
    for (std::size_t i = 0; i < beats64.size(); ++i)
        beats[i] = static_cast<int>(beats64[i]);

    const auto result = reamix::analysis::BeatSync::sync(
        mel.data.data(), r.nMels, r.tFrames, beats);

    if (result.nSlices != r.nSlices) {
        std::printf("[FAIL] %s — cpp nSlices %zu != py nSlices %zu\n",
                    r.name.c_str(), result.nSlices, r.nSlices);
        return r;
    }

    double peak = 0.0;
    for (double v : ref.data) {
        const double a = std::abs(v);
        if (a > peak) peak = a;
    }

    r.peakAbs = peak;
    r.diff    = compare(ref.data, result.data);
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

    std::printf("%-28s  %6s  %6s  %6s  %6s  %12s  %12s  %12s  %s\n",
                "track", "n_mels", "T", "beats", "slices",
                "peak_abs", "max_diff", "mean_diff", "status");
    std::printf("%-28s  %6s  %6s  %6s  %6s  %12s  %12s  %12s  %s\n",
                "----------------------------", "------", "------", "------", "------",
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

        std::printf("%-28s  %6zu  %6zu  %6zu  %6zu  %12.4e  %12.4e  %12.4e  %s\n",
                    r.name.c_str(), r.nMels, r.tFrames, r.nBeats, r.nSlices,
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
