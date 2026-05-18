// Parity: reamix::dsp::PitchTrack::estimateTuning vs
// `librosa.estimate_tuning(S=|STFT|², sr=22050, bins_per_octave=12)` —
// the exact scalar call `librosa.feature.chroma_stft(tuning=None)` makes
// internally (librosa/feature/spectral.py L1274-1275). See ADR-010.
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   stft_magnitude.npy   float64, (bins=1025, frames)   (|librosa.stft(y)|)
//   tuning_est.npy       float64, (1,) scalar           (reference)
//
// The C++ side reconstructs |STFT|² in float32 from stft_magnitude.npy
// (cast down — lossless on |·| since the python dump itself upcast
// float32→float64 after the abs) and squares in float32, matching
// librosa's internal complex64 → float32 abs → float32 square path.
// This isolates PitchTrack from any STFT numerical drift — pairing with
// test_stft, which bounds STFT independently.
//
// Formal pass threshold: |diff| ≤ 1e-4 (ADR-010 target). Floor: ≤ 0.01
// (one tuning-resolution bin = 1 cent) — any larger drift means a
// histogram boundary crossed, which is a real algorithmic divergence and
// deserves an ADR-008-pattern escape.

#include "dsp/PitchTrack.h"
#include "NpyIO.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TrackResult {
    std::string name;
    bool ran = false;
    std::size_t frames = 0;
    std::size_t bins = 0;
    double pyTuning = 0.0;
    double cppTuning = 0.0;
    double absDiff = 0.0;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path stftPath   = dumpDir / "stft_magnitude.npy";
    const fs::path tuningPath = dumpDir / "tuning_est.npy";
    if (!fs::exists(stftPath) || !fs::exists(tuningPath)) {
        std::printf("[skip] %s — missing stft_magnitude or tuning_est\n", r.name.c_str());
        return r;
    }

    auto pyStft   = reamix::test::loadNpy2DFloat64(stftPath.string());    // (bins, frames)
    auto pyTuning = reamix::test::loadNpy1DFloat64(tuningPath.string());  // (1,)

    if (pyTuning.size() != 1) {
        std::printf("[FAIL] %s — tuning_est shape != (1,)\n", r.name.c_str());
        return r;
    }

    r.bins   = pyStft.rows;
    r.frames = pyStft.cols;

    // S[frame][bin] = (|stft|[bin,frame] cast to f32) ² in float32.
    std::vector<std::vector<float>> sPower(
        r.frames, std::vector<float>(r.bins, 0.0f));
    for (std::size_t b = 0; b < r.bins; ++b) {
        for (std::size_t t = 0; t < r.frames; ++t) {
            const float m = static_cast<float>(pyStft.at(b, t));
            sPower[t][b] = m * m;
        }
    }

    reamix::dsp::PitchTrack pt;
    r.cppTuning = static_cast<double>(pt.estimateTuning(sPower, 22050.0f));
    r.pyTuning  = pyTuning[0];
    r.absDiff   = std::fabs(r.cppTuning - r.pyTuning);
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

    constexpr double kThreshold = 1e-4;
    constexpr double kFloor     = 0.01;  // one resolution bin = 1 cent

    std::printf("%-28s  %6s  %5s  %14s  %14s  %12s  %s\n",
                "track", "frames", "bins",
                "py_tuning", "cpp_tuning", "abs_diff", "status");
    std::printf("%-28s  %6s  %5s  %14s  %14s  %12s  %s\n",
                "----------------------------", "------", "-----",
                "--------------", "--------------",
                "------------", "------");

    int ran = 0, failures = 0;
    double overallMax = 0.0;
    int bitwiseCount = 0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool bitwise = (r.absDiff == 0.0);
        if (bitwise) ++bitwiseCount;
        const bool pass = (r.absDiff <= kThreshold)
                       || (r.absDiff <= kFloor);  // accept 1-bin drift as floor
        const bool strictPass = (r.absDiff <= kThreshold);
        if (!pass) ++failures;
        if (r.absDiff > overallMax) overallMax = r.absDiff;

        const char* status = bitwise  ? "PASS (bitwise)"
                           : strictPass ? "PASS"
                           : pass      ? "PASS (floor)"
                           : "FAIL";
        std::printf("%-28s  %6zu  %5zu  %+14.10f  %+14.10f  %12.4e  %s\n",
                    r.name.c_str(), r.frames, r.bins,
                    r.pyTuning, r.cppTuning, r.absDiff, status);
    }

    std::printf("\noverall max_diff = %.6e  (threshold %.0e, floor %.0e)\n",
                overallMax, kThreshold, kFloor);
    std::printf("ran %d track(s), %d bitwise, %d failure(s)\n",
                ran, bitwiseCount, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
