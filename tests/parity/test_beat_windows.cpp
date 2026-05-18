// Parity: reamix::analysis::BeatWindows vs feature_extractor.py (L317-463)
// step-9 helpers, using dumps from tools/dump_python_features.py.
//
// Four sub-tests per track, all L∞ ≤ 1e-3 (VALIDATION.md, phase-2):
//
//   edge_features — _extract_edge_features on mel_power (cast to f32), with
//     n_edge_frames=4. Output f64 [nBeats × 128], per-beat L2-normalized.
//     Two arrays diffed (start + end) — reported as the max of both.
//
//   edge_rms — _extract_edge_rms on y + beat_times, edge_ms=50.0. Output
//     f64 [nBeats], jointly [0,1]-normalized via shared max of start + end.
//     Two arrays diffed — reported as the max of both.
//
//   boundary_waveforms — _extract_waveform_snippets(pre=35, post=120).
//     Output f32 [nBeats × 3417] at sr=22050, per-row DC-remove + Hann +
//     RMS-normalize (1e-8 additive guard).
//
//   transition_waveforms — same as boundary with (pre=280, post=320).
//     Output f32 [nBeats × 13230].
//
// Inputs (per track, under references/golden/phase-2/dumps/<track>/):
//   mel_power.npy                f64, (n_mels=128, T)       — feature fixture
//   y_audio.npy                  f32, (n_samples,)          — raw audio
//   beat_frames.npy              i64, (n_beats,)            — stride-20 synth
//   beat_times.npy               f64, (n_beats,)            — beat_frames·hop/sr
//   edge_features_start.npy      f64, (n_beats, 128)        — ref
//   edge_features_end.npy        f64, (n_beats, 128)        — ref
//   edge_rms_start.npy           f64, (n_beats,)            — ref
//   edge_rms_end.npy             f64, (n_beats,)            — ref
//   boundary_waveforms.npy       f32, (n_beats, 3417)       — ref
//   transition_waveforms.npy     f32, (n_beats, 13230)      — ref
//
// Expected parity floors (pessimistic, per trap-scan analysis):
//
//   edge_features: ≤1e-12. n_edge = 4 → mean is pairwise-equivalent; only
//     the L2 sum-of-128 + sqrt contributes drift (BLAS dnrm2 vs naive sum).
//
//   edge_rms: ≤1e-6. Naive f32 sum on ~1102 samples (50 ms window) vs
//     numpy pairwise — ~log₂(1102) ≈ 10 ULP of sum, divided by 1102, under
//     sqrt; then divided by shared max of order 0.1.
//
//   boundary_waveforms: ≤1e-4 realized after RMS normalization on ~3417-
//     term sums. Budget: per-sample ULP drift in the f32 rms division
//     dominates, ~log₂(3417) ≈ 11 ULP on sumSq; rms then divides a snippet
//     whose post-normalization scale is O(1).
//
//   transition_waveforms: similar order; 13230-term sum = ~13 ULP drift.
//
// Formal pass threshold: L∞ ≤ 1e-3 across all four sub-tests, per VALIDATION.md.

#include "analysis/BeatWindows.h"
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

template <typename Tpy, typename Tcpp>
DiffStats compare(const std::vector<Tpy>& python,
                  const std::vector<Tcpp>& cpp)
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

struct SubResult {
    const char* name = "";
    bool ran = false;
    DiffStats diff;
};

struct TrackResult {
    std::string name;
    bool ran = false;
    std::size_t nBeats = 0;
    SubResult edgeFeat;
    SubResult edgeRms;
    SubResult boundary;
    SubResult transition;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path melPath          = dumpDir / "mel_power.npy";
    const fs::path yPath            = dumpDir / "y_audio.npy";
    const fs::path beatFramesPath   = dumpDir / "beat_frames.npy";
    const fs::path beatTimesPath    = dumpDir / "beat_times.npy";
    const fs::path edgeStartPath    = dumpDir / "edge_features_start.npy";
    const fs::path edgeEndPath      = dumpDir / "edge_features_end.npy";
    const fs::path edgeRmsStartPath = dumpDir / "edge_rms_start.npy";
    const fs::path edgeRmsEndPath   = dumpDir / "edge_rms_end.npy";
    const fs::path boundaryPath     = dumpDir / "boundary_waveforms.npy";
    const fs::path transitionPath   = dumpDir / "transition_waveforms.npy";

    for (const auto& p : {melPath, yPath, beatFramesPath, beatTimesPath,
                          edgeStartPath, edgeEndPath,
                          edgeRmsStartPath, edgeRmsEndPath,
                          boundaryPath, transitionPath}) {
        if (!fs::exists(p)) {
            std::printf("[skip] %s — missing %s\n",
                        r.name.c_str(), p.filename().string().c_str());
            return r;
        }
    }

    // Load shared inputs.
    const auto mel        = reamix::test::loadNpy2DFloat64(melPath.string());
    const auto y          = reamix::test::loadNpy1DFloat32(yPath.string());
    const auto beats64    = reamix::test::loadNpy1DInt64(beatFramesPath.string());
    const auto beatTimes  = reamix::test::loadNpy1DFloat64(beatTimesPath.string());
    std::vector<int> beatFrames(beats64.size());
    for (std::size_t i = 0; i < beats64.size(); ++i)
        beatFrames[i] = static_cast<int>(beats64[i]);
    r.nBeats = beats64.size();
    const int sr = 22050;

    // ---- Sub 1: edge_features (on mel_power cast to float32) -------------
    {
        r.edgeFeat.name = "edge_features";
        const auto refStart = reamix::test::loadNpy2DFloat64(edgeStartPath.string());
        const auto refEnd   = reamix::test::loadNpy2DFloat64(edgeEndPath.string());

        // Cast mel_power (f64 on disk) to the float32 dtype used in the
        // real pipeline's stacked_features (MFCC/chroma/contrast float32).
        std::vector<float> melF32(mel.data.size());
        for (std::size_t i = 0; i < mel.data.size(); ++i)
            melF32[i] = static_cast<float>(mel.data[i]);

        const auto out = reamix::analysis::BeatWindows::extractEdgeFeatures(
            melF32.data(), mel.rows, mel.cols, beatFrames, r.nBeats);

        DiffStats ds = compare(refStart.data, out.start);
        DiffStats de = compare(refEnd.data,   out.end);
        r.edgeFeat.diff.maxAbs  = std::max(ds.maxAbs, de.maxAbs);
        r.edgeFeat.diff.meanAbs = 0.5 * (ds.meanAbs + de.meanAbs);
        r.edgeFeat.diff.count   = ds.count + de.count;
        r.edgeFeat.ran = true;
    }

    // ---- Sub 2: edge_rms ------------------------------------------------
    {
        r.edgeRms.name = "edge_rms";
        const auto refStart = reamix::test::loadNpy1DFloat64(edgeRmsStartPath.string());
        const auto refEnd   = reamix::test::loadNpy1DFloat64(edgeRmsEndPath.string());

        const auto out = reamix::analysis::BeatWindows::extractEdgeRms(
            y.data(), y.size(), sr, beatTimes, r.nBeats);

        DiffStats ds = compare(refStart, out.start);
        DiffStats de = compare(refEnd,   out.end);
        r.edgeRms.diff.maxAbs  = std::max(ds.maxAbs, de.maxAbs);
        r.edgeRms.diff.meanAbs = 0.5 * (ds.meanAbs + de.meanAbs);
        r.edgeRms.diff.count   = ds.count + de.count;
        r.edgeRms.ran = true;
    }

    // ---- Sub 3: boundary_waveforms (pre=35, post=120) -------------------
    {
        r.boundary.name = "boundary";
        const auto ref = reamix::test::loadNpy2DFloat32(boundaryPath.string());

        const auto out = reamix::analysis::BeatWindows::extractWaveformSnippets(
            y.data(), y.size(), sr, beatTimes, r.nBeats,
            /*preMs=*/35.0, /*postMs=*/120.0);

        r.boundary.diff = compare(ref.data, out);
        r.boundary.ran  = true;
    }

    // ---- Sub 4: transition_waveforms (pre=280, post=320) ----------------
    {
        r.transition.name = "transition";
        const auto ref = reamix::test::loadNpy2DFloat32(transitionPath.string());

        const auto out = reamix::analysis::BeatWindows::extractWaveformSnippets(
            y.data(), y.size(), sr, beatTimes, r.nBeats,
            /*preMs=*/280.0, /*postMs=*/320.0);

        r.transition.diff = compare(ref.data, out);
        r.transition.ran  = true;
    }

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

    std::printf("%-28s  %6s  %14s  %14s  %14s  %14s  %s\n",
                "track", "beats",
                "edge_feat_max", "edge_rms_max",
                "boundary_max", "transition_max", "status");
    std::printf("%-28s  %6s  %14s  %14s  %14s  %14s  %s\n",
                "----------------------------", "------",
                "--------------", "--------------",
                "--------------", "--------------", "------");

    int ran = 0, failures = 0;
    double overallEdgeFeat = 0.0, overallEdgeRms = 0.0;
    double overallBoundary = 0.0, overallTransition = 0.0;

    for (const auto& d : trackDirs) {
        auto r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool pass =
            (r.edgeFeat.diff.maxAbs   <= kThreshold) &&
            (r.edgeRms.diff.maxAbs    <= kThreshold) &&
            (r.boundary.diff.maxAbs   <= kThreshold) &&
            (r.transition.diff.maxAbs <= kThreshold);
        if (!pass) ++failures;

        overallEdgeFeat   = std::max(overallEdgeFeat,   r.edgeFeat.diff.maxAbs);
        overallEdgeRms    = std::max(overallEdgeRms,    r.edgeRms.diff.maxAbs);
        overallBoundary   = std::max(overallBoundary,   r.boundary.diff.maxAbs);
        overallTransition = std::max(overallTransition, r.transition.diff.maxAbs);

        std::printf("%-28s  %6zu  %14.4e  %14.4e  %14.4e  %14.4e  %s\n",
                    r.name.c_str(), r.nBeats,
                    r.edgeFeat.diff.maxAbs, r.edgeRms.diff.maxAbs,
                    r.boundary.diff.maxAbs, r.transition.diff.maxAbs,
                    pass ? "PASS" : "FAIL");
    }

    std::printf("\noverall max_diff:\n");
    std::printf("  edge_features        = %.6e\n", overallEdgeFeat);
    std::printf("  edge_rms             = %.6e\n", overallEdgeRms);
    std::printf("  boundary_waveforms   = %.6e\n", overallBoundary);
    std::printf("  transition_waveforms = %.6e\n", overallTransition);
    std::printf("  threshold            = %.0e\n", kThreshold);
    std::printf("ran %d track(s), %d failure(s)\n", ran, failures);

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
