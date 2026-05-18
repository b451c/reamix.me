// Parity: reamix::analysis::CBMSegmenter vs
// python-source/analysis/cbm_segmenter.py on the 10-track phase-2 corpus.
//
// Fixture A — DP pipeline (session 5):
//   compute_bar_features + compute_bar_autosimilarity + cbm_segment.
//   Gate: cbm_bar_segments exact integer match on 10/10 tracks.
//
// Fixture B — labelSegments on GOLDEN inputs (session 6):
//   Tests the labeling algorithm (seg centroids → L2 norm → sim matmul →
//   greedy single-linkage → energy rank → position override) in isolation
//   from bar_energy drift. Feeds golden cbm_bar_features + cbm_bar_times
//   + cbm_bar_segments + cbm_bar_energy. Gate: start/end/confidence/
//   cluster_id/label exact match on 10/10.
//
// Fixture C — cbmAnalyze orchestrator (session 6):
//   End-to-end including C++ bar_energy computation from audio_mono.
//   Gate: all 5 segment fields + boundaries f64 bitwise match.
//   Also asserts bar_energy L∞ ≤ 1e-5 as a sanity bisection gate.
//
// Bisection aids printed on failure: cbm_seg_normed_centroids (f32) +
// cbm_seg_sim (f32) + cbm_bar_energy (f64).
//
// billie_jean_real is skipped — its --real-beats dump path does not emit
// phase-3-CBM keys (matches session-3 test_recurrence.cpp convention).

#include "analysis/CBMSegmenter.h"
#include "NpyIO.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DiffStats
{
    double maxAbs  = 0.0;
    double meanAbs = 0.0;
    std::size_t count = 0;
};

template <typename PyT, typename CppT>
DiffStats compareFlat(const PyT* py, const CppT* cpp, std::size_t N)
{
    DiffStats s;
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double d = std::abs(
            static_cast<double>(py[i]) - static_cast<double>(cpp[i]));
        if (d > s.maxAbs) s.maxAbs = d;
        sum += d;
    }
    s.count = N;
    s.meanAbs = N ? sum / static_cast<double>(N) : 0.0;
    return s;
}

struct SegmentDiff
{
    int  mismatches = 0;
    int  shapeCpp   = 0;
    int  shapePy    = 0;
};

SegmentDiff compareBarSegments(
    const reamix::test::NpyMatrixI64& pySegs,
    const std::vector<reamix::analysis::CBMSegmenter::BarSegment>& cppSegs)
{
    SegmentDiff d;
    d.shapePy  = static_cast<int>(pySegs.rows);
    d.shapeCpp = static_cast<int>(cppSegs.size());

    if (d.shapePy != d.shapeCpp) {
        d.mismatches = std::max(d.shapePy, d.shapeCpp);
        return d;
    }
    for (int i = 0; i < d.shapePy; ++i) {
        const std::int64_t pyStart = pySegs.at(i, 0);
        const std::int64_t pyEnd   = pySegs.at(i, 1);
        if (pyStart != cppSegs[i].start || pyEnd != cppSegs[i].end) {
            ++d.mismatches;
        }
    }
    return d;
}

// Session 6: read a label file dumped by tools/dump_python_features.py via
// _save_text — UTF-8, one label per line, trailing newline. Empty file =
// empty vector (degenerate-track case). No parser surface: std::getline
// respects \n, labels are ASCII (intro/outro/chorus/verse/bridge).
std::vector<std::string> loadTextLines(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open text file: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        // Strip trailing \r for CRLF safety (our writer uses \n only, but
        // belt-and-braces for cross-platform).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(std::move(line));
    }
    return out;
}

// Field-by-field comparison for Fixture B and C.
struct FieldDiff
{
    int  nStartMismatch      = 0;
    int  nEndMismatch        = 0;
    int  nConfidenceMismatch = 0;
    int  nClusterMismatch    = 0;
    int  nLabelMismatch      = 0;
    double maxAbsStart = 0.0;
    double maxAbsEnd   = 0.0;
    int  worstStartIdx = -1;
    int  worstEndIdx   = -1;
    std::string firstMismatchMsg;  // first divergence for diagnostics
};

FieldDiff compareSegmentFields(
    const reamix::test::NpyMatrixF64&     pyFields,   // (n_segs, 4) f64
    const std::vector<std::string>&       pyLabels,
    const std::vector<reamix::analysis::CBMSegmenter::Segment>& cpp)
{
    FieldDiff d;
    const int n = static_cast<int>(cpp.size());
    if (static_cast<int>(pyFields.rows) != n ||
        static_cast<int>(pyLabels.size()) != n)
    {
        d.firstMismatchMsg = "shape mismatch py=" +
            std::to_string(pyFields.rows) + "/" +
            std::to_string(pyLabels.size()) + " cpp=" +
            std::to_string(n);
        d.nStartMismatch = d.nEndMismatch = d.nConfidenceMismatch
            = d.nClusterMismatch = d.nLabelMismatch = std::max(
                static_cast<int>(pyFields.rows), n);
        return d;
    }
    for (int i = 0; i < n; ++i) {
        const double pyStart = pyFields.at(i, 0);
        const double pyEnd   = pyFields.at(i, 1);
        const double pyConf  = pyFields.at(i, 2);
        const int    pyClus  = static_cast<int>(pyFields.at(i, 3));

        if (pyStart != cpp[i].start) {
            ++d.nStartMismatch;
            const double ad = std::abs(pyStart - cpp[i].start);
            if (ad > d.maxAbsStart) { d.maxAbsStart = ad; d.worstStartIdx = i; }
            if (d.firstMismatchMsg.empty()) {
                char b[128];
                std::snprintf(b, sizeof(b),
                    "seg %d: start py=%.6f cpp=%.6f", i, pyStart, cpp[i].start);
                d.firstMismatchMsg = b;
            }
        }
        if (pyEnd != cpp[i].end) {
            ++d.nEndMismatch;
            const double ad = std::abs(pyEnd - cpp[i].end);
            if (ad > d.maxAbsEnd) { d.maxAbsEnd = ad; d.worstEndIdx = i; }
            if (d.firstMismatchMsg.empty()) {
                char b[128];
                std::snprintf(b, sizeof(b),
                    "seg %d: end py=%.6f cpp=%.6f", i, pyEnd, cpp[i].end);
                d.firstMismatchMsg = b;
            }
        }
        if (pyConf != cpp[i].confidence) {
            ++d.nConfidenceMismatch;
            if (d.firstMismatchMsg.empty()) {
                char b[128];
                std::snprintf(b, sizeof(b),
                    "seg %d: conf py=%.6f cpp=%.6f",
                    i, pyConf, cpp[i].confidence);
                d.firstMismatchMsg = b;
            }
        }
        if (pyClus != cpp[i].cluster_id) {
            ++d.nClusterMismatch;
            if (d.firstMismatchMsg.empty()) {
                char b[128];
                std::snprintf(b, sizeof(b),
                    "seg %d: cluster_id py=%d cpp=%d",
                    i, pyClus, cpp[i].cluster_id);
                d.firstMismatchMsg = b;
            }
        }
        if (pyLabels[i] != cpp[i].label) {
            ++d.nLabelMismatch;
            if (d.firstMismatchMsg.empty()) {
                d.firstMismatchMsg =
                    "seg " + std::to_string(i) +
                    ": label py=\"" + pyLabels[i] +
                    "\" cpp=\"" + cpp[i].label + "\"";
            }
        }
    }
    return d;
}

struct TrackResult
{
    std::string name;
    bool ran = false;

    // Fixture A — DP pipeline
    int  nBeatsCpp = 0;
    int  nBars     = 0;
    int  nSegsPy   = 0;
    int  nSegsCpp  = 0;
    int  dpMismatches = 0;
    DiffStats diffBarFeatures;
    DiffStats diffBarTimes;
    DiffStats diffAutosim;

    // Fixture B — labelSegments on golden inputs
    FieldDiff fixtureB;
    // Bisection aids (Fixture B context)
    DiffStats diffNormedCentroids;
    DiffStats diffSegSim;

    // Fixture C — full cbmAnalyze on audio
    FieldDiff fixtureC;
    DiffStats diffBarEnergy;
    DiffStats diffBoundaries;
    bool cAnalyzeSuccess = false;

    // Python cbm_analyze degenerate short-circuit (L406-408 `n_bars<4` or
    // L418-420 `len(bar_segments)<3`). Detected from dumped pyBarSegs.rows.
    // When set, Python golden `cbm_segments_fields.npy` has shape (0,4) and
    // `cbm_segments_labels.txt` is empty — Python never called labelSegments,
    // so Fixture B is N/A (no reference semantics to match), and Fixture C
    // must accept empty-both (cAnalyzeSuccess=false is the expected signal).
    // First exercised by session-15 alice_in_chains_nutshell (bar_segments=2).
    bool pythonCbmDegenerate = false;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    // ---- Fixture A inputs (session 5) -----------------------------------
    const fs::path featPath     = dumpDir / "feature_matrix_cbm.npy";
    const fs::path beatsPath    = dumpDir / "beats_cpp.npy";
    const fs::path dbPath       = dumpDir / "downbeats_cpp.npy";
    const fs::path barFeatPath  = dumpDir / "cbm_bar_features.npy";
    const fs::path barTimesPath = dumpDir / "cbm_bar_times.npy";
    const fs::path autosimPath  = dumpDir / "cbm_autosim.npy";
    const fs::path barSegsPath  = dumpDir / "cbm_bar_segments.npy";

    // ---- Fixture B / C inputs (session 6) -------------------------------
    const fs::path yAudioPath        = dumpDir / "y_audio.npy";
    const fs::path barEnergyPath     = dumpDir / "cbm_bar_energy.npy";
    const fs::path normedCentPath    = dumpDir / "cbm_seg_normed_centroids.npy";
    const fs::path segSimPath        = dumpDir / "cbm_seg_sim.npy";
    const fs::path segFieldsPath     = dumpDir / "cbm_segments_fields.npy";
    const fs::path segLabelsPath     = dumpDir / "cbm_segments_labels.txt";
    const fs::path boundariesPath    = dumpDir / "cbm_boundaries.npy";

    if (!fs::exists(featPath) || !fs::exists(barSegsPath)) {
        std::printf("[skip] %s — no phase-3 CBM dumps\n", r.name.c_str());
        return r;
    }

    // Load inputs. feature_matrix_cbm is saved as f64 (lossless upcast
    // from f32 production); cast back to f32 here — f32→f64→f32 round-trip
    // is identity.
    const auto features64 = reamix::test::loadNpy2DFloat64(featPath.string());
    const auto beats      = reamix::test::loadNpy1DFloat64(beatsPath.string());
    const auto downbeats  = reamix::test::loadNpy1DFloat64(dbPath.string());
    std::vector<float> features32(features64.data.size());
    for (std::size_t i = 0; i < features64.data.size(); ++i)
        features32[i] = static_cast<float>(features64.data[i]);
    const int nBeatsCpp = static_cast<int>(features64.rows);
    const int nFeat     = static_cast<int>(features64.cols);

    // Load Fixture A goldens.
    const auto pyBarFeat  = reamix::test::loadNpy2DFloat32(barFeatPath.string());
    const auto pyBarTimes = reamix::test::loadNpy1DFloat64(barTimesPath.string());
    const auto pyAutosim  = reamix::test::loadNpy2DFloat32(autosimPath.string());
    const auto pyBarSegs  = reamix::test::loadNpy2DInt64(barSegsPath.string());

    // --- Fixture A — DP pipeline -----------------------------------------
    const auto barRes = reamix::analysis::CBMSegmenter::computeBarFeatures(
        features32.data(), nBeatsCpp, nFeat,
        beats.data(), static_cast<int>(beats.size()),
        downbeats.data(), static_cast<int>(downbeats.size()));
    r.nBeatsCpp = nBeatsCpp;
    r.nBars     = barRes.nBars;

    if (static_cast<std::size_t>(r.nBars) != pyBarFeat.rows ||
        barRes.barFeatures.size() != pyBarFeat.data.size())
    {
        std::printf("[FAIL] %s — bar_features shape mismatch "
                    "(py %zux%zu, cpp %dx%d)\n",
                    r.name.c_str(),
                    pyBarFeat.rows, pyBarFeat.cols,
                    r.nBars, nFeat);
        return r;
    }
    r.diffBarFeatures = compareFlat(
        pyBarFeat.data.data(), barRes.barFeatures.data(),
        pyBarFeat.data.size());
    r.diffBarTimes = compareFlat(
        pyBarTimes.data(), barRes.barTimes.data(),
        pyBarTimes.size());

    const auto cppAutosim = reamix::analysis::CBMSegmenter::computeBarAutosimilarity(
        barRes.barFeatures, barRes.nBars, barRes.nFeat);
    r.diffAutosim = compareFlat(
        pyAutosim.data.data(), cppAutosim.data(), pyAutosim.data.size());

    const auto cppBarSegs = reamix::analysis::CBMSegmenter::cbmSegment(
        cppAutosim, barRes.nBars);
    const SegmentDiff segDiff = compareBarSegments(pyBarSegs, cppBarSegs);
    r.nSegsPy       = segDiff.shapePy;
    r.nSegsCpp      = segDiff.shapeCpp;
    r.dpMismatches  = segDiff.mismatches;

    // Fixtures B / C require session-6 aids + goldens.
    if (!fs::exists(segFieldsPath) || !fs::exists(segLabelsPath) ||
        !fs::exists(barEnergyPath) || !fs::exists(yAudioPath) ||
        !fs::exists(boundariesPath))
    {
        r.ran = true;  // Fixture A completed
        return r;
    }

    const auto pyBarEnergy    = reamix::test::loadNpy1DFloat64(barEnergyPath.string());
    const auto pyNormedCent   = reamix::test::loadNpy2DFloat32(normedCentPath.string());
    const auto pySegSim       = reamix::test::loadNpy2DFloat32(segSimPath.string());
    const auto pySegFields    = reamix::test::loadNpy2DFloat64(segFieldsPath.string());
    const auto pySegLabels    = loadTextLines(segLabelsPath.string());
    const auto pyBoundaries   = reamix::test::loadNpy1DFloat64(boundariesPath.string());
    const auto yAudio         = reamix::test::loadNpy1DFloat32(yAudioPath.string());

    // Python cbm_analyze L418-420 degenerate short-circuit check. When
    // bar_segments<3 Python returns None before step 5 (labelSegments), so
    // the labels golden is empty and Fixture B has no reference to match
    // against. Skip Fixture B; leave fixtureB at default-zero (trivial pass).
    r.pythonCbmDegenerate = (pyBarSegs.rows < 3);

    // --- Fixture B — labelSegments on golden inputs ----------------------
    // Build BarSegment vector from golden cbm_bar_segments.npy. DP-stage
    // drift (if any) does NOT contaminate this test.
    if (!r.pythonCbmDegenerate) {
        std::vector<reamix::analysis::CBMSegmenter::BarSegment> goldenBarSegs;
        goldenBarSegs.reserve(pyBarSegs.rows);
        for (std::size_t i = 0; i < pyBarSegs.rows; ++i) {
            goldenBarSegs.push_back({
                static_cast<int>(pyBarSegs.at(i, 0)),
                static_cast<int>(pyBarSegs.at(i, 1))
            });
        }
        // Use golden bar_features + bar_times (not C++-computed) so Fixture B
        // isolates the labeling algorithm from upstream ULP.
        std::vector<float> goldenBarFeats(pyBarFeat.data.begin(),
                                          pyBarFeat.data.end());
        std::vector<double> goldenBarTimes(pyBarTimes.begin(), pyBarTimes.end());
        std::vector<double> goldenBarEnergy(pyBarEnergy.begin(), pyBarEnergy.end());

        const auto segsB = reamix::analysis::CBMSegmenter::labelSegments(
            goldenBarSegs,
            goldenBarFeats,
            static_cast<int>(pyBarFeat.rows),
            static_cast<int>(pyBarFeat.cols),
            goldenBarTimes,
            goldenBarEnergy);
        r.fixtureB = compareSegmentFields(pySegFields, pySegLabels, segsB);
    }

    // --- Fixture C — cbmAnalyze orchestrator on audio --------------------
    const auto resC = reamix::analysis::CBMSegmenter::cbmAnalyze(
        features32.data(), nBeatsCpp, nFeat,
        beats.data(), static_cast<int>(beats.size()),
        downbeats.data(), static_cast<int>(downbeats.size()),
        yAudio.data(), yAudio.size(), /*sampleRate=*/22050);
    r.cAnalyzeSuccess = resC.success;
    r.fixtureC = compareSegmentFields(pySegFields, pySegLabels, resC.segments);

    // Bar-energy sanity gate (C++ pairwiseSumSquaresF32 vs numpy pairwise).
    if (resC.success) {
        // Need to re-compute bar_energy independently — cbmAnalyze doesn't
        // expose it. Regenerate via a second call-in path: we already know
        // the result from `pyBarEnergy`; construct a synthetic C++ compute
        // via the same formula to measure diff at THIS level. Instead of
        // plumbing a setter, we rely on the fact that labelSegments in
        // Fixture C consumed whatever bar_energy cbmAnalyze computed —
        // if that diverged materially from golden, we'd see label flips.
        // For an explicit gate we compute bar_energy here mirroring the
        // production path.
        const int nSamples = static_cast<int>(yAudio.size());
        std::vector<double> cppBarEnergy(pyBarEnergy.size(), 0.0);
        const int nBarsExpected = static_cast<int>(pyBarEnergy.size());
        const int lastIdx = static_cast<int>(pyBarTimes.size()) - 1;
        for (int i = 0; i < nBarsExpected; ++i) {
            const double tStart = pyBarTimes[i];
            const double tEnd   = pyBarTimes[std::min(i + 1, lastIdx)];
            std::int64_t s0 = static_cast<std::int64_t>(tStart * 22050.0);
            std::int64_t s1 = static_cast<std::int64_t>(tEnd   * 22050.0);
            if (s0 < 0) s0 = 0;
            if (s0 > static_cast<std::int64_t>(nSamples) - 1)
                s0 = static_cast<std::int64_t>(nSamples) - 1;
            if (s1 < s0 + 1) s1 = s0 + 1;
            if (s1 > static_cast<std::int64_t>(nSamples))
                s1 = static_cast<std::int64_t>(nSamples);
            const std::size_t n = static_cast<std::size_t>(s1 - s0);
            // Duplicates cbmAnalyze's inner loop exactly.
            // (pairwise f32 via recursive split, 8-leaf.)
            // Inline the recursion here as a lambda to avoid another
            // anonymous helper — mirrors CBMSegmenter.cpp::pairwiseSumSquaresF32.
            const float* xBase = yAudio.data() + s0;
            std::function<float(const float*, std::size_t)> pairF32 =
                [&pairF32](const float* x, std::size_t m) -> float {
                    if (m <= 8u) {
                        float s = 0.0f;
                        for (std::size_t j = 0; j < m; ++j) s += x[j] * x[j];
                        return s;
                    }
                    const std::size_t half = m / 2;
                    return pairF32(x, half) + pairF32(x + half, m - half);
                };
            const float sumSq = pairF32(xBase, n);
            const float meanSq = sumSq / static_cast<float>(n);
            cppBarEnergy[static_cast<std::size_t>(i)]
                = static_cast<double>(std::sqrt(meanSq));
        }
        r.diffBarEnergy = compareFlat(
            pyBarEnergy.data(), cppBarEnergy.data(), pyBarEnergy.size());
    }

    // Boundaries parity.
    if (!resC.boundaries.empty() && resC.boundaries.size() == pyBoundaries.size()) {
        r.diffBoundaries = compareFlat(
            pyBoundaries.data(), resC.boundaries.data(),
            pyBoundaries.size());
    } else if (resC.success) {
        r.diffBoundaries.maxAbs = std::numeric_limits<double>::infinity();
        r.diffBoundaries.count = std::max(resC.boundaries.size(), pyBoundaries.size());
    }

    // Bisection aids — normed_centroids + seg_sim vs Python goldens.
    // Reproduce the two intermediates inside the test so diagnostics are
    // available without re-running Python. (Same algebra the port uses.)
    {
        const int nSegs = static_cast<int>(pyBarSegs.rows);
        const int nBarsGolden = static_cast<int>(pyBarFeat.rows);
        const int nFeatGolden = static_cast<int>(pyBarFeat.cols);
        std::vector<float> cppCentroids(
            static_cast<std::size_t>(nSegs) * nFeatGolden, 0.0f);
        for (int s = 0; s < nSegs; ++s) {
            const int sb = static_cast<int>(pyBarSegs.at(s, 0));
            const int eb = std::min(
                static_cast<int>(pyBarSegs.at(s, 1)), nBarsGolden);
            float* dst = cppCentroids.data()
                       + static_cast<std::size_t>(s) * nFeatGolden;
            if (eb > sb) {
                for (int rr = sb; rr < eb; ++rr) {
                    const float* row = pyBarFeat.data.data()
                                     + static_cast<std::size_t>(rr) * nFeatGolden;
                    for (int k = 0; k < nFeatGolden; ++k) dst[k] += row[k];
                }
                const float denom = static_cast<float>(eb - sb);
                for (int k = 0; k < nFeatGolden; ++k) dst[k] /= denom;
            }
        }
        std::vector<float> cppNormed(cppCentroids.size(), 0.0f);
        for (int s = 0; s < nSegs; ++s) {
            const float* src = cppCentroids.data()
                             + static_cast<std::size_t>(s) * nFeatGolden;
            float sumSq = 0.0f;
            for (int k = 0; k < nFeatGolden; ++k) sumSq += src[k] * src[k];
            float norm = std::sqrt(sumSq);
            if (norm < 1e-8f) norm = 1.0f;
            float* dst = cppNormed.data()
                       + static_cast<std::size_t>(s) * nFeatGolden;
            for (int k = 0; k < nFeatGolden; ++k) dst[k] = src[k] / norm;
        }
        r.diffNormedCentroids = compareFlat(
            pyNormedCent.data.data(), cppNormed.data(),
            pyNormedCent.data.size());

        std::vector<float> cppSim(
            static_cast<std::size_t>(nSegs) * nSegs, 0.0f);
        for (int i = 0; i < nSegs; ++i) {
            const float* ri = cppNormed.data()
                            + static_cast<std::size_t>(i) * nFeatGolden;
            for (int j = 0; j < nSegs; ++j) {
                const float* rj = cppNormed.data()
                                + static_cast<std::size_t>(j) * nFeatGolden;
                float sum = 0.0f;
                for (int k = 0; k < nFeatGolden; ++k) sum += ri[k] * rj[k];
                cppSim[static_cast<std::size_t>(i) * nSegs + j] = sum;
            }
        }
        r.diffSegSim = compareFlat(
            pySegSim.data.data(), cppSim.data(), pySegSim.data.size());
    }

    r.ran = true;
    return r;
}

void printTrackRow(const TrackResult& r, bool pass)
{
    // Row: track | 5 bar-level ints | 5 field-mismatch ints | 4 diff doubles | status
    // Matches the 16-column header below.
    std::printf("%-28s  %6d  %5d  %5d  %5d  %5d  "
                "%5d  %5d  %5d  %5d  %5d  "
                "%10.3e  %10.3e  %10.3e  %10.3e  %s\n",
        r.name.c_str(),
        r.nBeatsCpp, r.nBars, r.nSegsPy, r.nSegsCpp, r.dpMismatches,
        r.fixtureB.nStartMismatch + r.fixtureB.nEndMismatch
            + r.fixtureB.nConfidenceMismatch + r.fixtureB.nClusterMismatch
            + r.fixtureB.nLabelMismatch,
        r.fixtureC.nStartMismatch + r.fixtureC.nEndMismatch
            + r.fixtureC.nConfidenceMismatch + r.fixtureC.nClusterMismatch
            + r.fixtureC.nLabelMismatch,
        r.fixtureC.nLabelMismatch,
        r.fixtureC.nClusterMismatch,
        r.fixtureC.nStartMismatch + r.fixtureC.nEndMismatch,
        r.diffNormedCentroids.maxAbs, r.diffSegSim.maxAbs,
        r.diffBarEnergy.maxAbs, r.diffBoundaries.maxAbs,
        pass ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char** argv)
{
    // FE_TONEAREST is the C/C++ default rounding mode per IEEE 754. CPython's
    // `round(x, 3)` is implemented via dtoa-based correct rounding, which
    // matches "%.3f" printf + strtod under FE_TONEAREST. If the platform or
    // compiler somehow changed the rounding mode, labelSegments' time field
    // would silently drift vs Python. Assert upfront.
    if (std::fegetround() != FE_TONEAREST) {
        std::fprintf(stderr,
            "FATAL: FE rounding mode is %d, expected FE_TONEAREST (%d). "
            "round3() will drift vs Python. Aborting.\n",
            std::fegetround(), FE_TONEAREST);
        return 3;
    }

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

    std::printf("%-28s  %6s  %5s  %5s  %5s  %5s  "
                "%5s  %5s  %5s  %5s  %5s  "
                "%10s  %10s  %10s  %10s  %s\n",
        "track", "nBeats", "nBars", "pySeg", "cpSeg", "dpMis",
        "Bmis", "Cmis", "Cfl",  "Ccid", "Cftm",
        "diffCent", "diffSim", "diffE", "diffBnd", "status");

    int ran = 0, failuresA = 0, failuresB = 0, failuresC = 0;
    bool allPass = true;

    for (const auto& d : trackDirs) {
        TrackResult r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool passA = (r.dpMismatches == 0 && r.nSegsPy == r.nSegsCpp);
        const bool passB = (r.fixtureB.nStartMismatch      == 0 &&
                            r.fixtureB.nEndMismatch        == 0 &&
                            r.fixtureB.nConfidenceMismatch == 0 &&
                            r.fixtureB.nClusterMismatch    == 0 &&
                            r.fixtureB.nLabelMismatch      == 0);
        // When Python cbm_analyze short-circuits (bar_segments<3), both sides
        // return empty; cAnalyzeSuccess=false is the expected signal, not an
        // error. Accept `pythonCbmDegenerate` as equivalent to cAnalyzeSuccess
        // for gate purposes. Field comparisons are still required (trivially
        // pass when both shapes are 0).
        const bool passC = ((r.cAnalyzeSuccess || r.pythonCbmDegenerate) &&
                            r.fixtureC.nStartMismatch      == 0 &&
                            r.fixtureC.nEndMismatch        == 0 &&
                            r.fixtureC.nConfidenceMismatch == 0 &&
                            r.fixtureC.nClusterMismatch    == 0 &&
                            r.fixtureC.nLabelMismatch      == 0 &&
                            r.diffBoundaries.maxAbs        == 0.0 &&
                            r.diffBarEnergy.maxAbs         <= 1e-5);
        const bool pass = passA && passB && passC;
        if (!pass) allPass = false;
        if (!passA) ++failuresA;
        if (!passB) ++failuresB;
        if (!passC) ++failuresC;

        printTrackRow(r, pass);

        // Diagnostics dump for first-mismatch on B or C.
        if (!passB && !r.fixtureB.firstMismatchMsg.empty()) {
            std::printf("   [B] %s\n", r.fixtureB.firstMismatchMsg.c_str());
        }
        if (!passC && !r.fixtureC.firstMismatchMsg.empty()) {
            std::printf("   [C] %s\n", r.fixtureC.firstMismatchMsg.c_str());
        }
    }

    std::printf("\nran %d track(s). Fixture A fail=%d, B fail=%d, C fail=%d\n",
                ran, failuresA, failuresB, failuresC);
    std::printf(
        "Column key: dpMis=Fixture A bar-segment mismatches; "
        "Bmis=Fixture B field mismatches (start+end+conf+cid+label); "
        "Cmis=Fixture C field mismatches; Cfl=Fixture C label mismatches; "
        "Ccid=Fixture C cluster_id mismatches; "
        "Cftm=Fixture C start+end mismatches; "
        "diffCent=max|normed_centroids_py - cpp|; "
        "diffSim=max|seg_sim_py - cpp|; "
        "diffE=max|bar_energy_py - cpp| (Fixture C, ≤ 1e-5 gate); "
        "diffBnd=max|boundaries_py - cpp| (Fixture C, = 0 gate).\n");

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had dumps to compare\n");
        return 2;
    }
    return allPass ? 0 : 1;
}
