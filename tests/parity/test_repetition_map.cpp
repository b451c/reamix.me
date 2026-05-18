// Parity: reamix::analysis::RepetitionMap (port-order step 8, session 11)
// vs references/python-source/analysis/repetition_map.py `build_repetition_map`
// on the 10-track phase-2 corpus.
//
// For each track:
//   Inputs  = y_audio.npy + beats_cpp.npy + downbeats_cpp.npy +
//             feature_matrix_cbm.npy (f64 → f32 round-trip; same feature matrix
//             the dispatcher consumed) + dispatched_segments_fields.npy +
//             dispatched_segments_labels.txt (from session-10 StructureAnalyzer
//             output — the segments RepetitionMap consumes in production).
//             boundary_waveforms are computed ON-THE-FLY in C++ via
//             BeatWindows::extractWaveformSnippets(y, sr, beats_cpp, 35 ms, 120 ms)
//             which is phase-2 step-9 validated bitwise against Python's
//             _extract_waveform_snippets (no separate golden needed).
//   Outputs = repetition_{from_beat,to_beat}.npy (i64) +
//             repetition_{waveform_similarity,chroma_correlation}.npy (f64) +
//             repetition_alignment_lag_samples.npy (i64) +
//             repetition_{from_section_idx,to_section_idx,from_bar,to_bar}.npy (i64) +
//             repetition_section_pairs.npy (i64 [n,2]) +
//             repetition_counts.txt ("n_sections_scanned=N\nn_pairs_verified=M").
//
// Gate design (phase-3 spec L35 + session-11 empirical):
//   Universal (per-track):
//     - n_sections_scanned equal.
//     - n_pairs_verified equal.
//     - section_pairs equal as unordered set of (idx_a, idx_b) pairs.
//   High-confidence agreement (the phase-3 acceptance gate):
//     - Set of jumps with waveform_similarity > 0.6 on the Python side.
//     - C++ must match at least 90 % of them (by (from_beat, to_beat,
//       from_section_idx, to_section_idx) composite key).
//     - For each matched pair: wf_sim L∞ ≤ 1e-6, chroma_corr L∞ ≤ 1e-6,
//       alignment_lag equal or off by at most 1 sample (FFT argmax ties).
//   Low-confidence (soft):
//     - Total n_jumps may differ by ULP drift at 0.08 / 0.10 diagonal
//       thresholds in phase 2. Reported for diagnostics, not gated.
//
//   ADR-022 exception:
//     - eminem_without_me — upstream cluster_id flip cascades through
//       dispatched segment labels which feed RepetitionMap's
//       _group_sections_by_label. from_section_idx / to_section_idx may
//       diverge from Python; the per-track gate is waived (same principled
//       exception as test_structure_analyzer / test_novelty_segmenter).

#include "analysis/RepetitionMap.h"
#include "analysis/StructureResult.h"
#include "analysis/BeatWindows.h"
#include "NpyIO.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int    kSampleRate        = 22050;
constexpr double kHighConfThreshold = 0.6;          // phase-3 spec L35
constexpr double kFieldTol          = 1e-6;
constexpr int    kMinAgreementPct   = 90;           // phase-3 spec L35
constexpr double kBoundaryPreMs     = 35.0;
constexpr double kBoundaryPostMs    = 120.0;

const std::set<std::string> kAdr022KmeansExceptions = {
    "eminem_without_me",
};

using reamix::analysis::RepetitionMap;
using reamix::analysis::RepetitionJump;
using reamix::analysis::RepetitionResult;
using reamix::analysis::Segment;

// ---- helpers ---------------------------------------------------------------

std::vector<std::string> loadTextLines(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open text file: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(std::move(line));
    }
    return out;
}

std::pair<int, int> parseCountsFile(const fs::path& p)
{
    int scanned = 0, verified = 0;
    const auto lines = loadTextLines(p.string());
    for (const auto& line : lines) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const int val = std::stoi(line.substr(eq + 1));
        if      (key == "n_sections_scanned") scanned  = val;
        else if (key == "n_pairs_verified")   verified = val;
    }
    return {scanned, verified};
}

// Key for matching jumps across Python and C++. Composite of all integer
// fields (waveform_sim / chroma_corr / lag are drift-prone; everything
// else is deterministic identity).
using JumpKey = std::tuple<int, int, int, int, int, int>;
// (fromBeat, toBeat, fromSectionIdx, toSectionIdx, fromBar, toBar)

JumpKey pyJumpKey(std::size_t i,
                  const std::vector<std::int64_t>& fromBeat,
                  const std::vector<std::int64_t>& toBeat,
                  const std::vector<std::int64_t>& fromSec,
                  const std::vector<std::int64_t>& toSec,
                  const std::vector<std::int64_t>& fromBar,
                  const std::vector<std::int64_t>& toBar)
{
    return {static_cast<int>(fromBeat[i]), static_cast<int>(toBeat[i]),
            static_cast<int>(fromSec[i]),  static_cast<int>(toSec[i]),
            static_cast<int>(fromBar[i]),  static_cast<int>(toBar[i])};
}

JumpKey cppJumpKey(const RepetitionJump& j)
{
    return {j.fromBeat, j.toBeat, j.fromSectionIdx, j.toSectionIdx,
            j.fromBar,  j.toBar};
}

// ---- per-track result + comparison ----------------------------------------

struct TrackResult
{
    std::string name;
    bool        ran = false;

    int scanPy = 0, scanCpp = 0;
    int verPy  = 0, verCpp  = 0;
    std::size_t nJumpPy = 0, nJumpCpp = 0;
    std::size_t nPairsPy = 0, nPairsCpp = 0;
    bool sectionPairsEqual = false;

    int nHighPy = 0, nHighCpp = 0;
    int nHighMatched = 0;                     // intersect of high-conf key sets
    int agreementPct = 100;

    double maxWfDiff    = 0.0;
    double maxChromaDiff = 0.0;
    int    maxLagDiff   = 0;

    std::string firstMismatchMsg;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    const fs::path yPath          = dumpDir / "y_audio.npy";
    const fs::path beatsPath      = dumpDir / "beats_cpp.npy";
    const fs::path downbeatsPath  = dumpDir / "downbeats_cpp.npy";
    const fs::path featurePath    = dumpDir / "feature_matrix_cbm.npy";
    const fs::path segFieldsPath  = dumpDir / "dispatched_segments_fields.npy";
    const fs::path segLabelsPath  = dumpDir / "dispatched_segments_labels.txt";
    const fs::path repFromBeat    = dumpDir / "repetition_from_beat.npy";
    const fs::path repToBeat      = dumpDir / "repetition_to_beat.npy";
    const fs::path repWfSim       = dumpDir / "repetition_waveform_similarity.npy";
    const fs::path repChromaCorr  = dumpDir / "repetition_chroma_correlation.npy";
    const fs::path repLag         = dumpDir / "repetition_alignment_lag_samples.npy";
    const fs::path repFromSec     = dumpDir / "repetition_from_section_idx.npy";
    const fs::path repToSec       = dumpDir / "repetition_to_section_idx.npy";
    const fs::path repFromBar     = dumpDir / "repetition_from_bar.npy";
    const fs::path repToBar       = dumpDir / "repetition_to_bar.npy";
    const fs::path repPairs       = dumpDir / "repetition_section_pairs.npy";
    const fs::path repCounts      = dumpDir / "repetition_counts.txt";

    if (!fs::exists(yPath) || !fs::exists(segFieldsPath)
        || !fs::exists(repFromBeat) || !fs::exists(repCounts))
    {
        std::printf("[skip] %s — session-11 goldens not present\n", r.name.c_str());
        return r;
    }

    // --- load inputs ---------------------------------------------------------
    const auto y        = reamix::test::loadNpy1DFloat32(yPath.string());
    const auto beats    = reamix::test::loadNpy1DFloat64(beatsPath.string());
    const auto downbeat = reamix::test::loadNpy1DFloat64(downbeatsPath.string());
    const auto feat64   = reamix::test::loadNpy2DFloat64(featurePath.string());
    std::vector<float> feat32(feat64.data.size());
    for (std::size_t i = 0; i < feat64.data.size(); ++i)
        feat32[i] = static_cast<float>(feat64.data[i]);

    const auto segFields = reamix::test::loadNpy2DFloat64(segFieldsPath.string());
    const auto segLabels = loadTextLines(segLabelsPath.string());

    // Build segment list for RepetitionMap input.
    std::vector<Segment> segs;
    segs.reserve(segFields.rows);
    for (std::size_t i = 0; i < segFields.rows; ++i) {
        Segment s;
        s.start      = segFields.at(i, 0);
        s.end        = segFields.at(i, 1);
        s.confidence = segFields.at(i, 2);
        s.cluster_id = static_cast<int>(segFields.at(i, 3));
        s.label      = (i < segLabels.size()) ? segLabels[i] : "unknown";
        segs.push_back(std::move(s));
    }

    // --- compute boundary_waveforms on beats_cpp (phase-2 step-9 parity) ---
    const auto boundaryWaves = reamix::analysis::BeatWindows::extractWaveformSnippets(
        y.data(), y.size(), kSampleRate, beats,
        /*nBeats=*/beats.size(),
        /*preMs=*/kBoundaryPreMs, /*postMs=*/kBoundaryPostMs);
    const int snippetLen =
        beats.empty() ? 0 : static_cast<int>(boundaryWaves.size() / beats.size());

    // --- call C++ RepetitionMap::build --------------------------------------
    const RepetitionResult res = RepetitionMap::build(
        beats.data(), static_cast<int>(beats.size()),
        downbeat.data(), static_cast<int>(downbeat.size()),
        feat32.data(), static_cast<int>(feat64.cols),
        segs,
        boundaryWaves.data(), snippetLen,
        kSampleRate);

    // --- load goldens --------------------------------------------------------
    const auto pyFromBeat = reamix::test::loadNpy1DInt64(repFromBeat.string());
    const auto pyToBeat   = reamix::test::loadNpy1DInt64(repToBeat.string());
    const auto pyWf       = reamix::test::loadNpy1DFloat64(repWfSim.string());
    const auto pyChroma   = reamix::test::loadNpy1DFloat64(repChromaCorr.string());
    const auto pyLag      = reamix::test::loadNpy1DInt64(repLag.string());
    const auto pyFromSec  = reamix::test::loadNpy1DInt64(repFromSec.string());
    const auto pyToSec    = reamix::test::loadNpy1DInt64(repToSec.string());
    const auto pyFromBar  = reamix::test::loadNpy1DInt64(repFromBar.string());
    const auto pyToBar    = reamix::test::loadNpy1DInt64(repToBar.string());
    const auto pyPairs    = reamix::test::loadNpy2DInt64(repPairs.string());
    const auto pyCounts   = parseCountsFile(repCounts);

    r.scanPy  = pyCounts.first;
    r.verPy   = pyCounts.second;
    r.scanCpp = res.nSectionsScanned;
    r.verCpp  = res.nPairsVerified;

    r.nJumpPy  = pyFromBeat.size();
    r.nJumpCpp = res.jumps.size();
    r.nPairsPy = pyPairs.rows;
    r.nPairsCpp = res.sectionPairs.size();

    // section_pairs set equality.
    std::set<std::pair<int, int>> pyPairsSet, cppPairsSet;
    for (std::size_t i = 0; i < pyPairs.rows; ++i) {
        pyPairsSet.emplace(static_cast<int>(pyPairs.at(i, 0)),
                           static_cast<int>(pyPairs.at(i, 1)));
    }
    for (const auto& p : res.sectionPairs) {
        cppPairsSet.emplace(p.first, p.second);
    }
    r.sectionPairsEqual = (pyPairsSet == cppPairsSet);

    // Build high-confidence key sets.
    std::set<JumpKey> pyHighSet, cppHighSet;
    for (std::size_t i = 0; i < pyFromBeat.size(); ++i) {
        if (pyWf[i] > kHighConfThreshold) {
            pyHighSet.insert(pyJumpKey(i, pyFromBeat, pyToBeat,
                                       pyFromSec, pyToSec,
                                       pyFromBar, pyToBar));
        }
    }
    for (const auto& j : res.jumps) {
        if (j.waveformSimilarity > kHighConfThreshold) {
            cppHighSet.insert(cppJumpKey(j));
        }
    }
    r.nHighPy  = static_cast<int>(pyHighSet.size());
    r.nHighCpp = static_cast<int>(cppHighSet.size());

    // Intersection count.
    std::set<JumpKey> inter;
    std::set_intersection(pyHighSet.begin(), pyHighSet.end(),
                          cppHighSet.begin(), cppHighSet.end(),
                          std::inserter(inter, inter.begin()));
    r.nHighMatched = static_cast<int>(inter.size());
    const int denom = std::max(r.nHighPy, r.nHighCpp);
    r.agreementPct = (denom == 0) ? 100 : (100 * r.nHighMatched / denom);

    // Field tolerance across matched jumps: walk Python jumps, find matching
    // C++ jump by composite key, compare scalars.
    std::vector<bool> cppUsed(res.jumps.size(), false);
    for (std::size_t i = 0; i < pyFromBeat.size(); ++i) {
        const JumpKey key = pyJumpKey(i, pyFromBeat, pyToBeat, pyFromSec, pyToSec,
                                      pyFromBar, pyToBar);
        for (std::size_t c = 0; c < res.jumps.size(); ++c) {
            if (cppUsed[c]) continue;
            if (cppJumpKey(res.jumps[c]) == key) {
                cppUsed[c] = true;
                const double wd = std::fabs(pyWf[i]     - res.jumps[c].waveformSimilarity);
                const double cd = std::fabs(pyChroma[i] - res.jumps[c].chromaCorrelation);
                const int    ld = std::abs(
                    static_cast<int>(pyLag[i]) - res.jumps[c].alignmentLagSamples);
                if (wd > r.maxWfDiff)     r.maxWfDiff = wd;
                if (cd > r.maxChromaDiff) r.maxChromaDiff = cd;
                if (ld > r.maxLagDiff)    r.maxLagDiff = ld;
                break;
            }
        }
    }

    r.ran = true;
    return r;
}

bool evaluateTrack(const TrackResult& r)
{
    // ADR-022 exception: waive gate if upstream cascade hits.
    if (kAdr022KmeansExceptions.count(r.name)) return true;

    if (r.scanPy != r.scanCpp) return false;
    if (r.verPy  != r.verCpp)  return false;
    if (!r.sectionPairsEqual)  return false;
    if (r.nPairsPy != r.nPairsCpp) return false;
    if (r.agreementPct < kMinAgreementPct) return false;

    // Field tolerance on matched jumps.
    if (r.maxWfDiff     > kFieldTol) return false;
    if (r.maxChromaDiff > kFieldTol) return false;
    if (r.maxLagDiff    > 1)         return false;

    return true;
}

void printHeader()
{
    std::printf(
        "%-26s  %-7s  %-5s  %-7s  %-9s  %-7s  %-9s  %-8s  %-7s  %s\n",
        "track", "scan", "ver", "pairs", "n_jumps",
        "high", "matched", "maxWfΔ", "maxLagΔ", "status");
}

void printRow(const TrackResult& r, bool pass)
{
    char scanCol[16], verCol[16], pairsCol[16], jumpsCol[16];
    char highCol[16], matchCol[16], wfCol[16], lagCol[16];
    std::snprintf(scanCol,  sizeof(scanCol),  "%d/%d", r.scanPy,  r.scanCpp);
    std::snprintf(verCol,   sizeof(verCol),   "%d/%d", r.verPy,   r.verCpp);
    std::snprintf(pairsCol, sizeof(pairsCol), "%zu/%zu", r.nPairsPy, r.nPairsCpp);
    std::snprintf(jumpsCol, sizeof(jumpsCol), "%zu/%zu", r.nJumpPy,  r.nJumpCpp);
    std::snprintf(highCol,  sizeof(highCol),  "%d/%d", r.nHighPy, r.nHighCpp);
    std::snprintf(matchCol, sizeof(matchCol), "%d(%d%%)",
                  r.nHighMatched, r.agreementPct);
    std::snprintf(wfCol,    sizeof(wfCol),    "%.1e", r.maxWfDiff);
    std::snprintf(lagCol,   sizeof(lagCol),   "%d",   r.maxLagDiff);
    std::printf(
        "%-26s  %-7s  %-5s  %-7s  %-9s  %-7s  %-9s  %-8s  %-7s  %s\n",
        r.name.c_str(),
        scanCol, verCol, pairsCol, jumpsCol, highCol, matchCol,
        wfCol, lagCol, pass ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char** argv)
{
    if (std::fegetround() != FE_TONEAREST) {
        std::fprintf(stderr,
            "FATAL: FE rounding mode is %d, expected FE_TONEAREST (%d).\n",
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

    printHeader();

    int ran = 0, failures = 0, totalHighPy = 0, totalHighMatched = 0;
    for (const auto& d : trackDirs) {
        const TrackResult r = runOne(d);
        if (!r.ran) continue;
        ++ran;
        totalHighPy      += r.nHighPy;
        totalHighMatched += r.nHighMatched;

        const bool pass = evaluateTrack(r);
        if (!pass) ++failures;
        printRow(r, pass);
        if (pass && kAdr022KmeansExceptions.count(r.name)) {
            std::printf("   [%s] PASS under ADR-022 K-means MC-sampling "
                        "exception (upstream cluster cascade; field-level "
                        "waived)\n", r.name.c_str());
        }
    }

    const int corpusPct = (totalHighPy == 0)
        ? 100
        : (100 * totalHighMatched / totalHighPy);
    std::printf("\nran %d track(s). corpus high-conf (wf>0.6) agreement = "
                "%d/%d (%d%%). failures=%d\n",
                ran, totalHighMatched, totalHighPy, corpusPct, failures);
    std::printf(
        "Gate: per-track n_sections_scanned / n_pairs_verified / "
        "section_pairs equal; ≥ 90%% high-confidence jump agreement by "
        "(from_beat, to_beat, from_section_idx, to_section_idx, from_bar, "
        "to_bar) key; matched-jump wf L∞ ≤ 1e-6, chroma L∞ ≤ 1e-6, lag "
        "within ±1; ADR-022 eminem_without_me waived.\n");

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had session-11 goldens\n");
        return 2;
    }
    return (failures == 0) ? 0 : 1;
}
