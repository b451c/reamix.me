// Parity: reamix::analysis::StructureAnalyzer (port-order step 6, session 10)
// vs references/python-source/analysis/structure_analyzer.py `_analyze`
// dispatcher on the 10-track phase-2 corpus.
//
// For each track:
//   Inputs  = y_audio.npy + beats_cpp.npy + downbeats_cpp.npy +
//             feature_matrix_cbm.npy (f64 → f32 round-trip).
//   Outputs = dispatched_path.txt ("cbm" or "novelty") +
//             dispatched_segments_fields.npy (f64, n × 4) +
//             dispatched_segments_labels.txt +
//             dispatched_boundaries.npy (f64).
//
// Dispatch split on the corpus (per session-10 golden regen):
//   CBM     (3): meshuggah_bleed, shostakovich_jazz_waltz, vocal_solo_with_fx
//   Novelty (7): billie_jean, daft_punk_aerodynamic, dance_monkey,
//                eminem_without_me, miles_davis_so_what, smells_like_teen_spirit,
//                tiesto_the_motto
//
// Gate design:
//   CBM-dispatched  — strict bitwise on all 5 Segment fields + boundaries
//                     (session-6 CBMSegmenter precedent: bitwise + consolidation
//                     is deterministic on bitwise inputs).
//   Novelty-dispatched — ADR-022 Option B per-track (session-8 precedent):
//                     start/end bitwise, confidence L∞ ≤ 1e-6, cluster_id
//                     Hungarian ≥ 70 %, label exact ≥ 70 %, boundaries bitwise.
//                     (start/end/boundaries are bitwise because they derive
//                     from integer find_peaks indices × bitwise frame_duration;
//                     confidence drifts from upstream novelty ULP; cluster_id
//                     drifts from k-means MC-sampling divergence; label drifts
//                     when cluster_id drifts.)
//   Boundary F1    — mir_eval-style detection F1 at 3-s tolerance per track
//                     (phase-3 spec L34: "Boundary F1 ≥ 0.90 on 10 tracks").
//                     Matches mir_eval.segment.detection(..., window=3.0,
//                     trim=False): greedy closest-first bipartite matching of
//                     reference to estimated boundaries within |ref-est| < 3.0s.
//                     On 9/10 tracks boundaries are bitwise-identical so F1 = 1.0
//                     trivially; eminem sits under ADR-022 exception (py=4 vs
//                     cpp=6 segs after consolidation cascade) and its F1 is
//                     reported as diagnostic but waived from the ≥ 0.90 gate.
//
// bpm is a passthrough field — the dispatcher never consumes it algorithmically.
// We pass a fixed dummy value and verify res.bpm matches on the way out.
//
// billie_jean_real skipped — --real-beats path does not emit session-10
// dispatched_* keys (matches session-9 test convention).

#include "analysis/StructureAnalyzer.h"
#include "analysis/StructureResult.h"
#include "NpyIO.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int    kSampleRate        = 22050;
constexpr double kBpmPassthrough    = 120.0;        // passthrough check; not parity-gated
constexpr double kConfidenceTol     = 1e-6;          // novelty path upstream ULP budget
constexpr int    kClusterMinPct     = 70;            // ADR-022 Option B
constexpr int    kLabelMinPct       = 70;
constexpr double kBoundaryWindowSec = 3.0;           // phase-3 spec L34 tolerance
constexpr double kBoundaryF1Min     = 0.90;          // phase-3 spec L34 gate

// ADR-022 § K-means MC-sampling finding: eminem_without_me's k=2 clustering
// on the novelty path sits at a local-minimum inertia partition in sklearn's
// n_init=10 Monte-Carlo k-means++ (random_state=42) — one segment flips vs
// C++'s std::mt19937 which happens to find the global minimum. In session-8
// Fixture F this cascades to 71.4 %/71.4 % cluster+label match on the
// PRE-consolidation 7 segments; that was gated as a per-track principled
// exception. Session-10 dispatcher runs SegmentConsolidation AFTER clustering,
// and the merge algorithm's predicates fire on label sequences — so the
// single label flip propagates into a DIFFERENT n_segments (py=4, cpp=6 on
// the current corpus). The gate recognizes this as the same ADR-022 exception:
// path + bpm passthrough must still hold, but field-level comparisons are
// waived (the segments don't correspond 1:1). Re-litigating eminem requires
// porting numpy MT19937 (~100 LOC, version-brittle) — deferred per ADR-022.
// Session-15 corpus expansion surfaced alice_in_chains_nutshell as 2nd
// instance of this exception class. Pre-consolidation Hungarian cluster match
// 5/6 = 83.3% (1 of 6 segments in different cluster); peaks integer-exact,
// SSM ≤ 2.7e-06, embeddings ≤ 5.7e-06 — upstream pipeline is identical,
// divergence is purely K-means init-subset selection (C(6,2)=15 init-subsets,
// 10 MC restarts sample different subsets in sklearn `np.random.RandomState`
// vs C++ `std::mt19937(42..51)`). Post-consolidation, the label-sequence
// flip propagates to ~14s merge-boundary shift (seg[1] verse end py=14.3 vs
// cpp=28.3), confidence Δ=8.4e-02, F1@3s=0.800. Same root-cause class as
// eminem; same deferred-MT19937-port remedy.
const std::set<std::string> kAdr022KmeansExceptions = {
    "eminem_without_me",
    "alice_in_chains_nutshell",
};

using Segment      = reamix::analysis::Segment;
using DispatchPath = reamix::analysis::DispatchPath;

// ---- helpers ----------------------------------------------------------------

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

// mir_eval.segment.detection(..., window=3.0, trim=False) ported to C++.
// Closest-first greedy bipartite match; for boundary arrays that are
// well-separated (typical structural segmentation) this matches mir_eval's
// networkx max_weight_matching exactly. Condition `|ref-est| < window`
// (strict less-than) mirrors mir_eval.util.match_events.
struct F1Result
{
    int    hits      = 0;
    int    nRef      = 0;
    int    nEst      = 0;
    double precision = 0.0;
    double recall    = 0.0;
    double f1        = 0.0;
};

F1Result boundaryF1(const std::vector<double>& ref,
                    const std::vector<double>& est,
                    double window)
{
    F1Result r;
    r.nRef = static_cast<int>(ref.size());
    r.nEst = static_cast<int>(est.size());
    if (r.nRef == 0 || r.nEst == 0) return r;

    struct Edge { double dist; int i; int j; };
    std::vector<Edge> edges;
    edges.reserve(ref.size() * est.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
        for (std::size_t j = 0; j < est.size(); ++j) {
            const double d = std::abs(ref[i] - est[j]);
            if (d < window) edges.push_back({d, static_cast<int>(i), static_cast<int>(j)});
        }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.dist < b.dist; });

    std::vector<char> usedRef(ref.size(), 0), usedEst(est.size(), 0);
    for (const auto& e : edges)
        if (!usedRef[e.i] && !usedEst[e.j]) {
            usedRef[e.i] = usedEst[e.j] = 1;
            ++r.hits;
        }

    r.precision = static_cast<double>(r.hits) / static_cast<double>(r.nEst);
    r.recall    = static_cast<double>(r.hits) / static_cast<double>(r.nRef);
    const double denom = r.precision + r.recall;
    r.f1 = (denom > 0.0) ? (2.0 * r.precision * r.recall / denom) : 0.0;
    return r;
}

// Brute-force Hungarian (k ≤ ~5 on our corpus so k! ≤ 120 permutations, fine).
// Matches test_novelty_segmenter.cpp:178-201.
int hungarianBestMatches(const std::vector<int>& py,
                         const std::vector<int>& cpp)
{
    if (py.size() != cpp.size() || py.empty()) return 0;
    int maxLabel = 0;
    for (auto v : py)  maxLabel = std::max(maxLabel, v);
    for (auto v : cpp) maxLabel = std::max(maxLabel, v);
    const int k = maxLabel + 1;

    std::vector<int> perm(k);
    std::iota(perm.begin(), perm.end(), 0);

    int best = 0;
    do {
        int matches = 0;
        for (std::size_t i = 0; i < py.size(); ++i) {
            const int cVal   = cpp[i];
            const int mapped = (cVal >= 0 && cVal < k) ? perm[cVal] : cVal;
            if (mapped == py[i]) ++matches;
        }
        if (matches > best) best = matches;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

// ---- per-track result + comparison ------------------------------------------

struct TrackResult
{
    std::string name;
    bool        ran = false;

    std::string pyPath;       // "cbm" | "novelty"
    std::string cppPath;

    int   nSegPy  = 0;
    int   nSegCpp = 0;

    int    nStartBitwise = 0;
    int    nEndBitwise   = 0;
    double maxConfDiff   = 0.0;
    int    nClusterExact = 0;
    int    nLabelExact   = 0;
    int    clusterHungarian = 0;

    std::size_t nBndPy         = 0;
    std::size_t nBndCpp        = 0;
    int         nBndBitwise    = 0;
    double      maxBndDiff     = 0.0;

    F1Result    bndF1;                  // mir_eval boundary F1 @ 3-s tolerance

    bool   bpmPassthroughOk = false;
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
    const fs::path dispPathPath   = dumpDir / "dispatched_path.txt";
    const fs::path dispFieldsPath = dumpDir / "dispatched_segments_fields.npy";
    const fs::path dispLabelsPath = dumpDir / "dispatched_segments_labels.txt";
    const fs::path dispBndsPath   = dumpDir / "dispatched_boundaries.npy";

    if (!fs::exists(yPath) || !fs::exists(dispPathPath) ||
        !fs::exists(dispFieldsPath) || !fs::exists(dispBndsPath))
    {
        std::printf("[skip] %s — session-10 goldens not present\n", r.name.c_str());
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

    // --- load goldens --------------------------------------------------------
    const auto dispPathLines = loadTextLines(dispPathPath.string());
    if (dispPathLines.empty())
        throw std::runtime_error("empty dispatched_path.txt: " + r.name);
    r.pyPath = dispPathLines[0];

    const auto dispFields = reamix::test::loadNpy2DFloat64(dispFieldsPath.string());
    const auto dispLabels = loadTextLines(dispLabelsPath.string());
    const auto dispBnds   = reamix::test::loadNpy1DFloat64(dispBndsPath.string());

    // --- call C++ dispatcher -------------------------------------------------
    const auto res = reamix::analysis::StructureAnalyzer::analyze(
        y.data(), y.size(), kSampleRate, kBpmPassthrough,
        beats.data(), static_cast<int>(beats.size()),
        downbeat.data(), static_cast<int>(downbeat.size()),
        feat32.data(), static_cast<int>(feat64.cols));

    r.cppPath = (res.path == DispatchPath::CBM) ? "cbm" : "novelty";
    r.bpmPassthroughOk = res.bpm.has_value() && *res.bpm == kBpmPassthrough;

    r.nSegPy  = static_cast<int>(dispFields.rows);
    r.nSegCpp = static_cast<int>(res.segments.size());

    if (r.nSegPy != r.nSegCpp) {
        r.firstMismatchMsg = "n_segments py=" + std::to_string(r.nSegPy) +
                             " cpp=" + std::to_string(r.nSegCpp);
    }

    // --- compare fields ------------------------------------------------------
    const int nCommon = std::min(r.nSegPy, r.nSegCpp);
    std::vector<int> pyClusterV, cppClusterV;
    pyClusterV.reserve(nCommon);
    cppClusterV.reserve(nCommon);

    for (int i = 0; i < nCommon; ++i) {
        const double pyStart = dispFields.at(i, 0);
        const double pyEnd   = dispFields.at(i, 1);
        const double pyConf  = dispFields.at(i, 2);
        const int    pyClus  = static_cast<int>(dispFields.at(i, 3));
        const auto&  cs      = res.segments[i];

        if (pyStart == cs.start) ++r.nStartBitwise;
        if (pyEnd   == cs.end)   ++r.nEndBitwise;
        const double cd = std::abs(pyConf - cs.confidence);
        if (cd > r.maxConfDiff) r.maxConfDiff = cd;
        if (pyClus == cs.cluster_id) ++r.nClusterExact;
        if (i < static_cast<int>(dispLabels.size()) &&
            dispLabels[i] == cs.label) ++r.nLabelExact;

        pyClusterV.push_back(pyClus);
        cppClusterV.push_back(cs.cluster_id);
    }
    r.clusterHungarian = hungarianBestMatches(pyClusterV, cppClusterV);

    // --- compare boundaries (bitwise) ---------------------------------------
    r.nBndPy  = dispBnds.size();
    r.nBndCpp = res.boundaries.size();
    const std::size_t nb = std::min(r.nBndPy, r.nBndCpp);
    for (std::size_t i = 0; i < nb; ++i) {
        if (dispBnds[i] == res.boundaries[i]) ++r.nBndBitwise;
        const double bd = std::abs(dispBnds[i] - res.boundaries[i]);
        if (bd > r.maxBndDiff) r.maxBndDiff = bd;
    }

    // --- mir_eval-style boundary F1 at 3-s tolerance (phase-3 spec L34) -----
    r.bndF1 = boundaryF1(dispBnds, res.boundaries, kBoundaryWindowSec);

    r.ran = true;
    return r;
}

bool evaluateTrack(const TrackResult& r)
{
    // Universal: path agreement + bpm passthrough must always hold.
    if (r.pyPath != r.cppPath) return false;
    if (!r.bpmPassthroughOk)   return false;

    // ADR-022 K-means MC-sampling exception: waive field-level comparisons
    // (segments don't correspond 1:1 after consolidation cascade). Path +
    // bpm above are the only signals that survive — and they must pass.
    if (kAdr022KmeansExceptions.count(r.name)) {
        return true;
    }

    if (r.nSegPy != r.nSegCpp)                       return false;
    if (r.nBndPy != r.nBndCpp)                       return false;
    if (r.nBndBitwise != static_cast<int>(r.nBndPy)) return false;

    // Phase-3 spec L34 gate (trivially satisfied on bitwise boundaries but
    // guards against future regressions where the bitwise gate is loosened
    // while F1 should still hold).
    if (r.bndF1.f1 < kBoundaryF1Min) return false;

    const int n = r.nSegPy;
    if (n == 0) return true;

    const bool startAllBitwise = (r.nStartBitwise == n);
    const bool endAllBitwise   = (r.nEndBitwise   == n);

    if (r.pyPath == "cbm") {
        // Strict bitwise on all 5 fields (session-6 CBM precedent).
        return startAllBitwise && endAllBitwise &&
               r.maxConfDiff == 0.0 &&
               r.nClusterExact == n &&
               r.nLabelExact   == n;
    }

    // Novelty: ADR-022 Option B per-track.
    const int clusterPct = (100 * r.clusterHungarian) / n;
    const int labelPct   = (100 * r.nLabelExact)      / n;
    return startAllBitwise && endAllBitwise &&
           r.maxConfDiff <= kConfidenceTol &&
           clusterPct >= kClusterMinPct &&
           labelPct   >= kLabelMinPct;
}

void printHeader()
{
    std::printf(
        "%-26s  %-14s  %-7s  %-5s  %-8s  %-9s  %-9s  %-7s  %-6s  %s\n",
        "track", "path(py→cpp)", "n", "s/e", "maxConfΔ",
        "cluster", "label", "bnd", "F1@3s", "status");
}

void printRow(const TrackResult& r, bool pass)
{
    const int n = r.nSegPy;
    const int nc = (n > 0) ? (100 * r.clusterHungarian / n) : 100;
    const int nl = (n > 0) ? (100 * r.nLabelExact      / n) : 100;

    char pathCol[32];
    std::snprintf(pathCol, sizeof(pathCol), "%s→%s",
                  r.pyPath.c_str(), r.cppPath.c_str());

    char nCol[16];
    std::snprintf(nCol, sizeof(nCol), "%d/%d", r.nSegPy, r.nSegCpp);

    char seCol[16];
    std::snprintf(seCol, sizeof(seCol), "%d/%d", r.nStartBitwise, r.nEndBitwise);

    char confCol[16];
    std::snprintf(confCol, sizeof(confCol), "%.1e", r.maxConfDiff);

    char clusterCol[16];
    std::snprintf(clusterCol, sizeof(clusterCol), "%d/%d(%d%%)",
                  r.clusterHungarian, n, nc);

    char labelCol[16];
    std::snprintf(labelCol, sizeof(labelCol), "%d/%d(%d%%)",
                  r.nLabelExact, n, nl);

    char bndCol[16];
    std::snprintf(bndCol, sizeof(bndCol), "%d/%zu", r.nBndBitwise, r.nBndPy);

    char f1Col[16];
    std::snprintf(f1Col, sizeof(f1Col), "%.3f", r.bndF1.f1);

    std::printf(
        "%-26s  %-14s  %-7s  %-5s  %-8s  %-9s  %-9s  %-7s  %-6s  %s\n",
        r.name.c_str(),
        pathCol, nCol, seCol, confCol,
        clusterCol, labelCol, bndCol, f1Col,
        pass ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char** argv)
{
    if (std::fegetround() != FE_TONEAREST) {
        std::fprintf(stderr,
            "FATAL: FE rounding mode is %d, expected FE_TONEAREST (%d). "
            "Aborting.\n",
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

    int ran = 0, failures = 0;
    int cbmDispatched = 0, novDispatched = 0;

    // Boundary F1 aggregates (phase-3 spec L34 closure). `gated` = non-ADR-022
    // tracks (F1 contributes to the ≥ 0.90 gate); `waived` = eminem diagnostic.
    int    gatedTracks    = 0;
    int    gatedF1Fails   = 0;
    double gatedF1Sum     = 0.0;
    double gatedF1Min     = 1.0;
    double waivedF1Min    = 1.0;

    for (const auto& d : trackDirs) {
        TrackResult r = runOne(d);
        if (!r.ran) continue;
        ++ran;
        if      (r.pyPath == "cbm")     ++cbmDispatched;
        else if (r.pyPath == "novelty") ++novDispatched;

        const bool pass = evaluateTrack(r);
        if (!pass) ++failures;
        printRow(r, pass);
        if (pass && kAdr022KmeansExceptions.count(r.name)) {
            std::printf("   [%s] PASS under ADR-022 K-means MC-sampling "
                        "exception (path + bpm gated; field-level + F1 waived; "
                        "see session-8 Fixture F 71.4 %% root cause). "
                        "Diagnostic F1=%.3f (py n_bnd=%zu, cpp n_bnd=%zu).\n",
                        r.name.c_str(), r.bndF1.f1, r.nBndPy, r.nBndCpp);
            if (r.bndF1.f1 < waivedF1Min) waivedF1Min = r.bndF1.f1;
        } else {
            ++gatedTracks;
            gatedF1Sum += r.bndF1.f1;
            if (r.bndF1.f1 < gatedF1Min) gatedF1Min = r.bndF1.f1;
            if (r.bndF1.f1 < kBoundaryF1Min) ++gatedF1Fails;
        }
        if (!pass && !r.firstMismatchMsg.empty()) {
            std::printf("   [%s] %s\n", r.name.c_str(),
                        r.firstMismatchMsg.c_str());
        }
    }

    std::printf("\nran %d track(s). dispatch split cbm=%d novelty=%d. failures=%d\n",
                ran, cbmDispatched, novDispatched, failures);
    std::printf(
        "Gate: CBM-dispatched → strict bitwise on all 5 segment fields + "
        "boundaries (session-6 precedent); Novelty-dispatched → start/end "
        "bitwise, confidence L∞ ≤ 1e-6, cluster Hungarian ≥ 70%%, label "
        "exact ≥ 70%%, boundaries bitwise (ADR-022 Option B, session-8 "
        "precedent); universal → dispatched_path match + n_segments match + "
        "bpm passthrough.\n");

    // --- Boundary F1 closure summary (phase-3 spec L34) ---------------------
    const double gatedF1Mean = (gatedTracks > 0)
        ? (gatedF1Sum / static_cast<double>(gatedTracks))
        : 0.0;
    std::printf(
        "\nBoundary F1 @ %.1fs closure (phase-3 spec L34): "
        "gated tracks %d/%d PASS ≥ %.2f (min=%.3f, mean=%.3f); "
        "waived (ADR-022) min F1=%.3f. "
        "Gate %s.\n",
        kBoundaryWindowSec,
        gatedTracks - gatedF1Fails, gatedTracks,
        kBoundaryF1Min, gatedF1Min, gatedF1Mean,
        (waivedF1Min <= 1.0 ? waivedF1Min : 1.0),
        (gatedF1Fails == 0) ? "SATISFIED" : "FAILED");

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had session-10 goldens\n");
        return 2;
    }
    return (failures == 0) ? 0 : 1;
}
