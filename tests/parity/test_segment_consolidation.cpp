// Parity: reamix::analysis::SegmentConsolidation vs
// references/python-source/analysis/segment_consolidation.py on the
// 10-track phase-2 corpus.
//
// Fixture A — CBM-path input (session 9):
//   Load golden cbm_segments_fields + cbm_segments_labels (session-6 dumps),
//   rebuild Segment list, call consolidate(input, duration), compare vs
//   cbm_consolidated_segments_fields + cbm_consolidated_segments_labels.
//
// Fixture B — Novelty-path input (session 9):
//   Same pattern on novelty_segments_* (session-8 dumps) →
//   novelty_consolidated_segments_*.
//
// Gate: strict bitwise on all 5 fields (start, end, confidence,
//       cluster_id, label). Consolidation is deterministic sequential
//       list processing + integer indexing; no RNG, no spectral divergence.
//
// Duration is computed from y_audio.npy at sr=22050 — matches dispatcher
// L119 `float(len(y)) / float(sr)` bit-exactly.
//
// billie_jean_real skipped — its --real-beats dump path does not emit
// phase-3 segment keys (matches test_recurrence / test_cbm_segmenter).

#include "analysis/SegmentConsolidation.h"
#include "NpyIO.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int    kSampleRate        = 22050;
constexpr double kSampleRateDouble  = static_cast<double>(kSampleRate);

using Segment = reamix::analysis::SegmentConsolidation::Segment;

// Read a label file dumped by tools/dump_python_features.py via _save_text —
// UTF-8, one label per line, trailing newline. Empty file = empty vector
// (degenerate-track case). Same helper as test_cbm_segmenter.cpp L100-113.
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

// Assemble a Segment vector from (fields f64, labels text) — matches the
// layout dumped by session-6/session-8 tools (rows of [start, end,
// confidence, cluster_id]).
std::vector<Segment>
assembleSegments(const reamix::test::NpyMatrixF64& fields,
                 const std::vector<std::string>&   labels)
{
    if (fields.rows != labels.size())
        throw std::runtime_error(
            "assembleSegments: fields rows=" + std::to_string(fields.rows) +
            " != labels size=" + std::to_string(labels.size()));
    if (fields.cols != 4 && fields.rows != 0)
        throw std::runtime_error(
            "assembleSegments: expected 4 columns, got " +
            std::to_string(fields.cols));

    std::vector<Segment> segs;
    segs.reserve(fields.rows);
    for (std::size_t i = 0; i < fields.rows; ++i) {
        segs.push_back(Segment{
            fields.at(i, 0),
            fields.at(i, 1),
            fields.at(i, 2),
            static_cast<int>(fields.at(i, 3)),
            labels[i],
        });
    }
    return segs;
}

// Field-by-field strict-bitwise comparison.
struct FieldDiff
{
    int  nStartMismatch      = 0;
    int  nEndMismatch        = 0;
    int  nConfidenceMismatch = 0;
    int  nClusterMismatch    = 0;
    int  nLabelMismatch      = 0;
    int  shapePy             = 0;
    int  shapeCpp            = 0;
    std::string firstMismatchMsg;
};

FieldDiff compareSegments(const reamix::test::NpyMatrixF64&   pyFields,
                          const std::vector<std::string>&     pyLabels,
                          const std::vector<Segment>&         cpp)
{
    FieldDiff d;
    d.shapePy  = static_cast<int>(pyFields.rows);
    d.shapeCpp = static_cast<int>(cpp.size());

    if (pyFields.rows != cpp.size() || pyLabels.size() != cpp.size()) {
        d.firstMismatchMsg = "shape mismatch py=" +
            std::to_string(pyFields.rows) + "/" +
            std::to_string(pyLabels.size()) + " cpp=" +
            std::to_string(cpp.size());
        const int n = std::max(d.shapePy, d.shapeCpp);
        d.nStartMismatch = d.nEndMismatch = d.nConfidenceMismatch
            = d.nClusterMismatch = d.nLabelMismatch = n;
        return d;
    }
    for (std::size_t i = 0; i < cpp.size(); ++i) {
        const double pyStart = pyFields.at(i, 0);
        const double pyEnd   = pyFields.at(i, 1);
        const double pyConf  = pyFields.at(i, 2);
        const int    pyClus  = static_cast<int>(pyFields.at(i, 3));

        auto recordFirst = [&](const std::string& m) {
            if (d.firstMismatchMsg.empty()) d.firstMismatchMsg = m;
        };

        if (pyStart != cpp[i].start) {
            ++d.nStartMismatch;
            char b[128];
            std::snprintf(b, sizeof(b),
                "seg %zu: start py=%.9f cpp=%.9f",
                i, pyStart, cpp[i].start);
            recordFirst(b);
        }
        if (pyEnd != cpp[i].end) {
            ++d.nEndMismatch;
            char b[128];
            std::snprintf(b, sizeof(b),
                "seg %zu: end py=%.9f cpp=%.9f",
                i, pyEnd, cpp[i].end);
            recordFirst(b);
        }
        if (pyConf != cpp[i].confidence) {
            ++d.nConfidenceMismatch;
            char b[160];
            std::snprintf(b, sizeof(b),
                "seg %zu: confidence py=%.17g cpp=%.17g (diff %.3e)",
                i, pyConf, cpp[i].confidence,
                std::abs(pyConf - cpp[i].confidence));
            recordFirst(b);
        }
        if (pyClus != cpp[i].cluster_id) {
            ++d.nClusterMismatch;
            char b[128];
            std::snprintf(b, sizeof(b),
                "seg %zu: cluster_id py=%d cpp=%d",
                i, pyClus, cpp[i].cluster_id);
            recordFirst(b);
        }
        if (pyLabels[i] != cpp[i].label) {
            ++d.nLabelMismatch;
            recordFirst(
                "seg " + std::to_string(i) +
                ": label py=\"" + pyLabels[i] +
                "\" cpp=\"" + cpp[i].label + "\"");
        }
    }
    return d;
}

struct TrackResult
{
    std::string name;
    bool ran = false;

    // Fixture A — CBM-path
    int  aSegsIn  = 0;
    int  aSegsPy  = 0;
    int  aSegsCpp = 0;
    FieldDiff fixtureA;

    // Fixture B — novelty-path
    int  bSegsIn  = 0;
    int  bSegsPy  = 0;
    int  bSegsCpp = 0;
    FieldDiff fixtureB;
};

TrackResult runOne(const fs::path& dumpDir)
{
    TrackResult r;
    r.name = dumpDir.filename().string();

    // Paths
    const fs::path yAudioPath       = dumpDir / "y_audio.npy";
    const fs::path cbmFieldsPath    = dumpDir / "cbm_segments_fields.npy";
    const fs::path cbmLabelsPath    = dumpDir / "cbm_segments_labels.txt";
    const fs::path cbmConsFieldsPath= dumpDir / "cbm_consolidated_segments_fields.npy";
    const fs::path cbmConsLabelsPath= dumpDir / "cbm_consolidated_segments_labels.txt";
    const fs::path novFieldsPath    = dumpDir / "novelty_segments_fields.npy";
    const fs::path novLabelsPath    = dumpDir / "novelty_segments_labels.txt";
    const fs::path novConsFieldsPath= dumpDir / "novelty_consolidated_segments_fields.npy";
    const fs::path novConsLabelsPath= dumpDir / "novelty_consolidated_segments_labels.txt";

    // Skip tracks without the new session-9 goldens (e.g. billie_jean_real).
    if (!fs::exists(yAudioPath) ||
        !fs::exists(cbmConsFieldsPath) || !fs::exists(novConsFieldsPath))
    {
        std::printf("[skip] %s — session-9 goldens not present\n",
                    r.name.c_str());
        return r;
    }

    // duration = len(y) / sr per structure_analyzer.py:119. y_audio was
    // saved from the clipped audio; sr is fixed at 22050 in the dump tool.
    const auto y = reamix::test::loadNpy1DFloat32(yAudioPath.string());
    const double duration =
        static_cast<double>(y.size()) / kSampleRateDouble;

    // --- Fixture A — CBM-path input --------------------------------------
    const auto cbmFields    = reamix::test::loadNpy2DFloat64(cbmFieldsPath.string());
    const auto cbmLabels    = loadTextLines(cbmLabelsPath.string());
    const auto cbmConsPy    = reamix::test::loadNpy2DFloat64(cbmConsFieldsPath.string());
    const auto cbmConsLblPy = loadTextLines(cbmConsLabelsPath.string());

    const auto cbmSegsIn = assembleSegments(cbmFields, cbmLabels);
    const auto cbmConsCpp =
        reamix::analysis::SegmentConsolidation::consolidate(cbmSegsIn, duration);

    r.aSegsIn  = static_cast<int>(cbmSegsIn.size());
    r.aSegsPy  = static_cast<int>(cbmConsPy.rows);
    r.aSegsCpp = static_cast<int>(cbmConsCpp.size());
    r.fixtureA = compareSegments(cbmConsPy, cbmConsLblPy, cbmConsCpp);

    // --- Fixture B — novelty-path input ----------------------------------
    const auto novFields    = reamix::test::loadNpy2DFloat64(novFieldsPath.string());
    const auto novLabels    = loadTextLines(novLabelsPath.string());
    const auto novConsPy    = reamix::test::loadNpy2DFloat64(novConsFieldsPath.string());
    const auto novConsLblPy = loadTextLines(novConsLabelsPath.string());

    const auto novSegsIn = assembleSegments(novFields, novLabels);
    const auto novConsCpp =
        reamix::analysis::SegmentConsolidation::consolidate(novSegsIn, duration);

    r.bSegsIn  = static_cast<int>(novSegsIn.size());
    r.bSegsPy  = static_cast<int>(novConsPy.rows);
    r.bSegsCpp = static_cast<int>(novConsCpp.size());
    r.fixtureB = compareSegments(novConsPy, novConsLblPy, novConsCpp);

    r.ran = true;
    return r;
}

int fixtureTotalMismatches(const FieldDiff& d)
{
    return d.nStartMismatch + d.nEndMismatch + d.nConfidenceMismatch
         + d.nClusterMismatch + d.nLabelMismatch;
}

void printTrackRow(const TrackResult& r, bool pass)
{
    std::printf("%-28s  %4d→%-4d  %4d→%-4d  %5d  %5d  %s\n",
        r.name.c_str(),
        r.aSegsIn, r.aSegsCpp,
        r.bSegsIn, r.bSegsCpp,
        fixtureTotalMismatches(r.fixtureA),
        fixtureTotalMismatches(r.fixtureB),
        pass ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char** argv)
{
    // FE_TONEAREST is the C/C++ default per IEEE 754; asserting matches
    // test_cbm_segmenter convention. Consolidation arithmetic is pure sum/
    // product/compare on f64 — no dtoa paths — but a non-default rounding
    // mode would still flip confidence bits on triplet-merges.
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

    std::printf("%-28s  %-9s  %-9s  %5s  %5s  %s\n",
        "track", "A:inCBM→", "B:inNov→", "Amis", "Bmis", "status");
    std::printf("%-28s  %-9s  %-9s  %5s  %5s  %s\n",
        "", "outCBM", "outNov", "", "", "");

    int ran = 0, failuresA = 0, failuresB = 0;
    bool allPass = true;

    for (const auto& d : trackDirs) {
        TrackResult r = runOne(d);
        if (!r.ran) continue;
        ++ran;

        const bool passA = (fixtureTotalMismatches(r.fixtureA) == 0);
        const bool passB = (fixtureTotalMismatches(r.fixtureB) == 0);
        const bool pass  = passA && passB;
        if (!pass)  allPass   = false;
        if (!passA) ++failuresA;
        if (!passB) ++failuresB;

        printTrackRow(r, pass);

        if (!passA && !r.fixtureA.firstMismatchMsg.empty()) {
            std::printf("   [A/CBM] %s\n", r.fixtureA.firstMismatchMsg.c_str());
        }
        if (!passB && !r.fixtureB.firstMismatchMsg.empty()) {
            std::printf("   [B/Nov] %s\n", r.fixtureB.firstMismatchMsg.c_str());
        }
    }

    std::printf("\nran %d track(s). Fixture A fail=%d, Fixture B fail=%d\n",
                ran, failuresA, failuresB);
    std::printf(
        "Column key: A = consolidate(CBM segs), B = consolidate(novelty segs); "
        "Amis/Bmis = total mismatches across start+end+confidence+"
        "cluster_id+label fields (strict bitwise).\n");

    if (ran == 0) {
        std::fprintf(stderr, "no tracks had session-9 goldens\n");
        return 2;
    }
    return allPass ? 0 : 1;
}
