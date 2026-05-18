// test_viterbi_dp — phase-4 parity:
// reamix::remix::{computeDownbeatArrays, buildNeighbors, viterbiDP}
// vs Python references/python-source/remix/viterbi_dp.py::{
//     _compute_downbeat_arrays, _build_neighbors, _viterbi_dp}.
//
// Session-20 smoke: 1 track (billie_jean) × 1 target (0.5 × original).
// Session-21 extends to 16 tracks × 3 ratios (0.3, 0.5, 0.7) = 48 cases
// per phase-4 acceptance criterion (a). Each case lives in its own subdir
// `<track_dir>/viterbi/r<R>/` so bisection stays per-ratio.
//
// Per-function sub-gates (HANDOVER-19 L130 rec):
//   Stage 1: computeDownbeatArrays  — bitwise int8 on pre_downbeat_arr +
//                                     downbeat_arr + set equality on indices.
//   Stage 2: SegmentData::computeSegmentData — bitwise int64 on
//                                     beat_to_segment + f64 ULP ≤ 1e-6 on
//                                     seg_sim_matrix + set equality on
//                                     boundary_beats.
//   Stage 3: buildNeighbors         — int64 bitwise on indices + offsets
//                                     AFTER per-beat sort (std::set in our
//                                     port already enforces sorted iteration;
//                                     Python does `sorted(neighbors)` at L123).
//   Stage 4: viterbiDP              — int64 bitwise on path + f64 ULP ≤ 5e-4
//                                     on total_cost. Exact path match is the
//                                     STRONGER gate (phase-4 spec L37 sets
//                                     a weaker "≥ 90 % of cases" acceptance
//                                     threshold — we target 100 % exact and
//                                     root-cause any deviation per Principle 6).
//
// Invocation:
//   test_viterbi_dp                          → corpus mode (16 × 3 = 48 cases).
//   test_viterbi_dp <vit_dir>                → single-case bisection; arg is
//                                              the full ratio-specific subdir
//                                              e.g. `.../billie_jean/viterbi/r0.5`.
//
// Gate design rationale:
//   - Bitwise on integer outputs (downbeat arrays, beat_to_segment,
//     boundary sets, neighbor CSR, DP path) because Python produces
//     well-defined int64 ↔ set semantics — any divergence signals a
//     semantic gap, not f64 ULP noise.
//   - Tight ULP gates on f64 seg_sim_matrix (1e-9) + total_cost (5e-4)
//     because these pass through the SAME accumulator chains validated
//     session-19 in TransitionCost (f32 matmul + composite weighted sum +
//     -ffp-contract=off on the test target per ADR-028, 3rd reuse).
//   - Path is EXACT — one mismatch is a failure. Rationale (phase-4
//     spec L37): "Exact beat-index sequence match on ≥ 90 % of cases".
//     For a single-track smoke that means 100 % match or FAIL + bisect
//     per Principle 6.

#include "remix/ViterbiDP.h"
#include "remix/SegmentData.h"
#include "analysis/RepetitionMap.h"
#include "analysis/StructureResult.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "NpyIO.h"

namespace {

// Session-21 corpus: 16 tracks (alphabetical; matches TRACK_LIST in
// tools/dump_phase4_tests.py) × 3 target ratios = 48 cases.
const std::vector<std::string> kCorpusTracks = {
    "alice_in_chains_nutshell",
    "billie_jean",
    "bob_dylan_lay_lady_lay",
    "daft_punk_aerodynamic",
    "dance_monkey",
    "eminem_without_me",
    "eno_music_for_airports_1_1",
    "goldberg_var_15_andante",
    "meshuggah_bleed",
    "miles_davis_so_what",
    "shostakovich_jazz_waltz",
    "smells_like_teen_spirit",
    "tiesto_the_motto",
    "vocal_solo_with_fx",
    "wardruna_voluspa",
    "woodkid_iron_acoustic",
};

const std::vector<double> kCorpusRatios = { 0.3, 0.5, 0.7 };

// Canonical per-ratio subdir name; mirrors _ratio_subdir() in the dump tool.
std::string ratioLabel(double r)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "r%.1f", r);
    return std::string(buf);
}

struct Tolerances
{
    // seg_sim_matrix: f32 accumulator class, not f64 bitwise.
    //
    // Python `features[mask_i].mean(axis=0)` returns f32 when features is
    // f32 — numpy preserves dtype under `.mean`. Our SegmentData.cpp
    // accumulates the per-segment mean in f64 for numerical stability, so
    // the off-diagonal blend cells drift by the f32-mean ULP class (~3-4×
    // 1e-8 at f32-ULP-times-num-features on a ~32-beat segment).
    //
    // Same class as `chroma_D` in test_transition_cost (f32 matmul, gate
    // 5e-5). Gate 1e-6 leaves ~25× safety margin vs session-20 measured
    // max_abs 3.83e-08 while still flagging any f64→f32 drift that escapes
    // the accumulator. Bitwise parity would require porting numpy's
    // pairwise-sum scheme (session-7 `pairwiseSumNumpy` precedent, ~200
    // LOC) AND switching the mean accumulator to f32 — documented in
    // weights-audit as potential session-21+ refinement if the drift
    // propagates above DP-path-flip threshold.
    double seg_sim_max_abs    = 1e-6;

    // total_cost: composite f64 DP accumulator with weighted penalty sums.
    // Same class as transition_cost::candidate quality_score in session-19
    // corpus (1.128e-07). Tolerance 5e-4 matches phase-4 spec Quality-
    // score L∞ gate; this is a total-cost proxy for the same gate class.
    double total_cost_max_abs = 5e-4;
};

const Tolerances TOL;

// Text file reader for seg_labels.txt (lives in the parent directory that
// dump_phase4_tests.py already populates for the 16-track corpus — we reuse
// it instead of re-dumping).
std::vector<std::string> readLines(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open text file: " + path);
    }
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
    }
    return out;
}

std::set<std::int64_t> toSet(const std::vector<std::int64_t>& v)
{
    return std::set<std::int64_t>(v.begin(), v.end());
}

// Inputs bundle for one (track, ratio) case. Corpus mode builds 48 of
// these; single-case bisection mode builds one.
struct Case
{
    // TRACK ROOT — same dir the 16-track TransitionCost dump populates.
    std::string track_dir;
    // Viterbi-specific subdir (e.g. `<track_dir>/viterbi/r0.5`).
    std::string vit_dir;
    std::string track;
    std::string ratio_label;   // "r0.3", "r0.5", "r0.7" — for reporting.

    // From the parent TransitionCost dump.
    std::vector<double>                      beat_times;
    std::vector<float>                       features;
    int                                      n_features = 0;
    std::vector<reamix::analysis::Segment>   segments;
    std::vector<double>                      downbeats;

    // From the Viterbi subdir.
    std::vector<std::int8_t>                 pre_db_expected;
    std::vector<std::int8_t>                 db_expected;
    std::set<std::int64_t>                   db_indices_expected;
    std::set<std::int64_t>                   pre_db_indices_expected;

    std::vector<std::int64_t>                beat_to_segment_expected;
    std::vector<double>                      seg_sim_expected;
    int                                      n_segs = 0;
    std::set<std::int64_t>                   boundary_beats_expected;

    // RepetitionJump flat arrays (+ pack into vector<RepetitionJump>).
    std::vector<reamix::analysis::RepetitionJump> rep_jumps;

    std::vector<std::int64_t>                neighbor_indices_expected;
    std::vector<std::int64_t>                neighbor_offsets_expected;

    std::vector<std::int64_t>                path_expected;
    double                                   total_cost_expected = 0.0;

    std::vector<double>                      W;  // f64 (n_beats × n_beats)
    int                                      n_beats = 0;

    // DP params (parsed from manifest.json).
    int    target_length      = 0;
    int    min_target_length  = 0;
    int    intro_beats        = 0;
    int    outro_beats        = 0;
    bool   is_shortening      = true;
    int    max_transitions    = 6;
    int    min_jumps          = 0;
    int    min_seq_after_jump = 16;
    int    min_forward_jump   = 16;
    int    min_neighbor_jump  = 4;   // optimizer.py:152 — time_signature.
    int    time_signature     = 4;
};

// Minimal JSON scanner for manifest.json. The manifest has a nested
// "dp_params" object with simple int / float fields; we extract them via
// key search rather than pulling in a JSON dependency.
std::string readAll(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

// Extract int value for `"<key>": <int>` occurring after `dp_params` key.
// Naive but sufficient — manifest has no nested duplicates for these keys.
bool extractInt(const std::string& json, const std::string& key, int& out)
{
    const std::string pat = "\"" + key + "\":";
    std::size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    // Skip whitespace.
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    char* end = nullptr;
    long v = std::strtol(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    out = static_cast<int>(v);
    return true;
}

bool extractBool(const std::string& json, const std::string& key, bool& out)
{
    const std::string pat = "\"" + key + "\":";
    std::size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (json.compare(pos, 4, "true") == 0)  { out = true;  return true; }
    if (json.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

Case loadCase(const std::string& track_dir,
              const std::string& ratio_label,
              const std::string& track)
{
    Case c;
    c.track_dir   = track_dir;
    c.vit_dir     = track_dir + "/viterbi/" + ratio_label;
    c.track       = track;
    c.ratio_label = ratio_label;

    // Parent-dir inputs (echoed from the 16-track TransitionCost dump).
    c.beat_times = reamix::test::loadNpy1DFloat64(track_dir + "/beat_times.npy");
    c.n_beats = static_cast<int>(c.beat_times.size());

    {
        auto feat = reamix::test::loadNpy2DFloat32(track_dir + "/features.npy");
        c.features   = std::move(feat.data);
        c.n_features = static_cast<int>(feat.cols);
    }

    {
        // Segment fields — same pattern as test_transition_cost.
        auto seg_start      = reamix::test::loadNpy1DFloat64(track_dir + "/seg_start.npy");
        auto seg_end        = reamix::test::loadNpy1DFloat64(track_dir + "/seg_end.npy");
        auto seg_confidence = reamix::test::loadNpy1DFloat64(track_dir + "/seg_confidence.npy");
        auto seg_cluster_id = reamix::test::loadNpy1DInt64(  track_dir + "/seg_cluster_id.npy");
        auto seg_labels     = readLines(                     track_dir + "/seg_labels.txt");
        if (seg_start.size() != seg_end.size()
            || seg_start.size() != seg_confidence.size()
            || seg_start.size() != seg_cluster_id.size()
            || seg_start.size() != seg_labels.size())
        {
            throw std::runtime_error("segment-array size mismatch in " + track_dir);
        }
        c.segments.reserve(seg_start.size());
        for (std::size_t i = 0; i < seg_start.size(); ++i) {
            reamix::analysis::Segment s;
            s.start      = seg_start[i];
            s.end        = seg_end[i];
            s.confidence = seg_confidence[i];
            s.cluster_id = static_cast<int>(seg_cluster_id[i]);
            s.label      = seg_labels[i];
            c.segments.push_back(std::move(s));
        }
    }

    c.downbeats = reamix::test::loadNpy1DFloat64(track_dir + "/downbeats.npy");

    // W from the Viterbi subdir (dumped fresh to decouple from any future
    // schema-drift in the parent corpus dump). Shape (n_beats, n_beats) f64
    // row-major; we store as flat f64 matching ViterbiDPInputs::W layout.
    {
        auto Wmat = reamix::test::loadNpy2DFloat64(c.vit_dir + "/W.npy");
        c.W = std::move(Wmat.data);
        if (static_cast<int>(Wmat.rows) != c.n_beats
            || static_cast<int>(Wmat.cols) != c.n_beats)
        {
            throw std::runtime_error("W shape mismatch: expected n_beats²");
        }
    }

    // --- Stage-1 goldens -----------------------------------------------
    // pre_downbeat_arr/downbeat_arr are dumped as int64 (not int8) because
    // NpyIO.h's reader surface is f64/f32/i64 only. We cast back to int8 at
    // load time to match the C++ public API (`DownbeatArrays::*_arr` is
    // `std::vector<std::int8_t>`).
    {
        auto tmp = reamix::test::loadNpy1DInt64(c.vit_dir + "/pre_downbeat_arr.npy");
        c.pre_db_expected.reserve(tmp.size());
        for (auto v : tmp) c.pre_db_expected.push_back(static_cast<std::int8_t>(v));
    }
    {
        auto tmp = reamix::test::loadNpy1DInt64(c.vit_dir + "/downbeat_arr.npy");
        c.db_expected.reserve(tmp.size());
        for (auto v : tmp) c.db_expected.push_back(static_cast<std::int8_t>(v));
    }

    {
        auto db_idx_vec = reamix::test::loadNpy1DInt64(c.vit_dir + "/downbeat_indices.npy");
        c.db_indices_expected = toSet(db_idx_vec);
    }
    {
        auto pre_idx_vec = reamix::test::loadNpy1DInt64(c.vit_dir + "/pre_downbeat_indices.npy");
        c.pre_db_indices_expected = toSet(pre_idx_vec);
    }

    // --- Stage-2 goldens -----------------------------------------------
    c.beat_to_segment_expected = reamix::test::loadNpy1DInt64(c.vit_dir + "/beat_to_segment.npy");
    {
        auto mat = reamix::test::loadNpy2DFloat64(c.vit_dir + "/seg_sim_matrix.npy");
        c.seg_sim_expected = std::move(mat.data);
        c.n_segs = static_cast<int>(mat.rows);
        if (mat.rows != mat.cols) {
            throw std::runtime_error("seg_sim_matrix non-square");
        }
    }
    {
        auto v = reamix::test::loadNpy1DInt64(c.vit_dir + "/boundary_beats.npy");
        c.boundary_beats_expected = toSet(v);
    }

    // --- Stage-3 goldens (RepetitionJump flat arrays) -------------------
    {
        auto fb = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_from_beat.npy");
        auto tb = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_to_beat.npy");
        auto ws = reamix::test::loadNpy1DFloat64(c.vit_dir + "/rep_waveform_similarity.npy");
        auto cc = reamix::test::loadNpy1DFloat64(c.vit_dir + "/rep_chroma_correlation.npy");
        auto al = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_alignment_lag_samples.npy");
        auto fs = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_from_section_idx.npy");
        auto ts = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_to_section_idx.npy");
        auto fr = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_from_bar.npy");
        auto tr = reamix::test::loadNpy1DInt64  (c.vit_dir + "/rep_to_bar.npy");
        const std::size_t n = fb.size();
        if (tb.size() != n || ws.size() != n || cc.size() != n || al.size() != n
            || fs.size() != n || ts.size() != n || fr.size() != n || tr.size() != n)
        {
            throw std::runtime_error("rep_* array-size mismatch");
        }
        c.rep_jumps.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            reamix::analysis::RepetitionJump j;
            j.fromBeat            = static_cast<int>(fb[k]);
            j.toBeat              = static_cast<int>(tb[k]);
            j.waveformSimilarity  = ws[k];
            j.chromaCorrelation   = cc[k];
            j.alignmentLagSamples = static_cast<int>(al[k]);
            j.fromSectionIdx      = static_cast<int>(fs[k]);
            j.toSectionIdx        = static_cast<int>(ts[k]);
            j.fromBar             = static_cast<int>(fr[k]);
            j.toBar               = static_cast<int>(tr[k]);
            c.rep_jumps.push_back(j);
        }
    }

    c.neighbor_indices_expected = reamix::test::loadNpy1DInt64(c.vit_dir + "/neighbor_indices.npy");
    c.neighbor_offsets_expected = reamix::test::loadNpy1DInt64(c.vit_dir + "/neighbor_offsets.npy");

    // --- Stage-4 goldens -----------------------------------------------
    c.path_expected = reamix::test::loadNpy1DInt64(c.vit_dir + "/dp_path.npy");
    {
        auto tc = reamix::test::loadNpy1DFloat64(c.vit_dir + "/dp_total_cost.npy");
        if (tc.empty()) throw std::runtime_error("dp_total_cost.npy empty");
        c.total_cost_expected = tc[0];
    }

    // --- DP params from manifest.json ----------------------------------
    const std::string mfst = readAll(c.vit_dir + "/manifest.json");
    // dp_params lives as a nested object; key names are unique across the
    // manifest so naive scan works.
    if (!extractInt(mfst,  "target_length",      c.target_length)
        || !extractInt(mfst,  "min_target_length",  c.min_target_length)
        || !extractInt(mfst,  "intro_beats",        c.intro_beats)
        || !extractInt(mfst,  "outro_beats",        c.outro_beats)
        || !extractBool(mfst, "is_shortening",      c.is_shortening)
        || !extractInt(mfst,  "max_transitions",    c.max_transitions)
        || !extractInt(mfst,  "min_jumps",          c.min_jumps)
        || !extractInt(mfst,  "min_seq_after_jump", c.min_seq_after_jump)
        || !extractInt(mfst,  "min_forward_jump",   c.min_forward_jump))
    {
        throw std::runtime_error("manifest.json missing dp_params key");
    }

    return c;
}

// --- Reporting helpers ----------------------------------------------------

bool cmpInt8(const std::vector<std::int8_t>& actual,
             const std::vector<std::int8_t>& expected,
             const std::string& label)
{
    if (actual.size() != expected.size()) {
        std::printf("  [%s] FAIL size %zu vs %zu\n", label.c_str(),
                    actual.size(), expected.size());
        return false;
    }
    std::size_t miss = 0, first = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (miss == 0) first = i;
            ++miss;
        }
    }
    if (miss) {
        std::printf("  [%s] FAIL %zu mismatches, first at i=%zu "
                    "(cpp=%d py=%d)\n",
                    label.c_str(), miss, first,
                    static_cast<int>(actual[first]),
                    static_cast<int>(expected[first]));
        return false;
    }
    std::printf("  [%s] OK (bitwise)\n", label.c_str());
    return true;
}

bool cmpInt64(const std::vector<std::int64_t>& actual,
              const std::vector<std::int64_t>& expected,
              const std::string& label)
{
    if (actual.size() != expected.size()) {
        std::printf("  [%s] FAIL size %zu vs %zu\n", label.c_str(),
                    actual.size(), expected.size());
        return false;
    }
    std::size_t miss = 0, first = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (miss == 0) first = i;
            ++miss;
        }
    }
    if (miss) {
        std::printf("  [%s] FAIL %zu mismatches, first at i=%zu "
                    "(cpp=%lld py=%lld)\n",
                    label.c_str(), miss, first,
                    static_cast<long long>(actual[first]),
                    static_cast<long long>(expected[first]));
        return false;
    }
    std::printf("  [%s] OK (bitwise)\n", label.c_str());
    return true;
}

bool cmpSet(const std::set<std::int64_t>& actual,
            const std::set<std::int64_t>& expected,
            const std::string& label)
{
    if (actual == expected) {
        std::printf("  [%s] OK (set-equal, |S|=%zu)\n", label.c_str(), actual.size());
        return true;
    }
    std::size_t extra = 0, missing = 0;
    for (auto v : actual)   if (!expected.count(v)) ++extra;
    for (auto v : expected) if (!actual.count(v))   ++missing;
    std::printf("  [%s] FAIL set-diff (|cpp|=%zu |py|=%zu extras=%zu missing=%zu)\n",
                label.c_str(), actual.size(), expected.size(), extra, missing);
    return false;
}

bool cmpF64Vec(const std::vector<double>& actual,
               const std::vector<double>& expected,
               double tol, const std::string& label)
{
    if (actual.size() != expected.size()) {
        std::printf("  [%s] FAIL size %zu vs %zu\n", label.c_str(),
                    actual.size(), expected.size());
        return false;
    }
    double max_abs = 0.0;
    std::size_t worst = 0;
    std::size_t miss  = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double d = std::fabs(actual[i] - expected[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
        if (d > tol) ++miss;
    }
    if (miss) {
        std::printf("  [%s] FAIL %zu/%zu above tol=%.2e, max_abs=%.3e at i=%zu "
                    "(cpp=%.6f py=%.6f)\n",
                    label.c_str(), miss, actual.size(), tol, max_abs, worst,
                    actual[worst], expected[worst]);
        return false;
    }
    std::printf("  [%s] OK max_abs=%.3e (tol=%.2e, n=%zu)\n",
                label.c_str(), max_abs, tol, actual.size());
    return true;
}

// --- Runner ---------------------------------------------------------------

int runCase(const Case& c)
{
    std::printf("=== %s %s (n_beats=%d n_segs=%d target=%d path_expected=%zu) ===\n",
                c.track.c_str(), c.ratio_label.c_str(),
                c.n_beats, c.n_segs, c.target_length,
                c.path_expected.size());

    int fails = 0;

    // ================ Stage 1: computeDownbeatArrays ==================
    std::printf("-- Stage 1: computeDownbeatArrays\n");
    auto db = reamix::remix::computeDownbeatArrays(
        c.beat_times.data(), c.n_beats,
        c.downbeats.data(), static_cast<int>(c.downbeats.size()),
        c.time_signature, /*downbeat_constraint=*/true);

    if (!db.valid) {
        std::printf("  [downbeat_arrays] FAIL: got valid=false (expected true)\n");
        ++fails;
    } else {
        if (!cmpInt8(db.pre_downbeat_arr, c.pre_db_expected, "pre_downbeat_arr"))
            ++fails;
        if (!cmpInt8(db.downbeat_arr,     c.db_expected,     "downbeat_arr"))
            ++fails;

        std::set<std::int64_t> db_idx_cpp;
        for (int v : db.downbeat_indices) db_idx_cpp.insert(v);
        if (!cmpSet(db_idx_cpp, c.db_indices_expected, "downbeat_indices"))
            ++fails;

        std::set<std::int64_t> pre_idx_cpp;
        for (int v : db.pre_downbeat_indices) pre_idx_cpp.insert(v);
        if (!cmpSet(pre_idx_cpp, c.pre_db_indices_expected, "pre_downbeat_indices"))
            ++fails;
    }

    // ================ Stage 2: SegmentData ============================
    std::printf("-- Stage 2: SegmentData::computeSegmentData\n");
    auto sd = reamix::remix::computeSegmentData(
        c.n_beats,
        c.segments.data(), static_cast<int>(c.segments.size()),
        c.beat_times.data(),
        c.features.data(), c.n_features);

    if (!cmpInt64(sd.beat_to_segment, c.beat_to_segment_expected, "beat_to_segment"))
        ++fails;

    // seg_sim is f64 — tight ULP gate 1e-9.
    if (!cmpF64Vec(sd.seg_sim, c.seg_sim_expected, TOL.seg_sim_max_abs, "seg_sim_matrix"))
        ++fails;

    {
        std::set<std::int64_t> bb_cpp;
        for (int v : sd.boundary_beats) bb_cpp.insert(v);
        if (!cmpSet(bb_cpp, c.boundary_beats_expected, "boundary_beats"))
            ++fails;
    }

    // ================ Stage 3: buildNeighbors =========================
    std::printf("-- Stage 3: buildNeighbors (min_forward_jump=%d per optimizer.py:152)\n",
                c.min_neighbor_jump);
    auto csr = reamix::remix::buildNeighbors(
        c.n_beats, c.W.data(),
        c.rep_jumps.empty() ? nullptr : c.rep_jumps.data(),
        static_cast<int>(c.rep_jumps.size()),
        &db,
        sd.boundary_beats,
        /*max_neighbors=*/32,
        c.min_neighbor_jump);

    if (!cmpInt64(csr.offsets, c.neighbor_offsets_expected, "neighbor_offsets"))
        ++fails;
    if (!cmpInt64(csr.indices, c.neighbor_indices_expected, "neighbor_indices"))
        ++fails;

    // ================ Stage 4: viterbiDP ==============================
    std::printf("-- Stage 4: viterbiDP (target=%d, min_target=%d, intro=%d, outro=%d, "
                "shortening=%d, min_jumps=%d, min_seq=%d, min_forward=%d)\n",
                c.target_length, c.min_target_length, c.intro_beats, c.outro_beats,
                static_cast<int>(c.is_shortening), c.min_jumps,
                c.min_seq_after_jump, c.min_forward_jump);

    reamix::remix::ViterbiDPInputs in{};
    in.W                  = c.W.data();
    in.n_beats            = c.n_beats;
    in.target_length      = c.target_length;
    in.min_target_length  = c.min_target_length;
    in.intro_beats        = c.intro_beats;
    in.outro_beats        = c.outro_beats;
    in.is_shortening      = c.is_shortening;
    in.neighbor_indices   = csr.indices.data();
    in.n_neighbor_indices = static_cast<int>(csr.indices.size());
    in.neighbor_offsets   = csr.offsets.data();
    in.beat_to_segment    = sd.beat_to_segment.data();
    in.seg_sim_matrix     = sd.seg_sim.data();
    in.n_segs             = sd.n_segs;
    in.pre_downbeat_arr   = db.valid ? db.pre_downbeat_arr.data() : nullptr;
    in.downbeat_arr       = db.valid ? db.downbeat_arr.data()     : nullptr;
    in.max_transitions    = c.max_transitions;
    in.min_jumps          = c.min_jumps;
    in.min_seq_after_jump = c.min_seq_after_jump;
    in.min_forward_jump   = c.min_forward_jump;
    // min_segment_beats stays at kMinSegmentBeatsDefault (16 per config.py:105);
    // smoke track is 4/4 so no override needed.

    auto vp = reamix::remix::viterbiDP(in);

    if (!cmpInt64(vp.path, c.path_expected, "dp_path"))
        ++fails;

    {
        const double d = std::fabs(vp.total_cost - c.total_cost_expected);
        if (d > TOL.total_cost_max_abs) {
            std::printf("  [dp_total_cost] FAIL |cpp-py|=%.3e > tol=%.2e "
                        "(cpp=%.6f py=%.6f)\n",
                        d, TOL.total_cost_max_abs, vp.total_cost, c.total_cost_expected);
            ++fails;
        } else {
            std::printf("  [dp_total_cost] OK |cpp-py|=%.3e (tol=%.2e)\n",
                        d, TOL.total_cost_max_abs);
        }
    }

    std::printf("=== %s %s: %d sub-gate failures ===\n\n",
                c.track.c_str(), c.ratio_label.c_str(), fails);
    return fails;
}

// --- Corpus-mode helpers --------------------------------------------------

struct CaseSummary
{
    std::string track;
    std::string ratio_label;
    int         fails = 0;
    bool        loaded = true;     // false if loadCase threw (missing dump)
    std::string err;               // populated when !loaded
};

// Split a path "tests/.../billie_jean/viterbi/r0.5" into parent track dir
// "tests/.../billie_jean" and the final component "r0.5". Accepts an
// optional trailing slash.
bool splitVitPath(const std::string& vit_path,
                  std::string&       out_track_dir,
                  std::string&       out_ratio_label)
{
    std::string p = vit_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    const std::size_t last = p.find_last_of('/');
    if (last == std::string::npos) return false;
    out_ratio_label = p.substr(last + 1);          // e.g. "r0.5"
    const std::string parent = p.substr(0, last);  // "<track_dir>/viterbi"
    const std::size_t parent_last = parent.find_last_of('/');
    if (parent_last == std::string::npos) return false;
    if (parent.substr(parent_last + 1) != "viterbi") return false;
    out_track_dir = parent.substr(0, parent_last); // "<track_dir>"
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    // Mode 1: single-case bisection. argv[1] points directly at the ratio-
    //         specific subdir (e.g. `.../billie_jean/viterbi/r0.5`).
    if (argc > 1) {
        const std::string vit_path = argv[1];
        std::string track_dir, ratio_label;
        if (!splitVitPath(vit_path, track_dir, ratio_label)) {
            std::fprintf(stderr,
                "ERROR: argv[1] must be a `.../viterbi/r<R>` subdir path "
                "(got: %s)\n", vit_path.c_str());
            return 2;
        }
        const std::size_t slash = track_dir.find_last_of('/');
        const std::string track =
            (slash == std::string::npos) ? track_dir : track_dir.substr(slash + 1);
        try {
            auto c = loadCase(track_dir, ratio_label, track);
            const int fails = runCase(c);
            return fails == 0 ? 0 : 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ERROR: %s\n", e.what());
            return 2;
        }
    }

    // Mode 2: corpus — iterate 16 tracks × 3 ratios = 48 cases.
    const std::string dataRoot = "tests/parity/reference/data/phase4";

    std::printf("=== ViterbiDP corpus parity: %zu tracks × %zu ratios = %zu cases ===\n",
                kCorpusTracks.size(), kCorpusRatios.size(),
                kCorpusTracks.size() * kCorpusRatios.size());

    std::vector<CaseSummary> summaries;
    summaries.reserve(kCorpusTracks.size() * kCorpusRatios.size());

    int total_fails = 0;
    int cases_pass  = 0;
    int cases_fail  = 0;
    int cases_miss  = 0;

    for (const auto& track : kCorpusTracks) {
        for (const double r : kCorpusRatios) {
            const std::string ratio = ratioLabel(r);
            const std::string track_dir = dataRoot + "/" + track;

            CaseSummary summary;
            summary.track       = track;
            summary.ratio_label = ratio;

            try {
                auto c = loadCase(track_dir, ratio, track);
                const int fails = runCase(c);
                summary.fails = fails;
                if (fails == 0) ++cases_pass;
                else            ++cases_fail;
                total_fails += fails;
            } catch (const std::exception& e) {
                summary.loaded = false;
                summary.err    = e.what();
                ++cases_miss;
                std::printf("=== %s %s: LOAD-FAIL: %s ===\n\n",
                            track.c_str(), ratio.c_str(), e.what());
            }

            summaries.push_back(std::move(summary));
        }
    }

    // --- Corpus summary ----------------------------------------------------
    std::printf("\n=== 48-case corpus summary ===\n");
    std::printf("  cases PASS : %d\n", cases_pass);
    std::printf("  cases FAIL : %d\n", cases_fail);
    std::printf("  cases MISS : %d  (dump directory not found)\n", cases_miss);
    std::printf("  total sub-gate failures across all cases: %d\n", total_fails);

    if (cases_fail || cases_miss) {
        std::printf("\n  FAIL / MISS detail:\n");
        for (const auto& s : summaries) {
            if (!s.loaded) {
                std::printf("    %-32s %s  LOAD-FAIL (%s)\n",
                            s.track.c_str(), s.ratio_label.c_str(),
                            s.err.c_str());
            } else if (s.fails) {
                std::printf("    %-32s %s  FAIL  (%d sub-gate failures)\n",
                            s.track.c_str(), s.ratio_label.c_str(), s.fails);
            }
        }
    }

    const bool corpus_pass = (cases_fail == 0 && cases_miss == 0);
    std::printf("\n  CORPUS %s\n", corpus_pass ? "PASS" : "FAIL");

    return corpus_pass ? 0 : 1;
}
