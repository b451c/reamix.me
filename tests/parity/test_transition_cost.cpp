// test_transition_cost — phase-4 parity:
// reamix::remix::computeTransitionCosts vs Python
// references/python-source/remix/transition_cost.py::compute_transition_costs.
//
// History:
//   Session 18: billie_jean smoke (129-beat librosa, segments=None,
//               downbeats=None). 235/235 candidates matched first-run.
//   Session 19: extended to 16-track corpus + segments enabled (phase-3
//               `dispatched_segments_*`) + downbeats enabled. Per-track
//               PASS/FAIL + corpus-wide per-signal max_abs table.
//
// Per-signal sub-gates (ADR-026) — empirically set on session-18 first run;
// tightened in subsequent sessions as drift sources are identified and
// fixed. Principle 6 kicks in on any violation: bisect (per-signal →
// per-constant → per-upstream-feature) before widening.
//
// Hard-gate outcomes (chroma prefilter, energy 8 dB, quality floor,
// VOCAL_SPLICE_ACTIVITY_MIN 0.35 via deceptive-splice detector) express
// as SPARSITY PATTERN of W: Python and C++ must mark identical (i, j) cells
// as INF vs finite. Any disagreement in the INF-mask is a semantic gap.
//
// Invocation:
//   test_transition_cost                              → run all 16 tracks (corpus, status quo phase4/).
//   test_transition_cost <trackDir>                   → single-track bisection mode.
//   test_transition_cost --mode=no-structure          → ADR-044 no-structure corpus from phase4_no_structure/.
//   test_transition_cost --mode=no-structure <track>  → single-track bisection on no-structure goldens.
//
// ADR-044 close-out (session 53): the no-structure corpus exercises the
// `n_segments=0` path of `computeTransitionCosts` against goldens dumped by
// `tools/dump_phase4_tests.py --no-structure`. Same per-signal tolerances
// (TOL) — gating only zeros section_sim / label_match / SPAN_PENALTY in
// scoring, no new sources of numerical drift.

#include "remix/TransitionCost.h"
#include "remix/Quality.h"

#include "analysis/StructureResult.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "NpyIO.h"

namespace {

// Per-signal tolerance budgets (f64 unless noted).
// Baseline set on session-18 billie_jean smoke (first-run max_abs × 2).
// Held constant for session 19 16-track first regen per HANDOVER-18 L131
// recommendation (tighten only after corpus-wide distribution is known).
struct Tolerances
{
    double W_max_abs            = 5e-4;

    double chroma_D_max_abs     = 5e-5;   // f32 matmul + f64 clip
    double importance_max_abs   = 0.0;    // bitwise (pure label lookup)

    double cand_waveform_max_abs         = 1e-6;   // f32 xcorr class
    double cand_successor_max_abs        = 5e-5;   // f32 matmul → f64 cast
    double cand_edge_splice_max_abs      = 5e-5;   // f32 matmul → f64 cast
    double cand_chroma_distance_max_abs  = 5e-5;   // f32 matmul → f64 (1-x)
    double cand_energy_diff_db_max_abs   = 5e-12;  // pure f64 log10+abs
    double cand_quality_score_max_abs    = 5e-4;   // composite weighted sum
    double cand_total_cost_max_abs       = 5e-4;   // 1 - quality
    int    cand_alignment_lag_tol        = 0;      // int, bitwise
};

const Tolerances TOL;

// Session-19 16-track corpus (alphabetical; matches TRACK_LIST in
// tools/dump_phase4_tests.py). Hard-coded for simplicity; if a 17th
// track lands in the future, extend both lists.
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

// --- Aggregated per-signal max_abs (for corpus summary) --------------------
struct SignalAcc
{
    double max_abs = 0.0;
    std::size_t total_n = 0;
    std::size_t total_mismatches = 0;
    std::string worst_track;
    std::size_t worst_index = 0;
};

struct CorpusAcc
{
    SignalAcc chroma_D;
    SignalAcc importance;
    SignalAcc W_finite;
    SignalAcc cand_quality_score;
    SignalAcc cand_waveform;
    SignalAcc cand_successor;
    SignalAcc cand_edge_splice;
    SignalAcc cand_chroma_distance;
    SignalAcc cand_energy_diff_db;
    SignalAcc cand_total_cost;
    std::size_t total_candidate_mismatches = 0;  // key-set mismatches
    std::size_t total_infmask_mismatches = 0;
    std::size_t tracks_pass = 0;
    std::size_t tracks_fail = 0;
};

// --- Per-track report ------------------------------------------------------
struct Report
{
    std::string name;
    std::size_t n = 0;
    std::size_t mismatches = 0;
    double      max_abs = 0.0;
    std::size_t first_fail = 0;
    bool        have_fail = false;
};

Report compareDouble(const std::string& name,
                     const std::vector<double>& a,
                     const std::vector<double>& b,
                     double tol)
{
    Report r; r.name = name; r.n = std::min(a.size(), b.size());
    if (a.size() != b.size()) {
        std::fprintf(stderr, "  [%s] size mismatch: %zu vs %zu\n",
                     name.c_str(), a.size(), b.size());
    }
    for (std::size_t i = 0; i < r.n; ++i) {
        const double d = std::abs(a[i] - b[i]);
        if (d > r.max_abs) r.max_abs = d;
        if (d > tol) {
            if (!r.have_fail) { r.first_fail = i; r.have_fail = true; }
            ++r.mismatches;
        }
    }
    return r;
}

Report compareInt(const std::string& name,
                  const std::vector<std::int64_t>& a,
                  const std::vector<std::int64_t>& b,
                  std::int64_t tol)
{
    Report r; r.name = name; r.n = std::min(a.size(), b.size());
    if (a.size() != b.size()) {
        std::fprintf(stderr, "  [%s] size mismatch: %zu vs %zu\n",
                     name.c_str(), a.size(), b.size());
    }
    for (std::size_t i = 0; i < r.n; ++i) {
        const std::int64_t d = std::abs(a[i] - b[i]);
        if (static_cast<double>(d) > r.max_abs) r.max_abs = static_cast<double>(d);
        if (d > tol) {
            if (!r.have_fail) { r.first_fail = i; r.have_fail = true; }
            ++r.mismatches;
        }
    }
    return r;
}

void printReport(const Report& r, double tol,
                 const std::vector<double>* a = nullptr,
                 const std::vector<double>* b = nullptr)
{
    std::printf("    %-28s  n=%zu  mismatches=%zu  max_abs=%.3e  tol=%.1e  %s\n",
                r.name.c_str(), r.n, r.mismatches, r.max_abs, tol,
                (r.mismatches == 0) ? "PASS" : "FAIL");
    if (r.have_fail && a != nullptr && b != nullptr) {
        std::printf("      first fail at i=%zu: cpp=%.17g python=%.17g diff=%.3e\n",
                    r.first_fail, (*a)[r.first_fail], (*b)[r.first_fail],
                    std::abs((*a)[r.first_fail] - (*b)[r.first_fail]));
    }
}

void accumulate(SignalAcc& acc, const Report& r, const std::string& track)
{
    if (r.max_abs > acc.max_abs) {
        acc.max_abs = r.max_abs;
        acc.worst_track = track;
        acc.worst_index = r.first_fail;
    }
    acc.total_n += r.n;
    acc.total_mismatches += r.mismatches;
}

// --- Track bundle ----------------------------------------------------------
struct TrackBundle
{
    std::string                       track;

    std::vector<double>               beat_times;
    std::vector<float>                features;
    std::vector<float>                boundary_waveforms;
    std::vector<float>                vocal_band_waveforms;
    std::vector<double>               edge_rms_start;
    std::vector<double>               edge_rms_end;
    std::vector<float>                edge_features_start;
    std::vector<float>                edge_features_end;
    std::vector<double>               rms_energy;
    std::vector<double>               onset_strength;
    std::vector<double>               spectral_centroid;
    std::vector<double>               vocal_activity;
    std::vector<double>               edge_vocal_activity_start;
    std::vector<double>               edge_vocal_activity_end;

    // Segment context (session 19+).
    std::vector<reamix::analysis::Segment> segments;
    std::vector<int>                       segment_boundaries;
    std::vector<double>                    downbeats;

    int n_beats = 0;
    int n_features = 0;
    int n_edge_features = 0;
    int n_samples_per_bnd = 0;
    int waveform_sample_rate = 0;

    // Expected outputs.
    std::vector<double>               W_expected;
    std::vector<double>               chroma_D_expected;
    std::vector<double>               importance_expected;

    std::vector<std::int64_t>         cand_from;
    std::vector<std::int64_t>         cand_to;
    std::vector<double>               cand_quality_score;
    std::vector<double>               cand_waveform_similarity;
    std::vector<double>               cand_successor_similarity;
    std::vector<double>               cand_edge_splice_similarity;
    std::vector<double>               cand_chroma_distance;
    std::vector<double>               cand_energy_diff_db;
    std::vector<std::int64_t>         cand_alignment_lag_samples;
    std::vector<double>               cand_total_cost;
};

std::vector<std::string> readLines(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open text file: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        // Strip trailing \r (in case of Windows line endings).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
    }
    return out;
}

TrackBundle loadTrackBundle(const std::string& dataDir, const std::string& track)
{
    TrackBundle b;
    b.track = track;

    b.beat_times = reamix::test::loadNpy1DFloat64(dataDir + "/beat_times.npy");
    b.n_beats = static_cast<int>(b.beat_times.size());

    auto feat_mat = reamix::test::loadNpy2DFloat32(dataDir + "/features.npy");
    b.features = std::move(feat_mat.data);
    b.n_features = static_cast<int>(feat_mat.cols);

    auto bnd_mat = reamix::test::loadNpy2DFloat32(dataDir + "/boundary_waveforms.npy");
    b.boundary_waveforms = std::move(bnd_mat.data);
    b.n_samples_per_bnd = static_cast<int>(bnd_mat.cols);

    auto vb_mat = reamix::test::loadNpy2DFloat32(dataDir + "/vocal_band_waveforms.npy");
    b.vocal_band_waveforms = std::move(vb_mat.data);

    b.edge_rms_start = reamix::test::loadNpy1DFloat64(dataDir + "/edge_rms_start.npy");
    b.edge_rms_end   = reamix::test::loadNpy1DFloat64(dataDir + "/edge_rms_end.npy");

    auto es_mat = reamix::test::loadNpy2DFloat32(dataDir + "/edge_features_start.npy");
    b.edge_features_start = std::move(es_mat.data);
    b.n_edge_features = static_cast<int>(es_mat.cols);
    auto ee_mat = reamix::test::loadNpy2DFloat32(dataDir + "/edge_features_end.npy");
    b.edge_features_end = std::move(ee_mat.data);

    b.rms_energy        = reamix::test::loadNpy1DFloat64(dataDir + "/rms_energy.npy");
    b.onset_strength    = reamix::test::loadNpy1DFloat64(dataDir + "/onset_strength.npy");
    b.spectral_centroid = reamix::test::loadNpy1DFloat64(dataDir + "/spectral_centroid.npy");
    b.vocal_activity    = reamix::test::loadNpy1DFloat64(dataDir + "/vocal_activity.npy");
    b.edge_vocal_activity_start = reamix::test::loadNpy1DFloat64(dataDir + "/edge_vocal_activity_start.npy");
    b.edge_vocal_activity_end   = reamix::test::loadNpy1DFloat64(dataDir + "/edge_vocal_activity_end.npy");

    // Segments (session 19+). Each per-track dump stores segment starts /
    // ends / confidences / cluster_ids as f64 1-D arrays, labels as a
    // newline-separated text file.
    auto seg_start      = reamix::test::loadNpy1DFloat64(dataDir + "/seg_start.npy");
    auto seg_end        = reamix::test::loadNpy1DFloat64(dataDir + "/seg_end.npy");
    auto seg_confidence = reamix::test::loadNpy1DFloat64(dataDir + "/seg_confidence.npy");
    auto seg_cluster_id = reamix::test::loadNpy1DInt64   (dataDir + "/seg_cluster_id.npy");
    auto seg_labels     = readLines                      (dataDir + "/seg_labels.txt");
    if (seg_start.size() != seg_end.size() ||
        seg_start.size() != seg_confidence.size() ||
        seg_start.size() != seg_cluster_id.size() ||
        seg_start.size() != seg_labels.size())
    {
        throw std::runtime_error("segment-array size mismatch in " + dataDir);
    }
    b.segments.reserve(seg_start.size());
    for (std::size_t i = 0; i < seg_start.size(); ++i) {
        reamix::analysis::Segment s;
        s.start      = seg_start[i];
        s.end        = seg_end[i];
        s.confidence = seg_confidence[i];
        s.cluster_id = static_cast<int>(seg_cluster_id[i]);
        s.label      = seg_labels[i];
        b.segments.push_back(std::move(s));
    }

    // segment_boundaries dumped as int64 by numpy; C++ input takes int.
    auto sb64 = reamix::test::loadNpy1DInt64(dataDir + "/segment_boundaries.npy");
    b.segment_boundaries.reserve(sb64.size());
    for (auto v : sb64) b.segment_boundaries.push_back(static_cast<int>(v));

    b.downbeats = reamix::test::loadNpy1DFloat64(dataDir + "/downbeats.npy");

    // Sample rate: all 16 tracks of the session-19 corpus dump at 22050.
    // If a future dump introduces a different rate, extend manifest.json
    // parsing (minimal JSON scan) instead of hard-coding.
    b.waveform_sample_rate = 22050;

    auto W_mat         = reamix::test::loadNpy2DFloat64(dataDir + "/W.npy");
    b.W_expected       = std::move(W_mat.data);
    auto chD_mat       = reamix::test::loadNpy2DFloat64(dataDir + "/chroma_D.npy");
    b.chroma_D_expected = std::move(chD_mat.data);
    b.importance_expected = reamix::test::loadNpy1DFloat64(dataDir + "/importance.npy");

    b.cand_from = reamix::test::loadNpy1DInt64(dataDir + "/cand_from.npy");
    b.cand_to   = reamix::test::loadNpy1DInt64(dataDir + "/cand_to.npy");
    b.cand_quality_score         = reamix::test::loadNpy1DFloat64(dataDir + "/cand_quality_score.npy");
    b.cand_waveform_similarity   = reamix::test::loadNpy1DFloat64(dataDir + "/cand_waveform_similarity.npy");
    b.cand_successor_similarity  = reamix::test::loadNpy1DFloat64(dataDir + "/cand_successor_similarity.npy");
    b.cand_edge_splice_similarity = reamix::test::loadNpy1DFloat64(dataDir + "/cand_edge_splice_similarity.npy");
    b.cand_chroma_distance       = reamix::test::loadNpy1DFloat64(dataDir + "/cand_chroma_distance.npy");
    b.cand_energy_diff_db        = reamix::test::loadNpy1DFloat64(dataDir + "/cand_energy_diff_db.npy");
    b.cand_alignment_lag_samples = reamix::test::loadNpy1DInt64(dataDir + "/cand_alignment_lag_samples.npy");
    b.cand_total_cost            = reamix::test::loadNpy1DFloat64(dataDir + "/cand_total_cost.npy");

    return b;
}

// Run a single track; returns true on PASS, false on FAIL. Accumulates
// per-signal max_abs into `corpus`.
bool runTrack(const std::string& dataDir, const std::string& track,
              CorpusAcc& corpus)
{
    std::printf("--- track: %s ---\n", track.c_str());

    TrackBundle b;
    try {
        b = loadTrackBundle(dataDir, track);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  [load error] %s\n", e.what());
        return false;
    }

    std::printf("  bundle: n_beats=%d n_feat=%d n_edge_feat=%d n_bnd=%d sr=%d"
                " n_seg=%zu n_sb=%zu n_db=%zu n_cand_py=%zu\n",
                b.n_beats, b.n_features, b.n_edge_features,
                b.n_samples_per_bnd, b.waveform_sample_rate,
                b.segments.size(), b.segment_boundaries.size(),
                b.downbeats.size(), b.cand_from.size());

    // Build inputs bundle for C++ API.
    reamix::remix::TransitionCostInputs in{};
    in.features  = b.features.data();
    in.n_beats   = b.n_beats;
    in.n_features = b.n_features;
    in.beat_times = b.beat_times.data();

    in.segments   = b.segments.empty() ? nullptr : b.segments.data();
    in.n_segments = static_cast<int>(b.segments.size());

    in.segment_boundaries   = b.segment_boundaries.empty()
        ? nullptr : b.segment_boundaries.data();
    in.n_segment_boundaries = static_cast<int>(b.segment_boundaries.size());

    in.boundary_waveforms = b.boundary_waveforms.data();
    in.n_boundary_waveforms = b.n_beats;
    in.n_samples_per_bnd = b.n_samples_per_bnd;
    in.waveform_sample_rate = b.waveform_sample_rate;

    in.vocal_band_waveforms = b.vocal_band_waveforms.data();

    in.edge_rms_start = b.edge_rms_start.data();
    in.edge_rms_end   = b.edge_rms_end.data();
    in.edge_features_start = b.edge_features_start.data();
    in.edge_features_end   = b.edge_features_end.data();
    in.n_edge_features = b.n_edge_features;

    in.rms_energy        = b.rms_energy.data();
    in.onset_strength    = b.onset_strength.data();
    in.spectral_centroid = b.spectral_centroid.data();
    in.vocal_activity    = b.vocal_activity.data();
    in.edge_vocal_activity_start = b.edge_vocal_activity_start.data();
    in.edge_vocal_activity_end   = b.edge_vocal_activity_end.data();

    in.downbeats = b.downbeats.empty() ? nullptr : b.downbeats.data();
    in.n_downbeats = static_cast<int>(b.downbeats.size());
    in.time_signature = 4;
    // Sesja 81 ADR-068 D1+D3 default flip — pin to legacy 10-component
    // simplex (kLegacyQualityWeights) + legacy floor 0.45 explicitly so
    // the Python-parity goldens stay bit-exact past production-default
    // changes (kDefaultQualityWeights now 6-D + harmonic, QUALITY_HARD_FLOOR
    // now 0.20).
    in.quality_floor = reamix::remix::LEGACY_QUALITY_HARD_FLOOR;
    in.quality_weights = &reamix::remix::kLegacyQualityWeights;
    in.chroma_prefilter = reamix::remix::CHROMA_PREFILTER_THRESHOLD;
    in.max_candidates_per_beat = 16;
    in.waveform_align_max_shift_ms = 30.0;
    in.prescreened_pairs = nullptr;
    in.n_prescreened_pairs = 0;

    reamix::remix::TransitionCostResult res =
        reamix::remix::computeTransitionCosts(in);

    bool track_pass = true;

    // --- Shared matrices -----------------------------------------------
    auto R_chroma = compareDouble("chroma_D", res.chroma_D, b.chroma_D_expected,
                                  TOL.chroma_D_max_abs);
    printReport(R_chroma, TOL.chroma_D_max_abs, &res.chroma_D, &b.chroma_D_expected);
    accumulate(corpus.chroma_D, R_chroma, track);
    if (R_chroma.mismatches > 0) track_pass = false;

    auto R_imp = compareDouble("importance", res.importance, b.importance_expected,
                               TOL.importance_max_abs);
    printReport(R_imp, TOL.importance_max_abs, &res.importance, &b.importance_expected);
    accumulate(corpus.importance, R_imp, track);
    if (R_imp.mismatches > 0) track_pass = false;

    // --- W (INF-mask + finite cells) ------------------------------------
    {
        const std::size_t N = res.W.size();
        std::size_t inf_mask_diff = 0;
        std::size_t first_diff = 0;
        bool have_diff = false;
        for (std::size_t k = 0; k < N; ++k) {
            const bool cpp_inf = (res.W[k] >= reamix::remix::INF * 0.9);
            const bool py_inf  = (b.W_expected[k] >= reamix::remix::INF * 0.9);
            if (cpp_inf != py_inf) {
                if (!have_diff) { first_diff = k; have_diff = true; }
                ++inf_mask_diff;
            }
        }
        std::printf("    %-28s  n=%zu  mismatches=%zu  %s\n",
                    "W INF-mask", N, inf_mask_diff,
                    (inf_mask_diff == 0) ? "PASS" : "FAIL");
        if (have_diff) {
            const int n = res.n_beats;
            const std::size_t i = first_diff / n;
            const std::size_t j = first_diff % n;
            std::printf("      first fail at (i=%zu, j=%zu): cpp=%.6g python=%.6g\n",
                        i, j, res.W[first_diff], b.W_expected[first_diff]);
        }
        corpus.total_infmask_mismatches += inf_mask_diff;
        if (inf_mask_diff > 0) track_pass = false;
    }

    {
        const std::size_t N = res.W.size();
        std::size_t n_both_finite = 0;
        std::size_t mismatches    = 0;
        double      max_abs       = 0.0;
        std::size_t first_fail    = 0;
        bool        have_fail     = false;
        for (std::size_t k = 0; k < N; ++k) {
            const bool cpp_fin = (res.W[k] < reamix::remix::INF * 0.9);
            const bool py_fin  = (b.W_expected[k] < reamix::remix::INF * 0.9);
            if (!(cpp_fin && py_fin)) continue;
            ++n_both_finite;
            const double d = std::abs(res.W[k] - b.W_expected[k]);
            if (d > max_abs) max_abs = d;
            if (d > TOL.W_max_abs) {
                if (!have_fail) { first_fail = k; have_fail = true; }
                ++mismatches;
            }
        }
        std::printf("    %-28s  n=%zu  mismatches=%zu  max_abs=%.3e  tol=%.1e  %s\n",
                    "W finite cells", n_both_finite, mismatches, max_abs,
                    TOL.W_max_abs, (mismatches == 0) ? "PASS" : "FAIL");
        if (have_fail) {
            const int n = res.n_beats;
            const std::size_t i = first_fail / n;
            const std::size_t j = first_fail % n;
            std::printf("      first fail at (i=%zu, j=%zu): cpp=%.17g python=%.17g\n",
                        i, j, res.W[first_fail], b.W_expected[first_fail]);
        }
        if (max_abs > corpus.W_finite.max_abs) {
            corpus.W_finite.max_abs = max_abs;
            corpus.W_finite.worst_track = track;
        }
        corpus.W_finite.total_n += n_both_finite;
        corpus.W_finite.total_mismatches += mismatches;
        if (mismatches > 0) track_pass = false;
    }

    // --- Candidates -----------------------------------------------------
    const std::size_t n_cpp  = res.candidates.size();
    const std::size_t n_py   = b.cand_from.size();

    if (n_cpp != n_py) {
        std::printf("    [WARN] candidate count differs (cpp=%zu python=%zu) — "
                    "emitting set-symmetry diagnostic.\n", n_cpp, n_py);
        std::vector<std::pair<int,int>> py_keys;
        py_keys.reserve(n_py);
        for (std::size_t k = 0; k < n_py; ++k) {
            py_keys.emplace_back(static_cast<int>(b.cand_from[k]),
                                 static_cast<int>(b.cand_to[k]));
        }
        std::sort(py_keys.begin(), py_keys.end());
        std::vector<std::pair<int,int>> cpp_keys;
        cpp_keys.reserve(n_cpp);
        for (const auto& kv : res.candidates) cpp_keys.push_back(kv.first);
        std::vector<std::pair<int,int>> cpp_only, py_only;
        std::set_difference(cpp_keys.begin(), cpp_keys.end(),
                            py_keys.begin(),  py_keys.end(),
                            std::back_inserter(cpp_only));
        std::set_difference(py_keys.begin(),  py_keys.end(),
                            cpp_keys.begin(), cpp_keys.end(),
                            std::back_inserter(py_only));
        std::printf("      cpp-only: %zu  python-only: %zu\n",
                    cpp_only.size(), py_only.size());
        const std::size_t show = std::min<std::size_t>(5, std::max(cpp_only.size(), py_only.size()));
        for (std::size_t k = 0; k < show; ++k) {
            if (k < cpp_only.size())
                std::printf("      cpp-only #%zu: (%d, %d)\n", k,
                            cpp_only[k].first, cpp_only[k].second);
            if (k < py_only.size())
                std::printf("      python-only #%zu: (%d, %d)\n", k,
                            py_only[k].first, py_only[k].second);
        }
        corpus.total_candidate_mismatches += (cpp_only.size() + py_only.size());
        track_pass = false;
    } else {
        std::printf("    candidates: cpp=%zu python=%zu (matched count)\n",
                    n_cpp, n_py);
    }

    std::vector<std::int64_t> cpp_from, cpp_to, cpp_lag;
    std::vector<double>       cpp_q, cpp_wf, cpp_succ, cpp_es, cpp_chD, cpp_edb, cpp_tc;
    cpp_from.reserve(n_cpp); cpp_to.reserve(n_cpp); cpp_lag.reserve(n_cpp);
    cpp_q.reserve(n_cpp); cpp_wf.reserve(n_cpp); cpp_succ.reserve(n_cpp);
    cpp_es.reserve(n_cpp); cpp_chD.reserve(n_cpp); cpp_edb.reserve(n_cpp);
    cpp_tc.reserve(n_cpp);
    for (const auto& [key, c] : res.candidates) {
        cpp_from.push_back(key.first);
        cpp_to  .push_back(key.second);
        cpp_q   .push_back(c.quality_score);
        cpp_wf  .push_back(c.waveform_similarity);
        cpp_succ.push_back(c.successor_similarity);
        cpp_es  .push_back(c.edge_splice_similarity);
        cpp_chD .push_back(c.chroma_distance);
        cpp_edb .push_back(c.energy_diff_db);
        cpp_lag .push_back(c.alignment_lag_samples);
        cpp_tc  .push_back(c.total_cost);
    }

    if (n_cpp == n_py) {
        auto R_from = compareInt("cand_from",           cpp_from, b.cand_from, 0);
        auto R_to   = compareInt("cand_to",             cpp_to,   b.cand_to,   0);
        auto R_lag  = compareInt("cand_alignment_lag",  cpp_lag,  b.cand_alignment_lag_samples,
                                 TOL.cand_alignment_lag_tol);
        printReport(R_from, 0);
        printReport(R_to,   0);
        printReport(R_lag,  static_cast<double>(TOL.cand_alignment_lag_tol));
        if (R_from.mismatches > 0 || R_to.mismatches > 0) {
            track_pass = false;
            std::printf("      [key mismatch] skipping field compare — fix keys first.\n");
        } else {
            auto R_q = compareDouble("cand_quality_score", cpp_q, b.cand_quality_score,
                                     TOL.cand_quality_score_max_abs);
            printReport(R_q, TOL.cand_quality_score_max_abs, &cpp_q, &b.cand_quality_score);
            accumulate(corpus.cand_quality_score, R_q, track);

            auto R_wf = compareDouble("cand_waveform_sim", cpp_wf, b.cand_waveform_similarity,
                                      TOL.cand_waveform_max_abs);
            printReport(R_wf, TOL.cand_waveform_max_abs, &cpp_wf, &b.cand_waveform_similarity);
            accumulate(corpus.cand_waveform, R_wf, track);

            auto R_succ = compareDouble("cand_successor_sim", cpp_succ, b.cand_successor_similarity,
                                        TOL.cand_successor_max_abs);
            printReport(R_succ, TOL.cand_successor_max_abs, &cpp_succ, &b.cand_successor_similarity);
            accumulate(corpus.cand_successor, R_succ, track);

            auto R_es = compareDouble("cand_edge_splice_sim", cpp_es, b.cand_edge_splice_similarity,
                                      TOL.cand_edge_splice_max_abs);
            printReport(R_es, TOL.cand_edge_splice_max_abs, &cpp_es, &b.cand_edge_splice_similarity);
            accumulate(corpus.cand_edge_splice, R_es, track);

            auto R_chD = compareDouble("cand_chroma_distance", cpp_chD, b.cand_chroma_distance,
                                       TOL.cand_chroma_distance_max_abs);
            printReport(R_chD, TOL.cand_chroma_distance_max_abs, &cpp_chD, &b.cand_chroma_distance);
            accumulate(corpus.cand_chroma_distance, R_chD, track);

            auto R_edb = compareDouble("cand_energy_diff_db", cpp_edb, b.cand_energy_diff_db,
                                       TOL.cand_energy_diff_db_max_abs);
            printReport(R_edb, TOL.cand_energy_diff_db_max_abs, &cpp_edb, &b.cand_energy_diff_db);
            accumulate(corpus.cand_energy_diff_db, R_edb, track);

            auto R_tc = compareDouble("cand_total_cost", cpp_tc, b.cand_total_cost,
                                      TOL.cand_total_cost_max_abs);
            printReport(R_tc, TOL.cand_total_cost_max_abs, &cpp_tc, &b.cand_total_cost);
            accumulate(corpus.cand_total_cost, R_tc, track);

            if (R_q.mismatches || R_wf.mismatches || R_succ.mismatches
                || R_es.mismatches || R_chD.mismatches || R_edb.mismatches
                || R_tc.mismatches || R_lag.mismatches) {
                track_pass = false;
            }
        }
    }

    std::printf("  -> track %s: %s\n\n", track.c_str(),
                track_pass ? "PASS" : "FAIL");
    if (track_pass) ++corpus.tracks_pass; else ++corpus.tracks_fail;
    return track_pass;
}

void printCorpusSummary(const CorpusAcc& c)
{
    std::printf("\n=== 16-track corpus summary ===\n");
    std::printf("  tracks PASS: %zu   FAIL: %zu\n",
                c.tracks_pass, c.tracks_fail);
    std::printf("  INF-mask mismatches (corpus sum): %zu\n",
                c.total_infmask_mismatches);
    std::printf("  candidate-set mismatches (corpus sum): %zu\n",
                c.total_candidate_mismatches);
    std::printf("\n");
    std::printf("  %-28s  %-10s  %-28s  %s\n",
                "signal", "max_abs", "worst_track", "mismatches");
    auto row = [&](const std::string& name, const SignalAcc& a) {
        std::printf("  %-28s  %.3e  %-28s  %zu / %zu\n",
                    name.c_str(), a.max_abs,
                    (a.worst_track.empty() ? "(none)" : a.worst_track.c_str()),
                    a.total_mismatches, a.total_n);
    };
    row("chroma_D",              c.chroma_D);
    row("importance",            c.importance);
    row("W finite",              c.W_finite);
    row("cand_quality_score",    c.cand_quality_score);
    row("cand_waveform_sim",     c.cand_waveform);
    row("cand_successor_sim",    c.cand_successor);
    row("cand_edge_splice_sim",  c.cand_edge_splice);
    row("cand_chroma_distance",  c.cand_chroma_distance);
    row("cand_energy_diff_db",   c.cand_energy_diff_db);
    row("cand_total_cost",       c.cand_total_cost);
}

} // namespace

int main(int argc, char** argv)
{
    std::string baseDir = "tests/parity/reference/data/phase4";
    bool noStructure    = false;
    std::string singleTrackArg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mode=no-structure") {
            noStructure = true;
            baseDir     = "tests/parity/reference/data/phase4_no_structure";
        } else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "Unknown flag: %s\n", a.c_str());
            std::fprintf(stderr,
                "Usage: test_transition_cost [--mode=no-structure] [<track-or-dir>]\n");
            return 2;
        } else {
            singleTrackArg = a;
        }
    }

    if (!singleTrackArg.empty()) {
        // Single-track bisection mode: arg is either a track name or a dir.
        std::string dataDir, trackName;
        if (singleTrackArg.find('/') == std::string::npos) {
            trackName = singleTrackArg;
            dataDir   = baseDir + "/" + singleTrackArg;
        } else {
            dataDir   = singleTrackArg;
            auto slash = singleTrackArg.find_last_of('/');
            trackName = (slash == std::string::npos)
                ? singleTrackArg : singleTrackArg.substr(slash + 1);
        }
        std::printf("=== test_transition_cost (single-track bisection: %s%s) ===\n",
                    trackName.c_str(),
                    noStructure ? " | mode=no-structure" : "");
        CorpusAcc c;
        const bool ok = runTrack(dataDir, trackName, c);
        return ok ? 0 : 1;
    }

    std::printf("=== test_transition_cost (%s — 16-track corpus) ===\n",
                noStructure ? "ADR-044 no-structure"
                            : "phase-4 session 19 status-quo");
    std::printf("base dir: %s\n\n", baseDir.c_str());

    CorpusAcc corpus;
    bool all_pass = true;
    for (const auto& t : kCorpusTracks) {
        const bool ok = runTrack(baseDir + "/" + t, t, corpus);
        if (!ok) all_pass = false;
    }

    printCorpusSummary(corpus);

    std::printf("\n=== transition_cost_parity%s: %s ===\n",
                noStructure ? " (no-structure)" : "",
                all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
