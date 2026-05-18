// test_region_optimizer — phase-4 session 23 parity:
// reamix::remix::RegionOptimizer + reamix::remix::computeRegionCosts vs
// Python references/python-source/remix/region_optimizer.py + region_cost.py.
//
// Session-23 smoke: billie_jean middle-50% × 0.8 (shortening). Corpus mode
// extends to 16 tracks × {0.5, 0.8, 1.2} = 48 cases; 1.2 exercises the
// `is_extending` branch with +0.5 backward-jump penalty at
// region_optimizer.py:158.
//
// Per-gate breakdown (6 stages):
//   Stage 0: region_W — f64 element-wise. Inherits session-19 W L∞ ≤ 0.02
//                       parity gate (corpus max_abs 1.128e-07). Tol 1e-6.
//   Stage 1: region_candidates — bitwise key-set equality + per-field sub-
//                       gates matching session-18/19 TransitionCost classes.
//   Stage 2: region_path — int64 bitwise. STRONGER than phase-4 spec L37
//                       ≥ 90% match; targeting 100% per Principle 6.
//   Stage 3: region_total_cost — f64 ULP ≤ 5e-4.
//   Stage 4: transition pair set — set-equal (lex-sorted on both sides).
//   Stage 5: per-transition metadata — 8 keys (subset of session-22's 10 —
//                       RegionRemixMixin emits no is_repetition_jump or
//                       chroma_correlation markers).
//
// Invocation:
//   test_region_optimizer                   → corpus mode (48 cases).
//   test_region_optimizer <region_dir>      → single-case bisection
//       e.g. `.../billie_jean/region/r0.8`.

#include "remix/Path.h"
#include "remix/RegionCost.h"
#include "remix/RegionOptimizer.h"
#include "remix/TransitionCost.h"
#include "analysis/StructureResult.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "NpyIO.h"

namespace {

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

// Region corpus ratios: 0.5/0.8 shortening, 1.2 extending.
const std::vector<double> kCorpusRatios = { 0.5, 0.8, 1.2 };

std::string ratioLabel(double r)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "r%.1f", r);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Tolerances (per-stage budget)
// ---------------------------------------------------------------------------

struct Tolerances
{
    // Stage 0: region_W — same ULP class as session-19 16-track corpus
    // (max_abs 1.128e-07). Gate 1e-6 (9× margin).
    double region_W_max_abs = 1e-6;

    // Stage 1: region_candidates — sub-gates match session-18 TransitionCost
    // per-signal class.
    double cand_quality_score_tol        = 5e-4;
    double cand_waveform_similarity_tol  = 1e-9;
    double cand_successor_similarity_tol = 5e-6;
    double cand_edge_splice_similarity_tol = 5e-6;
    double cand_chroma_distance_tol      = 5e-6;
    double cand_energy_diff_db_tol       = 1e-12;
    double cand_alignment_offset_sec_tol = 1e-12;
    double cand_total_cost_tol           = 5e-4;

    // Stage 3: region_total_cost — composite DP accumulator.
    double total_cost_max_abs = 5e-4;

    // Stage 5: per-key metadata — identical classes to Stage 1 candidates
    // since metadata values are directly copied from matched candidate.
    double meta_quality_score_tol        = 5e-4;
    double meta_waveform_similarity_tol  = 1e-9;
    double meta_successor_similarity_tol = 5e-6;
    double meta_edge_splice_similarity_tol = 5e-6;
    double meta_chroma_distance_tol      = 5e-6;
    double meta_energy_diff_db_tol       = 1e-12;
    double meta_alignment_offset_sec_tol = 1e-12;
    double meta_total_cost_tol           = 5e-4;
};

const Tolerances TOL;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

std::vector<std::string> readLines(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open text file: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
    }
    return out;
}

std::string readAll(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

bool extractDouble(const std::string& json, const std::string& key, double& out)
{
    const std::string pat = "\"" + key + "\":";
    std::size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    char* end = nullptr;
    double v = std::strtod(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return false;
    out = v;
    return true;
}

bool extractInt(const std::string& json, const std::string& key, long long& out)
{
    const std::string pat = "\"" + key + "\":";
    std::size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    char* end = nullptr;
    long long v = std::strtoll(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    out = v;
    return true;
}

// ---------------------------------------------------------------------------
// Case bundle
// ---------------------------------------------------------------------------

struct Case
{
    std::string track_dir;
    std::string region_dir;
    std::string track;
    std::string ratio_label;

    // Track-root inputs (reused from session-18/19 dumps).
    std::vector<double>                       beat_times;
    std::vector<float>                        features;
    int                                       n_features = 0;
    std::vector<reamix::analysis::Segment>    segments;
    std::vector<double>                       downbeats;
    int                                       n_beats = 0;

    // Edge arrays + per-beat scalars — loaded from track root for RegionCost.
    std::vector<float>  edge_features_start;
    std::vector<float>  edge_features_end;
    int                 n_edge_features = 0;
    std::vector<double> edge_rms_start;
    std::vector<double> edge_rms_end;
    std::vector<double> rms_energy;
    std::vector<double> onset_strength;
    std::vector<double> spectral_centroid;
    std::vector<double> vocal_activity;
    std::vector<double> edge_vocal_activity_start;
    std::vector<double> edge_vocal_activity_end;
    // boundary_waveforms — (n_bnd, n_samples) row-major f32.
    std::vector<float>  boundary_waveforms;
    int                 n_boundary_waveforms = 0;
    int                 n_samples_per_bnd    = 0;
    int                 waveform_sample_rate = 0;

    // Global candidates map (session-18) — merged into region candidates at
    // _region_transitions per region_optimizer.py:328-330.
    std::map<std::pair<int, int>, reamix::remix::TransitionCandidate> global_candidates;

    // -- RegionCost goldens ----------------------------------------------
    std::vector<double>                                                  region_W_expected;
    std::map<std::pair<int, int>, reamix::remix::TransitionCandidate>    region_candidates_expected;

    // -- RegionOptimizer goldens ------------------------------------------
    std::vector<std::int64_t>                 region_path_expected;
    double                                    region_total_cost_expected = 0.0;
    std::vector<std::pair<int, int>>          meta_pairs_expected;
    std::vector<double> meta_quality_score;
    std::vector<double> meta_waveform_similarity;
    std::vector<double> meta_successor_similarity;
    std::vector<double> meta_edge_splice_similarity;
    std::vector<double> meta_chroma_distance;
    std::vector<double> meta_energy_diff_db;
    std::vector<double> meta_alignment_offset_sec;
    std::vector<double> meta_total_cost;

    // From manifest.
    double target_duration    = 0.0;
    double region_start_sec   = 0.0;
    double region_end_sec     = 0.0;
    double avg_beat_duration  = 0.0;
    double duration_tolerance_sec = 5.0;
    int    entry_beat_expected = 0;
    int    exit_beat_expected  = 0;
    int    n_region_expected   = 0;
};

template <typename T>
std::vector<T> loadOpt(const std::string& path,
                        std::vector<T> (*fn)(const std::string&))
{
    try { return fn(path); } catch (...) { return {}; }
}

Case loadCase(const std::string& track_dir,
              const std::string& ratio_label,
              const std::string& track)
{
    Case c;
    c.track_dir   = track_dir;
    c.region_dir  = track_dir + "/region/" + ratio_label;
    c.track       = track;
    c.ratio_label = ratio_label;

    // -- Track-root inputs -------------------------------------------------
    c.beat_times = reamix::test::loadNpy1DFloat64(track_dir + "/beat_times.npy");
    c.n_beats    = static_cast<int>(c.beat_times.size());

    {
        auto feat = reamix::test::loadNpy2DFloat32(track_dir + "/features.npy");
        c.features   = std::move(feat.data);
        c.n_features = static_cast<int>(feat.cols);
    }

    // Segments from track dumps.
    {
        auto seg_start      = reamix::test::loadNpy1DFloat64(track_dir + "/seg_start.npy");
        auto seg_end        = reamix::test::loadNpy1DFloat64(track_dir + "/seg_end.npy");
        auto seg_confidence = reamix::test::loadNpy1DFloat64(track_dir + "/seg_confidence.npy");
        auto seg_cluster_id = reamix::test::loadNpy1DInt64(  track_dir + "/seg_cluster_id.npy");
        auto seg_labels     = readLines(                     track_dir + "/seg_labels.txt");
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

    // Edge arrays + scalars (session-18 dumps).
    {
        auto ef_start = reamix::test::loadNpy2DFloat32(track_dir + "/edge_features_start.npy");
        auto ef_end   = reamix::test::loadNpy2DFloat32(track_dir + "/edge_features_end.npy");
        c.edge_features_start = std::move(ef_start.data);
        c.edge_features_end   = std::move(ef_end.data);
        c.n_edge_features     = static_cast<int>(ef_start.cols);
    }
    c.edge_rms_start            = reamix::test::loadNpy1DFloat64(track_dir + "/edge_rms_start.npy");
    c.edge_rms_end              = reamix::test::loadNpy1DFloat64(track_dir + "/edge_rms_end.npy");
    c.rms_energy                = reamix::test::loadNpy1DFloat64(track_dir + "/rms_energy.npy");
    c.onset_strength            = reamix::test::loadNpy1DFloat64(track_dir + "/onset_strength.npy");
    c.spectral_centroid         = reamix::test::loadNpy1DFloat64(track_dir + "/spectral_centroid.npy");
    c.vocal_activity            = reamix::test::loadNpy1DFloat64(track_dir + "/vocal_activity.npy");
    c.edge_vocal_activity_start = reamix::test::loadNpy1DFloat64(track_dir + "/edge_vocal_activity_start.npy");
    c.edge_vocal_activity_end   = reamix::test::loadNpy1DFloat64(track_dir + "/edge_vocal_activity_end.npy");

    {
        auto bw = reamix::test::loadNpy2DFloat32(track_dir + "/boundary_waveforms.npy");
        c.boundary_waveforms    = std::move(bw.data);
        c.n_boundary_waveforms  = static_cast<int>(bw.rows);
        c.n_samples_per_bnd     = static_cast<int>(bw.cols);
    }

    // Session-18 track-level manifest has waveform_sample_rate.
    {
        const std::string track_manifest = readAll(track_dir + "/manifest.json");
        long long sr_ll = 0;
        if (extractInt(track_manifest, "waveform_sample_rate", sr_ll)) {
            c.waveform_sample_rate = static_cast<int>(sr_ll);
        } else {
            // Fallback: try "sample_rate" or default to 22050.
            c.waveform_sample_rate = 22050;
        }
    }

    // -- Global candidates map (session-18) ------------------------------
    {
        auto cf  = reamix::test::loadNpy1DInt64(track_dir + "/cand_from.npy");
        auto ct  = reamix::test::loadNpy1DInt64(track_dir + "/cand_to.npy");
        auto cq  = reamix::test::loadNpy1DFloat64(track_dir + "/cand_quality_score.npy");
        auto cw  = reamix::test::loadNpy1DFloat64(track_dir + "/cand_waveform_similarity.npy");
        auto cs  = reamix::test::loadNpy1DFloat64(track_dir + "/cand_successor_similarity.npy");
        auto ce  = reamix::test::loadNpy1DFloat64(track_dir + "/cand_edge_splice_similarity.npy");
        auto cd  = reamix::test::loadNpy1DFloat64(track_dir + "/cand_chroma_distance.npy");
        auto cen = reamix::test::loadNpy1DFloat64(track_dir + "/cand_energy_diff_db.npy");
        auto cal = reamix::test::loadNpy1DInt64  (track_dir + "/cand_alignment_lag_samples.npy");
        auto ctc = reamix::test::loadNpy1DFloat64(track_dir + "/cand_total_cost.npy");
        const std::size_t n_cand = cf.size();
        for (std::size_t k = 0; k < n_cand; ++k) {
            reamix::remix::TransitionCandidate tc;
            tc.from_beat              = static_cast<int>(cf[k]);
            tc.to_beat                = static_cast<int>(ct[k]);
            tc.quality_score          = cq[k];
            tc.waveform_similarity    = cw[k];
            tc.successor_similarity   = cs[k];
            tc.edge_splice_similarity = ce[k];
            tc.chroma_distance        = cd[k];
            tc.energy_diff_db         = cen[k];
            tc.alignment_lag_samples  = static_cast<int>(cal[k]);
            tc.total_cost             = ctc[k];
            c.global_candidates[{tc.from_beat, tc.to_beat}] = tc;
        }
    }

    // -- RegionCost goldens (this case) -----------------------------------
    c.region_W_expected = reamix::test::loadNpy1DFloat64(c.region_dir + "/region_W.npy");
    {
        auto cp  = reamix::test::loadNpy1DInt64  (c.region_dir + "/cand_pairs.npy");
        auto cq  = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_quality_score.npy");
        auto cw  = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_waveform_similarity.npy");
        auto cs  = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_successor_similarity.npy");
        auto ce  = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_edge_splice_similarity.npy");
        auto cd  = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_chroma_distance.npy");
        auto cen = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_energy_diff_db.npy");
        auto cal = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_alignment_lag_samples.npy");
        auto ctc = reamix::test::loadNpy1DFloat64(c.region_dir + "/cand_total_cost.npy");
        const std::size_t n_cand = cp.size() / 2;
        for (std::size_t k = 0; k < n_cand; ++k) {
            reamix::remix::TransitionCandidate tc;
            tc.from_beat              = static_cast<int>(cp[2 * k + 0]);
            tc.to_beat                = static_cast<int>(cp[2 * k + 1]);
            tc.quality_score          = cq[k];
            tc.waveform_similarity    = cw[k];
            tc.successor_similarity   = cs[k];
            tc.edge_splice_similarity = ce[k];
            tc.chroma_distance        = cd[k];
            tc.energy_diff_db         = cen[k];
            tc.alignment_lag_samples  = static_cast<int>(cal[k]);
            tc.total_cost             = ctc[k];
            c.region_candidates_expected[{tc.from_beat, tc.to_beat}] = tc;
        }
    }

    // -- RegionOptimizer goldens ------------------------------------------
    c.region_path_expected = reamix::test::loadNpy1DInt64(c.region_dir + "/region_path.npy");
    {
        auto tc_vec = reamix::test::loadNpy1DFloat64(c.region_dir + "/region_total_cost.npy");
        if (tc_vec.size() != 1) {
            throw std::runtime_error("region_total_cost.npy expected size-1 vector");
        }
        c.region_total_cost_expected = tc_vec[0];
    }
    {
        auto flat = reamix::test::loadNpy1DInt64(c.region_dir + "/meta_pairs.npy");
        if (flat.size() % 2 != 0) {
            throw std::runtime_error("meta_pairs.npy size must be even");
        }
        c.meta_pairs_expected.reserve(flat.size() / 2);
        for (std::size_t k = 0; k + 1 < flat.size(); k += 2) {
            c.meta_pairs_expected.emplace_back(static_cast<int>(flat[k]),
                                               static_cast<int>(flat[k + 1]));
        }
    }
    c.meta_quality_score          = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_quality_score.npy");
    c.meta_waveform_similarity    = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_waveform_similarity.npy");
    c.meta_successor_similarity   = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_successor_similarity.npy");
    c.meta_edge_splice_similarity = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_edge_splice_similarity.npy");
    c.meta_chroma_distance        = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_chroma_distance.npy");
    c.meta_energy_diff_db         = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_energy_diff_db.npy");
    c.meta_alignment_offset_sec   = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_alignment_offset_sec.npy");
    c.meta_total_cost             = reamix::test::loadNpy1DFloat64(c.region_dir + "/meta_total_cost.npy");

    // Manifest.
    const std::string manifest_json = readAll(c.region_dir + "/manifest.json");
    if (!extractDouble(manifest_json, "target_duration",        c.target_duration))        throw std::runtime_error("Missing target_duration");
    if (!extractDouble(manifest_json, "region_start_sec",       c.region_start_sec))       throw std::runtime_error("Missing region_start_sec");
    if (!extractDouble(manifest_json, "region_end_sec",         c.region_end_sec))         throw std::runtime_error("Missing region_end_sec");
    if (!extractDouble(manifest_json, "avg_beat_duration",      c.avg_beat_duration))      throw std::runtime_error("Missing avg_beat_duration");
    if (!extractDouble(manifest_json, "duration_tolerance_sec", c.duration_tolerance_sec)) c.duration_tolerance_sec = 5.0;
    long long v = 0;
    if (extractInt(manifest_json, "entry_beat", v)) c.entry_beat_expected = static_cast<int>(v);
    if (extractInt(manifest_json, "exit_beat",  v)) c.exit_beat_expected  = static_cast<int>(v);
    if (extractInt(manifest_json, "n_region",   v)) c.n_region_expected   = static_cast<int>(v);

    return c;
}

// ---------------------------------------------------------------------------
// Per-case comparison
// ---------------------------------------------------------------------------

struct CaseSummary
{
    std::string track;
    std::string ratio_label;
    bool        loaded               = false;
    bool        pass                 = false;
    bool        region_W_pass        = false;
    bool        region_cands_pass    = false;
    bool        path_bitwise         = false;
    bool        pairs_equal          = false;
    double      region_W_max_diff    = 0.0;
    double      total_cost_diff      = 0.0;
    int         n_cands_cpp          = 0;
    int         n_cands_py           = 0;
    int         n_trans_cpp          = 0;
    int         n_trans_py           = 0;
    int         path_len_cpp         = 0;
    int         path_len_py          = 0;
    // per-key worst-case diff.
    double worst_quality     = 0.0;
    double worst_waveform    = 0.0;
    double worst_successor   = 0.0;
    double worst_edge_splice = 0.0;
    double worst_chroma_dist = 0.0;
    double worst_energy_db   = 0.0;
    double worst_align_off   = 0.0;
    double worst_total_cost  = 0.0;
    std::string err;
};

std::pair<bool, int> comparePathInt(const std::vector<int>&         cpp,
                                    const std::vector<std::int64_t>& py)
{
    if (cpp.size() != py.size()) return {false, 0};
    for (std::size_t k = 0; k < cpp.size(); ++k) {
        if (static_cast<std::int64_t>(cpp[k]) != py[k]) {
            return {false, static_cast<int>(k)};
        }
    }
    return {true, -1};
}

std::pair<bool, double>
compareMetaKey(const std::string&                                           key_name,
               const std::vector<std::pair<int, int>>&                      py_pairs,
               const std::vector<double>&                                   py_values,
               const std::map<std::pair<int, int>, std::map<std::string, double>>&
                                                                            cpp_meta,
               double                                                       tol,
               std::string&                                                 err_out)
{
    double worst = 0.0;
    bool   pass  = true;
    for (std::size_t k = 0; k < py_pairs.size(); ++k) {
        const auto& pr    = py_pairs[k];
        const bool py_nan = std::isnan(py_values[k]);
        auto outer_it = cpp_meta.find(pr);
        const bool cpp_has_trans = (outer_it != cpp_meta.end());
        const bool cpp_has_key   =
            cpp_has_trans && outer_it->second.count(key_name) > 0;

        if (py_nan && cpp_has_key) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "[%s] (%d, %d) py=absent but cpp=%g",
                key_name.c_str(), pr.first, pr.second,
                outer_it->second.at(key_name));
            err_out = buf; pass = false;
        } else if (!py_nan && !cpp_has_key) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "[%s] (%d, %d) py=%g but cpp=absent",
                key_name.c_str(), pr.first, pr.second, py_values[k]);
            err_out = buf; pass = false;
        } else if (!py_nan && cpp_has_key) {
            const double cpp_v = outer_it->second.at(key_name);
            const double diff  = std::abs(cpp_v - py_values[k]);
            if (diff > worst) worst = diff;
            if (diff > tol) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "[%s] (%d, %d) py=%g cpp=%g |diff|=%g > tol=%g",
                    key_name.c_str(), pr.first, pr.second,
                    py_values[k], cpp_v, diff, tol);
                err_out = buf; pass = false;
            }
        }
    }
    return {pass, worst};
}

CaseSummary runCase(const Case& c)
{
    CaseSummary out;
    out.track       = c.track;
    out.ratio_label = c.ratio_label;
    out.loaded      = true;
    out.n_cands_py  = static_cast<int>(c.region_candidates_expected.size());
    out.n_trans_py  = static_cast<int>(c.meta_pairs_expected.size());
    out.path_len_py = static_cast<int>(c.region_path_expected.size());

    std::printf("\n=== %s [%s] ===\n", c.track.c_str(), c.ratio_label.c_str());
    std::printf("  n_beats=%d entry=%d exit=%d n_region=%d "
                "target_dur=%.4fs region_dur=%.4fs\n",
                c.n_beats, c.entry_beat_expected, c.exit_beat_expected,
                c.n_region_expected, c.target_duration,
                c.region_end_sec - c.region_start_sec);

    // -- Stage 0: compute region_W ----------------------------------------
    reamix::remix::RegionCostInputs rci{};
    rci.entry_beat           = c.entry_beat_expected;
    rci.exit_beat            = c.exit_beat_expected;
    rci.features             = c.features.data();
    rci.n_total              = c.n_beats;
    rci.n_features           = c.n_features;
    rci.beat_times           = c.beat_times.data();
    rci.segments             = c.segments.empty() ? nullptr : c.segments.data();
    rci.n_segments           = static_cast<int>(c.segments.size());
    rci.boundary_waveforms   = c.boundary_waveforms.empty() ? nullptr : c.boundary_waveforms.data();
    rci.n_boundary_waveforms = c.n_boundary_waveforms;
    rci.n_samples_per_bnd    = c.n_samples_per_bnd;
    rci.waveform_sample_rate = c.waveform_sample_rate;
    rci.edge_rms_start            = c.edge_rms_start.empty() ? nullptr : c.edge_rms_start.data();
    rci.edge_rms_end              = c.edge_rms_end.empty()   ? nullptr : c.edge_rms_end.data();
    rci.edge_features_start       = c.edge_features_start.empty() ? nullptr : c.edge_features_start.data();
    rci.edge_features_end         = c.edge_features_end.empty()   ? nullptr : c.edge_features_end.data();
    rci.n_edge_features           = c.n_edge_features;
    rci.rms_energy                = c.rms_energy.empty() ? nullptr : c.rms_energy.data();
    rci.onset_strength            = c.onset_strength.empty() ? nullptr : c.onset_strength.data();
    rci.spectral_centroid         = c.spectral_centroid.empty() ? nullptr : c.spectral_centroid.data();
    rci.vocal_activity            = c.vocal_activity.empty() ? nullptr : c.vocal_activity.data();
    rci.edge_vocal_activity_start = c.edge_vocal_activity_start.empty() ? nullptr : c.edge_vocal_activity_start.data();
    rci.edge_vocal_activity_end   = c.edge_vocal_activity_end.empty()   ? nullptr : c.edge_vocal_activity_end.data();
    rci.downbeats                 = c.downbeats.empty() ? nullptr : c.downbeats.data();
    rci.n_downbeats               = static_cast<int>(c.downbeats.size());
    rci.time_signature            = 4;  // session-23 corpus all 4/4
    // Sesja 81 ADR-068 D1+D4 default flip — pin to legacy 10-component
    // simplex so phase-4 RegionCost + RegionOptimizer goldens stay bit-exact
    // past kDefaultQualityWeights shipping the new 6-D production simplex.
    rci.quality_weights           = &reamix::remix::kLegacyQualityWeights;

    reamix::remix::RegionCostResult rc_out = reamix::remix::computeRegionCosts(rci);
    out.n_cands_cpp = static_cast<int>(rc_out.candidates.size());

    // Stage 0: region_W element-wise.
    double worst_W = 0.0;
    const int n_region = rc_out.n_region;
    if (static_cast<int>(c.region_W_expected.size()) != n_region * n_region) {
        std::printf("  [FAIL] region_W size mismatch: cpp=%d² vs py=%zu (expected %d)\n",
                    n_region, c.region_W_expected.size(), n_region * n_region);
    } else {
        for (int i = 0; i < n_region * n_region; ++i) {
            double cpp_v = rc_out.region_W[i];
            double py_v  = c.region_W_expected[i];
            // Both INF → skip (cap ULP noise on sentinels).
            if (cpp_v >= reamix::remix::INF && py_v >= reamix::remix::INF) continue;
            const double diff = std::abs(cpp_v - py_v);
            if (diff > worst_W) worst_W = diff;
        }
    }
    out.region_W_max_diff = worst_W;
    out.region_W_pass     = worst_W <= TOL.region_W_max_abs;
    std::printf("  [%s] region_W: worst=%.3e (tol %.0e, n_region=%d)\n",
                out.region_W_pass ? "PASS" : "FAIL", worst_W,
                TOL.region_W_max_abs, n_region);

    // -- Stage 1: region_candidates key set + per-field sub-gates ---------
    bool cands_keyset_ok = (out.n_cands_cpp == out.n_cands_py);
    if (cands_keyset_ok) {
        auto it_cpp = rc_out.candidates.begin();
        auto it_py  = c.region_candidates_expected.begin();
        for (; it_cpp != rc_out.candidates.end(); ++it_cpp, ++it_py) {
            if (it_cpp->first != it_py->first) {
                cands_keyset_ok = false;
                std::printf("  [FAIL] cand key mismatch: cpp=(%d,%d) py=(%d,%d)\n",
                            it_cpp->first.first, it_cpp->first.second,
                            it_py->first.first, it_py->first.second);
                break;
            }
        }
    }
    double worst_cand_q = 0.0, worst_cand_w = 0.0, worst_cand_s = 0.0,
           worst_cand_e = 0.0, worst_cand_cd = 0.0, worst_cand_edb = 0.0,
           worst_cand_aos = 0.0, worst_cand_tc = 0.0;
    bool cands_sub_ok = true;
    if (cands_keyset_ok) {
        auto it_cpp = rc_out.candidates.begin();
        auto it_py  = c.region_candidates_expected.begin();
        for (; it_cpp != rc_out.candidates.end(); ++it_cpp, ++it_py) {
            const auto& a = it_cpp->second;
            const auto& b = it_py->second;
            auto upd = [](double& worst, double diff, double tol, bool& ok,
                          const char* key, int i, int j) {
                if (diff > worst) worst = diff;
                if (diff > tol) {
                    std::printf("    [FAIL cand][%s] (%d,%d) |diff|=%g > tol=%g\n",
                                key, i, j, diff, tol);
                    ok = false;
                }
            };
            const int i = a.from_beat, j = a.to_beat;
            upd(worst_cand_q, std::abs(a.quality_score - b.quality_score),        TOL.cand_quality_score_tol,        cands_sub_ok, "q",   i, j);
            upd(worst_cand_w, std::abs(a.waveform_similarity - b.waveform_similarity), TOL.cand_waveform_similarity_tol, cands_sub_ok, "w",   i, j);
            upd(worst_cand_s, std::abs(a.successor_similarity - b.successor_similarity), TOL.cand_successor_similarity_tol, cands_sub_ok, "s", i, j);
            upd(worst_cand_e, std::abs(a.edge_splice_similarity - b.edge_splice_similarity), TOL.cand_edge_splice_similarity_tol, cands_sub_ok, "e", i, j);
            upd(worst_cand_cd, std::abs(a.chroma_distance - b.chroma_distance),  TOL.cand_chroma_distance_tol,      cands_sub_ok, "cd",  i, j);
            upd(worst_cand_edb, std::abs(a.energy_diff_db - b.energy_diff_db),   TOL.cand_energy_diff_db_tol,       cands_sub_ok, "edb", i, j);
            // alignment_lag_samples: bitwise int
            if (a.alignment_lag_samples != b.alignment_lag_samples) {
                std::printf("    [FAIL cand][lag] (%d,%d) cpp=%d py=%d\n",
                            i, j, a.alignment_lag_samples, b.alignment_lag_samples);
                cands_sub_ok = false;
            }
            upd(worst_cand_tc, std::abs(a.total_cost - b.total_cost),            TOL.cand_total_cost_tol,           cands_sub_ok, "tc",  i, j);
            (void)worst_cand_aos;  // computed in meta stage instead
        }
    }
    out.region_cands_pass = cands_keyset_ok && cands_sub_ok;
    std::printf("  [%s] region_candidates: n_cpp=%d n_py=%d keyset=%s sub-gates=%s\n"
                "         worst q=%.3e w=%.3e s=%.3e e=%.3e cd=%.3e edb=%.3e tc=%.3e\n",
                out.region_cands_pass ? "PASS" : "FAIL",
                out.n_cands_cpp, out.n_cands_py,
                cands_keyset_ok ? "OK" : "MISMATCH",
                cands_sub_ok    ? "OK" : "MISMATCH",
                worst_cand_q, worst_cand_w, worst_cand_s, worst_cand_e,
                worst_cand_cd, worst_cand_edb, worst_cand_tc);

    // -- RegionOptimizer invocation ---------------------------------------
    reamix::remix::RegionOptimizerInputs roi{};
    roi.n_beats               = c.n_beats;
    roi.beat_times            = c.beat_times.data();
    roi.avg_beat_duration     = c.avg_beat_duration;
    roi.duration_tolerance_sec = c.duration_tolerance_sec;
    roi.candidates            = &c.global_candidates;
    roi.sample_rate           = c.waveform_sample_rate;

    reamix::remix::RegionOptimizer ro(roi);
    reamix::remix::RemixPath rp = ro.remix(
        c.target_duration,
        c.region_start_sec,
        c.region_end_sec,
        rc_out.region_W.data(),
        rc_out.n_region,
        &rc_out.candidates);
    out.n_trans_cpp  = static_cast<int>(rp.transitions.size());
    out.path_len_cpp = static_cast<int>(rp.beat_indices.size());

    std::printf("  entry=%d exit=%d n_region=%d is_extending=%d\n",
                ro.entryBeat(), ro.exitBeat(), rc_out.n_region,
                ro.isExtending() ? 1 : 0);
    std::printf("  C++ path_len=%d n_trans=%d total_cost=%.6f\n",
                out.path_len_cpp, out.n_trans_cpp, rp.total_cost);

    // -- Stage 2: region_path ---------------------------------------------
    auto [path_pass, path_mis_idx] = comparePathInt(rp.beat_indices, c.region_path_expected);
    out.path_bitwise = path_pass;
    if (!path_pass) {
        if (static_cast<std::size_t>(path_mis_idx) < rp.beat_indices.size()
            && static_cast<std::size_t>(path_mis_idx) < c.region_path_expected.size())
        {
            std::printf("  [FAIL] region_path mismatch at idx=%d: cpp=%d vs py=%lld (sizes %d vs %d)\n",
                        path_mis_idx,
                        rp.beat_indices[path_mis_idx],
                        static_cast<long long>(c.region_path_expected[path_mis_idx]),
                        out.path_len_cpp, out.path_len_py);
        } else {
            std::printf("  [FAIL] region_path size mismatch: cpp=%d vs py=%d\n",
                        out.path_len_cpp, out.path_len_py);
        }
    } else {
        std::printf("  [PASS] region_path bitwise exact (%d beats)\n", out.path_len_cpp);
    }

    // -- Stage 3: total_cost ----------------------------------------------
    out.total_cost_diff = std::abs(rp.total_cost - c.region_total_cost_expected);
    const bool tc_pass  = out.total_cost_diff <= TOL.total_cost_max_abs;
    std::printf("  [%s] region_total_cost: cpp=%.6f py=%.6f |diff|=%.6e (tol %.0e)\n",
                tc_pass ? "PASS" : "FAIL",
                rp.total_cost, c.region_total_cost_expected,
                out.total_cost_diff, TOL.total_cost_max_abs);

    // -- Stage 4: transition pair set -------------------------------------
    std::vector<std::pair<int, int>> cpp_pairs_sorted;
    cpp_pairs_sorted.reserve(rp.transition_metadata.size());
    for (const auto& [pr, _] : rp.transition_metadata) {
        cpp_pairs_sorted.push_back(pr);
    }
    bool pairs_pass = (cpp_pairs_sorted == c.meta_pairs_expected);
    out.pairs_equal = pairs_pass;
    if (!pairs_pass) {
        std::printf("  [FAIL] transition pair set mismatch: cpp_n=%zu py_n=%zu\n",
                    cpp_pairs_sorted.size(), c.meta_pairs_expected.size());
    } else {
        std::printf("  [PASS] transition pair set equal (%zu pairs)\n",
                    c.meta_pairs_expected.size());
    }

    // -- Stage 5: per-key metadata ----------------------------------------
    bool meta_pass = true;
    if (pairs_pass) {
        struct KeyProbe {
            const char*                key;
            const std::vector<double>* values;
            double                     tol;
            double*                    worst_out;
        };
        const KeyProbe probes[] = {
            {"quality_score",          &c.meta_quality_score,          TOL.meta_quality_score_tol,        &out.worst_quality},
            {"waveform_similarity",    &c.meta_waveform_similarity,    TOL.meta_waveform_similarity_tol,  &out.worst_waveform},
            {"successor_similarity",   &c.meta_successor_similarity,   TOL.meta_successor_similarity_tol, &out.worst_successor},
            {"edge_splice_similarity", &c.meta_edge_splice_similarity, TOL.meta_edge_splice_similarity_tol, &out.worst_edge_splice},
            {"chroma_distance",        &c.meta_chroma_distance,        TOL.meta_chroma_distance_tol,      &out.worst_chroma_dist},
            {"energy_diff_db",         &c.meta_energy_diff_db,         TOL.meta_energy_diff_db_tol,       &out.worst_energy_db},
            {"alignment_offset_sec",   &c.meta_alignment_offset_sec,   TOL.meta_alignment_offset_sec_tol, &out.worst_align_off},
            {"total_cost",             &c.meta_total_cost,             TOL.meta_total_cost_tol,           &out.worst_total_cost},
        };
        for (const auto& p : probes) {
            std::string err;
            auto [pass, worst] = compareMetaKey(p.key, c.meta_pairs_expected,
                                                *p.values,
                                                rp.transition_metadata,
                                                p.tol, err);
            *p.worst_out = worst;
            std::printf("  [%s] meta[%s]: worst=%.3e (tol %.0e) n_pairs=%zu%s%s\n",
                        pass ? "PASS" : "FAIL",
                        p.key, worst, p.tol,
                        c.meta_pairs_expected.size(),
                        err.empty() ? "" : " — ", err.c_str());
            if (!pass) meta_pass = false;
        }
    } else {
        std::printf("  [SKIP] per-key metadata (pairs mismatched; not comparable)\n");
        meta_pass = false;
    }

    out.pass = out.region_W_pass && out.region_cands_pass
               && path_pass && tc_pass && pairs_pass && meta_pass;
    std::printf("  *** CASE %s ***\n", out.pass ? "PASS" : "FAIL");
    return out;
}

// Split `.../<track_dir>/region/r<R>` → (track_dir, ratio_label).
bool splitRegionPath(const std::string& region_path,
                     std::string&       out_track_dir,
                     std::string&       out_ratio_label)
{
    std::string p = region_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    const std::size_t last = p.find_last_of('/');
    if (last == std::string::npos) return false;
    out_ratio_label = p.substr(last + 1);
    if (out_ratio_label.empty() || out_ratio_label[0] != 'r') return false;
    std::string middle = p.substr(0, last);
    const std::size_t mid = middle.find_last_of('/');
    if (mid == std::string::npos) return false;
    if (middle.substr(mid + 1) != "region") return false;
    out_track_dir = middle.substr(0, mid);
    return true;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    // ADR-044 close-out (session 53): `--mode=no-structure` switches both the
    // base goldens dir AND the per-track segment context to empty, exercising
    // the C++ RegionCost noStructure gating (`section_sim=0` formula skip,
    // SPAN_PENALTY_CROSS_SECTION halved-branch skip). Tolerances unchanged.
    std::string PHASE4_ROOT =
        "tests/parity/reference/data/phase4";
    bool noStructure = false;
    std::string singleArg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mode=no-structure") {
            noStructure = true;
            PHASE4_ROOT = "tests/parity/reference/data/phase4_no_structure";
        } else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "Unknown flag: %s\n", a.c_str());
            std::fprintf(stderr,
                "Usage: test_region_optimizer [--mode=no-structure] "
                "[<track_dir>/region/r<R>]\n");
            return 2;
        } else {
            singleArg = a;
        }
    }

    if (!singleArg.empty()) {
        std::string region_path = singleArg;
        std::string track_dir, ratio_label;
        if (!splitRegionPath(region_path, track_dir, ratio_label)) {
            std::fprintf(stderr,
                "Usage: test_region_optimizer [--mode=no-structure] "
                "[<track_dir>/region/r<R>]\n");
            return 2;
        }
        const std::size_t tk = track_dir.find_last_of('/');
        const std::string track = (tk == std::string::npos) ? track_dir
                                                            : track_dir.substr(tk + 1);
        try {
            Case c = loadCase(track_dir, ratio_label, track);
            CaseSummary s = runCase(c);
            return s.pass ? 0 : 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "LOAD-FAIL: %s\n", e.what());
            return 3;
        }
    }

    std::printf("=== test_region_optimizer (%s) ===\n",
                noStructure ? "ADR-044 no-structure"
                            : "phase-4 status-quo");
    std::printf("base dir: %s\n", PHASE4_ROOT.c_str());

    // Corpus mode.
    int total = 0, pass_n = 0, fail_n = 0, miss_n = 0;
    std::vector<CaseSummary> summaries;

    for (const auto& track : kCorpusTracks) {
        for (double r : kCorpusRatios) {
            const std::string track_dir   = PHASE4_ROOT + "/" + track;
            const std::string ratio_label = ratioLabel(r);
            ++total;
            try {
                Case c = loadCase(track_dir, ratio_label, track);
                CaseSummary s = runCase(c);
                summaries.push_back(s);
                if (s.pass) ++pass_n; else ++fail_n;
            } catch (const std::exception& e) {
                CaseSummary s;
                s.track       = track;
                s.ratio_label = ratio_label;
                s.err         = e.what();
                summaries.push_back(s);
                ++miss_n;
                std::printf("\n=== %s [%s] === LOAD-FAIL: %s\n",
                            track.c_str(), ratio_label.c_str(), e.what());
            }
        }
    }

    std::printf("\n================== CORPUS SUMMARY ==================\n");
    std::printf("PASS: %d / FAIL: %d / MISS: %d / TOTAL: %d\n",
                pass_n, fail_n, miss_n, total);
    for (const auto& s : summaries) {
        const char* tag = s.loaded ? (s.pass ? "PASS" : "FAIL") : "MISS";
        if (s.loaded) {
            std::printf(" [%s] %-32s %s path=%d/%d trans=%d/%d "
                        "cands=%d/%d W_max=%.2e cost_diff=%.2e\n",
                        tag, s.track.c_str(), s.ratio_label.c_str(),
                        s.path_len_cpp, s.path_len_py,
                        s.n_trans_cpp, s.n_trans_py,
                        s.n_cands_cpp, s.n_cands_py,
                        s.region_W_max_diff,
                        s.total_cost_diff);
        } else {
            std::printf(" [%s] %-32s %s %s\n", tag, s.track.c_str(),
                        s.ratio_label.c_str(), s.err.c_str());
        }
    }

    if (fail_n == 0 && pass_n > 0) return 0;
    if (pass_n == 0 && miss_n == total) {
        std::printf("(no cases loaded; treat as infrastructure-only check)\n");
        return 0;
    }
    return fail_n > 0 ? 1 : 0;
}
