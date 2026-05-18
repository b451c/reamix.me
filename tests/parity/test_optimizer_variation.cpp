// test_optimizer_variation — phase-6 session 58 / ADR-048 / DEV-027 parity:
// reamix::remix::CleanOptimizer::remix_k_best + remix_variation vs Python
// references/python-source/remix/optimizer.py::CleanOptimizer.remix_k_best.
//
// Corpus: 3 tracks × 2 ratios = 6 cases × variation 0..3 = 24 variation
// lookups. Tracks chosen mid-session for path-coverage diversity:
//   billie_jean, woodkid_iron_acoustic — n_trans=1 → 2 unique paths;
//                exercises clamp v∈{2,3} → paths.back() = paths[1].
//   daft_punk_aerodynamic              — n_trans={2,3} → 3 unique paths;
//                exercises full v=0,1,2 unique + clamp at v=3.
//
// Per-gate breakdown (mirrors test_optimizer.cpp class):
//   Stage 1: paths.size() — equals manifest n_paths_returned.
//   Stage 2: per-path remix_path — int64 bitwise.
//   Stage 3: per-path total_cost — f64 |diff| ≤ 5e-4 (same class as session
//                                  -22 test_optimizer Stage 2).
//   Stage 4: per-path transition pair set equality.
//   Stage 5: per-path per-key metadata — same per-signal tolerances as
//                                        test_optimizer.cpp.
//   Stage 6: remix_variation(target, v) for v=0..3 returns paths[min(v,
//                                        len-1)] bitwise. This validates
//                                        the convenience wrapper independently
//                                        of the underlying k-best machinery.
//
// Full 16-track × 4-variation = 64-case gate deferred to phase-6 close
// (sesja 65) per ADR-034.
//
// Invocation:
//   test_optimizer_variation                 → corpus mode (6 cases).
//   test_optimizer_variation <variations_dir> → single-case bisection;
//      arg is `.../<track>/optimizer/r<R>/variations` directory.

#include "remix/Optimizer.h"
#include "remix/TransitionCost.h"
#include "analysis/RepetitionMap.h"
#include "analysis/StructureResult.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "NpyIO.h"

namespace {

const std::vector<std::string> kCorpusTracks = {
    "billie_jean",
    "woodkid_iron_acoustic",
    "daft_punk_aerodynamic",
};
const std::vector<double> kCorpusRatios = { 0.3, 0.5 };

std::string ratioLabel(double r)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "r%.1f", r);
    return std::string(buf);
}

// -- Tolerances — same class as test_optimizer.cpp ---------------------------
struct Tolerances
{
    double total_cost_max_abs = 5e-4;
    double meta_quality_score_tol        = 5e-4;
    double meta_waveform_similarity_tol  = 1e-9;
    double meta_successor_similarity_tol = 5e-6;
    double meta_edge_splice_similarity_tol = 5e-6;
    double meta_chroma_distance_tol      = 5e-6;
    double meta_energy_diff_db_tol       = 1e-12;
    double meta_alignment_offset_sec_tol = 1e-12;
    double meta_total_cost_tol           = 5e-4;
    double meta_is_repetition_jump_tol   = 1e-12;
    double meta_chroma_correlation_tol   = 1e-6;
};
const Tolerances TOL;

// -- Small helpers (mirror test_optimizer.cpp) -------------------------------
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

bool extractInt(const std::string& json, const std::string& key, int& out)
{
    double v = 0.0;
    if (!extractDouble(json, key, v)) return false;
    out = static_cast<int>(v);
    return true;
}

// -- Per-path expectation bundle (one per `variations/v<N>/`) ----------------
struct PathGolden
{
    std::vector<std::int64_t>          remix_path;
    double                             remix_total_cost = 0.0;
    std::vector<std::pair<int, int>>   meta_pairs;
    std::vector<double> meta_quality_score;
    std::vector<double> meta_waveform_similarity;
    std::vector<double> meta_successor_similarity;
    std::vector<double> meta_edge_splice_similarity;
    std::vector<double> meta_chroma_distance;
    std::vector<double> meta_energy_diff_db;
    std::vector<double> meta_alignment_offset_sec;
    std::vector<double> meta_total_cost;
    std::vector<double> meta_is_repetition_jump;
    std::vector<double> meta_chroma_correlation;
};

PathGolden loadPathGolden(const std::string& path_dir)
{
    PathGolden p;
    p.remix_path = reamix::test::loadNpy1DInt64(path_dir + "/remix_path.npy");
    {
        auto tc_vec = reamix::test::loadNpy1DFloat64(path_dir + "/remix_total_cost.npy");
        if (tc_vec.size() != 1) {
            throw std::runtime_error("remix_total_cost.npy expected size-1 vector in " + path_dir);
        }
        p.remix_total_cost = tc_vec[0];
    }
    {
        auto flat = reamix::test::loadNpy1DInt64(path_dir + "/meta_pairs.npy");
        if (flat.size() % 2 != 0) {
            throw std::runtime_error("meta_pairs.npy size must be even in " + path_dir);
        }
        p.meta_pairs.reserve(flat.size() / 2);
        for (std::size_t k = 0; k + 1 < flat.size(); k += 2) {
            p.meta_pairs.emplace_back(static_cast<int>(flat[k]),
                                      static_cast<int>(flat[k + 1]));
        }
    }
    p.meta_quality_score          = reamix::test::loadNpy1DFloat64(path_dir + "/meta_quality_score.npy");
    p.meta_waveform_similarity    = reamix::test::loadNpy1DFloat64(path_dir + "/meta_waveform_similarity.npy");
    p.meta_successor_similarity   = reamix::test::loadNpy1DFloat64(path_dir + "/meta_successor_similarity.npy");
    p.meta_edge_splice_similarity = reamix::test::loadNpy1DFloat64(path_dir + "/meta_edge_splice_similarity.npy");
    p.meta_chroma_distance        = reamix::test::loadNpy1DFloat64(path_dir + "/meta_chroma_distance.npy");
    p.meta_energy_diff_db         = reamix::test::loadNpy1DFloat64(path_dir + "/meta_energy_diff_db.npy");
    p.meta_alignment_offset_sec   = reamix::test::loadNpy1DFloat64(path_dir + "/meta_alignment_offset_sec.npy");
    p.meta_total_cost             = reamix::test::loadNpy1DFloat64(path_dir + "/meta_total_cost.npy");
    p.meta_is_repetition_jump     = reamix::test::loadNpy1DFloat64(path_dir + "/meta_is_repetition_jump.npy");
    p.meta_chroma_correlation     = reamix::test::loadNpy1DFloat64(path_dir + "/meta_chroma_correlation.npy");
    return p;
}

// -- Variation case: track-root inputs + N PathGoldens -----------------------
struct VariationCase
{
    std::string track;
    std::string ratio_label;
    std::string track_dir;
    std::string vit_dir;
    std::string variations_dir;

    std::vector<double>                                                beat_times;
    std::vector<float>                                                 features;
    int                                                                n_features = 0;
    std::vector<reamix::analysis::Segment>                             segments;
    std::vector<double>                                                downbeats;
    std::vector<double>                                                W;
    int                                                                n_beats = 0;
    std::map<std::pair<int, int>, reamix::remix::TransitionCandidate>  candidates;
    std::vector<reamix::analysis::RepetitionJump>                      rep_jumps;

    double target_duration = 0.0;
    int    k_requested     = 0;
    int    n_paths         = 0;

    std::vector<PathGolden> paths;
};

VariationCase loadVariationCase(const std::string& track_dir,
                                const std::string& ratio_label,
                                const std::string& track)
{
    VariationCase c;
    c.track          = track;
    c.ratio_label    = ratio_label;
    c.track_dir      = track_dir;
    c.vit_dir        = track_dir + "/viterbi/"   + ratio_label;
    c.variations_dir = track_dir + "/optimizer/" + ratio_label + "/variations";

    // Track-root inputs (shared with test_optimizer.cpp loader pattern).
    c.beat_times = reamix::test::loadNpy1DFloat64(track_dir + "/beat_times.npy");
    c.n_beats    = static_cast<int>(c.beat_times.size());

    {
        auto feat = reamix::test::loadNpy2DFloat32(track_dir + "/features.npy");
        c.features   = std::move(feat.data);
        c.n_features = static_cast<int>(feat.cols);
    }

    {
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

    {
        auto Wmat = reamix::test::loadNpy2DFloat64(track_dir + "/W.npy");
        c.W = std::move(Wmat.data);
        if (static_cast<int>(Wmat.rows) != c.n_beats
            || static_cast<int>(Wmat.cols) != c.n_beats)
        {
            throw std::runtime_error("W shape mismatch: expected n_beats² = "
                + std::to_string(c.n_beats) + "², got "
                + std::to_string(Wmat.rows) + "×" + std::to_string(Wmat.cols));
        }
    }

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
        if (ct.size() != n_cand || cq.size() != n_cand || cw.size() != n_cand
            || cs.size() != n_cand || ce.size() != n_cand || cd.size() != n_cand
            || cen.size() != n_cand || cal.size() != n_cand || ctc.size() != n_cand)
        {
            throw std::runtime_error("cand_* arrays size mismatch in " + track_dir);
        }
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
            c.candidates[{tc.from_beat, tc.to_beat}] = tc;
        }
    }

    {
        auto rfb  = reamix::test::loadNpy1DInt64(  c.vit_dir + "/rep_from_beat.npy");
        auto rtb  = reamix::test::loadNpy1DInt64(  c.vit_dir + "/rep_to_beat.npy");
        auto rws  = reamix::test::loadNpy1DFloat64(c.vit_dir + "/rep_waveform_similarity.npy");
        auto rcc  = reamix::test::loadNpy1DFloat64(c.vit_dir + "/rep_chroma_correlation.npy");
        auto ral  = reamix::test::loadNpy1DInt64(  c.vit_dir + "/rep_alignment_lag_samples.npy");
        auto rfs  = reamix::test::loadNpy1DInt64(  c.vit_dir + "/rep_from_section_idx.npy");
        auto rts  = reamix::test::loadNpy1DInt64(  c.vit_dir + "/rep_to_section_idx.npy");
        auto rfbar = reamix::test::loadNpy1DInt64( c.vit_dir + "/rep_from_bar.npy");
        auto rtbar = reamix::test::loadNpy1DInt64( c.vit_dir + "/rep_to_bar.npy");
        const std::size_t n_rep = rfb.size();
        c.rep_jumps.reserve(n_rep);
        for (std::size_t k = 0; k < n_rep; ++k) {
            reamix::analysis::RepetitionJump j;
            j.fromBeat            = static_cast<int>(rfb[k]);
            j.toBeat              = static_cast<int>(rtb[k]);
            j.waveformSimilarity  = rws[k];
            j.chromaCorrelation   = rcc[k];
            j.alignmentLagSamples = static_cast<int>(ral[k]);
            j.fromSectionIdx      = static_cast<int>(rfs[k]);
            j.toSectionIdx        = static_cast<int>(rts[k]);
            j.fromBar             = static_cast<int>(rfbar[k]);
            j.toBar               = static_cast<int>(rtbar[k]);
            c.rep_jumps.push_back(j);
        }
    }

    // Variations manifest.
    const std::string manifest_json = readAll(c.variations_dir + "/manifest.json");
    if (!extractDouble(manifest_json, "target_duration", c.target_duration)) {
        throw std::runtime_error("Missing target_duration in variations/manifest.json");
    }
    if (!extractInt(manifest_json, "k_requested", c.k_requested)) {
        throw std::runtime_error("Missing k_requested in variations/manifest.json");
    }
    if (!extractInt(manifest_json, "n_paths_returned", c.n_paths)) {
        throw std::runtime_error("Missing n_paths_returned in variations/manifest.json");
    }

    // Load each path golden v0..vN-1.
    c.paths.reserve(static_cast<std::size_t>(c.n_paths));
    for (int v = 0; v < c.n_paths; ++v) {
        char vbuf[8];
        std::snprintf(vbuf, sizeof(vbuf), "v%d", v);
        const std::string v_dir = c.variations_dir + "/" + vbuf;
        c.paths.push_back(loadPathGolden(v_dir));
    }

    return c;
}

// -- Comparators (mirror test_optimizer.cpp) ---------------------------------
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
compareMetaKey(const std::string& key_name,
               const std::vector<std::pair<int, int>>& py_pairs,
               const std::vector<double>&              py_values,
               const std::map<std::pair<int, int>, std::map<std::string, double>>&
                                                     cpp_meta,
               double                                  tol,
               std::string&                            err_out)
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
            err_out = buf;
            pass = false;
        } else if (!py_nan && !cpp_has_key) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "[%s] (%d, %d) py=%g but cpp=absent",
                key_name.c_str(), pr.first, pr.second, py_values[k]);
            err_out = buf;
            pass = false;
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
                err_out = buf;
                pass = false;
            }
        }
    }
    return {pass, worst};
}

struct PathSummary
{
    bool   path_bitwise   = false;
    bool   pairs_equal    = false;
    bool   meta_pass      = false;
    double total_cost_diff = 0.0;
    bool   total_cost_pass = false;
};

PathSummary comparePath(int                                     v_idx,
                        const reamix::remix::RemixPath&         cpp,
                        const PathGolden&                       py)
{
    PathSummary s;
    std::printf("    -- v%d --\n", v_idx);

    auto [path_pass, path_mis_idx] = comparePathInt(cpp.beat_indices, py.remix_path);
    s.path_bitwise = path_pass;
    if (!path_pass) {
        if (static_cast<std::size_t>(path_mis_idx) < cpp.beat_indices.size()
            && static_cast<std::size_t>(path_mis_idx) < py.remix_path.size())
        {
            std::printf("      [FAIL] remix_path mismatch idx=%d cpp=%d py=%lld (sizes %zu vs %zu)\n",
                        path_mis_idx,
                        cpp.beat_indices[path_mis_idx],
                        static_cast<long long>(py.remix_path[path_mis_idx]),
                        cpp.beat_indices.size(), py.remix_path.size());
        } else {
            std::printf("      [FAIL] remix_path size mismatch: cpp=%zu vs py=%zu\n",
                        cpp.beat_indices.size(), py.remix_path.size());
        }
    } else {
        std::printf("      [PASS] remix_path bitwise (%zu beats)\n",
                    cpp.beat_indices.size());
    }

    s.total_cost_diff = std::abs(cpp.total_cost - py.remix_total_cost);
    s.total_cost_pass = s.total_cost_diff <= TOL.total_cost_max_abs;
    std::printf("      [%s] total_cost: cpp=%.6f py=%.6f |diff|=%.3e\n",
                s.total_cost_pass ? "PASS" : "FAIL",
                cpp.total_cost, py.remix_total_cost, s.total_cost_diff);

    std::vector<std::pair<int, int>> cpp_pairs_sorted;
    cpp_pairs_sorted.reserve(cpp.transition_metadata.size());
    for (const auto& [pr, _] : cpp.transition_metadata) {
        cpp_pairs_sorted.push_back(pr);
    }
    s.pairs_equal = (cpp_pairs_sorted == py.meta_pairs);
    if (!s.pairs_equal) {
        std::printf("      [FAIL] pair set mismatch: cpp_n=%zu py_n=%zu\n",
                    cpp_pairs_sorted.size(), py.meta_pairs.size());
    } else {
        std::printf("      [PASS] pair set equal (%zu pairs)\n",
                    py.meta_pairs.size());
    }

    s.meta_pass = true;
    if (s.pairs_equal) {
        struct KeyProbe {
            const char*                key;
            const std::vector<double>* values;
            double                     tol;
        };
        const KeyProbe probes[] = {
            {"quality_score",          &py.meta_quality_score,          TOL.meta_quality_score_tol},
            {"waveform_similarity",    &py.meta_waveform_similarity,    TOL.meta_waveform_similarity_tol},
            {"successor_similarity",   &py.meta_successor_similarity,   TOL.meta_successor_similarity_tol},
            {"edge_splice_similarity", &py.meta_edge_splice_similarity, TOL.meta_edge_splice_similarity_tol},
            {"chroma_distance",        &py.meta_chroma_distance,        TOL.meta_chroma_distance_tol},
            {"energy_diff_db",         &py.meta_energy_diff_db,         TOL.meta_energy_diff_db_tol},
            {"alignment_offset_sec",   &py.meta_alignment_offset_sec,   TOL.meta_alignment_offset_sec_tol},
            {"total_cost",             &py.meta_total_cost,             TOL.meta_total_cost_tol},
            {"is_repetition_jump",     &py.meta_is_repetition_jump,     TOL.meta_is_repetition_jump_tol},
            {"chroma_correlation",     &py.meta_chroma_correlation,     TOL.meta_chroma_correlation_tol},
        };
        double worst_overall = 0.0;
        for (const auto& p : probes) {
            std::string err;
            auto [ok, worst] = compareMetaKey(p.key, py.meta_pairs, *p.values,
                                              cpp.transition_metadata, p.tol, err);
            if (worst > worst_overall) worst_overall = worst;
            if (!ok) {
                std::printf("      [FAIL] meta[%s]: worst=%.3e (tol %.0e) — %s\n",
                            p.key, worst, p.tol, err.c_str());
                s.meta_pass = false;
            }
        }
        if (s.meta_pass) {
            std::printf("      [PASS] meta (10 keys, worst=%.3e across all)\n",
                        worst_overall);
        }
    } else {
        std::printf("      [SKIP] per-key metadata (pairs mismatched)\n");
        s.meta_pass = false;
    }
    return s;
}

bool runVariationCase(const VariationCase& c)
{
    std::printf("\n=== %s [%s] variations ===\n",
                c.track.c_str(), c.ratio_label.c_str());
    std::printf("  n_beats=%d target_duration=%.4fs k_requested=%d n_paths=%d\n",
                c.n_beats, c.target_duration, c.k_requested, c.n_paths);

    // Build optimizer (mirrors test_optimizer.cpp::runCase).
    std::vector<double> W_copy = c.W;
    reamix::remix::CleanOptimizerInputs in{};
    in.W                     = W_copy.data();
    in.candidates            = &c.candidates;
    in.n_beats               = c.n_beats;
    in.beat_times            = c.beat_times.data();
    in.segments              = c.segments.empty() ? nullptr : c.segments.data();
    in.n_segments            = static_cast<int>(c.segments.size());
    in.features              = c.features.empty() ? nullptr : c.features.data();
    in.n_features            = c.n_features;
    in.downbeats             = c.downbeats.empty() ? nullptr : c.downbeats.data();
    in.n_downbeats           = static_cast<int>(c.downbeats.size());
    in.jumps                 = c.rep_jumps.empty() ? nullptr : c.rep_jumps.data();
    in.n_jumps               = static_cast<int>(c.rep_jumps.size());

    reamix::remix::CleanOptimizer optimizer(in);

    // Stage 1: remix_k_best returns expected number of paths.
    auto cpp_paths = optimizer.remix_k_best(c.target_duration, c.k_requested);
    std::printf("  remix_k_best returned %zu paths (expected %d)\n",
                cpp_paths.size(), c.n_paths);
    if (static_cast<int>(cpp_paths.size()) != c.n_paths) {
        std::printf("  [FAIL] paths.size() mismatch — Python returned %d, C++ returned %zu\n",
                    c.n_paths, cpp_paths.size());
        return false;
    }

    // Stages 2-5: per-path comparison.
    bool all_pass = true;
    for (int v = 0; v < c.n_paths; ++v) {
        PathSummary ps = comparePath(v, cpp_paths[static_cast<std::size_t>(v)],
                                     c.paths[static_cast<std::size_t>(v)]);
        if (!(ps.path_bitwise && ps.pairs_equal && ps.meta_pass && ps.total_cost_pass)) {
            all_pass = false;
        }
    }

    // Stage 6: remix_variation(target, v) for v=0..3 returns paths[min(v, n_paths-1)].
    std::printf("  -- remix_variation lookup probe --\n");
    for (int v = 0; v < 4; ++v) {
        // Need a fresh W copy because optimizer mutates it during k-best loops;
        // the previous remix_k_best call restored W on each iter via RAII, but
        // a fresh CleanOptimizer instance avoids any concern.
        std::vector<double> W_copy2 = c.W;
        reamix::remix::CleanOptimizerInputs in2 = in;
        in2.W = W_copy2.data();
        reamix::remix::CleanOptimizer optimizer2(in2);

        auto cpp_var = optimizer2.remix_variation(c.target_duration, v);
        const int expected_idx = std::min(v, c.n_paths - 1);
        const PathGolden& expected = c.paths[static_cast<std::size_t>(expected_idx)];

        const auto [match, mismatch_idx] = comparePathInt(cpp_var.beat_indices,
                                                          expected.remix_path);
        if (!match) {
            std::printf("    [FAIL] remix_variation(v=%d) → path[%d]: bitwise mismatch at idx=%d\n",
                        v, expected_idx, mismatch_idx);
            all_pass = false;
        } else {
            const double cost_diff = std::abs(cpp_var.total_cost - expected.remix_total_cost);
            if (cost_diff > TOL.total_cost_max_abs) {
                std::printf("    [FAIL] remix_variation(v=%d) → path[%d]: total_cost |diff|=%.3e > %.0e\n",
                            v, expected_idx, cost_diff, TOL.total_cost_max_abs);
                all_pass = false;
            } else {
                std::printf("    [PASS] remix_variation(v=%d) → path[%d] (clamp=%s)\n",
                            v, expected_idx, (v >= c.n_paths) ? "yes" : "no");
            }
        }
    }

    std::printf("  *** CASE %s ***\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

bool splitVariationsPath(const std::string& var_path,
                         std::string&       out_track_dir,
                         std::string&       out_ratio_label)
{
    std::string p = var_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    // Expect path to end with .../optimizer/r<R>/variations
    const std::string suffix = "/variations";
    if (p.size() < suffix.size()) return false;
    if (p.compare(p.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
    std::string parent = p.substr(0, p.size() - suffix.size());
    // parent = .../<track_dir>/optimizer/r<R>
    const std::size_t last = parent.find_last_of('/');
    if (last == std::string::npos) return false;
    out_ratio_label = parent.substr(last + 1);
    if (out_ratio_label.empty() || out_ratio_label[0] != 'r') return false;
    std::string middle = parent.substr(0, last);
    const std::size_t mid = middle.find_last_of('/');
    if (mid == std::string::npos) return false;
    if (middle.substr(mid + 1) != "optimizer") return false;
    out_track_dir = middle.substr(0, mid);
    return true;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    const std::string PHASE4_ROOT = "tests/parity/reference/data/phase4";

    if (argc > 1) {
        std::string var_path = argv[1];
        std::string track_dir, ratio_label;
        if (!splitVariationsPath(var_path, track_dir, ratio_label)) {
            std::fprintf(stderr,
                "Usage: test_optimizer_variation [<track_dir>/optimizer/r<R>/variations]\n");
            return 2;
        }
        const std::size_t tk = track_dir.find_last_of('/');
        const std::string track = (tk == std::string::npos) ? track_dir
                                                            : track_dir.substr(tk + 1);
        try {
            VariationCase c = loadVariationCase(track_dir, ratio_label, track);
            return runVariationCase(c) ? 0 : 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ERROR] %s\n", e.what());
            return 2;
        }
    }

    int n_pass = 0, n_fail = 0, n_skip = 0;
    for (const auto& track : kCorpusTracks) {
        const std::string track_dir = PHASE4_ROOT + "/" + track;
        for (const auto& r : kCorpusRatios) {
            const std::string lbl = ratioLabel(r);
            try {
                VariationCase c = loadVariationCase(track_dir, lbl, track);
                if (runVariationCase(c)) ++n_pass;
                else                     ++n_fail;
            } catch (const std::exception& e) {
                std::printf("[SKIP] %s [%s] — %s\n", track.c_str(), lbl.c_str(), e.what());
                ++n_skip;
            }
        }
    }

    const int n_total = n_pass + n_fail + n_skip;
    std::printf("\n=== SUMMARY === pass=%d fail=%d skip=%d total=%d\n",
                n_pass, n_fail, n_skip, n_total);
    return (n_fail == 0 && n_pass > 0) ? 0 : 1;
}
