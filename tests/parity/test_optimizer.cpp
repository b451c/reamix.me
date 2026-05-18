// test_optimizer — phase-4 session 22 parity:
// reamix::remix::CleanOptimizer vs Python references/python-source/remix/
// optimizer.py::CleanOptimizer.
//
// Session-22 smoke: billie_jean × 0.5. Corpus mode (16 × 3 = 48 cases)
// supported via argc-1 entry — stretch goal after smoke PASSes.
//
// FIRST session to exercise the label-aware intro/outro + adaptive scaling
// + post-DP outro extension chain (optimizer.py L144-146 + L218-227 +
// L289-301). These branches were NOT covered at sessions 20/21 ViterbiDP
// parity gates because dump_viterbi_smoke uses a simplified compute_dp_
// params helper with default intro=4/outro=2 (ignoring segment labels).
// Session 22 is the first validation that C++ facade output matches Python
// `CleanOptimizer.remix()` end-to-end.
//
// Per-gate breakdown:
//   Stage 1: remix_path — int64 bitwise. STRONGER than phase-4 spec L37
//                         "≥ 90% exact beat-index match"; targeting 100%
//                         per Principle 6 root-cause-over-widening.
//   Stage 2: remix_total_cost — f64 ULP ≤ 5e-4 (same class as dp_total_cost
//                         session 20/21, composes through DP penalties +
//                         -ffp-contract=off per ADR-028 4th reuse).
//   Stage 3: transition pair set — set-equal. Python and C++ both sort
//                         by (from, to) at dump/compare time so the flat
//                         meta_pairs int64 vector has deterministic order.
//   Stage 4: per-transition metadata — for each golden (from, to) pair, for
//                         each of 10 possible keys (8 candidate-sourced + 2
//                         repetition-sourced): Python NaN ⇔ C++ key absent;
//                         else |diff| under per-key tolerance (matches the
//                         session-19 test_transition_cost sub-gate class
//                         per signal — f32 ULP 5e-6 for signals, f64 ULP
//                         1e-4 for composites).
//
// Invocation:
//   test_optimizer                   → corpus mode (16 × 3 = 48 cases;
//                                      LOAD-FAIL on subdirs without
//                                      optimizer dumps — safe for smoke).
//   test_optimizer <opt_dir>         → single-case bisection; arg is the
//                                      full ratio-specific subdir
//                                      e.g. `.../billie_jean/optimizer/r0.5`.

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
#include <string>
#include <utility>
#include <vector>

#include "NpyIO.h"

// Session 26 (ADR-027 integration parity): this same translation unit is
// compiled twice — once as `test_optimizer` against the session-22 librosa-
// grid goldens under `tests/parity/reference/data/phase4/` and once as
// `test_optimizer_e2e` against the beat-this-grid goldens under
// `tests/parity/reference/data/phase4_e2e/`. The difference is two compile-
// time flags set via CMake `target_compile_definitions`.
#ifndef PHASE4_DATA_SUBDIR
#define PHASE4_DATA_SUBDIR "phase4"
#endif
#ifndef PHASE4_E2E_MODE
#define PHASE4_E2E_MODE 0
#endif

namespace {

// Corpus: session-22 uses all 16 tracks × {0.3, 0.5, 0.7}; session-26 E2E
// smoke narrowed to billie_jean until session-27 regenerated the full 48
// cases on the C++-BeatDetector grid. Session-27 expanded E2E to full 16.
// Lists duplicated intentionally: future asymmetric trims (e.g. cutting
// 1 track from E2E for ctest speed) should not silently affect librosa grid.
#if PHASE4_E2E_MODE
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
#else
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
#endif

const std::vector<double> kCorpusRatios = { 0.3, 0.5, 0.7 };

std::string ratioLabel(double r)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "r%.1f", r);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Tolerances (per-signal budget)
// ---------------------------------------------------------------------------

struct Tolerances
{
    // total_cost: composite f64 DP accumulator. Same class as session-21
    // dp_total_cost (worst corpus 1.477e-08 on woodkid r0.3). Gate 5e-4
    // matches phase-4 spec L38 Quality-score L∞ budget; total_cost is a
    // proxy for the same gate class (`total_cost = 1 - quality`).
    double total_cost_max_abs = 5e-4;

    // Per-key metadata budgets — inherit from session-19 test_transition_
    // cost sub-gates since the values are a subset of TransitionCandidate
    // fields. Cite session-19 baselines:
    //   quality_score      corpus max 1.128e-07 → gate 5e-4 (phase-4 L38).
    //   waveform_similarity      ≤ 2.554e-15   → gate 1e-9 (xcorr-bitwise).
    //   successor_similarity     ≤ 4.768e-07   → gate 5e-6 (f32 matmul ULP).
    //   edge_splice_similarity   ≤ 4.768e-07   → gate 5e-6 (f32 matmul ULP).
    //   chroma_distance          ≤ 3.576e-07   → gate 5e-6 (f32 matmul ULP).
    //   energy_diff_db           bitwise 0     → gate 1e-12 (pure f64 log10).
    //   alignment_offset_sec     bitwise 0     → gate 1e-12 (int / int).
    //   total_cost (per-trans)   ≤ 1.128e-07   → gate 5e-4 (phase-4 L38).
    //   is_repetition_jump       bitwise 1.0   → gate 1e-12 (marker float).
    //   chroma_correlation       bitwise 0     → gate 1e-6 (from RepetitionMap).
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

// ---------------------------------------------------------------------------
// Small helpers (mirror test_viterbi_dp.cpp style)
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

// Extract `"<key>": <float>` from a JSON blob. Does not parse nested keys
// beyond simple first-occurrence. Sufficient for our flat manifest fields.
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

// ---------------------------------------------------------------------------
// Case bundle
// ---------------------------------------------------------------------------

struct Case
{
    std::string track_dir;
    std::string vit_dir;   // `<track_dir>/viterbi/r<R>` — used for rep_* reload
    std::string opt_dir;   // `<track_dir>/optimizer/r<R>` — goldens live here
    std::string track;
    std::string ratio_label;

    // Track-root inputs.
    std::vector<double>                       beat_times;
    std::vector<float>                        features;
    int                                       n_features = 0;
    std::vector<reamix::analysis::Segment>    segments;
    std::vector<double>                       downbeats;
    std::vector<double>                       W;  // n_beats × n_beats, f64 row-major
    int                                       n_beats = 0;

    // Candidates map built from cand_* flat arrays.
    std::map<std::pair<int, int>, reamix::remix::TransitionCandidate> candidates;

    // Repetition jumps.
    std::vector<reamix::analysis::RepetitionJump> rep_jumps;

    // Goldens.
    std::vector<std::int64_t>                 remix_path_expected;
    double                                    remix_total_cost_expected = 0.0;
    std::vector<std::pair<int, int>>          meta_pairs_expected;

    // Per-key golden arrays (aligned to meta_pairs_expected order; NaN =
    // absent in Python dict).
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

    // From manifest.
    double target_duration = 0.0;
};

Case loadCase(const std::string& track_dir,
              const std::string& ratio_label,
              const std::string& track)
{
    Case c;
    c.track_dir   = track_dir;
    c.vit_dir     = track_dir + "/viterbi/"   + ratio_label;
    c.opt_dir     = track_dir + "/optimizer/" + ratio_label;
    c.track       = track;
    c.ratio_label = ratio_label;

    // Track-root inputs (shared across ratios).
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

    // -- Candidates map from cand_* flat arrays (session-18/19 dumps) -----
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

    // -- Repetition jumps from viterbi-subdir rep_* arrays -----------------
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

    // -- Optimizer goldens -------------------------------------------------
    c.remix_path_expected = reamix::test::loadNpy1DInt64(c.opt_dir + "/remix_path.npy");
    {
        auto tc_vec = reamix::test::loadNpy1DFloat64(c.opt_dir + "/remix_total_cost.npy");
        if (tc_vec.size() != 1) {
            throw std::runtime_error("remix_total_cost.npy expected size-1 vector");
        }
        c.remix_total_cost_expected = tc_vec[0];
    }
    {
        auto flat = reamix::test::loadNpy1DInt64(c.opt_dir + "/meta_pairs.npy");
        if (flat.size() % 2 != 0) {
            throw std::runtime_error("meta_pairs.npy size must be even");
        }
        c.meta_pairs_expected.reserve(flat.size() / 2);
        for (std::size_t k = 0; k + 1 < flat.size(); k += 2) {
            c.meta_pairs_expected.emplace_back(static_cast<int>(flat[k]),
                                               static_cast<int>(flat[k + 1]));
        }
    }
    c.meta_quality_score          = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_quality_score.npy");
    c.meta_waveform_similarity    = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_waveform_similarity.npy");
    c.meta_successor_similarity   = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_successor_similarity.npy");
    c.meta_edge_splice_similarity = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_edge_splice_similarity.npy");
    c.meta_chroma_distance        = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_chroma_distance.npy");
    c.meta_energy_diff_db         = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_energy_diff_db.npy");
    c.meta_alignment_offset_sec   = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_alignment_offset_sec.npy");
    c.meta_total_cost             = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_total_cost.npy");
    c.meta_is_repetition_jump     = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_is_repetition_jump.npy");
    c.meta_chroma_correlation     = reamix::test::loadNpy1DFloat64(c.opt_dir + "/meta_chroma_correlation.npy");

    // Manifest — target_duration.
    const std::string manifest_json = readAll(c.opt_dir + "/manifest.json");
    if (!extractDouble(manifest_json, "target_duration", c.target_duration)) {
        throw std::runtime_error("Missing target_duration in manifest");
    }

    return c;
}

// ---------------------------------------------------------------------------
// Per-case comparison
// ---------------------------------------------------------------------------

struct CaseSummary
{
    std::string track;
    std::string ratio_label;
    bool        loaded           = false;
    bool        pass             = false;
    bool        path_bitwise     = false;
    bool        pairs_equal      = false;
    double      total_cost_diff  = 0.0;
    // Per-signal worst-case absolute diff over transitions (NaN→skip).
    double worst_quality        = 0.0;
    double worst_waveform       = 0.0;
    double worst_successor      = 0.0;
    double worst_edge_splice    = 0.0;
    double worst_chroma_dist    = 0.0;
    double worst_energy_db      = 0.0;
    double worst_align_offset   = 0.0;
    double worst_total_cost     = 0.0;
    double worst_is_rep_jump    = 0.0;
    double worst_chroma_corr    = 0.0;
    int    n_trans_cpp          = 0;
    int    n_trans_py           = 0;
    int    path_len_cpp         = 0;
    int    path_len_py          = 0;
    std::string err;
};

// Compare C++ remix_path (vector<int>) vs golden (vector<int64>) bitwise
// promoted to int. Returns (pass, index_of_first_mismatch_or_-1).
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

// Per-key comparison. Python NaN ⇔ C++ key absent.
//   - If golden is NaN and C++ dict has the key → FAIL.
//   - If golden is non-NaN and C++ dict lacks the key → FAIL.
//   - If both present: |diff| ≤ tol.
//   - Both absent (NaN → absent) → pass (no contribution).
// Returns (pass, worst_abs_diff_this_key_across_all_transitions).
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

CaseSummary runCase(const Case& c)
{
    CaseSummary out;
    out.track       = c.track;
    out.ratio_label = c.ratio_label;
    out.loaded      = true;
    out.n_trans_py  = static_cast<int>(c.meta_pairs_expected.size());
    out.path_len_py = static_cast<int>(c.remix_path_expected.size());

    std::printf("\n=== %s [%s] ===\n", c.track.c_str(), c.ratio_label.c_str());
    std::printf("  n_beats=%d target_duration=%.4fs path_len_py=%d n_trans_py=%d\n",
                c.n_beats, c.target_duration, out.path_len_py, out.n_trans_py);

    // -- Build CleanOptimizerInputs + invoke optimizer.remix() ------------
    // NOTE on W ownership: `remix()` may mutate W via blocked_transitions
    // (we pass nullptr here, so no mutation). The W buffer in `c.W` is
    // owned by the Case — the optimizer holds a raw pointer.
    std::vector<double> W_copy = c.W;  // decouple from Case.W for safety

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

    // Defaults for session-22 smoke match Python ctor defaults (production
    // UI path applies overrides via server/_remix_options.py).

    reamix::remix::CleanOptimizer optimizer(in);

    std::printf("  effective_intro=%d effective_outro=%d avg_beat_duration=%.6f\n",
                optimizer.effectiveIntro(),
                optimizer.effectiveOutro(),
                optimizer.avgBeatDuration());

    reamix::remix::RemixPath rp = optimizer.remix(c.target_duration);
    out.n_trans_cpp  = static_cast<int>(rp.transitions.size());
    out.path_len_cpp = static_cast<int>(rp.beat_indices.size());

    std::printf("  C++ path_len=%d n_trans=%d total_cost=%.6f\n",
                out.path_len_cpp, out.n_trans_cpp, rp.total_cost);

    // -- Stage 1: remix_path bitwise --------------------------------------
    auto [path_pass, path_mis_idx] = comparePathInt(rp.beat_indices, c.remix_path_expected);
    out.path_bitwise = path_pass;
    if (!path_pass) {
        if (static_cast<std::size_t>(path_mis_idx) < rp.beat_indices.size()
            && static_cast<std::size_t>(path_mis_idx) < c.remix_path_expected.size())
        {
            std::printf("  [FAIL] remix_path mismatch at idx=%d: cpp=%d vs py=%lld (sizes %d vs %d)\n",
                        path_mis_idx,
                        rp.beat_indices[path_mis_idx],
                        static_cast<long long>(c.remix_path_expected[path_mis_idx]),
                        out.path_len_cpp, out.path_len_py);
        } else {
            std::printf("  [FAIL] remix_path size mismatch: cpp=%d vs py=%d\n",
                        out.path_len_cpp, out.path_len_py);
        }
    } else {
        std::printf("  [PASS] remix_path bitwise exact (%d beats)\n", out.path_len_cpp);
    }

    // -- Stage 2: total_cost ----------------------------------------------
    out.total_cost_diff = std::abs(rp.total_cost - c.remix_total_cost_expected);
    const bool tc_pass  = out.total_cost_diff <= TOL.total_cost_max_abs;
    std::printf("  [%s] remix_total_cost: cpp=%.6f py=%.6f |diff|=%.6e (tol %.0e)\n",
                tc_pass ? "PASS" : "FAIL",
                rp.total_cost, c.remix_total_cost_expected,
                out.total_cost_diff, TOL.total_cost_max_abs);

    // -- Stage 3: transition pair set equality ----------------------------
    // Python meta_pairs is sorted-by-(from,to). C++ iterates transition_
    // metadata which is a std::map<pair, ...> — already lex-sorted. So we
    // extract C++ sorted keys + compare.
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
        for (std::size_t k = 0; k < std::min(cpp_pairs_sorted.size(), c.meta_pairs_expected.size()); ++k) {
            if (cpp_pairs_sorted[k] != c.meta_pairs_expected[k]) {
                std::printf("    idx=%zu cpp=(%d,%d) py=(%d,%d)\n", k,
                            cpp_pairs_sorted[k].first, cpp_pairs_sorted[k].second,
                            c.meta_pairs_expected[k].first, c.meta_pairs_expected[k].second);
                break;
            }
        }
    } else {
        std::printf("  [PASS] transition pair set equal (%zu pairs)\n",
                    c.meta_pairs_expected.size());
    }

    // -- Stage 4: per-key metadata ----------------------------------------
    // Only meaningful when pairs are equal (otherwise the per-key vector
    // alignment is broken).
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
            {"alignment_offset_sec",   &c.meta_alignment_offset_sec,   TOL.meta_alignment_offset_sec_tol, &out.worst_align_offset},
            {"total_cost",             &c.meta_total_cost,             TOL.meta_total_cost_tol,           &out.worst_total_cost},
            {"is_repetition_jump",     &c.meta_is_repetition_jump,     TOL.meta_is_repetition_jump_tol,   &out.worst_is_rep_jump},
            {"chroma_correlation",     &c.meta_chroma_correlation,     TOL.meta_chroma_correlation_tol,   &out.worst_chroma_corr},
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

    out.pass = path_pass && tc_pass && pairs_pass && meta_pass;
    std::printf("  *** CASE %s ***\n", out.pass ? "PASS" : "FAIL");
    return out;
}

// Split an optimizer-dir path `.../<track_dir>/optimizer/r<R>` → (track_dir, ratio_label).
bool splitOptPath(const std::string& opt_path,
                  std::string&       out_track_dir,
                  std::string&       out_ratio_label)
{
    // Strip trailing slash.
    std::string p = opt_path;
    while (!p.empty() && p.back() == '/') p.pop_back();

    const std::size_t last = p.find_last_of('/');
    if (last == std::string::npos) return false;
    out_ratio_label = p.substr(last + 1);
    if (out_ratio_label.empty() || out_ratio_label[0] != 'r') return false;

    std::string middle = p.substr(0, last);
    const std::size_t mid = middle.find_last_of('/');
    if (mid == std::string::npos) return false;
    if (middle.substr(mid + 1) != "optimizer") return false;

    out_track_dir = middle.substr(0, mid);
    return true;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    const std::string PHASE4_ROOT =
        "tests/parity/reference/data/" PHASE4_DATA_SUBDIR;

    if (argc > 1) {
        // Single-case bisection mode.
        std::string opt_path = argv[1];
        std::string track_dir, ratio_label;
        if (!splitOptPath(opt_path, track_dir, ratio_label)) {
            std::fprintf(stderr,
                "Usage: test_optimizer [<track_dir>/optimizer/r<R>]\n");
            return 2;
        }
        // Extract track name from track_dir basename.
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
            std::printf(" [%s] %-32s %s path=%d/%d trans=%d/%d cost_diff=%.2e\n",
                        tag, s.track.c_str(), s.ratio_label.c_str(),
                        s.path_len_cpp, s.path_len_py,
                        s.n_trans_cpp, s.n_trans_py,
                        s.total_cost_diff);
        } else {
            std::printf(" [%s] %-32s %s %s\n", tag, s.track.c_str(),
                        s.ratio_label.c_str(), s.err.c_str());
        }
    }

    if (fail_n == 0 && pass_n > 0) return 0;
    if (pass_n == 0 && miss_n == total) {
        // No goldens at all — treat as missing, not failure (smoke-only
        // mode where only 1 of 48 subdirs is populated would hit this).
        std::printf("(no cases loaded; treat as infrastructure-only check)\n");
        return 0;
    }
    return fail_n > 0 ? 1 : 0;
}
