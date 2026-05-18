// test_block_assembly — phase-4 session 24 parity:
// reamix::remix::{buildBlocks, computeBlockCompatibility, assembleBlocks}
// vs Python references/python-source/remix/block_assembly.py.
//
// BlockAssembly is ratio-independent (arrangement is user-driven; we
// exercise the canonical [0, 1, 2, ..., n_blocks-1] sequence). Corpus =
// 16 tracks, one canonical arrangement each.
//
// Per-case gates (4-stage):
//   Stage 1: block fields — int64 bitwise (start_beat / end_beat /
//            n_beats / cluster_id), text equal (label / display_name),
//            f64 bitwise (start_sec / end_sec / duration_sec).
//   Stage 2: compatibility matrix — quality (n*n) f64 ULP, splice_from/to
//            (n*n) int64 bitwise, top_k_quality (n*n*K) f64 ULP,
//            top_k_from/to (n*n*K) int64 bitwise.
//   Stage 3: assembled path — beat_indices int64 bitwise, total_cost f64
//            ULP.
//   Stage 4: transition metadata — 3 keys (quality_score / block_transition
//            / crossfade_ms) with NaN-means-absent semantic.
//
// Invocation:
//   test_block_assembly                 → corpus mode (16 tracks).
//   test_block_assembly <track_dir>     → single-case bisection
//       e.g. `.../billie_jean`.

#include "remix/BlockAssembly.h"
#include "analysis/StructureResult.h"

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

struct Tolerances
{
    // Stage 2: quality/top_k_quality — f64 composition of 10 weighted signals
    // via computeQualityScore + context_sim (f32→f64 widening) + xcorr (f64).
    // Session-18 baseline: 9.35e-08. Gate 5e-6 (50× margin).
    double quality_tol       = 5e-6;

    // Stage 3: total_cost — accumulator of (1 - quality) over transitions.
    // With up to ~10 transitions per arrangement, f64 drift compounds by <N×ULP.
    // Gate 5e-4 (consistent with session-22/23 region_total_cost gate).
    double total_cost_tol    = 5e-4;

    // Stage 4: metadata keys — class of session-23 Region 8-key metadata.
    double meta_quality_tol  = 5e-6;
    double meta_crossfade_tol = 1e-12;   // direct constant lookup
    double meta_block_tol    = 1e-12;    // constant marker
};

const Tolerances TOL;

std::string readAll(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

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

int extractInt(const std::string& json, const std::string& key, int fallback)
{
    const std::string pat = "\"" + key + "\":";
    std::size_t pos = json.find(pat);
    if (pos == std::string::npos) return fallback;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    return std::atoi(json.c_str() + pos);
}

struct CaseBundle
{
    std::string track;
    std::string track_dir;
    std::string ba_dir;

    // Inputs.
    std::vector<double>                    beat_times;
    std::vector<float>                     features;
    int                                    n_features = 0;
    std::vector<reamix::analysis::Segment> segments;
    std::vector<double>                    downbeats;
    int                                    n_beats = 0;

    std::vector<float>  edge_features_start;
    std::vector<float>  edge_features_end;
    int                 n_edge_features = 0;
    std::vector<double> edge_rms_start;
    std::vector<double> edge_rms_end;
    std::vector<double> rms_energy;
    std::vector<double> spectral_centroid;
    std::vector<double> vocal_activity;
    std::vector<float>  boundary_waveforms;
    int                 n_boundary_waveforms = 0;
    int                 n_samples_per_bnd    = 0;
    int                 waveform_sample_rate = 22050;

    // Block fields (golden).
    std::vector<std::int64_t> py_block_segment_idx;
    std::vector<std::int64_t> py_block_start_beat;
    std::vector<std::int64_t> py_block_end_beat;
    std::vector<std::int64_t> py_block_n_beats;
    std::vector<std::int64_t> py_block_cluster_id;
    std::vector<double>       py_block_start_sec;
    std::vector<double>       py_block_end_sec;
    std::vector<double>       py_block_duration_sec;
    std::vector<std::string>  py_block_labels;
    std::vector<std::string>  py_block_display_names;

    // Compat matrix (golden).
    std::vector<double>       py_compat_quality;
    std::vector<std::int64_t> py_compat_splice_from;
    std::vector<std::int64_t> py_compat_splice_to;
    std::vector<double>       py_compat_top_k_quality;
    std::vector<std::int64_t> py_compat_top_k_from;
    std::vector<std::int64_t> py_compat_top_k_to;

    // Assembled path (golden).
    std::vector<std::int64_t> py_block_sequence;
    std::vector<std::int64_t> py_assembled_beat_indices;
    double                    py_assembled_total_cost = 0.0;
    std::vector<std::int64_t> py_assembled_trans_pairs;
    std::vector<double>       py_meta_quality_score;
    std::vector<double>       py_meta_block_transition;
    std::vector<double>       py_meta_crossfade_ms;

    int n_blocks = 0;
    int top_k    = 0;
};

CaseBundle loadCase(const std::string& track_dir, const std::string& track)
{
    CaseBundle c;
    c.track     = track;
    c.track_dir = track_dir;
    c.ba_dir    = track_dir + "/block_assembly";

    c.beat_times = reamix::test::loadNpy1DFloat64(track_dir + "/beat_times.npy");
    c.n_beats    = static_cast<int>(c.beat_times.size());

    auto feat_npy = reamix::test::loadNpy2DFloat32(track_dir + "/features.npy");
    c.features   = std::move(feat_npy.data);
    c.n_features = static_cast<int>(feat_npy.cols);

    {
        auto seg_start      = reamix::test::loadNpy1DFloat64(track_dir + "/seg_start.npy");
        auto seg_end        = reamix::test::loadNpy1DFloat64(track_dir + "/seg_end.npy");
        auto seg_conf       = reamix::test::loadNpy1DFloat64(track_dir + "/seg_confidence.npy");
        auto seg_cluster_id = reamix::test::loadNpy1DInt64  (track_dir + "/seg_cluster_id.npy");
        auto seg_labels     = readLines(                     track_dir + "/seg_labels.txt");
        c.segments.reserve(seg_start.size());
        for (std::size_t i = 0; i < seg_start.size(); ++i) {
            reamix::analysis::Segment s;
            s.start      = seg_start[i];
            s.end        = seg_end[i];
            s.confidence = seg_conf[i];
            s.cluster_id = static_cast<int>(seg_cluster_id[i]);
            s.label      = seg_labels[i];
            c.segments.push_back(std::move(s));
        }
    }

    c.downbeats = reamix::test::loadNpy1DFloat64(track_dir + "/downbeats.npy");

    {
        auto ef_start = reamix::test::loadNpy2DFloat32(track_dir + "/edge_features_start.npy");
        auto ef_end   = reamix::test::loadNpy2DFloat32(track_dir + "/edge_features_end.npy");
        c.edge_features_start = std::move(ef_start.data);
        c.edge_features_end   = std::move(ef_end.data);
        c.n_edge_features     = static_cast<int>(ef_start.cols);
    }
    c.edge_rms_start    = reamix::test::loadNpy1DFloat64(track_dir + "/edge_rms_start.npy");
    c.edge_rms_end      = reamix::test::loadNpy1DFloat64(track_dir + "/edge_rms_end.npy");
    c.rms_energy        = reamix::test::loadNpy1DFloat64(track_dir + "/rms_energy.npy");
    c.spectral_centroid = reamix::test::loadNpy1DFloat64(track_dir + "/spectral_centroid.npy");
    c.vocal_activity    = reamix::test::loadNpy1DFloat64(track_dir + "/vocal_activity.npy");

    {
        auto bw = reamix::test::loadNpy2DFloat32(track_dir + "/boundary_waveforms.npy");
        c.boundary_waveforms   = std::move(bw.data);
        c.n_boundary_waveforms = static_cast<int>(bw.rows);
        c.n_samples_per_bnd    = static_cast<int>(bw.cols);
    }

    {
        const std::string mj = readAll(track_dir + "/manifest.json");
        c.waveform_sample_rate = extractInt(mj, "sr", 22050);
    }

    // BlockAssembly goldens.
    c.py_block_segment_idx  = reamix::test::loadNpy1DInt64  (c.ba_dir + "/block_segment_idx.npy");
    c.py_block_start_beat   = reamix::test::loadNpy1DInt64  (c.ba_dir + "/block_start_beat.npy");
    c.py_block_end_beat     = reamix::test::loadNpy1DInt64  (c.ba_dir + "/block_end_beat.npy");
    c.py_block_n_beats      = reamix::test::loadNpy1DInt64  (c.ba_dir + "/block_n_beats.npy");
    c.py_block_cluster_id   = reamix::test::loadNpy1DInt64  (c.ba_dir + "/block_cluster_id.npy");
    c.py_block_start_sec    = reamix::test::loadNpy1DFloat64(c.ba_dir + "/block_start_sec.npy");
    c.py_block_end_sec      = reamix::test::loadNpy1DFloat64(c.ba_dir + "/block_end_sec.npy");
    c.py_block_duration_sec = reamix::test::loadNpy1DFloat64(c.ba_dir + "/block_duration_sec.npy");
    c.py_block_labels       = readLines(c.ba_dir + "/block_labels.txt");
    c.py_block_display_names = readLines(c.ba_dir + "/block_display_names.txt");

    c.py_compat_quality       = reamix::test::loadNpy1DFloat64(c.ba_dir + "/compat_quality.npy");
    c.py_compat_splice_from   = reamix::test::loadNpy1DInt64  (c.ba_dir + "/compat_splice_from.npy");
    c.py_compat_splice_to     = reamix::test::loadNpy1DInt64  (c.ba_dir + "/compat_splice_to.npy");
    c.py_compat_top_k_quality = reamix::test::loadNpy1DFloat64(c.ba_dir + "/compat_top_k_quality.npy");
    c.py_compat_top_k_from    = reamix::test::loadNpy1DInt64  (c.ba_dir + "/compat_top_k_from.npy");
    c.py_compat_top_k_to      = reamix::test::loadNpy1DInt64  (c.ba_dir + "/compat_top_k_to.npy");

    c.py_block_sequence       = reamix::test::loadNpy1DInt64  (c.ba_dir + "/block_sequence.npy");
    c.py_assembled_beat_indices = reamix::test::loadNpy1DInt64(c.ba_dir + "/assembled_beat_indices.npy");
    {
        auto v = reamix::test::loadNpy1DFloat64(c.ba_dir + "/assembled_total_cost.npy");
        if (v.size() != 1) throw std::runtime_error("assembled_total_cost.npy expected size-1");
        c.py_assembled_total_cost = v[0];
    }
    c.py_assembled_trans_pairs = reamix::test::loadNpy1DInt64(c.ba_dir + "/assembled_trans_pairs.npy");
    c.py_meta_quality_score    = reamix::test::loadNpy1DFloat64(c.ba_dir + "/assembled_meta_quality_score.npy");
    c.py_meta_block_transition = reamix::test::loadNpy1DFloat64(c.ba_dir + "/assembled_meta_block_transition.npy");
    c.py_meta_crossfade_ms     = reamix::test::loadNpy1DFloat64(c.ba_dir + "/assembled_meta_crossfade_ms.npy");

    const std::string mj = readAll(c.ba_dir + "/manifest.json");
    c.n_blocks = extractInt(mj, "n_blocks", 0);
    c.top_k    = extractInt(mj, "top_k", reamix::remix::BLOCK_TOP_K);

    return c;
}

struct CaseSummary
{
    std::string track;
    bool        loaded = false;
    bool        pass   = false;
    bool        blocks_equal = true;
    bool        compat_pass  = true;
    bool        path_pass    = true;
    bool        meta_pass    = true;
    double      worst_quality = 0.0;
    double      total_cost_diff = 0.0;
    double      worst_meta_q = 0.0;
    int         n_blocks = 0;
    int         path_len_cpp = 0;
    int         path_len_py  = 0;
    int         n_trans_cpp  = 0;
    int         n_trans_py   = 0;
    std::string err;
};

CaseSummary runCase(const CaseBundle& c)
{
    CaseSummary s;
    s.track = c.track;
    s.loaded = true;
    s.n_blocks = c.n_blocks;
    s.path_len_py = static_cast<int>(c.py_assembled_beat_indices.size());
    s.n_trans_py  = static_cast<int>(c.py_assembled_trans_pairs.size() / 2);

    std::printf("\n=== %s ===\n", c.track.c_str());
    std::printf("  n_beats=%d n_blocks=%d top_k=%d\n",
                c.n_beats, c.n_blocks, c.top_k);

    // ------------------ Stage 1: buildBlocks -----------------------------
    auto cpp_blocks = reamix::remix::buildBlocks(
        c.segments.empty() ? nullptr : c.segments.data(),
        static_cast<int>(c.segments.size()),
        c.beat_times.data(),
        c.n_beats);

    if (static_cast<int>(cpp_blocks.size()) != c.n_blocks) {
        s.blocks_equal = false;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "n_blocks: py=%d cpp=%zu", c.n_blocks, cpp_blocks.size());
        s.err = buf;
    } else {
        for (int i = 0; i < c.n_blocks; ++i) {
            const auto& b = cpp_blocks[i];
            if (static_cast<std::int64_t>(b.segment_idx) != c.py_block_segment_idx[i] ||
                static_cast<std::int64_t>(b.start_beat)  != c.py_block_start_beat[i] ||
                static_cast<std::int64_t>(b.end_beat)    != c.py_block_end_beat[i] ||
                static_cast<std::int64_t>(b.n_beats)     != c.py_block_n_beats[i] ||
                static_cast<std::int64_t>(b.cluster_id)  != c.py_block_cluster_id[i] ||
                b.label        != c.py_block_labels[i] ||
                b.display_name != c.py_block_display_names[i] ||
                b.start_sec    != c.py_block_start_sec[i] ||
                b.end_sec      != c.py_block_end_sec[i] ||
                b.duration_sec != c.py_block_duration_sec[i])
            {
                s.blocks_equal = false;
                char buf[384];
                std::snprintf(buf, sizeof(buf),
                    "block[%d] mismatch: py(label='%s' dn='%s' sb=%lld eb=%lld) "
                    "cpp(label='%s' dn='%s' sb=%d eb=%d)",
                    i,
                    c.py_block_labels[i].c_str(),
                    c.py_block_display_names[i].c_str(),
                    static_cast<long long>(c.py_block_start_beat[i]),
                    static_cast<long long>(c.py_block_end_beat[i]),
                    b.label.c_str(), b.display_name.c_str(),
                    b.start_beat, b.end_beat);
                if (s.err.empty()) s.err = buf;
                break;
            }
        }
    }
    std::printf("  Stage 1 blocks: %s\n", s.blocks_equal ? "EQ" : "MISMATCH");

    // ------------------ Stage 2: computeBlockCompatibility ---------------
    reamix::remix::BlockCompatInputs bci{};
    bci.blocks               = cpp_blocks.data();
    bci.n_blocks             = static_cast<int>(cpp_blocks.size());
    bci.beat_times           = c.beat_times.data();
    bci.n_beats              = c.n_beats;
    bci.features             = c.features.data();
    bci.n_features           = c.n_features;
    bci.boundary_waveforms   = c.boundary_waveforms.empty() ? nullptr : c.boundary_waveforms.data();
    bci.n_boundary_waveforms = c.n_boundary_waveforms;
    bci.n_samples_per_bnd    = c.n_samples_per_bnd;
    bci.waveform_sample_rate = c.waveform_sample_rate;
    bci.edge_features_start  = c.edge_features_start.empty() ? nullptr : c.edge_features_start.data();
    bci.edge_features_end    = c.edge_features_end.empty()   ? nullptr : c.edge_features_end.data();
    bci.n_edge_features      = c.n_edge_features;
    bci.edge_rms_start       = c.edge_rms_start.empty() ? nullptr : c.edge_rms_start.data();
    bci.edge_rms_end         = c.edge_rms_end.empty()   ? nullptr : c.edge_rms_end.data();
    bci.rms_energy           = c.rms_energy.empty()        ? nullptr : c.rms_energy.data();
    bci.spectral_centroid    = c.spectral_centroid.empty() ? nullptr : c.spectral_centroid.data();
    bci.vocal_activity       = c.vocal_activity.empty()    ? nullptr : c.vocal_activity.data();
    bci.downbeats            = c.downbeats.empty() ? nullptr : c.downbeats.data();
    bci.n_downbeats          = static_cast<int>(c.downbeats.size());
    bci.time_signature       = 4;
    // Sesja 81 ADR-068 D1+D4 default flip — pin to legacy 10-component
    // simplex so phase-4 BA goldens stay bit-exact past kDefaultQualityWeights
    // shipping the new 6-D production simplex (with sequential_continuity
    // collapse + harmonic mean).
    bci.quality_weights      = &reamix::remix::kLegacyQualityWeights;

    auto cpp_compat = reamix::remix::computeBlockCompatibility(bci);

    const int n = c.n_blocks;
    const int K = c.top_k;
    if (cpp_compat.n != n ||
        static_cast<int>(cpp_compat.quality.size()) != n * n ||
        static_cast<int>(cpp_compat.top_k_quality.size()) != n * n * K)
    {
        s.compat_pass = false;
        if (s.err.empty()) s.err = "compat shape mismatch";
    } else {
        // Primary quality matrix.
        double worst = 0.0;
        for (std::size_t i = 0; i < cpp_compat.quality.size(); ++i) {
            const double d = std::abs(cpp_compat.quality[i] - c.py_compat_quality[i]);
            if (d > worst) worst = d;
        }
        s.worst_quality = worst;
        if (worst > TOL.quality_tol) {
            s.compat_pass = false;
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "compat.quality worst=%.6e > tol=%.1e",
                worst, TOL.quality_tol);
            if (s.err.empty()) s.err = buf;
        }
        // splice_from/to bitwise.
        for (std::size_t i = 0; i < cpp_compat.splice_from.size(); ++i) {
            if (cpp_compat.splice_from[i] != c.py_compat_splice_from[i] ||
                cpp_compat.splice_to[i]   != c.py_compat_splice_to[i])
            {
                s.compat_pass = false;
                if (s.err.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "splice[%zu]: py(%lld,%lld) cpp(%lld,%lld)",
                        i,
                        static_cast<long long>(c.py_compat_splice_from[i]),
                        static_cast<long long>(c.py_compat_splice_to[i]),
                        static_cast<long long>(cpp_compat.splice_from[i]),
                        static_cast<long long>(cpp_compat.splice_to[i]));
                    s.err = buf;
                }
                break;
            }
        }
        // Top-K quality.
        double worst_tk = 0.0;
        for (std::size_t i = 0; i < cpp_compat.top_k_quality.size(); ++i) {
            const double d = std::abs(cpp_compat.top_k_quality[i] - c.py_compat_top_k_quality[i]);
            if (d > worst_tk) worst_tk = d;
        }
        if (worst_tk > TOL.quality_tol) {
            s.compat_pass = false;
            if (s.err.empty()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "top_k_quality worst=%.6e > tol=%.1e",
                    worst_tk, TOL.quality_tol);
                s.err = buf;
            }
        }
        // Top-K from/to bitwise.
        for (std::size_t i = 0; i < cpp_compat.top_k_from.size(); ++i) {
            if (cpp_compat.top_k_from[i] != c.py_compat_top_k_from[i] ||
                cpp_compat.top_k_to[i]   != c.py_compat_top_k_to[i])
            {
                s.compat_pass = false;
                if (s.err.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "top_k[%zu]: py(%lld,%lld) cpp(%lld,%lld)",
                        i,
                        static_cast<long long>(c.py_compat_top_k_from[i]),
                        static_cast<long long>(c.py_compat_top_k_to[i]),
                        static_cast<long long>(cpp_compat.top_k_from[i]),
                        static_cast<long long>(cpp_compat.top_k_to[i]));
                    s.err = buf;
                }
                break;
            }
        }
    }
    std::printf("  Stage 2 compat: %s worst_q=%.3e\n",
                s.compat_pass ? "PASS" : "FAIL", s.worst_quality);

    // ------------------ Stage 3: assembleBlocks --------------------------
    std::vector<int> seq;
    seq.reserve(c.py_block_sequence.size());
    for (auto v : c.py_block_sequence) seq.push_back(static_cast<int>(v));

    auto cpp_path = reamix::remix::assembleBlocks(
        seq, cpp_blocks, c.beat_times.data(), c.n_beats, cpp_compat,
        /*variation=*/0, /*junction_variations=*/nullptr);

    s.path_len_cpp = static_cast<int>(cpp_path.beat_indices.size());
    s.n_trans_cpp  = static_cast<int>(cpp_path.transitions.size());

    if (s.path_len_cpp != s.path_len_py) {
        s.path_pass = false;
        if (s.err.empty()) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "path_len: py=%d cpp=%d", s.path_len_py, s.path_len_cpp);
            s.err = buf;
        }
    } else {
        for (int k = 0; k < s.path_len_cpp; ++k) {
            if (static_cast<std::int64_t>(cpp_path.beat_indices[k]) !=
                c.py_assembled_beat_indices[k])
            {
                s.path_pass = false;
                if (s.err.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "beat_indices[%d]: py=%lld cpp=%d", k,
                        static_cast<long long>(c.py_assembled_beat_indices[k]),
                        cpp_path.beat_indices[k]);
                    s.err = buf;
                }
                break;
            }
        }
    }

    s.total_cost_diff = std::abs(cpp_path.total_cost - c.py_assembled_total_cost);
    if (s.total_cost_diff > TOL.total_cost_tol) {
        s.path_pass = false;
        if (s.err.empty()) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "total_cost: py=%.6f cpp=%.6f diff=%.3e",
                c.py_assembled_total_cost, cpp_path.total_cost, s.total_cost_diff);
            s.err = buf;
        }
    }
    std::printf("  Stage 3 path: %s len_py=%d len_cpp=%d cost_diff=%.3e\n",
                s.path_pass ? "PASS" : "FAIL",
                s.path_len_py, s.path_len_cpp, s.total_cost_diff);

    // ------------------ Stage 4: transition metadata ---------------------
    // Compare (from, to) pairs set + per-key values (NaN = absent).
    if (s.n_trans_cpp != s.n_trans_py) {
        s.meta_pass = false;
        if (s.err.empty()) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "n_transitions: py=%d cpp=%d", s.n_trans_py, s.n_trans_cpp);
            s.err = buf;
        }
    } else {
        double worst_q = 0.0;
        for (int k = 0; k < s.n_trans_cpp; ++k) {
            const int fb_py = static_cast<int>(c.py_assembled_trans_pairs[2 * k + 0]);
            const int tb_py = static_cast<int>(c.py_assembled_trans_pairs[2 * k + 1]);

            // Find matching transition in cpp_path.transition_metadata (sorted
            // by std::pair<int,int> by key, matching Python sorted dump).
            const std::pair<int, int> pr{fb_py, tb_py};
            auto it = cpp_path.transition_metadata.find(pr);
            if (it == cpp_path.transition_metadata.end()) {
                s.meta_pass = false;
                if (s.err.empty()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "meta missing key (%d, %d) in cpp", fb_py, tb_py);
                    s.err = buf;
                }
                break;
            }

            auto check_key = [&](const std::string& key, const std::vector<double>& py_arr, double tol) {
                const bool py_nan = std::isnan(py_arr[k]);
                auto mit = it->second.find(key);
                const bool cpp_has = (mit != it->second.end());
                if (py_nan && cpp_has) {
                    s.meta_pass = false;
                    if (s.err.empty()) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "meta[%s] (%d,%d) py=absent cpp=%g",
                            key.c_str(), fb_py, tb_py, mit->second);
                        s.err = buf;
                    }
                } else if (!py_nan && !cpp_has) {
                    s.meta_pass = false;
                    if (s.err.empty()) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "meta[%s] (%d,%d) py=%g cpp=absent",
                            key.c_str(), fb_py, tb_py, py_arr[k]);
                        s.err = buf;
                    }
                } else if (!py_nan && cpp_has) {
                    const double d = std::abs(mit->second - py_arr[k]);
                    if (key == "quality_score" && d > worst_q) worst_q = d;
                    if (d > tol) {
                        s.meta_pass = false;
                        if (s.err.empty()) {
                            char buf[256];
                            std::snprintf(buf, sizeof(buf),
                                "meta[%s] (%d,%d) py=%g cpp=%g diff=%.3e > tol=%.1e",
                                key.c_str(), fb_py, tb_py,
                                py_arr[k], mit->second, d, tol);
                            s.err = buf;
                        }
                    }
                }
            };
            check_key("quality_score",    c.py_meta_quality_score,    TOL.meta_quality_tol);
            check_key("block_transition", c.py_meta_block_transition, TOL.meta_block_tol);
            check_key("crossfade_ms",     c.py_meta_crossfade_ms,     TOL.meta_crossfade_tol);
        }
        s.worst_meta_q = worst_q;
    }
    std::printf("  Stage 4 meta: %s worst_q=%.3e\n",
                s.meta_pass ? "PASS" : "FAIL", s.worst_meta_q);

    s.pass = s.blocks_equal && s.compat_pass && s.path_pass && s.meta_pass;
    return s;
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> tracks_to_run = kCorpusTracks;
    std::string reference_root = "tests/parity/reference/data/phase4";

    if (argc > 1) {
        std::string arg = argv[1];
        if (!arg.empty() && arg.back() == '/') arg.pop_back();
        const std::string suffix = "/block_assembly";
        if (arg.size() >= suffix.size() &&
            arg.substr(arg.size() - suffix.size()) == suffix) {
            arg = arg.substr(0, arg.size() - suffix.size());
        }
        std::size_t slash = arg.find_last_of('/');
        std::string track = (slash == std::string::npos) ? arg : arg.substr(slash + 1);

        std::printf("[bisection] %s\n", arg.c_str());
        try {
            auto c = loadCase(arg, track);
            auto r = runCase(c);
            std::printf("%s: %s err=%s\n",
                        track.c_str(), r.pass ? "PASS" : "FAIL",
                        r.err.c_str());
            return r.pass ? 0 : 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    }

    std::printf("=== BlockAssembly corpus parity (%zu tracks) ===\n",
                kCorpusTracks.size());
    std::printf("%-32s  %7s  %6s  %12s  %12s  %12s  %s\n",
                "track", "blocks", "plen",
                "worst_q", "cost_diff", "worst_mq", "status");
    std::printf("%-32s  %7s  %6s  %12s  %12s  %12s  %s\n",
                "--------------------------------", "-------", "------",
                "------------", "------------", "------------", "------");

    int passed = 0, failed = 0, skipped = 0;
    double overall_q = 0.0, overall_cost = 0.0;
    for (const auto& track : tracks_to_run) {
        const std::string dir = reference_root + "/" + track;
        try {
            auto c = loadCase(dir, track);
            auto r = runCase(c);
            if (r.worst_quality > overall_q) overall_q = r.worst_quality;
            if (r.total_cost_diff > overall_cost) overall_cost = r.total_cost_diff;
            std::printf("%-32s  %7d  %6d  %12.4e  %12.4e  %12.4e  %s\n",
                        track.c_str(), r.n_blocks, r.path_len_cpp,
                        r.worst_quality, r.total_cost_diff, r.worst_meta_q,
                        r.pass ? "PASS" : "FAIL");
            if (!r.pass && !r.err.empty()) std::printf("  err: %s\n", r.err.c_str());
            if (r.pass) ++passed; else ++failed;
        } catch (const std::exception& e) {
            std::printf("%-32s  skip: %s\n", track.c_str(), e.what());
            ++skipped;
        }
    }

    std::printf("\n=== Summary: %d pass, %d fail, %d skip, "
                "overall worst_q=%.3e cost_diff=%.3e ===\n",
                passed, failed, skipped, overall_q, overall_cost);
    return failed == 0 ? 0 : 1;
}
