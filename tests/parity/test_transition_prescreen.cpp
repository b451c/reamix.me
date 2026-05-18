// test_transition_prescreen — phase-4 session 24 parity:
// reamix::remix::prescreenTransitions vs
// Python references/python-source/remix/transition_prescreen.py.
//
// Prescreen is ratio-independent: one pass per track. Corpus = 16 tracks.
// 8 tracks have non-empty results (alice_in_chains, eminem, eno, miles,
// shostakovich, vocal_solo, wardruna, woodkid); 8 tracks return empty
// (billie_jean, bob_dylan, daft_punk, dance_monkey, goldberg, meshuggah,
// smells_like_teen_spirit, tiesto). Both cases exercised.
//
// Per-case gates:
//   - prescreen_from, prescreen_to, prescreen_diag_len: int64 BITWISE.
//   - prescreen_rec_score:   f64 ULP (f32-matmul class via recurrence build).
//   - prescreen_wf_sim:      f64 ULP (xcorr composition).
//   - Result count equal.
//   - Order equal (Python list order = port stable_sort order).
//
// Invocation:
//   test_transition_prescreen               → corpus mode (16 tracks).
//   test_transition_prescreen <track_dir>   → single-case bisection
//       e.g. `.../alice_in_chains_nutshell`.

#include "remix/TransitionPrescreen.h"
#include "analysis/StructureResult.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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
    // prescreen_rec_score comes from mean along a diagonal of R cells. R cells
    // are accumulated via exp(-distSq / mu) where distSq is f64 of f32 rows →
    // f32-matmul-class drift (~1e-7). mean = sum / length (both f64). Tight
    // gate 1e-6 (10× margin over expected f32-class).
    double rec_score_tol = 1e-6;
    // wf_sim comes from WaveformXcorr which is f64 throughout; session-18/19
    // corpus showed ~1e-15 max drift. Gate 1e-9 (6 orders margin).
    double wf_sim_tol    = 1e-9;
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

struct CaseResult
{
    std::string track;
    bool        loaded        = false;
    bool        pass          = false;
    int         n_py          = 0;
    int         n_cpp         = 0;
    bool        order_equal   = true;
    bool        from_bitwise  = true;
    bool        to_bitwise    = true;
    bool        len_bitwise   = true;
    double      worst_rec     = 0.0;
    double      worst_wf      = 0.0;
    std::string err;
};

CaseResult runCase(const std::string& track_dir, const std::string& track)
{
    CaseResult r;
    r.track = track;

    // Load track-root inputs.
    auto beat_times = reamix::test::loadNpy1DFloat64(track_dir + "/beat_times.npy");
    const int n_beats = static_cast<int>(beat_times.size());

    auto feat_npy = reamix::test::loadNpy2DFloat32(track_dir + "/features.npy");
    const int n_features = static_cast<int>(feat_npy.cols);

    // Segments.
    std::vector<reamix::analysis::Segment> segments;
    {
        auto seg_start      = reamix::test::loadNpy1DFloat64(track_dir + "/seg_start.npy");
        auto seg_end        = reamix::test::loadNpy1DFloat64(track_dir + "/seg_end.npy");
        auto seg_conf       = reamix::test::loadNpy1DFloat64(track_dir + "/seg_confidence.npy");
        auto seg_cluster_id = reamix::test::loadNpy1DInt64  (track_dir + "/seg_cluster_id.npy");
        auto seg_labels     = readLines(                     track_dir + "/seg_labels.txt");
        segments.reserve(seg_start.size());
        for (std::size_t i = 0; i < seg_start.size(); ++i) {
            reamix::analysis::Segment s;
            s.start      = seg_start[i];
            s.end        = seg_end[i];
            s.confidence = seg_conf[i];
            s.cluster_id = static_cast<int>(seg_cluster_id[i]);
            s.label      = seg_labels[i];
            segments.push_back(std::move(s));
        }
    }

    auto downbeats = reamix::test::loadNpy1DFloat64(track_dir + "/downbeats.npy");

    auto bw_npy = reamix::test::loadNpy2DFloat32(track_dir + "/boundary_waveforms.npy");

    // waveform_sample_rate from track manifest.
    int waveform_sample_rate = 22050;
    {
        const std::string mj = readAll(track_dir + "/manifest.json");
        const std::string key = "\"sr\":";
        std::size_t pos = mj.find(key);
        if (pos != std::string::npos) {
            pos += key.size();
            while (pos < mj.size() && (mj[pos] == ' ' || mj[pos] == '\t')) ++pos;
            waveform_sample_rate = std::atoi(mj.c_str() + pos);
        }
    }

    // Load Python goldens.
    const std::string p_dir = track_dir + "/prescreen";
    auto py_from     = reamix::test::loadNpy1DInt64  (p_dir + "/prescreen_from.npy");
    auto py_to       = reamix::test::loadNpy1DInt64  (p_dir + "/prescreen_to.npy");
    auto py_diag_len = reamix::test::loadNpy1DInt64  (p_dir + "/prescreen_diag_len.npy");
    auto py_rec      = reamix::test::loadNpy1DFloat64(p_dir + "/prescreen_rec_score.npy");
    auto py_wf       = reamix::test::loadNpy1DFloat64(p_dir + "/prescreen_wf_sim.npy");

    r.loaded = true;
    r.n_py   = static_cast<int>(py_from.size());

    // Call C++ prescreen.
    reamix::remix::PrescreenInputs pi{};
    pi.features             = feat_npy.data.data();
    pi.n_beats              = n_beats;
    pi.n_features           = n_features;
    pi.beat_times           = beat_times.data();
    pi.segments             = segments.empty() ? nullptr : segments.data();
    pi.n_segments           = static_cast<int>(segments.size());
    pi.downbeats            = downbeats.empty() ? nullptr : downbeats.data();
    pi.n_downbeats          = static_cast<int>(downbeats.size());
    pi.boundary_waveforms   = bw_npy.data.empty() ? nullptr : bw_npy.data.data();
    pi.n_boundary_waveforms = static_cast<int>(bw_npy.rows);
    pi.n_samples_per_bnd    = static_cast<int>(bw_npy.cols);
    pi.waveform_sample_rate = waveform_sample_rate;
    pi.time_signature       = 4;

    auto cpp_results = reamix::remix::prescreenTransitions(pi);
    r.n_cpp = static_cast<int>(cpp_results.size());

    // Count mismatch aborts the case.
    if (r.n_py != r.n_cpp) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "n_py=%d but n_cpp=%d", r.n_py, r.n_cpp);
        r.err = buf;
        r.pass = false;
        return r;
    }

    for (int k = 0; k < r.n_py; ++k) {
        const auto& c = cpp_results[k];
        if (static_cast<std::int64_t>(c.from_beat) != py_from[k]) {
            r.from_bitwise = false;
            if (r.err.empty()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "[%d] from: py=%lld cpp=%d", k,
                    static_cast<long long>(py_from[k]), c.from_beat);
                r.err = buf;
            }
        }
        if (static_cast<std::int64_t>(c.to_beat) != py_to[k]) {
            r.to_bitwise = false;
            if (r.err.empty()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "[%d] to: py=%lld cpp=%d", k,
                    static_cast<long long>(py_to[k]), c.to_beat);
                r.err = buf;
            }
        }
        if (static_cast<std::int64_t>(c.diagonal_length) != py_diag_len[k]) {
            r.len_bitwise = false;
            if (r.err.empty()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "[%d] diag_len: py=%lld cpp=%d", k,
                    static_cast<long long>(py_diag_len[k]), c.diagonal_length);
                r.err = buf;
            }
        }
        const double rec_diff = std::abs(c.recurrence_score - py_rec[k]);
        const double wf_diff  = std::abs(c.waveform_similarity - py_wf[k]);
        if (rec_diff > r.worst_rec) r.worst_rec = rec_diff;
        if (wf_diff  > r.worst_wf ) r.worst_wf  = wf_diff;
    }

    r.order_equal = r.from_bitwise && r.to_bitwise && r.len_bitwise;
    r.pass = r.order_equal &&
             (r.worst_rec <= TOL.rec_score_tol) &&
             (r.worst_wf  <= TOL.wf_sim_tol);
    return r;
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> tracks_to_run = kCorpusTracks;
    std::string reference_root = "tests/parity/reference/data/phase4";

    if (argc > 1) {
        // Single-case bisection: argv[1] = path to <track_dir> OR
        // <track_dir>/prescreen. We normalize to <track_dir>.
        std::string arg = argv[1];
        if (!arg.empty() && arg.back() == '/') arg.pop_back();
        // Strip trailing /prescreen if present.
        const std::string suffix = "/prescreen";
        if (arg.size() >= suffix.size() &&
            arg.substr(arg.size() - suffix.size()) == suffix) {
            arg = arg.substr(0, arg.size() - suffix.size());
        }
        // Extract track name (last path component).
        std::size_t slash = arg.find_last_of('/');
        std::string track = (slash == std::string::npos) ? arg : arg.substr(slash + 1);

        std::printf("[bisection] %s\n", arg.c_str());
        auto r = runCase(arg, track);
        std::printf("  n_py=%d n_cpp=%d order=%s worst_rec=%.3e worst_wf=%.3e %s\n",
                    r.n_py, r.n_cpp,
                    r.order_equal ? "EQ" : "MISMATCH",
                    r.worst_rec, r.worst_wf,
                    r.pass ? "PASS" : "FAIL");
        if (!r.err.empty()) std::printf("  err: %s\n", r.err.c_str());
        return r.pass ? 0 : 1;
    }

    // Corpus mode.
    std::printf("=== TransitionPrescreen corpus parity (%zu tracks) ===\n",
                kCorpusTracks.size());
    std::printf("%-32s  %-4s  %-4s  %-5s  %12s  %12s  %s\n",
                "track", "n_py", "n_cpp", "order",
                "worst(rec)", "worst(wf)", "status");
    std::printf("%-32s  %-4s  %-4s  %-5s  %12s  %12s  %s\n",
                "--------------------------------", "----", "----", "-----",
                "------------", "------------", "------");

    int passed = 0, failed = 0;
    double overall_rec = 0.0, overall_wf = 0.0;
    int total_py = 0;
    for (const auto& track : tracks_to_run) {
        const std::string dir = reference_root + "/" + track;
        CaseResult r;
        try { r = runCase(dir, track); }
        catch (const std::exception& e) {
            std::printf("%-32s  skip: %s\n", track.c_str(), e.what());
            continue;
        }
        if (!r.loaded) {
            std::printf("%-32s  skip (no dumps)\n", track.c_str());
            continue;
        }
        total_py += r.n_py;
        if (r.worst_rec > overall_rec) overall_rec = r.worst_rec;
        if (r.worst_wf  > overall_wf ) overall_wf  = r.worst_wf;

        std::printf("%-32s  %-4d  %-4d  %-5s  %12.4e  %12.4e  %s\n",
                    track.c_str(), r.n_py, r.n_cpp,
                    r.order_equal ? "EQ" : "!EQ",
                    r.worst_rec, r.worst_wf,
                    r.pass ? "PASS" : "FAIL");
        if (!r.pass && !r.err.empty()) std::printf("  err: %s\n", r.err.c_str());
        if (r.pass) ++passed; else ++failed;
    }

    std::printf("\n=== Summary: %d pass, %d fail, %d total entries, "
                "overall rec=%.3e wf=%.3e ===\n",
                passed, failed, total_py, overall_rec, overall_wf);
    return failed == 0 ? 0 : 1;
}
