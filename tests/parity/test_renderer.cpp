// test_renderer — phase-5 session 36 parity:
//   reamix::render::Renderer (build_edit_plan + _render_edit_plan + render)
//   vs Python RemixRenderer (references/python-source/remix/renderer.py
//   L70-475).
//
// Cases (in-session smoke seed per HANDOVER session 36 + ADR-034 mandate
// of full 16-track corpus at phase-close):
//   billie_jean              r=0.5  canonical happy-path (1 transition, 2 clips)
//   alice_in_chains_nutshell r=0.3  fallback (path_len=12, n_trans=0, 1 clip)
//   meshuggah_bleed          r=0.7  polyrhythmic grid    (1 transition, 2 clips)
//
// Per-case files under references/golden/phase-5/renderer/<track>/r<R>/:
//   audio_in.npy / beat_times.npy / remix_path.npy / remix_pairs.npy /
//   remix_meta_keys.json / remix_meta_<key>.npy (NaN where absent) /
//   render_audio.npy / transition_times.npy / edit_plan.json / manifest.json
//
// Gates:
//   * Per-sample audio max_abs ≤ 5e-4 (phase-5 spec L27 starting gate, mirror
//     of phase-4 DP cost class). Threshold refined ONLY via bisection per
//     ADR-027 + ADR-034 + memory/feedback_root_cause_over_widening.md —
//     never widened without root-cause.
//   * transition_times f64 max_abs ≤ 1e-9.
//   * EditClip integer fields bit-exact (clip_index, start_beat, end_beat).
//   * EditClip f64 sec-fields max_abs ≤ 1e-9.
//   * EditPlan duration / nTransitions / nClips integer-exact.
//
// `-ffp-contract=off` 16th ADR-028 reuse — Renderer composes already-
// validated modules (Splice + Crossfade) plus per-sample fade-multiply
// (`f32 *= f64`) and equal-power np.cos/np.sin ramps. All scalar accumulator
// classes covered by ADR-028 reasoning.

#include "render/Renderer.h"
#include "remix/Path.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "NpyIO.h"

namespace {

constexpr double TOL_AUDIO_F32_ABS  = 5e-4;   // phase-5 spec L27
constexpr double TOL_TRANSITION_SEC = 1e-9;
constexpr double TOL_CLIP_SEC       = 1e-9;

struct Case {
    std::string track;
    std::string ratioDir;          // "r03", "r05", "r07"
    std::string label;
};

// Phase-5 ADR-034 phase-close gate: 16 tracks × 3 ratios = 48 cases.
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

inline std::string ratioLabel(double r)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "r%02d",
                  static_cast<int>(std::lround(r * 10.0)));
    return std::string(buf);
}

// --- File helpers --------------------------------------------------------

std::string loadFile(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string dataRoot(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') return std::string(argv[i]);
    }
    return std::string("references/golden/phase-5/renderer");
}

// --- JSON micro-reader (top-level array of objects + flat key:value) -----

std::size_t skipWS(const std::string& txt, std::size_t i)
{
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '\t' || txt[i] == '\n' || txt[i] == '\r')) ++i;
    return i;
}

// Find balanced object span starting at txt[i] == '{'. Returns the index of
// the matching '}'. Throws on unbalanced.
std::size_t findObjectEnd(const std::string& txt, std::size_t start)
{
    if (start >= txt.size() || txt[start] != '{') {
        throw std::runtime_error("findObjectEnd: not at '{'");
    }
    int depth = 0;
    bool inStr = false;
    bool esc   = false;
    for (std::size_t i = start; i < txt.size(); ++i) {
        const char c = txt[i];
        if (inStr) {
            if (esc) { esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"')  { inStr = false; }
            continue;
        }
        if (c == '"') { inStr = true; continue; }
        if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) return i;
        }
    }
    throw std::runtime_error("findObjectEnd: unbalanced");
}

// Extract list of top-level objects from a JSON array text.
std::vector<std::string> jsonArrayObjects(const std::string& txt)
{
    std::vector<std::string> out;
    std::size_t i = skipWS(txt, 0);
    if (i >= txt.size() || txt[i] != '[') {
        throw std::runtime_error("jsonArrayObjects: not an array");
    }
    ++i;
    while (true) {
        i = skipWS(txt, i);
        if (i >= txt.size()) break;
        if (txt[i] == ']') break;
        if (txt[i] != '{') {
            // skip stray comma / whitespace
            ++i;
            continue;
        }
        const std::size_t end = findObjectEnd(txt, i);
        out.emplace_back(txt.substr(i, end - i + 1));
        i = end + 1;
        i = skipWS(txt, i);
        if (i < txt.size() && txt[i] == ',') ++i;
    }
    return out;
}

std::size_t keyColon(const std::string& obj, const char* key)
{
    const std::string needle = std::string("\"") + key + "\"";
    auto p = obj.find(needle);
    if (p == std::string::npos) return std::string::npos;
    auto q = obj.find(':', p);
    if (q == std::string::npos) return std::string::npos;
    ++q;
    while (q < obj.size() && (obj[q] == ' ' || obj[q] == '\t')) ++q;
    return q;
}

double objNumber(const std::string& obj, const char* key)
{
    const std::size_t q = keyColon(obj, key);
    if (q == std::string::npos) {
        throw std::runtime_error(std::string("missing key: ") + key);
    }
    return std::atof(obj.c_str() + q);
}

std::int64_t objInt64(const std::string& obj, const char* key)
{
    const std::size_t q = keyColon(obj, key);
    if (q == std::string::npos) {
        throw std::runtime_error(std::string("missing key: ") + key);
    }
    return static_cast<std::int64_t>(std::atoll(obj.c_str() + q));
}

// Parse remix_meta_keys.json (array of strings).
std::vector<std::string> parseStringArray(const std::string& txt)
{
    std::vector<std::string> out;
    std::size_t i = skipWS(txt, 0);
    if (i >= txt.size() || txt[i] != '[') return out;
    ++i;
    while (true) {
        i = skipWS(txt, i);
        if (i >= txt.size() || txt[i] == ']') break;
        if (txt[i] != '"') { ++i; continue; }
        const auto end = txt.find('"', i + 1);
        if (end == std::string::npos) break;
        out.emplace_back(txt.substr(i + 1, end - i - 1));
        i = end + 1;
        i = skipWS(txt, i);
        if (i < txt.size() && txt[i] == ',') ++i;
    }
    return out;
}

// --- Per-case runner -----------------------------------------------------

struct CaseStats {
    int           nClips           = 0;
    int           nClipFieldErrors = 0;
    double        maxClipSecDiff   = 0.0;
    int           nSamples         = 0;
    double        maxAudioAbs      = 0.0;
    int           nOverTol         = 0;
    double        maxTransitionDiff = 0.0;
    int           nTransitions     = 0;
};

bool compareEditClip(const reamix::render::EditClip& cpp,
                     const std::string&             pyObj,
                     CaseStats&                     stats,
                     std::string&                   errMsg)
{
    bool ok = true;

    auto checkInt = [&](const char* key, int cppVal) {
        const std::int64_t pyVal = objInt64(pyObj, key);
        if (cppVal != static_cast<int>(pyVal)) {
            ok = false;
            ++stats.nClipFieldErrors;
            errMsg += std::string(" ") + key + "(cpp=" + std::to_string(cppVal)
                   + " py=" + std::to_string(pyVal) + ")";
        }
    };
    auto checkSec = [&](const char* key, double cppVal) {
        const double pyVal = objNumber(pyObj, key);
        const double d = std::fabs(cppVal - pyVal);
        if (d > stats.maxClipSecDiff) stats.maxClipSecDiff = d;
        if (d > TOL_CLIP_SEC) {
            ok = false;
            ++stats.nClipFieldErrors;
            errMsg += std::string(" ") + key + "(diff=" + std::to_string(d) + ")";
        }
    };

    checkInt("clip_index",        cpp.clipIndex);
    checkInt("start_beat",        cpp.startBeat);
    checkInt("end_beat",          cpp.endBeat);
    checkSec("source_start_sec",   cpp.sourceStartSec);
    checkSec("source_end_sec",     cpp.sourceEndSec);
    checkSec("timeline_start_sec", cpp.timelineStartSec);
    checkSec("timeline_end_sec",   cpp.timelineEndSec);
    checkSec("duration_sec",       cpp.durationSec);
    checkSec("fade_in_sec",        cpp.fadeInSec);
    checkSec("fade_out_sec",       cpp.fadeOutSec);
    checkSec("overlap_before_sec", cpp.overlapBeforeSec);
    checkSec("overlap_after_sec",  cpp.overlapAfterSec);
    checkSec("gap_after_sec",      cpp.gapAfterSec);
    return ok;
}

int runCase(const std::string& root, const Case& c, CaseStats& stats)
{
    using namespace reamix::test;
    const std::string caseDir = root + "/" + c.track + "/" + c.ratioDir;

    // --- Inputs ----------------------------------------------------------
    NpyMatrixF32 audioIn = loadNpy2DFloat32(caseDir + "/audio_in.npy");
    std::vector<double>       beatTimes  = loadNpy1DFloat64(caseDir + "/beat_times.npy");
    std::vector<std::int64_t> remixPath  = loadNpy1DInt64  (caseDir + "/remix_path.npy");
    std::vector<std::int64_t> remixPairs = loadNpy1DInt64  (caseDir + "/remix_pairs.npy");

    // sample_rate from manifest.
    const std::string manifestTxt = loadFile(caseDir + "/manifest.json");
    const std::size_t srPos = keyColon(manifestTxt, "sample_rate");
    if (srPos == std::string::npos) throw std::runtime_error("no sample_rate");
    const int sr = static_cast<int>(std::atoi(manifestTxt.c_str() + srPos));

    const std::size_t nCh = audioIn.rows;
    const std::size_t nSamples = audioIn.cols;

    // --- Build RemixPath from dumps -------------------------------------
    reamix::remix::RemixPath path;
    path.beat_indices.assign(remixPath.begin(), remixPath.end());
    path.duration_beats = static_cast<int>(remixPath.size());

    if (remixPairs.size() % 2 != 0) throw std::runtime_error("pairs not even");
    const std::size_t nTrans = remixPairs.size() / 2;
    path.transitions.reserve(nTrans);
    for (std::size_t t = 0; t < nTrans; ++t) {
        path.transitions.emplace_back(
            static_cast<int>(remixPairs[2 * t]),
            static_cast<int>(remixPairs[2 * t + 1]));
    }

    const std::vector<std::string> metaKeys =
        parseStringArray(loadFile(caseDir + "/remix_meta_keys.json"));
    for (const std::string& k : metaKeys) {
        std::string safe = k;
        for (auto& ch : safe) {
            if (ch == '/' || ch == '\\') ch = '_';
        }
        std::vector<double> vec =
            loadNpy1DFloat64(caseDir + "/remix_meta_" + safe + ".npy");
        if (vec.size() != nTrans) {
            throw std::runtime_error("meta vec length != nTrans for " + k);
        }
        for (std::size_t t = 0; t < nTrans; ++t) {
            const double v = vec[t];
            if (std::isnan(v)) continue; // absent in this transition
            path.transition_metadata[path.transitions[t]][k] = v;
        }
    }

    // total_cost not strictly needed for renderer parity; leave 0.0.
    path.total_cost = 0.0;

    // --- Construct + run Renderer ---------------------------------------
    reamix::render::RendererConfig cfg; // defaults from config.py:147-178
    reamix::render::Renderer renderer(
        /*audioPath*/ std::string("phase5-test:") + c.track,
        audioIn.data.data(), nCh, nSamples, sr,
        beatTimes.data(), beatTimes.size(),
        /*crossfadeMsOrNeg*/ -1.0,  // use crossfade_beats by default
        cfg);

    reamix::render::RenderResult result = renderer.render(path);

    // --- Compare audio (per-sample max_abs, f32 vs f32) -----------------
    NpyMatrixF32 audioExp = loadNpy2DFloat32(caseDir + "/render_audio.npy");
    if (audioExp.rows != result.nChannels
        || audioExp.cols != result.nSamples) {
        std::printf("  shape mismatch: cpp=(%zu,%zu) py=(%zu,%zu)\n",
                    result.nChannels, result.nSamples,
                    audioExp.rows, audioExp.cols);
        return 1;
    }
    stats.nSamples = static_cast<int>(result.nChannels * result.nSamples);
    for (std::size_t i = 0; i < result.audio.size(); ++i) {
        const double d = std::fabs(static_cast<double>(result.audio[i])
                                   - static_cast<double>(audioExp.data[i]));
        if (d > stats.maxAudioAbs) stats.maxAudioAbs = d;
        if (d > TOL_AUDIO_F32_ABS) ++stats.nOverTol;
    }

    // --- Compare transition_times ---------------------------------------
    std::vector<double> ttExp =
        loadNpy1DFloat64(caseDir + "/transition_times.npy");
    if (ttExp.size() != result.transitionTimes.size()) {
        std::printf("  transition_times count mismatch: cpp=%zu py=%zu\n",
                    result.transitionTimes.size(), ttExp.size());
        return 1;
    }
    stats.nTransitions = static_cast<int>(ttExp.size());
    for (std::size_t t = 0; t < ttExp.size(); ++t) {
        const double d = std::fabs(result.transitionTimes[t] - ttExp[t]);
        if (d > stats.maxTransitionDiff) stats.maxTransitionDiff = d;
    }

    // --- Compare edit_plan ----------------------------------------------
    const std::string planTxt = loadFile(caseDir + "/edit_plan.json");
    const std::vector<std::string> pyClips = jsonArrayObjects(planTxt);
    if (pyClips.size() != result.editPlan.clips.size()) {
        std::printf("  edit_plan clip count mismatch: cpp=%zu py=%zu\n",
                    result.editPlan.clips.size(), pyClips.size());
        return 1;
    }
    stats.nClips = static_cast<int>(result.editPlan.clips.size());
    int allOk = 0;
    for (std::size_t i = 0; i < result.editPlan.clips.size(); ++i) {
        std::string err;
        if (!compareEditClip(result.editPlan.clips[i], pyClips[i], stats, err)) {
            std::printf("  clip[%zu] mismatch:%s\n", i, err.c_str());
            allOk = 1;
        }
    }
    return allOk;
}

} // anon

int main(int argc, char** argv)
{
    const std::string root = dataRoot(argc, argv);

    // Build full 48-case corpus dynamically from kCorpusTracks × kCorpusRatios.
    std::vector<Case> cases;
    cases.reserve(kCorpusTracks.size() * kCorpusRatios.size());
    for (const std::string& track : kCorpusTracks) {
        for (double r : kCorpusRatios) {
            Case c;
            c.track    = track;
            c.ratioDir = ratioLabel(r);
            c.label    = track + " " + c.ratioDir;
            cases.push_back(c);
        }
    }

    int totalFailed = 0;
    int casesPassed = 0;
    int casesMissing = 0;
    double worstAudioAbs = 0.0;
    double worstTtAbs    = 0.0;
    double worstClipAbs  = 0.0;
    std::vector<std::string> failLabels;

    for (const Case& c : cases) {
        // Skip cases without dumped goldens (graceful MISS — phase-4 pattern).
        const std::string caseDir = root + "/" + c.track + "/" + c.ratioDir;
        std::ifstream probe(caseDir + "/manifest.json");
        if (!probe.good()) {
            ++casesMissing;
            std::printf("[%s] MISS (no manifest.json under %s)\n",
                        c.label.c_str(), caseDir.c_str());
            continue;
        }

        CaseStats stats;
        std::printf("[%s]\n", c.label.c_str());
        int rc = 0;
        try {
            rc = runCase(root, c, stats);
        } catch (const std::exception& e) {
            std::printf("  EXCEPTION: %s\n", e.what());
            rc = 1;
        }

        const bool audioPass = stats.nOverTol == 0
                            && stats.maxAudioAbs <= TOL_AUDIO_F32_ABS;
        const bool ttPass    = stats.maxTransitionDiff <= TOL_TRANSITION_SEC;
        const bool clipPass  = stats.nClipFieldErrors == 0;
        const bool casePass  = rc == 0 && audioPass && ttPass && clipPass;

        if (stats.maxAudioAbs       > worstAudioAbs) worstAudioAbs = stats.maxAudioAbs;
        if (stats.maxTransitionDiff > worstTtAbs)    worstTtAbs    = stats.maxTransitionDiff;
        if (stats.maxClipSecDiff    > worstClipAbs)  worstClipAbs  = stats.maxClipSecDiff;

        std::printf("  clips=%d  trans=%d  audio_samples=%d  audio_max_abs=%.3e  over_tol=%d"
                    "  tt_max=%.3e  clip_sec_max=%.3e  -> %s\n",
                    stats.nClips, stats.nTransitions,
                    stats.nSamples,
                    stats.maxAudioAbs,
                    stats.nOverTol,
                    stats.maxTransitionDiff,
                    stats.maxClipSecDiff,
                    casePass ? "PASS" : "FAIL");

        if (casePass) ++casesPassed;
        else {
            ++totalFailed;
            failLabels.push_back(c.label);
        }
    }

    std::printf("\n================== RENDERER CORPUS SUMMARY ==================\n");
    std::printf("PASS: %d / FAIL: %d / MISS: %d / TOTAL: %zu\n",
                casesPassed, totalFailed, casesMissing, cases.size());
    std::printf("Worst audio_max_abs = %.3e (gate %.1e)\n",
                worstAudioAbs, TOL_AUDIO_F32_ABS);
    std::printf("Worst tt_max_abs    = %.3e (gate %.1e)\n",
                worstTtAbs, TOL_TRANSITION_SEC);
    std::printf("Worst clip_sec_max  = %.3e (gate %.1e)\n",
                worstClipAbs, TOL_CLIP_SEC);
    if (!failLabels.empty()) {
        std::printf("\nFailed cases:\n");
        for (const auto& l : failLabels) std::printf("  - %s\n", l.c_str());
    }
    return totalFailed == 0 ? 0 : 1;
}
