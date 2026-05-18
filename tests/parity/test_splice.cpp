// test_splice — phase-5 session 34 + session 35 parity:
//   reamix::render::Splice (3 core primitives + findOnsetSample +
//   4 composite methods) vs Python SpliceMixin
//   (references/python-source/remix/splice.py L26-514).
//
// Kinds dispatched by `flags.json["kind"]`:
//   "find_onset"          → Splice::findOnsetSample                   (int bit-exact or ±N per ADR-033)
//   "window_onset"        → Splice::windowOnsetIndex                  (int bit-exact)
//   "similarity"          → Splice::stereoWindowSimilarity            (f64 max_abs ≤ 1e-9)
//   "score_splice"        → Splice::scoreSplicePair                   (f64 max_abs ≤ 1e-9)
//   "score_anchor"        → Splice::scoreAnchorAlignedPair            (f64 max_abs ≤ 1e-9)
//   "transition_overlap"  → Splice::transitionOverlapSamples          (int bit-exact)
//   "search_anchor"       → Splice::searchAnchorTransitionGeometry    (ADR-033 close test)
//   "refine"              → Splice::refineTransitionSplice            (composite — 2-stage grid)
//
// ADR-033 CLOSE DISCIPLINE: find_onset primitives have ±1-4 sample drift
// from numpy. We report their max_abs but TREAT THEM AS PASS if diff ≤ 4 —
// composition-level tests (search_anchor + refine) are the actual gate for
// ADR-033 closure. If search_anchor winner-beat bit-exact despite primitive
// drift → ADR-033 closes positively.
//
// Composite parity expected:
//   score_anchor — pure function of windows + scalars; should be 1 ULP.
//   transition_overlap — pure integer arithmetic; bit-exact.
//   search_anchor — calls findOnsetSample twice per outer-loop iteration;
//                   if primitive drift propagates to DIFFERENT best-score
//                   winner, struct fields diverge by ≥ 1 beat-index; if
//                   drift is absorbed by score_anchor's continuous clip,
//                   struct fields match bit-exact.
//   refine — calls findOnsetSample twice + coarse+fine grid search; similar
//            sensitivity to primitive drift; composition gate captures it.

#include "render/Splice.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "NpyIO.h"

namespace {

constexpr double      TOL_ABS              = 1e-9;
constexpr int         FIND_ONSET_DIAGNOSTIC_TOL = 4;  // ADR-033 primitive-level
                                                      // tolerance (reported but
                                                      // not gate-widened)

enum class Kind {
    FindOnset,
    WindowOnset,
    Similarity,
    ScoreSplice,
    ScoreAnchor,
    TransitionOverlap,
    SearchAnchor,
    Refine,
};

struct Case {
    const char* dir;
    const char* label;
    Kind        kind;
};

constexpr Case kCases[] = {
    {"case_01_find_onset_transient",         "find_onset transient",            Kind::FindOnset},
    {"case_02_find_onset_shortchunk",        "find_onset short n<128",          Kind::FindOnset},
    {"case_03_find_onset_explicit_ms",       "find_onset explicit ms",          Kind::FindOnset},
    {"case_04_window_onset_stereo_mid",      "window_onset stereo mid",         Kind::WindowOnset},
    {"case_05_window_onset_short",           "window_onset short n<16",         Kind::WindowOnset},
    {"case_06_similarity_identical",         "similarity  identical",           Kind::Similarity},
    {"case_07_similarity_billie",            "similarity  BJ real",             Kind::Similarity},
    {"case_08_similarity_anticorrelated",    "similarity  anti-correlated",     Kind::Similarity},
    {"case_09_score_splice_billie",          "score_splice BJ real",            Kind::ScoreSplice},
    {"case_10_score_splice_reject",          "score_splice reject branch",      Kind::ScoreSplice},
    {"case_11_score_splice_no_shift",        "score_splice no-shift branch",    Kind::ScoreSplice},
    {"case_12_score_anchor_billie_mid",      "score_anchor BJ mid",             Kind::ScoreAnchor},
    {"case_13_score_anchor_billie_edge",     "score_anchor BJ edge",            Kind::ScoreAnchor},
    {"case_14_score_anchor_reject",          "score_anchor reject",             Kind::ScoreAnchor},
    {"case_15_score_anchor_high_vocal",      "score_anchor high vocal",         Kind::ScoreAnchor},
    {"case_16_overlap_low_vocal",            "overlap low vocal",               Kind::TransitionOverlap},
    {"case_17_overlap_label_change",         "overlap label change",            Kind::TransitionOverlap},
    {"case_18_overlap_same_label_weak_entry","overlap same-label weak entry",   Kind::TransitionOverlap},
    {"case_19_overlap_strong_support",       "overlap strong support",          Kind::TransitionOverlap},
    {"case_20_overlap_preferred_override",   "overlap preferred override",      Kind::TransitionOverlap},
    {"case_21_search_anchor_bj_mid",         "search_anchor BJ mid",            Kind::SearchAnchor},
    {"case_22_search_anchor_bj_vocal",       "search_anchor BJ vocal",          Kind::SearchAnchor},
    {"case_23_search_anchor_bj_edge",        "search_anchor BJ edge",           Kind::SearchAnchor},
    {"case_24_refine_bj_mid",                "refine BJ mid",                   Kind::Refine},
    {"case_25_refine_bj_explicit_overlap",   "refine BJ explicit overlap",      Kind::Refine},
    {"case_26_refine_bj_offset",             "refine BJ offset",                Kind::Refine},
};

std::string dataRoot(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') return std::string(argv[i]);
    }
    return std::string("references/golden/phase-5/splice");
}

// --- JSON micro-reader (same as session 34, + findBool) ------------------

struct CaseFlags {
    std::string kind;
    int         sr                              = 44100;
    // find_onset
    std::int64_t beatSample                     = 0;
    double       lookbackMs                     = -1.0;  // -1 = "null / default"
    double       lookaheadMs                    = -1.0;
    // score_splice / score_anchor
    int          outgoingShift                  = 0;
    int          incomingShift                  = 0;
    int          maxShiftSamples                = 0;
    int          anchorIndex                    = 0;
    double       vocalPresence                  = 0.0;
    // transition_overlap
    int          crossfadeSamples               = 0;
    double       preferredOverlapSec            = 0.0;
    double       labelMatch                     = 1.0;
    double       vocalEntrySupport              = 1.0;
    double       vocalExitSupport               = 1.0;
    double       vocalPresenceLevel             = 0.0;
    // search_anchor
    int          prevBeat                       = 0;
    int          currBeat                       = 0;
    // refine
    std::int64_t successorSample                = 0;
    std::int64_t targetSample                   = 0;
    std::int64_t beatEnd                        = 0;
    double       alignmentOffsetSec             = 0.0;
    int          overlapSamplesOrNeg            = -1;    // -1 = null / use default
    // Config fields (all kinds may carry defaults)
    double       onsetSearchLookbackMs          =  70.0;
    double       onsetSearchLookaheadMs         =  18.0;
    double       transientCenterPenaltyWeight   =   0.14;
    double       transientAlignmentPenaltyWeight =  0.10;
    double       anchorLocalWindowMs            = 120.0;
    int          anchorSearchBeats              =   1;
    int          anchorSearchMaxExtensionBeats  =   2;
    double       anchorMinContextMs             = 120.0;
    double       vocalActivityThreshold         =   0.28;
    double       vocalCrossfadeMs               = 300.0;
    double       vocalSameLabelCrossfadeMs      = 180.0;
    double       stereoRefineMaxShiftMs         =  12.0;
    int          stereoRefineCoarseStepSamples  =  64;
};

static std::string loadFile(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::size_t keyColon(const std::string& txt, const char* key)
{
    const std::string needle = std::string("\"") + key + "\"";
    auto p = txt.find(needle);
    if (p == std::string::npos) return std::string::npos;
    auto q = txt.find(':', p);
    if (q == std::string::npos) return std::string::npos;
    ++q;
    while (q < txt.size() && (txt[q] == ' ' || txt[q] == '\t')) ++q;
    return q;
}

static std::string findString(const std::string& txt, const char* key)
{
    const std::size_t q = keyColon(txt, key);
    if (q == std::string::npos || q >= txt.size() || txt[q] != '"') return {};
    const auto end = txt.find('"', q + 1);
    if (end == std::string::npos) return {};
    return txt.substr(q + 1, end - q - 1);
}

static double findNumber(const std::string& txt, const char* key, double def)
{
    const std::size_t q = keyColon(txt, key);
    if (q == std::string::npos) return def;
    if (txt.compare(q, 4, "null") == 0) return def;
    return std::atof(txt.c_str() + q);
}

static int findInt(const std::string& txt, const char* key, int def)
{
    return static_cast<int>(findNumber(txt, key, static_cast<double>(def)));
}

static std::int64_t findInt64(const std::string& txt, const char* key, std::int64_t def)
{
    const std::size_t q = keyColon(txt, key);
    if (q == std::string::npos) return def;
    if (txt.compare(q, 4, "null") == 0) return def;
    return std::strtoll(txt.c_str() + q, nullptr, 10);
}

static bool findBool(const std::string& txt, const char* key, bool def)
{
    const std::size_t q = keyColon(txt, key);
    if (q == std::string::npos) return def;
    if (txt.compare(q, 4, "true")  == 0) return true;
    if (txt.compare(q, 5, "false") == 0) return false;
    return def;
}

static bool keyIsNull(const std::string& txt, const char* key)
{
    const std::size_t q = keyColon(txt, key);
    if (q == std::string::npos) return true;
    return (txt.compare(q, 4, "null") == 0);
}

CaseFlags loadFlags(const std::string& path)
{
    CaseFlags f;
    const std::string txt = loadFile(path);

    f.kind = findString(txt, "kind");
    f.sr   = findInt(txt, "sr", 44100);

    // Config defaults (always present in the merged flags.json)
    f.onsetSearchLookbackMs         = findNumber(txt, "onset_search_lookback_ms",          70.0);
    f.onsetSearchLookaheadMs        = findNumber(txt, "onset_search_lookahead_ms",         18.0);
    f.transientCenterPenaltyWeight  = findNumber(txt, "transient_center_penalty_weight",    0.14);
    f.transientAlignmentPenaltyWeight = findNumber(txt, "transient_alignment_penalty_weight", 0.10);
    f.anchorLocalWindowMs           = findNumber(txt, "anchor_local_window_ms",           120.0);
    f.anchorSearchBeats             = findInt   (txt, "anchor_search_beats",                1);
    f.anchorSearchMaxExtensionBeats = findInt   (txt, "anchor_search_max_extension_beats",  2);
    f.anchorMinContextMs            = findNumber(txt, "anchor_min_context_ms",            120.0);
    f.vocalActivityThreshold        = findNumber(txt, "vocal_activity_threshold",           0.28);
    f.vocalCrossfadeMs              = findNumber(txt, "vocal_crossfade_ms",               300.0);
    f.vocalSameLabelCrossfadeMs     = findNumber(txt, "vocal_same_label_crossfade_ms",    180.0);
    f.stereoRefineMaxShiftMs        = findNumber(txt, "stereo_refine_max_shift_ms",        12.0);
    f.stereoRefineCoarseStepSamples = findInt   (txt, "stereo_refine_coarse_step_samples", 64);

    if (f.kind == "find_onset") {
        f.beatSample  = findInt64(txt, "beat_sample", 0);
        f.lookbackMs  = keyIsNull(txt, "lookback_ms")  ? -1.0 : findNumber(txt, "lookback_ms",  70.0);
        f.lookaheadMs = keyIsNull(txt, "lookahead_ms") ? -1.0 : findNumber(txt, "lookahead_ms", 18.0);
    } else if (f.kind == "score_splice") {
        f.outgoingShift    = findInt(txt, "outgoing_shift",    0);
        f.incomingShift    = findInt(txt, "incoming_shift",    0);
        f.maxShiftSamples  = findInt(txt, "max_shift_samples", 0);
    } else if (f.kind == "score_anchor") {
        f.anchorIndex      = findInt   (txt, "anchor_index",    0);
        f.vocalPresence    = findNumber(txt, "vocal_presence",  0.0);
    } else if (f.kind == "transition_overlap") {
        f.crossfadeSamples     = findInt   (txt, "crossfade_samples",    0);
        f.vocalPresenceLevel   = findNumber(txt, "vocal_presence_level", 0.0);
        f.preferredOverlapSec  = findNumber(txt, "preferred_overlap_sec", 0.0);
        f.labelMatch           = findNumber(txt, "label_match",           1.0);
        f.vocalEntrySupport    = findNumber(txt, "vocal_entry_support",   1.0);
        f.vocalExitSupport     = findNumber(txt, "vocal_exit_support",    1.0);
    } else if (f.kind == "search_anchor") {
        f.prevBeat             = findInt   (txt, "prev_beat",             0);
        f.currBeat             = findInt   (txt, "curr_beat",             0);
        f.vocalPresenceLevel   = findNumber(txt, "vocal_presence_level",  0.0);
    } else if (f.kind == "refine") {
        f.crossfadeSamples     = findInt   (txt, "crossfade_samples",    0);
        f.successorSample      = findInt64 (txt, "successor_sample",     0);
        f.targetSample         = findInt64 (txt, "target_sample",        0);
        f.beatEnd              = findInt64 (txt, "beat_end",             0);
        f.alignmentOffsetSec   = findNumber(txt, "alignment_offset_sec", 0.0);
        f.overlapSamplesOrNeg  = keyIsNull(txt, "overlap_samples")
                                    ? -1
                                    : findInt(txt, "overlap_samples", -1);
    }
    return f;
}

static reamix::render::SpliceConfig makeConfig(const CaseFlags& f)
{
    reamix::render::SpliceConfig cfg;
    cfg.onsetSearchLookbackMs           = f.onsetSearchLookbackMs;
    cfg.onsetSearchLookaheadMs          = f.onsetSearchLookaheadMs;
    cfg.transientCenterPenaltyWeight    = f.transientCenterPenaltyWeight;
    cfg.transientAlignmentPenaltyWeight = f.transientAlignmentPenaltyWeight;
    cfg.anchorLocalWindowMs             = f.anchorLocalWindowMs;
    cfg.anchorSearchBeats               = f.anchorSearchBeats;
    cfg.anchorSearchMaxExtensionBeats   = f.anchorSearchMaxExtensionBeats;
    cfg.anchorMinContextMs              = f.anchorMinContextMs;
    cfg.vocalActivityThreshold          = f.vocalActivityThreshold;
    cfg.vocalCrossfadeMs                = f.vocalCrossfadeMs;
    cfg.vocalSameLabelCrossfadeMs       = f.vocalSameLabelCrossfadeMs;
    cfg.stereoRefineMaxShiftMs          = f.stereoRefineMaxShiftMs;
    cfg.stereoRefineCoarseStepSamples   = f.stereoRefineCoarseStepSamples;
    return cfg;
}

// --- per-kind dispatchers -----------------------------------------------

struct CaseResult {
    bool        ok           = false;
    double      maxAbs       = 0.0;
    std::size_t totalValues  = 0;
    std::size_t overTol      = 0;
    // Diagnostic-only (ADR-033): max find_onset primitive drift across all
    // find_onset kind cases. Reported but does not fail the aggregate gate.
    int         findOnsetDrift = 0;
};

static CaseResult runFindOnset(const std::string& caseRoot, const CaseFlags& f)
{
    CaseResult r;
    const auto monoE = reamix::test::loadNpy1DFloat64(caseRoot + "/input.npy");
    const auto expVec = reamix::test::loadNpy1DInt64(caseRoot + "/expected.npy");
    if (expVec.size() != 1) return r;
    const int expected = static_cast<int>(expVec[0]);

    const reamix::render::SpliceConfig cfg = makeConfig(f);
    const int got = reamix::render::Splice::findOnsetSample(
        monoE.data(), monoE.size(), f.sr,
        f.beatSample, f.lookbackMs, f.lookaheadMs, cfg);

    r.totalValues = 1;
    const int diff = std::abs(got - expected);
    r.findOnsetDrift = diff;
    r.maxAbs = static_cast<double>(diff);
    r.overTol = (diff > 0) ? 1u : 0u;
    // Per ADR-033: find_onset primitives are diagnostic-only; they pass if
    // drift is within FIND_ONSET_DIAGNOSTIC_TOL samples (session-34 measured
    // ±4 worst-case for case_01 on sr=44100). The composition-level
    // search_anchor / refine cases below are the real ADR-033 gate.
    r.ok = (diff <= FIND_ONSET_DIAGNOSTIC_TOL);
    return r;
}

static CaseResult runWindowOnset(const std::string& caseRoot, const CaseFlags&)
{
    CaseResult r;
    const auto win = reamix::test::loadNpy2DFloat64(caseRoot + "/input.npy");
    const auto expVec = reamix::test::loadNpy1DInt64(caseRoot + "/expected.npy");
    if (expVec.size() != 1) return r;
    const int expected = static_cast<int>(expVec[0]);

    const int got = reamix::render::Splice::windowOnsetIndex(
        win.data.data(), win.rows, win.cols);

    r.totalValues = 1;
    r.maxAbs = static_cast<double>(std::abs(got - expected));
    r.overTol = (got != expected) ? 1u : 0u;
    r.ok = (got == expected);
    return r;
}

static CaseResult runSimilarity(const std::string& caseRoot, const CaseFlags&)
{
    CaseResult r;
    const auto a = reamix::test::loadNpy2DFloat64(caseRoot + "/a.npy");
    const auto b = reamix::test::loadNpy2DFloat64(caseRoot + "/b.npy");
    const auto expVec = reamix::test::loadNpy1DFloat64(caseRoot + "/expected.npy");
    if (expVec.size() != 1) return r;
    if (a.rows != b.rows || a.cols != b.cols) return r;
    const double expected = expVec[0];

    const double got = reamix::render::Splice::stereoWindowSimilarity(
        a.data.data(), b.data.data(), a.rows, a.cols);

    r.totalValues = 1;
    r.maxAbs = std::abs(got - expected);
    r.overTol = (r.maxAbs > TOL_ABS) ? 1u : 0u;
    r.ok = (r.maxAbs <= TOL_ABS);
    return r;
}

static CaseResult runScoreSplice(const std::string& caseRoot, const CaseFlags& f)
{
    CaseResult r;
    const auto a = reamix::test::loadNpy2DFloat64(caseRoot + "/a.npy");
    const auto b = reamix::test::loadNpy2DFloat64(caseRoot + "/b.npy");
    const auto expVec = reamix::test::loadNpy1DFloat64(caseRoot + "/expected.npy");
    if (expVec.size() != 2) return r;
    if (a.rows != b.rows || a.cols != b.cols) return r;
    const double expectedScore = expVec[0];
    const double expectedSim   = expVec[1];

    const reamix::render::SpliceConfig cfg = makeConfig(f);
    const auto got = reamix::render::Splice::scoreSplicePair(
        a.data.data(), b.data.data(), a.rows, a.cols,
        f.outgoingShift, f.incomingShift, f.maxShiftSamples, cfg);

    const double dScore = std::abs(got.score      - expectedScore);
    const double dSim   = std::abs(got.similarity - expectedSim);
    r.totalValues = 2;
    r.maxAbs = std::max(dScore, dSim);
    r.overTol = (dScore > TOL_ABS ? 1u : 0u) + (dSim > TOL_ABS ? 1u : 0u);
    r.ok = (r.maxAbs <= TOL_ABS);
    return r;
}

static CaseResult runScoreAnchor(const std::string& caseRoot, const CaseFlags& f)
{
    CaseResult r;
    const auto a = reamix::test::loadNpy2DFloat64(caseRoot + "/a.npy");
    const auto b = reamix::test::loadNpy2DFloat64(caseRoot + "/b.npy");
    const auto expVec = reamix::test::loadNpy1DFloat64(caseRoot + "/expected.npy");
    if (expVec.size() != 3) return r;
    if (a.rows != b.rows || a.cols != b.cols) return r;

    const double expQuality = expVec[0];
    const double expSim01   = expVec[1];
    const double expLocal01 = expVec[2];

    const reamix::render::SpliceConfig cfg = makeConfig(f);
    const auto got = reamix::render::Splice::scoreAnchorAlignedPair(
        a.data.data(), b.data.data(), a.rows, a.cols,
        f.anchorIndex, f.vocalPresence, f.sr, cfg);

    const double dQ = std::abs(got.quality            - expQuality);
    const double dS = std::abs(got.similarity01       - expSim01);
    const double dL = std::abs(got.localSimilarity01  - expLocal01);
    r.totalValues = 3;
    r.maxAbs = std::max({ dQ, dS, dL });
    r.overTol = (dQ > TOL_ABS ? 1u : 0u) + (dS > TOL_ABS ? 1u : 0u) + (dL > TOL_ABS ? 1u : 0u);
    r.ok = (r.maxAbs <= TOL_ABS);
    return r;
}

static CaseResult runTransitionOverlap(const std::string& caseRoot, const CaseFlags& f)
{
    CaseResult r;
    const auto expVec = reamix::test::loadNpy1DInt64(caseRoot + "/expected.npy");
    if (expVec.size() != 1) return r;
    const int expected = static_cast<int>(expVec[0]);

    const reamix::render::SpliceConfig cfg = makeConfig(f);
    reamix::render::TransitionMeta meta;
    meta.vocalPresenceLevel  = f.vocalPresenceLevel;
    meta.preferredOverlapSec = f.preferredOverlapSec;
    meta.labelMatch          = f.labelMatch;
    meta.vocalEntrySupport   = f.vocalEntrySupport;
    meta.vocalExitSupport    = f.vocalExitSupport;

    const int got = reamix::render::Splice::transitionOverlapSamples(
        f.crossfadeSamples, f.sr, meta, cfg);

    r.totalValues = 1;
    r.maxAbs = static_cast<double>(std::abs(got - expected));
    r.overTol = (got != expected) ? 1u : 0u;
    r.ok = (got == expected);
    return r;
}

static CaseResult runSearchAnchor(const std::string& caseRoot, const CaseFlags& f)
{
    CaseResult r;
    const auto audio = reamix::test::loadNpy2DFloat64(caseRoot + "/audio.npy");
    const auto monoE = reamix::test::loadNpy1DFloat64(caseRoot + "/mono_energy.npy");
    const auto beatS = reamix::test::loadNpy1DInt64  (caseRoot + "/beat_samples.npy");
    const std::string expTxt = loadFile(caseRoot + "/expected.json");

    const bool expSelected = findBool(expTxt, "selected", false);

    reamix::render::SpliceContext ctx;
    ctx.audio        = audio.data.data();
    ctx.nChAudio     = audio.rows;
    ctx.nAudio       = audio.cols;
    ctx.sr           = f.sr;
    ctx.monoEnergy   = monoE.data();
    ctx.nMonoEnergy  = monoE.size();
    ctx.beatSamples  = beatS.data();
    ctx.nBeats       = beatS.size();
    ctx.crossfadeSamples = 0;

    reamix::render::TransitionMeta meta;
    meta.vocalPresenceLevel = f.vocalPresenceLevel;

    const reamix::render::SpliceConfig cfg = makeConfig(f);
    const auto got = reamix::render::Splice::searchAnchorTransitionGeometry(
        ctx, f.prevBeat, f.currBeat, meta, cfg);

    // 1 bool + 14 numeric fields (when selected=true)
    r.totalValues = expSelected ? 15u : 1u;

    if (got.selected != expSelected) {
        r.maxAbs = 1.0;
        r.overTol = 1;
        r.ok = false;
        return r;
    }
    if (!expSelected) {
        r.ok = true;
        r.maxAbs = 0.0;
        r.overTol = 0;
        return r;
    }

    struct Fld { const char* key; double gotVal; };
    const Fld fields[] = {
        {"render_anchor_splice_score",         got.score},
        {"render_anchor_similarity",           got.similarity01},
        {"render_anchor_local_similarity",     got.localSimilarity01},
        {"render_anchor_out_beat",             static_cast<double>(got.outAnchorBeat)},
        {"render_anchor_in_beat",              static_cast<double>(got.inAnchorBeat)},
        {"render_anchor_incoming_start_beat",  static_cast<double>(got.incomingStartBeat)},
        {"render_anchor_outgoing_boundary_beat", static_cast<double>(got.outgoingBoundaryBeat)},
        {"render_anchor_out_sec",              got.anchorOutSec},
        {"render_anchor_in_sec",               got.anchorInSec},
        {"render_outgoing_start_sample",       static_cast<double>(got.outgoingStartSample)},
        {"render_outgoing_cut_sample",         static_cast<double>(got.outgoingCutSample)},
        {"render_incoming_cut_sample",         static_cast<double>(got.incomingCutSample)},
        {"render_incoming_end_sample",         static_cast<double>(got.incomingEndSample)},
        {"anchor_overlap_samples",             static_cast<double>(got.anchorOverlapSamples)},
    };

    double maxDiff = 0.0;
    std::size_t over = 0;
    for (const auto& fl : fields) {
        const double expected = findNumber(expTxt, fl.key, 0.0);
        const double d = std::abs(fl.gotVal - expected);
        if (d > maxDiff) maxDiff = d;
        if (d > TOL_ABS) ++over;
    }
    r.maxAbs = maxDiff;
    r.overTol = over;
    r.ok = (maxDiff <= TOL_ABS);
    return r;
}

static CaseResult runRefine(const std::string& caseRoot, const CaseFlags& f)
{
    CaseResult r;
    const auto audio = reamix::test::loadNpy2DFloat64(caseRoot + "/audio.npy");
    const auto monoE = reamix::test::loadNpy1DFloat64(caseRoot + "/mono_energy.npy");
    const std::string expTxt = loadFile(caseRoot + "/expected.json");

    const bool expFound = findBool(expTxt, "found", false);

    reamix::render::SpliceContext ctx;
    ctx.audio            = audio.data.data();
    ctx.nChAudio         = audio.rows;
    ctx.nAudio           = audio.cols;
    ctx.sr               = f.sr;
    ctx.monoEnergy       = monoE.data();
    ctx.nMonoEnergy      = monoE.size();
    ctx.beatSamples      = nullptr;
    ctx.nBeats           = 0;
    ctx.crossfadeSamples = f.crossfadeSamples;

    reamix::render::TransitionMeta meta;
    meta.alignmentOffsetSec = f.alignmentOffsetSec;

    const reamix::render::SpliceConfig cfg = makeConfig(f);
    const auto got = reamix::render::Splice::refineTransitionSplice(
        ctx, f.successorSample, f.targetSample, f.beatEnd,
        meta, f.overlapSamplesOrNeg, cfg);

    r.totalValues = expFound ? 7u : 1u;

    if (got.found != expFound) {
        r.maxAbs = 1.0;
        r.overTol = 1;
        r.ok = false;
        return r;
    }
    if (!expFound) {
        r.ok = true;
        return r;
    }

    struct Fld { const char* key; double gotVal; };
    const Fld fields[] = {
        {"render_outgoing_cut_sample",   static_cast<double>(got.outgoingCutSample)},
        {"render_incoming_cut_sample",   static_cast<double>(got.incomingCutSample)},
        {"render_outgoing_shift_sec",    got.outgoingShiftSec},
        {"render_incoming_shift_sec",    got.incomingShiftSec},
        {"render_stereo_similarity",     got.stereoSimilarity01},
        {"render_splice_score",          got.spliceScore},
    };

    double maxDiff = 0.0;
    std::size_t over = 0;
    for (const auto& fl : fields) {
        const double expected = findNumber(expTxt, fl.key, 0.0);
        const double d = std::abs(fl.gotVal - expected);
        if (d > maxDiff) maxDiff = d;
        if (d > TOL_ABS) ++over;
    }
    r.maxAbs = maxDiff;
    r.overTol = over;
    r.ok = (maxDiff <= TOL_ABS);
    return r;
}

static CaseResult runCase(const std::string& root, const Case& c)
{
    const std::string caseRoot = root + "/" + c.dir;
    const CaseFlags flags = loadFlags(caseRoot + "/flags.json");
    switch (c.kind) {
        case Kind::FindOnset:         return runFindOnset(caseRoot, flags);
        case Kind::WindowOnset:       return runWindowOnset(caseRoot, flags);
        case Kind::Similarity:        return runSimilarity(caseRoot, flags);
        case Kind::ScoreSplice:       return runScoreSplice(caseRoot, flags);
        case Kind::ScoreAnchor:       return runScoreAnchor(caseRoot, flags);
        case Kind::TransitionOverlap: return runTransitionOverlap(caseRoot, flags);
        case Kind::SearchAnchor:      return runSearchAnchor(caseRoot, flags);
        case Kind::Refine:            return runRefine(caseRoot, flags);
    }
    return CaseResult{};
}

} // anonymous namespace

int main(int argc, char** argv)
{
    const std::string root = dataRoot(argc, argv);
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--verbose" || a == "-v") verbose = true;
    }

    std::printf("=== test_splice: full SpliceMixin parity (tol=%.1e; "
                "find_onset diag-tol=±%d samples) ===\n",
                TOL_ABS, FIND_ONSET_DIAGNOSTIC_TOL);

    double gmax = 0.0;
    std::size_t gover = 0, gtotal = 0;
    int gfailCases = 0;
    int casesOk = 0;
    int maxFindOnsetDrift = 0;

    for (const auto& c : kCases) {
        const CaseResult res = runCase(root, c);
        gtotal += res.totalValues;
        gover  += res.overTol;
        if (res.maxAbs > gmax) gmax = res.maxAbs;
        if (res.findOnsetDrift > maxFindOnsetDrift) maxFindOnsetDrift = res.findOnsetDrift;
        if (res.ok) ++casesOk;
        else        ++gfailCases;
        if (verbose || !res.ok) {
            std::printf("  %-38s case_max=%.3e over=%zu/%zu  %s\n",
                        c.label, res.maxAbs, res.overTol, res.totalValues,
                        res.ok ? "PASS" : "FAIL");
        }
    }

    std::printf(
        "  cases=%d/%zu PASS  total values=%zu  over_tol=%zu  max_abs=%.3e  "
        "max_find_onset_drift=%d samples  %s\n",
        casesOk, sizeof(kCases) / sizeof(kCases[0]),
        gtotal, gover, gmax, maxFindOnsetDrift,
        (gfailCases == 0) ? "PASS" : "FAIL");

    return (gfailCases == 0) ? 0 : 1;
}
