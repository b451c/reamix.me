// test_shape_planner — sesja 131 (ADR-117) self-validation.
//
// The shape-first planner (src/remix/ShapePlanner.h): whole sections of the
// section map in the original order, seams only at section boundaries judged
// by a boundary seam judge, phrase trims when whole sections cannot make the
// length. No Python ground truth (C++-canonical per ADR-065); the perceptual
// gate is the blinded round on the sesja-130 cases.
//
// Fixture: 128 beats of 4/4 at 120 BPM (64 s). Sections: intro bars 0-3
// (beats 0-16, 8 s), verse bars 4-11 (16-48, 16 s), chorus bars 12-27
// (48-112, 32 s, one internal phrase start at bar 20 = beat 80), outro bars
// 28-31 (112-128, 8 s). The seam judge is a table the test controls.
//
// Checks:
//   1. tier A, chorus kept: target 40 s -> intro + chorus + outro (48 s, +8),
//      one seam 15 -> 48, chorus -> outro adjacent (no seam), over the equally
//      long intro + verse + outro (the no-chorus tax).
//   2. a gated seam is never used: with 15 -> 48 rejected the plan is
//      intro + verse + outro (seam 47 -> 112).
//   3. tier B trims: only 15 -> 80 allowed -> intro + chorus/tail + outro.
//   4. no plan when the sections cannot make the window (target 3 s).
//   5. toPath: consecutive beats inside pieces, one transition per seam with
//      family 1 + quality_score metadata; the path starts at 0 and ends at n-1.
//   6. dynamics tie-break: equal seams, the plan whose seam arrives quieter
//      than the song did (positive excess) loses.
//   7. shapeSectionsFromSeconds: contiguous cover of [0, n), first at 0, last at n.

#include "remix/ShapePlanner.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <set>
#include <vector>

namespace
{
#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) { std::printf ("FAIL: %s — %s\n", #expr, msg); return false; }\
} while (0)

using namespace reamix::remix;

constexpr int    kBeats  = 128;
constexpr double kPeriod = 0.5;
constexpr double kTrack  = kBeats * kPeriod;   // 64 s

struct Fixture
{
    std::vector<double>        bt;
    std::set<int>              db;
    std::vector<ShapeSection>  secs;
    std::vector<double>        rms;
    std::function<std::optional<ShapeSeamScore>(int, int)> judge;

    Fixture()
    {
        for (int b = 0; b < kBeats; ++b) bt.push_back (b * kPeriod);
        for (int b = 0; b < kBeats; b += 4) db.insert (b);
        secs = { { 0, 16, 0 }, { 16, 48, 1 }, { 48, 112, 3 }, { 112, 128, 11 } };
        rms.assign ((std::size_t) kBeats, 0.8);
        judge = [] (int, int) { return ShapeSeamScore { 0.70, 1.0, 0.5 }; };
    }

    ShapePlannerInputs inputs (double target) const
    {
        ShapePlannerInputs in;
        in.beat_times = bt.data();  in.n_beats = kBeats;  in.track_sec = kTrack;
        in.sections   = secs.data(); in.n_sections = (int) secs.size();
        in.db_set     = &db;         in.rms_energy = rms.data();
        in.target_sec = target;
        in.seam       = judge;
        return in;
    }
};

bool sameKinds (const ShapePlan& p, std::vector<int> kinds)
{
    if (p.pieces.size() != kinds.size()) return false;
    for (std::size_t k = 0; k < kinds.size(); ++k) if (p.pieces[k].kind != kinds[k]) return false;
    return true;
}

bool test1_chorus_kept()
{
    Fixture f;
    // The chorus seam scores LOWER than the verse seam: the raw-cheaper plan
    // without a chorus must not shadow the chorus plan inside the DP (the
    // no-chorus tax is applied at pick time; the chorus flag is DP state).
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j == 48) return ShapeSeamScore { 0.60, 1.0, 0.5 };
        return ShapeSeamScore { 0.70, 1.0, 0.5 };
    };
    const ShapePlan p = planShape (f.inputs (40.0));
    CHECK (p.ok, "plan expected");
    CHECK (p.tier == 'A', "whole sections suffice");
    CHECK (sameKinds (p, { 0, 3, 11 }), "intro + chorus + outro");
    CHECK (p.seams.size() == 1 && p.seams[0].i == 15 && p.seams[0].j == 48, "one seam intro end -> chorus start");
    CHECK (std::fabs (p.est_sec - 48.0) < 1e-9, "8 + 32 + 8 = 48 s");
    CHECK (std::fabs (p.dev_sec - 8.0) < 1e-9, "dev +8 inside the window");
    return true;
}

bool test2_gated_seam()
{
    Fixture f;
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j == 48) return std::nullopt;
        return ShapeSeamScore { 0.70, 1.0, 0.5 };
    };
    const ShapePlan p = planShape (f.inputs (40.0));
    CHECK (p.ok, "plan expected");
    CHECK (sameKinds (p, { 0, 1, 11 }), "intro + verse + outro when intro -> chorus is gated");
    CHECK (p.seams.size() == 1 && p.seams[0].i == 47 && p.seams[0].j == 112, "seam verse end -> outro start");
    return true;
}

bool test3_trim_tier()
{
    Fixture f;
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j == 80) return ShapeSeamScore { 0.65, 1.0, 0.5 };
        return std::nullopt;
    };
    const ShapePlan p = planShape (f.inputs (32.0));
    CHECK (p.ok, "plan expected");
    CHECK (p.tier == 'B', "trims needed");
    CHECK (p.pieces.size() == 3, "intro + chorus tail + outro");
    CHECK (p.pieces[1].trim == ShapePiece::Trim::Tail && p.pieces[1].b0 == 80 && p.pieces[1].b1 == 112, "chorus tail from its phrase start");
    CHECK (std::fabs (p.est_sec - 32.0) < 1e-9, "8 + 16 + 8");
    return true;
}

bool test4_no_plan()
{
    Fixture f;
    const ShapePlan p = planShape (f.inputs (3.0));
    CHECK (! p.ok, "intro + outro = 16 s cannot reach 3 s within 10 s");
    return true;
}

bool test5_to_path()
{
    Fixture f;
    const ShapePlan p = planShape (f.inputs (40.0));
    CHECK (p.ok, "plan expected");
    const RemixPath path = p.toPath();
    CHECK (path.beat_indices.front() == 0 && path.beat_indices.back() == kBeats - 1, "head and tail reachable");
    CHECK (path.beat_indices.size() == 16 + 64 + 16, "beat count = kept beats");
    int jumps = 0;
    for (std::size_t k = 1; k < path.beat_indices.size(); ++k)
        if (path.beat_indices[k] != path.beat_indices[k - 1] + 1) ++jumps;
    CHECK (jumps == (int) path.transitions.size() && jumps == 1, "one non-sequential step = one transition");
    const auto md = path.transition_metadata.at ({ 15, 48 });
    CHECK (md.at ("family") == 1.0 && std::fabs (md.at ("quality_score") - 0.70) < 1e-12, "boundary metadata");
    return true;
}

bool test6_dynamics_tiebreak()
{
    Fixture f;
    // intro, verse A, verse B, outro — no chorus; 96 beats.
    f.secs = { { 0, 16, 0 }, { 16, 48, 1 }, { 48, 80, 1 }, { 80, 96, 11 } };
    for (int b = 0; b < 16; ++b) f.rms[(std::size_t) b] = 0.3;   // quiet intro
    ShapePlannerInputs in = f.inputs (32.0);
    in.n_beats = 96; in.track_sec = 48.0;
    const ShapePlan p = planShape (in);
    CHECK (p.ok, "plan expected");
    // intro + verse B + outro would seam 15 -> 48 with the loud verse A as the
    // song's own predecessor of 48 (excess +8.5 dB); intro + verse A + outro
    // seams 47 -> 80 with no excess.
    CHECK (p.seams.size() == 1 && p.seams[0].i == 47 && p.seams[0].j == 80, "the seam without a loudness excess wins");
    CHECK (p.pieces[1].b0 == 16, "verse A kept");
    return true;
}

bool test8_relaxed_tier()
{
    Fixture f;
    f.judge = [] (int, int) -> std::optional<ShapeSeamScore> { return std::nullopt; };   // strict: everything gated
    ShapePlannerInputs in = f.inputs (40.0);
    in.seam_relaxed = [] (int, int) -> std::optional<ShapeSeamScore> { return ShapeSeamScore { 0.60, 1.0, 0.5 }; };
    const ShapePlan p = planShape (in);
    CHECK (p.ok && p.tier == 'D', "relaxed judge tier when the strict judge starves the plan");
    CHECK (sameKinds (p, { 0, 3, 11 }), "same shape as tier A");
    in.seam_relaxed = nullptr;
    CHECK (! planShape (in).ok, "no relaxed judge = no plan");
    return true;
}

bool test9_best_effort()
{
    Fixture f;
    const ShapePlan p = planShape (f.inputs (3.0));   // intro + outro = 16 s is the closest complete plan
    CHECK (! p.ok && p.best_effort && p.tier == 'F', "best effort outside every window");
    CHECK (std::fabs (p.est_sec - 16.0) < 1e-9, "closest length: intro + outro");
    CHECK (p.seams.size() == 1 && p.seams[0].i == 15 && p.seams[0].j == 112, "seam intro end -> outro start");
    return true;
}

bool test7_sections_from_seconds()
{
    Fixture f;
    const std::vector<double> starts { 0.0, 8.05, 24.0, 55.9 };
    const std::vector<double> ends   { 8.05, 24.0, 55.9, 64.0 };
    const std::vector<int>    kinds  { 0, 1, 3, 11 };
    const auto secs = shapeSectionsFromSeconds (f.bt.data(), kBeats, starts.data(), ends.data(), kinds.data(), 4);
    CHECK (secs.size() == 4, "four sections");
    CHECK (secs[0].b0 == 0 && secs[3].b1 == kBeats, "covers the track");
    for (std::size_t s = 1; s < secs.size(); ++s) CHECK (secs[s].b0 == secs[s - 1].b1, "contiguous");
    CHECK (secs[1].b0 == 17, "8.05 s -> first beat at/after = beat 17 (8.5 s)");
    CHECK (secs[3].b0 == 112, "55.9 s -> beat 112 (56.0 s)");
    return true;
}
} // namespace

int main()
{
    int failed = 0;
    auto run = [&] (const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::printf ("%s %s\n", ok ? "PASS" : "FAIL", name);
        if (! ok) ++failed;
    };
    run ("1 tier A keeps the chorus",          test1_chorus_kept);
    run ("2 gated seam never used",            test2_gated_seam);
    run ("3 tier B phrase trims",              test3_trim_tier);
    run ("4 no plan outside the window",       test4_no_plan);
    run ("5 toPath transitions + metadata",    test5_to_path);
    run ("6 dynamics excess tie-break",        test6_dynamics_tiebreak);
    run ("7 sections from seconds",            test7_sections_from_seconds);
    run ("8 relaxed judge tier D",             test8_relaxed_tier);
    run ("9 best effort tier F",               test9_best_effort);
    std::printf (failed == 0 ? "ALL PASS\n" : "%d FAILED\n", failed);
    return failed == 0 ? 0 : 1;
}
