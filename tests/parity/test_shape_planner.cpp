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
    // Sesja 135: the chorus start is a zone (48 .. 51) - score all of it 0.60.
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j >= 48 && j < 52) return ShapeSeamScore { 0.60, 1.0, 0.5 };
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
    // Sesja 135: the section start is a zone (48 .. 51) - gate all of it.
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j >= 48 && j < 52) return std::nullopt;
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
    // Sesja 135: intro + the outro from the last beat of its start zone (115)
    // = 8 + 6.5 s is the closest complete plan (was intro + whole outro, 16 s).
    const ShapePlan p = planShape (f.inputs (3.0));
    CHECK (! p.ok && p.best_effort && p.tier == 'F', "best effort outside every window");
    CHECK (std::fabs (p.est_sec - 14.5) < 1e-9, "closest length: intro + outro from beat 115");
    CHECK (p.seams.size() == 1 && p.seams[0].i == 15 && p.seams[0].j == 115, "seam intro end -> outro zone");
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
// Sesja 134 (ADR-117 step 3, DEV-121).
bool test10_bar_ends()
{
    Fixture f;
    // Only the intro -> outro seam exists, at bar granularity: 15 -> 112 is
    // the whole-section seam (16 s), the bar-end seams are shorter pieces.
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (j >= 112 && i < 16) return ShapeSeamScore { 0.70, 1.0, 0.5 };
        return std::nullopt;
    };
    ShapePlannerInputs in = f.inputs (12.0);
    in.window_sec = 1.0; in.window_relaxed_sec = 1.0;   // only 12 s fits (10 / 14 s cost the same: one trim tax)
    CHECK (! planShape (in).ok, "whole intro + outro = 16 s misses a 1 s window; phrase trims cannot help");
    in.bar_ends = true;
    const ShapePlan p = planShape (in);
    CHECK (p.ok && p.tier == 'B', "bar-granular ends make the length");
    CHECK (p.pieces.size() == 2, "intro head + outro tail");
    CHECK (p.pieces[0].b0 == 0 && p.pieces[0].trim == ShapePiece::Trim::Head, "first piece from beat 0, cut at a bar");
    // 4 s of intro + the whole 8 s outro is the cheapest 12 s (one trim tax);
    // a bar-granular outro tail would cost a second trim tax.
    CHECK (p.pieces[1].b1 == kBeats, "last piece to the tail");
    CHECK (p.pieces[0].b1 % 4 == 0 && p.pieces[0].b1 < 16 && p.pieces[1].b0 % 4 == 0, "bar boundaries, intro shorter than the section");
    CHECK (std::fabs (p.est_sec - 12.0) < 1e-9, "6 + 6 s");
    return true;
}

bool test11_open_seam_no_floor()
{
    Fixture f;
    f.judge = [] (int, int) -> std::optional<ShapeSeamScore> { return std::nullopt; };   // strict: everything gated
    ShapePlannerInputs in = f.inputs (16.0);
    CHECK (! planShape (in).ok, "strict judge starves the plan");
    // The open judge accepts every seam at q 0.20 - below min_q.
    in.seam_open = [] (int, int) -> std::optional<ShapeSeamScore> { return ShapeSeamScore { 0.20, 9.0, 1.5 }; };
    const ShapePlan p = planShape (in);
    CHECK (p.ok && p.tier == 'A', "an open seam (landing on the outro start) needs no q floor");
    CHECK (p.seams.size() == 1 && p.seams[0].i == 15 && p.seams[0].j == 112 && p.seams[0].open, "intro end -> outro start, open");
    CHECK (std::fabs (p.min_q - 0.20) < 1e-12, "plan min_q reports the open seam");
    // A landing inside a middle section (chorus phrase start 80) is not open:
    // with the strict judge gated it stays unreachable.
    in.target_sec = 32.0;
    const ShapePlan q = planShape (in);
    CHECK (! q.ok || q.seams.empty() || q.seams[0].j != 80, "the chorus phrase start is not an open landing");
    // Seam crossfade metadata: one beat at the landing.
    in.target_sec = 16.0; in.seam_crossfade_beats = 1.0;
    const ShapePlan r = planShape (in);
    CHECK (r.ok && std::fabs (r.seams[0].overlap_sec - kPeriod) < 1e-12, "one beat overlap");
    const RemixPath path = r.toPath();
    const auto md = path.transition_metadata.at ({ 15, 112 });
    CHECK (std::fabs (md.at ("preferred_overlap_sec") - kPeriod) < 1e-12 && md.at ("open_seam") == 1.0, "metadata carries the overlap + open flag");
    return true;
}

// Sesja 135 (DEV-121): an open landing INSIDE the last section keeps the q
// floor; only a section-start landing is floor-free.
bool test12_inside_last_section_floor()
{
    Fixture f;
    f.judge = [] (int, int) -> std::optional<ShapeSeamScore> { return std::nullopt; };   // strict: everything gated
    ShapePlannerInputs in = f.inputs (14.0);
    in.bar_ends = true;
    in.window_sec = 0.25; in.window_relaxed_sec = 0.25;   // sesja 135: excludes the 14.5 s zone plan (outro from 115)
    // Open judge: q 0.20 everywhere (below min_q 0.45). 14 s = the whole
    // intro (8 s) + an outro tail from beat 116 (6 s) - an inside-the-outro
    // landing - or an intro head to beat 12 (6 s) + the whole outro (8 s),
    // landing on the outro start. Both cost one trim tax; the inside landing
    // is below the floor, so only the section-start plan may be picked.
    in.seam_open = [] (int, int) -> std::optional<ShapeSeamScore> { return ShapeSeamScore { 0.20, 9.0, 1.5 }; };
    const ShapePlan p = planShape (in);
    CHECK (p.ok, "plan expected");
    CHECK (p.seams.size() == 1 && p.seams[0].j == 112 && p.seams[0].open, "the open seam lands on the outro START");
    CHECK (p.pieces.size() == 2 && p.pieces[0].b1 == 12 && p.pieces[1].b0 == 112, "intro head to beat 12 + whole outro");
    // With the inside landing at q 0.60 (above the floor) it becomes eligible:
    // the whole intro (no trim tax) + outro tail from 116 (one trim tax) costs
    // 0.40 + 0.10 + 0.15 = 0.65 against 0.80 + 0.10 + 0.15 = 1.05 -> it wins.
    in.seam_open = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        return ShapeSeamScore { (i == 15 && j == 116) ? 0.60 : 0.20, 9.0, 1.5 };
    };
    const ShapePlan q = planShape (in);
    CHECK (q.ok && q.seams.size() == 1 && q.seams[0].i == 15 && q.seams[0].j == 116, "an inside landing at or above the floor is allowed");
    CHECK (std::fabs (q.min_q - 0.60) < 1e-12, "plan min_q tracks the inside landing");
    // Sesja 135 (DEV-122): a landing on a lattice beat is floor-free. With
    // beats 116 .. 127 flagged synthetic, the q 0.20 landing at 116 is allowed
    // again (the whole intro + the outro from 116 = 14 s, one trim tax = the
    // same cost as the intro-head plan; the earlier piece index wins the tie).
    in.seam_open = [] (int, int) -> std::optional<ShapeSeamScore> { return ShapeSeamScore { 0.20, 9.0, 1.5 }; };
    std::vector<bool> synth ((std::size_t) kBeats, false);
    for (int b = 116; b < kBeats; ++b) synth[(std::size_t) b] = true;
    in.beat_is_synthetic = &synth;
    // (With the chorus tax in maximin units the planner keeps a 2-bar chorus
    // head: intro head + chorus head + outro from 116, every seam open.)
    const ShapePlan r = planShape (in);
    CHECK (r.ok && ! r.seams.empty(), "plan with the lattice landing allowed");
    for (const auto& sm : r.seams)
        CHECK (sm.open && (sm.j == 48 || sm.j == 112 || sm.j >= 116), "every landing is a section start or a lattice beat, never a floored one");
    CHECK (std::fabs (r.est_sec - 14.0) < 1e-9, "14 s");
    return true;
}

// Sesja 135 (round-1 verdict): the section start is a zone - a landing 1 ..
// bar_beats-1 beats after the start is offered in every tier as a taxed head
// trim, open and floor-free; the judge's preference decides.
bool test13_section_start_zone()
{
    Fixture f;
    f.judge = [] (int, int) -> std::optional<ShapeSeamScore> { return std::nullopt; };   // strict: gated
    ShapePlannerInputs in = f.inputs (40.0);
    in.bar_beats = 4;
    // Open judge: intro end -> chorus start scores 0.20, two beats later 0.70
    // (the chord ringing over the bar line has faded); outro landings 0.70.
    in.seam_open = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j == 48) return ShapeSeamScore { 0.20, 9.0, 1.5 };
        if (i == 15 && j == 50) return ShapeSeamScore { 0.70, 2.0, 0.5 };
        if (j == 112)           return ShapeSeamScore { 0.70, 1.0, 0.5 };
        return std::nullopt;
    };
    const ShapePlan p = planShape (in);
    CHECK (p.ok && p.tier == 'A', "a zone landing is available in tier A");
    CHECK (p.seams.size() == 1 && p.seams[0].i == 15 && p.seams[0].j == 50 && p.seams[0].open, "lands two beats into the chorus (0.30 + tax 0.15 beats 0.80)");
    CHECK (p.pieces.size() == 3 && p.pieces[1].b0 == 50 && p.pieces[1].b1 == 112 && p.pieces[1].trim == ShapePiece::Trim::Head, "chorus from beat 50 to its end");
    CHECK (std::fabs (p.est_sec - 47.0) < 1e-9, "8 + 31 + 8 s");
    // A landing a whole bar in (52) is not part of the zone: never offered here.
    in.seam_open = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 15 && j == 52) return ShapeSeamScore { 0.90, 1.0, 0.5 };
        if (j == 112)           return ShapeSeamScore { 0.70, 1.0, 0.5 };
        return std::nullopt;
    };
    const ShapePlan r = planShape (in);
    CHECK (! r.ok || r.seams.empty() || r.seams[0].j != 52, "beat 52 (a full bar in) is outside the zone: never landed on");
    return true;
}

// Sesja 135: bar-granular ends for every section in the trim tiers.
bool test14_bar_ends_everywhere()
{
    Fixture f;
    f.judge = [] (int, int) -> std::optional<ShapeSeamScore> { return std::nullopt; };
    ShapePlannerInputs in = f.inputs (20.0);
    in.window_sec = 1.0; in.window_relaxed_sec = 1.0;
    in.seam_open = [] (int, int) -> std::optional<ShapeSeamScore> { return ShapeSeamScore { 0.70, 1.0, 0.5 }; };
    in.seam = [] (int, int) -> std::optional<ShapeSeamScore> { return ShapeSeamScore { 0.70, 1.0, 0.5 }; };
    in.bar_ends = false;
    CHECK (! planShape (in).ok, "20 s needs a 2-bar head of a middle section");
    in.bar_ends = true;
    const ShapePlan p = planShape (in);
    CHECK (p.ok && p.tier == 'B', "bar-granular head of a middle section makes the length");
    bool middle_bar_piece = false;
    for (const auto& pc : p.pieces)
        if ((pc.section == 1 || pc.section == 2) && pc.trim != ShapePiece::Trim::Whole && (pc.b1 - pc.b0) % 4 == 0 && pc.b1 - pc.b0 < 32)
            middle_bar_piece = true;
    CHECK (middle_bar_piece, "a bar-granular piece of a middle section is in the plan");
    CHECK (std::fabs (p.est_sec - 20.0) <= 1.0 + 1e-9, "length within the 1 s window");
    return true;
}

// Sesja 135 (round-1 verdict): the worst seam decides. Two seams at q 0.60
// beat one seam at q 0.45 although they cost more (1.0 vs 0.65).
bool test15_maximin()
{
    Fixture f;
    // intro, A, B, C, outro - no chorus, 96 beats.
    f.secs = { { 0, 16, 0 }, { 16, 40, 1 }, { 40, 64, 1 }, { 64, 80, 1 }, { 80, 96, 11 } };
    f.judge = [] (int i, int j) -> std::optional<ShapeSeamScore> {
        if (i == 39 && j == 80) return ShapeSeamScore { 0.45, 1.0, 0.5 };   // intro + A + outro: one seam
        if (i == 15 && j == 40) return ShapeSeamScore { 0.60, 1.0, 0.5 };   // intro + B + outro: two seams
        if (i == 63 && j == 80) return ShapeSeamScore { 0.60, 1.0, 0.5 };
        return std::nullopt;
    };
    // Target 30 +-5: intro + A + outro and intro + B + outro (28 s each) fit,
    // the seamless intro + A + B + outro (40 s) does not.
    ShapePlannerInputs in = f.inputs (30.0);
    in.n_beats = 96; in.track_sec = 48.0;
    in.window_sec = 5.0; in.window_relaxed_sec = 5.0;
    const ShapePlan p = planShape (in);
    CHECK (p.ok, "plan expected");
    CHECK (p.seams.size() == 2 && p.seams[0].j == 40 && p.seams[1].j == 80, "the plan with the better worst seam wins over the cheaper one");
    CHECK (std::fabs (p.min_q - 0.60) < 1e-12 && p.pieces[1].b0 == 40, "intro + B + outro");
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
    run ("10 bar-granular ends",               test10_bar_ends);
    run ("11 open seam, no floor, one beat",   test11_open_seam_no_floor);
    run ("12 inside-last-section floor",       test12_inside_last_section_floor);
    run ("13 section-start zone",              test13_section_start_zone);
    run ("14 bar-granular ends everywhere",    test14_bar_ends_everywhere);
    run ("15 maximin: the worst seam decides", test15_maximin);
    std::printf (failed == 0 ? "ALL PASS\n" : "%d FAILED\n", failed);
    return failed == 0 ? 0 : 1;
}
