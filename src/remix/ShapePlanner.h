// ShapePlanner — sesja 131 (ADR-117): shape-first engine for extreme
// shortening (target below kShapePlannerMaxRatio of the track).
//
// The user decision that closed sesja 130: for targets far below the track
// length the fault the ear reports is the FRAGMENT (which sections survive,
// a quiet passage jumping into a loud one, a vocal phrase cut in half), not
// the seam. So the plan is built from WHOLE SECTIONS of the grid-snapped
// section map, in the original order (shortening never repeats), starting
// with the song's first section (the renderer keeps the file head) and ending
// with its last (the renderer appends the tail). Every seam leaves a section
// at its end and lands on the next kept section's start - by construction a
// boundary cut (BoundaryFamily.h: phrase end -> phrase start) - and is judged
// by the boundary seam judge the caller supplies (SeamJudge.h). When whole
// sections cannot make the length, a section may be trimmed at its internal
// phrase starts (8 grid bars): its head (start .. phrase start) or its tail
// (phrase start .. end), so the seam stays a boundary cut.
//
// Tiers (the first that yields a plan wins):
//   A  whole sections only, length within `window_sec` (+-8 s, user spec)
//   B  phrase trims allowed, same window
//   C  phrase trims allowed, `window_relaxed_sec` (10 s, the user's cap)
//   D  as B with the relaxed judge (`seam_relaxed`: no per-track p98 loudness
//      reject - the song's own section changes are that distribution's tail;
//      the 8 dB hard block stays)
//   E  as C with the relaxed judge
//   F  best effort (`best_effort` = true, `ok` = false): the complete plan
//      closest to the target at any length, when no tier makes the window.
//      The caller uses it only when its own fallback misses the length by
//      more (the requested length outranks clean-but-long, sesja 127).
// A plan needs every seam at or above `min_q` (kAcceptMinQ). No plan = the
// caller falls back to the beat-level engine.
//
// Sesja 134 (ADR-117 step 3, DEV-121) - Audition's structure: the windows
// are kShapeLengthSlackSec (5 s) at every tier, the trim tiers also offer
// bar-granular ends (`bar_ends`), a seam landing on a section start or inside
// the last section is OPEN (`seam_open`: every loudness gate off, the score is
// a cost only) and every seam carries a one-beat crossfade
// (`seam_crossfade_beats`) for the Renderer. Isolation round (Audition's exact
// segments through our Renderer): 10 ties / 1 Audition / 1 both bad - the
// structure is the lever, the one-beat seam only smooths the big-step cut-ins.
// Sesja 135 (DEV-121, the lost "dynamika" cut-ins): the q floor is waived
// only for landings exactly on a section start; an open landing inside the
// last section keeps the floor (the composite separates the rated verdicts
// there), and the open judge carries the brightness-collapse cap
// (PairScorer.h gate 4) for the verse -> filtered-outro family.
//
// Selection inside a tier (sesja 135, round-1 verdict): the plan whose WORST
// seam is best wins (kShapeMaximinBucket steps); the cost below breaks ties.
// Cost (lower wins inside a step): sum(1 - q) over seams + kSeamTax per seam
// + kTrimTax per trimmed piece + kNoChorusTax when the track has a chorus and
// the plan keeps none (the hook must survive) + a soft dynamics term: the
// section-context loudness excess of the seam in the substitution view
// (dB(first 8 beats of the landing) - dB(last 8 beats before the cut), minus
// the same step the original makes into that landing), positive excess only,
// / kDynamicsDbPerCost. Sesja-131 measurement: on 51 user-rated cuts the
// beat-RMS excess does not separate bad from ok (AUC 0.49-0.56 at 8 / 16 /
// 32 / 64 beats), so it is a tie-breaker, never a gate (DEV-117 b).
//
// C++-canonical (ADR-065): self-validated by tests/parity/test_shape_planner.cpp
// on hand-built fixtures; the perceptual gate is the blinded round.
#pragma once

#include "remix/Path.h"

#include <functional>
#include <optional>
#include <set>
#include <vector>

namespace reamix::remix
{

// Duration targets below this fraction of the track go through the planner
// (Duration mode, v2). The corpus tables at 0.15 / 0.25 / 0.33 are the
// planner's cases; 0.5 / 0.75 / 1.25 stay on the beat-level engine.
inline constexpr double kShapePlannerMaxRatio = 0.35;

// Sesja 134 (ADR-117 step 3, DEV-121): the length slack below the switch.
// Adobe Audition's Remix (blinded round sesja 133: Audition 8 / reamix 0 /
// tie 4) never misses its request by more than its 5 s slack; the user's
// verdicts put the requested length above a clean-but-long version. Replaces
// the +-8 / 10 s windows for the planner's tiers (the beat-level fallback
// keeps its own cap).
inline constexpr double kShapeLengthSlackSec = 5.0;

struct ShapeSection
{
    int b0   = 0;     // first beat (inclusive)
    int b1   = 0;     // one past the last beat; the last section ends at n_beats
    int kind = 1;     // reamix::theme::SegmentKind as int (3 = Chorus)
};

struct ShapeSeamScore
{
    double q              = 0.0;
    double energy_diff_db = 0.0;
    double edge_distance  = -1.0;
};

struct ShapePlannerInputs
{
    const double*       beat_times  = nullptr;
    int                 n_beats     = 0;
    double              track_sec   = 0.0;    // file length (the last section runs to it)
    const ShapeSection* sections    = nullptr;
    int                 n_sections  = 0;
    const std::set<int>* db_set     = nullptr; // downbeat beat indices (cleaned grid)
    int                 phrase_bars = 8;      // PhraseAlign::kPhraseBars
    // Sesja 135 (round-1 verdict on Daft Punk x0.25 / x0.33): a section
    // START is a zone, not a beat - the model boundary sits on the bar line
    // while the previous section's last chord still rings over it (Daft
    // Punk 283: -9.6 dB decaying, q 0.10; Audition landed two beats later
    // at -13.6 dB, q 0.59, rated clean). Every section but the first also
    // offers pieces starting 1 .. bar_beats-1 beats after its start (head
    // trims, taxed, every tier), and a landing anywhere inside that first
    // bar is an OPEN, floor-free landing like the exact start.
    int                 bar_beats   = 4;
    // Sesja 135 (DEV-122): the analysis grid's lattice mask (AnalysisBundle::
    // beatIsSynthetic, same length as beat_times; null = none). A landing on
    // a lattice beat (a beatless intro / outro / hole) is open and floor-free
    // like a section-start zone: the q floor guards against landing inside a
    // phrase, and a lattice zone has no phrases - Audition's rated-clean Drake
    // x0.15 lands 17 s into the beatless outro (our open judge: q 0.25).
    const std::vector<bool>* beat_is_synthetic = nullptr;
    const double*       rms_energy  = nullptr; // per beat, optional (soft dynamics term)
    double              target_sec  = 0.0;
    double              window_sec         = 8.0;
    double              window_relaxed_sec = 10.0;
    double              min_q              = 0.45;   // kAcceptMinQ
    // Boundary seam judge: leave at the end of beat i, land on beat j.
    // nullopt = a hard gate fired (the seam is not allowed).
    std::function<std::optional<ShapeSeamScore>(int i, int j)> seam;
    // The same judge without the p98 loudness reject (tiers D / E); empty =
    // those tiers are skipped.
    std::function<std::optional<ShapeSeamScore>(int i, int j)> seam_relaxed;
    // Sesja 134 (DEV-121): the judge with every loudness gate off, for OPEN
    // seams = landings on a section start or anywhere inside the last section
    // (the ending piece). Audition's intro -> ending cut-ins with +7..+11 dB
    // steps were rated clean, and on the 16 Audition seams our composite does
    // not separate the one cut the ear rejected (q 0.62) from the accepted
    // ones (q 0.26 .. 0.93): on a section-start landing the score is a cost
    // only, never a gate, and `min_q` does not apply. Sesja 135: a landing
    // inside the last section that is not its start keeps the `min_q` floor.
    // Empty = open seams are judged like every other seam (sesja-132
    // behaviour).
    std::function<std::optional<ShapeSeamScore>(int i, int j)> seam_open;
    // Sesja 134: bar-granular ends in the trim tiers - the first piece may
    // stop at any downbeat of the first section, the last piece may start at
    // any downbeat of the last section (Audition: 1 s intros, 6.5 s endings).
    // Sesja 135: every section offers bar-granular head / tail pieces in
    // the trim tiers (Audition's rated-clean Daft Punk x0.25 keeps 11 beats
    // of the outro's first part before its 6 s ending; our whole-section
    // alternative was a quiet -> loud jump rated "dynamika").
    bool bar_ends = false;
    // Sesja 134: seam crossfade in beats written to the path's metadata
    // (`preferred_overlap_sec`; the Renderer overlays the whole window with
    // one equal-power crossfade above the 200 ms band cap). 0 = the
    // Renderer's default (0.15 beat multi-band). Audition: one beat.
    double seam_crossfade_beats = 0.0;
    // Sesja 136 (ADR-117 step 5, "Audition's recipe"): recipe_mode != 0
    // bypasses the tiers, the maximin selection and the chorus tax above.
    // Round 2 (sesja 135) lost 0/7 to Audition; the audit
    // (meta/research/sesja-136-recipe-audit.md) found the tier order (whole
    // sections first), the chorus tax in maximin units, a 0.88 s crossfade and
    // length-driven fragments behind the losses, and q separating rated ok from
    // bad seams at AUC 0.74 only. Audition's 28 rated cut points on our grid:
    // the plan always starts at 0 and ends at the file end, +-5 s, 1-3 seams,
    // landings NOT bound to section starts (5/28 on downbeats, 0/28 on phrase
    // starts, often the pickup beat before a bar line, 9/28 in lattice zones).
    // Recipe: cut candidates = downbeats, the beat before each downbeat, the
    // first bar of every section, lattice bar starts (db_set carries them);
    // every seam is judged OPEN (`seam_open`: the edge / rms caps and the
    // brightness-collapse cap are the only gates) above `recipe_q_floor`;
    // pieces >= `recipe_min_piece_beats`; the FEWEST seams win outright, then
    // the mode ranks the plans of that seam count (2 s / 0.1 q steps):
    //   1 "ending"   longest last piece (the song's own ending), then the
    //                longest shortest piece, landing class, closest length
    //   2 "q"        best worst q, then the longest shortest piece, landing
    //                class, closest length
    //   3 "balanced" longest shortest piece, landing class, closest length
    // Landing class: section-start zone 2, downbeat / pickup / lattice 1.
    // Prototype: tools/dev/shape_eval/recipe_planner.py (C++ == prototype on
    // the round-3 cases). A recipe plan reports tier 'R'.
    int    recipe_mode            = 0;
    double recipe_q_floor         = 0.25;
    int    recipe_min_piece_beats = 8;
    int    recipe_max_seams       = 3;
    // Sesja 136: cap on the seam crossfade in seconds (0 = none; applies to
    // every planner seam). Drake x0.25's one-beat seam at 67 BPM = 0.88 s
    // overlapped two different 16th-note patterns ("podwojne uderzenie");
    // Audition's crossfades are 0.44-0.61 s on every corpus case.
    double seam_crossfade_max_sec = 0.0;
};

struct ShapePiece
{
    enum class Trim { Whole, Head, Tail };
    int  section = 0;
    int  b0 = 0, b1 = 0;
    int  kind = 1;
    Trim trim = Trim::Whole;
};

struct ShapeSeam
{
    int            i = 0, j = 0;   // cut leaves after beat i, lands on beat j
    ShapeSeamScore score;
    double         excess_db = 0.0;
    bool           open      = false;   // sesja 134: judged by `seam_open`, no q floor
    double         overlap_sec = 0.0;   // sesja 134: crossfade for the Renderer (0 = default)
};

struct ShapePlan
{
    bool   ok      = false;        // a tier A..E plan
    bool   best_effort = false;    // tier F: complete plan outside every window (ok stays false)
    char   tier    = '-';          // 'A' .. 'F'
    double est_sec = 0.0;          // estimated render length (head + pieces + tail)
    double dev_sec = 0.0;          // est - target
    double cost    = 0.0;
    double min_q   = 1.0;          // over every seam (section-start landings included; the floor applies to the others)
    std::vector<ShapePiece> pieces;
    std::vector<ShapeSeam>  seams;

    // Why a plan was or was not found (REAMIX_DURATION_DEBUG).
    struct Diag
    {
        int    pieces_whole = 0, pieces_trim = 0;
        int    seams_tried = 0, seams_strict = 0, seams_relaxed = 0;   // section / phrase seams judged, passing (q >= min_q)
        double closest_dev_whole = 0.0, closest_dev_trim = 0.0;        // best |est - target| reachable with passing seams (any window)
        struct Judged { int i, j; bool strict_ok, relaxed_ok; double q; };
        std::vector<Judged> judged;                                    // every seam the search asked about (strict judge; relaxed when asked)
    } diag;

    // Beat path for the renderer: consecutive beats inside every piece, one
    // transition per seam with quality_score / family (1) / energy_diff_db /
    // edge_distance metadata.
    RemixPath toPath() const;
};

// Cost constants (see the header comment).
inline constexpr double kShapeSeamTax          = 0.10;
// Sesja 135: maximin plan selection - plans are ranked by the quality of
// their weakest seam in steps of this size (cost breaks ties inside a step).
inline constexpr double kShapeMaximinBucket    = 0.05;
inline constexpr double kShapeTrimTax          = 0.15;
inline constexpr double kShapeNoChorusTax      = 0.50;
// Sesja 136: the no-chorus tax in maximin steps. Two rated data points bound
// it: Tiesto x0.25's no-chorus 1-seam plan (q 0.39, rated ok) must beat its
// chorus plan (worst seam 0.25, rated bad) - so at most 2 steps; Avicii x0.25
// (sesja 132) needed its chorus plan (0.54) over the no-chorus one (0.60,
// rated "no drop") - so at least 2 steps. Sesja 135's 10 steps forced the
// Tiesto regression.
inline constexpr int    kShapeNoChorusTaxSteps = 2;
inline constexpr double kShapeDynamicsDbPerCost= 10.0;
inline constexpr int    kShapeContextBeats     = 8;
inline constexpr double kShapeBinSec           = 0.5;
inline constexpr int    kShapeMinSections      = 3;
// Sesja 134: a trimmed piece (phrase or bar trim, head or tail) is at least
// this many beats - Audition's shortest ending is 6.4 s (its minimum loop is
// 8 beats); with the edge caps the DP otherwise closes with 2 beats of the
// last chorus + the file tail (Dance Monkey x0.15: 336-338), a truncation.
inline constexpr int    kShapeMinEndBeats      = 8;
// Sesja 136 (ADR-117 step 5): structure limits from Audition's 12 rated
// plans (1-3 seams, 4 once; intro pieces down to one beat) and the round-3 /
// s136 audit (the merged whole + trim search otherwise chains 8-beat
// fragments of middle sections through six high-q seams: Daft Punk x0.25
// 9 pieces, Calvin Harris 1:09 a 2-bar chorus - the ear's "fragment").
// Bar / half-bar ends exist for the FIRST section (heads) and the LAST
// section (tails) only; middle sections are whole or phrase-trimmed.
inline constexpr int    kShapeMaxSeams         = 3;
// The first piece (intro head) and the last piece (ending tail) may be one
// bar: Audition's rated-clean plans open with 1-6 s and close with 6.4 s
// (Daft Punk x0.25: 4.9 s intro -> 141.6-177.7 -> the last 6.4 s = beats
// 407-413 + the file tail; our 8-beat minimum blocked that ending and the
// planner took the rated-bad inst -> outro-start seam instead).
inline constexpr int    kShapeMinEndBars       = 1;
// A trimmed MIDDLE piece shorter than kShapeFragmentBeats costs
// kShapeFragmentSteps maximin steps (the ear's "fragment"): without it the
// merged search chained five 8-beat pieces through q 0.88 seams on Calvin
// Harris 1:09 (a 2-bar chorus) against the rated-great whole-section plan
// (worst q 0.59); with 1 step the chain still won (17 - 5 vs 11), with 2 it
// loses (7 vs 11) while the rated-ok Alice x0.25 plan (one 8-beat chorus
// head, worst 0.60 -> 10) keeps beating its 8-beat-tail alternative (9) and
// Calvin 0:19's rated-great plan (2-bar chorus head, 0.65 -> 10) beats the
// 1-seam intro -> ending plan (0.47 -> 9). First / last pieces are exempt.
inline constexpr int    kShapeFragmentBeats    = 16;
inline constexpr int    kShapeFragmentSteps    = 2;
// Round 4 (sesja 136): the planner added a second seam to Daft Punk x0.33
// for +0.02 q - a skip of two bars INSIDE the outro (347-371 -> 379, rated
// "nie ten fragment z taktu") where Audition's 1-seam plan (the same first
// cut) was rated clean three times. A general per-seam step reshuffled the
// rated-ok plans of Alice x0.25, Daft Punk x0.25 and Calvin Harris 1:09, so
// the step is narrow: a seam that leaves AND lands in outro-kind sections
// (a skip inside the ending, where the ear expects a verbatim run - every
// Audition ending is one) costs one maximin step. Daft Punk x0.25's rated-ok
// outro seam (366 -> 399, a 32-beat skip) keeps its plan under it.
inline constexpr int    kShapeEndingSkipSteps  = 1;
inline constexpr int    kOutroKind             = 11;  // reamix::theme::SegmentKind::Outro
// Seam crossfade = one beat (Audition), but Audition detects the double
// tempo on slow tracks (Drake 0.444 s = half our 0.85 s period) and the
// round-4 ear called our 0.5 s cap "za dlugi" on Drake and heard the seams
// "na fejdach" on Dance Monkey (0.5 s cap under its 0.61 s beat): a beat
// period above this many seconds is halved instead of capped.
inline constexpr double kShapeSeamHalveAboveSec = 0.65;

ShapePlan planShape(const ShapePlannerInputs& in);

// Grid-snapped sections in seconds -> beat ranges covering [0, n_beats):
// a boundary maps to the first beat at/after it; the first section starts at
// beat 0 and the last ends at n_beats.
std::vector<ShapeSection> shapeSectionsFromSeconds(const double* beat_times, int n_beats,
                                                   const double* starts, const double* ends,
                                                   const int* kinds, int n_sections);

} // namespace reamix::remix
