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
// Cost (lower wins inside a tier): sum(1 - q) over seams + kSeamTax per seam
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
};

struct ShapePlan
{
    bool   ok      = false;        // a tier A..E plan
    bool   best_effort = false;    // tier F: complete plan outside every window (ok stays false)
    char   tier    = '-';          // 'A' .. 'F'
    double est_sec = 0.0;          // estimated render length (head + pieces + tail)
    double dev_sec = 0.0;          // est - target
    double cost    = 0.0;
    double min_q   = 1.0;
    std::vector<ShapePiece> pieces;
    std::vector<ShapeSeam>  seams;

    // Why a plan was or was not found (REAMIX_DURATION_DEBUG).
    struct Diag
    {
        int    pieces_whole = 0, pieces_trim = 0;
        int    seams_tried = 0, seams_strict = 0, seams_relaxed = 0;   // section / phrase seams judged, passing (q >= min_q)
        double closest_dev_whole = 0.0, closest_dev_trim = 0.0;        // best |est - target| reachable with passing seams (any window)
    } diag;

    // Beat path for the renderer: consecutive beats inside every piece, one
    // transition per seam with quality_score / family (1) / energy_diff_db /
    // edge_distance metadata.
    RemixPath toPath() const;
};

// Cost constants (see the header comment).
inline constexpr double kShapeSeamTax          = 0.10;
inline constexpr double kShapeTrimTax          = 0.15;
inline constexpr double kShapeNoChorusTax      = 0.50;
inline constexpr double kShapeDynamicsDbPerCost= 10.0;
inline constexpr int    kShapeContextBeats     = 8;
inline constexpr double kShapeBinSec           = 0.5;
inline constexpr int    kShapeMinSections      = 3;

ShapePlan planShape(const ShapePlannerInputs& in);

// Grid-snapped sections in seconds -> beat ranges covering [0, n_beats):
// a boundary maps to the first beat at/after it; the first section starts at
// beat 0 and the last ends at n_beats.
std::vector<ShapeSection> shapeSectionsFromSeconds(const double* beat_times, int n_beats,
                                                   const double* starts, const double* ends,
                                                   const int* kinds, int n_sections);

} // namespace reamix::remix
