// RepetitionPrior - candidate mask from repetition diagonals (ADR-115 E4).
//
// Sesja 115. Plachouras & Miron (ICASSP 2023, arXiv 2305.07347) beat Adobe
// Audition Remix in a listening test by allowing a jump i -> j only when the
// pair lies on a repetition diagonal of the beat-level recurrence matrix
// that is at least one measure long with a meter-multiple offset. Their
// failure case for Audition ("Dancing Queen": single beats matched with
// mismatched surroundings) is exactly what a per-beat similarity top-k
// produces. This module turns the existing, parity-tested
// `analysis::Recurrence::build` (unused in production since ADR-044) into a
// boolean mask over (pre-downbeat source, downbeat target) pairs:
//
//   allowed(i, j)  <=>  at least `min_run` of the 2 x TS diagonal cells
//                       R[i+1+k, j+k], k in [-TS, TS), are above
//                       Recurrence::kDiagonalThreshold
//
// i.e. the bar before and the bar after the junction, read in the
// successor view (beat j must resemble beat i+1 in context), contain at
// least one measure's worth of mutual-kNN recurrence hits. Labels are not
// needed (ADR-044 stands). When the track yields too few allowed pairs the
// prior deactivates itself (`active == false`) and the caller falls back
// to the E3 bar-aligned candidate set.
//
// C++-canonical (ADR-065); validated by tests/parity/test_repetition_prior.cpp
// on synthetic recurrence matrices.
#pragma once

#include <cstdint>
#include <set>
#include <vector>

namespace reamix::remix {

struct RepetitionPrior
{
    int  n_beats   { 0 };
    bool active    { false };
    int  n_allowed { 0 };          // allowed (i, j) pairs
    int  n_sources { 0 };          // pre-downbeat sources with >= 1 allowed target
    int  min_run_used { 0 };       // diagonal cells required (TS = one measure; TS/2 after relax)
    std::vector<std::uint8_t> mask; // row-major n x n, 1 = allowed; empty when inactive

    int  outro_exempt_from { 0 };   // targets j >= this index bypass the mask

    bool allowed(int i, int j) const noexcept
    {
        if (! active) return true;
        if (i < 0 || j < 0 || i >= n_beats || j >= n_beats) return false;
        if (j >= outro_exempt_from) return true;
        return mask[static_cast<std::size_t>(i) * n_beats + j] != 0;
    }

    // Minimum share of pre-downbeat sources that must keep at least one
    // allowed target for the prior to stay active (fallback rule).
    static constexpr double kMinSourceCoverage = 0.25;
    // Recurrence k for the mask (RepetitionMap uses 12; denser than the
    // structure default 10, so short songs still form diagonals).
    static constexpr int kRecurrenceK = 12;
    // Minimum diagonal run required, in measures (1.0 = one full measure of
    // recurrence hits within the two bars around the junction).
    static constexpr double kMinRunBars = 1.0;
    // Sparse-repetition relax (sesja 115 listening round 2): when the
    // one-measure rule leaves fewer than this many allowed targets per
    // pre-downbeat source (jazz / through-composed material: Miles Davis
    // 1.8, Periphery 1.5), the rule drops to half a measure. Below that the
    // mask starves the DP and the user rated both engines "both bad".
    static constexpr double kMinAllowedPerSource = 3.0;
    static constexpr double kRelaxedMinRunBars   = 0.5;
    // Outro exemption (sesja 115 user smoke, Billie Jean at 0:24): a song's
    // ending (fade-out, coda) repeats nothing, so a pure repetition mask can
    // leave the DP no jump INTO the outro and the remix stops mid-song. Targets
    // in the last kOutroExemptBars bars bypass the mask (bar alignment still
    // applies), so every remix can still reach the real ending.
    static constexpr int kOutroExemptBars = 8;

    // `R` is the combined recurrence matrix [n x n] from Recurrence::build.
    // `min_run` defaults to one measure (time_signature cells).
    static RepetitionPrior fromRecurrence(const double* R, int n,
                                          const std::set<int>& pre_db_set,
                                          const std::set<int>& db_set,
                                          int time_signature,
                                          int min_run = 0,
                                          double threshold = 0.1);

    // Convenience: builds the recurrence from features, then the mask.
    static RepetitionPrior build(const float* features, int n, int n_feat,
                                 const std::set<int>& pre_db_set,
                                 const std::set<int>& db_set,
                                 int time_signature);
};

} // namespace reamix::remix
