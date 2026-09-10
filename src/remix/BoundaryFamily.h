#pragma once
// ---------------------------------------------------------------------------
// BoundaryFamily — the second splice-candidate family (ADR-116 step 3,
// sesja 130). CANONICAL DEFINITION (no Python reference); self-validated by
// tests/parity/test_boundary_family.cpp.
//
// The continuation family (TransitionCost main loop) admits a cut that
// keeps repeating material: same phrase position, on a repetition diagonal,
// judged by the continuity composite. A BOUNDARY cut is a different act: it
// leaves at the END of a phrase (beat i is the last beat before a phrase
// start) and lands on the START of a phrase (beat j opens one), so the
// arrangement changes on purpose - the radio-edit cut (intro -> last chorus),
// the Blocks junction, the whole-phrase Region loop. Evidence (sesja 128
// seam probe, meta/DECISIONS.md ADR-116 STATUS UPDATE 1): four of five such
// placements were rated usable although the continuity composite scores
// them 0.2-0.5, because its waveform term compares start_{i+1} with start_j
// - two different sections by design.
//
// Phrase start = a downbeat whose bar offset inside its section is 0 mod
// phrase_bars (PhraseAlign::barOffsets: bars from the section start, or
// from the first downbeat when the track has no section map). Every section
// start is a phrase start by construction (offset 0). Beat 0 never lands
// (no original context before it).
// ---------------------------------------------------------------------------

#include "analysis/StructureResult.h"
#include "remix/PhraseAlign.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace reamix::remix {

struct BoundaryFamily
{
    bool active      { false };
    int  n_beats     { 0 };
    int  phrase_bars { 0 };
    int  n_starts    { 0 };
    std::vector<std::uint8_t> phrase_start;   // per beat, 1 = downbeat opening a phrase

    static constexpr int kPhraseBars = PhraseAlign::kPhraseBars;   // 8 grid bars
    static constexpr int kMinStarts  = 2;                          // one to leave from, one to land on

    bool isPhraseStart(int b) const noexcept
    {
        return active && b >= 0 && b < n_beats && phrase_start[static_cast<std::size_t>(b)] != 0;
    }
    // The cut after beat i leaves at a phrase end when beat i + 1 opens a phrase.
    bool leavesAtPhraseEnd(int i) const noexcept { return isPhraseStart(i + 1); }
    // Beat j lands on a phrase start; j == 0 has no original context (excluded).
    bool landsAtPhraseStart(int j) const noexcept { return j > 0 && isPhraseStart(j); }
    bool pairOk(int i, int j) const noexcept { return leavesAtPhraseEnd(i) && landsAtPhraseStart(j); }

    // Declare beat b a phrase start after build (Blocks: the beat after a
    // user block - leaving at the block's end is leaving at a phrase end,
    // whatever the 8-bar count from the block start says).
    void markStart(int b) noexcept
    {
        if (b < 0 || b >= n_beats || phrase_start.empty()) return;
        if (phrase_start[static_cast<std::size_t>(b)] != 0) return;
        phrase_start[static_cast<std::size_t>(b)] = 1;
        ++n_starts;
        active = n_starts >= kMinStarts;
    }

    static BoundaryFamily build(const double* beat_times, int n,
                                const std::set<int>& db_set,
                                const analysis::Segment* segs, int n_segs,
                                int phrase_bars = kPhraseBars)
    {
        BoundaryFamily f;
        f.n_beats     = n;
        f.phrase_bars = phrase_bars;
        if (beat_times == nullptr || n <= 0 || db_set.empty() || phrase_bars <= 0) return f;
        const std::vector<int> offset = PhraseAlign::barOffsets(beat_times, n, db_set, segs, n_segs);
        f.phrase_start.assign(static_cast<std::size_t>(n), 0);
        for (const int b : db_set) {
            if (b < 0 || b >= n) continue;
            const int off = offset[static_cast<std::size_t>(b)];
            if (off >= 0 && off % phrase_bars == 0) {
                f.phrase_start[static_cast<std::size_t>(b)] = 1;
                ++f.n_starts;
            }
        }
        f.active = f.n_starts >= kMinStarts;
        return f;
    }
};

} // namespace reamix::remix
