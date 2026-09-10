#pragma once
// ---------------------------------------------------------------------------
// PhraseAlign — phrase-position alignment gate for splice candidates
// (sesja 126, DEV-116, ADR-115 E12). CANONICAL DEFINITION (no Python
// reference); self-validated by tests/parity/test_phrase_align.cpp.
//
// Listening evidence (sesja 126, Drake "Hotline Bling" + Bob Dylan "Lay Lady
// Lay", 11 user-rated cuts): every cut the user heard as clean keeps the
// POSITION IN THE PHRASE - it leaves bar k of one section and lands on bar
// k of another section of the same kind (chorus +7.1 s -> chorus +7.1 s,
// verse +10.7 s -> verse +10.7 s) - or lands on a phrase start; every cut
// the user heard as bad is bar-aligned, spectrally matched, on a repetition
// diagonal, and 1-4 bars off inside the phrase (verse +10.7 s -> verse
// +3.6 s: the lyric jumps mid-phrase). None of the 15 scoring signals
// separates the two groups; the bar grid plus the section map does.
//
// Rule (pair i -> j = the cut after beat i, landing on beat j):
//     offset(i + 1) == offset(j)  (mod phrase_bars)   or   offset(j) == 0 (mod phrase_bars)
// where offset(b) = bars from the start of the section containing beat b
// (bars from the first downbeat when the track has no section map).
// phrase_bars = 8 GRID bars (the hypermeter of popular music; the section
// model's segments are ~8 bars). On a half-bar grid (Bob Dylan: measured bar
// = 2 beats at the true tempo) that is 4 real bars - and the evidence says
// so: a 16-grid-bar rule rejected the Dylan 7.5 s -> 33.2 s cut the user
// rated "super" (offsets 5 -> 13), while every bad Dylan cut fails mod 8.
// Relax to 4 when the gate would starve the DP (fewer than
// kMinAllowedPerSource targets per source on average, the RepetitionPrior
// rule), then off.
// ---------------------------------------------------------------------------

#include "analysis/StructureResult.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <vector>

namespace reamix::remix {

struct PhraseAlign
{
    bool             active      { false };
    int              n_beats     { 0 };
    int              phrase_bars { 0 };   // 8, 4, or 0 when inactive
    std::vector<int> bar_offset;          // per beat; -1 = unknown (before the first downbeat)
    int              n_allowed   { 0 };   // (pre-downbeat, downbeat) pairs that pass every gate
    int              n_sources   { 0 };   // pre-downbeat sources with >= 1 allowed target

    static constexpr int    kPhraseBars          = 8;
    static constexpr int    kRelaxedPhraseBars   = 4;
    static constexpr double kMinAllowedPerSource = 3.0;   // RepetitionPrior::kMinAllowedPerSource

    // Position rule only (no starvation logic).
    static bool positionOk(int off_out, int off_in, int phrase_bars) noexcept
    {
        if (phrase_bars <= 0) return true;
        if (off_out < 0 || off_in < 0) return true;     // unknown position: never block
        const int a = off_out % phrase_bars;
        const int b = off_in  % phrase_bars;
        return a == b || b == 0;
    }

    bool allowed(int i, int j) const noexcept
    {
        if (! active) return true;
        if (i < 0 || j < 0 || i + 1 >= n_beats || j >= n_beats) return false;
        return positionOk(bar_offset[static_cast<std::size_t>(i + 1)],
                          bar_offset[static_cast<std::size_t>(j)], phrase_bars);
    }

    // Bar offsets per beat. `db_set` = beat indices that are downbeats.
    // Sections (may be null / empty) are [start, end) in seconds on the same
    // grid; a beat inside no section counts bars from the first downbeat.
    static std::vector<int> barOffsets(const double* beat_times, int n,
                                       const std::set<int>& db_set,
                                       const analysis::Segment* segs, int n_segs)
    {
        std::vector<int> bar_index(static_cast<std::size_t>(n < 0 ? 0 : n), -1);
        int bar = -1;
        for (int b = 0; b < n; ++b) {
            if (db_set.count(b) > 0) ++bar;
            bar_index[static_cast<std::size_t>(b)] = bar;
        }
        std::vector<int> out = bar_index;
        if (segs == nullptr || n_segs <= 0) return out;
        // Section start bar = bar of the first beat at/after the section start.
        std::vector<int> seg_bar(static_cast<std::size_t>(n_segs), -1);
        for (int s = 0; s < n_segs; ++s) {
            for (int b = 0; b < n; ++b) {
                if (beat_times[b] >= segs[s].start - 1e-6) {
                    seg_bar[static_cast<std::size_t>(s)] = bar_index[static_cast<std::size_t>(b)];
                    break;
                }
            }
        }
        for (int b = 0; b < n; ++b) {
            const double t = beat_times[b];
            for (int s = 0; s < n_segs; ++s) {
                if (t >= segs[s].start - 1e-6 && t < segs[s].end - 1e-6) {
                    const int sb = seg_bar[static_cast<std::size_t>(s)];
                    const int bi = bar_index[static_cast<std::size_t>(b)];
                    out[static_cast<std::size_t>(b)] = (sb < 0 || bi < 0) ? -1 : bi - sb;
                    break;
                }
            }
        }
        return out;
    }

    // `base_allowed(i, j)` = the other v2 gates (downbeat target + repetition
    // prior); the starvation rule counts only pairs that pass them.
    template <typename BaseAllowed>
    static PhraseAlign build(const double* beat_times, int n,
                             const std::set<int>& db_set, const std::set<int>& pre_db_set,
                             const analysis::Segment* segs, int n_segs,
                             BaseAllowed&& base_allowed)
    {
        PhraseAlign p;
        p.n_beats = n;
        if (beat_times == nullptr || n <= 0 || db_set.empty() || pre_db_set.empty()) return p;
        p.bar_offset = barOffsets(beat_times, n, db_set, segs, n_segs);
        for (const int bars : { kPhraseBars, kRelaxedPhraseBars }) {
            int allowed = 0, sources = 0;
            for (const int i : pre_db_set) {
                if (i + 1 >= n) continue;
                bool any = false;
                for (const int j : db_set) {
                    if (! base_allowed(i, j)) continue;
                    if (! positionOk(p.bar_offset[static_cast<std::size_t>(i + 1)],
                                     p.bar_offset[static_cast<std::size_t>(j)], bars)) continue;
                    ++allowed;
                    any = true;
                }
                if (any) ++sources;
            }
            const double per_source = sources > 0 ? static_cast<double>(allowed) / sources : 0.0;
            if (per_source >= kMinAllowedPerSource) {
                p.active = true; p.phrase_bars = bars; p.n_allowed = allowed; p.n_sources = sources;
                return p;
            }
        }
        return p;   // inactive: the gate would starve the DP on this track
    }
};

} // namespace reamix::remix
