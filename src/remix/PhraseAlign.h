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
    // DEV-117 (d) (sesja 127): grid bar per beat and the section id per beat
    // (-1 = none) for the short-loop rule below.
    std::vector<int> bar_index;
    std::vector<int> section_of;

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

    // DEV-117 (d) (sesja 127) - Region / Blocks short loops. A backward pair
    // that repeats L grid bars of ONE section (j < i, same section, L in
    // {1, 2, 4}) is a loop of its own size: it keeps the phrase when it
    // starts on a multiple of L inside the phrase (a 2-bar loop at bar 0,
    // 2, 4, 6 of the 8-bar phrase; a 4-bar loop at 0 or 4; a 1-bar loop
    // anywhere). The mod-8 rule would reject every 2- / 4-bar loop that
    // does not start a phrase, and Region lives on those loops (sesja 122:
    // 2-bar loop spots are a user decision). Everything else = `allowed`.
    bool loopAllowed(int i, int j) const noexcept
    {
        if (! active) return true;
        if (i < 0 || j < 0 || i + 1 >= n_beats || j >= n_beats) return false;
        if (j < i && bar_index.size() == bar_offset.size() && section_of.size() == bar_offset.size()) {
            // The looped material is [j, i]; i + 1 may already start the
            // next section (a loop ending on a section end), so the section
            // test uses beat i, the bar count the bar after it.
            const std::size_t a = static_cast<std::size_t>(i + 1), b = static_cast<std::size_t>(j);
            const int sec_i = section_of[static_cast<std::size_t>(i)], sec_b = section_of[b];
            const int L = bar_index[a] - bar_index[b];
            // A loop of a whole number of phrases (8, 16, ... grid bars)
            // returns to its own phrase position by construction, whatever
            // the section map says about its two ends (Billie Jean 51.6 ->
            // 68.0 s: an 8-bar loop across the verse / chorus boundary).
            if (L > 0 && L % kPhraseBars == 0 && bar_index[b] >= 0) return true;
            if (sec_i == sec_b && sec_i >= 0 && bar_offset[b] >= 0
                && (L == 1 || L == 2 || L == 4))
                return bar_offset[b] % L == 0;
        }
        return allowed(i, j);
    }

    // Bar offsets per beat. `db_set` = beat indices that are downbeats.
    // Sections (may be null / empty) are [start, end) in seconds on the same
    // grid; a beat inside no section counts bars from the first downbeat.
    // `snap_to_downbeat` (DEV-120, sesja 132): section boundaries snapped to
    // the nearest downbeat within two beats (the rule buildUiSegments uses).
    // ON for the boundary family (BoundaryFamily::build) so every section
    // start the UI shows is a phrase start; OFF for the continuation phrase
    // gate, whose behaviour on the 50-125 % ratios the ear confirmed (sesje
    // 126-129) - measured in sesja 132: snapping the gate too changed 14 of 51
    // normal-ratio cases (Woodkid x0.75 to a red cut, Goldberg to one loop x4).
    static std::vector<int> barOffsets(const double* beat_times, int n,
                                       const std::set<int>& db_set,
                                       const analysis::Segment* segs, int n_segs,
                                       bool snap_to_downbeat = false)
    {
        std::vector<int> bar_index(static_cast<std::size_t>(n < 0 ? 0 : n), -1);
        int bar = -1;
        for (int b = 0; b < n; ++b) {
            if (db_set.count(b) > 0) ++bar;
            bar_index[static_cast<std::size_t>(b)] = bar;
        }
        std::vector<int> out = bar_index;
        if (segs == nullptr || n_segs <= 0) return out;
        // DEV-120 (sesja 132): section boundaries come from the section model
        // on its own beat grid and can sit a beat off the engine's downbeat.
        // "The first beat at/after the raw start" then lands in the previous
        // bar whenever the raw boundary precedes the downbeat, shifting every
        // offset of that section by one bar - its real start stops being a
        // phrase start (High Hopes final chorus, the Without Me outros were
        // unlandable for the boundary family). With `snap_to_downbeat` each
        // boundary snaps to the NEAREST downbeat (the rule buildUiSegments
        // uses for the UI) when one lies within two beats; otherwise (and
        // always for the continuation gate) the first beat at/after it.
        std::vector<double> snapped(static_cast<std::size_t>(n_segs), 0.0);
        std::vector<int>    seg_bar(static_cast<std::size_t>(n_segs), -1);
        const double period = (n >= 2) ? (beat_times[n - 1] - beat_times[0]) / static_cast<double>(n - 1) : 0.0;
        for (int s = 0; s < n_segs; ++s) {
            int best = -1;
            double best_dt = 2.0 * period + 1e-6;
            if (snap_to_downbeat)
                for (const int d : db_set) {
                    if (d < 0 || d >= n) continue;
                    const double dt = std::abs(beat_times[d] - segs[s].start);
                    if (dt < best_dt) { best_dt = dt; best = d; }
                }
            if (best < 0) {
                for (int b = 0; b < n; ++b)
                    if (beat_times[b] >= segs[s].start - 1e-6) { best = b; break; }
            }
            snapped[static_cast<std::size_t>(s)] = best >= 0 ? beat_times[best] : segs[s].start;
            seg_bar[static_cast<std::size_t>(s)] = best >= 0 ? bar_index[static_cast<std::size_t>(best)] : -1;
        }
        for (int b = 0; b < n; ++b) {
            const double t = beat_times[b];
            for (int s = 0; s < n_segs; ++s) {
                const double s_start = snap_to_downbeat ? snapped[static_cast<std::size_t>(s)] : segs[s].start;
                const double s_end   = snap_to_downbeat
                    ? ((s + 1 < n_segs) ? snapped[static_cast<std::size_t>(s) + 1] : segs[s].end)
                    : segs[s].end;
                if (t >= s_start - 1e-6 && t < s_end - 1e-6) {
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
        // DEV-117 (d): grid bar + section id per beat (same lookup as barOffsets).
        p.bar_index.assign(static_cast<std::size_t>(n), -1);
        p.section_of.assign(static_cast<std::size_t>(n), -1);
        {
            int bar = -1;
            for (int b = 0; b < n; ++b) {
                if (db_set.count(b) > 0) ++bar;
                p.bar_index[static_cast<std::size_t>(b)] = bar;
                if (segs == nullptr) continue;
                for (int s2 = 0; s2 < n_segs; ++s2)
                    if (beat_times[b] >= segs[s2].start - 1e-6 && beat_times[b] < segs[s2].end - 1e-6) {
                        p.section_of[static_cast<std::size_t>(b)] = s2;
                        break;
                    }
            }
        }
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
