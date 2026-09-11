// ShapePlanner.cpp — sesja 131 (ADR-117). See ShapePlanner.h.
#include "remix/ShapePlanner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace reamix::remix
{
namespace
{

constexpr int kChorusKind = 3;   // reamix::theme::SegmentKind::Chorus

struct Piece
{
    ShapePiece p;
    double     dur = 0.0;
};

struct State
{
    double cost      = std::numeric_limits<double>::infinity();
    double min_q     = 1.0;     // over the floored seams (the min_q gate)
    double min_q_all = 1.0;     // sesja 135: over EVERY seam (the maximin key)
    int    prev_p    = -1;
    int    prev_bin  = -1;
    int    prev_c    = 0;
    int    n_seams   = 0;       // sesja 136: kShapeMaxSeams
    int    fragments = 0;       // sesja 136: trimmed middle pieces shorter than kShapeFragmentBeats
    int    ending_skips = 0;    // sesja 136: seams leaving and landing in outro-kind sections
    bool   chorus    = false;
    bool   valid() const noexcept { return std::isfinite(cost); }
};

// Sesja 135 (round-1 verdict): the worst seam decides. Plans are ranked by
// the quality of their weakest seam in kShapeMaximinBucket steps, cost
// breaks ties inside a step. On Daft Punk x0.25 the cheapest plan was one
// intro -> outro seam at q 0.25 (rated "zgrzyt, dynamika, fragment"), while
// Audition's structure (intro -> solo -> outro head -> ending, two seams at
// q 0.40) cost more; every rated-clean open seam but Tiesto's two intro ->
// outro cut-ins scores >= 0.39, the rated-bad ones mostly below 0.42.
int maximinBucket(double q) { return static_cast<int>(std::floor(q / kShapeMaximinBucket + 1e-9)); }
// Sesja 136: the maximin key of a state = the worst seam's bucket minus the
// fragment steps (short trimmed middle pieces, see kShapeFragmentSteps).
int stateBucket(const State& st)
{
    return maximinBucket(st.min_q_all) - kShapeFragmentSteps * st.fragments
         - kShapeEndingSkipSteps * st.ending_skips;
}
bool betterState(const State& a, const State& b)
{
    const int ba = stateBucket(a), bb = stateBucket(b);
    return ba != bb ? ba > bb : a.cost < b.cost;
}

double beatEndTime(const ShapePlannerInputs& in, int b)
{
    return (b >= in.n_beats) ? in.track_sec : in.beat_times[b];
}

// Phrase starts per section: the section start and every `phrase_bars`-th
// downbeat inside it (PhraseAlign::barOffsets with the grid-snapped sections).
std::set<int> phraseStarts(const ShapePlannerInputs& in)
{
    std::set<int> ps;
    if (in.db_set == nullptr) return ps;
    for (int s = 0; s < in.n_sections; ++s) {
        const ShapeSection& sec = in.sections[s];
        int k = 0;
        for (auto it = in.db_set->lower_bound(sec.b0); it != in.db_set->end() && *it < sec.b1; ++it, ++k)
            if (k % std::max(1, in.phrase_bars) == 0) ps.insert(*it);
    }
    return ps;
}

std::vector<Piece> buildPieces(const ShapePlannerInputs& in, bool allow_trim, const std::set<int>& ps)
{
    std::vector<Piece> out;
    for (int s = 0; s < in.n_sections; ++s) {
        const ShapeSection& sec = in.sections[s];
        if (sec.b1 <= sec.b0) continue;
        auto push = [&] (int b0, int b1, ShapePiece::Trim trim) {
            // Sesja 134: a trimmed piece is >= kShapeMinEndBeats; sesja 136:
            // the first section's head and the last section's tail may be as
            // short as kShapeMinEndBars.
            const bool endPiece = (s == 0 && trim == ShapePiece::Trim::Head)
                               || (s == in.n_sections - 1 && trim == ShapePiece::Trim::Tail);
            const int minBeats = endPiece ? kShapeMinEndBars * std::max(1, in.bar_beats) : kShapeMinEndBeats;
            if (trim != ShapePiece::Trim::Whole && b1 - b0 < minBeats) return;
            Piece pc;
            pc.p.section = s; pc.p.b0 = b0; pc.p.b1 = b1; pc.p.kind = sec.kind; pc.p.trim = trim;
            pc.dur = beatEndTime(in, b1) - in.beat_times[b0];
            out.push_back(pc);
        };
        push(sec.b0, sec.b1, ShapePiece::Trim::Whole);
        // Sesja 135: the section start is a zone - pieces starting inside the
        // first bar (every tier; the piece is a head trim, so the tax keeps the
        // exact start unless the judge prefers a later landing). Sesja 136: the
        // bar-phase rule (phaseOk) admits only the half-bar offset from a
        // bar-end departure.
        if (s > 0)
            for (int k = 1; k < std::max(1, in.bar_beats) && sec.b0 + k < sec.b1; ++k)
                push(sec.b0 + k, sec.b1, ShapePiece::Trim::Head);
        // Sesja 136 (ADR-117 step 5): the FIRST section's head pieces compete
        // with whole sections in every tier - Audition leaves the intro after
        // 1-6 s on every corpus case; tier A's whole 15 s intro forced Daft
        // Punk x0.25 / x0.33 onto rated-bad seams (q 0.26 / 0.40) while the
        // 9-beat intro reaches Audition's rated-clean landing (q 0.59). Head
        // pieces end at every downbeat and every half-bar of the section
        // (the half-bar end pairs with a half-bar zone landing under the
        // bar-phase rule; Audition's Daft Punk cut = beat 10 -> 285).
        if (s == 0 && in.bar_ends && in.db_set != nullptr) {
            const int half = std::max(1, in.bar_beats) / 2;
            for (auto it = in.db_set->upper_bound(sec.b0); it != in.db_set->end() && *it < sec.b1; ++it) {
                push(sec.b0, *it, ShapePiece::Trim::Head);
                if (half > 0 && *it + half < sec.b1) push(sec.b0, *it + half, ShapePiece::Trim::Head);
            }
        }
        if (! allow_trim) continue;
        for (auto it = ps.upper_bound(sec.b0); it != ps.end() && *it < sec.b1; ++it) {
            push(sec.b0, *it, ShapePiece::Trim::Head);
            push(*it, sec.b1, ShapePiece::Trim::Tail);
        }
        // Sesja 134: bar-granular ends - head pieces ending at every downbeat,
        // tail pieces starting at every downbeat (skipping the phrase starts
        // already pushed above); sesja 135: every section; sesja 136: head
        // pieces may also end at the half-bar (the departure offset that pairs
        // with a half-bar zone landing under the bar-phase rule), and a short
        // middle piece costs kShapeFragmentSteps (see the header).
        if (in.bar_ends && in.db_set != nullptr) {
            const int half = std::max(1, in.bar_beats) / 2;
            for (auto it = in.db_set->upper_bound(sec.b0); it != in.db_set->end() && *it < sec.b1; ++it) {
                if (! ps.count(*it)) {
                    if (s != 0) push(sec.b0, *it, ShapePiece::Trim::Head);   // s == 0: pushed above
                    push(*it, sec.b1, ShapePiece::Trim::Tail);
                }
                if (s != 0 && half > 0 && *it + half < sec.b1) push(sec.b0, *it + half, ShapePiece::Trim::Head);
            }
        }
    }
    std::sort(out.begin(), out.end(), [] (const Piece& a, const Piece& b) {
        return a.p.b0 != b.p.b0 ? a.p.b0 < b.p.b0 : a.p.b1 < b.p.b1;
    });
    return out;
}

double contextDb(const ShapePlannerInputs& in, int a, int b)
{
    a = std::max(0, a); b = std::min(in.n_beats, b);
    if (in.rms_energy == nullptr || b <= a) return 0.0;
    double m = 0.0;
    for (int k = a; k < b; ++k) m += in.rms_energy[k];
    m /= static_cast<double>(b - a);
    return 20.0 * std::log10(std::max(m, 1e-6));
}

// Substitution-view loudness excess: what preceded the landing in the
// original vs what precedes it in the remix (positive = the remix steps up
// more than the song did). The landing's own level cancels out.
double seamExcessDb(const ShapePlannerInputs& in, int i, int j)
{
    if (in.rms_energy == nullptr) return 0.0;
    const int K = kShapeContextBeats;
    return contextDb(in, j - K, j) - contextDb(in, i - K + 1, i + 1);
}

// Seam crossfade for the Renderer: `seam_crossfade_beats` beat periods at the
// landing (its predecessor's on the last beat), capped by
// `seam_crossfade_max_sec` (sesja 136) when set; 0 = the Renderer default.
double seamOverlap(const ShapePlannerInputs& in, int j)
{
    if (in.seam_crossfade_beats <= 0.0 || in.n_beats < 2) return 0.0;
    const int k = std::min(j, in.n_beats - 2);
    double period = in.beat_times[k + 1] - in.beat_times[k];
    if (period > kShapeSeamHalveAboveSec) period *= 0.5;   // sesja 136: the double-tempo beat on slow grids
    double ov = in.seam_crossfade_beats * period;
    if (in.seam_crossfade_max_sec > 0.0) ov = std::min(ov, in.seam_crossfade_max_sec);
    return ov;
}

// DP table over (piece, length bin, chorus kept). The chorus flag is part of
// the state: the no-chorus tax is applied when a plan is picked, so two plans
// of equal length that differ in whether a chorus survives must both reach the
// end (sesja 132, Avicii x0.25: the raw-cheaper intro + verse + verse -> ending
// plan (q 0.60) shadowed intro + verse + chorus -> ending (q 0.54) at the same
// bin and the tax never saw the chorus plan).
struct Search
{
    std::vector<Piece>  pieces;
    std::vector<State>  table;   // pieces x bins x 2
    int                 nb = 0;
    State& at(int p, int bin, int c) { return table[(static_cast<std::size_t>(p) * nb + bin) * 2 + c]; }
};

// Sesja 134: an OPEN seam lands on a section start or inside the last
// section (see ShapePlannerInputs::seam_open).
// Sesja 135 (DEV-121): an open seam is free of the q floor only when it lands
// in a section-start ZONE (the first bar of a section, see
// ShapePlannerInputs::bar_beats). Inside the last section beyond its first
// bar (a bar-granular ending piece) the open judge still drops every
// loudness gate, but the composite must reach `min_q`: on the rated
// inside-section landings the composite separates the ear's verdicts (bad
// 0.33 / 0.37 / 0.39 / 0.40 vs clean >= 0.48; Dance Monkey 47 -> 328, Daft
// Punk 127 -> 411, Tiesto 51 -> 308), while Audition's accepted section-start
// cut-ins score as low as 0.11.
bool sectionStartLanding(const ShapePlannerInputs& in, int j)
{
    const int zone = std::max(1, in.bar_beats);
    for (int s = 1; s < in.n_sections; ++s)
        if (j >= in.sections[s].b0 && j < std::min(in.sections[s].b1, in.sections[s].b0 + zone)) return true;
    return false;
}

// Sesja 135 (DEV-122): a landing on a lattice beat (see ShapePlannerInputs::
// beat_is_synthetic) is floor-free like a section-start zone.
bool latticeLanding(const ShapePlannerInputs& in, int j)
{
    return in.beat_is_synthetic != nullptr && j >= 0 && j < static_cast<int>(in.beat_is_synthetic->size())
        && (*in.beat_is_synthetic)[static_cast<std::size_t>(j)];
}

bool openLanding(const ShapePlannerInputs& in, int j)
{
    if (! in.seam_open) return false;
    if (in.n_sections > 0 && j >= in.sections[in.n_sections - 1].b0) return true;
    return sectionStartLanding(in, j) || latticeLanding(in, j);
}

// Sesja 136 (ADR-117 step 5): the bar-phase rule. A seam leaves at the START
// of beat c = i + 1 and lands on beat j; both sit at some offset inside their
// bar (beats since the last downbeat). Over the 75 rated seams of sesje
// 133-136 an odd offset change (1 or 3 beats in 4/4) was rated ok 4 / bad 15
// (the 4 = lattice or 2-beat-bar cases), the same offset or a half-bar change
// ok 33 / bad 23; Adobe Audition keeps the phase on 21 of its 22 clean cuts.
// The round-3 recipe broke it on 15 of 20 seams (0 ok, "nie w takt"), and the
// sesja-135 zone landings (+1 / +3 beats after a section start from a bar-end
// departure) broke it on Tiesto x0.25 and Daft Punk x0.33 (both "zgrzyt").
// Rule: 2 * (off(j) - off(c)) == 0 (mod bar_beats); a lattice beat on either
// side has no phase and is exempt.
int barOffset(const ShapePlannerInputs& in, int b)
{
    if (in.db_set == nullptr || in.db_set->empty()) return 0;
    auto it = in.db_set->upper_bound(b);
    if (it == in.db_set->begin()) return 0;
    --it;
    return (b - *it) % std::max(1, in.bar_beats);
}

bool phaseOk(const ShapePlannerInputs& in, int c, int j)
{
    const int bar = std::max(1, in.bar_beats);
    if (bar == 1 || latticeLanding(in, j) || latticeLanding(in, c)) return true;
    const int d = ((barOffset(in, j) - barOffset(in, c)) % bar + bar) % bar;
    if (d == 0) return true;
    // A half-bar change is admitted only when the half-bar is at least two
    // beats: on a 2-beat grid (Drake, a half-time detection) "half a bar" is
    // one beat = the off-beat, rated "nie w takt" in round 4.
    const int half = bar / 2;
    return half >= 2 && 2 * half == bar && d == half;
}

using SeamFn    = std::function<std::optional<ShapeSeamScore>(int, int)>;
struct SeamCache
{
    const SeamFn* fn = nullptr;
    std::map<std::pair<int, int>, std::optional<ShapeSeamScore>> map;
};

const std::optional<ShapeSeamScore>& seamScore(SeamCache& cache, int i, int j)
{
    auto it = cache.map.find({i, j});
    if (it == cache.map.end())
        it = cache.map.emplace(std::make_pair(i, j),
                               (cache.fn && *cache.fn) ? (*cache.fn)(i, j) : std::nullopt).first;
    return it->second;
}

void runSearch(const ShapePlannerInputs& in, Search& s, SeamCache& cache, SeamCache& open, double max_len)
{
    const int P = static_cast<int>(s.pieces.size());
    s.nb = static_cast<int>(std::ceil(max_len / kShapeBinSec)) + 2;
    s.table.assign(static_cast<std::size_t>(P) * s.nb * 2, State{});
    const double head = in.beat_times[0];
    auto binOf = [] (double sec) { return static_cast<int>(std::lround(sec / kShapeBinSec)); };
    auto put = [&] (int p, int bin, const State& st) {
        if (bin < 0 || bin >= s.nb) return;
        State& cur = s.at(p, bin, st.chorus ? 1 : 0);
        if (! cur.valid() || betterState(st, cur)) cur = st;
    };
    for (int p = 0; p < P; ++p) {
        const Piece& pc = s.pieces[p];
        if (pc.p.b0 != 0) continue;
        State st;
        st.cost   = pc.p.trim == ShapePiece::Trim::Whole ? 0.0 : kShapeTrimTax;
        st.chorus = pc.p.kind == kChorusKind;
        put(p, binOf(head + pc.dur), st);
    }
    for (int p = 0; p < P; ++p) {
        const Piece& pc = s.pieces[p];
        for (int bin = 0; bin < s.nb; ++bin)
        for (int c = 0; c < 2; ++c) {
            const State cur = s.at(p, bin, c);
            if (! cur.valid()) continue;
            for (int q = p + 1; q < P; ++q) {
                const Piece& qc = s.pieces[q];
                if (qc.p.b0 < pc.p.b1) continue;
                double add   = qc.p.trim == ShapePiece::Trim::Whole ? 0.0 : kShapeTrimTax;
                double min_q = cur.min_q;
                double min_q_all = cur.min_q_all;
                int    n_seams = cur.n_seams;
                int    ending_skips = cur.ending_skips;
                if (qc.p.b0 != pc.p.b1) {
                    const int i = pc.p.b1 - 1, j = qc.p.b0;
                    if (++n_seams > kShapeMaxSeams) continue;   // sesja 136: Audition never needs more
                    if (pc.p.kind == kOutroKind && qc.p.kind == kOutroKind) ++ending_skips;   // sesja 136
                    if (! phaseOk(in, i + 1, j)) continue;   // sesja 136: the bar-phase rule
                    const bool is_open    = openLanding(in, j);
                    const bool floor_free = is_open && (sectionStartLanding(in, j) || latticeLanding(in, j));   // sesja 135
                    const auto& sc = seamScore(is_open ? open : cache, i, j);
                    if (! sc.has_value() || (! floor_free && sc->q < in.min_q)) continue;
                    add  += (1.0 - sc->q) + kShapeSeamTax
                          + std::max(0.0, seamExcessDb(in, i, j)) / kShapeDynamicsDbPerCost;
                    if (! floor_free) min_q = std::min(min_q, sc->q);   // the floor skips section-start landings only
                    min_q_all = std::min(min_q_all, sc->q);
                }
                State st;
                st.cost = cur.cost + add; st.min_q = min_q; st.min_q_all = min_q_all; st.prev_p = p; st.prev_bin = bin; st.prev_c = c;
                st.n_seams = n_seams; st.ending_skips = ending_skips;
                st.fragments = cur.fragments
                    + ((qc.p.trim != ShapePiece::Trim::Whole && qc.p.b1 != in.n_beats && qc.p.b1 - qc.p.b0 < kShapeFragmentBeats) ? 1 : 0);
                st.chorus = cur.chorus || qc.p.kind == kChorusKind;
                put(q, bin + binOf(qc.dur), st);
            }
        }
    }
}

double closestDev(const ShapePlannerInputs& in, Search& s)
{
    double best = std::numeric_limits<double>::infinity();
    for (int p = 0; p < static_cast<int>(s.pieces.size()); ++p) {
        if (s.pieces[p].p.b1 != in.n_beats) continue;
        for (int bin = 0; bin < s.nb; ++bin)
        for (int c = 0; c < 2; ++c) {
            const State& st = s.at(p, bin, c);
            if (! st.valid() || st.min_q < in.min_q) continue;
            best = std::min(best, std::fabs(bin * kShapeBinSec - in.target_sec));
        }
    }
    return std::isfinite(best) ? best : -1.0;
}

int passing(const SeamCache& c, double min_q)
{
    int n = 0;
    for (const auto& kv : c.map) if (kv.second.has_value() && kv.second->q >= min_q) ++n;
    return n;
}

std::optional<ShapePlan> pickPlan(const ShapePlannerInputs& in, Search& s, SeamCache& cache, SeamCache& open,
                                  double window, bool has_chorus, char tier)
{
    const int P = static_cast<int>(s.pieces.size());
    const double head = in.beat_times[0];
    double best_cost = std::numeric_limits<double>::infinity();
    int best_bucket = std::numeric_limits<int>::min();
    int best_p = -1, best_bin = -1, best_c = 0;
    for (int p = 0; p < P; ++p) {
        if (s.pieces[p].p.b1 != in.n_beats) continue;
        for (int bin = 0; bin < s.nb; ++bin)
        for (int c = 0; c < 2; ++c) {
            const State& st = s.at(p, bin, c);
            if (! st.valid() || st.min_q < in.min_q) continue;
            const double dev = bin * kShapeBinSec - in.target_sec;
            if (std::fabs(dev) > window) continue;
            // Inside a window: the best worst seam wins (sesja 135 maximin),
            // the cheapest plan inside that step; with no window (tier F) the
            // closest length wins, cost breaks ties.
            const double cost = std::isfinite(window)
                ? st.cost + ((has_chorus && ! st.chorus) ? kShapeNoChorusTax : 0.0)
                : std::fabs(dev) * 1e3 + st.cost;
            // Sesja 136: the no-chorus tax is a COST tie-break only. In
            // maximin units (sesja 135: 0.5 = 10 steps) it forced Tiesto x0.25
            // from its rated-ok 1-seam plan (q 0.39, no chorus) into a 2-seam
            // chorus plan whose worst seam (q 0.25, a zone landing off the bar
            // phase) was rated "zgrzyt, dynamika, wokal".
            const int bucket = std::isfinite(window)
                ? stateBucket(st) - ((has_chorus && ! st.chorus) ? kShapeNoChorusTaxSteps : 0)
                : 0;
            if (bucket > best_bucket || (bucket == best_bucket && cost < best_cost)) {
                best_bucket = bucket; best_cost = cost; best_p = p; best_bin = bin; best_c = c;
            }
        }
    }
    if (best_p < 0) return std::nullopt;

    ShapePlan plan;
    plan.ok = true; plan.tier = tier; plan.cost = best_cost;
    int p = best_p, bin = best_bin, c = best_c;
    while (p >= 0) {
        const State& st = s.at(p, bin, c);
        plan.pieces.push_back(s.pieces[p].p);
        const int pp = st.prev_p, pb = st.prev_bin, pcx = st.prev_c;
        p = pp; bin = pb; c = pcx;
    }
    std::reverse(plan.pieces.begin(), plan.pieces.end());
    plan.est_sec = head;
    for (const auto& pc : plan.pieces) plan.est_sec += beatEndTime(in, pc.b1) - in.beat_times[pc.b0];
    plan.dev_sec = plan.est_sec - in.target_sec;
    for (std::size_t k = 1; k < plan.pieces.size(); ++k) {
        const auto& a = plan.pieces[k - 1];
        const auto& b = plan.pieces[k];
        if (b.b0 == a.b1) continue;
        ShapeSeam sm;
        sm.i = a.b1 - 1; sm.j = b.b0;
        sm.open      = openLanding(in, sm.j);
        sm.score     = seamScore(sm.open ? open : cache, sm.i, sm.j).value_or(ShapeSeamScore{});
        sm.excess_db = seamExcessDb(in, sm.i, sm.j);
        sm.overlap_sec = seamOverlap(in, sm.j);
        plan.min_q   = std::min(plan.min_q, sm.score.q);
        plan.seams.push_back(sm);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Sesja 136 (ADR-117 step 5): "Audition's recipe" - see ShapePlannerInputs::
// recipe_mode. Pieces are [0, c1) [j1, c2) ... [jk, n): every cut point c and
// landing j comes from the candidate set, the seam is (c - 1, j).
struct RState
{
    double min_piece = 1e9;   // shortest piece so far, seconds
    int    land      = 0;     // landing-class sum
    double min_q     = 1.0;   // worst seam so far
    int    prev_j    = 0, prev_bin = 0, cut_c = -1;   // back pointer: the previous piece's start + bin, this seam's cut point
};

int landingClass(const ShapePlannerInputs& in, int j)
{
    if (sectionStartLanding(in, j)) return 2;
    if (latticeLanding(in, j)) return 1;
    if (in.db_set != nullptr && (in.db_set->count(j) != 0 || in.db_set->count(j + 1) != 0)) return 1;
    return 0;
}

using RKey = std::array<long long, 5>;

long long bucket2s(double sec) { return static_cast<long long>(std::floor(sec / 2.0 + 1e-9)); }
long long bucketQ(double q)    { return static_cast<long long>(std::floor(q * 10.0 + 1e-9)); }

// Dominance inside one (piece start, length bin) state.
RKey recipeDomKey(const ShapePlannerInputs& in, const RState& st)
{
    if (in.recipe_mode == 2) return { bucketQ(st.min_q), bucket2s(st.min_piece), st.land, 0, 0 };
    return { bucket2s(st.min_piece), st.land, bucketQ(st.min_q), 0, 0 };
}

// Ranking of complete plans with the same seam count.
RKey recipeFinalKey(const ShapePlannerInputs& in, const RState& st, double last_sec, double dev_sec)
{
    const long long closeness = -static_cast<long long>(std::llround(std::fabs(dev_sec) * 1000.0));
    if (in.recipe_mode == 1) return { bucket2s(last_sec), bucket2s(st.min_piece), st.land, closeness, bucketQ(st.min_q) };
    if (in.recipe_mode == 2) return { bucketQ(st.min_q), bucket2s(st.min_piece), st.land, closeness, 0 };
    return { bucket2s(st.min_piece), st.land, closeness, bucketQ(st.min_q), 0 };
}

std::optional<ShapePlan> planRecipe(const ShapePlannerInputs& in, SeamCache& open, ShapePlan::Diag& diag)
{
    const int n = in.n_beats;
    std::set<int> C;
    if (in.db_set != nullptr)
        for (const int d : *in.db_set) {
            if (d >= 1 && d < n) C.insert(d);
            if (d - 1 >= 1 && d - 1 < n) C.insert(d - 1);   // the pickup beat before a bar line
        }
    const int zone = std::max(1, in.bar_beats);
    for (int s = 0; s < in.n_sections; ++s)
        for (int k = 0; k < zone; ++k) {
            const int b = in.sections[s].b0 + k;
            if (b >= 1 && b < n && b < in.sections[s].b1) C.insert(b);
        }
    const std::vector<int> cands(C.begin(), C.end());
    diag.pieces_whole = static_cast<int>(cands.size());
    const int    minP  = std::max(1, in.recipe_min_piece_beats);
    const double slack = in.window_sec;
    const int    nb    = static_cast<int>(std::ceil((in.target_sec + slack) / kShapeBinSec)) + 2;
    auto binOf    = [] (double sec) { return static_cast<int>(std::lround(sec / kShapeBinSec)); };
    auto pieceLen = [&] (int j, int c) {
        const double a = (j == 0) ? 0.0 : in.beat_times[j];
        const double b = (c >= n) ? in.track_sec : in.beat_times[c];
        return b - a;
    };
    auto seamOk = [&] (int i, int j) -> const ShapeSeamScore* {
        if (! phaseOk(in, i + 1, j)) return nullptr;         // sesja 136: the bar-phase rule
        const auto& sc = seamScore(open, i, j);
        if (! sc.has_value() || sc->q < in.recipe_q_floor) return nullptr;
        return &*sc;
    };

    using Key = std::pair<int, int>;   // (piece start j, length bin at that start)
    const int maxSeams = std::max(1, in.recipe_max_seams);
    std::vector<std::map<Key, RState>> layers(static_cast<std::size_t>(maxSeams) + 1);
    layers[0][{0, 0}] = RState{};
    struct Best { RKey key {}; int k = 0, j = 0, bin = 0; double est = 0.0; RState st; bool set = false; } best;
    for (int k = 0; k < maxSeams; ++k) {
        auto& nxt = layers[static_cast<std::size_t>(k) + 1];
        for (const auto& [key, st] : layers[static_cast<std::size_t>(k)]) {
            const int j = key.first, bin = key.second;
            for (const int c : cands) {
                if (c <= j || c - j < minP) continue;
                const double plen = pieceLen(j, c);
                const int nbin = bin + binOf(plen);
                if (nbin >= nb) break;                      // cands ascend: every later cut is longer
                for (const int j2 : cands) {
                    if (j2 <= c) continue;
                    const ShapeSeamScore* sc = seamOk(c - 1, j2);
                    if (sc == nullptr) continue;
                    RState st2;
                    st2.min_piece = std::min(st.min_piece, plen);
                    st2.land      = st.land + landingClass(in, j2);
                    st2.min_q     = std::min(st.min_q, sc->q);
                    st2.prev_j = j; st2.prev_bin = bin; st2.cut_c = c;
                    auto it = nxt.find({j2, nbin});
                    if (it == nxt.end() || recipeDomKey(in, st2) > recipeDomKey(in, it->second)) nxt[{j2, nbin}] = st2;
                }
            }
        }
        for (const auto& [key, st] : nxt) {                   // close with the ending piece [j, n)
            const int j = key.first, bin = key.second;
            if (n - j < minP) continue;
            const double last = pieceLen(j, n);
            const double est  = bin * kShapeBinSec + last;
            const double dev  = est - in.target_sec;
            if (std::fabs(dev) > slack) continue;
            RState fin = st; fin.min_piece = std::min(st.min_piece, last);
            const RKey fk = recipeFinalKey(in, fin, last, dev);
            if (! best.set || fk > best.key) best = Best { fk, k + 1, j, bin, est, fin, true };
        }
        if (best.set) break;                                  // the fewest seams win outright
    }
    diag.seams_tried  = static_cast<int>(open.map.size());
    diag.seams_strict = passing(open, in.recipe_q_floor);
    for (const auto& kv : open.map) {
        ShapePlan::Diag::Judged jd{kv.first.first, kv.first.second, false, false, -1.0};
        if (kv.second.has_value()) { jd.q = kv.second->q; jd.strict_ok = jd.relaxed_ok = kv.second->q >= in.recipe_q_floor; }
        diag.judged.push_back(jd);
    }
    if (! best.set) return std::nullopt;

    std::vector<std::pair<int, int>> cuts;                   // (i, j) in plan order
    {
        int j = best.j, bin = best.bin;
        for (int k = best.k; k >= 1; --k) {
            const RState& st = layers[static_cast<std::size_t>(k)].at({j, bin});
            cuts.emplace_back(st.cut_c - 1, j);
            j = st.prev_j; bin = st.prev_bin;
        }
        std::reverse(cuts.begin(), cuts.end());
    }
    auto sectionOf = [&] (int b) {
        for (int s = 0; s < in.n_sections; ++s)
            if (b >= in.sections[s].b0 && b < in.sections[s].b1) return s;
        return in.n_sections - 1;
    };
    ShapePlan plan;
    plan.ok = true; plan.tier = 'R'; plan.cost = static_cast<double>(cuts.size());
    int start = 0;
    for (std::size_t k = 0; k <= cuts.size(); ++k) {
        const int end = (k < cuts.size()) ? cuts[k].first + 1 : n;
        ShapePiece pc;
        pc.section = sectionOf(start); pc.b0 = start; pc.b1 = end;
        const ShapeSection& sec = in.sections[pc.section];
        pc.kind = sec.kind;
        pc.trim = (start == sec.b0 && end == sec.b1) ? ShapePiece::Trim::Whole
                : (start == sec.b0 ? ShapePiece::Trim::Head : ShapePiece::Trim::Tail);
        plan.pieces.push_back(pc);
        if (k < cuts.size()) start = cuts[k].second;
    }
    plan.est_sec = best.est; plan.dev_sec = best.est - in.target_sec;
    for (const auto& [i, j] : cuts) {
        ShapeSeam sm;
        sm.i = i; sm.j = j; sm.open = true;
        sm.score       = seamScore(open, i, j).value_or(ShapeSeamScore{});
        sm.excess_db   = seamExcessDb(in, i, j);
        sm.overlap_sec = seamOverlap(in, j);
        plan.min_q     = std::min(plan.min_q, sm.score.q);
        plan.seams.push_back(sm);
    }
    plan.diag = diag;
    return plan;
}

} // namespace

RemixPath ShapePlan::toPath() const
{
    RemixPath path;
    for (const auto& pc : pieces)
        for (int b = pc.b0; b < pc.b1; ++b) path.beat_indices.push_back(b);
    for (const auto& sm : seams) {
        path.transitions.emplace_back(sm.i, sm.j);
        auto& md = path.transition_metadata[{sm.i, sm.j}];
        md["quality_score"]  = sm.score.q;
        md["energy_diff_db"] = sm.score.energy_diff_db;
        md["edge_distance"]  = sm.score.edge_distance;
        md["family"]         = 1.0;
        md["shape_seam"]     = 1.0;
        md["open_seam"]      = sm.open ? 1.0 : 0.0;
        if (sm.overlap_sec > 0.0) md["preferred_overlap_sec"] = sm.overlap_sec;
    }
    path.duration_beats = static_cast<int>(path.beat_indices.size());
    path.total_cost     = cost;
    return path;
}

ShapePlan planShape(const ShapePlannerInputs& in)
{
    ShapePlan none;
    if (in.beat_times == nullptr || in.n_beats < 2 || in.sections == nullptr
        || in.n_sections < kShapeMinSections || in.target_sec <= 0.0 || ! in.seam)
        return none;
    if (in.sections[0].b0 != 0 || in.sections[in.n_sections - 1].b1 != in.n_beats) return none;

    bool has_chorus = false;
    for (int s = 0; s < in.n_sections; ++s) has_chorus = has_chorus || in.sections[s].kind == kChorusKind;

    // Sesja 136: Audition's recipe replaces the tiers when asked for.
    if (in.recipe_mode != 0) {
        if (! in.seam_open) return none;
        SeamCache ropen; ropen.fn = &in.seam_open;
        ShapePlan::Diag rdiag;
        if (auto p = planRecipe(in, ropen, rdiag)) return *p;
        none.diag = rdiag;
        return none;
    }

    const std::set<int> ps = phraseStarts(in);
    const double max_len = in.target_sec + std::max(in.window_sec, in.window_relaxed_sec) + kShapeBinSec;
    SeamCache strict;  strict.fn  = &in.seam;
    SeamCache relaxed; relaxed.fn = &in.seam_relaxed;
    SeamCache open;    open.fn    = &in.seam_open;   // sesja 134: open seams, every tier

    ShapePlan::Diag diag;
    auto finish = [&] (ShapePlan p) {
        diag.seams_tried   = static_cast<int>(strict.map.size());
        diag.seams_strict  = passing(strict, in.min_q);
        diag.seams_relaxed = passing(relaxed, in.min_q);
        for (const auto& kv : strict.map) {
            ShapePlan::Diag::Judged jd{kv.first.first, kv.first.second, false, false, -1.0};
            if (kv.second.has_value()) { jd.q = kv.second->q; jd.strict_ok = kv.second->q >= in.min_q; }
            auto rit = relaxed.map.find(kv.first);
            if (rit != relaxed.map.end() && rit->second.has_value()) {
                jd.relaxed_ok = rit->second->q >= in.min_q;
                if (jd.q < 0.0) jd.q = rit->second->q;
            }
            diag.judged.push_back(jd);
        }
        p.diag = diag;
        return p;
    };

    // Sesja 136 (ADR-117 step 5): ONE search over whole sections and every
    // trim (phrase / bar / half-bar heads, bar tails, zone heads) - the tiers
    // A (whole first) and B (trims) are merged, maximin picks the plan with
    // the best worst seam across both, the trim tax breaks ties. Round 2
    // (sesja 135, 0/7 vs Audition): tier A's whole 15 s intro forced Daft
    // Punk onto rated-bad seams while a 9-beat intro reached Audition's
    // rated-clean landing; the tier letter now only reports whether a
    // trimmed piece is in the plan ('B') or not ('A').
    Search wide;
    wide.pieces = buildPieces(in, true, ps);
    for (const auto& pc : wide.pieces)
        (pc.p.trim == ShapePiece::Trim::Whole ? diag.pieces_whole : diag.pieces_trim)++;
    auto letter = [] (ShapePlan& p, char c) {
        if (c == 'A' || c == 'B') {
            bool trimmed = false;
            for (const auto& pc : p.pieces) trimmed = trimmed || pc.trim != ShapePiece::Trim::Whole;
            p.tier = trimmed ? 'B' : 'A';
        }
        return p;
    };
    runSearch(in, wide, strict, open, max_len);
    diag.closest_dev_whole = closestDev(in, wide);
    diag.closest_dev_trim  = diag.closest_dev_whole;
    if (auto p = pickPlan(in, wide, strict, open, in.window_sec, has_chorus, 'A')) return finish(letter(*p, 'A'));
    if (auto p = pickPlan(in, wide, strict, open, in.window_relaxed_sec, has_chorus, 'C')) return finish(*p);
    if (in.seam_relaxed) {
        runSearch(in, wide, relaxed, open, max_len);
        diag.closest_dev_trim = closestDev(in, wide);
        if (auto p = pickPlan(in, wide, relaxed, open, in.window_sec, has_chorus, 'D')) return finish(*p);
        if (auto p = pickPlan(in, wide, relaxed, open, in.window_relaxed_sec, has_chorus, 'E')) return finish(*p);
    }

    // Tier F: the complete plan closest to the target at any length (the
    // table is widened so long plans are represented too); the relaxed judge
    // when the caller has one, the strict judge otherwise.
    SeamCache& fcache = in.seam_relaxed ? relaxed : strict;
    Search effort;
    effort.pieces = wide.pieces;
    runSearch(in, effort, fcache, open, std::max(max_len, 2.0 * in.target_sec + 20.0));
    if (auto p = pickPlan(in, effort, fcache, open, std::numeric_limits<double>::infinity(), has_chorus, 'F')) {
        p->ok = false; p->best_effort = true;
        return finish(*p);
    }
    return finish(none);
}

std::vector<ShapeSection> shapeSectionsFromSeconds(const double* beat_times, int n_beats,
                                                   const double* starts, const double* ends,
                                                   const int* kinds, int n_sections)
{
    std::vector<ShapeSection> out;
    if (beat_times == nullptr || n_beats <= 0 || n_sections <= 0) return out;
    auto firstAtOrAfter = [&] (double t) {
        int b = 0;
        while (b < n_beats && beat_times[b] < t - 1e-3) ++b;
        return b;
    };
    for (int s = 0; s < n_sections; ++s) {
        ShapeSection sec;
        sec.b0   = firstAtOrAfter(starts[s]);
        sec.b1   = (s + 1 < n_sections) ? firstAtOrAfter(ends[s]) : n_beats;
        sec.kind = kinds[s];
        out.push_back(sec);
    }
    out.front().b0 = 0;
    for (std::size_t s = 1; s < out.size(); ++s) out[s].b0 = out[s - 1].b1;
    out.erase(std::remove_if(out.begin(), out.end(), [] (const ShapeSection& x) { return x.b1 <= x.b0; }),
              out.end());
    if (! out.empty()) { out.front().b0 = 0; out.back().b1 = n_beats; }
    return out;
}

} // namespace reamix::remix
