// ShapePlanner.cpp — sesja 131 (ADR-117). See ShapePlanner.h.
#include "remix/ShapePlanner.h"

#include <algorithm>
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
    double min_q     = 1.0;
    int    prev_p    = -1;
    int    prev_bin  = -1;
    int    prev_c    = 0;
    bool   chorus    = false;
    bool   valid() const noexcept { return std::isfinite(cost); }
};

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
            if (trim != ShapePiece::Trim::Whole && b1 - b0 < kShapeMinEndBeats) return;   // sesja 134
            Piece pc;
            pc.p.section = s; pc.p.b0 = b0; pc.p.b1 = b1; pc.p.kind = sec.kind; pc.p.trim = trim;
            pc.dur = beatEndTime(in, b1) - in.beat_times[b0];
            out.push_back(pc);
        };
        push(sec.b0, sec.b1, ShapePiece::Trim::Whole);
        if (! allow_trim) continue;
        for (auto it = ps.upper_bound(sec.b0); it != ps.end() && *it < sec.b1; ++it) {
            push(sec.b0, *it, ShapePiece::Trim::Head);
            push(*it, sec.b1, ShapePiece::Trim::Tail);
        }
        // Sesja 134: bar-granular ends. First section: head pieces ending at
        // every downbeat; last section: tail pieces starting at every downbeat
        // (both skip the phrase starts already pushed above).
        if (in.bar_ends && in.db_set != nullptr && (s == 0 || s == in.n_sections - 1)) {
            for (auto it = in.db_set->upper_bound(sec.b0); it != in.db_set->end() && *it < sec.b1; ++it) {
                if (ps.count(*it)) continue;
                if (s == 0)                 push(sec.b0, *it, ShapePiece::Trim::Head);
                if (s == in.n_sections - 1) push(*it, sec.b1, ShapePiece::Trim::Tail);
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
bool openLanding(const ShapePlannerInputs& in, int j)
{
    if (! in.seam_open) return false;
    if (in.n_sections > 0 && j >= in.sections[in.n_sections - 1].b0) return true;
    for (int s = 0; s < in.n_sections; ++s) if (in.sections[s].b0 == j) return true;
    return false;
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
        if (! cur.valid() || st.cost < cur.cost) cur = st;
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
                if (qc.p.b0 != pc.p.b1) {
                    const int i = pc.p.b1 - 1, j = qc.p.b0;
                    const bool is_open = openLanding(in, j);
                    const auto& sc = seamScore(is_open ? open : cache, i, j);
                    if (! sc.has_value() || (! is_open && sc->q < in.min_q)) continue;
                    add  += (1.0 - sc->q) + kShapeSeamTax
                          + std::max(0.0, seamExcessDb(in, i, j)) / kShapeDynamicsDbPerCost;
                    if (! is_open) min_q = std::min(min_q, sc->q);   // the floor tracks gated seams only
                }
                State st;
                st.cost = cur.cost + add; st.min_q = min_q; st.prev_p = p; st.prev_bin = bin; st.prev_c = c;
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
    int best_p = -1, best_bin = -1, best_c = 0;
    for (int p = 0; p < P; ++p) {
        if (s.pieces[p].p.b1 != in.n_beats) continue;
        for (int bin = 0; bin < s.nb; ++bin)
        for (int c = 0; c < 2; ++c) {
            const State& st = s.at(p, bin, c);
            if (! st.valid() || st.min_q < in.min_q) continue;
            const double dev = bin * kShapeBinSec - in.target_sec;
            if (std::fabs(dev) > window) continue;
            // Inside a window the cheapest plan wins; with no window (tier F)
            // the closest length wins, cost breaks ties.
            const double cost = std::isfinite(window)
                ? st.cost + ((has_chorus && ! st.chorus) ? kShapeNoChorusTax : 0.0)
                : std::fabs(dev) * 1e3 + st.cost;
            if (cost < best_cost) { best_cost = cost; best_p = p; best_bin = bin; best_c = c; }
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
        if (in.seam_crossfade_beats > 0.0 && in.n_beats > 1) {
            const int k = std::min(sm.j, in.n_beats - 2);   // the beat period at the landing (its predecessor's on the last beat)
            sm.overlap_sec = in.seam_crossfade_beats * (in.beat_times[k + 1] - in.beat_times[k]);
        }
        plan.min_q   = std::min(plan.min_q, sm.score.q);
        plan.seams.push_back(sm);
    }
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

    Search whole;
    whole.pieces = buildPieces(in, false, ps);
    diag.pieces_whole = static_cast<int>(whole.pieces.size());
    runSearch(in, whole, strict, open, max_len);
    diag.closest_dev_whole = closestDev(in, whole);
    if (auto p = pickPlan(in, whole, strict, open, in.window_sec, has_chorus, 'A')) return finish(*p);

    Search trimmed;
    trimmed.pieces = buildPieces(in, true, ps);
    const bool trims = trimmed.pieces.size() > whole.pieces.size();
    diag.pieces_trim = trims ? static_cast<int>(trimmed.pieces.size()) : 0;
    if (trims) {
        runSearch(in, trimmed, strict, open, max_len);
        diag.closest_dev_trim = closestDev(in, trimmed);
        if (auto p = pickPlan(in, trimmed, strict, open, in.window_sec, has_chorus, 'B')) return finish(*p);
        if (auto p = pickPlan(in, trimmed, strict, open, in.window_relaxed_sec, has_chorus, 'C')) return finish(*p);
    } else if (auto p = pickPlan(in, whole, strict, open, in.window_relaxed_sec, has_chorus, 'C')) {
        return finish(*p);
    }
    Search& wide = trims ? trimmed : whole;
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
