#include "remix/TransitionPrescreen.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

#include "analysis/Recurrence.h"
#include "dsp/WaveformXcorr.h"

namespace reamix::remix {

namespace {

// PARITY: transition_prescreen.py:91-94 — convert seg.{start,end} seconds
// to beat indices via argmin(|beat_times - t|) + clamp end ≥ start+1.
struct SegBeats
{
    int s;        // start beat
    int e;        // end beat (exclusive-ish, post clamp)
    // Python keeps label too but prescreen doesn't read it after assignment,
    // so we drop it. Check: transition_prescreen.py:105 takes label into
    // `label_i, label_j` but never uses them after. Verified by grep in
    // session-24 audit.
};

int argminAbs(const double* beat_times, int n_beats, double target)
{
    int    best_i = 0;
    double best_v = std::abs(beat_times[0] - target);
    for (int i = 1; i < n_beats; ++i) {
        const double v = std::abs(beat_times[i] - target);
        if (v < best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

} // namespace

std::vector<PrescreenedTransition>
prescreenTransitions(const PrescreenInputs& in)
{
    std::vector<PrescreenedTransition> empty;

    // PARITY: transition_prescreen.py:69-71 — `n < 8 OR n_segments < 2 → []`.
    if (in.features == nullptr || in.n_beats < PRESCREEN_MIN_BEATS ||
        in.n_segments < 2 || in.segments == nullptr)
    {
        return empty;
    }

    const int n = in.n_beats;

    // PARITY: transition_prescreen.py:73-74 — build_recurrence_matrix with
    // k_neighbors=12 (CRITICAL CALLER OVERRIDE, 5th confirmation of pattern).
    auto recResult = analysis::Recurrence::build(in.features,
                                                 n,
                                                 in.n_features,
                                                 PRESCREEN_K_NEIGHBORS);
    const double* R = recResult.R.data();

    // PARITY: transition_prescreen.py:77-84 — waveform verification setup.
    // has_wf requires all three conditions; max_lag = 3 ms in samples.
    const bool has_wf =
        in.boundary_waveforms != nullptr &&
        in.waveform_sample_rate > 0 &&
        in.n_boundary_waveforms >= n;

    int max_lag = 0;
    if (has_wf) {
        max_lag = static_cast<int>(PRESCREEN_MAX_LAG_MS *
                                   static_cast<double>(in.waveform_sample_rate) / 1000.0);
    }

    // PARITY: transition_prescreen.py:87-94 — seg_beats tuples.
    std::vector<SegBeats> seg_beats;
    seg_beats.reserve(static_cast<std::size_t>(in.n_segments));
    for (int k = 0; k < in.n_segments; ++k) {
        const auto& seg = in.segments[k];
        // Note: Python uses `seg.get("start", 0.0)` / `"end"` dict access; C++
        // Segment struct has direct `startSec` / `endSec` fields. Requires
        // beat_times != nullptr (checked: pre-entry gate + Segment is a
        // phase-3 C++ type, we always have beat_times on the production path).
        const int s = argminAbs(in.beat_times, n, seg.start);
        int       e = argminAbs(in.beat_times, n, seg.end);
        e = std::max(s + 1, std::min(e, n));
        seg_beats.push_back({s, e});
    }

    // PARITY: transition_prescreen.py:97-98 — search_radius = 4 × TS.
    const int search_radius = PRESCREEN_SEARCH_BARS * in.time_signature;

    // PARITY: transition_prescreen.py:100-145 — outer loop over segment pairs.
    // Collects ALL diagonals (before dedup). `results` analog.
    std::vector<PrescreenedTransition> results;

    for (int idx = 0; idx < in.n_segments; ++idx) {
        for (int jdx = 0; jdx < in.n_segments; ++jdx) {
            if (idx == jdx) continue;

            const int e_i = seg_beats[idx].e;
            const int s_j = seg_beats[jdx].s;
            // s_i + e_j unused — prescreen only centers search on exit(i) +
            // entry(j). Python transition_prescreen.py:105-106 binds them
            // to locals `s_i, e_i, label_i` via tuple unpack but the loop
            // never reads `s_i` or `label_i` afterward. Port drops them.

            // PARITY: transition_prescreen.py:109-112 — search window around
            // exit(i) → entry(j), ±search_radius each side.
            const int row_start = std::max(0, e_i - search_radius);
            const int row_end   = std::min(n, e_i + search_radius);
            const int col_start = std::max(0, s_j - search_radius);
            const int col_end   = std::min(n, s_j + search_radius);

            // PARITY: transition_prescreen.py:114-115 — degenerate skip.
            if (row_end <= row_start || col_end <= col_start) continue;

            // PARITY: transition_prescreen.py:117-121 — findDiagonals with
            // min_length = time_signature (≥1 bar) and bar_size = TS.
            auto diags = analysis::Recurrence::findDiagonals(
                R, n,
                row_start, row_end,
                col_start, col_end,
                in.time_signature,  // min_length
                in.time_signature   // bar_size
            );

            // PARITY: transition_prescreen.py:123 — keep top 3 per pair.
            const int keep = std::min<int>(PRESCREEN_TOP_DIAGONALS_PER_PAIR,
                                           static_cast<int>(diags.size()));
            for (int di = 0; di < keep; ++di) {
                const auto& d = diags[di];

                // PARITY: transition_prescreen.py:124-125 — mean_sim gate.
                if (d.meanSim < PRESCREEN_MIN_RECURRENCE_SIM) continue;

                // PARITY: transition_prescreen.py:128-134 — waveform verification.
                double wf_sim = 0.0;
                // has_wf AND r_mid + 1 < n guards (Python uses `min(r_mid+1, n-1)`
                // when indexing to avoid OOB — port mirrors via min-clamp).
                if (has_wf && d.rMid + 1 < n) {
                    const int src_row = std::min(d.rMid + 1, n - 1);
                    const int tgt_row = d.cMid;
                    const float* src_wave =
                        in.boundary_waveforms +
                        static_cast<std::size_t>(src_row) * in.n_samples_per_bnd;
                    const float* tgt_wave =
                        in.boundary_waveforms +
                        static_cast<std::size_t>(tgt_row) * in.n_samples_per_bnd;
                    auto [sim, _lag] = dsp::WaveformXcorr::compute(
                        src_wave, tgt_wave,
                        static_cast<std::size_t>(in.n_samples_per_bnd),
                        static_cast<std::size_t>(in.n_samples_per_bnd),
                        max_lag);
                    wf_sim = sim;
                }

                // PARITY: transition_prescreen.py:136-137 — waveform gate fires
                // ONLY when has_wf=true (Python guards with `if has_wf and …`).
                if (has_wf && wf_sim < in.min_waveform_similarity) continue;

                PrescreenedTransition t;
                t.from_beat           = d.rMid;
                t.to_beat             = d.cMid;
                t.diagonal_length     = d.length;
                t.recurrence_score    = d.meanSim;
                t.waveform_similarity = wf_sim;
                results.push_back(t);
            }
        }
    }

    // PARITY: transition_prescreen.py:147-152 — deduplicate by (from, to),
    // keeping highest recurrence_score. std::map gives sorted iteration by
    // key; replaces Python dict with identical semantic (insertion-order loss
    // is irrelevant here because we sort again below).
    std::map<std::pair<int, int>, PrescreenedTransition> best;
    for (const auto& t : results) {
        const std::pair<int, int> key{t.from_beat, t.to_beat};
        auto it = best.find(key);
        if (it == best.end()) {
            best.emplace(key, t);
        } else if (t.recurrence_score > it->second.recurrence_score) {
            it->second = t;
        }
    }

    // PARITY: transition_prescreen.py:155-158 — final sort DESC by
    // length × rec_score × max(0.1, waveform_sim). Python `sorted(...)` is
    // stable (Timsort); dict iteration order on Python 3.7+ is insertion
    // order but here we iterate `best.values()` which follows first-insert
    // order. std::map iterates by key order → breaks tie resolution. We use
    // std::stable_sort over a FIRST-SEEN-ORDER vector to match Python's
    // stable tie-break, which requires preserving insertion order of `best`.
    //
    // Implementation: after dedup, rebuild output in FIRST-SEEN-ORDER via a
    // secondary ordering map. Python's `best.values()` on 3.7+ iterates in
    // insertion order — duplicate replacements do NOT change the insertion
    // slot, so keep the first-insertion index.
    //
    // Session-24 note: sort key ties are vanishingly rare on realistic
    // corpus (fp64 product of 3 f64 scalars) — the stability choice only
    // matters for degenerate cases but the explicit handling avoids silent
    // drift if a tie ever surfaces.
    std::vector<std::pair<int, PrescreenedTransition>> ordered;
    ordered.reserve(best.size());
    std::map<std::pair<int, int>, int> first_seen_idx;
    int insert_counter = 0;
    for (const auto& t : results) {
        const std::pair<int, int> key{t.from_beat, t.to_beat};
        if (first_seen_idx.find(key) == first_seen_idx.end()) {
            first_seen_idx.emplace(key, insert_counter++);
        }
    }
    for (const auto& kv : best) {
        const int insert_idx = first_seen_idx.at(kv.first);
        ordered.emplace_back(insert_idx, kv.second);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<PrescreenedTransition> out;
    out.reserve(ordered.size());
    for (auto& p : ordered) out.push_back(p.second);

    std::stable_sort(out.begin(), out.end(),
        [](const PrescreenedTransition& a, const PrescreenedTransition& b) {
            const double ka = static_cast<double>(a.diagonal_length) *
                              a.recurrence_score *
                              std::max(PRESCREEN_SORT_WF_FLOOR, a.waveform_similarity);
            const double kb = static_cast<double>(b.diagonal_length) *
                              b.recurrence_score *
                              std::max(PRESCREEN_SORT_WF_FLOOR, b.waveform_similarity);
            return ka > kb;  // DESC
        });

    return out;
}

} // namespace reamix::remix
