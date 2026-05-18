#include "remix/ViterbiDP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace reamix::remix {

namespace {

// ---------------------------------------------------------------------------
// Constants — VERBATIM port of viterbi_dp.py L24-35 + L178.
// ---------------------------------------------------------------------------
//
// Per session-20 pre-port audit (weights-audit.md):
//   - INF                       CLEAN      mathematical DP sentinel
//   - COOLDOWN_BARS             CLEAN      meter-aware × time_signature
//   - MIN_SEQ_AFTER_JUMP        FALLBACK   caller overrides via optimizer
//   - MIN_FORWARD_JUMP_BEATS    FALLBACK   caller overrides via optimizer
//   - BACKWARD_PENALTY          UNJUSTIFIED (LOW)   session-17 classification
//   - JUMP_PENALTY_BASE         UNJUSTIFIED (HIGH)  dominates candidate weighting
//   - BOUNDARY_BONUS            CLEAN (config.py:98-102 inline rationale)
//   - SECTION_PENALTY           UNJUSTIFIED (HIGH)  × section_scale can reach 1.8
//   - DURATION_TOLERANCE        DEAD       (declared but never read in remix/ — not ported)
//   - DURATION_PENALTY_WEIGHT   UNJUSTIFIED (MEDIUM) distinct from config.duration_penalty_weight=10.0
//   - SPAN_PENALTY_THRESHOLD    UNJUSTIFIED (MEDIUM)
//   - LINF_PENALTY_WEIGHT       UNJUSTIFIED (HIGH)  path-shape dominator
//
// All placed in anonymous namespace to avoid INF collision with
// `TransitionCost.h` in future CleanOptimizer.cpp (session 21+).

// Source-of-truth: viterbi_dp.py:24 (2026-04-21).
constexpr double INF = 1e9;

// Source-of-truth: viterbi_dp.py:29 (2026-04-21).
// UNJUSTIFIED per weights-audit.md (LOW priority). Additive cost on backward
// jumps when shortening; ~2× typical transition cost.
constexpr double BACKWARD_PENALTY = 4.0;

// JUMP_PENALTY_BASE moved to ViterbiDP.h public constexpr (session 23:
// 2nd TU consumer = RegionOptimizer.cpp, Hard Rule #1 single-source-of-
// truth). Use `reamix::remix::JUMP_PENALTY_BASE` directly below.

// Source-of-truth: viterbi_dp.py:31 + config.py:98-102 (2026-04-21).
// CLEAN — config.py docstring provides motivation ("Boundary jumps pay
// only 10 % of jump_penalty"). Applied as `(1 - BOUNDARY_BONUS)` multiplier.
constexpr double BOUNDARY_BONUS = 0.9;

// Source-of-truth: viterbi_dp.py:32 (2026-04-21).
// UNJUSTIFIED (HIGH). Scaled by `max(1, 2 - 2·target_ratio)` — reaches 1.8
// at aggressive shortening (ratio 0.1).
constexpr double SECTION_PENALTY = 1.0;

// DURATION_PENALTY_WEIGHT moved to ViterbiDP.h public constexpr (session
// 23: 2nd TU consumer = RegionOptimizer.cpp, Hard Rule #1). Use
// `reamix::remix::DURATION_PENALTY_WEIGHT` directly below.

// Source-of-truth: viterbi_dp.py:35 (2026-04-21).
// UNJUSTIFIED (MEDIUM). Threshold above which span penalty `(span - 0.7)² × 3.0`
// fires.
constexpr double SPAN_PENALTY_THRESHOLD = 0.7;

// Source-of-truth: viterbi_dp.py:178 (2026-04-21). Function-scope in Python;
// hoisted to module scope in C++ (no semantic change — compile-time constant).
// UNJUSTIFIED (HIGH). Penalizes the SINGLE WORST transition in the path.
constexpr double LINF_PENALTY_WEIGHT = 8.0;

// Inline magic literals below are all UNJUSTIFIED per session-20 audit
// (see weights-audit.md session-20 entries). Hoisted to named constexpr
// here for (1) traceability — each appears once, not scattered across the
// DP kernel — and (2) harness-readable symbol names. Values unchanged
// from Python; cited per-constant.

// Source-of-truth: viterbi_dp.py:185 (2026-04-21). Clamps adaptive
// `jump_penalty_scale` when target_ratio shrinks below 0.15.
constexpr double JUMP_PENALTY_SCALE_FLOOR = 0.15;

// Source-of-truth: viterbi_dp.py:185 (2026-04-21). Upper clamp (ratio ≥ 1
// means "no shortening", no discount).
constexpr double JUMP_PENALTY_SCALE_CEILING = 1.0;

// Source-of-truth: viterbi_dp.py:189 (2026-04-21). `ssj[1, 0] = 999`
// sentinel for "no recent jump" — must exceed any plausible min_seq_after_jump.
constexpr std::int64_t SSJ_NO_RECENT_JUMP_SENTINEL = 999;

// Source-of-truth: viterbi_dp.py:266 (2026-04-21). Aggressive-shortening
// sequential-penalty activation threshold — below this ratio, the DP
// nudges toward jumps to spend duration budget.
constexpr double SEQUENTIAL_PENALTY_ACTIVATION_RATIO = 0.5;

// Source-of-truth: viterbi_dp.py:267 (2026-04-21). Coefficient inside the
// `seq_penalty = 0.02 × (0.5 - target_ratio)` nudge.
constexpr double SEQUENTIAL_PENALTY_COEFF = 0.02;

// Source-of-truth: viterbi_dp.py:297 (2026-04-21). Section-scale at ratio 0:
// `max(1, 2 - 2·ratio)` — slope-2 linear to ratio 0.5, flat at 1.0 above.
constexpr double SECTION_SCALE_BASE = 2.0;
constexpr double SECTION_SCALE_SLOPE = 2.0;

// Source-of-truth: viterbi_dp.py:304 (2026-04-21). Quadratic-span penalty
// coefficient inline `* 3.0` — should be a named constant per Python
// convention (other multipliers at L267 are). Keeping the shape.
constexpr double SPAN_PENALTY_WEIGHT = 3.0;

// Source-of-truth: viterbi_dp.py:344 (2026-04-21). Outro-weight linear
// scale: `max(1, 3 × (1 - ratio))` — at aggressive shortening (ratio 0.1)
// reaches 2.7×; at high ratio (0.9) falls to 1.0×.
constexpr double OUTRO_WEIGHT_FLOOR = 1.0;
constexpr double OUTRO_WEIGHT_SLOPE = 3.0;

// Source-of-truth: viterbi_dp.py:348 (2026-04-21). Per-unit-distance-from-
// end bonus factor (negative cost ⇒ reward) applied when beat i lies
// within the outro region.
constexpr double TERMINAL_IN_OUTRO_BONUS = -0.5;

// Source-of-truth: viterbi_dp.py:349 (2026-04-21). "Very near end"
// threshold — the last two beats get an additional flat reward.
constexpr int TERMINAL_END_PROXIMITY_BEATS = 2;

// Source-of-truth: viterbi_dp.py:350 (2026-04-21). Flat bonus for
// ending within `TERMINAL_END_PROXIMITY_BEATS` of n_beats.
constexpr double TERMINAL_END_BONUS = -1.0;

// Source-of-truth: viterbi_dp.py:354 (2026-04-21). Slope for non-outro
// termination penalty: `min(dist × 0.2 × outro_weight, 8.0)`.
constexpr double TERMINAL_NON_OUTRO_SLOPE = 0.2;

// Source-of-truth: viterbi_dp.py:354 (2026-04-21). Cap on non-outro
// termination penalty.
constexpr double TERMINAL_NON_OUTRO_CAP = 8.0;

// Source-of-truth: viterbi_dp.py:361 (2026-04-21). Precision-boost shape:
// `max(1, 2 - ratio)` — at ratio 0.15 ≈ 1.85; at ratio 0.9 ≈ 1.1.
constexpr double PRECISION_BOOST_FLOOR = 1.0;
constexpr double PRECISION_BOOST_BASE  = 2.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// `argmin(|beat_times - db_time|)` — direct port of `_compute_downbeat_arrays`
// L488. First-winner tie-break (Python `np.argmin` returns first index of
// minimum; std::min_element does likewise when comparator is `<`).
int nearestBeat(const double* beat_times, int n_beats, double db_time)
{
    double min_abs = std::numeric_limits<double>::infinity();
    int    best    = 0;
    for (int i = 0; i < n_beats; ++i) {
        const double d = std::fabs(beat_times[i] - db_time);
        if (d < min_abs) {
            min_abs = d;
            best    = i;
        }
    }
    return best;
}

} // namespace

// ---------------------------------------------------------------------------
// computeDownbeatArrays
// ---------------------------------------------------------------------------
//
// Direct port of `_compute_downbeat_arrays` (viterbi_dp.py:468-507).

DownbeatArrays
computeDownbeatArrays(const double* beat_times,
                      int           n_beats,
                      const double* downbeats,
                      int           n_downbeats,
                      int           time_signature,
                      bool          downbeat_constraint)
{
    DownbeatArrays out;

    // Python L481-482: `if not DEFAULT_CONFIG.remix.downbeat_constraint:
    //                      return None, None, None`.
    if (!downbeat_constraint) {
        return out;  // valid = false, empty arrays.
    }

    // Python L485-491: build downbeat_indices set.
    if (downbeats != nullptr && n_downbeats > 0) {
        for (int k = 0; k < n_downbeats; ++k) {
            const int idx = nearestBeat(beat_times, n_beats, downbeats[k]);
            out.downbeat_indices.insert(idx);
        }
    } else {
        // Fallback: `set(range(0, n_beats, time_signature))`.
        for (int i = 0; i < n_beats; i += time_signature) {
            out.downbeat_indices.insert(i);
        }
    }

    // Python L493-494: empty set → return None.
    if (out.downbeat_indices.empty()) {
        out = DownbeatArrays{};
        return out;  // valid = false.
    }

    // Python L496: pre_db_indices = {db - 1 for db in db_indices if db > 0}.
    for (int db : out.downbeat_indices) {
        if (db > 0) {
            out.pre_downbeat_indices.insert(db - 1);
        }
    }

    // Python L498-505: int8 boolean arrays.
    out.pre_downbeat_arr.assign(static_cast<std::size_t>(n_beats), 0);
    out.downbeat_arr.assign    (static_cast<std::size_t>(n_beats), 0);
    for (int idx : out.pre_downbeat_indices) {
        if (idx >= 0 && idx < n_beats) out.pre_downbeat_arr[static_cast<std::size_t>(idx)] = 1;
    }
    for (int idx : out.downbeat_indices) {
        if (idx >= 0 && idx < n_beats) out.downbeat_arr[static_cast<std::size_t>(idx)] = 1;
    }

    out.valid = true;
    return out;
}

// ---------------------------------------------------------------------------
// buildNeighbors
// ---------------------------------------------------------------------------
//
// Direct port of `_build_neighbors` (viterbi_dp.py:41-138).
//
// `jumps` can be nullptr (no repetition map); filtered inline per-beat
// (jumps.size is typically < 200 even at corpus max, so O(n_beats · n_jumps)
// linear scan stays sub-millisecond; pre-indexing would be premature).

NeighborCSR
buildNeighbors(int                                n_beats,
               const double*                      W,
               const analysis::RepetitionJump*    jumps,
               int                                n_jumps,
               const DownbeatArrays*              downbeat_arrays,
               const std::set<int>&               segment_boundaries,
               int                                max_neighbors,
               int                                min_forward_jump)
{
    NeighborCSR out;
    if (n_beats <= 0) {
        out.offsets.assign(1, 0);
        return out;
    }

    // Python L64-68: has_bar_info predicate.
    const bool has_bar_info = (downbeat_arrays != nullptr
                               && downbeat_arrays->valid
                               && !downbeat_arrays->downbeat_indices.empty()
                               && !downbeat_arrays->pre_downbeat_indices.empty());

    // Per-beat sorted unique neighbor lists.
    std::vector<std::vector<int>> all_neighbors(static_cast<std::size_t>(n_beats));

    // Scratch row buffer for top-K masking (copies W[i, :], overwrites with
    // INF to block self / micro-skip / non-downbeat entries).
    std::vector<double> w_row(static_cast<std::size_t>(n_beats), 0.0);

    for (int i = 0; i < n_beats; ++i) {
        std::set<int> neighbors;

        // Python L73-75: sequential.
        if (i + 1 < n_beats) {
            neighbors.insert(i + 1);
        }

        // Python L77-81: repetition-based jumps (always included if verified).
        if (jumps != nullptr && n_jumps > 0) {
            for (int k = 0; k < n_jumps; ++k) {
                if (jumps[k].fromBeat == i) {
                    const int to = jumps[k].toBeat;
                    if (to >= 0 && to < n_beats) {
                        neighbors.insert(to);
                    }
                }
            }
        }

        // Python L83-86: can_jump predicate.
        bool can_jump = true;
        if (has_bar_info) {
            can_jump = (downbeat_arrays->pre_downbeat_indices.count(i) != 0);
        }

        if (can_jump) {
            // Python L89: w_row = W[i].copy()
            for (int j = 0; j < n_beats; ++j) {
                w_row[static_cast<std::size_t>(j)] =
                    W[static_cast<std::size_t>(i) * n_beats + j];
            }

            // Python L91: w_row[i] = INF.
            w_row[static_cast<std::size_t>(i)] = INF;

            // Python L92-97: block micro-skip window.
            const int block_lo = std::max(0, i - min_forward_jump + 1);
            const int block_hi = std::min(n_beats, i + min_forward_jump);
            for (int b = block_lo; b < block_hi; ++b) {
                if (b != i + 1) {
                    w_row[static_cast<std::size_t>(b)] = INF;
                }
            }

            // Python L100-103: downbeat-only filter.
            if (has_bar_info) {
                for (int j = 0; j < n_beats; ++j) {
                    if (j == i + 1) continue;
                    if (downbeat_arrays->downbeat_indices.count(j) == 0) {
                        w_row[static_cast<std::size_t>(j)] = INF;
                    }
                }
            }

            // Python L105-110: top-K by cost.
            const int k = std::min(max_neighbors, n_beats);
            if (k < n_beats) {
                // np.argpartition(w_row, k)[:k] returns the k smallest
                // entries (unsorted). std::nth_element does the same:
                // elements before the k-th are ≤ the k-th, elements after
                // are ≥. Validated at 13 254-pair corpus scale session 19.
                std::vector<int> idx(static_cast<std::size_t>(n_beats));
                for (int j = 0; j < n_beats; ++j) idx[static_cast<std::size_t>(j)] = j;
                std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                                 [&](int a, int b) { return w_row[a] < w_row[b]; });
                for (int pos = 0; pos < k; ++pos) {
                    const int j = idx[static_cast<std::size_t>(pos)];
                    if (w_row[static_cast<std::size_t>(j)] < INF) {
                        neighbors.insert(j);
                    }
                }
            } else {
                // Python L110: top_k = np.arange(n_beats). Enumerate all.
                for (int j = 0; j < n_beats; ++j) {
                    if (w_row[static_cast<std::size_t>(j)] < INF) {
                        neighbors.insert(j);
                    }
                }
            }
        }

        // Python L117-121: always include segment boundaries as candidates.
        for (int b : segment_boundaries) {
            if (b == i) continue;
            if (std::abs(b - i) < min_forward_jump) continue;
            if (has_bar_info && downbeat_arrays->downbeat_indices.count(b) == 0) continue;
            neighbors.insert(b);
        }

        // Python L123: all_neighbors[i] = sorted(neighbors). std::set
        // iterates in sorted order.
        all_neighbors[static_cast<std::size_t>(i)].assign(neighbors.begin(), neighbors.end());
    }

    // Python L125-136: flatten to CSR.
    std::int64_t total = 0;
    for (const auto& nb : all_neighbors) {
        total += static_cast<std::int64_t>(nb.size());
    }
    out.indices.assign(static_cast<std::size_t>(total), 0);
    out.offsets.assign(static_cast<std::size_t>(n_beats) + 1, 0);

    std::int64_t pos = 0;
    for (int i = 0; i < n_beats; ++i) {
        out.offsets[static_cast<std::size_t>(i)] = pos;
        for (int j : all_neighbors[static_cast<std::size_t>(i)]) {
            out.indices[static_cast<std::size_t>(pos)] = static_cast<std::int64_t>(j);
            ++pos;
        }
    }
    out.offsets[static_cast<std::size_t>(n_beats)] = pos;

    return out;
}

// ---------------------------------------------------------------------------
// viterbiDP
// ---------------------------------------------------------------------------
//
// Direct port of `_viterbi_dp` (viterbi_dp.py:144-381).

ViterbiPath
viterbiDP(const ViterbiDPInputs& in)
{
    ViterbiPath result;
    result.total_cost = INF;

    if (in.n_beats <= 0 || in.target_length <= 0) {
        return result;
    }

    const int n_beats     = in.n_beats;
    const int T           = in.target_length;
    const int T_rows      = T + 1;      // dp shape (target_length + 1, n_beats)

    // Allocations (Python L167-173).
    std::vector<double>       dp        (static_cast<std::size_t>(T_rows) * n_beats, INF);
    std::vector<std::int64_t> parent    (static_cast<std::size_t>(T_rows) * n_beats, -1);
    std::vector<std::int64_t> ssj       (static_cast<std::size_t>(T_rows) * n_beats, 0);
    std::vector<std::int64_t> n_jumps   (static_cast<std::size_t>(T_rows) * n_beats, 0);
    std::vector<double>       worst_jump(static_cast<std::size_t>(T_rows) * n_beats, 0.0);

    const bool has_bar_info = (in.pre_downbeat_arr != nullptr && in.downbeat_arr != nullptr);

    // Python L184-185.
    const double target_ratio      = static_cast<double>(T) / std::max(1, n_beats);
    const double jump_penalty_scale =
        std::max(JUMP_PENALTY_SCALE_FLOOR,
                 std::min(JUMP_PENALTY_SCALE_CEILING, target_ratio));

    // Python L188-189: initialize.
    dp [1 * n_beats + 0] = 0.0;
    ssj[1 * n_beats + 0] = SSJ_NO_RECENT_JUMP_SENTINEL;

    // Python L191-195: boundary_set from beat_to_segment transitions.
    std::vector<std::uint8_t> is_boundary(static_cast<std::size_t>(n_beats), 0);
    if (in.n_segs > 1) {
        for (int i = 1; i < n_beats; ++i) {
            if (in.beat_to_segment[i] != in.beat_to_segment[i - 1]) {
                is_boundary[static_cast<std::size_t>(i)] = 1;
            }
        }
    }

    // Python L197-320: fill DP table.
    for (int t = 1; t < T; ++t) {
        const std::size_t row_t   = static_cast<std::size_t>(t)     * n_beats;
        const std::size_t row_t1  = static_cast<std::size_t>(t + 1) * n_beats;

        for (int i = 0; i < n_beats; ++i) {
            const double dp_ti = dp[row_t + i];
            if (dp_ti >= INF) continue;

            // Python L204-214: intro-lock path.
            if (t < in.intro_beats) {
                const int j = i + 1;
                if (j < n_beats) {
                    const double cost = dp_ti + in.W[static_cast<std::size_t>(i) * n_beats + j];
                    if (cost < dp[row_t1 + j]) {
                        dp        [row_t1 + j] = cost;
                        parent    [row_t1 + j] = i;
                        ssj       [row_t1 + j] = ssj[row_t + i] + 1;
                        worst_jump[row_t1 + j] = worst_jump[row_t + i];
                        n_jumps   [row_t1 + j] = n_jumps   [row_t + i];
                    }
                }
                continue;
            }

            // Python L216-227: cooldown path.
            if (ssj[row_t + i] < in.min_seq_after_jump) {
                const int j = i + 1;
                if (j < n_beats) {
                    const double cost = dp_ti + in.W[static_cast<std::size_t>(i) * n_beats + j];
                    if (cost < dp[row_t1 + j]) {
                        dp        [row_t1 + j] = cost;
                        parent    [row_t1 + j] = i;
                        ssj       [row_t1 + j] = ssj[row_t + i] + 1;
                        worst_jump[row_t1 + j] = worst_jump[row_t + i];
                        n_jumps   [row_t1 + j] = n_jumps   [row_t + i];
                    }
                }
                continue;
            }

            // Python L229-320: explore neighbors.
            const std::int64_t start = in.neighbor_offsets[i];
            const std::int64_t end   = in.neighbor_offsets[i + 1];

            for (std::int64_t ni = start; ni < end; ++ni) {
                const int j = static_cast<int>(in.neighbor_indices[ni]);

                // Python L237-238: backward must respect min_segment.
                if (j < i && (i - j) < in.min_segment_beats) continue;

                // Python L241-242: don't jump backward into intro.
                if (j < i && j < in.intro_beats) continue;

                // Python L245-246: forward micro-skips blocked.
                if (j > i + 1 && (j - i) < in.min_forward_jump) continue;

                // Python L249-253: bar alignment.
                if (j != i + 1 && has_bar_info) {
                    if (in.pre_downbeat_arr[i] != 1) continue;
                    if (in.downbeat_arr[j]    != 1) continue;
                }

                // Python L256-258: base cost + INF guard.
                const double raw_w = in.W[static_cast<std::size_t>(i) * n_beats + j];
                if (raw_w >= INF) continue;

                double cost = dp_ti + raw_w;

                // Python L266-268: aggressive-shortening sequential nudge.
                if (j == i + 1 && target_ratio < SEQUENTIAL_PENALTY_ACTIVATION_RATIO) {
                    const double seq_penalty =
                        SEQUENTIAL_PENALTY_COEFF * (SEQUENTIAL_PENALTY_ACTIVATION_RATIO - target_ratio);
                    cost += seq_penalty;
                }

                // Python L270-272: backward penalty.
                if (in.is_shortening && j < i) {
                    cost += BACKWARD_PENALTY;
                }

                // Python L274-304: non-sequential jump penalties.
                if (j != i + 1) {
                    // Python L277-278: max transitions gate.
                    if (n_jumps[row_t + i] >= in.max_transitions) continue;

                    // ADR-084 sesja 93 (REVISED) — Edit Length MULTIPLICATIVE
                    // jump-cost scale applied to TOTAL jump-tax (quality +
                    // section + span). Initial sesja-93 build only scaled
                    // quality_factor term (~5-10% of typical jump tax),
                    // which was dwarfed by section + span at typical
                    // target_ratio → 16× slider span produced no path
                    // change. User smoke verdict on Joe Jackson 1:45 confirmed.
                    // Now accumulate full jump-extra into local var, multiply
                    // by jump_scale at end. Default 1.0 → bit-exact baseline.
                    double jump_extra = 0.0;

                    // Python L280-289: quadratic jump + boundary bonus.
                    const double quality_factor = raw_w * raw_w;
                    const double scaled_penalty =
                        JUMP_PENALTY_BASE * jump_penalty_scale;
                    const bool at_boundary =
                           is_boundary[static_cast<std::size_t>(j)]
                        || is_boundary[static_cast<std::size_t>(i)];
                    if (at_boundary) {
                        jump_extra += scaled_penalty * quality_factor * (1.0 - BOUNDARY_BONUS);
                    } else {
                        jump_extra += scaled_penalty * quality_factor;
                    }

                    // Python L291-298: section compatibility.
                    const std::int64_t seg_i = in.beat_to_segment[i];
                    const std::int64_t seg_j = in.beat_to_segment[j];
                    if (in.seg_sim_matrix != nullptr
                        && seg_i < in.n_segs
                        && seg_j < in.n_segs)
                    {
                        const double s_sim = in.seg_sim_matrix[
                            static_cast<std::size_t>(seg_i) * in.n_segs + seg_j];
                        const double section_scale =
                            std::max(1.0, SECTION_SCALE_BASE - target_ratio * SECTION_SCALE_SLOPE);
                        jump_extra += SECTION_PENALTY * section_scale * (1.0 - s_sim);
                    }

                    // Python L300-304: large-span penalty.
                    if (j > i + 1) {
                        const double span_frac = static_cast<double>(j - i) / n_beats;
                        if (span_frac > SPAN_PENALTY_THRESHOLD) {
                            const double dx = span_frac - SPAN_PENALTY_THRESHOLD;
                            jump_extra += dx * dx * SPAN_PENALTY_WEIGHT;
                        }
                    }

                    // Apply Edit Length scale to entire jump-tax.
                    cost += jump_extra * in.edit_length_jump_scale;
                }

                // Python L306-313: L-inf tracking.
                double       new_worst;
                std::int64_t new_jumps;
                if (j != i + 1) {
                    const double prev_worst = worst_jump[row_t + i];
                    new_worst = std::max(prev_worst, raw_w);
                    cost += LINF_PENALTY_WEIGHT * std::max(0.0, new_worst - prev_worst);
                    new_jumps = n_jumps[row_t + i] + 1;
                } else {
                    new_worst = worst_jump[row_t + i];
                    new_jumps = n_jumps   [row_t + i];
                }

                // Python L315-320: relax DP cell.
                if (cost < dp[row_t1 + j]) {
                    dp        [row_t1 + j] = cost;
                    parent    [row_t1 + j] = i;
                    ssj       [row_t1 + j] = (j == i + 1) ? ssj[row_t + i] + 1 : 0;
                    worst_jump[row_t1 + j] = new_worst;
                    n_jumps   [row_t1 + j] = new_jumps;
                }
            }
        }
    }

    // Python L322-368: endpoint search with terminal + duration scoring.
    int    best_end  = -1;
    double best_cost = INF;
    int    best_t    = T;
    const int outro_start = (in.outro_beats > 0)
                                ? std::max(0, n_beats - in.outro_beats)
                                : n_beats;

    const int search_start = (in.min_target_length > 0) ? in.min_target_length : T;

    for (int t = search_start; t <= T; ++t) {
        const std::size_t row_t = static_cast<std::size_t>(t) * n_beats;

        for (int i = 0; i < n_beats; ++i) {
            const double dp_ti = dp[row_t + i];
            if (dp_ti >= INF) continue;

            // Python L338-339: min_jumps gate.
            if (n_jumps[row_t + i] < in.min_jumps) continue;

            // Python L341-356: terminal scoring.
            double terminal_penalty;
            if (in.outro_beats > 0) {
                const double outro_weight =
                    std::max(OUTRO_WEIGHT_FLOOR, OUTRO_WEIGHT_SLOPE * (1.0 - target_ratio));
                if (i >= outro_start) {
                    const double bars_from_end =
                        static_cast<double>(n_beats - 1 - i) / std::max(1, in.outro_beats);
                    terminal_penalty = TERMINAL_IN_OUTRO_BONUS * outro_weight * (1.0 - bars_from_end);
                    if (i >= n_beats - TERMINAL_END_PROXIMITY_BEATS) {
                        terminal_penalty += TERMINAL_END_BONUS * outro_weight;
                    }
                } else {
                    const double dist = static_cast<double>(outro_start - i);
                    terminal_penalty = std::min(dist * TERMINAL_NON_OUTRO_SLOPE * outro_weight,
                                                TERMINAL_NON_OUTRO_CAP);
                }
            } else {
                terminal_penalty = 0.0;
            }

            // Python L358-362: duration deviation.
            const double dev =
                std::fabs(static_cast<double>(t - T)) / std::max(1, T);
            const double precision_boost =
                std::max(PRECISION_BOOST_FLOOR, PRECISION_BOOST_BASE - target_ratio);
            const double duration_penalty = dev * DURATION_PENALTY_WEIGHT * precision_boost;

            const double total = dp_ti + terminal_penalty + duration_penalty;
            if (total < best_cost) {
                best_cost = total;
                best_end  = i;
                best_t    = t;
            }
        }
    }

    if (best_end < 0) {
        // Python L370-371: empty path.
        return result;  // path empty, total_cost = INF.
    }

    // Python L373-381: backtrace.
    std::vector<std::int64_t> path_rev;
    path_rev.reserve(static_cast<std::size_t>(best_t));
    int cur = best_end;
    for (int t = best_t; t > 0; --t) {
        path_rev.push_back(cur);
        cur = static_cast<int>(parent[static_cast<std::size_t>(t) * n_beats + cur]);
    }
    result.path.assign(path_rev.rbegin(), path_rev.rend());
    result.total_cost = best_cost;
    return result;
}

} // namespace reamix::remix
