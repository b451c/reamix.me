#include "remix/RegionOptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>      // sesja 94 diagnostic log (region_beta path)
#include <cstdlib>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>       // sesja 94 diagnostic log iter 2 (top-backward dump)
#include <utility>
#include <vector>

namespace reamix::remix {

namespace {

// REGION_COOLDOWN = 8 (region_optimizer.py:226).
// UNJUSTIFIED (R7) per session-23 audit — no empirical citation for 8.
// Distinct from global `COOLDOWN_BARS × time_signature` (= 16 for 4/4);
// region DP operates on shorter scales where 8-beat cooldown allows more
// viable splice points.
constexpr std::int64_t REGION_COOLDOWN = 8;

// R6 neighbor top-k — region_optimizer.py:195.
// UNJUSTIFIED (R6) — differs from global `viterbi_max_neighbors=12`.
constexpr int REGION_NEIGHBOR_TOPK = 8;

// R8 initial-state cooldown sentinel — region_optimizer.py:233.
// MARKER (R8) — `cooldown_ssj[1, 0] = 999` encodes "no recent jump";
// must exceed any plausible REGION_COOLDOWN. Session-22 F2 MARKER sub-class.
constexpr std::int64_t REGION_SSJ_NO_RECENT_JUMP_SENTINEL = 999;

// R3/R5 backward-jump penalty (extending mode) — region_optimizer.py:158 + :176.
// UNJUSTIFIED (R3) + BENIGN-DUPLICATION (R5). Port verbatim, DO NOT DRY —
// session-23 corpus validates both paths bit-exact with the duplicate.
constexpr double REGION_BACKWARD_JUMP_PENALTY_EXTEND = 0.5;

// R9 jump-cost cap + scale — region_optimizer.py:260.
// `cost += JUMP_PENALTY_BASE * min(1.0, raw_w * 2.0)`.
// UNJUSTIFIED (R9).
constexpr double REGION_JUMP_COST_CAP   = 1.0;
constexpr double REGION_JUMP_COST_SCALE = 2.0;

// R10 terminal-penalty scale — region_optimizer.py:279.
// `terminal_penalty = dist * 3.0 if dist > 1 else 0.0`.
// UNJUSTIFIED (R10).
constexpr double REGION_TERMINAL_SCALE  = 3.0;
constexpr int    REGION_TERMINAL_DIST_THRESHOLD = 1;  // R11 DEFENSIVE

// R13 fallback repeats guards — region_optimizer.py:296.
// `repeats = max(1, int(round(target_duration / max(0.1, region_duration))))`.
// DEFENSIVE (R13).
constexpr double REGION_FALLBACK_MIN_DURATION = 0.1;
constexpr int    REGION_FALLBACK_MIN_REPEATS  = 1;

// R2 min region/target/tolerance guards — DEFENSIVE.
// region_optimizer.py:78,90,93,97 all use the literal 2 as "minimum viable DP"
// sentinel. Four occurrences — all the same class, documented as one.
constexpr int REGION_MIN_VIABLE_DP = 2;

// Python `int(round(x))` banker's rounding — std::nearbyint with default
// FE_TONEAREST. Same helper as Optimizer.cpp session 22.
int pyIntRound(double x)
{
    return static_cast<int>(std::nearbyint(x));
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor — stores inputs, no heavy work.
// ---------------------------------------------------------------------------
RegionOptimizer::RegionOptimizer(const RegionOptimizerInputs& in)
    : n_beats_(in.n_beats)
    , beat_times_(in.beat_times)
    , avg_beat_duration_(in.avg_beat_duration)
    , duration_tolerance_sec_(in.duration_tolerance_sec)
    , candidates_(in.candidates)
    , sample_rate_(in.sample_rate)
    , downbeats_(in.downbeats)
    , n_downbeats_(in.n_downbeats)
    , splice_flex_beats_(in.splice_flex_beats)
    , edit_length_jump_scale_(in.edit_length_jump_scale)  // ADR-084 sesja 93
    , min_seq_after_jump_override_(in.min_seq_after_jump_override)  // ADR-083 sesja 92
    , region_beta_(in.region_beta)  // ADR-081 STATUS UPDATE 1 sesja 94
    , entry_beat_override_(in.entry_beat_override)  // ADR-081 STATUS UPDATE 2 sesja 94
    , exit_beat_override_(in.exit_beat_override)
{
}

// ---------------------------------------------------------------------------
// Main entry.
// Port of `remix_region` (region_optimizer.py:35-137).
// ---------------------------------------------------------------------------
RemixPath RegionOptimizer::remix(double                                                    target_duration,
                                  double                                                    region_start_sec,
                                  double                                                    region_end_sec,
                                  const double*                                             region_W,
                                  int                                                       n_region_input,
                                  const std::map<std::pair<int, int>, TransitionCandidate>* region_candidates,
                                  const std::set<std::pair<int, int>>*                      blocked_transitions)
{
    // Sesja 100 (DEV-032) — stash blocked set for regionDp + regionLoopSynthesize
    // to consume. RAII guard ensures member resets even on exception path.
    blocked_transitions_ = blocked_transitions;
    struct BlockedGuard {
        RegionOptimizer* self;
        ~BlockedGuard() { self->blocked_transitions_ = nullptr; }
    } blockedGuard { this };

    // region_optimizer.py:66-70 short-circuit.
    if (n_beats_ < REGION_MIN_VIABLE_DP) {
        RemixPath p;
        p.total_cost     = 0.0;
        p.duration_beats = 0;
        return p;
    }

    // Find region beat boundaries.
    //
    // Two paths:
    //
    // (1) DEV-033 / ADR-054 soft-boundary path — when caller passes downbeats
    //     and a non-zero splice_flex_beats. Searches ±W beats around user's
    //     region.startSec/endSec for the closest downbeat (phrase boundary
    //     = musically natural splice point). Falls back to directional snap
    //     (first beat at-or-after for entry; last beat at-or-before for exit)
    //     when no downbeat falls in the window. Used by production Region
    //     mode in RemixPipeline.
    //
    // (2) Legacy closest-beat snap — preserved for the parity test against
    //     the Python port and any caller that doesn't populate the new
    //     RegionOptimizerInputs fields.
    auto argminAbsDiff = [&](double target) {
        int    idx  = 0;
        double best = std::abs(beat_times_[0] - target);
        for (int i = 1; i < n_beats_; ++i) {
            const double d = std::abs(beat_times_[i] - target);
            if (d < best) { best = d; idx = i; }
        }
        return idx;
    };

    int entry_beat;
    int exit_beat;

    // ADR-081 STATUS UPDATE 2 sesja 94 — caller-override path. When the
    // caller (RemixPipeline) provides entry_beat_override / exit_beat_override
    // ≥ 0, use them directly. This guarantees stride consistency with the
    // region_W matrix the caller built. Fixes latent stride bug where local
    // argminAbsDiff diverged from RemixPipeline's lower_bound/upper_bound by
    // 1 beat for user regions whose boundaries don't align with beat times.
    const bool useCallerOverride = (entry_beat_override_ >= 0)
                                 && (exit_beat_override_ >= 0)
                                 && (exit_beat_override_ > entry_beat_override_);

    const bool useSoftBoundary = (! useCallerOverride)
                              && (splice_flex_beats_ > 0)
                              && (downbeats_ != nullptr)
                              && (n_downbeats_ > 0);

    if (useCallerOverride) {
        entry_beat = entry_beat_override_;
        exit_beat  = exit_beat_override_;
    }
    else if (useSoftBoundary)
    {
        const double window_sec =
            static_cast<double>(splice_flex_beats_) * avg_beat_duration_;

        // Find closest downbeat to a target time within ±window_sec, return
        // -1 if none. Linear scan — n_downbeats is O(track_bars).
        auto closestDownbeatInWindow = [&](double target) -> double {
            double best_t = -1.0;
            double best_d = window_sec + 1.0;
            for (int i = 0; i < n_downbeats_; ++i) {
                const double t = downbeats_[i];
                const double d = std::abs(t - target);
                if (d <= window_sec && d < best_d) {
                    best_d = d;
                    best_t = t;
                }
            }
            return best_t;
        };

        // Map a time to the index of the closest beat (used to convert a
        // chosen downbeat time into a beat index for DP).
        auto closestBeatIndex = [&](double target) {
            int    idx  = 0;
            double best = std::abs(beat_times_[0] - target);
            for (int i = 1; i < n_beats_; ++i) {
                const double d = std::abs(beat_times_[i] - target);
                if (d < best) { best = d; idx = i; }
            }
            return idx;
        };

        // -- entry_beat ------------------------------------------------------
        const double db_entry = closestDownbeatInWindow(region_start_sec);
        if (db_entry >= 0.0) {
            entry_beat = closestBeatIndex(db_entry);
        } else {
            // Fallback: first beat at-or-after region.startSec (directional
            // snap). Pre-region item trim on Insert side covers any tiny
            // forward jump — content stays continuous.
            entry_beat = 0;
            while (entry_beat < n_beats_
                   && beat_times_[entry_beat] < region_start_sec) {
                ++entry_beat;
            }
            if (entry_beat >= n_beats_) entry_beat = n_beats_ - 1;
        }

        // -- exit_beat -------------------------------------------------------
        const double db_exit = closestDownbeatInWindow(region_end_sec);
        if (db_exit >= 0.0) {
            exit_beat = closestBeatIndex(db_exit);
        } else {
            // Fallback: last beat at-or-before region.endSec.
            exit_beat = n_beats_ - 1;
            while (exit_beat > 0 && beat_times_[exit_beat] > region_end_sec) {
                --exit_beat;
            }
        }
    }
    else
    {
        // Legacy closest-beat snap — preserves parity test goldens.
        entry_beat = argminAbsDiff(region_start_sec);
        exit_beat  = argminAbsDiff(region_end_sec);
    }

    // R1 DEFENSIVE: `max(entry_beat + 1, min(exit_beat, n_beats - 1))`.
    exit_beat = std::max(entry_beat + 1, std::min(exit_beat, n_beats_ - 1));

    entry_beat_ = entry_beat;
    exit_beat_  = exit_beat;

    int region_beats = exit_beat - entry_beat;
    // region_optimizer.py:77-84.
    if (region_beats < REGION_MIN_VIABLE_DP) {  // R2 DEFENSIVE
        RemixPath p;
        p.beat_indices.reserve(static_cast<std::size_t>(region_beats));
        for (int b = entry_beat; b < exit_beat; ++b) {
            p.beat_indices.push_back(b);  // RemixPath::beat_indices is std::vector<int>
        }
        p.total_cost     = 0.0;
        p.duration_beats = region_beats;
        return p;
    }

    // region_optimizer.py:86-90.
    region_duration_ = beat_times_[exit_beat] - beat_times_[entry_beat];
    is_extending_   = target_duration > region_duration_;
    // R2 DEFENSIVE: `max(2, int(round(target_duration / avg_beat_duration)))`.
    target_beats_ = std::max(REGION_MIN_VIABLE_DP,
                             pyIntRound(target_duration / avg_beat_duration_));

    // Duration flexibility — region_optimizer.py:92-98.
    // R2 DEFENSIVE: tolerance_beats floor 2.
    const int tolerance_beats = std::max(
        REGION_MIN_VIABLE_DP,
        pyIntRound(duration_tolerance_sec_ / avg_beat_duration_));
    // R2 DEFENSIVE: min_target floor 2.
    min_target_ = std::max(REGION_MIN_VIABLE_DP, target_beats_ - tolerance_beats);
    max_target_ = target_beats_ + tolerance_beats;

    // Build constrained cost matrix — region_optimizer.py:100-104.
    // n_region from input (caller should have already computed exit-entry).
    // For parity, trust our local computation:
    const int n_region = region_beats;
    // Sanity: caller-provided n_region should match. If not, trust ours.
    (void) n_region_input;

    buildRegionCostMatrix(n_region, region_W, is_extending_);

    // Build neighbor lists — region_optimizer.py:106-107.
    buildRegionNeighbors(n_region);

    // ADR-081 STATUS UPDATE 1 sesja 94 — extended diagnostic dump (Path B
    // iteration 2). Logs candidate-space telemetry BEFORE DP runs so we can
    // diagnose whether fallback hits are due to (a) cost-function bias
    // (β-model intent) or (b) candidate-space exhaustion (hard gates rejected
    // all viable backward candidates → β-model has nothing to choose from).
    // sesja-94-only debug aid; removed at handover.
    if (region_beta_) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            std::string log_path = std::string(home) + "/Desktop/reamix_diag.log";
            std::FILE* f = std::fopen(log_path.c_str(), "a");
            if (f != nullptr) {
                int n_backward_viable = 0;
                int n_forward_skip_viable = 0;  // forward jumps (rj > ri+1)
                std::vector<std::tuple<double, int, int>> backward_cells;
                backward_cells.reserve(static_cast<std::size_t>(n_region) * n_region);

                for (int ri = 0; ri < n_region; ++ri) {
                    for (int rj = 0; rj < n_region; ++rj) {
                        if (rj == ri || rj == ri + 1) continue;
                        const double v = rW_[static_cast<std::size_t>(ri) * n_region + rj];
                        if (v >= INF) continue;
                        if (rj < ri) {
                            ++n_backward_viable;
                            backward_cells.emplace_back(v, ri, rj);
                        } else {
                            ++n_forward_skip_viable;
                        }
                    }
                }

                std::sort(backward_cells.begin(), backward_cells.end());

                int n_neighbors_total = static_cast<int>(nb_indices_.size());
                int n_rows_with_extra_neighbor = 0;
                for (int ri = 0; ri < n_region; ++ri) {
                    int row_count = static_cast<int>(
                        nb_offsets_[static_cast<std::size_t>(ri + 1)]
                        - nb_offsets_[static_cast<std::size_t>(ri)]);
                    // > 1 means more than just sequential ri+1 in neighbor list.
                    if (row_count > 1) ++n_rows_with_extra_neighbor;
                }

                std::fprintf(f, "REGION_BETA_RW_DUMP region_start_sec=%.3f region_end_sec=%.3f "
                             "n_region=%d n_backward_viable=%d n_forward_skip_viable=%d "
                             "n_neighbors_total=%d n_rows_with_extra_neighbor=%d "
                             "top_backward=[",
                             region_start_sec, region_end_sec, n_region,
                             n_backward_viable, n_forward_skip_viable,
                             n_neighbors_total, n_rows_with_extra_neighbor);
                int n_dump = std::min<int>(10, static_cast<int>(backward_cells.size()));
                for (int k = 0; k < n_dump; ++k) {
                    double v = std::get<0>(backward_cells[(std::size_t) k]);
                    int ri = std::get<1>(backward_cells[(std::size_t) k]);
                    int rj = std::get<2>(backward_cells[(std::size_t) k]);
                    int abs_i = entry_beat_ + ri;
                    int abs_j = entry_beat_ + rj;
                    std::fprintf(f, "(%d->%d,raw_w=%.4f)%s",
                                 abs_i, abs_j, v,
                                 (k + 1 < n_dump) ? "," : "");
                }
                std::fprintf(f, "]\n");
                std::fclose(f);
            }
        }
    }

    // ADR-081 STATUS UPDATE 2 sesja 94 — Region β-model loop synthesizer.
    // When region_beta_ AND is_extending_, try explicit (i, j, N) synthesis
    // BEFORE falling through to standard regionDp. Synthesis builds an
    // explicit multi-iteration loop path matching user intent ("find ideal
    // inner loop and repeat 3×"); standard DP minimizes total body cost
    // which structurally biases toward 1 transition (47/48 corpus cases).
    //
    // If synth produces empty path (no viable (i, j, N) in tolerance window),
    // fall back to DP. DP path also retains cost-rebalance constants from
    // STATUS UPDATE 1 — both knobs work together.
    //
    // ADR-083 sesja 92 — Min cut UI override applied here too: synth uses
    // same effective_cooldown as DP for consistency (loops shorter than
    // user's Min cut would not satisfy "minimum segment length" UX intent).
    const std::int64_t effective_cooldown_synth =
        (min_seq_after_jump_override_ > 0)
            ? static_cast<std::int64_t>(min_seq_after_jump_override_)
            : REGION_COOLDOWN;
    if (region_beta_ && is_extending_) {
        LoopSynthResult synth = regionLoopSynthesize(
            n_region, min_target_, max_target_, target_beats_,
            static_cast<int>(effective_cooldown_synth), entry_beat);
        if (! synth.path.empty()) {
            // Use synth result directly.
            RemixPath out;
            out.beat_indices.reserve(synth.path.size());
            for (std::int64_t idx : synth.path) {
                out.beat_indices.push_back(static_cast<int>(idx));
            }
            out.total_cost     = synth.total_cost;
            out.duration_beats = static_cast<int>(out.beat_indices.size());

            // Build metadata for transitions from candidates map (same logic
            // as regionTransitions but path is already built).
            std::map<std::pair<int, int>, std::map<std::string, double>> metadata;
            std::map<std::pair<int, int>, TransitionCandidate> all_candidates;
            if (candidates_ != nullptr) {
                for (const auto& kv : *candidates_) {
                    all_candidates.insert_or_assign(kv.first, kv.second);
                }
            }
            if (region_candidates != nullptr) {
                for (const auto& kv : *region_candidates) {
                    all_candidates.insert_or_assign(kv.first, kv.second);
                }
            }
            for (const auto& tr : synth.transitions) {
                std::map<std::string, double> meta;
                auto it = all_candidates.find(tr);
                if (it != all_candidates.end()) {
                    const TransitionCandidate& cand = it->second;
                    meta["quality_score"]          = cand.quality_score;
                    meta["waveform_similarity"]    = cand.waveform_similarity;
                    meta["successor_similarity"]   = cand.successor_similarity;
                    meta["edge_splice_similarity"] = cand.edge_splice_similarity;
                    meta["chroma_distance"]        = cand.chroma_distance;
                    meta["energy_diff_db"]         = cand.energy_diff_db;
                    meta["alignment_offset_sec"]   =
                        static_cast<double>(cand.alignment_lag_samples)
                        / static_cast<double>(std::max(1, sample_rate_));
                    meta["total_cost"]             = cand.total_cost;
                }
                metadata[tr] = std::move(meta);
            }
            out.transitions         = std::move(synth.transitions);
            out.transition_metadata = std::move(metadata);
            return out;
        }
        // Synth produced no valid result → fall through to standard DP.
    }

    // Run DP — region_optimizer.py:109-113.
    DpResult dp_res = regionDp(n_region, min_target_, max_target_, target_beats_);

    // Check failure — region_optimizer.py:115-119.
    if (dp_res.best_cost >= INF) {
        // ADR-081 STATUS UPDATE 1 sesja 94 — diagnostic log for fallback
        // path (sesja-94-only debug aid; matches success-path log below).
        if (region_beta_) {
            const char* home = std::getenv("HOME");
            if (home != nullptr) {
                std::string log_path = std::string(home) + "/Desktop/reamix_diag.log";
                std::FILE* f = std::fopen(log_path.c_str(), "a");
                if (f != nullptr) {
                    std::fprintf(f, "REGION_BETA_FALLBACK region_start_sec=%.3f "
                                 "region_end_sec=%.3f entry_beat=%d exit_beat=%d "
                                 "n_region=%d target_beats=%d min_target=%d "
                                 "max_target=%d is_extending=%d region_beta=%d "
                                 "fallback=tile_whole_region\n",
                                 region_start_sec, region_end_sec,
                                 entry_beat, exit_beat, exit_beat - entry_beat,
                                 target_beats_, min_target_, max_target_,
                                 is_extending_ ? 1 : 0, region_beta_ ? 1 : 0);
                    std::fclose(f);
                }
            }
        }
        return regionFallback(entry_beat, exit_beat, target_beats_,
                              target_duration, region_duration_);
    }

    // Backtrace — region_optimizer.py:121-124.
    std::vector<std::int64_t> path = regionBacktrace(
        dp_res.best_t, dp_res.best_ri, dp_res.parent,
        n_region, max_target_, entry_beat);

    // Extract transitions + metadata — region_optimizer.py:126-129.
    std::vector<std::pair<int, int>>                             transitions;
    std::map<std::pair<int, int>, std::map<std::string, double>> metadata;
    regionTransitions(path, region_candidates, transitions, metadata);

    // Path.h convention: beat_indices is std::vector<int> (int64 → int demote
    // session-22 Optimizer.cpp precedent at :616-622; beat count fits in int).
    RemixPath out;
    out.beat_indices.reserve(path.size());
    for (std::int64_t idx : path) out.beat_indices.push_back(static_cast<int>(idx));
    out.total_cost         = dp_res.best_cost;
    out.duration_beats     = static_cast<int>(out.beat_indices.size());
    out.transitions        = std::move(transitions);
    out.transition_metadata = std::move(metadata);

    // ADR-081 STATUS UPDATE 1 sesja 94 — diagnostic log for inner-loop β-model.
    // Append-only telemetry per Region remix() call. REMOVE at sesja-94
    // handover (sesja-94-only debug aid; same pattern as sesja-93 ADR-084
    // Optimizer.cpp diag log). Logs only when region_beta_ flag is set so
    // baseline parity tests are not polluted.
    if (region_beta_) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            std::string log_path = std::string(home) + "/Desktop/reamix_diag.log";
            std::FILE* f = std::fopen(log_path.c_str(), "a");
            if (f != nullptr) {
                std::fprintf(f, "REGION_BETA region_start_sec=%.3f region_end_sec=%.3f "
                             "entry_beat=%d exit_beat=%d n_region=%d "
                             "target_beats=%d min_target=%d max_target=%d "
                             "is_extending=%d region_beta=%d "
                             "n_transitions=%d total_cost=%.6f path_len=%d ",
                             region_start_sec, region_end_sec,
                             entry_beat_, exit_beat_, exit_beat_ - entry_beat_,
                             target_beats_, min_target_, max_target_,
                             is_extending_ ? 1 : 0, region_beta_ ? 1 : 0,
                             static_cast<int>(out.transitions.size()),
                             out.total_cost, out.duration_beats);
                // Log up to first 5 transitions: (from_beat, to_beat, raw_w
                // sourced from region_W as (ri,rj)-relative cells).
                int limit = std::min(5, static_cast<int>(out.transitions.size()));
                std::fprintf(f, "transitions=[");
                for (int t = 0; t < limit; ++t) {
                    int from_abs = out.transitions[t].first;
                    int to_abs   = out.transitions[t].second;
                    int from_rel = from_abs - entry_beat_;
                    int to_rel   = to_abs - entry_beat_;
                    double raw_w = INF;
                    if (from_rel >= 0 && from_rel < (exit_beat_ - entry_beat_)
                        && to_rel >= 0 && to_rel < (exit_beat_ - entry_beat_)) {
                        raw_w = rW_[static_cast<std::size_t>(from_rel)
                                    * (exit_beat_ - entry_beat_)
                                    + static_cast<std::size_t>(to_rel)];
                    }
                    std::fprintf(f, "(%d,%d,raw_w=%.4f)%s",
                                 from_abs, to_abs, raw_w,
                                 (t + 1 < limit) ? "," : "");
                }
                std::fprintf(f, "]\n");
                std::fclose(f);
            }
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Build region cost matrix — Q2 rec B (session 23): production path only,
// `region_W != nullptr`. Fallback branch (region_W==None, L160-178) skipped.
// ---------------------------------------------------------------------------
void RegionOptimizer::buildRegionCostMatrix(int           n_region,
                                             const double* region_W,
                                             bool          is_extending)
{
    rW_.assign(static_cast<std::size_t>(n_region) * n_region, INF);

    // region_optimizer.py:152-158 — copy input + add +0.5 to backward cells
    // when extending.
    for (int ri = 0; ri < n_region; ++ri) {
        for (int rj = 0; rj < n_region; ++rj) {
            rW_[static_cast<std::size_t>(ri) * n_region + rj] =
                region_W[static_cast<std::size_t>(ri) * n_region + rj];
        }
    }
    if (is_extending) {
        // R3 UNJUSTIFIED — region_optimizer.py:157-158.
        // ADR-081 STATUS UPDATE 1 sesja 94 — beta path uses tenfold-reduced
        // backward penalty (0.05 vs 0.5) so multi-iteration loops on quality-
        // rich inner content compete with single-jump alternatives.
        // REGION_COOLDOWN=8 still bounds runaway loops structurally.
        const double effective_backward_penalty =
            region_beta_ ? 0.05 : REGION_BACKWARD_JUMP_PENALTY_EXTEND;
        for (int ri = 0; ri < n_region; ++ri) {
            for (int rj = 0; rj < ri; ++rj) {
                double& cell = rW_[static_cast<std::size_t>(ri) * n_region + rj];
                if (cell < INF) cell += effective_backward_penalty;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build CSR neighbor list.
// Port of `_build_region_neighbors` (region_optimizer.py:181-213).
// ---------------------------------------------------------------------------
void RegionOptimizer::buildRegionNeighbors(int n_region)
{
    std::vector<std::vector<int>> region_neighbors(static_cast<std::size_t>(n_region));

    for (int ri = 0; ri < n_region; ++ri) {
        std::set<int> neighbors;
        if (ri + 1 < n_region) neighbors.insert(ri + 1);

        // Copy row costs, mask self + ri+1.
        std::vector<double> costs(static_cast<std::size_t>(n_region));
        for (int j = 0; j < n_region; ++j) {
            costs[static_cast<std::size_t>(j)] =
                rW_[static_cast<std::size_t>(ri) * n_region + j];
        }
        costs[static_cast<std::size_t>(ri)] = INF;
        if (ri + 1 < n_region) costs[static_cast<std::size_t>(ri + 1)] = INF;

        const int k = std::min(REGION_NEIGHBOR_TOPK, n_region);
        if (k > 0 && k < n_region) {
            // Python `np.argpartition(costs, k)[:k]` — unsorted top-k by cost
            // (smallest k). std::nth_element matches semantics (session-19
            // corpus-scale validated on TransitionCost candidate selection).
            std::vector<int> idx(static_cast<std::size_t>(n_region));
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(
                idx.begin(), idx.begin() + k, idx.end(),
                [&costs](int a, int b) { return costs[a] < costs[b]; });
            idx.resize(static_cast<std::size_t>(k));
            for (int j : idx) {
                if (costs[static_cast<std::size_t>(j)] < INF) {
                    neighbors.insert(j);
                }
            }
        }
        // `sorted(set(neighbors))` — std::set iteration order = ascending.
        region_neighbors[static_cast<std::size_t>(ri)].assign(
            neighbors.begin(), neighbors.end());
    }

    int total_nb = 0;
    for (const auto& nb : region_neighbors) total_nb += static_cast<int>(nb.size());
    nb_indices_.assign(static_cast<std::size_t>(total_nb), 0);
    nb_offsets_.assign(static_cast<std::size_t>(n_region + 1), 0);

    int pos = 0;
    for (int ri = 0; ri < n_region; ++ri) {
        nb_offsets_[static_cast<std::size_t>(ri)] = pos;
        for (int j : region_neighbors[static_cast<std::size_t>(ri)]) {
            nb_indices_[static_cast<std::size_t>(pos)] = static_cast<std::int64_t>(j);
            ++pos;
        }
    }
    nb_offsets_[static_cast<std::size_t>(n_region)] = pos;
}

// ---------------------------------------------------------------------------
// Region Viterbi DP.
// Port of `_region_dp` (region_optimizer.py:215-288).
// ---------------------------------------------------------------------------
RegionOptimizer::DpResult
RegionOptimizer::regionDp(int n_region,
                           int min_target,
                           int max_target,
                           int target_beats) const
{
    const std::size_t rows = static_cast<std::size_t>(max_target + 1);
    const std::size_t cols = static_cast<std::size_t>(n_region);
    const std::size_t sz   = rows * cols;

    std::vector<double>       dp(sz, INF);
    std::vector<std::int64_t> parent(sz, -1);
    std::vector<std::int64_t> cooldown_ssj(sz, 0);

    // region_optimizer.py:230.
    dp[1 * cols + 0] = 0.0;
    // R8 MARKER — region_optimizer.py:233.
    cooldown_ssj[1 * cols + 0] = REGION_SSJ_NO_RECENT_JUMP_SENTINEL;

    // ADR-083 sesja 92 — Min cut UI override. Default 0 → legacy
    // REGION_COOLDOWN=8 → bit-exact baseline. UI passes user slider value
    // 4-32 beats when slider dragged off default 16.
    const std::int64_t effective_cooldown =
        (min_seq_after_jump_override_ > 0)
            ? static_cast<std::int64_t>(min_seq_after_jump_override_)
            : REGION_COOLDOWN;

    // ADR-081 STATUS UPDATE 1 sesja 94 — beta-path tunables. Default
    // (region_beta_=false) preserves bit-exact baseline. Beta values
    // chosen to make multi-iteration short loops competitive with
    // single long backward jump:
    //   - cap raised 1.0 → 5.0 so quality discrimination survives across
    //     raw_w range (current cap saturates uniformly for raw_w ≥ 0.5,
    //     collapsing all backward jumps to identical cost).
    //   - base reduced 0.8 → 0.3 so per-jump tax doesn't dominate
    //     multi-iteration paths (3 × 0.3 × 0.16 = 0.14 < 1 × 0.8 × 0.8 = 0.64).
    const double effective_jump_cap =
        region_beta_ ? 5.0 : REGION_JUMP_COST_CAP;
    const double effective_jump_base =
        region_beta_ ? 0.3 : JUMP_PENALTY_BASE;

    // DP main loop.
    for (int t = 1; t < max_target; ++t) {
        const std::size_t t_row = static_cast<std::size_t>(t);
        for (int ri = 0; ri < n_region; ++ri) {
            const double d_ri = dp[t_row * cols + static_cast<std::size_t>(ri)];
            if (d_ri >= INF) continue;

            const std::int64_t ssj_ri = cooldown_ssj[t_row * cols + static_cast<std::size_t>(ri)];
            // Sequential-only branch while in cooldown — region_optimizer.py:240-248.
            if (ssj_ri < effective_cooldown) {
                const int rj = ri + 1;
                if (rj < n_region) {
                    const double w_seq = rW_[static_cast<std::size_t>(ri) * n_region + static_cast<std::size_t>(rj)];
                    const double cost  = d_ri + w_seq;
                    const std::size_t dst = (t_row + 1) * cols + static_cast<std::size_t>(rj);
                    if (cost < dp[dst]) {
                        dp[dst]           = cost;
                        parent[dst]       = static_cast<std::int64_t>(ri);
                        cooldown_ssj[dst] = ssj_ri + 1;
                    }
                }
                continue;
            }

            // Non-sequential allowed — region_optimizer.py:250-267.
            const int start = static_cast<int>(nb_offsets_[static_cast<std::size_t>(ri)]);
            const int end_  = static_cast<int>(nb_offsets_[static_cast<std::size_t>(ri + 1)]);
            for (int ni = start; ni < end_; ++ni) {
                const int rj = static_cast<int>(nb_indices_[static_cast<std::size_t>(ni)]);
                const double raw_w = rW_[static_cast<std::size_t>(ri) * n_region + static_cast<std::size_t>(rj)];
                if (raw_w >= INF) continue;

                // Sesja 100 (DEV-032) — skip transitions in blocked set.
                // Translate local (ri, rj) → global (entry+ri, entry+rj).
                if (blocked_transitions_ != nullptr && rj != ri + 1) {
                    const std::pair<int, int> key {
                        entry_beat_ + ri, entry_beat_ + rj
                    };
                    if (blocked_transitions_->find (key) != blocked_transitions_->end())
                        continue;
                }

                double cost = d_ri + raw_w;
                if (rj != ri + 1) {
                    // ADR-084 sesja 93 — Edit Length MULTIPLICATIVE jump-cost
                    // scale. Default 1.0 = bit-exact baseline. Slider=100 →
                    // 4× cost (fewer cuts); slider=0 → 0.25× (more cuts).
                    // R9 UNJUSTIFIED — region_optimizer.py:260.
                    // ADR-081 STATUS UPDATE 1 sesja 94 — beta path uses
                    // raised cap + reduced base; cf. effective_jump_cap +
                    // effective_jump_base above.
                    cost += edit_length_jump_scale_
                          * effective_jump_base
                          * std::min(effective_jump_cap, raw_w * REGION_JUMP_COST_SCALE);
                }

                const std::size_t dst = (t_row + 1) * cols + static_cast<std::size_t>(rj);
                if (cost < dp[dst]) {
                    dp[dst]     = cost;
                    parent[dst] = static_cast<std::int64_t>(ri);
                    // region_optimizer.py:265-267 — cooldown reset on non-seq,
                    // increment on seq.
                    cooldown_ssj[dst] = (rj == ri + 1) ? (ssj_ri + 1) : 0;
                }
            }
        }
    }

    // Find best endpoint — region_optimizer.py:270-288.
    double best_cost = INF;
    int    best_t    = max_target;
    int    best_ri   = n_region - 1;

    for (int t = min_target; t <= max_target; ++t) {
        const std::size_t t_row = static_cast<std::size_t>(t);
        for (int ri = 0; ri < n_region; ++ri) {
            const double d_val = dp[t_row * cols + static_cast<std::size_t>(ri)];
            if (d_val >= INF) continue;
            const int dist = std::abs(ri - (n_region - 1));
            // R10 + R11 — region_optimizer.py:279.
            const double terminal_penalty =
                (dist > REGION_TERMINAL_DIST_THRESHOLD)
                    ? static_cast<double>(dist) * REGION_TERMINAL_SCALE
                    : 0.0;
            // R12 DEFENSIVE — region_optimizer.py:281.
            const double duration_dev =
                std::abs(static_cast<double>(t - target_beats))
                / static_cast<double>(std::max(1, target_beats));
            // Reuse promoted ViterbiDP constant (R9/R10 audit — session-23
            // Q4 rec A, 2nd TU consumer promotion).
            const double duration_penalty = duration_dev * DURATION_PENALTY_WEIGHT;
            const double total = d_val + terminal_penalty + duration_penalty;
            if (total < best_cost) {
                best_cost = total;
                best_t    = t;
                best_ri   = ri;
            }
        }
    }

    DpResult out;
    out.best_cost = best_cost;
    out.best_t    = best_t;
    out.best_ri   = best_ri;
    out.parent    = std::move(parent);
    return out;
}

// ---------------------------------------------------------------------------
// ADR-081 STATUS UPDATE 2 sesja 94 — Region β-model loop synthesizer.
//
// Empirical motivation: sesja 94 corpus diagnostic showed standard regionDp
// produces n_transitions=1 in 47/48 corpus cases despite 800-1700+ viable
// backward candidates per region. DP minimizes total body cost; non-zero
// per-jump cost structurally biases toward fewer jumps. User intent ("find
// ideal 8-sec inner loop and repeat 3×") cannot emerge from cost rebalancing
// alone — requires explicit multi-iteration path structure.
//
// Algorithm: enumerate viable backward (i, j) candidates from rW_; for each,
// compute optimal N iterations to land target_beats; pick (i, j, N) with
// minimum cost. Path structure:
//
//     [0, 1, ..., i] + N × [j, j+1, ..., i] + [i+1, ..., n_region-1]
//
// Each iteration is "jump i→j followed by sequential return j..i". Total
// path length = n_region + N × (i - j + 1). Cooldown constraint L = i-j ≥
// effective_cooldown (so within-iteration sequential satisfies DP cooldown
// semantics; equivalent to making path representable as DP-valid trajectory).
//
// Cost composition:
//   cost = N × per_jump_cost(i, j)
//        + (small) sequential body cost
//        + duration_penalty(t_actual, target_beats)
// where per_jump_cost uses the same formula as regionDp (raw_w + base ×
// min(cap, raw_w * scale)) — we don't double-rebalance here. Cost rebalance
// constants from STATUS UPDATE 1 still effect per-jump cost.
//
// Returns LoopSynthResult with `path` empty if no valid (i, j, N) found
// within [min_target, max_target]; caller falls back to regionDp.
// ---------------------------------------------------------------------------
RegionOptimizer::LoopSynthResult
RegionOptimizer::regionLoopSynthesize(int    n_region,
                                       int    min_target,
                                       int    max_target,
                                       int    target_beats,
                                       int    effective_cooldown,
                                       int    entry_beat) const
{
    LoopSynthResult best;
    best.total_cost = INF;

    // Beta-path constants (mirror regionDp branch values).
    const double effective_jump_cap =
        region_beta_ ? 5.0 : REGION_JUMP_COST_CAP;
    const double effective_jump_base =
        region_beta_ ? 0.3 : JUMP_PENALTY_BASE;

    // Sequential body cost — sum of rW_[ri][ri+1] for ri in pre-loop, loop,
    // and post-loop traversals. Cheap to precompute.
    double seq_cost_full_traversal = 0.0;
    for (int ri = 0; ri + 1 < n_region; ++ri) {
        const double v = rW_[static_cast<std::size_t>(ri) * n_region + (ri + 1)];
        if (v < INF) seq_cost_full_traversal += v;
    }

    // Enumerate viable (i, j) backward candidates with i - j ≥ cooldown.
    // i ∈ [cooldown, n_region - 2] (must have at least one post-loop seq edge).
    // j ∈ [0, i - cooldown].
    //
    // Sesja-94 iter 4 — adaptive cooldown floor. When user-set Min cut
    // (= effective_cooldown) exceeds n_region - 2, the strict constraint
    // would skip ALL (i, j) → synth empty → DP fallback → tile_whole_region
    // (the dreaded user-perceived "loop the whole region" failure mode).
    // For small regions (e.g., n_region=11 with Min cut=16 default), this
    // happens silently. Adaptive: when strict cd exceeds region-fit, try
    // again with a region-scaled floor. Best effort beats tile-region.
    int cd = std::max(1, effective_cooldown);
    if (cd > n_region - 2) {
        cd = std::max(2, n_region / 3);
    }
    int candidates_evaluated = 0;
    int chosen_i_local = -1, chosen_j_local = -1, chosen_N_local = 0;

    for (int i = cd; i <= n_region - 2; ++i) {
        for (int j = 0; j <= i - cd; ++j) {
            const double raw_w = rW_[static_cast<std::size_t>(i) * n_region + j];
            if (raw_w >= INF) continue;

            // Sesja 100 (DEV-032) — skip blocked (i, j) tuples. β-path
            // emits N-loop variants where the same (i, j) repeats N times;
            // blocking the pair removes the entire family from K-best.
            // Translate local → global (entry_beat_ + i, entry_beat_ + j).
            if (blocked_transitions_ != nullptr) {
                const std::pair<int, int> key {
                    entry_beat_ + i, entry_beat_ + j
                };
                if (blocked_transitions_->find (key) != blocked_transitions_->end())
                    continue;
            }
            ++candidates_evaluated;

            const int loop_step_len = i - j + 1;  // 1 jump + (i-j) seq

            // N optimal: min cost via N × (i-j+1) ≈ target_beats - n_region.
            // Range search for N: pick N closest to (target_beats - n_region) /
            // loop_step_len, then check if t_actual ∈ [min_target, max_target].
            const int target_extra = target_beats - n_region;
            if (target_extra <= 0) continue;  // Not extending sufficiently

            const int N_round =
                static_cast<int>(std::round(static_cast<double>(target_extra)
                                            / loop_step_len));
            // Try N_round and ±1 to find best fit within tolerance window.
            for (int dn = -1; dn <= 1; ++dn) {
                const int N = N_round + dn;
                if (N < 1) continue;
                const int t_actual = n_region + N * loop_step_len;
                if (t_actual < min_target || t_actual > max_target) continue;

                // Compute body cost. ADR-084 sesja 93 — edit_length_jump_scale_
                // multiplies the jump-tax term (mirrors DP path at
                // RegionOptimizer.cpp:443-445). Slider=100 → scale=4.0 →
                // jumps 4× more expensive → synth prefers fewer iterations
                // (larger L per iteration). Slider=0 → scale=0.25 → jumps
                // 0.25× → synth prefers more iterations (shorter L). Default
                // scale=1.0 = bit-exact baseline.
                const double per_jump_cost = raw_w
                    + edit_length_jump_scale_
                    * effective_jump_base
                    * std::min(effective_jump_cap, raw_w * REGION_JUMP_COST_SCALE);

                // Sequential cost: pre-loop (i edges) + N × within-loop (i-j edges)
                // + post-loop (n_region-1-i edges). Approximate by:
                //   per_seq_avg = seq_cost_full_traversal / max(1, n_region-1)
                //   total_seq_cost = (i + N×(i-j) + n_region-1-i) × per_seq_avg
                const double per_seq_avg =
                    (n_region > 1)
                        ? seq_cost_full_traversal / static_cast<double>(n_region - 1)
                        : 0.0;
                const int seq_edges = i + N * (i - j) + (n_region - 1 - i);
                const double seq_cost = static_cast<double>(seq_edges) * per_seq_avg;

                const double duration_dev =
                    std::abs(static_cast<double>(t_actual - target_beats))
                    / static_cast<double>(std::max(1, target_beats));
                const double duration_penalty =
                    duration_dev * DURATION_PENALTY_WEIGHT;

                const double total_cost =
                    static_cast<double>(N) * per_jump_cost
                    + seq_cost
                    + duration_penalty;

                if (total_cost < best.total_cost) {
                    best.total_cost = total_cost;
                    chosen_i_local = i;
                    chosen_j_local = j;
                    chosen_N_local = N;
                }
            }
        }
    }

    // No viable (i, j, N) — return empty result; caller falls back to DP.
    if (chosen_i_local < 0) {
        return best;  // best.path is empty
    }

    // Build path explicitly:
    //   [0, 1, ..., i] + N × [j, j+1, ..., i] + [i+1, ..., n_region-1]
    const int i = chosen_i_local;
    const int j = chosen_j_local;
    const int N = chosen_N_local;

    std::vector<std::int64_t>& path = best.path;
    path.reserve(static_cast<std::size_t>(n_region + N * (i - j + 1)));

    // Pre-loop: 0, 1, ..., i.
    for (int p = 0; p <= i; ++p) {
        path.push_back(static_cast<std::int64_t>(entry_beat + p));
    }
    // N iterations: each appends j, j+1, ..., i.
    for (int n = 0; n < N; ++n) {
        for (int p = j; p <= i; ++p) {
            path.push_back(static_cast<std::int64_t>(entry_beat + p));
        }
    }
    // Post-loop: i+1, i+2, ..., n_region-1.
    for (int p = i + 1; p < n_region; ++p) {
        path.push_back(static_cast<std::int64_t>(entry_beat + p));
    }

    // Transitions: each loop iteration contributes one (i, j) backward jump
    // in absolute beat indices.
    best.transitions.reserve(static_cast<std::size_t>(N));
    for (int n = 0; n < N; ++n) {
        best.transitions.emplace_back(entry_beat + i, entry_beat + j);
    }

    best.chosen_i = i;
    best.chosen_j = j;
    best.chosen_N = N;

    // Diagnostic log — sesja-94-only debug aid (removed at handover).
    if (region_beta_) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            std::string log_path = std::string(home) + "/Desktop/reamix_diag.log";
            std::FILE* f = std::fopen(log_path.c_str(), "a");
            if (f != nullptr) {
                const double final_raw_w =
                    rW_[static_cast<std::size_t>(i) * n_region + j];
                const double per_jump_cost_logged = final_raw_w
                    + edit_length_jump_scale_
                    * effective_jump_base
                    * std::min(effective_jump_cap,
                               final_raw_w * REGION_JUMP_COST_SCALE);
                std::fprintf(f, "REGION_BETA_LOOPSYNTH n_region=%d target_beats=%d "
                             "min_target=%d max_target=%d "
                             "candidates_evaluated=%d "
                             "chosen_i=%d chosen_j=%d chosen_N=%d "
                             "loop_length_beats=%d t_actual=%d "
                             "raw_w=%.4f per_jump_cost=%.4f total_cost=%.4f "
                             "edit_length_scale=%.3f effective_cooldown=%d "
                             "abs_jump=%d->%d\n",
                             n_region, target_beats, min_target, max_target,
                             candidates_evaluated,
                             i, j, N, i - j, n_region + N * (i - j + 1),
                             final_raw_w,
                             per_jump_cost_logged, best.total_cost,
                             edit_length_jump_scale_, effective_cooldown,
                             entry_beat + i, entry_beat + j);
                std::fclose(f);
            }
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Fallback: play region straight, repeated if extending.
// Port of `_region_fallback` (region_optimizer.py:290-305).
// ---------------------------------------------------------------------------
RemixPath RegionOptimizer::regionFallback(int    entry_beat,
                                           int    exit_beat,
                                           int    target_beats,
                                           double target_duration,
                                           double region_duration) const
{
    // R13 DEFENSIVE — region_optimizer.py:296.
    const int repeats = std::max(
        REGION_FALLBACK_MIN_REPEATS,
        pyIntRound(target_duration / std::max(REGION_FALLBACK_MIN_DURATION, region_duration)));

    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(repeats) * (exit_beat - entry_beat));
    for (int r = 0; r < repeats; ++r) {
        for (int b = entry_beat; b < exit_beat; ++b) {
            indices.push_back(b);
        }
    }
    // region_optimizer.py:301: `indices[:target_beats]`.
    if (static_cast<int>(indices.size()) > target_beats) {
        indices.resize(static_cast<std::size_t>(target_beats));
    }

    RemixPath p;
    p.beat_indices   = std::move(indices);
    p.total_cost     = 0.0;
    p.duration_beats = std::min(static_cast<int>(p.beat_indices.size()), target_beats);
    return p;
}

// ---------------------------------------------------------------------------
// Backtrace.
// Port of `_region_backtrace` (region_optimizer.py:307-320).
// ---------------------------------------------------------------------------
std::vector<std::int64_t>
RegionOptimizer::regionBacktrace(int                              best_t,
                                  int                              best_ri,
                                  const std::vector<std::int64_t>& parent,
                                  int                              n_region,
                                  int                              max_target,
                                  int                              entry_beat) const
{
    (void) max_target;  // only used for shape/bounds check already done
    const std::size_t cols = static_cast<std::size_t>(n_region);

    std::vector<int> path_region;
    int ri = best_ri;
    for (int t = best_t; t > 0; --t) {
        path_region.push_back(ri);
        ri = static_cast<int>(parent[static_cast<std::size_t>(t) * cols + static_cast<std::size_t>(ri)]);
    }
    std::reverse(path_region.begin(), path_region.end());

    std::vector<std::int64_t> out;
    out.reserve(path_region.size());
    for (int r : path_region) {
        out.push_back(static_cast<std::int64_t>(entry_beat + r));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Extract transitions + metadata.
// Port of `_region_transitions` (region_optimizer.py:322-357).
// ---------------------------------------------------------------------------
void RegionOptimizer::regionTransitions(
    const std::vector<std::int64_t>&                              path,
    const std::map<std::pair<int, int>, TransitionCandidate>*     region_candidates,
    std::vector<std::pair<int, int>>&                             out_transitions,
    std::map<std::pair<int, int>, std::map<std::string, double>>& out_metadata) const
{
    // Build `all_candidates = dict(self._candidates)` + `update(region_candidates)`.
    // region_optimizer.py:328-330. Python dict update: region_candidates keys
    // override global. C++ std::map::insert-or-assign loop matches semantics.
    std::map<std::pair<int, int>, TransitionCandidate> all_candidates;
    if (candidates_ != nullptr) {
        for (const auto& kv : *candidates_) {
            all_candidates.insert_or_assign(kv.first, kv.second);
        }
    }
    if (region_candidates != nullptr) {
        for (const auto& kv : *region_candidates) {
            all_candidates.insert_or_assign(kv.first, kv.second);
        }
    }

    for (std::size_t idx = 0; idx + 1 < path.size(); ++idx) {
        const int i = static_cast<int>(path[idx]);
        const int j = static_cast<int>(path[idx + 1]);
        // R14 CLEAN — j != i+1 consecutive-beat semantic (session-22 reuse).
        if (j != i + 1) {
            out_transitions.emplace_back(i, j);
            std::map<std::string, double> meta;
            auto it = all_candidates.find({i, j});
            if (it != all_candidates.end()) {
                const TransitionCandidate& cand = it->second;
                meta["quality_score"]         = cand.quality_score;
                meta["waveform_similarity"]   = cand.waveform_similarity;
                meta["successor_similarity"]  = cand.successor_similarity;
                meta["edge_splice_similarity"] = cand.edge_splice_similarity;
                meta["chroma_distance"]       = cand.chroma_distance;
                meta["energy_diff_db"]        = cand.energy_diff_db;
                // R15 CLEAN — alignment_offset_sec divisor.
                // region_optimizer.py:349-352: `max(1, sample_rate)`.
                meta["alignment_offset_sec"]  =
                    static_cast<double>(cand.alignment_lag_samples)
                    / static_cast<double>(std::max(1, sample_rate_));
                meta["total_cost"]            = cand.total_cost;
            }
            out_metadata[{i, j}] = std::move(meta);
        }
    }
}

// ---------------------------------------------------------------------------
// remix_k_best — Region mirror of CleanOptimizer::remix_k_best (sesja 100,
// DEV-032, ADR-091 — agent-side mirror of ADR-048 Duration K-best for Region).
// ---------------------------------------------------------------------------
//
// Algorithm: get base region path → for each (fb, tb) jump in base.transitions,
// build augmented_blocked = caller_blocked ∪ {(fb, tb)}, re-call remix() with
// augmented set, dedup by beat_indices, accumulate up to k unique paths.
// Both DP paths (regionDp + regionLoopSynthesize β-path) honour blocked_set
// via blocked_transitions_ member set at remix() entry.
std::vector<RemixPath>
RegionOptimizer::remix_k_best(double                                                    target_duration,
                              double                                                    region_start_sec,
                              double                                                    region_end_sec,
                              const double*                                             region_W,
                              int                                                       n_region_input,
                              const std::map<std::pair<int, int>, TransitionCandidate>* region_candidates,
                              int                                                       k,
                              const std::set<std::pair<int, int>>*                      blocked_transitions)
{
    // Mirror Optimizer.cpp L816 short-circuit semantics for n_beats_<2.
    if (n_beats_ < REGION_MIN_VIABLE_DP) {
        return std::vector<RemixPath>{ remix(target_duration, region_start_sec,
                                             region_end_sec, region_W,
                                             n_region_input, region_candidates,
                                             blocked_transitions) };
    }

    // Base path with caller's blocked set applied.
    RemixPath best = remix(target_duration, region_start_sec, region_end_sec,
                           region_W, n_region_input, region_candidates,
                           blocked_transitions);

    if (best.transitions.empty() || k <= 1) {
        return std::vector<RemixPath>{ std::move(best) };
    }

    std::vector<RemixPath> paths;
    paths.reserve (static_cast<std::size_t>(k));
    paths.push_back (std::move(best));

    std::set<std::vector<int>> seen;
    seen.insert (paths.front().beat_indices);

    const std::vector<std::pair<int, int>> base_transitions = paths.front().transitions;

    for (const auto& tr : base_transitions) {
        if (paths.size() >= static_cast<std::size_t>(k)) break;

        std::set<std::pair<int, int>> augmented;
        if (blocked_transitions != nullptr) {
            augmented = *blocked_transitions;
        }
        augmented.insert (tr);

        RemixPath alt = remix(target_duration, region_start_sec, region_end_sec,
                              region_W, n_region_input, region_candidates,
                              &augmented);

        if (! alt.beat_indices.empty()
            && seen.find (alt.beat_indices) == seen.end()) {
            seen.insert (alt.beat_indices);
            paths.push_back (std::move(alt));
        }
    }

    return paths;
}

// ---------------------------------------------------------------------------
// remix_variation — Region mirror of CleanOptimizer::remix_variation
// (sesja 100, DEV-032, ADR-091).
// ---------------------------------------------------------------------------
RemixPath
RegionOptimizer::remix_variation(double                                                    target_duration,
                                 double                                                    region_start_sec,
                                 double                                                    region_end_sec,
                                 const double*                                             region_W,
                                 int                                                       n_region_input,
                                 const std::map<std::pair<int, int>, TransitionCandidate>* region_candidates,
                                 int                                                       variation_idx,
                                 const std::set<std::pair<int, int>>*                      blocked_transitions)
{
    const int v = std::max (0, variation_idx);
    const int k = std::max (2, v + 1);   // mirror Python `_remix.py:143`
    auto paths = remix_k_best (target_duration, region_start_sec, region_end_sec,
                               region_W, n_region_input, region_candidates,
                               k, blocked_transitions);
    if (paths.empty()) {
        return RemixPath{};
    }
    const int idx = std::min (v, static_cast<int>(paths.size()) - 1);  // mirror `_remix.py:146`
    return paths[static_cast<std::size_t>(idx)];
}

} // namespace reamix::remix
