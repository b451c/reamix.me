#include "remix/RegionOptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// ADR-115 E8 (sesja 116, DEV-090) — v2 region path search constants.
// C++-canonical (ADR-065), no library citation; validated by
// tests/parity/test_region_dp_v2.cpp on synthetic pools.
//   kRegionQualityBand   non-sequential pairs whose cost exceeds the region's
//                        best pair by more than this are dropped before the
//                        DP (cost = 1 - quality; 0.08 = "within 8 quality
//                        points of the best"). Sesja-115 Billie Jean pools:
//                        best 0.59 vs chosen 0.47 / 0.54 loops would be cut.
//   kRegionRepeatPenalty paid by a jump that repeats (or lies within a
//                        Gaussian neighbourhood, sigma = one bar, of) the
//                        previous jump's pair; smaller than a jump tax
//                        (0.3 x min(5, 2 w) ~ 0.3 at w = 0.5) so the DP
//                        alternates between band pairs but never adds a cut
//                        to avoid a repeat.
//   kRegionMaxBandPairs  cap on the jump alphabet (DP state carries the
//                        previous pair: (t, beat, pair) states).
constexpr double kRegionQualityBand   = 0.08;
constexpr double kRegionRepeatPenalty = 0.15;
constexpr int    kRegionMaxBandPairs  = 16;

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
    , v2_scoring_(in.v2_scoring)  // ADR-115 E8 sesja 116
    , bar_beats_(std::max(1, in.bar_beats))
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
    buildRegionNeighbors(n_region, rW_);

    // ADR-115 E8 (sesja 116, DEV-090) — v2 region path search: quality band
    // + repetition-penalty DP with cooldown = one measured bar (Min cut UI
    // override still honoured until P3 removes it). Falls through to the
    // synthesizer / legacy DP when it finds no path.
    if (v2_scoring_) {
        const std::int64_t cooldown_v2 =
            (min_seq_after_jump_override_ > 0)
                ? static_cast<std::int64_t>(min_seq_after_jump_override_)
                : static_cast<std::int64_t>(bar_beats_);
        V2Result v2 = regionDpV2(n_region, min_target_, max_target_, target_beats_,
                                 cooldown_v2, entry_beat);
        if (v2.ok) {
            std::vector<std::pair<int, int>>                             transitions;
            std::map<std::pair<int, int>, std::map<std::string, double>> metadata;
            regionTransitions(v2.path, region_candidates, transitions, metadata);
            RemixPath out;
            out.beat_indices.reserve(v2.path.size());
            for (std::int64_t idx : v2.path) out.beat_indices.push_back(static_cast<int>(idx));
            out.total_cost          = v2.score;
            out.duration_beats      = static_cast<int>(out.beat_indices.size());
            out.transitions         = std::move(transitions);
            out.transition_metadata = std::move(metadata);
            return out;
        }
        buildRegionNeighbors(n_region, rW_);   // restore the plain lists for the fallbacks
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
    // ADR-083 sesja 92 — Min cut UI override. Default 0 → legacy
    // REGION_COOLDOWN=8 → bit-exact baseline. UI passes user slider value
    // 4-32 beats when slider dragged off default 16.
    const std::int64_t effective_cooldown =
        (min_seq_after_jump_override_ > 0)
            ? static_cast<std::int64_t>(min_seq_after_jump_override_)
            : REGION_COOLDOWN;
    DpResult dp_res = regionDp(n_region, min_target_, max_target_, target_beats_,
                               rW_, effective_cooldown);

    // Check failure — region_optimizer.py:115-119.
    if (dp_res.best_cost >= INF) {
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
void RegionOptimizer::buildRegionNeighbors(int n_region, const std::vector<double>& W)
{
    std::vector<std::vector<int>> region_neighbors(static_cast<std::size_t>(n_region));

    for (int ri = 0; ri < n_region; ++ri) {
        std::set<int> neighbors;
        if (ri + 1 < n_region) neighbors.insert(ri + 1);

        // Copy row costs, mask self + ri+1.
        std::vector<double> costs(static_cast<std::size_t>(n_region));
        for (int j = 0; j < n_region; ++j) {
            costs[static_cast<std::size_t>(j)] =
                W[static_cast<std::size_t>(ri) * n_region + j];
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
                           int target_beats,
                           const std::vector<double>& W,
                           std::int64_t cooldown) const
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

    // Cooldown: legacy REGION_COOLDOWN (8) or the Min cut UI override
    // (ADR-083 sesja 92); one measured bar on the v2 path (sesja 116).
    const std::int64_t effective_cooldown = cooldown;

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
                    const double w_seq = W[static_cast<std::size_t>(ri) * n_region + static_cast<std::size_t>(rj)];
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
                const double raw_w = W[static_cast<std::size_t>(ri) * n_region + static_cast<std::size_t>(rj)];
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
// ADR-115 E8 (sesja 116, DEV-090) — v2 region path search.
// See RegionOptimizer.h for the algorithm summary.
//
// Why a "last pair" state instead of iterative re-weighting: an additive DP
// under a matrix that penalises every use of a pair only ever switches to
// another pair wholesale (pure A x N -> pure B x M); it cannot express "A
// then B then A". The patent applies its Gaussian repetition penalty
// greedily after each chosen segment, which alternates naturally; carrying
// the previous jump's pair in the DP state is the exact equivalent (a jump
// that repeats or neighbours the previous one pays the penalty once).
// ---------------------------------------------------------------------------
RegionOptimizer::V2Result
RegionOptimizer::regionDpV2(int          n_region,
                            int          min_target,
                            int          max_target,
                            int          target_beats,
                            std::int64_t cooldown,
                            int          entry_beat)
{
    V2Result best;
    const std::size_t n = static_cast<std::size_t>(n_region);
    auto cell = [&](int ri, int rj) { return static_cast<std::size_t>(ri) * n + static_cast<std::size_t>(rj); };

    auto isBlocked = [&](int ri, int rj) {
        if (blocked_transitions_ == nullptr) return false;
        return blocked_transitions_->find({entry_beat_ + ri, entry_beat_ + rj})
               != blocked_transitions_->end();
    };

    // 1. Quality band around the region's best non-sequential pair; the
    //    surviving pairs (at most kRegionMaxBandPairs, best first) are the
    //    DP's jump alphabet.
    struct Pair { int ri, rj; double w; };
    std::vector<Pair> pool;
    for (int ri = 0; ri < n_region; ++ri)
        for (int rj = 0; rj < n_region; ++rj) {
            if (rj == ri || rj == ri + 1) continue;
            const double w = rW_[cell(ri, rj)];
            if (w >= INF || isBlocked(ri, rj)) continue;
            pool.push_back({ri, rj, w});
        }
    if (pool.empty()) return best;
    std::sort(pool.begin(), pool.end(), [](const Pair& a, const Pair& b) {
        return a.w < b.w || (a.w == b.w && (a.ri < b.ri || (a.ri == b.ri && a.rj < b.rj)));
    });
    const double c_best = pool.front().w;
    std::vector<Pair> pairs;
    for (const Pair& p : pool) {
        if (p.w > c_best + kRegionQualityBand) break;
        if (static_cast<int>(pairs.size()) >= kRegionMaxBandPairs) break;
        pairs.push_back(p);
    }
    const int P = static_cast<int>(pairs.size());   // index P = "no jump yet"
    std::vector<int> pair_idx(n * n, -1);
    for (int q = 0; q < P; ++q) pair_idx[cell(pairs[q].ri, pairs[q].rj)] = q;

    // 2. DP over (t, ri, last pair). Sequential cost from rW_; jump cost =
    //    w + jump tax (beta constants) + repetition penalty vs the last pair.
    const double jump_cap  = region_beta_ ? 5.0 : REGION_JUMP_COST_CAP;
    const double jump_base = region_beta_ ? 0.3 : JUMP_PENALTY_BASE;
    const double sigma2    = 2.0 * static_cast<double>(bar_beats_) * static_cast<double>(bar_beats_);
    auto repeatPenalty = [&](int p, int q) {
        if (p >= P) return 0.0;
        const double di = static_cast<double>(pairs[p].ri - pairs[q].ri);
        const double dj = static_cast<double>(pairs[p].rj - pairs[q].rj);
        return kRegionRepeatPenalty * std::exp(-(di * di + dj * dj) / sigma2);
    };

    const std::size_t L    = static_cast<std::size_t>(P + 1);
    const std::size_t rows = static_cast<std::size_t>(max_target + 1);
    const std::size_t sz   = rows * n * L;
    auto st = [&](int t, int ri, int p) {
        return (static_cast<std::size_t>(t) * n + static_cast<std::size_t>(ri)) * L + static_cast<std::size_t>(p);
    };
    std::vector<double>       dp(sz, INF);
    std::vector<std::int32_t> par_ri(sz, -1);
    std::vector<std::int32_t> par_p(sz, -1);
    std::vector<std::int32_t> ssj(sz, 0);

    dp[st(1, 0, P)]  = 0.0;
    ssj[st(1, 0, P)] = static_cast<std::int32_t>(REGION_SSJ_NO_RECENT_JUMP_SENTINEL);

    auto relax = [&](std::size_t dst, double cost, int from_ri, int from_p, std::int32_t new_ssj) {
        if (cost < dp[dst]) {
            dp[dst]     = cost;
            par_ri[dst] = from_ri;
            par_p[dst]  = from_p;
            ssj[dst]    = new_ssj;
        }
    };

    for (int t = 1; t < max_target; ++t) {
        for (int ri = 0; ri < n_region; ++ri) {
            const double w_seq = (ri + 1 < n_region) ? rW_[cell(ri, ri + 1)] : INF;
            for (int p = 0; p <= P; ++p) {
                const std::size_t src = st(t, ri, p);
                const double d = dp[src];
                if (d >= INF) continue;
                const std::int32_t s = ssj[src];
                if (w_seq < INF)
                    relax(st(t + 1, ri + 1, p), d + w_seq, ri, p, s + 1);
                if (s < cooldown) continue;   // sequential only while in cooldown
                for (int q = 0; q < P; ++q) {
                    if (pairs[q].ri != ri) continue;
                    const double w = pairs[q].w;
                    const double cost = d + w
                        + edit_length_jump_scale_ * jump_base * std::min(jump_cap, w * REGION_JUMP_COST_SCALE)
                        + repeatPenalty(p, q);
                    relax(st(t + 1, pairs[q].rj, q), cost, ri, p, 0);
                }
            }
        }
    }

    // 3. Endpoint: terminal + duration penalties as in regionDp.
    int best_t = -1, best_ri = -1, best_p = -1;
    for (int t = min_target; t <= max_target; ++t)
        for (int ri = 0; ri < n_region; ++ri) {
            const int dist = std::abs(ri - (n_region - 1));
            const double terminal = (dist > REGION_TERMINAL_DIST_THRESHOLD)
                ? static_cast<double>(dist) * REGION_TERMINAL_SCALE : 0.0;
            const double duration_dev = std::abs(static_cast<double>(t - target_beats))
                                      / static_cast<double>(std::max(1, target_beats));
            const double extra = terminal + duration_dev * DURATION_PENALTY_WEIGHT;
            for (int p = 0; p <= P; ++p) {
                const double d = dp[st(t, ri, p)];
                if (d >= INF) continue;
                const double total = d + extra;
                if (total < best.score) { best.score = total; best_t = t; best_ri = ri; best_p = p; }
            }
        }
    if (best_t < 0) return best;

    std::vector<int> rel;
    for (int t = best_t, ri = best_ri, p = best_p; t > 0; --t) {
        rel.push_back(ri);
        const std::size_t s = st(t, ri, p);
        const int pri = par_ri[s], pp = par_p[s];
        ri = pri; p = pp;
    }
    std::reverse(rel.begin(), rel.end());
    best.path.reserve(rel.size());
    for (int r : rel) best.path.push_back(static_cast<std::int64_t>(entry_beat + r));
    best.ok = true;

    // Dev-only diagnostic (sesja 116): REAMIX_REGION_DEBUG=<path> appends the
    // band and the chosen path's jumps.
    if (const char* dbg = std::getenv("REAMIX_REGION_DEBUG")) {
        if (FILE* f = std::fopen(dbg, "a")) {
            std::fprintf(f, "# v2 n_region=%d target=%d cooldown=%lld c_best=%.4f pool=%d band=%d score=%.4f len=%d band:",
                         n_region, target_beats, static_cast<long long>(cooldown), c_best,
                         static_cast<int>(pool.size()), P, best.score, static_cast<int>(best.path.size()));
            for (const Pair& p : pairs)
                std::fprintf(f, " %d->%d(%.3f)", entry_beat + p.ri, entry_beat + p.rj, p.w);
            std::fprintf(f, " jumps:");
            for (std::size_t k = 0; k + 1 < best.path.size(); ++k)
                if (best.path[k + 1] != best.path[k] + 1)
                    std::fprintf(f, " %lld->%lld", static_cast<long long>(best.path[k]),
                                 static_cast<long long>(best.path[k + 1]));
            std::fprintf(f, "\n");
            std::fclose(f);
        }
    }
    return best;
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

    // Sesja 115 diagnostic (dev only): REAMIX_REGION_DEBUG=<path> appends one
    // line per viable (i, j) with raw_w, N_round and the chosen tuple so the
    // loop-pair choice can be audited from the calibration harness.
    if (const char* dbg = std::getenv("REAMIX_REGION_DEBUG")) {
        if (FILE* f = std::fopen(dbg, "a")) {
            std::fprintf(f, "# n_region=%d target=%d cd=%d evaluated=%d chosen=(%d,%d,N=%d) cost=%.4f\n",
                         n_region, target_beats, cd, candidates_evaluated,
                         chosen_i_local, chosen_j_local, chosen_N_local, best.total_cost);
            for (int i2 = cd; i2 <= n_region - 2; ++i2)
                for (int j2 = 0; j2 <= i2 - cd; ++j2) {
                    const double w = rW_[static_cast<std::size_t>(i2) * n_region + j2];
                    if (w < INF) std::fprintf(f, "%d,%d,%d,%.4f\n", entry_beat + i2, entry_beat + j2, i2 - j2, w);
                }
            std::fclose(f);
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
