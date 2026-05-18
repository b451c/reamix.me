// test_region_inner_loop — sesja 94 (ADR-081 STATUS UPDATE 1) self-validation.
//
// Self-validation parity test for the Region β-model "inner-loop synthesizer"
// cost-function rebalance. Per ADR-065 + memory
// `feedback_python_no_longer_source_of_truth.md`: parity tests for new C++-
// canonical extensions validate against hand-computed values + invariants,
// not vs Python ground truth.
//
// CANONICAL SOURCE: `RegionOptimizer.cpp::buildRegionCostMatrix` +
// `RegionOptimizer.cpp::regionDp` (post ADR-081 STATUS UPDATE 1 cap raise +
// backward penalty reduce + jump base reduce when `region_beta_=true`).
//
// REBALANCE (ADR-081 STATUS UPDATE 1 sesja 94 — supersedes ADR-081 PROPOSED
// § B source-wide-transitions sketch per pre-code audit empirical 48/48 = 1
// transition finding):
//   - REGION_JUMP_COST_CAP raised 1.0 → 5.0 in beta path (preserves quality
//     discrimination across raw_w range; cap retained at 5.0 as outlier guard).
//   - REGION_BACKWARD_JUMP_PENALTY_EXTEND reduced 0.5 → 0.05 in beta path
//     (extending mode; cooldown still bounds runaway loops).
//   - JUMP_PENALTY_BASE effective scale reduced 0.8 → 0.3 in beta path
//     (multi-iteration paths competitive with single-jump alternatives).
//
// Test invariants:
//   1. `region_beta` field default = false in `RegionOptimizerInputs` (bit-
//      exact baseline preservation across all 48 corpus parity cases).
//   2. Legacy path (`region_beta=false`) completes successfully on synthetic
//      extending fixture with finite total_cost (no degeneracy regression).
//   3. Beta path (`region_beta=true`) completes successfully on same fixture
//      with finite total_cost; cost arithmetic distinct from legacy
//      (verifies the rebalance is actually wired through DP).
//   4. Beta-on-low-quality-uniform-fixture: no degenerate path, no NaN/INF
//      leak, cost stays bounded.

#include "remix/Path.h"
#include "remix/RegionOptimizer.h"
#include "remix/TransitionCost.h"  // INF, TransitionCandidate

#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

using reamix::remix::INF;
using reamix::remix::RegionOptimizer;
using reamix::remix::RegionOptimizerInputs;
using reamix::remix::RemixPath;
using reamix::remix::TransitionCandidate;

namespace {

// Synthetic 12-beat region fixture. Track has n_total=13 beats (one buffer
// beat past the region) so exit_beat=12 and n_region (= exit_beat - entry_beat)
// = 12 — matches our fixture's rW dimension exactly. Without the buffer
// beat, RegionOptimizer.cpp:209 clamps exit_beat to n_beats-1, producing
// n_region=11 and silently mis-indexing into a 12×12 rW (INF cells where
// loop point should be → fallback path → total_cost=0).
//
// All sequential edges rW[i][i+1] = 0.0 (free forward traversal). One
// backward-jump opportunity at rW[8][3] = `loop_cost` parameter (cheap loop
// = 0.05; medium = 0.4; very-low-quality = 0.95). All other backward /
// forward-skip cells = INF.
//
// avg_beat_duration = 0.5 sec → 12-beat region = 6.0 sec. Target durations
// in tests below select extending mode (target > region_duration).
struct Fixture {
    std::vector<double> rW;          // (n_region * n_region) row-major
    std::vector<double> beat_times;  // (n_total,)
    std::map<std::pair<int, int>, TransitionCandidate> region_candidates;
    int    n_region        = 12;
    int    n_total         = 13;  // n_region + 1 buffer beat past region
    double avg_beat        = 0.5;
    double region_start    = 0.0;
    double region_end      = 6.0;  // exit_beat = argminAbsDiff(6.0) = 12
    double tolerance_sec   = 3.0;
};

Fixture buildFixture(double loop_cost) {
    Fixture f;
    const int n = f.n_region;
    f.rW.assign(static_cast<std::size_t>(n) * n, INF);

    // Sequential cells: rW[i][i+1] = 0.0 for i ∈ [0, n-2].
    for (int i = 0; i + 1 < n; ++i) {
        f.rW[(std::size_t) i * n + (i + 1)] = 0.0;
    }

    // Single backward-jump opportunity: rW[8][3] = loop_cost.
    f.rW[(std::size_t) 8 * n + 3] = loop_cost;

    // beat_times: linear 0.0, 0.5, 1.0, ..., 5.5 (n=12 beats).
    f.beat_times.resize(static_cast<std::size_t>(f.n_total));
    for (int i = 0; i < f.n_total; ++i) {
        f.beat_times[(std::size_t) i] = static_cast<double>(i) * f.avg_beat;
    }
    return f;
}

bool runRegionDP(const Fixture& fix,
                 bool region_beta,
                 double target_duration,
                 RemixPath& out_path)
{
    RegionOptimizerInputs roin{};
    roin.n_beats               = fix.n_total;
    roin.beat_times            = fix.beat_times.data();
    roin.avg_beat_duration     = fix.avg_beat;
    roin.duration_tolerance_sec = fix.tolerance_sec;
    roin.candidates            = nullptr;
    roin.sample_rate           = 22050;
    roin.downbeats             = nullptr;
    roin.n_downbeats           = 0;
    roin.splice_flex_beats     = 0;        // legacy closest-beat snap
    roin.region_beta           = region_beta;

    RegionOptimizer ropt(roin);
    out_path = ropt.remix(target_duration,
                          fix.region_start, fix.region_end,
                          fix.rW.data(), fix.n_region,
                          &fix.region_candidates);
    return ! out_path.beat_indices.empty();
}

// ---------------------------------------------------------------------------
// Test 1 — Default field value: region_beta = false.
//
// Bit-exact baseline preservation regression guard. Default false means
// existing 48-case test_region_optimizer parity gate stays unchanged
// (parity tests construct RegionOptimizerInputs as `{}` then populate
// caller-relevant fields; region_beta defaults to false → legacy path).
// ---------------------------------------------------------------------------
bool test_default_field_value()
{
    RegionOptimizerInputs roin{};
    if (roin.region_beta != false) {
        std::fprintf(stderr,
                     "[FAIL] RegionOptimizerInputs::region_beta default=%d, expected false\n",
                     roin.region_beta ? 1 : 0);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — Legacy path completes on extending fixture.
//
// region_beta=false on synthetic 12-beat region with target=8.0s (extending
// from 6.0s region duration). DP must produce non-empty path with finite
// total_cost. Validates baseline path computation has not regressed.
// ---------------------------------------------------------------------------
bool test_legacy_path_completes()
{
    Fixture fix = buildFixture(/*loop_cost=*/0.4);
    RemixPath p;
    if (! runRegionDP(fix, /*region_beta=*/false, /*target=*/8.0, p)) {
        std::fprintf(stderr, "[FAIL] legacy path empty (no DP solution found)\n");
        return false;
    }
    if (! std::isfinite(p.total_cost)) {
        std::fprintf(stderr, "[FAIL] legacy total_cost not finite: %.20g\n",
                     p.total_cost);
        return false;
    }
    if (p.total_cost < 0.0) {
        std::fprintf(stderr, "[FAIL] legacy total_cost negative: %.20g\n",
                     p.total_cost);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — Beta path distinct from legacy on identical fixture.
//
// region_beta=true on same 12-beat extending fixture must produce DIFFERENT
// total_cost than legacy (because cost arithmetic constants changed: cap
// 1.0→5.0, backward penalty 0.5→0.05, base 0.8→0.3). If costs are bit-
// identical, the rebalance flag is not actually being threaded through to
// regionDp (silent regression).
//
// Fixture forces DP to USE the backward-jump path: target_duration=10.0 sec
// → target_beats=20 → min_target=14. Sequential-only path stops at t=12
// (row 11 has no forward neighbors), which is BELOW min_target → DP must
// reach t ≥ 14 via the jump-then-sequential trajectory: 0..8 sequential +
// jump 8→3 + 3..11 sequential = t=18 at ri=11.
//
// On the chosen jump cell (ri=8, rj=3) with `is_extending=true`:
//
//   LEGACY: raw_w = rW[8][3] + 0.5   (backward penalty adds 0.5 to cell)
//           jump_increment = 0.8 × min(1.0, raw_w × 2.0)
//
//   BETA:   raw_w = rW[8][3] + 0.05  (reduced backward penalty)
//           jump_increment = 0.3 × min(5.0, raw_w × 2.0)
//
// For loop_cost = 0.4: raw_w_legacy = 0.9, raw_w_beta = 0.45.
// Legacy increment = 0.8 × min(1.0, 1.8) = 0.8 (saturated cap).
// Beta increment   = 0.3 × min(5.0, 0.9) = 0.27.
// Total at jump cell: legacy = 0.9 + 0.8 = 1.7; beta = 0.45 + 0.27 = 0.72.
// Beta cost is observably lower at the jump arithmetic.
// ---------------------------------------------------------------------------
bool test_beta_distinct_from_legacy()
{
    Fixture fix_legacy = buildFixture(/*loop_cost=*/0.4);
    Fixture fix_beta   = buildFixture(/*loop_cost=*/0.4);  // identical content

    RemixPath p_legacy;
    RemixPath p_beta;

    // target=10.0s forces min_target=14 > sequential-only-max t=12 → DP must
    // pick jump path to reach end-state range [14, 26].
    if (! runRegionDP(fix_legacy, /*region_beta=*/false, /*target=*/10.0, p_legacy)) {
        std::fprintf(stderr, "[FAIL] legacy path empty\n");
        return false;
    }
    if (! runRegionDP(fix_beta, /*region_beta=*/true, /*target=*/10.0, p_beta)) {
        std::fprintf(stderr, "[FAIL] beta path empty\n");
        return false;
    }

    // Beta cost arithmetic must be observably different from legacy.
    // Tight equality would mean rebalance flag is not threaded into DP.
    const double diff = std::abs(p_beta.total_cost - p_legacy.total_cost);
    if (diff < 1e-6) {
        std::fprintf(stderr,
                     "[FAIL] beta total_cost (%.10f) bit-identical to legacy (%.10f); "
                     "rebalance flag not threaded into DP\n",
                     p_beta.total_cost, p_legacy.total_cost);
        return false;
    }

    // Beta cost should be ≤ legacy when DP picks the same path (lower per-
    // jump tax). Allow small numerical tolerance for path-divergence cases.
    if (p_beta.total_cost > p_legacy.total_cost + 1e-6) {
        std::fprintf(stderr,
                     "[FAIL] beta total_cost (%.10f) exceeds legacy (%.10f); "
                     "expected ≤ given lower per-jump tax\n",
                     p_beta.total_cost, p_legacy.total_cost);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// SynthFixture — larger fixture for loop synthesizer tests (ADR-081 STATUS
// UPDATE 2). n_region=20 to satisfy cooldown=8 + i+j+spread requirements;
// target extends to force synth path. Loop point at rW[i_loop][j_loop] =
// `loop_cost`; i_loop - j_loop must be ≥ 8 (cooldown) for synth to consider.
// ---------------------------------------------------------------------------
struct SynthFixture {
    std::vector<double> rW;
    std::vector<double> beat_times;
    std::map<std::pair<int, int>, TransitionCandidate> region_candidates;
    int    n_region        = 20;
    int    n_total         = 21;  // n_region + 1 buffer beat
    double avg_beat        = 0.5;
    double region_start    = 0.0;
    double region_end      = 10.0;  // 20 beats × 0.5 sec/beat
    double tolerance_sec   = 5.0;
};

SynthFixture buildSynthFixture(int i_loop, int j_loop, double loop_cost) {
    SynthFixture f;
    const int n = f.n_region;
    f.rW.assign(static_cast<std::size_t>(n) * n, INF);
    for (int i = 0; i + 1 < n; ++i) {
        f.rW[(std::size_t) i * n + (i + 1)] = 0.0;
    }
    f.rW[(std::size_t) i_loop * n + j_loop] = loop_cost;
    f.beat_times.resize(static_cast<std::size_t>(f.n_total));
    for (int i = 0; i < f.n_total; ++i) {
        f.beat_times[(std::size_t) i] = static_cast<double>(i) * f.avg_beat;
    }
    return f;
}

bool runSynthRegionDP(const SynthFixture& fix,
                     bool region_beta,
                     double target_duration,
                     RemixPath& out_path)
{
    RegionOptimizerInputs roin{};
    roin.n_beats               = fix.n_total;
    roin.beat_times            = fix.beat_times.data();
    roin.avg_beat_duration     = fix.avg_beat;
    roin.duration_tolerance_sec = fix.tolerance_sec;
    roin.candidates            = nullptr;
    roin.sample_rate           = 22050;
    roin.downbeats             = nullptr;
    roin.n_downbeats           = 0;
    roin.splice_flex_beats     = 0;
    roin.region_beta           = region_beta;

    RegionOptimizer ropt(roin);
    out_path = ropt.remix(target_duration,
                          fix.region_start, fix.region_end,
                          fix.rW.data(), fix.n_region,
                          &fix.region_candidates);
    return ! out_path.beat_indices.empty();
}

// ---------------------------------------------------------------------------
// Test 5 — Loop synthesizer produces N≥2 transitions on extending fixture.
//
// SynthFixture with rW[18][8]=0.1 (loop length 10, ≥ cooldown=8). Target
// duration 22.0 sec → target_beats=44, n_region=20, target_extra=24.
// loop_step_len = 18 - 8 + 1 = 11. N_round = round(24/11) = 2.
// t_actual = 20 + 2×11 = 42 ∈ [min_target=34, max_target=54] (tolerance 5
// sec / avg_beat 0.5 = 10 beats). Synth picks (i=18, j=8, N=2).
//
// Expected path: [0..18] + 2 × [8..18] + [19] = 19 + 22 + 1 = 42 beats.
// Expected transitions: 2 × (18, 8) absolute (entry_beat=0, so abs ≡ rel).
// ---------------------------------------------------------------------------
bool test_synth_picks_iteration_path()
{
    SynthFixture fix = buildSynthFixture(/*i=*/18, /*j=*/8, /*loop_cost=*/0.1);
    RemixPath p;
    if (! runSynthRegionDP(fix, /*region_beta=*/true, /*target=*/22.0, p)) {
        std::fprintf(stderr, "[FAIL] synth path empty\n");
        return false;
    }
    if (p.transitions.size() < 2) {
        std::fprintf(stderr, "[FAIL] synth produced %zu transitions, expected >= 2\n",
                     p.transitions.size());
        return false;
    }
    // All transitions should be the same (i, j) jump.
    for (std::size_t k = 0; k < p.transitions.size(); ++k) {
        if (p.transitions[k].first != 18 || p.transitions[k].second != 8) {
            std::fprintf(stderr, "[FAIL] transitions[%zu]=(%d,%d), expected (18, 8)\n",
                         k, p.transitions[k].first, p.transitions[k].second);
            return false;
        }
    }
    // Path length: 1 + i + N×(i-j+1) + (n_region-1-i) = n_region + N×(i-j+1)
    // For N=2: 20 + 2×11 = 42.
    if (p.beat_indices.size() != 42) {
        std::fprintf(stderr, "[FAIL] path length %zu, expected 42\n",
                     p.beat_indices.size());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 6 — Loop synthesizer skipped on shortening (is_extending=false).
//
// SynthFixture with same loop point but target shorter than region. Synth
// is gated on `is_extending_=true` so should not fire. Path comes from
// standard regionDp.
// ---------------------------------------------------------------------------
bool test_synth_skipped_for_shortening()
{
    SynthFixture fix = buildSynthFixture(18, 8, 0.1);
    RegionOptimizerInputs roin{};
    roin.n_beats               = fix.n_total;
    roin.beat_times            = fix.beat_times.data();
    roin.avg_beat_duration     = fix.avg_beat;
    roin.duration_tolerance_sec = fix.tolerance_sec;
    roin.candidates            = nullptr;
    roin.sample_rate           = 22050;
    roin.splice_flex_beats     = 0;
    roin.region_beta           = true;

    RegionOptimizer ropt(roin);
    // target=6.0 < region_duration=10.0 → shortening → is_extending=false.
    RemixPath p = ropt.remix(/*target=*/6.0, fix.region_start, fix.region_end,
                             fix.rW.data(), fix.n_region, &fix.region_candidates);
    if (p.beat_indices.empty()) {
        std::fprintf(stderr, "[FAIL] shortening path empty\n");
        return false;
    }
    if (ropt.isExtending()) {
        std::fprintf(stderr, "[FAIL] expected is_extending=false for target<region\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 7 — Loop synthesizer falls back to DP when no viable (i, j, N).
//
// Fixture with all backward cells INF (no viable loop point). Synth must
// return empty path → caller falls through to standard regionDp. DP path
// must still produce a finite-cost result (sequential-only or fallback).
// ---------------------------------------------------------------------------
bool test_synth_falls_back_to_dp()
{
    // Fixture: only sequential edges, NO backward candidates at all.
    SynthFixture fix;
    const int n = fix.n_region;
    fix.rW.assign(static_cast<std::size_t>(n) * n, INF);
    for (int i = 0; i + 1 < n; ++i) {
        fix.rW[(std::size_t) i * n + (i + 1)] = 0.0;
    }
    fix.beat_times.resize(static_cast<std::size_t>(fix.n_total));
    for (int i = 0; i < fix.n_total; ++i) {
        fix.beat_times[(std::size_t) i] = static_cast<double>(i) * fix.avg_beat;
    }

    RemixPath p;
    if (! runSynthRegionDP(fix, /*region_beta=*/true, /*target=*/15.0, p)) {
        std::fprintf(stderr, "[FAIL] fallback path empty\n");
        return false;
    }
    if (! std::isfinite(p.total_cost)) {
        std::fprintf(stderr, "[FAIL] fallback total_cost not finite: %.20g\n",
                     p.total_cost);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — Beta path on uniform-low-quality fixture: no degenerate output.
//
// region_beta=true on fixture where the only backward-jump candidate has
// very high raw cost (loop_cost=0.95). DP must still produce a finite-cost
// path (sequential traversal + duration_penalty for falling short of target).
// Guards against NaN/INF leak under unusual cost matrix shapes.
// ---------------------------------------------------------------------------
bool test_beta_uniform_no_degenerate()
{
    Fixture fix = buildFixture(/*loop_cost=*/0.95);
    RemixPath p;
    if (! runRegionDP(fix, /*region_beta=*/true, /*target=*/10.0, p)) {
        std::fprintf(stderr, "[FAIL] beta uniform path empty\n");
        return false;
    }
    if (! std::isfinite(p.total_cost)) {
        std::fprintf(stderr, "[FAIL] beta uniform total_cost not finite: %.20g\n",
                     p.total_cost);
        return false;
    }
    if (p.total_cost < 0.0 || p.total_cost > 100.0) {
        std::fprintf(stderr,
                     "[FAIL] beta uniform total_cost out of expected range [0, 100]: %.20g\n",
                     p.total_cost);
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    std::printf("test_region_inner_loop — sesja 94 ADR-081 STATUS UPDATE 1 self-validation\n");

    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"default_field_value",          test_default_field_value},
        {"legacy_path_completes",        test_legacy_path_completes},
        {"beta_distinct_from_legacy",    test_beta_distinct_from_legacy},
        {"beta_uniform_no_degenerate",   test_beta_uniform_no_degenerate},
        {"synth_picks_iteration_path",   test_synth_picks_iteration_path},
        {"synth_skipped_for_shortening", test_synth_skipped_for_shortening},
        {"synth_falls_back_to_dp",       test_synth_falls_back_to_dp},
    };

    int pass = 0, fail = 0;
    for (const auto& c : cases) {
        const bool ok = c.fn();
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (ok) ++pass; else ++fail;
    }

    std::printf("\nResults: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
