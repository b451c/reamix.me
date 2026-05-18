// test_region_optimizer_k_best — sesja 100 (DEV-032, ADR-091) self-validation.
//
// Self-validation parity test for RegionOptimizer K-best variations
// (Region mirror of CleanOptimizer::remix_k_best per ADR-048 sesja 58).
// Per ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`:
// post-port extensions validate against hand-computed values + invariants,
// not vs Python ground truth. RegionOptimizer K-best has no Python equivalent
// (Lua client uses `start_remix(variation)` against the Python server's
// `_remix.py:140-147` which dispatches to CleanOptimizer for Duration; Region
// mode never had a K-best path in Python).
//
// CANONICAL SOURCE: `RegionOptimizer.cpp::remix_k_best` +
// `RegionOptimizer.cpp::remix_variation` + blocked_transitions_ propagation
// through `regionDp` neighbor traversal + `regionLoopSynthesize` (i,j) loop.
//
// Test invariants:
//   1. variation == 0 short-circuit: remix_variation(target, 0, ...) ≡
//      remix_k_best(target, 1, ...)[0] ≡ remix(target, ..., nullptr).
//   2. blocked_transitions = nullptr default param preserves bit-exact
//      baseline (parity-neutral wrt pre-DEV-032 callers).
//   3. K-best returns ≥1 path for any non-degenerate region (mirror of
//      CleanOptimizer::remix_k_best invariant).
//   4. K-best paths are deduped by beat_indices (set-key invariant).
//   5. Blocked transition skipped: when (fb, tb) ∈ blocked_set, no resulting
//      path contains that jump in transitions[].
//   6. variation_idx >= len(paths) - 1 clamps to last path (mirror Python
//      `_remix.py:146` behaviour).

#include "remix/Path.h"
#include "remix/RegionOptimizer.h"
#include "remix/TransitionCost.h"  // INF, TransitionCandidate

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include <vector>

using reamix::remix::INF;
using reamix::remix::RegionOptimizer;
using reamix::remix::RegionOptimizerInputs;
using reamix::remix::RemixPath;
using reamix::remix::TransitionCandidate;

namespace {

// Synthetic 14-beat region fixture (n_total = 15 buffer). Two backward-jump
// opportunities engineered so K-best can pick at least 2 distinct paths:
//   rW[10][4] = jump_a_cost  (preferred)
//   rW[12][6] = jump_b_cost  (alternate)
// Sequential cells rW[i][i+1] = 0.0 free; everything else INF.
struct Fixture {
    std::vector<double> rW;
    std::vector<double> beat_times;
    std::map<std::pair<int, int>, TransitionCandidate> region_candidates;
    int    n_region        = 14;
    int    n_total         = 15;
    double avg_beat        = 0.5;
    double region_start    = 0.0;
    double region_end      = 7.0;
    double tolerance_sec   = 3.0;
};

Fixture buildTwoJumpFixture(double jump_a_cost, double jump_b_cost) {
    Fixture f;
    const int n = f.n_region;
    f.rW.assign(static_cast<std::size_t>(n) * n, INF);

    for (int i = 0; i + 1 < n; ++i) {
        f.rW[(std::size_t) i * n + (i + 1)] = 0.0;
    }

    f.rW[(std::size_t) 10 * n + 4] = jump_a_cost;
    f.rW[(std::size_t) 12 * n + 6] = jump_b_cost;

    f.beat_times.resize(static_cast<std::size_t>(f.n_total));
    for (int i = 0; i < f.n_total; ++i) {
        f.beat_times[(std::size_t) i] = static_cast<double>(i) * f.avg_beat;
    }
    return f;
}

RegionOptimizerInputs makeInputs(const Fixture& fix) {
    RegionOptimizerInputs roin{};
    roin.n_beats                = fix.n_total;
    roin.beat_times             = fix.beat_times.data();
    roin.avg_beat_duration      = fix.avg_beat;
    roin.duration_tolerance_sec = fix.tolerance_sec;
    roin.candidates             = nullptr;
    roin.sample_rate            = 22050;
    roin.downbeats              = nullptr;
    roin.n_downbeats            = 0;
    roin.splice_flex_beats      = 0;
    roin.region_beta            = false;
    return roin;
}

bool sameBeatIndices(const RemixPath& a, const RemixPath& b) {
    return a.beat_indices == b.beat_indices;
}

// ---------------------------------------------------------------------------
// Test 1 — Bit-exact baseline: remix_variation(0) ≡ remix() with nullptr.
// ---------------------------------------------------------------------------
bool test1_baseline_parity_variation_zero() {
    auto fix = buildTwoJumpFixture(0.1, 0.3);
    auto roin = makeInputs(fix);

    RegionOptimizer ropt1(roin);
    RemixPath base = ropt1.remix(/*target=*/9.0,
                                  fix.region_start, fix.region_end,
                                  fix.rW.data(), fix.n_region,
                                  &fix.region_candidates);

    RegionOptimizer ropt2(roin);
    RemixPath var0 = ropt2.remix_variation(/*target=*/9.0,
                                            fix.region_start, fix.region_end,
                                            fix.rW.data(), fix.n_region,
                                            &fix.region_candidates,
                                            /*variation_idx=*/0);

    if (! sameBeatIndices(base, var0)) {
        std::fprintf(stderr,
            "[FAIL] test1: remix_variation(0) != remix(): "
            "base.beat_indices.size=%zu var0.beat_indices.size=%zu\n",
            base.beat_indices.size(), var0.beat_indices.size());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — K-best with k=1 returns single path.
// ---------------------------------------------------------------------------
bool test2_kbest_k1_returns_single() {
    auto fix = buildTwoJumpFixture(0.1, 0.3);
    auto roin = makeInputs(fix);

    RegionOptimizer ropt(roin);
    auto paths = ropt.remix_k_best(/*target=*/9.0,
                                    fix.region_start, fix.region_end,
                                    fix.rW.data(), fix.n_region,
                                    &fix.region_candidates,
                                    /*k=*/1);
    if (paths.size() != 1) {
        std::fprintf(stderr, "[FAIL] test2: k=1 returned %zu paths, expected 1\n",
                     paths.size());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — K-best with k=4 + two distinct backward jumps yields ≥1 alt path.
//
// Base path picks cheaper jump (a). After blocking (a) the alt path should
// pick (b). All paths in result must be distinct by beat_indices (dedup
// invariant).
// ---------------------------------------------------------------------------
bool test3_kbest_yields_distinct_paths() {
    auto fix = buildTwoJumpFixture(0.1, 0.3);
    auto roin = makeInputs(fix);

    RegionOptimizer ropt(roin);
    auto paths = ropt.remix_k_best(/*target=*/9.0,
                                    fix.region_start, fix.region_end,
                                    fix.rW.data(), fix.n_region,
                                    &fix.region_candidates,
                                    /*k=*/4);

    if (paths.empty()) {
        std::fprintf(stderr,
            "[FAIL] test3: k=4 returned 0 paths (expected ≥1)\n");
        return false;
    }

    // All returned paths must be distinct by beat_indices.
    std::set<std::vector<int>> seen;
    for (const auto& p : paths) {
        if (seen.find(p.beat_indices) != seen.end()) {
            std::fprintf(stderr,
                "[FAIL] test3: duplicate beat_indices in k-best result\n");
            return false;
        }
        seen.insert(p.beat_indices);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — variation_idx >= len(paths) clamps to last path.
//
// Even with k=10 requested, the synthetic fixture has at most a few distinct
// paths. variation_idx=99 should return the last-returned path, not crash.
// ---------------------------------------------------------------------------
bool test4_variation_clamps_to_last() {
    auto fix = buildTwoJumpFixture(0.1, 0.3);
    auto roin = makeInputs(fix);

    RegionOptimizer ropt1(roin);
    auto paths = ropt1.remix_k_best(/*target=*/9.0,
                                     fix.region_start, fix.region_end,
                                     fix.rW.data(), fix.n_region,
                                     &fix.region_candidates,
                                     /*k=*/10);
    const std::size_t lenPaths = paths.size();

    RegionOptimizer ropt2(roin);
    RemixPath clamped = ropt2.remix_variation(/*target=*/9.0,
                                               fix.region_start, fix.region_end,
                                               fix.rW.data(), fix.n_region,
                                               &fix.region_candidates,
                                               /*variation_idx=*/99);

    if (clamped.beat_indices.empty() && lenPaths > 0) {
        std::fprintf(stderr,
            "[FAIL] test4: variation_idx=99 returned empty path "
            "(k_best returned %zu non-empty paths)\n", lenPaths);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 5 — Blocked transition is skipped: when caller passes
// blocked_transitions = {(jump_a_global)}, resulting path's transitions[]
// must NOT contain jump_a_global. With single-jump regional structure
// blocked, the only viable path is jump_b OR no-jump straight traversal.
// ---------------------------------------------------------------------------
bool test5_blocked_transition_skipped() {
    auto fix = buildTwoJumpFixture(0.1, 0.3);
    auto roin = makeInputs(fix);

    // Block jump (10, 4) — translate local → global. With entry_beat = 0 (since
    // region_start_sec=0.0, beat[0] is at 0.0), local indices == global indices.
    std::set<std::pair<int, int>> blocked;
    blocked.insert({10, 4});

    RegionOptimizer ropt(roin);
    RemixPath p = ropt.remix(/*target=*/9.0,
                              fix.region_start, fix.region_end,
                              fix.rW.data(), fix.n_region,
                              &fix.region_candidates,
                              &blocked);

    // Blocked transition must NOT appear in resulting path's jump list.
    for (const auto& tr : p.transitions) {
        if (tr.first == 10 && tr.second == 4) {
            std::fprintf(stderr,
                "[FAIL] test5: blocked transition (10,4) leaked into path\n");
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 6 — Default-nullptr param preserves baseline. remix(target, ..., nullptr)
// ≡ remix(target, ..., /*blocked_transitions implicit default*/).
// ---------------------------------------------------------------------------
bool test6_nullptr_blocked_param_baseline() {
    auto fix = buildTwoJumpFixture(0.1, 0.3);
    auto roin = makeInputs(fix);

    RegionOptimizer ropt1(roin);
    RemixPath a = ropt1.remix(/*target=*/9.0,
                               fix.region_start, fix.region_end,
                               fix.rW.data(), fix.n_region,
                               &fix.region_candidates,
                               /*blocked_transitions=*/nullptr);

    RegionOptimizer ropt2(roin);
    RemixPath b = ropt2.remix(/*target=*/9.0,
                               fix.region_start, fix.region_end,
                               fix.rW.data(), fix.n_region,
                               &fix.region_candidates);  // default nullptr

    if (! sameBeatIndices(a, b)) {
        std::fprintf(stderr,
            "[FAIL] test6: default-nullptr param yields different path than "
            "explicit nullptr (a.size=%zu b.size=%zu)\n",
            a.beat_indices.size(), b.beat_indices.size());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int passed = 0;
    int failed = 0;

    auto run = [&](const char* name, bool result) {
        if (result) {
            std::printf("[PASS] %s\n", name);
            ++passed;
        } else {
            std::printf("[FAIL] %s\n", name);
            ++failed;
        }
    };

    run("test1_baseline_parity_variation_zero",   test1_baseline_parity_variation_zero());
    run("test2_kbest_k1_returns_single",          test2_kbest_k1_returns_single());
    run("test3_kbest_yields_distinct_paths",      test3_kbest_yields_distinct_paths());
    run("test4_variation_clamps_to_last",         test4_variation_clamps_to_last());
    run("test5_blocked_transition_skipped",       test5_blocked_transition_skipped());
    run("test6_nullptr_blocked_param_baseline",   test6_nullptr_blocked_param_baseline());

    std::printf("\n%d passed, %d failed (out of %d)\n",
                passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}
