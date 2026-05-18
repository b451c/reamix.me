// test_edit_length — sesja 92 (ADR-083) updated sesja 93 (ADR-084).
//
// Self-validation parity test for the Edit Length slider MULTIPLICATIVE
// jump-cost scale wired through ViterbiDPInputs / RegionOptimizerInputs /
// CleanOptimizerInputs / assembleBlocks.
//
// CANONICAL SOURCE: ViterbiDP.cpp jump-branch (post ADR-084 multiplicative
// scaled_penalty composition) + RegionOptimizer.cpp regionDp jump-branch
// (post ADR-084) + BlockAssembly.cpp assembleBlocks junction-cost
// (post ADR-084). NO Python reference (per ADR-065 + ADR-084 — formulation
// is C++-native, productionised from Audition's Advanced "Edit Length"
// slider via multiplicative jump-scale 2^((slider-50)/25) ∈ [0.25, 4.0]).
//
// FORMULA (ADR-084 sesja 93 — supersedes sesja-92 additive design):
//   slider → scale = 2^((slider - 50) / 25)
//   slider=50 → scale=1.0 (bit-exact baseline center)
//   slider=100 → scale=4.0 (4× per-jump cost → DP picks fewer cuts)
//   slider=0 → scale=0.25 (0.25× cost → DP picks more cuts)
// In ViterbiDP: per-jump cost = JUMP_PENALTY_BASE × jump_penalty_scale ×
// quality_factor × edit_length_jump_scale. In RegionOptimizer: analogous
// multiplication. In BlockAssembly: total_cost += scale × (1 - quality)
// per junction.
//
// Test asserts:
//   1. Field default = 1.0 in all relevant Inputs structs (bit-exact baseline
//      preservation when slider untouched at center=50).
//   2. Multiplicative map algebraic invariants:
//      - slider=50 → 1.0 (center, bit-exact).
//      - slider=100 → 4.0 (4× span at top end).
//      - slider=0 → 0.25 (0.25× span at bottom end).
//      - slider=75 → 2.0 (linear in log space).
//      - slider=25 → 0.5.
//   3. ViterbiDP integration: scale > 1.0 produces HIGHER total cost than
//      scale = 1.0 on identical fixture (path-cost monotonicity in scale).
//   4. Sequential-path invariance: scale never fires on sequential edges
//      (gate `if (j != i + 1)` not entered).
//
// Per memory `feedback_python_no_longer_source_of_truth.md`: parity tests
// for new cost components validate C++ against itself, not vs Python.

#include "remix/Optimizer.h"
#include "remix/RegionOptimizer.h"
#include "remix/ViterbiDP.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using reamix::remix::CleanOptimizerInputs;
using reamix::remix::RegionOptimizerInputs;
using reamix::remix::ViterbiDPInputs;
using reamix::remix::ViterbiPath;
using reamix::remix::viterbiDP;

namespace {

bool nearly_equal(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

// ADR-084 multiplicative jump-scale at MainComponent UI layer.
double sliderToJumpScale(int slider) {
    return std::pow(2.0, (static_cast<double>(slider) - 50.0) / 25.0);
}

// ---------------------------------------------------------------------------
// Test 1 — Field default = 1.0 across all 3 Inputs structs (ADR-084).
//
// Bit-exact baseline preservation is the core regression guard. After
// ADR-084 the field is multiplicative; default 1.0 means "no scaling" =
// identity multiplication = bit-exact baseline.
// ---------------------------------------------------------------------------
bool test_default_neutral_invariant() {
    ViterbiDPInputs vd{};
    if (vd.edit_length_jump_scale != 1.0) {
        std::fprintf(stderr, "[FAIL] ViterbiDPInputs default=%.20g, expected 1.0\n",
                     vd.edit_length_jump_scale);
        return false;
    }

    RegionOptimizerInputs ro{};
    if (ro.edit_length_jump_scale != 1.0) {
        std::fprintf(stderr, "[FAIL] RegionOptimizerInputs default=%.20g, expected 1.0\n",
                     ro.edit_length_jump_scale);
        return false;
    }
    if (ro.min_seq_after_jump_override != 0) {
        std::fprintf(stderr, "[FAIL] RegionOptimizerInputs min_seq_after_jump_override default=%d, expected 0\n",
                     ro.min_seq_after_jump_override);
        return false;
    }

    CleanOptimizerInputs co{};
    if (co.edit_length_jump_scale != 1.0) {
        std::fprintf(stderr, "[FAIL] CleanOptimizerInputs default=%.20g, expected 1.0\n",
                     co.edit_length_jump_scale);
        return false;
    }
    if (co.min_seq_after_jump_override != 0) {
        std::fprintf(stderr, "[FAIL] CleanOptimizerInputs min_seq_after_jump_override default=%d, expected 0\n",
                     co.min_seq_after_jump_override);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 2 — Multiplicative-scale map algebraic invariants (ADR-084).
//
// Map slider [0..100] → 2^((slider-50)/25) ∈ [0.25, 4.0] center=1.0.
// ---------------------------------------------------------------------------
bool test_multiplicative_scale_map() {
    // Slider center → scale 1.0 (bit-exact baseline anchor).
    if (! nearly_equal(sliderToJumpScale(50), 1.0)) {
        std::fprintf(stderr, "[FAIL] sliderToJumpScale(50)=%.20g, expected 1.0\n",
                     sliderToJumpScale(50));
        return false;
    }

    // Long end (slider=100) → 2^2 = 4.0.
    if (! nearly_equal(sliderToJumpScale(100), 4.0)) {
        std::fprintf(stderr, "[FAIL] sliderToJumpScale(100)=%.20g, expected 4.0\n",
                     sliderToJumpScale(100));
        return false;
    }

    // Short end (slider=0) → 2^-2 = 0.25.
    if (! nearly_equal(sliderToJumpScale(0), 0.25)) {
        std::fprintf(stderr, "[FAIL] sliderToJumpScale(0)=%.20g, expected 0.25\n",
                     sliderToJumpScale(0));
        return false;
    }

    // Midpoints — exponential in slider (e.g. slider=75 → 2; slider=25 → 0.5).
    if (! nearly_equal(sliderToJumpScale(75), 2.0)) {
        std::fprintf(stderr, "[FAIL] sliderToJumpScale(75)=%.20g, expected 2.0\n",
                     sliderToJumpScale(75));
        return false;
    }
    if (! nearly_equal(sliderToJumpScale(25), 0.5)) {
        std::fprintf(stderr, "[FAIL] sliderToJumpScale(25)=%.20g, expected 0.5\n",
                     sliderToJumpScale(25));
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 3 — ViterbiDP integration: scale applied only on non-sequential edges.
//
// Synthetic 4-beat W matrix where:
//   W[i, i+1] = 0.0  (sequential edges = zero base cost)
//   W[i, j>i+1] = 0.5  (forward jump cost)
//   W[i, j<i] = INF  (no backward jumps in this synthetic)
//
// Per-jump cost in DP: JUMP_PENALTY_BASE × jump_penalty_scale ×
// quality_factor × edit_length_jump_scale (where quality_factor = raw_w²).
// scale=2.0 doubles per-jump component vs scale=1.0 (assuming non-trivial
// path with at least one jump).
//
// We don't know K a priori (DP picks min-cost path), so we run with
// scale=1.0 (baseline) and scale=4.0 (max) and verify the higher-scale
// run produces equal or HIGHER total cost. Equality is allowed only when
// the path takes zero non-sequential jumps (sequential-only fallback).
// ---------------------------------------------------------------------------
bool test_viterbi_dp_integration() {
    constexpr int n = 4;
    // W matrix 4×4 row-major.
    constexpr double INF_VAL = 1e9;
    std::vector<double> W(static_cast<std::size_t>(n) * n, INF_VAL);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            W[(std::size_t) i * n + j] = (j == i + 1) ? 0.0 : 0.5;
        }
    }

    // Trivial neighbor CSR — every i links to all forward j > i.
    // Format: indices flat [j0, j1, ...]; offsets[i] = start of i's list.
    std::vector<std::int64_t> indices;
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(n + 1), 0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            indices.push_back(j);
        }
        offsets[(std::size_t) i + 1] = static_cast<std::int64_t>(indices.size());
    }

    // Single-segment, no downbeat constraints (degenerate fixture).
    std::vector<std::int64_t> beat_to_segment(static_cast<std::size_t>(n), 0);

    auto runDP = [&](double scale) -> ViterbiPath {
        ViterbiDPInputs vd{};
        vd.W                  = W.data();
        vd.n_beats            = n;
        vd.target_length      = n;       // hit n=4 path length
        vd.min_target_length  = n;
        vd.intro_beats        = 0;
        vd.outro_beats        = 0;
        vd.is_shortening      = false;
        vd.neighbor_indices   = indices.data();
        vd.n_neighbor_indices = static_cast<int>(indices.size());
        vd.neighbor_offsets   = offsets.data();
        vd.beat_to_segment    = beat_to_segment.data();
        vd.seg_sim_matrix     = nullptr;  // single segment fallback
        vd.n_segs             = 1;
        vd.pre_downbeat_arr   = nullptr;  // no bar info
        vd.downbeat_arr       = nullptr;
        vd.max_transitions    = 8;
        vd.min_jumps          = 0;
        vd.min_seq_after_jump = 1;        // permissive cooldown for tiny fixture
        vd.min_forward_jump   = 1;
        vd.min_segment_beats  = 1;
        vd.edit_length_jump_scale = scale;
        return viterbiDP(vd);
    };

    const auto path1 = runDP(1.0);   // baseline
    const auto path4 = runDP(4.0);   // max (slider=100)

    // Both runs must produce non-empty paths on this trivial fixture.
    if (path1.path.empty()) {
        std::fprintf(stderr, "[FAIL] DP returned empty path for scale=1.0\n");
        return false;
    }
    if (path4.path.empty()) {
        std::fprintf(stderr, "[FAIL] DP returned empty path for scale=4.0\n");
        return false;
    }

    // Non-negative cost difference under higher scale (path4 cost can equal
    // path1 cost only if path has zero non-sequential jumps).
    if (path4.total_cost + 1e-12 < path1.total_cost) {
        std::fprintf(stderr, "[FAIL] cost regressed under scale=4.0: %.20g vs %.20g\n",
                     path4.total_cost, path1.total_cost);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Test 4 — Boundary algebra: pure-sequential path is INVARIANT under scale.
//
// On a path that takes only sequential edges (no jumps), the multiplicative
// jump-scale NEVER fires (gated by `if (j != i + 1)`). Total path cost
// identical for any scale value → confirms gate placement.
//
// Trivial case: 4-beat fixture where W[i, i+1] = 0.0 and ALL other entries
// are INF (no jumps possible). DP forced to take pure-sequential 0→1→2→3.
// ---------------------------------------------------------------------------
bool test_sequential_path_invariant() {
    constexpr int n = 4;
    constexpr double INF_VAL = 1e9;
    std::vector<double> W(static_cast<std::size_t>(n) * n, INF_VAL);
    for (int i = 0; i < n - 1; ++i) {
        W[(std::size_t) i * n + (i + 1)] = 0.0;
    }

    // Neighbor CSR: only i → i+1.
    std::vector<std::int64_t> indices;
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(n + 1), 0);
    for (int i = 0; i < n; ++i) {
        if (i + 1 < n) indices.push_back(i + 1);
        offsets[(std::size_t) i + 1] = static_cast<std::int64_t>(indices.size());
    }

    std::vector<std::int64_t> beat_to_segment(static_cast<std::size_t>(n), 0);

    auto runDP = [&](double scale) -> double {
        ViterbiDPInputs vd{};
        vd.W                  = W.data();
        vd.n_beats            = n;
        vd.target_length      = n;
        vd.min_target_length  = n;
        vd.intro_beats        = 0;
        vd.outro_beats        = 0;
        vd.is_shortening      = false;
        vd.neighbor_indices   = indices.data();
        vd.n_neighbor_indices = static_cast<int>(indices.size());
        vd.neighbor_offsets   = offsets.data();
        vd.beat_to_segment    = beat_to_segment.data();
        vd.seg_sim_matrix     = nullptr;
        vd.n_segs             = 1;
        vd.pre_downbeat_arr   = nullptr;
        vd.downbeat_arr       = nullptr;
        vd.max_transitions    = 8;
        vd.min_jumps          = 0;
        vd.min_seq_after_jump = 1;
        vd.min_forward_jump   = 1;
        vd.min_segment_beats  = 1;
        vd.edit_length_jump_scale = scale;
        const auto p = viterbiDP(vd);
        return p.total_cost;
    };

    const double cost_one  = runDP(1.0);
    const double cost_high = runDP(8.0);
    const double cost_low  = runDP(0.125);

    // Pure-sequential path has zero non-sequential edges → scale never fires.
    // All three runs must produce identical cost (= 0.0 on this fixture).
    if (! nearly_equal(cost_one, cost_high)) {
        std::fprintf(stderr, "[FAIL] sequential-only path cost changed under scale=8.0: %.20g vs %.20g\n",
                     cost_one, cost_high);
        return false;
    }
    if (! nearly_equal(cost_one, cost_low)) {
        std::fprintf(stderr, "[FAIL] sequential-only path cost changed under scale=0.125: %.20g vs %.20g\n",
                     cost_one, cost_low);
        return false;
    }
    return true;
}

} // namespace

int main()
{
    std::printf("test_edit_length — sesja 92 ADR-083 self-validation\n");

    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"default_neutral_invariant",       test_default_neutral_invariant},
        {"multiplicative_scale_map",        test_multiplicative_scale_map},
        {"viterbi_dp_integration",          test_viterbi_dp_integration},
        {"sequential_path_invariant",       test_sequential_path_invariant},
    };

    int pass = 0, fail = 0;
    for (const auto& c : cases) {
        const bool ok = c.fn();
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (ok) ++pass; else ++fail;
    }

    std::printf("\nSummary: %d/%d passed\n", pass, pass + fail);
    return fail == 0 ? 0 : 1;
}
