// test_edit_density — sesja 124 (ADR-115 P3 / DEV-112) self-validation.
//
// Validates the Edit density "More cuts" mechanism for Duration:
// `ViterbiDPInputs::no_backward_jumps` (forward-only shortening) and
// `viterbiDPWithJumpFloor` (smallest per-jump bonus whose optimal path
// carries the requested number of cuts; best effort otherwise), plus the
// detent table in `ui/EditDensity.h`. No Python reference (C++-canonical
// per ADR-065; the maximum-run gate and the DP's `min_jumps` filter were
// measured and rejected in sesja 124).
//
// Fixture: 64 beats in 4/4 (regular downbeat grid), sequential edges free,
// every jump edge raw cost 0.3, target 48 beats (shortening ratio 0.75),
// intro lock 4, outro 4, cooldown 3, no transition cap.
//
// Asserts:
//   1. Defaults: no_backward_jumps = false, min_jumps_floor = 0,
//      no_backward_when_shortening = false (bit-exact baseline).
//   2. Neutral: no_backward_jumps = true changes nothing when the legacy
//      path has no backward jump (this fixture: one forward skip).
//   3. Floor 4 yields >= 4 jumps, all forward under no_backward_jumps,
//      every jump bar-aligned, the target length unchanged (the reported
//      total_cost carries the bonus, so it is not compared).
//   4. Infeasible floor (40 cuts in 48 beats): best effort, never empty.
//   4b. Floors monotone: jumps(0) <= jumps(2) <= jumps(4), each floor met.
//   5. Detent table: densityMinCuts 16/8/4 -> 0, 2 -> 2, 1 -> 4; the
//      Duration labels are distinct and the default reads "Default".

#include "remix/Optimizer.h"
#include "remix/ViterbiDP.h"
#include "ui/EditDensity.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using reamix::remix::CleanOptimizerInputs;
using reamix::remix::ViterbiDPInputs;
using reamix::remix::ViterbiPath;
using reamix::remix::viterbiDP;
using reamix::remix::viterbiDPWithJumpFloor;

namespace {

constexpr double INF_VAL = 1e9;
constexpr int    kN      = 64;
constexpr int    kTS     = 4;
constexpr int    kTarget = 48;
constexpr int    kIntro  = 4;
constexpr int    kOutro  = 4;

struct Fixture
{
    std::vector<double>       W;
    std::vector<std::int64_t> indices;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> beat_to_segment;
    std::vector<std::int8_t>  pre_downbeat;
    std::vector<std::int8_t>  downbeat;

    Fixture()
        : W(static_cast<std::size_t>(kN) * kN, INF_VAL),
          offsets(static_cast<std::size_t>(kN + 1), 0),
          beat_to_segment(static_cast<std::size_t>(kN), 0),
          pre_downbeat(static_cast<std::size_t>(kN), 0),
          downbeat(static_cast<std::size_t>(kN), 0)
    {
        for (int i = 0; i < kN; ++i) {
            downbeat    [(std::size_t) i] = (i % kTS == 0)       ? 1 : 0;
            pre_downbeat[(std::size_t) i] = ((i + 1) % kTS == 0) ? 1 : 0;
            for (int j = 0; j < kN; ++j) {
                if (j == i) continue;
                W[(std::size_t) i * kN + j] = (j == i + 1) ? 0.0 : 0.3;
                indices.push_back(j);
            }
            offsets[(std::size_t) i + 1] = static_cast<std::int64_t>(indices.size());
        }
    }

    ViterbiPath run(int floor, bool no_backward) const
    {
        ViterbiDPInputs vd{};
        vd.W                  = W.data();
        vd.n_beats            = kN;
        vd.target_length      = kTarget;
        vd.min_target_length  = kTarget;
        vd.intro_beats        = kIntro;
        vd.outro_beats        = kOutro;
        vd.is_shortening      = true;
        vd.neighbor_indices   = indices.data();
        vd.n_neighbor_indices = static_cast<int>(indices.size());
        vd.neighbor_offsets   = offsets.data();
        vd.beat_to_segment    = beat_to_segment.data();
        vd.seg_sim_matrix     = nullptr;
        vd.n_segs             = 1;
        vd.pre_downbeat_arr   = pre_downbeat.data();
        vd.downbeat_arr       = downbeat.data();
        vd.max_transitions    = kN;
        vd.min_seq_after_jump = kTS - 1;
        vd.min_forward_jump   = kTS;
        vd.min_segment_beats  = kTS;
        vd.no_backward_jumps  = no_backward;
        return viterbiDPWithJumpFloor(vd, floor);
    }
};

int countJumps(const std::vector<std::int64_t>& p, int* backward = nullptr)
{
    int n = 0, b = 0;
    for (std::size_t k = 1; k < p.size(); ++k) {
        if (p[k] != p[k - 1] + 1) { ++n; if (p[k] < p[k - 1]) ++b; }
    }
    if (backward) *backward = b;
    return n;
}

bool test_defaults()
{
    ViterbiDPInputs vd{};
    CleanOptimizerInputs oin{};
    if (vd.no_backward_jumps || oin.min_jumps_floor != 0 || oin.no_backward_when_shortening) {
        std::fprintf(stderr, "[FAIL] defaults are not the bit-exact baseline\n");
        return false;
    }
    std::fprintf(stderr, "[PASS] defaults off (bit-exact baseline)\n");
    return true;
}

bool test_neutral(const Fixture& f)
{
    const ViterbiPath a = f.run(0, false);
    const ViterbiPath b = f.run(0, true);
    int back = 0;
    countJumps(a.path, &back);
    if (a.path.empty() || back != 0 || a.path != b.path || a.total_cost != b.total_cost) {
        std::fprintf(stderr, "[FAIL] forward-only changed a path without backward jumps (%zu / %zu beats, %d backward)\n",
                     a.path.size(), b.path.size(), back);
        return false;
    }
    std::fprintf(stderr, "[PASS] forward-only neutral on the legacy path (%d jumps)\n", countJumps(a.path));
    return true;
}

bool test_floor(const Fixture& f)
{
    const ViterbiPath off = f.run(0, false);
    const ViterbiPath r4  = f.run(4, true);
    int back = 0;
    const int j4 = countJumps(r4.path, &back);
    if (r4.path.empty() || j4 < 4 || back != 0 || r4.path.size() != off.path.size()) {
        std::fprintf(stderr, "[FAIL] floor 4: %d jumps (%d backward), %zu beats vs legacy %zu\n",
                     j4, back, r4.path.size(), off.path.size());
        return false;
    }
    for (std::size_t k = 1; k < r4.path.size(); ++k) {
        const std::int64_t a = r4.path[k - 1], b = r4.path[k];
        if (b == a + 1) continue;
        if (f.pre_downbeat[(std::size_t) a] != 1 || f.downbeat[(std::size_t) b] != 1) {
            std::fprintf(stderr, "[FAIL] floor 4: jump %lld -> %lld off the bar grid\n",
                         (long long) a, (long long) b);
            return false;
        }
    }
    std::fprintf(stderr, "[PASS] floor 4: %d forward, bar-aligned jumps (legacy %d), same target length\n",
                 j4, countJumps(off.path));
    return true;
}

bool test_infeasible_floor(const Fixture& f)
{
    // 40 cuts cannot fit 48 beats at cooldown 3: best effort = a path with
    // as many cuts as the largest bonus buys, never empty.
    const ViterbiPath r = f.run(40, true);
    const int j = countJumps(r.path);
    if (r.path.empty() || j >= 40 || j < 4) {
        std::fprintf(stderr, "[FAIL] infeasible floor: %zu beats, %d jumps\n", r.path.size(), j);
        return false;
    }
    std::fprintf(stderr, "[PASS] infeasible floor -> best effort (%d jumps, non-empty)\n", j);
    return true;
}

bool test_monotone(const Fixture& f)
{
    const int j2 = countJumps(f.run(2, true).path);
    const int j4 = countJumps(f.run(4, true).path);
    const int j0 = countJumps(f.run(0, true).path);
    if (!(j0 <= j2 && j2 <= j4) || j2 < 2 || j4 < 4) {
        std::fprintf(stderr, "[FAIL] floors not monotone: 0 -> %d, 2 -> %d, 4 -> %d\n", j0, j2, j4);
        return false;
    }
    std::fprintf(stderr, "[PASS] floors monotone: 0 -> %d, 2 -> %d, 4 -> %d jumps\n", j0, j2, j4);
    return true;
}

bool test_detent_table()
{
    using reamix::ui::densityMinCuts;
    using reamix::ui::durationDensityLabel;
    const bool cuts_ok = densityMinCuts(16) == 0 && densityMinCuts(8) == 0 && densityMinCuts(4) == 0
                      && densityMinCuts(2) == 2 && densityMinCuts(1) == 4;
    const int bars[5] = { 16, 8, 4, 2, 1 };
    bool labels_ok = std::strcmp(durationDensityLabel(4), "Default") == 0;
    for (int a = 0; a < 5 && labels_ok; ++a)
        for (int b = a + 1; b < 5; ++b)
            if (std::strcmp(durationDensityLabel(bars[a]), durationDensityLabel(bars[b])) == 0) labels_ok = false;
    if (!cuts_ok || !labels_ok) {
        std::fprintf(stderr, "[FAIL] detent table (cuts %d, labels %d)\n", cuts_ok, labels_ok);
        return false;
    }
    std::fprintf(stderr, "[PASS] detent table: 2 bars -> 2 cuts, 1 bar -> 4 cuts, labels distinct\n");
    return true;
}

} // namespace

int main()
{
    const Fixture f;
    bool ok = true;
    ok = test_defaults()          && ok;
    ok = test_neutral(f)          && ok;
    ok = test_floor(f)            && ok;
    ok = test_infeasible_floor(f) && ok;
    ok = test_monotone(f)         && ok;
    ok = test_detent_table()      && ok;
    std::fprintf(stderr, ok ? "test_edit_density: ALL PASS\n" : "test_edit_density: FAIL\n");
    return ok ? 0 : 1;
}
