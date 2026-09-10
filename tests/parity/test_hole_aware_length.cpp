// test_hole_aware_length — sesja 126 (DEV-114) self-validation.
//
// The Duration DP counts length in beats; a beat that spans a beat-tracker
// hole counted as ONE beat, so the remix overshot the target (Drake x0.5
// +30 s) and the path ranked by a wrong length. `ViterbiDPInputs::
// beat_weights` gives every beat its weight in period slots and the DP
// advances t by that weight. No Python reference (C++-canonical, ADR-065).
//
// Fixture: the sesja-124 density fixture (64 beats in 4/4, sequential edges
// free, every jump edge raw cost 0.3, target 48 = min target, intro 4,
// outro 4, cooldown 3) with beat 20 spanning a 7-period hole.
//
// Asserts:
//   1. Null weights == explicit all-ones weights (path + cost bit-exact).
//   2. With the hole weight the path's weighted length == the target; the
//      unweighted path counts 48 beats regardless of the hole, so whenever
//      it crosses beat 20 its real length is 54 slots.
//   3. holeAwareBeatWeights: uniform grid -> all ones + period; a 6.0-period
//      gap -> 6, 1.3 -> 1, 1.6 -> 2; the last beat always 1.

#include "remix/ViterbiDP.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using reamix::remix::ViterbiDPInputs;
using reamix::remix::ViterbiPath;
using reamix::remix::viterbiDP;
using reamix::remix::holeAwareBeatWeights;

namespace {

constexpr double INF_VAL = 1e9;
constexpr int    kN      = 64;
constexpr int    kTS     = 4;
constexpr int    kTarget = 48;
constexpr int    kHole   = 20;   // beat index spanning the hole
constexpr int    kHoleW  = 7;    // period slots it occupies

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

    ViterbiPath run(const std::vector<int>* weights) const
    {
        ViterbiDPInputs vd{};
        vd.W                  = W.data();
        vd.n_beats            = kN;
        vd.target_length      = kTarget;
        vd.min_target_length  = kTarget;
        vd.intro_beats        = 4;
        vd.outro_beats        = 4;
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
        if (weights != nullptr) {
            vd.beat_weights = weights->data();
            int total = 0;
            for (const int w : *weights) total += w;
            vd.total_weight = total;
        }
        return viterbiDP(vd);
    }
};

int weightedLength(const std::vector<std::int64_t>& path, const std::vector<int>& w)
{
    int n = 0;
    for (const auto b : path) n += w[static_cast<std::size_t>(b)];
    return n;
}

bool test_null_equals_ones(const Fixture& f)
{
    const std::vector<int> ones(static_cast<std::size_t>(kN), 1);
    const ViterbiPath a = f.run(nullptr);
    const ViterbiPath b = f.run(&ones);
    if (a.path.empty() || a.path != b.path || a.total_cost != b.total_cost) {
        std::fprintf(stderr, "[FAIL] all-ones weights differ from null (paths %zu / %zu, cost %.6f / %.6f)\n",
                     a.path.size(), b.path.size(), a.total_cost, b.total_cost);
        return false;
    }
    std::fprintf(stderr, "[PASS] null weights == all-ones weights (%zu beats, cost %.4f)\n",
                 a.path.size(), a.total_cost);
    return true;
}

bool test_hole_weight(const Fixture& f)
{
    std::vector<int> w(static_cast<std::size_t>(kN), 1);
    w[kHole] = kHoleW;
    const ViterbiPath base = f.run(nullptr);
    const ViterbiPath hole = f.run(&w);
    if (hole.path.empty()) {
        std::fprintf(stderr, "[FAIL] weighted DP returned an empty path\n");
        return false;
    }
    const int base_len = weightedLength(base.path, w);
    const int hole_len = weightedLength(hole.path, w);
    const bool base_crosses = base_len != static_cast<int>(base.path.size());
    if (hole_len != kTarget) {
        std::fprintf(stderr, "[FAIL] weighted length %d != target %d (%zu beats)\n",
                     hole_len, kTarget, hole.path.size());
        return false;
    }
    if (base_crosses && base_len != kTarget + kHoleW - 1) {
        std::fprintf(stderr, "[FAIL] unweighted path crosses the hole but weighs %d\n", base_len);
        return false;
    }
    std::fprintf(stderr, "[PASS] weighted path = %d slots over %zu beats (unweighted: %zu beats = %d slots, crosses hole: %s)\n",
                 hole_len, hole.path.size(), base.path.size(), base_len, base_crosses ? "yes" : "no");
    return true;
}

bool test_weights_helper()
{
    // Uniform 0.5 s grid with one 6.0-period gap, one 1.3 and one 1.6.
    std::vector<double> t;
    double x = 0.0;
    for (int k = 0; k < 40; ++k) {
        t.push_back(x);
        if (k == 10)      x += 6.0 * 0.5;
        else if (k == 20) x += 1.3 * 0.5;
        else if (k == 30) x += 1.6 * 0.5;
        else              x += 0.5;
    }
    double period = 0.0;
    const std::vector<int> w = holeAwareBeatWeights(t.data(), static_cast<int>(t.size()), &period);
    bool ok = std::fabs(period - 0.5) < 1e-9 && w.size() == t.size()
           && w[10] == 6 && w[20] == 1 && w[30] == 2 && w.back() == 1;
    for (std::size_t k = 0; ok && k + 1 < w.size(); ++k)
        if (k != 10 && k != 30 && w[k] != 1) ok = false;
    // Uniform grid = all ones.
    std::vector<double> u;
    for (int k = 0; k < 16; ++k) u.push_back(0.6 * k);
    double pu = 0.0;
    const std::vector<int> wu = holeAwareBeatWeights(u.data(), 16, &pu);
    for (const int v : wu) if (v != 1) ok = false;
    if (std::fabs(pu - 0.6) > 1e-9) ok = false;
    if (!ok) {
        std::fprintf(stderr, "[FAIL] holeAwareBeatWeights: period %.3f, w[10]=%d w[20]=%d w[30]=%d last=%d\n",
                     period, w[10], w[20], w[30], w.back());
        return false;
    }
    std::fprintf(stderr, "[PASS] holeAwareBeatWeights: 6.0 -> 6, 1.3 -> 1, 1.6 -> 2, uniform -> ones\n");
    return true;
}

} // namespace

int main()
{
    const Fixture f;
    bool ok = true;
    ok = test_null_equals_ones(f) && ok;
    ok = test_hole_weight(f)      && ok;
    ok = test_weights_helper()    && ok;
    std::fprintf(stderr, ok ? "== test_hole_aware_length PASS ==\n" : "== test_hole_aware_length FAIL ==\n");
    return ok ? 0 : 1;
}
