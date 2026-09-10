// test_tail_reach — sesja 127 (DEV-117 a) self-validation.
//
// The Duration DP must end at the song's ending (the renderer appends the
// file head and tail only then). With the sesja-126 rule the tail was
// required only inside the length window; when the pool's only loop
// overshot the window by a second the fallback ended the path early and the
// remix lost its head and tail (vocal_solo x1.25: -40 s). `tail_search_band`
// / `tail_search_extension` widen the window one tolerance-band per pass and
// take the first band with a tail endpoint. No Python reference
// (C++-canonical, ADR-065).
//
// Fixture: 64 beats in 4/4, sequential edges free, ONE loop edge 59 -> 28
// (raw 0.3, span 31), intro 4, outro 4, cooldown 3. Sequential-only = 64
// slots, one loop = 96, two = 128.
//
// Asserts:
//   1. Target 84 +-4 ([80, 88]) without bands: no tail endpoint in the
//      window -> the fallback ends early (the sesja-126 behaviour).
//   2. Same target with band 4 / extension 16: band 2 ([72, 96]) holds the
//      one-loop tail path -> ends at beat 63, t = 96, exactly one cut.
//   3. Target 96 +-4 (in-window tail reachable): the banded run == the
//      un-banded run bit-exact (rows past T never change rows <= T).
//   4. end_within_last = 0 (legacy) with bands set == legacy without.

#include "remix/ViterbiDP.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using reamix::remix::ViterbiDPInputs;
using reamix::remix::ViterbiPath;
using reamix::remix::viterbiDP;

namespace {

constexpr double INF_VAL = 1e9;
constexpr int    kN      = 64;
constexpr int    kTS     = 4;
constexpr int    kFrom   = 59;   // pre-downbeat beat (60 % 4 == 0)
constexpr int    kTo     = 28;   // downbeat

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
            if (i + 1 < kN) {
                W[(std::size_t) i * kN + (i + 1)] = 0.0;
                indices.push_back(i + 1);
            }
            if (i == kFrom) {
                W[(std::size_t) i * kN + kTo] = 0.3;
                indices.push_back(kTo);
            }
            offsets[(std::size_t) i + 1] = static_cast<std::int64_t>(indices.size());
        }
    }

    ViterbiPath run(int target, int tol, int end_within_last, int band, int extension) const
    {
        ViterbiDPInputs vd{};
        vd.W                  = W.data();
        vd.n_beats            = kN;
        vd.target_length      = target + tol;
        vd.min_target_length  = target - tol;
        vd.intro_beats        = 4;
        vd.outro_beats        = 4;
        vd.is_shortening      = false;
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
        vd.end_within_last    = end_within_last;
        vd.tail_search_band      = band;
        vd.tail_search_extension = extension;
        return viterbiDP(vd);
    }
};

int countJumps(const std::vector<std::int64_t>& p)
{
    int n = 0;
    for (std::size_t k = 1; k < p.size(); ++k) if (p[k] != p[k - 1] + 1) ++n;
    return n;
}

bool same(const ViterbiPath& a, const ViterbiPath& b)
{
    return a.path == b.path && a.total_cost == b.total_cost;
}

bool test_no_bands_ends_early(const Fixture& f)
{
    const ViterbiPath r = f.run(84, 4, 3, 0, 0);
    if (r.path.empty() || r.at_tail || r.path.back() >= kN - 3
        || static_cast<int>(r.path.size()) < 80 || static_cast<int>(r.path.size()) > 88) {
        std::fprintf(stderr, "[FAIL] no bands: %zu beats, last %lld, at_tail %d\n",
                     r.path.size(), r.path.empty() ? -1LL : (long long) r.path.back(), (int) r.at_tail);
        return false;
    }
    std::fprintf(stderr, "[PASS] no bands: fallback ends early at beat %lld (%zu slots)\n",
                 (long long) r.path.back(), r.path.size());
    return true;
}

bool test_bands_reach_tail(const Fixture& f)
{
    const ViterbiPath r = f.run(84, 4, 3, 4, 16);
    if (r.path.empty() || !r.at_tail || r.path.back() != kN - 1 || r.end_t != 96
        || r.path.size() != 96 || countJumps(r.path) != 1) {
        std::fprintf(stderr, "[FAIL] bands: %zu beats, last %lld, end_t %d, at_tail %d, jumps %d\n",
                     r.path.size(), r.path.empty() ? -1LL : (long long) r.path.back(), r.end_t,
                     (int) r.at_tail, countJumps(r.path));
        return false;
    }
    std::fprintf(stderr, "[PASS] bands: one loop, ends at beat %lld, t = %d (band 2)\n",
                 (long long) r.path.back(), r.end_t);
    return true;
}

bool test_in_window_bit_exact(const Fixture& f)
{
    const ViterbiPath a = f.run(96, 4, 3, 0, 0);
    const ViterbiPath b = f.run(96, 4, 3, 4, 16);
    if (a.path.empty() || !a.at_tail || !same(a, b) || b.end_t != 96) {
        std::fprintf(stderr, "[FAIL] in-window: %zu / %zu beats, cost %.6f / %.6f, end_t %d\n",
                     a.path.size(), b.path.size(), a.total_cost, b.total_cost, b.end_t);
        return false;
    }
    std::fprintf(stderr, "[PASS] in-window tail path identical with and without bands (%zu slots)\n",
                 a.path.size());
    return true;
}

bool test_legacy_ignores_bands(const Fixture& f)
{
    const ViterbiPath a = f.run(84, 4, 0, 0, 0);
    const ViterbiPath b = f.run(84, 4, 0, 4, 16);
    if (a.path.empty() || !same(a, b) || b.at_tail) {
        std::fprintf(stderr, "[FAIL] legacy: %zu / %zu beats, cost %.6f / %.6f\n",
                     a.path.size(), b.path.size(), a.total_cost, b.total_cost);
        return false;
    }
    std::fprintf(stderr, "[PASS] legacy (end_within_last 0) ignores the bands (%zu slots)\n", a.path.size());
    return true;
}

} // namespace

int main()
{
    const Fixture f;
    bool ok = true;
    ok = test_no_bands_ends_early(f)  && ok;
    ok = test_bands_reach_tail(f)     && ok;
    ok = test_in_window_bit_exact(f)  && ok;
    ok = test_legacy_ignores_bands(f) && ok;
    std::fprintf(stderr, ok ? "== test_tail_reach PASS ==\n" : "== test_tail_reach FAIL ==\n");
    return ok ? 0 : 1;
}
