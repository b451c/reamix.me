// test_repetition_prior - ADR-115 E4 (sesja 115) self-validation of the
// repetition-diagonal candidate prior on synthetic recurrence matrices.
// C++-canonical, hand-computed expectations, no Python reference.
#include "remix/RepetitionPrior.h"

#include <cstdio>
#include <set>
#include <vector>

using namespace reamix::remix;

static int g_fail = 0;
static void expectTrue(const char* what, bool v)
{
    if (! v) { std::printf("FAIL %s\n", what); ++g_fail; }
    else       std::printf("ok   %s\n", what);
}

int main()
{
    // 32 beats, 4/4: downbeats 0,4,8,...; pre-downbeats 3,7,11,...
    const int n = 32, ts = 4;
    std::set<int> db, pre;
    for (int b = 0; b < n; b += ts) { db.insert(b); if (b - 1 >= 0) pre.insert(b - 1); }
    pre.insert(n - 1);

    // Recurrence: bars 0-1 (beats 0-7) repeat as bars 4-5 (beats 16-23):
    // R[b, b+16] = R[b+16, b] = 0.5 for b in [0, 8). Plus an isolated
    // single-beat match (11, 27) - the "Dancing Queen" class.
    std::vector<double> R(static_cast<std::size_t>(n) * n, 0.0);
    auto set = [&](int r, int c, double v) { R[(std::size_t) r * n + c] = v; R[(std::size_t) c * n + r] = v; };
    for (int b = 0; b < 8; ++b) set(b, b + 16, 0.5);
    set(11, 27, 0.9);

    const auto p = RepetitionPrior::fromRecurrence(R.data(), n, pre, db, ts);
    // Junction 3 -> 20: successor view (4, 20) lies on the diagonal offset 16
    // with hits at k in [-4, 4): rows 0..7 vs cols 16..23 -> 8 hits >= 4.
    expectTrue("prior built", p.n_allowed > 0);
    expectTrue("3 -> 20 allowed (one measure of context on the diagonal)", p.allowed(3, 20));
    // Junction 19 -> 4: successor view (20, 4) is the mirrored diagonal.
    expectTrue("19 -> 4 allowed (backward jump on the same repeat)", p.allowed(19, 4));
    // Junction 7 -> 24: successor view (8, 24): rows 4..11 vs cols 20..27 ->
    // hits only for rows 4..7 (4 hits) = exactly one measure -> allowed.
    expectTrue("7 -> 24 allowed (exactly one measure)", p.allowed(7, 24));
    // Junction 11 -> 28: successor view (12, 28): rows 8..15 vs cols 24..31 ->
    // 0 hits on the repeat; the isolated (11, 27) cell is 1 hit < 4 -> blocked.
    expectTrue("11 -> 28 blocked (isolated single-beat match)", ! p.allowed(11, 28));
    expectTrue("3 -> 12 blocked (no repetition)", ! p.allowed(3, 12));
    // Coverage: allowed sources are 3 (->20), 7 (->24), 19 (->4), 23 (->8) ...
    expectTrue("active: source coverage >= 25 %", p.active);
    std::printf("     n_allowed=%d n_sources=%d\n", p.n_allowed, p.n_sources);

    // Fallback: a matrix with a single tiny repeat -> too few sources -> inactive,
    // and an inactive prior allows everything.
    std::vector<double> R2(static_cast<std::size_t>(n) * n, 0.0);
    for (int b = 0; b < 4; ++b) R2[(std::size_t) b * n + (b + 16)] = 0.5;
    const auto p2 = RepetitionPrior::fromRecurrence(R2.data(), n, pre, db, ts);
    expectTrue("sparse repeat -> prior inactive (fallback to E3)", ! p2.active);
    expectTrue("inactive prior allows every pair", p2.allowed(11, 28) && p2.allowed(3, 12));

    // Threshold: cells at or below 0.1 do not count.
    std::vector<double> R3(static_cast<std::size_t>(n) * n, 0.1);
    const auto p3 = RepetitionPrior::fromRecurrence(R3.data(), n, pre, db, ts);
    expectTrue("R == threshold everywhere -> no hits -> inactive", ! p3.active && p3.n_allowed == 0);

    std::printf(g_fail == 0 ? "test_repetition_prior PASS\n" : "test_repetition_prior FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
