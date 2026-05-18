// Parity test: reamix::TempoEstimator must produce byte-identical output
// to REABeat's reference TempoEstimator for the same input.
//
// Success criterion: bit-identical float32 return value (compared via
// memcpy to uint32_t) on every case. A single `==` compare would let
// NaN-equivalents slip through; TempoEstimator does not produce NaNs on
// these cases, but bit-compare is the correct parity assertion.
//
// TempoEstimator is pure stdlib + <cmath>: std::fmod, std::sin, std::cos,
// std::atan2, std::round, std::sort, std::accumulate. Both ports call the
// same libm on the same bits, so bitwise parity is structurally guaranteed.
// The cases below exist to cover branches, not to stress numerics.
//
// Cases (1:1 branch coverage):
//   1) too_short           — beats.size() < 2 → fallback 120 BPM
//   2) steady_4_4_120bpm   — 10 s of 120 BPM (0.5 s gap) → happy path, r² > 0.99
//   3) steady_3_4_waltz    — 150 BPM (0.4 s gap) in [78, 185] → no octave shift
//   4) octave_correct_half — 40 BPM (1.5 s gap) → *2 into range (80 BPM)
//   5) octave_correct_double — 240 BPM (0.25 s gap) → /2 into range (120 BPM)
//   6) sparse_irregular    — jittered beats, r² < 0.99 → mean-fallback path
//   7) all_filtered_out    — all dt outside [0.2, 2.0] s → empty intervals → fallback
//
// Each case also exercises octaveCorrect() directly via compute()'s return
// path, so we skip a separate sub-suite for it.

#include "analysis/TempoEstimator.h"
#include "TempoEstimatorReabeat.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint32_t bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

struct Case
{
    std::string name;
    std::vector<float> beats;
};

// --- Cases --------------------------------------------------------------

Case tooShort()
{
    Case c;
    c.name = "too_short";
    // size() < 2 → fallback 120 BPM.
    c.beats = {1.0f};
    return c;
}

Case steady4_4_120bpm()
{
    Case c;
    c.name = "steady_4_4_120bpm";
    // 120 BPM → 0.5 s interval. 20 beats over 10 s. Regular → r² ≈ 1.0,
    // slope ≈ 0.5, 60/0.5 = 120 BPM already in [78, 185].
    for (int i = 0; i < 20; ++i)
        c.beats.push_back(static_cast<float>(i) * 0.5f);
    return c;
}

Case steady3_4Waltz()
{
    Case c;
    c.name = "steady_3_4_waltz";
    // 150 BPM → 0.4 s interval. 25 beats over 10 s. In [78, 185] → no octave shift.
    for (int i = 0; i < 25; ++i)
        c.beats.push_back(static_cast<float>(i) * 0.4f);
    return c;
}

Case octaveCorrectHalf()
{
    Case c;
    c.name = "octave_correct_half";
    // 40 BPM → 1.5 s interval. Below 78 BPM → doubled to 80 BPM.
    // 1.5 s is inside [0.2, 2.0] s interval window so it's not filtered out.
    for (int i = 0; i < 10; ++i)
        c.beats.push_back(static_cast<float>(i) * 1.5f);
    return c;
}

Case octaveCorrectDouble()
{
    Case c;
    c.name = "octave_correct_double";
    // 240 BPM → 0.25 s interval. Above 185 BPM → halved to 120 BPM.
    // 0.25 s is inside [0.2, 2.0] s interval window.
    for (int i = 0; i < 40; ++i)
        c.beats.push_back(static_cast<float>(i) * 0.25f);
    return c;
}

Case sparseIrregular()
{
    Case c;
    c.name = "sparse_irregular";
    // Jittered around 0.5 s median but with large per-interval drift so
    // the linear regression r² falls below 0.99. Drives the mean-fallback
    // return path: return octaveCorrect(60 / avgInterval).
    // Intervals chosen so that the ±15% median filter keeps several of
    // them (median will be 0.5, ±15% band is [0.425, 0.575]).
    c.beats = {
        0.00f,
        0.55f,  // dt = 0.55  (inside band)
        1.00f,  // dt = 0.45  (inside band)
        1.70f,  // dt = 0.70  (outside band)
        2.15f,  // dt = 0.45  (inside band)
        2.50f,  // dt = 0.35  (outside band)
        3.05f,  // dt = 0.55  (inside band)
        3.85f,  // dt = 0.80  (outside band)
        4.30f,  // dt = 0.45  (inside band)
        4.90f,  // dt = 0.60  (outside band)
    };
    return c;
}

Case allFilteredOut()
{
    Case c;
    c.name = "all_filtered_out";
    // All intervals are 0.1 s (below kIntervalMinSec=0.2) → filtered out
    // → intervals.empty() → fallback 120 BPM.
    for (int i = 0; i < 20; ++i)
        c.beats.push_back(static_cast<float>(i) * 0.1f);
    return c;
}

// --- Runner -------------------------------------------------------------

int runCase(const Case& c)
{
    float ours = reamix::TempoEstimator::compute(c.beats);
    float ref  = TempoEstimatorReabeat::compute(c.beats);

    uint32_t ob = bits(ours);
    uint32_t rb = bits(ref);

    std::printf("[%s] in=%zu ours=%.9g (0x%08x) ref=%.9g (0x%08x)\n",
                c.name.c_str(), c.beats.size(),
                static_cast<double>(ours), ob,
                static_cast<double>(ref),  rb);

    if (ob != rb)
    {
        std::fprintf(stderr, "FAIL [%s]: bit mismatch (ours=0x%08x ref=0x%08x)\n",
                     c.name.c_str(), ob, rb);
        return 1;
    }
    return 0;
}

} // namespace

int main()
{
    int rc = 0;
    rc |= runCase(tooShort());
    rc |= runCase(steady4_4_120bpm());
    rc |= runCase(steady3_4Waltz());
    rc |= runCase(octaveCorrectHalf());
    rc |= runCase(octaveCorrectDouble());
    rc |= runCase(sparseIrregular());
    rc |= runCase(allFilteredOut());

    if (rc != 0)
    {
        std::fprintf(stderr, "FAIL: at least one case diverged from REABeat reference\n");
        return 1;
    }

    std::printf("PASS: bit-identical to REABeat reference on all 7 cases\n");
    return 0;
}
