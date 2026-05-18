// Parity: reamix::util::percentile vs `np.percentile(a, q, method='linear')`
// (numpy 1.26.4). Hand-crafted hardcoded cases with expected values computed
// once via `python -c "import numpy as np; np.percentile(a, q)"`. Bitwise
// target — algorithm is pure sort + lerp in f64.
//
// Coverage:
//   c1  — odd length monotonic f64, q=50 (centre, exact-half index).
//   c2  — even length monotonic f64, q=95 (typical vocal-pipeline q).
//   c3  — q=0 returns min.
//   c4  — q=100 returns max.
//   c5  — single element, q=50 (i+1 >= n branch).
//   c6  — unsorted input, q=25 (sort internally).
//   c7  — duplicates, q=75 (no interpolation edge case).
//   c8  — negatives mixed, q=50.
//   c9  — float32 input, q=95 (exercises f32→f64 promotion on lerp).
//
// Production-scale parity (2500+ frames, f32 input from librosa) is
// validated implicitly at phase-2b's downstream steps (HPSS → SpectralFlatness
// → VocalFeatures), where percentile is called inside the pipeline and its
// output feeds flatness_inv / voice_band_ratio / vocal_rise_frame scales.

#include "util/Percentile.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct FailCount { int n = 0; };

// Bitwise comparison via bit_cast to uint64.
bool bitwiseEqual(double a, double b)
{
    std::uint64_t ai, bi;
    std::memcpy(&ai, &a, sizeof(ai));
    std::memcpy(&bi, &b, sizeof(bi));
    return ai == bi;
}

template <typename T>
void runCase(const char* name, const std::vector<T>& values, double q,
             double expected, FailCount& fc)
{
    const double got = reamix::util::percentile(values, q);
    const double diff = std::abs(got - expected);
    const bool pass = bitwiseEqual(got, expected);
    std::printf("%-32s  n=%4zu  q=%6.2f  expected=%+.17e  got=%+.17e  diff=%.2e  %s\n",
                name, values.size(), q, expected, got, diff,
                pass ? "PASS" : "FAIL");
    if (!pass) ++fc.n;
}

} // namespace

int main()
{
    FailCount fc;

    // c1 — odd length monotonic f64, q=50.
    {
        std::vector<double> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        runCase("c1_odd_mono_q50_f64", v, 50.0, 4.5, fc);
    }
    // c2 — even length monotonic f64, q=95  → 0.95*19 = 18.05.
    {
        std::vector<double> v;
        v.reserve(20);
        for (int i = 0; i < 20; ++i) v.push_back(static_cast<double>(i));
        runCase("c2_even_mono_q95_f64", v, 95.0, 18.05, fc);
    }
    // c3 — q=0 returns min.
    {
        std::vector<double> v = {3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0};
        runCase("c3_q0_min_f64", v, 0.0, 1.0, fc);
    }
    // c4 — q=100 returns max.
    {
        std::vector<double> v = {3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0};
        runCase("c4_q100_max_f64", v, 100.0, 9.0, fc);
    }
    // c5 — single element (i+1 == n branch).
    {
        std::vector<double> v = {7.25};
        runCase("c5_single_f64", v, 50.0, 7.25, fc);
    }
    // c6 — unsorted, q=25  → sorted=[-1,0,2,3,4,5]; h=0.25*5=1.25; lerp(0,2,0.25)=0.5.
    {
        std::vector<double> v = {5.0, -1.0, 3.0, 2.0, 4.0, 0.0};
        runCase("c6_unsorted_q25_f64", v, 25.0, 0.5, fc);
    }
    // c7 — duplicates, q=75  → sorted=[1,1,1,2,2,3,3,3,3,3]; h=0.75*9=6.75; lerp(3,3,0.75)=3.
    {
        std::vector<double> v = {1, 1, 1, 2, 2, 3, 3, 3, 3, 3};
        runCase("c7_duplicates_q75_f64", v, 75.0, 3.0, fc);
    }
    // c8 — negatives mixed, q=50  → sorted=[-5,-3,-1,1,3,5]; h=0.5*5=2.5; lerp(-1,1,0.5)=0.
    {
        std::vector<double> v = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
        runCase("c8_negatives_q50_f64", v, 50.0, 0.0, fc);
    }
    // c9 — float32 input, q=95. Expected from numpy:
    //   n=15, sorted ascending; h = 0.95*14 = 13.3; values at idx 13,14 are
    //   0.30f, 0.35f; lerp in f64 = 0.30 + 0.3*(0.35 - 0.30) = 0.3150000065565109.
    {
        std::vector<float> v = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f,
                                0.08f, 0.09f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f};
        runCase<float>("c9_f32_small_q95", v, 95.0, 0.3150000065565109, fc);
    }

    std::printf("\nfailures: %d\n", fc.n);
    return fc.n == 0 ? 0 : 1;
}
