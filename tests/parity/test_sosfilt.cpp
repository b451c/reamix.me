// test_sosfilt — phase-4 session 18 parity: reamix::dsp::SosFilt vs
// scipy.signal.sosfilt on the Butterworth 4th-order bandpass [250, 3400] Hz
// at fs=22050 (billie_jean boundary_waveforms[0] as input, 3417 samples).
//
// Gate: ULP-class (max_abs ≤ 1e-12). Bitwise is not achievable against
// scipy's C implementation because scipy's wheel is compiled with FMA
// contraction enabled (single-rounding `a*b + c`) while our test target
// runs `-ffp-contract=off` per ADR-028 to match CPython's scalar
// arithmetic elsewhere in phase-4. The biquad kernel is identical on both
// sides; the drift is a 1-per-op rounding difference that accumulates
// through the 4-section × 3417-sample cascade. Max drift observed on
// billie_jean boundary_waveforms[0]: ~2.3e-14 abs, ~1e-13 relative —
// perceptually inaudible and well below any downstream signal.
//
// Goldens dumped by tools/dump_phase4_tests.py (session 18).

#include "dsp/SosFilt.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "NpyIO.h"

int main(int argc, char** argv)
{
    const std::string dataDir = (argc > 1)
        ? argv[1]
        : "tests/parity/reference/data/phase4/billie_jean";

    constexpr double TOL_ABS = 1e-12;

    auto sos_mat      = reamix::test::loadNpy2DFloat64(dataDir + "/sos_coeffs.npy");
    auto input_f64    = reamix::test::loadNpy1DFloat64(dataDir + "/sosfilt_input_f64.npy");
    auto expected_f64 = reamix::test::loadNpy1DFloat64(dataDir + "/sosfilt_output_f64.npy");

    if (sos_mat.cols != 6) {
        std::fprintf(stderr, "sos_coeffs has cols=%zu, expected 6\n", sos_mat.cols);
        return 1;
    }
    if (input_f64.size() != expected_f64.size()) {
        std::fprintf(stderr, "sosfilt input/output length mismatch: %zu vs %zu\n",
                     input_f64.size(), expected_f64.size());
        return 1;
    }

    std::vector<double> actual(input_f64.size(), 0.0);
    reamix::dsp::SosFilt::apply(sos_mat.data.data(),
                                sos_mat.rows,
                                input_f64.data(),
                                input_f64.size(),
                                actual.data());

    std::size_t over_tol  = 0;
    double      max_abs   = 0.0;
    std::size_t first_fail = 0;
    bool        have_fail  = false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double d = std::abs(actual[i] - expected_f64[i]);
        if (d > max_abs) max_abs = d;
        if (d > TOL_ABS) {
            if (!have_fail) { first_fail = i; have_fail = true; }
            ++over_tol;
        }
    }

    std::printf("sosfilt: n_sections=%zu n_samples=%zu over_tol=%zu/%zu "
                "max_abs=%.3e tol=%.1e  %s\n",
                sos_mat.rows, actual.size(), over_tol, actual.size(),
                max_abs, TOL_ABS, (over_tol == 0) ? "PASS" : "FAIL");
    if (have_fail) {
        std::printf("  first fail at i=%zu: cpp=%.17g python=%.17g\n",
                    first_fail, actual[first_fail], expected_f64[first_fail]);
    }

    return (over_tol == 0) ? 0 : 1;
}
