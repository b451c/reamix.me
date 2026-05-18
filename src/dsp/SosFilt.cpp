#include "dsp/SosFilt.h"

#include <vector>

namespace reamix::dsp {

void SosFilt::apply(const double* sos,
                    std::size_t   n_sections,
                    const double* input,
                    std::size_t   n_samples,
                    double*       output)
{
    // Pass-through when no sections: avoids a no-op loop + allows callers
    // to compose a dynamic cascade.
    if (n_sections == 0) {
        for (std::size_t n = 0; n < n_samples; ++n) {
            output[n] = input[n];
        }
        return;
    }

    // Per-section state [s1, s2], zero initial condition (scipy zi=None).
    // Allocate once per call; small (2 × n_sections doubles, typically
    // 4 × 8 = 32 bytes for a 4th-order bandpass).
    std::vector<double> state(2 * n_sections, 0.0);

    // Process sample-by-sample through the full cascade. Matches scipy
    // `_sosfilt_float64` kernel iteration order (outer: samples, inner:
    // sections). Reverse order (sections outer, samples inner) would
    // need per-section intermediate buffers — same math but higher
    // memory + ULP-identical only in the rounding-associative case.
    for (std::size_t n = 0; n < n_samples; ++n) {
        double x = input[n];
        for (std::size_t k = 0; k < n_sections; ++k) {
            const double* sect = sos + k * 6;
            const double  b0 = sect[0];
            const double  b1 = sect[1];
            const double  b2 = sect[2];
            // sect[3] = a0, assumed 1.0 (scipy normalizes in zpk2sos).
            const double  a1 = sect[4];
            const double  a2 = sect[5];

            double&       s1 = state[2 * k + 0];
            double&       s2 = state[2 * k + 1];

            // Transposed direct form II (matches scipy implementation):
            //   y    = b0*x + s1
            //   s1'  = b1*x - a1*y + s2
            //   s2'  = b2*x - a2*y
            const double y = b0 * x + s1;
            s1 = b1 * x - a1 * y + s2;
            s2 = b2 * x - a2 * y;

            x = y; // Cascade into next section.
        }
        output[n] = x;
    }
}

void SosFilt::applyWithZi(const double* sos,
                          std::size_t   n_sections,
                          const double* zi_init,
                          const double* input,
                          std::size_t   n_samples,
                          double*       output,
                          double*       zf_out)
{
    if (n_sections == 0) {
        for (std::size_t n = 0; n < n_samples; ++n) {
            output[n] = input[n];
        }
        return;
    }

    // Initialize state from caller-provided zi: matches scipy's
    // `zi * x_0` initialization for sosfiltfilt forward/backward passes.
    std::vector<double> state(2 * n_sections, 0.0);
    for (std::size_t k = 0; k < n_sections; ++k) {
        state[2 * k + 0] = zi_init[2 * k + 0];
        state[2 * k + 1] = zi_init[2 * k + 1];
    }

    // Same kernel as apply(); only the initial state differs.
    for (std::size_t n = 0; n < n_samples; ++n) {
        double x = input[n];
        for (std::size_t k = 0; k < n_sections; ++k) {
            const double* sect = sos + k * 6;
            const double  b0 = sect[0];
            const double  b1 = sect[1];
            const double  b2 = sect[2];
            const double  a1 = sect[4];
            const double  a2 = sect[5];

            double&       s1 = state[2 * k + 0];
            double&       s2 = state[2 * k + 1];

            const double y = b0 * x + s1;
            s1 = b1 * x - a1 * y + s2;
            s2 = b2 * x - a2 * y;

            x = y;
        }
        output[n] = x;
    }

    if (zf_out != nullptr) {
        for (std::size_t k = 0; k < n_sections; ++k) {
            zf_out[2 * k + 0] = state[2 * k + 0];
            zf_out[2 * k + 1] = state[2 * k + 1];
        }
    }
}

} // namespace reamix::dsp
