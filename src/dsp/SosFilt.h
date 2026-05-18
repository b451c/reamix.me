#pragma once

#include <cstddef>

namespace reamix::dsp {

// IIR filter in Second-Order Sections (SOS) form — transposed direct form II.
// Port of `scipy.signal.sosfilt` for zero-initial-state case (zi=None).
//
// Consumed by:
//   - TransitionCost (phase-4 session 18) — vocal-band (250-3400 Hz)
//     bandpass for deceptive-splice detection on boundary waveforms.
//   - Phase-5 renderer multi-band crossover (future).
//
// Input format matches scipy `butter(N, [lo, hi], btype, fs=sr, output="sos")`:
//   sos shape (n_sections, 6) row-major, each row = [b0, b1, b2, a0, a1, a2]
//   where a0 is always 1.0 (scipy normalizes). n_sections for a bandpass
//   of order N is N (scipy doubles analog order N → digital 2N-pole filter
//   in N biquads).
//
// Transposed Direct Form II biquad (per section):
//     y = b0 * x + s1
//     s1 = b1 * x - a1 * y + s2
//     s2 = b2 * x - a2 * y
// State (s1, s2) starts at zero. Sections are cascaded: input of section
// k+1 is output of section k.
//
// PARITY notes vs `scipy.signal.sosfilt`:
//   - scipy uses the exact same transposed direct form II kernel in its
//     C implementation (`scipy/signal/_sosfilt.pyx`). Bitwise parity is
//     achievable with matching state semantics + same arithmetic order.
//   - We process sample-by-sample (not block-by-block) to match scipy's
//     scalar evaluation and avoid SIMD-induced reorder drift.
//   - Input/output are f64 throughout. Caller converts f32 → f64 before
//     calling (matches Python L353 `.astype(np.float64)`) and casts the
//     output back if needed (Python L353 `.astype(np.float32)`).
//   - Expects a0 == 1.0 in every section; no explicit normalization.
//
// Edge cases:
//   - n_sections == 0 → output = input (pass-through).
//   - n_samples   == 0 → no-op, output untouched.
//   - NaN/Inf in input propagates deterministically through the state
//     variables. No silent fallback (ADR-023 port guidance).
class SosFilt
{
public:
    // Apply SOS cascade to input in-place-safe manner (output may alias
    // input). Zero initial state. Caller owns all buffers.
    //
    //   sos:        (n_sections × 6) row-major, f64. Rows are
    //               [b0, b1, b2, a0, a1, a2]; a0 must equal 1.0.
    //   input:      (n_samples,) f64.
    //   output:     (n_samples,) f64. Written with the filtered signal.
    static void apply(const double* sos,
                      std::size_t   n_sections,
                      const double* input,
                      std::size_t   n_samples,
                      double*       output);

    // Like apply() but with caller-provided initial state per section
    // (matches scipy `sosfilt(sos, x, zi=zi_init)`; phase-5 session 30).
    //
    //   zi_init:    (n_sections × 2) row-major, f64. Each row is the
    //               pair (s1_0, s2_0) = initial transposed-direct-form-II
    //               state for that section. For a zero-phase forward-
    //               backward filter, this is `sosfilt_zi(sos) * x_edge`
    //               where x_edge is the first sample of the (padded)
    //               input for the forward pass.
    //   zf_out:     (n_sections × 2) row-major, f64. Final state
    //               (s1_end, s2_end) written back for caller's next pass.
    //               Pass nullptr to skip writeback.
    //
    // Arithmetic order is identical to apply() — biquad kernel per
    // section (y = b0*x + s1; s1 = b1*x - a1*y + s2; s2 = b2*x - a2*y)
    // executed sample-outer / section-inner.
    static void applyWithZi(const double* sos,
                            std::size_t   n_sections,
                            const double* zi_init,
                            const double* input,
                            std::size_t   n_samples,
                            double*       output,
                            double*       zf_out);
};

} // namespace reamix::dsp
