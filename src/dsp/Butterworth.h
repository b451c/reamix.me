#pragma once

#include <cstddef>

namespace reamix::dsp {

// Butterworth IIR filter design + zero-phase forward-backward filtering.
// Port of `scipy.signal.butter(..., output='sos')` + `scipy.signal.sosfiltfilt`
// with default `padtype='odd'`, `padlen = 3*(2*n_sections+1 - zero_pair_count)`.
//
// Consumed by:
//   - phase-5 renderer crossfade (`src/render/Crossfade.cpp`) for the
//     3-band split (bass LP @200 Hz / mid BP [200, 4000] Hz / treble HP
//     @4000 Hz) from `references/python-source/remix/crossfade.py:BANDS`.
//
// Design algorithm (port of scipy, ADR-031 choice A):
//   1. Analog prototype poles: p_k = -exp(j*pi*(2k-N+1)/(2N)) for k=1..N.
//   2. Frequency pre-warp: warped = 2*fs*tan(pi*Wn/fs) with internal fs=2.
//   3. btype transform:
//      LP (`_zpklp2lp`): p_a = warped*p; z_a = empty; k_a = warped^N.
//      HP (`_zpklp2hp`): p_a = warped/p; z_a = N zeros at origin;
//                        k_a = 1/real(prod(-p_proto)).
//      BP (`_zpklp2bp`): for each p_proto, two poles
//                        p_a,± = p_lp ± sqrt(p_lp^2 - wo^2), p_lp = p*bw/2;
//                        N zeros at origin; k_a = bw^N.
//   4. Bilinear (`bilinear_zpk` with fs=2): p_d = (4+p_a)/(4-p_a),
//      z_d = (4+z_a)/(4-z_a), N-count zeros added at z=-1 for degree
//      balancing; k_d = k_a * real(prod(4-z_a)/prod(4-p_a)).
//   5. zpk2sos (`pairing='nearest'`): sort poles by closeness to unit
//      circle; process worst-first into last section; pair each complex
//      pole p1 with conjugate p2=p1.conj(); find closest real zero,
//      then closest real partner; gain k consolidated into sos[0].
//
// sosfiltfilt algorithm (port of scipy, empirically verified session 30):
//   1. ntaps = 2*n_sections + 1 - min(count(sos[:,2]==0), count(sos[:,5]==0)).
//      For Butterworth order 4: LP/HP -> ntaps=5, BP -> ntaps=9.
//   2. edge = 3*ntaps (default padlen). Odd-extend input by `edge` each side.
//   3. zi = sosfilt_zi(sos): per-section steady-state for step response
//      (stable DC response); scaled through sections by running gain product.
//   4. Forward pass with initial state = zi * x_ext[0].
//   5. Reverse, second pass with initial state = zi * y_fwd[-1].
//   6. Reverse, strip padding `edge` samples each side.
//
// Parity class:
//   - SOS coefficient design: target 1e-12 bit-exact vs scipy per-element.
//     See ADR-031. Row order matches scipy (worst-pole last).
//   - sosfiltfilt output: target 1e-9 on synthetic signals (sine sweep +
//     white noise + unit impulse) × 3 band configs × 2 sample rates.
//     Builds on SosFilt session-18 ULP baseline ~2.3e-14 + 2× reverse-and-
//     re-apply accumulation.
class Butterworth
{
public:
    // Max sections ever produced by this class at order 4:
    //   LP/HP -> 2, BP -> 4. Callers can allocate `kMaxSections * 6` f64.
    static constexpr std::size_t kMaxSections = 4;

    // Design a Butterworth lowpass. `Wn` is the cutoff frequency
    // normalized to Nyquist (i.e., `cutoff_hz / (sr/2)`), in (0, 1).
    //
    //   order:          filter order N.
    //   Wn:             normalized cutoff (fraction of Nyquist).
    //   sosOut:         output buffer, at least (N/2) × 6 f64
    //                   (2 sections for order 4).
    //   nSectionsOut:   written with the number of sections produced.
    //
    // Rows are `[b0, b1, b2, a0, a1, a2]`, a0 == 1.0, matching scipy.
    // Row order: worst pole (closest to unit circle) last; gain in
    // sos[0][:3].
    static void designLowpass(int          order,
                              double       Wn,
                              double*      sosOut,
                              std::size_t* nSectionsOut);

    // Design a Butterworth highpass. Same conventions as designLowpass.
    static void designHighpass(int          order,
                               double       Wn,
                               double*      sosOut,
                               std::size_t* nSectionsOut);

    // Design a Butterworth bandpass. `WnLow` and `WnHigh` are the
    // passband edges normalized to Nyquist, 0 < WnLow < WnHigh < 1.
    //
    //   sosOut:         output buffer, at least N × 6 f64 (4 sections
    //                   for order 4 — scipy doubles analog order for BP).
    static void designBandpass(int          order,
                               double       WnLow,
                               double       WnHigh,
                               double*      sosOut,
                               std::size_t* nSectionsOut);

    // Compute per-section initial conditions for the stable DC step
    // response, matching `scipy.signal.sosfilt_zi(sos)`.
    //
    //   sos:         (n_sections × 6) f64 row-major.
    //   ziOut:       (n_sections × 2) f64 row-major; each row = (s1, s2).
    static void sosfiltZi(const double* sos,
                          std::size_t   n_sections,
                          double*       ziOut);

    // Zero-phase forward-backward filter. Matches scipy
    // `sosfiltfilt(sos, x, padtype='odd', padlen=default)`. Requires
    // n_samples > 3 * (2*n_sections + 1 - zero_pair_count); throws
    // std::invalid_argument otherwise (matches scipy's ValueError).
    //
    //   sos:         (n_sections × 6) f64 row-major.
    //   input:       (n_samples,) f64.
    //   output:      (n_samples,) f64. May alias input.
    //
    // Internally allocates a padded buffer of size (n_samples + 2*edge).
    // Stateless — no pre-existing init, no retention.
    static void sosFiltFilt(const double* sos,
                            std::size_t   n_sections,
                            const double* input,
                            std::size_t   n_samples,
                            double*       output);
};

} // namespace reamix::dsp
