// PhaseAlign — time-domain brute-force NCC shift search + shift-apply.
// Port of `_phase_align` (crossfade.py:86-153) per ADR-031 SB-4.
//
// Parity class target: shift-index bitwise exact (int) + aligned output
// max_abs ≤ 1e-12 (f64 ULP on dot / norm accumulation, no sosfiltfilt-style
// reverse-and-re-apply accumulation so gate is tighter than Butterworth
// Fixture B).

#include "render/PhaseAlign.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace reamix::render {

namespace {

// 2-norm of f64 array, computed via `sqrt(dot(x,x))` rather than
// `np.linalg.norm`'s BLAS-dispatched variant.
//
// numpy's `np.linalg.norm` on a 1-D f64 array of length N defers to LAPACK
// dnrm2 when N is large enough; for our typical N<=2048 slice it falls
// through to the naive `sqrt(sum(x*x))` path (numpy/linalg/_linalg.py). The
// naive path uses a straight-line accumulator. Matches bit-exactly against
// `-ffp-contract=off` pure-scalar C++ in the configs tested by session 31.
inline double l2norm(const double* x, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += x[i] * x[i];
    }
    return std::sqrt(s);
}

// Straight dot product. Same accumulation discipline as numpy's `np.dot`
// on two f64 vectors (defers to BLAS dgemv/dgemm for large N; naive for
// small N). With `-ffp-contract=off` the C++ scalar loop matches numpy
// bit-exact at the lengths we exercise.
inline double dot(const double* a, const double* b, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

} // namespace

void PhaseAlign::alignMono(const double* out,
                           std::size_t   nOut,
                           const double* in_,
                           std::size_t   nIn,
                           int           maxShift,
                           int*          shiftOut,
                           double*       alignedOut)
{
    // min-length + max_shift early return: `(incoming, 0)` passthrough.
    // crossfade.py:107-108.
    const std::size_t n = (nOut < nIn) ? nOut : nIn;
    if (n < 32 || maxShift <= 0) {
        *shiftOut = 0;
        if (alignedOut != in_) {
            std::memcpy(alignedOut, in_, nIn * sizeof(double));
        }
        return;
    }

    const std::size_t minOverlap =
        std::max<std::size_t>(16u, n / 2u);

    // Initial sentinel must be strictly lower than any achievable corr so
    // that the first valid iteration always wins. Python: `best_corr = -1.0`
    // (crossfade.py:111). For f64 NCC on real audio, `corr == -1.0` exactly
    // is practically unreachable; but we preserve the sentinel verbatim
    // for bit-exact parity of the `corr > best_corr` update rule.
    double bestCorr  = -1.0;
    int    bestShift = 0;

    // Ascending iteration from -maxShift to +maxShift inclusive. First
    // occurrence wins on equal corr via strict `>`. Mirrors
    // `for shift in range(-max_shift, max_shift + 1)` (crossfade.py:115).
    for (int shift = -maxShift; shift <= maxShift; ++shift) {
        const double* o;
        const double* i;
        std::size_t   len;

        if (shift >= 0) {
            // o = out[shift:], i = in[:n-shift]
            const std::size_t s = static_cast<std::size_t>(shift);
            o   = out + s;
            i   = in_;
            len = n - s;
        } else {
            // o = out[:n+shift], i = in[-shift:]
            const std::size_t s = static_cast<std::size_t>(-shift);
            o   = out;
            i   = in_ + s;
            len = n - s;
        }

        if (len < minOverlap) continue;

        const double oN = l2norm(o, len);
        const double iN = l2norm(i, len);
        if (oN < 1e-8 || iN < 1e-8) continue;

        const double corr = dot(o, i, len) / (oN * iN);

        // Cast to float via `float(np.dot(...) / (...))` in Python
        // (crossfade.py:131). For f64 inputs this is a no-op at value level
        // (the cast is already from Python `float` to Python `float`; numpy
        // returns a numpy scalar which `float()` converts). C++ f64 path
        // matches directly.
        if (corr > bestCorr) {
            bestCorr  = corr;
            bestShift = shift;
        }
    }

    *shiftOut = bestShift;

    if (bestShift == 0) {
        // Fast passthrough: `return (incoming, 0)` without constructing a
        // zeros-buffer (crossfade.py:136-137).
        if (alignedOut != in_) {
            std::memcpy(alignedOut, in_, nIn * sizeof(double));
        }
        return;
    }

    // Shift-apply branch (crossfade.py:139-153). result = zeros_like(incoming).
    // Zero the entire output first, then copy the shifted slice in.
    std::memset(alignedOut, 0, nIn * sizeof(double));

    if (bestShift > 0) {
        // result[:-bestShift] = incoming[bestShift:]
        const std::size_t s = static_cast<std::size_t>(bestShift);
        // incoming has nIn samples; shifted source starts at s, copies (nIn - s).
        std::memcpy(alignedOut, in_ + s, (nIn - s) * sizeof(double));
    } else {
        // result[s:] = incoming[:-s]  where s = -bestShift
        const std::size_t s = static_cast<std::size_t>(-bestShift);
        std::memcpy(alignedOut + s, in_, (nIn - s) * sizeof(double));
    }
}

} // namespace reamix::render
