#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace reamix::util {

// 2-D median filter matching `scipy.ndimage.median_filter(size=(kR, kC),
// mode='reflect')` semantics, scipy 1.15.3.
//
// Used by phase-2b HPSS with two kernel shapes:
//   (1, 31) — filter along time axis   → harmonic enhancement
//   (31, 1) — filter along freq axis   → percussive enhancement
// (librosa.effects.hpss source: librosa/decompose.py::hpss, kernel_size=31
// default — one value, applied as `(1, k)` / `(k, 1)`.)
//
// Boundary mode: 'reflect' (scipy vocabulary). Pattern for a 1-D array of
// length n=4: `… 3 2 1 0 | 0 1 2 3 | 3 2 1 0 …`. This reflects about the
// *edge of the first/last pixel*, NOT about the pixel itself (that would be
// 'mirror' in scipy). Off-by-one here causes ~30-frame drift at each track
// edge — flagged in meta/HANDOVER.md session-1 gotchas.
//
// Kernel rule: odd-sized only (assert-enforced). scipy's median_filter with
// even kernel returns the (k/2)-th rank (upper of two middles), NOT the
// true mean-of-two-middles median. Since HPSS uses kernel 31 we only need
// odd. Removing the assert requires a decision on even-kernel semantics
// (and an ADR), not a code-only change.
//
// Bitwise parity target: median is the true (kR*kC/2)-th order statistic
// — well-defined on finite sets. `std::nth_element` is an
// implementation-defined partition (IntroSelect in libc++/libstdc++) but
// returns the exact k-th smallest, which is what scipy computes via
// rank_filter(rank=size//2). Bit-identical for every finite input.
//
// Complexity: O(nRows · nCols · kRows · kCols) — naive, single-threaded.
// On (1025, 200) × (1, 31) measures ~35 ms on M1. phase-2b budget has room;
// two-heap O(nRows · nCols · log k) optimization deferred (spec § Risk #5).

namespace detail {

// `scipy` 'reflect' mode. idx may be negative or >= n. n must be >= 1.
// For n == 1 every lookup folds to 0 (the sole valid index).
inline std::size_t reflectIndex(std::ptrdiff_t idx, std::size_t n) noexcept
{
    if (n == 1) return 0;
    const std::ptrdiff_t period = 2 * static_cast<std::ptrdiff_t>(n);
    std::ptrdiff_t k = idx % period;
    if (k < 0) k += period;
    if (k >= static_cast<std::ptrdiff_t>(n)) {
        k = period - 1 - k;
    }
    return static_cast<std::size_t>(k);
}

} // namespace detail

// Apply a 2-D median filter with `mode='reflect'` to a row-major `T` matrix
// of shape (nRows, nCols). Output matrix must be separately allocated (no
// in-place). Both kernel dims must be odd.
template <typename T>
inline void medianFilter2DReflect(
    const T* input, std::size_t nRows, std::size_t nCols,
    std::size_t kRows, std::size_t kCols,
    T* output)
{
    assert((kRows % 2u) == 1u && "kRows must be odd");
    assert((kCols % 2u) == 1u && "kCols must be odd");
    assert(nRows >= 1 && nCols >= 1 && "empty input");

    const std::size_t kRhalf = (kRows - 1u) / 2u;
    const std::size_t kChalf = (kCols - 1u) / 2u;
    const std::size_t windowSize = kRows * kCols;
    const std::size_t medianRank = windowSize / 2u;

    // Reused buffer per output cell. Avoids per-cell allocation.
    std::vector<T> window(windowSize);

    for (std::size_t i = 0; i < nRows; ++i) {
        for (std::size_t j = 0; j < nCols; ++j) {
            std::size_t idx = 0;
            for (std::size_t kr = 0; kr < kRows; ++kr) {
                const std::ptrdiff_t ri =
                    static_cast<std::ptrdiff_t>(i) +
                    static_cast<std::ptrdiff_t>(kr) -
                    static_cast<std::ptrdiff_t>(kRhalf);
                const std::size_t riRef = detail::reflectIndex(ri, nRows);
                const std::size_t rowOff = riRef * nCols;
                for (std::size_t kc = 0; kc < kCols; ++kc) {
                    const std::ptrdiff_t cj =
                        static_cast<std::ptrdiff_t>(j) +
                        static_cast<std::ptrdiff_t>(kc) -
                        static_cast<std::ptrdiff_t>(kChalf);
                    const std::size_t cjRef = detail::reflectIndex(cj, nCols);
                    window[idx++] = input[rowOff + cjRef];
                }
            }
            std::nth_element(window.begin(),
                             window.begin() + static_cast<std::ptrdiff_t>(medianRank),
                             window.end());
            output[i * nCols + j] = window[medianRank];
        }
    }
}

} // namespace reamix::util
