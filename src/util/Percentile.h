#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace reamix::util {

// Percentile over a numeric span, matching numpy's default
// `np.percentile(a, q, method='linear')` semantics:
//
//   h      = (q / 100) * (n - 1)
//   i      = floor(h)
//   frac   = h - i
//   result = a_sorted[i] + frac * (a_sorted[i+1] - a_sorted[i])   if i+1 < n
//            a_sorted[i]                                           otherwise
//
// numpy promotes to float64 for the interpolation even when the input is
// float32 (default `out=None` → f64). We replicate that by casting the
// pivot values to double before the lerp, regardless of T.
//
// PARITY: numpy.lib.function_base._quantile_unchecked (method='linear') +
//         numpy 1.26.4. Used by phase-2b vocal pipeline at three sites:
//           - vocal_features.py:85  np.percentile(flatness, 95)         (f32)
//           - vocal_features.py:99  np.percentile(voice_band_ratio, 95) (f64)
//           - vocal_features.py:162 np.percentile(vocal_rise_frame, 95) (f64)
//           - vocal_features.py:165 np.percentile(vocal_fall_frame, 95) (f64)
//
// Bitwise target at the scalar level. Algorithm is pure sort + lerp; the
// lerp form `a[i] + frac * (a[i+1] - a[i])` matches numpy's source
// verbatim (not the alternative `(1-f)*a + f*b` which drifts by 1 ULP).
//
// q is in [0, 100] matching np.percentile (NOT [0, 1] like np.quantile).
// Empty input is undefined (caller must guard — numpy raises).
template <typename T>
inline double percentile(std::span<const T> values, double q)
{
    const std::size_t n = values.size();
    // Copy + sort. std::sort is introsort; deterministic for POD.
    std::vector<T> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());

    // PARITY: numpy 1.26 _quantile_unchecked with method='linear'.
    // q/100 in double; (n-1) in double; floor/frac in double.
    const double h = (q / 100.0) * static_cast<double>(n - 1);
    const double floorH = std::floor(h);
    const std::size_t i = static_cast<std::size_t>(floorH);
    const double frac = h - floorH;

    const double lo = static_cast<double>(sorted[i]);
    if (i + 1 < n) {
        const double hi = static_cast<double>(sorted[i + 1]);
        return lo + frac * (hi - lo);
    }
    return lo;
}

// Convenience overload for std::vector<T>.
template <typename T>
inline double percentile(const std::vector<T>& values, double q)
{
    return percentile(std::span<const T>(values.data(), values.size()), q);
}

} // namespace reamix::util
