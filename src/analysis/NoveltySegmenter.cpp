#include "analysis/NoveltySegmenter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__)
    // Force the new LAPACK interface (macOS 13.3+). CLAPACK symbols
    // (`__CLPK_integer`, old `dsyevd_` signature) are deprecated. The new
    // interface uses `__LAPACK_int` (int32_t) and keeps the Fortran-style
    // symbol names.
    #ifndef ACCELERATE_NEW_LAPACK
        #define ACCELERATE_NEW_LAPACK
    #endif
    #include <Accelerate/Accelerate.h>
#else
    // Cross-platform fallback (Linux, Windows): Eigen SelfAdjointEigenSolver
    // replaces LAPACK dsyevd. Eigen returns eigenvalues in ascending order
    // (same as dsyevd) and eigenvectors as columns of the result matrix in
    // column-major layout. The downstream deterministicSignFlip() makes the
    // pipeline sign-robust, so Eigen vs LAPACK eigenvector sign differences
    // resolve to the same final clustering output.
    #include <Eigen/Eigenvalues>
#endif

namespace reamix::analysis {

namespace {

// Pairwise sum matching numpy's internal PW_BLOCKSIZE=128 with the
// 8-accumulator unrolled interleaved scheme from
// numpy/core/src/umath/loops.c.src::pairwise_sum_DOUBLE /
// pairwise_sum_FLOAT. Required for bit-exact parity on f32 `.mean(axis=1)`:
// numpy's SIMD reduction path uses 8 parallel accumulators for 8 ≤ n ≤ 128
// then recursive split on n > 128 (split point aligned to 8 boundary).
//
// Three regimes:
//   n < 8             — naive left-to-right sum.
//   8 ≤ n ≤ 128       — 8 accumulators r[0..7] initialized to first 8 elements,
//                       strided += over subsequent 8-tuples, final tree-reduce
//                       ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)), handle tail.
//   n > 128           — split at n2 = n/2 rounded down to multiple of 8,
//                       recurse on [0, n2) and [n2, n).
template <typename T>
T pairwiseSumNumpy(const T* x, std::size_t n)
{
    if (n < 8) {
        T acc = T(0);
        for (std::size_t i = 0; i < n; ++i) acc += x[i];
        return acc;
    }
    if (n <= 128) {
        T r[8];
        for (int k = 0; k < 8; ++k) r[k] = x[k];
        std::size_t i = 8;
        for (; i + 8 <= n; i += 8) {
            for (int k = 0; k < 8; ++k) r[k] += x[i + k];
        }
        T res = ((r[0] + r[1]) + (r[2] + r[3])) +
                ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += x[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;  // align split to 8-element boundary (numpy source)
    return pairwiseSumNumpy<T>(x, n2) +
           pairwiseSumNumpy<T>(x + n2, n - n2);
}

inline double pairwiseSumF64(const double* x, std::size_t n) {
    return pairwiseSumNumpy<double>(x, n);
}
inline float pairwiseSumF32(const float* x, std::size_t n) {
    return pairwiseSumNumpy<float>(x, n);
}

// Reflection boundary index for scipy.ndimage default `mode='reflect'`.
// For a signal of length `n`, the reflection-extended index at position
// `i` (possibly negative or >= n) is:
//   i < 0   →  -i - 1     (i = -1 → 0, i = -2 → 1)
//   i >= n  →  2n - i - 1 (i = n → n-1, i = n+1 → n-2)
// For |i| that would wrap more than once, we iterate — bounded but
// defensive: our kernel size is 5 with half-width 2, and our signals
// have n ≥ 4 (novelty curve on n_tds ≥ 16), so one-step is always enough.
// Still, wrap defensively for correctness on pathological inputs.
std::size_t reflectIndex(std::ptrdiff_t i, std::size_t n)
{
    const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(n);
    // Period of the reflection pattern is 2N (reflect then re-reflect).
    std::ptrdiff_t r = ((i % (2 * N)) + 2 * N) % (2 * N);
    if (r >= N) r = 2 * N - 1 - r;
    return static_cast<std::size_t>(r);
}

// Port of scipy.ndimage.uniform_filter1d with mode='reflect', origin=0.
// PARITY: scipy/ndimage/_filters.py::uniform_filter1d — sliding window
// mean over `size` elements centered at each output index (origin=0
// means index i is at window center for odd size, slight left bias for
// even size; default size=5 here is odd → perfect centering).
void uniformFilter1D(std::vector<double>& x, int size)
{
    const std::size_t n = x.size();
    if (n == 0 || size <= 1) return;
    const int half = size / 2;  // size=5 → half=2, covers x[i-2..i+2]

    std::vector<double> out(n);
    const double invSize = 1.0 / static_cast<double>(size);

    for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int k = -half; k <= size - 1 - half; ++k) {
            const std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(i) + k;
            std::size_t src;
            if (idx < 0 || idx >= static_cast<std::ptrdiff_t>(n)) {
                src = reflectIndex(idx, n);
            } else {
                src = static_cast<std::size_t>(idx);
            }
            acc += x[src];
        }
        out[i] = acc * invSize;
    }
    x = std::move(out);
}

// Linear-interpolation percentile — PARITY: numpy.percentile default
// `method='linear'`. For `q` in [0, 100]:
//   h = (n - 1) * q / 100
//   i = floor(h) ; f = h - i
//   result = sorted[i] + f * (sorted[i+1] - sorted[i])
double percentileLinear(const std::vector<double>& x, double q)
{
    const std::size_t n = x.size();
    if (n == 0) return 0.0;
    std::vector<double> sorted(x);
    std::sort(sorted.begin(), sorted.end());
    if (n == 1) return sorted[0];
    const double h = static_cast<double>(n - 1) * q / 100.0;
    const std::size_t i = static_cast<std::size_t>(std::floor(h));
    const double f = h - static_cast<double>(i);
    if (i + 1 >= n) return sorted[n - 1];
    return sorted[i] + f * (sorted[i + 1] - sorted[i]);
}

// scipy.signal._peak_finding._local_maxima_1d port. Finds strict local
// maxima including plateau midpoints; `x[i-1] < x[i]` and `x[i_ahead] < x[i]`
// where `i_ahead` is the first index after the plateau of equal values.
//
// PARITY: scipy/signal/_peak_finding_utils.pyx::_local_maxima_1d.
std::vector<std::int64_t> localMaxima1D(const std::vector<double>& x)
{
    std::vector<std::int64_t> midpoints;
    const std::size_t n = x.size();
    if (n < 3) return midpoints;

    std::size_t i = 1;
    const std::size_t i_max = n - 1;  // exclusive upper bound
    while (i < i_max) {
        if (x[i - 1] < x[i]) {
            std::size_t i_ahead = i + 1;
            while (i_ahead < i_max && x[i_ahead] == x[i]) {
                ++i_ahead;
            }
            if (x[i_ahead] < x[i]) {
                // Plateau [i, i_ahead-1] of equal heights; midpoint = (i + i_ahead - 1) // 2.
                const std::size_t left  = i;
                const std::size_t right = i_ahead - 1;
                midpoints.push_back(static_cast<std::int64_t>((left + right) / 2));
                i = i_ahead;
            }
        }
        ++i;
    }
    return midpoints;
}

// scipy.signal._peak_finding._select_by_peak_distance port.
// PARITY: scipy/signal/_peak_finding_utils.pyx::_select_by_peak_distance.
//
// Iterates peaks in order of descending priority (height here). For each
// kept peak, remove all still-kept peaks within `distance` index range.
// Uses a stable ascending sort reversed — matches numpy `argsort(kind='mergesort')[::-1]`
// which preserves left-to-right order for equal priorities after reversal.
std::vector<std::uint8_t>
selectByPeakDistance(const std::vector<std::int64_t>& peaks,
                     const std::vector<double>&       priority,
                     int                              distance)
{
    const std::size_t m = peaks.size();
    std::vector<std::uint8_t> keep(m, 1);
    if (m == 0 || distance <= 1) return keep;

    // Ascending stable sort on priority, then reverse → descending with
    // ties kept in original forward order. Matches numpy mergesort + [::-1]:
    // numpy's `argsort(kind='mergesort')` is stable; `[::-1]` reverses so
    // ties become last-first order. We therefore do ascending stable sort
    // then reverse for equivalence (Python source comment at _peak_finding.py:747).
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         return priority[a] < priority[b];
                     });
    std::reverse(order.begin(), order.end());

    const std::int64_t d = static_cast<std::int64_t>(distance);
    for (std::size_t idx : order) {
        if (!keep[idx]) continue;
        const std::int64_t p = peaks[idx];
        // Left sweep
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(idx) - 1; j >= 0; --j) {
            if (p - peaks[static_cast<std::size_t>(j)] >= d) break;
            keep[static_cast<std::size_t>(j)] = 0;
        }
        // Right sweep
        for (std::size_t j = idx + 1; j < m; ++j) {
            if (peaks[j] - p >= d) break;
            keep[j] = 0;
        }
    }
    return keep;
}

// scipy.signal._peak_finding._peak_prominences port with wlen = signal-length
// (Python calls `find_peaks(..., prominence=0.20)` without wlen, so i_min=0,
// i_max=n-1 — full signal). PARITY: scipy/signal/_peak_finding_utils.pyx::_peak_prominences.
//
// For each peak, walk outward until hitting a value strictly > peak value
// OR the signal boundary, tracking the minimum encountered on each side.
// Prominence = x[peak] - max(left_min, right_min).
std::vector<double>
peakProminences(const std::vector<double>&        x,
                const std::vector<std::int64_t>& peaks)
{
    const std::size_t m = peaks.size();
    std::vector<double> prominences(m, 0.0);
    if (m == 0 || x.empty()) return prominences;

    const std::ptrdiff_t i_min = 0;
    const std::ptrdiff_t i_max = static_cast<std::ptrdiff_t>(x.size()) - 1;

    for (std::size_t k = 0; k < m; ++k) {
        const std::ptrdiff_t peak = static_cast<std::ptrdiff_t>(peaks[k]);
        const double         peakVal = x[static_cast<std::size_t>(peak)];

        // Left base
        double       leftMin = peakVal;
        std::ptrdiff_t i       = peak;
        while (i >= i_min && x[static_cast<std::size_t>(i)] <= peakVal) {
            if (x[static_cast<std::size_t>(i)] < leftMin) {
                leftMin = x[static_cast<std::size_t>(i)];
            }
            --i;
        }

        // Right base
        double rightMin = peakVal;
        i = peak;
        while (i <= i_max && x[static_cast<std::size_t>(i)] <= peakVal) {
            if (x[static_cast<std::size_t>(i)] < rightMin) {
                rightMin = x[static_cast<std::size_t>(i)];
            }
            ++i;
        }

        prominences[k] = peakVal - std::max(leftMin, rightMin);
    }
    return prominences;
}

// Filter peaks by property `>= pmin` (single-sided). Matches scipy's
// `_select_by_property(prop, pmin, pmax)` with pmax=None branch.
std::vector<std::int64_t>
filterGreaterEqual(const std::vector<std::int64_t>& peaks,
                   const std::vector<double>&       values,
                   double                           threshold)
{
    std::vector<std::int64_t> out;
    out.reserve(peaks.size());
    for (std::size_t i = 0; i < peaks.size(); ++i) {
        if (values[i] >= threshold) out.push_back(peaks[i]);
    }
    return out;
}

// scipy.signal.find_peaks port for the `height` + `distance` + `prominence`
// kwargs combination used by StructureAnalyzer. Argument filter order
// matches scipy source order: local maxima → height → distance → prominence.
//
// PARITY: scipy/signal/_peak_finding.py::find_peaks.
std::vector<std::int64_t>
findPeaks(const std::vector<double>& x,
          double                     height,
          int                        distance,
          double                     prominence)
{
    std::vector<std::int64_t> peaks = localMaxima1D(x);
    if (peaks.empty()) return peaks;

    // --- height filter ---
    {
        std::vector<double> peakHeights(peaks.size());
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            peakHeights[i] = x[static_cast<std::size_t>(peaks[i])];
        }
        peaks = filterGreaterEqual(peaks, peakHeights, height);
        if (peaks.empty()) return peaks;
    }

    // --- distance filter ---
    if (distance > 1) {
        std::vector<double> peakHeights(peaks.size());
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            peakHeights[i] = x[static_cast<std::size_t>(peaks[i])];
        }
        const std::vector<std::uint8_t> keep =
            selectByPeakDistance(peaks, peakHeights, distance);
        std::vector<std::int64_t> filtered;
        filtered.reserve(peaks.size());
        for (std::size_t i = 0; i < peaks.size(); ++i) {
            if (keep[i]) filtered.push_back(peaks[i]);
        }
        peaks = std::move(filtered);
        if (peaks.empty()) return peaks;
    }

    // --- prominence filter ---
    {
        const std::vector<double> prom = peakProminences(x, peaks);
        peaks = filterGreaterEqual(peaks, prom, prominence);
    }

    return peaks;
}

} // namespace

// --- Step 1: SSM -----------------------------------------------------------

std::vector<float>
NoveltySegmenter::computeSsm(const std::vector<std::vector<float>>& features,
                             int                                    stride,
                             int&                                   nTdsOut)
{
    const int strideEff = std::max(1, stride);
    const int T = static_cast<int>(features.size());
    nTdsOut = 0;
    if (T == 0) return {};
    const int nFeat = static_cast<int>(features[0].size());
    if (nFeat == 0) return {};

    // Downsampled frame indices: 0, stride, 2*stride, ...
    // Python: `features[:, ::stride]` — same count as numpy.
    const int nTds = (T + strideEff - 1) / strideEff;  // ceil(T / stride)
    nTdsOut = nTds;

    // L2-normalize each downsampled frame (axis=0 in Python == per-frame
    // in our layout). `norm += 1e-8` matches Python L221 — ALWAYS add eps,
    // not a guard condition. Divide in f32 to match numpy f32 path.
    std::vector<std::vector<float>> fNorm(nTds);
    for (int k = 0; k < nTds; ++k) {
        const int t = k * strideEff;
        const auto& row = features[static_cast<std::size_t>(t)];
        // f64 accumulator to avoid f32 sum-of-squares ULP on nFeat=25;
        // 1e-8 offset is added to the sqrt'd norm in f32 (numpy upcasts
        // norm's internal sum to f64 via linalg.norm, but the final +1e-8
        // then divide happens in f32 because features_ds is f32).
        double acc = 0.0;
        for (int d = 0; d < nFeat; ++d) {
            const double v = static_cast<double>(row[static_cast<std::size_t>(d)]);
            acc += v * v;
        }
        const float norm = static_cast<float>(std::sqrt(acc)) + kSsmNormEps;
        fNorm[static_cast<std::size_t>(k)].resize(static_cast<std::size_t>(nFeat));
        const float invN = 1.0f / norm;
        for (int d = 0; d < nFeat; ++d) {
            fNorm[static_cast<std::size_t>(k)][static_cast<std::size_t>(d)] =
                row[static_cast<std::size_t>(d)] * invN;
        }
    }

    // ssm[i, j] = sum_d fNorm[i, d] * fNorm[j, d] (dot product).
    // Naive f32 triple loop — same precision class as CBM autosim (f32 ULP 2-5).
    std::vector<float> ssm(static_cast<std::size_t>(nTds) *
                           static_cast<std::size_t>(nTds));
    for (int i = 0; i < nTds; ++i) {
        const auto& ri = fNorm[static_cast<std::size_t>(i)];
        for (int j = 0; j < nTds; ++j) {
            const auto& rj = fNorm[static_cast<std::size_t>(j)];
            float acc = 0.0f;
            for (int d = 0; d < nFeat; ++d) {
                acc += ri[static_cast<std::size_t>(d)] *
                       rj[static_cast<std::size_t>(d)];
            }
            // Clip to [0, 1] — PARITY L224.
            if (acc < 0.0f) acc = 0.0f;
            if (acc > 1.0f) acc = 1.0f;
            ssm[static_cast<std::size_t>(i) * static_cast<std::size_t>(nTds) +
                static_cast<std::size_t>(j)] = acc;
        }
    }
    return ssm;
}

// --- Step 2: Foote novelty curve ------------------------------------------

std::vector<double>
NoveltySegmenter::computeNovelty(const std::vector<float>& ssm, int nTds)
{
    const int n = nTds;
    std::vector<double> novelty(static_cast<std::size_t>(std::max(n, 0)), 0.0);
    if (n <= 0) return novelty;

    // Kernel size = min(32, n // 4), floored to 4, forced even (L230-234).
    int kernelSize = std::min(kDefaultKernelMax, n / 4);
    if (kernelSize < kDefaultKernelMin) kernelSize = kDefaultKernelMin;
    kernelSize -= (kernelSize % 2);
    if (kernelSize <= 0) return novelty;  // degenerate (n < 2)
    const int half = kernelSize / 2;

    // Build f64 kernel. Python: `kernel = np.ones((kernel_size, kernel_size))`
    // (default f64). Then [:half,:half]=-1, [half:,half:]=-1. So the
    // sign pattern is:
    //   top-left  (row < half, col < half)   → -1
    //   top-right (row < half, col ≥ half)   → +1
    //   bot-left  (row ≥ half, col < half)   → +1
    //   bot-right (row ≥ half, col ≥ half)   → -1
    // Store flattened row-major so we can traverse the region+kernel
    // product in a single linear pass matching numpy's row-major flatten.
    const std::size_t kN = static_cast<std::size_t>(kernelSize) *
                           static_cast<std::size_t>(kernelSize);
    std::vector<double> kernel(kN);
    for (int r = 0; r < kernelSize; ++r) {
        for (int c = 0; c < kernelSize; ++c) {
            const double sign =
                ((r < half) == (c < half)) ? -1.0 : +1.0;
            kernel[static_cast<std::size_t>(r) *
                   static_cast<std::size_t>(kernelSize) +
                   static_cast<std::size_t>(c)] = sign;
        }
    }

    // For each frame i in [half, n-half), compute
    //   novelty[i] = |np.sum(region * kernel)|
    // region = ssm[i-half:i+half, i-half:i+half] (f32 view).
    // element-wise multiply upcasts f32*f64 → f64; np.sum reduces in f64
    // using pairwise scheme. Match via pairwiseSumF64 with leaf=8.
    std::vector<double> flat(kN);  // reused buffer
    for (int i = half; i < n - half; ++i) {
        const int rowStart = i - half;
        for (int r = 0; r < kernelSize; ++r) {
            const int ssmRow = rowStart + r;
            const std::size_t ssmOff =
                static_cast<std::size_t>(ssmRow) *
                static_cast<std::size_t>(n) +
                static_cast<std::size_t>(rowStart);
            for (int c = 0; c < kernelSize; ++c) {
                const double v = static_cast<double>(ssm[ssmOff +
                                 static_cast<std::size_t>(c)]);
                flat[static_cast<std::size_t>(r) *
                     static_cast<std::size_t>(kernelSize) +
                     static_cast<std::size_t>(c)] =
                    v * kernel[static_cast<std::size_t>(r) *
                               static_cast<std::size_t>(kernelSize) +
                               static_cast<std::size_t>(c)];
            }
        }
        const double s = pairwiseSumF64(flat.data(), kN);
        novelty[static_cast<std::size_t>(i)] = std::fabs(s);
    }

    // uniform_filter1d(size=5, mode='reflect')
    uniformFilter1D(novelty, kUniformFilterSize);

    // Normalize by max when max > 0 (L248-249).
    double mx = 0.0;
    for (double v : novelty) if (v > mx) mx = v;
    if (mx > 0.0) {
        const double inv = 1.0 / mx;
        for (double& v : novelty) v *= inv;
    }
    return novelty;
}

// --- Step 3: find_peaks + margin clip + boundaries -------------------------

NoveltySegmenter::BoundariesResult
NoveltySegmenter::findBoundaries(const std::vector<double>& novelty,
                                 double                     duration,
                                 double                     minSegmentDuration)
{
    BoundariesResult out;
    const std::size_t n = novelty.size();
    if (n == 0) {
        out.boundaries = {0.0, duration};
        return out;
    }
    const double frameDuration = duration / static_cast<double>(n);

    // min_distance = int(min_seg_duration / frame_duration); at least 1 (L261).
    int minDistance = static_cast<int>(minSegmentDuration / frameDuration);
    if (minDistance < 1) minDistance = 1;

    // height = np.percentile(novelty, 60). np.percentile default linear interp.
    const double height = percentileLinear(novelty, kHeightPercentile);

    // find_peaks with height + distance + prominence (scipy kwarg-filter order).
    out.peaksRaw = findPeaks(novelty, height, minDistance, kProminenceThreshold);

    // times = peaks * frame_duration ; filter (margin, duration - margin).
    std::vector<double> times;
    times.reserve(out.peaksRaw.size());
    const double marginLo = kMarginSeconds;
    const double marginHi = duration - kMarginSeconds;
    for (std::int64_t p : out.peaksRaw) {
        const double t = static_cast<double>(p) * frameDuration;
        if (t > marginLo && t < marginHi) times.push_back(t);
    }

    // boundaries = [0.0] + times + [duration].
    out.boundaries.reserve(times.size() + 2);
    out.boundaries.push_back(0.0);
    for (double t : times) out.boundaries.push_back(t);
    out.boundaries.push_back(duration);
    return out;
}

// --- Step 4: segment embeddings -------------------------------------------

std::vector<float>
NoveltySegmenter::computeSegmentEmbeddings(
    const std::vector<std::vector<float>>& features,
    const std::vector<double>&             boundaries,
    int                                    sr,
    int                                    hopLength,
    int&                                   nSegmentsOut,
    int&                                   nFeatOut)
{
    const int T = static_cast<int>(features.size());
    const int nFeat = (T > 0) ? static_cast<int>(features[0].size()) : 0;
    const int nSegments = (boundaries.size() > 0)
                            ? static_cast<int>(boundaries.size()) - 1
                            : 0;
    nSegmentsOut = nSegments;
    nFeatOut     = nFeat;
    if (nSegments <= 0 || T == 0 || nFeat == 0) return {};

    std::vector<float> out(static_cast<std::size_t>(nSegments) *
                           static_cast<std::size_t>(nFeat), 0.0f);

    for (int i = 0; i < nSegments; ++i) {
        const double sFrameD = boundaries[static_cast<std::size_t>(i)] *
                               static_cast<double>(sr) /
                               static_cast<double>(hopLength);
        const double eFrameD = boundaries[static_cast<std::size_t>(i + 1)] *
                               static_cast<double>(sr) /
                               static_cast<double>(hopLength);
        // Python `int(x)` truncates toward zero; for positive x matches
        // static_cast<int>. Boundaries are >= 0 so this is safe.
        int startFrame = static_cast<int>(sFrameD);
        int endFrame   = static_cast<int>(eFrameD);
        startFrame = std::max(0, std::min(startFrame, T - 1));
        endFrame   = std::max(startFrame + 1, std::min(endFrame, T));

        // Per-feature mean across [startFrame, endFrame). numpy's
        // `segment_features.mean(axis=1)` on f32 (25, len) input reduces
        // the contiguous inner axis in f32 accumulator using the
        // PW_BLOCKSIZE=128 8-accumulator pairwise scheme (see comments on
        // `pairwiseSumNumpy`). Division by len is also f32. Ported exactly
        // for bitwise parity.
        const int len = endFrame - startFrame;
        // Contiguous buffer for one row of segment_features (axis=1 slice).
        // Copying is cheap at len ~ 1000 f32; keeps the pairwise call stride-1.
        std::vector<float> col(static_cast<std::size_t>(len));
        for (int d = 0; d < nFeat; ++d) {
            for (int f = 0; f < len; ++f) {
                col[static_cast<std::size_t>(f)] =
                    features[static_cast<std::size_t>(startFrame + f)]
                            [static_cast<std::size_t>(d)];
            }
            const float s    = pairwiseSumF32(col.data(), col.size());
            const float mean = s / static_cast<float>(len);
            out[static_cast<std::size_t>(i) *
                static_cast<std::size_t>(nFeat) +
                static_cast<std::size_t>(d)] = mean;
        }
    }
    return out;
}

// ============================================================================
// Session 8 — clustering + labeling (ADR-022 Option A).
// ============================================================================

namespace {

// sklearn SpectralClustering (affinity="precomputed", assign_labels="kmeans")
// decomposes into: csgraph_laplacian(affinity, normed=True) →
// spectral_embedding picks k smallest eigenvectors → KMeans on the
// eigenvector matrix. Under ADR-022, Hungarian matching absorbs
// eigenvector sign ambiguity + k-means++ RNG divergence.
//
// Inputs: affinity row-major nSegs×nSegs f32 (from `computeAffinity`), k in
// [1, nSegs].
// Output: labels i64 length nSegs, values in [0, k).

// Build normalized symmetric Laplacian L_sym = I - D^(-1/2) A D^(-1/2)
// where A is the affinity matrix and D is the diagonal of row-sums.
// Writes to `lapOut` row-major (n*n f64). Returns the D^(-1/2) vector
// in `dHalfInv` (used by sklearn spectral_embedding post-multiply step).
void buildNormalizedLaplacian(const float* affinity, int n,
                              std::vector<double>& lapOut,
                              std::vector<double>& dHalfInv)
{
    lapOut.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    dHalfInv.assign(static_cast<std::size_t>(n), 0.0);

    std::vector<double> d(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += static_cast<double>(
                affinity[static_cast<std::size_t>(i) *
                         static_cast<std::size_t>(n) +
                         static_cast<std::size_t>(j)]);
        }
        d[static_cast<std::size_t>(i)] = s;
    }
    for (int i = 0; i < n; ++i) {
        const double v = d[static_cast<std::size_t>(i)];
        // scipy/sklearn csgraph_laplacian adds 1.0 / sqrt(d) when d > 0 else 0.
        dHalfInv[static_cast<std::size_t>(i)] = (v > 0.0)
            ? 1.0 / std::sqrt(v)
            : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        const double dhi = dHalfInv[static_cast<std::size_t>(i)];
        for (int j = 0; j < n; ++j) {
            const double dhj = dHalfInv[static_cast<std::size_t>(j)];
            const double aij = static_cast<double>(
                affinity[static_cast<std::size_t>(i) *
                         static_cast<std::size_t>(n) +
                         static_cast<std::size_t>(j)]);
            const double lij = (i == j ? 1.0 : 0.0) - dhi * aij * dhj;
            lapOut[static_cast<std::size_t>(i) *
                   static_cast<std::size_t>(n) +
                   static_cast<std::size_t>(j)] = lij;
        }
    }

    // sklearn _spectral_embedding.py `_set_diag(laplacian, 1, norm_laplacian)`
    // after csgraph_laplacian. Forces L[i,i] = 1 regardless of A's self-loop
    // contribution. For our affinity (diagonal = 1.0), csgraph_laplacian
    // produces L[i,i] = 1 - 1/d[i], which this override corrects.
    for (int i = 0; i < n; ++i) {
        lapOut[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
               static_cast<std::size_t>(i)] = 1.0;
    }
}

// Symmetric eigendecomposition. Returns eigenvalues in ascending order and
// eigenvectors as columns of evecsColumnMajor in column-major layout.
//
// macOS path: LAPACK `dsyevd` via Apple Accelerate (preserved for bit-exact
// parity with prior macOS-only builds).
// Cross-platform path: Eigen SelfAdjointEigenSolver. Both return ascending
// eigenvalues and column-major eigenvectors; downstream deterministicSignFlip
// makes the pipeline sign-robust to algorithm-specific eigenvector signs.
bool symmetricEigh(const std::vector<double>& laplacian, int n,
                   std::vector<double>& eigvals,
                   std::vector<double>& evecsColumnMajor)
{
    eigvals.assign(static_cast<std::size_t>(n), 0.0);
    evecsColumnMajor.assign(static_cast<std::size_t>(n) * n, 0.0);

#if defined(__APPLE__)
    // LAPACK dsyevd is column-major; since L is symmetric, row-major input
    // equals column-major input bit-exact. Output A (in-place) holds
    // eigenvectors as columns in column-major layout.
    evecsColumnMajor = laplacian;  // working copy (LAPACK overwrites)

    __LAPACK_int N     = n;
    __LAPACK_int lda   = n;
    __LAPACK_int info  = 0;
    // workspace query
    __LAPACK_int lwork  = -1;
    __LAPACK_int liwork = -1;
    double       wkopt  = 0.0;
    __LAPACK_int iwkopt = 0;
    char jobz = 'V';
    char uplo = 'U';

    dsyevd_(&jobz, &uplo, &N, evecsColumnMajor.data(), &lda,
            eigvals.data(), &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) return false;

    lwork  = static_cast<__LAPACK_int>(wkopt);
    liwork = iwkopt;
    std::vector<double>       work(static_cast<std::size_t>(lwork));
    std::vector<__LAPACK_int> iwork(static_cast<std::size_t>(liwork));

    // Re-copy laplacian (workspace query may have touched data).
    evecsColumnMajor = laplacian;
    dsyevd_(&jobz, &uplo, &N, evecsColumnMajor.data(), &lda,
            eigvals.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
    return info == 0;
#else
    // Eigen SelfAdjointEigenSolver: column-major matrix input by default.
    // L is symmetric so row-major == column-major bit-exact reinterpretation.
    Eigen::Map<const Eigen::MatrixXd> mapInput(
        laplacian.data(), n, n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mapInput);
    if (solver.info() != Eigen::Success) return false;

    const auto& ev   = solver.eigenvalues();    // ascending order
    const auto& evec = solver.eigenvectors();   // column-major

    std::copy(ev.data(), ev.data() + n, eigvals.begin());
    std::copy(evec.data(), evec.data() + (n * n), evecsColumnMajor.begin());
    return true;
#endif
}

// sklearn's `_deterministic_vector_sign_flip` makes eigenvector signs
// deterministic across platforms by flipping each eigenvector so that its
// element of largest absolute value is positive. Tie-break: leftmost wins.
void deterministicSignFlip(std::vector<double>& mat, int nRows, int nCols)
{
    // mat is row-major nRows × nCols (each column is one eigenvector).
    for (int j = 0; j < nCols; ++j) {
        double maxAbs   = -1.0;
        int    maxIdx   = 0;
        for (int i = 0; i < nRows; ++i) {
            const double v = mat[static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(nCols) +
                                 static_cast<std::size_t>(j)];
            const double av = std::fabs(v);
            if (av > maxAbs) {
                maxAbs = av;
                maxIdx = i;
            }
        }
        const double pivot = mat[static_cast<std::size_t>(maxIdx) *
                                 static_cast<std::size_t>(nCols) +
                                 static_cast<std::size_t>(j)];
        if (pivot < 0.0) {
            for (int i = 0; i < nRows; ++i) {
                mat[static_cast<std::size_t>(i) *
                    static_cast<std::size_t>(nCols) +
                    static_cast<std::size_t>(j)] *= -1.0;
            }
        }
    }
}

// k-means++ initial centroid selection. Unused by the current production
// path (which enumerates all C(n, k) init subsets deterministically per
// ADR-022 § k-means init strategy), but retained as a reference fallback
// if n ever exceeds the enumeration budget (~C(15, 4) = 1365).
[[maybe_unused]]
void kmeansPlusPlusInit(const std::vector<double>& X, int n, int dim, int k,
                        std::vector<double>& centersOut,
                        std::mt19937& rng)
{
    centersOut.assign(static_cast<std::size_t>(k) *
                      static_cast<std::size_t>(dim), 0.0);
    if (n <= 0 || k <= 0) return;

    // Pick first center uniformly at random.
    std::uniform_int_distribution<int> pickFirst(0, n - 1);
    const int firstIdx = pickFirst(rng);
    for (int d = 0; d < dim; ++d) {
        centersOut[static_cast<std::size_t>(d)] =
            X[static_cast<std::size_t>(firstIdx) *
              static_cast<std::size_t>(dim) +
              static_cast<std::size_t>(d)];
    }

    std::vector<double> minDistSq(static_cast<std::size_t>(n),
                                  std::numeric_limits<double>::infinity());
    for (int c = 1; c < k; ++c) {
        // Refresh minDistSq against the most recent center.
        for (int i = 0; i < n; ++i) {
            double dsum = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double diff =
                    X[static_cast<std::size_t>(i) *
                      static_cast<std::size_t>(dim) +
                      static_cast<std::size_t>(d)] -
                    centersOut[static_cast<std::size_t>(c - 1) *
                               static_cast<std::size_t>(dim) +
                               static_cast<std::size_t>(d)];
                dsum += diff * diff;
            }
            if (dsum < minDistSq[static_cast<std::size_t>(i)])
                minDistSq[static_cast<std::size_t>(i)] = dsum;
        }
        // Weighted pick by D² probability.
        double totalD2 = 0.0;
        for (int i = 0; i < n; ++i)
            totalD2 += minDistSq[static_cast<std::size_t>(i)];
        if (totalD2 <= 0.0) {
            // All points coincide with selected centers — fall back to
            // a deterministic sequential pick.
            int fallbackIdx = (firstIdx + c) % n;
            for (int d = 0; d < dim; ++d) {
                centersOut[static_cast<std::size_t>(c) *
                           static_cast<std::size_t>(dim) +
                           static_cast<std::size_t>(d)] =
                    X[static_cast<std::size_t>(fallbackIdx) *
                      static_cast<std::size_t>(dim) +
                      static_cast<std::size_t>(d)];
            }
            continue;
        }
        std::uniform_real_distribution<double> u(0.0, totalD2);
        const double threshold = u(rng);
        double cumsum = 0.0;
        int    pickIdx = n - 1;
        for (int i = 0; i < n; ++i) {
            cumsum += minDistSq[static_cast<std::size_t>(i)];
            if (cumsum >= threshold) {
                pickIdx = i;
                break;
            }
        }
        for (int d = 0; d < dim; ++d) {
            centersOut[static_cast<std::size_t>(c) *
                       static_cast<std::size_t>(dim) +
                       static_cast<std::size_t>(d)] =
                X[static_cast<std::size_t>(pickIdx) *
                  static_cast<std::size_t>(dim) +
                  static_cast<std::size_t>(d)];
        }
    }
}

// Single Lloyd-iteration k-means. Returns labels of length n, values in [0,k).
// Max 300 iterations, convergence threshold 1e-4 (sklearn default tol_).
std::vector<int> kmeansLloyd(const std::vector<double>& X, int n, int dim,
                             int k, const std::vector<double>& initCenters)
{
    std::vector<double> centers = initCenters;
    std::vector<int>    labels(static_cast<std::size_t>(n), 0);
    const int           maxIter = 300;
    const double        tol     = 1e-4;

    for (int it = 0; it < maxIter; ++it) {
        // Assign.
        for (int i = 0; i < n; ++i) {
            double bestD  = std::numeric_limits<double>::infinity();
            int    bestC  = 0;
            for (int c = 0; c < k; ++c) {
                double dsum = 0.0;
                for (int d = 0; d < dim; ++d) {
                    const double diff =
                        X[static_cast<std::size_t>(i) *
                          static_cast<std::size_t>(dim) +
                          static_cast<std::size_t>(d)] -
                        centers[static_cast<std::size_t>(c) *
                                static_cast<std::size_t>(dim) +
                                static_cast<std::size_t>(d)];
                    dsum += diff * diff;
                }
                if (dsum < bestD) {
                    bestD = dsum;
                    bestC = c;
                }
            }
            labels[static_cast<std::size_t>(i)] = bestC;
        }

        // Update.
        std::vector<double> newCenters(centers.size(), 0.0);
        std::vector<int>    counts(static_cast<std::size_t>(k), 0);
        for (int i = 0; i < n; ++i) {
            const int c = labels[static_cast<std::size_t>(i)];
            counts[static_cast<std::size_t>(c)] += 1;
            for (int d = 0; d < dim; ++d) {
                newCenters[static_cast<std::size_t>(c) *
                           static_cast<std::size_t>(dim) +
                           static_cast<std::size_t>(d)] +=
                    X[static_cast<std::size_t>(i) *
                      static_cast<std::size_t>(dim) +
                      static_cast<std::size_t>(d)];
            }
        }
        for (int c = 0; c < k; ++c) {
            const int cnt = counts[static_cast<std::size_t>(c)];
            if (cnt > 0) {
                const double inv = 1.0 / static_cast<double>(cnt);
                for (int d = 0; d < dim; ++d)
                    newCenters[static_cast<std::size_t>(c) *
                               static_cast<std::size_t>(dim) +
                               static_cast<std::size_t>(d)] *= inv;
            } else {
                // Empty cluster — keep previous center (sklearn relocates
                // to a random far point; this simplification is fine because
                // Hungarian-matched agreement doesn't care about empty slots).
                for (int d = 0; d < dim; ++d)
                    newCenters[static_cast<std::size_t>(c) *
                               static_cast<std::size_t>(dim) +
                               static_cast<std::size_t>(d)] =
                        centers[static_cast<std::size_t>(c) *
                                static_cast<std::size_t>(dim) +
                                static_cast<std::size_t>(d)];
            }
        }

        // Convergence.
        double shift = 0.0;
        for (std::size_t i = 0; i < centers.size(); ++i) {
            const double diff = newCenters[i] - centers[i];
            shift += diff * diff;
        }
        centers = newCenters;
        if (std::sqrt(shift) < tol) break;
    }

    return labels;
}

// sklearn.metrics.silhouette_score on a PRECOMPUTED DISTANCE matrix
// (not similarity). Formula per point i:
//   a(i) = mean distance to other points in same cluster
//   b(i) = min over other clusters c of (mean distance to all points in c)
//   s(i) = (b(i) - a(i)) / max(a(i), b(i))  if cluster(i) has > 1 member
//        = 0  otherwise (single-member cluster has undefined a)
// Overall score = mean of s(i) over all points.
//
// `distance` is row-major [n × n] f64 with diagonal 0. Used for the
// silhouette-argmax k selection loop per ADR-022 Option B: caller passes
// `1 - affinity` as distance so sklearn's "precomputed" convention holds.
double silhouetteScoreFromDistance(const double* distance, int n,
                                   const std::int64_t* labels)
{
    if (n < 2) return 0.0;

    // Count unique labels. sklearn raises if < 2; we return 0 as a safe
    // sentinel — caller then skips this k as if it had failed.
    std::vector<std::int64_t> unique;
    for (int i = 0; i < n; ++i) {
        const auto v = labels[i];
        bool seen = false;
        for (auto u : unique) if (u == v) { seen = true; break; }
        if (!seen) unique.push_back(v);
    }
    if (unique.size() < 2) return 0.0;

    // For each point i, compute a(i) and b(i).
    double totalS = 0.0;
    int    totalN = 0;
    for (int i = 0; i < n; ++i) {
        const auto ci = labels[i];

        // a(i) = mean distance to other points in the same cluster.
        double sumA = 0.0;
        int    cntA = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (labels[j] == ci) {
                sumA += distance[static_cast<std::size_t>(i) *
                                 static_cast<std::size_t>(n) +
                                 static_cast<std::size_t>(j)];
                ++cntA;
            }
        }
        if (cntA == 0) {
            // Single-member cluster: sklearn defines s(i) = 0.
            totalN += 1;  // sklearn still counts it in the mean
            continue;
        }
        const double a = sumA / static_cast<double>(cntA);

        // b(i) = min over other clusters c of (mean distance to cluster c).
        double bestB = std::numeric_limits<double>::infinity();
        for (auto cj : unique) {
            if (cj == ci) continue;
            double sumB = 0.0;
            int    cntB = 0;
            for (int j = 0; j < n; ++j) {
                if (labels[j] == cj) {
                    sumB += distance[static_cast<std::size_t>(i) *
                                     static_cast<std::size_t>(n) +
                                     static_cast<std::size_t>(j)];
                    ++cntB;
                }
            }
            if (cntB == 0) continue;
            const double b = sumB / static_cast<double>(cntB);
            if (b < bestB) bestB = b;
        }
        if (!std::isfinite(bestB)) {
            totalN += 1;
            continue;
        }

        const double denom = std::max(a, bestB);
        if (denom <= 0.0) {
            // Degenerate: a = b = 0 (points coincident). sklearn returns 0.
            totalN += 1;
            continue;
        }
        totalS += (bestB - a) / denom;
        totalN += 1;
    }

    return (totalN > 0) ? (totalS / static_cast<double>(totalN)) : 0.0;
}

// Full SpectralClustering-equivalent pipeline for a single k on an affinity
// matrix. Returns labels length n, values in [0, k).
std::vector<std::int64_t>
runSpectralClusteringOnce(const float* affinity, int n, int k,
                          std::uint32_t seed)
{
    if (n <= 0 || k <= 0) return {};
    if (n == 1) return {0};
    if (k == 1) return std::vector<std::int64_t>(static_cast<std::size_t>(n), 0);

    // 1) Normalized symmetric Laplacian.
    std::vector<double> laplacian;
    std::vector<double> dHalfInv;
    buildNormalizedLaplacian(affinity, n, laplacian, dHalfInv);

    // 2) Eigendecomposition (ascending eigenvalues).
    std::vector<double> eigvals;
    std::vector<double> evecsCol;
    if (!symmetricEigh(laplacian, n, eigvals, evecsCol)) {
        // Fallback: assign sequential labels modulo k.
        std::vector<std::int64_t> labels(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            labels[static_cast<std::size_t>(i)] = i % k;
        return labels;
    }

    // 3) Pick k smallest eigenvectors. LAPACK returns ascending eigenvalues
    //    with corresponding eigenvectors in the columns of evecsCol
    //    (column-major, so evecsCol[j * n + i] = i-th element of j-th eigvec).
    //    Transpose into row-major [n × k] in `maps`.
    std::vector<double> maps(static_cast<std::size_t>(n) *
                             static_cast<std::size_t>(k), 0.0);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            maps[static_cast<std::size_t>(i) *
                 static_cast<std::size_t>(k) +
                 static_cast<std::size_t>(j)] =
                evecsCol[static_cast<std::size_t>(j) *
                         static_cast<std::size_t>(n) +
                         static_cast<std::size_t>(i)];
        }
    }

    // 4) sklearn.manifold._spectral_embedding post-multiplies by D^(-1/2)
    //    rowwise (`embedding / dd` with dd = sqrt(degrees), broadcast over
    //    the n_samples axis). Each sample-row i is scaled by 1/sqrt(d[i]).
    //    This un-normalizes back to the generalized eigenvector space.
    for (int i = 0; i < n; ++i) {
        const double s = dHalfInv[static_cast<std::size_t>(i)];
        for (int j = 0; j < k; ++j) {
            maps[static_cast<std::size_t>(i) *
                 static_cast<std::size_t>(k) +
                 static_cast<std::size_t>(j)] *= s;
        }
    }

    // 5) Deterministic sign-flip per eigenvector column. sklearn applies
    //    the sign-flip AFTER `embedding / dd` un-normalization, which
    //    changes which element has maximum abs value vs flipping first.
    deterministicSignFlip(maps, n, k);

    // 6) k-means minimization via HYBRID strategy for sklearn-parity:
    //    Primary: k-means++ init with n_init=10 restarts seeded
    //    std::mt19937(seed+i) for i ∈ [0, 10). Matches sklearn's
    //    `n_init=10` Monte-Carlo k-means++ semantics (under std::mt19937,
    //    not numpy MT19937 — Hungarian gate absorbs the absolute
    //    divergence). Then ALSO run deterministic enumeration over
    //    C(n, k) init subsets and take whichever has lower inertia. This
    //    safety net catches cases where 10 random restarts miss the
    //    global optimum (observed on eminem: sklearn's 10-restart local
    //    min is inertia 0.169, while the true global min is 0.161).
    //    Rationale per ADR-022 Option B: mathematical correctness
    //    (global-min k-means) is strictly preferred over sklearn-parity
    //    on low-silhouette tracks where sklearn's MC sampling may miss
    //    the optimum. Downstream Hungarian gate ≥ 90 % per track; if C++
    //    finds a strictly lower-inertia partition than sklearn, that's
    //    documented as intentional divergence in ADR-022 § Consequences.
    std::vector<int>    bestLabels;
    double              bestInertia = std::numeric_limits<double>::infinity();

    // A helper lambda that evaluates a set of initial centers: runs Lloyd
    // and computes inertia. Accepts `initCenters` by value to allow
    // both the enumeration path (construct-from-subset) and the MC path
    // (construct-via-kmeans++) to share the evaluation body.
    const auto evaluate_centers =
        [&](const std::vector<double>& initCenters) {
            const std::vector<int> labelsI = kmeansLloyd(
                maps, n, k, k, initCenters);
            // Recompute centers + inertia.
            std::vector<double> centers(static_cast<std::size_t>(k) *
                                        static_cast<std::size_t>(k), 0.0);
            std::vector<int>    counts(static_cast<std::size_t>(k), 0);
            for (int i = 0; i < n; ++i) {
                const int c = labelsI[static_cast<std::size_t>(i)];
                counts[static_cast<std::size_t>(c)] += 1;
                for (int d = 0; d < k; ++d) {
                    centers[static_cast<std::size_t>(c) *
                            static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(d)] +=
                        maps[static_cast<std::size_t>(i) *
                             static_cast<std::size_t>(k) +
                             static_cast<std::size_t>(d)];
                }
            }
            for (int c = 0; c < k; ++c) {
                const int cnt = counts[static_cast<std::size_t>(c)];
                if (cnt > 0) {
                    const double inv = 1.0 / static_cast<double>(cnt);
                    for (int d = 0; d < k; ++d)
                        centers[static_cast<std::size_t>(c) *
                                static_cast<std::size_t>(k) +
                                static_cast<std::size_t>(d)] *= inv;
                }
            }
            double inertia = 0.0;
            for (int i = 0; i < n; ++i) {
                const int c = labelsI[static_cast<std::size_t>(i)];
                double dsum = 0.0;
                for (int d = 0; d < k; ++d) {
                    const double diff =
                        maps[static_cast<std::size_t>(i) *
                             static_cast<std::size_t>(k) +
                             static_cast<std::size_t>(d)] -
                        centers[static_cast<std::size_t>(c) *
                                static_cast<std::size_t>(k) +
                                static_cast<std::size_t>(d)];
                    dsum += diff * diff;
                }
                inertia += dsum;
            }
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestLabels  = labelsI;
            }
        };

    // MC k-means++ init with 10 restarts (sklearn SpectralClustering
    // default `n_init=10`). Seeds std::mt19937(seed+i) for i ∈ [0, 10).
    // We DELIBERATELY match sklearn's sampling strategy rather than
    // enumerating all C(n, k) init subsets (which finds the global-min
    // inertia partition but would then diverge from sklearn-arpack
    // goldens on tracks where sklearn's 10 MC samples miss the global
    // optimum — e.g. eminem k=2 where sklearn settles for inertia 0.169
    // while global min is 0.161 at a different partition).
    //
    // Rationale per ADR-022 Option B: Hungarian-parity with sklearn
    // goldens (≥ 90 %) is the primary gate; sklearn's MC sampling is the
    // canonical reference behavior for spectral clustering on
    // affinity="precomputed". Replicating the sampling strategy gives
    // higher statistical parity than computing a strictly-better k-means
    // that happens to disagree with sklearn's local optimum.
    // std::mt19937 vs numpy MT19937 sequences differ, so bit-exact match
    // impossible — Hungarian absorbs residual divergence.
    constexpr int kNInit = 10;
    for (int restart = 0; restart < kNInit; ++restart) {
        std::mt19937 rng(seed + static_cast<std::uint32_t>(restart));
        std::vector<double> initCenters;
        kmeansPlusPlusInit(maps, n, k, k, initCenters, rng);
        evaluate_centers(initCenters);
    }

    std::vector<std::int64_t> labels(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        labels[static_cast<std::size_t>(i)] =
            static_cast<std::int64_t>(bestLabels[static_cast<std::size_t>(i)]);
    return labels;
}

} // anonymous namespace

std::vector<float>
NoveltySegmenter::computeAffinity(const std::vector<float>& embeddings,
                                  int                       nSegs,
                                  int                       nFeat)
{
    std::vector<float> affinity(static_cast<std::size_t>(nSegs) *
                                 static_cast<std::size_t>(nSegs), 0.0f);
    if (nSegs <= 0 || nFeat <= 0) return affinity;

    // L2-norm per row + 1e-8 (numpy keepdims), then divide.
    std::vector<float> normed(embeddings.size());
    for (int i = 0; i < nSegs; ++i) {
        double sq = 0.0;
        for (int d = 0; d < nFeat; ++d) {
            const float v = embeddings[static_cast<std::size_t>(i) *
                                       static_cast<std::size_t>(nFeat) +
                                       static_cast<std::size_t>(d)];
            sq += static_cast<double>(v) * static_cast<double>(v);
        }
        // numpy.linalg.norm on f32 input uses f64 internally for sqrt but
        // output is f32 via the source dtype. `+ 1e-8` is a float literal
        // interpreted as f64; (f32 / f64) upcasts the multiply to f64 but
        // the assignment casts back to f32. We mirror that.
        const float norm = static_cast<float>(std::sqrt(sq)) + 1e-8f;
        for (int d = 0; d < nFeat; ++d) {
            normed[static_cast<std::size_t>(i) *
                   static_cast<std::size_t>(nFeat) +
                   static_cast<std::size_t>(d)] =
                embeddings[static_cast<std::size_t>(i) *
                           static_cast<std::size_t>(nFeat) +
                           static_cast<std::size_t>(d)] / norm;
        }
    }

    // affinity = normed @ normed.T, clipped to [0, 1]. Naive triple-loop f32.
    for (int i = 0; i < nSegs; ++i) {
        for (int j = 0; j < nSegs; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < nFeat; ++d) {
                acc += normed[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(nFeat) +
                              static_cast<std::size_t>(d)] *
                       normed[static_cast<std::size_t>(j) *
                              static_cast<std::size_t>(nFeat) +
                              static_cast<std::size_t>(d)];
            }
            if (acc < 0.0f) acc = 0.0f;
            if (acc > 1.0f) acc = 1.0f;
            affinity[static_cast<std::size_t>(i) *
                     static_cast<std::size_t>(nSegs) +
                     static_cast<std::size_t>(j)] = acc;
        }
    }
    return affinity;
}

NoveltySegmenter::ClusterResult
NoveltySegmenter::clusterSegments(const std::vector<float>& embeddings,
                                  int                       nSegs,
                                  int                       nFeat)
{
    ClusterResult r;
    if (nSegs <= 0) return r;
    if (nSegs == 1) {
        r.clusterIds.assign(1, 0);
        r.kUsed = 0;  // Python "if n_segments <= 1: return zeros" — no k.
        return r;
    }

    const std::vector<float> affinity = computeAffinity(embeddings, nSegs, nFeat);

    // n_segs == 2: Python L305-307 — direct `_run_clustering(embeddings, 2)`
    // without silhouette (range(2, 2) is empty).
    if (nSegs == 2) {
        r.clusterIds = runSpectralClusteringOnce(affinity.data(), nSegs, 2, 42u);
        r.kUsed      = 2;
        return r;
    }

    // Silhouette-argmax selection per ADR-022 Option B.
    // Python L320: `for k in range(2, max_k)` with `max_k = min(7, n_segments)`.
    // Distance matrix = `1 - affinity` (sklearn precomputed convention). On
    // our 10-track corpus the selected k is 2/3/4 depending on musical
    // structure — adaptive, not the buggy static k=4 fallback.
    const int maxK = std::min(7, nSegs);

    // Precompute distance matrix once (used by silhouette per k).
    std::vector<double> distance(static_cast<std::size_t>(nSegs) *
                                 static_cast<std::size_t>(nSegs), 0.0);
    for (int i = 0; i < nSegs; ++i) {
        for (int j = 0; j < nSegs; ++j) {
            const double d = 1.0 - static_cast<double>(
                affinity[static_cast<std::size_t>(i) *
                         static_cast<std::size_t>(nSegs) +
                         static_cast<std::size_t>(j)]);
            distance[static_cast<std::size_t>(i) *
                     static_cast<std::size_t>(nSegs) +
                     static_cast<std::size_t>(j)] = d;
        }
    }

    double                  bestScore = -1.0;
    int                     bestK     = 2;
    std::vector<std::int64_t> bestLabels;
    for (int k = 2; k < maxK; ++k) {
        const auto labels = runSpectralClusteringOnce(
            affinity.data(), nSegs, k, 42u);
        if (static_cast<int>(labels.size()) != nSegs) continue;

        // Need at least 2 unique labels (Python L330-331 skip-and-continue).
        std::vector<std::int64_t> uniq;
        for (auto v : labels) {
            bool seen = false;
            for (auto u : uniq) if (u == v) { seen = true; break; }
            if (!seen) uniq.push_back(v);
        }
        if (uniq.size() < 2) continue;

        const double score = silhouetteScoreFromDistance(
            distance.data(), nSegs, labels.data());
        if (score > bestScore) {
            bestScore  = score;
            bestK      = k;
            bestLabels = labels;
        }
    }

    if (!bestLabels.empty()) {
        r.clusterIds = std::move(bestLabels);
        r.kUsed      = bestK;
        return r;
    }

    // Fallback: `_run_clustering(min(4, n_segs))` per Python L344.
    const int fallbackK = std::min(4, nSegs);
    r.clusterIds = runSpectralClusteringOnce(
        affinity.data(), nSegs, fallbackK, 42u);
    r.kUsed      = fallbackK;
    return r;
}

std::vector<NoveltySegmenter::Segment>
NoveltySegmenter::createSegments(const std::vector<double>&        boundaries,
                                 const std::vector<std::int64_t>&  clusterIds,
                                 double                             duration,
                                 const std::vector<double>&         novelty,
                                 const std::vector<float>&          audio,
                                 int                                sr)
{
    std::vector<Segment> segments;
    const int nSegments = static_cast<int>(boundaries.size()) - 1;
    if (nSegments <= 0) return segments;

    // Per-segment RMS energy (Python L392-401 — int-cast truncates positives).
    std::vector<double> segEnergy(static_cast<std::size_t>(nSegments), 0.0);
    const int audioLen = static_cast<int>(audio.size());
    for (int i = 0; i < nSegments; ++i) {
        int startSample = static_cast<int>(boundaries[static_cast<std::size_t>(i)] *
                                           static_cast<double>(sr));
        int endSample   = static_cast<int>(boundaries[static_cast<std::size_t>(i + 1)] *
                                           static_cast<double>(sr));
        startSample = std::max(0, std::min(startSample, audioLen - 1));
        endSample   = std::max(startSample + 1, std::min(endSample, audioLen));
        const int len = endSample - startSample;
        // Mean of squares → RMS.
        double sumSq = 0.0;
        for (int s = startSample; s < endSample; ++s) {
            const double v = static_cast<double>(audio[static_cast<std::size_t>(s)]);
            sumSq += v * v;
        }
        segEnergy[static_cast<std::size_t>(i)] =
            std::sqrt(sumSq / static_cast<double>(len));
    }

    // Cluster count (insertion order preserved via vector-of-pairs + stable sort).
    // Python's Counter in 3.7+ preserves insertion order; we emulate that for
    // most_common tie-breaks.
    std::vector<int>              clusterOrder;          // first-seen order
    std::unordered_map<int, int>  clusterCount;
    for (int i = 0; i < nSegments; ++i) {
        const int c = static_cast<int>(clusterIds[static_cast<std::size_t>(i)]);
        if (clusterCount.find(c) == clusterCount.end()) {
            clusterOrder.push_back(c);
            clusterCount[c] = 1;
        } else {
            clusterCount[c] += 1;
        }
    }

    // Repeating clusters (count ≥ 2) ranked by mean per-segment energy.
    std::vector<int> repeating;
    for (int c : clusterOrder) {
        if (clusterCount[c] >= 2) repeating.push_back(c);
    }
    std::vector<std::pair<int, double>> clusterEnergyPairs;
    for (int c : repeating) {
        double sum = 0.0;
        int    cnt = 0;
        for (int i = 0; i < nSegments; ++i) {
            if (static_cast<int>(clusterIds[static_cast<std::size_t>(i)]) == c) {
                sum += segEnergy[static_cast<std::size_t>(i)];
                ++cnt;
            }
        }
        const double meanE = (cnt > 0) ? sum / static_cast<double>(cnt) : 0.0;
        clusterEnergyPairs.emplace_back(c, meanE);
    }
    // Python L414: sorted(cluster_energy, key=cluster_energy.get, reverse=True)
    // — descending by mean energy. Python's sorted is stable → for ties,
    // preserves dict insertion order (which is insertion-order of `repeating`
    // above since we iterate clusterOrder). Use std::stable_sort with energy
    // descending to mirror.
    std::stable_sort(clusterEnergyPairs.begin(), clusterEnergyPairs.end(),
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second > b.second;
        });

    std::map<int, std::string> clusterToLabel;
    for (std::size_t rank = 0; rank < clusterEnergyPairs.size(); ++rank) {
        const int c = clusterEnergyPairs[rank].first;
        if (rank == 0)      clusterToLabel[c] = "chorus";
        else if (rank == 1) clusterToLabel[c] = "verse";
        else                clusterToLabel[c] = "bridge";
    }

    // Non-repeating clusters → "bridge" (Python L424-427).
    for (int c : clusterOrder) {
        if (clusterToLabel.find(c) == clusterToLabel.end())
            clusterToLabel[c] = "bridge";
    }

    // If NO repeating clusters → Counter.most_common fallback (L429-438).
    if (repeating.empty()) {
        // Counter.most_common in Python sorts by count descending, stable on
        // insertion order for ties. All counts are 1 (since nothing repeats),
        // so order == clusterOrder.
        for (std::size_t rank = 0; rank < clusterOrder.size(); ++rank) {
            const int c = clusterOrder[rank];
            if (rank == 0)      clusterToLabel[c] = "chorus";
            else if (rank == 1) clusterToLabel[c] = "verse";
            else                clusterToLabel[c] = "bridge";
        }
    }

    // Confidence from novelty peak height + position overrides.
    const int nFrames = static_cast<int>(novelty.size());
    const double frameDuration = (nFrames > 0)
        ? duration / static_cast<double>(nFrames)
        : 1.0;

    for (int i = 0; i < nSegments; ++i) {
        Segment s;
        s.start = boundaries[static_cast<std::size_t>(i)];
        s.end   = boundaries[static_cast<std::size_t>(i + 1)];
        // int64_t → int narrowing: cluster ids are tiny (≤ kMaxSegments=12),
        // sklearn returns int64; unified Segment uses int (matches Python).
        s.cluster_id = static_cast<int>(clusterIds[static_cast<std::size_t>(i)]);

        // Position override (15% / 85% for novelty path).
        if (i == 0 && s.start < duration * 0.15) {
            s.label = "intro";
        } else if (i == nSegments - 1 && s.end > duration * 0.85) {
            s.label = "outro";
        } else {
            auto it = clusterToLabel.find(s.cluster_id);
            s.label = (it != clusterToLabel.end()) ? it->second : std::string("verse");
        }

        // Confidence: i==0 → 0.8 literal; i>0 → min(1, 0.5 + 0.5 * nov[bf]).
        if (i > 0 && nFrames > 0) {
            int bf = static_cast<int>(s.start / frameDuration);
            if (bf >= nFrames) bf = nFrames - 1;
            double c = 0.5 + 0.5 * novelty[static_cast<std::size_t>(bf)];
            if (c > 1.0) c = 1.0;
            s.confidence = c;
        } else {
            s.confidence = 0.8;
        }

        segments.push_back(s);
    }
    return segments;
}

} // namespace reamix::analysis
