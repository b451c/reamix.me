#include "analysis/CBMSegmenter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace reamix::analysis {

namespace {

// PARITY: cbm_segmenter.py:104-120 `_build_band_kernel`.
//
// The kernel is a size×size float64 matrix with 1.0 at |i-j| in [1, nBands],
// 0.0 elsewhere (diagonal = 0). Python constructs it via `np.zeros` +
// `np.eye(size, k=d) + np.eye(size, k=-d)` accumulation in float64 — so we
// return std::vector<double> row-major.
std::vector<double> buildBandKernel(int size, int nBands)
{
    if (size <= 1) {
        // Python L115-116: returns a zeros((1,1)). Matches size=1 case,
        // though cbm_segment never builds a kernel for size < minSize.
        return std::vector<double>(1, 0.0);
    }
    std::vector<double> K(static_cast<std::size_t>(size) * size, 0.0);
    const int dMax = std::min(nBands + 1, size);
    for (int d = 1; d < dMax; ++d) {
        // Superdiagonal d and subdiagonal -d. For d < size, there are
        // (size - d) ones on each of the two diagonals.
        for (int i = 0; i + d < size; ++i) {
            K[static_cast<std::size_t>(i)     * size + (i + d)] = 1.0;
            K[static_cast<std::size_t>(i + d) * size + (i)    ] = 1.0;
        }
    }
    return K;
}

// PARITY: cbm_segmenter.py:123-132 `_convolution_cost`.
//
// Homogeneity score for a size p × p diagonal block of the autosimilarity
// matrix. `aRow0Col0Ptr` points at A[seg_start, seg_start] with row stride
// `aStride` (== nBars). Kernel is a row-major size p × p float64 block
// sliced from a precomputed full-size kernel.
//
// Replicates Python's f32*f64 → f64 upcast at the elementwise multiply,
// with the sum in f64.
double convolutionCost(const float*  aRow0Col0Ptr,
                       int           aStride,
                       const double* kernelPtr,
                       int           p)
{
    if (p <= 1) return 0.0;
    double s = 0.0;
    for (int i = 0; i < p; ++i) {
        const float*  aRow = aRow0Col0Ptr + static_cast<std::size_t>(i) * aStride;
        const double* kRow = kernelPtr    + static_cast<std::size_t>(i) * p;
        for (int j = 0; j < p; ++j) {
            // f32 → f64 implicit promotion at the multiply — exactly matches
            // numpy broadcast rule A_block_f32 * kernel_f64 = f64.
            s += static_cast<double>(aRow[j]) * kRow[j];
        }
    }
    return s / (static_cast<double>(p) * p);
}

// PARITY: cbm_segmenter.py:141-151 `_penalty_modulo8`. Favors 8-bar then
// 4-bar-multiples then any even length. Pure scalar lookup; lives here so
// cbm_segment can inline the constant-class dispatch.
double penaltyModulo8(int length)
{
    if (length <= 0) return 1.0;
    if (length == 8) return 0.0;
    if (length % 4 == 0) return 0.25;
    if (length % 2 == 0) return 0.50;
    return 1.0;
}

// PARITY: numpy `np.sum` pairwise reduction for float32. numpy uses a
// binary tree recursion with a naive-sum leaf at `PAIRWISE_BLOCKSIZE = 8`
// (see numpy/_core/src/umath/loops_utils.h.src). `np.mean` on f32 input
// accumulates in f32 via this scheme. For `cbm_analyze` step 4
// (cbm_segmenter.py:433 `np.mean(audio[s0:s1] ** 2)`), C++ naive summation
// would diverge by ~O(N)·ULP(f32) on ~11 000-sample spans (one bar at
// sr=22050) where pairwise accumulates ~O(log N)·ULP(f32). This is not
// hypothetical — ~100 ULP drift in mean-squared values can flip energy-
// rank ties on close-clustered groups, so we match numpy's scheme upfront
// to keep the energy-sort boundary clean.
//
// Operates on `x[i] * x[i]` elements; keeps the square+sum fused in leaf
// base case to mirror numpy's `x**2` → `np.sum` fused reduction path.
float pairwiseSumSquaresF32(const float* x, std::size_t n)
{
    constexpr std::size_t kLeaf = 8;
    if (n <= kLeaf) {
        float s = 0.0f;
        for (std::size_t i = 0; i < n; ++i) s += x[i] * x[i];
        return s;
    }
    const std::size_t half = n / 2;
    return pairwiseSumSquaresF32(x, half)
         + pairwiseSumSquaresF32(x + half, n - half);
}

// PARITY: `round(x, 3)` — CPython's builtin round() for floats uses
// dtoa-based correct rounding (see Objects/floatobject.c::float___round___impl
// which calls `_Py_dg_stdnan` / `_Py_dg_dtoa`). IEEE-754 "%.3f" printf +
// strtod round-trip implements the same round-to-nearest-even semantics
// bitwise for values in the normal f64 range; `std::nearbyint(x*1000)/1000`
// would drift at 0.0005 boundaries because FP multiply is not exact.
// Used by labelSegments to emit start/end bit-exact with Python JSON.
double round3(double x)
{
    char   buf[32] = { 0 };
    // Negative zero path: printf("%.3f", -0.0) yields "-0.000"; strtod
    // parses as -0.0. Python's round(-0.0, 3) also returns -0.0. Same.
    // NaN/inf: "%.3f" yields "nan"/"inf"; strtod returns NaN/inf. Same.
    std::snprintf(buf, sizeof(buf), "%.3f", x);
    return std::strtod(buf, nullptr);
}

} // namespace

// ---------------------------------------------------------------------------
// Step 1 — bar-level features
// ---------------------------------------------------------------------------

CBMSegmenter::BarFeaturesResult
CBMSegmenter::computeBarFeatures(const float*  features,
                                 int           nBeats,
                                 int           nFeat,
                                 const double* beatTimes,
                                 int           nBeatTimes,
                                 const double* downbeats,
                                 int           nDownbeats)
{
    BarFeaturesResult out;
    out.nFeat = nFeat;

    // PARITY: cbm_segmenter.py:51 — degenerate return `np.empty((0, n_feat))`.
    if (nDownbeats < 2 || nBeatTimes < 2 || nBeats < 2) {
        return out;
    }

    // Map each downbeat to nearest beat index. PARITY: L55-58 argmin over
    // |beat_times - db_time|. Tiebreak: numpy argmin returns the FIRST
    // minimum index; std::min_element does the same on equal keys.
    std::vector<int> dbBeatIdx;
    dbBeatIdx.reserve(static_cast<std::size_t>(nDownbeats));
    for (int d = 0; d < nDownbeats; ++d) {
        const double dbTime = downbeats[d];
        int   bestIdx  = 0;
        double bestDiff = std::abs(beatTimes[0] - dbTime);
        for (int b = 1; b < nBeatTimes; ++b) {
            const double diff = std::abs(beatTimes[b] - dbTime);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx  = b;
            }
        }
        dbBeatIdx.push_back(bestIdx);
    }

    // PARITY: L59 `sorted(set(db_beat_indices))` — dedupe + ascending sort.
    // Meshuggah hits this heavily (114 downbeats → 97 unique beat indices).
    std::set<int> dbSet(dbBeatIdx.begin(), dbBeatIdx.end());
    std::vector<int> dbSorted(dbSet.begin(), dbSet.end());

    // PARITY: L63 bar_boundaries = db_beat_indices + [n_beats].
    std::vector<int> boundaries = dbSorted;
    boundaries.push_back(nBeats);

    // PARITY: L66-77 per-bar mean + bar_times[lo].
    std::vector<float> barFeats;
    barFeats.reserve(boundaries.size() * nFeat);
    std::vector<double> barTimesList;
    barTimesList.reserve(boundaries.size());

    const int nBoundaries = static_cast<int>(boundaries.size());
    for (int i = 0; i < nBoundaries - 1; ++i) {
        const int lo = boundaries[i];
        const int hi = boundaries[i + 1];
        if (hi <= lo) continue;  // PARITY: L71-72 guard.

        // Mean of rows [lo, hi) along axis 0. numpy's `.mean(axis=0)` on
        // float32 input accumulates in float32 and divides by n as float32.
        // Replicate bit-for-bit: naive-sequential f32 sum, f32 division.
        const int span = hi - lo;
        std::vector<float> bar(nFeat, 0.0f);
        for (int r = lo; r < hi; ++r) {
            const float* srcRow = features + static_cast<std::size_t>(r) * nFeat;
            for (int k = 0; k < nFeat; ++k) bar[k] += srcRow[k];
        }
        const float denom = static_cast<float>(span);
        for (int k = 0; k < nFeat; ++k) bar[k] /= denom;
        barFeats.insert(barFeats.end(), bar.begin(), bar.end());

        barTimesList.push_back(beatTimes[lo]);
    }

    if (barTimesList.empty()) {
        // PARITY: L76-77 empty-bar return.
        return out;
    }

    // PARITY: L79-82 append extrapolated song-end. Python stores beat_times
    // as float64 via np.array; (beats[-1] - beats[-2]) inherits that.
    const double lastBeat = beatTimes[nBeatTimes - 1];
    const double delta    = (nBeatTimes > 1)
                              ? (lastBeat - beatTimes[nBeatTimes - 2])
                              : 0.5;
    barTimesList.push_back(lastBeat + delta);

    out.nBars       = static_cast<int>(barTimesList.size()) - 1;
    out.barFeatures = std::move(barFeats);
    out.barTimes    = std::move(barTimesList);
    return out;
}

// ---------------------------------------------------------------------------
// Step 2 — bar autosimilarity
// ---------------------------------------------------------------------------

std::vector<float>
CBMSegmenter::computeBarAutosimilarity(const std::vector<float>& barFeatures,
                                       int                       nBars,
                                       int                       nFeat)
{
    // PARITY: cbm_segmenter.py:87-97. L2-normalize rows (norm < 1e-8 → 1.0),
    // compute A = X_norm @ X_norm.T, clip to [0, 1]. All arithmetic stays
    // in f32 to match numpy's dtype-preserving matmul + clip.
    const std::size_t matSize = static_cast<std::size_t>(nBars) * nBars;
    std::vector<float> A(matSize, 0.0f);
    if (nBars <= 0 || nFeat <= 0) return A;

    // L2-normalize in place into a scratch buffer.
    std::vector<float> Xn(static_cast<std::size_t>(nBars) * nFeat, 0.0f);
    for (int i = 0; i < nBars; ++i) {
        const float* src = barFeatures.data() + static_cast<std::size_t>(i) * nFeat;
        // Naive-sequential f32 sum of squares — matches numpy f32 norm
        // for small nFeat (59). numpy uses BLAS snrm2 for larger arrays
        // which may pairwise-sum; bitwise parity on 59-dim rows has been
        // stable against naive sum in phase-2 precedent (NoveltyFeatures).
        float sumSq = 0.0f;
        for (int k = 0; k < nFeat; ++k) sumSq += src[k] * src[k];
        float norm = std::sqrt(sumSq);
        if (norm < 1e-8f) norm = 1.0f;
        float* dst = Xn.data() + static_cast<std::size_t>(i) * nFeat;
        for (int k = 0; k < nFeat; ++k) dst[k] = src[k] / norm;
    }

    // A = Xn @ Xn.T, f32 throughout, then clip [0, 1].
    // Matmul is naive triple-loop in f32 — numpy's f32 @ f32 also returns
    // f32; default numpy matmul does NOT auto-promote to f64 for the accum.
    for (int i = 0; i < nBars; ++i) {
        const float* row_i = Xn.data() + static_cast<std::size_t>(i) * nFeat;
        for (int j = 0; j < nBars; ++j) {
            const float* row_j = Xn.data() + static_cast<std::size_t>(j) * nFeat;
            float s = 0.0f;
            for (int k = 0; k < nFeat; ++k) s += row_i[k] * row_j[k];
            if (s < 0.0f)      s = 0.0f;
            else if (s > 1.0f) s = 1.0f;
            A[static_cast<std::size_t>(i) * nBars + j] = s;
        }
    }
    return A;
}

// ---------------------------------------------------------------------------
// Step 3 — CBM dynamic programming
// ---------------------------------------------------------------------------

std::vector<CBMSegmenter::BarSegment>
CBMSegmenter::cbmSegment(const std::vector<float>& A,
                         int                       nBars,
                         int                       minSize,
                         int                       maxSize,
                         double                    penaltyWeight,
                         int                       nBands)
{
    // PARITY: L177-179 short-circuit.
    if (nBars < minSize) {
        if (nBars > 0) return { { 0, nBars } };
        return {};
    }

    // PARITY: L182-184 precompute kernels for each legal segment size.
    // Store as a dense vector indexed by size (slot 0..minSize-1 unused).
    const int maxKernelSize = std::min(maxSize + 1, nBars + 1);
    std::vector<std::vector<double>> kernels(maxKernelSize);
    for (int s = minSize; s < maxKernelSize; ++s) {
        kernels[s] = buildBandKernel(s, nBands);
    }

    // PARITY: L186-197 normalization — max_conv over size-8 sliding windows.
    // Falls back to max_conv = 1.0 when the whole matrix is ~zero (L196-197).
    const int normSize = std::min(8, nBars);
    double maxConv = 0.0;
    if (normSize >= minSize) {
        const std::vector<double>& normKern = kernels[normSize];
        for (int i = 0; i + normSize <= nBars; ++i) {
            const float* blockPtr = A.data()
                                  + static_cast<std::size_t>(i) * nBars + i;
            const double c = convolutionCost(
                blockPtr, /*aStride=*/nBars,
                normKern.data(), /*p=*/normSize
            );
            if (c > maxConv) maxConv = c;
        }
    }
    if (maxConv < 1e-10) maxConv = 1.0;

    // PARITY: L199-219 DP forward pass.
    //
    // scores[j] = best score of a segmentation ending at position j
    // (last segment closes at bar index j, half-open end). backtrack[j]
    // records the start bar of that final segment.
    constexpr double kNegInf = -std::numeric_limits<double>::infinity();
    constexpr double kReachThreshold = -1e30;  // Python L214 "unreachable" guard

    std::vector<double>   scores(nBars, kNegInf);
    std::vector<std::int64_t> backtrack(nBars, 0);

    for (int j = minSize; j < nBars; ++j) {
        const int segStartLo = std::max(0, j - maxSize);
        const int segStartHi = j - minSize;  // inclusive upper bound
        for (int segStart = segStartLo; segStart <= segStartHi; ++segStart) {
            const int segLen = j - segStart;
            if (segLen < minSize || segLen >= static_cast<int>(kernels.size())
                || kernels[segLen].empty()) continue;

            const float* blockPtr = A.data()
                                  + static_cast<std::size_t>(segStart) * nBars
                                  + segStart;
            const double h = convolutionCost(
                blockPtr, /*aStride=*/nBars,
                kernels[segLen].data(), /*p=*/segLen
            );
            const double penalty = penaltyModulo8(segLen);
            const double cost    = h * segLen - penalty * penaltyWeight * maxConv;

            // PARITY: L213 `prev_score = scores[seg_start] if seg_start > 0 else 0.0`.
            const double prevScore = (segStart > 0) ? scores[segStart] : 0.0;

            // PARITY: L214-215 unreachable-previous guard.
            if (segStart > 0 && prevScore < kReachThreshold) continue;

            const double total = prevScore + cost;
            if (total > scores[j]) {
                scores[j]    = total;
                backtrack[j] = segStart;
            }
        }
    }

    // PARITY: L221-224 "DP couldn't reach last bar" full-span fallback.
    if (scores[nBars - 1] < kReachThreshold) {
        return { { 0, nBars } };
    }

    // PARITY: L226-241 backtrack + cover-from-zero patch.
    std::vector<BarSegment> segments;
    int pos = nBars - 1;
    while (pos >= minSize) {
        const int start = static_cast<int>(backtrack[pos]);
        segments.push_back({ start, pos + 1 });  // +1 for exclusive end
        if (start <= 0) break;
        pos = start - 1;
    }

    // Ensure first bar is covered. Python L237-238.
    if (!segments.empty() && segments.back().start > 0) {
        segments.push_back({ 0, segments.back().start });
    }

    std::reverse(segments.begin(), segments.end());
    return segments;
}

// ---------------------------------------------------------------------------
// Step 5 — label segments (intro/verse/chorus/bridge/outro)
// ---------------------------------------------------------------------------

std::vector<CBMSegmenter::Segment>
CBMSegmenter::labelSegments(const std::vector<BarSegment>& barSegments,
                            const std::vector<float>&      barFeatures,
                            int                            nBars,
                            int                            nFeat,
                            const std::vector<double>&     barTimes,
                            const std::vector<double>&     barEnergy)
{
    const int nSegs = static_cast<int>(barSegments.size());
    if (nSegs == 0) return {};

    // PARITY: L276 `duration = float(bar_times[-1]) if len(bar_times) > 0 else 0.0`.
    const double duration = !barTimes.empty() ? barTimes.back() : 0.0;

    // --- Compute seg centroids --------------------------------------------
    // PARITY: L278-286. Per-seg mean of bar_features rows; empty slabs
    // (end <= start) emit a zero row. Dtype matches bar_features (f32 in
    // production; we keep f32 here). Naive-sequential f32 sum matches
    // numpy's `.mean(axis=0)` on small slabs (typical bar_features slice
    // is < 20 bars × 59 features — numpy pairwise within 1 ULP of naive
    // at this size). Session-5 precedent: `compute_bar_features` uses
    // the same naive-sequential scheme and hit bitwise parity.
    std::vector<float> centroids(
        static_cast<std::size_t>(nSegs) * nFeat, 0.0f);
    for (int s = 0; s < nSegs; ++s) {
        const int startBar = barSegments[s].start;
        const int endBar   = std::min(barSegments[s].end, nBars);
        float* dst = centroids.data() + static_cast<std::size_t>(s) * nFeat;
        if (endBar > startBar) {
            for (int r = startBar; r < endBar; ++r) {
                const float* row = barFeatures.data()
                                 + static_cast<std::size_t>(r) * nFeat;
                for (int k = 0; k < nFeat; ++k) dst[k] += row[k];
            }
            const float denom = static_cast<float>(endBar - startBar);
            for (int k = 0; k < nFeat; ++k) dst[k] /= denom;
        }
        // else: dst stays at zeros (matches np.zeros L285).
    }

    // --- L2-normalize centroids (f32) -------------------------------------
    // PARITY: L289-291. `norms[norms < 1e-8] = 1.0` guards against
    // divide-by-zero on all-zero centroid rows. numpy's linalg.norm on
    // f32 returns f32; we stay in f32 throughout.
    std::vector<float> normed(centroids.size(), 0.0f);
    for (int s = 0; s < nSegs; ++s) {
        const float* src = centroids.data() + static_cast<std::size_t>(s) * nFeat;
        float sumSq = 0.0f;
        for (int k = 0; k < nFeat; ++k) sumSq += src[k] * src[k];
        float norm = std::sqrt(sumSq);
        if (norm < 1e-8f) norm = 1.0f;
        float* dst = normed.data() + static_cast<std::size_t>(s) * nFeat;
        for (int k = 0; k < nFeat; ++k) dst[k] = src[k] / norm;
    }

    // --- Similarity matrix (f32 matmul) -----------------------------------
    // PARITY: L295 `sim_matrix = normed @ normed.T`. numpy's f32 matmul
    // may use BLAS (2-4 f32 ULP drift vs our naive triple-loop per the
    // session-5 autosim finding). On the 10-track corpus no off-diagonal
    // sim value lands within ±0.02 of the 0.80 threshold, so ULP drift
    // cannot flip clustering membership — held empirically before the
    // C++ port. A future corpus track near the threshold would surface
    // as a cluster-flip gate failure.
    const std::size_t simN  = static_cast<std::size_t>(nSegs) * nSegs;
    std::vector<float> sim(simN, 0.0f);
    for (int i = 0; i < nSegs; ++i) {
        const float* ri = normed.data() + static_cast<std::size_t>(i) * nFeat;
        for (int j = 0; j < nSegs; ++j) {
            const float* rj = normed.data() + static_cast<std::size_t>(j) * nFeat;
            float s = 0.0f;
            for (int k = 0; k < nFeat; ++k) s += ri[k] * rj[k];
            sim[static_cast<std::size_t>(i) * nSegs + j] = s;
        }
    }

    // --- Greedy single-linkage grouping -----------------------------------
    // PARITY: L294, L296-309. Threshold 0.80 strict `>=`, iteration order
    // is `for i in range(n), for j in range(i+1, n)`. `assigned` set
    // prevents double-claiming a segment.
    constexpr float kSimThreshold = 0.80f;
    std::vector<std::vector<int>> groups;
    std::vector<char> assigned(static_cast<std::size_t>(nSegs), 0);
    for (int i = 0; i < nSegs; ++i) {
        if (assigned[i]) continue;
        std::vector<int> group;
        group.push_back(i);
        assigned[i] = 1;
        for (int j = i + 1; j < nSegs; ++j) {
            if (assigned[j]) continue;
            const float v = sim[static_cast<std::size_t>(i) * nSegs + j];
            if (v >= kSimThreshold) {
                group.push_back(j);
                assigned[j] = 1;
            }
        }
        groups.push_back(std::move(group));
    }

    // --- Per-group energy + repetition count ------------------------------
    // PARITY: L312-326. `energy = mean-of-mean-bar-RMS across segs in group`.
    // When bar_energy is empty, energy stays 0.0 for every group — all
    // repeating groups tie, stable_sort degenerates to insertion order.
    struct GroupInfo
    {
        std::vector<int> indices;
        int              count  = 0;
        double           energy = 0.0;
    };
    std::vector<GroupInfo> groupInfo;
    groupInfo.reserve(groups.size());
    const int nBarEnergy = static_cast<int>(barEnergy.size());
    for (const auto& g : groups) {
        GroupInfo gi;
        gi.indices = g;
        gi.count   = static_cast<int>(g.size());
        if (nBarEnergy > 0) {
            double totalEnergy = 0.0;
            for (int segIdx : g) {
                const int s = barSegments[segIdx].start;
                int       e = std::min(barSegments[segIdx].end, nBarEnergy);
                if (e > s) {
                    // numpy.mean on f64 accumulates in f64 — single naive
                    // sum is bit-close within 1 ULP of numpy's pairwise
                    // for spans of ~10-20 bars encountered here.
                    double sum = 0.0;
                    for (int k = s; k < e; ++k) sum += barEnergy[k];
                    totalEnergy += sum / static_cast<double>(e - s);
                }
            }
            gi.energy = totalEnergy / static_cast<double>(gi.count);
        }
        groupInfo.push_back(std::move(gi));
    }

    // --- Sort repeating groups by energy descending (stable) --------------
    // PARITY: L329-330 `repeating.sort(key=energy, reverse=True)`. Python's
    // `sort`/`sorted` is Timsort (stable). `std::stable_sort` in C++ matches
    // this on equal keys — essential when two groups tie on energy (rare
    // but possible on corpora with flat-energy tracks).
    std::vector<GroupInfo> repeating;
    repeating.reserve(groupInfo.size());
    for (const auto& gi : groupInfo) {
        if (gi.count >= 2) repeating.push_back(gi);
    }
    std::stable_sort(
        repeating.begin(), repeating.end(),
        [](const GroupInfo& a, const GroupInfo& b) {
            return a.energy > b.energy;  // strict descending → ties preserve insertion order
        });

    // --- Assign cluster IDs + initial labels ------------------------------
    // PARITY: L332-345. Every segment starts as "bridge"; then repeating
    // groups overwrite by rank (0=chorus, 1=verse, 2+=bridge). Non-repeating
    // groups stay "bridge" (matches `else` arm at L343).
    std::vector<std::string> segLabels(
        static_cast<std::size_t>(nSegs), "bridge");
    std::vector<int> segCluster(static_cast<std::size_t>(nSegs), 0);
    for (int i = 0; i < nSegs; ++i) segCluster[i] = i;
    for (std::size_t gid = 0; gid < groups.size(); ++gid) {
        for (int segIdx : groups[gid]) {
            segCluster[segIdx] = static_cast<int>(gid);
        }
    }
    for (std::size_t rank = 0; rank < repeating.size(); ++rank) {
        const char* lbl = (rank == 0) ? "chorus"
                        : (rank == 1) ? "verse"
                                      : "bridge";
        for (int segIdx : repeating[rank].indices) {
            segLabels[segIdx] = lbl;
        }
    }

    // --- Position overrides: first → intro, last → outro ------------------
    // PARITY: L347-354. Override only applies when n_segs >= 3. Times come
    // from bar_times (f64 pure arithmetic, bitwise vs Python per session
    // 5), so the 0.20/0.80 ratio tests are bit-stable.
    if (nSegs >= 3 && !barTimes.empty()) {
        const int lastBarTimesIdx = static_cast<int>(barTimes.size()) - 1;
        const int firstEndIdx = std::min(barSegments[0].end, lastBarTimesIdx);
        const double firstEndT = barTimes[firstEndIdx];
        if (firstEndT < duration * 0.20) {
            segLabels[0] = "intro";
        }
        const int lastStartIdx = std::min(barSegments[nSegs - 1].start, lastBarTimesIdx);
        const double lastStartT = barTimes[lastStartIdx];
        if (lastStartT > duration * 0.80) {
            segLabels[static_cast<std::size_t>(nSegs) - 1] = "outro";
        }
    }

    // --- Build output Segments --------------------------------------------
    // PARITY: L356-367. Times rounded to 3 decimals via dtoa-matching
    // round3 helper. confidence is constant 0.8.
    std::vector<Segment> out;
    out.reserve(static_cast<std::size_t>(nSegs));
    const int lastBarTimesIdx = barTimes.empty()
        ? 0
        : static_cast<int>(barTimes.size()) - 1;
    for (int i = 0; i < nSegs; ++i) {
        const int startIdx = barTimes.empty()
            ? 0
            : std::min(barSegments[i].start, lastBarTimesIdx);
        const int endIdx = barTimes.empty()
            ? 0
            : std::min(barSegments[i].end, lastBarTimesIdx);
        Segment seg;
        seg.start      = barTimes.empty() ? 0.0 : round3(barTimes[startIdx]);
        seg.end        = barTimes.empty() ? 0.0 : round3(barTimes[endIdx]);
        seg.label      = segLabels[i];
        seg.confidence = 0.8;
        seg.cluster_id = segCluster[i];
        out.push_back(std::move(seg));
    }
    return out;
}

// ---------------------------------------------------------------------------
// cbm_analyze — full CBM pipeline orchestrator
// ---------------------------------------------------------------------------

CBMSegmenter::Result
CBMSegmenter::cbmAnalyze(const float*  features,
                         int           nBeats,
                         int           nFeat,
                         const double* beatTimes,
                         int           nBeatTimes,
                         const double* downbeats,
                         int           nDownbeats,
                         const float*  audioMono,
                         std::size_t   nSamples,
                         int           sampleRate,
                         int           minBars,
                         int           maxBars,
                         double        penaltyWeight,
                         int           nBands)
{
    Result result;

    // Step 1: bar-level features.
    BarFeaturesResult bar = computeBarFeatures(
        features, nBeats, nFeat,
        beatTimes, nBeatTimes,
        downbeats, nDownbeats);
    // PARITY: L406-408 degenerate short-circuit.
    if (bar.nBars < 4) return result;

    // Step 2: autosimilarity.
    std::vector<float> A = computeBarAutosimilarity(
        bar.barFeatures, bar.nBars, bar.nFeat);

    // Step 3: DP segmentation.
    std::vector<BarSegment> barSegs = cbmSegment(
        A, bar.nBars, minBars, maxBars, penaltyWeight, nBands);
    // PARITY: L418-420 too-few-segments short-circuit.
    if (barSegs.size() < 3) return result;

    // Step 4: bar-level RMS energy (for label ranking).
    // PARITY: L422-433. `bar_energy[i] = float(sqrt(mean(audio[s0:s1]**2)))`
    // where `s0 = int(t_start * sr)` is Python `int()` truncation (toward
    // zero, matches C++ static_cast<int64_t>). Clamp order matches Python:
    // s0 into [0, n-1], then s1 into [s0+1, n].
    std::vector<double> barEnergy;
    if (audioMono != nullptr && sampleRate > 0 && nSamples > 0) {
        barEnergy.assign(static_cast<std::size_t>(bar.nBars), 0.0);
        const std::int64_t nSamplesI = static_cast<std::int64_t>(nSamples);
        const int lastBarTimesIdx = static_cast<int>(bar.barTimes.size()) - 1;
        for (int i = 0; i < bar.nBars; ++i) {
            const double tStart = bar.barTimes[i];
            const double tEnd   = bar.barTimes[std::min(i + 1, lastBarTimesIdx)];

            // Python L429-430: `int()` truncates toward zero. For positive
            // products this is equivalent to `static_cast<std::int64_t>`.
            std::int64_t s0 = static_cast<std::int64_t>(tStart * sampleRate);
            std::int64_t s1 = static_cast<std::int64_t>(tEnd   * sampleRate);

            // PARITY: L431-432 clamp sequence.
            if (s0 < 0) s0 = 0;
            if (s0 > nSamplesI - 1) s0 = nSamplesI - 1;
            if (s1 < s0 + 1) s1 = s0 + 1;
            if (s1 > nSamplesI) s1 = nSamplesI;

            const std::size_t n = static_cast<std::size_t>(s1 - s0);
            const float sumSq = pairwiseSumSquaresF32(
                audioMono + s0, n);
            const float meanSq = sumSq / static_cast<float>(n);
            // Python L433 `float(np.sqrt(np.mean(...)))` — sqrt is done on
            // the f32 scalar, then upcast exactly to f64 on the `float()`.
            barEnergy[static_cast<std::size_t>(i)]
                = static_cast<double>(std::sqrt(meanSq));
        }
    }

    // Step 5: assign labels.
    result.segments = labelSegments(
        barSegs, bar.barFeatures, bar.nBars, bar.nFeat,
        bar.barTimes, barEnergy);
    result.success = true;

    // Build boundaries: [seg[0].start, seg[0].end, seg[1].end, ...]
    // (length = n_segments + 1). Matches the dump-tool construction.
    if (!result.segments.empty()) {
        result.boundaries.reserve(result.segments.size() + 1);
        result.boundaries.push_back(result.segments.front().start);
        for (const auto& seg : result.segments) {
            result.boundaries.push_back(seg.end);
        }
    }
    return result;
}

} // namespace reamix::analysis
