#include "analysis/Recurrence.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace reamix::analysis {

namespace {

// PARITY: config.py:20-24 chroma_range(n_features).
inline std::pair<int, int> chromaRange(int nFeat)
{
    const int start = std::max(0, nFeat - Recurrence::kNChromaDims - Recurrence::kNContrastDims);
    const int end   = std::min(start + Recurrence::kNChromaDims, nFeat);
    return {start, end};
}

// PARITY: recurrence.py:69-73 _l2_normalize. Zero-norm rows get norm := 1.0
// (NOT tiny; the "zero vector stays zero vector" semantic of the original).
std::vector<std::vector<double>>
l2NormalizeRows(const std::vector<std::vector<double>>& X)
{
    const int n = static_cast<int>(X.size());
    std::vector<std::vector<double>> Y(n);
    for (int i = 0; i < n; ++i) {
        const int d = static_cast<int>(X[i].size());
        double norm = 0.0;
        for (int k = 0; k < d; ++k) norm += X[i][k] * X[i][k];
        norm = std::sqrt(norm);
        if (norm == 0.0) norm = 1.0;
        Y[i].resize(d);
        for (int k = 0; k < d; ++k) Y[i][k] = X[i][k] / norm;
    }
    return Y;
}

// PARITY: numpy.median — average of two middle values on even-sized input.
// nth_element runs in O(n); max_element on the first half recovers the
// lower middle on even sizes (nth_element only guarantees the pivot element
// is at its final position, not that both halves are fully partitioned).
double medianDestructive(std::vector<double>& v)
{
    const int m = static_cast<int>(v.size());
    if (m == 0) return 0.0;
    const int mid = m / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (m % 2 == 1) return v[mid];
    const double upper = v[mid];
    const double lower = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5 * (lower + upper);
}

// PARITY: recurrence.py:76-114 _mutual_knn_recurrence.
//
// `X` is already L2-normalized (callers: feat_norm or chroma_norm). Writes
// into flat row-major `R` of size n × n (pre-zero assumed). Uses brute-force
// pairwise distances — n < 500 per recurrence.py docstring, so O(n² · d) is
// cheap and sklearn-free keeps the port dep surface clean.
void mutualKnnRecurrence(
    const std::vector<std::vector<double>>& X,
    int                kNeighbors,
    double*            R,
    int                n)
{
    int k = std::min(kNeighbors, n - 1);
    if (k < 1) {
        // PARITY: recurrence.py:90-91 — return identity on degenerate input.
        for (int i = 0; i < n; ++i) R[i * n + i] = 1.0;
        return;
    }

    const int d = static_cast<int>(X[0].size());

    // ---- Pairwise distance² (symmetric) -----------------------------------
    // Store distance-squared to avoid O(n²) sqrt and to reuse downstream in
    // the exp(-distSq / mu) fill. Sorting by distSq preserves the same
    // ordering as sorting by dist (sqrt is monotonic).
    std::vector<double> distSq(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double s = 0.0;
            const auto& xi = X[i];
            const auto& xj = X[j];
            for (int t = 0; t < d; ++t) {
                const double diff = xi[t] - xj[t];
                s += diff * diff;
            }
            distSq[i * n + j] = s;
            distSq[j * n + i] = s;
        }
    }

    // ---- Per-row k-NN ------------------------------------------------------
    // Python's `NearestNeighbors(n_neighbors=k+1).kneighbors(X_norm)` returns
    // the k+1 closest including self at index 0 (dist=0). `indices[:, 1:]`
    // drops self. We partial-sort k+1 smallest per row and take entries 1..k.
    // Tiebreak ascending by j: if two rows have identical dist (near-
    // impossible on 59-dim L2-normalized real-music features but defensive),
    // lower-index wins — matches numpy's stable sort + self-index=i landing
    // first when all zero-dist rows tie.
    std::vector<std::vector<int>> knnIdx(n);
    std::vector<double> closestDist(n, 0.0);     // distances[:, 1] in Python, for mu.
    std::vector<std::pair<double, int>> row(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            row[j] = {distSq[i * n + j], j};
        }
        std::partial_sort(row.begin(), row.begin() + (k + 1), row.end(),
            [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
        knnIdx[i].reserve(k);
        for (int m = 1; m <= k; ++m) knnIdx[i].push_back(row[m].second);
        // row[1] is the 1st non-self NN; closestDist (not distSq) goes into mu median.
        closestDist[i] = std::sqrt(row[1].first);
    }

    // ---- Gaussian bandwidth: mu = median(closestDist), floor 1e-8 ---------
    double mu = medianDestructive(closestDist);    // closestDist is consumed.
    if (mu < Recurrence::kMuFloor) mu = Recurrence::kMuFloor;

    // ---- Symmetric (union) k-NN fill --------------------------------------
    // PARITY: recurrence.py:101-112. For each i, iterate knn_set[i]; mirror
    // the cell to (j, i). The `if R[i, j] > 0: continue` check skips cells
    // already filled by j's earlier loop — cheap correctness, avoids double
    // exp() on the symmetric pair when j is also in knnIdx[i].
    for (int i = 0; i < n; ++i) {
        for (int j : knnIdx[i]) {
            if (R[i * n + j] > 0.0) continue;
            const double dSq = distSq[i * n + j];
            const double val = std::exp(-dSq / mu);
            R[i * n + j] = val;
            R[j * n + i] = val;
        }
    }
}

// PARITY: recurrence.py:117-146 _homogeneity_matrix.
// Fills R[i, i+1] = R[i+1, i] = exp(-||x_{i+1} - x_i||² / sigma²) for
// 0 ≤ i < n-1; everything else stays zero. sigma² = median of consecutive
// dist², floored at 1e-8.
//
// Vestigial arg note: Python takes `beat_times` here but never uses it
// (sigma comes from row diffs, not beat timing). Dropped in port.
void homogeneityMatrix(
    const std::vector<std::vector<double>>& X,
    double*            R,
    int                n)
{
    if (n < 2) return;

    const int d = static_cast<int>(X[0].size());
    std::vector<double> distSq(n - 1, 0.0);
    for (int i = 0; i < n - 1; ++i) {
        double s = 0.0;
        const auto& a = X[i];
        const auto& b = X[i + 1];
        for (int t = 0; t < d; ++t) {
            const double diff = b[t] - a[t];
            s += diff * diff;
        }
        distSq[i] = s;
    }

    std::vector<double> tmp = distSq;
    double sigmaSq = medianDestructive(tmp);
    if (sigmaSq < Recurrence::kMuFloor) sigmaSq = Recurrence::kMuFloor;

    for (int i = 0; i < n - 1; ++i) {
        const double val = std::exp(-distSq[i] / sigmaSq);
        R[i * n + (i + 1)] = val;
        R[(i + 1) * n + i] = val;
    }
}

} // namespace

Recurrence::Result
Recurrence::build(const float* features, int nBeats, int nFeat, int kNeighbors)
{
    Result res;
    res.nBeats = nBeats;
    const int n = nBeats;
    if (n <= 0 || nFeat <= 0 || features == nullptr) return res;

    const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    res.R      .assign(nn, 0.0);
    res.rFeat  .assign(nn, 0.0);
    res.rChroma.assign(nn, 0.0);
    res.rHomo  .assign(nn, 0.0);

    // PARITY: recurrence.py:48-49 — n < 4 short-circuits to identity.
    if (n < 4) {
        for (int i = 0; i < n; ++i) res.R[i * n + i] = 1.0;
        return res;
    }

    // ---- Upcast f32 input to f64 (single copy) ----------------------------
    std::vector<std::vector<double>> X(n, std::vector<double>(nFeat));
    for (int i = 0; i < n; ++i) {
        const float* row = features + static_cast<std::size_t>(i) * nFeat;
        for (int j = 0; j < nFeat; ++j)
            X[i][j] = static_cast<double>(row[j]);
    }

    // ---- L2-normalize full feature rows -----------------------------------
    const auto featNorm = l2NormalizeRows(X);

    // ---- Extract chroma slice from RAW features (pre-normalize) + normalize
    // PARITY: recurrence.py:52-53 — `chroma = features[:, cs:ce].copy()`
    // takes the slice from the original `features`, then L2-normalizes the
    // SLICE (not the full-row-normalized chroma columns). The chroma slice
    // has its own norm (≠ 1 in general, since only 12 of nFeat dimensions).
    const auto [cs, ce] = chromaRange(nFeat);
    const int chromaWidth = ce - cs;
    std::vector<std::vector<double>> chromaRaw(n, std::vector<double>(chromaWidth));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < chromaWidth; ++j)
            chromaRaw[i][j] = X[i][cs + j];
    const auto chromaNorm = l2NormalizeRows(chromaRaw);

    // ---- Build component matrices -----------------------------------------
    mutualKnnRecurrence(featNorm,   kNeighbors, res.rFeat  .data(), n);
    mutualKnnRecurrence(chromaNorm, kNeighbors, res.rChroma.data(), n);
    homogeneityMatrix  (featNorm,               res.rHomo  .data(), n);

    // ---- Weighted combine -------------------------------------------------
    // PARITY: recurrence.py:65. f64 throughout.
    for (std::size_t idx = 0; idx < nn; ++idx) {
        res.R[idx] = kWFeatures     * res.rFeat  [idx]
                   + kWChroma       * res.rChroma[idx]
                   + kWHomogeneity  * res.rHomo  [idx];
    }

    return res;
}

// PARITY: recurrence.py:149-228 find_diagonals. Added phase-4 session 24 for
// TransitionPrescreen. See header Diagonal struct docstring for algorithm.
std::vector<Recurrence::Diagonal>
Recurrence::findDiagonals(const double* R,
                          int           n,
                          int           rowStart,
                          int           rowEnd,
                          int           colStart,
                          int           colEnd,
                          int           minLength,
                          int           barSize)
{
    std::vector<Diagonal> results;
    if (R == nullptr || n <= 0 || barSize < 1 || minLength < 1) return results;

    const int delta = colStart - rowStart;

    // PARITY: recurrence.py:176-177 — `range(Δ - bar, Δ + bar + 1, bar)` emits
    // {Δ-bar, Δ, Δ+bar} for bar ≥ 1. Enumerate the 3 explicitly; skip 0.
    const int offsets[3] = { delta - barSize, delta, delta + barSize };

    for (int oi = 0; oi < 3; ++oi) {
        const int offset = offsets[oi];
        if (offset == 0) continue;  // recurrence.py:178-179

        // Trace diagonal: collect (r, c) positions + vals where bounds hold.
        // PARITY: recurrence.py:184-188 — row range [max(rowStart, 0),
        // min(rowEnd, n)); per-r check `colStart ≤ c < colEnd AND 0 ≤ c < n`.
        std::vector<double> vals;
        std::vector<std::pair<int, int>> positions;
        const int rLo = std::max(rowStart, 0);
        const int rHi = std::min(rowEnd, n);
        vals.reserve(static_cast<std::size_t>(std::max(0, rHi - rLo)));
        positions.reserve(vals.capacity());
        for (int r = rLo; r < rHi; ++r) {
            const int c = r + offset;
            if (c < colStart || c >= colEnd) continue;
            if (c < 0 || c >= n) continue;
            vals.push_back(R[static_cast<std::size_t>(r) * n + c]);
            positions.emplace_back(r, c);
        }

        if (static_cast<int>(vals.size()) < minLength) continue;

        // PARITY: recurrence.py:193-217 — find the BEST contiguous run of
        // values strictly above kDiagonalThreshold (0.1). Tracks `run_len`,
        // `run_sum`, and records the best-so-far when `run_len > best_run_len`.
        // On a below-threshold value, both `run_len` and `run_sum` reset to 0
        // (run_start implicit via idx on next above).
        int    best_run_start = 0;
        int    best_run_len   = 0;
        double best_run_sum   = 0.0;
        int    run_start = 0;
        int    run_len   = 0;
        double run_sum   = 0.0;

        const int m = static_cast<int>(vals.size());
        for (int idx = 0; idx < m; ++idx) {
            if (vals[idx] > kDiagonalThreshold) {
                if (run_len == 0) run_start = idx;
                run_len += 1;
                run_sum += vals[idx];
                if (run_len > best_run_len) {
                    best_run_len   = run_len;
                    best_run_start = run_start;
                    best_run_sum   = run_sum;
                }
            } else {
                run_len = 0;
                run_sum = 0.0;
            }
        }

        if (best_run_len >= minLength) {
            // PARITY: recurrence.py:220-224 — mid = best_run_start + best_run_len // 2
            // (Python floor division on int). `if mid < len(diag_positions)`
            // always true here because best_run_start + best_run_len ≤ m.
            const int mid = best_run_start + best_run_len / 2;
            if (mid < static_cast<int>(positions.size())) {
                Diagonal d;
                d.rMid    = positions[mid].first;
                d.cMid    = positions[mid].second;
                d.length  = best_run_len;
                d.meanSim = best_run_sum / static_cast<double>(best_run_len);
                results.push_back(d);
            }
        }
    }

    // PARITY: recurrence.py:227 — sort DESC by length × mean_similarity.
    // std::sort is not stable; Python's `list.sort` IS stable (Timsort).
    // For ties (length × sim), Python preserves insertion order (= offset
    // enumeration order). std::stable_sort matches that.
    std::stable_sort(results.begin(), results.end(),
        [](const Diagonal& a, const Diagonal& b) {
            return (a.length * a.meanSim) > (b.length * b.meanSim);
        });

    return results;
}

} // namespace reamix::analysis
