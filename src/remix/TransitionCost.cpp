#include "remix/TransitionCost.h"

#include "remix/Quality.h"
#include "remix/SegmentData.h"
#include "dsp/WaveformXcorr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace reamix::remix {

namespace {

// chromaRange moved to TransitionCost.h public (session 23: 2nd TU
// consumer = RegionCost.cpp, Hard Rule #1). N_CHROMA_DIMS + N_CONTRAST_DIMS
// also public in the header.

// ---------------------------------------------------------------------------
// Utility: L2-normalized matrix from f32 (n_rows × n_cols). Zero-norm rows
// divide by 1.0 (matches numpy `norms[norms == 0] = 1.0`). Caller owns
// output buffer of size n_rows × n_cols.
// ---------------------------------------------------------------------------
void l2NormalizeRows(const float* input,
                     int          n_rows,
                     int          n_cols,
                     float*       output)
{
    for (int i = 0; i < n_rows; ++i) {
        const float* src = input  + static_cast<std::size_t>(i) * n_cols;
        float*       dst = output + static_cast<std::size_t>(i) * n_cols;

        // Naive-sequential f32 sum of squares — matches numpy f32 norm
        // for small n_cols (≤ 59 in our pipeline) under the bitwise-parity
        // discipline established in CBMSegmenter (session 6).
        float sumSq = 0.0f;
        for (int k = 0; k < n_cols; ++k) sumSq += src[k] * src[k];
        float norm = std::sqrt(sumSq);
        // Python: `norms[norms == 0] = 1.0`. We use == 0 exactly (not a
        // tolerance). A zero L2 norm on non-negative inputs means every
        // element is exactly 0; a non-zero sum of squares sqrts to
        // non-zero (modulo denormals, which we don't care about here).
        if (norm == 0.0f) norm = 1.0f;

        for (int k = 0; k < n_cols; ++k) dst[k] = src[k] / norm;
    }
}

// ---------------------------------------------------------------------------
// Utility: matmul A @ B.T in f32, naive triple-loop. Matches numpy's
// `normed @ normed.T` dtype-preserving semantics (f32 @ f32 = f32).
// Writes output shape (n_a, n_b). BLAS-free to keep accumulation order
// deterministic; for n_beats ≤ ~800 and n_features ≤ 59, performance is
// fine (~15 ms at 500×500×59 on M1).
// ---------------------------------------------------------------------------
void matmulATransposeF32(const float* A,
                         int          n_a,
                         const float* B,
                         int          n_b,
                         int          n_cols,
                         float*       out)
{
    for (int i = 0; i < n_a; ++i) {
        const float* row_i = A + static_cast<std::size_t>(i) * n_cols;
        for (int j = 0; j < n_b; ++j) {
            const float* row_j = B + static_cast<std::size_t>(j) * n_cols;
            float s = 0.0f;
            for (int k = 0; k < n_cols; ++k) s += row_i[k] * row_j[k];
            out[static_cast<std::size_t>(i) * n_b + j] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Utility: cosine similarity on two f64 vectors (Python `_cosine_sim`
// at transition_cost.py:129-135). Returns 0 if either norm is < 1e-8.
// Else returns clip(dot(a,b) / (na*nb), -1, 1).
// Used for context_sim on window-mean vectors (which are f64 because
// numpy `mean(axis=0)` on f32 promotes to f64).
// ---------------------------------------------------------------------------
double cosineSim(const double* a, const double* b, int n)
{
    double na = 0.0, nb = 0.0, ab = 0.0;
    for (int k = 0; k < n; ++k) na += a[k] * a[k];
    for (int k = 0; k < n; ++k) nb += b[k] * b[k];
    na = std::sqrt(na);
    nb = std::sqrt(nb);
    if (na < COSINE_DEGENERATE_FLOOR || nb < COSINE_DEGENERATE_FLOOR) return 0.0;

    for (int k = 0; k < n; ++k) ab += a[k] * b[k];
    double cos = ab / (na * nb);
    if (cos < -1.0)      cos = -1.0;
    else if (cos >  1.0) cos =  1.0;
    return cos;
}

// ---------------------------------------------------------------------------
// Utility: window mean over `features[lo:hi]` (row range), returning an
// f64 vector of length n_features. Clamps lo/hi to [0, n_beats]; returns
// zeros if lo >= hi after clamp.
//
// PARITY: transition_cost.py::_window_mean (L138-144). numpy `.mean(axis=0)`
// on f32 promotes to f64 (per numpy dtype rules for reduction operators —
// float types promote to float64 for accumulation to avoid ULP loss).
// ---------------------------------------------------------------------------
std::vector<double> windowMeanF64(const float* features,
                                  int          n_beats,
                                  int          n_features,
                                  int          lo,
                                  int          hi)
{
    std::vector<double> out(static_cast<std::size_t>(n_features), 0.0);

    lo = std::max(0, lo);
    hi = std::min(n_beats, hi);
    if (lo >= hi) return out;

    const int count = hi - lo;
    // Accumulate in f64 then divide by count (numpy mean semantics).
    for (int i = lo; i < hi; ++i) {
        const float* row = features + static_cast<std::size_t>(i) * n_features;
        for (int k = 0; k < n_features; ++k) {
            out[k] += static_cast<double>(row[k]);
        }
    }
    const double inv = 1.0 / static_cast<double>(count);
    for (int k = 0; k < n_features; ++k) out[k] *= inv;
    return out;
}

// ---------------------------------------------------------------------------
// Build per-beat string labels from segments + beat_times.
// PARITY: transition_cost.py::_build_beat_labels (L235-251). Last segment
// whose [start, end) contains the beat wins (Python overwrites in order).
// Default "unknown" when no segment contains the beat.
// ---------------------------------------------------------------------------
std::vector<std::string>
buildBeatLabels(const analysis::Segment* segments,
                int                      n_segments,
                const double*            beat_times,
                int                      n_beats)
{
    std::vector<std::string> labels(static_cast<std::size_t>(n_beats), "unknown");
    if (segments == nullptr || n_segments == 0) return labels;

    for (int s = 0; s < n_segments; ++s) {
        // Python `label.lower()`. Our Segment.label is already lowercase
        // by session-10 StructureAnalyzer convention (clusterToLabel emits
        // "chorus"/"verse"/"bridge"/"intro"/"outro"/"unknown"). Still,
        // apply `.lower()` defensively to match Python L245 exactly.
        std::string label = segments[s].label;
        std::transform(label.begin(), label.end(), label.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const double start = segments[s].start;
        const double end   = segments[s].end;
        for (int b = 0; b < n_beats; ++b) {
            if (beat_times[b] >= start && beat_times[b] < end) {
                labels[b] = label;
            }
        }
    }
    return labels;
}

// _compute_segment_data (viterbi_dp.py:387-462) was originally ported inline
// here in session 18. Session 19 extracted it to `src/remix/SegmentData.{h,cpp}`
// so ViterbiDP (session 20+) can share the same implementation without
// duplicating code. Zero semantic change across the extraction: the 16-track
// corpus `test_transition_cost` parity sweep (session-19 VALIDATION row)
// still PASSes bit-identically to the session-18 inline-version numbers.

// ---------------------------------------------------------------------------
// Edge energy → dB pair. (transition_cost.py:200-208). Clips the RMS at
// 1e-6 before `20 * log10(x)` to avoid log(0). Returns both end-dB and
// start-dB arrays; caller indexes `[0][i]` for source edge, `[1][j]` for
// destination edge, same as Python's returned tuple.
// ---------------------------------------------------------------------------
struct EdgeDB
{
    std::vector<double> end_dB;
    std::vector<double> start_dB;
    bool                available = false;
};

EdgeDB computeEdgeEnergyDB(const double* edge_rms_end,
                           const double* edge_rms_start,
                           int           n_beats)
{
    EdgeDB out;
    if (edge_rms_end == nullptr || edge_rms_start == nullptr) return out;
    out.available = true;
    out.end_dB.resize(static_cast<std::size_t>(n_beats));
    out.start_dB.resize(static_cast<std::size_t>(n_beats));
    for (int i = 0; i < n_beats; ++i) {
        const double safe_end   = std::max(edge_rms_end[i],   EDGE_DB_FLOOR);
        const double safe_start = std::max(edge_rms_start[i], EDGE_DB_FLOOR);
        out.end_dB[i]   = 20.0 * std::log10(safe_end);
        out.start_dB[i] = 20.0 * std::log10(safe_start);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Successor similarity matrix (transition_cost.py::_compute_successor_sim_matrix
// L163-178). Row-shifted full-feature cosine.
// ---------------------------------------------------------------------------
std::vector<float>
computeSuccessorSimMatrix(const float* features, int n_beats, int n_features)
{
    const std::size_t N = static_cast<std::size_t>(n_beats) * n_beats;
    std::vector<float> S(N, 0.0f);
    if (n_beats <= 0) return S;

    std::vector<float> normed(static_cast<std::size_t>(n_beats) * n_features);
    l2NormalizeRows(features, n_beats, n_features, normed.data());
    matmulATransposeF32(normed.data(), n_beats, normed.data(), n_beats,
                        n_features, S.data());

    // Clip to [-1, 1]. Python does `np.clip(S, -1, 1)` in-place.
    for (std::size_t k = 0; k < N; ++k) {
        if      (S[k] < -1.0f) S[k] = -1.0f;
        else if (S[k] >  1.0f) S[k] =  1.0f;
    }

    // Row-shift: S[:-1, :] = S[1:, :]; S[-1, :] = 0.0.
    // The numpy assignment reads the "future" rows first (it's a copy,
    // not an alias). We copy into a scratch then write back.
    if (n_beats > 1) {
        for (int i = 0; i < n_beats - 1; ++i) {
            std::copy_n(S.data() + static_cast<std::size_t>(i + 1) * n_beats,
                        n_beats,
                        S.data() + static_cast<std::size_t>(i) * n_beats);
        }
    }
    for (int j = 0; j < n_beats; ++j) {
        S[static_cast<std::size_t>(n_beats - 1) * n_beats + j] = 0.0f;
    }
    return S;
}

// ---------------------------------------------------------------------------
// Edge splice similarity matrix (transition_cost.py::_compute_edge_splice_sim
// L181-194). NOT row-shifted — direct pairwise at splice point.
// ---------------------------------------------------------------------------
std::vector<float>
computeEdgeSpliceSim(const float* edge_features_end,
                     const float* edge_features_start,
                     int          n_beats,
                     int          n_edge_features)
{
    const std::size_t N = static_cast<std::size_t>(n_beats) * n_beats;
    std::vector<float> M(N, 0.0f);
    if (n_beats <= 0 || n_edge_features <= 0) return M;

    std::vector<float> nEnd  (static_cast<std::size_t>(n_beats) * n_edge_features);
    std::vector<float> nStart(static_cast<std::size_t>(n_beats) * n_edge_features);
    l2NormalizeRows(edge_features_end,   n_beats, n_edge_features, nEnd.data());
    l2NormalizeRows(edge_features_start, n_beats, n_edge_features, nStart.data());
    matmulATransposeF32(nEnd.data(), n_beats, nStart.data(), n_beats,
                        n_edge_features, M.data());

    for (std::size_t k = 0; k < N; ++k) {
        if      (M[k] < -1.0f) M[k] = -1.0f;
        else if (M[k] >  1.0f) M[k] =  1.0f;
    }
    return M;
}

// ---------------------------------------------------------------------------
// _fill_sequential_costs (transition_cost.py:214-229).
// Writes W[i, i+1] for i in [0, n-1):
//   - If BOTH i and i+1 are non-boundary: W = min(chroma_D[i,i+1]*0.3, 0.12).
//   - Else: W = chroma_D[i,i+1]*0.5 + 0.1.
// ---------------------------------------------------------------------------
void fillSequentialCosts(double*              W,
                         const double*        chroma_D,
                         int                  n,
                         const std::set<int>& bounds_set)
{
    for (int i = 0; i < n - 1; ++i) {
        const double chD_ii1 = chroma_D[static_cast<std::size_t>(i) * n + (i + 1)];
        const bool   boundary = (bounds_set.count(i) > 0) || (bounds_set.count(i + 1) > 0);
        double       w;
        if (!boundary) {
            w = std::min(chD_ii1 * SEQUENTIAL_COEFF_NONBOUND, SEQUENTIAL_FLOOR);
        } else {
            w = chD_ii1 * SEQUENTIAL_COEFF_BOUNDARY + SEQUENTIAL_BIAS_BOUNDARY;
        }
        W[static_cast<std::size_t>(i) * n + (i + 1)] = w;
    }
}

// ---------------------------------------------------------------------------
// Resolve downbeat → pre-downbeat index sets.
// Python L306-316 + L468 order:
//   - if downbeats provided and non-empty: db_indices = { argmin(|beat_times-t|) }
//   - else: db_indices = { 0, ts, 2*ts, ... }
//   - pre_db_indices = { db - 1 for db in db_indices if db > 0 }
// ---------------------------------------------------------------------------
struct DbIndices
{
    std::set<int> db_set;
    std::set<int> pre_db_set;
};

DbIndices resolveDbIndices(const double* downbeats,
                           int           n_downbeats,
                           const double* beat_times,
                           int           n_beats,
                           int           time_signature)
{
    DbIndices out;
    if (downbeats != nullptr && n_downbeats > 0) {
        for (int d = 0; d < n_downbeats; ++d) {
            const double t = downbeats[d];
            int best = 0;
            double best_val = std::abs(beat_times[0] - t);
            for (int b = 1; b < n_beats; ++b) {
                const double v = std::abs(beat_times[b] - t);
                if (v < best_val) { best_val = v; best = b; }
            }
            out.db_set.insert(best);
        }
    } else if (n_beats > 0) {
        const int ts = std::max(1, time_signature);
        for (int b = 0; b < n_beats; b += ts) out.db_set.insert(b);
    }
    for (int db : out.db_set) {
        if (db > 0) out.pre_db_set.insert(db - 1);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Top-k unsorted selection matching `np.argpartition(arr, k)[:k]`.
// Returns indices (unsorted, but first k are the k smallest values).
// Uses std::nth_element which matches numpy's introselect.
// ---------------------------------------------------------------------------
std::vector<int> argPartitionTopK(const std::vector<double>& arr, int k)
{
    const int n = static_cast<int>(arr.size());
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[i] = i;
    if (k >= n) return idx;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&arr](int a, int b) { return arr[a] < arr[b]; });
    idx.resize(static_cast<std::size_t>(k));
    return idx;
}

} // namespace

// chromaRange public implementation (promoted from anon namespace session 23).
std::pair<int, int> chromaRange(int n_features)
{
    const int start = std::max(0, n_features - N_CHROMA_DIMS - N_CONTRAST_DIMS);
    const int end   = std::min(start + N_CHROMA_DIMS, n_features);
    return {start, end};
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

double sectionImportance(const std::string& label)
{
    // transition_cost.py:93-103 SECTION_IMPORTANCE dict + default 0.5.
    // We use an if-ladder (not std::unordered_map) for clarity + bit-exact
    // branch behavior at single comparison-to-string-literal.
    if (label == "chorus")       return 1.0;
    if (label == "verse")        return 0.7;
    if (label == "bridge")       return 0.5;
    if (label == "pre-chorus")   return 0.6;
    if (label == "post-chorus")  return 0.6;
    if (label == "instrumental") return 0.4;
    if (label == "intro")        return 0.3;
    if (label == "outro")        return 0.3;
    if (label == "unknown")      return 0.5;
    return 0.5;
}

std::vector<double>
computeImportance(int                      n_beats,
                  const analysis::Segment* segments,
                  int                      n_segments,
                  const double*            beat_times)
{
    // transition_cost.py::compute_importance (L106-122). Default 0.5.
    std::vector<double> importance(static_cast<std::size_t>(n_beats), 0.5);
    if (segments == nullptr || beat_times == nullptr || n_beats == 0) {
        return importance;
    }
    for (int s = 0; s < n_segments; ++s) {
        std::string label = segments[s].label;
        std::transform(label.begin(), label.end(), label.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const double score = sectionImportance(label);
        const double start = segments[s].start;
        const double end   = segments[s].end;
        for (int b = 0; b < n_beats; ++b) {
            if (beat_times[b] >= start && beat_times[b] < end) {
                importance[b] = score;
            }
        }
    }
    return importance;
}

std::vector<double>
computeChromaDistance(const float* features, int n_beats, int n_features)
{
    // transition_cost.py::compute_chroma_distance (L147-160).
    const std::size_t N = static_cast<std::size_t>(n_beats) * n_beats;
    std::vector<double> D(N, 0.0);
    if (n_beats <= 0 || n_features <= 0) return D;

    const auto [cs, ce] = chromaRange(n_features);
    const int chroma_n = ce - cs;
    if (chroma_n <= 0) {
        // Degenerate: no chroma columns. Python path still runs matmul on
        // empty slice, producing a (n_beats, n_beats) zero matrix, then D
        // = 1 - 0 = 1; with row-shift last row = 1.0 → D is all 1.0.
        std::fill(D.begin(), D.end(), 1.0);
        return D;
    }

    // Extract chroma submatrix into contiguous buffer.
    std::vector<float> chroma(static_cast<std::size_t>(n_beats) * chroma_n);
    for (int i = 0; i < n_beats; ++i) {
        const float* src = features + static_cast<std::size_t>(i) * n_features + cs;
        float*       dst = chroma.data() + static_cast<std::size_t>(i) * chroma_n;
        for (int k = 0; k < chroma_n; ++k) dst[k] = src[k];
    }

    std::vector<float> normed(static_cast<std::size_t>(n_beats) * chroma_n);
    l2NormalizeRows(chroma.data(), n_beats, chroma_n, normed.data());

    std::vector<float> sim(N, 0.0f);
    matmulATransposeF32(normed.data(), n_beats, normed.data(), n_beats,
                        chroma_n, sim.data());

    // Python: `D = 1.0 - np.clip(sim, -1, 1)`. Numpy keeps the array dtype
    // when combining f32 array with a Python-scalar literal (`np.subtract`
    // returns f32 when the array side is f32). So compute in f32 and widen
    // to f64 only on storage. Widening is value-preserving on f32 → f64, so
    // C++ `D[k] == static_cast<double>(f32_result)` bit-exactly matches
    // Python's f32 value after the loader upcasts it for comparison.
    for (std::size_t k = 0; k < N; ++k) {
        float s = sim[k];
        if      (s < -1.0f) s = -1.0f;
        else if (s >  1.0f) s =  1.0f;
        const float d_f32 = 1.0f - s;
        D[k] = static_cast<double>(d_f32);
    }

    // Row-shift: D[:-1, :] = D[1:, :]; D[-1, :] = 1.0.
    if (n_beats > 1) {
        for (int i = 0; i < n_beats - 1; ++i) {
            std::copy_n(D.data() + static_cast<std::size_t>(i + 1) * n_beats,
                        n_beats,
                        D.data() + static_cast<std::size_t>(i) * n_beats);
        }
    }
    for (int j = 0; j < n_beats; ++j) {
        D[static_cast<std::size_t>(n_beats - 1) * n_beats + j] = 1.0;
    }
    return D;
}

// ---------------------------------------------------------------------------
// Main entry point (transition_cost.py::compute_transition_costs L257-559).
// ---------------------------------------------------------------------------
TransitionCostResult computeTransitionCosts(const TransitionCostInputs& in)
{
    TransitionCostResult res;
    const int n = in.n_beats;
    res.n_beats = n;

    if (n <= 0) {
        res.W.clear();
        res.chroma_D.clear();
        res.importance.clear();
        return res;
    }

    // --- Pre-compute shared matrices ---------------------------------------
    res.chroma_D = computeChromaDistance(in.features, n, in.n_features);
    std::vector<float> successor_sim =
        computeSuccessorSimMatrix(in.features, n, in.n_features);

    res.importance = computeImportance(n, in.segments, in.n_segments, in.beat_times);

    EdgeDB edge_db = computeEdgeEnergyDB(in.edge_rms_end, in.edge_rms_start, n);

    std::vector<std::string> beat_labels =
        buildBeatLabels(in.segments, in.n_segments, in.beat_times, n);

    std::vector<float> edge_splice_matrix;
    bool has_edge_splice = false;
    if (in.edge_features_end != nullptr && in.edge_features_start != nullptr
        && in.n_edge_features > 0) {
        edge_splice_matrix = computeEdgeSpliceSim(in.edge_features_end,
                                                  in.edge_features_start,
                                                  n, in.n_edge_features);
        has_edge_splice = true;
    }

    SegmentData seg_data = computeSegmentData(n, in.segments, in.n_segments,
                                              in.beat_times, in.features, in.n_features);

    // ADR-044: auto path passes n_segments == 0 (no structure). In that case
    // section_sim and the SPAN_PENALTY branches are gated to 0 below to avoid
    // injecting structure-derived signal that we have already rejected as
    // strictly-worse-than-absent (49.5 % vs 64.2 % label_match accuracy on
    // Dua Lipa, research-scout 2026-04-25). label_match itself naturally
    // resolves to 0 because buildBeatLabels returns "unknown" for every beat
    // when segments == nullptr || n_segments == 0; no extra gate needed there.
    // BlockAssembly.cpp is unaffected (always has user labels by mode).
    const bool noStructure = (in.n_segments <= 0);

    DbIndices db_idx = resolveDbIndices(in.downbeats, in.n_downbeats,
                                        in.beat_times, n, in.time_signature);

    // ADR-064 (sesja 75): pre-compute normalised onset strength once for
    // per-pair `transient_continuity` composition inside the inner scoring
    // loop. Empty when onset_strength was not provided → qi.transient_
    // continuity stays nullopt → null-guard in computeQualityScore drops the
    // contribution → bit-exact parity preserved with kDefaultQualityWeights.
    const std::vector<double> onset_norm =
        computeOnsetNorm(in.onset_strength, n);

    // ADR-066 (sesja 77): pre-compute MFCC + delta-MFCC L2 similarity matrix
    // once for per-pair `mfcc_continuity` composition inside the inner
    // scoring loop. Empty when features were not provided → qi.mfcc_
    // continuity stays nullopt → null-guard drops the contribution → parity
    // preserved with kDefaultQualityWeights.
    const std::vector<double> mfcc_continuity_matrix =
        computeMfccContinuityMatrix(in.features, n, in.n_features);

    // ADR-080 RESCOPE + ADR-083 (sesja 92): pre-compute full-mix chroma
    // similarity matrix once for per-pair `full_mix_chroma_continuity`
    // composition inside the inner scoring loop. Tone slider blend logic in
    // computeQualityScore reads this when weights.harmonic_vs_timbre > 0.
    // Default 0.0 weight → blend bypassed → matrix unused → bit-exact baseline.
    //
    // Chroma slice extracted from 59-dim features columns
    // [chroma_start, chroma_end) per chromaRange(). computeChromaContinuityMatrix
    // expects a tightly-packed (n_beats × n_chroma) buffer, so we copy the
    // chroma columns once. Cost: ~48 KB per analyze for 1000-beat track,
    // trivial vs O(n²) DP main loop.
    const auto [chroma_start, chroma_end] = chromaRange(in.n_features);
    const int  n_chroma = chroma_end - chroma_start;
    std::vector<float>  chroma_slice;
    std::vector<double> chroma_continuity_matrix;
    if (n_chroma > 0 && in.features != nullptr) {
        chroma_slice.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n_chroma));
        for (int i = 0; i < n; ++i) {
            const float* src = in.features
                             + static_cast<std::size_t>(i) * in.n_features
                             + chroma_start;
            float* dst = chroma_slice.data()
                       + static_cast<std::size_t>(i) * n_chroma;
            for (int k = 0; k < n_chroma; ++k) dst[k] = src[k];
        }
        chroma_continuity_matrix =
            computeChromaContinuityMatrix(chroma_slice.data(), n, n_chroma);
    }

    // segment_boundaries argument set (for sequential fill).
    std::set<int> bounds_set;
    if (in.segment_boundaries != nullptr) {
        for (int k = 0; k < in.n_segment_boundaries; ++k) {
            const int b = in.segment_boundaries[k];
            if (b >= 0 && b < n) bounds_set.insert(b);
        }
    }

    // --- Initialize W with INF + sequential fill ---------------------------
    res.W.assign(static_cast<std::size_t>(n) * n, INF);
    fillSequentialCosts(res.W.data(), res.chroma_D.data(), n, bounds_set);

    // --- Waveform setup ----------------------------------------------------
    int max_lag_samples = 0;
    if (in.waveform_sample_rate > 0) {
        max_lag_samples = static_cast<int>(in.waveform_align_max_shift_ms
                                           * in.waveform_sample_rate / 1000.0);
    }
    const bool has_waveforms =
        in.boundary_waveforms != nullptr
        && in.n_boundary_waveforms >= n
        && in.waveform_sample_rate > 0
        && max_lag_samples > 0;

    // --- Track-level vocal detection --------------------------------------
    bool track_has_vocals = false;
    if (in.vocal_activity != nullptr) {
        double mx = 0.0;
        bool   any = false;
        for (int b = 0; b < n; ++b) {
            const double v = in.vocal_activity[b];
            if (!any || v > mx) { mx = v; any = true; }
        }
        if (any && mx >= TRACK_VOCAL_THRESHOLD) track_has_vocals = true;
    }

    // vocal_band_waveforms in Python is computed inside compute_transition_costs
    // (L344-356) via scipy butter/sosfilt. In C++ we consume it pre-computed
    // by caller (test path — dump tool; production path — future butter
    // port + SosFilt::apply). Gate: only used when track_has_vocals AND
    // waveform_sample_rate >= VOCAL_BAND_MIN_SR (8000).
    const bool has_vocal_band =
        has_waveforms
        && track_has_vocals
        && in.waveform_sample_rate >= VOCAL_BAND_MIN_SR
        && in.vocal_band_waveforms != nullptr;

    // Prescreened pairs set.
    std::set<std::pair<int, int>> prescreen_set;
    if (in.prescreened_pairs != nullptr) {
        for (int k = 0; k < in.n_prescreened_pairs; ++k) {
            prescreen_set.insert({in.prescreened_pairs[2 * k],
                                  in.prescreened_pairs[2 * k + 1]});
        }
    }

    // Adaptive micro-skip: 4 bars = 4 × time_signature beats.
    const int micro_skip = 4 * in.time_signature;
    const int k_target   = std::min(in.max_candidates_per_beat, n);

    // --- Main loop over source beats --------------------------------------
    for (int i = 0; i < n - 1; ++i) {
        const int source_boundary = i + 1;

        // Copy chroma row, block micro-skip region with INF.
        std::vector<double> chroma_row(static_cast<std::size_t>(n));
        const double* src_row = res.chroma_D.data() + static_cast<std::size_t>(i) * n;
        for (int k = 0; k < n; ++k) chroma_row[k] = src_row[k];

        const int block_lo = std::max(0, i - micro_skip + 1);
        const int block_hi = std::min(n, i + micro_skip);
        for (int k = block_lo; k < block_hi; ++k) chroma_row[k] = INF;

        // Top-k by chroma (UNSORTED — matches np.argpartition).
        std::vector<int> top_indices = argPartitionTopK(chroma_row, k_target);

        // Prescreened targets for this source beat.
        std::vector<int> prescreen_targets;
        for (const auto& p : prescreen_set) {
            if (p.first == i) prescreen_targets.push_back(p.second);
        }

        // Combined evaluation list. Duplicates allowed (top-k ∩ prescreen);
        // last write wins — same as Python dict semantics.
        std::vector<int> eval_indices = top_indices;
        eval_indices.insert(eval_indices.end(),
                            prescreen_targets.begin(), prescreen_targets.end());

        for (int j_raw : eval_indices) {
            const int j = j_raw;
            const bool is_prescreened = prescreen_set.count({i, j}) > 0;

            if (!is_prescreened) {
                if (chroma_row[j] >= INF || chroma_row[j] > in.chroma_prefilter) {
                    continue;
                }
            }

            // --- Hard gate: energy -------------------------------------
            double energy_diff = 0.0;
            if (edge_db.available) {
                energy_diff = std::abs(edge_db.end_dB[i] - edge_db.start_dB[j]);
                if (energy_diff > ENERGY_HARD_BLOCK_DB) continue;
            }

            // --- Vocal activity scalars --------------------------------
            double va_i = 0.0, va_j = 0.0;
            if (in.vocal_activity != nullptr) {
                if (i < n) va_i = in.vocal_activity[i];
                if (j < n) va_j = in.vocal_activity[j];
            }

            // --- Waveform xcorr + deceptive-splice detection -----------
            std::optional<double> waveform_sim;
            int alignment_lag = 0;
            if (has_waveforms) {
                const float* src = in.boundary_waveforms
                                   + static_cast<std::size_t>(source_boundary)
                                     * in.n_samples_per_bnd;
                const float* tgt = in.boundary_waveforms
                                   + static_cast<std::size_t>(j)
                                     * in.n_samples_per_bnd;
                auto [ws, lag] = dsp::WaveformXcorr::compute(
                    src, tgt,
                    static_cast<std::size_t>(in.n_samples_per_bnd),
                    static_cast<std::size_t>(in.n_samples_per_bnd),
                    max_lag_samples);
                waveform_sim = ws;
                alignment_lag = lag;

                // Deceptive splice: when both sides have active vocals
                // (>0.35 runtime gate; see VOCAL_GATE_THRESHOLD/RUNTIME
                // drift note in TransitionCost.h), compute vocal-band
                // xcorr and replace waveform_sim if gap > 0.3.
                if (waveform_sim.has_value()
                    && has_vocal_band
                    && std::min(va_i, va_j) > VOCAL_SPLICE_ACTIVITY_MIN) {
                    const float* vb_src = in.vocal_band_waveforms
                                          + static_cast<std::size_t>(source_boundary)
                                            * in.n_samples_per_bnd;
                    const float* vb_tgt = in.vocal_band_waveforms
                                          + static_cast<std::size_t>(j)
                                            * in.n_samples_per_bnd;
                    auto [vb_sim, /*vb_lag unused*/ _ignored] = dsp::WaveformXcorr::compute(
                        vb_src, vb_tgt,
                        static_cast<std::size_t>(in.n_samples_per_bnd),
                        static_cast<std::size_t>(in.n_samples_per_bnd),
                        max_lag_samples);
                    (void)_ignored;
                    const double gap = waveform_sim.value() - vb_sim;
                    if (gap > DECEPTIVE_SPLICE_GAP_THRESHOLD) {
                        waveform_sim = vb_sim;
                    }
                }
            }

            // --- 10 quality signals ------------------------------------
            // 1. Successor similarity (row-shifted full features).
            const double successor_sim_v = static_cast<double>(
                successor_sim[static_cast<std::size_t>(i) * n + j]);

            // 2. Edge splice similarity (direct pairwise).
            std::optional<double> edge_splice_sim;
            if (has_edge_splice) {
                edge_splice_sim = static_cast<double>(
                    edge_splice_matrix[static_cast<std::size_t>(i) * n + j]);
            }

            // 3. Context similarity.
            std::vector<double> ctx_before = windowMeanF64(
                in.features, n, in.n_features,
                i + CONTEXT_BEFORE_LO_OFFSET, i + CONTEXT_BEFORE_HI_OFFSET);
            std::vector<double> ctx_after = windowMeanF64(
                in.features, n, in.n_features,
                j + CONTEXT_AFTER_LO_OFFSET, j + CONTEXT_AFTER_HI_OFFSET);
            const double context_sim_v =
                cosineSim(ctx_before.data(), ctx_after.data(), in.n_features);

            // 4. Label match.
            const double label_match =
                (beat_labels[i] != "unknown" && beat_labels[i] == beat_labels[j])
                    ? 1.0 : 0.0;

            // 5. Section similarity. ADR-044: 0 in no-structure mode
            // (computeSegmentData would otherwise return seg_sim=[1.0] from
            // its empty-segments fallback, leaking a max-similarity bonus).
            const std::int64_t seg_i = seg_data.beat_to_segment[i];
            const std::int64_t seg_j = seg_data.beat_to_segment[j];
            const double section_sim_v = noStructure ? 0.0
                : seg_data.seg_sim[
                    static_cast<std::size_t>(seg_i) * seg_data.n_segs + seg_j];

            // 6. Bar alignment.
            const double bar_aligned =
                (db_idx.pre_db_set.count(i) > 0 && db_idx.db_set.count(j) > 0)
                    ? 1.0 : 0.0;

            // 7. Energy match (whole-beat RMS).
            double energy_match = 1.0;
            if (in.rms_energy != nullptr) {
                const double rms_diff = std::abs(in.rms_energy[i] - in.rms_energy[j]);
                energy_match = std::max(0.0, 1.0 - rms_diff * RMS_DIFF_SCALE);
            }

            // 8. Edge energy match (soft-penalty saturation at 12 dB).
            double edge_energy_match = 1.0;
            if (edge_db.available) {
                edge_energy_match = std::max(
                    0.0,
                    1.0 - std::min(energy_diff, EDGE_ENERGY_SATURATION_DB)
                          / EDGE_ENERGY_SATURATION_DB);
            }

            // 9. Centroid match.
            double centroid_match = 1.0;
            if (in.spectral_centroid != nullptr) {
                const double c_diff =
                    std::abs(in.spectral_centroid[i] - in.spectral_centroid[j]);
                centroid_match = std::max(0.0, 1.0 - c_diff * CENTROID_DIFF_SCALE);
            }

            // --- Composite quality (Quality::computeQualityScore) ------
            QualityInputs qi{};
            qi.waveform_sim       = waveform_sim;
            qi.successor_sim      = successor_sim_v;
            qi.edge_splice_sim    = edge_splice_sim;
            qi.context_sim        = context_sim_v;
            qi.label_match        = label_match;
            qi.section_sim        = section_sim_v;
            qi.bar_aligned        = bar_aligned;
            qi.energy_match       = energy_match;
            qi.edge_energy_match  = edge_energy_match;
            qi.centroid_match     = centroid_match;
            // ADR-064 sesja 75 — transient continuity per-pair similarity.
            // Computed inline from pre-normalised onset values; nullopt when
            // onset_strength was not provided to this run (parity test path).
            if (! onset_norm.empty()) {
                qi.transient_continuity =
                    1.0 - std::abs(onset_norm[(std::size_t) i]
                                 - onset_norm[(std::size_t) j]);
            }
            // ADR-066 sesja 77 — MFCC continuity per-pair similarity from
            // pre-computed matrix; nullopt when features were not provided.
            if (! mfcc_continuity_matrix.empty()) {
                qi.mfcc_continuity =
                    mfcc_continuity_matrix[(std::size_t) i * n + j];
            }
            // ADR-080 RESCOPE + ADR-083 sesja 92 — full-mix chroma continuity
            // per-pair similarity from pre-computed matrix. Consumed by Tone
            // slider blend in computeQualityScore (weights.harmonic_vs_timbre).
            // nullopt when chroma slice empty → blend bypassed → bit-exact
            // baseline preserved.
            if (! chroma_continuity_matrix.empty()) {
                qi.full_mix_chroma_continuity =
                    chroma_continuity_matrix[(std::size_t) i * n + j];
            }
            // DEV-028 sesja 74 expressivity scratch slot — calibration
            // harness populates `in.extra1_per_pair` as a pre-computed n×n
            // matrix (one signal source per cell). Production default
            // nullptr → nullopt → no contribution.
            if (in.extra1_per_pair != nullptr && in.extra1_n == n) {
                qi.extra1_value = in.extra1_per_pair[(std::size_t) i * n + j];
            }
            // ADR-088 sesja 98 STATUS UPDATE 1 — vocal phrase continuity, fixed
            // formula. Sygnały edge_vocal_onset_start / release_end FIRE w
            // momentach phrase boundaries (rising / falling derivative peaks),
            // NIE w mid-phrase. Wcześniej formuła `1 - max(...)` punishingała
            // boundary alignment (= najlepsze splice'y), powodując "no splices"
            // w vocal-dense tracks (user-flagged sesja 98 smoke).
            //
            // Fix: HIGH boundary signals → HIGH quality (reward alignment).
            //      Plus vocal_activity gate: gdy oba strony silence/instrumental,
            //      q = 1.0 (clean splice).
            //      Plus soft floor 0.5: harmonic mean × kHarmonicMeanEpsilon=1e-3
            //      crashował composite Q poniżej QUALITY_HARD_FLOOR=0.20 dla
            //      każdej pary mid-phrase. Floor 0.5 → harmonic term `w / 0.5`
            //      = 2w, manageable amplification.
            if (in.edge_vocal_onset_start != nullptr
                && in.edge_vocal_release_end != nullptr
                && i < n && j < n) {
                const double rel_i = in.edge_vocal_release_end[(std::size_t) i];
                const double on_j  = in.edge_vocal_onset_start[(std::size_t) j];
                const double boundary = std::max(rel_i, on_j);

                double vocal_density = 0.0;
                if (in.vocal_activity != nullptr) {
                    const double va_i = in.vocal_activity[(std::size_t) i];
                    const double va_j = in.vocal_activity[(std::size_t) j];
                    vocal_density = std::max(va_i, va_j);
                }

                constexpr double kSilenceThreshold = 0.1;
                if (vocal_density < kSilenceThreshold) {
                    qi.vocal_continuity = 1.0;  // instrumental gap — always safe
                } else {
                    qi.vocal_continuity = 0.5 + 0.5 * boundary;
                }
            }

            double quality        = computeQualityScore(
                qi,
                in.quality_weights != nullptr ? *in.quality_weights : kDefaultQualityWeights);

            // Span penalty (additive, negative contribution).
            // ADR-044: skipped in no-structure mode (label_match=0 would
            // otherwise blanket-fire CROSS_SECTION on every short jump).
            if (! noStructure) {
                const int jump_beats = std::abs(j - i);
                if (jump_beats < SPAN_PENALTY_MAX_BEATS && label_match < 0.5) {
                    quality = std::max(0.0, quality - SPAN_PENALTY_CROSS_SECTION);
                } else if (jump_beats < SPAN_PENALTY_MAX_BEATS) {
                    quality = std::max(0.0, quality - SPAN_PENALTY_SAME_SECTION);
                }
            }

            // Vocal penalty.
            if (track_has_vocals) {
                std::optional<double> eva_end;
                std::optional<double> eva_start;
                if (in.edge_vocal_activity_end != nullptr && i < n) {
                    eva_end = in.edge_vocal_activity_end[i];
                }
                if (in.edge_vocal_activity_start != nullptr && j < n) {
                    eva_start = in.edge_vocal_activity_start[j];
                }
                const double vp = computeVocalPenalty(va_i, va_j, eva_end, eva_start);
                quality = std::max(0.0, quality - vp);
            }

            // Onset penalty on destination.
            if (in.onset_strength != nullptr) {
                std::optional<double> os_j;
                if (j < n) os_j = in.onset_strength[j];
                quality = std::max(0.0, quality - computeOnsetPenalty(os_j));
            }

            // Hard quality floor.
            if (quality < in.quality_floor) continue;

            const double total_cost = 1.0 - quality;
            res.W[static_cast<std::size_t>(i) * n + j] = total_cost;

            TransitionCandidate cand;
            cand.from_beat              = i;
            cand.to_beat                = j;
            cand.quality_score          = quality;
            cand.waveform_similarity    = waveform_sim.has_value() ? waveform_sim.value() : 0.0;
            cand.successor_similarity   = successor_sim_v;
            cand.edge_splice_similarity = edge_splice_sim.has_value() ? edge_splice_sim.value() : 0.0;
            cand.chroma_distance        = res.chroma_D[static_cast<std::size_t>(i) * n + j];
            cand.energy_diff_db         = energy_diff;
            cand.alignment_lag_samples  = alignment_lag;
            cand.total_cost             = total_cost;
            // DEV-028 sesja 80 — D1 correlation matrix per-pair components.
            cand.context_similarity     = context_sim_v;
            cand.label_match            = label_match;
            cand.section_similarity     = section_sim_v;
            cand.bar_aligned            = bar_aligned;
            cand.energy_match           = energy_match;
            cand.edge_energy_match      = edge_energy_match;
            cand.centroid_match         = centroid_match;
            cand.transient_continuity   = qi.transient_continuity.has_value()
                                            ? qi.transient_continuity.value() : 0.0;
            cand.mfcc_continuity        = qi.mfcc_continuity.has_value()
                                            ? qi.mfcc_continuity.value() : 0.0;
            res.candidates[{i, j}]      = cand;
        }
    }

    return res;
}

} // namespace reamix::remix
