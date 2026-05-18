#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "StructureResult.h"

namespace reamix::analysis {

// Novelty-path structure segmentation — port of `StructureAnalyzer`'s
// novelty branch (`structure_analyzer.py::_compute_ssm / _compute_novelty /
// _find_boundaries / _compute_segment_embeddings`, L212-296).
//
// ADR-021 split the NoveltySegmenter port across two sessions:
//   Session 7 (this file)  — SSM → novelty curve → find_peaks → boundaries
//                            → segment embeddings (pre-clustering).
//   Session 8 (deferred)   — spectral clustering + silhouette selection +
//                            _create_segments labeling. Clustering decision
//                            per ADR-021 is a full port of sklearn
//                            SpectralClustering with Hungarian-matched
//                            cluster_id parity.
//
// Precision policy per ADR-021 § Parity gate design:
//   SSM          — f32 matmul, expected 1-5 ULP drift vs numpy BLAS
//                  (matches CBM autosim precedent), gate ≤ 1e-6 L∞.
//   novelty      — f64 checkerboard sum + f64 uniform_filter1d + f64
//                  normalize by max. Gate ≤ 1e-9 L∞ (f64 pairwise reduction
//                  drift ~1e-11 worst case on 1024-element sums).
//   peaks_raw    — integer exact.
//   boundaries   — f64 bitwise (deterministic arithmetic on peak indices).
//   embeddings   — f32 bitwise (per-segment mean of bitwise features).
class NoveltySegmenter
{
public:
    // PARITY: structure_analyzer.py L71, L214-217, L66, L230-234, L246,
    // L271, L266, L267.
    static constexpr int    kDefaultSsmStride            = 4;
    static constexpr double kDefaultMinSegmentDuration   = 8.0;
    static constexpr int    kDefaultKernelMax            = 32;
    static constexpr int    kDefaultKernelMin            = 4;
    static constexpr int    kUniformFilterSize           = 5;
    static constexpr double kMarginSeconds               = 2.0;
    static constexpr double kHeightPercentile            = 60.0;
    static constexpr double kProminenceThreshold         = 0.20;
    static constexpr float  kSsmNormEps                  = 1e-8f;

    // Step 1: SSM on stride-downsampled features.
    // PARITY: structure_analyzer.py:212-225 StructureAnalyzer._compute_ssm.
    //
    // `features` uses C++ native layout [T][nFeat] (row = time frame).
    // Python's (nFeat, T) `features_ds = features[:, ::stride]` becomes
    // rows `features[0], features[stride], features[2*stride], ...` in
    // our layout. L2-normalization along axis=0 (Python) == per-row L2
    // normalization in our C++ layout.
    //
    // `nTdsOut` receives the number of downsampled frames (== len(features)
    // stride-sliced with `0, stride, 2*stride, ...`, same as Python's
    // `ceil(T / stride)`). Output is row-major [nTds × nTds] float32.
    static std::vector<float>
    computeSsm(const std::vector<std::vector<float>>& features,
               int                                    stride,
               int&                                   nTdsOut);

    // Step 2: Foote-kernel novelty curve + uniform_filter1d smoothing +
    // normalize by max.
    // PARITY: structure_analyzer.py:227-251 StructureAnalyzer._compute_novelty.
    //
    // Kernel size = min(32, n // 4), floored to ≥ 4, forced even.
    // Kernel pattern (row-major):
    //   [:half, :half] = -1 ; [:half, half:] = +1
    //   [half:, :half] = +1 ; [half:, half:] = -1
    // Novelty at frame i (half ≤ i < n-half) = |Σ region * kernel|
    // where region = ssm[i-half:i+half, i-half:i+half] (f32), kernel is
    // default-f64 numpy ones-matrix; the element-wise multiply upcasts
    // f32*f64 → f64, np.sum reduces in f64.
    //
    // Post-processing: `uniform_filter1d(novelty, size=5)` with scipy
    // default `mode='reflect'` (boundary extension reflects the edge
    // element: index -1 ↔ x[0], -2 ↔ x[1], ...). Final normalize by max
    // when max > 0 (else left as-is to avoid divide-by-zero).
    static std::vector<double>
    computeNovelty(const std::vector<float>& ssm, int nTds);

    // Step 3: scipy.signal.find_peaks port + margin clip + boundary build.
    // PARITY: structure_analyzer.py:253-275 StructureAnalyzer._find_boundaries.
    struct BoundariesResult
    {
        // Raw output of find_peaks BEFORE margin clip (bisection aid for
        // any flip at the peak-detection stage). Matches Python's `peaks`
        // array before the margin filter at L272.
        std::vector<std::int64_t> peaksRaw;

        // Final boundaries array: [0.0, t1, t2, ..., duration]. Times are
        // peaks_raw * frame_duration, filtered to strictly inside
        // (margin, duration - margin). Length = n_segments + 1.
        std::vector<double> boundaries;
    };

    // `minSegmentDuration` is the StructureAnalyzer.__init__ default 8.0;
    // exposed for tests if non-default paths are ever validated.
    static BoundariesResult
    findBoundaries(const std::vector<double>& novelty,
                   double                     duration,
                   double                     minSegmentDuration =
                                                  kDefaultMinSegmentDuration);

    // Step 4: per-segment mean of features sliced by frame-converted boundaries.
    // PARITY: structure_analyzer.py:277-296 StructureAnalyzer._compute_segment_embeddings.
    //
    // `features` is [T][nFeat] (same layout as SSM input). Boundaries are
    // converted to frames via `int(b * sr / hopLength)` (Python `int()`
    // truncates toward zero on positive values → `static_cast<int>` in C++),
    // clamped to valid indices. Output is row-major [nSegments × nFeat]
    // float32, matching Python `np.array(embeddings)` on list-of-f32 rows.
    static std::vector<float>
    computeSegmentEmbeddings(const std::vector<std::vector<float>>& features,
                             const std::vector<double>&             boundaries,
                             int                                    sr,
                             int                                    hopLength,
                             int&                                   nSegmentsOut,
                             int&                                   nFeatOut);

    // ------------------------------------------------------------------
    // Session 8 — ADR-022 — clustering + labeling (Option A: preserve
    // Python's `silhouette_score(affinity, ..., metric="precomputed")`
    // diagonal-zero bug which makes the k-search loop dead code; Python
    // always falls through to `_run_clustering(embeddings, min(4, n_segs))`).
    // ------------------------------------------------------------------

    // Step 5a: clipped-cosine affinity on L2-normalized embeddings.
    // PARITY: structure_analyzer.py L310-313 (inside _cluster_segments).
    //   norms = ||embedding||_2 + 1e-8 (per row, keepdims)
    //   normed = embeddings / norms
    //   affinity = clip(normed @ normed.T, 0, 1)  (symmetric, diagonal ~ 1.0)
    // Row-major [nSegs × nSegs] f32 output.
    static std::vector<float>
    computeAffinity(const std::vector<float>& embeddings,
                    int                       nSegs,
                    int                       nFeat);

    struct ClusterResult
    {
        std::vector<std::int64_t> clusterIds;  // length = nSegs
        int                       kUsed = 0;   // k picked (0 for n≤1, 2 for n==2, min(4, n) else)
    };

    // Step 5b: cluster segment embeddings per ADR-022 Option A dispatch.
    // PARITY: structure_analyzer.py:298-348 _cluster_segments — observed
    // behavior, not docstring intent.
    //   n_segs == 0 → {}
    //   n_segs == 1 → [0]
    //   n_segs == 2 → runSpectralClustering(embeddings, 2)   (L305-307)
    //   n_segs ≥ 3  → runSpectralClustering(embeddings, min(4, n_segs))
    //                 (L344, the silhouette-loop fallback that always fires
    //                 per ADR-022 finding).
    static ClusterResult
    clusterSegments(const std::vector<float>& embeddings,
                    int                       nSegs,
                    int                       nFeat);

    // Step 5c: labeled segments from boundaries + cluster_ids + novelty + audio.
    // PARITY: structure_analyzer.py:369-475 _create_segments. Position
    // overrides: i==0 and start < 0.15·duration → "intro"; i==last and
    // end > 0.85·duration → "outro". Label assignment: repeating clusters
    // (count ≥ 2) ranked by mean per-segment RMS energy → chorus/verse/bridge;
    // non-repeating clusters → "bridge". If ALL clusters non-repeating,
    // Counter.most_common rank → chorus/verse/bridge. Confidence:
    //   i == 0 → 0.8 literal
    //   i >  0 → min(1.0, 0.5 + 0.5 · novelty[floor(start/frame_dur)])
    //            with boundary_frame clamped to n_frames - 1.
    // No rounding on start/end.
    //
    // Session 10: aliased to the unified `reamix::analysis::Segment` in
    // StructureResult.h. Original local struct held `std::int64_t clusterId`
    // (camelCase, 64-bit); unified struct uses `int cluster_id` (matching
    // Python + CBMSegmenter + SegmentConsolidation). `ClusterResult::clusterIds`
    // remains `std::int64_t` — that's sklearn's native output type and
    // participates in Hungarian matching against int64 goldens.
    using Segment = reamix::analysis::Segment;

    static std::vector<Segment>
    createSegments(const std::vector<double>&        boundaries,
                   const std::vector<std::int64_t>&  clusterIds,
                   double                             duration,
                   const std::vector<double>&         novelty,
                   const std::vector<float>&          audio,
                   int                                sr);
};

} // namespace reamix::analysis
