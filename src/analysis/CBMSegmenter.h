#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "StructureResult.h"

namespace reamix::analysis {

// Barwise Correlation Block-Matching segmentation (TISMIR 2023).
// Port of `cbm_segmenter.py` (python-source/analysis/cbm_segmenter.py).
//
// Session-5 scope: DP pipeline only — bar-level feature computation +
// autosimilarity + the CBM dynamic-programming core. `label_segments`
// + `cbm_analyze` orchestrator land in session 6.
//
// Precision: bar-level features + autosimilarity run in float32 to match
// the Python production path, where librosa's MFCC/chroma/contrast are
// float32 and `beat_features` arrives at `cbm_analyze` as float32 (see
// structure_analyzer.py:135, feature_extractor.py:232-235). The DP core
// runs in float64 because Python's `_convolution_cost` upcasts via
// `A_block * kernel` (kernel is np.zeros/np.eye default float64 → f32*f64
// = f64), then sums in f64. Bit-identical DP output requires replicating
// this f32→f64 upcast at the convolution step.
class CBMSegmenter
{
public:
    // PARITY: cbm_segmenter.py:156-159 `cbm_segment` default arguments.
    static constexpr int    kDefaultMinSize       = 2;
    static constexpr int    kDefaultMaxSize       = 16;
    static constexpr double kDefaultPenaltyWeight = 1.0;
    static constexpr int    kDefaultNBands        = 7;

    // Bar-index segment pair, half-open [start, end). `end == n_bars` on the
    // last segment of a track. Matches Python tuple (start_bar, end_bar)
    // returned by `cbm_segment`.
    struct BarSegment
    {
        int start;  // inclusive
        int end;    // exclusive
    };

    // Output of step 1 — mapping beat-level features to bar-level averages.
    struct BarFeaturesResult
    {
        // Row-major [nBars × nFeat] float32, one row per bar.
        std::vector<float>  barFeatures;
        int                 nBars = 0;
        int                 nFeat = 0;

        // Bar boundary times in seconds, size n_bars + 1. Last entry is an
        // extrapolated song-end time (cbm_segmenter.py:80-82).
        std::vector<double> barTimes;
    };

    // Step 1: map each downbeat to its nearest beat, deduplicate + sort,
    // and average beat-level features within each resulting bar slab.
    // PARITY: cbm_segmenter.py:35-84 `compute_bar_features`.
    //
    // `features` is row-major [nBeats × nFeat] float32. `beatTimes` and
    // `downbeats` are float64 seconds.
    //
    // Degenerate inputs (`nDownbeats < 2` OR `nBeats < 2`) return empty
    // result (matches Python L51-52).
    static BarFeaturesResult
    computeBarFeatures(const float*  features,
                       int           nBeats,
                       int           nFeat,
                       const double* beatTimes,
                       int           nBeatTimes,
                       const double* downbeats,
                       int           nDownbeats);

    // Step 2: cosine-similarity autosimilarity matrix.
    // PARITY: cbm_segmenter.py:87-97 `compute_bar_autosimilarity`.
    //
    // L2-normalizes each bar row (norm < 1e-8 → norm := 1.0), computes
    // `A = X_norm @ X_norm.T`, clips to [0, 1].
    //
    // Returns row-major [nBars × nBars] float32 (matches Python's f32
    // numpy matmul + clip).
    static std::vector<float>
    computeBarAutosimilarity(const std::vector<float>& barFeatures,
                             int                       nBars,
                             int                       nFeat);

    // Step 3: CBM dynamic programming over the autosimilarity matrix.
    // PARITY: cbm_segmenter.py:154-241 `cbm_segment`.
    //
    // Maximizes Σ (h · seg_len  -  penalty · penaltyWeight · max_conv)
    // over legal (minSize .. maxSize) segmentations of [0, nBars). `h`
    // is the mean of the element-wise product of the A sub-block with a
    // banded kernel (band width = nBands). `max_conv` is the maximum such
    // score taken over all size-8 windows (normalizer, cbm_segmenter.py
    // L186-197). Penalty favors 8-bar segments, then 4-bar multiples,
    // then any even length.
    //
    // Returns the bar-index segmentation of `[0, nBars)` as a list of
    // half-open `[start, end)` pairs covering the whole range (Python
    // guarantees L227-241 segment reversal + cover-from-zero patch).
    static std::vector<BarSegment>
    cbmSegment(const std::vector<float>& A,
               int                       nBars,
               int                       minSize       = kDefaultMinSize,
               int                       maxSize       = kDefaultMaxSize,
               double                    penaltyWeight = kDefaultPenaltyWeight,
               int                       nBands        = kDefaultNBands);

    // Labeled segment record. `confidence` is a constant 0.8 placeholder
    // from Python L365 (cluster-quality score is not yet computed). Times
    // are rounded to 3 decimals via `snprintf("%.3f")` + `strtod` to match
    // CPython's dtoa-based `round(x, 3)` (FP-multiply by 1000 would drift
    // at 0.0005 boundaries under some rounding modes).
    //
    // Session 10: the previously-local `Segment` struct is now an alias to
    // the unified `reamix::analysis::Segment` in StructureResult.h. Field
    // set + semantics are identical; CBMSegmenter always sets `confidence`
    // explicitly (see CBMSegmenter.cpp:599) so the unified struct's default
    // `confidence = 0.0` does not change behavior.
    using Segment = reamix::analysis::Segment;

    // Full-pipeline output. Python's `cbm_analyze` returns
    // `Optional[List[dict]]` = `None` on degenerate input (n_bars < 4 OR
    // len(bar_segments) < 3). C++ expresses the None distinction via
    // `success = false` + empty vectors — callers must check `success`
    // before indexing into segments/boundaries.
    struct Result
    {
        bool                    success = false;
        std::vector<Segment>    segments;
        // Boundaries array per Python build:
        // `[segs[0].start, segs[0].end, segs[1].end, ..., segs[n-1].end]`
        // length = n_segments + 1. f64 seconds.
        std::vector<double>     boundaries;
    };

    // Label each bar-index segment as intro/verse/chorus/bridge/outro.
    // PARITY: cbm_segmenter.py:248-369 `label_segments`.
    //
    // `barFeatures` / `nBars` / `nFeat` are the session-5 step-1 output
    // (f32 row-major). `barTimes` has n_bars + 1 entries (step-1 output).
    // `barEnergy` is f64 per-bar RMS from `cbm_analyze` step 4; pass an
    // empty vector to disable energy-based ranking (all repeating groups
    // tie at energy=0, stable-sort falls back to insertion order).
    //
    // Times in returned Segments are rounded to 3 decimals; `confidence`
    // is a fixed 0.8; `cluster_id` matches group index assigned during
    // greedy single-linkage clustering.
    static std::vector<Segment>
    labelSegments(const std::vector<BarSegment>& barSegments,
                  const std::vector<float>&      barFeatures,
                  int                            nBars,
                  int                            nFeat,
                  const std::vector<double>&     barTimes,
                  const std::vector<double>&     barEnergy);

    // Full CBM pipeline. PARITY: cbm_segmenter.py:376-443 `cbm_analyze`.
    //
    // Degenerate-case handling:
    //   n_bars < 4         → `Result{ success=false, {}, {} }` (Python L406-408)
    //   len(segments) < 3  → `Result{ success=false, {}, {} }` (Python L418-420)
    //
    // `audioMono` + `nSamples` + `sampleRate`: mono f32 audio at the same
    // SR used for feature extraction. Pass `nullptr` / 0 / 0 to skip the
    // bar-energy step (labels fall back to insertion-order ranking).
    static Result
    cbmAnalyze(const float*  features,
               int           nBeats,
               int           nFeat,
               const double* beatTimes,
               int           nBeatTimes,
               const double* downbeats,
               int           nDownbeats,
               const float*  audioMono,
               std::size_t   nSamples,
               int           sampleRate,
               int           minBars       = kDefaultMinSize,
               int           maxBars       = kDefaultMaxSize,
               double        penaltyWeight = kDefaultPenaltyWeight,
               int           nBands        = kDefaultNBands);
};

} // namespace reamix::analysis
