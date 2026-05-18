#pragma once

#include <cstddef>
#include <vector>

namespace reamix::analysis {

// Combined recurrence matrix for music structure and transition analysis.
// Port of `build_recurrence_matrix`
// (python-source/analysis/recurrence.py L32-146) on librosa 0.11.0 + sklearn:
//
//   R = 0.15·R_feat + 0.35·R_chroma + 0.50·R_homogeneity
//
// R_feat       : mutual k-NN on full beat feature vectors (global timbral).
// R_chroma     : mutual k-NN on chroma sub-vectors (harmonic similarity).
// R_homogeneity: Gaussian kernel on consecutive-beat distances (|i-j|=1 only).
//
// Input precision note: the Python production path passes FLOAT32 features
// in and numpy broadcasts up to FLOAT64 internally (sklearn kneighbors
// returns f64 distances, np.exp returns f64). C++ port upcasts to f64 once
// and does all distance math in f64 — simplifies the path and lands within
// f32-ULP of Python (~1e-6 drift, 3 orders below the 1e-3 parity gate).
//
// Vestigial-argument note: Python signature is
// `build_recurrence_matrix(features, beat_times, k_neighbors=10)` but the
// `beat_times` argument is UNUSED inside the function body (source audit
// per Hard Rule #8 — recurrence.py:117-146 `_homogeneity_matrix` computes
// sigma² from row diffs of X_norm, never indexes beat_times). Port drops
// the parameter.
class Recurrence
{
public:
    struct Result
    {
        // Primary output: weighted combined matrix, row-major [nBeats × nBeats].
        // R(i,j) ∈ [0, 1]. Diagonal is zero (mutual-kNN excludes self; homo
        // only fills |i-j|=1).
        std::vector<double> R;

        // Component matrices — bisection aids for parity debugging. Not
        // consumed by downstream production code; the dump tool emits the
        // same three for golden-side comparison. Row-major [nBeats × nBeats].
        std::vector<double> rFeat;
        std::vector<double> rChroma;
        std::vector<double> rHomo;

        int nBeats = 0;
    };

    // PARITY: recurrence.py:27-29 weights and config.py:16-17 chroma/contrast
    // dimensionality. Feature matrix layout at input is assumed to be
    // [MFCC(variable), Chroma(12), Contrast(7)] per
    // python-source/config.py:13-14 — chromaRange picks up the 12-dim chroma
    // sub-vector from any feature width (59 std, 39 fast, or future variants).
    static constexpr int    kDefaultK       = 10;
    static constexpr double kWFeatures      = 0.15;
    static constexpr double kWChroma        = 0.35;
    static constexpr double kWHomogeneity   = 0.50;
    static constexpr int    kNChromaDims    = 12;
    static constexpr int    kNContrastDims  = 7;
    static constexpr double kMuFloor        = 1e-8;  // Python L99, L138

    // Build combined recurrence matrix from per-beat features.
    //
    // `features` is row-major [nBeats × nFeat] float32 — matches
    // `FeatureExtractor::Result::features`. Rows may or may not be
    // L2-normalized (internal L2-normalize is applied regardless; idempotent
    // on pre-normalized rows).
    //
    // `kNeighbors` clamped to nBeats-1 internally per recurrence.py:89.
    //
    // n < 4: returns identity R (and zero components) per recurrence.py:48-49.
    static Result build(const float* features,
                        int nBeats,
                        int nFeat,
                        int kNeighbors = kDefaultK);

    // ---------------------------------------------------------------------
    // findDiagonals — bar-aligned diagonal search in a recurrence submatrix.
    //
    // Port of `recurrence.py::find_diagonals` (L149-228, 2026-04-22). Added
    // phase-4 session 24 for `TransitionPrescreen` which locates structurally
    // repeated patterns near segment boundaries; the caller (prescreen)
    // verifies the best diagonals with waveform xcorr.
    //
    // Algorithm:
    //   1. Iterate 3 bar-aligned offsets: {Δ - bar, Δ, Δ + bar} where
    //      Δ = col_start - row_start. Skip offset == 0 (self-diagonal).
    //   2. For each offset, trace the diagonal r ∈ [max(row_start, 0),
    //      min(row_end, n)); include (r, c=r+offset) only if
    //      col_start ≤ c < col_end AND 0 ≤ c < n.
    //   3. If diagonal length ≥ min_length, find the BEST contiguous run of
    //      values above `kDiagonalThreshold` (0.1). If best run length ≥
    //      min_length, emit (r_mid, c_mid, length, mean_similarity) where
    //      mid = best_run_start + best_run_len // 2 (Python floor division)
    //      relative to the trace array.
    //   4. Sort results DESC by `length × mean_similarity`.
    //
    // Returns list of (r_mid, c_mid, diag_len, mean_sim) tuples. C++ signature
    // uses a POD struct for cache-friendly flat array.
    //
    // Numerical parity notes:
    //   - Python uses numpy array comparison `vals > threshold`; C++ mirrors
    //     with direct double compare (vals are f64 R cells from build()).
    //   - Python accumulates `run_sum` with Python double add — C++ uses
    //     same-order f64 add. -ffp-contract=off required on test target for
    //     bitwise parity of the mean_sim division.
    //   - Python `range(start, stop, step)` with `stop = Δ + bar_size + 1`
    //     and `step = bar_size` produces 3 values for bar_size ≥ 1: {Δ-bar,
    //     Δ, Δ+bar}. Port enumerates these 3 explicitly (clearer than Python
    //     range arithmetic).
    //
    // Source-of-truth: recurrence.py:149-228 (2026-04-22).
    struct Diagonal
    {
        int    rMid;    // row index of diagonal midpoint
        int    cMid;    // col index of diagonal midpoint
        int    length;  // length in beats of the matched run
        double meanSim; // average R value along the best run
    };

    // PARITY: recurrence.py:195 `threshold = 0.1`.
    static constexpr double kDiagonalThreshold = 0.1;

    // `R` is row-major [n × n] from Recurrence::Result::R. `n` is the matrix
    // side (nBeats). Caller clamps row/col ranges; internal max/min enforce
    // `[0, n)` bounds exactly like the Python `0 <= c < n` guard.
    static std::vector<Diagonal>
    findDiagonals(const double* R,
                  int           n,
                  int           rowStart,
                  int           rowEnd,
                  int           colStart,
                  int           colEnd,
                  int           minLength = 4,
                  int           barSize   = 4);
};

} // namespace reamix::analysis
