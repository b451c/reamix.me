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

    // PARITY: recurrence.py:195 `threshold = 0.1`. Diagonal-hit threshold on
    // R cells; the only remaining consumer is remix/RepetitionPrior (ADR-115 E4).
    static constexpr double kDiagonalThreshold = 0.1;
};

} // namespace reamix::analysis
