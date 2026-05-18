#pragma once

#include <cstddef>
#include <vector>

namespace reamix::analysis {

// Beat-synchronous feature aggregation, matching
// `librosa.util.sync(data, idx, aggregate=np.mean, pad=True, axis=-1)`
// on librosa 0.11.0. Concretely, for a 2-D feature matrix `data` of shape
// [D × T] and an integer `beatFrames` list, aggregate frames between
// consecutive beat boundaries by arithmetic mean, producing [D × nSlices].
//
// Boundary construction mirrors librosa's `fix_frames(frames, x_min=0,
// x_max=T, pad=True)` + `index_to_slice`:
//
//   1. Clip each beat frame to [0, T].
//   2. Prepend {0, T} to the frame set.
//   3. `np.unique(...).astype(int)` — sorted, deduplicated boundaries.
//   4. Slices are `[b[i], b[i+1])` for i in 0..len(b)-2.
//
// Result invariants:
//   - For typical beat lists (all beats strictly inside (0, T)),
//     `nSlices == len(unique_sorted_beatFrames) + 1`. Python call sites
//     then truncate with `synced[:, :n_beats]` to drop the trailing
//     "after-last-beat" segment — this truncation is the caller's
//     responsibility, NOT this class's.
//   - `nSlices >= 1` whenever `T >= 1` (np.unique guarantees no empty
//     slices, so np.mean never sees a zero-length axis).
//
// Aggregation is done in float64 with a naive sequential sum. NumPy uses
// pairwise summation under the hood for `np.mean` on large arrays; for
// typical beat-sized slices (~20-40 frames) the discrepancy is well under
// 1e-12 on well-behaved mel/chroma/contrast data. The formal parity gate
// is L∞ ≤ 1e-3 (VALIDATION.md phase-2), with the expected realized floor
// four-plus orders under that.
//
// PARITY (function sources on librosa 0.11.0):
//   - `librosa/util/utils.py::sync`       (L1549)
//   - `librosa/util/utils.py::index_to_slice` (L1490)
//   - `librosa/util/utils.py::fix_frames` (L580)
//
// Call sites in the Python reference (reused across 2-D features and 1-D
// side-channels via `[np.newaxis, :]` wrapping):
//   - `python-source/analysis/feature_extractor.py::_beat_sync_features` (L465)
//   - `python-source/analysis/feature_extractor.py::_beat_sync_1d`       (L477)
//
// The 1-D case (for RMS / onset / centroid) is handled by callers that
// shape a single-row matrix [1 × T] → result [1 × nSlices], exactly
// mirroring the Python `values[np.newaxis, :]` convention.
class BeatSync
{
public:
    struct Result
    {
        std::vector<double> data;   // [D × nSlices], row-major.
        std::size_t nSlices = 0;
    };

    // Aggregate `data` (row-major, [D × T]) over beat boundaries.
    //
    // `beatFrames` may be unordered, contain duplicates, or fall outside
    // [0, T] — fix_frames normalizes silently. Negative entries are
    // skipped (librosa raises on those; phase-2 test inputs are always
    // non-negative, so we take the permissive branch).
    static Result sync(const double* data,
                       std::size_t D,
                       std::size_t T,
                       const std::vector<int>& beatFrames);
};

} // namespace reamix::analysis
