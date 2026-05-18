#include "analysis/BeatSync.h"

#include <set>

namespace reamix::analysis {

namespace {

// Mirrors `librosa.util.fix_frames(frames, x_min=0, x_max=T, pad=True)`:
// clip to [0, T], prepend {0, T}, dedupe + sort. std::set handles both
// the sort and the uniqueness in one pass.
//
// PARITY: librosa/util/utils.py::fix_frames (L580).
std::vector<std::size_t>
fixFrames(const std::vector<int>& frames, std::size_t T)
{
    std::set<std::size_t> s;
    s.insert(0);
    s.insert(T);
    for (int f : frames) {
        if (f < 0) continue;  // librosa raises; we skip (phase-2 inputs are
                              // never negative). Port stays total.
        const std::size_t v = (static_cast<std::size_t>(f) > T)
                                ? T
                                : static_cast<std::size_t>(f);
        s.insert(v);
    }
    return std::vector<std::size_t>(s.begin(), s.end());
}

} // namespace

BeatSync::Result
BeatSync::sync(const double* data,
               std::size_t D,
               std::size_t T,
               const std::vector<int>& beatFrames)
{
    const std::vector<std::size_t> boundaries = fixFrames(beatFrames, T);
    const std::size_t nSlices =
        boundaries.size() > 0 ? boundaries.size() - 1 : 0;

    Result r;
    r.nSlices = nSlices;
    r.data.assign(D * nSlices, 0.0);

    if (T == 0 || D == 0 || nSlices == 0) return r;

    for (std::size_t i = 0; i < nSlices; ++i) {
        const std::size_t lo = boundaries[i];
        const std::size_t hi = boundaries[i + 1];
        const std::size_t n  = hi - lo;
        // fix_frames returns unique sorted → n >= 1 for every slice.
        // np.mean on empty axis would emit NaN (librosa/feature_extractor
        // does not guard this). Port inherits the invariant instead of
        // adding a dead branch.
        const double invN = 1.0 / static_cast<double>(n);

        for (std::size_t d = 0; d < D; ++d) {
            const double* row = data + d * T;
            double sum = 0.0;
            for (std::size_t t = lo; t < hi; ++t) sum += row[t];
            r.data[d * nSlices + i] = sum * invN;
        }
    }
    return r;
}

} // namespace reamix::analysis
