#include "remix/SegmentData.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace reamix::remix {

SegmentData
computeSegmentData(int                      n_beats,
                   const analysis::Segment* segments,
                   int                      n_segments,
                   const double*            beat_times,
                   const float*             features,
                   int                      n_features)
{
    SegmentData out;

    if (segments == nullptr || n_segments == 0) {
        // Python viterbi_dp.py L400-405 empty-segments path.
        out.beat_to_segment.assign(static_cast<std::size_t>(n_beats), 0);
        out.seg_sim.assign(1, 1.0);
        out.n_segs = 1;
        return out;
    }

    out.n_segs = n_segments;
    out.beat_to_segment.assign(static_cast<std::size_t>(n_beats), 0);

    // Python L411-416: last segment whose [start, end) contains beat wins.
    for (int s = 0; s < n_segments; ++s) {
        const double start = segments[s].start;
        const double end   = segments[s].end;
        for (int b = 0; b < n_beats; ++b) {
            if (beat_times[b] >= start && beat_times[b] < end) {
                out.beat_to_segment[b] = static_cast<std::int64_t>(s);
            }
        }
    }

    // Python L418-421: boundary set.
    for (int b = 1; b < n_beats; ++b) {
        if (out.beat_to_segment[b] != out.beat_to_segment[b - 1]) {
            out.boundary_beats.insert(b);
        }
    }

    // Python L423-434: seg_sim baseline (label-based).
    std::vector<std::string> labels(static_cast<std::size_t>(n_segments));
    for (int s = 0; s < n_segments; ++s) {
        std::string lab = segments[s].label;
        std::transform(lab.begin(), lab.end(), lab.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        labels[s] = std::move(lab);
    }

    out.seg_sim.assign(static_cast<std::size_t>(n_segments) * n_segments, 0.0);
    for (int i = 0; i < n_segments; ++i) {
        for (int j = 0; j < n_segments; ++j) {
            double v;
            if (i == j)                       v = 1.0;
            else if (labels[i] == labels[j])  v = 0.9;
            else                              v = 0.2;
            out.seg_sim[static_cast<std::size_t>(i) * n_segments + j] = v;
        }
    }

    // Python L436-460: refine upper-triangle with feature cosine blend.
    if (features != nullptr && n_features > 0) {
        // Per-segment mean feature vector + norm, f64 (matches numpy
        // `features[mask_i].mean(axis=0)` semantics).
        std::vector<std::vector<double>> seg_mean(static_cast<std::size_t>(n_segments));
        std::vector<double>              seg_norm(static_cast<std::size_t>(n_segments), 0.0);
        std::vector<int>                 seg_count(static_cast<std::size_t>(n_segments), 0);

        for (int b = 0; b < n_beats; ++b) {
            seg_count[static_cast<std::size_t>(out.beat_to_segment[b])]++;
        }

        for (int s = 0; s < n_segments; ++s) {
            if (seg_count[s] == 0) continue;
            std::vector<double> mean(static_cast<std::size_t>(n_features), 0.0);
            for (int b = 0; b < n_beats; ++b) {
                if (out.beat_to_segment[b] != static_cast<std::int64_t>(s)) continue;
                const float* row = features + static_cast<std::size_t>(b) * n_features;
                for (int k = 0; k < n_features; ++k) {
                    mean[k] += static_cast<double>(row[k]);
                }
            }
            const double inv = 1.0 / static_cast<double>(seg_count[s]);
            for (int k = 0; k < n_features; ++k) mean[k] *= inv;
            double n2 = 0.0;
            for (int k = 0; k < n_features; ++k) n2 += mean[k] * mean[k];
            seg_norm[s] = std::sqrt(n2);
            seg_mean[s] = std::move(mean);
        }

        // Python L447-460: blend 0.6 × label_sim + 0.4 × max(0, cos).
        for (int i = 0; i < n_segments; ++i) {
            if (seg_count[i] == 0 || seg_norm[i] < COSINE_DEGENERATE_FLOOR) continue;
            for (int j = i + 1; j < n_segments; ++j) {
                if (seg_count[j] == 0 || seg_norm[j] < COSINE_DEGENERATE_FLOOR) continue;

                double dot = 0.0;
                for (int k = 0; k < n_features; ++k) dot += seg_mean[i][k] * seg_mean[j][k];
                const double cos_sim = dot / (seg_norm[i] * seg_norm[j]);
                const double clipped = std::max(0.0, cos_sim);

                const std::size_t idx_ij = static_cast<std::size_t>(i) * n_segments + j;
                const std::size_t idx_ji = static_cast<std::size_t>(j) * n_segments + i;
                const double label_sim = out.seg_sim[idx_ij];
                const double blended   = 0.6 * label_sim + 0.4 * clipped;
                out.seg_sim[idx_ij] = blended;
                out.seg_sim[idx_ji] = blended;
            }
        }
    }

    return out;
}

} // namespace reamix::remix
