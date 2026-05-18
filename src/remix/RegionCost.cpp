#include "remix/RegionCost.h"

#include "remix/Quality.h"
#include "dsp/WaveformXcorr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace reamix::remix {

namespace {

// ---------------------------------------------------------------------------
// L2-normalized row matrix (f64 variant — region uses f64 accumulator to
// match Python `np.linalg.norm` with dtype defaulting to the input).
// Python region_cost.py:163-166 + 173-175: `np.linalg.norm(chroma, axis=1,
// keepdims=True)` on f32 input returns f32; we promote to f64 for dot
// products below to match the f32→f64 widening at `clip(... @ ...)`.
// Actually we keep f32 to match Python more faithfully — `cn @ cn.T` in
// numpy returns f32 when both operands are f32.
// ---------------------------------------------------------------------------
void l2NormalizeRowsF32(const float* input,
                         int          n_rows,
                         int          n_cols,
                         std::vector<float>& output)
{
    output.assign(static_cast<std::size_t>(n_rows) * n_cols, 0.0f);
    for (int i = 0; i < n_rows; ++i) {
        const float* src = input + static_cast<std::size_t>(i) * n_cols;
        float*       dst = output.data() + static_cast<std::size_t>(i) * n_cols;

        float sumSq = 0.0f;
        for (int k = 0; k < n_cols; ++k) sumSq += src[k] * src[k];
        float norm = std::sqrt(sumSq);
        if (norm == 0.0f) norm = 1.0f;  // region_cost.py:164,197

        for (int k = 0; k < n_cols; ++k) dst[k] = src[k] / norm;
    }
}

// Row-shifted chroma distance for region subset.
// Port of `_precompute_region_matrices` chroma branch (region_cost.py:154-170).
// Output size = n × n (f64, row-major).
std::vector<double> precomputeRegionChromaD(const float* features,
                                             int          entry_beat,
                                             int          exit_beat,
                                             int          n_features)
{
    const int n = exit_beat - entry_beat;
    std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);

    // Slice chroma dims. chroma_range() at region_cost.py:161.
    const auto [cs, ce] = chromaRange(n_features);
    const int  n_chroma = ce - cs;

    // Extract chroma rows into contiguous buffer.
    std::vector<float> chroma(static_cast<std::size_t>(n) * n_chroma, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* src = features + static_cast<std::size_t>(entry_beat + i) * n_features + cs;
        float*       dst = chroma.data() + static_cast<std::size_t>(i) * n_chroma;
        for (int k = 0; k < n_chroma; ++k) dst[k] = src[k];
    }

    // L2-normalize rows in f32.
    std::vector<float> cn;
    l2NormalizeRowsF32(chroma.data(), n, n_chroma, cn);

    // S[i, j] = cn[i] · cn[j]  (f32 matmul — matches Python `cn @ cn.T` on f32 input).
    // Clip to [-1, 1] in f32 then widen to f64 for `1 - S`.
    // Session-18 TransitionCost established the naive f32 triple-loop as the
    // canonical bitwise-matching pattern for `cn @ cn.T` vs numpy.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            const float* ri = cn.data() + static_cast<std::size_t>(i) * n_chroma;
            const float* rj = cn.data() + static_cast<std::size_t>(j) * n_chroma;
            for (int k = 0; k < n_chroma; ++k) s += ri[k] * rj[k];
            // numpy clip on f32.
            s = std::clamp(s, -1.0f, 1.0f);
            // Widen to f64 for 1.0 - s then store.
            out[static_cast<std::size_t>(i) * n + j] = 1.0 - static_cast<double>(s);
        }
    }

    // Row shift: chroma_D[:-1, :] = chroma_D[1:, :]; last row = 1.0.
    // region_cost.py:168-170.
    if (n > 1) {
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < n; ++j) {
                out[static_cast<std::size_t>(i) * n + j] =
                    out[static_cast<std::size_t>(i + 1) * n + j];
            }
        }
    }
    // Last row = 1.0 (region_cost.py:170).
    for (int j = 0; j < n; ++j) {
        out[static_cast<std::size_t>(n - 1) * n + j] = 1.0;  // CLEAN (C6)
    }

    return out;
}

// Row-shifted successor similarity for region subset (full-feature cosine).
// Port of `_precompute_region_matrices` S branch (region_cost.py:172-179).
std::vector<double> precomputeRegionSuccessorSim(const float* features,
                                                  int          entry_beat,
                                                  int          exit_beat,
                                                  int          n_features)
{
    const int n = exit_beat - entry_beat;
    std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);

    // Extract full-feature rows.
    std::vector<float> feats(static_cast<std::size_t>(n) * n_features, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* src = features + static_cast<std::size_t>(entry_beat + i) * n_features;
        float*       dst = feats.data() + static_cast<std::size_t>(i) * n_features;
        for (int k = 0; k < n_features; ++k) dst[k] = src[k];
    }

    // L2-normalize rows in f32.
    std::vector<float> normed;
    l2NormalizeRowsF32(feats.data(), n, n_features, normed);

    // S = clip(normed @ normed.T, -1, 1).
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            const float* ri = normed.data() + static_cast<std::size_t>(i) * n_features;
            const float* rj = normed.data() + static_cast<std::size_t>(j) * n_features;
            for (int k = 0; k < n_features; ++k) s += ri[k] * rj[k];
            s = std::clamp(s, -1.0f, 1.0f);
            out[static_cast<std::size_t>(i) * n + j] = static_cast<double>(s);
        }
    }

    // Row shift: S[:-1, :] = S[1:, :]; last row = 0.0.
    // region_cost.py:177-179.
    if (n > 1) {
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < n; ++j) {
                out[static_cast<std::size_t>(i) * n + j] =
                    out[static_cast<std::size_t>(i + 1) * n + j];
            }
        }
    }
    for (int j = 0; j < n; ++j) {
        out[static_cast<std::size_t>(n - 1) * n + j] = 0.0;  // CLEAN (C7)
    }

    return out;
}

// Edge splice similarity matrix (region subset).
// Port of `_precompute_edge_splice` (region_cost.py:184-199).
// Returns empty vector when edge_end/edge_start are null.
std::vector<double> precomputeEdgeSplice(const float* edge_end,
                                          const float* edge_start,
                                          int          entry_beat,
                                          int          exit_beat,
                                          int          n_edge_features)
{
    if (edge_end == nullptr || edge_start == nullptr || n_edge_features <= 0) {
        return {};
    }

    const int n = exit_beat - entry_beat;
    std::vector<float> ee(static_cast<std::size_t>(n) * n_edge_features, 0.0f);
    std::vector<float> es(static_cast<std::size_t>(n) * n_edge_features, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* se = edge_end + static_cast<std::size_t>(entry_beat + i) * n_edge_features;
        const float* ss = edge_start + static_cast<std::size_t>(entry_beat + i) * n_edge_features;
        float*       de = ee.data() + static_cast<std::size_t>(i) * n_edge_features;
        float*       ds = es.data() + static_cast<std::size_t>(i) * n_edge_features;
        for (int k = 0; k < n_edge_features; ++k) { de[k] = se[k]; ds[k] = ss[k]; }
    }

    std::vector<float> ee_n, es_n;
    l2NormalizeRowsF32(ee.data(), n, n_edge_features, ee_n);
    l2NormalizeRowsF32(es.data(), n, n_edge_features, es_n);

    std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            const float* ri = ee_n.data() + static_cast<std::size_t>(i) * n_edge_features;
            const float* rj = es_n.data() + static_cast<std::size_t>(j) * n_edge_features;
            for (int k = 0; k < n_edge_features; ++k) s += ri[k] * rj[k];
            s = std::clamp(s, -1.0f, 1.0f);
            out[static_cast<std::size_t>(i) * n + j] = static_cast<double>(s);
        }
    }
    return out;
}

// Edge dB arrays for region subset.
// Port of `_precompute_edge_energy` (region_cost.py:202-213).
// Returns (db_end, db_start) — each empty when edge_rms_* is null.
std::pair<std::vector<double>, std::vector<double>>
precomputeEdgeEnergy(const double* edge_rms_end,
                     const double* edge_rms_start,
                     int           entry_beat,
                     int           exit_beat)
{
    if (edge_rms_end == nullptr || edge_rms_start == nullptr) {
        return {{}, {}};
    }
    const int n = exit_beat - entry_beat;
    std::vector<double> db_end(static_cast<std::size_t>(n), 0.0);
    std::vector<double> db_start(static_cast<std::size_t>(n), 0.0);
    // region_cost.py:211-212:
    //   20.0 * np.log10(np.maximum(arr, 1e-6))
    // CLEAN (C8) — shared pattern with TransitionCost session 18.
    for (int i = 0; i < n; ++i) {
        const double re = edge_rms_end[entry_beat + i];
        const double rs = edge_rms_start[entry_beat + i];
        db_end[i]   = 20.0 * std::log10(std::max(re, 1e-6));
        db_start[i] = 20.0 * std::log10(std::max(rs, 1e-6));
    }
    return {std::move(db_end), std::move(db_start)};
}

// Beat labels for region subset. Port of `_region_beat_labels` (region_cost.py:216-236).
std::vector<std::string> regionBeatLabels(const analysis::Segment* segments,
                                           int                      n_segments,
                                           const double*            beat_times,
                                           int                      entry_beat,
                                           int                      exit_beat,
                                           int                      n_total)
{
    const int n_region = exit_beat - entry_beat;
    std::vector<std::string> labels(static_cast<std::size_t>(n_region), "unknown");
    if (segments == nullptr || n_segments <= 0) return labels;

    for (int s = 0; s < n_segments; ++s) {
        const auto& seg = segments[s];
        std::string label = seg.label;
        // Python `seg.get("label", "unknown").lower()`. C++ lowercase ASCII.
        for (char& ch : label) {
            if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
        }
        const double start_t = seg.start;
        const double end_t   = seg.end;
        for (int ri = 0; ri < n_region; ++ri) {
            const int abs_b = entry_beat + ri;
            if (abs_b < n_total && start_t <= beat_times[abs_b] && beat_times[abs_b] < end_t) {
                labels[static_cast<std::size_t>(ri)] = label;
            }
        }
    }
    return labels;
}

// Region-relative downbeat + pre-downbeat index sets computed inline in
// computeRegionCosts (needs access to the full-track beat_times via n_total
// for the argmin loop at region_cost.py:251). No standalone helper.

// Max lag in samples for waveform xcorr. Port of `_waveform_max_lag`
// (region_cost.py:260-265).
int waveformMaxLag(int waveform_sample_rate)
{
    if (waveform_sample_rate <= 0) return 0;
    // region_cost.py:264: DEFAULT_CONFIG.remix.waveform_alignment_max_shift_ms = 30.0
    // CLEAN (C9) — double-citation agreement with TransitionCost session 18.
    constexpr double max_lag_ms = 30.0;  // config.py:165
    // region_cost.py:265: int(max_lag_ms * sr / 1000.0)
    return static_cast<int>(max_lag_ms * static_cast<double>(waveform_sample_rate) / 1000.0);  // CLEAN (C10)
}

// Safe window mean for context similarity.
// Port of `_safe_window_mean` (region_cost.py:423-431).
std::vector<double> safeWindowMean(const float* feats_region,
                                    int          n_region,
                                    int          n_dim,
                                    int          lo,
                                    int          hi)
{
    std::vector<double> out(static_cast<std::size_t>(n_dim), 0.0);
    const int lo_c = std::max(0, lo);
    const int hi_c = std::min(n_region, hi);
    if (lo_c >= hi_c) return out;  // all zeros
    const int count = hi_c - lo_c;
    // Mean over [lo_c, hi_c) rows.
    for (int i = lo_c; i < hi_c; ++i) {
        const float* row = feats_region + static_cast<std::size_t>(i) * n_dim;
        for (int k = 0; k < n_dim; ++k) out[k] += static_cast<double>(row[k]);
    }
    const double inv = 1.0 / static_cast<double>(count);
    for (int k = 0; k < n_dim; ++k) out[k] *= inv;
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
RegionCostResult computeRegionCosts(const RegionCostInputs& in)
{
    const int n_total  = in.n_total;
    const int n_region = in.exit_beat - in.entry_beat;

    RegionCostResult out;
    out.n_region = n_region;

    // region_cost.py:77-78 short-circuit.
    if (n_region < 2) {
        const int n_safe = std::max(1, n_region);
        out.region_W.assign(static_cast<std::size_t>(n_safe) * n_safe, INF);  // CLEAN
        out.n_region = n_region;
        return out;
    }

    // Pre-computed matrices.
    std::vector<double> chroma_D = precomputeRegionChromaD(
        in.features, in.entry_beat, in.exit_beat, in.n_features);
    std::vector<double> successor_sim = precomputeRegionSuccessorSim(
        in.features, in.entry_beat, in.exit_beat, in.n_features);
    std::vector<double> edge_splice = precomputeEdgeSplice(
        in.edge_features_end, in.edge_features_start,
        in.entry_beat, in.exit_beat, in.n_edge_features);
    auto [edge_db_end, edge_db_start] = precomputeEdgeEnergy(
        in.edge_rms_end, in.edge_rms_start, in.entry_beat, in.exit_beat);
    std::vector<std::string> beat_labels = regionBeatLabels(
        in.segments, in.n_segments, in.beat_times,
        in.entry_beat, in.exit_beat, n_total);

    // ADR-044: auto-path Region Remix passes n_segments == 0. label_match
    // is naturally 0 (regionBeatLabels returns empty labels), but section_sim
    // and SPAN_PENALTY need explicit gating below — see corresponding sites.
    const bool noStructure = (in.n_segments <= 0);

    // Region-relative downbeat sets.
    // argmin over FULL beat_times (region_cost.py:251), then filter to [entry, exit).
    std::set<int> db_set;
    if (in.downbeats != nullptr && in.n_downbeats > 0) {
        for (int k = 0; k < in.n_downbeats; ++k) {
            const double dbt = in.downbeats[k];
            int    abs_idx = 0;
            double best    = std::abs(in.beat_times[0] - dbt);
            for (int b = 1; b < n_total; ++b) {
                const double d = std::abs(in.beat_times[b] - dbt);
                if (d < best) { best = d; abs_idx = b; }
            }
            if (in.entry_beat <= abs_idx && abs_idx < in.exit_beat) {
                db_set.insert(abs_idx - in.entry_beat);
            }
        }
    } else if (n_region > 0) {
        // Fallback: synthesized downbeats region_cost.py:254-255.
        for (int ri = 0; ri < n_region; ri += in.time_signature) {  // CLEAN (CALLER OVERRIDE)
            db_set.insert(ri);
        }
    }
    std::set<int> pre_db_set;
    for (int db : db_set) if (db > 0) pre_db_set.insert(db - 1);

    // Track-level vocal detection. region_cost.py:98-102.
    bool track_has_vocals = false;
    if (in.vocal_activity != nullptr && n_total > 0) {
        double max_va = 0.0;
        for (int i = 0; i < n_total; ++i) max_va = std::max(max_va, in.vocal_activity[i]);
        track_has_vocals = max_va >= TRACK_VOCAL_THRESHOLD;  // CLEAN (sesja 18)
    }

    // Waveform xcorr setup.
    const int  max_lag_samples = waveformMaxLag(in.waveform_sample_rate);
    const bool has_waveforms =
        in.boundary_waveforms != nullptr
        && in.n_boundary_waveforms >= n_total
        && max_lag_samples > 0;

    // ADR-064 (sesja 75): pre-compute normalised onset over the full track
    // (n_total length, indexed by absolute abs_i / abs_j). Empty when
    // onset_strength was not provided → qi.transient_continuity stays
    // nullopt → null-guard drops contribution → bit-exact parity preserved.
    const std::vector<double> onset_norm =
        computeOnsetNorm(in.onset_strength, n_total);

    // ADR-066 (sesja 77): pre-compute MFCC continuity matrix over the full
    // track (n_total × n_total, indexed by abs_i / abs_j). Empty when
    // features were not provided → qi.mfcc_continuity stays nullopt → parity
    // preserved.
    const std::vector<double> mfcc_continuity_matrix =
        computeMfccContinuityMatrix(in.features, n_total, in.n_features);

    // ADR-080 RESCOPE + ADR-083 (sesja 92): pre-compute full-mix chroma
    // continuity matrix once over the full track. Indexed by abs_i / abs_j.
    // Empty when features absent → qi.full_mix_chroma_continuity stays
    // nullopt → Tone slider blend bypassed → bit-exact baseline.
    const auto [chroma_start, chroma_end] = chromaRange(in.n_features);
    const int  n_chroma = chroma_end - chroma_start;
    std::vector<float>  chroma_slice;
    std::vector<double> chroma_continuity_matrix;
    if (n_chroma > 0 && in.features != nullptr) {
        chroma_slice.resize(static_cast<std::size_t>(n_total) * static_cast<std::size_t>(n_chroma));
        for (int i = 0; i < n_total; ++i) {
            const float* src = in.features
                             + static_cast<std::size_t>(i) * in.n_features
                             + chroma_start;
            float* dst = chroma_slice.data()
                       + static_cast<std::size_t>(i) * n_chroma;
            for (int k = 0; k < n_chroma; ++k) dst[k] = src[k];
        }
        chroma_continuity_matrix =
            computeChromaContinuityMatrix(chroma_slice.data(), n_total, n_chroma);
    }

    // Build cost matrix (region_cost.py:113-148).
    out.region_W.assign(static_cast<std::size_t>(n_region) * n_region, INF);  // CLEAN

    // Sequential costs (region_cost.py:117-118).
    for (int ri = 0; ri < n_region - 1; ++ri) {
        const double cd = chroma_D[static_cast<std::size_t>(ri) * n_region + (ri + 1)];
        // UNJUSTIFIED (C5)
        out.region_W[static_cast<std::size_t>(ri) * n_region + (ri + 1)] =
            std::min(cd * REGION_SEQUENTIAL_COEFF, REGION_SEQUENTIAL_CAP);
    }

    // Non-sequential: evaluate viable region-internal pairs.
    // Features for region (f32 pointer into full array).
    const float* feats_region = in.features + static_cast<std::size_t>(in.entry_beat) * in.n_features;

    for (int ri = 0; ri < n_region - 1; ++ri) {
        const int abs_i = in.entry_beat + ri;

        for (int rj = 0; rj < n_region; ++rj) {
            if (rj == ri || rj == ri + 1) continue;
            if (std::abs(rj - ri) < REGION_MICRO_SKIP_BEATS) continue;  // C1
            const double cd = chroma_D[static_cast<std::size_t>(ri) * n_region + rj];
            if (cd > REGION_CHROMA_PREFILTER) continue;  // C2

            const int abs_j = in.entry_beat + rj;
            const int source_boundary = abs_i + 1;

            // --- Hard gates ---
            double energy_diff = 0.0;
            const bool have_edge_db = !edge_db_end.empty() && !edge_db_start.empty();
            if (have_edge_db) {
                energy_diff = std::abs(edge_db_end[ri] - edge_db_start[rj]);
                if (energy_diff > ENERGY_HARD_BLOCK_DB) continue;  // CLEAN (sesja 18)
            }

            // Vocal features.
            double va_i = 0.0;
            double va_j = 0.0;
            if (track_has_vocals && in.vocal_activity != nullptr) {
                if (abs_i < n_total) va_i = in.vocal_activity[abs_i];
                if (abs_j < n_total) va_j = in.vocal_activity[abs_j];
            }

            // --- Quality signals ---
            std::optional<double> waveform_sim;
            int lag = 0;
            if (has_waveforms && source_boundary < n_total) {
                auto [ws, l] = dsp::WaveformXcorr::compute(
                    in.boundary_waveforms + static_cast<std::size_t>(source_boundary) * in.n_samples_per_bnd,
                    in.boundary_waveforms + static_cast<std::size_t>(abs_j) * in.n_samples_per_bnd,
                    static_cast<std::size_t>(in.n_samples_per_bnd),
                    static_cast<std::size_t>(in.n_samples_per_bnd),
                    max_lag_samples);
                waveform_sim = ws;
                lag          = l;
            }

            const double successor_sim_v =
                successor_sim[static_cast<std::size_t>(ri) * n_region + rj];

            std::optional<double> edge_splice_sim;
            if (!edge_splice.empty()) {
                edge_splice_sim = edge_splice[static_cast<std::size_t>(ri) * n_region + rj];
            }

            // Context similarity (2-beat window — region_cost.py:337-338).
            // C11 UNJUSTIFIED-DRIFT: hardcoded 2 before + 3 after, NOT
            // config.py::context_window_beats=2 (DEAD-CONFIG).
            std::vector<double> ctx_before = safeWindowMean(
                feats_region, n_region, in.n_features,
                std::max(0, ri - REGION_CONTEXT_BEFORE_BEATS),
                ri + 1);
            std::vector<double> ctx_after = safeWindowMean(
                feats_region, n_region, in.n_features,
                rj,
                std::min(n_region, rj + REGION_CONTEXT_AFTER_BEATS));
            double na = 0.0, nb = 0.0, dot = 0.0;
            for (int k = 0; k < in.n_features; ++k) {
                na  += ctx_before[k] * ctx_before[k];
                nb  += ctx_after[k]  * ctx_after[k];
                dot += ctx_before[k] * ctx_after[k];
            }
            na = std::sqrt(na);
            nb = std::sqrt(nb);
            double context_sim = 0.0;
            if (na > 1e-8 && nb > 1e-8) {  // DEFENSIVE (C12)
                context_sim = std::clamp(dot / (na * nb), -1.0, 1.0);
            }

            const double label_match = (
                beat_labels[static_cast<std::size_t>(ri)] != "unknown"
                && beat_labels[static_cast<std::size_t>(ri)] == beat_labels[static_cast<std::size_t>(rj)]
            ) ? 1.0 : 0.0;

            const double bar_aligned =
                (pre_db_set.count(ri) > 0 && db_set.count(rj) > 0) ? 1.0 : 0.0;  // MARKER (C13)

            // Energy / centroid matches (absolute indices).
            double energy_match = 1.0;
            if (in.rms_energy != nullptr) {
                const double rms_diff = std::abs(in.rms_energy[abs_i] - in.rms_energy[abs_j]);
                energy_match = std::max(0.0, 1.0 - rms_diff * 5.0);  // UNJUSTIFIED (C14)
            }

            double edge_energy_match = 1.0;
            if (have_edge_db) {
                // UNJUSTIFIED-DRIFT (C15) — 12.0 cap matches session-16 soft-penalty saturation pattern.
                // Session-24 retrofit: 2nd consumer of public EDGE_ENERGY_SATURATION_DB
                // (from TransitionCost.h, promoted session 18). Hard Rule #1 single-source-of-truth —
                // see ADR-029 for consolidation of 3 consumers (TransitionCost + RegionCost + BlockAssembly).
                edge_energy_match = std::max(
                    0.0,
                    1.0 - std::min(energy_diff, EDGE_ENERGY_SATURATION_DB) / EDGE_ENERGY_SATURATION_DB);
            }

            double centroid_match = 1.0;
            if (in.spectral_centroid != nullptr) {
                const double c_diff = std::abs(in.spectral_centroid[abs_i] - in.spectral_centroid[abs_j]);
                centroid_match = std::max(0.0, 1.0 - c_diff * 5.0);  // UNJUSTIFIED (C16)
            }

            // Section sim — region_cost.py:368. ADR-044: 0 in no-structure
            // mode (otherwise label_match=0 would still leave the BIAS=0.1 floor).
            const double section_sim = noStructure ? 0.0
                : (label_match * REGION_SECTION_SIM_SCALE + REGION_SECTION_SIM_BIAS);  // UNJUSTIFIED (C17)

            // Compose quality score.
            QualityInputs q;
            q.waveform_sim     = waveform_sim;
            q.successor_sim    = successor_sim_v;
            q.edge_splice_sim  = edge_splice_sim;
            q.context_sim      = context_sim;
            q.label_match      = label_match;
            q.section_sim      = section_sim;
            q.bar_aligned      = bar_aligned;
            q.energy_match     = energy_match;
            q.edge_energy_match = edge_energy_match;
            q.centroid_match   = centroid_match;
            // ADR-064 sesja 75 — transient continuity (absolute indices).
            if (! onset_norm.empty()
                && abs_i < n_total && abs_j < n_total) {
                q.transient_continuity =
                    1.0 - std::abs(onset_norm[(std::size_t) abs_i]
                                 - onset_norm[(std::size_t) abs_j]);
            }
            // ADR-066 sesja 77 — MFCC continuity (absolute indices).
            if (! mfcc_continuity_matrix.empty()
                && abs_i < n_total && abs_j < n_total) {
                q.mfcc_continuity = mfcc_continuity_matrix[
                    (std::size_t) abs_i * (std::size_t) n_total
                  + (std::size_t) abs_j];
            }
            // ADR-080 RESCOPE + ADR-083 sesja 92 — full-mix chroma continuity
            // (absolute indices). Consumed by Tone slider blend in
            // computeQualityScore. nullopt → blend bypassed.
            if (! chroma_continuity_matrix.empty()
                && abs_i < n_total && abs_j < n_total) {
                q.full_mix_chroma_continuity = chroma_continuity_matrix[
                    (std::size_t) abs_i * (std::size_t) n_total
                  + (std::size_t) abs_j];
            }
            // ADR-088 sesja 98 STATUS UPDATE 1 — vocal phrase continuity, fixed
            // formula (see TransitionCost.cpp for rationale).
            //   HIGH boundary signals → HIGH quality (reward alignment).
            //   Silence-gate: vocal_density<0.1 → q=1.0 (instrumental clean).
            //   Soft floor 0.5: prevents harmonic mean composite crash via
            //   epsilon-clamped term.
            if (in.edge_vocal_onset_start != nullptr
                && in.edge_vocal_release_end != nullptr
                && abs_i < n_total && abs_j < n_total) {
                const double rel_i = in.edge_vocal_release_end[(std::size_t) abs_i];
                const double on_j  = in.edge_vocal_onset_start[(std::size_t) abs_j];
                const double boundary = std::max(rel_i, on_j);

                double vocal_density = 0.0;
                if (in.vocal_activity != nullptr) {
                    const double va_i = in.vocal_activity[(std::size_t) abs_i];
                    const double va_j = in.vocal_activity[(std::size_t) abs_j];
                    vocal_density = std::max(va_i, va_j);
                }

                constexpr double kSilenceThreshold = 0.1;
                if (vocal_density < kSilenceThreshold) {
                    q.vocal_continuity = 1.0;
                } else {
                    q.vocal_continuity = 0.5 + 0.5 * boundary;
                }
            }

            double quality = computeQualityScore(
                q,
                in.quality_weights != nullptr ? *in.quality_weights : kDefaultQualityWeights);

            // Span penalty — region_cost.py:384-387.
            // ADR-044: skipped in no-structure mode (label_match=0 would
            // otherwise blanket-fire on every short jump).
            if (! noStructure) {
                const int jump_beats = std::abs(rj - ri);
                if (jump_beats < SPAN_PENALTY_MAX_BEATS
                    && label_match < REGION_LABEL_MATCH_THRESHOLD) {  // C18
                    // UNJUSTIFIED (C19) — halved vs. global full penalty.
                    quality = std::max(0.0, quality - SPAN_PENALTY_CROSS_SECTION * REGION_SPAN_PENALTY_HALVING);
                }
                // No same-section span penalty (region_cost.py:387 comment).
            }

            // Vocal penalty.
            if (track_has_vocals && in.vocal_activity != nullptr) {
                std::optional<double> eva_end;
                std::optional<double> eva_start;
                if (in.edge_vocal_activity_end != nullptr && abs_i < n_total)
                    eva_end = in.edge_vocal_activity_end[abs_i];
                if (in.edge_vocal_activity_start != nullptr && abs_j < n_total)
                    eva_start = in.edge_vocal_activity_start[abs_j];
                const double vp = computeVocalPenalty(va_i, va_j, eva_end, eva_start);
                quality = std::max(0.0, quality - vp);
            }

            // Onset penalty.
            if (in.onset_strength != nullptr) {
                std::optional<double> os_j;
                if (abs_j < n_total) os_j = in.onset_strength[abs_j];
                quality = std::max(0.0, quality - computeOnsetPenalty(os_j));
            }

            const double total_cost = 1.0 - quality;  // CLEAN (C20)

            // Write into region_W + candidates.
            out.region_W[static_cast<std::size_t>(ri) * n_region + rj] = total_cost;
            TransitionCandidate cand;
            cand.from_beat              = abs_i;
            cand.to_beat                = abs_j;
            cand.quality_score          = quality;
            cand.waveform_similarity    = waveform_sim.value_or(0.0);
            cand.successor_similarity   = successor_sim_v;
            cand.edge_splice_similarity = edge_splice_sim.value_or(0.0);
            cand.chroma_distance        = cd;
            cand.energy_diff_db         = energy_diff;
            cand.alignment_lag_samples  = lag;
            cand.total_cost             = total_cost;
            out.candidates[{abs_i, abs_j}] = cand;
        }
    }

    return out;
}

} // namespace reamix::remix
