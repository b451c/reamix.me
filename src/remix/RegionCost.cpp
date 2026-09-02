#include "remix/RegionCost.h"

#include "remix/Quality.h"
#include "remix/SignalNorm.h"
#include "remix/RepetitionPrior.h"  // ADR-115 E4 (sesja 115)
#include "remix/PairScorer.h"      // sesja 119 (DEV-096) shared pair scorer

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>    // dev-only REAMIX_REGION_DEBUG pool dump (sesja 116)
#include <cstdlib>
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
    // Edge dB over the WHOLE track, indexed by absolute beat (DEV-090 sesja
    // 116): the v2 baselines below must describe the track's own consecutive
    // steps, not the region slice (a 31-beat slice is below kMinSamples and a
    // region-local array indexed the wrong beats when combined with the
    // full-track rms / centroid / onset pointers).
    auto [edge_db_end, edge_db_start] = precomputeEdgeEnergy(
        in.edge_rms_end, in.edge_rms_start, 0, n_total);
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
    std::set<int> abs_db_set;   // whole-track downbeat indices (v2 prior, sesja 116)
    if (in.downbeats != nullptr && in.n_downbeats > 0) {
        for (int k = 0; k < in.n_downbeats; ++k) {
            const double dbt = in.downbeats[k];
            int    abs_idx = 0;
            double best    = std::abs(in.beat_times[0] - dbt);
            for (int b = 1; b < n_total; ++b) {
                const double d = std::abs(in.beat_times[b] - dbt);
                if (d < best) { best = d; abs_idx = b; }
            }
            abs_db_set.insert(abs_idx);
            if (in.entry_beat <= abs_idx && abs_idx < in.exit_beat) {
                db_set.insert(abs_idx - in.entry_beat);
            }
        }
    } else if (n_region > 0) {
        // Fallback: synthesized downbeats region_cost.py:254-255.
        for (int ri = 0; ri < n_region; ri += in.time_signature) {  // CLEAN (CALLER OVERRIDE)
            db_set.insert(ri);
            abs_db_set.insert(in.entry_beat + ri);
        }
    }
    std::set<int> pre_db_set;
    for (int db : db_set) if (db > 0) pre_db_set.insert(db - 1);
    std::set<int> abs_pre_db_set;
    for (int db : abs_db_set) if (db > 0) abs_pre_db_set.insert(db - 1);

    // ADR-115 E1/E3 (sesja 114) — v2 scoring state: baselines over the whole
    // track's consecutive beats (all arrays absolute-indexed, n_total long).
    const bool v2 = in.v2_scoring;
    const SignalBaselines baselines = v2
        ? buildSignalBaselines(in.rms_energy, in.spectral_centroid, in.onset_strength,
                               edge_db_end.empty()   ? nullptr : edge_db_end.data(),
                               edge_db_start.empty() ? nullptr : edge_db_start.data(),
                               n_total)
        : SignalBaselines{};
    const bool v2_bar_constraint = v2 && db_set.size() >= 2 && ! pre_db_set.empty();
    const QualityWeights& v2_default_weights = v2 ? kV2QualityWeights : kDefaultQualityWeights;

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

    // ADR-115 E4 (sesja 115) + E8 (sesja 116, DEV-090): repetition-diagonal
    // prior built on the WHOLE track (absolute indices) and consulted for the
    // region's pairs. Sesja 115 built it on the region slice, where a 31-beat
    // recurrence with k = 12 mutual neighbours is noise and left Billie Jean's
    // 16 s region with 3 loop pairs from a single source. Region fallback: when
    // the prior leaves fewer than kRegionMinPriorPairsPerSource allowed pairs
    // per region source, it is switched off for the region (E3 bar alignment
    // only) - a short region has too little material to also demand
    // structural recurrence.
    RepetitionPrior rep_prior = v2_bar_constraint && ! in.disable_repetition_prior
        ? RepetitionPrior::build(in.features, n_total, in.n_features,
                                 abs_pre_db_set, abs_db_set, in.time_signature)
        : RepetitionPrior{};
    int n_pairs_bar = 0, n_pairs_prior = 0;
    if (rep_prior.active) {
        for (int ri : pre_db_set) {
            for (int rj : db_set) {
                if (rj == ri + 1 || std::abs(rj - ri) < REGION_MICRO_SKIP_BEATS) continue;
                ++n_pairs_bar;
                if (rep_prior.allowed(in.entry_beat + ri, in.entry_beat + rj)) ++n_pairs_prior;
            }
        }
        if (n_pairs_prior < kRegionMinPriorPairsPerSource * static_cast<double>(pre_db_set.size())) {
            rep_prior.active = false;
            rep_prior.mask.clear();
        }
    }
    out.prior_active   = rep_prior.active;
    out.n_pairs_bar    = n_pairs_bar;
    out.n_pairs_prior  = n_pairs_prior;
    int n_gate_chroma = 0, n_gate_energy = 0, n_gate_loud = 0;   // dev dump counters

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
    // Sesja 119 (DEV-096): the per-pair scoring body is shared with Block
    // Assembly (remix/PairScorer.h). Region keeps its own candidate gates
    // (bar constraint, prior, micro-skip, chroma prefilter) and the span
    // penalty; the precomputed f32 successor / edge-splice matrices are
    // passed through so the legacy path stays bit-exact.
    PairScorerTrack track{};
    track.n_total             = n_total;
    track.n_features          = in.n_features;
    track.features            = in.features;
    track.boundary_waveforms  = in.boundary_waveforms;
    track.n_samples_per_bnd   = in.n_samples_per_bnd;
    track.max_lag             = max_lag_samples;
    track.has_waveforms       = has_waveforms;
    track.edge_db_end         = edge_db_end.empty()   ? nullptr : edge_db_end.data();
    track.edge_db_start       = edge_db_start.empty() ? nullptr : edge_db_start.data();
    track.rms_energy          = in.rms_energy;
    track.spectral_centroid   = in.spectral_centroid;
    track.onset_strength      = in.onset_strength;
    track.vocal_activity      = in.vocal_activity;
    track.edge_vocal_activity_start = in.edge_vocal_activity_start;
    track.edge_vocal_activity_end   = in.edge_vocal_activity_end;
    track.edge_vocal_onset_start    = in.edge_vocal_onset_start;
    track.edge_vocal_release_end    = in.edge_vocal_release_end;
    track.onset_norm          = onset_norm.empty() ? nullptr : onset_norm.data();
    track.onset_norm_n        = static_cast<int>(onset_norm.size());
    track.mfcc_continuity_matrix   = mfcc_continuity_matrix.empty() ? nullptr : mfcc_continuity_matrix.data();
    track.chroma_continuity_matrix = chroma_continuity_matrix.empty() ? nullptr : chroma_continuity_matrix.data();
    track.ctx_lo              = in.entry_beat;
    track.ctx_hi              = in.exit_beat;
    track.v2                  = v2;
    track.baselines           = v2 ? &baselines : nullptr;
    track.weights             = in.quality_weights != nullptr ? in.quality_weights : &v2_default_weights;
    track.track_has_vocals    = track_has_vocals;
    track.gate                = v2 ? PairGate::SuccessorView : PairGate::LegacyAbsolute;

    for (int ri = 0; ri < n_region - 1; ++ri) {
        const int abs_i = in.entry_beat + ri;

        for (int rj = 0; rj < n_region; ++rj) {
            if (rj == ri || rj == ri + 1) continue;
            if (std::abs(rj - ri) < REGION_MICRO_SKIP_BEATS) continue;  // C1
            // v2: bar alignment as a candidate constraint (ADR-115 E3)
            if (v2_bar_constraint && ! (pre_db_set.count(ri) > 0 && db_set.count(rj) > 0)) continue;
            if (v2_bar_constraint && ! rep_prior.allowed(abs_i, in.entry_beat + rj)) continue;   // ADR-115 E4 (whole-track prior, sesja 116)
            const double cd = chroma_D[static_cast<std::size_t>(ri) * n_region + rj];
            if (cd > REGION_CHROMA_PREFILTER) { ++n_gate_chroma; continue; }  // C2

            const int abs_j = in.entry_beat + rj;

            const double label_match = (
                beat_labels[static_cast<std::size_t>(ri)] != "unknown"
                && beat_labels[static_cast<std::size_t>(ri)] == beat_labels[static_cast<std::size_t>(rj)]
            ) ? 1.0 : 0.0;

            const double bar_aligned =
                (pre_db_set.count(ri) > 0 && db_set.count(rj) > 0) ? 1.0 : 0.0;  // MARKER (C13)

            PairScorerRequest req{};
            req.abs_i         = abs_i;
            req.abs_j         = abs_j;
            req.successor_sim = successor_sim[static_cast<std::size_t>(ri) * n_region + rj];
            if (!edge_splice.empty())
                req.edge_splice_sim = edge_splice[static_cast<std::size_t>(ri) * n_region + rj];
            req.label_match = label_match;
            req.bar_aligned = bar_aligned;
            // Section sim — region_cost.py:368. ADR-044: 0 in no-structure
            // mode (otherwise label_match=0 would still leave the BIAS=0.1 floor).
            req.section_sim = noStructure ? 0.0
                : (label_match * REGION_SECTION_SIM_SCALE + REGION_SECTION_SIM_BIAS);  // UNJUSTIFIED (C17)

            const PairScore score = scorePair(track, req);
            if (score.rejected) {
                if (score.gate == 1) ++n_gate_energy; else ++n_gate_loud;
                continue;
            }
            double quality = score.quality;

            // Span penalty — region_cost.py:384-387.
            // ADR-044: skipped in no-structure mode (label_match=0 would
            // otherwise blanket-fire on every short jump). Applied after the
            // shared vocal / onset penalties: every penalty is >= 0 and
            // clamped, so the order does not change the value.
            if (! noStructure) {
                const int jump_beats = std::abs(rj - ri);
                if (jump_beats < SPAN_PENALTY_MAX_BEATS
                    && label_match < REGION_LABEL_MATCH_THRESHOLD) {  // C18
                    // UNJUSTIFIED (C19) — halved vs. global full penalty.
                    quality = std::max(0.0, quality - SPAN_PENALTY_CROSS_SECTION * REGION_SPAN_PENALTY_HALVING);
                }
                // No same-section span penalty (region_cost.py:387 comment).
            }

            const double total_cost = 1.0 - quality;  // CLEAN (C20)

            // Write into region_W + candidates.
            out.region_W[static_cast<std::size_t>(ri) * n_region + rj] = total_cost;
            TransitionCandidate cand;
            cand.from_beat              = abs_i;
            cand.to_beat                = abs_j;
            cand.quality_score          = quality;
            cand.waveform_similarity    = score.waveform_sim;
            cand.successor_similarity   = score.successor_sim;
            cand.edge_splice_similarity = score.edge_splice_sim;
            cand.chroma_distance        = cd;
            cand.energy_diff_db         = score.energy_diff_db;
            cand.alignment_lag_samples  = score.lag;
            cand.total_cost             = total_cost;
            out.candidates[{abs_i, abs_j}] = cand;
        }
    }

    // Dev-only pool diagnostic (sesja 116, DEV-090): REAMIX_REGION_DEBUG=<path>
    // appends one line describing the region candidate pool; pairs with
    // RegionOptimizer's loop-pair dump under the same variable.
    if (const char* dbg = std::getenv("REAMIX_REGION_DEBUG")) {
        if (FILE* f = std::fopen(dbg, "a")) {
            double best_q = 0.0;
            for (const auto& kv : out.candidates) best_q = std::max(best_q, kv.second.quality_score);
            // natural edge step |end(i) - start(i+1)| over the track: all pairs / bar-aligned pairs
            std::vector<double> step_all, step_bar;
            if (! edge_db_end.empty())
                for (int i = 0; i + 1 < n_total; ++i) {
                    const double d = std::abs(edge_db_end[i] - edge_db_start[i + 1]);
                    step_all.push_back(d);
                    if (abs_pre_db_set.count(i)) step_bar.push_back(d);
                }
            auto pct = [](std::vector<double> v, double q) {
                if (v.empty()) return 0.0;
                std::sort(v.begin(), v.end());
                return v[static_cast<std::size_t>(q * (v.size() - 1))];
            };
            std::fprintf(f, "# natural edge step dB: all p50=%.2f p90=%.2f p98=%.2f | bar p50=%.2f p90=%.2f p98=%.2f (n=%d)\n",
                         pct(step_all, 0.5), pct(step_all, 0.9), pct(step_all, 0.98),
                         pct(step_bar, 0.5), pct(step_bar, 0.9), pct(step_bar, 0.98), static_cast<int>(step_bar.size()));
            std::fprintf(f, "# pool entry=%d n_region=%d bar=%d sources=%d bar_pairs=%d prior_pairs=%d prior_active=%d gate_chroma=%d gate_energy=%d gate_loud=%d scored=%d best_q=%.3f\n",
                         in.entry_beat, n_region, in.time_signature,
                         static_cast<int>(pre_db_set.size()), n_pairs_bar, n_pairs_prior,
                         rep_prior.active ? 1 : 0, n_gate_chroma, n_gate_energy, n_gate_loud,
                         static_cast<int>(out.candidates.size()), best_q);
            std::fclose(f);
        }
    }

    return out;
}

} // namespace reamix::remix
