#include "remix/PairScorer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "dsp/WaveformXcorr.h"
#include "remix/TransitionCost.h"  // ENERGY_HARD_BLOCK_DB, EDGE_ENERGY_SATURATION_DB

namespace reamix::remix {

namespace {

// Block Assembly legacy graduated energy penalty (block_assembly.py:310-316):
// 0 below 6 dB, +0.025 per dB, capped at 0.30.
constexpr double kGraduatedThresholdDb = 6.0;
constexpr double kGraduatedCap         = 0.30;
constexpr double kGraduatedSlope       = 0.025;

// Context window: 2 beats before the source, 3 beats from the destination
// (region_cost.py:337-338 / block_assembly.py:348-349).
constexpr int kContextBefore = 2;
constexpr int kContextAfter  = 3;

constexpr double kNormFloor = 1e-8;

// Mean of feature rows [lo, hi) in f64 (region_cost.py `_safe_window_mean`).
// lo >= hi leaves the vector at zero, which the cosine guard turns into 0.
void windowMean(const float* features, int n_features, int lo, int hi, std::vector<double>& out)
{
    out.assign(static_cast<std::size_t>(n_features), 0.0);
    if (lo >= hi) return;
    for (int r = lo; r < hi; ++r) {
        const float* row = features + static_cast<std::size_t>(r) * n_features;
        for (int k = 0; k < n_features; ++k) out[static_cast<std::size_t>(k)] += static_cast<double>(row[k]);
    }
    const double inv = 1.0 / static_cast<double>(hi - lo);
    for (int k = 0; k < n_features; ++k) out[static_cast<std::size_t>(k)] *= inv;
}

// Cosine of two f32 rows accumulated in f64, clipped to [-1, 1]; nullopt
// when either norm is degenerate (block_assembly.py:337-345 edge splice,
// :334-335 successor on pre-normalised rows).
std::optional<double> rowCosine(const float* a, const float* b, int n)
{
    double na = 0.0, nb = 0.0, dot = 0.0;
    for (int k = 0; k < n; ++k) {
        const double x = static_cast<double>(a[k]);
        const double y = static_cast<double>(b[k]);
        na  += x * x;
        nb  += y * y;
        dot += x * y;
    }
    na = std::sqrt(na);
    nb = std::sqrt(nb);
    if (na <= kNormFloor || nb <= kNormFloor) return std::nullopt;
    return std::clamp(dot / (na * nb), -1.0, 1.0);
}

} // namespace

PairScore scorePair(const PairScorerTrack& t, const PairScorerRequest& req)
{
    PairScore out;
    const int  n_total = t.n_total;
    const int  i = req.abs_i;
    const int  j = req.abs_j;
    const int  source_boundary = std::min(i + 1, n_total - 1);
    const bool have_edge_db = t.edge_db_end != nullptr && t.edge_db_start != nullptr;

    // --- Hard gates (edge energy) ------------------------------------------
    double energy_diff = 0.0;
    if (have_edge_db) {
        energy_diff = std::abs(t.edge_db_end[i] - t.edge_db_start[j]);
        out.energy_diff_db = energy_diff;
        bool ok = true;
        switch (t.gate) {
            case PairGate::None: break;
            case PairGate::LegacyAbsolute:
                ok = energy_diff <= ENERGY_HARD_BLOCK_DB;
                break;
            case PairGate::SuccessorView:
                // ADR-115 E8 (sesja 116, DEV-090 / DEV-091): the incoming
                // attack must resemble the one the listener expects
                // (start(j) vs start(i+1)) and the outgoing tail must resemble
                // what normally precedes j (end(i) vs end(j-1)).
                ok = i + 1 < n_total && j > 0
                  && std::abs(t.edge_db_start[j] - t.edge_db_start[i + 1]) <= ENERGY_HARD_BLOCK_DB
                  && std::abs(t.edge_db_end[i] - t.edge_db_end[j - 1]) <= ENERGY_HARD_BLOCK_DB;
                break;
        }
        if (! ok) { out.rejected = true; out.gate = 1; return out; }
    }

    // --- Vocal readouts (track-level gate, region_cost.py:98-102) -----------
    double va_i = 0.0, va_j = 0.0;
    if (t.track_has_vocals && t.vocal_activity != nullptr) {
        if (i < n_total) va_i = t.vocal_activity[i];
        if (j < n_total) va_j = t.vocal_activity[j];
    }

    // --- Waveform xcorr -----------------------------------------------------
    std::optional<double> waveform_sim;
    if (t.has_waveforms && source_boundary < n_total) {
        auto [ws, lag] = dsp::WaveformXcorr::compute(
            t.boundary_waveforms + static_cast<std::size_t>(source_boundary) * t.n_samples_per_bnd,
            t.boundary_waveforms + static_cast<std::size_t>(j) * t.n_samples_per_bnd,
            static_cast<std::size_t>(t.n_samples_per_bnd),
            static_cast<std::size_t>(t.n_samples_per_bnd),
            t.max_lag);
        waveform_sim     = ws;
        out.has_waveform = true;
        out.waveform_sim = ws;
        out.lag          = lag;
    }

    // --- Successor similarity (row-shifted full-feature cosine) ------------
    double successor_sim = 0.0;
    if (req.successor_sim.has_value()) {
        successor_sim = *req.successor_sim;
    } else {
        successor_sim = rowCosine(t.features + static_cast<std::size_t>(source_boundary) * t.n_features,
                                  t.features + static_cast<std::size_t>(j) * t.n_features,
                                  t.n_features).value_or(0.0);
    }
    out.successor_sim = successor_sim;

    // --- Edge splice similarity --------------------------------------------
    std::optional<double> edge_splice_sim = req.edge_splice_sim;
    if (! edge_splice_sim.has_value()
        && t.edge_features_start != nullptr && t.edge_features_end != nullptr && t.n_edge_features > 0) {
        edge_splice_sim = rowCosine(t.edge_features_end + static_cast<std::size_t>(i) * t.n_edge_features,
                                    t.edge_features_start + static_cast<std::size_t>(j) * t.n_edge_features,
                                    t.n_edge_features);
    }
    out.edge_splice_sim = edge_splice_sim.value_or(0.0);

    // --- Context similarity -------------------------------------------------
    std::vector<double> ctx_a, ctx_b;
    windowMean(t.features, t.n_features, std::max(t.ctx_lo, i - kContextBefore), std::min(t.ctx_hi, i + 1), ctx_a);
    windowMean(t.features, t.n_features, std::max(t.ctx_lo, j), std::min(t.ctx_hi, j + kContextAfter), ctx_b);
    double na = 0.0, nb = 0.0, dot = 0.0;
    for (int k = 0; k < t.n_features; ++k) {
        na  += ctx_a[static_cast<std::size_t>(k)] * ctx_a[static_cast<std::size_t>(k)];
        nb  += ctx_b[static_cast<std::size_t>(k)] * ctx_b[static_cast<std::size_t>(k)];
        dot += ctx_a[static_cast<std::size_t>(k)] * ctx_b[static_cast<std::size_t>(k)];
    }
    na = std::sqrt(na);
    nb = std::sqrt(nb);
    double context_sim = 0.0;
    if (na > kNormFloor && nb > kNormFloor) context_sim = std::clamp(dot / (na * nb), -1.0, 1.0);
    out.context_sim = context_sim;

    // --- Scalar matches -----------------------------------------------------
    double energy_match = 1.0;
    if (t.rms_energy != nullptr)
        energy_match = std::max(0.0, 1.0 - std::abs(t.rms_energy[i] - t.rms_energy[j]) * 5.0);

    double edge_energy_match = 1.0;
    if (have_edge_db)
        edge_energy_match = std::max(
            0.0, 1.0 - std::min(energy_diff, EDGE_ENERGY_SATURATION_DB) / EDGE_ENERGY_SATURATION_DB);

    double centroid_match = 1.0;
    if (t.spectral_centroid != nullptr)
        centroid_match = std::max(0.0, 1.0 - std::abs(t.spectral_centroid[i] - t.spectral_centroid[j]) * 5.0);

    if (t.v2 && t.baselines != nullptr) {   // ADR-115 E1 / E2
        if (t.rms_energy != nullptr && have_edge_db
            && loudnessRejectV2(*t.baselines, t.rms_energy[i], t.rms_energy[j], energy_diff)) {
            out.rejected = true; out.gate = 2; return out;
        }
        if (t.rms_energy != nullptr)
            energy_match = energyQualityV2(*t.baselines, t.rms_energy[i], t.rms_energy[j], energy_match);
        if (have_edge_db)
            edge_energy_match = edgeEnergyQualityV2(*t.baselines, energy_diff, edge_energy_match);
        if (t.spectral_centroid != nullptr)
            centroid_match = centroidQualityV2(*t.baselines, t.spectral_centroid[i], t.spectral_centroid[j], centroid_match);
    }

    // --- Composite ----------------------------------------------------------
    QualityInputs q{};
    q.waveform_sim      = waveform_sim;
    q.successor_sim     = successor_sim;
    q.edge_splice_sim   = edge_splice_sim;
    q.context_sim       = context_sim;
    q.label_match       = req.label_match;
    q.section_sim       = req.section_sim;
    q.bar_aligned       = req.bar_aligned;
    q.energy_match      = energy_match;
    q.edge_energy_match = edge_energy_match;
    q.centroid_match    = centroid_match;
    if (t.onset_norm != nullptr && i < t.onset_norm_n && j < t.onset_norm_n) {   // ADR-064
        q.transient_continuity = 1.0 - std::abs(t.onset_norm[i] - t.onset_norm[j]);
        if (t.v2 && t.baselines != nullptr && t.onset_strength != nullptr)
            q.transient_continuity = onsetQualityV2(*t.baselines, t.onset_strength[i], t.onset_strength[j],
                                                    *q.transient_continuity);
    }
    if (t.mfcc_continuity_matrix != nullptr && i < n_total && j < n_total)   // ADR-066
        q.mfcc_continuity = t.mfcc_continuity_matrix[static_cast<std::size_t>(i) * n_total + j];
    if (t.chroma_continuity_matrix != nullptr && i < n_total && j < n_total)   // ADR-083
        q.full_mix_chroma_continuity = t.chroma_continuity_matrix[static_cast<std::size_t>(i) * n_total + j];
    if (t.edge_vocal_onset_start != nullptr && t.edge_vocal_release_end != nullptr
        && i < n_total && j < n_total) {   // ADR-088 STATUS UPDATE 1
        const double boundary = std::max(t.edge_vocal_release_end[i], t.edge_vocal_onset_start[j]);
        double vocal_density = 0.0;
        if (t.vocal_activity != nullptr)
            vocal_density = std::max(t.vocal_activity[i], t.vocal_activity[j]);
        constexpr double kSilenceThreshold = 0.1;
        q.vocal_continuity = vocal_density < kSilenceThreshold ? 1.0 : 0.5 + 0.5 * boundary;
    }

    double quality = computeQualityScore(q, *t.weights);

    // --- Penalties (all >= 0; sequential clamps equal one final clamp) ------
    if (t.graduated_energy_penalty && have_edge_db && energy_diff > kGraduatedThresholdDb)
        quality -= std::min(kGraduatedCap, (energy_diff - kGraduatedThresholdDb) * kGraduatedSlope);

    if (t.track_has_vocals && t.vocal_activity != nullptr) {
        std::optional<double> eva_end, eva_start;
        if (t.edge_vocal_activity_end != nullptr && i < n_total)   eva_end   = t.edge_vocal_activity_end[i];
        if (t.edge_vocal_activity_start != nullptr && j < n_total) eva_start = t.edge_vocal_activity_start[j];
        quality = std::max(0.0, quality - computeVocalPenalty(va_i, va_j, eva_end, eva_start));
    }
    if (t.onset_strength != nullptr) {
        std::optional<double> os_j;
        if (j < n_total) os_j = t.onset_strength[j];
        quality = std::max(0.0, quality - computeOnsetPenalty(os_j));
    }

    out.quality = std::max(0.0, quality);
    return out;
}

} // namespace reamix::remix
