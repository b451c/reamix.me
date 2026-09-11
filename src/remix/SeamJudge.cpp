// SeamJudge.cpp — sesja 131. See SeamJudge.h.
#include "remix/SeamJudge.h"

#include "remix/Quality.h"          // computeOnsetNorm, continuity matrices, kV2QualityWeights
#include "remix/TransitionCost.h"   // chromaRange, TRACK_VOCAL_THRESHOLD

#include <algorithm>
#include <cmath>

namespace reamix::remix
{

BoundarySeamJudge::BoundarySeamJudge(const BlockCompatInputs& in)
{
    const int n = in.n_beats;
    if (n <= 1 || in.features == nullptr || in.n_features <= 0 || in.beat_times == nullptr)
        return;

    // Waveform setup (diagnostic only on a boundary cut; kept so the record
    // carries the xcorr the harness columns expect). Mirrors BlockAssembly.
    int  max_lag = 0;
    bool has_wf  = false;
    if (in.boundary_waveforms != nullptr && in.waveform_sample_rate > 0) {
        const double max_lag_ms = 30.0;
        max_lag = static_cast<int>(max_lag_ms * static_cast<double>(in.waveform_sample_rate) / 1000.0);
        has_wf  = in.n_boundary_waveforms >= n && max_lag > 0;
    }

    const bool has_edge_db = in.edge_rms_start != nullptr && in.edge_rms_end != nullptr;
    if (has_edge_db) {
        edge_db_start_.resize(static_cast<std::size_t>(n));
        edge_db_end_.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            edge_db_end_  [static_cast<std::size_t>(i)] = 20.0 * std::log10(std::max(in.edge_rms_end  [i], BLOCK_DB_FLOOR));
            edge_db_start_[static_cast<std::size_t>(i)] = 20.0 * std::log10(std::max(in.edge_rms_start[i], BLOCK_DB_FLOOR));
        }
    }

    onset_norm_       = computeOnsetNorm(in.onset_strength, n);
    mfcc_continuity_  = computeMfccContinuityMatrix(in.features, n, in.n_features);
    const auto [chroma_start, chroma_end] = chromaRange(in.n_features);
    const int n_chroma = chroma_end - chroma_start;
    if (n_chroma > 0) {
        chroma_slice_.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n_chroma));
        for (int b = 0; b < n; ++b) {
            const float* src = in.features + static_cast<std::size_t>(b) * in.n_features + chroma_start;
            float*       dst = chroma_slice_.data() + static_cast<std::size_t>(b) * n_chroma;
            for (int k = 0; k < n_chroma; ++k) dst[k] = src[k];
        }
        chroma_continuity_ = computeChromaContinuityMatrix(chroma_slice_.data(), n, n_chroma);
    }

    baselines_ = buildSignalBaselines(in.rms_energy, in.spectral_centroid, in.onset_strength,
                                      has_edge_db ? edge_db_end_.data()   : nullptr,
                                      has_edge_db ? edge_db_start_.data() : nullptr, n,
                                      in.edge_mel_end, in.n_edge_mel, in.time_signature);

    bool track_has_vocals = false;
    if (in.vocal_activity != nullptr) {
        double max_va = 0.0;
        for (int b = 0; b < n; ++b) max_va = std::max(max_va, in.vocal_activity[b]);
        track_has_vocals = max_va >= TRACK_VOCAL_THRESHOLD;
    }

    track_.n_total                  = n;
    track_.n_features               = in.n_features;
    track_.features                 = in.features;
    track_.boundary_waveforms       = in.boundary_waveforms;
    track_.n_samples_per_bnd        = in.n_samples_per_bnd;
    track_.max_lag                  = max_lag;
    track_.has_waveforms            = has_wf;
    track_.edge_db_end              = has_edge_db ? edge_db_end_.data()   : nullptr;
    track_.edge_db_start            = has_edge_db ? edge_db_start_.data() : nullptr;
    track_.edge_features_start      = in.edge_features_start;
    track_.edge_features_end        = in.edge_features_end;
    track_.n_edge_features          = in.n_edge_features;
    track_.rms_energy               = in.rms_energy;
    track_.spectral_centroid        = in.spectral_centroid;
    track_.onset_strength           = in.onset_strength;
    track_.vocal_activity           = in.vocal_activity;
    track_.edge_vocal_activity_start= in.edge_vocal_activity_start;
    track_.edge_vocal_activity_end  = in.edge_vocal_activity_end;
    track_.edge_vocal_onset_start   = in.edge_vocal_onset_start;
    track_.edge_vocal_release_end   = in.edge_vocal_release_end;
    track_.edge_mel_end             = in.edge_mel_end;
    track_.n_edge_mel               = in.n_edge_mel;
    track_.onset_norm               = onset_norm_.empty() ? nullptr : onset_norm_.data();
    track_.onset_norm_n             = static_cast<int>(onset_norm_.size());
    track_.mfcc_continuity_matrix   = mfcc_continuity_.empty() ? nullptr : mfcc_continuity_.data();
    track_.chroma_continuity_matrix = chroma_continuity_.empty() ? nullptr : chroma_continuity_.data();
    track_.ctx_lo                   = 0;
    track_.ctx_hi                   = n;
    track_.v2                       = true;                 // the boundary judge is v2-only
    track_.baselines                = &baselines_;
    track_.weights                  = &kV2QualityWeights;   // unused on the boundary path (kV2BoundaryQualityWeights)
    track_.track_has_vocals         = track_has_vocals;
    track_.gate                     = PairGate::None;       // boundary gates are inside scorePair
    track_.graduated_energy_penalty = false;
    valid_ = true;
}

PairScore BoundarySeamJudge::score(int i, int j, bool relaxed, bool open) const
{
    PairScore out;
    if (! valid_ || i < 0 || j <= 0 || i >= track_.n_total || j >= track_.n_total) {
        out.rejected = true;
        return out;
    }
    PairScorerRequest req;
    req.abs_i       = i;
    req.abs_j       = j;
    req.bar_aligned = 1.0;
    req.boundary    = true;
    req.skip_loudness_reject = relaxed || open;
    req.skip_energy_gate     = open;
    return scorePair(track_, req);
}

std::optional<double> BoundarySeamJudge::quality(int i, int j, bool relaxed, bool open) const
{
    const PairScore s = score(i, j, relaxed, open);
    if (s.rejected) return std::nullopt;
    return s.quality;
}

} // namespace reamix::remix
