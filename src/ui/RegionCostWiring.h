#pragma once

#include "AnalysisBundle.h"
#include "remix/RegionCost.h"

#include <cstddef>
#include <vector>

// RegionCostInputs wiring from an AnalysisBundle. Shared by the Region remix
// branch (RemixPipeline) and the loop-spot map (LoopSpotsBuilder, ADR-115 E11,
// sesja 117) so both see the same pool: same features, boundary waveforms,
// edge arrays, per-beat scalars, vocal signals and cleaned grid.
//
// The caller sets entry_beat / exit_beat / v2_scoring / quality_weights.
// `gridDownbeats` + `gridBarBeats` must outlive the inputs (raw pointers).

namespace reamix::ui
{

inline constexpr int kRegionCostAnalysisSampleRate = 22050;

inline void fillRegionCostInputs (reamix::remix::RegionCostInputs& rcin,
                                  const AnalysisBundle&            bundle,
                                  const std::vector<double>&       gridDownbeats,
                                  int                              gridBarBeats)
{
    rcin.features   = bundle.feat.features.data();
    rcin.n_total    = bundle.tc.n_beats;
    rcin.n_features = bundle.feat.nFeat;
    rcin.beat_times = bundle.beatTimes.data();

    rcin.segments   = bundle.structure.segments.data();
    rcin.n_segments = (int) bundle.structure.segments.size();

    // Sesja 60 hot-fix — boundary waveforms drive the xcorr phase-alignment
    // branch (RegionCost.cpp). FeatureExtractor stage 3 populates them
    // unconditionally (155 ms window per beat @ 22050 Hz ≈ 3417 samples).
    const auto& bw = bundle.feat.boundaryWaveforms;
    if (! bw.empty() && bundle.tc.n_beats > 0)
    {
        rcin.boundary_waveforms   = bw.data();
        rcin.n_boundary_waveforms = bundle.tc.n_beats;
        rcin.n_samples_per_bnd    = (int) (bw.size() / (std::size_t) bundle.tc.n_beats);
    }
    else
    {
        rcin.boundary_waveforms   = nullptr;
        rcin.n_boundary_waveforms = 0;
        rcin.n_samples_per_bnd    = 0;
    }
    rcin.waveform_sample_rate = kRegionCostAnalysisSampleRate;

    rcin.edge_rms_start = bundle.feat.edgeRmsStart.empty() ? nullptr : bundle.feat.edgeRmsStart.data();
    rcin.edge_rms_end   = bundle.feat.edgeRmsEnd.empty()   ? nullptr : bundle.feat.edgeRmsEnd.data();
    rcin.edge_features_start = bundle.feat.edgeFeaturesStart.empty() ? nullptr
                               : reinterpret_cast<const float*> (bundle.feat.edgeFeaturesStart.data());
    rcin.edge_features_end   = bundle.feat.edgeFeaturesEnd.empty()   ? nullptr
                               : reinterpret_cast<const float*> (bundle.feat.edgeFeaturesEnd.data());
    rcin.n_edge_features     = bundle.feat.edgeFeaturesStart.empty() ? 0 : bundle.feat.nFeat;

    rcin.rms_energy        = bundle.feat.rmsEnergy.empty()        ? nullptr : bundle.feat.rmsEnergy.data();
    rcin.onset_strength    = bundle.feat.onsetStrength.empty()    ? nullptr : bundle.feat.onsetStrength.data();
    rcin.spectral_centroid = bundle.feat.spectralCentroid.empty() ? nullptr : bundle.feat.spectralCentroid.data();
    rcin.vocal_activity    = bundle.feat.vocalActivity.empty()    ? nullptr : bundle.feat.vocalActivity.data();
    rcin.edge_vocal_activity_start = bundle.feat.edgeVocalActivityStart.empty()
                                      ? nullptr : bundle.feat.edgeVocalActivityStart.data();
    rcin.edge_vocal_activity_end   = bundle.feat.edgeVocalActivityEnd.empty()
                                      ? nullptr : bundle.feat.edgeVocalActivityEnd.data();
    // ADR-088 sesja 98 — vocal phrase boundary signals.
    rcin.edge_vocal_onset_start    = bundle.feat.edgeVocalOnsetStart.empty()
                                      ? nullptr : bundle.feat.edgeVocalOnsetStart.data();
    rcin.edge_vocal_release_end    = bundle.feat.edgeVocalReleaseEnd.empty()
                                      ? nullptr : bundle.feat.edgeVocalReleaseEnd.data();

    rcin.downbeats   = gridDownbeats.empty() ? nullptr : gridDownbeats.data();
    rcin.n_downbeats = (int) gridDownbeats.size();
    rcin.time_signature = gridBarBeats;
}

} // namespace reamix::ui
