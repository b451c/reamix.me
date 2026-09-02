#pragma once

#include "AnalysisBundle.h"
#include "remix/BlockAssembly.h"

#include <algorithm>
#include <cstddef>
#include <vector>

// BlockCompatInputs wiring from an AnalysisBundle. Shared by the Blocks
// branch of RemixPipeline (Assemble) and the live junction preview in
// MainComponent (sesja 120, DEV-097) so both score a junction with exactly
// the same signals, windows and beta constants - two copies of this body
// were the DEV-096 drift. Mirrors RegionCostWiring.h.
//
// The caller sets blocks / n_blocks, v2_scoring, quality_weights,
// drift_penalty_weight, block_assembly_beta, block_energy_gate and the
// (lazy) block_sequence. `gridDownbeats` must outlive the inputs.

namespace reamix::ui
{

inline constexpr int kBlockCompatAnalysisSampleRate = 22050;

inline void fillBlockCompatInputs (reamix::remix::BlockCompatInputs& bin,
                                   const AnalysisBundle&             bundle,
                                   const std::vector<double>&        gridDownbeats,
                                   int                               gridBarBeats)
{
    const auto& bt = bundle.beatTimes;
    bin.beat_times = bt.data();
    bin.n_beats    = (int) bt.size();
    bin.features   = bundle.feat.features.data();
    bin.n_features = bundle.feat.nFeat;

    const auto& bw = bundle.feat.boundaryWaveforms;
    if (! bw.empty() && bundle.tc.n_beats > 0)
    {
        bin.boundary_waveforms   = bw.data();
        bin.n_boundary_waveforms = bundle.tc.n_beats;
        bin.n_samples_per_bnd    = (int) (bw.size() / (std::size_t) bundle.tc.n_beats);
        bin.waveform_sample_rate = kBlockCompatAnalysisSampleRate;
    }
    else
    {
        bin.boundary_waveforms   = nullptr;
        bin.n_boundary_waveforms = 0;
        bin.n_samples_per_bnd    = 0;
        bin.waveform_sample_rate = 0;
    }

    bin.edge_rms_start  = bundle.feat.edgeRmsStart.empty() ? nullptr : bundle.feat.edgeRmsStart.data();
    bin.edge_rms_end    = bundle.feat.edgeRmsEnd.empty()   ? nullptr : bundle.feat.edgeRmsEnd.data();
    bin.edge_features_start = bundle.feat.edgeFeaturesStart.empty() ? nullptr
                               : reinterpret_cast<const float*> (bundle.feat.edgeFeaturesStart.data());
    bin.edge_features_end   = bundle.feat.edgeFeaturesEnd.empty()   ? nullptr
                               : reinterpret_cast<const float*> (bundle.feat.edgeFeaturesEnd.data());
    bin.n_edge_features     = bundle.feat.edgeFeaturesStart.empty() ? 0 : bundle.feat.nFeat;

    bin.rms_energy        = bundle.feat.rmsEnergy.empty()        ? nullptr : bundle.feat.rmsEnergy.data();
    bin.spectral_centroid = bundle.feat.spectralCentroid.empty() ? nullptr : bundle.feat.spectralCentroid.data();
    bin.vocal_activity    = bundle.feat.vocalActivity.empty()    ? nullptr : bundle.feat.vocalActivity.data();
    // FIX-IN-PORT (sesja 71, ADR-059) - onset-sustain penalty for Block
    // Assembly, missing in Python `block_assembly.py`. Null in parity tests
    // preserves Python ground truth; non-null here activates the fix.
    bin.onset_strength    = bundle.feat.onsetStrength.empty()    ? nullptr : bundle.feat.onsetStrength.data();
    // ADR-088 sesja 98 - vocal phrase boundary signals.
    bin.edge_vocal_onset_start = bundle.feat.edgeVocalOnsetStart.empty() ? nullptr
                                                                        : bundle.feat.edgeVocalOnsetStart.data();
    bin.edge_vocal_release_end = bundle.feat.edgeVocalReleaseEnd.empty() ? nullptr
                                                                        : bundle.feat.edgeVocalReleaseEnd.data();
    // Sesja 119 (DEV-096) - edge-resolution vocal activity for the shared
    // vocal penalty (same inputs Region passes).
    bin.edge_vocal_activity_start = bundle.feat.edgeVocalActivityStart.empty() ? nullptr
                                                                               : bundle.feat.edgeVocalActivityStart.data();
    bin.edge_vocal_activity_end   = bundle.feat.edgeVocalActivityEnd.empty() ? nullptr
                                                                             : bundle.feat.edgeVocalActivityEnd.data();

    bin.downbeats   = gridDownbeats.empty() ? nullptr : gridDownbeats.data();
    bin.n_downbeats = (int) gridDownbeats.size();
    bin.time_signature = gridBarBeats;

    // Sesja 119 (DEV-096): every "bar-ish" beat constant derives from the
    // measured bar (ADR-115 E5) instead of the 4/4 literals. The drift
    // penalty is normalised by two bars (= the old W=8 at 4/4). Beta
    // fields: sesja-69 captured design + ADR-081 sketch in measured bars -
    // outside window 2 bars, min jump / top-K separation / short-block
    // threshold 1 bar.
    const int barBeats = std::max (1, gridBarBeats);
    bin.search_window_beats        = 2 * barBeats;
    bin.fragment_penalty_weight    = 0.03;
    bin.short_block_threshold_beats= barBeats;
    bin.top_k_min_separation_beats = barBeats;
    bin.outside_window_beats       = 2 * barBeats;
    bin.min_jump_beats             = barBeats;
    bin.downbeat_only_splices      = true;
    bin.block_sequence_lazy        = true;
}

// UserBlock -> BlockInfo mapping shared by the same two callers (DEV-095
// sesja 119 remap rules: blocks shorter than 2 beats are dropped and the
// queue is translated through `infoIdxOfUser`). `kindLabels` order =
// reamix::theme::SegmentKind.
struct BlockInfoMap
{
    std::vector<reamix::remix::BlockInfo> infos;
    std::vector<int>                      infoIdxOfUser;   // -1 = dropped
    std::vector<juce::String>             display;         // per info (labelOverride or kind label)
    int                                   nDropped { 0 };
};

inline BlockInfoMap mapUserBlocksToInfos (const std::vector<reamix::ui::UserBlock>& userBlocks,
                                          const std::vector<double>&                bt)
{
    static const char* const kKindLabels[12] = {
        "intro", "verse", "pre-chorus", "chorus", "post-chorus",
        "bridge", "buildup", "drop", "breakdown", "solo", "instrumental", "outro"
    };
    BlockInfoMap m;
    if (bt.size() < 2) return m;
    auto nearestBeatIdx = [&] (double t) -> int
    {
        auto it = std::lower_bound (bt.begin(), bt.end(), t);
        if (it == bt.begin()) return 0;
        if (it == bt.end())   return (int) bt.size() - 1;
        const int after  = (int) std::distance (bt.begin(), it);
        const int before = after - 1;
        return (std::abs (bt[(std::size_t) before] - t)
                <= std::abs (bt[(std::size_t) after] - t)) ? before : after;
    };
    m.infos.reserve (userBlocks.size());
    m.infoIdxOfUser.assign (userBlocks.size(), -1);
    std::vector<int> kindCounts (12, 0);
    for (std::size_t bi = 0; bi < userBlocks.size(); ++bi)
    {
        const auto& ub = userBlocks[bi];
        const int sb = nearestBeatIdx (ub.startSec);
        const int eb = nearestBeatIdx (ub.endSec);
        if (eb - sb < 2) { ++m.nDropped; continue; }

        reamix::remix::BlockInfo info{};
        info.segment_idx = (int) bi;
        const int kindInt = (int) ub.kind;
        const bool known = kindInt >= 0 && kindInt < 12;
        info.label = known ? kKindLabels[kindInt] : "unknown";
        const int n = known ? ++kindCounts[(std::size_t) kindInt] : 1;
        info.display_name = info.label;
        if (n > 1) info.display_name += " " + std::to_string (n);
        info.start_beat = sb;
        info.end_beat   = eb;
        info.start_sec  = bt[(std::size_t) sb];
        info.end_sec    = bt[(std::size_t) eb];
        info.n_beats    = eb - sb;
        info.duration_sec = info.end_sec - info.start_sec;
        info.cluster_id = kindInt;
        m.infoIdxOfUser[bi] = (int) m.infos.size();
        m.display.push_back (ub.labelOverride.has_value() && ub.labelOverride->isNotEmpty()
                                 ? *ub.labelOverride : juce::String (info.label));
        m.infos.push_back (info);
    }
    return m;
}

} // namespace reamix::ui
