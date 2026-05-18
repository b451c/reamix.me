#include "ui/RemixPipeline.h"

#include <juce_audio_formats/juce_audio_formats.h>

#include "remix/BlockAssembly.h"
#include "remix/Optimizer.h"
#include "remix/Path.h"
#include "remix/RegionCost.h"
#include "remix/RegionOptimizer.h"
#include "remix/TransitionCost.h"
#include "render/Renderer.h"
#include "ui/RemixCache.h"

#include <algorithm>
#include <map>

#include <utility>

namespace reamix::ui
{

namespace
{
    constexpr int kAnalysisSampleRate = 22050;

    // Stage-budget per ADR-047 § 2 — RemixPipeline owns its own [0.0, 1.0]
    // progress range. Optimizer is fast (~5 % of total), Renderer dominates.
    constexpr double kPOptimize = 0.20;
    constexpr double kPRender   = 0.85;
    constexpr double kPWav      = 1.00;

    bool writeTmpWav (const std::vector<float>& channelMajor,
                      std::size_t nChannels,
                      std::size_t nSamplesPerCh,
                      int         sr,
                      const juce::File& outFile,
                      juce::String& err)
    {
        if (nChannels == 0 || nSamplesPerCh == 0 || sr <= 0)
        {
            err = "Render result has zero dimension";
            return false;
        }

        // Remove any stale file so writer can create fresh.
        (void) outFile.deleteFile();

        auto stream = std::unique_ptr<juce::FileOutputStream> (outFile.createOutputStream());
        if (stream == nullptr || ! stream->openedOk())
        {
            err = "Unable to open tmp WAV for writing: " + outFile.getFullPathName();
            return false;
        }

        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer (
            wav.createWriterFor (stream.get(),
                                 (double) sr,
                                 (unsigned int) nChannels,
                                 /*bitsPerSample*/ 24,
                                 {}, 0));
        if (writer == nullptr)
        {
            err = "Unable to create WAV writer";
            return false;
        }
        (void) stream.release();

        std::vector<const float*> planes (nChannels);
        for (std::size_t ch = 0; ch < nChannels; ++ch)
            planes[ch] = channelMajor.data() + ch * nSamplesPerCh;

        if (! writer->writeFromFloatArrays (planes.data(), (int) nChannels, (int) nSamplesPerCh))
        {
            err = "WAV write failed";
            return false;
        }
        writer->flush();
        return true;
    }
}

RemixPipeline::RemixPipeline (Input        in,
                              ProgressCb   onProgress,
                              CompleteCb   onComplete,
                              juce::String tmpWavPath)
    : juce::Thread ("reamix.remix"),
      in_ (std::move (in)),
      onProgress_ (std::move (onProgress)),
      onComplete_ (std::move (onComplete)),
      tmpWavPath_ (std::move (tmpWavPath)),
      alive_ (std::make_shared<std::atomic<bool>> (true))
{
}

RemixPipeline::~RemixPipeline()
{
    alive_->store (false);
    stopThread (5000);
}

void RemixPipeline::postProgress (const juce::String& step, double p01)
{
    if (! onProgress_) return;
    const double clamped = juce::jlimit (0.0, 1.0, p01);
    juce::String s = step;
    auto alive = alive_;
    ProgressCb cb = onProgress_;
    juce::MessageManager::callAsync ([cb, s, clamped, alive]
    {
        if (alive && alive->load()) cb (s, clamped);
    });
}

void RemixPipeline::postCompletion (RemixOutput out)
{
    if (! onComplete_) return;
    auto alive = alive_;
    CompleteCb cb = onComplete_;
    juce::MessageManager::callAsync ([cb, out = std::move (out), alive]() mutable
    {
        if (alive && alive->load()) cb (std::move (out));
    });
}

void RemixPipeline::run()
{
    RemixOutput out;
    out.sourcePath     = in_.bundle ? in_.bundle->sourcePath : juce::String();
    out.itemGuid       = in_.itemGuid; // ADR-056 (sesja 66) echo for composite cache key
    out.targetSec      = in_.targetDurationSec;
    out.regionStartSec = in_.regionStartSec.value_or (0.0);
    out.regionEndSec   = in_.regionEndSec.value_or (0.0);
    // DEV-033 — default actual bounds = user bounds; overwritten in Region
    // branch after RegionOptimizer picks soft-boundary entry/exit beats.
    out.actualRegionStartSec = out.regionStartSec;
    out.actualRegionEndSec   = out.regionEndSec;
    out.variation      = in_.variation;
    out.blockedTransitions = in_.blockedTransitions;
    // ADR-080 RESCOPE + ADR-083 (sesja 92) — echo audition slider hash so
    // handleRemixComplete inserts under the same cache key kickRemixPipeline
    // looked up under.
    out.auditionHash = reamix::ui::hashAuditionParams (
        in_.harmonic_vs_timbre, in_.edit_length_slider,
        (int) std::lround (in_.allow_pm_seconds), in_.min_cut_beats == 0 ? 16 : in_.min_cut_beats);

    if (in_.bundle == nullptr)
    {
        out.errorMessage = "RemixPipeline started with null bundle";
        postCompletion (std::move (out));
        return;
    }

    // Bundle is conceptually const for this worker, but
    // CleanOptimizerInputs / TransitionCostInputs declare some pointer
    // fields as non-const (legacy from when AnalyzeWorker built them
    // from local non-const stack values). Take a mutable reference here
    // — this worker never writes to bundle members.
    auto& bundle = *in_.bundle;

    // ── Stage 6 — Optimizer (phase-4 second half) ──────────────────
    // Three branches:
    //   - Blocks  (ADR-051): user-marked + arranged sections; soft-boundary
    //     algorithm via BlockAssembly::computeBlockCompatibility +
    //     assembleBlocks.
    //   - Region  (sesja 60, step 6): regionStart/EndSec set; RegionOptimizer.
    //   - Auto    (default): CleanOptimizer.
    const bool blocksMode =
        in_.userBlocks.size() >= 1 && in_.userBlocksQueue.size() >= 2;

    const bool regionMode = ! blocksMode
        && in_.regionStartSec.has_value() && in_.regionEndSec.has_value()
        && (*in_.regionEndSec) > (*in_.regionStartSec);

    postProgress (blocksMode ? "Assembling blocks"
                  : regionMode ? "Computing region remix path"
                               : "Computing remix path",
                  0.0);

    reamix::remix::RemixPath path;
    try
    {
        const std::set<std::pair<int,int>>* blockedPtr =
            in_.blockedTransitions.empty() ? nullptr : &in_.blockedTransitions;

        if (blocksMode)
        {
            // ADR-051 phase J — Block Assembly path. Convert UserBlock list to
            // BlockInfo via beat-time mapping; compute compatibility matrix
            // with soft-boundary penalty; assemble user-ordered queue.

            const auto& bt = bundle.beatTimes;
            if (bt.size() < 2)
                throw std::runtime_error ("Bundle has < 2 beats — Block Assembly not possible");

            // Helper: nearest beat index for a given time. Uses lower_bound to
            // find the first beat at or after t; corrects to nearest.
            auto nearestBeatIdx = [&] (double t) -> int
            {
                auto it = std::lower_bound (bt.begin(), bt.end(), t);
                if (it == bt.begin()) return 0;
                if (it == bt.end())   return (int) bt.size() - 1;
                const int after = (int) std::distance (bt.begin(), it);
                const int before = after - 1;
                return (std::abs (bt[(std::size_t) before] - t)
                        <= std::abs (bt[(std::size_t) after] - t)) ? before : after;
            };

            // Map UserBlocks → BlockInfo. Skip degenerate (n_beats < 2) blocks.
            // Disambiguate display_name by counting prior occurrences per kind.
            std::vector<reamix::remix::BlockInfo> infos;
            infos.reserve (in_.userBlocks.size());
            std::map<int,int> kindCounts;

            static const std::array<const char*, 12> kKindLabels = {
                "intro", "verse", "pre-chorus", "chorus", "post-chorus",
                "bridge", "buildup", "drop", "breakdown", "solo", "instrumental", "outro"
            };

            for (std::size_t bi = 0; bi < in_.userBlocks.size(); ++bi)
            {
                const auto& ub = in_.userBlocks[bi];
                const int sb = nearestBeatIdx (ub.startSec);
                const int eb = nearestBeatIdx (ub.endSec);
                if (eb - sb < 2) continue;

                reamix::remix::BlockInfo info{};
                info.segment_idx = (int) bi;
                const int kindInt = (int) ub.kind;
                info.label = (kindInt >= 0 && kindInt < (int) kKindLabels.size())
                                ? kKindLabels[(std::size_t) kindInt]
                                : "unknown";
                const int n = ++kindCounts[kindInt];
                info.display_name = info.label;
                if (n > 1) info.display_name += " " + std::to_string (n);
                info.start_beat = sb;
                info.end_beat   = eb;
                info.start_sec  = bt[(std::size_t) sb];
                info.end_sec    = bt[(std::size_t) eb];
                info.n_beats    = eb - sb;
                info.duration_sec = info.end_sec - info.start_sec;
                info.cluster_id = kindInt;
                infos.push_back (info);
            }

            if (infos.size() < 2)
                throw std::runtime_error ("Block Assembly needs at least 2 valid blocks");

            // Build BlockCompatInputs — mirror RegionCost wiring for shared
            // signals (boundary waveforms, edge features, etc).
            reamix::remix::BlockCompatInputs bin{};
            bin.blocks    = infos.data();
            bin.n_blocks  = (int) infos.size();
            bin.beat_times = bt.data();
            bin.n_beats    = (int) bt.size();
            bin.features   = bundle.feat.features.data();
            bin.n_features = bundle.feat.nFeat;

            const auto& bw = bundle.feat.boundaryWaveforms;
            if (! bw.empty() && bundle.tc.n_beats > 0)
            {
                bin.boundary_waveforms   = bw.data();
                bin.n_boundary_waveforms = bundle.tc.n_beats;
                bin.n_samples_per_bnd    =
                    (int) (bw.size() / (std::size_t) bundle.tc.n_beats);
                bin.waveform_sample_rate = kAnalysisSampleRate;
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
            // FIX-IN-PORT (sesja 71, ADR-059) — onset-sustain penalty for
            // Block Assembly, missing in Python `block_assembly.py`. Null in
            // parity tests preserves Python ground truth; non-null here
            // activates the fix in production. See dev-028-lessons.md.
            bin.onset_strength    = bundle.feat.onsetStrength.empty()    ? nullptr : bundle.feat.onsetStrength.data();
            // ADR-088 sesja 98 — vocal phrase boundary signals.
            bin.edge_vocal_onset_start = bundle.feat.edgeVocalOnsetStart.empty() ? nullptr
                                                                                : bundle.feat.edgeVocalOnsetStart.data();
            bin.edge_vocal_release_end = bundle.feat.edgeVocalReleaseEnd.empty() ? nullptr
                                                                                : bundle.feat.edgeVocalReleaseEnd.data();

            bin.downbeats   = bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data();
            bin.n_downbeats = (int) bundle.downbeatTimes.size();
            bin.time_signature = juce::jmax (1, (int) bundle.timeSigNum);

            bin.search_window_beats   = std::max (1, in_.spliceFlexBeats);
            bin.drift_penalty_weight  = in_.driftPenaltyWeight;

            // ADR-058 — calibration weight override (sesja 71). nullptr →
            // kDefaultQualityWeights → preserves production baseline + parity.
            bin.quality_weights       = in_.qualityWeightsOverride.has_value()
                ? &(*in_.qualityWeightsOverride)
                : nullptr;

            // ADR-081 (sesja 96) — β-model candidate-space expansion. Filter
            // queue first (used by lazy compute in β-mode). Default
            // block_assembly_beta=false preserves legacy ±W path.
            std::vector<int> validQueue;
            validQueue.reserve (in_.userBlocksQueue.size());
            for (int q : in_.userBlocksQueue)
            {
                if (q >= 0 && q < (int) infos.size())
                    validQueue.push_back (q);
            }
            if (validQueue.size() < 2)
                throw std::runtime_error ("Queue has fewer than 2 valid blocks");

            bin.block_assembly_beta = in_.block_assembly_beta;
            // β-fields below have no effect when block_assembly_beta=false.
            // Defaults match sesja-69 captured design + ADR-081 sketch.
            bin.fragment_penalty_weight    = 0.03;
            bin.short_block_threshold_beats= 4;
            bin.top_k_min_separation_beats = 4;
            bin.outside_window_beats       = 8;
            bin.min_jump_beats             = 4;
            bin.downbeat_only_splices      = true;
            bin.block_sequence_lazy        = true;
            bin.block_sequence             = validQueue.empty() ? nullptr : validQueue.data();
            bin.n_block_sequence           = (int) validQueue.size();

            const auto compat = reamix::remix::computeBlockCompatibility (bin);

            path = reamix::remix::assembleBlocks (validQueue, infos,
                                                   bt.data(), (int) bt.size(),
                                                   compat,
                                                   /*variation=*/0,
                                                   in_.junctionVariations.empty()
                                                     ? nullptr
                                                     : &in_.junctionVariations,
                                                   in_.edit_length_jump_scale,   // ADR-084 sesja 93
                                                   /*allow_outside_window=*/in_.block_assembly_beta);
            (void) blockedPtr; // assembleBlocks does not consume blocked set
        }
        else if (regionMode)
        {
            // ── Resolve entry/exit beat indices ──────────────────
            // Lua `remix_insert.get_item_region` returns item-relative
            // seconds; in_.regionStartSec/EndSec are stored item-relative
            // (beat_times are also item-relative, both reference the
            // analyzed source file).
            const auto& bt = bundle.beatTimes;
            if (bt.size() < 2)
                throw std::runtime_error ("Bundle has < 2 beats — region remix not possible");

            const double regStart = *in_.regionStartSec;
            const double regEnd   = *in_.regionEndSec;

            // entry_beat = first beat with time >= regStart (forward into the region).
            auto entryIt = std::lower_bound (bt.begin(), bt.end(), regStart);
            int entry_beat = (int) std::distance (bt.begin(), entryIt);
            if (entry_beat < 0) entry_beat = 0;
            if (entry_beat >= (int) bt.size()) entry_beat = (int) bt.size() - 1;

            // exit_beat = first beat with time > regEnd (one past the last in-region beat).
            auto exitIt = std::upper_bound (bt.begin(), bt.end(), regEnd);
            int exit_beat = (int) std::distance (bt.begin(), exitIt);
            if (exit_beat <= entry_beat + 1)
                throw std::runtime_error ("Region too small (entry/exit beat overlap)");
            if (exit_beat > (int) bt.size())
                exit_beat = (int) bt.size();

            // ── Build RegionCostInputs (mirror AnalyzePipeline TransitionCost wiring) ──
            reamix::remix::RegionCostInputs rcin{};
            rcin.entry_beat = entry_beat;
            rcin.exit_beat  = exit_beat;
            rcin.features   = bundle.feat.features.data();
            rcin.n_total    = bundle.tc.n_beats;
            rcin.n_features = bundle.feat.nFeat;
            rcin.beat_times = bundle.beatTimes.data();

            rcin.segments   = bundle.structure.segments.data();
            rcin.n_segments = (int) bundle.structure.segments.size();

            // Sesja 60 hot-fix — pass boundary waveforms so RegionCost runs
            // its xcorr-based phase-alignment branch (RegionCost.cpp:373-434).
            // Without this, splice point selection lacks audio-level phase
            // matching and falls back to chroma + feature distance only,
            // producing "rigid" splice picks. FeatureExtractor stage 3
            // populates boundaryWaveforms unconditionally (155 ms window per
            // beat @ 22050 Hz ≈ 3417 samples).
            const auto& bw = bundle.feat.boundaryWaveforms;
            if (! bw.empty() && bundle.tc.n_beats > 0)
            {
                rcin.boundary_waveforms   = bw.data();
                rcin.n_boundary_waveforms = bundle.tc.n_beats;
                rcin.n_samples_per_bnd    =
                    (int) (bw.size() / (std::size_t) bundle.tc.n_beats);
            }
            else
            {
                rcin.boundary_waveforms   = nullptr;
                rcin.n_boundary_waveforms = 0;
                rcin.n_samples_per_bnd    = 0;
            }
            rcin.waveform_sample_rate = kAnalysisSampleRate;

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

            rcin.downbeats   = bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data();
            rcin.n_downbeats = (int) bundle.downbeatTimes.size();
            rcin.time_signature = juce::jmax (1, (int) bundle.timeSigNum);

            // ADR-058 — calibration weight override (sesja 71). nullptr →
            // kDefaultQualityWeights → preserves production baseline + parity.
            rcin.quality_weights = in_.qualityWeightsOverride.has_value()
                ? &(*in_.qualityWeightsOverride)
                : nullptr;

            auto rcr = reamix::remix::computeRegionCosts (rcin);

            // ── Build RegionOptimizerInputs ──────────────────────
            reamix::remix::RegionOptimizerInputs roin{};
            roin.n_beats    = bundle.tc.n_beats;
            roin.beat_times = bundle.beatTimes.data();
            // CleanOptimizer constructor (Optimizer.cpp:212-216):
            //   avg_beat_duration = (last - first) / (n_beats - 1)
            roin.avg_beat_duration =
                (bt.back() - bt.front()) / (double) (bt.size() - 1);
            // ADR-083 sesja 92 — Allow ± slider drives duration_tolerance_sec.
            // Default 5.0 (matches kDurationToleranceSecDefault) → bit-exact.
            roin.duration_tolerance_sec = in_.allow_pm_seconds;
            roin.candidates = &bundle.tc.candidates;
            roin.sample_rate = kAnalysisSampleRate;
            // ADR-084 sesja 93 — Edit Length multiplicative + Min cut override.
            roin.edit_length_jump_scale      = in_.edit_length_jump_scale;
            roin.min_seq_after_jump_override = in_.min_cut_beats;

            // ADR-057 (sesja 68) — Region boundaries → mechanical match to
            // user-selection edges. splice_flex_beats=0 activates legacy
            // argminAbsDiff path (closest beat to region.startSec/endSec);
            // Renderer + Insert pipeline override first/last clip source
            // bounds to user-selection so boundaries are sample-exact match.
            // Supersedes ADR-054 soft-boundary downbeat search.
            roin.downbeats         = bundle.downbeatTimes.empty() ? nullptr
                                                                  : bundle.downbeatTimes.data();
            roin.n_downbeats       = (int) bundle.downbeatTimes.size();
            roin.splice_flex_beats = 0;

            // ADR-081 STATUS UPDATE 1 sesja 94 — Region β-model "inner-loop
            // synthesizer" production flip. Default false in struct preserves
            // parity test 48/48 PASS for test_region_optimizer; production
            // path here flips true so user-facing Region mode finds multi-
            // iteration short loops on quality-rich inner content (cost-
            // function rebalance: cap 1.0→5.0, backward penalty 0.5→0.05,
            // jump base 0.8→0.3 in beta path).
            roin.region_beta = true;

            // ADR-081 STATUS UPDATE 2 sesja 94 — pass entry/exit beats to
            // RegionOptimizer so it uses the SAME boundaries that RegionCost
            // used to build region_W. Fixes latent stride bug where
            // RegionOptimizer's local argminAbsDiff diverged from this
            // pipeline's lower_bound/upper_bound by 1 beat → SCRAMBLED rW
            // reads → loop synthesizer picked wrong (i, j) loop points
            // (sesja 94 user smoke iter 2 surface trigger).
            roin.entry_beat_override = entry_beat;
            roin.exit_beat_override  = exit_beat;

            reamix::remix::RegionOptimizer ropt (roin);

            // Sesja 100 (DEV-032) — Region "Try different splice" K-best
            // variations. variation == 0 → standard remix() with caller's
            // blocked set; variation > 0 → remix_variation() builds k-best
            // and returns the (variation_idx)-th distinct path. Mirrors
            // Duration mode wiring (CleanOptimizer::remix_variation).
            if (in_.variation > 0)
            {
                path = ropt.remix_variation (in_.targetDurationSec,
                                              regStart, regEnd,
                                              rcr.region_W.data(),
                                              rcr.n_region,
                                              &rcr.candidates,
                                              in_.variation,
                                              blockedPtr);
            }
            else
            {
                path = ropt.remix (in_.targetDurationSec,
                                    regStart, regEnd,
                                    rcr.region_W.data(),
                                    rcr.n_region,
                                    &rcr.candidates,
                                    blockedPtr);
            }

            // ADR-057 (sesja 68) — capture source-time positions where WAV's
            // first/last samples actually live so Insert pipeline can split
            // pre/post-region items at exactly those positions for sample-
            // exact boundary content match.
            //
            // ENTRY: WAV's first sample = source[beat_times[entry_beat]]
            // (Renderer's first run sourceStartSample = beatSamples_[entry]).
            //
            // EXIT — OFF-BY-ONE FIX: Renderer's last run uses sourceEndSample
            // = beatEndSample(exit_beat) = beatSamples_[exit_beat + 1]
            // (Renderer.cpp:213-225, _beat_end_sample port). So WAV's LAST
            // sample is at source[beat_times[exit_beat + 1]], NOT
            // source[beat_times[exit_beat]]. Pre-step-2c we set
            // actualRegionEndSec = beat_times[exit_beat], causing Insert to
            // split post-region one beat EARLIER than where WAV ends —
            // result: the source-time interval [beat_times[exit_beat],
            // beat_times[exit_beat + 1]] played BOTH in WAV's tail AND in
            // post-region's head, audibly perceived as a beat-skip ("mijanka
            // w beatach na łączeniu końcowym" — sesja 68 user smoke).
            //
            // Fix: actualRegionEndSec follows Renderer's beatEndSample
            // semantics (next-beat boundary), with same fallback for
            // out-of-range exit_beat + 1.
            const int chosenEntry = ropt.entryBeat();
            const int chosenExit  = ropt.exitBeat();
            if (chosenEntry >= 0 && chosenEntry < (int) bt.size())
                out.actualRegionStartSec = bt[(std::size_t) chosenEntry];
            if (chosenExit > 0 && chosenExit < (int) bt.size())
            {
                if ((std::size_t) (chosenExit + 1) < bt.size())
                {
                    out.actualRegionEndSec = bt[(std::size_t) (chosenExit + 1)];
                }
                else if (bt.size() >= 2)
                {
                    // Fallback mirrors Renderer::beatEndSample line 219-224:
                    // when exit_beat is the last detected beat, extrapolate
                    // by 2× average beat duration capped at source duration.
                    const double avgBeat =
                        (bt.back() - bt.front())
                        / (double) (bt.size() - 1);
                    const double srcDur = (bundle.nativeSr > 0)
                        ? (double) bundle.nativeSamples / (double) bundle.nativeSr
                        : bt.back() + avgBeat;
                    out.actualRegionEndSec = std::min (
                        bt[(std::size_t) chosenExit] + avgBeat * 2.0, srcDur);
                }
                else
                {
                    out.actualRegionEndSec = bt[(std::size_t) chosenExit];
                }
            }

            // Sesja 100 (DEV-032) — RegionOptimizer now consumes blockedPtr
            // (above call sites). Comment retained as historical marker for
            // pre-DEV-032 (void) cast that this branch used to suppress
            // unused-parameter warning.
        }
        else
        {
            // DEV-044 Path A (sesja 93) — Tone slider Duration mode fix.
            // When the override sets harmonic_vs_timbre > 0, re-compute the
            // transition cost matrix locally with that weight so the Tone
            // blend in computeQualityScore actually fires. The cached
            // bundle.tc.W was baked at analyze-time without the override
            // (AnalyzePipeline.cpp:269-313 pins quality_weights = nullptr).
            // Region + Block paths already do their own remix-time matrix
            // build via computeRegionCosts / computeBlockCompatibility, so
            // they pick up the override directly. Mirrors AnalyzePipeline
            // setup. ~50-200 ms vs ~20 ms cache lookup, acceptable per
            // ADR-080 § Decision 150-300 ms total UX target.
            reamix::remix::TransitionCostResult freshTc;
            reamix::remix::TransitionCostResult* tcSrc = &bundle.tc;

            // Sesja-98 ADR-087 STATUS UPDATE 1 — extend Path A guard to fire
            // for ANY override that differs from kDefaultQualityWeights, not
            // only when harmonic_vs_timbre > 0. Dev calibration (sesja 98)
            // tweaks 7-component simplex weights via dev panel sliders;
            // without this extension, Duration mode would always use the
            // analyze-time cached W and silently ignore the override —
            // breaking the very feature the dev calibration build provides.
            // qualityWeightsAtDefault is bit-exact, so production users with
            // override unset (nullopt) AND override == default both skip the
            // re-compute and stay on the cached W path.
            const bool overrideRequiresPathA =
                  in_.qualityWeightsOverride.has_value()
               && ! reamix::ui::qualityWeightsAtDefault (*in_.qualityWeightsOverride);

            if (overrideRequiresPathA)
            {
                reamix::remix::TransitionCostInputs tcin{};
                tcin.features    = bundle.feat.features.data();
                tcin.n_beats     = bundle.feat.nBeats;
                tcin.n_features  = bundle.feat.nFeat;
                tcin.beat_times  = bundle.beatTimes.data();

                tcin.segments   = bundle.structure.segments.data();
                tcin.n_segments = (int) bundle.structure.segments.size();

                tcin.rms_energy        = bundle.feat.rmsEnergy.empty()        ? nullptr : bundle.feat.rmsEnergy.data();
                tcin.onset_strength    = bundle.feat.onsetStrength.empty()    ? nullptr : bundle.feat.onsetStrength.data();
                tcin.spectral_centroid = bundle.feat.spectralCentroid.empty() ? nullptr : bundle.feat.spectralCentroid.data();
                tcin.vocal_activity    = bundle.feat.vocalActivity.empty()    ? nullptr : bundle.feat.vocalActivity.data();

                const auto& bw = bundle.feat.boundaryWaveforms;
                if (! bw.empty() && bundle.feat.nBeats > 0)
                {
                    tcin.boundary_waveforms   = bw.data();
                    tcin.n_boundary_waveforms = bundle.feat.nBeats;
                    tcin.n_samples_per_bnd    =
                        (int) (bw.size() / (std::size_t) bundle.feat.nBeats);
                    tcin.waveform_sample_rate = kAnalysisSampleRate;
                }

                tcin.edge_vocal_activity_start = bundle.feat.edgeVocalActivityStart.empty() ? nullptr : bundle.feat.edgeVocalActivityStart.data();
                tcin.edge_vocal_activity_end   = bundle.feat.edgeVocalActivityEnd.empty()   ? nullptr : bundle.feat.edgeVocalActivityEnd.data();
                // ADR-088 sesja 98 — vocal phrase boundary signals.
                tcin.edge_vocal_onset_start    = bundle.feat.edgeVocalOnsetStart.empty()    ? nullptr : bundle.feat.edgeVocalOnsetStart.data();
                tcin.edge_vocal_release_end    = bundle.feat.edgeVocalReleaseEnd.empty()    ? nullptr : bundle.feat.edgeVocalReleaseEnd.data();
                tcin.edge_rms_start            = bundle.feat.edgeRmsStart.empty()           ? nullptr : bundle.feat.edgeRmsStart.data();
                tcin.edge_rms_end              = bundle.feat.edgeRmsEnd.empty()             ? nullptr : bundle.feat.edgeRmsEnd.data();

                tcin.edge_features_start = bundle.feat.edgeFeaturesStart.empty() ? nullptr
                                          : reinterpret_cast<const float*> (bundle.feat.edgeFeaturesStart.data());
                tcin.edge_features_end   = bundle.feat.edgeFeaturesEnd.empty()   ? nullptr
                                          : reinterpret_cast<const float*> (bundle.feat.edgeFeaturesEnd.data());
                tcin.n_edge_features     = bundle.feat.edgeFeaturesStart.empty() ? 0 : bundle.feat.nFeat;

                tcin.downbeats   = bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data();
                tcin.n_downbeats = (int) bundle.downbeatTimes.size();

                tcin.time_signature  = juce::jmax (1, (int) bundle.timeSigNum);
                tcin.quality_weights = &(*in_.qualityWeightsOverride);

                freshTc = reamix::remix::computeTransitionCosts (tcin);
                tcSrc   = &freshTc;
            }

            reamix::remix::CleanOptimizerInputs oin{};
            oin.W           = tcSrc->W.data();
            oin.candidates  = &tcSrc->candidates;
            oin.n_beats     = tcSrc->n_beats;
            oin.beat_times  = bundle.beatTimes.data();

            oin.segments    = bundle.structure.segments.data();
            oin.n_segments  = (int) bundle.structure.segments.size();

            oin.features    = bundle.feat.features.data();
            oin.n_features  = bundle.feat.nFeat;

            oin.downbeats   = bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data();
            oin.n_downbeats = (int) bundle.downbeatTimes.size();

            oin.time_signature = juce::jmax (1, (int) bundle.timeSigNum);
            oin.sample_rate    = kAnalysisSampleRate;

            // ADR-083 + ADR-084 sesja 92-93 — AuditionBar slider params
            // propagated to Duration mode optimizer. Defaults bit-exact baseline.
            oin.duration_tolerance_sec      = in_.allow_pm_seconds;
            oin.edit_length_jump_scale      = in_.edit_length_jump_scale;
            oin.min_seq_after_jump_override = in_.min_cut_beats;

            reamix::remix::CleanOptimizer opt (oin);

            // DEV-027 fix landed sesja 58 (ADR-048): when variation > 0, dispatch
            // through `remix_variation` which calls `remix_k_best(target,
            // max(2, v+1), blocked)` and indexes `paths[min(v, len-1)]`. For
            // variation == 0 stay on the fast path `remix(target, blocked)` —
            // identical result, skips k-best machinery.
            // Empty blocked set ⇒ pass nullptr per CleanOptimizer::remix signature.
            path = (in_.variation > 0)
                 ? opt.remix_variation (in_.targetDurationSec, in_.variation, blockedPtr)
                 : opt.remix           (in_.targetDurationSec,                blockedPtr);
        }
    }
    catch (const std::exception& e)
    {
        out.errorMessage = juce::String ("Optimizer failed: ") + e.what();
        postCompletion (std::move (out));
        return;
    }
    if (threadShouldExit()) return;

    // ── Stage 7 — Renderer (phase-5) ───────────────────────────────
    postProgress ("Rendering remix", kPOptimize);

    reamix::render::RendererConfig rcfg{};
    reamix::render::RenderResult renderOut;
    try
    {
        reamix::render::Renderer renderer (
            bundle.sourcePath.toStdString(),
            bundle.stereoNative.data(),
            (std::size_t) bundle.nChannels,
            bundle.nativeSamples,
            bundle.nativeSr,
            bundle.beatTimes.data(), bundle.beatTimes.size(),
            /*crossfadeMsOrNeg*/ -1.0,
            rcfg);

        // ADR-057 (sesja 68 step 2c) — Region user-selection-exact boundary
        // override. Default Renderer::render path uses beat-aligned source
        // boundaries (first clip starts at source[beat_times[entry_beat]],
        // last clip ends at source[beat_times[exit_beat + 1]] per beatEndSample
        // semantics). For Region mode user explicitly mandates that pre-region
        // / post-region splits respect the user's exact selection edges so
        // boundaries match the original song's selection points (sesja 68
        // user verbatim: "ostatni fragment ma sie laczyc w tym meijscu w
        // ktorhym byl w oryginalnym utworze"). Inline render() body so we
        // can mutate the plan between buildEditPlan and renderEditPlan.
        auto plan = renderer.buildEditPlan (path);

        if (in_.regionStartSec.has_value()
            && in_.regionEndSec.has_value()
            && ! plan.clips.empty())
        {
            // Step 2e — overlap-crossfade boundary. WAV first/last clip's
            // source range extends `halfFade` beyond user-selection edges so
            // Insert pipeline can place pre-region / first-WAV with halfFade
            // overlap centered on user edge, equal-power crossfade between
            // sample-exact identical content (pre-region's source content
            // [regStart, regStart+halfFade] = WAV[halfFade..2*halfFade] =
            // source[regStart, regStart+halfFade] by construction). Same on
            // exit boundary.
            constexpr double kFadeOverlapSec = 0.010;
            const double halfFade = kFadeOverlapSec * 0.5;

            const double regStart = std::max (0.0, *in_.regionStartSec);
            const double regEnd   = *in_.regionEndSec;
            const double srcDur   = (bundle.nativeSr > 0)
                ? (double) bundle.nativeSamples / (double) bundle.nativeSr
                : regEnd;

            // Clamp lead amounts to source bounds (degenerate to no-overlap
            // fade when region is at very start/end of source).
            const double leadIn  = std::min (halfFade, regStart);
            const double leadOut = std::min (halfFade, std::max (0.0, srcDur - regEnd));
            const double newFcStart = regStart - leadIn;
            const double newLcEnd   = regEnd + leadOut;

            // First clip — extend backward to (regStart - leadIn).
            auto& fc = plan.clips.front();
            const double deltaFirst = fc.sourceStartSec - newFcStart;
            fc.sourceStartSec = newFcStart;
            fc.durationSec   += deltaFirst;
            fc.timelineEndSec = fc.timelineStartSec + fc.durationSec;
            for (std::size_t i = 1; i < plan.clips.size(); ++i)
            {
                plan.clips[i].timelineStartSec += deltaFirst;
                plan.clips[i].timelineEndSec   += deltaFirst;
            }

            // Last clip — extend forward to (regEnd + leadOut).
            auto& lc = plan.clips.back();
            const double deltaLast = newLcEnd - lc.sourceEndSec;
            lc.sourceEndSec   = newLcEnd;
            lc.durationSec   += deltaLast;
            lc.timelineEndSec = lc.timelineStartSec + lc.durationSec;

            plan.duration = plan.clips.back().timelineEndSec;

            // RemixOutput exposes user-selection edges + actual lead amounts
            // (Insert reads boundaryLeadIn/OutSec to size overlap windows).
            out.actualRegionStartSec = regStart;
            out.actualRegionEndSec   = std::min (regEnd, srcDur);
            out.boundaryLeadInSec    = leadIn;
            out.boundaryLeadOutSec   = leadOut;
        }

        renderOut.editPlan   = plan;
        renderOut.sampleRate = renderer.sampleRate();
        renderer.renderEditPlan (plan,
                                  renderOut.audio,
                                  renderOut.nChannels,
                                  renderOut.nSamples,
                                  renderOut.transitionTimes);
        renderOut.duration     = (renderOut.sampleRate > 0)
            ? (double) renderOut.nSamples / (double) renderOut.sampleRate
            : 0.0;
        renderOut.nTransitions = (int) renderOut.transitionTimes.size();
    }
    catch (const std::exception& e)
    {
        out.errorMessage = juce::String ("Render failed: ") + e.what();
        postCompletion (std::move (out));
        return;
    }
    if (threadShouldExit()) return;

    // ── Stage 8 — WAV write ────────────────────────────────────────
    postProgress ("Writing tmp WAV", kPRender);

    juce::File wav (tmpWavPath_);
    juce::String err;
    if (! writeTmpWav (renderOut.audio,
                       renderOut.nChannels,
                       renderOut.nSamples,
                       renderOut.sampleRate,
                       wav, err))
    {
        out.errorMessage = err;
        postCompletion (std::move (out));
        return;
    }

    // ── Assemble output ────────────────────────────────────────────
    out.ok                 = true;
    out.nTransitions       = renderOut.nTransitions;
    out.transitionTimesSec = renderOut.transitionTimes;
    out.remixDurationSec   = renderOut.duration;
    out.tmpWavPath         = wav.getFullPathName();
    out.editPlan           = std::move (renderOut.editPlan);

    // Per-transition diagnostic vectors for SpliceMarker (session 57).
    // Sourced from RemixPath::transition_metadata (parity with Python
    // server/handlers/_remix.py:154-185). Order = path.transitions order;
    // matches transitionTimesSec from Renderer (both walk the path in
    // sequence).
    {
        const auto& segs = bundle.structure.segments;
        auto labelAtBeat = [&] (int beat) -> juce::String
        {
            if (beat < 0 || (std::size_t) beat >= bundle.beatTimes.size())
                return {};
            const double bt = bundle.beatTimes[(std::size_t) beat];
            for (const auto& seg : segs)
                if (bt >= seg.start && bt < seg.end)
                    return juce::String (seg.label);
            return {};
        };

        out.transitionQualities.reserve (path.transitions.size());
        out.transitionFromBeats.reserve (path.transitions.size());
        out.transitionToBeats.reserve (path.transitions.size());
        out.transitionEnergyDiffsDb.reserve (path.transitions.size());
        out.transitionFromLabels.reserve (path.transitions.size());
        out.transitionToLabels.reserve (path.transitions.size());

        for (const auto& tr : path.transitions)
        {
            const int fb = tr.first;
            const int tb = tr.second;
            float quality   = 0.0f;
            float energyDb  = 0.0f;
            auto it = path.transition_metadata.find (tr);
            if (it != path.transition_metadata.end())
            {
                auto qit = it->second.find ("quality_score");
                if (qit != it->second.end()) quality = (float) qit->second;
                auto eit = it->second.find ("energy_diff_db");
                if (eit != it->second.end()) energyDb = (float) eit->second;
            }
            out.transitionFromBeats.push_back (fb);
            out.transitionToBeats.push_back (tb);
            out.transitionQualities.push_back (quality);
            out.transitionEnergyDiffsDb.push_back (energyDb);
            out.transitionFromLabels.push_back (labelAtBeat (fb));
            out.transitionToLabels.push_back (labelAtBeat (tb));
        }
    }

    postProgress ("Done", kPWav);
    postCompletion (std::move (out));
}

} // namespace reamix::ui
