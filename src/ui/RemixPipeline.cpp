#include "ui/RemixPipeline.h"
#include "remix/BeatGrid.h"   // ADR-115 E5 (sesja 115)
#include "RegionCostWiring.h"  // ADR-115 E11 (sesja 117)
#include "BlockCompatWiring.h"  // sesja 120 (DEV-097)

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
#include <array>
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
    out.uiMode             = in_.uiMode;             // DEV-101 (sesja 123)
    out.blocksHash         = in_.blocksHash;
    out.qualityWeightsHash = in_.qualityWeightsHash;
    out.targetSec      = in_.targetDurationSec;
    out.regionStartSec = in_.regionStartSec.value_or (0.0);
    out.regionEndSec   = in_.regionEndSec.value_or (0.0);
    // DEV-033 — default actual bounds = user bounds; overwritten in Region
    // branch after RegionOptimizer picks soft-boundary entry/exit beats.
    out.actualRegionStartSec = out.regionStartSec;
    out.actualRegionEndSec   = out.regionEndSec;
    out.variation      = in_.variation;
    out.blockedTransitions = in_.blockedTransitions;
    // ADR-115 P3 (sesja 123) — echo the Edit density hash so handleRemixComplete
    // inserts under the same cache key kickRemixPipeline looked up under.
    out.editDensityHash = reamix::ui::hashEditDensity (in_.edit_density_bars);

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

    // ADR-115 E5 (sesja 115): on the v2 path every engine input gets the
    // cleaned beat grid (downbeats snapped onto beats with tolerance, hole-
    // adjacent downbeats dropped, bar length measured from downbeat spacing)
    // instead of the raw detector downbeats + detected time signature. The
    // legacy path keeps the raw grid (parity).
    const reamix::remix::BeatGridResult v2Grid = in_.v2_scoring
        ? reamix::remix::cleanBeatGrid (bundle.beatTimes.data(), (int) bundle.beatTimes.size(),
                                        bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data(),
                                        (int) bundle.downbeatTimes.size(),
                                        juce::jmax (1, (int) bundle.timeSigNum))
        : reamix::remix::BeatGridResult{};
    const std::vector<double>& gridDownbeats = in_.v2_scoring ? v2Grid.downbeats : bundle.downbeatTimes;
    const int gridBarBeats = in_.v2_scoring ? v2Grid.bar_beats : juce::jmax (1, (int) bundle.timeSigNum);

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
    // Sesja 119 (DEV-096): Blocks-mode junction labels (from -> to block),
    // indexed by junction; the metadata key "junction_idx" maps a transition
    // back to its junction (adjacent continuations record no transition).
    std::vector<std::pair<juce::String, juce::String>> blockJunctionLabels;
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

            // Map UserBlocks -> BlockInfo (nearest beat; blocks shorter than
            // 2 beats dropped) and remap the queue through userIdx -> infoIdx
            // (DEV-095 sesja 119). Shared with the live junction preview
            // (BlockCompatWiring.h, sesja 120).
            const BlockInfoMap bmap = mapUserBlocksToInfos (in_.userBlocks, bt);
            const auto& infos = bmap.infos;
            if (infos.size() < 2)
                throw std::runtime_error ("Block Assembly needs at least 2 valid blocks");

            std::vector<int> validQueue;
            validQueue.reserve (in_.userBlocksQueue.size());
            int nSkippedQueue = 0;
            for (int q : in_.userBlocksQueue)
            {
                if (q >= 0 && q < (int) bmap.infoIdxOfUser.size() && bmap.infoIdxOfUser[(std::size_t) q] >= 0)
                    validQueue.push_back (bmap.infoIdxOfUser[(std::size_t) q]);
                else
                    ++nSkippedQueue;
            }
            if (validQueue.size() < 2)
                throw std::runtime_error ("Queue has fewer than 2 valid blocks");
            if (bmap.nDropped > 0 || nSkippedQueue > 0)
                out.warningMessage = juce::String (nSkippedQueue) + " queued block"
                                   + (nSkippedQueue == 1 ? "" : "s")
                                   + " shorter than 2 beats skipped";
            for (std::size_t k = 0; k + 1 < validQueue.size(); ++k)
                blockJunctionLabels.emplace_back (bmap.display[(std::size_t) validQueue[k]],
                                                  bmap.display[(std::size_t) validQueue[k + 1]]);

            // Build BlockCompatInputs - shared wiring (signals, grid, beta
            // constants), then the per-run policy fields.
            reamix::remix::BlockCompatInputs bin{};
            fillBlockCompatInputs (bin, bundle, gridDownbeats, gridBarBeats);
            bin.v2_scoring = in_.v2_scoring;   // ADR-115 v2 scoring
            bin.blocks     = infos.data();
            bin.n_blocks   = (int) infos.size();
            bin.drift_penalty_weight  = in_.driftPenaltyWeight;
            // ADR-058 - calibration weight override (sesja 71). nullptr ->
            // kDefaultQualityWeights -> preserves production baseline + parity.
            bin.quality_weights       = in_.qualityWeightsOverride.has_value()
                ? &(*in_.qualityWeightsOverride)
                : nullptr;
            // ADR-081 (sesja 96) - beta-model candidate-space expansion.
            // Default block_assembly_beta=false preserves legacy +-W path.
            bin.block_assembly_beta = in_.block_assembly_beta;
            bin.block_energy_gate   = in_.block_energy_gate;   // sesja 119
            bin.block_sequence      = validQueue.data();
            bin.n_block_sequence    = (int) validQueue.size();
            const int barBeats      = std::max (1, gridBarBeats);

            const auto compat = reamix::remix::computeBlockCompatibility (bin);

            path = reamix::remix::assembleBlocks (validQueue, infos,
                                                   bt.data(), (int) bt.size(),
                                                   compat,
                                                   /*variation=*/0,
                                                   in_.junctionVariations.empty()
                                                     ? nullptr
                                                     : &in_.junctionVariations,
                                                   /*edit_length_jump_scale=*/1.0,   // ADR-115 P3: density has no Blocks axis
                                                   /*allow_outside_window=*/in_.block_assembly_beta,
                                                   /*min_keep_beats=*/barBeats);   // DEV-094 sesja 119
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
            rcin.v2_scoring = in_.v2_scoring;   // ADR-115 v2 scoring
            rcin.entry_beat = entry_beat;
            rcin.exit_beat  = exit_beat;
            // Shared with the loop-spot map (RegionCostWiring.h, ADR-115 E11).
            fillRegionCostInputs (rcin, bundle, gridDownbeats, gridBarBeats);

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
            roin.duration_tolerance_sec = reamix::remix::kDurationToleranceSecDefault;
            roin.candidates = &bundle.tc.candidates;
            roin.sample_rate = kAnalysisSampleRate;
            // ADR-115 P3 (sesja 123) — Edit density in Region = the minimum
            // loop length in bars. Default (0 / 1 bar) = the E8 cooldown of
            // one measured bar (bit-exact); a longer detent raises the
            // cooldown, clamped so at least two runs fit the region (a
            // 16-bar minimum inside an 8-bar region would leave the DP no
            // path). The jump tax is not scaled here: the E8 band DP is
            // quality-first and the cooldown is the axis that changes what
            // the user hears.
            roin.edit_length_jump_scale      = 1.0;
            {
                const int bars = in_.edit_density_bars;
                int overrideBeats = 0;
                if (bars > 1)
                {
                    const int regionBeats = std::max (1, exit_beat - entry_beat);
                    overrideBeats = std::min (bars * gridBarBeats,
                                              std::max (gridBarBeats, regionBeats / 2));
                    if (overrideBeats <= gridBarBeats) overrideBeats = 0;   // = default
                }
                roin.min_seq_after_jump_override = overrideBeats;
            }

            // ADR-057 (sesja 68) — Region boundaries → mechanical match to
            // user-selection edges. splice_flex_beats=0 activates legacy
            // argminAbsDiff path (closest beat to region.startSec/endSec);
            // Renderer + Insert pipeline override first/last clip source
            // bounds to user-selection so boundaries are sample-exact match.
            // Supersedes ADR-054 soft-boundary downbeat search.
            roin.downbeats         = gridDownbeats.empty() ? nullptr : gridDownbeats.data();
            roin.n_downbeats       = (int) gridDownbeats.size();
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

            // ADR-115 E8 (sesja 116, DEV-090) — v2 region path search with
            // the measured bar as cooldown / repetition sigma.
            roin.v2_scoring = in_.v2_scoring;
            roin.bar_beats  = gridBarBeats;

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
                tcin.v2_scoring  = in_.v2_scoring;   // ADR-115 v2 scoring
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

                tcin.downbeats   = gridDownbeats.empty() ? nullptr : gridDownbeats.data();
                tcin.n_downbeats = (int) gridDownbeats.size();

                tcin.time_signature  = gridBarBeats;
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

            oin.downbeats   = gridDownbeats.empty() ? nullptr : gridDownbeats.data();
            oin.n_downbeats = (int) gridDownbeats.size();

            oin.time_signature = gridBarBeats;
            oin.sample_rate    = kAnalysisSampleRate;

            // ADR-115 P3 (sesja 123) — Edit density in Duration. The default
            // (0 / 4 bars = COOLDOWN_BARS) keeps the legacy path bit-exact:
            // no cooldown override (the adaptive scaling for ratios < 0.5
            // stays), jump scale 1.0, transition cap 6. Another detent sets
            // the cooldown to bars x TS (bypassing the adaptive scaling, as
            // the Min cut override did), scales the jump tax by bars / 4
            // (16 bars = 4x, 1 bar = 0.25x - the ADR-084 range over the five
            // detents) and scales the transition cap inversely (16 bars: 2,
            // 8: 3, 4: 6, 2: 12, 1: 16) so "more cuts" can actually add cuts.
            oin.duration_tolerance_sec      = reamix::remix::kDurationToleranceSecDefault;
            {
                constexpr int kDefaultBars = reamix::remix::COOLDOWN_BARS;
                const int bars = in_.edit_density_bars > 0 ? in_.edit_density_bars : kDefaultBars;
                if (bars != kDefaultBars)
                {
                    oin.min_seq_after_jump_override = bars * gridBarBeats;
                    oin.edit_length_jump_scale      = (double) bars / (double) kDefaultBars;
                    oin.max_transitions             = juce::jlimit (2, 16,
                        juce::roundToInt ((double) reamix::remix::kMaxTransitionsDefaultOpt
                                          * (double) kDefaultBars / (double) bars));
                }
            }

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

        for (std::size_t ti = 0; ti < path.transitions.size(); ++ti)
        {
            const auto& tr = path.transitions[ti];
            const int fb = tr.first;
            const int tb = tr.second;
            float quality   = 0.0f;
            float energyDb  = 0.0f;
            float overlap   = 0.0f;
            int   anchor    = 0;
            // DEV-109 (sesja 123): per-occurrence junction index when the
            // Blocks path carries it; the beat-pair metadata is the fallback.
            int   junction  = ti < path.transition_junctions.size()
                              ? path.transition_junctions[ti] : -1;
            int   fallback  = 0;
            auto it = path.transition_metadata.find (tr);
            if (it != path.transition_metadata.end())
            {
                auto qit = it->second.find ("quality_score");
                if (qit != it->second.end()) quality = (float) qit->second;
                auto eit = it->second.find ("energy_diff_db");
                if (eit != it->second.end()) energyDb = (float) eit->second;
                auto jit = it->second.find ("junction_idx");
                if (junction < 0 && jit != it->second.end()) junction = (int) jit->second;
                auto fit = it->second.find ("junction_fallback");   // sesja 119 DEV-094
                if (fit != it->second.end()) fallback = (int) fit->second;
                auto oit = it->second.find ("resolved_overlap_sec");   // DEV-087
                if (oit != it->second.end()) overlap = (float) oit->second;
                anchor = it->second.count ("anchor_overlap_samples") ? 1 : 0;
            }
            out.transitionFallbacks.push_back (fallback);
            out.transitionJunctions.push_back (junction);
            out.transitionFromBeats.push_back (fb);
            out.transitionToBeats.push_back (tb);
            out.transitionQualities.push_back (quality);
            out.transitionEnergyDiffsDb.push_back (energyDb);
            out.transitionOverlapSec.push_back (overlap);
            out.transitionAnchorAccepted.push_back (anchor);
            if (junction >= 0 && junction < (int) blockJunctionLabels.size())
            {
                out.transitionFromLabels.push_back (blockJunctionLabels[(std::size_t) junction].first);
                out.transitionToLabels.push_back (blockJunctionLabels[(std::size_t) junction].second);
            }
            else
            {
                out.transitionFromLabels.push_back (labelAtBeat (fb));
                out.transitionToLabels.push_back (labelAtBeat (tb));
            }
        }
    }

    postProgress ("Done", kPWav);
    postCompletion (std::move (out));
}

} // namespace reamix::ui
