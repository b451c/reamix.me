#include "ui/RemixPipeline.h"
#include "remix/BeatGrid.h"   // ADR-115 E5 (sesja 115)
#include "RegionCostWiring.h"  // ADR-115 E11 (sesja 117)
#include "BlockCompatWiring.h"  // sesja 120 (DEV-097)

#include <juce_audio_formats/juce_audio_formats.h>

#include "remix/BlockAssembly.h"
#include "remix/Optimizer.h"
#include "remix/Path.h"
#include "ui/EditDensity.h"   // DEV-112 (sesja 124): density detent -> cut floor
#include "remix/RegionCost.h"
#include "remix/RegionOptimizer.h"
#include "remix/TransitionCost.h"
#include "remix/SpliceAcceptance.h"   // ADR-116 step 3 (sesja 130)
#include "remix/SeamJudge.h"          // ADR-117 (sesja 131)
#include "remix/ShapePlanner.h"       // ADR-117 (sesja 131)
#include "render/Renderer.h"
#include "ui/RemixCache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>

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
            bin.disable_boundary_family = in_.disable_boundary_family;   // sesja 130 (ADR-116 step 3)
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
            auto runRegion = [&] (const double* W)
            {
                return (in_.variation > 0)
                    ? ropt.remix_variation (in_.targetDurationSec, regStart, regEnd, W,
                                            rcr.n_region, &rcr.candidates, in_.variation, blockedPtr)
                    : ropt.remix (in_.targetDurationSec, regStart, regEnd, W,
                                  rcr.n_region, &rcr.candidates, blockedPtr);
            };
            // DEV-117 (d) (sesja 127): the Duration waveform floor (DEV-116)
            // in Region - the unfiltered run first (the baseline), then tiers
            // 0.80 / 0.70 / 0.60 masked in region_W; a tier is accepted when
            // the Region DP still returns >= 1 cut (when the target differs
            // from the region's own length), every cut at q >= 0.45, the
            // path length inside the tolerance (+2 s) and NO MORE cuts than
            // the baseline (Woodkid 60-90 s -> 90 s: the 0.60 tier left one
            // 2-bar loop and the DP took it 15 times where the baseline had
            // 8 cuts - a floor must not buy its cleanliness with cuts). The
            // legacy path (v2 off) = the single unfiltered run.
            if (! in_.v2_scoring)
            {
                path = runRegion (rcr.region_W.data());
            }
            else
            {
                constexpr double kAcceptMinQ = 0.45;
                const double tol      = roin.duration_tolerance_sec + 2.0;
                const double regionLen = bt[(std::size_t) std::min (exit_beat, (int) bt.size() - 1)]
                                         - bt[(std::size_t) entry_beat];
                const bool needsCut   = std::fabs (regionLen - in_.targetDurationSec) > roin.duration_tolerance_sec;
                std::vector<double> maskedW;
                reamix::remix::RemixPath baseline = runRegion (rcr.region_W.data());
                const std::size_t baselineCuts = baseline.transitions.size();
                bool haveBest = false;
                for (const double tier : { 0.80, 0.70, 0.60 })
                {
                    const double* W = rcr.region_W.data();
                    if (baselineCuts == 0) break;
                    {
                        maskedW = rcr.region_W;
                        for (const auto& kv : rcr.candidates)
                            if (kv.second.waveform_similarity < tier)
                            {
                                const int ri = kv.first.first - entry_beat, rj = kv.first.second - entry_beat;
                                if (ri >= 0 && rj >= 0 && ri < rcr.n_region && rj < rcr.n_region)
                                    maskedW[(std::size_t) ri * (std::size_t) rcr.n_region + (std::size_t) rj]
                                        = reamix::remix::INF;
                            }
                        W = maskedW.data();
                    }
                    reamix::remix::RemixPath cand = runRegion (W);
                    bool ok = ! cand.beat_indices.empty() && (! needsCut || ! cand.transitions.empty())
                              && cand.transitions.size() <= baselineCuts;
                    double minQ = 1.0;
                    for (const auto& tr : cand.transitions)
                    {
                        auto it = cand.transition_metadata.find (tr);
                        const double q = (it != cand.transition_metadata.end() && it->second.count ("quality_score"))
                                         ? it->second.at ("quality_score") : 0.0;
                        minQ = std::min (minQ, q);
                    }
                    if (minQ < kAcceptMinQ) ok = false;
                    double len = 0.0;
                    for (const int b : cand.beat_indices)
                        len += (b + 1 < (int) bt.size()) ? bt[(std::size_t) b + 1] - bt[(std::size_t) b]
                                                         : bt[bt.size() - 1] - bt[bt.size() - 2];
                    if (std::fabs (len - in_.targetDurationSec) > tol) ok = false;
                    if (const char* dbg = std::getenv ("REAMIX_REGION_DEBUG"))
                        if (FILE* f = std::fopen (dbg, "a"))
                        {
                            std::fprintf (f, "# region tier %.2f: cuts %d minQ %.3f len %.1f target %.1f -> %s\n",
                                          tier, (int) cand.transitions.size(), minQ, len,
                                          in_.targetDurationSec, ok ? "ACCEPT" : "reject");
                            std::fclose (f);
                        }
                    if (ok) { path = std::move (cand); out.waveformFloorUsed = tier; haveBest = true; break; }
                }
                if (! haveBest) path = std::move (baseline);
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
                // Sesja 129 — edge continuity (voice-band mel END edges).
                tcin.edge_mel_end = bundle.feat.edgeMelEnd.empty() ? nullptr : bundle.feat.edgeMelEnd.data();
                tcin.n_edge_mel   = bundle.feat.edgeMelEnd.empty() ? 0 : reamix::analysis::FeatureExtractor::kEdgeMelBands;
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
                tcin.disable_boundary_family = in_.disable_boundary_family;   // sesja 130

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
            oin.hole_aware_length = in_.v2_scoring;   // DEV-114 sesja 126
            if (in_.v2_scoring)                        // DEV-116 sesja 126: +-8 s flat
            {
                oin.duration_tolerance_sec = reamix::remix::kDurationToleranceSecV2;
                oin.flat_tolerance         = true;
            }
            oin.sample_rate    = kAnalysisSampleRate;

            // ADR-115 P3 (sesja 123) + DEV-112 (sesja 124) — Edit density in
            // Duration. The default (0 / 4 bars = COOLDOWN_BARS) keeps the
            // legacy path bit-exact: no cooldown override (the adaptive
            // scaling for ratios < 0.5 stays), jump scale 1.0, transition
            // cap 6, no density floor.
            // "Fewer cuts" (8 / 16 bars): cooldown bars x TS (bypassing the
            // adaptive scaling, as the Min cut override did), jump tax x
            // bars / 4 (2x / 4x) and the transition cap 6 x 4 / bars (3 / 2)
            // - a preference the DP already leans to, so it changes little.
            // "More cuts" (detents 2 / 1): at least 2 / 4 cuts, forward-only
            // when shortening, phrase cooldown untouched; best effort when no
            // jump bonus reaches the floor (viterbiDPWithJumpFloor). The
            // detent value is still "bars" for the cache key / harness; the
            // Duration bar labels read the cut counts (EditTuningBar).
            // DEV-112: a maximum-run gate (cut every N bars) was measured
            // and rejected - sparse pools strand or loop the path.
            oin.duration_tolerance_sec      = reamix::remix::kDurationToleranceSecDefault;
            {
                constexpr int kDefaultBars = reamix::remix::COOLDOWN_BARS;
                const int bars = in_.edit_density_bars > 0 ? in_.edit_density_bars : kDefaultBars;
                if (bars > kDefaultBars)
                {
                    oin.min_seq_after_jump_override = bars * gridBarBeats;
                    oin.edit_length_jump_scale      = (double) bars / (double) kDefaultBars;
                    oin.max_transitions             = juce::jlimit (2, 16,
                        juce::roundToInt ((double) reamix::remix::kMaxTransitionsDefaultOpt
                                          * (double) kDefaultBars / (double) bars));
                }
                else if (bars < kDefaultBars)
                {
                    oin.min_jumps_floor             = reamix::ui::densityMinCuts (bars);
                    oin.no_backward_when_shortening = true;
                }
            }

            // DEV-027 fix landed sesja 58 (ADR-048): when variation > 0, dispatch
            // through `remix_variation` which calls `remix_k_best(target,
            // max(2, v+1), blocked)` and indexes `paths[min(v, len-1)]`. For
            // variation == 0 stay on the fast path `remix(target, blocked)` —
            // identical result, skips k-best machinery.
            // Empty blocked set ⇒ pass nullptr per CleanOptimizer::remix signature.
            // DEV-116 (sesja 126): the optimizer's duration universe is
            // [first beat, last beat + period] while the renderer keeps the
            // un-beated head (0 .. first beat) and tail (last beat .. file end)
            // verbatim (Renderer: runs.front/back extension), so the DP target
            // must exclude them or every remix overshoots by head + tail
            // (Alice in Chains: 33 s intro + 23 s tail = +53 s at every ratio).
            double dpTarget = in_.targetDurationSec;
            double tailSec  = 0.0;
            if (in_.v2_scoring && bundle.beatTimes.size() >= 2 && bundle.nativeSr > 0)
            {
                const double trackSec = (double) bundle.nativeSamples / (double) bundle.nativeSr;
                const double period   = (bundle.beatTimes.back() - bundle.beatTimes.front())
                                        / (double) (bundle.beatTimes.size() - 1);
                const double head     = bundle.beatTimes.front();
                tailSec  = juce::jmax (0.0, trackSec - (bundle.beatTimes.back() + period));
                dpTarget = juce::jmax (4.0 * period, in_.targetDurationSec - head - tailSec);
            }
            // The renderer appends the file tail only when the path ends
            // within its last 3 beats (Renderer: isRegion / skipExtension), so
            // the v2 DP is told to end there (end_within_last_beats = 3,
            // ViterbiDP fallback when unreachable) and the target excludes the
            // tail consistently.
            oin.end_within_last_beats = in_.v2_scoring ? 3 : 0;
            const int  nBeatsAll = (int) bundle.beatTimes.size();
            auto endsAtTail = [&] (const reamix::remix::RemixPath& p)
            {
                return ! p.beat_indices.empty() && p.beat_indices.back() >= nBeatsAll - 3;
            };
            auto runDp = [&] (reamix::remix::CleanOptimizer& opt)
            {
                return (in_.variation > 0) ? opt.remix_variation (dpTarget, in_.variation, blockedPtr)
                                           : opt.remix (dpTarget, blockedPtr);
            };
            // DEV-116 (sesja 126): waveform-similarity floor with a DP retry.
            // Below ~0.8 the crossfade blends misaligned waveforms (Dylan
            // 0.64 / 0.77 = the two cuts rated bad after the phrase gate,
            // every cut rated ok had >= 0.84). Tiers 0.8 / 0.7 / 0.6 / none.
            // DEV-117 (sesja 127): a tier is accepted only when the path
            // ends at the song's ending (the renderer appends the head and
            // the tail only then; vocal_solo x1.25 was accepted at 0.80 with
            // the path 14 beats short and lost 45 s of head + tail), every
            // cut is at or above kAcceptMinQ, and the estimated render
            // length is within kMaxLengthDevSec of the target (user spec:
            // 5-8 s, 10 max). When no tier manages the length, the highest
            // tier that ends at the tail wins (length off, logged); the
            // unfiltered pool is the last resort. Legacy path (v2_scoring
            // off) = the single unfiltered run.
            // ADR-116 step 3 (sesja 130): the tier vocabulary lives in
            // SpliceAcceptance.h; a tier masks a continuation candidate by
            // its waveform xcorr and a boundary candidate by its edge distance.
            constexpr double kAcceptMinQ      = reamix::remix::kAcceptMinQ;
            const     double kMaxLengthDevSec = in_.maxLengthDevSec;
            const auto&      tiers            = reamix::remix::kAcceptTiers;
            struct TierCand
            {
                double tier;
                reamix::remix::RemixPath path;
                bool   qOk, tail;
                double dev;
            };
            std::vector<TierCand> cands;
            std::size_t passInsert = 0;   // sesja 130: pass-1 candidates are inserted at the front, in tier order
            std::vector<double> maskedW;
            double floorUsed = 0.0;
            reamix::remix::RemixPath best;
            bool haveBest = false;
            const double trackSec = (bundle.nativeSr > 0)
                                    ? (double) bundle.nativeSamples / (double) bundle.nativeSr : 0.0;
            // Estimated render length: beat durations along the path; with
            // the tail reached the renderer plays the file head and the tail
            // verbatim (Renderer: runs.front/back extension).
            auto estimateSec = [&] (const reamix::remix::RemixPath& p, bool tail)
            {
                const auto& bt = bundle.beatTimes;
                if (p.beat_indices.empty() || bt.size() < 2) return 0.0;
                auto dur = [&] (int b)
                {
                    return (b + 1 < (int) bt.size()) ? bt[(std::size_t) b + 1] - bt[(std::size_t) b]
                                                     : bt[bt.size() - 1] - bt[bt.size() - 2];
                };
                double len = 0.0;
                const std::size_t n = p.beat_indices.size();
                for (std::size_t k = 0; k + (tail ? 1 : 0) < n; ++k) len += dur (p.beat_indices[k]);
                if (tail) len += bt.front() + juce::jmax (0.0, trackSec - bt[(std::size_t) p.beat_indices.back()]);
                return len;
            };
            // ADR-116 step 3 (sesja 130): the boundary family is a RESCUE, not
            // a competitor. Its composite (no waveform term) sits on another
            // scale than the continuation composite, so with both families
            // in one pool the DP swapped 34 of 51 normal-ratio corpus cases
            // (Woodkid x0.75 to a red cut, Periphery x1.25 to one boundary
            // loop x6). Pass 0 = the continuation pool alone (the tiers
            // exactly as sesja 129, bit-exact when a tier accepts); pass 1 =
            // both families, only when no tier accepted in pass 0 (extreme
            // ratios: High Hopes 3:11 -> 0:30, corpus 0.15-0.33). The
            // fallback then prefers the pass-1 pool (richer) over pass 0.
            std::optional<reamix::remix::RemixPath> shapeEffort;   // ADR-117 tier F (see the fallback below)
            double shapeEffortDev = 0.0;
            // ADR-117 (sesja 131): shape-first planner for extreme shortening.
            // Below kShapePlannerMaxRatio of the track the plan is whole
            // sections of the grid-snapped section map in the original order,
            // seams only at section boundaries judged as boundary cuts
            // (SeamJudge on the shared pair scorer); no plan = the beat-level
            // engine below runs as before. Ratios >= the switch never enter.
            if (in_.v2_scoring && ! in_.disable_shape_planner && trackSec > 0.0
                && in_.targetDurationSec < reamix::remix::kShapePlannerMaxRatio * trackSec
                && (int) bundle.uiSegments.size() >= reamix::remix::kShapeMinSections)
            {
                std::vector<double> segStarts, segEnds;
                std::vector<int>    segKinds;
                for (const auto& s : bundle.uiSegments)
                {
                    segStarts.push_back (s.startSec);
                    segEnds.push_back (s.endSec);
                    segKinds.push_back ((int) s.kind);
                }
                const auto sections = reamix::remix::shapeSectionsFromSeconds (
                    bundle.beatTimes.data(), nBeatsAll, segStarts.data(), segEnds.data(),
                    segKinds.data(), (int) segStarts.size());
                std::set<int> dbSet (v2Grid.downbeat_idx.begin(), v2Grid.downbeat_idx.end());
                // Sesja 135 (DEV-122): bar starts continued into the lattice
                // zones, for the planner only (the engine grid has none there).
                for (int idx : reamix::remix::latticeDownbeatIdx (bundle.beatIsSynthetic, v2Grid.downbeat_idx, v2Grid.bar_beats))
                    dbSet.insert (idx);

                reamix::remix::BlockCompatInputs jin{};
                fillBlockCompatInputs (jin, bundle, gridDownbeats, gridBarBeats);
                jin.v2_scoring = true;
                const reamix::remix::BoundarySeamJudge judge (jin);

                reamix::remix::ShapePlannerInputs sin;
                sin.beat_times  = bundle.beatTimes.data();
                sin.n_beats     = nBeatsAll;
                sin.track_sec   = trackSec;
                sin.sections    = sections.data();
                sin.n_sections  = (int) sections.size();
                sin.db_set      = &dbSet;
                sin.rms_energy  = bundle.feat.rmsEnergy.empty() ? nullptr : bundle.feat.rmsEnergy.data();
                sin.target_sec  = in_.targetDurationSec;
                // Sesja 134 (ADR-117 step 3, DEV-121): Audition's structure -
                // the length slack is 5 s at every tier, the first / last piece
                // may stop / start at any bar of the first / last section, a
                // seam landing on a section start or inside the last section is
                // judged with every loudness gate off and no q floor (the
                // Audition round: such cut-ins with +7..+11 dB steps were rated
                // clean; our composite does not separate the one rejected cut
                // from the accepted ones), and the seam crossfade is one beat.
                sin.window_sec  = reamix::remix::kShapeLengthSlackSec;
                sin.window_relaxed_sec = reamix::remix::kShapeLengthSlackSec;
                sin.min_q       = kAcceptMinQ;
                sin.bar_ends    = true;
                sin.bar_beats   = gridBarBeats;   // sesja 135: section-start zone = the first bar
                sin.beat_is_synthetic = bundle.beatIsSynthetic.empty() ? nullptr : &bundle.beatIsSynthetic;   // sesja 135 (DEV-122)
                sin.seam_crossfade_beats = in_.shapeSeamCrossfadeBeats;
                sin.recipe_mode           = in_.shapeRecipeMode;   // sesja 136 (ADR-117 step 5)
                sin.seam_crossfade_max_sec = in_.shapeSeamMaxSec;
                auto judgeFn = [&judge] (bool relaxed, bool open)
                {
                    return [&judge, relaxed, open] (int i, int j) -> std::optional<reamix::remix::ShapeSeamScore>
                    {
                        const auto s = judge.score (i, j, relaxed, open);
                        if (s.rejected) return std::nullopt;
                        return reamix::remix::ShapeSeamScore { s.quality, s.energy_diff_db, s.edge_distance };
                    };
                };
                sin.seam         = judgeFn (false, false);
                sin.seam_relaxed = judgeFn (true, false);
                sin.seam_open    = judgeFn (true, true);
                const reamix::remix::ShapePlan plan = judge.valid() ? reamix::remix::planShape (sin)
                                                                    : reamix::remix::ShapePlan{};
                if (const char* dbg = std::getenv ("REAMIX_DURATION_DEBUG"))
                {
                    if (FILE* f = std::fopen (dbg, "a"))
                    {
                        std::fprintf (f, "shape: sections %d judge %s -> %s tier %c pieces %d seams %d minQ %.3f est %.1f dev %+.1f"
                                         " | pieces whole %d trim %d, seams tried %d strict-ok %d relaxed-ok %d, closest |dev| whole %.1f relaxed/trim %.1f\n",
                                      (int) sections.size(), judge.valid() ? "ok" : "invalid",
                                      plan.ok ? "PLAN" : "no plan", plan.tier, (int) plan.pieces.size(),
                                      (int) plan.seams.size(), plan.min_q, plan.est_sec, plan.dev_sec,
                                      plan.diag.pieces_whole, plan.diag.pieces_trim, plan.diag.seams_tried,
                                      plan.diag.seams_strict, plan.diag.seams_relaxed,
                                      plan.diag.closest_dev_whole, plan.diag.closest_dev_trim);
                        for (const auto& pc : plan.pieces)
                            std::fprintf (f, "  piece section %d kind %d beats %d-%d %s\n", pc.section, pc.kind, pc.b0, pc.b1,
                                          pc.trim == reamix::remix::ShapePiece::Trim::Whole ? "whole"
                                          : pc.trim == reamix::remix::ShapePiece::Trim::Head ? "head" : "tail");
                        for (const auto& sm : plan.seams)
                            std::fprintf (f, "  seam %d -> %d q %.3f ed %.2f excess %+.1f dB%s xfade %.3f s\n",
                                          sm.i, sm.j, sm.score.q, sm.score.edge_distance, sm.excess_db,
                                          sm.open ? " OPEN" : "", sm.overlap_sec);
                        if (std::getenv ("REAMIX_SHAPE_SEAMS") != nullptr)
                        {
                            // Every seam the search asked about: section-boundary pairs with the
                            // judge's verdict (gate 1 = 8 dB hard block, 2 = p98 loudness reject).
                            auto sectionOf = [&] (int b) {
                                for (std::size_t k = 0; k < sections.size(); ++k)
                                    if (b >= sections[k].b0 && b < sections[k].b1) return (int) k;
                                return -1; };
                            for (const auto& jd : plan.diag.judged)
                            {
                                const auto sc = judge.score (jd.i, jd.j);
                                std::fprintf (f, "  judged %d -> %d (sec %d kind %d -> sec %d kind %d) %s q %.3f ed %.2f gate %d excess %+.1f\n",
                                              jd.i, jd.j, sectionOf (jd.i), sectionOf (jd.i) >= 0 ? sections[(std::size_t) sectionOf (jd.i)].kind : -1,
                                              sectionOf (jd.j), sectionOf (jd.j) >= 0 ? sections[(std::size_t) sectionOf (jd.j)].kind : -1,
                                              jd.strict_ok ? "OK" : (jd.relaxed_ok ? "relaxed-only" : (sc.rejected ? "GATED" : "low-q")),
                                              sc.rejected ? -1.0 : sc.quality, sc.edge_distance, sc.gate,
                                              [&] { const int K = reamix::remix::kShapeContextBeats; auto ctx = [&] (int a, int b) {
                                                        a = std::max (0, a); b = std::min (nBeatsAll, b); if (b <= a || sin.rms_energy == nullptr) return 0.0;
                                                        double m = 0; for (int k = a; k < b; ++k) m += sin.rms_energy[k]; return 20.0 * std::log10 (std::max (m / (b - a), 1e-6)); };
                                                    return ctx (jd.j - K, jd.j) - ctx (jd.i - K + 1, jd.i + 1); }());
                            }
                        }
                        std::fclose (f);
                    }
                }
                if (plan.ok)
                {
                    best = plan.toPath();
                    haveBest = true;
                    out.shapePlanUsed = true;
                }
                else if (plan.best_effort)
                {
                    shapeEffort    = plan.toPath();
                    shapeEffortDev = plan.dev_sec;
                }
            }

            bool hasBoundary = false;
            for (const auto& kv : tcSrc->candidates)
                if (kv.second.family == reamix::remix::TransitionCandidate::kFamilyBoundary) { hasBoundary = true; break; }
            for (int pass = 0; pass < 2 && ! haveBest; ++pass)
            {
            if (pass == 1 && (! hasBoundary || ! in_.v2_scoring)) break;
            const bool withBoundary = pass == 1;
            for (const double tier : tiers)
            {
                if (tier > 0.0 && ! in_.v2_scoring) continue;
                if (tier > 0.0 || (hasBoundary && ! withBoundary))
                {
                    maskedW = tcSrc->W;
                    for (const auto& kv : tcSrc->candidates)
                        if ((! withBoundary && kv.second.family == reamix::remix::TransitionCandidate::kFamilyBoundary)
                            || reamix::remix::maskedAtTier (kv.second, tier))
                            maskedW[(std::size_t) kv.first.first * (std::size_t) tcSrc->n_beats
                                    + (std::size_t) kv.first.second] = reamix::remix::INF;
                    oin.W = maskedW.data();
                }
                else
                {
                    oin.W = tcSrc->W.data();
                }
                reamix::remix::CleanOptimizer opt (oin);
                reamix::remix::RemixPath cand = runDp (opt);
                if (! in_.v2_scoring) { best = std::move (cand); floorUsed = 0.0; haveBest = true; break; }
                // A remix that changes the length needs >= 1 cut: the DP's
                // empty-path fallback (a straight run of the first beats =
                // a truncated song) is never accepted at a tier.
                bool   qOk  = ! cand.beat_indices.empty() && ! cand.transitions.empty();
                double minQ = 1.0;
                for (const auto& tr : cand.transitions)
                {
                    auto it = cand.transition_metadata.find (tr);
                    const double q = (it != cand.transition_metadata.end() && it->second.count ("quality_score"))
                                     ? it->second.at ("quality_score") : 0.0;
                    minQ = std::min (minQ, q);
                }
                if (minQ < kAcceptMinQ) qOk = false;
                int bnd = 0;   // sesja 130: boundary-family cuts on the path
                for (const auto& tr : cand.transitions)
                {
                    auto ci = tcSrc->candidates.find (tr);
                    if (ci != tcSrc->candidates.end()
                        && ci->second.family == reamix::remix::TransitionCandidate::kFamilyBoundary) ++bnd;
                }
                const bool   tail = endsAtTail (cand);
                const double est  = estimateSec (cand, tail);
                const double dev  = est - in_.targetDurationSec;
                const bool   ok   = qOk && tail && std::fabs (dev) <= kMaxLengthDevSec;
                if (const char* dbg = std::getenv ("REAMIX_DURATION_DEBUG"))
                {
                    if (FILE* f = std::fopen (dbg, "a"))
                    {
                        std::fprintf (f, "pass %d tier %.2f: cuts %d bnd %d minQ %.3f est %.1f dev %+.1f dp-target %.1f last %d/%d %s -> %s\n",
                                      pass, tier, (int) cand.transitions.size(), bnd, minQ, est, dev, dpTarget,
                                      cand.beat_indices.empty() ? -1 : cand.beat_indices.back(), nBeatsAll,
                                      tail ? "ends@tail" : "ends-early", ok ? "ACCEPT" : "reject");
                        std::fclose (f);
                    }
                }
                if (ok) { best = std::move (cand); floorUsed = tier; haveBest = true; break; }
                // Pass-1 candidates go first so the fallback prefers the
                // richer pool (a boundary cut over a red continuation cut).
                if (withBoundary) cands.insert (cands.begin() + passInsert++, TierCand { tier, std::move (cand), qOk, tail, dev });
                else              cands.push_back ({ tier, std::move (cand), qOk, tail, dev });
            }
            }
            if (! haveBest)
            {
                // No tier makes the length with clean cuts. The user's length
                // comes first (High Hopes 3:11 -> 0:30: the pool has one
                // intro -> outro pair at q 0.25; a 36 s remix with that red
                // cut is the request, a 51 s remix with two orange cuts is
                // not - "totalna klapa"): the highest tier whose path ends at
                // the tail inside the cap, whatever its cuts; then the
                // highest tier ending at the tail with clean cuts (length
                // off); then the unfiltered pool's path as it is.
                double chosenDev   = 0.0;
                bool   insideCap   = false;
                for (const auto& c : cands)
                    if (c.tail && std::fabs (c.dev) <= kMaxLengthDevSec) { best = c.path; floorUsed = c.tier; haveBest = true; chosenDev = c.dev; insideCap = true; break; }
                if (! haveBest)
                    for (const auto& c : cands)
                        if (c.qOk && c.tail) { best = c.path; floorUsed = c.tier; haveBest = true; chosenDev = c.dev; break; }
                if (! haveBest && ! cands.empty()) { best = cands.back().path; floorUsed = cands.back().tier; haveBest = true; chosenDev = cands.back().dev; }
                // ADR-117 (sesja 131): when the beat-level fallback also misses
                // the cap, the shape planner's best-effort plan wins if it is
                // closer to the requested length (Drake x0.25: 80 s vs 125 s).
                bool shapeTaken = false;
                if (shapeEffort.has_value() && ! insideCap
                    && (! haveBest || std::fabs (shapeEffortDev) < std::fabs (chosenDev)))
                {
                    best = *shapeEffort; floorUsed = 0.0; haveBest = true; shapeTaken = true;
                    out.shapePlanUsed = true;
                }
                if (const char* dbg = std::getenv ("REAMIX_DURATION_DEBUG"))
                {
                    if (FILE* f = std::fopen (dbg, "a"))
                    {
                        std::fprintf (f, "fallback: floor %.2f (%s)%s\n", floorUsed,
                                      haveBest && endsAtTail (best) ? "ends@tail, length off" : "ends-early",
                                      shapeTaken ? " -> shape best effort (closer to the target)" : "");
                        std::fclose (f);
                    }
                }
            }
            if (! haveBest)
            {
                oin.W = tcSrc->W.data();
                reamix::remix::CleanOptimizer opt (oin);
                best = runDp (opt);
            }
            out.waveformFloorUsed = floorUsed;
            path = std::move (best);
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
    // DEV-115 (sesja 126): on the v2 path an anchor splice may overlay the
    // clips for at most 1 s (the corpus anchor overlaps were 1.8-3.5 s and
    // the user hears them as two passages at once); legacy = uncapped.
    if (in_.v2_scoring)
        rcfg.anchorMaxOverlapSec = 1.0;
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
            int   family    = -1;   // sesja 130
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
                auto fam = it->second.find ("family");   // sesja 130
                if (fam != it->second.end()) family = (int) fam->second;
            }
            out.transitionFamilies.push_back (family);
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
