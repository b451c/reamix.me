// Renderer — phase-5 final module port
// (references/python-source/remix/renderer.py, 482 LOC). Per ADR-031 module
// ordering: Butterworth → PhaseAlign → Crossfade → Splice → **Renderer**.
//
// Port discipline: Python-verbatim with -ffp-contract=off per ADR-028 (16th
// reuse). Composite methods (transition resolution + render loop) compose
// already-validated Splice + Crossfade modules — no novel DSP. Numerical
// surface area is dominated by:
//   * np.linspace + np.cos + np.sin (f64) for fade ramps, applied in-place
//     to f32 audio buffers via `f32 *= f64` (numpy widens to f64, casts back).
//   * adaptive_crossfade(outgoing.astype(f64), incoming.astype(f64)).astype(f32).
//   * Integer arithmetic on sample positions / beat indices.
//
// Parity class targets:
//   * Per-sample max_abs ≤ 5e-4 on rendered audio f32 (phase-5 spec).
//   * Edit plan int-bit-exact on sample-position fields.
//   * EditClip f64 sec-fields max_abs ≤ 1e-9.
//   * transition_times f64 max_abs ≤ 1e-9.

#include "render/Renderer.h"
#include <cstdint>

#include "render/Crossfade.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace reamix::render {

namespace {

// pi/2 as a literal (matches Crossfade.cpp's kPi convention) — np.pi/2 in
// numpy is float64-precise and equals this value to the ULP.
constexpr double kPiHalf = 1.5707963267948966192313216916397514;

// f32 *= f64 in-place multiply matching numpy's `f32_buf[slice] *= f64_arr`
// semantic: numpy widens both operands to f64 for the arithmetic and casts
// the result back to f32 before storing. Done element-wise to match exactly.
inline void mulInPlaceF32F64(float* dst, const double* gain, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(static_cast<double>(dst[i]) * gain[i]);
    }
}

// Port of `np.linspace(start, stop, num)` for num >= 2 (inclusive endpoints).
// Matches numpy's `step = (stop - start) / (num - 1); out[i] = start + step*i`
// with the explicit `out[num-1] = stop` overwrite numpy applies (linspace.py).
// For num == 1 returns [start]; for num == 0 returns empty.
inline void linspace(double start, double stop, std::size_t num,
                     std::vector<double>& out)
{
    out.resize(num);
    if (num == 0) return;
    if (num == 1) {
        out[0] = start;
        return;
    }
    const double step = (stop - start) / static_cast<double>(num - 1);
    for (std::size_t i = 0; i < num; ++i) {
        out[i] = start + step * static_cast<double>(i);
    }
    out[num - 1] = stop; // numpy explicit endpoint write
}

// Map (from_beat, to_beat) → transition_metadata dict, with empty fallback.
// Mirror of Python `dict(path.transition_metadata.get(key, {}))`.
inline std::map<std::string, double> getTransitionMetaCopy(
    const reamix::remix::RemixPath& path, int prevBeat, int currBeat)
{
    auto key = std::make_pair(prevBeat, currBeat);
    auto it = path.transition_metadata.find(key);
    if (it == path.transition_metadata.end()) {
        return {};
    }
    return it->second;
}

inline double metaGet(const std::map<std::string, double>& meta,
                      const std::string& key, double fallback)
{
    auto it = meta.find(key);
    return it == meta.end() ? fallback : it->second;
}

inline bool metaHas(const std::map<std::string, double>& meta,
                    const std::string& key)
{
    return meta.find(key) != meta.end();
}

// Build a TransitionMeta struct (Splice consumer view) from the dict-of-floats
// representation Python uses. Only fields read by Splice composite methods.
inline TransitionMeta toSpliceMeta(const std::map<std::string, double>& m)
{
    TransitionMeta tm;
    tm.vocalPresenceLevel = metaGet(m, "vocal_presence_level", 0.0);
    tm.preferredOverlapSec = metaGet(m, "preferred_overlap_sec", 0.0);
    tm.labelMatch         = metaGet(m, "label_match", 1.0);
    tm.vocalEntrySupport  = metaGet(m, "vocal_entry_support", 1.0);
    tm.vocalExitSupport   = metaGet(m, "vocal_exit_support", 1.0);
    tm.alignmentOffsetSec = metaGet(m, "alignment_offset_sec", 0.0);
    return tm;
}

} // anon namespace

// ---------------------------------------------------------------------------
// Constructor (port of __init__, renderer.py:70-99)
// ---------------------------------------------------------------------------
Renderer::Renderer(const std::string&  audioPath,
                   const float*        audioF32,
                   std::size_t         nChannels,
                   std::size_t         nSamples,
                   int                 sampleRate,
                   const double*       beatTimes,
                   std::size_t         nBeats,
                   double              crossfadeMsOrNeg,
                   const RendererConfig& cfg)
    : audioPath_(audioPath)
    , nChannels_(nChannels == 0 ? 1 : nChannels)
    , nSamples_(nSamples)
    , sr_(sampleRate)
    , nBeats_(nBeats)
    , cfg_(cfg)
{
    if (audioF32 == nullptr || nSamples == 0) {
        throw std::invalid_argument("Renderer: empty audio buffer");
    }
    if (beatTimes == nullptr || nBeats == 0) {
        throw std::invalid_argument("Renderer: empty beat grid");
    }

    // Copy audio. Python L88-89: `if self._audio.ndim == 1: self._audio =
    // self._audio[np.newaxis, :]`. Caller is responsible for the (1, n)
    // reshape on mono inputs (we just preserve channel-major layout).
    audio_.assign(audioF32, audioF32 + nChannels_ * nSamples_);

    // Resolve crossfade_ms (Python L80-86). If crossfade_beats > 0 and
    // multiple beats present: ms = beats * avg_beat_dur * 1000. Otherwise
    // fall back to the explicit ms (or default).
    beatTimes_.assign(beatTimes, beatTimes + nBeats);
    avgBeatDuration_ = 0.5;
    if (nBeats > 1) {
        avgBeatDuration_ = (beatTimes_.back() - beatTimes_.front())
                         / static_cast<double>(nBeats - 1);
    }

    double crossfadeMs = crossfadeMsOrNeg;
    if (cfg_.crossfadeBeats > 0.0 && nBeats > 1) {
        crossfadeMs = cfg_.crossfadeBeats * avgBeatDuration_ * 1000.0;
    } else if (crossfadeMs < 0.0) {
        crossfadeMs = cfg_.crossfadeMs;
    }
    crossfadeSamples_ = static_cast<int>(crossfadeMs * sampleRate / 1000.0);

    // mono_energy = sqrt(mean(audio.astype(f64) ** 2, axis=0)) — Python L97.
    monoEnergy_.assign(nSamples_, 0.0);
    for (std::size_t i = 0; i < nSamples_; ++i) {
        double acc = 0.0;
        for (std::size_t c = 0; c < nChannels_; ++c) {
            const double v = static_cast<double>(audio_[c * nSamples_ + i]);
            acc += v * v;
        }
        monoEnergy_[i] = std::sqrt(acc / static_cast<double>(nChannels_));
    }

    // beat_samples = (beat_times * sample_rate).astype(np.int64) — Python L99.
    // numpy float→int64 cast is C-style truncation toward zero.
    beatSamples_.assign(nBeats, 0);
    for (std::size_t i = 0; i < nBeats; ++i) {
        beatSamples_[i] = static_cast<std::int64_t>(beatTimes_[i] * sampleRate);
    }

    // f64 mirror of audio buffer for Splice composite consumption. See
    // Renderer.h `audioF64Mirror_` comment for the memory-cost trade-off.
    audioF64Mirror_.assign(nChannels_ * nSamples_, 0.0);
    for (std::size_t i = 0; i < nChannels_ * nSamples_; ++i) {
        audioF64Mirror_[i] = static_cast<double>(audio_[i]);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
SpliceContext Renderer::makeSpliceContext() const
{
    SpliceContext ctx;
    // Splice expects f64 channel-major audio; we hold f32. Splice composite
    // methods that touch audio (searchAnchorTransitionGeometry,
    // refineTransitionSplice) cast on read via stereoWindowSimilarity which
    // is f64-only. To bridge cleanly: build an f64 view on demand.
    //
    // Lazy promotion strategy is too leaky here — Splice expects raw pointer.
    // Instead Renderer holds a persistent f64 audio mirror via a lazily-
    // populated cache. For now: build the mirror up front (memory cost
    // ~2× audio buffer; for an 8-min stereo @ 44100 Hz that's 21 MB f32 +
    // 42 MB f64 = 63 MB total — acceptable). See Renderer member audioF64Mirror_.
    ctx.audio        = audioF64Mirror_.data();
    ctx.nChAudio     = nChannels_;
    ctx.nAudio       = nSamples_;
    ctx.sr           = sr_;
    ctx.monoEnergy   = monoEnergy_.data();
    ctx.nMonoEnergy  = monoEnergy_.size();
    ctx.beatSamples  = beatSamples_.data();
    ctx.nBeats       = beatSamples_.size();
    ctx.crossfadeSamples = crossfadeSamples_;
    return ctx;
}

std::int64_t Renderer::beatEndSample(int beatIdx) const
{
    // Port of _beat_end_sample (renderer.py:101-109).
    if (static_cast<std::size_t>(beatIdx + 1) < beatSamples_.size()) {
        return beatSamples_[beatIdx + 1];
    }
    const std::size_t denom = std::max<std::size_t>(1, beatSamples_.size() - 1);
    const std::int64_t avgBeatSamples = static_cast<std::int64_t>(
        (beatSamples_.back() - beatSamples_.front())
        / static_cast<std::int64_t>(denom));
    const std::int64_t cap = static_cast<std::int64_t>(nSamples_);
    return std::min(beatSamples_[beatIdx] + avgBeatSamples * 2, cap);
}

// ---------------------------------------------------------------------------
// resolveTransitionEdit — port of _resolve_transition_edit (renderer.py:119-227)
// ---------------------------------------------------------------------------
Renderer::ResolvedTransition Renderer::resolveTransitionEdit(
    reamix::remix::RemixPath& path, int prevBeat, int currBeat)
{
    auto key = std::make_pair(prevBeat, currBeat);
    auto meta = getTransitionMetaCopy(path, prevBeat, currBeat);

    ResolvedTransition res;

    // Teaser-fade branch (Python L126-148).
    if (metaGet(meta, "teaser_fade", 0.0) >= 0.5) {
        const double feo = metaGet(meta, "teaser_fade_out_ms",
                                   cfg_.ultraShortTeaserFadeOutMs);
        const double fei = metaGet(meta, "teaser_fade_in_ms",
                                   cfg_.ultraShortTeaserFadeInMs);
        const double gap = metaGet(meta, "teaser_gap_ms",
                                   cfg_.ultraShortTeaserGapMs);
        const std::int64_t fadeOut =
            static_cast<std::int64_t>(std::nearbyint(feo * sr_ / 1000.0));
        const std::int64_t fadeIn  =
            static_cast<std::int64_t>(std::nearbyint(fei * sr_ / 1000.0));
        const std::int64_t gapS    =
            static_cast<std::int64_t>(std::nearbyint(gap * sr_ / 1000.0));
        path.transition_metadata[key] = meta; // Python writes back even on teaser path
        res.teaserFade        = true;
        res.fadeOutSamples    = std::max<std::int64_t>(0, fadeOut);
        res.fadeInSamples     = std::max<std::int64_t>(0, fadeIn);
        res.gapSamples        = std::max<std::int64_t>(0, gapS);
        return res;
    }

    // Cached anchor overlap path (Python L150-165).
    if (metaHas(meta, "anchor_overlap_samples")
        && metaHas(meta, "render_outgoing_cut_sample")
        && metaHas(meta, "render_incoming_cut_sample")) {
        const std::int64_t overlap = std::max<std::int64_t>(
            0, static_cast<std::int64_t>(std::nearbyint(
                metaGet(meta, "anchor_overlap_samples", 0.0))));
        const std::int64_t outCut = static_cast<std::int64_t>(
            metaGet(meta, "render_outgoing_cut_sample", 0.0));
        const std::int64_t inCut = static_cast<std::int64_t>(
            metaGet(meta, "render_incoming_cut_sample", 0.0));
        const std::int64_t bEnd = beatEndSample(currBeat);
        res.teaserFade           = false;
        res.overlapSamples       = overlap;
        res.outgoingCutSample    = outCut;
        res.incomingStartSample  = inCut;
        res.beatEndSample        = bEnd;
        return res;
    }

    // Main transition resolution (Python L167-220).
    const int successorBeat = prevBeat + 1;
    std::int64_t successorSample;
    if (static_cast<std::size_t>(successorBeat) < nBeats_) {
        successorSample = beatSamples_[successorBeat];
    } else {
        const std::int64_t end = beatEndSample(prevBeat);
        const std::int64_t cap = static_cast<std::int64_t>(nSamples_) - 1;
        successorSample = std::min(std::max<std::int64_t>(0, end), cap);
    }
    const std::int64_t beatStart = beatSamples_[currBeat];
    const std::int64_t beatEnd   = beatEndSample(currBeat);

    // Splice context + meta view.
    SpliceContext ctx = makeSpliceContext();
    TransitionMeta tm = toSpliceMeta(meta);

    const int transitionOverlap =
        Splice::transitionOverlapSamples(crossfadeSamples_, sr_, tm, cfg_.splice);
    const int halfXfade = transitionOverlap / 2;

    // refineTransitionSplice (Python L178-181).
    RefineResult refinement = Splice::refineTransitionSplice(
        ctx, successorSample, beatStart, beatEnd, tm,
        transitionOverlap, cfg_.splice);
    if (refinement.found) {
        // Update meta with refinement fields. Python L182-183: meta.update(refinement).
        // refineTransitionSplice writes 6 keys per Python L505-512:
        //   render_outgoing_cut_sample, render_incoming_cut_sample,
        //   render_outgoing_shift_sec, render_incoming_shift_sec,
        //   render_stereo_similarity, render_splice_score.
        meta["render_outgoing_cut_sample"] = static_cast<double>(refinement.outgoingCutSample);
        meta["render_incoming_cut_sample"] = static_cast<double>(refinement.incomingCutSample);
        meta["render_outgoing_shift_sec"]  = refinement.outgoingShiftSec;
        meta["render_incoming_shift_sec"]  = refinement.incomingShiftSec;
        meta["render_stereo_similarity"]   = refinement.stereoSimilarity01;
        meta["render_splice_score"]        = refinement.spliceScore;
    }

    // searchAnchorTransitionGeometry (Python L185-202).
    AnchorSearchResult anchor = Splice::searchAnchorTransitionGeometry(
        ctx, prevBeat, currBeat, tm, cfg_.splice);
    if (anchor.selected) {
        const double baselineQuality = metaGet(meta, "render_stereo_similarity",
            metaGet(meta, "waveform_similarity", 0.0));
        const double anchorQuality = anchor.score;        // render_anchor_splice_score
        const double localQuality  = anchor.localSimilarity01; // render_anchor_local_similarity
        const double improvementThreshold = cfg_.anchorImprovementThreshold;
        const double vocalThreshold       = cfg_.splice.vocalActivityThreshold;
        const double vocalPresence        = metaGet(meta, "vocal_presence_level", 0.0);
        const bool accept =
            (anchorQuality >= baselineQuality + improvementThreshold)
            || (vocalPresence >= vocalThreshold
                && localQuality >= baselineQuality + 0.01);
        if (accept) {
            // Python L202: meta.update(anchor_geometry). Anchor writes 15 keys
            // per searchAnchorTransitionGeometry (splice.py:319-330):
            //   render_anchor_out_beat, render_anchor_in_beat,
            //   render_anchor_incoming_start_beat, render_anchor_outgoing_boundary_beat,
            //   render_anchor_out_sec, render_anchor_in_sec,
            //   render_outgoing_start_sample, render_outgoing_cut_sample,
            //   render_incoming_cut_sample, render_incoming_end_sample,
            //   anchor_overlap_samples, render_anchor_splice_score,
            //   render_anchor_similarity, render_anchor_local_similarity,
            //   render_stereo_similarity (= render_anchor_similarity per L317).
            meta["render_anchor_out_beat"]               = static_cast<double>(anchor.outAnchorBeat);
            meta["render_anchor_in_beat"]                = static_cast<double>(anchor.inAnchorBeat);
            meta["render_anchor_incoming_start_beat"]    = static_cast<double>(anchor.incomingStartBeat);
            meta["render_anchor_outgoing_boundary_beat"] = static_cast<double>(anchor.outgoingBoundaryBeat);
            meta["render_anchor_out_sec"]                = anchor.anchorOutSec;
            meta["render_anchor_in_sec"]                 = anchor.anchorInSec;
            meta["render_outgoing_start_sample"]         = static_cast<double>(anchor.outgoingStartSample);
            meta["render_outgoing_cut_sample"]           = static_cast<double>(anchor.outgoingCutSample);
            meta["render_incoming_cut_sample"]           = static_cast<double>(anchor.incomingCutSample);
            meta["render_incoming_end_sample"]           = static_cast<double>(anchor.incomingEndSample);
            meta["anchor_overlap_samples"]               = static_cast<double>(anchor.anchorOverlapSamples);
            meta["render_anchor_splice_score"]           = anchor.score;
            meta["render_anchor_similarity"]             = anchor.similarity01;
            meta["render_anchor_local_similarity"]       = anchor.localSimilarity01;
            meta["render_stereo_similarity"]             = anchor.similarity01;
        }
    }

    // outgoing_cut / incoming_start clamping (Python L204-218).
    std::int64_t outgoingCut = static_cast<std::int64_t>(std::nearbyint(
        metaGet(meta, "render_outgoing_cut_sample",
                static_cast<double>(successorSample + halfXfade))));
    std::int64_t incomingStart = static_cast<std::int64_t>(std::nearbyint(
        metaGet(meta, "render_incoming_cut_sample",
                static_cast<double>(beatStart - halfXfade))));
    outgoingCut = std::max<std::int64_t>(transitionOverlap,
        std::min(outgoingCut, static_cast<std::int64_t>(nSamples_)));
    incomingStart = std::max<std::int64_t>(0, std::min(incomingStart, beatEnd));

    const std::int64_t metaOverlap = static_cast<std::int64_t>(std::nearbyint(
        metaGet(meta, "anchor_overlap_samples",
                static_cast<double>(transitionOverlap))));
    const std::int64_t inEnd = static_cast<std::int64_t>(std::nearbyint(
        metaGet(meta, "render_incoming_end_sample",
                static_cast<double>(beatEnd))));
    std::int64_t overlap = std::min({metaOverlap,
                                     outgoingCut,
                                     std::max<std::int64_t>(0, inEnd - incomingStart)});
    meta["resolved_overlap_sec"] = static_cast<double>(overlap) / sr_;
    path.transition_metadata[key] = meta;

    res.teaserFade           = false;
    res.overlapSamples       = overlap;
    res.outgoingCutSample    = outgoingCut;
    res.incomingStartSample  = incomingStart;
    res.beatEndSample        = beatEnd;
    return res;
}

// ---------------------------------------------------------------------------
// buildEditPlan — port of build_edit_plan (renderer.py:229-339)
// ---------------------------------------------------------------------------
EditPlan Renderer::buildEditPlan(reamix::remix::RemixPath& path)
{
    if (path.beat_indices.empty()) {
        throw std::invalid_argument("Renderer: empty remix path");
    }

    const int maxBeat = static_cast<int>(beatSamples_.size()) - 1;
    std::vector<int> beatIndices(path.beat_indices.size());
    for (std::size_t i = 0; i < path.beat_indices.size(); ++i) {
        beatIndices[i] = std::clamp(path.beat_indices[i], 0, maxBeat);
    }

    // Run grouping (Python L236-260).
    std::vector<Run> runs;
    int runStart = beatIndices[0];
    int runEnd   = runStart;
    for (std::size_t i = 1; i < beatIndices.size(); ++i) {
        const int prev = beatIndices[i - 1];
        const int curr = beatIndices[i];
        if (curr == prev + 1) {
            runEnd = curr;
            continue;
        }
        Run r;
        r.startBeat = runStart;
        r.endBeat   = runEnd;
        r.sourceStartSample = beatSamples_[runStart];
        r.sourceEndSample   = beatEndSample(runEnd);
        runs.push_back(r);
        runStart = curr;
        runEnd   = curr;
    }
    Run last;
    last.startBeat = runStart;
    last.endBeat   = runEnd;
    last.sourceStartSample = beatSamples_[runStart];
    last.sourceEndSample   = beatEndSample(runEnd);
    runs.push_back(last);

    // Block-assembly / region-remix detection (Python L264-272).
    bool isBlockAssembly = false;
    for (const auto& kv : path.transition_metadata) {
        if (metaGet(kv.second, "block_transition", 0.0) > 0.0) {
            isBlockAssembly = true;
            break;
        }
    }
    const int firstBeat = beatIndices.front();
    const int lastBeat  = beatIndices.back();
    const bool isRegion = firstBeat > 2
                       || lastBeat < static_cast<int>(nBeats_) - 3;
    const bool skipExtension = isBlockAssembly || isRegion;
    if (!runs.empty() && !skipExtension) {
        runs.front().sourceStartSample = 0;
        runs.back().sourceEndSample    = static_cast<std::int64_t>(nSamples_);
    }

    // Per-transition fade/overlap fields (Python L279-309).
    for (std::size_t idx = 0; idx + 1 < runs.size(); ++idx) {
        const int prevB = runs[idx].endBeat;
        const int currB = runs[idx + 1].startBeat;
        ResolvedTransition resolved = resolveTransitionEdit(path, prevB, currB);

        if (resolved.teaserFade) {
            const double fadeOutSec = static_cast<double>(resolved.fadeOutSamples) / sr_;
            const double fadeInSec  = static_cast<double>(resolved.fadeInSamples)  / sr_;
            const double gapSec     = static_cast<double>(resolved.gapSamples)     / sr_;
            runs[idx].fadeOutSec     = fadeOutSec;
            runs[idx].gapAfterSec    = gapSec;
            runs[idx + 1].fadeInSec  = fadeInSec;
            continue;
        }

        const std::int64_t overlapSamples = resolved.overlapSamples;
        if (overlapSamples <= 0) continue;

        std::int64_t outgoingCut    = resolved.outgoingCutSample;
        std::int64_t incomingStart  = resolved.incomingStartSample;

        outgoingCut = std::max(runs[idx].sourceStartSample + 1,
                               std::min(outgoingCut, static_cast<std::int64_t>(nSamples_)));
        incomingStart = std::max<std::int64_t>(0,
            std::min(incomingStart, runs[idx + 1].sourceEndSample - 1));

        const double overlapSec = static_cast<double>(overlapSamples) / sr_;
        runs[idx].sourceEndSample      = outgoingCut;
        runs[idx].fadeOutSec           = overlapSec;
        runs[idx].overlapAfterSec      = overlapSec;
        runs[idx + 1].sourceStartSample = incomingStart;
        runs[idx + 1].fadeInSec        = overlapSec;
        runs[idx + 1].overlapBeforeSec = overlapSec;
    }

    // Emit clips (Python L311-339).
    EditPlan plan;
    plan.clips.reserve(runs.size());
    double timelinePos = 0.0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const Run& run = runs[i];
        std::int64_t srcStart = run.sourceStartSample;
        std::int64_t srcEnd   = run.sourceEndSample;
        srcStart = std::max<std::int64_t>(0,
            std::min(srcStart, static_cast<std::int64_t>(nSamples_) - 1));
        srcEnd   = std::max(srcStart + 1,
            std::min(srcEnd, static_cast<std::int64_t>(nSamples_)));
        const double durSec = static_cast<double>(srcEnd - srcStart) / sr_;

        EditClip clip;
        clip.clipIndex        = static_cast<int>(i + 1);
        clip.sourcePath       = audioPath_;
        clip.startBeat        = run.startBeat;
        clip.endBeat          = run.endBeat;
        clip.sourceStartSec   = static_cast<double>(srcStart) / sr_;
        clip.sourceEndSec     = static_cast<double>(srcEnd)   / sr_;
        clip.timelineStartSec = timelinePos;
        clip.timelineEndSec   = timelinePos + durSec;
        clip.durationSec      = durSec;
        clip.fadeInSec        = run.fadeInSec;
        clip.fadeOutSec       = run.fadeOutSec;
        clip.overlapBeforeSec = run.overlapBeforeSec;
        clip.overlapAfterSec  = run.overlapAfterSec;
        clip.gapAfterSec      = run.gapAfterSec;
        plan.clips.push_back(clip);
        timelinePos = clip.timelineEndSec - clip.overlapAfterSec + clip.gapAfterSec;
    }
    plan.duration     = plan.clips.empty() ? 0.0 : plan.clips.back().timelineEndSec;
    plan.nTransitions = static_cast<int>(path.transitions.size());
    return plan;
}

// ---------------------------------------------------------------------------
// renderEditPlan — port of _render_edit_plan (renderer.py:406-460)
// ---------------------------------------------------------------------------
void Renderer::renderEditPlan(const EditPlan&         plan,
                              std::vector<float>&     audioOut,
                              std::size_t&            nChannelsOut,
                              std::size_t&            nSamplesOut,
                              std::vector<double>&    transitionTimesOut) const
{
    nChannelsOut = nChannels_;
    transitionTimesOut.clear();

    if (plan.clips.empty()) {
        // Python L408-409: zeros((nCh, 1)).
        audioOut.assign(nChannels_, 0.0f);
        nSamplesOut = 1;
        return;
    }

    // Pre-render each clip as f32 channel-major slice (Python L412-417).
    struct ClipBuf
    {
        std::size_t nSamp = 0;     // samples per channel
        std::vector<float> data;   // channel-major (nCh × nSamp)
    };
    std::vector<ClipBuf> rendered(plan.clips.size());
    const std::int64_t cap = static_cast<std::int64_t>(nSamples_);
    for (std::size_t i = 0; i < plan.clips.size(); ++i) {
        const EditClip& clip = plan.clips[i];
        std::int64_t srcStart = static_cast<std::int64_t>(
            std::nearbyint(clip.sourceStartSec * sr_));
        std::int64_t srcEnd   = static_cast<std::int64_t>(
            std::nearbyint(clip.sourceEndSec   * sr_));
        srcStart = std::max<std::int64_t>(0, std::min(srcStart, cap - 1));
        srcEnd   = std::max(srcStart + 1, std::min(srcEnd, cap));
        const std::size_t nSamp = static_cast<std::size_t>(srcEnd - srcStart);
        rendered[i].nSamp = nSamp;
        rendered[i].data.resize(nChannels_ * nSamp);
        for (std::size_t c = 0; c < nChannels_; ++c) {
            std::memcpy(rendered[i].data.data() + c * nSamp,
                        audio_.data()           + c * nSamples_
                                                + static_cast<std::size_t>(srcStart),
                        nSamp * sizeof(float));
        }
    }

    // result = rendered_clips[0] (Python L418).
    std::size_t resultLen = rendered[0].nSamp;
    std::vector<float> result = std::move(rendered[0].data);

    std::vector<double> ramp; // scratch

    for (std::size_t idx = 1; idx < rendered.size(); ++idx) {
        const EditClip& clip       = plan.clips[idx];
        const EditClip& prevClip   = plan.clips[idx - 1];
        const double overlapSec    = clip.overlapBeforeSec;
        const double gapSec        = prevClip.gapAfterSec;

        if (gapSec > 0.0) {
            // Teaser-gap branch (Python L423-438).
            std::size_t fadeOut = static_cast<std::size_t>(std::min<std::int64_t>(
                static_cast<std::int64_t>(resultLen),
                static_cast<std::int64_t>(std::nearbyint(prevClip.fadeOutSec * sr_))));
            if (fadeOut > 0) {
                linspace(0.0, kPiHalf, fadeOut, ramp);
                std::vector<double> gain(fadeOut);
                for (std::size_t k = 0; k < fadeOut; ++k) gain[k] = std::cos(ramp[k]);
                // Apply per-channel to result tail.
                for (std::size_t c = 0; c < nChannels_; ++c) {
                    mulInPlaceF32F64(result.data() + c * resultLen
                                     + (resultLen - fadeOut), gain.data(), fadeOut);
                }
            }
            const std::size_t gapSamples = static_cast<std::size_t>(
                std::nearbyint(gapSec * sr_));

            // Apply incoming fade-in (Python L433-437).
            ClipBuf& inc = rendered[idx];
            std::size_t fadeIn = static_cast<std::size_t>(std::min<std::int64_t>(
                static_cast<std::int64_t>(inc.nSamp),
                static_cast<std::int64_t>(std::nearbyint(clip.fadeInSec * sr_))));
            if (fadeIn > 0) {
                linspace(0.0, kPiHalf, fadeIn, ramp);
                std::vector<double> gain(fadeIn);
                for (std::size_t k = 0; k < fadeIn; ++k) gain[k] = std::sin(ramp[k]);
                for (std::size_t c = 0; c < nChannels_; ++c) {
                    mulInPlaceF32F64(inc.data.data() + c * inc.nSamp,
                                     gain.data(), fadeIn);
                }
            }

            // Concatenate result + zeros(gapSamples) + incoming.
            const std::size_t newLen = resultLen + gapSamples + inc.nSamp;
            std::vector<float> next(nChannels_ * newLen, 0.0f);
            for (std::size_t c = 0; c < nChannels_; ++c) {
                std::memcpy(next.data() + c * newLen,
                            result.data() + c * resultLen,
                            resultLen * sizeof(float));
                // gap region is zero by construction
                std::memcpy(next.data() + c * newLen + resultLen + gapSamples,
                            inc.data.data() + c * inc.nSamp,
                            inc.nSamp * sizeof(float));
            }
            result   = std::move(next);
            resultLen = newLen;
        } else if (overlapSec > 0.0) {
            // Adaptive crossfade branch (Python L439-454).
            std::int64_t overlapSamples = static_cast<std::int64_t>(
                std::nearbyint(overlapSec * sr_));
            overlapSamples = std::min<std::int64_t>(overlapSamples,
                std::min<std::int64_t>(static_cast<std::int64_t>(resultLen),
                                       static_cast<std::int64_t>(rendered[idx].nSamp)));
            ClipBuf& inc = rendered[idx];
            if (overlapSamples > 0) {
                const std::size_t ov = static_cast<std::size_t>(overlapSamples);
                // outgoing = result[..., -ov:]   (f32 view, cast to f64 for crossfade)
                // incoming = inc.data[..., :ov]  (f32 view, cast to f64)
                std::vector<double> outgoingF64(nChannels_ * ov);
                std::vector<double> incomingF64(nChannels_ * ov);
                for (std::size_t c = 0; c < nChannels_; ++c) {
                    const float* src = result.data() + c * resultLen + (resultLen - ov);
                    for (std::size_t k = 0; k < ov; ++k) {
                        outgoingF64[c * ov + k] = static_cast<double>(src[k]);
                    }
                    const float* src2 = inc.data.data() + c * inc.nSamp;
                    for (std::size_t k = 0; k < ov; ++k) {
                        incomingF64[c * ov + k] = static_cast<double>(src2[k]);
                    }
                }
                const std::size_t blendedLen = Crossfade::computeResultLen(
                    ov, ov, sr_, nullptr, 0);
                std::vector<double> blendedF64(nChannels_ * blendedLen);
                Crossfade::adaptiveCrossfade(
                    outgoingF64.data(), nChannels_, ov,
                    incomingF64.data(), ov,
                    sr_, nullptr, 0,
                    /*energyCompensate*/ true,
                    /*phaseAlign*/       true,
                    /*maxPhaseShiftMs*/  3.0,
                    blendedF64.data(), blendedLen);

                // Cast back to f32 (Python `.astype(result.dtype)`).
                std::vector<float> blendedF32(nChannels_ * blendedLen);
                for (std::size_t k = 0; k < nChannels_ * blendedLen; ++k) {
                    blendedF32[k] = static_cast<float>(blendedF64[k]);
                }

                // pre_overlap = result[..., :-ov] ; rest = inc[..., ov:]
                const std::size_t preLen  = resultLen - ov;
                const std::size_t restLen = inc.nSamp - ov;
                const std::size_t newLen  = preLen + blendedLen + restLen;
                std::vector<float> next(nChannels_ * newLen);
                for (std::size_t c = 0; c < nChannels_; ++c) {
                    std::memcpy(next.data() + c * newLen,
                                result.data() + c * resultLen,
                                preLen * sizeof(float));
                    std::memcpy(next.data() + c * newLen + preLen,
                                blendedF32.data() + c * blendedLen,
                                blendedLen * sizeof(float));
                    std::memcpy(next.data() + c * newLen + preLen + blendedLen,
                                inc.data.data() + c * inc.nSamp + ov,
                                restLen * sizeof(float));
                }
                result   = std::move(next);
                resultLen = newLen;
            } else {
                // overlap clamped to 0 — fall through to plain concat (Python L453).
                const std::size_t newLen = resultLen + inc.nSamp;
                std::vector<float> next(nChannels_ * newLen);
                for (std::size_t c = 0; c < nChannels_; ++c) {
                    std::memcpy(next.data() + c * newLen,
                                result.data() + c * resultLen,
                                resultLen * sizeof(float));
                    std::memcpy(next.data() + c * newLen + resultLen,
                                inc.data.data() + c * inc.nSamp,
                                inc.nSamp * sizeof(float));
                }
                result   = std::move(next);
                resultLen = newLen;
            }
        } else {
            // Plain concat (Python L455-456).
            ClipBuf& inc = rendered[idx];
            const std::size_t newLen = resultLen + inc.nSamp;
            std::vector<float> next(nChannels_ * newLen);
            for (std::size_t c = 0; c < nChannels_; ++c) {
                std::memcpy(next.data() + c * newLen,
                            result.data() + c * resultLen,
                            resultLen * sizeof(float));
                std::memcpy(next.data() + c * newLen + resultLen,
                            inc.data.data() + c * inc.nSamp,
                            inc.nSamp * sizeof(float));
            }
            result   = std::move(next);
            resultLen = newLen;
        }

        // transition_times.append((result.shape[-1] - inc.nSamp) / sr) — Python L457-458.
        const double tt = static_cast<double>(resultLen - rendered[idx].nSamp) / sr_;
        transitionTimesOut.push_back(tt);
    }

    audioOut    = std::move(result);
    nSamplesOut = resultLen;
}

// ---------------------------------------------------------------------------
// render — port of render(path) (renderer.py:462-475)
// ---------------------------------------------------------------------------
RenderResult Renderer::render(reamix::remix::RemixPath& path)
{
    if (path.beat_indices.empty()) {
        throw std::invalid_argument("Renderer: empty remix path");
    }
    EditPlan plan = buildEditPlan(path);

    RenderResult res;
    res.editPlan   = plan;
    res.sampleRate = sr_;
    renderEditPlan(plan, res.audio, res.nChannels, res.nSamples, res.transitionTimes);
    res.duration     = static_cast<double>(res.nSamples) / sr_;
    res.nTransitions = static_cast<int>(res.transitionTimes.size());
    return res;
}

} // namespace reamix::render
