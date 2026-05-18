#pragma once

#include <cstddef>
#include <cstdint>

namespace reamix::render {

// Splice — phase-5 SpliceMixin port
// (references/python-source/remix/splice.py, 514 LOC). Per ADR-031 module
// ordering: Butterworth → PhaseAlign → Crossfade (adaptive + simple) →
// **Splice** → Renderer.
//
// Session-34 scope — 3 core primitives (splice.py L85-196):
//   - _window_onset_index        → windowOnsetIndex
//   - _get_hanning               → (inlined; not a public entry point)
//   - _stereo_window_similarity  → stereoWindowSimilarity
//   - _score_splice_pair         → scoreSplicePair
//
// Session-35 scope — findOnsetSample re-enable + 4 composite methods:
//   - _find_onset_sample                     → findOnsetSample
//   - _score_anchor_aligned_pair             → scoreAnchorAlignedPair
//   - _search_anchor_transition_geometry     → searchAnchorTransitionGeometry
//   - _transition_overlap_samples            → transitionOverlapSamples
//   - _refine_transition_splice              → refineTransitionSplice
//
// Per ADR-033: findOnsetSample primitive parity is DEFERRED (±1-4 sample
// argmax drift from numpy np.convolve's BLAS-dispatched accumulator at K=32
// smoothing kernel — no tested C++ accumulator matches bit-exact). Session
// 35 tests findOnsetSample at composition level via
// searchAnchorTransitionGeometry — if the winner-anchor-beat-pair matches
// Python despite primitive ULP drift, ADR-033 closes positively.
//
// Per SB-5 (ADR-031): stereoWindowSimilarity is a DISTINCT metric from
// `PhaseAlign::alignMono` — Hanning-taper + zero-mean + 0.8/0.2
// static/derivative NCC weighting vs plain NCC. Do NOT alias.
//
// Channel-major row-major f64 I/O (same convention as Crossfade).
// For a stereo window of nCh=2, n=1024: `a_buf[c * n + i]` at channel c,
// sample i. Mono inputs pass nCh=1.

// Config — DEFAULT_CONFIG.remix.* fields consumed by Splice methods.
// Defaults below mirror references/python-source/config.py L160-180, L194.
struct SpliceConfig
{
    // Consumed by findOnsetSample (ms → sample conversion requires sr):
    double onsetSearchLookbackMs           =  70.0;  // config.py:172
    double onsetSearchLookaheadMs          =  18.0;  // config.py:173

    // Consumed by scoreSplicePair / scoreAnchorAlignedPair / refineTransitionSplice:
    double transientCenterPenaltyWeight    =   0.14; // config.py:179
    double transientAlignmentPenaltyWeight =   0.10; // config.py:180

    // Consumed by scoreAnchorAlignedPair:
    double anchorLocalWindowMs             = 120.0;  // config.py:176

    // Consumed by searchAnchorTransitionGeometry:
    int    anchorSearchBeats               =   1;    // config.py:174
    int    anchorSearchMaxExtensionBeats   =   2;    // config.py:175
    double anchorMinContextMs              = 120.0;  // config.py:177

    // Consumed by transitionOverlapSamples:
    double vocalActivityThreshold          =   0.28; // config.py:194
    double vocalCrossfadeMs                = 300.0;  // config.py:160
    double vocalSameLabelCrossfadeMs       = 180.0;  // config.py:161

    // Consumed by refineTransitionSplice:
    double stereoRefineMaxShiftMs          =  12.0;  // config.py:170
    int    stereoRefineCoarseStepSamples   =  64;    // config.py:171
};

// Transition metadata — Python's transition_meta `Dict[str, float]`.
// Only fields actually consumed by session-35 methods are exposed; defaults
// match Python `.get(key, DEFAULT)` fallbacks.
struct TransitionMeta
{
    double vocalPresenceLevel   = 0.0;   // search / overlap
    double preferredOverlapSec  = 0.0;   // overlap
    double labelMatch           = 1.0;   // overlap
    double vocalEntrySupport    = 1.0;   // overlap
    double vocalExitSupport     = 1.0;   // overlap
    double alignmentOffsetSec   = 0.0;   // refine
};

// State bundle for the composite methods (audio, beat grid, mono energy,
// default crossfade). Caller populates; Splice methods read only.
struct SpliceContext
{
    const double*        audio             = nullptr;   // channel-major 2D
    std::size_t          nChAudio          = 0;
    std::size_t          nAudio            = 0;         // samples per channel
    int                  sr                = 0;

    const double*        monoEnergy        = nullptr;   // 1D f64 (findOnsetSample)
    std::size_t          nMonoEnergy       = 0;

    const std::int64_t*  beatSamples       = nullptr;   // 1D int64
    std::size_t          nBeats            = 0;

    int                  crossfadeSamples  = 0;         // default overlap
};

// Output of scoreSplicePair (session-34).
struct SplicePairScore
{
    double score;       // composite; -1.0 on reject (similarity ≤ -0.99)
    double similarity;  // raw stereoWindowSimilarity
};

// Output of scoreAnchorAlignedPair (session-35).
struct AnchorScore
{
    double quality;            // clip((1-lw)*sim01 + lw*local01 - penalties, 0, 1)
    double similarity01;       // clip((similarity + 1) * 0.5, 0, 1)
    double localSimilarity01;  // clip((local_similarity + 1) * 0.5, 0, 1)
};

// Output of searchAnchorTransitionGeometry (session-35).
// `selected=false` ↔ Python returned `{}` (no candidate improved best_score).
struct AnchorSearchResult
{
    bool                 selected               = false;
    double               score                  = 0.0;
    double               similarity01           = 0.0;
    double               localSimilarity01      = 0.0;
    int                  outAnchorBeat          = 0;
    int                  inAnchorBeat           = 0;
    int                  incomingStartBeat      = 0;
    int                  outgoingBoundaryBeat   = 0;
    double               anchorOutSec           = 0.0;
    double               anchorInSec            = 0.0;
    std::int64_t         outgoingStartSample    = 0;
    std::int64_t         outgoingCutSample      = 0;
    std::int64_t         incomingCutSample      = 0;   // == incoming_start_sample
    std::int64_t         incomingEndSample      = 0;
    std::int64_t         anchorOverlapSamples   = 0;
};

// Output of refineTransitionSplice (session-35).
// `found=false` ↔ Python returned `{}` (any early-return gate fired).
struct RefineResult
{
    bool                 found               = false;
    std::int64_t         outgoingCutSample   = 0;
    std::int64_t         incomingCutSample   = 0;
    double               outgoingShiftSec    = 0.0;
    double               incomingShiftSec    = 0.0;
    double               stereoSimilarity01  = 0.0;   // clip((sim + 1) * 0.5, 0, 1)
    double               spliceScore         = 0.0;
};

class Splice
{
public:
    // --- windowOnsetIndex -------------------------------------------------
    // Port of _window_onset_index (splice.py:85-98). Locate dominant onset
    // inside a splice window (relative to window start).
    // Returns: sample index in [0, n-1], or `max(0, n/2)` on n < 16.
    static int windowOnsetIndex(const double* window,
                                std::size_t   nCh,
                                std::size_t   n);

    // --- stereoWindowSimilarity -------------------------------------------
    // Port of _stereo_window_similarity (splice.py:108-149).
    // Hanning-taper + zero-mean + 0.8/0.2 static/derivative NCC.
    // Returns clipped similarity ∈ [-1.0, +1.0], or -1.0 on short / degenerate.
    static double stereoWindowSimilarity(const double* windowA,
                                         const double* windowB,
                                         std::size_t   nCh,
                                         std::size_t   n);

    // --- scoreSplicePair --------------------------------------------------
    // Port of _score_splice_pair (splice.py:151-196).
    // Composite: similarity - 0.10*energy - 0.15*position - wC*center - wA*alignment.
    // Short-circuits to { -1.0, similarity } on similarity ≤ -0.99.
    static SplicePairScore scoreSplicePair(const double*        outgoing,
                                           const double*        incoming,
                                           std::size_t          nCh,
                                           std::size_t          n,
                                           int                  outgoingShift,
                                           int                  incomingShift,
                                           int                  maxShiftSamples,
                                           const SpliceConfig&  cfg);

    // --- findOnsetSample --------------------------------------------------
    // Port of _find_onset_sample (splice.py:26-83).
    //
    // Asymmetric onset search on a mono_energy array around a beat sample.
    // (1) chunk = mono[max(0, beat-lookback) : min(N, beat+lookahead)]
    // (2) if len(chunk) < 128 → return beat_sample (short-chunk fallback)
    // (3) kernel_size = min(64, len(chunk) // 2); if < 4 → return beat_sample
    // (4) envelope = boxConvolveSame(chunk, kernel_size)
    // (5) diff = np.diff(envelope)  [length N-1]
    //     if len(diff) < 16 → return beat_sample
    // (6) smooth_size = min(32, len(diff) // 2); if ≥ 2:
    //         diff = boxConvolveSame(diff, smooth_size)    ← THE second convolve
    //                                                        that drifts per ADR-033
    // (7) return start + int(np.argmax(diff))
    //
    // `lookbackMs` / `lookaheadMs` < 0 means "use default" (cfg fields).
    // This matches the Python `None` sentinel used for the cache-hit branch.
    //
    // NOTE (ADR-033): no instance cache is maintained in the C++ port — cache
    // is a performance-only optimization; output is deterministic given inputs.
    // Parity target at composition level (searchAnchorTransitionGeometry).
    static int findOnsetSample(const double*       monoEnergy,
                               std::size_t         nMonoEnergy,
                               int                 sr,
                               std::int64_t        beatSample,
                               double              lookbackMs,
                               double              lookaheadMs,
                               const SpliceConfig& cfg);

    // --- scoreAnchorAlignedPair -------------------------------------------
    // Port of _score_anchor_aligned_pair (splice.py:198-245).
    //
    // Composite score for an anchor-aligned splice pair:
    //   similarity    = stereoWindowSimilarity(outgoing, incoming)
    //   local_sim     = stereoWindowSimilarity(win[..., lo:hi], win[..., lo:hi])
    //                    where lo/hi = anchor ± local_radius (clamped to [0, n])
    //   local_weight  = min(0.70, 0.40 + 0.25 * clip(vocal_presence, 0, 1))
    //   quality       = (1 - lw) * clip((sim+1)*0.5, 0, 1)
    //                 + lw       * clip((local_sim+1)*0.5, 0, 1)
    //                 - 0.08 * energy_penalty
    //                 - 0.04 * edge_penalty
    //
    // Returns (0,0,0) on `similarity <= -0.99` reject (Python L207-208).
    //
    // The local_radius is `round(anchorLocalWindowMs * sr / 2000.0)` clamped
    // to ≥ 16 samples (splice.py:209-212). The 2000.0 (not 1000) is a HALF-
    // window convention — the local window spans 2 × local_radius centered
    // at the anchor.
    static AnchorScore scoreAnchorAlignedPair(const double*        outgoing,
                                              const double*        incoming,
                                              std::size_t          nCh,
                                              std::size_t          n,
                                              int                  anchorIndex,
                                              double               vocalPresence,
                                              int                  sr,
                                              const SpliceConfig&  cfg);

    // --- searchAnchorTransitionGeometry -----------------------------------
    // Port of _search_anchor_transition_geometry (splice.py:247-330).
    //
    // 4-level nested loop over (out_anchor_beat × in_anchor_beat ×
    // incoming_start_beat × outgoing_boundary_beat). For each valid
    // quadruple, constructs outgoing/incoming windows aligned so the anchor
    // onset lands at the same offset in both, then scores via
    // scoreAnchorAlignedPair. Keeps argmax-best.
    //
    // Returns AnchorSearchResult with `selected=false` when every candidate
    // fails the inner gates (min_context / overlap / audio-bounds / shape).
    //
    // This is the CRITICAL ADR-033 close test: if findOnsetSample primitive
    // drift does not flip the winner-anchor-beat-pair on real music, ADR-033
    // closes positively. If it does flip, ADR-034 opens with empirical
    // winner-flip evidence.
    static AnchorSearchResult searchAnchorTransitionGeometry(
        const SpliceContext&    ctx,
        int                     prevBeat,
        int                     currBeat,
        const TransitionMeta&   meta,
        const SpliceConfig&     cfg);

    // --- transitionOverlapSamples -----------------------------------------
    // Port of _transition_overlap_samples (splice.py:332-355).
    //
    // Rule-based overlap selection:
    //   base = max(crossfadeSamples, round(preferred_overlap_sec * sr))
    //   if vocal_presence < vocalActivityThreshold:        return base
    //   if label_match < 0.5:                              target = vocalCrossfadeMs
    //   elif min(entry, exit) < 0.85:                      target = vocalSameLabelCrossfadeMs
    //   else:                                              return base
    //   return max(base, round(target * sr / 1000))
    //
    // Pure integer arithmetic; f64 only as intermediate for rounding.
    static int transitionOverlapSamples(int                    crossfadeSamples,
                                        int                    sr,
                                        const TransitionMeta&  meta,
                                        const SpliceConfig&    cfg);

    // --- refineTransitionSplice -------------------------------------------
    // Port of _refine_transition_splice (splice.py:357-514).
    //
    // 2-stage coarse→fine grid search around the nominal cut/start positions:
    //   Stage 1: fast mono-sum NCC on out_positions × in_positions (step =
    //            coarse_step); keep pairs with score > -0.5.
    //   Stage 2: full composite evaluation (scoreSplicePair-style) on top-8
    //            coarse candidates (sorted by -score).
    //   Stage 3: fine-radius exhaustive search around the best (radius =
    //            max(1, coarse_step // 8)).
    //
    // Returns RefineResult with `found=false` on any of:
    //   - overlap ≤ 16 or half_xfade ≤ 0 (Python L368-369 early return)
    //   - max_shift_samples ≤ 0 after clamping (L383-384)
    //   - out_min ≥ out_max or in_min ≥ in_max (L397-398)
    //   - best_score < -0.5 after all stages (L505-506)
    //
    // `overlapSamplesOrNeg = -1` means "use ctx.crossfadeSamples" (Python's
    // `overlap_samples=None`).
    static RefineResult refineTransitionSplice(const SpliceContext&   ctx,
                                               std::int64_t           successorSample,
                                               std::int64_t           targetSample,
                                               std::int64_t           beatEnd,
                                               const TransitionMeta&  meta,
                                               int                    overlapSamplesOrNeg,
                                               const SpliceConfig&    cfg);
};

} // namespace reamix::render
