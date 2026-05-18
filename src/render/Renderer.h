#pragma once

#include "remix/Path.h"
#include "render/Splice.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace reamix::render {

// Renderer — phase-5 final module port
// (references/python-source/remix/renderer.py, 482 LOC). Per ADR-031 module
// ordering: Butterworth → PhaseAlign → Crossfade → Splice → **Renderer**.
//
// Composition strategy:
//   Python `class RemixRenderer(SpliceMixin)` inherits SpliceMixin and reads
//   _audio / _sr / _mono_energy / _beat_samples / _crossfade_samples directly
//   on `self`. C++ uses **composition**: Renderer holds the same state as
//   plain f64/int64/string members; constructs a `SpliceContext` on demand
//   to pass into `reamix::render::Splice::*` static composite methods.
//
// Dtype discipline (matches Python verbatim — critical for parity):
//   * Audio buffer is f32 throughout the render pipeline (matches
//     `librosa.load(..., sr=sample_rate, mono=False)` which returns float32).
//   * mono_energy is f64 (Python: `sqrt(np.mean(self._audio.astype(f64)**2,
//     axis=0))` at __init__).
//   * Fade ramps are computed in f64 (np.linspace + np.cos + np.sin), then
//     applied as in-place `f32 *= f64_ramp`. Numpy in-place multiply
//     internally uses the higher precision and casts back to f32.
//   * adaptive_crossfade is called with explicit `outgoing.astype(f64),
//     incoming.astype(f64)` and the result is cast `.astype(result.dtype)` =
//     f32 before being merged into the output buffer.
//
// Audio buffer layout convention: channel-major row-major (nChannels x
// nSamples). Mono uses nChannels=1.
//
// Parity targets (per phase-5 spec acceptance):
//   * Per-sample max_abs ≤ 5e-4 on rendered audio f32 (mirror phase-4 DP cost
//     class; ADR-027 + ADR-034 discipline — refined via bisection only).
//   * Edit plan int-bit-exact on clip_index / start_beat / end_beat /
//     source_start_sample / source_end_sample.
//   * EditClip f64 sec-fields max_abs ≤ 1e-9 (timeline + duration + fade).
//   * transition_times f64 max_abs ≤ 1e-9.

// Mirror of Python `EditClip` dataclass (renderer.py:33-50). One non-
// destructive clip in the rendered timeline.
struct EditClip
{
    int          clipIndex            = 0;
    std::string  sourcePath;                  // copy of audioPath ctor arg
    int          startBeat            = 0;
    int          endBeat              = 0;
    double       sourceStartSec       = 0.0;
    double       sourceEndSec         = 0.0;
    double       timelineStartSec     = 0.0;
    double       timelineEndSec       = 0.0;
    double       durationSec          = 0.0;
    double       fadeInSec            = 0.0;
    double       fadeOutSec           = 0.0;
    double       overlapBeforeSec     = 0.0;
    double       overlapAfterSec      = 0.0;
    double       gapAfterSec          = 0.0;
};

// Mirror of Python `EditPlan` dataclass (renderer.py:53-59).
struct EditPlan
{
    std::vector<EditClip> clips;
    double                duration       = 0.0;
    int                   nTransitions   = 0;
};

// Mirror of Python `RenderResult` dataclass (renderer.py:21-30).
// Note: Python `audio` is np.ndarray of shape (n,) for mono squeezed and
// (nCh, n) for stereo. C++ keeps channel-major shape always — caller may
// squeeze to 1-D when nChannels == 1.
struct RenderResult
{
    std::vector<float>   audio;          // channel-major (nChannels x nSamples)
    std::size_t          nChannels       = 0;
    std::size_t          nSamples        = 0;
    int                  sampleRate      = 0;
    double               duration        = 0.0;
    int                  nTransitions    = 0;
    std::vector<double>  transitionTimes;
    EditPlan             editPlan;
};

// DEFAULT_CONFIG.remix.* fields consumed by Renderer (in addition to the
// SpliceConfig fields consumed by composite Splice methods).
struct RendererConfig
{
    double crossfadeMs                  = 75.0;   // config.py:158 — fallback if crossfade_beats is None
    double crossfadeBeats               = 0.15;   // config.py:159 — beat-adaptive crossfade
    double ultraShortTeaserFadeOutMs    = 320.0;  // config.py:147 — teaser-fade branch override
    double ultraShortTeaserFadeInMs     = 180.0;  // config.py:148 — teaser-fade branch override
    double ultraShortTeaserGapMs        = 60.0;   // config.py:149 — teaser-fade branch override
    double anchorImprovementThreshold   = 0.02;   // config.py:178 — anchor accept gate
    SpliceConfig splice;                          // delegated to Splice composite methods
};

class Renderer
{
public:
    // Constructor (port of __init__, renderer.py:70-99).
    //
    //   audioPath        forwarded into EditClip.sourcePath; never opened by C++.
    //   audioF32         channel-major row-major float32 audio buffer
    //                    (matches librosa.load output). Caller owns memory;
    //                    Renderer holds an owning copy.
    //   nChannels        1 (mono) or 2 (stereo). Mono inputs preserved as
    //                    (1, nSamples) per Python L88-89 reshape.
    //   nSamples         samples per channel.
    //   sampleRate       Hz. Matches librosa `sr=` kwarg.
    //   beatTimes        f64 beat times in seconds. Length nBeats.
    //   nBeats           number of beats. Required ≥ 1.
    //   crossfadeMsOrNeg if ≥ 0: explicit crossfade override (Python
    //                    crossfade_ms kwarg). If < 0: fall back to
    //                    cfg.crossfadeBeats * avg_beat_dur (Python L80-86).
    //   cfg              renderer config; SpliceConfig defaults from config.py.
    Renderer(const std::string&  audioPath,
             const float*        audioF32,
             std::size_t         nChannels,
             std::size_t         nSamples,
             int                 sampleRate,
             const double*       beatTimes,
             std::size_t         nBeats,
             double              crossfadeMsOrNeg,
             const RendererConfig& cfg);

    // Port of build_edit_plan(path) (renderer.py:229-339).
    //
    // Mutates `path.transition_metadata` per Python L142, L202, L219-220
    // (caches anchor/teaser overlap fields for replay; matches Python).
    EditPlan buildEditPlan(reamix::remix::RemixPath& path);

    // Port of _render_edit_plan(edit_plan) (renderer.py:406-460).
    // Returns rendered f32 audio buffer + transition times (timeline-relative).
    void renderEditPlan(const EditPlan&         plan,
                        std::vector<float>&     audioOut,
                        std::size_t&            nChannelsOut,
                        std::size_t&            nSamplesOut,
                        std::vector<double>&    transitionTimesOut) const;

    // Port of render(path) (renderer.py:462-475).
    RenderResult render(reamix::remix::RemixPath& path);

    // --- Accessors (for testing / dump tool) -----------------------------
    int                 sampleRate()       const { return sr_; }
    std::size_t         nChannels()        const { return nChannels_; }
    std::size_t         nSamples()         const { return nSamples_; }
    int                 crossfadeSamples() const { return crossfadeSamples_; }
    double              avgBeatDuration()  const { return avgBeatDuration_; }
    const std::vector<std::int64_t>& beatSamples() const { return beatSamples_; }
    const std::vector<double>&       monoEnergy()  const { return monoEnergy_; }
    const std::vector<float>&        audio()       const { return audio_; }

    // Port of _beat_end_sample(beat_idx) (renderer.py:101-109). Public for
    // dump tool / parity testing.
    std::int64_t beatEndSample(int beatIdx) const;

private:
    // Port of _resolve_transition_edit (renderer.py:119-227). Mutates
    // path.transition_metadata for cache replay.
    struct ResolvedTransition
    {
        bool          teaserFade        = false;
        std::int64_t  fadeOutSamples    = 0;     // teaser branch
        std::int64_t  fadeInSamples     = 0;     // teaser branch
        std::int64_t  gapSamples        = 0;     // teaser branch
        std::int64_t  overlapSamples    = 0;     // crossfade branch
        std::int64_t  outgoingCutSample = 0;     // crossfade branch
        std::int64_t  incomingStartSample = 0;   // crossfade branch
        std::int64_t  beatEndSample     = 0;     // crossfade branch
    };
    ResolvedTransition resolveTransitionEdit(reamix::remix::RemixPath& path,
                                             int prevBeat, int currBeat);

    // Build SpliceContext referencing this Renderer's state. Used by
    // resolveTransitionEdit to call Splice composite methods.
    SpliceContext makeSpliceContext() const;

    // Run-grouping intermediate (corresponds to Python's `runs` list of dicts
    // at renderer.py:236-260). Floats in Python; we use exact int64 sample
    // positions internally and only convert to seconds at clip emission.
    struct Run
    {
        int           startBeat;
        int           endBeat;
        std::int64_t  sourceStartSample;
        std::int64_t  sourceEndSample;
        double        fadeInSec        = 0.0;
        double        fadeOutSec       = 0.0;
        double        overlapBeforeSec = 0.0;
        double        overlapAfterSec  = 0.0;
        double        gapAfterSec      = 0.0;
    };

    // State (matches Python `self._*` attributes at renderer.py:79-99).
    std::string                 audioPath_;
    std::vector<float>          audio_;             // channel-major (nCh x nSamples)
    std::size_t                 nChannels_      = 0;
    std::size_t                 nSamples_       = 0;
    int                         sr_             = 0;
    std::vector<double>         beatTimes_;
    std::size_t                 nBeats_         = 0;
    double                      avgBeatDuration_ = 0.5;
    int                         crossfadeSamples_ = 0;
    std::vector<double>         monoEnergy_;        // f64, length nSamples
    std::vector<std::int64_t>   beatSamples_;       // int64, length nBeats
    // f64 mirror of `audio_` for Splice composite methods (which take
    // `const double*` audio). Memory cost ~2x audio buffer; built once at
    // construction and held read-only thereafter. For an 8-min stereo @
    // 44100 Hz clip this is ~21 MB f32 + ~42 MB f64.
    std::vector<double>         audioF64Mirror_;
    RendererConfig              cfg_;
};

} // namespace reamix::render
