#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reamix::analysis {

// Per-beat window extraction helpers — step 9 of the phase-2 port order.
// Three distinct operations, all keyed off a beat list but operating on
// different input domains:
//
//   1. extractEdgeFeatures:   feature-matrix domain. Mean of the first
//      and last `nEdge` frames of each beat's slice of a stacked feature
//      matrix (59-dim in the real pipeline), followed by per-beat L2
//      normalization. Used for splice-point feature similarity.
//
//   2. extractEdgeRms:        sample domain. RMS of the first and last
//      ~50 ms of each beat's audio window, then both arrays are jointly
//      normalized to [0, 1] against a SHARED max. Used as a
//      transient-position cue at beat edges.
//
//   3. extractWaveformSnippets: sample domain. DC-removed, Hann-windowed,
//      RMS-normalized mono snippet centered on each beat, with caller-
//      chosen pre/post ms. Called twice at orchestrator level — once
//      with (35, 120) for "boundary" waveforms (splice cross-correlation)
//      and once with (280, 320) for "transition" waveforms (overlap scoring).
//
// PARITY (Python reference, librosa 0.11.0):
//   - references/python-source/analysis/feature_extractor.py
//       * _extract_edge_features      (L317–351)
//       * _extract_edge_rms           (L353–392)
//       * _extract_waveform_snippets  (L424–463)
//   - references/python-source/config.py
//       * boundary_waveform_pre_ms / post_ms   (L62–63)
//       * transition_window_pre_ms / post_ms   (L64–65)
//
// Precision notes (per spec.md "Known traps per remaining step — Step 9"):
//
//   - Two rounding conventions coexist within the Python helpers, by
//     design, and MUST be reproduced separately:
//       * `librosa.time_to_frames` uses round-half-to-even (np.round).
//         — not used here; relevant upstream at the orchestrator.
//       * `int(ms * sr / 1000.0)` and `(beat_times * sr).astype(int64)`
//         use truncation toward zero. — this is what
//         extractEdgeRms / extractWaveformSnippets use.
//     We use `static_cast<int>(...)` / `static_cast<std::int64_t>(...)`
//     for the truncation branch.
//
//   - Three epsilon guards coexist, each with distinct physical meaning:
//       * 1e-10 — additive-under-sqrt for edge RMS, also the floor used
//         by the joint [0,1] normalization.
//       * 1e-8  — additive-under-sqrt for snippet RMS.
//       * zero-guard `if (norm == 0) norm = 1` for edge-feature L2
//         (NOT tiny/1e-10; preserves zero rows exactly).
//
//   - numpy computes scalar arithmetic at float64 when a float32 result
//     meets a Python float literal (`float32(meanSq) + 1e-8` → f64).
//     `np.sqrt` then returns f64. The subsequent `snippet /= rms` with
//     an out=float32 array casts rms back to f32 at divide-time.
//     Port mirrors that: sqrt in f64, divide in f32.
//
//   - `np.hanning(N)` is f64 then `.astype(np.float32)` downcasts.
//     We synthesize the window directly as f32 via
//     `0.5 * (1 - cos(2π·n / (N-1)))` evaluated in f64 then rounded.
//     `max(8, pre+post)` upstream ensures N ≥ 8, so the `N=1 → all zeros`
//     librosa fallback is unreachable; we do not guard it.
//
//   - `(beat_times * sr).astype(int64)` — note the float64 multiply before
//     cast. We preserve the f64 intermediate exactly to match numpy's
//     rounding-to-zero on the f64 product (not on a float-32-ed
//     intermediate).
class BeatWindows
{
public:
    struct EdgeFeatures
    {
        std::vector<double> start; // [nBeats × F], row-major, L2-normalized.
        std::vector<double> end;   // same layout.
    };

    struct EdgeRms
    {
        std::vector<double> start; // [nBeats], normalized [0, 1] vs shared max.
        std::vector<double> end;   // same.
    };

    // Mean of the first `nEdge` and last `nEdge` frames of each beat's
    // slice of `features` [F × T] (row-major, float32), then per-beat L2
    // normalization in float64. Output rows correspond to beats, columns
    // to features. `n_edge = min(nEdgeFrames, max(1, beat_len/2))` — short
    // beats use half their length rounded down. Zero-length beats leave
    // the corresponding output rows as zero (pre-L2) which then stay zero
    // post-L2 by the `if (norm == 0) norm = 1` convention.
    //
    // PARITY: feature_extractor.py::_extract_edge_features (L317).
    static EdgeFeatures extractEdgeFeatures(const float* features,
                                            std::size_t F,
                                            std::size_t T,
                                            const std::vector<int>& beatFrames,
                                            std::size_t nBeats,
                                            int nEdgeFrames = 4);

    // RMS of the first and last `edgeMs` milliseconds of each beat's audio
    // window, with 1e-10 additive-under-sqrt guard. Output arrays are
    // then jointly normalized to [0, 1] against the larger of
    // (max(start), max(end), 1e-10). Beat boundaries in samples are
    // `(beatTimes * sr).astype(int64)` (truncation, not round). Short
    // beats shrink n_edge to `max(1, beat_len/2)`.
    //
    // PARITY: feature_extractor.py::_extract_edge_rms (L353).
    static EdgeRms extractEdgeRms(const float* y,
                                  std::size_t nSamples,
                                  int sr,
                                  const std::vector<double>& beatTimes,
                                  std::size_t nBeats,
                                  double edgeMs = 50.0);

    // Extract one DC-removed, Hann-windowed, RMS-normalized mono snippet
    // per beat, centered at `beat_sample = int(beat_time * sr)` with
    // `preSamples` before and `postSamples` after. `total = max(8, pre +
    // post)`. Snippets near the audio edges are zero-padded on the side
    // that falls outside `[0, nSamples)`. Silent beats are left as the
    // pre-window-multiplied output (post-DC-remove, zero-length audio
    // contribution) — but the 1e-8 floor under sqrt prevents NaN.
    //
    // Pipeline per beat (must be in this order — DC-remove → window → RMS):
    //   1. Zero-fill snippet [total]; copy y[clip_start..clip_end] into it
    //      at offset `clip_start - start` (handles negative-start correction).
    //   2. snippet -= mean(snippet)    (float32 sum, naive-sequential)
    //   3. snippet *= hann(total)      (float32 window)
    //   4. rms = sqrt(float64(mean(snippet²)) + 1e-8)  (scalar f64)
    //   5. snippet /= float32(rms)     (out=float32 forces rms cast)
    //
    // PARITY: feature_extractor.py::_extract_waveform_snippets (L424).
    static std::vector<float>
    extractWaveformSnippets(const float* y,
                            std::size_t nSamples,
                            int sr,
                            const std::vector<double>& beatTimes,
                            std::size_t nBeats,
                            double preMs,
                            double postMs);
};

} // namespace reamix::analysis
