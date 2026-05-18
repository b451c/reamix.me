#pragma once

#include "analysis/StructureResult.h"

#include <utility>
#include <vector>

namespace reamix::analysis {

// Bar-level repetition detection for inaudible-splice candidate mining.
// Port of `build_repetition_map` (python-source/analysis/repetition_map.py
// L284-532). Finds pairs of bars across the song that are musically
// identical repetitions (e.g., verse1 bar 4 ↔ verse2 bar 4) — these pairs
// are the BEST candidates for inaudible transitions because the actual
// audio content is nearly the same (Adobe US 9,280,313 method).
//
// Pipeline (two phases):
//   1. Cross-section scan (L349-448). Group dispatched segments by label
//      (intro/outro/unknown skipped), for each pair of same-label sections
//      compute bar-level chroma cross-correlation with offsets ∈ [-4, +4]
//      bars, threshold at 0.80; map corresponding bars with the winning
//      offset; verify each bar-boundary via waveform cross-correlation at
//      ±30 ms lag; emit both forward and reverse jumps.
//   2. Internal repetitions via recurrence diagonals (L450-528). Inside
//      segments long enough for 4 × (2-bar min phrase), search bar-aligned
//      diagonal lags in the full recurrence matrix; emit the top-5
//      diagonals with mean_sim ≥ 0.10 and lag ≥ 2 bars; relaxed waveform
//      gate at 0.75 × min_waveform_similarity.
// Final sort: `(-waveform_similarity, -chroma_correlation)` (L530-531).
//
// Dependency graph:
//   - chroma indices via `chromaRange(nFeat)` (mirrors config.chroma_range).
//   - WaveformXcorr::compute (phase-3 session-11 port; matches
//     waveform_utils.py::waveform_xcorr).
//   - Recurrence::build(..., kNeighbors = 12) for phase-2 diagonals.
//
// Determinism: no RNG, no spectral decomposition. All f64 in helper math.
// Label comparisons are lowercase-fold (matches Python `.lower()`). Group
// iteration preserves INSERTION ORDER of labels in the `segments` input
// (matches Python 3.7+ dict order) — implemented via std::vector<pair>.
//
// PARITY NOTES:
//   1. `segments.index(seg) if seg in segments else 0` (L515): Python
//      identity-based lookup into the original segments list. C++ equivalent
//      is the outer-loop index when iterating `segments` directly.
//   2. `segment.cluster_id` and `segment.label` flow from StructureAnalyzer.
//      On the eminem_without_me track they diverge from Python per
//      ADR-022 K-means MC-sampling finding; the C++ parity test applies
//      the same `kAdr022KmeansExceptions` waiver used in
//      test_structure_analyzer.cpp.
//   3. Phase-2 diagonal search does NOT import `find_diagonals` from
//      Python's recurrence.py (the `from ... import find_diagonals` line is
//      dead code — the algorithm uses its own inline bar-multiple lag
//      loop at L473-494). We mirror the inline pattern; no findDiagonals
//      helper is needed.
struct RepetitionJump
{
    int     fromBeat;                // Last beat of outgoing bar (pre-downbeat)
    int     toBeat;                  // First beat of incoming bar (downbeat)
    double  waveformSimilarity;      // 0-1, from waveform xcorr at bar boundary
    double  chromaCorrelation;       // 0-1, bar-level chroma correlation
    int     alignmentLagSamples;     // Micro-alignment offset in samples
    int     fromSectionIdx;
    int     toSectionIdx;
    int     fromBar;                 // Bar index within section
    int     toBar;
};

struct RepetitionResult
{
    std::vector<RepetitionJump>            jumps;
    std::vector<std::pair<int, int>>       sectionPairs;
    int                                    nSectionsScanned = 0;
    int                                    nPairsVerified   = 0;
};

class RepetitionMap
{
public:
    // PARITY: repetition_map.py L292-293 defaults.
    static constexpr double kDefaultMinChromaCorr   = 0.80;
    static constexpr double kDefaultMinWaveformSim  = 0.40;
    static constexpr int    kDefaultTimeSignature   = 4;
    // Recurrence k_neighbors passed at L457: `k_neighbors=12`.
    static constexpr int    kRecurrenceKNeighbors   = 12;
    // Chroma/contrast layout mirrors config.chroma_range (n - 12 - 7, n - 7).
    static constexpr int    kNChromaDims            = 12;
    static constexpr int    kNContrastDims          = 7;

    // Build the repetition map.
    //
    // `beatTimes`:       (nBeats,) f64, seconds.
    // `downbeats`:       (nDownbeats,) f64, seconds. Pass nullptr / nDownbeats=0
    //                    for librosa-style synthetic-downbeat fallback.
    // `features`:        [nBeats × nFeat] row-major f32. L2-normalized (matches
    //                    production FeatureExtractor output); function re-
    //                    normalizes bar chromas anyway.
    // `segments`:        dispatcher-consolidated segments (see StructureResult).
    // `boundaryWaveforms`: [nBeats × snippetLen] row-major f32. Pass nullptr
    //                    (with snippetLen = 0) to skip waveform verification —
    //                    phase-1 jumps then accept on chroma alone, phase-2
    //                    gets wfSim = 0. (Mirrors Python `has_waveforms=False`.)
    // `waveformSampleRate`: sample rate for `max_lag_samples = int(30 ms · SR)`.
    // `timeSignature`, `minChromaCorr`, `minWaveformSim`: see Python defaults.
    //
    // Early-out per L317-318: `nBeats < 8 OR segments.size() < 2` → empty result.
    // Early-out per L329-330: `bars.size() < 4` → empty result.
    static RepetitionResult build(const double* beatTimes,
                                  int           nBeats,
                                  const double* downbeats,
                                  int           nDownbeats,
                                  const float*  features,
                                  int           nFeat,
                                  const std::vector<Segment>& segments,
                                  const float*  boundaryWaveforms,
                                  int           snippetLen,
                                  int           waveformSampleRate,
                                  int           timeSignature  = kDefaultTimeSignature,
                                  double        minChromaCorr  = kDefaultMinChromaCorr,
                                  double        minWaveformSim = kDefaultMinWaveformSim);
};

} // namespace reamix::analysis
