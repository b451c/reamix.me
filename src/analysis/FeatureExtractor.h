#pragma once

#include <cstddef>
#include <vector>

namespace reamix::analysis {

// Phase-2 feature-extraction orchestrator — step 10 of the phase-2 port order.
// Wires the 11 DSP modules (STFT + MelSpectrogramLibrosa + DCT + Mfcc +
// ChromaSTFT + PitchTrack + SpectralContrast + OnsetStrength + RMS +
// SpectralCentroid + BeatSync) together with the step-9 BeatWindows helpers,
// mirroring references/python-source/analysis/feature_extractor.py::
// FeatureExtractor.extract (L189–305 on librosa 0.11.0).
//
// Pipeline (59-dim):
//   y  (f32 audio @ sr=22050)
//     → STFT.magnitude                                [frame × bin] f32
//     → (square in place for power spec)              [frame × bin] f32
//     → Mfcc(40).compute                              [frame × 40]  f32
//     → PitchTrack.estimateTuning(|STFT|²)            scalar f32
//     → ChromaSTFT.compute(|STFT|², tuning)           [frame × 12]  f32
//     → SpectralContrast.compute(|STFT|)              [frame × 7]   f32
//     → vstack to [59 × T] row-major f32              stacked_features
//     → BeatSync.sync → [59 × nSlices] f64 → transpose → [nSlices × 59]
//     → truncate-or-pad-with-last-row to nBeats
//     → cast to f32, L2-normalize per row (norm == 0 → 1)
//   Side channels (RMS, OnsetStrength, SpectralCentroid):
//     → compute per-frame → sync to per-beat mean → truncate to nBeats
//       (NOTE: truncate only, NOT pad-with-last — intentional asymmetry
//        with the main matrix; matches Python L244-254 `[0][:n_beats]`.)
//     → max-normalize [0, 1] with 1e-10 floor
//   Edge + boundary + transition windows:
//     → BeatWindows::extractEdgeFeatures on PRE-L2 stacked matrix
//     → BeatWindows::extractEdgeRms        on y + beat_times
//     → BeatWindows::extractWaveformSnippets ×2 for boundary (35/120 ms)
//       and transition (280/320 ms)
//
// Result struct layout is a flat phase-2-specific struct (not a mirror of
// Python's 18-field FeatureResult with vocal/f0 Optional slots). Phase-2b
// extended this in place per ADR-015 with 8 vocal fields.
//
// PARITY THRESHOLDS (VALIDATION.md phase-2 step-10):
//   features (L2-norm):         L∞ ≤ 1e-3; expected realized ≤ 1e-5.
//   edge*/boundary/transition:  per step-9 row (BeatWindows unchanged).
//   rmsEnergy / onsetStrength / spectralCentroid: L∞ ≤ 1e-3;
//     expected ≤ 1e-6 after [0,1] normalize.
//   vocal*:                     L∞ ≤ 1e-3 on meaningful fields;
//                               == 0.0 bitwise on ADR-014 zero-stubs.
class FeatureExtractor
{
public:
    struct Result
    {
        // Main feature matrix — row-major [nBeats × nFeat], L2-normalized
        // per row. Row k = 59-dim feature vector for beat k, aligned to
        // beatTimes[k]. Float32 to match the dtype of MFCC/chroma/contrast
        // in the real Python pipeline.
        std::vector<float>  features;
        int                 nBeats = 0;
        int                 nFeat  = 0;  // 59

        // Beat times echoed from input — convenience for downstream code
        // that wants a self-contained Result handle.
        std::vector<double> beatTimes;

        // Side channels, beat-synced, max-normalized to [0, 1] with 1e-10
        // floor. Length may be ≤ nBeats if the beat_frames list produced
        // fewer BeatSync slices than there are beats — intentional match
        // with Python's truncate-only `[0][:n_beats]` slice (L244-254).
        std::vector<double> rmsEnergy;
        std::vector<double> onsetStrength;
        std::vector<double> spectralCentroid;

        // Per-beat edge windows (step-9 BeatWindows outputs).
        std::vector<double> edgeFeaturesStart;   // [nBeats × nFeat] f64, L2
        std::vector<double> edgeFeaturesEnd;     // [nBeats × nFeat] f64, L2
        std::vector<double> edgeRmsStart;        // [nBeats], [0, 1]
        std::vector<double> edgeRmsEnd;          // [nBeats], [0, 1]
        std::vector<float>  boundaryWaveforms;   // [nBeats × 3417] f32
        std::vector<float>  transitionWaveforms; // [nBeats × 13230] f32

        // Phase-2b vocal outputs (ADR-015). Names match VocalFeatures::Result
        // 1:1. Populated via HPSS → STFT → VocalFeatures. Meaningful fields
        // are ≤ 1e-3 vs Python; voicedRatio / f0Hz / f0Confidence are
        // constant-zero per ADR-014 D2-D4 and must compare bitwise == 0.0.
        std::vector<double> vocalActivity;          // [nBeats] meaningful
        std::vector<double> voicedRatio;            // [nBeats] zeros (ADR-014 D4)
        std::vector<double> f0Hz;                   // [nBeats] zeros (ADR-014 D2)
        std::vector<double> f0Confidence;           // [nBeats] zeros (ADR-014 D3)
        std::vector<double> edgeVocalActivityStart; // [nBeats] meaningful
        std::vector<double> edgeVocalActivityEnd;   // [nBeats] meaningful
        std::vector<double> edgeVocalOnsetStart;    // [nBeats] meaningful
        std::vector<double> edgeVocalReleaseEnd;    // [nBeats] meaningful
    };

    // Run the 59-dim (or 39-dim) orchestrator on raw mono audio plus a list
    // of beat times in seconds. The beat source is irrelevant — phase-1
    // BeatDetector, librosa.beat.beat_track, or user-supplied; the
    // orchestrator is beat-source-agnostic.
    //
    // Preconditions:
    //   - y non-null for nSamples > 0; sr > 0; hop_length = 512 fixed.
    //   - beatTimes non-empty (throws std::invalid_argument otherwise).
    //   - beatTimes should be within [0, nSamples/sr] for meaningful
    //     output; out-of-range values are silently clipped by downstream
    //     BeatSync / BeatWindows (matches Python's librosa behavior).
    static Result extract(const float* y,
                          std::size_t nSamples,
                          int sr,
                          const std::vector<double>& beatTimes);

    // librosa.time_to_frames port — exposed for unit testing.
    //
    // PARITY (IMPLEMENTATION, not docstring): librosa/core/convert.py
    // on librosa 0.11.0 composes two integer conversions:
    //
    //   samples = (times * sr).astype(int)      # truncate toward zero
    //   frames  = samples // hop                 # floor-divide
    //
    // The librosa DOCSTRING claims `floor(times·sr/hop)` but the code
    // inserts an int-cast between the two steps — functionally different.
    // Spec.md § Step 10 trap-scan initially claimed np.round (round-half-
    // to-even); this was a session-9 trap-scan error corrected session 14.
    // Never use std::nearbyint / std::round here.
    static std::vector<int> timeToFrames(const std::vector<double>& times,
                                         int sr, int hop);
};

} // namespace reamix::analysis
