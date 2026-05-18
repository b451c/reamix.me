#pragma once

#include <vector>

namespace reamix::analysis {

// Vocal-features orchestrator — port of `_extract_vocal_features` from
// `references/python-source/analysis/vocal_features.py` L40-218 in
// **default-mode only** (`vocal_use_pyin=False` — see ADR-014 for the
// pYIN defer rationale). Returns 8 per-beat f64 vectors; 3 of them
// (`voicedRatio`, `f0Hz`, `f0Confidence`) are constant-zero in default
// mode, matching Python's L123-124 + L157 else branches.
//
// The meaningful composite is:
//
//   harmonic_ratio      = clip(RMS(y_harm) / max(RMS(y), 1e-8) / 1.15, 0, 1)
//   flatness            = SpectralFlatness(|STFT(y_harm)|)
//   flatness_inv        = clip(1 - flatness / max(p95(flatness), 1e-6), 0, 1)
//   voice_band_ratio    = clip(mean(|S|[250..3400Hz]) /
//                               max(mean(|S|[80..7000Hz]), 1e-8)
//                               / max(p95(...), 1e-6), 0, 1)
//   vocal_activity_frame = clip(0.40*harm + 0.33*vb + 0.27*flat_inv, 0, 1)
//
// Frame-level outputs are then split into rise / fall via prepended diff,
// percentile-scaled, and aggregated to per-beat via BeatSync::sync +
// edge-window helpers.
//
// API rationale: takes pre-computed upstream `yHarmonic` and
// `magHarm = |STFT(yHarmonic)|` to isolate parity tests from HPSS + STFT
// drift (same pattern as SpectralFlatness / SpectralCentroid). The
// FeatureExtractor orchestrator (phase-2b step 6) computes both once from
// raw `y` and passes them here.
//
// All per-frame arithmetic is done in f64 in C++ (Python runs in f32 until
// the final `np.ascontiguousarray(..., dtype=np.float64)` in the dump).
// Drift from numpy's f32 path is at f32-ULP level (~1e-7), comfortably under
// the 1e-3 per-beat parity gate.
//
// DEFERRED (ADR-014): three outputs are structurally present but constant-zero
// in this phase. Future `phase-2c-pyin` replaces the zero-stubs with real pYIN
// outputs — no API change (ADR-015 struct is stable).
class VocalFeatures
{
public:
    struct Result
    {
        std::vector<double> vocalActivity;            // n_beats — meaningful
        std::vector<double> voicedRatio;              // n_beats — zeros (ADR-014 D4)
        std::vector<double> f0Hz;                     // n_beats — zeros (ADR-014 D2)
        std::vector<double> f0Confidence;             // n_beats — zeros (ADR-014 D3)
        std::vector<double> edgeVocalActivityStart;   // n_beats — meaningful
        std::vector<double> edgeVocalActivityEnd;     // n_beats — meaningful
        std::vector<double> edgeVocalOnsetStart;      // n_beats — meaningful
        std::vector<double> edgeVocalReleaseEnd;      // n_beats — meaningful
    };

    // PARITY: python-source/config.py:60-75 + vocal_features.py:L76-104.
    static constexpr int    kNfft             = 2048;
    static constexpr int    kHopLength        = 512;
    static constexpr double kVoiceLowHz       = 250.0;
    static constexpr double kVoiceHighHz      = 3400.0;
    static constexpr double kBodyLowHz        = 80.0;
    static constexpr double kBodyHighHz       = 7000.0;
    static constexpr double kHarmonicDivisor  = 1.15;  // harm ratio headroom
    static constexpr double kRmsFloor         = 1e-8;  // guards /max(full_rms, 1e-8)
    static constexpr double kBandFloor        = 1e-8;  // guards /max(body, 1e-8)
    static constexpr double kScaleFloor       = 1e-6;  // guards /max(p95, 1e-6)
    static constexpr double kVocalMaxHz       = 1000.0;  // config.py:69
    static constexpr double kF0ClipFactor     = 1.2;
    static constexpr int    kNEdgeFrames      = 4;     // config.py::vocal_edge_frames
    static constexpr double kCompHarm         = 0.40;  // L150 weights
    static constexpr double kCompVoiceBand    = 0.33;
    static constexpr double kCompFlatnessInv  = 0.27;

    VocalFeatures() = default;

    // Inputs:
    //   y          — raw float32 audio (used only for full-track RMS).
    //   yHarmonic  — HPSS harmonic output, float32 (upstream).
    //   magHarm    — |STFT(yHarmonic)|, [frame][bin] float32 (upstream).
    //   sr         — sample rate, Hz.
    //   beatFrames — int64 → int beat positions (STFT-frame index).
    //   nBeats     — expected output length.
    static Result compute(const std::vector<float>& y,
                          const std::vector<float>& yHarmonic,
                          const std::vector<std::vector<float>>& magHarm,
                          double sr,
                          const std::vector<int>& beatFrames,
                          int nBeats);
};

} // namespace reamix::analysis
