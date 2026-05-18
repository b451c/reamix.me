#include "analysis/FeatureExtractor.h"
#include <cstdint>

#include "analysis/BeatSync.h"
#include "analysis/BeatWindows.h"
#include "analysis/Mfcc.h"
#include "analysis/VocalFeatures.h"
#include "dsp/ChromaSTFT.h"
#include "dsp/HPSS.h"
#include "dsp/MelSpectrogramLibrosa.h"
#include "dsp/OnsetStrength.h"
#include "dsp/PitchTrack.h"
#include "dsp/RMS.h"
#include "dsp/STFT.h"
#include "dsp/SpectralCentroid.h"
#include "dsp/SpectralContrast.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace reamix::analysis {

namespace {

// PARITY: librosa.time_to_frames (librosa/core/convert.py, librosa 0.11.0).
//
// IMPORTANT: the librosa DOCSTRING says `frames[i] = floor(times[i]·sr/hop)`
// but the IMPLEMENTATION composes two distinct integer conversions:
//
//   samples = (times * sr).astype(int)     # truncate toward zero
//   frames  = samples // hop_length        # floor-divide
//
// The inner `.astype(int)` and outer `//` behave identically on positive
// inputs (all our beat_times) but they are NOT equivalent to a single
// `floor(times * sr / hop)` — the intermediate int-cast can push values
// to one side of a hop-integer boundary. Nor is it round-half-to-even:
// phase-2 spec.md § Step 10 trap-scan claims `np.round` rounding; this is
// a session-9 trap-scan error (see session-14 log). For synthetic beats
// `beat_frames * hop / sr → time → frames` the two steps together CAN
// return a frame one less than the original beat_frame (e.g. beat_frame
// 30 → time 0.69659... → samples 15359.99... → int 15359 → 15359//512 =
// 29, vs `round(30) = 30`). Downstream BeatSync/BeatWindows then operate
// on the wrong boundary list, producing O(10 %) drift in features.
//
// Implementation: faithful two-step composition.
inline std::vector<int>
timeToFramesImpl(const std::vector<double>& times, int sr, int hop)
{
    std::vector<int> out(times.size());
    const double sr_d   = static_cast<double>(sr);
    const std::int64_t hop_i = static_cast<std::int64_t>(hop);
    for (std::size_t i = 0; i < times.size(); ++i) {
        // PARITY: librosa.time_to_samples (librosa/core/convert.py):
        //   return (np.asanyarray(times) * sr).astype(int)
        // `.astype(int)` truncates toward zero; on positive samples this
        // equals floor. All phase-2 beat_times are non-negative.
        const std::int64_t samples = static_cast<std::int64_t>(times[i] * sr_d);

        // PARITY: librosa.samples_to_frames (librosa/core/convert.py):
        //   return np.asarray(np.floor((samples - offset) // hop_length), dtype=int)
        // Python `//` is floor-div; on non-negative samples C++ `/` is
        // truncation-toward-zero which equals floor. offset=0 (we don't
        // pass n_fft), so the expression simplifies to `samples // hop`.
        out[i] = static_cast<int>(samples / hop_i);
    }
    return out;
}

// Element-wise square of a [frame × bin] float32 amplitude spectrogram.
// PARITY: librosa/feature/spectral.py::_spectrogram(power=2) — complex64 →
// float32 abs → float32 square. The two modules we feed from this (PitchTrack
// and ChromaSTFT) both expect |STFT|² float32, matching librosa's internal
// shape exactly.
inline std::vector<std::vector<float>>
amplitudeToPower(const std::vector<std::vector<float>>& sAmp)
{
    std::vector<std::vector<float>> sPow(sAmp.size());
    for (std::size_t t = 0; t < sAmp.size(); ++t) {
        sPow[t].resize(sAmp[t].size());
        for (std::size_t b = 0; b < sAmp[t].size(); ++b) {
            const float a = sAmp[t][b];
            sPow[t][b] = a * a;
        }
    }
    return sPow;
}

// Stack [frame × nMfcc] + [frame × 12] + [frame × 7] into row-major
// [F × T] float32 matrix matching Python's np.vstack([mfcc, chroma, contrast]).
//
// Layout convention: stacked[f * T + t] = value of feature f at time t,
// with feature ordering MFCC (0..nMfcc) → chroma (nMfcc..nMfcc+12) →
// contrast (nMfcc+12..nMfcc+19). Any permutation silently corrupts the
// 10-signal cost matrix in phase 4 — see spec.md § Step 10.
//
// PARITY: feature_extractor.py:210 stacked_features = np.vstack(all_features).
inline std::vector<float>
stackRowMajor(const std::vector<std::vector<float>>& mfcc,
              const std::vector<std::vector<float>>& chroma,
              const std::vector<std::vector<float>>& contrast,
              std::size_t& F, std::size_t& T)
{
    if (mfcc.empty()) { F = 0; T = 0; return {}; }
    const std::size_t nMfcc     = mfcc[0].size();
    const std::size_t nChroma   = 12;
    const std::size_t nContrast = 7;
    F = nMfcc + nChroma + nContrast;
    T = mfcc.size();
    std::vector<float> out(F * T, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t m = 0; m < nMfcc; ++m)
            out[m * T + t] = mfcc[t][m];
        for (std::size_t c = 0; c < nChroma; ++c)
            out[(nMfcc + c) * T + t] = chroma[t][c];
        for (std::size_t b = 0; b < nContrast; ++b)
            out[(nMfcc + nChroma + b) * T + t] = contrast[t][b];
    }
    return out;
}

// Per-beat mean-aggregation of a 1-D frame sequence, followed by
// truncate-only (NOT pad-with-last) to nBeats, then max-normalize to [0, 1]
// with 1e-10 floor.
//
// LANDMINE — intentional asymmetry with the main feature-matrix path:
//   - feature-matrix path uses the full _beat_sync_features helper
//     (feature_extractor.py:465-475) which does truncate-or-pad-with-last.
//   - side-channel path uses `[0][:n_beats]` (L245/L249/L253) which
//     silently truncates without padding.
// Python preserves this asymmetry and so must we. If nSlices < nBeats,
// the returned vector is genuinely shorter than nBeats — callers that
// assume size == nBeats on these fields will need to check.
inline std::vector<double>
syncAndNormalize1DTruncate(const std::vector<float>& x_f32,
                           const std::vector<int>& beatFrames,
                           std::size_t nBeats)
{
    // Upcast f32 → f64 for BeatSync's double* API. librosa.util.sync on
    // f32 inputs aggregates in f32; our f64 accumulation differs by
    // ~ULP(f32_sum) ≈ few × 1e-7 on typical mean magnitudes (expected
    // drift well under the 1e-3 gate).
    const std::size_t T = x_f32.size();
    std::vector<double> x_f64(T);
    for (std::size_t i = 0; i < T; ++i)
        x_f64[i] = static_cast<double>(x_f32[i]);

    auto res = BeatSync::sync(x_f64.data(), 1, T, beatFrames);
    // res.data layout for D=1: [0 * nSlices + k] = mean over slice k.

    const std::size_t n = std::min(res.nSlices, nBeats);
    std::vector<double> synced(n);
    for (std::size_t k = 0; k < n; ++k)
        synced[k] = res.data[k];

    // PARITY: feature_extractor.py:246/250/254 — `/ max(x.max(), 1e-10)`.
    // Max-normalize with 1e-10 floor. Python computes max on the f32
    // vector then promotes to f64 at division; our f64 max is equivalent
    // up to ULP.
    double m = 0.0;
    for (double v : synced) if (v > m) m = v;
    const double denom = std::max(m, 1e-10);
    for (double& v : synced) v /= denom;
    return synced;
}

} // anonymous namespace

std::vector<int>
FeatureExtractor::timeToFrames(const std::vector<double>& times, int sr, int hop)
{
    return timeToFramesImpl(times, sr, hop);
}

FeatureExtractor::Result
FeatureExtractor::extract(const float* y, std::size_t nSamples, int sr,
                          const std::vector<double>& beatTimes)
{
    if (beatTimes.empty())
        throw std::invalid_argument("FeatureExtractor::extract: beatTimes is empty");
    if (y == nullptr && nSamples > 0)
        throw std::invalid_argument("FeatureExtractor::extract: null y with nSamples>0");

    constexpr int nMfccDim   = 40;
    const std::size_t nBeats = beatTimes.size();

    Result out;
    out.nBeats    = static_cast<int>(nBeats);
    out.nFeat     = nMfccDim + 12 + 7;
    out.beatTimes = beatTimes;

    const std::vector<float> yVec(y, y + nSamples);

    // ---- STFT (amplitude + power) ---------------------------------------
    // Single STFT computation shared with SpectralContrast + SpectralCentroid
    // (amplitude) and PitchTrack + ChromaSTFT (power). Mfcc and MelLibrosa
    // do redo STFT internally — accepted redundancy for a clean first pass;
    // performance optimization is stretch scope per ROADMAP.
    reamix::dsp::STFT stft;
    const auto stftMag = stft.magnitude(yVec);               // [frame][bin] f32
    const auto stftPow = amplitudeToPower(stftMag);          // [frame][bin] f32

    // ---- MFCC (width depends on mode) -----------------------------------
    // PARITY: feature_extractor.py:307-315 _extract_mfcc calls
    //   librosa.feature.mfcc(y, sr, n_mfcc=self._n_mfcc, hop, n_fft).
    Mfcc mfccMod(nMfccDim);
    const auto mfcc = mfccMod.compute(yVec);                 // [frame][nMfcc] f32

    // ---- Chroma — EXPLICIT tuning pipeline ------------------------------
    // librosa.feature.chroma_stft(y=y) without tuning= triggers internal
    // estimate_tuning(S=|STFT|², bins_per_octave=12). We replicate
    // explicitly — PARITY: feature_extractor.py:197,
    // librosa/feature/spectral.py::chroma_stft (librosa 0.11.0).
    //
    // Jawne wywołanie eliminuje hidden-coupling: ChromaSTFT::compute
    // requires tuning as an explicit parameter and will NOT compute it
    // internally if omitted.
    const float tuning = reamix::dsp::PitchTrack{}.estimateTuning(
        stftPow, static_cast<float>(sr), 12);
    const auto chroma = reamix::dsp::ChromaSTFT{}.compute(
        stftPow, tuning, static_cast<float>(sr));            // [frame][12] f32

    // ---- SpectralContrast (amplitude input) -----------------------------
    // PARITY: feature_extractor.py:203-208.
    const auto contrast = reamix::dsp::SpectralContrast{}.compute(
        stftMag, static_cast<float>(sr));                    // [frame][7]  f32

    // ---- Stack [F × T] row-major f32 -------------------------------------
    std::size_t F = 0, T = 0;
    const std::vector<float> stacked = stackRowMajor(mfcc, chroma, contrast, F, T);
    assert(F == static_cast<std::size_t>(out.nFeat));

    // ---- Beat-frame conversion (round-half-to-even) ---------------------
    const std::vector<int> beatFrames = timeToFramesImpl(
        beatTimes, sr, reamix::dsp::STFT::kHopLength);

    // ---- BeatSync on stacked (→ main feature matrix) --------------------
    // BeatSync's API takes double*; upcast f32 → f64 for the call.
    // The precision asymmetry vs Python's f32 sync is well under 1e-6
    // (confirmed by side-channel parity budget above).
    std::vector<double> stackedF64(F * T);
    for (std::size_t i = 0; i < F * T; ++i)
        stackedF64[i] = static_cast<double>(stacked[i]);
    const auto syncedFeat = BeatSync::sync(
        stackedF64.data(), F, T, beatFrames);
    // syncedFeat.data layout: [F × nSlices] row-major (matches BeatSync API).

    // Transpose to [nSlices × F] row-major (matches Python's
    // _beat_sync_features' `synced.T` step at L469).
    std::vector<double> syncedT(syncedFeat.nSlices * F);
    for (std::size_t k = 0; k < syncedFeat.nSlices; ++k)
        for (std::size_t f = 0; f < F; ++f)
            syncedT[k * F + f] = syncedFeat.data[f * syncedFeat.nSlices + k];

    // Truncate-or-pad-with-last-row to exactly nBeats.
    // LANDMINE — PARITY: feature_extractor.py:470-474
    //   _beat_sync_features truncates to n_beats if nSlices > n_beats;
    //   pads with tile(synced[-1:], (pad, 1)) if nSlices < n_beats.
    // DO NOT "simplify" to truncate-only or drop the pad branch; the
    // main-matrix path DOES pad, unlike the side-channel path below.
    {
        std::size_t nSlices = syncedFeat.nSlices;
        if (nSlices > nBeats) {
            syncedT.resize(nBeats * F);
            nSlices = nBeats;
        } else if (nSlices < nBeats) {
            const std::size_t oldSize = syncedT.size();
            syncedT.resize(nBeats * F);
            if (nSlices == 0) {
                // Unreachable for T >= 1 (librosa.util.sync guarantees at
                // least one slice); zero-fill for defense.
                std::fill(syncedT.begin(), syncedT.end(), 0.0);
            } else {
                const std::size_t lastBase = (nSlices - 1) * F;
                for (std::size_t k = nSlices; k < nBeats; ++k) {
                    const std::size_t base = k * F;
                    for (std::size_t f = 0; f < F; ++f)
                        syncedT[base + f] = syncedT[lastBase + f];
                }
            }
            (void)oldSize; // silence unused in release
        }
    }

    // Downcast to f32 (matches Python: beat_features is f32 pre-L2 because
    // librosa.util.sync on f32 input stays f32).
    std::vector<float> beatFeaturesF32(nBeats * F);
    for (std::size_t i = 0; i < nBeats * F; ++i)
        beatFeaturesF32[i] = static_cast<float>(syncedT[i]);

    // ---- Edge features — BEFORE L2 normalize on the main matrix ---------
    // PARITY: feature_extractor.py:216-222. Edge features operate on the
    // PRE-L2 stacked_features matrix; they have their own internal L2 per
    // row inside BeatWindows::extractEdgeFeatures.
    {
        auto edgeFeat = BeatWindows::extractEdgeFeatures(
            stacked.data(), F, T, beatFrames, nBeats);
        out.edgeFeaturesStart = std::move(edgeFeat.start);
        out.edgeFeaturesEnd   = std::move(edgeFeat.end);
    }

    // ---- Edge RMS (sample-domain) ---------------------------------------
    {
        auto edgeRms = BeatWindows::extractEdgeRms(
            y, nSamples, sr, beatTimes, nBeats);
        out.edgeRmsStart = std::move(edgeRms.start);
        out.edgeRmsEnd   = std::move(edgeRms.end);
    }

    // ---- Boundary + transition waveforms --------------------------------
    out.boundaryWaveforms = BeatWindows::extractWaveformSnippets(
        y, nSamples, sr, beatTimes, nBeats,
        /*preMs=*/35.0, /*postMs=*/120.0);
    out.transitionWaveforms = BeatWindows::extractWaveformSnippets(
        y, nSamples, sr, beatTimes, nBeats,
        /*preMs=*/280.0, /*postMs=*/320.0);

    // ---- L2 normalize per beat row (zero-guard: norm := 1) --------------
    // PARITY: feature_extractor.py:233-235.
    //   norms = np.linalg.norm(beat_features, axis=1, keepdims=True)
    //   norms[norms == 0] = 1.0
    //   beat_features = beat_features / norms
    // Zero-guard MUST assign 1.0, NOT tiny or 1e-10 — preserves zero rows
    // exactly as zero in the output.
    for (std::size_t k = 0; k < nBeats; ++k) {
        double sumSq = 0.0;
        const std::size_t base = k * F;
        for (std::size_t f = 0; f < F; ++f) {
            const double v = static_cast<double>(beatFeaturesF32[base + f]);
            sumSq += v * v;
        }
        const double norm = std::sqrt(sumSq);
        const float normF = (norm == 0.0) ? 1.0f : static_cast<float>(norm);
        for (std::size_t f = 0; f < F; ++f)
            beatFeaturesF32[base + f] /= normF;
    }
    out.features = std::move(beatFeaturesF32);

    // ---- Side channels (truncate-only, max-normalize [0, 1]) ------------
    // PARITY: feature_extractor.py:244-254.
    reamix::dsp::RMS rmsMod;
    const auto rmsFrames = rmsMod.compute(yVec);              // [T] f32

    reamix::dsp::MelSpectrogramLibrosa melMod;
    const auto melPower = melMod.power(yVec);                 // [frame][mel] f32

    const auto onsetFrames = reamix::dsp::OnsetStrength{}.compute(melPower); // [T] f32

    const auto centroidFrames = reamix::dsp::SpectralCentroid{}.compute(
        stftMag, static_cast<float>(sr));                     // [T] f32

    out.rmsEnergy        = syncAndNormalize1DTruncate(rmsFrames,      beatFrames, nBeats);
    out.onsetStrength    = syncAndNormalize1DTruncate(onsetFrames,    beatFrames, nBeats);
    out.spectralCentroid = syncAndNormalize1DTruncate(centroidFrames, beatFrames, nBeats);

    // ---- Phase-2b vocal features ----------------------------------------
    // PARITY: feature_extractor.py:264-281 — default-mode composite proxy
    // (vocal_use_pyin=False per ADR-014).
    //
    // The Python reference passes the same beat_frames computed at L211-213
    // (via librosa.time_to_frames) into _extract_vocal_features. We use the
    // C++ `beatFrames` (built via timeToFramesImpl above) unchanged — the
    // two-step int-cast trap (ADR-013) is applied consistently to both the
    // feature-matrix BeatSync and vocal BeatSync slicings.
    //
    // New work on top of phase-2: HPSS::harmonic(y) (~5.3 s on 180-s clip)
    // plus STFT::magnitude(yHarmonic) (separate from the phase-2 `stftMag`
    // which runs on raw y). VocalFeatures is an orchestrator over these two
    // plus raw y; the magnitude-input API isolates parity from upstream
    // STFT/HPSS drift (see session-4 log).
    {
        const auto yHarmonic = reamix::dsp::HPSS::harmonic(yVec);
        const auto magHarm   = stft.magnitude(yHarmonic);
        auto vf = VocalFeatures::compute(
            yVec, yHarmonic, magHarm,
            static_cast<double>(sr), beatFrames, static_cast<int>(nBeats));
        out.vocalActivity          = std::move(vf.vocalActivity);
        out.voicedRatio            = std::move(vf.voicedRatio);
        out.f0Hz                   = std::move(vf.f0Hz);
        out.f0Confidence           = std::move(vf.f0Confidence);
        out.edgeVocalActivityStart = std::move(vf.edgeVocalActivityStart);
        out.edgeVocalActivityEnd   = std::move(vf.edgeVocalActivityEnd);
        out.edgeVocalOnsetStart    = std::move(vf.edgeVocalOnsetStart);
        out.edgeVocalReleaseEnd    = std::move(vf.edgeVocalReleaseEnd);
    }

    return out;
}

} // namespace reamix::analysis
