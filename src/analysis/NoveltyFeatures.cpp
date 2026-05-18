#include "analysis/NoveltyFeatures.h"

#include "analysis/Mfcc.h"
#include "dsp/ChromaSTFT.h"
#include "dsp/PitchTrack.h"
#include "dsp/STFT.h"

#include <cmath>

namespace reamix::analysis {

namespace {

// Element-wise square of a [frame × bin] amplitude spectrogram, float32.
// Matches librosa's `_spectrogram(power=2)` internal path:
// complex64 → float32 abs → float32 square — the input PitchTrack and
// ChromaSTFT expect. Duplicated from FeatureExtractor.cpp to keep
// NoveltyFeatures self-contained (Principle 3 Surgical Changes: no
// refactor out to a shared header while there are only two callers).
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

// Per-coefficient (per-row in Python's (dim, T) layout) z-score in place.
// Mirrors PARITY: structure_analyzer.py:202-207
//   x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
//
// Input frames is our C++ [frame][dim] layout; Python's axis=1 on (dim, T)
// is "across time per coefficient" which maps to "iterate dim at outer,
// frame at inner" here.
//
// Accumulators run in double to stay below the 1e-7 f32 ULP floor on
// sum/sum-of-squares; the final subtraction + divide downcast the
// mean/std scalars to float32 before touching the data, matching Python's
// f32 numerator = (x - cast(mean)) and denominator = (cast(std) + 1e-8).
// Population variance (ddof=0) — numpy's default for `.std`.
inline void
zscoreInPlace(std::vector<std::vector<float>>& frames, int dim)
{
    const std::size_t T = frames.size();
    if (T == 0) return;
    for (int d = 0; d < dim; ++d) {
        double sum = 0.0;
        for (std::size_t t = 0; t < T; ++t)
            sum += static_cast<double>(frames[t][d]);
        const double mean = sum / static_cast<double>(T);

        double sumSq = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            const double diff = static_cast<double>(frames[t][d]) - mean;
            sumSq += diff * diff;
        }
        const double variance = sumSq / static_cast<double>(T);
        const double stddev   = std::sqrt(variance);

        const float meanF = static_cast<float>(mean);
        // PARITY: Python `std + 1e-8` operates in float32 (np.std returns
        // float32 on float32 input; scalar 1e-8 keeps array dtype). Cast
        // stddev to float32 before adding the guard.
        const float denomF = static_cast<float>(stddev) + NoveltyFeatures::kStdEps;

        for (std::size_t t = 0; t < T; ++t)
            frames[t][d] = (frames[t][d] - meanF) / denomF;
    }
}

} // namespace

std::vector<std::vector<float>>
NoveltyFeatures::extract(const float* y, std::size_t nSamples, int sr)
{
    if (y == nullptr || nSamples == 0) return {};

    const std::vector<float> yVec(y, y + nSamples);

    // ---- STFT once, shared between the chroma-path (|·|²) and the MFCC
    // sub-pipeline (which redoes its own STFT inside MelSpectrogramLibrosa::power).
    // Redundancy mirrors phase-2 FeatureExtractor — intentionally not optimized
    // at this port step; simplicity + parity isolation over performance.
    reamix::dsp::STFT stft;
    const auto stftMag = stft.magnitude(yVec);                  // [frame][bin] f32
    const auto stftPow = amplitudeToPower(stftMag);             // [frame][bin] f32

    // ---- MFCC(13) ---------------------------------------------------------
    // PARITY: structure_analyzer.py:195-197. Mfcc::compute applies
    //   power_to_db(mel_power, ref=1.0) → DCT(n_mels=128, n_mfcc=13) — the
    // exact pipeline `librosa.feature.mfcc(y, sr, n_mfcc=13, hop=512)` uses.
    Mfcc mfccMod(kNMfcc);
    auto mfcc = mfccMod.compute(yVec);                          // [frame][13] f32

    // ---- Chroma(12) with tuning estimated from the same |STFT|² -----------
    // PARITY: structure_analyzer.py:198-200. Python's
    //   librosa.feature.chroma_stft(y, sr, hop=512, n_fft=2048)
    // with `tuning=None` triggers `estimate_tuning(S=|STFT|², bpo=n_chroma=12)`
    // internally — the same scalar PitchTrack::estimateTuning produces here.
    // ADR-010 closed this equivalence at phase-2.
    const float tuning = reamix::dsp::PitchTrack{}.estimateTuning(
        stftPow, static_cast<float>(sr), 12);
    auto chroma = reamix::dsp::ChromaSTFT{}.compute(
        stftPow, tuning, static_cast<float>(sr));               // [frame][12] f32

    // mfcc, chroma must share frame grid (same STFT parameters → same T
    // under center=True pad). Defend with a run-time size match rather than
    // an assert — empty input fell out early, so nonempty-T mismatch here
    // would indicate a future wiring regression.
    const std::size_t T = mfcc.size();
    if (T == 0 || chroma.size() != T) return {};

    // ---- Per-row z-score (ddof=0, std + 1e-8 guard) -----------------------
    zscoreInPlace(mfcc,   kNMfcc);
    zscoreInPlace(chroma, kNChroma);

    // ---- Vstack [mfcc * 1.5, chroma] → (T, 25) ----------------------------
    // PARITY: structure_analyzer.py:209. Python's (25, T) matrix in our
    // [frame][25] layout is MFCC rows 0..12 followed by chroma rows 0..11.
    std::vector<std::vector<float>> features(T, std::vector<float>(kNFeat));
    for (std::size_t t = 0; t < T; ++t) {
        for (int i = 0; i < kNMfcc; ++i)
            features[t][i] = mfcc[t][i] * kMfccWeight;
        for (int j = 0; j < kNChroma; ++j)
            features[t][kNMfcc + j] = chroma[t][j];
    }
    return features;
}

} // namespace reamix::analysis
