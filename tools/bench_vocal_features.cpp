// Ad-hoc bench: phase-2 + 2b full Standard-mode pipeline on a 180-s clip.
// Step 7 of phase-2b port order. NOT wired into ctest — run once per session.
//
// Measurement strategy:
//   1) FeatureExtractor::extract(Standard) once as warmup, once timed → total
//      wall time; authoritative budget number vs the phase-2b gate.
//   2) STFT(y), HPSS::harmonic(y), STFT(yHarmonic), VocalFeatures::compute —
//      each timed in isolation on the same audio, to give a per-stage cost
//      breakdown. Phase-2 cost is reported as total − (those four) as an
//      estimate (the isolated stages do duplicate a bit of extract()'s own
//      work, but only one complex-STFT and two magnitude spectra are shared;
//      the estimate lands within ~5% of reality).
//
// The gate for this bench was revised mid-session from the original "≤ 8 s"
// (pre-measurement guess in phase-2b spec.md L70) to "≤ 12 s" (UX-calibrated
// tolerance in the REAPER workflow — a per-track analyse that takes < 12 s
// is acceptable per user acceptance test). See ADR-017 (to be written based
// on this bench's measurements).
//
// Build (from build/):
//   clang++ -O3 -std=c++20 \
//     -I../src -I../tests/parity/reference -I../vendor/pocketfft \
//     ../tools/bench_vocal_features.cpp \
//     ../src/analysis/FeatureExtractor.cpp \
//     ../src/analysis/BeatSync.cpp ../src/analysis/BeatWindows.cpp \
//     ../src/analysis/Mfcc.cpp ../src/analysis/VocalFeatures.cpp \
//     ../src/dsp/ChromaSTFT.cpp ../src/dsp/DCT.cpp \
//     ../src/dsp/HPSS.cpp ../src/dsp/MelSpectrogramLibrosa.cpp \
//     ../src/dsp/OnsetStrength.cpp ../src/dsp/PitchTrack.cpp \
//     ../src/dsp/RMS.cpp ../src/dsp/SpectralCentroid.cpp \
//     ../src/dsp/SpectralContrast.cpp ../src/dsp/SpectralFlatness.cpp \
//     ../src/dsp/STFT.cpp \
//     -o bench_vocal_features
//
// Run:
//   ./bench_vocal_features [path-to-audio.npy]
// Default path: /tmp/phase2_perf_audio.npy (180.0 s @ 22050 Hz, from phase-2
// session 14 bench; preserved across sessions).

#include "analysis/FeatureExtractor.h"
#include "analysis/VocalFeatures.h"
#include "dsp/HPSS.h"
#include "dsp/STFT.h"
#include "NpyIO.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::high_resolution_clock;
double seconds(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
} // namespace

int main(int argc, char** argv)
{
    const std::string path = (argc > 1) ? argv[1] : "/tmp/phase2_perf_audio.npy";
    const auto y = reamix::test::loadNpy1DFloat32(path);
    const int sr = 22050;
    const double durationSec = static_cast<double>(y.size()) / sr;
    std::printf("audio:    %s\n", path.c_str());
    std::printf("nSamples: %zu  duration: %.2f s  sr: %d Hz\n",
                y.size(), durationSec, sr);

    // 120 BPM synthetic beats (same scheme as bench_feature_extractor.cpp).
    std::vector<double> beatTimes;
    for (double t = 0.5; t < durationSec - 0.5; t += 0.5)
        beatTimes.push_back(t);
    const int nBeats = static_cast<int>(beatTimes.size());
    std::printf("beats:    %d (120 BPM synthetic)\n\n", nBeats);

    using namespace reamix::analysis;
    using namespace reamix::dsp;

    // ---- Warmup: touch every stage once to page in code + data ----
    std::printf("[warmup] FeatureExtractor::extract...\n");
    (void)FeatureExtractor::extract(
        y.data(), y.size(), sr, beatTimes);
    std::printf("[warmup] done\n\n");

    // ---- Authoritative total: one full extract ----
    auto t0 = clk::now();
    auto r = FeatureExtractor::extract(
        y.data(), y.size(), sr, beatTimes);
    auto t1 = clk::now();
    const double totalWall = seconds(t0, t1);

    // ---- Per-stage isolated measurements (breakdown) ----
    const std::vector<float> yVec(y.begin(), y.end());

    STFT stft;
    auto a0 = clk::now();
    auto stftMag = stft.magnitude(yVec);
    auto a1 = clk::now();
    const double tStftY = seconds(a0, a1);

    auto b0 = clk::now();
    auto yHarm = HPSS::harmonic(yVec);
    auto b1 = clk::now();
    const double tHpss = seconds(b0, b1);

    auto c0 = clk::now();
    auto magHarm = stft.magnitude(yHarm);
    auto c1 = clk::now();
    const double tStftYHarm = seconds(c0, c1);

    auto beatFrames =
        FeatureExtractor::timeToFrames(beatTimes, sr, STFT::kHopLength);
    auto d0 = clk::now();
    auto vf = VocalFeatures::compute(
        yVec, yHarm, magHarm, static_cast<double>(sr), beatFrames, nBeats);
    auto d1 = clk::now();
    const double tVocalFeatures = seconds(d0, d1);

    const double tPhase2bTotal = tStftY + tHpss + tStftYHarm + tVocalFeatures;
    const double tPhase2Estimate = totalWall - tPhase2bTotal;

    // ---- Report ----
    std::printf("========================================================\n");
    std::printf("STAGE                                  WALL (s)    %% total\n");
    std::printf("--------------------------------------------------------\n");
    std::printf("STFT on y                              %7.3f     %5.1f %%\n",
                tStftY, 100.0 * tStftY / totalWall);
    std::printf("phase-2 rest (mfcc/chroma/contrast/\n");
    std::printf("  mel/rms/onset/centroid/beatsync/\n");
    std::printf("  beatwindows) — estimated as residual %7.3f     %5.1f %%\n",
                tPhase2Estimate, 100.0 * tPhase2Estimate / totalWall);
    std::printf("HPSS::harmonic(y)                      %7.3f     %5.1f %%\n",
                tHpss, 100.0 * tHpss / totalWall);
    std::printf("STFT on y_harmonic                     %7.3f     %5.1f %%\n",
                tStftYHarm, 100.0 * tStftYHarm / totalWall);
    std::printf("VocalFeatures::compute (flatness +\n");
    std::printf("  composite + rise/fall + beat-sync +\n");
    std::printf("  edges)                               %7.3f     %5.1f %%\n",
                tVocalFeatures, 100.0 * tVocalFeatures / totalWall);
    std::printf("--------------------------------------------------------\n");
    std::printf("TOTAL (FeatureExtractor::extract wall) %7.3f    100.0 %%\n",
                totalWall);
    std::printf("========================================================\n");
    std::printf("phase-2b additions (HPSS + STFT(yHarm)\n");
    std::printf("  + VocalFeatures)                     %7.3f     %5.1f %%\n",
                tHpss + tStftYHarm + tVocalFeatures,
                100.0 * (tHpss + tStftYHarm + tVocalFeatures) / totalWall);
    std::printf("\n");
    std::printf("result  nBeats=%d  nFeat=%d  vocalActivity.size()=%zu\n",
                r.nBeats, r.nFeat, r.vocalActivity.size());
    std::printf("\n");
    std::printf("Gate (revised session 2026-04-21 per user, supersedes 8 s\n");
    std::printf("  in spec.md L70): ≤ 12 s on M1 for 180-s clip.\n");
    std::printf("Gate result: %s (%.3f s vs 12.00 s budget).\n",
                totalWall <= 12.0 ? "PASS" : "FAIL",
                totalWall);

    return totalWall <= 12.0 ? 0 : 1;
}
