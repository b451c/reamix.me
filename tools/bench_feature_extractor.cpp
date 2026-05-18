// Ad-hoc bench: FeatureExtractor on a 3-minute clip (ROADMAP target
// ≤ 15 s on M1 CPU). Not wired into ctest — run once per session-end.
//
// Build: (from build/)
//   clang++ -O3 -std=c++17 -I../src -I../tests/parity/reference \
//     -I../vendor ../tools/bench_feature_extractor.cpp \
//     ../src/analysis/FeatureExtractor.cpp \
//     ../src/analysis/BeatSync.cpp \
//     ../src/analysis/BeatWindows.cpp \
//     ../src/analysis/Mfcc.cpp \
//     ../src/dsp/ChromaSTFT.cpp ../src/dsp/DCT.cpp \
//     ../src/dsp/MelSpectrogramLibrosa.cpp ../src/dsp/OnsetStrength.cpp \
//     ../src/dsp/PitchTrack.cpp ../src/dsp/RMS.cpp ../src/dsp/STFT.cpp \
//     ../src/dsp/SpectralCentroid.cpp ../src/dsp/SpectralContrast.cpp \
//     -o bench_feature_extractor

#include "analysis/FeatureExtractor.h"
#include "NpyIO.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const std::string path = (argc > 1) ? argv[1] : "/tmp/phase2_perf_audio.npy";
    const auto y = reamix::test::loadNpy1DFloat32(path);
    const int sr = 22050;
    const double durationSec = static_cast<double>(y.size()) / sr;
    std::printf("audio: %s  nSamples=%zu  duration=%.1fs\n",
                path.c_str(), y.size(), durationSec);

    // 120 BPM synthetic beats.
    std::vector<double> beatTimes;
    for (double t = 0.5; t < durationSec - 0.5; t += 0.5) beatTimes.push_back(t);
    std::printf("beats: %zu (120 BPM synthetic)\n", beatTimes.size());

    // Warmup + measure.
    using clk = std::chrono::high_resolution_clock;
    (void)reamix::analysis::FeatureExtractor::extract(
        y.data(), y.size(), sr, beatTimes);

    auto t0 = clk::now();
    auto r = reamix::analysis::FeatureExtractor::extract(
        y.data(), y.size(), sr, beatTimes);
    auto t1 = clk::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    std::printf("extract() wall: %.3f s  (features %dx%d)\n",
                wall, r.nBeats, r.nFeat);
    std::printf("target ≤ 15 s (ROADMAP loose budget, phase-2-features).\n");
    return wall <= 15.0 ? 0 : 1;
}
