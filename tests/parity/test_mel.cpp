// Parity test: reamix::MelSpectrogram must produce byte-identical output
// to REABeat's reference MelSpectrogram for the same input.
//
// Success criterion (phase 1 spec): max |Δ| == 0.0 across all frames × mels.
// VALIDATION.md per-frame mel threshold is 1e-3, so bitwise is 1000x tighter.
//
// Input: synthetic deterministic signal (440 Hz sine @ 22050 Hz, 2 s).
// A tiny golden dump of 10 frames is also written to
// references/golden/phase-1/mel_sine440_2s.txt on first run, for regression
// tracking independent of the REABeat reference being compiled in.

#include "analysis/MelSpectrogram.h"
#include "MelSpectrogramReabeat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

namespace {

std::vector<float> makeSine(int sampleRate, float freqHz, float durationSec)
{
    int n = static_cast<int>(sampleRate * durationSec);
    std::vector<float> out(n);
    const float w = 2.0f * static_cast<float>(std::numbers::pi) * freqHz / static_cast<float>(sampleRate);
    for (int i = 0; i < n; ++i)
        out[i] = 0.5f * std::sin(w * static_cast<float>(i));
    return out;
}

double maxAbsDiff(const std::vector<std::vector<float>>& a,
                  const std::vector<std::vector<float>>& b)
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].size() != b[i].size()) return std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < a[i].size(); ++j)
        {
            double d = std::abs(static_cast<double>(a[i][j]) - static_cast<double>(b[i][j]));
            if (d > m) m = d;
        }
    }
    return m;
}

void writeGoldenDump(const std::vector<std::vector<float>>& mel,
                     const std::string& path,
                     int framesToDump = 10)
{
    std::filesystem::path p(path);
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(path);
    int n = std::min<int>(framesToDump, static_cast<int>(mel.size()));
    f << "# mel_sine440_2s : first " << n << " frames, " << (mel.empty() ? 0 : mel[0].size()) << " mels each\n";
    f << "# source: reamix::MelSpectrogram on 2 s of 440 Hz sine @ 22050 Hz, amplitude 0.5\n";
    f.setf(std::ios::fixed);
    f.precision(9);
    for (int i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < mel[i].size(); ++j)
        {
            f << mel[i][j];
            f << (j + 1 == mel[i].size() ? '\n' : ' ');
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    const std::string goldenPath = (argc > 1)
        ? argv[1]
        : "references/golden/phase-1/mel_sine440_2s.txt";

    auto audio = makeSine(22050, 440.0f, 2.0f);

    reamix::MelSpectrogram ours;
    MelSpectrogramReabeat ref;

    auto melOurs = ours.compute(audio);
    auto melRef  = ref.compute(audio);

    std::printf("frames ours=%zu, ref=%zu, mels=%zu\n",
                melOurs.size(),
                melRef.size(),
                melOurs.empty() ? 0 : melOurs[0].size());

    double md = maxAbsDiff(melOurs, melRef);
    std::printf("max_abs_diff = %.17g\n", md);

    writeGoldenDump(melOurs, goldenPath, /*framesToDump=*/10);
    std::printf("wrote golden dump -> %s\n", goldenPath.c_str());

    // Phase-1 spec: byte-identical vs REABeat reference.
    if (md != 0.0)
    {
        std::fprintf(stderr, "FAIL: output diverges from REABeat reference (max_abs_diff=%.17g)\n", md);
        return 1;
    }

    std::printf("PASS: byte-identical to REABeat reference\n");
    return 0;
}
