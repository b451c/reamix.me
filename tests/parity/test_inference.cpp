// Parity test: reamix::InferenceProcessor must produce byte-identical
// output to REABeat's reference InferenceProcessor for the same input
// spectrogram when both share one Ort::Session.
//
// Success criterion: max |Δ| == 0.0 across beat_logits and downbeat_logits.
// ORT CPU EP is documented to be run-to-run deterministic on one machine
// (github.com/microsoft/onnxruntime#4611); we additionally pin
// intra_op_num_threads=1, inter_op_num_threads=1, and
// session.use_deterministic_compute=1 in the TEST ONLY to eliminate the
// last degree of freedom (thread-count differences across machines).
// Production code keeps ORT defaults.
//
// Two test cases exercise both chunking paths:
//   A) short input — 101 frames (single-chunk path; no overlap aggregation).
//   B) long input  — 5000 frames (multi-chunk path with keep_first overlap).
//
// Model discovery: in order,
//   1. argv[1], if passed.
//   2. env var REAMIX_TEST_ONNX_MODEL.
//   3. ~/.reabeat/models/beat_this_final0.onnx (REABeat's cache).
// If none exist, prints SKIPPED and exits 0 so ctest marks the test as passed.

#include "analysis/InferenceProcessor.h"
#include "analysis/MelSpectrogram.h"
#include "InferenceProcessorReabeat.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace {

std::vector<float> makeSine(int sampleRate, float freqHz, float durationSec)
{
    int n = static_cast<int>(sampleRate * durationSec);
    std::vector<float> out(n);
    const float w = 2.0f * static_cast<float>(std::numbers::pi) * freqHz
                    / static_cast<float>(sampleRate);
    for (int i = 0; i < n; ++i)
        out[i] = 0.5f * std::sin(w * static_cast<float>(i));
    return out;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}

std::string resolveModelPath(int argc, char** argv)
{
    if (argc > 1 && std::filesystem::exists(argv[1]))
        return argv[1];

    if (const char* env = std::getenv("REAMIX_TEST_ONNX_MODEL"))
    {
        if (std::filesystem::exists(env))
            return env;
    }

    if (const char* home = std::getenv("HOME"))
    {
        std::filesystem::path p = std::filesystem::path(home)
            / ".reabeat" / "models" / "beat_this_final0.onnx";
        if (std::filesystem::exists(p))
            return p.string();
    }

    return {};
}

bool runCase(const char* label,
             Ort::Session& session,
             const std::vector<std::vector<float>>& spect)
{
    std::printf("[%s] frames=%zu mels=%zu\n", label, spect.size(),
                spect.empty() ? 0 : spect[0].size());

    reamix::InferenceProcessor ours(session);
    InferenceProcessorReabeat   ref(session);

    auto [beatOurs, downOurs] = ours.process(spect);
    auto [beatRef,  downRef]  = ref.process(spect);

    double beatDiff = maxAbsDiff(beatOurs, beatRef);
    double downDiff = maxAbsDiff(downOurs, downRef);

    std::printf("[%s] beat_max_abs_diff = %.17g, downbeat_max_abs_diff = %.17g\n",
                label, beatDiff, downDiff);

    if (beatOurs.size() != beatRef.size() || downOurs.size() != downRef.size())
    {
        std::fprintf(stderr, "[%s] FAIL: output length mismatch "
                             "(ours beat=%zu down=%zu, ref beat=%zu down=%zu)\n",
                     label, beatOurs.size(), downOurs.size(),
                     beatRef.size(),  downRef.size());
        return false;
    }

    if (beatDiff != 0.0 || downDiff != 0.0)
    {
        std::fprintf(stderr, "[%s] FAIL: non-zero diff (beat=%.17g, down=%.17g)\n",
                     label, beatDiff, downDiff);
        return false;
    }

    std::printf("[%s] PASS: byte-identical\n", label);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string modelPath = resolveModelPath(argc, argv);
    if (modelPath.empty())
    {
        std::printf("SKIPPED: no ONNX model available.\n"
                    "  Set REAMIX_TEST_ONNX_MODEL=<path-to>/beat_this_final0.onnx,\n"
                    "  pass the path as argv[1], or place it at\n"
                    "  $HOME/.reabeat/models/beat_this_final0.onnx.\n");
        return 0;
    }
    std::printf("using model: %s\n", modelPath.c_str());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "reamix_test_inference");

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.AddConfigEntry("session.use_deterministic_compute", "1");

#ifdef _WIN32
    // ORT 1.24.4 on Windows uses ORTCHAR_T = wchar_t for the path argument.
    // Mirror BeatDetector::loadModel pattern (src/analysis/BeatDetector.cpp:50-55).
    std::wstring widePath(modelPath.begin(), modelPath.end());
    Ort::Session session(env, widePath.c_str(), opts);
#else
    Ort::Session session(env, modelPath.c_str(), opts);
#endif

    // Case A — short input: single-chunk path.
    // 2 s of 440 Hz sine @ 22050 Hz → 101 mel frames (same input as test_mel).
    reamix::MelSpectrogram mel;
    auto specShort = mel.compute(makeSine(22050, 440.0f, 2.0f));

    // Case B — long input: multi-chunk + overlap aggregation.
    // 40 s of 440 Hz sine → ~2000 frames → 2 chunks (step = 1488).
    auto specLong  = mel.compute(makeSine(22050, 440.0f, 40.0f));

    bool okShort = runCase("short", session, specShort);
    bool okLong  = runCase("long",  session, specLong);

    if (!(okShort && okLong))
        return 1;

    std::printf("PASS: InferenceProcessor byte-identical to REABeat reference "
                "on short+long inputs\n");
    return 0;
}
