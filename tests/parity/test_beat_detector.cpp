// End-to-end parity test: reamix::BeatDetector must produce a byte-identical
// DetectionResult to REABeat's reference BeatDetectorReabeat for the same
// input audio when both share one Ort::Session.
//
// Shared-session strategy matches test_inference.cpp: construct one
// deterministic Ort::Session (threads=1, use_deterministic_compute=1) and
// inject it into both detector instances via their test-only constructor.
// This eliminates the "two independent session loads might diverge" risk.
//
// Input: 10 s of synthetic click train at 120 BPM, sampled at 22050 Hz so
// resampleTo22050() short-circuits (LagrangeInterpolator not exercised here;
// it will be validated separately when AudioLoader lands). Exponential-decay
// impulses (~5 ms decay) produce clear spectral content for the ONNX model
// to find beats on, and RMS > 0.001 passes the silence guard.
//
// Compares every bit of DetectionResult that comes from pure computation:
//   beats, downbeats, tempo, timeSigNum, confidence, peaks, duration
// (detectionTime is elapsed wall-clock — excluded.)
//
// Model discovery (same as test_inference.cpp): argv[1] -> env
// REAMIX_TEST_ONNX_MODEL -> ~/.reabeat/models/beat_this_final0.onnx ->
// SKIPPED (exit 0) if nothing found.

#include "analysis/BeatDetector.h"
#include "BeatDetectorReabeat.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

// Click train: impulses at every (60/bpm) s, each an exponential-decay burst.
std::vector<float> makeClickTrain(int sampleRate, float bpm, float durationSec)
{
    int n = static_cast<int>(sampleRate * durationSec);
    std::vector<float> out(n, 0.0f);

    float period = 60.0f / bpm;  // seconds per beat
    float decayTau = 0.005f;     // 5 ms decay
    int decaySamples = static_cast<int>(decayTau * 8 * sampleRate);  // 8 tau ~ -70 dB

    for (float t = 0.0f; t < durationSec; t += period)
    {
        int start = static_cast<int>(t * sampleRate);
        for (int j = 0; j < decaySamples && start + j < n; ++j)
        {
            float env = std::exp(-static_cast<float>(j) / (decayTau * sampleRate));
            out[start + j] += 0.8f * env;
        }
    }
    return out;
}

bool bitEqualF(float a, float b)
{
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
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

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "reamix_test_beat_detector");

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

    // Case: 10 s of 120 BPM click train @ 22050 Hz (no resampling).
    const int sampleRate = 22050;
    const float bpm = 120.0f;
    const float durationSec = 10.0f;
    auto audio = makeClickTrain(sampleRate, bpm, durationSec);
    std::printf("[click_120bpm_10s] samples=%zu, sr=%d\n", audio.size(), sampleRate);

    reamix::BeatDetector   ours(session);
    BeatDetectorReabeat    ref(session);

    auto rOurs = ours.detect(audio, sampleRate);
    auto rRef  = ref.detect(audio, sampleRate);

    if (!rOurs.error.empty())
    {
        std::fprintf(stderr, "FAIL: reamix detect returned error: %s\n",
                     rOurs.error.c_str());
        return 1;
    }
    if (!rRef.error.empty())
    {
        std::fprintf(stderr, "FAIL: reabeat ref detect returned error: %s\n",
                     rRef.error.c_str());
        return 1;
    }

    double beatDiff  = maxAbsDiff(rOurs.beats,     rRef.beats);
    double downDiff  = maxAbsDiff(rOurs.downbeats, rRef.downbeats);
    double peaksDiff = maxAbsDiff(rOurs.peaks,     rRef.peaks);

    std::printf("  beats:      n=%zu vs %zu, max_abs_diff=%.17g\n",
                rOurs.beats.size(), rRef.beats.size(), beatDiff);
    std::printf("  downbeats:  n=%zu vs %zu, max_abs_diff=%.17g\n",
                rOurs.downbeats.size(), rRef.downbeats.size(), downDiff);
    std::printf("  peaks:      n=%zu vs %zu, max_abs_diff=%.17g\n",
                rOurs.peaks.size(), rRef.peaks.size(), peaksDiff);
    std::printf("  tempo:      %.9g vs %.9g\n",     rOurs.tempo,      rRef.tempo);
    std::printf("  timeSigNum: %d vs %d\n",          rOurs.timeSigNum, rRef.timeSigNum);
    std::printf("  confidence: %.9g vs %.9g\n",     rOurs.confidence, rRef.confidence);
    std::printf("  duration:   %.9g vs %.9g\n",     rOurs.duration,   rRef.duration);

    bool ok = true;

    if (rOurs.beats.size() != rRef.beats.size() || beatDiff != 0.0)
    {
        std::fprintf(stderr, "FAIL: beats diverge (max_abs_diff=%.17g)\n", beatDiff);
        ok = false;
    }
    if (rOurs.downbeats.size() != rRef.downbeats.size() || downDiff != 0.0)
    {
        std::fprintf(stderr, "FAIL: downbeats diverge (max_abs_diff=%.17g)\n", downDiff);
        ok = false;
    }
    if (rOurs.peaks.size() != rRef.peaks.size() || peaksDiff != 0.0)
    {
        std::fprintf(stderr, "FAIL: peaks diverge (max_abs_diff=%.17g)\n", peaksDiff);
        ok = false;
    }
    if (!bitEqualF(rOurs.tempo, rRef.tempo))
    {
        std::fprintf(stderr, "FAIL: tempo not bit-equal\n");
        ok = false;
    }
    if (rOurs.timeSigNum != rRef.timeSigNum)
    {
        std::fprintf(stderr, "FAIL: timeSigNum diverge (%d vs %d)\n",
                     rOurs.timeSigNum, rRef.timeSigNum);
        ok = false;
    }
    if (!bitEqualF(rOurs.confidence, rRef.confidence))
    {
        std::fprintf(stderr, "FAIL: confidence not bit-equal\n");
        ok = false;
    }
    if (!bitEqualF(rOurs.duration, rRef.duration))
    {
        std::fprintf(stderr, "FAIL: duration not bit-equal\n");
        ok = false;
    }

    if (!ok) return 1;

    std::printf("PASS: BeatDetector byte-identical to REABeat reference "
                "on click_120bpm_10s\n");
    return 0;
}
