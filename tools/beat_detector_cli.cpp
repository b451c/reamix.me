// Phase-1 closing-gate CLI runner.
//
// Loads an audio file, runs reamix::BeatDetector end-to-end, emits
// DetectionResult as JSON. Used by tools/closing_gate.py to diff
// against Python's references/python-source/analysis/beat_detector.py.
//
// Not shipped with the plugin; pure tooling target.

#include "analysis/BeatDetector.h"
#include "io/AudioLoader.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

juce::var floatsToVar(const std::vector<float>& v)
{
    juce::Array<juce::var> arr;
    arr.ensureStorageAllocated((int) v.size());
    for (float x : v)
        arr.add((double) x);
    return juce::var(std::move(arr));
}

int printUsage(const char* argv0)
{
    std::fprintf(stderr,
        "usage: %s --audio <path> --model <path.onnx> [--output <json-path>]\n",
        argv0);
    return 2;
}

} // namespace

int main(int argc, char** argv)
{
    std::string audioPath, modelPath, outputPath;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if      (a == "--audio"  && i + 1 < argc) audioPath  = argv[++i];
        else if (a == "--model"  && i + 1 < argc) modelPath  = argv[++i];
        else if (a == "--output" && i + 1 < argc) outputPath = argv[++i];
        else                                      return printUsage(argv[0]);
    }
    if (audioPath.empty() || modelPath.empty())
        return printUsage(argv[0]);

    juce::File audioFile{juce::String(audioPath)};
    int srcRate = 0;
    std::string err;

    auto audioOpt = reamix::AudioLoader::loadMono(audioFile, srcRate, err);
    if (!audioOpt)
    {
        std::fprintf(stderr, "AudioLoader::loadMono failed: %s\n", err.c_str());
        return 1;
    }

    reamix::BeatDetector detector;
    if (!detector.loadModel(modelPath))
    {
        std::fprintf(stderr, "BeatDetector::loadModel failed for %s\n", modelPath.c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto result = detector.detect(*audioOpt, srcRate);
    const auto t1 = std::chrono::steady_clock::now();

    if (!result.error.empty())
    {
        std::fprintf(stderr, "BeatDetector::detect failed: %s\n", result.error.c_str());
        return 1;
    }

    const double wallSec = std::chrono::duration<double>(t1 - t0).count();

    auto* obj = new juce::DynamicObject();
    obj->setProperty("backend",        juce::String("reamix-cpp"));
    obj->setProperty("audio_path",     juce::String(audioPath));
    obj->setProperty("source_rate",    srcRate);
    obj->setProperty("duration",       (double) result.duration);
    obj->setProperty("tempo",          (double) result.tempo);
    obj->setProperty("time_signature", result.timeSigNum);
    obj->setProperty("confidence",     (double) result.confidence);
    obj->setProperty("beats",          floatsToVar(result.beats));
    obj->setProperty("downbeats",      floatsToVar(result.downbeats));
    obj->setProperty("n_beats",        (int) result.beats.size());
    obj->setProperty("n_downbeats",    (int) result.downbeats.size());
    obj->setProperty("wall_seconds",   wallSec);

    juce::var root(obj);
    juce::String json = juce::JSON::toString(root, true);

    if (outputPath.empty())
    {
        std::cout << json << "\n";
    }
    else
    {
        juce::File out{juce::String(outputPath)};
        out.getParentDirectory().createDirectory();
        if (!out.replaceWithText(json))
        {
            std::fprintf(stderr, "Failed to write %s\n", outputPath.c_str());
            return 1;
        }
    }
    return 0;
}
