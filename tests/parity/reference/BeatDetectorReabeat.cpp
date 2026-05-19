// Copy of references/reabeat-template/src/BeatDetector.cpp with class/struct
// renamed; see BeatDetectorReabeat.h for policy.

#include "BeatDetectorReabeat.h"
#include "MelSpectrogramReabeat.h"
#include "PostprocessorReabeat.h"
#include "TempoEstimatorReabeat.h"
#include "DownbeatCleanerReabeat.h"
#include "TimeSigDetectorReabeat.h"
#include "BeatInterpolatorReabeat.h"
#include "OnsetRefinementReabeat.h"

#if REABEAT_HAS_ONNX
#include "InferenceProcessorReabeat.h"
#endif

#if REABEAT_WITH_REAPER_IO
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"
#endif

#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

BeatDetectorReabeat::BeatDetectorReabeat() = default;
BeatDetectorReabeat::~BeatDetectorReabeat() = default;

#if REABEAT_HAS_ONNX
BeatDetectorReabeat::BeatDetectorReabeat(Ort::Session& externalSession)
    : sessionPtr_(&externalSession)
{
    modelLoaded_ = true;
}
#endif

bool BeatDetectorReabeat::loadModel(const std::string& modelPath)
{
#if REABEAT_HAS_ONNX
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ReaBeat");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring widePath(modelPath.begin(), modelPath.end());
        ownedSession_ = std::make_unique<Ort::Session>(*env_, widePath.c_str(), opts);
#else
        ownedSession_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), opts);
#endif
        sessionPtr_ = ownedSession_.get();
        modelLoaded_ = true;
        return true;
    }
    catch (const Ort::Exception&)
    {
        modelLoaded_ = false;
        return false;
    }
#else
    (void)modelPath;
    return false;
#endif
}

bool BeatDetectorReabeat::isReady() const
{
    return modelLoaded_;
}

std::vector<float> BeatDetectorReabeat::resampleTo22050(const std::vector<float>& audio, int srcRate)
{
    if (srcRate == 22050)
        return audio;

    double ratio = 22050.0 / srcRate;
    auto outSize = static_cast<size_t>(std::ceil(audio.size() * ratio));
    std::vector<float> output(outSize);

    juce::LagrangeInterpolator interpolator;
    interpolator.reset();

    int numUsed = 0;
    interpolator.process(1.0 / ratio, audio.data(), output.data(),
                         static_cast<int>(outSize), static_cast<int>(audio.size()),
                         numUsed);

    return output;
}

float BeatDetectorReabeat::computeConfidence(const std::vector<float>& beats, float tempo)
{
    if (beats.size() < 4)
        return 0.5f;

    float expectedIbi = 60.0f / tempo;
    int consistent = 0;

    for (size_t i = 1; i < beats.size(); ++i)
    {
        float ibi = beats[i] - beats[i - 1];
        float deviation = std::abs(ibi - expectedIbi) / expectedIbi;
        if (deviation < 0.10f)
            ++consistent;
    }

    return std::min(1.0f, static_cast<float>(consistent) / static_cast<float>(beats.size() - 1));
}


DetectionResultReabeat BeatDetectorReabeat::detect(const std::vector<float>& audioMono,
                                                    int sampleRate,
                                                    std::function<void(const std::string&, float)> progressCb)
{
    DetectionResultReabeat result;
    auto t0 = std::chrono::steady_clock::now();

    if (!modelLoaded_)
    {
        result.error = "Model not loaded";
        return result;
    }

    if (progressCb) progressCb("Preparing audio...", 0.0f);

    result.duration = static_cast<float>(audioMono.size()) / sampleRate;
    if (result.duration < 2.0f)
    {
        result.error = "Audio too short (minimum 2 seconds)";
        return result;
    }

    double rmsTotal = 0;
    for (float s : audioMono) rmsTotal += s * s;
    rmsTotal = std::sqrt(rmsTotal / audioMono.size());
    if (rmsTotal < 0.001)
    {
        result.error = "Audio is silent";
        return result;
    }

    {
        int peaksPerSec = 100;
        int hop = std::max(1, sampleRate / peaksPerSec);
        int nFrames = static_cast<int>(audioMono.size()) / hop;
        result.peaks.resize(nFrames);

        for (int i = 0; i < nFrames; ++i)
        {
            float sum = 0;
            int base = i * hop;
            for (int j = 0; j < hop && base + j < static_cast<int>(audioMono.size()); ++j)
            {
                float s = audioMono[base + j];
                sum += s * s;
            }
            result.peaks[i] = std::sqrt(sum / hop);
        }

        if (!result.peaks.empty())
        {
            auto sorted = result.peaks;
            std::sort(sorted.begin(), sorted.end());
            float p98 = sorted[static_cast<size_t>(sorted.size() * 0.98)];
            if (p98 > 0)
                for (auto& p : result.peaks)
                    p = std::min(1.0f, p / p98);
        }
    }

    auto audio22k = resampleTo22050(audioMono, sampleRate);

    if (progressCb) progressCb("Computing spectrogram...", 0.1f);

    MelSpectrogramReabeat mel;
    auto spectrogram = mel.compute(audio22k);
    if (spectrogram.empty())
    {
        result.error = "Failed to compute spectrogram";
        return result;
    }
    if (!spectrogram.empty() && !spectrogram[0].empty())
    {
        float v = spectrogram[0][0];
        if (std::isnan(v) || std::isinf(v))
        {
            result.error = "Spectrogram contains NaN/Inf values";
            return result;
        }
    }

#if REABEAT_HAS_ONNX
    if (progressCb) progressCb("Running neural network...", 0.2f);

    InferenceProcessorReabeat inference(*sessionPtr_);
    try {
    auto [beatLogits, downbeatLogits] = inference.process(spectrogram,
        [&](float frac) {
            if (progressCb)
                progressCb("Running neural network...", 0.2f + frac * 0.5f);
        });
    if (!beatLogits.empty() && (std::isnan(beatLogits[0]) || std::isinf(beatLogits[0])))
    {
        result.error = "Neural network output contains NaN/Inf values";
        return result;
    }

    if (progressCb) progressCb("Postprocessing...", 0.75f);

    PostprocessorReabeat postproc(50.0f);
    auto ppResult = postproc.process(beatLogits, downbeatLogits);

    if (ppResult.beatTimes.size() < 2)
    {
        result.error = "Not enough beats detected";
        return result;
    }

    auto interpolatedBeats = BeatInterpolatorReabeat::interpolate(
        ppResult.beatTimes, ppResult.beatLogits, 50.0f);

    if (interpolatedBeats.size() >= 5)
    {
        std::vector<float> ivals;
        ivals.reserve(interpolatedBeats.size() - 1);
        for (size_t k = 1; k < interpolatedBeats.size(); ++k)
            ivals.push_back(interpolatedBeats[k] - interpolatedBeats[k - 1]);
        auto sortedIvals = ivals;
        std::sort(sortedIvals.begin(), sortedIvals.end());
        float medianIval = sortedIvals[sortedIvals.size() / 2];

        std::vector<float> consistent;
        consistent.push_back(interpolatedBeats[0]);
        for (size_t k = 1; k + 1 < interpolatedBeats.size(); ++k)
        {
            float prevGap = interpolatedBeats[k] - interpolatedBeats[k - 1];
            float nextGap = interpolatedBeats[k + 1] - interpolatedBeats[k];
            float prevDev = std::abs(prevGap - medianIval) / medianIval;
            float nextDev = std::abs(nextGap - medianIval) / medianIval;
            if (prevDev > 0.25f && nextDev > 0.25f)
                continue;
            consistent.push_back(interpolatedBeats[k]);
        }
        consistent.push_back(interpolatedBeats.back());
        interpolatedBeats = std::move(consistent);
    }

    if (progressCb) progressCb("Refining to transients...", 0.80f);

    auto refinedBeats = OnsetRefinementReabeat::refine(audio22k, 22050, interpolatedBeats);
    auto rawDownbeats = OnsetRefinementReabeat::refine(audio22k, 22050, ppResult.downbeatTimes);

    result.beats = refinedBeats;

    if (progressCb) progressCb("Computing tempo...", 0.85f);

    result.tempo = TempoEstimatorReabeat::compute(result.beats);

    if (rawDownbeats.size() >= 2)
    {
        result.timeSigNum = TimeSigDetectorReabeat::detect(result.beats, rawDownbeats);
        result.timeSigDenom = 4;
        result.downbeats = DownbeatCleanerReabeat::clean(rawDownbeats, result.tempo, result.timeSigNum);
    }
    else
    {
        result.timeSigNum = 4;
        result.timeSigDenom = 4;
        for (size_t i = 0; i < result.beats.size(); i += result.timeSigNum)
            result.downbeats.push_back(result.beats[i]);
    }

    result.confidence = computeConfidence(result.beats, result.tempo);

    auto t1 = std::chrono::steady_clock::now();
    result.detectionTime = std::chrono::duration<float>(t1 - t0).count();

    if (progressCb) progressCb("Done", 1.0f);
    } catch (const std::exception& e) {
        result.error = std::string("Detection failed: ") + e.what();
    } catch (...) {
        result.error = "Detection failed: unknown error";
    }
#else
    result.error = "ONNX Runtime not available";
#endif

    return result;
}

#if REABEAT_WITH_REAPER_IO
DetectionResultReabeat BeatDetectorReabeat::detectFile(const std::string& filePath,
                                                       std::function<void(const std::string&, float)> progressCb)
{
    if (!PCM_Source_CreateFromFile)
    {
        DetectionResultReabeat result;
        result.error = "REAPER API not available";
        return result;
    }

    PCM_source* source = PCM_Source_CreateFromFile(filePath.c_str());
    if (!source)
    {
        DetectionResultReabeat result;
        result.error = "Cannot read audio file: " + filePath;
        return result;
    }

    auto sampleRate = static_cast<int>(source->GetSampleRate());
    auto numChannels = source->GetNumChannels();
    auto lengthSec = source->GetLength();
    auto numSamples = static_cast<int>(lengthSec * sampleRate);

    if (sampleRate < 1 || numSamples < 1)
    {
        delete source;
        DetectionResultReabeat result;
        result.error = "Invalid audio source (MIDI or empty)";
        return result;
    }

    std::vector<ReaSample> buffer(static_cast<size_t>(numSamples) * numChannels);

    PCM_source_transfer_t transfer = {};
    transfer.time_s = 0.0;
    transfer.samplerate = sampleRate;
    transfer.nch = numChannels;
    transfer.length = numSamples;
    transfer.samples = buffer.data();

    source->GetSamples(&transfer);
    delete source;

    int samplesRead = transfer.samples_out;
    if (samplesRead < 1)
    {
        DetectionResultReabeat result;
        result.error = "Failed to read audio samples";
        return result;
    }

    std::vector<float> mono(samplesRead);
    if (numChannels == 1)
    {
        for (int i = 0; i < samplesRead; ++i)
            mono[i] = static_cast<float>(buffer[i]);
    }
    else
    {
        for (int i = 0; i < samplesRead; ++i)
        {
            double sum = 0;
            for (int ch = 0; ch < numChannels; ++ch)
                sum += buffer[static_cast<size_t>(i) * numChannels + ch];
            mono[i] = static_cast<float>(sum / numChannels);
        }
    }

    return detect(mono, sampleRate, progressCb);
}
#endif // REABEAT_WITH_REAPER_IO
