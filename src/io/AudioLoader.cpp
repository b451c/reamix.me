#include "io/AudioLoader.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <cmath>
#include <memory>

namespace reamix {

std::optional<std::vector<float>> AudioLoader::loadMono(const juce::File& file,
                                                        int& outSampleRate,
                                                        std::string& outError)
{
    if (!file.existsAsFile())
    {
        outError = "File does not exist: " + file.getFullPathName().toStdString();
        return std::nullopt;
    }

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (!reader)
    {
        outError = "Unsupported or unreadable audio format: "
                   + file.getFullPathName().toStdString();
        return std::nullopt;
    }

    const auto numSamples  = static_cast<int>(reader->lengthInSamples);
    const auto numChannels = static_cast<int>(reader->numChannels);
    outSampleRate          = static_cast<int>(reader->sampleRate);

    if (numSamples < 1 || numChannels < 1 || outSampleRate < 1)
    {
        outError = "Invalid audio stream (empty or zero channels/rate)";
        return std::nullopt;
    }

    juce::AudioBuffer<float> buffer(numChannels, numSamples);
    if (!reader->read(&buffer, 0, numSamples, 0, true, true))
    {
        outError = "Failed to read audio samples";
        return std::nullopt;
    }

    std::vector<float> mono(static_cast<size_t>(numSamples));
    if (numChannels == 1)
    {
        const float* src = buffer.getReadPointer(0);
        std::copy(src, src + numSamples, mono.begin());
    }
    else
    {
        const float inv = 1.0f / static_cast<float>(numChannels);
        for (int i = 0; i < numSamples; ++i)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < numChannels; ++ch)
                sum += buffer.getReadPointer(ch)[i];
            mono[static_cast<size_t>(i)] = sum * inv;
        }
    }

    return mono;
}

std::vector<float> AudioLoader::resample(const std::vector<float>& audio,
                                         int srcRate,
                                         int dstRate)
{
    if (srcRate == dstRate)
        return audio;

    const double ratio = static_cast<double>(dstRate) / static_cast<double>(srcRate);
    const auto outSize = static_cast<size_t>(std::ceil(audio.size() * ratio));
    std::vector<float> output(outSize);

    juce::LagrangeInterpolator interpolator;
    interpolator.reset();

    int numUsed = 0;
    interpolator.process(1.0 / ratio,
                         audio.data(),
                         output.data(),
                         static_cast<int>(outSize),
                         static_cast<int>(audio.size()),
                         numUsed);

    return output;
}

} // namespace reamix
