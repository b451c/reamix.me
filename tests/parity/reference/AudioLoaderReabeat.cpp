// Reference: verbatim body of REABeat BeatDetector::resampleTo22050
// (references/reabeat-template/src/BeatDetector.cpp:61-80), with the
// hardcoded 22050 replaced by a `dstRate` parameter so the parity test
// can drive non-trivial source/destination combinations.
//
// Everything else — the ratio formula, ceil(outSize), LagrangeInterpolator
// construction, reset(), process() call order and arguments — must match
// the upstream text 1:1. Divergence here breaks the bitwise parity
// guarantee on reamix::AudioLoader::resample.

#include "AudioLoaderReabeat.h"

#include <juce_audio_basics/juce_audio_basics.h>

#include <cmath>

namespace reabeat_ref {

std::vector<float> resample(const std::vector<float>& audio, int srcRate, int dstRate)
{
    if (srcRate == dstRate)
        return audio;

    double ratio = static_cast<double>(dstRate) / srcRate;
    auto outSize = static_cast<size_t>(std::ceil(audio.size() * ratio));
    std::vector<float> output(outSize);

    // JUCE LagrangeInterpolator for high-quality resampling
    juce::LagrangeInterpolator interpolator;
    interpolator.reset();

    int numUsed = 0;
    interpolator.process(1.0 / ratio, audio.data(), output.data(),
                         static_cast<int>(outSize), static_cast<int>(audio.size()),
                         numUsed);

    return output;
}

} // namespace reabeat_ref
