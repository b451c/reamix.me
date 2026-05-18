#pragma once

#include <optional>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>

// Audio loading + resampling helpers used in dev/test paths.
//
// In production, BeatDetector::detectFile() reads audio via REAPER's
// PCM_Source decoders (format coverage for free — mp3/flac/ogg/aac/etc).
// This module provides a parallel path via juce::AudioFormatManager for
// environments without REAPER (parity tests, CLI tools, automated runs)
// and for the "Detect Beats" UI button's file-drag convenience case.
//
// resample() is a verbatim port of REABeat's resampleTo22050 (see
// references/reabeat-template/src/BeatDetector.cpp:61-80), generalized
// to accept an arbitrary destination rate.

namespace reamix {

class AudioLoader
{
public:
    // Decode audio file to mono float samples via juce::AudioFormatManager.
    // Registers basic formats (wav/aiff/flac/ogg — all JUCE static codecs).
    // Multi-channel input is mixed down to mono by channel-wise averaging.
    // Returns std::nullopt on error; outError contains a diagnostic string.
    // outSampleRate is the file's native rate (no resampling performed here —
    // caller decides whether to resample).
    static std::optional<std::vector<float>> loadMono(const juce::File& file,
                                                      int& outSampleRate,
                                                      std::string& outError);

    // Resample mono audio from srcRate to dstRate via juce::LagrangeInterpolator.
    // Returns the input unchanged if srcRate == dstRate (same short-circuit as
    // REABeat). Verbatim of references/reabeat-template/src/BeatDetector.cpp:61-80.
    static std::vector<float> resample(const std::vector<float>& audio,
                                       int srcRate,
                                       int dstRate);
};

} // namespace reamix
