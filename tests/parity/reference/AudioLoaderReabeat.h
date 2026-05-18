// Reference copy of REABeat's resampling logic, extracted from
// references/reabeat-template/src/BeatDetector.cpp:61-80 as a free
// function. Used only by tests/parity/test_audio_loader.cpp to verify
// reamix::AudioLoader::resample() is bit-identical to the upstream
// computation.
//
// REABeat itself has no "AudioLoader" module — its production path goes
// through REAPER's PCM_Source decoders (see BeatDetector::detectFile).
// Resampling is inlined in BeatDetector::resampleTo22050 as a private
// member. This file is a minimal standalone reimplementation of that
// same function body, generalized from a hardcoded 22050 to an arbitrary
// dstRate, so the parity test can exercise non-trivial resampling
// ratios beyond 22050.
//
// Do not edit except to re-apply the source text after an upstream sync.
#pragma once

#include <vector>

namespace reabeat_ref {

std::vector<float> resample(const std::vector<float>& audio, int srcRate, int dstRate);

} // namespace reabeat_ref
