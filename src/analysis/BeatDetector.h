#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>

#if REAMIX_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

// End-to-end beat/downbeat detection orchestrator.
// Wires: resample -> MelSpectrogram -> InferenceProcessor -> Postprocessor
//   -> BeatInterpolator -> beat-consistency pass (25% deviation filter)
//   -> OnsetRefinement -> TempoEstimator -> TimeSigDetector -> DownbeatCleaner
//   -> confidence heuristic.
// Source: references/reabeat-template/src/BeatDetector.{h,cpp}

namespace reamix {

struct DetectionResult
{
    std::vector<float> beats;
    std::vector<float> downbeats;
    float tempo = 0.0f;
    int timeSigNum = 4;
    int timeSigDenom = 4;
    float confidence = 0.0f;
    float duration = 0.0f;
    float detectionTime = 0.0f;
    std::vector<float> peaks;  // RMS waveform envelope (100 values/sec, 0-1 range)
    std::string error;  // non-empty on failure
};

class BeatDetector
{
public:
    BeatDetector();
    ~BeatDetector();

#if REAMIX_HAS_ONNX
    // Test-only constructor: inject an externally-owned Ort::Session so
    // the parity test can share one session between reamix::BeatDetector
    // and BeatDetectorReabeat. Not used in production (production code
    // constructs with the default ctor and calls loadModel()).
    explicit BeatDetector(Ort::Session& externalSession);
#endif

    // Load ONNX model. Returns false on failure.
    bool loadModel(const std::string& modelPath);

    // Is model loaded and ready?
    bool isReady() const;

    // Run detection on mono audio at given sample rate.
    DetectionResult detect(const std::vector<float>& audioMono,
                           int sampleRate,
                           std::function<void(const std::string&, float)> progressCb = nullptr);

#if REAMIX_WITH_REAPER_IO
    // Run detection on audio file path via REAPER's PCM_Source decoders.
    // Only available in plugin builds; excluded from test binaries.
    DetectionResult detectFile(const std::string& filePath,
                               std::function<void(const std::string&, float)> progressCb = nullptr);
#endif

private:
    std::vector<float> resampleTo22050(const std::vector<float>& audio, int srcRate);
    float computeConfidence(const std::vector<float>& beats, float tempo);

#if REAMIX_HAS_ONNX
    std::unique_ptr<Ort::Env>     env_;            // owned: only when loadModel() was called
    std::unique_ptr<Ort::Session> ownedSession_;   // owned: only when loadModel() was called
    Ort::Session*                 sessionPtr_ = nullptr;  // points to ownedSession_.get() OR external
#endif
    bool modelLoaded_ = false;
};

} // namespace reamix
