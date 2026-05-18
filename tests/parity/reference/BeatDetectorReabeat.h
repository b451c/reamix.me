// Copy of references/reabeat-template/src/BeatDetector.h with class/struct
// renamed (BeatDetector -> BeatDetectorReabeat, DetectionResult ->
// DetectionResultReabeat) so it can be compiled alongside
// reamix::BeatDetector in the same parity-test binary. Do not edit except
// to re-apply the rename after an upstream sync.
//
// Also modified (clearly marked with "reamix test-only"): a second
// constructor that accepts an externally-owned Ort::Session, to enable
// shared-session parity testing (same model weights, same kernel
// invocations — the only variation between reamix and reabeat is the C++
// code on either side).
//
// REABEAT_HAS_ONNX / REABEAT_WITH_REAPER_IO guards are preserved
// verbatim from upstream; the test target defines REABEAT_HAS_ONNX=1 and
// leaves REABEAT_WITH_REAPER_IO undefined (detectFile is plugin-only).
#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>

#if REABEAT_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

struct DetectionResultReabeat
{
    std::vector<float> beats;
    std::vector<float> downbeats;
    float tempo = 0.0f;
    int timeSigNum = 4;
    int timeSigDenom = 4;
    float confidence = 0.0f;
    float duration = 0.0f;
    std::vector<float> peaks;
    std::string error;
    float detectionTime = 0.0f;
};

class BeatDetectorReabeat
{
public:
    BeatDetectorReabeat();
    ~BeatDetectorReabeat();

#if REABEAT_HAS_ONNX
    // reamix test-only: inject external Ort::Session for shared-session parity.
    explicit BeatDetectorReabeat(Ort::Session& externalSession);
#endif

    bool loadModel(const std::string& modelPath);
    bool isReady() const;

    DetectionResultReabeat detect(const std::vector<float>& audioMono,
                                  int sampleRate,
                                  std::function<void(const std::string&, float)> progressCb = nullptr);

#if REABEAT_WITH_REAPER_IO
    DetectionResultReabeat detectFile(const std::string& filePath,
                                      std::function<void(const std::string&, float)> progressCb = nullptr);
#endif

private:
    std::vector<float> resampleTo22050(const std::vector<float>& audio, int srcRate);
    float computeConfidence(const std::vector<float>& beats, float tempo);

#if REABEAT_HAS_ONNX
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> ownedSession_;
    Ort::Session*                 sessionPtr_ = nullptr;
#endif
    bool modelLoaded_ = false;
};
