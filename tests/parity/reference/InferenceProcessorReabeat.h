// Copy of references/reabeat-template/src/InferenceProcessor.h with class
// renamed to InferenceProcessorReabeat so it can be compiled alongside
// reamix::InferenceProcessor in the same parity-test binary. Do not edit —
// if REABeat's reference changes, re-copy the file and re-apply the rename.
//
// The upstream header gates everything on REABEAT_HAS_ONNX; we compile
// the test target with -DREABEAT_HAS_ONNX=1 so the guard is inert here.
#pragma once

#if REABEAT_HAS_ONNX

#include <vector>
#include <functional>
#include <onnxruntime_cxx_api.h>

class InferenceProcessorReabeat
{
public:
    explicit InferenceProcessorReabeat(Ort::Session& session);

    std::pair<std::vector<float>, std::vector<float>>
    process(const std::vector<std::vector<float>>& spectrogram,
            std::function<void(float)> progressCb = nullptr);

private:
    Ort::Session& session_;

    static constexpr int kChunkSize = 1500;
    static constexpr int kBorderSize = 6;

    struct SplitResult
    {
        std::vector<std::vector<std::vector<float>>> chunks;
        std::vector<int> starts;
    };

    SplitResult splitPiece(const std::vector<std::vector<float>>& spect);

    std::pair<std::vector<float>, std::vector<float>>
    runInference(const std::vector<std::vector<float>>& chunk);

    std::pair<std::vector<float>, std::vector<float>>
    aggregate(const std::vector<std::pair<std::vector<float>, std::vector<float>>>& predChunks,
              const std::vector<int>& starts,
              int fullSize);
};

#endif // REABEAT_HAS_ONNX
