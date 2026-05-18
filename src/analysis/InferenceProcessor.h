#pragma once

#if REAMIX_HAS_ONNX

#include <functional>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <vector>

namespace reamix {

// ONNX model inference with chunking for beat detection.
// Direct port of REABeat's InferenceProcessor (foundation: beat_this_cpp, MIT)
// with two inherited fixes:
//   - splitPiece returns `starts` alongside chunks (upstream computed them twice).
//   - progress callback for long files.
// Chunking: 1500-frame windows, 6-frame border overlap, keep_first aggregation.
class InferenceProcessor
{
public:
    explicit InferenceProcessor(Ort::Session& session);

    // Process full spectrogram, return {beat_logits, downbeat_logits}.
    std::pair<std::vector<float>, std::vector<float>>
    process(const std::vector<std::vector<float>>& spectrogram,
            std::function<void(float)> progressCb = nullptr);

private:
    Ort::Session& session_;

    static constexpr int kChunkSize  = 1500;  // 30 s at 50 fps
    static constexpr int kBorderSize = 6;     // 120 ms overlap

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

} // namespace reamix

#endif // REAMIX_HAS_ONNX
