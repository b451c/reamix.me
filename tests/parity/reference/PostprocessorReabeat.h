// Copy of references/reabeat-template/src/Postprocessor.h with class renamed
// to PostprocessorReabeat so it can be compiled alongside reamix::Postprocessor
// in the same parity-test binary. Do not edit — if REABeat's reference changes,
// re-copy the file and re-apply the rename.
#pragma once
#include <vector>

// Convert neural network logits to beat/downbeat timestamps.
// Port of beat_this_cpp Postprocessor with fixes:
// - FIXED: max_pool applied separately to beat and downbeat logits
//   (original concatenated them, causing cross-boundary interference)
// - Added raw logit output for BeatInterpolator

struct PostprocessResultReabeat
{
    std::vector<float> beatTimes;
    std::vector<float> downbeatTimes;
    std::vector<float> beatLogits;     // raw logits for gap-filling
    std::vector<float> downbeatLogits; // raw logits for gap-filling
};

class PostprocessorReabeat
{
public:
    explicit PostprocessorReabeat(float fps = 50.0f);

    PostprocessResultReabeat process(const std::vector<float>& beatLogits,
                                     const std::vector<float>& downbeatLogits);

private:
    float fps_;

    // Detect peaks: local max in max_pool window AND logit > 0
    std::vector<int> detectPeaks(const std::vector<float>& logits);

    // Merge peaks within 'width' frames using running mean
    std::vector<int> deduplicatePeaks(const std::vector<int>& peaks, int width);

    // 1D max pooling with same-padding
    std::vector<float> maxPool1d(const std::vector<float>& input, int kernelSize);

    static constexpr int kMaxPoolKernel = 7;
    static constexpr int kDedupWidth = 1;
};
