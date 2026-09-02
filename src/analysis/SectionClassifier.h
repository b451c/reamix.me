#pragma once

#include <memory>
#include <string>
#include <vector>

#if REAMIX_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

// LinkSeg section classifier (Buisson, McFee, Essid, ISMIR 2024) — functional
// section boundaries + labels from the full mix and the beat grid. ADR-115
// Block Assembly base layer (DEV-098, sesja 121): the Blocks tab shows the
// model's sections as clickable blocks; auto mode takes the model output
// as-is (user decision sesja 118).
//
// Model: ONNX export of the upstream 9-class checkpoint (7 classes for the
// Hugging Face `elicwhite/linkseg-7c-onnx` file), input `mel` (N, 1, 64, 64)
// float32, outputs `bound` (N-1) sigmoid boundary activations and `label`
// (N, C) logits. The mel front-end and the decoder are C++-canonical ports
// of tools/dev/section_eval/linkseg_infer.py (itself a port of the
// LinkSeg-web TypeScript, validated byte-exact vs the PyTorch model):
//
//   beats -> requantise to 256-sample frames (unique, frame 0 added, stride-2
//            until <= 1500)
//         -> one 16382-sample window centred on each beat (edge-padded 22050 Hz
//            mono) -> STFT n_fft 1024 hop 256 periodic Hann reflect-centre,
//            power spectrum, 64 HTK mel bands (no norm), 10*log10(max(x,1e-10))
//         -> ONNX -> peak-picked boundaries (8 s neighbourhood) + majority-vote
//            labels per segment.
//
// Class index order (LinkSeg data_utils.indices_9classes):
//   0 silence, 1 verse, 2 chorus, 3 intro, 4 outro, 5 inst, 6 bridge,
//   7 pre-chorus, 8 post-chorus (7-class models stop at 6).

namespace reamix::analysis {

struct SectionModelOutput
{
    std::vector<double> beatTimes;    // the requantised beats the model saw (N)
    std::vector<float>  bound;        // N-1
    std::vector<float>  labelLogits;  // N * nClasses, row-major
    int                 nClasses = 0;
};

struct SectionSegment
{
    double      start      = 0.0;
    double      end        = 0.0;
    int         cls        = 0;
    std::string label;
    double      confidence = 0.0;     // mean softmax probability of `cls` over the segment's beats
};

class SectionClassifier
{
public:
    static constexpr int kSampleRate  = 22050;
    static constexpr int kHop         = 256;
    static constexpr int kMaxBeats    = 1500;
    static constexpr int kPad         = (kHop * 64 - 2) / 2;   // 8191
    static constexpr int kWindow      = 2 * kPad;              // 16382 samples
    static constexpr int kMelBands    = 64;
    static constexpr int kMelFrames   = 64;
    static constexpr int kMinBeats    = 4;

    SectionClassifier();
    ~SectionClassifier();

    bool loadModel (const std::string& modelPath);
    bool isLoaded() const { return loaded_; }

    // Full pipeline on 22050 Hz mono. Empty output + `err` on failure or when
    // fewer than kMinBeats beats are given.
    SectionModelOutput run (const float* mono22k, std::size_t numSamples,
                            const std::vector<double>& beatTimes,
                            std::string* err = nullptr);

    // Front-end pieces (static, testable without ORT).
    static std::vector<double> requantiseBeats (const std::vector<double>& beatTimes);
    // (N * 64 * 64) floats, layout [n][mel][frame].
    static std::vector<float> melWindows (const float* mono22k, std::size_t numSamples,
                                          const std::vector<double>& beatTimes);

    // Decoder: peak-pick `bound` with a +-windowSec neighbourhood, majority
    // vote of argmax labels per segment. Segments cover [0, durationSec].
    static std::vector<SectionSegment> decode (const SectionModelOutput& out,
                                               double durationSec,
                                               double windowSec = 8.0,
                                               double tau = 0.0);

    static const char* labelName (int cls);

private:
    bool loaded_ = false;
#if REAMIX_HAS_ONNX
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
#endif
};

} // namespace reamix::analysis
