#pragma once

#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "AnalysisBundle.h"

// AnalyzePipeline — runs phase-1..phase-4-first-half on a background
// juce::Thread, producing an AnalysisBundle that RemixPipeline consumes.
//
// Stages owned by this worker (per ADR-047 § 2):
//   1. Load audio  (juce::AudioFormatManager + AudioLoader::resample)
//   2. BeatDetector::detect(mono_22050, 22050)
//   3. FeatureExtractor::extract(mono_22050, 22050, beats, mode)
//   4. StructureAnalyzer::analyze(...)        — SKIPPED on auto path (ADR-044)
//   5. computeTransitionCosts(features + segments + ...)
//
// All stages run sequentially. Progress callbacks marshal to the message
// thread via juce::MessageManager::callAsync; the completion callback fires
// on the message thread with std::shared_ptr<AnalysisBundle> on success or
// a populated error string on failure.
//
// Cancellation: threadShouldExit() polled between stages; BeatDetector
// (~2 s of ONNX inference) cannot be interrupted mid-call but its outer
// progress callback is. `MainComponent` uses the same zombie-reaping
// pattern as the deleted AnalyzeWorker (DEV-015 fix preserved): a
// superseded pipeline moves into `stoppingWorkers_` and the dtor waits.

namespace reamix {
    class BeatDetector;
}

namespace reamix::ui
{

class AnalyzePipeline : public juce::Thread
{
public:
    struct Input
    {
        juce::String sourcePath;
    };

    using ProgressCb = std::function<void (juce::String label, double p01)>;
    using CompleteCb = std::function<void (AnalysisBundlePtr bundle, juce::String error)>;

    AnalyzePipeline (Input                in,
                     reamix::BeatDetector& beatDetector,
                     ProgressCb           onProgress,
                     CompleteCb           onComplete);

    ~AnalyzePipeline() override;

    void run() override;

    // Lock-free signal that the holder of this worker is about to discard
    // it (item switch, fast-toggle). Lambdas posted to the message thread
    // capture a shared_ptr<atomic<bool>> and bail out if it is set, so
    // late-completion cannot dereference a destroyed MainComponent.
    std::shared_ptr<std::atomic<bool>> aliveFlag() { return alive_; }

private:
    void postProgress (const juce::String& step, double p01);
    void postCompletion (AnalysisBundlePtr bundle, const juce::String& error);

    struct LoadedAudio
    {
        std::vector<float> stereoNative;
        int                nChannels     { 0 };
        int                nativeSr      { 0 };
        std::size_t        nativeSamples { 0 };
        std::vector<float> mono22050;
    };
    bool loadAudio (const juce::String& path, LoadedAudio& out, juce::String& err);

    Input                  in_;
    reamix::BeatDetector&  beatDetector_;
    ProgressCb             onProgress_;
    CompleteCb             onComplete_;
    std::shared_ptr<std::atomic<bool>> alive_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnalyzePipeline)
};

} // namespace reamix::ui
