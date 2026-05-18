#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <functional>
#include <memory>

#include "WaveformView.h"

// FilePeaksProvider — reads peaks directly from the current item's audio
// file via juce::AudioFormatManager. Replaces SyntheticPeaksProvider from
// step 2 (DEV-002).
//
// ADR-037 spec: source-path change invalidates the cached reader + file;
// getPeakBlock() reads a lazy coarse peak summary on first call per path,
// then services bin queries from that summary.
//
// Design:
//   - One file open at a time; setSourcePath() replaces + invalidates.
//   - Coarse summary (N = max 8192 bins) cached in-memory per path; chunked
//     readMaxLevels reads source min/max per bin once. Subsequent
//     getPeakBlock() downsamples the coarse summary to the requested bin
//     count via cheap nearest-sample pick.
//   - Revision bumps each time path changes so WaveformView drops its
//     downstream PeaksCache.
//
// DEV-024 (sesja 95 ADR-085) — peaks build is now async on a background
// thread. setSourcePath() does the cheap metadata read (audio file header
// → durationSec + sampleRate; <10 ms typical) on the message thread to
// preserve DEV-012 (waveform view-range needs duration before first paint),
// then kicks a background worker that runs the heavier 8192-bin coarse
// summary build (50-200 ms typical for FLAC; was the 50-150 ms item-switch
// lag user reported sesja 49). getPeakBlock() returns zero peaks until
// coarseReady_ flips true; an optional onPeaksReady callback fires on the
// message thread when the summary is ready so WaveformView can repaint.
//
// Thread-safety: setSourcePath() + getPeakBlock() must be called on the
// message thread. Background peaks worker writes coarseMin_/coarseMax_
// THEN sets coarseReady_=true (release semantics); message-thread reads of
// coarseReady_ act as acquire — vectors are fully visible when flag is
// true. setSourcePath waits for any in-flight worker to exit before
// mutating member state (signalThreadShouldExit + abortRebuild_ flag
// trigger fast-bail inside the inner 8192-bin loop).

namespace reamix::ui
{

class FilePeaksProvider : public PeaksProvider
{
public:
    FilePeaksProvider();
    ~FilePeaksProvider() override;

    // Empty path clears the provider (returns 0 duration + empty bins).
    // Non-empty path triggers sync metadata read (duration available
    // immediately) + async background peaks build.
    void setSourcePath (const juce::String& absPath);

    // DEV-024 (sesja 95) — register a callback fired on the message thread
    // when async peaks build completes. Caller (MainComponent) typically
    // wires this to waveformView_.repaint() so the silhouette appears
    // without further user interaction once peaks are ready.
    using OnPeaksReady = std::function<void()>;
    void setOnPeaksReady (OnPeaksReady cb);

    double       getTotalDurationSeconds() const override;
    void         getPeakBlock (double startSec, double endSec, int nPx,
                               float* minOut, float* maxOut) override;
    std::int64_t getRevision() const override { return revision_.load(); }

private:
    class PeaksWorker;

    void rebuildSummaryIfNeeded();
    void loadReader();
    void cancelInFlightWorker();
    void startBackgroundRebuild();

    static constexpr int kCoarseBins = 8192;

    juce::AudioFormatManager formatManager_;
    juce::String             sourcePath_;
    std::unique_ptr<juce::AudioFormatReader> reader_;
    double   durationSec_ {0.0};

    // Coarse summary covering [0, durationSec_] across kCoarseBins bins.
    std::vector<float> coarseMin_;
    std::vector<float> coarseMax_;
    std::atomic<bool>  coarseReady_ {false};

    std::atomic<std::int64_t> revision_ {0};

    // DEV-024 (sesja 95) — background peaks build coordination.
    std::unique_ptr<PeaksWorker> peaksWorker_;
    std::atomic<bool>            abortRebuild_ {false};
    OnPeaksReady                 onPeaksReady_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FilePeaksProvider)
};

} // namespace reamix::ui
