#include "ui/FilePeaksProvider.h"

#include <algorithm>
#include <cmath>

namespace reamix::ui
{

// DEV-024 (sesja 95 ADR-085) — background worker that runs the 8192-bin
// coarse summary build off the message thread. setSourcePath() launches
// one of these per source-path change; abortRebuild_ + threadShouldExit()
// inside the inner loop guarantee a fast bail when the user rapidly
// switches items. On successful completion (no abort), posts a repaint
// trigger via MessageManager::callAsync so WaveformView paints the
// silhouette without further user interaction.
class FilePeaksProvider::PeaksWorker : public juce::Thread
{
public:
    explicit PeaksWorker (FilePeaksProvider& owner)
        : juce::Thread ("ReamixPeaksRebuild"), owner_ (owner) {}

    void run() override
    {
        owner_.rebuildSummaryIfNeeded();

        if (! threadShouldExit()
            && ! owner_.abortRebuild_.load (std::memory_order_acquire)
            && owner_.coarseReady_.load (std::memory_order_acquire)
            && owner_.onPeaksReady_)
        {
            // Capture by value so the callback survives even if the worker
            // (and its owner reference) outlives the next setSourcePath
            // mutation. onPeaksReady_ is set once at MainComponent ctor and
            // not mutated thereafter — safe to capture by value here.
            auto cb = owner_.onPeaksReady_;
            juce::MessageManager::callAsync ([cb] { if (cb) cb(); });
        }
    }

private:
    FilePeaksProvider& owner_;
};

FilePeaksProvider::FilePeaksProvider()
{
    formatManager_.registerBasicFormats();
}

FilePeaksProvider::~FilePeaksProvider()
{
    cancelInFlightWorker();
}

void FilePeaksProvider::setOnPeaksReady (OnPeaksReady cb)
{
    onPeaksReady_ = std::move (cb);
}

void FilePeaksProvider::setSourcePath (const juce::String& absPath)
{
    if (absPath == sourcePath_)
        return;

    // DEV-024 (sesja 95) — wait for any in-flight peaks worker to exit
    // BEFORE mutating shared state. Otherwise the worker could write to
    // coarseMin_/coarseMax_ (or read reader_) while we clear them — UB.
    cancelInFlightWorker();

    sourcePath_  = absPath;
    reader_.reset();
    coarseMin_.clear();
    coarseMax_.clear();
    coarseReady_.store (false, std::memory_order_release);
    durationSec_  = 0.0;
    revision_.fetch_add (1);

    if (sourcePath_.isEmpty()) return;

    // Sync metadata read — DEV-012 (sesja 44) requires duration before the
    // first WaveformView paint cycle to scale view-range correctly.
    // loadReader is fast (<10 ms typical for header read); not the source
    // of the DEV-024 lag.
    loadReader();

    // DEV-024 (sesja 95) — heavy 8192-bin coarse summary build moves to a
    // background thread. Was synchronous on message thread → blocked
    // applySelectedItem return → blocked next paint up to 50-150 ms (user-
    // reported lag sesja 49). Now: getPeakBlock returns zeros until worker
    // completes; onPeaksReady callback triggers WaveformView repaint when
    // ready (typically 50-200 ms wall on FLAC; ~5-20 ms on WAV).
    if (reader_ != nullptr)
        startBackgroundRebuild();
}

double FilePeaksProvider::getTotalDurationSeconds() const
{
    return durationSec_;
}

void FilePeaksProvider::loadReader()
{
    if (reader_ != nullptr || sourcePath_.isEmpty())
        return;

    juce::File f (sourcePath_);
    if (! f.existsAsFile())
        return;

    reader_.reset (formatManager_.createReaderFor (f));
    if (reader_ == nullptr)
        return;

    if (reader_->sampleRate <= 0.0 || reader_->lengthInSamples <= 0)
    {
        reader_.reset();
        return;
    }

    durationSec_ = (double) reader_->lengthInSamples / reader_->sampleRate;
}

void FilePeaksProvider::cancelInFlightWorker()
{
    if (peaksWorker_ == nullptr) return;

    // Signal the abort flag first (faster path inside the 8192-bin loop)
    // then call stopThread which combines signalThreadShouldExit + join
    // with timeout. 1 s is generous — typical bail is <10 ms because the
    // inner loop checks abortRebuild_ every iteration.
    abortRebuild_.store (true, std::memory_order_release);
    peaksWorker_->stopThread (1000);
    peaksWorker_.reset();
    abortRebuild_.store (false, std::memory_order_release);
}

void FilePeaksProvider::startBackgroundRebuild()
{
    peaksWorker_ = std::make_unique<PeaksWorker> (*this);
    peaksWorker_->startThread();
}

void FilePeaksProvider::rebuildSummaryIfNeeded()
{
    if (coarseReady_.load (std::memory_order_acquire))
        return;

    loadReader();
    if (reader_ == nullptr)
        return;

    const juce::int64 totalSamples = reader_->lengthInSamples;
    const int         numChannels  = (int) reader_->numChannels;
    if (totalSamples <= 0 || numChannels <= 0)
        return;

    const int nBins = (int) std::min<juce::int64> (kCoarseBins, totalSamples);

    // Build into temporary local vectors first; only swap into coarseMin_/
    // coarseMax_ + flip coarseReady_ at the end. This guarantees that
    // getPeakBlock readers either see fully-built peaks (coarseReady_=true)
    // or empty/stale state (coarseReady_=false) — never partial state.
    std::vector<float> localMin ((size_t) nBins, 0.0f);
    std::vector<float> localMax ((size_t) nBins, 0.0f);

    juce::ignoreUnused (numChannels);

    for (int i = 0; i < nBins; ++i)
    {
        // DEV-024 (sesja 95) — abort fast-path checked every iteration.
        // Worker thread bails within one bin-read of a setSourcePath call.
        if (abortRebuild_.load (std::memory_order_acquire))
            return;

        const juce::int64 startSample = (juce::int64) (((juce::int64) i * totalSamples) / nBins);
        const juce::int64 endSample   = (juce::int64) (((juce::int64) (i + 1) * totalSamples) / nBins);
        const juce::int64 numSamples  = juce::jmax<juce::int64> (1, endSample - startSample);

        juce::Range<float> lo_hi (0.0f, 0.0f);
        reader_->readMaxLevels (startSample, numSamples, &lo_hi, 1);

        localMin[(size_t) i] = lo_hi.getStart();
        localMax[(size_t) i] = lo_hi.getEnd();
    }

    // Final abort check before publishing — race window is tiny but real:
    // setSourcePath might fire just as we're about to commit.
    if (abortRebuild_.load (std::memory_order_acquire))
        return;

    coarseMin_ = std::move (localMin);
    coarseMax_ = std::move (localMax);
    coarseReady_.store (true, std::memory_order_release);

    // DEV-024 (sesja 95) — bump revision AFTER peaks are published so
    // WaveformView::PeaksCache invalidates on the next paint. Without this
    // the cache holds zero-bins fetched during the brief window after
    // setSourcePath kicked the worker but before peaks landed; subsequent
    // paints (including the one triggered by onPeaksReady callback) would
    // hit the stale cache and never display the silhouette. revision_ is
    // also bumped in setSourcePath itself (when sourcePath changes) — this
    // is the SECOND bump that signals "peaks now valid for current path".
    revision_.fetch_add (1);
}

void FilePeaksProvider::getPeakBlock (double startSec, double endSec, int nPx,
                                      float* minOut, float* maxOut)
{
    if (nPx <= 0 || minOut == nullptr || maxOut == nullptr)
        return;

    // DEV-024 (sesja 95) — no longer triggers rebuildSummaryIfNeeded
    // synchronously. Background worker (kicked by setSourcePath) handles
    // it. During the brief window between setSourcePath return and
    // worker-completion, getPeakBlock returns zeros (existing degenerate
    // path below). onPeaksReady callback triggers WaveformView repaint
    // when peaks land.
    if (! coarseReady_.load (std::memory_order_acquire)
        || durationSec_ <= 0.0
        || coarseMin_.empty())
    {
        for (int i = 0; i < nPx; ++i) { minOut[i] = 0.0f; maxOut[i] = 0.0f; }
        return;
    }

    const int nCoarse = (int) coarseMin_.size();
    const double range = std::max (1e-9, endSec - startSec);

    // Two regimes:
    //   (a) zoom-out — UI bin spans >1 coarse bin → min-of-mins / max-of-maxes
    //       (anti-aliasing: preserves visual peaks across the averaged range).
    //   (b) zoom-in  — UI bin fits inside a single coarse bin → linear-interp
    //       between the two flanking coarse bins (REABeat WaveformView.cpp:489
    //       style, adapted to our bar-per-bin model; eliminates the staircase
    //       that otherwise emerges at high zoom when many UI bins sample the
    //       same coarse source bin).
    for (int i = 0; i < nPx; ++i)
    {
        const double t0 = startSec + range * (double) i       / (double) nPx;
        const double t1 = startSec + range * (double) (i + 1) / (double) nPx;

        const double u0 = juce::jlimit (0.0, 1.0, t0 / durationSec_);
        const double u1 = juce::jlimit (0.0, 1.0, t1 / durationSec_);

        const double coarse0 = u0 * nCoarse;
        const double coarse1 = u1 * nCoarse;

        float mn = 0.0f;
        float mx = 0.0f;

        if (coarse1 - coarse0 >= 1.0)
        {
            // Zoom-out: aggregate min/max across the covered coarse bins.
            int i0 = juce::jlimit (0, nCoarse - 1, (int) std::floor (coarse0));
            int i1 = juce::jlimit (i0 + 1, nCoarse, (int) std::ceil  (coarse1));
            mn =  1.0f;
            mx = -1.0f;
            for (int k = i0; k < i1; ++k)
            {
                mn = std::min (mn, coarseMin_[(size_t) k]);
                mx = std::max (mx, coarseMax_[(size_t) k]);
            }
            if (mn > mx) { mn = 0.0f; mx = 0.0f; } // degenerate fallback
        }
        else
        {
            // Zoom-in: linear-interpolate between the two flanking coarse bins
            // using the midpoint of the UI bin time range.
            const double coarseMid = 0.5 * (coarse0 + coarse1);
            const int iA = juce::jlimit (0, nCoarse - 1, (int) std::floor (coarseMid));
            const int iB = juce::jlimit (0, nCoarse - 1, iA + 1);
            const float frac = (float) (coarseMid - std::floor (coarseMid));
            mn = coarseMin_[(size_t) iA] * (1.0f - frac) + coarseMin_[(size_t) iB] * frac;
            mx = coarseMax_[(size_t) iA] * (1.0f - frac) + coarseMax_[(size_t) iB] * frac;
        }

        minOut[i] = mn;
        maxOut[i] = mx;
    }
}

} // namespace reamix::ui
