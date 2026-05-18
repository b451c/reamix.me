// warm_cache — DEV-028 Stage-1 calibration prerequisite.
//
// Headless driver that runs `AnalyzePipeline` end-to-end on a list of audio
// files and writes each resulting `AnalysisBundle` to `AnalysisDiskCache`
// (~/Library/Application Support/reamix.me/cache/<sha1(path)>.bundle).
//
// Why this exists (sesja 71b decision): the calibration harness must read
// production-equivalent analysis state for each corpus track. Reading from
// `references/golden/phase-2/dumps/` (Option A, considered + rejected) would
// inherit (a) 60-second clip semantics — not full-song generalisation, (b)
// synthetic beats in 15/16 tracks instead of beat-this ONNX output, (c) any
// staleness vs current AnalyzePipeline. Per the user's quality-first mandate,
// we pay a one-time bootstrap cost (~10–15 min for 16 tracks on M-series Mac)
// to get bit-exact production analysis cached on disk; subsequent harness
// invocations read from the cache in ~ms.
//
// Usage:
//   warm_cache --audio <path> [--audio <path> ...]
//   warm_cache --corpus references/golden/phase-1 references/golden/phase-2
//
// `--corpus <dir>` walks each directory and analyses every .mp3/.flac/.wav
// found. Existing cache entries are skipped unless --force is passed. Exit
// code is 0 on full success, 1 if any track failed.
//
// Threading: AnalyzePipeline is a juce::Thread that posts progress and
// completion via juce::MessageManager::callAsync. We construct a
// ScopedJuceInitialiser_GUI so the message thread is alive, then drive the
// dispatch loop ourselves until each pipeline signals done.

#include "ui/AnalyzePipeline.h"
#include "ui/AnalysisDiskCache.h"
#include "analysis/BeatDetector.h"
#include "analysis/ModelManager.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Args
{
    std::vector<juce::String> audioPaths;
    std::vector<juce::String> corpusDirs;
    bool                      force { false };
};

int printUsage (const char* argv0)
{
    std::fprintf (stderr,
        "usage: %s [--audio <path>]... [--corpus <dir>]... [--force]\n"
        "\n"
        "Walks each --corpus directory for .mp3/.flac/.wav and analyses each\n"
        "audio file via AnalyzePipeline, writing the result to\n"
        "AnalysisDiskCache. Existing entries are skipped unless --force.\n",
        argv0);
    return 2;
}

bool parseArgs (int argc, char** argv, Args& out)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--audio" && i + 1 < argc)
        {
            out.audioPaths.push_back (juce::String (argv[++i]));
        }
        else if (a == "--corpus" && i + 1 < argc)
        {
            out.corpusDirs.push_back (juce::String (argv[++i]));
        }
        else if (a == "--force")
        {
            out.force = true;
        }
        else
        {
            std::fprintf (stderr, "unknown arg: %s\n", a.c_str());
            return false;
        }
    }
    if (out.audioPaths.empty() && out.corpusDirs.empty()) return false;
    return true;
}

// Walk a directory for audio files (non-recursive — corpus directories are
// flat). Sorted alphabetically so output order is deterministic across runs.
std::vector<juce::String> findAudio (const juce::File& dir)
{
    std::vector<juce::String> paths;
    if (! dir.isDirectory())
    {
        std::fprintf (stderr, "  warning: %s is not a directory\n",
                      dir.getFullPathName().toRawUTF8());
        return paths;
    }
    auto matches = dir.findChildFiles (juce::File::findFiles, false,
                                       "*.mp3;*.flac;*.wav");
    for (const auto& f : matches) paths.push_back (f.getFullPathName());
    std::sort (paths.begin(), paths.end());
    return paths;
}

// Drive AnalyzePipeline to completion synchronously. Returns true on success
// and assigns *outBundle on completion. Pumps the message dispatch loop in
// 50 ms slices so progress + completion callbacks fire predictably.
bool runPipeline (const juce::String&             sourcePath,
                  reamix::BeatDetector&           beatDetector,
                  reamix::ui::AnalysisBundlePtr&  outBundle,
                  juce::String&                   outError)
{
    reamix::ui::AnalyzePipeline::Input in;
    in.sourcePath = sourcePath;

    std::atomic<bool>             done { false };
    reamix::ui::AnalysisBundlePtr resultBundle;
    juce::String                  resultError;

    // Last-printed progress for inline progress display.
    std::atomic<int> lastPct { -1 };
    juce::String     lastStage;

    auto progressCb = [&] (juce::String step, double p01)
    {
        const int pct = (int) std::round (p01 * 100.0);
        if (pct != lastPct.load() || step != lastStage)
        {
            lastPct.store (pct);
            lastStage = step;
            std::fprintf (stderr, "\r    %3d%% %-40s",
                          pct, step.toRawUTF8());
            std::fflush (stderr);
        }
    };
    auto completeCb = [&] (reamix::ui::AnalysisBundlePtr bundle, juce::String error)
    {
        resultBundle = std::move (bundle);
        resultError  = std::move (error);
        done.store (true, std::memory_order_release);
    };

    auto pipeline = std::make_unique<reamix::ui::AnalyzePipeline> (
        std::move (in), beatDetector, std::move (progressCb), std::move (completeCb));
    pipeline->startThread();

    while (! done.load (std::memory_order_acquire))
    {
        juce::MessageManager::getInstance()->runDispatchLoopUntil (50);
    }
    // Newline after the inline progress line.
    std::fprintf (stderr, "\n");

    pipeline.reset();  // joins the thread (~immediate since done already)

    if (resultBundle == nullptr)
    {
        outError = resultError.isNotEmpty() ? resultError
                                            : juce::String ("AnalyzePipeline returned null bundle");
        return false;
    }
    outBundle = std::move (resultBundle);
    return true;
}

} // namespace

int main (int argc, char** argv)
{
    Args args;
    if (! parseArgs (argc, argv, args)) return printUsage (argv[0]);

    // Initialise JUCE message thread + GUI subsystem (required by
    // AudioFormatManager and MessageManager::callAsync used inside
    // AnalyzePipeline).
    juce::ScopedJuceInitialiser_GUI juceInit;

    // Ensure beat-this ONNX model is on disk + load it once. ModelManager
    // downloads to ~/Library/Application Support/reamix.me/models/<sha>.onnx;
    // the warm-cache run shares this with the plugin so the user only pays
    // the download once.
    std::fprintf (stderr, "step 0: ensuring beat-this model is downloaded\n");
    std::string downloadErr;
    if (! reamix::ModelManager::ensureDownloaded (nullptr, &downloadErr))
    {
        std::fprintf (stderr, "  ERROR: model download failed: %s\n",
                      downloadErr.c_str());
        return 1;
    }
    reamix::BeatDetector beatDetector;
    if (! beatDetector.loadModel (
            reamix::ModelManager::modelPath().getFullPathName().toStdString()))
    {
        std::fprintf (stderr, "  ERROR: BeatDetector::loadModel failed\n");
        return 1;
    }
    std::fprintf (stderr, "  OK model ready (%s)\n",
                  reamix::ModelManager::modelPath().getFullPathName().toRawUTF8());

    // Build full audio path list.
    std::vector<juce::String> audioPaths = args.audioPaths;
    for (const auto& dirStr : args.corpusDirs)
    {
        const auto dirPaths = findAudio (juce::File (dirStr));
        audioPaths.insert (audioPaths.end(), dirPaths.begin(), dirPaths.end());
    }
    if (audioPaths.empty())
    {
        std::fprintf (stderr, "no audio files to analyze\n");
        return 1;
    }

    std::fprintf (stderr, "step 1: analyzing %d audio file(s)\n",
                  (int) audioPaths.size());
    std::fprintf (stderr, "cache dir: %s\n",
                  reamix::ui::AnalysisDiskCache::getCacheDirectory()
                      .getFullPathName().toRawUTF8());

    int nOk = 0, nSkipped = 0, nFailed = 0;
    const auto wallStart = std::chrono::steady_clock::now();

    for (size_t i = 0; i < audioPaths.size(); ++i)
    {
        const auto& path = audioPaths[i];
        const auto  shortName = juce::File (path).getFileName();

        std::fprintf (stderr, "\n[%d/%d] %s\n",
                      (int) (i + 1), (int) audioPaths.size(),
                      shortName.toRawUTF8());

        if (! args.force)
        {
            auto cachedBundle = reamix::ui::AnalysisDiskCache::tryLoad (path);
            if (cachedBundle != nullptr)
            {
                std::fprintf (stderr,
                              "    SKIP (cache hit, --force to override)\n");
                ++nSkipped;
                continue;
            }
        }

        const auto t0 = std::chrono::steady_clock::now();
        reamix::ui::AnalysisBundlePtr bundle;
        juce::String                  err;
        if (! runPipeline (path, beatDetector, bundle, err))
        {
            std::fprintf (stderr, "    FAIL: %s\n", err.toRawUTF8());
            ++nFailed;
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double> (t1 - t0).count();

        if (! reamix::ui::AnalysisDiskCache::save (*bundle))
        {
            std::fprintf (stderr, "    FAIL: AnalysisDiskCache::save returned false\n");
            ++nFailed;
            continue;
        }

        const int    nBeats   = (int) bundle->beatTimes.size();
        const double bpm      = bundle->bpm;
        const int    nSeg     = (int) bundle->structure.segments.size();
        std::fprintf (stderr,
                      "    OK %.1f s wall · %d beats · %.1f BPM · %d segments\n",
                      sec, nBeats, bpm, nSeg);
        ++nOk;
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    const double wallTotal = std::chrono::duration<double> (wallEnd - wallStart).count();

    std::fprintf (stderr, "\n=== summary ===\n");
    std::fprintf (stderr, "ok=%d skipped=%d failed=%d wall=%.1f s cache=%lld bytes (%d entries)\n",
                  nOk, nSkipped, nFailed, wallTotal,
                  (long long) reamix::ui::AnalysisDiskCache::totalSizeBytes(),
                  reamix::ui::AnalysisDiskCache::countEntries());

    return (nFailed > 0) ? 1 : 0;
}
