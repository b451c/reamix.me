#pragma once

#include "AnalysisBundle.h"

#include <juce_core/juce_core.h>

namespace reamix::ui
{

// AnalysisDiskCache — persistent on-disk cache for AnalysisBundle objects.
//
// ADR-053 (sesja 63 BUG-19 follow-up) — closes the asymmetry where user-
// blocks survived REAPER project save/reopen via P_EXT (BUG-19 fix) but
// the analysis bundle did not, forcing a manual Analyze click before the
// user could resume real work on a previously-analyzed track. Per user
// mandate: *"jak juz odpalam to sie powinna mysle uruchomic i plugin i
// jej stan po analizie wraz z blokami"*.
//
// Layout: `~/Library/Application Support/reamix.me/cache/<sha1>.bundle`
// where `<sha1>` is the SHA1 of the absolute source path. Cross-project
// by design — the same audio file analyzed in any REAPER project shares
// one cache entry. Magic/version header rejects stale formats cleanly
// (any breaking change bumps kFormatVersion → next plugin run treats old
// caches as miss and overwrites on next analyze).
//
// What is NOT cached: `stereoNative` (the raw audio buffer). It is
// re-decoded from the source file on cache hit using the same
// `AudioFormatManager` path AnalyzePipeline uses. Trade-off: 1-2 s of
// audio decode vs ~36 MB / track of disk weight; the heavy DSP outputs
// (TransitionCost matrix, BeatWindows boundary waveforms, FeatureExtractor
// 59-dim feature matrix) are what take the time to compute, and those
// ARE cached.
//
// All methods are static; the class has no instance state.

class AnalysisDiskCache
{
public:
    // Returns `~/Library/Application Support/reamix.me/cache/`. Creates
    // the directory tree if it does not exist. Returns an invalid juce::File
    // only on filesystem permission failure (rare in user-app sandbox).
    static juce::File getCacheDirectory();

    // Maps an absolute source path to its cache file (whether or not the
    // file currently exists). Filename = SHA1(sourcePath) + ".bundle".
    static juce::File cacheFileForSource (const juce::String& sourcePath);

    // Attempts to load + reconstitute a bundle for `sourcePath`. Returns
    // nullptr on: missing file, version mismatch, truncated/corrupt data,
    // source-file re-decode failure. On success the returned bundle has
    // `stereoNative` populated by re-reading the source audio file.
    static AnalysisBundlePtr tryLoad (const juce::String& sourcePath);

    // Writes `bundle` to its cache file. Returns true on success. Skips
    // `stereoNative` from the on-disk payload (re-decoded on load). Safe
    // to call on background thread.
    static bool save (const AnalysisBundle& bundle);

    // Total size of all .bundle files in the cache directory, in bytes.
    static juce::int64 totalSizeBytes();

    // Number of .bundle files in the cache directory.
    static int countEntries();

    // Deletes every .bundle file in the cache directory. Returns the
    // number of files actually removed. Does not delete the directory
    // itself (next save recreates entries).
    static int clearAll();

    // Triggers OS reveal of the cache directory in Finder. juce::File::
    // revealToUser opens the parent and selects the target; for a
    // directory target this opens the directory itself.
    static void revealInFinder();

private:
    AnalysisDiskCache() = delete;
};

} // namespace reamix::ui
