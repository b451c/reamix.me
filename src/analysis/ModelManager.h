#pragma once

#include <functional>
#include <string>

#include <juce_core/juce_core.h>

// beat-this ONNX model cache manager.
//
// Per-product cache per ADR-006:
//   macOS:   ~/Library/Application Support/reamix/models/
//   Windows: %APPDATA%\reamix\models\
//   Linux:   ~/.config/reamix/models/
//
// All three resolve via juce::File::userApplicationDataDirectory +
// "reamix"/"models". Deliberately different from REABeat's ~/.reabeat/models
// (ADR-006 Option B — independent per-product caches).
//
// Integrity: file size must fall in [kExpectedSizeMin, kExpectedSizeMax] and
// (when pinned) SHA-256 must match kExpectedSha256. Current file pinned is
// the beat-this ONNX served from the REABeat GitHub release (v2.0.0-model).
// If upstream rotates the release, bump kExpectedSha256 and record an ADR.

namespace reamix {

class ModelManager
{
public:
    // Progress callback receives fraction in [0, 1] during download.
    // Called from the download thread; keep handler lock-free.
    using ProgressCb = std::function<void(float)>;

    // Resolve the per-product model directory. Creates the directory tree if
    // missing. On a hostile filesystem (read-only home) the returned File may
    // not exist on disk — callers that need existence must check.
    static juce::File modelDir();

    // modelDir() / kModelFilename. Does not require the file to exist.
    static juce::File modelPath();

    // True iff the file exists, its size is in the accepted range, AND its
    // SHA-256 matches kExpectedSha256. A corrupt or truncated cache returns
    // false here and triggers re-download by ensureDownloaded().
    static bool isCached();

    // Ensure the model is present and integrity-checked on disk.
    //   - If isCached() returns true: no-op, returns true.
    //   - Else downloads kModelUrl to modelPath(), re-verifies size + SHA-256,
    //     deletes the file and returns false on any failure.
    // outError (optional) receives a human-readable diagnostic on failure.
    static bool ensureDownloaded(ProgressCb cb = nullptr,
                                 std::string* outError = nullptr);

    // SHA-256 hex digest (64 lowercase chars) of an on-disk file, computed
    // via juce::SHA256. Returns empty string on read failure.
    static juce::String computeSha256(const juce::File& file);

    // SHA-256 hex digest of an in-memory byte range. Used by the unit test
    // against NIST vectors to validate the SHA-256 plumbing without touching
    // the filesystem.
    static juce::String computeSha256(const void* data, size_t numBytes);

    static constexpr const char* kModelFilename = "beat_this_final0.onnx";

    // Same file REABeat ships (ADR-006 Option B still accepts shared source;
    // only the cache path is per-product). If this URL rotates or we mirror
    // to our own CDN, record an ADR and update both URL + SHA.
    static constexpr const char* kModelUrl =
        "https://github.com/b451c/ReaBeat/releases/download/v2.0.0-model/beat_this_final0.onnx";

    // Integrity bounds. Observed file is 83'077'779 bytes (~79 MB); the
    // 70–100 MB window mirrors REABeat's tolerance for future minor rebuilds.
    static constexpr juce::int64 kExpectedSizeMin = 70'000'000;
    static constexpr juce::int64 kExpectedSizeMax = 100'000'000;

    // Pinned SHA-256 of beat_this_final0.onnx as served from kModelUrl on
    // 2026-04-20 (computed locally from ~/.reabeat/models/beat_this_final0.onnx
    // via `shasum -a 256`; size 83'077'779 B).
    static constexpr const char* kExpectedSha256 =
        "552dfc2b0d705e8eba77f75d8e4635de121d1884bf8bb6f1e4ad6c882ff5e384";
};

} // namespace reamix
