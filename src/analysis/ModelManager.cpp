#include "analysis/ModelManager.h"

#include <juce_cryptography/juce_cryptography.h>

namespace reamix {

juce::File ModelManager::modelDir()
{
    // ADR-006 paths per platform:
    //   macOS:   ~/Library/Application Support/reamix/models/
    //   Windows: %APPDATA%\reamix\models\
    //   Linux:   ~/.config/reamix/models/
    //
    // JUCE's userApplicationDataDirectory maps to %APPDATA% on Windows and
    // ~/.config on Linux (matches ADR directly), but to ~/Library on macOS —
    // so on macOS we append "Application Support" manually.
    auto base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
   #if JUCE_MAC
    base = base.getChildFile("Application Support");
   #endif
    auto dir = base.getChildFile("reamix").getChildFile("models");
    if (!dir.isDirectory())
        dir.createDirectory();
    return dir;
}

juce::File ModelManager::modelPath()
{
    return modelDir().getChildFile(kModelFilename);
}

juce::String ModelManager::computeSha256(const juce::File& file)
{
    juce::FileInputStream in(file);
    if (!in.openedOk())
        return {};
    return juce::SHA256(in).toHexString();
}

juce::String ModelManager::computeSha256(const void* data, size_t numBytes)
{
    return juce::SHA256(data, numBytes).toHexString();
}

bool ModelManager::isCached()
{
    auto path = modelPath();
    if (!path.existsAsFile())
        return false;

    const auto size = path.getSize();
    if (size < kExpectedSizeMin || size > kExpectedSizeMax)
        return false;

    const auto digest = computeSha256(path);
    return digest == juce::String(kExpectedSha256);
}

bool ModelManager::ensureDownloaded(ProgressCb cb, std::string* outError)
{
    auto setErr = [outError](const char* msg) {
        if (outError) *outError = msg;
    };

    if (isCached())
        return true;

    auto dest = modelPath();

    // Sesja 111 — REABeat coexistence (handover P0 #3).
    // REABeat caches the exact same beat_this_final0.onnx (same kModelUrl,
    // same SHA-256) at ~/.reabeat/models/. If a user already has REABeat
    // installed, copy that file instead of re-downloading 80 MB. JUCE's
    // userHomeDirectory resolves to /Users/<u>/, /home/<u>/, C:\Users\<u>\
    // automatically, so a single path works cross-platform.
    {
        auto reabeatPath = juce::File::getSpecialLocation(juce::File::userHomeDirectory)
                               .getChildFile(".reabeat")
                               .getChildFile("models")
                               .getChildFile(kModelFilename);
        if (reabeatPath.existsAsFile()
            && reabeatPath.getSize() >= kExpectedSizeMin
            && reabeatPath.getSize() <= kExpectedSizeMax
            && computeSha256(reabeatPath) == juce::String(kExpectedSha256))
        {
            dest.deleteFile();
            if (reabeatPath.copyFileTo(dest) && isCached())
                return true;
            // Copy succeeded structurally but integrity check failed, or copy
            // itself failed (e.g. permission). Fall through to fresh download.
            dest.deleteFile();
        }
    }

    dest.deleteFile(); // clear any partial / corrupt cache before network fetch

    juce::URL url(kModelUrl);
    auto in = url.createInputStream(
        juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
            .withConnectionTimeoutMs(30000));

    if (!in)
    {
        setErr("Failed to open model URL input stream");
        return false;
    }

    const auto totalLen = in->getTotalLength();

    auto out = dest.createOutputStream();
    if (!out)
    {
        setErr("Failed to open model cache output stream");
        return false;
    }

    constexpr int kBufferSize = 65536;
    juce::HeapBlock<char> buffer(kBufferSize);
    juce::int64 totalRead = 0;

    while (!in->isExhausted())
    {
        const auto n = in->read(buffer.getData(), kBufferSize);
        if (n <= 0)
            break;
        out->write(buffer.getData(), static_cast<size_t>(n));
        totalRead += n;

        if (cb)
            cb(totalRead, totalLen);
    }

    out->flush();
    out.reset();

    if (!isCached())
    {
        dest.deleteFile();
        setErr("Downloaded model failed size or SHA-256 integrity check");
        return false;
    }

    return true;
}

} // namespace reamix
