// Unit test for reamix::ModelManager.
//
// No parity-vs-REABeat component: ADR-006 deliberately diverges cache paths
// between the two products (reamix uses ~/Library/Application Support/reamix
// per platform conventions; REABeat uses ~/.reabeat). Diffing would either
// manufacture fake parity or break the ADR.
//
// What this test validates:
//
//   (1) SHA-256 plumbing. Three NIST-style vectors + empty string. If the
//       juce::SHA256 wrapper is hooked up correctly, all four digests match
//       the canonical hex strings byte-for-byte.
//
//   (2) modelDir() / modelPath() path shape: directory is created on call,
//       filename component matches kModelFilename, both live under
//       userApplicationDataDirectory.
//
// What this test does NOT validate:
//
//   - Actual network download (requires HTTP, non-deterministic, flaky in CI;
//     deferred to manual REAPER-load validation per handover).
//   - isCached() on a real 79 MB model (requires the model file; covered by
//     manual validation — if the plugin loads beat-this on first run in
//     REAPER, caching works).
//
// Exit 0 on pass, 1 on first failure.

#include "analysis/ModelManager.h"

#include <juce_core/juce_core.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct HashCase
{
    const char* name;
    std::vector<std::uint8_t> bytes;
    const char* expectedHex;
};

int runHashCase(const HashCase& tc)
{
    auto got = reamix::ModelManager::computeSha256(tc.bytes.data(), tc.bytes.size());
    const bool ok = (got == juce::String(tc.expectedHex));
    std::printf("sha256/%s: N=%zu digest=%s %s\n",
                tc.name, tc.bytes.size(), got.toRawUTF8(), ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int runPathShape()
{
    auto dir = reamix::ModelManager::modelDir();
    auto path = reamix::ModelManager::modelPath();

    const bool dirIsDir       = dir.isDirectory();
    const bool filenameOk     = (path.getFileName()
                                 == juce::String(reamix::ModelManager::kModelFilename));
    const bool pathUnderDir   = path.getParentDirectory() == dir;
    const bool underAppData   = dir.isAChildOf(
        juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory));

    std::printf("path/shape: dir=%s filename=%s under_app_data=%s\n",
                dir.getFullPathName().toRawUTF8(),
                path.getFileName().toRawUTF8(),
                underAppData ? "yes" : "no");

    if (!dirIsDir || !filenameOk || !pathUnderDir || !underAppData)
    {
        std::printf("  FAIL (dirIsDir=%d filenameOk=%d pathUnderDir=%d underAppData=%d)\n",
                    dirIsDir, filenameOk, pathUnderDir, underAppData);
        return 1;
    }
    return 0;
}

std::vector<std::uint8_t> bytesFrom(const char* s)
{
    const auto n = std::strlen(s);
    return { reinterpret_cast<const std::uint8_t*>(s),
             reinterpret_cast<const std::uint8_t*>(s) + n };
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
{
    // NIST SHA-256 vectors (https://www.di-mgt.com.au/sha_testvectors.html):
    //   "" → e3b0c442...b7852b855
    //   "abc" → ba7816bf...f20015ad
    //   "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" → 248d6a61...9db06c1
    //   441 'a's → a long-input sanity vector ~same length as a tiny file.
    std::vector<std::uint8_t> aaa441(441, 'a');

    std::vector<HashCase> cases = {
        {"empty", bytesFrom(""),
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", bytesFrom("abc"),
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"448bit_msg",
         bytesFrom("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"441_a", aaa441,
         "a06005644e2fb28c22902aac0777d83e039a69381c3899a076e920e88cdeb578"},
    };

    int fails = 0;
    for (const auto& tc : cases)
        fails += runHashCase(tc);

    fails += runPathShape();

    if (fails == 0)
    {
        std::printf("model_manager: all cases PASS\n");
        return 0;
    }
    std::printf("model_manager: %d case(s) FAILED\n", fails);
    return 1;
}
