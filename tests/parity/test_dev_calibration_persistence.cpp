// test_dev_calibration_persistence — sesja 98 (ADR-087 STATUS UPDATE 1) self-validation.
// Sesja 106 (ADR-098): perceptual mapping tests removed; schema bumped v1→v2.
// Sesja 107 (ADR-097): relocated src/dev/ → src/ui/, namespace reamix::dev →
//   reamix::ui, compiled into every build (production dylib ships the panel
//   as opt-in window per ADR-097).
//
// Validates the atDefault sentinel semantics (the JSONL storage it was
// written for is gone; the two RemixCache.h helpers are live).
//
// Per ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`:
// hand-computed invariants, no Python ground truth (advanced weights is
// C++-canonical infrastructure).

#include "ui/RemixCache.h"
#include "remix/Quality.h"

#include <juce_core/juce_core.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace
{

#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) {                                                            \
        std::printf ("FAIL: %s — %s\n", #expr, msg);                           \
        return false;                                                          \
    }                                                                          \
} while (0)

bool testHashQualityWeightsAtDefault()
{
    auto def = reamix::remix::kDefaultQualityWeights;
    CHECK (reamix::ui::hashQualityWeights (def) == 0,
           "kDefaultQualityWeights should hash to 0 (atDefault sentinel)");
    auto modified = def;
    modified.waveform = 0.55;  // tweak
    CHECK (reamix::ui::hashQualityWeights (modified) != 0,
           "modified weights should hash to non-zero");
    return true;
}

bool testQualityWeightsAtDefaultHelper()
{
    auto def = reamix::remix::kDefaultQualityWeights;
    CHECK (reamix::ui::qualityWeightsAtDefault (def),
           "kDefaultQualityWeights should match atDefault helper");
    auto modified = def;
    modified.transient_continuity = 0.20;
    CHECK (! reamix::ui::qualityWeightsAtDefault (modified),
           "modified weights should not match atDefault");
    return true;
}

} // namespace

int main()
{
    bool ok = true;
    struct TestCase { const char* name; bool (*fn)(); };
    const TestCase cases [] = {
        { "hashQualityWeightsAtDefault",       testHashQualityWeightsAtDefault },
        { "qualityWeightsAtDefaultHelper",     testQualityWeightsAtDefaultHelper },
    };
    for (auto& tc : cases)
    {
        const bool pass = tc.fn();
        std::printf ("%s %s\n", pass ? "PASS" : "FAIL", tc.name);
        if (! pass) ok = false;
    }
    std::printf ("\n%s — dev calibration persistence\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
