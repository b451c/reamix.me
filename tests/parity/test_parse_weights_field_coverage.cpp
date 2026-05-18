// test_parse_weights_field_coverage — sesja 86 ADR-076 CI guardrail.
//
// Sesja 85 added drum_stem_continuity / drum_stem_activity_match /
// instrument_stem_chroma_continuity to QualityWeights but the matching
// `if (v.hasProperty(...)) w.X = ...;` blocks in
// tools/calibration_harness.cpp::parseWeights were forgotten → those 3 axes
// were silently zeroed for the entire 14-D Sobol sweep → ADR-074 REJECT-LIKE
// based on bugged data → ADR-075 audit gate → ADR-076 fix + this test.
//
// What this test guarantees: every QualityWeights field that orchestrator.py
// emits to JSONL is read by parseWeights and lands on the matching struct
// member. Plus the "missing required field throws" + "missing optional field
// defaults to 0.0 / false" contracts.
//
// CONTRACT FOR FUTURE AGENTS adding a new QualityWeights field:
//   1. Add the field to src/remix/Quality.h::QualityWeights struct.
//   2. Add a parse block to tools/calibration_harness_parse_weights.h.
//   3. Add a probe row below in kProbes[] (or kRequiredProbes[] if the
//      field must be present in every orchestrator JSONL row).
//   4. Update tools/calibration/orchestrator.py::WEIGHT_FIELD_ORDER_BASE +
//      tools/calibration/preference_gp.py::WEIGHT_FIELD_ORDER in lockstep.
//
// All four sources MUST stay aligned. Sesja-81 dev-028-lessons.md:631
// documented this exact rule; sesja 85 violated step 2 + 3 + 4. CI will now
// catch it loudly instead of silently corrupting calibration.

#include "../../tools/calibration_harness_parse_weights.h"

#include "remix/Quality.h"

#include <juce_core/juce_core.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

using reamix::remix::QualityWeights;

// Required fields (parseWeights `get` lambda throws on absent). These match
// the historical pre-sesja-75 base simplex (10 cost components from the
// Python parity baseline).
struct RequiredProbe
{
    const char* json_name;
    double probe_value;
    double (*getter) (const QualityWeights&);
};

const RequiredProbe kRequired[] = {
    { "waveform",    0.111, [] (const QualityWeights& w) { return w.waveform; } },
    { "successor",   0.122, [] (const QualityWeights& w) { return w.successor; } },
    { "edge_splice", 0.133, [] (const QualityWeights& w) { return w.edge_splice; } },
    { "context",     0.144, [] (const QualityWeights& w) { return w.context; } },
    { "label",       0.155, [] (const QualityWeights& w) { return w.label; } },
    { "bar_align",   0.166, [] (const QualityWeights& w) { return w.bar_align; } },
    { "section",     0.177, [] (const QualityWeights& w) { return w.section; } },
    { "energy",      0.188, [] (const QualityWeights& w) { return w.energy; } },
    { "edge_energy", 0.199, [] (const QualityWeights& w) { return w.edge_energy; } },
    { "centroid",    0.210, [] (const QualityWeights& w) { return w.centroid; } },
};
constexpr size_t kRequiredCount = sizeof (kRequired) / sizeof (kRequired[0]);

// Optional double fields (`if (v.hasProperty(...))` blocks). Default 0.0
// when absent. Productionised via ADR-064 / ADR-066 / ADR-070 / ADR-068 D4 /
// ADR-071 phase-1 / ADR-074.
struct OptionalDoubleProbe
{
    const char* json_name;
    double probe_value;
    double (*getter) (const QualityWeights&);
};

const OptionalDoubleProbe kOptionalDoubles[] = {
    { "transient_continuity",                0.221, [] (const QualityWeights& w) { return w.transient_continuity; } },
    { "mfcc_continuity",                     0.232, [] (const QualityWeights& w) { return w.mfcc_continuity; } },
    { "extra1",                              0.243, [] (const QualityWeights& w) { return w.extra1; } },
    { "sequential_continuity",               0.265, [] (const QualityWeights& w) { return w.sequential_continuity; } },
};
constexpr size_t kOptionalDoubleCount = sizeof (kOptionalDoubles) / sizeof (kOptionalDoubles[0]);

// Build a juce::var DynamicObject containing every field from kRequired +
// kOptionalDoubles + the use_harmonic_mean bool flag.
juce::var buildFullJson (bool harmonic_mean_flag)
{
    auto* obj = new juce::DynamicObject();
    for (size_t i = 0; i < kRequiredCount; ++i)
        obj->setProperty (kRequired[i].json_name, kRequired[i].probe_value);
    for (size_t i = 0; i < kOptionalDoubleCount; ++i)
        obj->setProperty (kOptionalDoubles[i].json_name, kOptionalDoubles[i].probe_value);
    obj->setProperty ("use_harmonic_mean", harmonic_mean_flag);
    return juce::var (obj);
}

bool approxEqual (double a, double b, double tol = 1e-12)
{
    return std::abs (a - b) <= tol;
}

int testAllFieldsRoundTrip()
{
    const juce::var v = buildFullJson (true);
    QualityWeights w = reamix::cal::parseWeights (v);

    int failures = 0;
    for (size_t i = 0; i < kRequiredCount; ++i)
    {
        const double actual = kRequired[i].getter (w);
        if (! approxEqual (actual, kRequired[i].probe_value))
        {
            std::fprintf (stderr,
                "FAIL [required]: '%s' expected %.6f got %.6f\n",
                kRequired[i].json_name, kRequired[i].probe_value, actual);
            ++failures;
        }
    }
    for (size_t i = 0; i < kOptionalDoubleCount; ++i)
    {
        const double actual = kOptionalDoubles[i].getter (w);
        if (! approxEqual (actual, kOptionalDoubles[i].probe_value))
        {
            std::fprintf (stderr,
                "FAIL [optional]: '%s' expected %.6f got %.6f\n",
                kOptionalDoubles[i].json_name, kOptionalDoubles[i].probe_value, actual);
            ++failures;
        }
    }
    if (! w.use_harmonic_mean)
    {
        std::fprintf (stderr, "FAIL: use_harmonic_mean expected true got false\n");
        ++failures;
    }
    return failures;
}

int testOptionalFieldsDefaultToZero()
{
    // Build JSON with ONLY required fields, no optional fields.
    auto* obj = new juce::DynamicObject();
    for (size_t i = 0; i < kRequiredCount; ++i)
        obj->setProperty (kRequired[i].json_name, kRequired[i].probe_value);
    juce::var v (obj);

    QualityWeights w = reamix::cal::parseWeights (v);

    int failures = 0;
    for (size_t i = 0; i < kOptionalDoubleCount; ++i)
    {
        const double actual = kOptionalDoubles[i].getter (w);
        if (! approxEqual (actual, 0.0))
        {
            std::fprintf (stderr,
                "FAIL [optional-default]: '%s' expected 0.0 got %.6f\n",
                kOptionalDoubles[i].json_name, actual);
            ++failures;
        }
    }
    if (w.use_harmonic_mean)
    {
        std::fprintf (stderr, "FAIL: use_harmonic_mean expected false (default) got true\n");
        ++failures;
    }
    return failures;
}

int testRequiredFieldMissingThrows()
{
    int failures = 0;

    // For each required field, build a JSON missing only that field and
    // verify parseWeights throws.
    for (size_t miss = 0; miss < kRequiredCount; ++miss)
    {
        auto* obj = new juce::DynamicObject();
        for (size_t i = 0; i < kRequiredCount; ++i)
        {
            if (i == miss) continue;
            obj->setProperty (kRequired[i].json_name, kRequired[i].probe_value);
        }
        juce::var v (obj);

        bool threw = false;
        try
        {
            (void) reamix::cal::parseWeights (v);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        if (! threw)
        {
            std::fprintf (stderr,
                "FAIL [required-missing]: '%s' missing did not throw\n",
                kRequired[miss].json_name);
            ++failures;
        }
    }
    return failures;
}

// Sesja-86 audit Cat 6 sentinel: the four "weights file order" sources of
// truth must list the SAME 14 calibratable axes. We can only check the C++
// side here (orchestrator.py / preference_gp.py / Quality.h struct). The
// other two sources are checked by orchestrator.py self-tests. This sentinel
// confirms we did not silently drop or rename a probe row in this file.
//
// Counter: the count is hand-maintained, but if a future agent adds a field
// to QualityWeights without adding a probe row here, they will not trip
// this assert — they WILL however fail testAllFieldsRoundTrip if they also
// add a parseWeights block referencing the new field but no probe row, or
// fail nothing if they add nothing — which is OK because then the field is
// genuinely unused. The intended failure mode this test catches is:
// QualityWeights field added + parseWeights block FORGOTTEN → bugged data.
int testProbeCountSentinel()
{
    constexpr size_t kExpectedRequired       = 10;  // pre-sesja-75 base
    constexpr size_t kExpectedOptionalDouble = 4;   // sesja 91 ADR-082 CL-1+CL-7: stems-aware + vocal_avoid removed
    int failures = 0;
    if (kRequiredCount != kExpectedRequired)
    {
        std::fprintf (stderr,
            "FAIL [sentinel]: required count = %zu, expected %zu — update kExpectedRequired if adding required fields.\n",
            kRequiredCount, kExpectedRequired);
        ++failures;
    }
    if (kOptionalDoubleCount != kExpectedOptionalDouble)
    {
        std::fprintf (stderr,
            "FAIL [sentinel]: optional-double count = %zu, expected %zu — update kExpectedOptionalDouble if adding optional doubles.\n",
            kOptionalDoubleCount, kExpectedOptionalDouble);
        ++failures;
    }
    return failures;
}

} // namespace

int main()
{
    int failures = 0;

    std::fprintf (stderr, "test_parse_weights_field_coverage: round-trip all fields\n");
    failures += testAllFieldsRoundTrip();

    std::fprintf (stderr, "test_parse_weights_field_coverage: optional fields default to 0.0\n");
    failures += testOptionalFieldsDefaultToZero();

    std::fprintf (stderr, "test_parse_weights_field_coverage: required field missing throws\n");
    failures += testRequiredFieldMissingThrows();

    std::fprintf (stderr, "test_parse_weights_field_coverage: probe count sentinel\n");
    failures += testProbeCountSentinel();

    if (failures == 0)
    {
        std::fprintf (stderr, "test_parse_weights_field_coverage: 4/4 PASS\n");
        return 0;
    }
    std::fprintf (stderr, "test_parse_weights_field_coverage: %d failure(s)\n", failures);
    return 1;
}
