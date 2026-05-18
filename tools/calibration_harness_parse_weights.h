// tools/calibration_harness_parse_weights.h
//
// Sesja 86 ADR-076 — extracted from tools/calibration_harness.cpp so that
// tests/parity/test_parse_weights_field_coverage.cpp can exercise the same
// JSONL → QualityWeights mapping without rebuilding the harness binary.
//
// CONTRACT (binding for any new QualityWeights field):
//   1. Add the field to src/remix/Quality.h::QualityWeights struct.
//   2. Add a `if (v.hasProperty("X")) w.X = (double) v.getProperty("X", 0.0);`
//      block here (or for required-by-orchestrator fields, add to the `get`
//      block — required fields throw on missing).
//   3. Add a probe row to tests/parity/test_parse_weights_field_coverage.cpp.
//   4. Update tools/calibration/orchestrator.py::WEIGHT_FIELD_ORDER_BASE +
//      tools/calibration/preference_gp.py::WEIGHT_FIELD_ORDER in the same
//      commit so all sources (struct + harness + orchestrator + GP) stay
//      in lockstep.

#pragma once

#include "remix/Quality.h"

#include <juce_core/juce_core.h>

#include <stdexcept>
#include <string>

namespace reamix::cal {

// Parse the QualityWeights block. Caller-side validation (sum to 1.0 within
// tolerance) is the orchestrator's responsibility — we accept what we receive
// and let the cost function compute. Throwing on missing required fields makes
// orchestrator bugs loud rather than silent. Optional fields default to 0.0
// (DEV-028 sesja 74+ expressivity slots; productionised weights named
// individually as ADR-064 / ADR-066 / ADR-070 land them).
inline reamix::remix::QualityWeights parseWeights (const juce::var& v)
{
    auto get = [&] (const char* key) -> double
    {
        if (! v.hasProperty (key))
            throw std::runtime_error (std::string ("weights missing field: ") + key);
        return (double) v.getProperty (key, 0.0);
    };
    reamix::remix::QualityWeights w{};
    w.waveform    = get ("waveform");
    w.successor   = get ("successor");
    w.edge_splice = get ("edge_splice");
    w.context     = get ("context");
    w.label       = get ("label");
    w.bar_align   = get ("bar_align");
    w.section     = get ("section");
    w.energy      = get ("energy");
    w.edge_energy = get ("edge_energy");
    w.centroid    = get ("centroid");
    // Sesja 75 ADR-064: transient_continuity productionised as named field.
    if (v.hasProperty ("transient_continuity"))
        w.transient_continuity = (double) v.getProperty ("transient_continuity", 0.0);
    // Sesja 77 ADR-066: mfcc_continuity productionised as named field.
    if (v.hasProperty ("mfcc_continuity"))
        w.mfcc_continuity = (double) v.getProperty ("mfcc_continuity", 0.0);
    if (v.hasProperty ("extra1"))
        w.extra1 = (double) v.getProperty ("extra1", 0.0);
    // Sesja 81 ADR-068 D4: sequential_continuity collapse weight.
    if (v.hasProperty ("sequential_continuity"))
        w.sequential_continuity = (double) v.getProperty ("sequential_continuity", 0.0);
    // Sesja 81 ADR-068 D2: harmonic-mean composition flag.
    if (v.hasProperty ("use_harmonic_mean"))
        w.use_harmonic_mean = (bool) v.getProperty ("use_harmonic_mean", false);
    return w;
}

} // namespace reamix::cal
