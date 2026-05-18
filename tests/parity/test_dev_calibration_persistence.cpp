// test_dev_calibration_persistence — sesja 98 (ADR-087 STATUS UPDATE 1) self-validation.
// Sesja 106 (ADR-098): perceptual mapping tests removed; schema bumped v1→v2.
// Sesja 107 (ADR-097): relocated src/dev/ → src/ui/, namespace reamix::dev →
//   reamix::ui, compiled into every build (production dylib ships the panel
//   as opt-in window per ADR-097).
//
// Validates advanced weights JSONL schema (D5) + storage round-trip +
// atDefault sentinel semantics + Block Assembly β-default helper.
//
// Per ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`:
// hand-computed invariants, no Python ground truth (advanced weights is
// C++-canonical infrastructure).

#include "ui/DevCalibrationStorage.h"
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

bool testBlockAssemblyBetaAtDefault()
{
    reamix::ui::BlockAssemblyBeta b;
    CHECK (reamix::ui::blockAssemblyBetaAtDefault (b),
           "default-constructed β should be at default");
    b.outside_window_beats = 16;
    CHECK (! reamix::ui::blockAssemblyBetaAtDefault (b),
           "modified β should not be at default");
    return true;
}

bool testJsonRoundTrip()
{
    using reamix::ui::DevCalibrationStorage;
    using reamix::ui::DevCalibrationRecord;

    DevCalibrationRecord rec;
    rec.timestamp           = "2026-05-06T15:30:00Z";
    rec.track_path          = "/path/to/song.flac";
    rec.track_sha256        = "abc123def456";
    rec.duration_sec        = 215.4;
    rec.bpm                 = 120.5;
    rec.weights_raw         = reamix::remix::kDefaultQualityWeights;
    rec.weights_raw.waveform = 0.55;  // tweak to verify round-trip
    rec.block_assembly_beta.fragment_penalty_weight = 0.05;
    rec.block_assembly_beta.outside_window_beats = 12;
    rec.block_assembly_beta.min_jump_beats = 6;
    rec.block_assembly_beta.downbeat_only_splices = false;
    rec.genre               = "rock";
    rec.user_note           = "test note ąęłóż";
    rec.mode_evaluated      = "blocks";
    rec.plugin_version      = "0.7.0-dev";

    const auto line = DevCalibrationStorage::toJsonLine (rec);
    CHECK (! line.contains ("\n"), "JSONL line must be single-line");
    CHECK (line.startsWith ("{"), "JSONL line must be a JSON object");

    DevCalibrationRecord parsed;
    CHECK (DevCalibrationStorage::fromJsonLine (line, parsed),
           "round-trip parse should succeed");

    CHECK (parsed.schema_version == 2, "schema_version v2 preserved");
    CHECK (parsed.timestamp == rec.timestamp, "timestamp preserved");
    CHECK (parsed.track_path == rec.track_path, "track_path preserved");
    CHECK (parsed.track_sha256 == rec.track_sha256, "track_sha256 preserved");
    CHECK (std::abs (parsed.duration_sec - rec.duration_sec) < 1e-6,
           "duration_sec preserved");
    CHECK (std::abs (parsed.bpm - rec.bpm) < 1e-6, "bpm preserved");
    CHECK (std::abs (parsed.weights_raw.waveform - rec.weights_raw.waveform) < 1e-9,
           "waveform weight preserved");
    CHECK (std::abs (parsed.block_assembly_beta.fragment_penalty_weight - 0.05) < 1e-6,
           "β fragment_penalty_weight preserved");
    CHECK (parsed.block_assembly_beta.outside_window_beats == 12,
           "β outside_window_beats preserved");
    CHECK (parsed.block_assembly_beta.downbeat_only_splices == false,
           "β downbeat_only_splices preserved");
    CHECK (parsed.genre == "rock", "genre preserved");
    CHECK (parsed.user_note == rec.user_note, "user_note preserved (utf-8 incl)");
    CHECK (parsed.mode_evaluated == "blocks", "mode_evaluated preserved");
    return true;
}

bool testLegacyV1RecordLoads()
{
    // Sesja 106 ADR-098: verify backward compat — v1 records with
    // perceptual_sliders + advanced_used fields still load (unknown fields
    // silently ignored; raw weights are always populated since v1).
    const juce::String v1Line = R"({"schema_version":1,)"
        R"("timestamp":"2026-05-06T10:00:00Z","track_path":"/x.flac","track_sha256":"",)"
        R"("duration_sec":0.0,"bpm":0.0,)"
        R"("weights_raw":{"waveform":0.55,"sequential_continuity":0.20,)"
        R"("transient_continuity":0.15,"energy":0.07,"edge_energy":0.04,)"
        R"("bar_align":0.02,"centroid":0.02,"vocal_continuity":0.0},)"
        R"("perceptual_sliders":{"audio_fit":0.7,"transient_fit":0.5,)"
        R"("energy_fit":0.5,"downbeat_hold":0.5},)"
        R"("block_assembly_beta":{"fragment_penalty_weight":0.03,)"
        R"("outside_window_beats":8,"min_jump_beats":4,"downbeat_only_splices":true},)"
        R"("advanced_used":true,"genre":"rock","user_note":"","mode_evaluated":"duration",)"
        R"("plugin_version":"0.7.0"})";

    reamix::ui::DevCalibrationRecord parsed;
    CHECK (reamix::ui::DevCalibrationStorage::fromJsonLine (v1Line, parsed),
           "legacy v1 record must parse OK");
    CHECK (parsed.schema_version == 1, "schema_version field carried through");
    CHECK (std::abs (parsed.weights_raw.waveform - 0.55) < 1e-9,
           "v1 weights_raw.waveform preserved");
    CHECK (parsed.block_assembly_beta.outside_window_beats == 8,
           "v1 β preserved");
    CHECK (parsed.genre == "rock", "v1 genre preserved");
    return true;
}

bool testStorageAppendAndLoad()
{
    using reamix::ui::DevCalibrationStorage;
    using reamix::ui::DevCalibrationRecord;

    auto tempDir = juce::File::getSpecialLocation (juce::File::tempDirectory);
    auto storeFile = tempDir.getNonexistentChildFile ("reamix-dev-calib-test", ".jsonl", false);
    storeFile.deleteFile();

    DevCalibrationStorage storage (storeFile);

    DevCalibrationRecord rec1;
    rec1.timestamp = "2026-05-06T10:00:00Z";
    rec1.track_path = "/path/A.flac";
    rec1.weights_raw = reamix::remix::kDefaultQualityWeights;
    rec1.genre = "rock";
    rec1.plugin_version = "0.7.0";

    DevCalibrationRecord rec2;
    rec2.timestamp = "2026-05-06T11:00:00Z";
    rec2.track_path = "/path/B.flac";
    rec2.weights_raw = reamix::remix::kDefaultQualityWeights;
    rec2.weights_raw.waveform = 0.40;
    rec2.weights_raw.sequential_continuity = 0.30;  // shifted from default
    rec2.genre = "pop";
    rec2.plugin_version = "0.7.0";

    DevCalibrationRecord rec3;
    rec3.timestamp = "2026-05-06T12:00:00Z";
    rec3.track_path = "/path/A.flac";  // same as rec1
    rec3.weights_raw = reamix::remix::kDefaultQualityWeights;
    rec3.weights_raw.transient_continuity = 0.25;  // shifted
    rec3.genre = "metal";
    rec3.plugin_version = "0.7.0";

    CHECK (storage.append (rec1), "append rec1");
    CHECK (storage.append (rec2), "append rec2");
    CHECK (storage.append (rec3), "append rec3");

    auto all = storage.loadAll();
    CHECK (all.size() == 3, "loadAll returns 3 records");
    CHECK (all[0].track_path == "/path/A.flac", "rec1 first");
    CHECK (all[1].track_path == "/path/B.flac", "rec2 second");
    CHECK (all[2].track_path == "/path/A.flac", "rec3 third");

    auto trackA = storage.loadForTrack ("/path/A.flac");
    CHECK (trackA.size() == 2, "loadForTrack /path/A returns 2");
    CHECK (trackA[0].genre == "rock", "first A entry rock");
    CHECK (trackA[1].genre == "metal", "second A entry metal");

    storeFile.deleteFile();
    return true;
}

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
        { "blockAssemblyBetaAtDefault",        testBlockAssemblyBetaAtDefault },
        { "jsonRoundTrip",                     testJsonRoundTrip },
        { "legacyV1RecordLoads",               testLegacyV1RecordLoads },
        { "storageAppendAndLoad",              testStorageAppendAndLoad },
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
