#pragma once

// Sesja 98 — ADR-087 STATUS UPDATE 1 D5 + D8.
// Sesja 106 — ADR-098 STATUS UPDATE: schema bumped v1 → v2.
//   v2 drops `perceptual_sliders` field + `advanced_used` flag — raw
//   weights are now sole source of truth (PerceptualMapping deleted).
//   Legacy v1 records still load (parser ignores unknown fields).
// Sesja 107 — ADR-097 implementation: relocated src/dev/ → src/ui/, namespace
//   reamix::dev → reamix::ui. Now compiled into every build (community
//   preset sharing per ADR-096 OSS direction).
//
// JSONL append-only writer + reader for advanced weights preset sharing.
// Path: ~/Library/Application Support/reamix/dev-calibration.jsonl (same
// dir as model cache per ADR-006). One JSON record per line.

#include <juce_core/juce_core.h>

#include <vector>

#include "remix/Quality.h"

namespace reamix::ui
{

// Block Assembly β-parameters. Lives here so both AdvancedWeightsPanel and
// DevCalibrationRecord can reference the type without circular include.
// Defaults match RemixPipeline.h Input field defaults (sesja 96).
struct BlockAssemblyBeta
{
    double fragment_penalty_weight = 0.03;
    int    outside_window_beats    = 8;
    int    min_jump_beats          = 4;
    bool   downbeat_only_splices   = true;
};

// Returns true when all 4 β-params are at default.
bool blockAssemblyBetaAtDefault (const BlockAssemblyBeta& b) noexcept;

// One save record (schema_version=2 since sesja 106 / ADR-098). Mirrors
// schema spec in meta/DECISIONS.md ADR-098 § Sub-decision 3.
struct DevCalibrationRecord
{
    int                          schema_version  = 2;  // v2 since ADR-098
    juce::String                 timestamp;        // ISO-8601 UTC
    juce::String                 track_path;       // absolute path
    juce::String                 track_sha256;     // hex digest
    double                       duration_sec    = 0.0;
    double                       bpm             = 0.0;
    reamix::remix::QualityWeights weights_raw   {};
    BlockAssemblyBeta            block_assembly_beta;
    juce::String                 genre;            // lowercase: rock/pop/...
    juce::String                 user_note;        // multiline, max 500 chars
    juce::String                 mode_evaluated;   // duration / region / blocks
    juce::String                 plugin_version;
};

// Returns absolute path to the JSONL store. Creates parent directory if
// missing. Path: ~/Library/Application Support/reamix/dev-calibration.jsonl
// on macOS (same dir as model cache per ADR-006).
juce::File defaultStorePath();

class DevCalibrationStorage
{
public:
    DevCalibrationStorage() = default;
    explicit DevCalibrationStorage (juce::File store) : store_ (std::move (store)) {}

    // Append one record to the JSONL store. Returns true on success.
    // Implementation: POSIX open(O_WRONLY|O_APPEND|O_CREAT, 0644) +
    // flock(LOCK_EX) + write line + \n + fsync + flock(LOCK_UN) + close.
    // Single-record write < 4096 bytes ensures atomicity per macOS POSIX.
    bool append (const DevCalibrationRecord& rec) const;

    // Read all valid records for a given track path. Skips malformed
    // lines with a warning (partial-write recovery). Returns records in
    // file order (oldest first). Empty vector when no saves exist or file
    // missing.
    std::vector<DevCalibrationRecord> loadForTrack (const juce::String& trackPath) const;

    // Read all valid records (used by sesja 102 aggregate analysis).
    std::vector<DevCalibrationRecord> loadAll() const;

    const juce::File& storePath() const noexcept { return store_; }

    // ----------------------------------------------------------------
    // Static helpers (exposed for test fixtures + introspection).
    // ----------------------------------------------------------------

    // Serialize record → single-line JSON string (NOT trailing newline).
    // Emits schema_version=2 fields only; legacy v1 perceptual_sliders +
    // advanced_used fields not written.
    static juce::String toJsonLine (const DevCalibrationRecord& rec);

    // Parse single JSON line → record. Returns false when malformed (not
    // a JSON object, missing schema_version, etc). Legacy v1 records load
    // OK — perceptual_sliders + advanced_used fields silently ignored
    // (raw weights are always populated since v1).
    static bool fromJsonLine (const juce::String& line, DevCalibrationRecord& out);

    // Returns ISO-8601 UTC timestamp string for "now" (used by append).
    static juce::String nowIsoUtc();

private:
    juce::File store_ { defaultStorePath() };
};

} // namespace reamix::ui
