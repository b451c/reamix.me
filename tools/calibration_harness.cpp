// calibration_harness — DEV-028 Stage-1 driver.
//
// Reads a JSONL batch of remix specifications (mode + weights + targets) for a
// single source audio file, executes each via RemixPipeline (Block / Region)
// or via direct CleanOptimizer w/ rebuilt TC matrix (Duration), and writes a
// rendered WAV + per-splice CSV per request. The Python orchestrator
// (tools/calibration_orchestrator.py) drives this binary with Sobol-sampled
// weight vectors during the Stage-1 coarse search.
//
// Why TWO paths for Duration vs. Block/Region (sesja-71b architecture
// decision, "Droga 1"):
//   - Block + Region build their cost inputs at REMIX-time inside
//     RemixPipeline::run, so the qualityWeightsOverride field plumbs into
//     BlockCompatInputs.quality_weights / RegionCostInputs.quality_weights
//     cleanly. We pass weights via RemixPipeline::Input.qualityWeightsOverride.
//   - Duration mode reads bundle.tc.W (Transition Cost matrix) which was
//     computed at ANALYZE-time inside AnalyzePipeline.cpp:272 with default
//     weights. The override has NO effect on the cached W. So we rebuild
//     bundle.tc by calling computeTransitionCosts(...) directly with the
//     custom weights, replace bundle.tc on a per-request basis, then drive
//     RemixPipeline through its CleanOptimizer branch (no override needed —
//     the weights are already baked into the rebuilt W).
//
// Plugin code is unchanged. Parity tests stay 51/51 PASS.
//
// Usage:
//   calibration_harness --source <abs/audio> --batch <abs/jsonl>
//
// JSONL row schema (one per remix request, see tools/calibration_orchestrator.py):
//   {"id":"r001","mode":"duration"|"region"|"blocks",
//    "weights":{"waveform":..,"successor":..,"edge_splice":..,"context":..,
//               "label":..,"bar_align":..,"section":..,"energy":..,
//               "edge_energy":..,"centroid":..},
//    "target_duration_sec": 90.0,
//    "region_start_sec": 30.0,            // region only
//    "region_end_sec":   60.0,            // region only
//    "user_blocks":[{"s":..,"e":..,"k":..}, ...],   // blocks only
//    "user_blocks_queue":[0,2,1],                    // blocks only
//    "variation": 0,
//    "out_wav": "/tmp/cal_r001.wav",
//    "out_csv": "/tmp/cal_r001.csv"}
//
// Output:
//   - WAV at out_wav (rendered remix)
//   - CSV at out_csv (one row per splice transition with diagnostic data)
//   - stderr: per-request progress + summary

#include "ui/AnalysisDiskCache.h"
#include "ui/AnalysisBundle.h"
#include "ui/RemixPipeline.h"
#include "ui/UserBlock.h"
#include "remix/Quality.h"
#include "remix/TransitionCost.h"
#include "calibration_harness_parse_weights.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Args
{
    juce::String sourcePath;
    juce::String batchPath;
    juce::String dumpBeatsJson;       // sesja 74 — optional, exits after dump
    juce::String dumpComponentsCsv;   // sesja 80 — D1 correlation matrix dump
};

int printUsage (const char* argv0)
{
    std::fprintf (stderr,
        "usage: %s --source <abs/audio> --batch <abs/jsonl>\n"
        "       %s --source <abs/audio> --dump-beats <out.json>           (sesja 74 helper)\n"
        "       %s --source <abs/audio> --dump-components <out.csv>       (sesja 80 D1 helper)\n",
        argv0, argv0, argv0);
    return 2;
}

bool parseArgs (int argc, char** argv, Args& out)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if      (a == "--source"          && i + 1 < argc) out.sourcePath          = juce::String (argv[++i]);
        else if (a == "--batch"           && i + 1 < argc) out.batchPath           = juce::String (argv[++i]);
        else if (a == "--dump-beats"      && i + 1 < argc) out.dumpBeatsJson       = juce::String (argv[++i]);
        else if (a == "--dump-components" && i + 1 < argc) out.dumpComponentsCsv   = juce::String (argv[++i]);
        else
        {
            std::fprintf (stderr, "unknown arg: %s\n", a.c_str());
            return false;
        }
    }
    if (out.sourcePath.isEmpty()) return false;
    return out.batchPath.isNotEmpty()
        || out.dumpBeatsJson.isNotEmpty()
        || out.dumpComponentsCsv.isNotEmpty();
}

// parseWeights — extracted to tools/calibration_harness_parse_weights.h
// (sesja 86 ADR-076) so test_parse_weights_field_coverage.cpp can verify
// the JSONL → QualityWeights mapping without depending on the harness binary.

// Cache the extra1 matrix per (key, n) so a 32-row batch builds/loads it
// once. Orchestrator pre-computes one matrix per cell and references it from
// every JSONL row in that cell. `key` is either an absolute file path
// (file-load path) or a "signal:<name>" tag (inline-computed path).
struct Extra1Matrix
{
    juce::String        key;
    int                 n      { 0 };
    std::vector<double> data;
};

// Compute an n×n extra1 matrix from bundle data using a named signal source.
// Returns a zero-size matrix when the signal is unknown — caller falls back to
// "no extra1 contribution" (production default).
Extra1Matrix computeExtra1Matrix (const reamix::ui::AnalysisBundle& b,
                                  const juce::String& signal)
{
    Extra1Matrix m;
    m.key = "signal:" + signal;
    m.n   = 0;

    const int n = b.feat.nBeats;
    if (n <= 0)
    {
        std::fprintf (stderr, "[harness] WARN computeExtra1Matrix: nBeats=%d\n", n);
        return m;
    }
    m.n = n;
    m.data.assign ((std::size_t) n * (std::size_t) n, 0.0);

    auto idx = [n] (int i, int j) -> std::size_t {
        return (std::size_t) i * (std::size_t) n + (std::size_t) j;
    };

    if (signal == "synthetic")
    {
        // Tier-1 generic structural-extension test: a deterministic signal
        // uncorrelated with any existing Quality.h cost term. Tests the
        // hypothesis "any 11th dimension breaks the default-Viterbi
        // attractor", isolated from candidate-specific signal coverage.
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                m.data[idx (i, j)] =
                    0.5 + 0.5 * std::sin ((double) i * 0.7) * std::cos ((double) j * 0.3);
    }
    else if (signal == "onset")
    {
        // Transient-onset signal: similarity = how aligned the splice
        // entry/exit are on transients (per ADR-062 + sesja-69 hypothesis).
        // Higher value = better alignment (= lower mismatch).
        const auto& os = b.feat.onsetStrength;
        if ((int) os.size() < n) {
            std::fprintf (stderr, "[harness] WARN onset_strength size %d < nBeats %d\n",
                          (int) os.size(), n);
            m.n = 0;
            m.data.clear();
            return m;
        }
        // Normalise to [0,1] using percentile bounds for stability across tracks.
        double mn =  std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < n; ++k) { mn = std::min (mn, os[k]); mx = std::max (mx, os[k]); }
        const double rng = std::max (1e-9, mx - mn);
        std::vector<double> osn (n);
        for (int k = 0; k < n; ++k) osn[k] = (os[k] - mn) / rng;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                m.data[idx (i, j)] = 1.0 - std::abs (osn[i] - osn[j]);
    }
    else if (signal == "ibi")
    {
        // IBI continuity: log-ratio of local IBI between exit (after i) and
        // entry (before j), with sigmoid through 5% JND knee per Ellis 2007 +
        // Quinn-Watt 2006 (research-scout sesja 74). Higher = closer tempo.
        const auto& bt = b.beatTimes;
        if ((int) bt.size() < n + 1) {
            std::fprintf (stderr, "[harness] WARN beatTimes size %d < nBeats+1 %d\n",
                          (int) bt.size(), n);
            m.n = 0;
            m.data.clear();
            return m;
        }
        auto ibiAt = [&] (int k) -> double {
            const int kk = std::clamp (k, 0, n - 2);
            return std::max (1e-6, bt[(std::size_t) kk + 1] - bt[(std::size_t) kk]);
        };
        const double JND = 0.05, SAT = 0.15;
        for (int i = 0; i < n; ++i) {
            const double ibi_l = ibiAt (i - 1);  // last IBI before splice from i
            for (int j = 0; j < n; ++j) {
                const double ibi_r = ibiAt (j);  // first IBI after splice to j
                const double r = std::abs (std::log (ibi_r) - std::log (ibi_l));
                double cost;
                if (r <= JND)      cost = 0.0;
                else if (r >= SAT) cost = 1.0;
                else               cost = (r - JND) / (SAT - JND);
                m.data[idx (i, j)] = 1.0 - cost;  // similarity, not cost
            }
        }
    }
    else if (signal == "mfcc_continuity")
    {
        // MFCC L2 + delta-MFCC L2 boundary continuity, per Stylianou-Syrdal
        // 2001 + Vepa-King 2002 (research-scout sesja 74). Uses first 13
        // dims of bundle.feat.features (per beat) as the MFCC vector.
        const auto& feat = b.feat.features;
        const int nFeat  = b.feat.nFeat;
        const int K      = std::min (13, nFeat);  // first 13 = MFCC bands
        if ((int) feat.size() < n * nFeat || nFeat <= 0 || K <= 0) {
            std::fprintf (stderr, "[harness] WARN features size %d / nFeat %d / K %d\n",
                          (int) feat.size(), nFeat, K);
            m.n = 0;
            m.data.clear();
            return m;
        }
        // Pre-compute pairwise L2 (across MFCC bands only) for normalisation.
        const double normRef = std::sqrt ((double) K);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s2_static  = 0.0;
                for (int k = 0; k < K; ++k) {
                    const double d =
                        (double) feat[(std::size_t) i * nFeat + k]
                      - (double) feat[(std::size_t) j * nFeat + k];
                    s2_static += d * d;
                }
                const double d_static = std::sqrt (s2_static) / normRef;
                // Delta proxy: difference of (i - prev) vs (next - j).
                const int ip = std::max (0, i - 1);
                const int jn = std::min (n - 1, j + 1);
                double s2_dyn = 0.0;
                for (int k = 0; k < K; ++k) {
                    const double dl =
                        (double) feat[(std::size_t) i  * nFeat + k]
                      - (double) feat[(std::size_t) ip * nFeat + k];
                    const double dr =
                        (double) feat[(std::size_t) jn * nFeat + k]
                      - (double) feat[(std::size_t) j  * nFeat + k];
                    const double dd = dl - dr;
                    s2_dyn += dd * dd;
                }
                const double d_dynamic = std::sqrt (s2_dyn) / normRef;
                // Vepa-King 2002 0.7 / 0.3 ratio. Saturation NORM_REF=0.6
                // empirical for our 22050 Hz pipeline (will be re-calibrated
                // sesja 75+ via corpus p95 if shipped).
                double raw = 0.7 * d_static + 0.3 * d_dynamic;
                raw = std::min (1.0, raw / 0.6);
                m.data[idx (i, j)] = 1.0 - raw;  // similarity
            }
        }
    }
    else
    {
        std::fprintf (stderr, "[harness] WARN unknown extra1_signal: %s\n",
                      signal.toRawUTF8());
        m.n = 0;
        m.data.clear();
        return m;
    }

    std::fprintf (stderr, "[harness] extra1 matrix computed: signal=%s n=%d\n",
                  signal.toRawUTF8(), m.n);
    return m;
}

// Load a row-major n×n double matrix from a raw binary file. Returns the
// matrix on success, or a zero-size struct on failure (caller sees `n==0`
// and falls back to the production default of "no extra1 contribution").
Extra1Matrix loadExtra1Matrix (const juce::String& absPath, int expectedN)
{
    Extra1Matrix m;
    m.key = absPath;
    m.n   = 0;

    juce::File file (absPath);
    if (! file.existsAsFile())
    {
        std::fprintf (stderr, "[harness] WARN extra1_path not a file: %s\n",
                      absPath.toRawUTF8());
        return m;
    }

    const auto sizeBytes = (juce::int64) file.getSize();
    const auto expectedBytes = (juce::int64) expectedN
                             * (juce::int64) expectedN
                             * (juce::int64) sizeof (double);
    if (sizeBytes != expectedBytes)
    {
        std::fprintf (stderr,
                      "[harness] WARN extra1 size mismatch: got %lld bytes, expected %lld (n=%d)\n",
                      (long long) sizeBytes, (long long) expectedBytes, expectedN);
        return m;
    }

    juce::FileInputStream in (file);
    if (! in.openedOk())
    {
        std::fprintf (stderr, "[harness] WARN extra1 file open failed: %s\n",
                      absPath.toRawUTF8());
        return m;
    }

    m.data.resize ((std::size_t) expectedN * (std::size_t) expectedN);
    const auto bytesRead = in.read (m.data.data(), (int) expectedBytes);
    if (bytesRead != (int) expectedBytes)
    {
        std::fprintf (stderr,
                      "[harness] WARN extra1 partial read: got %d bytes, expected %lld\n",
                      bytesRead, (long long) expectedBytes);
        m.data.clear();
        return m;
    }

    m.n = expectedN;
    std::fprintf (stderr, "[harness] extra1 matrix loaded: n=%d (%lld bytes)\n",
                  expectedN, (long long) expectedBytes);
    return m;
}

// Build TransitionCostInputs from a bundle, mirroring AnalyzePipeline.cpp:272-316.
// Caller sets `out.quality_weights` after this returns.
reamix::remix::TransitionCostInputs buildTcInputs (const reamix::ui::AnalysisBundle& b)
{
    reamix::remix::TransitionCostInputs in{};
    in.features    = b.feat.features.data();
    in.n_beats     = b.feat.nBeats;
    in.n_features  = b.feat.nFeat;
    in.beat_times  = b.beatTimes.data();

    in.segments    = b.structure.segments.data();
    in.n_segments  = (int) b.structure.segments.size();

    in.rms_energy        = b.feat.rmsEnergy.empty()        ? nullptr : b.feat.rmsEnergy.data();
    in.onset_strength    = b.feat.onsetStrength.empty()    ? nullptr : b.feat.onsetStrength.data();
    in.spectral_centroid = b.feat.spectralCentroid.empty() ? nullptr : b.feat.spectralCentroid.data();
    in.vocal_activity    = b.feat.vocalActivity.empty()    ? nullptr : b.feat.vocalActivity.data();

    const auto& bw = b.feat.boundaryWaveforms;
    if (! bw.empty() && b.feat.nBeats > 0)
    {
        in.boundary_waveforms   = bw.data();
        in.n_boundary_waveforms = b.feat.nBeats;
        in.n_samples_per_bnd    = (int) (bw.size() / (std::size_t) b.feat.nBeats);
        in.waveform_sample_rate = 22050;
    }
    in.edge_vocal_activity_start = b.feat.edgeVocalActivityStart.empty() ? nullptr : b.feat.edgeVocalActivityStart.data();
    in.edge_vocal_activity_end   = b.feat.edgeVocalActivityEnd.empty()   ? nullptr : b.feat.edgeVocalActivityEnd.data();

    in.edge_rms_start = b.feat.edgeRmsStart.empty() ? nullptr : b.feat.edgeRmsStart.data();
    in.edge_rms_end   = b.feat.edgeRmsEnd.empty()   ? nullptr : b.feat.edgeRmsEnd.data();

    in.edge_features_start = b.feat.edgeFeaturesStart.empty() ? nullptr
                              : reinterpret_cast<const float*> (b.feat.edgeFeaturesStart.data());
    in.edge_features_end   = b.feat.edgeFeaturesEnd.empty()   ? nullptr
                              : reinterpret_cast<const float*> (b.feat.edgeFeaturesEnd.data());
    in.n_edge_features     = b.feat.edgeFeaturesStart.empty() ? 0 : b.feat.nFeat;

    in.downbeats   = b.downbeatTimes.empty() ? nullptr : b.downbeatTimes.data();
    in.n_downbeats = (int) b.downbeatTimes.size();

    in.time_signature = juce::jmax (1, (int) b.timeSigNum);

    return in;
}

// One remix request, parsed from a JSONL row.
struct Run
{
    juce::String id;
    juce::String mode;     // "duration" | "region" | "blocks"
    reamix::remix::QualityWeights weights;
    double       targetSec        { 0.0 };
    std::optional<double> regStart;
    std::optional<double> regEnd;
    std::vector<reamix::ui::UserBlock> userBlocks;
    std::vector<int>                   userBlocksQueue;
    int          variation        { 0 };
    juce::String outWav;
    juce::String outCsv;

    // DEV-028 sesja 74 expressivity slot — EITHER a path to a pre-computed
    // n×n matrix file (MERT/MuQ Python pre-compute path) OR a signal name
    // for inline C++ computation (synthetic / onset / ibi / mfcc_continuity
    // / etc; cheap signals derivable from existing bundle data). All rows
    // in a single batch typically share the same source — harness caches
    // by (key, n) where key="signal:<name>" or absolute path.
    juce::String extra1Path;
    juce::String extra1Signal;
    int          extra1N        { 0 };

    // Sesja 81 ADR-068 D3 — optional Duration-mode quality_floor override.
    // Block/Region modes use AnalyzePipeline-cached bundle.tc which baked in
    // the floor at warm-cache time; Duration mode rebuilds tc here, so this
    // override threads into TransitionCostInputs.quality_floor. Defaults to
    // QUALITY_HARD_FLOOR (= 0.20 post-D3-flip). For "before" listening A/B
    // pass `LEGACY_QUALITY_HARD_FLOOR = 0.45` explicitly via JSONL.
    std::optional<double> qualityFloor;
};

Run parseRun (const juce::var& v)
{
    Run r;
    r.id   = v.getProperty ("id",   juce::String()).toString();
    r.mode = v.getProperty ("mode", juce::String()).toString();
    if (r.mode != "duration" && r.mode != "region" && r.mode != "blocks")
        throw std::runtime_error ("mode must be duration|region|blocks (got: "
                                  + r.mode.toStdString() + ")");

    r.weights = reamix::cal::parseWeights (v.getProperty ("weights", juce::var()));
    r.targetSec = (double) v.getProperty ("target_duration_sec", 0.0);
    if (v.hasProperty ("quality_floor"))
        r.qualityFloor = (double) v.getProperty ("quality_floor", 0.0);

    if (r.mode == "region")
    {
        if (! v.hasProperty ("region_start_sec") || ! v.hasProperty ("region_end_sec"))
            throw std::runtime_error ("region mode requires region_start_sec + region_end_sec");
        r.regStart = (double) v.getProperty ("region_start_sec", 0.0);
        r.regEnd   = (double) v.getProperty ("region_end_sec",   0.0);
    }
    if (r.mode == "blocks")
    {
        const auto blocksVar = v.getProperty ("user_blocks", juce::var());
        if (! blocksVar.isArray() || blocksVar.size() < 2)
            throw std::runtime_error ("blocks mode requires user_blocks[] with ≥ 2 entries");
        for (int i = 0; i < blocksVar.size(); ++i)
        {
            if (auto ub = reamix::ui::userBlockFromVar (blocksVar[i]); ub.has_value())
                r.userBlocks.push_back (*ub);
        }
        const auto queueVar = v.getProperty ("user_blocks_queue", juce::var());
        if (! queueVar.isArray() || queueVar.size() < 2)
            throw std::runtime_error ("blocks mode requires user_blocks_queue[] with ≥ 2 entries");
        for (int i = 0; i < queueVar.size(); ++i)
            r.userBlocksQueue.push_back ((int) queueVar[i]);
    }
    r.variation = (int) v.getProperty ("variation", 0);
    r.outWav = v.getProperty ("out_wav", juce::String()).toString();
    r.outCsv = v.getProperty ("out_csv", juce::String()).toString();
    if (r.outWav.isEmpty() || r.outCsv.isEmpty())
        throw std::runtime_error ("out_wav and out_csv are required");

    // Optional extra1 matrix (sesja 74 expressivity pre-check). Two sources:
    if (v.hasProperty ("extra1_signal"))
    {
        r.extra1Signal = v.getProperty ("extra1_signal", juce::String()).toString();
    }
    if (v.hasProperty ("extra1_path"))
    {
        r.extra1Path = v.getProperty ("extra1_path", juce::String()).toString();
        r.extra1N    = (int) v.getProperty ("extra1_n", 0);
        if (r.extra1Path.isNotEmpty() && r.extra1N <= 0)
            throw std::runtime_error ("extra1_n must be > 0 when extra1_path is set");
    }
    if (r.extra1Signal.isNotEmpty() && r.extra1Path.isNotEmpty())
        throw std::runtime_error ("extra1_signal and extra1_path are mutually exclusive");

    return r;
}

// Drive RemixPipeline to completion. Returns the populated RemixOutput on
// success; on failure ok=false + errorMessage set.
reamix::ui::RemixOutput driveRemixPipeline (
    reamix::ui::AnalysisBundlePtr bundle,
    const Run&                    run,
    bool                          useOverride)
{
    reamix::ui::RemixPipeline::Input pin{};
    pin.bundle             = bundle;
    pin.targetDurationSec  = run.targetSec;
    pin.variation          = run.variation;
    if (run.regStart.has_value()) pin.regionStartSec = *run.regStart;
    if (run.regEnd.has_value())   pin.regionEndSec   = *run.regEnd;
    pin.userBlocks         = run.userBlocks;
    pin.userBlocksQueue    = run.userBlocksQueue;
    if (useOverride)
        pin.qualityWeightsOverride = run.weights;

    std::atomic<bool>          done { false };
    reamix::ui::RemixOutput    result;
    auto progressCb = [] (juce::String, double) {};
    auto completeCb = [&] (reamix::ui::RemixOutput out)
    {
        result = std::move (out);
        done.store (true, std::memory_order_release);
    };

    auto pipeline = std::make_unique<reamix::ui::RemixPipeline> (
        std::move (pin), std::move (progressCb), std::move (completeCb), run.outWav);
    pipeline->startThread();

    while (! done.load (std::memory_order_acquire))
        juce::MessageManager::getInstance()->runDispatchLoopUntil (50);

    pipeline.reset();
    return result;
}

// Emit per-splice CSV. Mirrors RemixOutput.transition* parallel arrays.
bool writeSplicesCsv (const juce::String& path,
                      const reamix::ui::RemixOutput& out,
                      const reamix::ui::AnalysisBundle& bundle)
{
    juce::File f (path);
    f.getParentDirectory().createDirectory();
    juce::FileOutputStream stream (f);
    if (! stream.openedOk()) return false;
    stream.setNewLineString ("\n");
    stream.setPosition (0);
    stream.truncate();

    stream << "splice_idx,from_beat,to_beat,from_time_sec,to_time_sec,"
              "splice_remix_time_sec,quality,energy_diff_db,from_label,to_label\n";

    const auto& bt = bundle.beatTimes;
    auto safeBt = [&] (int b) -> double
    {
        if (b < 0 || (std::size_t) b >= bt.size()) return 0.0;
        return bt[(std::size_t) b];
    };

    for (int i = 0; i < out.nTransitions; ++i)
    {
        const int    fb     = (i < (int) out.transitionFromBeats.size())     ? out.transitionFromBeats[i]     : -1;
        const int    tb     = (i < (int) out.transitionToBeats.size())       ? out.transitionToBeats[i]       : -1;
        const double remT   = (i < (int) out.transitionTimesSec.size())      ? out.transitionTimesSec[i]      : 0.0;
        const double q      = (i < (int) out.transitionQualities.size())     ? (double) out.transitionQualities[i] : 0.0;
        const double edb    = (i < (int) out.transitionEnergyDiffsDb.size()) ? (double) out.transitionEnergyDiffsDb[i] : 0.0;
        const auto   fromLb = (i < (int) out.transitionFromLabels.size())    ? out.transitionFromLabels[i]    : juce::String();
        const auto   toLb   = (i < (int) out.transitionToLabels.size())      ? out.transitionToLabels[i]      : juce::String();

        stream << i << ','
               << fb << ',' << tb << ','
               << juce::String (safeBt (fb), 6) << ','
               << juce::String (safeBt (tb), 6) << ','
               << juce::String (remT,        6) << ','
               << juce::String (q,           6) << ','
               << juce::String (edb,         3) << ','
               << fromLb << ',' << toLb << '\n';
    }
    return true;
}

} // namespace

int main (int argc, char** argv)
{
    Args args;
    if (! parseArgs (argc, argv, args)) return printUsage (argv[0]);

    juce::ScopedJuceInitialiser_GUI juceInit;

    // Bundle is loaded ONCE per harness invocation. The orchestrator batches
    // per-source so this disk-load + audio-decode cost is amortised across
    // hundreds-to-thousands of weight vectors.
    std::fprintf (stderr, "loading cached bundle for %s\n",
                  args.sourcePath.toRawUTF8());
    auto bundle = reamix::ui::AnalysisDiskCache::tryLoad (args.sourcePath);
    if (bundle == nullptr)
    {
        std::fprintf (stderr, "  ERROR: no cache entry — run warm_cache first\n");
        return 1;
    }
    std::fprintf (stderr, "  OK %d beats · %.1f BPM · %d native samples\n",
                  (int) bundle->beatTimes.size(), bundle->bpm,
                  (int) bundle->nativeSamples);

    // sesja 74 helper: --dump-beats writes beat_times to JSON and exits.
    // Used by precompute_extra1_mert.py to align Python embeddings with the
    // exact beat indexing the harness will use during the sweep.
    if (args.dumpBeatsJson.isNotEmpty())
    {
        juce::File outf (args.dumpBeatsJson);
        outf.getParentDirectory().createDirectory();
        juce::FileOutputStream s (outf);
        if (! s.openedOk())
        {
            std::fprintf (stderr, "  ERROR: cannot open %s\n",
                          args.dumpBeatsJson.toRawUTF8());
            return 1;
        }
        s.setNewLineString ("\n");
        s.setPosition (0);
        s.truncate();
        s << "{\n  \"n_beats\": " << (int) bundle->beatTimes.size() << ",\n";
        s << "  \"bpm\": " << juce::String (bundle->bpm, 4) << ",\n";
        s << "  \"beat_times\": [";
        for (std::size_t i = 0; i < bundle->beatTimes.size(); ++i)
        {
            if (i > 0) s << ", ";
            s << juce::String (bundle->beatTimes[i], 6);
        }
        s << "]\n}\n";
        std::fprintf (stderr, "[harness] wrote %s (%d beats)\n",
                      args.dumpBeatsJson.toRawUTF8(),
                      (int) bundle->beatTimes.size());
        return 0;
    }

    // sesja 80 helper: --dump-components writes per-surviving-pair component
    // values to CSV and exits. ADR-068 § D1 — Pearson correlation matrix
    // input. Runs computeTransitionCosts with default weights, iterates the
    // surviving-pairs candidate map, dumps all 12 cost components per pair.
    if (args.dumpComponentsCsv.isNotEmpty())
    {
        std::fprintf (stderr, "[harness] dump-components: building TC inputs...\n");
        auto tcin = buildTcInputs (*bundle);
        // Default weights — we want the production gate stack (chroma_prefilter
        // 0.45 + energy_hard_block 8 dB + composite HARD_FLOOR 0.45) so the
        // surviving-pair pool matches the production Viterbi candidate space.
        std::fprintf (stderr, "[harness] dump-components: running computeTransitionCosts...\n");
        auto tc = reamix::remix::computeTransitionCosts (tcin);
        std::fprintf (stderr, "[harness] dump-components: %d surviving pairs\n",
                      (int) tc.candidates.size());

        juce::File outf (args.dumpComponentsCsv);
        outf.getParentDirectory().createDirectory();
        juce::FileOutputStream s (outf);
        if (! s.openedOk())
        {
            std::fprintf (stderr, "  ERROR: cannot open %s\n",
                          args.dumpComponentsCsv.toRawUTF8());
            return 1;
        }
        s.setNewLineString ("\n");
        s.setPosition (0);
        s.truncate();
        // Header: 12 components + composite quality + chroma_distance + energy_diff_db
        s << "from_beat,to_beat,quality,"
             "waveform,successor,edge_splice,context,label,section,bar_align,"
             "energy,edge_energy,centroid,transient_continuity,mfcc_continuity,"
             "chroma_distance,energy_diff_db\n";
        for (const auto& kv : tc.candidates)
        {
            const auto& c = kv.second;
            s << c.from_beat << ',' << c.to_beat << ','
              << juce::String (c.quality_score,         6) << ','
              << juce::String (c.waveform_similarity,   6) << ','
              << juce::String (c.successor_similarity,  6) << ','
              << juce::String (c.edge_splice_similarity,6) << ','
              << juce::String (c.context_similarity,    6) << ','
              << juce::String (c.label_match,           6) << ','
              << juce::String (c.section_similarity,    6) << ','
              << juce::String (c.bar_aligned,           6) << ','
              << juce::String (c.energy_match,          6) << ','
              << juce::String (c.edge_energy_match,     6) << ','
              << juce::String (c.centroid_match,        6) << ','
              << juce::String (c.transient_continuity,  6) << ','
              << juce::String (c.mfcc_continuity,       6) << ','
              << juce::String (c.chroma_distance,       6) << ','
              << juce::String (c.energy_diff_db,        3) << '\n';
        }
        std::fprintf (stderr, "[harness] wrote %s (%d pairs)\n",
                      args.dumpComponentsCsv.toRawUTF8(),
                      (int) tc.candidates.size());
        return 0;
    }

    std::ifstream batch (args.batchPath.toStdString());
    if (! batch.is_open())
    {
        std::fprintf (stderr, "  ERROR: cannot open batch file %s\n",
                      args.batchPath.toRawUTF8());
        return 1;
    }

    int nOk = 0, nFailed = 0;
    int rowIdx = 0;
    const auto wallStart = std::chrono::steady_clock::now();

    // Cache extra1 matrices loaded from disk so a multi-row batch hits one
    // load only. Key = (path, n) so distinct sources don't alias.
    std::vector<Extra1Matrix> extra1Cache;

    std::string line;
    while (std::getline (batch, line))
    {
        ++rowIdx;
        // Skip blank lines + comments (lines starting with '#').
        const auto trimmed = juce::String (line).trim();
        if (trimmed.isEmpty() || trimmed.startsWith ("#")) continue;

        Run run;
        try
        {
            const juce::var v = juce::JSON::parse (trimmed);
            if (! v.isObject())
                throw std::runtime_error ("row is not a JSON object");
            run = parseRun (v);
        }
        catch (const std::exception& e)
        {
            std::fprintf (stderr, "[row %d] PARSE ERROR: %s\n", rowIdx, e.what());
            ++nFailed;
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();

        // Look up (or build/load on first use) the extra1 matrix for this row.
        const Extra1Matrix* extra1 = nullptr;
        const juce::String  extra1Key =
            run.extra1Signal.isNotEmpty() ? juce::String ("signal:") + run.extra1Signal
                                           : run.extra1Path;
        if (extra1Key.isNotEmpty())
        {
            const int wantN = run.extra1Signal.isNotEmpty() ? bundle->feat.nBeats
                                                              : run.extra1N;
            for (const auto& m : extra1Cache)
            {
                if (m.key == extra1Key && m.n == wantN)
                {
                    extra1 = &m;
                    break;
                }
            }
            if (extra1 == nullptr)
            {
                if (run.extra1Signal.isNotEmpty())
                    extra1Cache.push_back (computeExtra1Matrix (*bundle, run.extra1Signal));
                else
                    extra1Cache.push_back (loadExtra1Matrix (run.extra1Path, run.extra1N));
                if (extra1Cache.back().n > 0)
                    extra1 = &extra1Cache.back();
            }
        }

        // Duration mode: rebuild bundle.tc with custom weights. RemixPipeline
        // then runs CleanOptimizer reading the freshly-built W matrix.
        bool useOverride;
        try
        {
            if (run.mode == "duration")
            {
                auto tcin = buildTcInputs (*bundle);
                tcin.quality_weights = &run.weights;
                if (run.qualityFloor.has_value())
                    tcin.quality_floor = *run.qualityFloor;
                if (extra1 != nullptr && extra1->n == bundle->feat.nBeats)
                {
                    tcin.extra1_per_pair = extra1->data.data();
                    tcin.extra1_n        = extra1->n;
                }
                else if (extra1 != nullptr)
                {
                    std::fprintf (stderr,
                                  "[row %d %s] WARN extra1 n=%d != bundle nBeats=%d — "
                                  "ignoring extra1\n",
                                  rowIdx, run.id.toRawUTF8(), extra1->n,
                                  (int) bundle->feat.nBeats);
                }
                bundle->tc = reamix::remix::computeTransitionCosts (tcin);
                useOverride = false;
            }
            else
            {
                useOverride = true;  // Block + Region read via override
            }
        }
        catch (const std::exception& e)
        {
            std::fprintf (stderr, "[row %d %s] TC REBUILD FAILED: %s\n",
                          rowIdx, run.id.toRawUTF8(), e.what());
            ++nFailed;
            continue;
        }

        auto out = driveRemixPipeline (bundle, run, useOverride);
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double> (t1 - t0).count();

        if (! out.ok)
        {
            std::fprintf (stderr, "[row %d %s %s] PIPELINE FAILED (%.2fs): %s\n",
                          rowIdx, run.id.toRawUTF8(), run.mode.toRawUTF8(), sec,
                          out.errorMessage.toRawUTF8());
            ++nFailed;
            continue;
        }

        if (! writeSplicesCsv (run.outCsv, out, *bundle))
        {
            std::fprintf (stderr, "[row %d %s] CSV WRITE FAILED to %s\n",
                          rowIdx, run.id.toRawUTF8(), run.outCsv.toRawUTF8());
            ++nFailed;
            continue;
        }

        std::fprintf (stderr,
                      "[row %d %s %s] OK %.2fs · %d splices · %.1fs remix\n",
                      rowIdx, run.id.toRawUTF8(), run.mode.toRawUTF8(),
                      sec, out.nTransitions, out.remixDurationSec);
        ++nOk;
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    const double wallTotal = std::chrono::duration<double> (wallEnd - wallStart).count();

    std::fprintf (stderr, "\n=== summary ===\n");
    std::fprintf (stderr, "ok=%d failed=%d wall=%.1fs\n", nOk, nFailed, wallTotal);
    return (nFailed > 0) ? 1 : 0;
}
