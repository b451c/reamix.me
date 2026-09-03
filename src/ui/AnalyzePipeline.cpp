#include "ui/AnalyzePipeline.h"
#include "analysis/GridConsistency.h"   // DEV-088 (sesja 124)
#include "ui/LoopSpotsBuilder.h"   // ADR-115 E11 (sesja 117)

#include <juce_audio_formats/juce_audio_formats.h>

#include "analysis/BeatDetector.h"
#include "analysis/SectionClassifier.h"
#include "analysis/FeatureExtractor.h"
#include "analysis/StructureResult.h"
#include "remix/BeatGrid.h"   // ADR-115 E5 (sesja 115)
#include "io/AudioLoader.h"
#include "remix/TransitionCost.h"

#include <algorithm>
#include <cmath>

namespace reamix::ui
{

namespace
{
    constexpr int kAnalysisSampleRate = 22050;

    // Stage-budget remap per ADR-047 § 2 — AnalyzePipeline owns its own
    // [0.0, 1.0] progress range. The former stage 4 (CBM / novelty
    // structure stack) was skipped per ADR-044 and deleted in sesja 122;
    // its budget is folded into stage 5 (TransitionCost).
    constexpr double kPLoad         = 0.05;
    constexpr double kPBeatDetect   = 0.45;
    constexpr double kPFeatures     = 0.70;
    constexpr double kPTransitions  = 1.00;

    std::vector<double> toDoubles (const std::vector<float>& v)
    {
        std::vector<double> out (v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = (double) v[i];
        return out;
    }

    // For each beat, check whether any downbeat time falls within half a beat
    // period — port of the Lua / Python beat_is_downbeat semantic used by the
    // WaveformView grid.
}

AnalyzePipeline::AnalyzePipeline (Input                 in,
                                  reamix::BeatDetector& beatDetector,
                                  ProgressCb            onProgress,
                                  CompleteCb            onComplete,
                                  reamix::analysis::SectionClassifier* sections)
    : juce::Thread ("reamix.analyze"),
      in_ (std::move (in)),
      beatDetector_ (beatDetector),
      sections_ (sections),
      onProgress_ (std::move (onProgress)),
      onComplete_ (std::move (onComplete)),
      alive_ (std::make_shared<std::atomic<bool>> (true))
{
}

AnalyzePipeline::~AnalyzePipeline()
{
    alive_->store (false);
    stopThread (5000);
}

void AnalyzePipeline::postProgress (const juce::String& step, double p01)
{
    if (! onProgress_) return;
    const double clamped = juce::jlimit (0.0, 1.0, p01);
    juce::String s = step;
    auto alive = alive_;
    ProgressCb cb = onProgress_;
    juce::MessageManager::callAsync ([cb, s, clamped, alive]
    {
        if (alive && alive->load()) cb (s, clamped);
    });
}

void AnalyzePipeline::postCompletion (AnalysisBundlePtr bundle, const juce::String& error)
{
    if (! onComplete_) return;
    auto alive = alive_;
    CompleteCb cb = onComplete_;
    juce::MessageManager::callAsync ([cb, bundle, error, alive]() mutable
    {
        if (alive && alive->load()) cb (std::move (bundle), error);
    });
}

bool AnalyzePipeline::loadAudio (const juce::String& path,
                                 LoadedAudio&        out,
                                 juce::String&       err)
{
    juce::File f (path);
    if (! f.existsAsFile())
    {
        err = "Source file not found: " + path;
        return false;
    }

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (f));
    if (reader == nullptr)
    {
        err = "Unable to decode audio file: " + path;
        return false;
    }

    const int nativeSr  = (int) reader->sampleRate;
    const int nChannels = juce::jmax (1, juce::jmin (2, (int) reader->numChannels));
    const juce::int64 totalSamples = reader->lengthInSamples;
    if (nativeSr <= 0 || totalSamples <= 0)
    {
        err = "Audio file has zero length or invalid sample rate";
        return false;
    }

    out.nChannels     = nChannels;
    out.nativeSr      = nativeSr;
    out.nativeSamples = (std::size_t) totalSamples;

    juce::AudioBuffer<float> buf (nChannels, (int) totalSamples);
    if (! reader->read (&buf, 0, (int) totalSamples, 0,
                        /*useLeft*/  nChannels >= 1,
                        /*useRight*/ nChannels >= 2))
    {
        err = "Audio read failed";
        return false;
    }

    out.stereoNative.resize ((std::size_t) nChannels * (std::size_t) totalSamples);
    for (int ch = 0; ch < nChannels; ++ch)
    {
        const float* src = buf.getReadPointer (ch);
        std::copy (src, src + totalSamples,
                   out.stereoNative.begin() + (std::size_t) ch * (std::size_t) totalSamples);
    }

    // Mono downmix → resample to 22050 for DSP stages.
    std::vector<float> monoNative ((std::size_t) totalSamples, 0.0f);
    if (nChannels == 1)
    {
        std::copy (out.stereoNative.begin(),
                   out.stereoNative.begin() + (std::ptrdiff_t) totalSamples,
                   monoNative.begin());
    }
    else
    {
        const float* L = out.stereoNative.data();
        const float* R = out.stereoNative.data() + totalSamples;
        for (juce::int64 i = 0; i < totalSamples; ++i)
            monoNative[(std::size_t) i] = 0.5f * (L[i] + R[i]);
    }

    out.mono22050 = reamix::AudioLoader::resample (monoNative, nativeSr, kAnalysisSampleRate);
    return true;
}

void AnalyzePipeline::run()
{
    auto bundle = std::make_shared<AnalysisBundle>();
    bundle->sourcePath = in_.sourcePath;

    // ── Stage 1 — load audio ────────────────────────────────────────
    postProgress ("Loading audio", 0.0);
    LoadedAudio audio;
    juce::String err;
    if (! loadAudio (in_.sourcePath, audio, err))
    {
        postCompletion (nullptr, err);
        return;
    }
    if (threadShouldExit()) return;

    bundle->stereoNative   = std::move (audio.stereoNative);
    bundle->nChannels      = audio.nChannels;
    bundle->nativeSr       = audio.nativeSr;
    bundle->nativeSamples  = audio.nativeSamples;

    // ── Stage 2 — BeatDetector (phase-1) ────────────────────────────
    postProgress ("Detecting beats", kPLoad);
    reamix::DetectionResult det = beatDetector_.detect (
        audio.mono22050, kAnalysisSampleRate,
        [this] (const std::string& label, float p01)
        {
            const double mapped = kPLoad + (kPBeatDetect - kPLoad) * juce::jlimit (0.0f, 1.0f, p01);
            postProgress (juce::String::fromUTF8 (label.c_str()), mapped);
        });

    if (! det.error.empty())
    {
        postCompletion (nullptr, "Beat detection failed: " + juce::String (det.error));
        return;
    }
    if (det.beats.empty())
    {
        postCompletion (nullptr, "No beats detected — audio may be silence or too short");
        return;
    }
    if (threadShouldExit()) return;

    // DEV-088 (sesja 124) — the bundle carries a consistent grid: phase-
    // consistent gaps filled, downbeats on the grid (off-grid / hole-
    // adjacent dropped), bar length measured, tempo = the grid's own
    // (the detector's octave-corrected BPM could be 2x the beat rate).
    {
        const auto grid = reamix::analysis::makeConsistentGrid (
            toDoubles (det.beats), toDoubles (det.downbeats), (int) det.timeSigNum);
        bundle->bpm            = grid.bpm;
        bundle->timeSigNum     = grid.bar_beats;
        bundle->beatTimes      = grid.beats;
        bundle->downbeatTimes  = grid.downbeats;
        bundle->beatIsDownbeat = grid.beatIsDownbeat;
    }

    // ── Stage 2b — SectionClassifier (sesja 121, DEV-098) ──────────
    // LinkSeg sections from the full mix + the beat grid, raw model times
    // and labels into bundle.structure (cached on disk, format 7);
    // LoopSpotsBuilder::ensureBeatGrid snaps them to the cleaned grid for
    // the UI. ~1 s CPU for a 5-minute track. Failure = no sections.
    bundle->structure = {};
    if (sections_ != nullptr && sections_->isLoaded())
    {
        postProgress ("Finding sections", kPBeatDetect);
        std::string sectionErr;
        const auto modelOut = sections_->run (audio.mono22050.data(), audio.mono22050.size(),
                                              bundle->beatTimes, &sectionErr);
        if (! modelOut.beatTimes.empty())
        {
            const double durationSec = (double) audio.mono22050.size() / kAnalysisSampleRate;
            for (const auto& s : reamix::analysis::SectionClassifier::decode (modelOut, durationSec))
            {
                reamix::analysis::Segment seg;
                seg.start      = s.start;
                seg.end        = s.end;
                seg.confidence = s.confidence;
                seg.cluster_id = s.cls;
                seg.label      = s.label;
                bundle->structure.segments.push_back (std::move (seg));
                bundle->structure.boundaries.push_back (s.start);
            }
        }
    }
    if (threadShouldExit()) return;

    // ── Stage 3 — FeatureExtractor (phase-2) ───────────────────────
    postProgress ("Extracting features", kPBeatDetect);
    try
    {
        bundle->feat = reamix::analysis::FeatureExtractor::extract (
            audio.mono22050.data(),
            audio.mono22050.size(),
            kAnalysisSampleRate,
            bundle->beatTimes);
    }
    catch (const std::exception& e)
    {
        postCompletion (nullptr, juce::String ("Feature extraction failed: ") + e.what());
        return;
    }
    if (bundle->feat.nBeats <= 0 || bundle->feat.nFeat <= 0)
    {
        postCompletion (nullptr, "Feature extraction produced empty result");
        return;
    }
    if (threadShouldExit()) return;

    // mono22050 not needed past stage 3 — drop to release ~16 MB on a 3-min
    // file. Renderer (in RemixPipeline) uses native-rate stereoNative.
    audio.mono22050.clear();
    audio.mono22050.shrink_to_fit();

    // ── Stage 5 — TransitionCost (phase-4 first half) ──────────────
    postProgress ("Computing transitions", kPFeatures);

    reamix::remix::TransitionCostInputs tcin{};
    // ADR-115 v2 scoring: production default since sesja 115 (user decision
    // after four blinded listening rounds: 11-track panel v2 2 / legacy 1 /
    // tie 8 / both-bad 1; see meta/research/sesja-115-signal-mapping.md).
    // Legacy stays reachable through the calibration harness batch field.
    tcin.v2_scoring  = true;
    tcin.features    = bundle->feat.features.data();
    tcin.n_beats     = bundle->feat.nBeats;
    tcin.n_features  = bundle->feat.nFeat;
    tcin.beat_times  = bundle->beatTimes.data();

    tcin.segments   = bundle->structure.segments.data();
    tcin.n_segments = (int) bundle->structure.segments.size();

    tcin.rms_energy        = bundle->feat.rmsEnergy.empty()        ? nullptr : bundle->feat.rmsEnergy.data();
    tcin.onset_strength    = bundle->feat.onsetStrength.empty()    ? nullptr : bundle->feat.onsetStrength.data();
    tcin.spectral_centroid = bundle->feat.spectralCentroid.empty() ? nullptr : bundle->feat.spectralCentroid.data();
    tcin.vocal_activity    = bundle->feat.vocalActivity.empty()    ? nullptr : bundle->feat.vocalActivity.data();

    // DEV-028 (sesja 63b) — xcorr wiring for Duration mode TransitionCost.
    // Mirror of RemixPipeline.cpp:246-254 (sesja 60 hot-fix for Region path).
    // Without these 4 fields, TransitionCost.cpp:548-552 has_waveforms guard
    // fails → audio-level phase-alignment branch is skipped → Duration splice
    // picks lack the most important inaudibility signal. FeatureExtractor
    // populates boundaryWaveforms unconditionally in stage 3.
    const auto& bw = bundle->feat.boundaryWaveforms;
    if (! bw.empty() && bundle->feat.nBeats > 0)
    {
        tcin.boundary_waveforms   = bw.data();
        tcin.n_boundary_waveforms = bundle->feat.nBeats;
        tcin.n_samples_per_bnd    =
            (int) (bw.size() / (std::size_t) bundle->feat.nBeats);
        tcin.waveform_sample_rate = kAnalysisSampleRate;
    }
    tcin.edge_vocal_activity_start = bundle->feat.edgeVocalActivityStart.empty() ? nullptr : bundle->feat.edgeVocalActivityStart.data();
    tcin.edge_vocal_activity_end   = bundle->feat.edgeVocalActivityEnd.empty()   ? nullptr : bundle->feat.edgeVocalActivityEnd.data();
    // ADR-088 sesja 98 — vocal phrase boundary signals.
    tcin.edge_vocal_onset_start    = bundle->feat.edgeVocalOnsetStart.empty()    ? nullptr : bundle->feat.edgeVocalOnsetStart.data();
    tcin.edge_vocal_release_end    = bundle->feat.edgeVocalReleaseEnd.empty()    ? nullptr : bundle->feat.edgeVocalReleaseEnd.data();

    tcin.edge_rms_start = bundle->feat.edgeRmsStart.empty() ? nullptr : bundle->feat.edgeRmsStart.data();
    tcin.edge_rms_end   = bundle->feat.edgeRmsEnd.empty()   ? nullptr : bundle->feat.edgeRmsEnd.data();

    tcin.edge_features_start = bundle->feat.edgeFeaturesStart.empty() ? nullptr
                              : reinterpret_cast<const float*> (bundle->feat.edgeFeaturesStart.data());
    tcin.edge_features_end   = bundle->feat.edgeFeaturesEnd.empty()   ? nullptr
                              : reinterpret_cast<const float*> (bundle->feat.edgeFeaturesEnd.data());
    tcin.n_edge_features     = bundle->feat.edgeFeaturesStart.empty() ? 0 : bundle->feat.nFeat;

    // ADR-115 E5 (sesja 115): cleaned grid on the v2 path (snapped downbeats,
    // measured bar length) - same helper RemixPipeline::run uses.
    const reamix::remix::BeatGridResult v2Grid = tcin.v2_scoring
        ? reamix::remix::cleanBeatGrid (bundle->beatTimes.data(), (int) bundle->beatTimes.size(),
                                        bundle->downbeatTimes.empty() ? nullptr : bundle->downbeatTimes.data(),
                                        (int) bundle->downbeatTimes.size(),
                                        juce::jmax (1, (int) bundle->timeSigNum))
        : reamix::remix::BeatGridResult{};
    const std::vector<double>& gridDownbeats = tcin.v2_scoring ? v2Grid.downbeats : bundle->downbeatTimes;
    tcin.downbeats   = gridDownbeats.empty() ? nullptr : gridDownbeats.data();
    tcin.n_downbeats = (int) gridDownbeats.size();

    tcin.time_signature = tcin.v2_scoring ? v2Grid.bar_beats : juce::jmax (1, (int) bundle->timeSigNum);

    try
    {
        bundle->tc = reamix::remix::computeTransitionCosts (tcin);
    }
    catch (const std::exception& e)
    {
        postCompletion (nullptr, juce::String ("Transition cost failed: ") + e.what());
        return;
    }
    if (threadShouldExit()) return;

    // ── Loop-spot map (ADR-115 E11, sesja 117) ─────────────────────
    // Whole-track Region pool -> bundle.loopSpots (Region tab suggestions).
    ensureLoopSpots (*bundle);
    if (threadShouldExit()) return;

    // uiSegments (model sections snapped to the cleaned grid) are built by
    // ensureBeatGrid inside ensureLoopSpots above (sesja 121).

    postProgress ("Done", kPTransitions);
    postCompletion (std::move (bundle), {});
}

} // namespace reamix::ui
