#include "AnalysisDiskCache.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>

namespace reamix::ui
{

namespace
{
    // Magic + version header. Bump kFormatVersion on any field/layout
    // change — old caches will silently miss and be overwritten on next
    // analyze.
    //
    // History:
    //   1 — original (sesja 63 ADR-053).
    //   2 — sesja 83b2 ADR-071 phase-1: 4 vocal-stem fields appended.
    //   3 — sesja 85 ADR-074: 3 drum/inst stem fields appended.
    //   4 — sesja 91 ADR-082 CL-1: stems-aware ONNX-bound removal —
    //       7 stems-aware tail fields removed.
    //   5 — sesja 91 ADR-082 CL-3: Fast mode removal — fastMode byte
    //       removed from header.
    constexpr char        kMagic[4]       = { 'R', 'X', 'B', 'C' };
    constexpr juce::uint32 kFormatVersion = 5;

    juce::String hashOf (const juce::String& s)
    {
        const auto utf = s.toUTF8();
        return juce::SHA256 ((const void*) utf.getAddress(),
                             (size_t) s.getNumBytesAsUTF8()).toHexString();
    }

    // ── Write helpers ─────────────────────────────────────────────────

    bool writeRaw (juce::FileOutputStream& s, const void* data, size_t bytes)
    {
        return s.write (data, bytes);
    }

    template <typename T>
    bool writePOD (juce::FileOutputStream& s, T v)
    {
        return writeRaw (s, &v, sizeof (T));
    }

    bool writeString (juce::FileOutputStream& s, const juce::String& str)
    {
        const auto utf = str.toRawUTF8();
        const juce::uint32 n = (juce::uint32) std::strlen (utf);
        return writePOD (s, n) && writeRaw (s, utf, n);
    }

    template <typename T>
    bool writeVec (juce::FileOutputStream& s, const std::vector<T>& v)
    {
        const juce::uint64 n = (juce::uint64) v.size();
        if (! writePOD (s, n)) return false;
        if (n == 0) return true;
        return writeRaw (s, v.data(), n * sizeof (T));
    }

    bool writeVecBool (juce::FileOutputStream& s, const std::vector<bool>& v)
    {
        const juce::uint64 n = (juce::uint64) v.size();
        if (! writePOD (s, n)) return false;
        std::vector<juce::uint8> bytes (v.size());
        for (std::size_t i = 0; i < v.size(); ++i) bytes[i] = v[i] ? 1 : 0;
        return n == 0 ? true : writeRaw (s, bytes.data(), bytes.size());
    }

    // ── Read helpers ──────────────────────────────────────────────────

    bool readRaw (juce::FileInputStream& s, void* data, size_t bytes)
    {
        return (size_t) s.read (data, (int) bytes) == bytes;
    }

    template <typename T>
    bool readPOD (juce::FileInputStream& s, T& v)
    {
        return readRaw (s, &v, sizeof (T));
    }

    bool readString (juce::FileInputStream& s, juce::String& out)
    {
        juce::uint32 n = 0;
        if (! readPOD (s, n)) return false;
        if (n == 0) { out = {}; return true; }
        std::vector<char> buf (n + 1, 0);
        if (! readRaw (s, buf.data(), n)) return false;
        out = juce::String::fromUTF8 (buf.data(), (int) n);
        return true;
    }

    template <typename T>
    bool readVec (juce::FileInputStream& s, std::vector<T>& v)
    {
        juce::uint64 n = 0;
        if (! readPOD (s, n)) return false;
        v.assign ((std::size_t) n, T {});
        if (n == 0) return true;
        return readRaw (s, v.data(), (size_t) n * sizeof (T));
    }

    bool readVecBool (juce::FileInputStream& s, std::vector<bool>& v)
    {
        juce::uint64 n = 0;
        if (! readPOD (s, n)) return false;
        v.assign ((std::size_t) n, false);
        if (n == 0) return true;
        std::vector<juce::uint8> bytes ((std::size_t) n, 0);
        if (! readRaw (s, bytes.data(), bytes.size())) return false;
        for (std::size_t i = 0; i < bytes.size(); ++i) v[i] = bytes[i] != 0;
        return true;
    }

    // ── FeatureExtractor::Result IO ───────────────────────────────────

    bool writeFeatureResult (juce::FileOutputStream& s,
                             const reamix::analysis::FeatureExtractor::Result& f)
    {
        return writeVec (s, f.features)
            && writePOD (s, (juce::int32) f.nBeats)
            && writePOD (s, (juce::int32) f.nFeat)
            && writeVec (s, f.beatTimes)
            && writeVec (s, f.rmsEnergy)
            && writeVec (s, f.onsetStrength)
            && writeVec (s, f.spectralCentroid)
            && writeVec (s, f.edgeFeaturesStart)
            && writeVec (s, f.edgeFeaturesEnd)
            && writeVec (s, f.edgeRmsStart)
            && writeVec (s, f.edgeRmsEnd)
            && writeVec (s, f.boundaryWaveforms)
            && writeVec (s, f.transitionWaveforms)
            && writeVec (s, f.vocalActivity)
            && writeVec (s, f.voicedRatio)
            && writeVec (s, f.f0Hz)
            && writeVec (s, f.f0Confidence)
            && writeVec (s, f.edgeVocalActivityStart)
            && writeVec (s, f.edgeVocalActivityEnd)
            && writeVec (s, f.edgeVocalOnsetStart)
            && writeVec (s, f.edgeVocalReleaseEnd);
    }

    bool readFeatureResult (juce::FileInputStream& s,
                            reamix::analysis::FeatureExtractor::Result& f)
    {
        juce::int32 nBeats = 0, nFeat = 0;
        if (! (readVec (s, f.features)
            && readPOD (s, nBeats)
            && readPOD (s, nFeat)
            && readVec (s, f.beatTimes)
            && readVec (s, f.rmsEnergy)
            && readVec (s, f.onsetStrength)
            && readVec (s, f.spectralCentroid)
            && readVec (s, f.edgeFeaturesStart)
            && readVec (s, f.edgeFeaturesEnd)
            && readVec (s, f.edgeRmsStart)
            && readVec (s, f.edgeRmsEnd)
            && readVec (s, f.boundaryWaveforms)
            && readVec (s, f.transitionWaveforms)
            && readVec (s, f.vocalActivity)
            && readVec (s, f.voicedRatio)
            && readVec (s, f.f0Hz)
            && readVec (s, f.f0Confidence)
            && readVec (s, f.edgeVocalActivityStart)
            && readVec (s, f.edgeVocalActivityEnd)
            && readVec (s, f.edgeVocalOnsetStart)
            && readVec (s, f.edgeVocalReleaseEnd))) return false;
        f.nBeats = (int) nBeats;
        f.nFeat  = (int) nFeat;
        return true;
    }

    // ── StructureResult IO ───────────────────────────────────────────

    bool writeStructure (juce::FileOutputStream& s,
                         const reamix::analysis::StructureResult& st)
    {
        const juce::uint32 nSeg = (juce::uint32) st.segments.size();
        if (! writePOD (s, nSeg)) return false;
        for (const auto& seg : st.segments)
        {
            if (! (writePOD (s, seg.start)
                && writePOD (s, seg.end)
                && writePOD (s, seg.confidence)
                && writePOD (s, (juce::int32) seg.cluster_id)
                && writeString (s, juce::String (seg.label)))) return false;
        }
        if (! writeVec (s, st.boundaries)) return false;
        const juce::uint8 hasBpm = st.bpm.has_value() ? 1 : 0;
        if (! writePOD (s, hasBpm)) return false;
        const double bpmVal = st.bpm.value_or (0.0);
        if (! writePOD (s, bpmVal)) return false;
        return writePOD (s, (juce::uint8) st.path);
    }

    bool readStructure (juce::FileInputStream& s,
                        reamix::analysis::StructureResult& st)
    {
        juce::uint32 nSeg = 0;
        if (! readPOD (s, nSeg)) return false;
        st.segments.resize (nSeg);
        for (auto& seg : st.segments)
        {
            juce::int32 cluster = 0;
            juce::String labelStr;
            if (! (readPOD (s, seg.start)
                && readPOD (s, seg.end)
                && readPOD (s, seg.confidence)
                && readPOD (s, cluster)
                && readString (s, labelStr))) return false;
            seg.cluster_id = (int) cluster;
            seg.label      = labelStr.toStdString();
        }
        if (! readVec (s, st.boundaries)) return false;
        juce::uint8 hasBpm = 0;
        double bpmVal = 0.0;
        if (! (readPOD (s, hasBpm) && readPOD (s, bpmVal))) return false;
        st.bpm = (hasBpm != 0) ? std::optional<double> (bpmVal) : std::nullopt;
        juce::uint8 pathByte = 0;
        if (! readPOD (s, pathByte)) return false;
        st.path = (reamix::analysis::DispatchPath) pathByte;
        return true;
    }

    // ── TransitionCostResult IO ──────────────────────────────────────

    bool writeTC (juce::FileOutputStream& s,
                  const reamix::remix::TransitionCostResult& tc)
    {
        if (! (writeVec (s, tc.W)
            && writeVec (s, tc.chroma_D)
            && writeVec (s, tc.importance))) return false;

        const juce::uint32 nCand = (juce::uint32) tc.candidates.size();
        if (! writePOD (s, nCand)) return false;
        for (const auto& kv : tc.candidates)
        {
            const auto& c = kv.second;
            if (! (writePOD (s, (juce::int32) c.from_beat)
                && writePOD (s, (juce::int32) c.to_beat)
                && writePOD (s, c.quality_score)
                && writePOD (s, c.waveform_similarity)
                && writePOD (s, c.successor_similarity)
                && writePOD (s, c.edge_splice_similarity)
                && writePOD (s, c.chroma_distance)
                && writePOD (s, c.energy_diff_db)
                && writePOD (s, (juce::int32) c.alignment_lag_samples)
                && writePOD (s, c.total_cost))) return false;
        }
        return writePOD (s, (juce::int32) tc.n_beats);
    }

    bool readTC (juce::FileInputStream& s,
                 reamix::remix::TransitionCostResult& tc)
    {
        if (! (readVec (s, tc.W)
            && readVec (s, tc.chroma_D)
            && readVec (s, tc.importance))) return false;

        juce::uint32 nCand = 0;
        if (! readPOD (s, nCand)) return false;
        tc.candidates.clear();
        for (juce::uint32 i = 0; i < nCand; ++i)
        {
            reamix::remix::TransitionCandidate c {};
            juce::int32 fromBeat = 0, toBeat = 0, lag = 0;
            if (! (readPOD (s, fromBeat)
                && readPOD (s, toBeat)
                && readPOD (s, c.quality_score)
                && readPOD (s, c.waveform_similarity)
                && readPOD (s, c.successor_similarity)
                && readPOD (s, c.edge_splice_similarity)
                && readPOD (s, c.chroma_distance)
                && readPOD (s, c.energy_diff_db)
                && readPOD (s, lag)
                && readPOD (s, c.total_cost))) return false;
            c.from_beat = (int) fromBeat;
            c.to_beat   = (int) toBeat;
            c.alignment_lag_samples = (int) lag;
            tc.candidates[{ c.from_beat, c.to_beat }] = c;
        }
        juce::int32 nBeats = 0;
        if (! readPOD (s, nBeats)) return false;
        tc.n_beats = (int) nBeats;
        return true;
    }

    // ── uiSegments IO ────────────────────────────────────────────────

    bool writeUiSegments (juce::FileOutputStream& s,
                          const std::vector<AnalysisSegment>& segs)
    {
        const juce::uint32 n = (juce::uint32) segs.size();
        if (! writePOD (s, n)) return false;
        for (const auto& sg : segs)
        {
            if (! (writePOD (s, sg.startSec)
                && writePOD (s, sg.endSec)
                && writePOD (s, (juce::int32) sg.kind))) return false;
        }
        return true;
    }

    bool readUiSegments (juce::FileInputStream& s,
                         std::vector<AnalysisSegment>& segs)
    {
        juce::uint32 n = 0;
        if (! readPOD (s, n)) return false;
        segs.resize (n);
        for (auto& sg : segs)
        {
            juce::int32 k = 0;
            if (! (readPOD (s, sg.startSec)
                && readPOD (s, sg.endSec)
                && readPOD (s, k))) return false;
            sg.kind = (reamix::theme::SegmentKind) k;
        }
        return true;
    }

    // ── Audio re-decode (mirrors AnalyzePipeline::loadAudio) ─────────

    bool reDecodeAudio (const juce::String& path, AnalysisBundle& bundle)
    {
        juce::File f (path);
        if (! f.existsAsFile()) return false;

        juce::AudioFormatManager fm;
        fm.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor (f));
        if (reader == nullptr) return false;

        const int nativeSr  = (int) reader->sampleRate;
        const int nChannels = juce::jmax (1, juce::jmin (2, (int) reader->numChannels));
        const juce::int64 totalSamples = reader->lengthInSamples;
        if (nativeSr <= 0 || totalSamples <= 0) return false;

        // Sanity: cached header must match what the file reports now. If
        // the user replaced the source file with a different track at the
        // same path, treat as miss (caller deletes cache file).
        if (nativeSr != bundle.nativeSr
            || nChannels != bundle.nChannels
            || (std::size_t) totalSamples != bundle.nativeSamples)
            return false;

        juce::AudioBuffer<float> buf (nChannels, (int) totalSamples);
        if (! reader->read (&buf, 0, (int) totalSamples, 0,
                            nChannels >= 1, nChannels >= 2)) return false;

        bundle.stereoNative.resize ((std::size_t) nChannels * (std::size_t) totalSamples);
        for (int ch = 0; ch < nChannels; ++ch)
        {
            const float* src = buf.getReadPointer (ch);
            std::copy (src, src + totalSamples,
                       bundle.stereoNative.begin()
                       + (std::size_t) ch * (std::size_t) totalSamples);
        }
        return true;
    }
} // anon namespace

juce::File AnalysisDiskCache::getCacheDirectory()
{
    // macOS quirk (mirrors ModelManager.cpp): JUCE's
    // userApplicationDataDirectory returns ~/Library on macOS, not
    // ~/Library/Application Support — append manually so cache lives at
    //   ~/Library/Application Support/reamix.me/cache/
    // matching where REAPER plugins / model files already live.
    auto base = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory);
   #if JUCE_MAC
    base = base.getChildFile ("Application Support");
   #endif
    auto dir = base.getChildFile ("reamix.me").getChildFile ("cache");
    if (! dir.isDirectory()) dir.createDirectory();
    return dir;
}

juce::File AnalysisDiskCache::cacheFileForSource (const juce::String& sourcePath)
{
    return getCacheDirectory().getChildFile (hashOf (sourcePath) + ".bundle");
}

bool AnalysisDiskCache::save (const AnalysisBundle& bundle)
{
    if (bundle.sourcePath.isEmpty()) return false;
    auto file = cacheFileForSource (bundle.sourcePath);

    auto tmp = file.getSiblingFile (file.getFileName() + ".tmp");
    tmp.deleteFile();

    {
        juce::FileOutputStream out (tmp);
        if (! out.openedOk()) return false;

        // Header
        if (! out.write (kMagic, 4)) return false;
        if (! writePOD (out, kFormatVersion)) return false;
        if (! writeString (out, bundle.sourcePath)) return false;
        if (! writePOD (out, (juce::int32) bundle.nativeSr)) return false;
        if (! writePOD (out, (juce::int32) bundle.nChannels)) return false;
        if (! writePOD (out, (juce::uint64) bundle.nativeSamples)) return false;

        // BeatDetector
        if (! writePOD (out, bundle.bpm)) return false;
        if (! writePOD (out, (juce::int32) bundle.timeSigNum)) return false;
        if (! writeVec (out, bundle.beatTimes)) return false;
        if (! writeVec (out, bundle.downbeatTimes)) return false;
        if (! writeVecBool (out, bundle.beatIsDownbeat)) return false;

        // FeatureExtractor / Structure / TransitionCost / uiSegments
        if (! writeFeatureResult (out, bundle.feat)) return false;
        if (! writeStructure (out, bundle.structure)) return false;
        if (! writeTC (out, bundle.tc)) return false;
        if (! writeUiSegments (out, bundle.uiSegments)) return false;

        out.flush();
    }

    // Atomic-ish replace: tmp → final. Best effort; on failure leave both.
    file.deleteFile();
    return tmp.moveFileTo (file);
}

AnalysisBundlePtr AnalysisDiskCache::tryLoad (const juce::String& sourcePath)
{
    if (sourcePath.isEmpty()) return nullptr;
    auto file = cacheFileForSource (sourcePath);
    if (! file.existsAsFile()) return nullptr;

    juce::FileInputStream in (file);
    if (! in.openedOk()) return nullptr;

    char magic[4] = {};
    if (! readRaw (in, magic, 4)) return nullptr;
    if (std::memcmp (magic, kMagic, 4) != 0) { file.deleteFile(); return nullptr; }

    juce::uint32 version = 0;
    if (! readPOD (in, version) || version != kFormatVersion)
    {
        file.deleteFile();
        return nullptr;
    }

    auto bundle = std::make_shared<AnalysisBundle>();

    juce::String storedPath;
    juce::int32  nativeSr = 0, nChannels = 0;
    juce::uint64 nativeSamples = 0;

    if (! (readString (in, storedPath)
        && readPOD (in, nativeSr)
        && readPOD (in, nChannels)
        && readPOD (in, nativeSamples))) { file.deleteFile(); return nullptr; }

    bundle->sourcePath    = storedPath;
    bundle->nativeSr      = (int) nativeSr;
    bundle->nChannels     = (int) nChannels;
    bundle->nativeSamples = (std::size_t) nativeSamples;

    juce::int32 timeSigNum = 4;
    if (! (readPOD (in, bundle->bpm)
        && readPOD (in, timeSigNum)
        && readVec (in, bundle->beatTimes)
        && readVec (in, bundle->downbeatTimes)
        && readVecBool (in, bundle->beatIsDownbeat)
        && readFeatureResult (in, bundle->feat)
        && readStructure (in, bundle->structure)
        && readTC (in, bundle->tc)
        && readUiSegments (in, bundle->uiSegments)))
    {
        file.deleteFile();
        return nullptr;
    }
    bundle->timeSigNum = (int) timeSigNum;

    if (! reDecodeAudio (sourcePath, *bundle))
    {
        // Audio file moved / format changed / size mismatch — cache is
        // no longer valid for this source. Drop it.
        file.deleteFile();
        return nullptr;
    }

    return bundle;
}

juce::int64 AnalysisDiskCache::totalSizeBytes()
{
    juce::int64 total = 0;
    auto dir = getCacheDirectory();
    for (auto entry : juce::RangedDirectoryIterator (dir, false, "*.bundle"))
        total += entry.getFile().getSize();
    return total;
}

int AnalysisDiskCache::countEntries()
{
    int n = 0;
    auto dir = getCacheDirectory();
    for (auto entry : juce::RangedDirectoryIterator (dir, false, "*.bundle"))
    {
        juce::ignoreUnused (entry);
        ++n;
    }
    return n;
}

int AnalysisDiskCache::clearAll()
{
    int n = 0;
    auto dir = getCacheDirectory();
    for (auto entry : juce::RangedDirectoryIterator (dir, false, "*.bundle"))
    {
        if (entry.getFile().deleteFile()) ++n;
    }
    return n;
}

void AnalysisDiskCache::revealInFinder()
{
    getCacheDirectory().revealToUser();
}

} // namespace reamix::ui
