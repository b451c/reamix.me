#include "DevCalibrationStorage.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cmath>
#include <ctime>

namespace reamix::ui
{

bool blockAssemblyBetaAtDefault (const BlockAssemblyBeta& b) noexcept
{
    constexpr double kEps = 1e-9;
    return std::abs (b.fragment_penalty_weight - 0.03) < kEps
        && b.outside_window_beats == 8
        && b.min_jump_beats       == 4
        && b.downbeat_only_splices == true;
}

juce::File defaultStorePath()
{
    // Same parent dir as model cache (ADR-006): on macOS this is
    // ~/Library/Application Support/reamix/. Reused for dev-calibration.jsonl
    // — clearly dev-only namespace, not in product user-data path.
    auto dir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                 .getChildFile ("reamix");
    if (! dir.exists()) dir.createDirectory();
    return dir.getChildFile ("dev-calibration.jsonl");
}

juce::String DevCalibrationStorage::nowIsoUtc()
{
    const auto now = std::time (nullptr);
    std::tm tm {};
    gmtime_r (&now, &tm);
    char buf [32];
    std::strftime (buf, sizeof (buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return juce::String (buf);
}

namespace
{

juce::var weightsToVar (const reamix::remix::QualityWeights& w)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("waveform",              w.waveform);
    obj->setProperty ("sequential_continuity", w.sequential_continuity);
    obj->setProperty ("transient_continuity",  w.transient_continuity);
    obj->setProperty ("energy",                w.energy);
    obj->setProperty ("edge_energy",           w.edge_energy);
    obj->setProperty ("bar_align",             w.bar_align);
    obj->setProperty ("centroid",              w.centroid);
    // ADR-088 sesja 98 — vocal_continuity field added to schema. Forward-
    // compat: reader ignores extra fields. Older JSONL records without this
    // key default to kDefaultQualityWeights.vocal_continuity (= 0.0).
    obj->setProperty ("vocal_continuity",      w.vocal_continuity);
    return juce::var (obj);
}

// Sesja 106 ADR-098: PerceptualSliders dropped. perceptualToVar removed.

juce::var betaToVar (const BlockAssemblyBeta& b)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("fragment_penalty_weight", b.fragment_penalty_weight);
    obj->setProperty ("outside_window_beats",    b.outside_window_beats);
    obj->setProperty ("min_jump_beats",          b.min_jump_beats);
    obj->setProperty ("downbeat_only_splices",   b.downbeat_only_splices);
    return juce::var (obj);
}

void weightsFromVar (const juce::var& v, reamix::remix::QualityWeights& out)
{
    if (! v.isObject()) return;
    // CRITICAL: start with kDefaultQualityWeights so flag fields not in JSONL
    // schema (use_harmonic_mean=true per ADR-068 D2, harmonic_vs_timbre=0.0)
    // are preserved at production defaults. Sesja-98 follow-up bug fix:
    // user reported "wczytuje zapisane to wczytuja sie domyslne" — root
    // cause was struct default QualityWeights{} has use_harmonic_mean=false
    // which silently flipped composite from harmonic to arithmetic mean →
    // different cost matrix → different/zero splices → "waveform się skraca
    // ale brak markers". Starting from kDefault preserves harmonic mean +
    // any future-added flag fields.
    out = reamix::remix::kDefaultQualityWeights;
    out.waveform              = v.getProperty ("waveform",              out.waveform);
    out.sequential_continuity = v.getProperty ("sequential_continuity", out.sequential_continuity);
    out.transient_continuity  = v.getProperty ("transient_continuity",  out.transient_continuity);
    out.energy                = v.getProperty ("energy",                out.energy);
    out.edge_energy           = v.getProperty ("edge_energy",           out.edge_energy);
    out.bar_align             = v.getProperty ("bar_align",             out.bar_align);
    out.centroid              = v.getProperty ("centroid",              out.centroid);
    out.vocal_continuity      = v.getProperty ("vocal_continuity",      out.vocal_continuity);
}

// Sesja 106 ADR-098: PerceptualSliders dropped. perceptualFromVar removed.
// Legacy v1 records with `perceptual_sliders` field are silently ignored
// (raw weights are always populated since schema v1 per sesja 98 D7).

void betaFromVar (const juce::var& v, BlockAssemblyBeta& out)
{
    if (! v.isObject()) return;
    out.fragment_penalty_weight = v.getProperty ("fragment_penalty_weight", 0.03);
    out.outside_window_beats    = (int) v.getProperty ("outside_window_beats", 8);
    out.min_jump_beats          = (int) v.getProperty ("min_jump_beats", 4);
    out.downbeat_only_splices   = (bool) v.getProperty ("downbeat_only_splices", true);
}

} // namespace

juce::String DevCalibrationStorage::toJsonLine (const DevCalibrationRecord& rec)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("schema_version",      rec.schema_version);
    obj->setProperty ("timestamp",           rec.timestamp);
    obj->setProperty ("track_path",          rec.track_path);
    obj->setProperty ("track_sha256",        rec.track_sha256);
    obj->setProperty ("duration_sec",        rec.duration_sec);
    obj->setProperty ("bpm",                 rec.bpm);
    obj->setProperty ("weights_raw",         weightsToVar (rec.weights_raw));
    // Sesja 106 ADR-098: perceptual_sliders + advanced_used fields dropped
    // in schema v2. Raw weights are sole source of truth.
    obj->setProperty ("block_assembly_beta", betaToVar (rec.block_assembly_beta));
    obj->setProperty ("genre",               rec.genre);
    obj->setProperty ("user_note",           rec.user_note);
    obj->setProperty ("mode_evaluated",      rec.mode_evaluated);
    obj->setProperty ("plugin_version",      rec.plugin_version);
    return juce::JSON::toString (juce::var (obj), /*allOnOneLine=*/true);
}

bool DevCalibrationStorage::fromJsonLine (const juce::String& line, DevCalibrationRecord& out)
{
    const juce::var v = juce::JSON::parse (line);
    if (! v.isObject()) return false;
    if (! v.hasProperty ("schema_version")) return false;

    out = DevCalibrationRecord{};
    out.schema_version = (int) v.getProperty ("schema_version", 1);
    out.timestamp      = v.getProperty ("timestamp",      "").toString();
    out.track_path     = v.getProperty ("track_path",     "").toString();
    out.track_sha256   = v.getProperty ("track_sha256",   "").toString();
    out.duration_sec   = v.getProperty ("duration_sec",   0.0);
    out.bpm            = v.getProperty ("bpm",            0.0);
    weightsFromVar    (v.getProperty ("weights_raw",         {}), out.weights_raw);
    // Sesja 106 ADR-098: perceptual_sliders + advanced_used dropped. Legacy
    // v1 records still load successfully — unknown fields silently ignored.
    betaFromVar       (v.getProperty ("block_assembly_beta", {}), out.block_assembly_beta);
    out.genre          = v.getProperty ("genre",          "").toString();
    out.user_note      = v.getProperty ("user_note",      "").toString();
    out.mode_evaluated = v.getProperty ("mode_evaluated", "").toString();
    out.plugin_version = v.getProperty ("plugin_version", "").toString();
    return true;
}

bool DevCalibrationStorage::append (const DevCalibrationRecord& rec) const
{
    const juce::String line = toJsonLine (rec) + "\n";
    const auto utf8 = line.toRawUTF8();
    const auto bytes = std::strlen (utf8);

    // Ensure parent dir exists.
    store_.getParentDirectory().createDirectory();

    // POSIX open with O_APPEND ensures atomic-against-other-O_APPEND writes
    // up to PIPE_BUF (4096 bytes) on macOS. flock adds belt-and-braces
    // mutual exclusion against non-O_APPEND readers.
    const auto path = store_.getFullPathName().toRawUTF8();
    const int fd = ::open (path, O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) return false;

    if (::flock (fd, LOCK_EX) != 0)
    {
        ::close (fd);
        return false;
    }

    bool ok = true;
    ssize_t written = 0;
    while (written < (ssize_t) bytes)
    {
        const ssize_t n = ::write (fd, utf8 + written, bytes - written);
        if (n < 0) { ok = false; break; }
        written += n;
    }

    if (ok) ::fsync (fd);
    ::flock (fd, LOCK_UN);
    ::close (fd);
    return ok;
}

std::vector<DevCalibrationRecord> DevCalibrationStorage::loadAll() const
{
    std::vector<DevCalibrationRecord> out;
    if (! store_.existsAsFile()) return out;

    const juce::String all = store_.loadFileAsString();
    juce::StringArray lines;
    lines.addLines (all);

    for (const auto& line : lines)
    {
        if (line.trim().isEmpty()) continue;
        DevCalibrationRecord rec;
        if (fromJsonLine (line, rec))
            out.push_back (std::move (rec));
        // Skip malformed silently (partial-write recovery).
    }
    return out;
}

std::vector<DevCalibrationRecord> DevCalibrationStorage::loadForTrack (const juce::String& trackPath) const
{
    std::vector<DevCalibrationRecord> out;
    for (auto& rec : loadAll())
        if (rec.track_path == trackPath)
            out.push_back (std::move (rec));
    return out;
}

} // namespace reamix::ui
