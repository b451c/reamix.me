#include "LoopSpotsBuilder.h"

#include "RegionCostWiring.h"
#include "remix/BeatGrid.h"
#include "remix/LoopSpots.h"
#include "remix/RegionCost.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

namespace reamix::ui
{

namespace
{
    reamix::theme::SegmentKind mapLabel (const std::string& label)
    {
        // LinkSeg 9-class labels (SectionClassifier::labelName). Unknown
        // tokens map to Verse so the section bar renders without a hole.
        using K = reamix::theme::SegmentKind;
        if (label == "intro")       return K::Intro;
        if (label == "outro")       return K::Outro;
        if (label == "chorus")      return K::Chorus;
        if (label == "verse")       return K::Verse;
        if (label == "bridge")      return K::Bridge;
        if (label == "inst")        return K::Instrumental;
        if (label == "pre-chorus")  return K::PreChorus;
        if (label == "post-chorus") return K::PostChorus;
        return K::Verse;
    }

    // Sesja 121 (DEV-098): model sections -> UI sections on the cleaned grid.
    // Mirrors tools/dev/section_eval/corpus.py snap_to_downbeats + relabel.
    void buildUiSegments (AnalysisBundle& b)
    {
        b.uiSegments.clear();
        const auto& segs = b.structure.segments;
        if (segs.empty()) return;
        const double duration = segs.back().end;
        const double bar      = std::max (0.0, b.barSec);
        const auto&  db       = b.gridDownbeats;

        std::vector<double> times { 0.0 };
        for (std::size_t i = 1; i < segs.size(); ++i)
        {
            double t = segs[i].start;
            if (! db.empty())
            {
                auto it = std::lower_bound (db.begin(), db.end(), t);
                double best = (it != db.end()) ? *it : db.back();
                if (it != db.begin() && std::abs (*(it - 1) - t) < std::abs (best - t)) best = *(it - 1);
                t = best;
            }
            if (t - times.back() >= bar - 1e-3 && duration - t >= bar - 1e-3) times.push_back (t);
        }
        times.push_back (duration);

        struct Raw { double s, e; std::string label; };
        std::vector<Raw> out;
        for (std::size_t i = 0; i + 1 < times.size(); ++i)
        {
            const double m = 0.5 * (times[i] + times[i + 1]);
            const reamix::analysis::Segment* raw = &segs.back();
            for (const auto& s : segs)
                if (m >= s.start && m < s.end) { raw = &s; break; }
            out.push_back ({ times[i], times[i + 1], raw->label });
        }
        // Silence has no block kind: hand its span to the following section
        // (leading silence) or the previous one (trailing / inner).
        for (std::size_t i = 0; i < out.size();)
        {
            if (out[i].label != "silence") { ++i; continue; }
            if (i + 1 < out.size())      out[i + 1].s = out[i].s;
            else if (i > 0)              out[i - 1].e = out[i].e;
            else { ++i; continue; }
            out.erase (out.begin() + (long) i);
        }
        b.uiSegments.reserve (out.size());
        for (const auto& r : out)
            b.uiSegments.push_back ({ r.s, r.e, mapLabel (r.label) });
    }
} // namespace

void ensureBeatGrid (AnalysisBundle& bundle)
{
    if (bundle.gridBuilt) return;
    bundle.gridBuilt = true;
    const int nBeats = (int) bundle.beatTimes.size();
    if (nBeats < 2) { buildUiSegments (bundle); return; }

    // ADR-115 E5 — same cleaned grid the Region / Blocks remix uses
    // (RemixPipeline, v2 path).
    const reamix::remix::BeatGridResult grid = reamix::remix::cleanBeatGrid (
        bundle.beatTimes.data(), nBeats,
        bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data(),
        (int) bundle.downbeatTimes.size(),
        std::max (1, (int) bundle.timeSigNum));
    bundle.gridDownbeats     = grid.downbeats;
    bundle.barBeats          = std::max (1, grid.bar_beats);
    bundle.loopSpotsBarBeats = bundle.barBeats;
    bundle.barSec            = grid.period_sec > 0.0
                                 ? grid.period_sec * bundle.barBeats
                                 : (bundle.beatTimes.back() - bundle.beatTimes.front())
                                     / (double) (nBeats - 1) * bundle.barBeats;

    // DEV-098: UI downbeat ticks from the cleaned grid (the raw detector mask
    // could put a block edge on a tick the engine does not treat as a bar).
    if (! grid.downbeat_idx.empty())
    {
        bundle.beatIsDownbeat.assign ((std::size_t) nBeats, false);
        for (int idx : grid.downbeat_idx)
            if (idx >= 0 && idx < nBeats) bundle.beatIsDownbeat[(std::size_t) idx] = true;
    }
    buildUiSegments (bundle);
}

void ensureLoopSpots (AnalysisBundle& bundle)
{
    if (bundle.loopSpotsBuilt) return;
    bundle.loopSpotsBuilt = true;          // one attempt per bundle, even on failure
    bundle.loopSpots.clear();
    bundle.sectionSpans.clear();
    ensureBeatGrid (bundle);

    const int nBeats = bundle.tc.n_beats;
    if (nBeats < 2 || bundle.beatTimes.size() < (std::size_t) nBeats
        || bundle.feat.features.empty())
        return;

    reamix::remix::RegionCostInputs rcin{};
    rcin.v2_scoring = true;                // v2 is the production default (ADR-115)
    rcin.entry_beat = 0;
    rcin.exit_beat  = nBeats;
    fillRegionCostInputs (rcin, bundle, bundle.gridDownbeats, bundle.barBeats);

    try
    {
        const auto pool = reamix::remix::computeRegionCosts (rcin);
        bundle.loopSpots = reamix::remix::extractLoopSpots (
            pool.candidates, bundle.beatTimes.data(), nBeats, bundle.barBeats);
        // Sesja 120 (DEV-097): the same pool as proposed blocks.
        bundle.sectionSpans = reamix::remix::extractSectionSpans (
            pool.candidates, bundle.beatTimes.data(), nBeats, bundle.barBeats);
    }
    catch (const std::exception&)
    {
        bundle.loopSpots.clear();          // Region tab then shows no suggestions
        bundle.sectionSpans.clear();
    }
}

} // namespace reamix::ui
