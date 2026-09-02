#include "LoopSpotsBuilder.h"

#include "RegionCostWiring.h"
#include "remix/BeatGrid.h"
#include "remix/LoopSpots.h"
#include "remix/RegionCost.h"

#include <algorithm>
#include <exception>

namespace reamix::ui
{

void ensureBeatGrid (AnalysisBundle& bundle)
{
    if (bundle.gridBuilt) return;
    bundle.gridBuilt = true;
    const int nBeats = (int) bundle.beatTimes.size();
    if (nBeats < 2) return;

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
