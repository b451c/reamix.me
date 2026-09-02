#include "LoopSpotsBuilder.h"

#include "RegionCostWiring.h"
#include "remix/BeatGrid.h"
#include "remix/LoopSpots.h"
#include "remix/RegionCost.h"

#include <algorithm>
#include <exception>

namespace reamix::ui
{

void ensureLoopSpots (AnalysisBundle& bundle)
{
    if (bundle.loopSpotsBuilt) return;
    bundle.loopSpotsBuilt = true;          // one attempt per bundle, even on failure
    bundle.loopSpots.clear();

    const int nBeats = bundle.tc.n_beats;
    if (nBeats < 2 || bundle.beatTimes.size() < (std::size_t) nBeats
        || bundle.feat.features.empty())
        return;

    // ADR-115 E5 — same cleaned grid the Region remix uses (RemixPipeline).
    const reamix::remix::BeatGridResult grid = reamix::remix::cleanBeatGrid (
        bundle.beatTimes.data(), (int) bundle.beatTimes.size(),
        bundle.downbeatTimes.empty() ? nullptr : bundle.downbeatTimes.data(),
        (int) bundle.downbeatTimes.size(),
        std::max (1, (int) bundle.timeSigNum));
    bundle.loopSpotsBarBeats = std::max (1, grid.bar_beats);

    reamix::remix::RegionCostInputs rcin{};
    rcin.v2_scoring = true;                // v2 is the production default (ADR-115)
    rcin.entry_beat = 0;
    rcin.exit_beat  = nBeats;
    fillRegionCostInputs (rcin, bundle, grid.downbeats, grid.bar_beats);

    try
    {
        const auto pool = reamix::remix::computeRegionCosts (rcin);
        bundle.loopSpots = reamix::remix::extractLoopSpots (
            pool.candidates, bundle.beatTimes.data(), nBeats, bundle.loopSpotsBarBeats);
    }
    catch (const std::exception&)
    {
        bundle.loopSpots.clear();          // Region tab then shows no suggestions
    }
}

} // namespace reamix::ui
