#pragma once

#include "AnalysisBundle.h"

// ADR-115 E11 (sesja 117) — whole-track loop-spot map for Region mode.
//
// Runs the Region candidate pool (`computeRegionCosts`, v2 scoring, same
// wiring as the Region remix via RegionCostWiring.h) over the whole track
// (entry_beat = 0, exit_beat = n_beats) and stores every backward bar-aligned
// pair as a LoopSpot in `bundle.loopSpots` (best quality first) together with
// the measured bar length. ~0.4 s for a 5-minute track (Billie Jean, 569
// beats: 687 scored pairs). Not part of the disk-cache payload (format 6
// unchanged) — called by AnalyzePipeline after stage 5 and by MainComponent
// after a disk-cache hit. Idempotent: a bundle that already has the map is
// left alone.

namespace reamix::ui
{

void ensureLoopSpots (AnalysisBundle& bundle);

} // namespace reamix::ui
