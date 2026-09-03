#pragma once

// EditDensity — ADR-115 P3 (sesja 123) + DEV-112 (sesja 124). The five
// "Edit density" detents are stored, hashed and passed to the engine as a
// bar count (16 / 8 / 4 / 2 / 1; see EditTuningBar::kDetentBars,
// RemixCache::hashEditDensity, RemixPipeline::Input::edit_density_bars);
// what a detent MEANS differs per mode and lives here, shared by the bar
// (labels) and the pipeline (engine mapping), with no JUCE dependency so the
// calibration harness sees the same table.
//
//   Region:   the loop-length cooldown in bars (E8): 1 bar = the engine
//             default, 2 / 4 / 8 / 16 = longer loops, fewer cuts.
//   Duration: 4 bars = the engine default (COOLDOWN_BARS, bit-exact legacy
//             path); 8 / 16 = "fewer cuts" (longer cooldown, jump tax x2 /
//             x4, transition cap 3 / 2); 2 / 1 = "more cuts" = a MINIMUM
//             number of cuts (2 / 4), forward-only when shortening, the
//             phrase cooldown untouched, fewer cuts when no such path
//             exists. DEV-112: the bar count is not a phrase length in
//             Duration - a maximum-run gate was measured and rejected
//             (sparse candidate pools strand or loop the path).

namespace reamix::ui
{

// Minimum number of cuts requested by a Duration "more cuts" detent
// (0 = no floor: the default and the "fewer cuts" detents).
inline constexpr int densityMinCuts (int bars) noexcept
{
    return bars == 1 ? 4 : (bars == 2 ? 2 : 0);
}

// Readout of a detent on the Duration tab (Region reads "N bars").
inline const char* durationDensityLabel (int bars) noexcept
{
    switch (bars)
    {
        case 16: return "Fewest cuts";
        case 8:  return "Fewer cuts";
        case 2:  return "At least 2 cuts";
        case 1:  return "At least 4 cuts";
        default: return "Default";
    }
}

} // namespace reamix::ui
