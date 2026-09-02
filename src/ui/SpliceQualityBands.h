#pragma once

#include "WaveformView.h"

// Sesja 120 (DEV-099) — one home for the splice-quality colour bands.
//
// Duration / Region keep the legacy buckets (Good > 0.70, Medium >= 0.50).
// Block Assembly junctions score lower by construction (the v2 geometric
// composite with floor 0.05 on cuts that are a whole section change), and
// the sesja-119 user smoke accepted junctions at q 0.45-0.62 that painted
// red / amber ("calkiem niezle mimo czerwonych linii"), so Blocks uses
// Good >= 0.60, Medium >= 0.45. PROVISIONAL until the blinded s119 cells /
// hands-on smoke re-calibrate; change the numbers here only. A junction
// that fell back to the authored boundary (no clean cut found) is always
// Bad regardless of its score.

namespace reamix::ui
{

struct QualityBands
{
    float good;     // q >  good   -> Good
    float medium;   // q >= medium -> Medium, else Bad
};

inline constexpr QualityBands kPathQualityBands   { 0.70f, 0.50f };   // Duration / Region
inline constexpr QualityBands kBlocksQualityBands { 0.60f, 0.45f };   // Block Assembly

inline WaveformView::SpliceQuality bucketQuality (float q, QualityBands b, bool fallback = false) noexcept
{
    if (fallback)        return WaveformView::SpliceQuality::Bad;
    if (q > b.good)      return WaveformView::SpliceQuality::Good;
    if (q >= b.medium)   return WaveformView::SpliceQuality::Medium;
    return WaveformView::SpliceQuality::Bad;
}

} // namespace reamix::ui
