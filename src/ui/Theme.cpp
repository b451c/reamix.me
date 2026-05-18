#include "Theme.h"

namespace reamix::theme
{

juce::Colour segmentColour (SegmentKind k) noexcept
{
    // Lookup table ordered to match SegmentKind enum (tokens.css:39-50).
    switch (k)
    {
        case SegmentKind::Intro:        return SegIntro;
        case SegmentKind::Verse:        return SegVerse;
        case SegmentKind::PreChorus:    return SegPreChorus;
        case SegmentKind::Chorus:       return SegChorus;
        case SegmentKind::PostChorus:   return SegPostChorus;
        case SegmentKind::Bridge:       return SegBridge;
        case SegmentKind::Buildup:      return SegBuildup;
        case SegmentKind::Drop:         return SegDrop;
        case SegmentKind::Breakdown:    return SegBreakdown;
        case SegmentKind::Solo:         return SegSolo;
        case SegmentKind::Instrumental: return SegInstrumental;
        case SegmentKind::Outro:        return SegOutro;
        case SegmentKind::NumKinds:     break;
    }
    return Fg3; // fallback — unknown kind
}

static juce::Font buildFont (const juce::String& family, float sizePx, int weight)
{
    // Map CSS font-weight → JUCE style string. JUCE uses style names the
    // typeface advertises; Inter + JetBrains Mono both expose Regular /
    // Medium / SemiBold / Bold as distinct style names.
    juce::String style;
    if      (weight >= 700) style = "Bold";
    else if (weight >= 600) style = "SemiBold";
    else if (weight >= 500) style = "Medium";
    else                    style = "Regular";

    juce::Font f (juce::FontOptions (family, sizePx, juce::Font::plain)
                      .withStyle (style));
    return f;
}

juce::Font uiFont (float sizePx, int weight)
{
    return buildFont (font::UiFamily, sizePx, weight);
}

juce::Font monoFont (float sizePx, int weight)
{
    return buildFont (font::MonoFamily, sizePx, weight);
}

} // namespace reamix::theme
