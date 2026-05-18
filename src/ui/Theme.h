#pragma once

#include <juce_graphics/juce_graphics.h>

// reamix.me design tokens — ported verbatim from
// phases/phase-6-ui/reamix.me JUCE/tokens.css (ADR-036 D5).
// Every constant cites its tokens.css line.
// Designer explicitly authored tokens.css for JUCE port (tokens.css:1-3).
//
// All juce::Colour values use ARGB with alpha = 0xFF unless a token in
// tokens.css defines an rgba() value — then the ARGB alpha is taken from
// the rgba alpha and the RGB from the matching hex accent.

namespace reamix::theme
{

// ── Base surfaces (warm-cast graphite) — tokens.css:5-13 ─────────────
inline const juce::Colour Bg0         { 0xFF0B0A09 }; // tokens.css:6   — window chrome outer
inline const juce::Colour Bg1         { 0xFF131210 }; // tokens.css:7   — window body
inline const juce::Colour Bg2         { 0xFF1A1816 }; // tokens.css:8   — panel
inline const juce::Colour Bg3         { 0xFF232120 }; // tokens.css:9   — raised panel / waveform well
inline const juce::Colour Bg4         { 0xFF2D2A28 }; // tokens.css:10  — input, button rest
inline const juce::Colour Bg5         { 0xFF3A3734 }; // tokens.css:11  — button hover
inline const juce::Colour Line        { 0xFF2A2724 }; // tokens.css:12  — 1px divider
inline const juce::Colour LineStrong  { 0xFF3D3935 }; // tokens.css:13  — stronger divider

// ── Foreground — tokens.css:15-20 ────────────────────────────────────
inline const juce::Colour Fg0         { 0xFFF2EEE8 }; // tokens.css:16  — titles, primary numerics
inline const juce::Colour Fg1         { 0xFFCFC9C0 }; // tokens.css:17  — body
inline const juce::Colour Fg2         { 0xFF8F897F }; // tokens.css:18  — secondary
inline const juce::Colour Fg3         { 0xFF5F5A53 }; // tokens.css:19  — tertiary / disabled label
inline const juce::Colour Fg4         { 0xFF3E3B37 }; // tokens.css:20  — ghost

// ── Accent (copper-amber) — tokens.css:22-27 ────────────────────────
inline const juce::Colour Accent      { 0xFFE8A15A }; // tokens.css:23
inline const juce::Colour AccentHi    { 0xFFF2B172 }; // tokens.css:24
inline const juce::Colour AccentLo    { 0xFFB8794A }; // tokens.css:25
// rgba(232,161,90,0.18) — tokens.css:26
inline const juce::Colour AccentDim   { juce::Colour::fromFloatRGBA (232.0f/255.0f, 161.0f/255.0f,  90.0f/255.0f, 0.18f) };
// rgba(232,161,90,0.35) — tokens.css:27
inline const juce::Colour AccentGlow  { juce::Colour::fromFloatRGBA (232.0f/255.0f, 161.0f/255.0f,  90.0f/255.0f, 0.35f) };

// ── Semantic — tokens.css:29-36 ──────────────────────────────────────
inline const juce::Colour Good        { 0xFF6CC28A }; // tokens.css:30
inline const juce::Colour GoodDim     { juce::Colour::fromFloatRGBA (108.0f/255.0f, 194.0f/255.0f, 138.0f/255.0f, 0.18f) }; // tokens.css:31
inline const juce::Colour Warn        { 0xFFE0B24A }; // tokens.css:32
inline const juce::Colour WarnDim     { juce::Colour::fromFloatRGBA (224.0f/255.0f, 178.0f/255.0f,  74.0f/255.0f, 0.18f) }; // tokens.css:33
inline const juce::Colour Bad         { 0xFFD5654C }; // tokens.css:34
inline const juce::Colour BadDim      { juce::Colour::fromFloatRGBA (213.0f/255.0f, 101.0f/255.0f,  76.0f/255.0f, 0.18f) }; // tokens.css:35
inline const juce::Colour Info        { 0xFF6A9BC4 }; // tokens.css:36

// ── Section palette (12 sections) — tokens.css:38-50 ─────────────────
// Ordering matches SegmentKind enum below.
enum class SegmentKind : int
{
    Intro = 0,
    Verse,
    PreChorus,
    Chorus,
    PostChorus,
    Bridge,
    Buildup,
    Drop,
    Breakdown,
    Solo,
    Instrumental,
    Outro,
    NumKinds
};

inline const juce::Colour SegIntro        { 0xFF7A8B6A }; // tokens.css:39  — sage
inline const juce::Colour SegVerse        { 0xFF6A8B8A }; // tokens.css:40  — teal
inline const juce::Colour SegPreChorus    { 0xFF7084B0 }; // tokens.css:41  — steel
inline const juce::Colour SegChorus       { 0xFFB07A5A }; // tokens.css:42  — copper
inline const juce::Colour SegPostChorus   { 0xFF9B7AA8 }; // tokens.css:43  — mauve
inline const juce::Colour SegBridge       { 0xFFAA7A8A }; // tokens.css:44  — rose
inline const juce::Colour SegBuildup      { 0xFFB08A50 }; // tokens.css:45  — ochre
inline const juce::Colour SegDrop         { 0xFFC56A52 }; // tokens.css:46  — rust
inline const juce::Colour SegBreakdown    { 0xFF6F6A9E }; // tokens.css:47  — indigo
inline const juce::Colour SegSolo         { 0xFFA89050 }; // tokens.css:48  — gold
inline const juce::Colour SegInstrumental { 0xFF5A8F7A }; // tokens.css:49  — jade
inline const juce::Colour SegOutro        { 0xFF7E7870 }; // tokens.css:50  — gray

juce::Colour segmentColour (SegmentKind k) noexcept;

// ── Type scale (px) — tokens.css:52-58 ───────────────────────────────
namespace fs
{
    constexpr float Display = 28.0f; // tokens.css:53  — transport readout
    constexpr float Xl      = 20.0f; // tokens.css:54
    constexpr float Lg      = 15.0f; // tokens.css:55
    constexpr float Md      = 13.0f; // tokens.css:56  — body default
    constexpr float Sm      = 11.0f; // tokens.css:57
    constexpr float Xs      =  9.0f; // tokens.css:58  — tiny caps
}

// ── Spacing scale (px) — tokens.css:60-68 ────────────────────────────
namespace s
{
    constexpr int S1 =  2; // tokens.css:61
    constexpr int S2 =  4; // tokens.css:62
    constexpr int S3 =  6; // tokens.css:63
    constexpr int S4 =  8; // tokens.css:64
    constexpr int S5 = 12; // tokens.css:65
    constexpr int S6 = 16; // tokens.css:66
    constexpr int S7 = 20; // tokens.css:67
    constexpr int S8 = 28; // tokens.css:68
}

// ── Radius — tokens.css:70-75 ────────────────────────────────────────
namespace r
{
    constexpr float R0   =   0.0f; // tokens.css:71
    constexpr float R1   =   2.0f; // tokens.css:72  — chip
    constexpr float R2   =   3.0f; // tokens.css:73  — standard
    constexpr float R3   =   5.0f; // tokens.css:74  — button
    constexpr float RFull= 999.0f; // tokens.css:75
}

// ── Elevation helpers — tokens.css:77-80 ─────────────────────────────
// CSS box-shadow ports as offset+blur+alpha tuple. JUCE-portable per spec
// comment (tokens.css:77 "blur ≤ 4px, JUCE-portable"). Consumers apply
// via juce::DropShadow.
struct Elevation { int offsetY; int blur; juce::uint8 shadowAlpha; };
constexpr Elevation Elev1   { 1, 0, static_cast<juce::uint8> (0.40f * 255) }; // tokens.css:78
constexpr Elevation Elev2   { 2, 4, static_cast<juce::uint8> (0.50f * 255) }; // tokens.css:79
constexpr Elevation ElevPop { 4, 4, static_cast<juce::uint8> (0.55f * 255) }; // tokens.css:80 (+ LineStrong 1px ring applied separately)

// ── Type families (engineer embeds TTFs) — tokens.css:82-84 ──────────
// Actual Typeface::Ptrs resolved by LookAndFeelReamix from BinaryData.
// These names must match the PostScript family names in the bundled TTFs.
namespace font
{
    inline const juce::String UiFamily   { "Inter" };
    inline const juce::String MonoFamily { "JetBrains Mono" };
}

// ── Named font helpers — map HandoffCard rows 1:1 to sized fonts ─────
// Requests Inter by family name; LookAndFeelReamix intercepts via
// getTypefaceForFont() to return the bundled TTF Typeface::Ptr.
juce::Font uiFont   (float sizePx, int weight = 400);
juce::Font monoFont (float sizePx, int weight = 400);

} // namespace reamix::theme
