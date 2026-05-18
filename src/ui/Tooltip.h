#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <vector>

// Tooltip — phase-6 step 5 skeleton (ADR-036 D3 row 14).
//
// Rich tooltip per plugin.css:554-580 + design-system.jsx:297-306:
//   TITLE         "Splice · chorus → verse"    (Xs Fg3 uppercase)
//   key    value  "quality"  / "74%"            (Sm Fg2 / mono Fg0 right)
//   key    value  "ΔEnergy"  / "+1.2 dB"
//   key    value  "beat"     / "116"
//   [qbar 0..1]                                 (3h bar, qbarColour fill)
//   hint          "Click to seek · Right-click to block"  (10px Fg3)
//
// Step-5 ships a data-driven skeleton not wired to any caller. Step 7 wires
// SpliceMarker hover → showAt() + hide() on hover-enter / hover-leave.
// Cursor-follow semantics deferred; skeleton renders at a fixed point.

namespace reamix::ui
{

class Tooltip : public juce::Component
{
public:
    struct Row  { juce::String k; juce::String v; };
    struct Data
    {
        juce::String         title;
        std::vector<Row>     rows;
        float                qbarPct = 0.0f;     // 0..1; <=0 hides qbar
        juce::Colour         qbarColour;         // transparent default
        juce::String         hint;
    };

    Tooltip();
    ~Tooltip() override = default;

    void setData (Data);

    // Shows the tooltip anchored at parent-space `anchor` (top-left of
    // tooltip). Clamps to parent bounds. Caller must first add the tooltip
    // as a child of a parent Component.
    void showAt (juce::Point<int> anchor);
    void hide();

    void paint (juce::Graphics&) override;

private:
    juce::Rectangle<int> computePreferredBounds() const;

    Data data_;

    static constexpr int kMinWidth  = 180; // plugin.css:565
    static constexpr int kPadX      = 10;  // plugin.css:560
    static constexpr int kPadY      = 8;   // plugin.css:560
    static constexpr int kRowHeight = 14;
    static constexpr int kRowGap    = 16;  // plugin.css:572 gap
    static constexpr int kTitleH    = 9;
    static constexpr int kTitleMargBot = 4; // plugin.css:570

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Tooltip)
};

} // namespace reamix::ui
