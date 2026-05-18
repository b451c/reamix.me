#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

#include "Theme.h"

// AuditionBar — sesja 92 (ADR-080 RESCOPE + ADR-083 PROPOSED).
//
// Audition's Remix Advanced section analog as a 4-control horizontal panel
// hosted in MainComponent layout between WaveformView and TransportBar.
// 4-row column layout, ~120 px tall, full plugin width.
//
// Controls (top to bottom):
//   1. Tone           — Timbre ↔ Harmonic, continuous 0.0-1.0, default 0.0
//                       (ADR-080 RESCOPE: lewy koniec / pure timbre / bit-
//                       exact baseline). Drag right to enable harmonic blend.
//   2. Edit Length    — Short ↔ Long, discrete 0-100 snap 1, default 50
//                       (ADR-083 center-zero map: penalty=0.0 at 50,
//                       penalty=+1.0 at 100, penalty=-1.0 at 0).
//   3. Allow ±        — discrete 0-15 sec snap 1, default 5 (matches
//                       optimizer.py duration_tolerance_sec=5.0; UI exposure
//                       of existing constraint, no new DSP).
//   4. Min cut        — discrete 4-32 beats snap 1, default 16 (matches
//                       COOLDOWN_BARS × 4 = 16 for 4/4; UI exposure of
//                       existing min_seq_after_jump constraint).
//
// All 4 defaults bit-exact replicate current production behavior — user
// MUST drag a slider off its default position to change algorithm output.
// Per CLAUDE.md hard rule #2 (parity preservation) + user "no compromise
// quality" sesja 92 directive.
//
// Plain-language labels ("Tone" / "Edit Length" / "Allow ±" / "Min cut")
// per research-scout sesja 92b premium-UX recommendation; dynamic readout
// next to each slider shows current value in user-friendly form.
//
// Slider drag debouncing (100 ms) is owned by MainComponent — AuditionBar
// fires onChanged callbacks synchronously on each user interaction; the
// host decides when to trigger Viterbi DP-only re-run.

namespace reamix::ui
{

class AuditionBar : public juce::Component,
                    private juce::Slider::Listener
{
public:
    AuditionBar();
    ~AuditionBar() override;

    // Setters — caller (MainComponent) syncs slider thumb position to
    // per-track persisted value at item-switch time. Setters do NOT fire
    // onChanged callbacks (avoids re-render loops on switch).
    void setTone           (double v);   // 0.0-1.0
    void setEditLength     (int    v);   // 0-100
    void setAllowPmSeconds (int    v);   // 0-15
    void setMinCutBeats    (int    v);   // 4-32

    // Read-only accessors (paint + caller introspection).
    double toneValue()           const noexcept { return tone_; }
    int    editLengthValue()     const noexcept { return editLength_; }
    int    allowPmSecondsValue() const noexcept { return allowPmSeconds_; }
    int    minCutBeatsValue()    const noexcept { return minCutBeats_; }

    // Callbacks — fired on user drag/click. Host (MainComponent) debounces
    // 100 ms then triggers Viterbi DP-only re-run.
    std::function<void(double)> onToneChanged;
    std::function<void(int)>    onEditLengthChanged;
    std::function<void(int)>    onAllowPmSecondsChanged;
    std::function<void(int)>    onMinCutBeatsChanged;

    // Sesja 108 — discrete 1-click shortcut to the AdvancedWeightsWindow,
    // placed as a 16×16 sliders icon in the top-right corner of the inner
    // 12 px top padding gap (zero impact on slider row layout). Replaces
    // the prior 3-click path (gear → SettingsPopover → "Advanced..." button).
    // Host wires this to MainComponent::onAdvancedToggled().
    std::function<void()> onAdvancedClicked;

    // Sesja 108 — collapse / expand the slider body. When collapsed, only
    // the 28 px header strip ("Audition ▼/▶ … icon") is visible and the
    // MainComponent reservation shrinks accordingly so WaveformView grows.
    // Host wires onCollapseToggled to persist via ExtState + re-run its
    // own resized() to pick up the new preferred height.
    void setCollapsed (bool collapsed);
    bool isCollapsed() const noexcept { return collapsed_; }
    int  getPreferredHeight() const noexcept;
    std::function<void()> onCollapseToggled;

    // juce::Component
    void paint     (juce::Graphics&) override;
    void resized   () override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;

private:
    // juce::Slider::Listener
    void sliderValueChanged (juce::Slider* s) override;

    // Helper: dynamic readout string based on current Tone slider value.
    juce::String toneReadout()       const;
    // ADR-084 sesja 93 — meaningful Edit length readout (Short / Default /
    // Long with %). Replaces sesja-92 raw int display ("47") that user
    // pushed back as "nieintuicyjne — nie wiadomo co to za jednostka".
    juce::String editLengthReadout() const;

    // Layout constants (matches Theme tokens).
    static constexpr int kRowHeight     = 36;     // each slider row 36h (22 slider + 14 endpoint scale labels)
    static constexpr int kRowGap        = 14;     // iter-9.c: 4 → 14 (user "ciasno"); breathes between rows
    static constexpr int kBodyHeight    = 4 * kRowHeight + 3 * kRowGap;  // = 186

    static constexpr int kLabelWidth    = 88;     // left "Tone" / "Edit Length" / etc.
    static constexpr int kReadoutWidth  = 110;    // right dynamic readout
    static constexpr int kPadding       = 12;     // outer L/R padding
    // Sesja 108 — asymmetric top/bottom padding to host quick-Advanced icon
    // in the 28 px top gap. Bottom stays 12 px (sesja-92 baseline). Total bar
    // height = 28 + kBodyHeight + 12 = 226 (MainComponent reserves this).
    static constexpr int kTopPadding    = 28;
    static constexpr int kBottomPadding = 12;

    // 4 JUCE sliders (continuous + 3 discrete).
    juce::Slider toneSlider_;          // 0.0-1.0 continuous
    juce::Slider editLengthSlider_;    // 0-100 discrete snap 1
    juce::Slider allowPmSlider_;       // 0-15 discrete snap 1
    juce::Slider minCutSlider_;        // 4-32 discrete snap 1

    // ADR-084 sesja 93 — title labels host the tooltips (NOT the sliders).
    // User pushback: "tooltipy nie powinny wyskakiwac przy najechaniu na
    // slajder tylko na tytul slajdera" — slider drag triggers tooltip too
    // often. Putting tooltip on the title text means it shows only on the
    // small label area at left of each row.
    juce::Label  toneLabel_;
    juce::Label  editLengthLabel_;
    juce::Label  allowPmLabel_;
    juce::Label  minCutLabel_;

    // Current values mirroring slider state for fast read access.
    double tone_           = 0.0;   // ADR-080 RESCOPE: bit-exact baseline default
    int    editLength_     = 50;    // ADR-083 center-zero default
    int    allowPmSeconds_ = 5;     // matches duration_tolerance_sec
    int    minCutBeats_    = 16;    // matches COOLDOWN_BARS × 4 (4/4)

    // Suppress callback re-fire when MainComponent calls setX() programmatically.
    bool suppressCallbacks_ = false;

    // Sesja 108 — quick-Advanced icon bounds + hover/pressed state.
    juce::Rectangle<int> advancedBounds_;
    bool advancedHover_   = false;
    bool advancedPressed_ = false;

    // Sesja 108 — collapse/expand state + header hit-test region (entire
    // top strip minus the advanced icon on the right). Chevron is drawn
    // next to the "Audition" label as a visual affordance.
    bool collapsed_ = false;
    juce::Rectangle<int> headerHitBounds_;
    bool headerHover_   = false;
    bool headerPressed_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AuditionBar)
};

} // namespace reamix::ui
