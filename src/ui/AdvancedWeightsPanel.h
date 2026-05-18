#pragma once

// Sesja 106 — ADR-097 + ADR-098 (ratified).
// Sesja 107 — ADR-097 implementation: relocated src/dev/ → src/ui/, namespace
// reamix::dev → reamix::ui, hosted by AdvancedWeightsWindow (juce::DocumentWindow).
//
// AdvancedWeightsPanel — power-user opt-in window content. Opened on demand
// from SettingsPopover "Advanced..." button. Lazy-created on first open;
// open-state + position persisted across REAPER restart via ExtState.
//
// Post-ADR-098 simplification: PerceptualMapping layer dropped. Panel shows
// 8 raw cost weights always visible (no expander) + 4 Block Assembly β-controls.
//
// Layout ~420 px tall:
//   Header bar 28h
//   Badge row 24h (4 states)
//   Section 1: COST WEIGHTS — active in all modes
//     8 raw sliders always shown
//   Section 2: BLOCK ASSEMBLY TUNING — active only in Blocks mode
//     4 β-controls (disabled-with-explanation when mode != Blocks)
//   Actions row 36h
//     [Save best...] [Load saved...] [Restore default]

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

#include "DevCalibrationStorage.h"

namespace reamix::ui
{

// BlockAssemblyBeta struct + blockAssemblyBetaAtDefault helper live in
// DevCalibrationStorage.h post-ADR-098 (formerly in PerceptualMapping.h
// which was deleted). Imported transitively via DevCalibrationStorage.h
// include above.

class AdvancedWeightsPanel : public juce::Component,
                             private juce::Slider::Listener,
                             private juce::Button::Listener
{
public:
    enum class AppMode { Duration, Region, Blocks };

    enum class BadgeState { NoSave, Modified, Saved, Loaded };

    AdvancedWeightsPanel();
    ~AdvancedWeightsPanel() override;

    // Setters — caller (MainComponent) syncs slider thumbs to per-track
    // persisted value at item-switch time. Setters do NOT fire onChanged
    // callbacks (avoids re-render loops).
    void setBlockAssemblyBeta (const BlockAssemblyBeta& b);
    void setRawWeights        (const reamix::remix::QualityWeights& w);

    // Read-only accessors.
    const BlockAssemblyBeta&             beta()    const noexcept { return beta_; }
    const reamix::remix::QualityWeights& weights() const noexcept { return weights_; }

    // Mode-awareness — β-section disabled-with-explanation when mode != Blocks.
    // Caller updates on app mode tab switch.
    void setAppMode (AppMode m);

    // Track context — used by Save flow (track_path + sha256 in JSONL),
    // Load flow (filter saves to current track), and badge state. Caller
    // sets at item-switch.
    void setTrackContext (juce::String trackPath, juce::String trackSha256,
                          double durationSec, double bpm);

    // Badge state — caller sets when entering "Modified" / "Saved" states
    // by examining per-track persistence.
    void setBadgeState (BadgeState s, juce::String savedDescription = {});

    // Plugin version for JSONL records.
    void setPluginVersion (juce::String v) { pluginVersion_ = std::move (v); }

    // Returns total panel height required by current layout (varies with
    // appMode_). Host calls this when laying out the AdvancedWeightsWindow.
    int getPreferredHeight() const noexcept;

    // Notifies host that panel needs re-layout (mode changed); host calls
    // its own resized() which re-reads getPreferredHeight().
    std::function<void()> onPreferredHeightChanged;

    // Callbacks — fired on user interaction. Host (MainComponent) routes to
    // qualityWeightsByPathMode_ persistence + kickRemixPipeline.
    std::function<void()> onAnyChanged;            // any slider drag/click → re-render
    std::function<void()> onRestoreDefault;        // Restore default → host clears override
    std::function<void()> onSaved;                 // saved JSONL OK → host can refresh badge
    std::function<juce::String()> getCurrentMode;  // host-supplied "duration"/"region"/"blocks"

    // Sesja 107 iter-10 — "Set as defaults" callback. Fired AFTER user
    // confirms via AlertWindow. Host persists the supplied weights + β to
    // ExtState ("reamix.me" / "advanced_defaults") so the new defaults
    // survive REAPER restarts.
    std::function<void(reamix::remix::QualityWeights, BlockAssemblyBeta)>
        onSetAsDefaultsConfirmed;

    // Setter for host to push persisted defaults (loaded from ExtState on
    // panel construction). Replaces compile-time kRawSpecs[i].defaultValue
    // for: tick markers, double-click-return-value, "Restore defaults" target.
    void setDefaultsForTicks (const reamix::remix::QualityWeights& w,
                              const BlockAssemblyBeta& b);

    // juce::Component
    void paint    (juce::Graphics&) override;
    void resized  () override;

private:
    // juce::Slider::Listener
    void sliderValueChanged (juce::Slider* s) override;
    // juce::Button::Listener
    void buttonClicked (juce::Button* b) override;

    // Helpers.
    void applyRawToWeights();
    void rebuildRawFromWeights();
    void openSavePopover();
    void openLoadPopover();
    void confirmRestoreDefault();
    void confirmSetAsDefaults();          // iter-10
    void applyDoubleClickReturnValues();  // iter-10: re-bind after defaults change
    void layoutColumns (juce::Rectangle<int> body);
    void updateBadgeText();

    // Layout constants.
    static constexpr int kHeaderHeight     = 28;
    static constexpr int kBadgeHeight      = 28;
    static constexpr int kSectionHeader    = 22;
    static constexpr int kRowHeight        = 36;
    static constexpr int kRowGap           = 4;
    static constexpr int kActionsHeight    = 36;
    static constexpr int kPanelPadding     = 12;

    // ---- State ----
    BlockAssemblyBeta            beta_       {};
    reamix::remix::QualityWeights weights_   {};
    AppMode                      appMode_ = AppMode::Duration;

    // Iter-10 — user-set defaults. Replaces compile-time kRawSpecs[i].defaultValue
    // semantics for tick markers + double-click reset + "Restore defaults".
    // Initialised from kRawSpecs at construction; overwritten when host calls
    // setDefaultsForTicks() (typically with values loaded from ExtState).
    std::array<double, 8> currentDefaultRaw_ {};
    BlockAssemblyBeta     currentDefaultBeta_ {};

    // Track context.
    juce::String trackPath_;
    juce::String trackSha256_;
    double       durationSec_ = 0.0;
    double       bpm_         = 0.0;
    juce::String pluginVersion_;

    // Badge.
    BadgeState   badgeState_ = BadgeState::NoSave;
    juce::String badgeDescription_;
    juce::Label  badgeLabel_;

    // Storage.
    DevCalibrationStorage storage_;

    // Suppress callback re-fire on programmatic write.
    bool suppressCallbacks_ = false;

    // ---- Children ----
    // Section 1: 8 raw sliders always visible (post-ADR-098 simplification).
    juce::Slider rawSliders_   [8];
    juce::Label  rawLabels_    [8];
    juce::Label  rawReadouts_  [8];

    // Section 2: 4 β-controls (Block Assembly mode only).
    juce::Slider     betaFragmentSlider_;
    juce::Slider     betaOutsideSlider_;
    juce::Slider     betaMinJumpSlider_;
    juce::ToggleButton betaDownbeatToggle_;
    juce::Label      betaFragmentLabel_;
    juce::Label      betaOutsideLabel_;
    juce::Label      betaMinJumpLabel_;
    juce::Label      betaFragmentReadout_;
    juce::Label      betaOutsideReadout_;
    juce::Label      betaMinJumpReadout_;
    juce::Label      betaSectionDisabledNotice_;

    // Section headers.
    juce::Label section1Header_;
    juce::Label section2Header_;

    // Actions row.
    juce::TextButton saveButton_;
    juce::TextButton loadButton_;
    juce::TextButton setAsDefaultsButton_;  // iter-10
    juce::TextButton restoreDefaultButton_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedWeightsPanel)
};

} // namespace reamix::ui
