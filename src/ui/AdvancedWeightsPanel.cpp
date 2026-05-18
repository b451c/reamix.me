#include "AdvancedWeightsPanel.h"

#include "Theme.h"

#include <array>

namespace reamix::ui
{

namespace th = reamix::theme;

// blockAssemblyBetaAtDefault implementation lives in DevCalibrationStorage.cpp
// (same translation unit as the struct's other helpers). Forward-declared in
// DevCalibrationStorage.h.

namespace
{

// 8 raw cost-matrix slider specs. Labels = QualityWeights field names
// (power-user audience reads source; CONTRIBUTING.md documents semantics).
// Tooltips give a one-paragraph English explanation of what each weight
// controls in splice selection.
struct RawSpec
{
    const char* label;
    const char* tooltip;
    double      defaultValue;  // matches kDefaultQualityWeights
};

const std::array<RawSpec, 8> kRawSpecs = { {
    { "waveform",
      "Sample-level cross-correlation cost across the splice boundary.\n"
      "Gates splice candidates by phase alignment of source and destination "
      "waveforms; clickless transitions require non-trivial weight.\n"
      "Lower values open more candidates at the cost of micro-discontinuities.",
      0.50 },
    { "sequential_continuity",
      "Penalty for backward or long-forward jumps in the source timeline.\n"
      "Increases preference for forward-only and short-jump splices.\n"
      "Lower values permit large structural rearrangement.",
      0.20 },
    { "transient_continuity",
      "Onset-envelope match between source and destination beats.\n"
      "Demotes splices that would land mid-attack (preserves percussion "
      "character).\n"
      "Lower values ignore transient alignment.",
      0.15 },
    { "energy",
      "RMS-level match between source and destination boundary windows.\n"
      "Penalizes splices between sections of disparate loudness.\n"
      "Lower values tolerate post-splice level jumps.",
      0.07 },
    { "edge_energy",
      "Sample-accurate RMS at the splice boundary edge.\n"
      "Penalizes splices landing into a high-level transient on the "
      "destination side.\n"
      "Lower values ignore boundary-edge level.",
      0.04 },
    { "bar_align",
      "Binary cost: 1.0 on downbeat, 0.0 elsewhere.\n"
      "Under the harmonic-mean composite, any non-zero weight hard-gates "
      "off-downbeat splices.\n"
      "Set to 0.0 to disable the gate entirely.",
      0.02 },
    { "centroid",
      "Spectral-centroid (brightness) match between source and destination.\n"
      "Prefers splices between tonally-matched sections.\n"
      "Lower values ignore spectral character.",
      0.02 },
    // ADR-088 sesja 98 — vocal phrase continuity. Default 0.0 = no contribution.
    { "vocal_continuity",
      "Penalty for splice boundaries landing inside a detected vocal segment.\n"
      "Protects vocal phrases at the cost of splice-point variety.\n"
      "Default 0.0 (disabled); raise for vocal-heavy material with "
      "mid-word cut artifacts.",
      0.00 },
} };

void styleSlider (juce::Slider& s, double minV, double maxV, double interval, double initial)
{
    s.setSliderStyle (juce::Slider::SliderStyle::LinearHorizontal);
    s.setRange (minV, maxV, interval);
    s.setValue (initial, juce::dontSendNotification);
    s.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    s.setColour (juce::Slider::trackColourId,      th::Bg4);
    s.setColour (juce::Slider::backgroundColourId, th::Bg3);
    s.setColour (juce::Slider::thumbColourId,      th::Accent);
    // Iter-10 — double-click return value is now controlled by the panel
    // (applyDoubleClickReturnValues) so it tracks user-set defaults; helper
    // only seeds the initial bind. Re-binding happens at construction and on
    // every "Set as defaults" confirm.
    s.setDoubleClickReturnValue (true, initial);
}

void styleLabel (juce::Label& l, const juce::String& text, juce::Colour fg, float size, int weight)
{
    l.setText (text, juce::dontSendNotification);
    l.setFont (th::uiFont (size, weight));
    l.setColour (juce::Label::textColourId, fg);
    l.setColour (juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    l.setJustificationType (juce::Justification::centredLeft);
}

void styleTextButton (juce::TextButton& b, juce::Colour fill, juce::Colour text)
{
    b.setColour (juce::TextButton::buttonColourId,   fill);
    b.setColour (juce::TextButton::buttonOnColourId, fill.brighter (0.15f));
    b.setColour (juce::TextButton::textColourOffId,  text);
    b.setColour (juce::TextButton::textColourOnId,   text);
}

// Helper — wraps a UTF-8 byte literal as a juce::String. Required for any
// non-ASCII char in labels/tooltips: raw `const char*` passed to juce::String
// uses platform-default encoding (Latin-1 on macOS via SWELL) producing
// mojibake on render. fromUTF8 forces UTF-8 interpretation. Use for em-dash,
// arrows, stars, etc — even in otherwise-English strings.
inline juce::String u8 (const char* utf8) { return juce::String::fromUTF8 (utf8); }

} // namespace

AdvancedWeightsPanel::AdvancedWeightsPanel()
{
    // Badge label.
    styleLabel (badgeLabel_, u8 ("No save \xe2\x80\x94 default weights"),
                th::Fg2, th::fs::Md, 400);
    badgeLabel_.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (badgeLabel_);

    // Section 1 header.
    styleLabel (section1Header_, u8 ("COST WEIGHTS \xe2\x80\x94 active in all modes"),
                th::Fg2, th::fs::Sm, 600);
    addAndMakeVisible (section1Header_);

    // 8 raw sliders (always visible post-ADR-098).
    for (int i = 0; i < 8; ++i)
    {
        styleSlider (rawSliders_[i], 0.0, 1.0, 0.001, kRawSpecs[i].defaultValue);
        rawSliders_[i].addListener (this);
        addAndMakeVisible (rawSliders_[i]);

        styleLabel (rawLabels_[i], kRawSpecs[i].label, th::Fg1, th::fs::Md, 500);
        // Iter-5: u8() because RawSpec::tooltip is const char* with UTF-8
        // em-dashes / arrows; juce::String(const char*) uses Latin-1 on SWELL
        // → mojibake. Same fix pattern as sesja 106 follow-up.
        rawLabels_[i].setTooltip (u8 (kRawSpecs[i].tooltip));
        rawLabels_[i].setInterceptsMouseClicks (true, false);
        addAndMakeVisible (rawLabels_[i]);

        styleLabel (rawReadouts_[i], juce::String (kRawSpecs[i].defaultValue, 3),
                    th::Fg2, th::fs::Sm, 500);
        rawReadouts_[i].setJustificationType (juce::Justification::centredRight);
        addAndMakeVisible (rawReadouts_[i]);
    }

    // Section 2 header.
    styleLabel (section2Header_, u8 ("BLOCK ASSEMBLY TUNING \xe2\x80\x94 active in Blocks mode only"),
                th::Fg2, th::fs::Sm, 600);
    addAndMakeVisible (section2Header_);

    // β-sliders.
    styleSlider (betaFragmentSlider_, 0.0, 0.20, 0.005, 0.03);
    betaFragmentSlider_.addListener (this);
    addAndMakeVisible (betaFragmentSlider_);
    styleLabel (betaFragmentLabel_, "Fragment penalty",
                th::Fg1, th::fs::Md, 500);
    betaFragmentLabel_.setTooltip (u8 (
        "Penalty for splices near a user-defined block boundary.\n"
        "Higher values bias the optimizer toward the block center.\n"
        "0.0: best-sounding splice even at the block edge.\n"
        "0.20: strict block-boundary respect."));
    addAndMakeVisible (betaFragmentLabel_);
    styleLabel (betaFragmentReadout_, "0.030", th::Fg2, th::fs::Sm, 500);
    betaFragmentReadout_.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (betaFragmentReadout_);

    styleSlider (betaOutsideSlider_, 0.0, 32.0, 1.0, 8.0);
    betaOutsideSlider_.addListener (this);
    addAndMakeVisible (betaOutsideSlider_);
    styleLabel (betaOutsideLabel_, "Outside-block search window",
                th::Fg1, th::fs::Md, 500);
    betaOutsideLabel_.setTooltip (u8 (
        "Optimizer search radius outside the chosen block, in beats.\n"
        "0: strict block boundaries.\n"
        "32: soft boundaries \xe2\x80\x94 block may extend or shrink."));
    addAndMakeVisible (betaOutsideLabel_);
    styleLabel (betaOutsideReadout_, "8 beats", th::Fg2, th::fs::Sm, 500);
    betaOutsideReadout_.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (betaOutsideReadout_);

    styleSlider (betaMinJumpSlider_, 0.0, 32.0, 1.0, 4.0);
    betaMinJumpSlider_.addListener (this);
    addAndMakeVisible (betaMinJumpSlider_);
    styleLabel (betaMinJumpLabel_, "Min spacing between splices",
                th::Fg1, th::fs::Md, 500);
    betaMinJumpLabel_.setTooltip (u8 (
        "Minimum spacing between two consecutive splices, in beats.\n"
        "0: dense splices (chopped character).\n"
        "32: sparse splices, long contiguous regions."));
    addAndMakeVisible (betaMinJumpLabel_);
    styleLabel (betaMinJumpReadout_, "4 beats", th::Fg2, th::fs::Sm, 500);
    betaMinJumpReadout_.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (betaMinJumpReadout_);

    betaDownbeatToggle_.setButtonText ("Splice on downbeat only");
    betaDownbeatToggle_.setToggleState (true, juce::dontSendNotification);
    betaDownbeatToggle_.setColour (juce::ToggleButton::textColourId, th::Fg1);
    betaDownbeatToggle_.setColour (juce::ToggleButton::tickColourId, th::Accent);
    betaDownbeatToggle_.setTooltip (u8 (
        "ON: restricts splices to bar-1 downbeats (rhythmically natural).\n"
        "OFF: permits splices on any beat (flexibility for irregular structures)."));
    betaDownbeatToggle_.addListener (this);
    addAndMakeVisible (betaDownbeatToggle_);

    // Disabled-section notice (shown when appMode != Blocks).
    styleLabel (betaSectionDisabledNotice_,
                "Active only in Block Assembly mode. "
                "Switch to the 'Blocks' tab to tune.",
                th::Fg3, th::fs::Sm, 400);
    addChildComponent (betaSectionDisabledNotice_);

    // Action buttons.
    saveButton_.setButtonText ("Save best...");
    styleTextButton (saveButton_, th::Accent, th::Bg0);
    saveButton_.addListener (this);
    addAndMakeVisible (saveButton_);

    loadButton_.setButtonText ("Load saved...");
    styleTextButton (loadButton_, th::Bg4, th::Fg1);
    loadButton_.addListener (this);
    addAndMakeVisible (loadButton_);

    // Iter-10 — Set as defaults button. Placed between Load and Restore so
    // the action row reads as: build preset → apply preset → make current the
    // baseline → reset to baseline.
    setAsDefaultsButton_.setButtonText ("Set as defaults");
    styleTextButton (setAsDefaultsButton_, th::Bg4, th::Fg1);
    setAsDefaultsButton_.addListener (this);
    addAndMakeVisible (setAsDefaultsButton_);

    restoreDefaultButton_.setButtonText (u8 ("\xe2\x86\xba Restore defaults"));
    styleTextButton (restoreDefaultButton_, th::Bg3, th::Fg2);
    restoreDefaultButton_.addListener (this);
    addAndMakeVisible (restoreDefaultButton_);

    // Initialise weights + ticks defaults.
    weights_ = reamix::remix::kDefaultQualityWeights;
    for (int i = 0; i < 8; ++i)
        currentDefaultRaw_[i] = kRawSpecs[i].defaultValue;
    currentDefaultBeta_ = BlockAssemblyBeta{};
    rebuildRawFromWeights();
    applyDoubleClickReturnValues();
    updateBadgeText();
}

AdvancedWeightsPanel::~AdvancedWeightsPanel() = default;

// ----------------------------------------------------------------
// Setters (suppressCallbacks_ pattern from AuditionBar).
// ----------------------------------------------------------------

void AdvancedWeightsPanel::setBlockAssemblyBeta (const BlockAssemblyBeta& b)
{
    suppressCallbacks_ = true;
    beta_ = b;
    betaFragmentSlider_.setValue (b.fragment_penalty_weight, juce::dontSendNotification);
    betaOutsideSlider_.setValue ((double) b.outside_window_beats, juce::dontSendNotification);
    betaMinJumpSlider_.setValue ((double) b.min_jump_beats, juce::dontSendNotification);
    betaDownbeatToggle_.setToggleState (b.downbeat_only_splices, juce::dontSendNotification);
    betaFragmentReadout_.setText (juce::String (b.fragment_penalty_weight, 3),
                                  juce::dontSendNotification);
    betaOutsideReadout_.setText (juce::String (b.outside_window_beats) + " beats",
                                 juce::dontSendNotification);
    betaMinJumpReadout_.setText (juce::String (b.min_jump_beats) + " beats",
                                 juce::dontSendNotification);
    suppressCallbacks_ = false;
    repaint();
}

void AdvancedWeightsPanel::setRawWeights (const reamix::remix::QualityWeights& w)
{
    suppressCallbacks_ = true;
    weights_ = w;
    rebuildRawFromWeights();
    suppressCallbacks_ = false;
    repaint();
}

void AdvancedWeightsPanel::setAppMode (AppMode m)
{
    appMode_ = m;
    const bool enabled = (m == AppMode::Blocks);
    betaFragmentSlider_.setEnabled (enabled);
    betaOutsideSlider_.setEnabled (enabled);
    betaMinJumpSlider_.setEnabled (enabled);
    betaDownbeatToggle_.setEnabled (enabled);
    betaSectionDisabledNotice_.setVisible (! enabled);
    if (onPreferredHeightChanged) onPreferredHeightChanged();
    resized();
    repaint();
}

void AdvancedWeightsPanel::setTrackContext (juce::String trackPath, juce::String trackSha256,
                                            double durationSec, double bpm)
{
    trackPath_   = std::move (trackPath);
    trackSha256_ = std::move (trackSha256);
    durationSec_ = durationSec;
    bpm_         = bpm;
    updateBadgeText();
}

void AdvancedWeightsPanel::setBadgeState (BadgeState s, juce::String desc)
{
    badgeState_       = s;
    badgeDescription_ = std::move (desc);
    updateBadgeText();
}

// ----------------------------------------------------------------
// Helpers.
// ----------------------------------------------------------------

void AdvancedWeightsPanel::applyRawToWeights()
{
    weights_.waveform              = rawSliders_[0].getValue();
    weights_.sequential_continuity = rawSliders_[1].getValue();
    weights_.transient_continuity  = rawSliders_[2].getValue();
    weights_.energy                = rawSliders_[3].getValue();
    weights_.edge_energy           = rawSliders_[4].getValue();
    weights_.bar_align             = rawSliders_[5].getValue();
    weights_.centroid              = rawSliders_[6].getValue();
    weights_.vocal_continuity      = rawSliders_[7].getValue();  // ADR-088 sesja 98

    for (int i = 0; i < 8; ++i)
        rawReadouts_[i].setText (juce::String (rawSliders_[i].getValue(), 3),
                                 juce::dontSendNotification);
}

void AdvancedWeightsPanel::rebuildRawFromWeights()
{
    const std::array<double, 8> vals = {
        weights_.waveform,
        weights_.sequential_continuity,
        weights_.transient_continuity,
        weights_.energy,
        weights_.edge_energy,
        weights_.bar_align,
        weights_.centroid,
        weights_.vocal_continuity
    };
    for (int i = 0; i < 8; ++i)
    {
        rawSliders_[i].setValue (vals[i], juce::dontSendNotification);
        rawReadouts_[i].setText (juce::String (vals[i], 3), juce::dontSendNotification);
    }
}

void AdvancedWeightsPanel::updateBadgeText()
{
    juce::String text;
    juce::Colour fg = th::Fg2;
    int weight = 400;
    switch (badgeState_)
    {
        case BadgeState::NoSave:
            text = u8 ("No save \xe2\x80\x94 default weights");
            fg = th::Fg2; weight = 400; break;
        case BadgeState::Modified:
            text = "Weights modified (unsaved)";
            fg = th::Warn; weight = 600; break;
        case BadgeState::Saved:
            text = badgeDescription_.isEmpty()
                 ? juce::String ("Saved")
                 : "Saved: " + badgeDescription_;
            fg = th::Good; weight = 500; break;
        case BadgeState::Loaded:
            text = "Loaded saved weights";
            fg = th::Info; weight = 400; break;
    }
    badgeLabel_.setText (text, juce::dontSendNotification);
    badgeLabel_.setFont (th::uiFont (th::fs::Md, weight));
    badgeLabel_.setColour (juce::Label::textColourId, fg);
}

// ----------------------------------------------------------------
// juce::Slider::Listener
// ----------------------------------------------------------------

void AdvancedWeightsPanel::sliderValueChanged (juce::Slider* s)
{
    if (suppressCallbacks_) return;

    // Section 1 raw sliders.
    for (int i = 0; i < 8; ++i)
    {
        if (s == &rawSliders_[i])
        {
            applyRawToWeights();
            if (badgeState_ == BadgeState::NoSave
                || badgeState_ == BadgeState::Saved
                || badgeState_ == BadgeState::Loaded)
                setBadgeState (BadgeState::Modified);
            if (onAnyChanged) onAnyChanged();
            return;
        }
    }
    // Section 2 β-sliders.
    if (s == &betaFragmentSlider_)
    {
        beta_.fragment_penalty_weight = s->getValue();
        betaFragmentReadout_.setText (juce::String (beta_.fragment_penalty_weight, 3),
                                      juce::dontSendNotification);
    }
    else if (s == &betaOutsideSlider_)
    {
        beta_.outside_window_beats = (int) s->getValue();
        betaOutsideReadout_.setText (juce::String (beta_.outside_window_beats) + " beats",
                                     juce::dontSendNotification);
    }
    else if (s == &betaMinJumpSlider_)
    {
        beta_.min_jump_beats = (int) s->getValue();
        betaMinJumpReadout_.setText (juce::String (beta_.min_jump_beats) + " beats",
                                     juce::dontSendNotification);
    }
    if (badgeState_ != BadgeState::Modified) setBadgeState (BadgeState::Modified);
    if (onAnyChanged) onAnyChanged();
}

// ----------------------------------------------------------------
// juce::Button::Listener
// ----------------------------------------------------------------

void AdvancedWeightsPanel::buttonClicked (juce::Button* b)
{
    if (b == &saveButton_)            { openSavePopover();      return; }
    if (b == &loadButton_)            { openLoadPopover();      return; }
    if (b == &setAsDefaultsButton_)   { confirmSetAsDefaults(); return; }
    if (b == &restoreDefaultButton_)  { confirmRestoreDefault(); return; }
    if (b == &betaDownbeatToggle_)
    {
        if (suppressCallbacks_) return;
        beta_.downbeat_only_splices = betaDownbeatToggle_.getToggleState();
        if (badgeState_ != BadgeState::Modified) setBadgeState (BadgeState::Modified);
        if (onAnyChanged) onAnyChanged();
        return;
    }
}

// ----------------------------------------------------------------
// Save / Load / Restore popovers.
// ----------------------------------------------------------------

void AdvancedWeightsPanel::openSavePopover()
{
    // Modal popover [genre dropdown + note textbox + Save/Cancel].
    static const std::array<const char*, 9> kGenres = {
        "rock", "pop", "hip-hop", "EDM", "acoustic",
        "jazz", "metal", "instrumental", "other"
    };

    juce::AlertWindow w ("Save best weights",
                         "Pick a genre and optionally add a note.",
                         juce::AlertWindow::NoIcon);
    juce::StringArray genreOptions;
    for (auto* g : kGenres) genreOptions.add (g);
    w.addComboBox ("genre", genreOptions, "Genre");
    w.addTextEditor ("note", "", "Note (optional)", false);
    if (auto* ed = w.getTextEditor ("note")) ed->setMultiLine (true, true);
    w.addButton ("Save",   1, juce::KeyPress (juce::KeyPress::returnKey));
    w.addButton ("Cancel", 0, juce::KeyPress (juce::KeyPress::escapeKey));

    if (w.runModalLoop() != 1) return;

    DevCalibrationRecord rec;
    rec.timestamp           = DevCalibrationStorage::nowIsoUtc();
    rec.track_path          = trackPath_;
    rec.track_sha256        = trackSha256_;
    rec.duration_sec        = durationSec_;
    rec.bpm                 = bpm_;
    rec.weights_raw         = weights_;
    rec.block_assembly_beta = beta_;
    if (auto* cb = w.getComboBoxComponent ("genre"))
        rec.genre = cb->getText().toLowerCase();
    if (auto* ed = w.getTextEditor ("note"))
        rec.user_note = ed->getText().substring (0, 500);
    rec.mode_evaluated = getCurrentMode ? getCurrentMode() : juce::String ("unknown");
    rec.plugin_version = pluginVersion_;

    if (storage_.append (rec))
    {
        setBadgeState (BadgeState::Saved,
                       rec.genre + u8 (" \xe2\x80\x94 ") + rec.timestamp.substring (0, 16).replaceCharacter ('T', ' '));
        if (onSaved) onSaved();
    }
    else
    {
        juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::WarningIcon,
            "Save error",
            "Could not write to " + storage_.storePath().getFullPathName());
    }
}

void AdvancedWeightsPanel::openLoadPopover()
{
    // Post-ADR-098: single source of truth (raw weights). Load record →
    // setRawWeights(r.weights_raw). Legacy v1 records with perceptual_sliders
    // field are ignored (raw is always populated per sesja 98 schema D7).
    auto records = storage_.loadAll();
    if (records.empty())
    {
        juce::AlertWindow::showMessageBoxAsync (juce::AlertWindow::NoIcon,
            "No saves",
            "The dev-calibration.jsonl file is empty.");
        return;
    }

    juce::PopupMenu menu;
    int idx = 1;
    for (auto& r : records)
    {
        const auto when = r.timestamp.substring (0, 16).replaceCharacter ('T', ' ');
        // Track filename only (not full path) for compact row.
        const auto trackName = juce::File (r.track_path).getFileNameWithoutExtension();
        const auto trackTag  = trackName.isEmpty() ? juce::String ("?")
                                                   : trackName.substring (0, 28);
        const auto note = r.user_note.substring (0, 50);
        // Mark current-track entries with a leading "★" so user can spot them
        // in cross-track list. Also handles trackPath_ empty (no current track).
        const bool currentTrack = (! trackPath_.isEmpty()) && (r.track_path == trackPath_);
        const juce::String marker = currentTrack ? u8 ("\xe2\x98\x85 ")
                                                 : juce::String ("  ");
        const auto label = marker + when + "  " + r.genre + "  [" + trackTag + "]"
                         + (note.isEmpty() ? juce::String() : u8 ("  \xe2\x80\x94 ") + note);
        menu.addItem (idx++, label);
    }
    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (loadButton_),
        [this, records = std::move (records)] (int chosen)
        {
            if (chosen <= 0 || chosen > (int) records.size()) return;
            const auto& r = records[(std::size_t) (chosen - 1)];

            // Post-ADR-098: raw is always source of truth. No more branch on
            // advanced_used (field dropped from schema v2).
            suppressCallbacks_ = true;
            setRawWeights (r.weights_raw);
            setBlockAssemblyBeta (r.block_assembly_beta);
            suppressCallbacks_ = false;

            setBadgeState (BadgeState::Loaded,
                           r.genre + u8 (" \xe2\x80\x94 ") + r.timestamp.substring (0, 16).replaceCharacter ('T', ' '));
            if (onAnyChanged) onAnyChanged();
        });
}

void AdvancedWeightsPanel::confirmRestoreDefault()
{
    juce::AlertWindow::showOkCancelBox (juce::AlertWindow::QuestionIcon,
        "Restore defaults",
        "Reset all sliders to default values?",
        "Yes", "Cancel", nullptr,
        juce::ModalCallbackFunction::create ([this] (int result)
        {
            if (result != 1) return;
            // Iter-10 — restore to the CURRENT defaults (which may be user-set
            // via "Set as defaults" — see currentDefaultRaw_/Beta_), not the
            // compile-time kDefaultQualityWeights baseline.
            reamix::remix::QualityWeights w;
            w.waveform              = currentDefaultRaw_[0];
            w.sequential_continuity = currentDefaultRaw_[1];
            w.transient_continuity  = currentDefaultRaw_[2];
            w.energy                = currentDefaultRaw_[3];
            w.edge_energy           = currentDefaultRaw_[4];
            w.bar_align             = currentDefaultRaw_[5];
            w.centroid              = currentDefaultRaw_[6];
            w.vocal_continuity      = currentDefaultRaw_[7];
            setRawWeights (w);
            setBlockAssemblyBeta (currentDefaultBeta_);
            setBadgeState (BadgeState::NoSave);
            if (onRestoreDefault) onRestoreDefault();
            if (onAnyChanged)     onAnyChanged();
        }));
}

void AdvancedWeightsPanel::confirmSetAsDefaults()
{
    juce::AlertWindow::showOkCancelBox (juce::AlertWindow::QuestionIcon,
        "Set as defaults",
        "Save current slider positions as the new defaults?\n"
        "Tick markers, double-click reset, and 'Restore defaults' will use these "
        "values until you set new defaults or reinstall the plugin.",
        "Yes", "Cancel", nullptr,
        juce::ModalCallbackFunction::create ([this] (int result)
        {
            if (result != 1) return;

            // Capture current slider state as the new defaults.
            currentDefaultRaw_[0] = weights_.waveform;
            currentDefaultRaw_[1] = weights_.sequential_continuity;
            currentDefaultRaw_[2] = weights_.transient_continuity;
            currentDefaultRaw_[3] = weights_.energy;
            currentDefaultRaw_[4] = weights_.edge_energy;
            currentDefaultRaw_[5] = weights_.bar_align;
            currentDefaultRaw_[6] = weights_.centroid;
            currentDefaultRaw_[7] = weights_.vocal_continuity;
            currentDefaultBeta_   = beta_;

            // Re-bind double-click reset targets to the new defaults so the
            // user can return to them with a dbl-click on any slider.
            applyDoubleClickReturnValues();

            // Re-render so the white tick markers jump to the new positions.
            repaint();

            // Host (MainComponent) persists to ExtState
            // ("reamix.me" / "advanced_defaults") so the new defaults survive
            // REAPER restart.
            if (onSetAsDefaultsConfirmed)
                onSetAsDefaultsConfirmed (weights_, beta_);
        }));
}

void AdvancedWeightsPanel::applyDoubleClickReturnValues()
{
    for (int i = 0; i < 8; ++i)
        rawSliders_[i].setDoubleClickReturnValue (true, currentDefaultRaw_[i]);
    betaFragmentSlider_.setDoubleClickReturnValue (true, currentDefaultBeta_.fragment_penalty_weight);
    betaOutsideSlider_ .setDoubleClickReturnValue (true, (double) currentDefaultBeta_.outside_window_beats);
    betaMinJumpSlider_ .setDoubleClickReturnValue (true, (double) currentDefaultBeta_.min_jump_beats);
}

void AdvancedWeightsPanel::setDefaultsForTicks (const reamix::remix::QualityWeights& w,
                                                const BlockAssemblyBeta& b)
{
    currentDefaultRaw_[0] = w.waveform;
    currentDefaultRaw_[1] = w.sequential_continuity;
    currentDefaultRaw_[2] = w.transient_continuity;
    currentDefaultRaw_[3] = w.energy;
    currentDefaultRaw_[4] = w.edge_energy;
    currentDefaultRaw_[5] = w.bar_align;
    currentDefaultRaw_[6] = w.centroid;
    currentDefaultRaw_[7] = w.vocal_continuity;
    currentDefaultBeta_   = b;
    applyDoubleClickReturnValues();
    repaint();
}

// ----------------------------------------------------------------
// Layout + paint.
// ----------------------------------------------------------------

int AdvancedWeightsPanel::getPreferredHeight() const noexcept
{
    // Computed sum: header + badge + section1 (header + 8 raw rows) +
    // section2 (header + disabled-notice when not Blocks + 3 β-rows +
    // downbeat toggle) + actions row + paddings.
    int h = kPanelPadding;
    h += kHeaderHeight + kRowGap;
    h += kBadgeHeight + kRowGap;
    h += kSectionHeader + kRowGap;
    for (int i = 0; i < 8; ++i)
        h += kRowHeight + kRowGap;
    h += kSectionHeader + kRowGap;
    if (appMode_ != AppMode::Blocks)
        h += kRowHeight + kRowGap;  // disabled-section notice
    for (int i = 0; i < 3; ++i)
        h += kRowHeight + kRowGap;  // β-sliders
    h += kRowHeight + kRowGap;       // downbeat toggle
    h += kActionsHeight;
    h += kPanelPadding;
    return h;
}

void AdvancedWeightsPanel::resized()
{
    auto body = getLocalBounds().reduced (kPanelPadding, kPanelPadding);

    auto headerStrip = body.removeFromTop (kHeaderHeight);
    (void) headerStrip;

    auto badgeRow = body.removeFromTop (kBadgeHeight);
    badgeLabel_.setBounds (badgeRow);
    body.removeFromTop (kRowGap);

    layoutColumns (body);
}

void AdvancedWeightsPanel::layoutColumns (juce::Rectangle<int> body)
{
    constexpr int kLabelColW   = 220;
    constexpr int kReadoutColW = 110;

    auto layoutSliderRow = [&body] (juce::Slider& s, juce::Label& lbl,
                                     juce::Label& read)
    {
        auto row = body.removeFromTop (kRowHeight);
        lbl.setBounds  (row.removeFromLeft (kLabelColW));
        read.setBounds (row.removeFromRight (kReadoutColW));
        s.setBounds    (row.reduced (8, 8));
        body.removeFromTop (kRowGap);
    };

    // Section 1 header.
    section1Header_.setBounds (body.removeFromTop (kSectionHeader));
    body.removeFromTop (kRowGap);

    for (int i = 0; i < 8; ++i)
        layoutSliderRow (rawSliders_[i], rawLabels_[i], rawReadouts_[i]);

    // Section 2 header.
    section2Header_.setBounds (body.removeFromTop (kSectionHeader));
    body.removeFromTop (kRowGap);

    if (appMode_ != AppMode::Blocks)
    {
        betaSectionDisabledNotice_.setBounds (body.removeFromTop (kRowHeight));
        body.removeFromTop (kRowGap);
    }

    layoutSliderRow (betaFragmentSlider_, betaFragmentLabel_, betaFragmentReadout_);
    layoutSliderRow (betaOutsideSlider_,  betaOutsideLabel_,  betaOutsideReadout_);
    layoutSliderRow (betaMinJumpSlider_,  betaMinJumpLabel_,  betaMinJumpReadout_);
    betaDownbeatToggle_.setBounds (body.removeFromTop (kRowHeight).withTrimmedLeft (kLabelColW));
    body.removeFromTop (kRowGap);

    // Actions row at bottom — pull from bottom of remaining body.
    // Iter-10 — 4 buttons: Save / Load / Set as defaults / Restore defaults.
    auto actions = body.removeFromBottom (kActionsHeight);
    const int btnW = (actions.getWidth() - 3 * kRowGap) / 4;
    saveButton_.setBounds         (actions.removeFromLeft (btnW));
    actions.removeFromLeft (kRowGap);
    loadButton_.setBounds         (actions.removeFromLeft (btnW));
    actions.removeFromLeft (kRowGap);
    setAsDefaultsButton_.setBounds (actions.removeFromLeft (btnW));
    actions.removeFromLeft (kRowGap);
    restoreDefaultButton_.setBounds (actions);
}

void AdvancedWeightsPanel::paint (juce::Graphics& g)
{
    g.fillAll (th::Bg2);

    // Header bar.
    auto bounds = getLocalBounds();
    auto header = bounds.removeFromTop (kHeaderHeight + kPanelPadding).reduced (kPanelPadding, 0);
    g.setColour (th::Fg0);
    g.setFont (th::uiFont (th::fs::Md, 600));
    g.drawText ("ADVANCED WEIGHTS", header, juce::Justification::centredLeft, false);

    // Default tick marks on each slider track. Painted on top of slider drawn
    // area. Position matches JUCE's internal slider sliderPos formula:
    //   sliderPos = (bounds.X + thumbR) + frac × (bounds.W - 2 × thumbR)
    // Earlier the tick used raw bounds (no thumbR offset), so for low default
    // values the tick crept under the row's left padding / label edge.
    // thumbR=7 mirrors LookAndFeelReamix::drawLinearSlider's thumbDiameter=14.
    auto drawDefaultTick = [&g] (const juce::Slider& s, double minV, double maxV, double dv)
    {
        if (! s.isShowing()) return;
        const auto sb = s.getBoundsInParent();
        // Iter-9: matches LookAndFeelReamix::getSliderThumbRadius (returns 6)
        // so tick and JUCE-computed sliderPos use the same offset → thumb
        // sits exactly on the tick when value == default.
        constexpr int thumbR = 6;
        const int trackLeft  = sb.getX() + thumbR;
        const int trackRight = sb.getRight() - thumbR;
        const int trackWidth = juce::jmax (1, trackRight - trackLeft);
        const double frac = juce::jlimit (0.0, 1.0, (dv - minV) / (maxV - minV));
        const int x = trackLeft + (int) std::lround (trackWidth * frac);
        // 1 px wide × 18 px tall — overhangs 4 px above + below the 10 px
        // thumb so the tick stays visible even when value == default.
        g.setColour (th::Fg1);
        g.fillRect (x, sb.getY() + sb.getHeight() / 2 - 9, 1, 18);
    };
    // Iter-10 — ticks reflect the CURRENT defaults (kRawSpecs at first run,
    // user-set after "Set as defaults"). currentDefaultRaw_ / currentDefaultBeta_
    // are kept in sync via applyDoubleClickReturnValues() + repaint().
    for (int i = 0; i < 8; ++i)
        drawDefaultTick (rawSliders_[i], 0.0, 1.0, currentDefaultRaw_[i]);
    drawDefaultTick (betaFragmentSlider_, 0.0, 0.20, currentDefaultBeta_.fragment_penalty_weight);
    drawDefaultTick (betaOutsideSlider_,  0.0, 32.0, (double) currentDefaultBeta_.outside_window_beats);
    drawDefaultTick (betaMinJumpSlider_,  0.0, 32.0, (double) currentDefaultBeta_.min_jump_beats);
}

} // namespace reamix::ui
