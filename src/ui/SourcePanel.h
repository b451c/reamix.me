#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <chrono>
#include <functional>

// SourcePanel — 68h grid: item name (row 1, col 1) + metadata row
// (duration/BPM/beats, row 2, col 1) + Analyze Again button (row 1-2, col 3).
// When `analyzing_=true`, grows to ~86h to expose the 4h progress bar +
// mono progress label.
//
// Ported from phases/phase-6-ui/reamix.me JUCE/plugin.css § .rx-source /
// .rx-source-name / .rx-source-meta / .rx-analyze-row / .rx-progress*
// (L80-127) + shell.jsx § RxSource (L25-66) + primitives.jsx § rxFmt
// (L45-49).
//
// Step-3 scope (ADR-036 D7):
// - SourceInfo passed in via setSource() — synthetic values in step 3 /
//   step 3b; ReaperBridge-backed at step 9 (ADR-036 D7).
// - onAnalyze signal fires on button click (stubbed to StatusBar in step 3;
//   worker-thread + phase-1..5 pipeline wiring in step 3b).
// - setAnalyzing(true, progress01) swaps button label to "Analyzing…" +
//   reveals progress bar; the panel's preferred height grows from 68 to 86.
// - Empty state (shell.jsx:26-36) renders "No item selected" + disabled
//   Analyze button.

namespace reamix::ui
{

struct SourceInfo
{
    juce::String name;         // e.g. "Nightdrift — Sable Orion (master v3).wav"
    double       duration { 0.0 };  // seconds
    double       bpm      { 0.0 };
    int          beats    { 0 };
    bool         empty    { false };  // true → "No item selected" state
};

class SourcePanel : public juce::Component,
                    private juce::Timer
{
public:
    SourcePanel();
    ~SourcePanel() override = default;

    void setSource    (SourceInfo);
    // Sesja 64 — added stageLabel (e.g. "Detecting beats", "Computing
    // transitions") rendered before the percent in the progress row. Empty
    // string falls back to the legacy "{pct}% · {eta}s" layout.
    void setAnalyzing (bool analyzing, double progress01 = 0.0,
                       juce::String stageLabel = {});

    // Sesja 60 — button label = "Analyze" when no AnalysisBundle exists yet
    // for the current source path (incl. fresh post-restart state), "Analyze
    // Again" when re-analyze would replace existing in-memory cache. Default
    // false (post-restart / no item selected). Caller (MainComponent) flips
    // on AnalysisBundle insert/erase + on item path change.
    void setHasAnalysis (bool);

    bool isAnalyzing() const noexcept { return analyzing_; }

    // Preferred height given current state — MainComponent queries this so
    // the progress-row expansion is reflected in the layout.
    int getPreferredHeight() const noexcept;

    std::function<void()>     onAnalyze;

    void paint     (juce::Graphics&) override;
    void resized   () override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;

private:
    enum class HitRegion { None, Analyze };

    HitRegion regionAt (juce::Point<int>) const;
    void setHover   (HitRegion);
    void setPressed (HitRegion);

    SourceInfo info_;
    bool   analyzing_   { false };
    double progress_    { 0.0 };    // [0..1]
    juce::String stageLabel_;       // sesja 64 — current pipeline stage label
    bool   hasAnalysis_ { false };  // sesja 60 — drives button label text

    // Sesja 108 — real-time elapsed-time counter during analysis. Decoupled
    // from progress percentage (which still updates step-wise from pipeline
    // callbacks). 10 Hz Timer started on the false → true edge of
    // setAnalyzing; paint() reads currentElapsedSec_ instead of synthetic
    // ETA. Switching item mid-analyze resets the counter (worker start time
    // not exposed by the pipeline).
    void timerCallback() override;
    std::chrono::steady_clock::time_point analyzeStart_;
    double currentElapsedSec_ = 0.0;
    static constexpr int kElapsedTimerHz = 10;

    juce::Rectangle<int> analyzeButtonBounds_;
    juce::Rectangle<int> progressRowBounds_;

    HitRegion hover_   { HitRegion::None };
    HitRegion pressed_ { HitRegion::None };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SourcePanel)
};

} // namespace reamix::ui
