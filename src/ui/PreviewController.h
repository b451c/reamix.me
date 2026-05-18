#pragma once

#include <juce_core/juce_core.h>

// PreviewController — phase-6 step 4 (ADR-036 D7 step 4, ADR-037 source-
// lifecycle). Owns the low-latency audition path for the Renderer output
// tmp-WAV written by AnalyzeWorker (reamix_current.wav in tempDirectory).
//
// Primary path (ADR-036 ⚑ P1 Y): SWS CF_Preview — PCM_Source_CreateFromFile
// on tmpWavPath_ → CF_CreatePreview → CF_Preview_Play/Stop/SetValue/GetValue.
// Chosen over REAPER core PlayPreview* because SWS ships the mutex-safe
// wrapper used by Lua preview_helper.lua:38-92; no tracks created, no
// save/reopen survival problem.
//
// Secondary path (ADR-036 ⚑ P1 Y — hidden-track fallback): DEV-019 open;
// session 46 ships graceful-degradation modal + StatusBar error instead.
// When CF_CreatePreview is unresolved (SWS not installed), play() returns
// PlayResult::NoSws; MainComponent shows a one-time AlertWindow + sticky
// StatusBar error pointing to sws-extension.org. Full hidden-track
// implementation deferred to a later session.
//
// Threading: all methods are message-thread only. No background polling
// owned by this class — MainComponent::timerCallback polls isPlaying() +
// getPositionSec() at the existing 100 ms tick and forwards playhead to
// WaveformView::setPlayhead.

namespace reamix::ui
{

class PreviewController
{
public:
    enum class PlayResult
    {
        OK,             // playback started; caller should flip UI to Playing
        NoSource,       // tmpWavPath empty or PCM_Source_CreateFromFile failed
        NoSws,          // SWS Extension not installed (CF_CreatePreview null)
        InternalError   // SWS resolved but CF_CreatePreview returned null
    };

    PreviewController();
    ~PreviewController();

    // Called from ReaperPluginEntry after REAPERAPI_LoadAPI, BEFORE the
    // MainComponent (and therefore before any PreviewController instance)
    // exists. Pass the rec->GetFunc pointer; the 5 SWS symbols
    // (CF_CreatePreview / CF_Preview_Play / CF_Preview_Stop /
    // CF_Preview_SetValue / CF_Preview_GetValue) are resolved into
    // translation-unit-static pointers. Safe to call once per plugin load;
    // subsequent calls are a no-op.
    //
    // GetFuncPtr is passed as void* to avoid leaking reaper_plugin.h
    // typedefs into this header; the .cpp casts back to the canonical
    // (void*(*)(const char*)) type.
    //
    // Returns true when all 5 symbols resolve (SWS Extension installed +
    // compatible). Subsequent instances of PreviewController all observe
    // the same availability via isSwsAvailable().
    static bool resolveSwsSymbols (void* getFuncPtr);

    // Post-resolution availability flag. False until resolveSwsSymbols
    // returns true; stable across the REAPER session afterwards.
    bool isSwsAvailable() const noexcept;

    // Load wavPath as a PCM_source, wrap via CF_CreatePreview, start
    // playback from startSec with volume 1.0 (DEV-018 — no slider in
    // mockup step-3; future ADR may expose setVolume).
    //
    // If endSec > startSec, isPlaying() additionally returns false once
    // position reaches endSec - 0.05 (range-limited playback, ADR-039
    // play-selection semantics — EditView DoLoopSelection precedent is
    // one-shot, not loop). endSec < 0 (default) disables range limit.
    //
    // Implicitly calls stop() + releases any prior source before loading
    // the new one (Lua preview_helper.load pattern, preview_helper.lua:42).
    //
    // Returns:
    //   OK            — isPlaying() will return true on the next tick
    //   NoSource      — wavPath empty or PCM_Source_CreateFromFile(nullptr)
    //   NoSws         — caller must surface "install SWS Extension" UI
    //   InternalError — SWS present but CF_CreatePreview returned null;
    //                   caller may retry but likely a source-format issue
    PlayResult play (const juce::String& wavPath,
                     double startSec = 0.0,
                     double endSec   = -1.0);

    // Move the playback position while a preview is active (ADR-039 —
    // click-to-seek on WaveformView, plus mid-playback seek). No-op when
    // not playing. Does not restart playback or alter volume / range.
    void seek (double sec);

    // Mid-playback range change (ADR-039 — drag-select during playback).
    // Seeks to startSec and updates the end-of-range so isPlaying()
    // auto-stops at endSec - 0.05. endSec <= startSec drops the range
    // limit (play continues until end of source). No-op when not playing.
    void updateRange (double startSec, double endSec);

    // Drop the range limit without moving the playhead (ADR-039 — Esc
    // clears selection mid-playback; user wants audio to keep playing).
    // No-op when not playing.
    void clearRange();

    // Set linear preview volume in [0.0, 1.0]. Persists across play()
    // calls — next play() applies the most recent setVolume value to the
    // CF_Preview D_VOLUME slot. Default 1.0. Outside-range values are
    // clamped silently. No-op when SWS not available; safe to call before
    // play() (caller stores requested level for next playback session).
    //
    // DEV-018 (sesja 100) — volume slider host = WaveformView overlay
    // icon (ADR-091 agent-side design; mirrors Edit overlay placement,
    // dyskretny ale funkcjonalny per user directive sesja 100).
    void setVolume (double linear01);

    // Current linear volume; reflects most recent setVolume or default 1.0.
    double getVolume() const noexcept { return volume_; }

    // CF_Preview_Stop + release PCM_source. Idempotent — safe to call
    // when nothing is loaded.
    void stop();

    // Tick-rate state query (intended for 100 ms timer). Uses
    // CF_Preview_GetValue(D_POSITION) + end-detection heuristic ported
    // from preview_helper.is_playing (preview_helper.lua:159-188):
    //   - position >= duration - 0.05 → finished
    //   - position unchanged for >10 consecutive calls → stopped externally
    // Both flip the internal flag; returns false thereafter until the
    // next play() call.
    bool isPlaying();

    // Current position in seconds from the preview start; 0 when nothing
    // playing. Safe to call at tick rate; queries CF_Preview_GetValue.
    double getPositionSec() const;

    // Source duration in seconds; 0 when nothing loaded.
    double getDurationSec() const;

private:
    // Opaque handles — actual types are PCM_source* and
    // preview_register_t* (SWS wraps the latter).
    void* pcmSource_ { nullptr };
    void* preview_   { nullptr };

    // End-detection state (mutable — isPlaying() is not logically const
    // but must match the conventional poll shape).
    bool   isPlayingFlag_ { false };
    double lastPosition_  { -1.0 };
    int    stallCount_    { 0 };

    // Range-limited playback (ADR-039). endSec_ < 0 = no range limit.
    double endSec_        { -1.0 };

    // DEV-018 (sesja 100) — linear preview volume [0.0, 1.0]. Applied at
    // every play() call to CF_Preview D_VOLUME. Default 1.0 preserves
    // pre-sesja-100 behaviour (volume hardcoded to 1.0 inline).
    double volume_        { 1.0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PreviewController)
};

} // namespace reamix::ui
