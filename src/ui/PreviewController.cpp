#include "ui/PreviewController.h"

#include <algorithm>

#if REAMIX_WITH_REAPER_IO
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"
#endif

namespace
{
#if REAMIX_WITH_REAPER_IO

// SWS CF_Preview API — resolved at plugin init via rec->GetFunc.
// Signatures from sws-extension source (sws-extension.org), matched to
// Lua bindings in preview_helper.lua:20-134.
using GetFuncT = void* (*) (const char* name);

using CF_CreatePreviewT  = preview_register_t* (*) (PCM_source* source);
using CF_PreviewPlayT    = bool (*) (preview_register_t* preview);
using CF_PreviewStopT    = bool (*) (preview_register_t* preview);
using CF_PreviewSetValT  = bool (*) (preview_register_t* preview,
                                     const char* param, double value);
using CF_PreviewGetValT  = bool (*) (preview_register_t* preview,
                                     const char* param, double* out);

CF_CreatePreviewT s_cfCreatePreview   = nullptr;
CF_PreviewPlayT   s_cfPreviewPlay     = nullptr;
CF_PreviewStopT   s_cfPreviewStop     = nullptr;
CF_PreviewSetValT s_cfPreviewSetValue = nullptr;
CF_PreviewGetValT s_cfPreviewGetValue = nullptr;

// Captured at plugin init from rec->GetFunc. Used for lazy SWS resolution
// in ensureSwsResolved() — see DIAG-SWS-48 rationale in resolveSwsSymbols.
GetFuncT s_getFunc = nullptr;

#endif // REAMIX_WITH_REAPER_IO

// Plugin-lifetime singletons. s_swsAvailable flips true on the first
// successful ensureSwsResolved(); once true it sticks. Failed resolutions do
// NOT set a resolved-sticky flag — we retry on every play() because REAPER
// load-order can place SWS after our plugin (session 48 DIAG-SWS-48 proved
// all 5 CF_* symbols null at ReaperPluginEntry time even with SWS installed).
bool s_swsAvailable = false;

} // anonymous namespace

namespace reamix::ui
{

PreviewController::PreviewController() = default;

PreviewController::~PreviewController()
{
    stop();
}

bool PreviewController::resolveSwsSymbols (void* getFuncPtr)
{
    // DIAG-SWS-48: this function was originally eager — it resolved all 5
    // CF_* symbols at plugin init and cached the result (pass/fail) for the
    // rest of the REAPER session. Session-48 diagnostic showed every symbol
    // returns nullptr at ReaperPluginEntry time even with SWS installed:
    //   CF_CreatePreview=0x0 Play=0x0 Stop=0x0 SetVal=0x0 GetVal=0x0 | SWSVer=0x0
    // Root cause: REAPER loads extension plugins alphabetically; `reamix`
    // comes before `sws64`, so SWS has not yet registered its C API with
    // REAPER when our entry point runs.
    //
    // Fix: capture the getFunc pointer here, but defer resolution to first
    // use via ensureSwsResolved() (called from play()). By the time the user
    // clicks Preview, SWS is fully loaded and the symbols resolve normally.
#if REAMIX_WITH_REAPER_IO
    if (getFuncPtr == nullptr)
        return false;
    s_getFunc = reinterpret_cast<GetFuncT> (getFuncPtr);
    return true;
#else
    (void) getFuncPtr;
    return false;
#endif
}

namespace
{
    // Lazy resolver — tries once per call until the 5 SWS symbols are all
    // non-null. Subsequent calls short-circuit on the `s_swsAvailable` flag.
    // Safe to call on every play() attempt: the probe is 5 string-keyed
    // function-table lookups, cheap enough for user-initiated clicks.
    bool ensureSwsResolved()
    {
#if REAMIX_WITH_REAPER_IO
        if (s_swsAvailable) return true;
        if (s_getFunc == nullptr) return false;

        s_cfCreatePreview   = reinterpret_cast<CF_CreatePreviewT>  (s_getFunc ("CF_CreatePreview"));
        s_cfPreviewPlay     = reinterpret_cast<CF_PreviewPlayT>    (s_getFunc ("CF_Preview_Play"));
        s_cfPreviewStop     = reinterpret_cast<CF_PreviewStopT>    (s_getFunc ("CF_Preview_Stop"));
        s_cfPreviewSetValue = reinterpret_cast<CF_PreviewSetValT>  (s_getFunc ("CF_Preview_SetValue"));
        s_cfPreviewGetValue = reinterpret_cast<CF_PreviewGetValT>  (s_getFunc ("CF_Preview_GetValue"));

        s_swsAvailable = s_cfCreatePreview   != nullptr
                      && s_cfPreviewPlay     != nullptr
                      && s_cfPreviewStop     != nullptr
                      && s_cfPreviewSetValue != nullptr
                      && s_cfPreviewGetValue != nullptr;
        return s_swsAvailable;
#else
        return false;
#endif
    }
}

bool PreviewController::isSwsAvailable() const noexcept
{
    return s_swsAvailable;
}

bool PreviewController::probeSwsAvailable()
{
    ensureSwsResolved();
    return s_swsAvailable;
}

PreviewController::PlayResult PreviewController::play (const juce::String& wavPath,
                                                        double startSec,
                                                        double endSec)
{
    if (wavPath.isEmpty())
        return PlayResult::NoSource;

#if REAMIX_WITH_REAPER_IO
    // Lazy SWS resolution (DIAG-SWS-48 session 48) — SWS loads after our
    // plugin due to alphabetical load order, so we probe symbols at first
    // Preview click rather than at plugin init.
    if (! ensureSwsResolved())
        return PlayResult::NoSws;

    // preview_helper.load (preview_helper.lua:38-61): stop current, create
    // new source, create new preview. We follow the same ordering.
    stop();

    PCM_source* source = PCM_Source_CreateFromFile (wavPath.toRawUTF8());
    if (source == nullptr)
        return PlayResult::NoSource;

    preview_register_t* preview = s_cfCreatePreview (source);
    if (preview == nullptr)
    {
        PCM_Source_Destroy (source);
        return PlayResult::InternalError;
    }

    pcmSource_ = source;
    preview_   = preview;

    // preview_helper.play (preview_helper.lua:63-83): start position, volume,
    // loop flag, then Play.
    if (startSec >= 0.0)
        s_cfPreviewSetValue (preview, "D_POSITION", startSec);

    // DEV-018 (sesja 100) — apply persistent linear volume member.
    // Default 1.0; user-set via WaveformView overlay → setVolume().
    s_cfPreviewSetValue (preview, "D_VOLUME", volume_);

    s_cfPreviewSetValue (preview, "B_LOOP", 0.0);
    s_cfPreviewPlay (preview);

    isPlayingFlag_ = true;
    lastPosition_  = -1.0;
    stallCount_    = 0;
    endSec_        = (endSec > startSec) ? endSec : -1.0;

    return PlayResult::OK;
#else
    (void) startSec;
    (void) endSec;
    return PlayResult::NoSource;
#endif
}

void PreviewController::seek (double sec)
{
#if REAMIX_WITH_REAPER_IO
    if (preview_ == nullptr || ! isPlayingFlag_ || s_cfPreviewSetValue == nullptr)
        return;
    s_cfPreviewSetValue (static_cast<preview_register_t*> (preview_),
                         "D_POSITION", sec);
    // Reset stall counter so end-detection does not fire if the seek
    // happens to land in a near-silent region.
    lastPosition_ = -1.0;
    stallCount_   = 0;
#else
    (void) sec;
#endif
}

void PreviewController::updateRange (double startSec, double endSec)
{
#if REAMIX_WITH_REAPER_IO
    if (preview_ == nullptr || ! isPlayingFlag_ || s_cfPreviewSetValue == nullptr)
        return;
    s_cfPreviewSetValue (static_cast<preview_register_t*> (preview_),
                         "D_POSITION", startSec);
    endSec_       = (endSec > startSec) ? endSec : -1.0;
    lastPosition_ = -1.0;
    stallCount_   = 0;
#else
    (void) startSec;
    (void) endSec;
#endif
}

void PreviewController::clearRange()
{
    if (! isPlayingFlag_) return;
    endSec_ = -1.0;
}

void PreviewController::setVolume (double linear01)
{
    // Clamp to [0.0, 1.0] silently. CF_Preview D_VOLUME accepts >1.0 but
    // semantics are "amplify" which we do not surface in UI; cap at unity
    // gain. Negative is invalid.
    volume_ = std::clamp (linear01, 0.0, 1.0);

#if REAMIX_WITH_REAPER_IO
    // Live update during playback so user hears the change immediately
    // while dragging the slider. No-op if SWS not resolved or not playing.
    if (preview_ != nullptr && isPlayingFlag_ && s_cfPreviewSetValue != nullptr)
    {
        s_cfPreviewSetValue (static_cast<preview_register_t*> (preview_),
                             "D_VOLUME", volume_);
    }
#endif
}

void PreviewController::stop()
{
#if REAMIX_WITH_REAPER_IO
    if (preview_ != nullptr && s_cfPreviewStop != nullptr)
        s_cfPreviewStop (static_cast<preview_register_t*> (preview_));

    preview_ = nullptr;

    if (pcmSource_ != nullptr)
    {
        PCM_Source_Destroy (static_cast<PCM_source*> (pcmSource_));
        pcmSource_ = nullptr;
    }
#endif

    isPlayingFlag_ = false;
    lastPosition_  = -1.0;
    stallCount_    = 0;
    endSec_        = -1.0;
}

bool PreviewController::isPlaying()
{
#if REAMIX_WITH_REAPER_IO
    if (preview_ == nullptr || ! isPlayingFlag_)
        return false;

    const double pos      = getPositionSec();
    const double duration = getDurationSec();

    // End-of-audio reached (preview_helper.lua:169-171). 0.05 s cushion
    // matches Lua; CF_Preview_GetValue polls from the audio thread, so
    // the exposed position lags the hardware by a buffer-period.
    if (duration > 0.0 && pos >= duration - 0.05)
    {
        isPlayingFlag_ = false;
        return false;
    }

    // Range-limited playback (ADR-039 play-selection). Same 0.05 s cushion.
    if (endSec_ > 0.0 && pos >= endSec_ - 0.05)
    {
        isPlayingFlag_ = false;
        return false;
    }

    // Stall detection (preview_helper.lua:174-184): if the position has
    // not advanced for >10 consecutive polls, assume playback was stopped
    // externally (e.g. user hit REAPER's transport, or another plugin
    // called CF_Preview_Stop).
    if (lastPosition_ >= 0.0 && std::abs (pos - lastPosition_) < 0.001)
    {
        ++stallCount_;
        if (stallCount_ > 10)
        {
            isPlayingFlag_ = false;
            return false;
        }
    }
    else
    {
        stallCount_ = 0;
    }
    lastPosition_ = pos;

    return true;
#else
    return false;
#endif
}

double PreviewController::getPositionSec() const
{
#if REAMIX_WITH_REAPER_IO
    if (preview_ == nullptr || s_cfPreviewGetValue == nullptr)
        return 0.0;
    double out = 0.0;
    if (s_cfPreviewGetValue (static_cast<preview_register_t*> (preview_),
                             "D_POSITION", &out))
        return out;
    return 0.0;
#else
    return 0.0;
#endif
}

double PreviewController::getDurationSec() const
{
#if REAMIX_WITH_REAPER_IO
    if (pcmSource_ == nullptr)
        return 0.0;
    return GetMediaSourceLength (static_cast<PCM_source*> (pcmSource_), nullptr);
#else
    return 0.0;
#endif
}

} // namespace reamix::ui
