// reamix native — REAPER extension entry point (JUCE + WDL/SWELL).
// Phase 0 scaffold: command registration + dockable window only.

#define REAPERAPI_IMPLEMENT
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"

#include <juce_gui_basics/juce_gui_basics.h>
#include "ui/DockableWindow.h"
#include "ui/MainComponent.h"
#include "ui/PreviewController.h"

#include <memory>

// --- Globals ---

static std::unique_ptr<DockableWindow> g_window;
static int g_cmdToggle = 0;
static bool g_juceInitialised = false;

// --- JUCE message pump ---
// Pumps JUCE's internal message queue from REAPER's timer.
// Needed for juce::MessageManager::callAsync, juce::Timer, repaint, etc.
// Lightweight no-op when JUCE is not initialised.

static void juceMessagePump()
{
    if (g_juceInitialised)
        juce::MessageManager::getInstance()->runDispatchLoopUntil(2);
}

// --- Action handler ---

static bool onAction(int command, int)
{
    if (command != g_cmdToggle)
        return false;

    if (!g_juceInitialised)
    {
        juce::initialiseJuce_GUI();
        g_juceInitialised = true;
    }

    if (!g_window)
    {
        g_window = std::make_unique<DockableWindow>();
        g_window->create();

        // Wire dock callbacks (REABeat main.cpp:79-80 parity). Must run
        // after create() populates g_window->getContent().
        if (auto* content = g_window->getContent())
        {
            content->onToggleDock = []
            {
                if (g_window) g_window->toggleDock();
            };
            content->onIsDocked = []
            {
                return g_window && g_window->isDocked();
            };
        }
        return true;
    }
    g_window->toggleVisibility();

    return true;
}

// --- Toggle state callback ---

static int toggleActionState(int command)
{
    if (command == g_cmdToggle)
    {
        if (g_window && g_window->isVisible())
            return 1;
        return 0;
    }
    return -1;
}

// --- Sesja 64 BUG-8 — keyboard accelerator hook ---
// In docked mode the plugin window is a child panel of REAPER's main window,
// so OS-level keyboard focus stays on REAPER → JUCE KeyListener never fires
// for Space / Esc / Enter. SWS-style accelerator hook intercepts keys at the
// REAPER message-pump level and dispatches to MainComponent when the user has
// interacted with our window (focused or mouse-over fallback for docked).
// Pattern lifted from /Volumes/@Basic/Projekty/EditView/cpp/src/main.cpp:152.

static int translateAccelReamix(MSG* msg, accelerator_register_t* /*ctx*/)
{
    if (!g_window || !g_window->isVisible() || !g_window->getHwnd())
        return 0;

    HWND ours = g_window->getHwnd();

    // Focus detection: docked WS_CHILD windows on Windows don't reliably
    // report focus via GetFocus, so cascade through several methods then
    // fall back to mouse-over.
    HWND focus = GetFocus();
    bool isFocused = (focus == ours) || (IsChild(ours, focus) != 0);
    if (!isFocused && g_window->isDocked())
    {
        POINT pt;
        GetCursorPos(&pt);
        RECT wr;
        GetWindowRect(ours, &wr);
        isFocused = PtInRect(&wr, pt) != 0;
    }
    if (!isFocused) return 0;

    if (msg->message == WM_KEYDOWN)
    {
        const int vk = (int) msg->wParam;
        // Only intercept the global transport shortcuts; let everything else
        // pass through to REAPER (return -666 = SWS magic for "we have focus
        // but pass to REAPER main").
        const bool isOurKey = (vk == 0x20 /*SPACE*/)
                           || (vk == 0x1B /*ESCAPE*/)
                           || (vk == 0x0D /*RETURN*/);
        if (isOurKey)
        {
            if (auto* content = g_window->getContent())
            {
                const bool ctrl  = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
                const bool shift = (GetAsyncKeyState(VK_SHIFT)   & 0x8000) != 0;
                if (content->handleAccelKey(vk, ctrl, shift)) return 1; // consumed
            }
        }
    }

    return -666; // our window has focus, pass remaining keys to REAPER main
}

static accelerator_register_t g_accelReg = { translateAccelReamix, true, nullptr };

// --- Cleanup ---

static void onExit()
{
    g_window.reset();

    if (g_juceInitialised)
    {
        juce::shutdownJuce_GUI();
        g_juceInitialised = false;
    }
}

// --- Entry point ---

extern "C" {

REAPER_PLUGIN_DLL_EXPORT int ReaperPluginEntry(
    HINSTANCE hInstance, reaper_plugin_info_t* rec)
{
    // Cleanup path
    if (!rec)
    {
        onExit();
        return 0;
    }

    // Version check
    if (rec->caller_version != REAPER_PLUGIN_VERSION)
        return 0;

    // Load all REAPER API functions
    if (REAPERAPI_LoadAPI(rec->GetFunc) != 0)
        return 0;

#ifdef _WIN32
    // Pre-load onnxruntime.dll from our DLL's directory (UserPlugins)
    // before delay-load resolves it from System32, where Windows 10/11
    // ship onnxruntime.dll v1.17 (Windows ML). Our binary needs the
    // v1.24.4 sidecar we install next to the plugin. CMakeLists.txt:210
    // sets /DELAYLOAD:onnxruntime.dll; this LoadLibraryW pre-binds the
    // correct version to the process before any ONNX symbol resolves.
    // Pattern ported verbatim from REABeat src/main.cpp (commit 7f5b17c).
    {
        wchar_t selfPath[MAX_PATH] = {};
        HMODULE hSelf = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)ReaperPluginEntry, &hSelf);
        if (hSelf && GetModuleFileNameW(hSelf, selfPath, MAX_PATH))
        {
            for (int i = (int)wcslen(selfPath) - 1; i >= 0; --i)
            {
                if (selfPath[i] == L'\\' || selfPath[i] == L'/')
                {
                    selfPath[i + 1] = 0;
                    break;
                }
            }
            // Sesja 111 v1.0.2 — try subdir first (ReaPack install ships
            // ORT to UserPlugins\reamix\ to avoid conflict with REABeat
            // which claims ownership of UserPlugins\onnxruntime.dll).
            // Fallback to root for manual install convention.
            wchar_t ortPath[MAX_PATH] = {};
            wcscpy_s(ortPath, selfPath);
            wcscat_s(ortPath, L"reamix\\onnxruntime.dll");
            HMODULE hOrt = LoadLibraryW(ortPath);
            if (!hOrt)
            {
                wchar_t rootPath[MAX_PATH] = {};
                wcscpy_s(rootPath, selfPath);
                wcscat_s(rootPath, L"onnxruntime.dll");
                LoadLibraryW(rootPath);
            }
        }
    }
#endif

    // Capture rec->GetFunc for lazy SWS resolution (ADR-036 ⚑ P1 primary
    // preview path). Actual CF_Preview_* symbol lookup happens at first
    // Preview click, not here — REAPER loads extensions alphabetically and
    // SWS ("sws64") loads AFTER our plugin ("reamix"), so symbols are not
    // yet registered at this point. Session-48 DIAG-SWS-48 confirmed all
    // 5 CF_* pointers + CF_GetSWSVersion null at entry time with SWS
    // installed. See PreviewController.cpp::ensureSwsResolved comment.
    reamix::ui::PreviewController::resolveSwsSymbols((void*) rec->GetFunc);

    // Register timer for JUCE message pump (always running, lightweight when JUCE not init)
    rec->Register("timer", (void*)(void(*)())juceMessagePump);

    // Register toggle action
    g_cmdToggle = rec->Register("command_id", (void*)"reamix_ShowWindow");
    if (!g_cmdToggle)
        return 0;

    static gaccel_register_t accel = {{0, 0, 0}, "reamix.me: Show/Hide Window"};
    accel.accel.cmd = static_cast<unsigned short>(g_cmdToggle);
    rec->Register("gaccel", &accel);

    rec->Register("hookcommand", (void*)onAction);
    rec->Register("toggleaction", (void*)toggleActionState);
    rec->Register("atexit", (void*)onExit);

    // Sesja 64 BUG-8 — register accelerator hook so Space/Esc/Enter reach
    // the plugin even when docked (REAPER's transport otherwise eats them).
    // "<accelerator" prefix = priority before standard handling.
    rec->Register("<accelerator", &g_accelReg);

    return 1;
}

} // extern "C"
