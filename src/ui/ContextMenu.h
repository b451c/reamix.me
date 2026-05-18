#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

// ContextMenu — phase-6 step 5 skeleton (ADR-036 D3 row 15).
//
// Splice-marker right-click menu per design-system.jsx:287-294 +
// plugin.css:582-601:
//
//   Try different splice                 R
//   Reset to best
//   ──────────────────────────────
//   Block this splice
//   Seek to marker
//
// Step-5 ships a static async helper that uses juce::PopupMenu under
// LookAndFeelReamix styling (drawPopupMenuBackground + drawPopupMenuItem
// already override .rx-menu). Step 7 SpliceMarker right-click calls
// showSpliceMenuAsync() with the clicked marker's screen position.

namespace reamix::ui
{

class ContextMenu
{
public:
    enum class SpliceResult : int
    {
        None          = 0,
        TryDifferent  = 1,
        ResetToBest   = 2,
        Block         = 3,
        Seek          = 4
    };

    static void showSpliceMenuAsync (juce::Component& source,
                                     juce::Point<int> screenPos,
                                     std::function<void (SpliceResult)> onResult);
};

} // namespace reamix::ui
