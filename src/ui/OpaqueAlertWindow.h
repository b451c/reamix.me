#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace reamix {

// `juce::AlertWindow::paint()` delegates to `LookAndFeel::drawAlertBox()`
// which paints a rounded rectangle inside local bounds. Pixels OUTSIDE the
// rounded rect (corners, any margin) are never written. macOS/Windows native
// peer windows initialize the backing buffer (default cleared), so the
// uninitialised pixels show as transparent or solid background. X11/Cairo
// on Linux does NOT clear a fresh peer window's backing buffer — the first
// frame leaks whatever bytes were in memory, manifesting as static-noise
// artefacts visible behind the rounded corners (DEV-085, sesja 113).
//
// Solution: fill the full bounds with the AlertWindow background colour
// before the base paint runs. Harmless on macOS/Windows (overpaints with
// the same colour the rounded rect uses anyway).
class OpaqueAlertWindow : public juce::AlertWindow
{
public:
    using juce::AlertWindow::AlertWindow;

    void paint (juce::Graphics& g) override
    {
        g.fillAll (findColour (juce::AlertWindow::backgroundColourId));
        juce::AlertWindow::paint (g);
    }
};

} // namespace reamix
