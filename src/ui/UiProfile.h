#pragma once

#include <juce_core/juce_core.h>

#include <cstdlib>

// UiProfile — DEV-102 (sesja 123) message-thread profile. Off unless the
// plugin runs with REAMIX_UI_PROFILE=<file>; then every mark appends one
// line "<ms since first mark>\t<label>\t<duration ms>" to that file. Used to
// find what keeps the plugin busy after analysis completes (the VM drops tab
// clicks for 1.7-3.9 s). Dev instrumentation, no product behaviour.

namespace reamix::ui
{

struct UiProfile
{
    static juce::File& file()
    {
        static juce::File f = []
        {
            const char* p = std::getenv ("REAMIX_UI_PROFILE");
            return (p != nullptr && p[0] != '\0') ? juce::File (juce::String::fromUTF8 (p)) : juce::File();
        }();
        return f;
    }

    static bool enabled() { return file() != juce::File(); }

    static double nowMs() { return juce::Time::getMillisecondCounterHiRes(); }

    static void mark (const char* label, double durationMs = -1.0)
    {
        if (! enabled()) return;
        static const double t0 = nowMs();
        juce::String line = juce::String (nowMs() - t0, 1) + "\t" + label;
        if (durationMs >= 0.0) line += "\t" + juce::String (durationMs, 1);
        file().appendText (line + "\n");
    }

    // Scope timer: marks the label with the elapsed time at scope exit when
    // it exceeds `minMs` (0 = always).
    struct Scope
    {
        const char* label;
        double      minMs;
        double      t0;
        Scope (const char* l, double minMilliseconds = 0.0)
            : label (l), minMs (minMilliseconds), t0 (nowMs()) {}
        ~Scope()
        {
            const double d = nowMs() - t0;
            if (d >= minMs) mark (label, d);
        }
    };
};

} // namespace reamix::ui
