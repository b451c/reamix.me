#include "ContextMenu.h"

#include <functional>

namespace reamix::ui
{

void ContextMenu::showSpliceMenuAsync (juce::Component& source,
                                       juce::Point<int> screenPos,
                                       std::function<void (SpliceResult)> onResult)
{
    juce::PopupMenu menu;
    menu.setLookAndFeel (&source.getLookAndFeel());

    // "Try different splice" with R shortcut chip per design-system.jsx:289.
    // Backed by `CleanOptimizer::remix_variation` (sesja 58, ADR-048).
    {
        juce::PopupMenu::Item item;
        item.itemID = static_cast<int> (SpliceResult::TryDifferent);
        item.text   = "Try different splice";
        item.shortcutKeyDescription = "R";
        menu.addItem (item);
    }
    menu.addItem (static_cast<int> (SpliceResult::ResetToBest), "Reset to best");
    menu.addSeparator();
    // Sesja 99 — DEV-048 label clarification per sesja 62 NOTE-3 user
    // pushback: "Block this splice" sugerowało freeze-in-place semantykę
    // zamiast exclude-from-future-variations. "Avoid this transition"
    // matches faktyczne zachowanie (per-source `blockedBySource_` set
    // persistent across slider changes — algorithm omits this junction
    // from future remix candidates).
    menu.addItem (static_cast<int> (SpliceResult::Block),       "Avoid this transition");
    menu.addItem (static_cast<int> (SpliceResult::Seek),        "Seek to marker");

    auto options = juce::PopupMenu::Options()
                       .withTargetComponent (&source)
                       .withTargetScreenArea ({ screenPos.x, screenPos.y, 1, 1 });

    menu.showMenuAsync (options,
                        [cb = std::move (onResult)] (int result)
                        {
                            if (cb) cb (static_cast<SpliceResult> (result));
                        });
}

} // namespace reamix::ui
