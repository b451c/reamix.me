#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

// SupportPopover — sesja 108. Lightweight donation-links popover anchored
// below the HeaderBar heart icon. Mirrors SettingsPopover's modal +
// click-outside-to-dismiss pattern, with a simpler 3-row body listing the
// donation channels documented in .github/FUNDING.yml (Ko-fi / Buy Me a
// Coffee / PayPal). Each row click opens the URL in the user's default
// browser via juce::URL::launchInDefaultBrowser().
//
// Visual + interaction parity with SettingsPopover (Bg2 rounded body, Bg3
// hover, Bg4 pressed, Fg2 text, Fg3 secondary, drop shadow). Anchored under
// the heart icon (NOT the gear) so the trigger affordance lines up.

namespace reamix::ui
{

class SupportPopover : public juce::Component
{
public:
    SupportPopover();
    ~SupportPopover() override;

    void show();
    void hideMe();
    bool isOpen() const noexcept { return isVisible(); }

    // juce::Component
    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseUp   (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void mouseExit (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;
    void inputAttemptWhenModal() override;

private:
    enum class HitRow { None, GitHubSponsors, KoFi, BuyMeCoffee, PayPal };

    void layOutHitRegions();
    HitRow rowAt (juce::Point<int>) const;

    juce::Rectangle<int> bodyRect_;
    juce::Rectangle<int> rowGitHub_;
    juce::Rectangle<int> rowKoFi_;
    juce::Rectangle<int> rowBuyMe_;
    juce::Rectangle<int> rowPayPal_;

    HitRow hover_   = HitRow::None;
    HitRow pressed_ = HitRow::None;

    // Body sized for "Support reamix.me" header + 4 rows + padding.
    static constexpr int kBodyWidth   = 260;
    static constexpr int kPadding     = 12;
    static constexpr int kAnchorRight = 32;  // right edge of heart icon = right edge of popover
    static constexpr int kAnchorTop   = 38;
    static constexpr int kTitleH      = 9;
    static constexpr int kTitleMargB  = 10;
    static constexpr int kRowH        = 44;  // taller — fits primary + subtitle without overlap
    static constexpr int kRowGap      = 4;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SupportPopover)
};

} // namespace reamix::ui
