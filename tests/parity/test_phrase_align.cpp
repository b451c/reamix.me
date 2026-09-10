// test_phrase_align — sesja 126 (DEV-116) self-validation of the
// phrase-position alignment gate (src/remix/PhraseAlign.h, header-only).
//
// Asserts:
//   1. The rule on the 11 user-rated cuts of sesja 126 (bar offsets inside
//      the section, from the audit table): every "bad" cut is rejected,
//      every "ok" cut is kept (mod 8, landing on a phrase start allowed).
//   2. barOffsets: uniform 4/4 grid with sections of 8 bars -> offsets
//      0..7 restart at every section; beats before the first downbeat = -1;
//      no sections -> bars from the first downbeat.
//   3. build: all-permissive base -> active at 8 bars, same-offset pair
//      allowed, one-bar-off pair rejected, phrase-start landing allowed;
//      a base that only leaves one-bar-off pairs -> relax to 4 fails too ->
//      inactive (allowed() == true, never starve the DP).

#include "remix/PhraseAlign.h"

#include <cstdio>
#include <set>
#include <vector>

using reamix::remix::PhraseAlign;

namespace {

bool test_rated_cuts()
{
    struct Cut { int off_out, off_in; bool ok; const char* name; };
    const Cut cuts[] = {
        {6, 2, false, "drake 60->117 verse+10.7s -> verse+3.6s"},
        {7, 6, false, "drake 28->173 chorus+12.5s -> chorus+10.7s"},
        {4, 5, false, "drake 104->25 chorus+7.1s -> chorus+8.9s"},
        {4, 4, true,  "drake 168->233 chorus+7.1s -> chorus+7.1s"},
        {6, 6, true,  "drake 124->61 verse+10.7s -> verse+10.7s"},
        {3, 8, true,  "drake 86->159 chorus+5.3s -> phrase start"},
        {5, 1, false, "dylan 121->210"},
        {10, 18, true, "dylan 67->108"},
        {2, 10, true, "dylan 75->196"},
        {2, 16, true, "dylan 51->104 (phrase start)"},
        {18, 10, true, "dylan 107->36"},
    };
    bool ok = true;
    for (const auto& c : cuts) {
        const bool got = PhraseAlign::positionOk(c.off_out, c.off_in, PhraseAlign::kPhraseBars);
        if (got != c.ok) {
            std::fprintf(stderr, "[FAIL] %s: gate %s, user %s\n", c.name,
                         got ? "keeps" : "rejects", c.ok ? "ok" : "bad");
            ok = false;
        }
    }
    if (ok) std::fprintf(stderr, "[PASS] 11 rated cuts: 4 bad rejected, 7 ok kept\n");
    return ok;
}

struct Grid
{
    std::vector<double>                    t;
    std::set<int>                          db, pre_db;
    std::vector<reamix::analysis::Segment> segs;
    Grid(int n_bars, int section_bars, int lead_beats = 0)
    {
        // 4 beats per bar, 0.5 s per beat, `lead_beats` beats before the first downbeat.
        for (int b = 0; b < lead_beats + n_bars * 4; ++b) {
            t.push_back(0.5 * b);
            const int k = b - lead_beats;
            if (k >= 0 && k % 4 == 0) db.insert(b);
            if (k >= 0 && (k + 1) % 4 == 0) pre_db.insert(b);
        }
        for (int s = 0; s * section_bars < n_bars; ++s) {
            reamix::analysis::Segment seg;
            seg.start = 0.5 * (lead_beats + s * section_bars * 4);
            seg.end   = 0.5 * (lead_beats + std::min(n_bars, (s + 1) * section_bars) * 4);
            seg.label = (s % 2 == 0) ? "verse" : "chorus";
            segs.push_back(seg);
        }
    }
};

bool test_bar_offsets()
{
    const Grid g(24, 8, 2);   // 24 bars, 8-bar sections, 2 lead beats
    const auto off = PhraseAlign::barOffsets(g.t.data(), (int) g.t.size(), g.db, g.segs.data(), (int) g.segs.size());
    bool ok = off[0] == -1 && off[1] == -1;
    for (int bar = 0; bar < 24 && ok; ++bar)
        for (int k = 0; k < 4; ++k)
            if (off[(std::size_t) (2 + bar * 4 + k)] != bar % 8) ok = false;
    const auto flat = PhraseAlign::barOffsets(g.t.data(), (int) g.t.size(), g.db, nullptr, 0);
    if (flat[(std::size_t) (2 + 13 * 4)] != 13) ok = false;
    if (!ok) { std::fprintf(stderr, "[FAIL] barOffsets\n"); return false; }
    std::fprintf(stderr, "[PASS] barOffsets: section-relative 0..7, lead beats -1, flat = bar index\n");
    return true;
}

bool test_loop_rule()
{
    // DEV-117 (d) (sesja 127): 32 bars, 8-bar sections, 4/4 at 0.5 s.
    const Grid g(32, 8);
    const int n = (int) g.t.size();
    auto all = [] (int, int) { return true; };
    const PhraseAlign p = PhraseAlign::build(g.t.data(), n, g.db, g.pre_db, g.segs.data(), (int) g.segs.size(), all);
    bool ok = p.active && p.phrase_bars == 8;
    // 2-bar loop: after bar 11 (beat 47) back to bar 10 (beat 40) = offset 2 -> allowed;
    // after bar 12 (beat 51) back to bar 11 (beat 44) = offset 3 -> rejected;
    // 4-bar loop: after bar 15 (beat 63) back to bar 12 (beat 48) = offset 4 -> allowed;
    // after bar 13 (beat 55) back to bar 10 (beat 40) = offset 2 -> rejected;
    // 1-bar loop anywhere: after bar 11 (beat 47) back to bar 11 (beat 44) -> allowed;
    // 3-bar loop falls back to the mod-8 rule: after bar 12 (beat 51) back to bar 10 (beat 40): 5 -> 2 rejected;
    // across sections (bar 8 -> bar 6, 2 bars): mod-8 rule, offsets 1 -> 6 rejected;
    // forward skip unchanged: after bar 3 (beat 15) -> bar 11 (beat 44) rejected, -> bar 12 (beat 48) allowed.
    ok = ok && p.loopAllowed(47, 40) && !p.loopAllowed(51, 44)
            && p.loopAllowed(63, 48) && !p.loopAllowed(55, 40)
            && p.loopAllowed(47, 44) && !p.loopAllowed(51, 40)
            && !p.loopAllowed(35, 24)
            && p.loopAllowed(43, 12)   // 8-bar loop across the section boundary (bars 3..10): whole phrase -> allowed
            && !p.loopAllowed(15, 44) && p.loopAllowed(15, 48);
    if (!ok) {
        std::fprintf(stderr, "[FAIL] loop rule: 47->40 %d 51->44 %d 63->48 %d 55->40 %d 47->44 %d 51->40 %d 35->24 %d 15->44 %d 15->48 %d\n",
                     (int) p.loopAllowed(47, 40), (int) p.loopAllowed(51, 44), (int) p.loopAllowed(63, 48),
                     (int) p.loopAllowed(55, 40), (int) p.loopAllowed(47, 44), (int) p.loopAllowed(51, 40),
                     (int) p.loopAllowed(35, 24), (int) p.loopAllowed(15, 44), (int) p.loopAllowed(15, 48));
        return false;
    }
    std::fprintf(stderr, "[PASS] loop rule: 2-bar loop at even offsets, 4-bar at 0 / 4, 1-bar anywhere, other spans mod 8\n");
    return true;
}

bool test_build()
{
    const Grid g(32, 8);
    const int n = (int) g.t.size();
    auto all = [] (int, int) { return true; };
    const PhraseAlign p = PhraseAlign::build(g.t.data(), n, g.db, g.pre_db, g.segs.data(), (int) g.segs.size(), all);
    bool ok = p.active && p.phrase_bars == 8;
    // Cut after bar 3 (beat 15) -> bar 11 (beat 44): offsets 4 -> 3 => rejected.
    // Cut after bar 3 (beat 15) -> bar 12 (beat 48): offsets 4 -> 4 => allowed.
    // Cut after bar 3 (beat 15) -> bar 16 (beat 64): offsets 4 -> 0 => allowed (phrase start).
    ok = ok && !p.allowed(15, 44) && p.allowed(15, 48) && p.allowed(15, 64);
    if (!ok) { std::fprintf(stderr, "[FAIL] build/allowed (active %d bars %d)\n", (int) p.active, p.phrase_bars); return false; }
    // Base that only leaves one-bar-off targets: 8 fails, 4 fails -> inactive.
    const auto off = p.bar_offset;
    auto one_off = [&] (int i, int j) { return ((off[(std::size_t) (i + 1)] + 1) % 8) == (off[(std::size_t) j] % 8) && off[(std::size_t) j] % 4 != 0; };
    const PhraseAlign q = PhraseAlign::build(g.t.data(), n, g.db, g.pre_db, g.segs.data(), (int) g.segs.size(), one_off);
    if (q.active || !q.allowed(15, 44)) { std::fprintf(stderr, "[FAIL] starvation should switch the gate off\n"); return false; }
    std::fprintf(stderr, "[PASS] build: 8 bars active, one-bar-off rejected, same offset + phrase start allowed, starvation -> off\n");
    return true;
}

} // namespace

int main()
{
    bool ok = true;
    ok = test_rated_cuts()  && ok;
    ok = test_bar_offsets() && ok;
    ok = test_build()       && ok;
    ok = test_loop_rule()   && ok;
    std::fprintf(stderr, ok ? "== test_phrase_align PASS ==\n" : "== test_phrase_align FAIL ==\n");
    return ok ? 0 : 1;
}
