// test_boundary_family — sesja 130 (ADR-116 step 3) self-validation.
//
// The boundary cut family (src/remix/BoundaryFamily.h): a cut that leaves
// at a phrase END and lands on a phrase START, admitted into the Duration
// pool beside the continuation family and scored in the substitution view
// (kV2BoundaryQualityWeights). No Python ground truth (C++-canonical per
// ADR-065); the perceptual evidence is the sesja-128 seam probe (4 of 5
// boundary placements usable) and the sesja-130 corpus tables + round.
//
// Checks:
//   1. phrase starts on a 4/4 grid with a section map: every section start
//      and every 8th bar inside a section; not the 8th bar from the song
//      start when a section began elsewhere; without sections = every 8th
//      bar from the first downbeat.
//   2. pairOk = leave at a phrase end AND land on a phrase start; beat 0
//      never lands.
//   3. computeTransitionCosts (v2): with the family on, every extra
//      candidate vs the family-off run is a boundary pair with family 1,
//      W == 1 - q, no alignment lag; every family-off candidate survives
//      unchanged (the continuation family is untouched).
//   4. SpliceAcceptance::maskedAtTier: continuation by waveform, boundary by
//      edge distance, unknown distance never masked, tier 0 never masks.

#include "remix/BoundaryFamily.h"
#include "remix/PhraseAlign.h"
#include "remix/SpliceAcceptance.h"
#include "remix/TransitionCost.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

namespace
{
#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) { std::printf ("FAIL: %s — %s\n", #expr, msg); return false; }\
} while (0)

using namespace reamix::remix;

constexpr int    kBeats   = 96;    // 24 bars of 4/4
constexpr int    kTS      = 4;
constexpr double kPeriod  = 0.5;   // 120 BPM
constexpr int    kFeat    = 59;

std::vector<double> beatTimes()
{
    std::vector<double> t ((std::size_t) kBeats);
    for (int b = 0; b < kBeats; ++b) t[(std::size_t) b] = b * kPeriod;
    return t;
}

std::vector<double> downbeatTimes()
{
    std::vector<double> d;
    for (int b = 0; b < kBeats; b += kTS) d.push_back (b * kPeriod);
    return d;
}

std::set<int> downbeatSet()
{
    std::set<int> s;
    for (int b = 0; b < kBeats; b += kTS) s.insert (b);
    return s;
}

// intro = bars 0-5, chorus = bars 6-21, outro = bars 22-23.
std::vector<reamix::analysis::Segment> sections()
{
    auto seg = [] (int barFrom, int barTo, const char* label)
    {
        reamix::analysis::Segment s;
        s.start = barFrom * kTS * kPeriod;
        s.end   = barTo   * kTS * kPeriod;
        s.label = label;
        return s;
    };
    return { seg (0, 6, "intro"), seg (6, 22, "chorus"), seg (22, 24, "outro") };
}

bool testPhraseStarts()
{
    const auto bt = beatTimes();
    const auto db = downbeatSet();
    const auto segs = sections();
    const BoundaryFamily f = BoundaryFamily::build (bt.data(), kBeats, db, segs.data(), (int) segs.size());
    CHECK (f.active, "family active with a section map");
    // Section starts: bar 0 (beat 0), bar 6 (beat 24), bar 22 (beat 88); the
    // 8th bar of the chorus: bar 14 (beat 56). Bars 8 / 16 (beats 32 / 64)
    // are bars 2 / 10 of the chorus - not phrase starts.
    CHECK (f.isPhraseStart (0),  "bar 0 = intro start");
    CHECK (f.isPhraseStart (24), "bar 6 = chorus start");
    CHECK (f.isPhraseStart (56), "bar 14 = chorus bar 8");
    CHECK (f.isPhraseStart (88), "bar 22 = outro start");
    CHECK (! f.isPhraseStart (32), "bar 8 is chorus bar 2");
    CHECK (! f.isPhraseStart (64), "bar 16 is chorus bar 10");
    CHECK (! f.isPhraseStart (25), "a non-downbeat never opens a phrase");
    CHECK (f.n_starts == 4, "four phrase starts");

    // Pairs: the cut after beat 23 leaves at the intro's end (beat 24 opens
    // the chorus); it may land on 56 or 88, not on 32 (chorus bar 2) and
    // not on 0 (no original context).
    CHECK (f.leavesAtPhraseEnd (23), "beat 23 = last beat before the chorus");
    CHECK (! f.leavesAtPhraseEnd (27), "beat 27 is inside the phrase");
    CHECK (f.pairOk (23, 56), "intro end -> chorus bar 8");
    CHECK (f.pairOk (23, 88), "intro end -> outro");
    CHECK (f.pairOk (55, 88), "chorus bar 7 end -> outro");
    CHECK (! f.pairOk (23, 32), "landing inside a phrase");
    CHECK (! f.pairOk (23, 0),  "beat 0 never lands");
    CHECK (! f.pairOk (27, 56), "leaving mid-phrase");

    // Without a section map: every 8th bar from the first downbeat.
    const BoundaryFamily g = BoundaryFamily::build (bt.data(), kBeats, db, nullptr, 0);
    CHECK (g.active, "family active without sections");
    CHECK (g.isPhraseStart (0) && g.isPhraseStart (32) && g.isPhraseStart (64), "bars 0 / 8 / 16");
    CHECK (! g.isPhraseStart (24) && ! g.isPhraseStart (56), "bars 6 / 14 are not phrase starts on the bare grid");
    CHECK (g.n_starts == 3, "three phrase starts");

    // Degenerate inputs.
    const BoundaryFamily h = BoundaryFamily::build (nullptr, 0, db, nullptr, 0);
    CHECK (! h.active && ! h.pairOk (0, 8), "empty input = inactive");
    return true;
}

// Synthetic features: a per-section base pattern plus a small per-beat
// wobble, so chroma / MFCC similarity is high inside a section and moderate
// across sections (below the 0.45 chroma prefilter for the pairs we assert).
std::vector<float> features (const std::vector<reamix::analysis::Segment>& segs, const std::vector<double>& bt)
{
    std::vector<float> f ((std::size_t) kBeats * kFeat, 0.0f);
    for (int b = 0; b < kBeats; ++b)
    {
        int sec = 0;
        for (int s = 0; s < (int) segs.size(); ++s)
            if (bt[(std::size_t) b] >= segs[(std::size_t) s].start && bt[(std::size_t) b] < segs[(std::size_t) s].end) sec = s;
        for (int k = 0; k < kFeat; ++k)
        {
            const double base   = 0.6 + 0.4 * std::cos (0.37 * k + 0.9 * sec);
            const double wobble = 0.05 * std::sin (0.11 * b + 0.23 * k);
            f[(std::size_t) b * kFeat + (std::size_t) k] = (float) (base + wobble);
        }
    }
    return f;
}

TransitionCostResult runPool (bool disableFamily, const std::vector<float>& feat,
                              const std::vector<double>& bt, const std::vector<double>& db,
                              const std::vector<reamix::analysis::Segment>& segs)
{
    TransitionCostInputs in {};
    in.features    = feat.data();
    in.n_beats     = kBeats;
    in.n_features  = kFeat;
    in.beat_times  = bt.data();
    in.segments    = segs.data();
    in.n_segments  = (int) segs.size();
    in.downbeats   = db.data();
    in.n_downbeats = (int) db.size();
    in.time_signature = kTS;
    in.v2_scoring  = true;
    in.disable_repetition_prior = true;   // synthetic features carry no recurrence
    in.disable_phrase_align     = true;   // the test is about the family, not the gate
    in.max_candidates_per_beat  = 1;      // a lean continuation pool leaves phrase pairs for the family
    in.disable_boundary_family  = disableFamily;
    return computeTransitionCosts (in);
}

// DEV-120 (sesja 132): a raw model boundary one beat BEFORE the downbeat
// (chorus start at 5.75 bars instead of bar 6) must still make the real
// chorus start (bar 6, beat 24) a phrase start for the boundary family
// (snapped to the nearest downbeat), while the continuation gate's offsets
// (snap off) keep the sesja-126 rule: the first beat at/after the raw start
// (beat 23, bar 5) is the section's start bar, so beat 24 is bar 1 of it.
bool testSnappedBoundary()
{
    const auto bt = beatTimes();
    const auto db = downbeatSet();
    auto segs = sections();
    segs[1].start = 23 * kPeriod;   // raw chorus start = beat 23 (one beat early)
    segs[0].end   = segs[1].start;
    const BoundaryFamily f = BoundaryFamily::build (bt.data(), kBeats, db, segs.data(), (int) segs.size());
    CHECK (f.isPhraseStart (24), "chorus start snapped to bar 6 = phrase start");
    CHECK (f.isPhraseStart (56), "chorus bar 8 counted from the snapped start");
    CHECK (! f.isPhraseStart (28), "bar 7 is not a phrase start after snapping");
    const auto gate = PhraseAlign::barOffsets (bt.data(), kBeats, db, segs.data(), (int) segs.size(), false);
    CHECK (gate[24] == 1, "continuation gate keeps the first-beat-at/after rule (beat 24 = bar 1 of the raw section)");
    const auto snapped = PhraseAlign::barOffsets (bt.data(), kBeats, db, segs.data(), (int) segs.size(), true);
    CHECK (snapped[24] == 0 && snapped[23] == 5, "snapped offsets: beat 24 opens the chorus, beat 23 still belongs to the intro");
    return true;
}

bool testPoolAdmission()
{
    const auto bt   = beatTimes();
    const auto db   = downbeatTimes();
    const auto segs = sections();
    const auto feat = features (segs, bt);
    const BoundaryFamily f = BoundaryFamily::build (bt.data(), kBeats, downbeatSet(), segs.data(), (int) segs.size());

    const TransitionCostResult off = runPool (true,  feat, bt, db, segs);
    const TransitionCostResult on  = runPool (false, feat, bt, db, segs);
    CHECK (! off.boundary_family_active && off.boundary_family_pairs == 0, "flag off = no family");
    CHECK (on.boundary_family_active, "family active on the synthetic grid");
    CHECK (on.boundary_family_starts == 4, "four phrase starts reported");
    CHECK (on.boundary_family_pairs > 0, "the family admits pairs the lean continuation pool did not");
    CHECK (on.candidates.size() == off.candidates.size() + (std::size_t) on.boundary_family_pairs,
           "extra candidates == family pairs");

    int extras = 0;
    for (const auto& kv : on.candidates)
    {
        const int i = kv.first.first, j = kv.first.second;
        const TransitionCandidate& c = kv.second;
        const auto it = off.candidates.find (kv.first);
        if (it == off.candidates.end())
        {
            ++extras;
            CHECK (c.family == TransitionCandidate::kFamilyBoundary, "an extra candidate is a boundary cut");
            CHECK (f.pairOk (i, j), "an extra candidate leaves at a phrase end and lands on a phrase start");
            CHECK (std::abs (j - i) >= 4 * kTS, "micro-skip block respected");
            CHECK (c.alignment_lag_samples == 0, "no alignment lag on a boundary cut");
            CHECK (c.quality_score >= QUALITY_HARD_FLOOR, "quality floor applied");
            CHECK (std::abs (on.W[(std::size_t) i * kBeats + j] - (1.0 - c.quality_score)) < 1e-12, "W == 1 - q");
            CHECK (c.chroma_distance <= CHROMA_PREFILTER_THRESHOLD, "chroma prefilter applied");
        }
        else
        {
            CHECK (c.family == TransitionCandidate::kFamilyContinuation, "a continuation candidate keeps family 0");
            CHECK (c.quality_score == it->second.quality_score, "continuation score untouched by the family");
            CHECK (on.W[(std::size_t) i * kBeats + j] == off.W[(std::size_t) i * kBeats + j], "continuation W untouched");
        }
    }
    CHECK (extras == on.boundary_family_pairs, "diagnostic count == extras");
    // The intro-end -> outro pair is the radio-edit cut this family exists
    // for; with a lean continuation pool it must come from the family.
    const auto ro = on.candidates.find ({ 23, 88 });
    CHECK (ro != on.candidates.end(), "intro end -> outro admitted");
    return true;
}

bool testAcceptanceMask()
{
    TransitionCandidate cont {};
    cont.family = TransitionCandidate::kFamilyContinuation;
    cont.waveform_similarity = 0.75;
    cont.edge_distance = 1.5;   // ignored for a continuation cut
    CHECK (maskedAtTier (cont, 0.80), "continuation 0.75 masked at 0.80");
    CHECK (! maskedAtTier (cont, 0.70), "continuation 0.75 passes 0.70");
    CHECK (! maskedAtTier (cont, 0.0),  "tier 0 never masks");

    TransitionCandidate bnd {};
    bnd.family = TransitionCandidate::kFamilyBoundary;
    bnd.waveform_similarity = 0.1;   // ignored for a boundary cut
    bnd.edge_distance = 0.70;
    CHECK (maskedAtTier (bnd, 0.80),   "boundary 0.70 masked at the 0.60 cap");
    CHECK (! maskedAtTier (bnd, 0.70), "boundary 0.70 passes the 0.80 cap");
    CHECK (! maskedAtTier (bnd, 0.60), "boundary 0.70 passes the 1.00 cap");
    bnd.edge_distance = 1.2;
    CHECK (maskedAtTier (bnd, 0.60),   "boundary 1.2 masked at the 1.00 cap");
    CHECK (! maskedAtTier (bnd, 0.0),  "unfiltered pool keeps it");
    bnd.edge_distance = -1.0;
    CHECK (! maskedAtTier (bnd, 0.80), "no edge scale on the track: never masked");
    CHECK (boundaryEdgeCap (0.80) == 0.60 && boundaryEdgeCap (0.70) == 0.80 && boundaryEdgeCap (0.60) == 1.00,
           "cap table");
    return true;
}

} // namespace

int main()
{
    int failed = 0;
    struct { const char* name; bool (*fn)(); } tests[] = {
        { "phrase starts + pairs", testPhraseStarts },
        { "snapped raw boundary (DEV-120)", testSnappedBoundary },
        { "pool admission",        testPoolAdmission },
        { "acceptance mask",       testAcceptanceMask },
    };
    for (const auto& t : tests)
    {
        const bool ok = t.fn();
        std::printf ("%s: %s\n", ok ? "PASS" : "FAIL", t.name);
        if (! ok) ++failed;
    }
    return failed == 0 ? 0 : 1;
}
