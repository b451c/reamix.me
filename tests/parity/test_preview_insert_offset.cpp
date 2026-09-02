// test_preview_insert_offset — sesja 122 (ADR-115 E7, DEV-087) self-validation.
//
// The edit plan (what REAPER Insert places) overlaps two clips by the full
// transition overlap; for an anchor-accepted splice that overlap is the
// onset-anchor window (>= 240 ms, up to ~2 beats). The preview WAV used to
// cap the blend at the widest crossfade band (200 ms), so the incoming clip
// started `overlap - 200 ms` late, the preview was longer than the insert
// and the two onsets the anchor aligned never coincided.
//
// Synthetic track: stereo 10 s @ 22050 Hz, silence except two unit clicks —
// one at the outgoing anchor (sample 62843, 2.85 s), one at the incoming
// anchor (sample 128993, 5.85 s). Path: beats 0..5 then 12..19 (one
// transition 5 -> 12) with the cached anchor geometry in the metadata, all
// in whole samples: overlap 13230 (0.60 s) = preRoll 7718 + postRoll 5512,
// outgoing cut = anchorOut + postRoll (3.10 s), incoming start = anchorIn -
// preRoll (5.50 s). By construction both clicks land on the same output
// sample when the plan is honoured.
//
// Invariants:
//   1. production: rendered length == plan duration (7.0 s); the splice time
//      == the incoming clip's timeline start (2.5 s); exactly one click peak,
//      at 2.85 s, with gain cos + sin > 1 (both clicks summed);
//   2. legacy cap: rendered length == plan + (0.6 - 0.2) s and the splice
//      time is 0.4 s late; two separate click peaks (the old defect);
//   3. an overlap within the band cap (0.10 s) renders identically under
//      both settings (the multi-band blend is untouched).
//
// Per ADR-065: hand-computed invariants, no Python ground truth.
#include "render/Renderer.h"
#include "remix/Path.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

constexpr int    kSr        = 22050;
constexpr double kDur       = 10.0;
constexpr double kBeatSec   = 0.5;     // 20 beats
constexpr int    kNBeats    = 20;
constexpr std::size_t kAnchorOutSample = 62843;    // 2.85 s
constexpr std::size_t kAnchorInSample  = 128993;   // 5.85 s

struct Rendered
{
    reamix::render::EditPlan plan;
    std::vector<float>       audio;
    std::size_t              nCh = 0, nSamp = 0;
    std::vector<double>      tt;
};

Rendered renderCase(std::size_t overlapSamples, bool legacyCap)
{
    const std::size_t n = static_cast<std::size_t>(kDur * kSr);
    std::vector<float> audio(2 * n, 0.0f);
    for (std::size_t c = 0; c < 2; ++c) {
        audio[c * n + kAnchorOutSample] = 1.0f;
        audio[c * n + kAnchorInSample]  = 1.0f;
    }
    std::vector<double> beats(kNBeats);
    for (int i = 0; i < kNBeats; ++i) beats[static_cast<std::size_t>(i)] = i * kBeatSec;

    reamix::render::RendererConfig cfg;
    cfg.legacyPreviewOverlapCap = legacyCap;
    reamix::render::Renderer renderer("synthetic.wav", audio.data(), 2, n, kSr,
                                      beats.data(), beats.size(), -1.0, cfg);

    reamix::remix::RemixPath path;
    for (int b = 0; b <= 5; ++b)  path.beat_indices.push_back(b);
    for (int b = 12; b < 20; ++b) path.beat_indices.push_back(b);
    path.transitions.push_back({5, 12});
    // Cached anchor geometry (resolveTransitionEdit's "anchor_overlap_samples"
    // branch): preRoll : postRoll = 7 : 5 so both anchors meet on the output.
    const std::size_t pre  = (overlapSamples * 7 + 6) / 12;
    const std::size_t post = overlapSamples - pre;
    std::map<std::string, double> meta;
    meta["anchor_overlap_samples"]     = static_cast<double>(overlapSamples);
    meta["render_outgoing_cut_sample"] = static_cast<double>(kAnchorOutSample + post);
    meta["render_incoming_cut_sample"] = static_cast<double>(kAnchorInSample  - pre);
    path.transition_metadata[{5, 12}] = meta;

    Rendered r;
    r.plan = renderer.buildEditPlan(path);
    renderer.renderEditPlan(r.plan, r.audio, r.nCh, r.nSamp, r.tt);
    return r;
}

std::vector<std::size_t> peaks(const Rendered& r, float thr)
{
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < r.nSamp; ++i)
        if (std::fabs(r.audio[i]) > thr) out.push_back(i);   // channel 0
    return out;
}

} // namespace

int main()
{
    std::printf("test_preview_insert_offset (DEV-087 / ADR-115 E7)\n");

    // ── 1. production: preview == plan ──────────────────────────────
    {
        const Rendered r = renderCase(13230, /*legacyCap*/ false);
        CHECK(r.plan.clips.size() == 2, "two clips");
        const double planSamples = r.plan.duration * kSr;
        std::printf("  full overlay: plan %.4f s, rendered %.4f s, splice %.4f s\n",
                    r.plan.duration, static_cast<double>(r.nSamp) / kSr,
                    r.tt.empty() ? -1.0 : r.tt[0]);
        CHECK(std::fabs(static_cast<double>(r.nSamp) - planSamples) <= 1.0,
              "rendered length == plan duration");
        CHECK(r.tt.size() == 1, "one splice time");
        CHECK(!r.tt.empty()
              && std::fabs(r.tt[0] - r.plan.clips[1].timelineStartSec) <= 1.0 / kSr,
              "splice time == incoming clip timeline start");
        CHECK(std::fabs(r.plan.clips[1].timelineStartSec - 2.50) < 1e-9,
              "incoming clip starts at 2.50 s on the timeline");
        const auto p = peaks(r, 0.1f);
        CHECK(p.size() == 1, "exactly one click in the output (both anchors on one sample)");
        if (p.size() == 1) {
            CHECK(p[0] == kAnchorOutSample, "the click sits at the outgoing anchor sample");
            CHECK(r.audio[p[0]] > 1.0f, "gain cos + sin > 1: the two clicks are summed");
        }
    }

    // ── 2. legacy cap reproduces the old offset ─────────────────────
    {
        const Rendered r = renderCase(13230, /*legacyCap*/ true);
        const double expected = (r.plan.duration + 0.60 - 0.20) * kSr;
        std::printf("  legacy cap:   plan %.4f s, rendered %.4f s, splice %.4f s\n",
                    r.plan.duration, static_cast<double>(r.nSamp) / kSr,
                    r.tt.empty() ? -1.0 : r.tt[0]);
        CHECK(std::fabs(static_cast<double>(r.nSamp) - expected) <= 2.0,
              "legacy render is overlap - 200 ms longer than the plan");
        CHECK(!r.tt.empty()
              && std::fabs(r.tt[0] - (r.plan.clips[1].timelineStartSec + 0.40)) <= 2.0 / kSr,
              "legacy splice time is 400 ms late");
        const auto p = peaks(r, 0.1f);
        CHECK(p.size() == 2, "legacy: the two clicks land on different samples");
    }

    // ── 3. overlaps within the band cap are untouched ───────────────
    {
        const Rendered a = renderCase(2205, false);
        const Rendered b = renderCase(2205, true);
        CHECK(a.nSamp == b.nSamp, "100 ms overlap: same length under both settings");
        CHECK(std::fabs(static_cast<double>(a.nSamp) - a.plan.duration * kSr) <= 1.0,
              "100 ms overlap: rendered length == plan duration");
        bool same = a.nSamp == b.nSamp;
        for (std::size_t i = 0; same && i < a.audio.size(); ++i)
            same = a.audio[i] == b.audio[i];
        CHECK(same, "100 ms overlap: bit-identical audio under both settings");
    }

    if (g_failures == 0) {
        std::printf("PASS\n");
        return EXIT_SUCCESS;
    }
    std::printf("FAIL (%d)\n", g_failures);
    return EXIT_FAILURE;
}
