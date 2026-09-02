// test_user_blocks_state — sesja 120 (DEV-097) self-validation.
//
// The P_EXT payload of Block Assembly carries blocks + queue + variations
// (UserBlocksState, v2 object). Invariants:
//   1. v2 round trip: blocks (kind, label, custom id), queue, variations;
//   2. v1 bare array still reads (queue empty, variations empty);
//   3. queue entries out of range are dropped, variations sized to junctions;
//   4. unsorted blocks are sorted on read so the queue indexes the same
//      blocks the writer had (userBlocks_ is kept sorted);
//   5. empty blocks -> empty payload (clears the P_EXT entry).
// Per ADR-065: hand-computed invariants, no Python ground truth.

#include "ui/UserBlock.h"

#include <cstdio>

using namespace reamix::ui;

namespace {
int g_failures = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); ++g_failures; } \
        else         { std::printf("  ok:   %s\n", msg); }                      \
    } while (0)
}

int main()
{
    std::printf("[1] v2 round trip\n");
    UserBlocksState st;
    UserBlock a; a.startSec = 0.5; a.endSec = 28.0; a.kind = reamix::theme::SegmentKind::Intro;
    UserBlock b; b.startSec = 28.0; b.endSec = 57.0; b.kind = reamix::theme::SegmentKind::Verse; b.labelOverride = juce::String ("Verse A");
    UserBlock c; c.startSec = 72.0; c.endSec = 96.0; c.kind = reamix::theme::SegmentKind::Chorus; c.customKindId = juce::String ("ck_1a2b3c4d");
    st.blocks = { a, b, c };
    st.queue  = { 0, 2, 1, 2 };
    st.variations = { 0, 1, 0 };
    const juce::String json = serializeUserBlocksState (st);
    CHECK (juce::JSON::parse (json).isObject() && (int) juce::JSON::parse (json).getProperty ("v", 0) == 2,
           "payload is a v2 object");
    const UserBlocksState rt = deserializeUserBlocksState (json);
    CHECK (rt.blocks.size() == 3, "3 blocks back");
    CHECK (rt.blocks.size() == 3 && rt.blocks[1].labelOverride.has_value() && *rt.blocks[1].labelOverride == "Verse A", "label override survives");
    CHECK (rt.blocks.size() == 3 && rt.blocks[2].customKindId.has_value() && *rt.blocks[2].customKindId == "ck_1a2b3c4d", "custom kind id survives");
    CHECK (rt.blocks.size() == 3 && rt.blocks[0].kind == reamix::theme::SegmentKind::Intro, "kind survives");
    CHECK (rt.queue == std::vector<int> ({ 0, 2, 1, 2 }), "queue survives");
    CHECK (rt.variations == std::vector<int> ({ 0, 1, 0 }), "variations survive");

    std::printf("[2] v1 bare array\n");
    const UserBlocksState v1 = deserializeUserBlocksState ("[{\"s\":12.5,\"e\":34.7,\"k\":3},{\"s\":34.7,\"e\":67.1,\"k\":1,\"l\":\"Verse 2\"}]");
    CHECK (v1.blocks.size() == 2, "two v1 blocks");
    CHECK (v1.queue.empty() && v1.variations.empty(), "v1 has no queue");

    std::printf("[3] queue validation\n");
    const UserBlocksState bad = deserializeUserBlocksState (
        "{\"v\":2,\"blocks\":[{\"s\":0,\"e\":10,\"k\":0},{\"s\":10,\"e\":20,\"k\":1}],\"queue\":[0,7,1,-1,1],\"var\":[3]}");
    CHECK (bad.queue == std::vector<int> ({ 0, 1, 1 }), "out-of-range queue entries dropped");
    CHECK (bad.variations.size() == 2 && bad.variations[0] == 3 && bad.variations[1] == 0, "variations sized to junctions, missing = 0");

    std::printf("[4] unsorted blocks\n");
    const UserBlocksState uns = deserializeUserBlocksState (
        "{\"v\":2,\"blocks\":[{\"s\":50,\"e\":60,\"k\":3},{\"s\":0,\"e\":10,\"k\":0}],\"queue\":[1,0]}");
    CHECK (uns.blocks.size() == 2 && uns.blocks[0].startSec == 0.0, "blocks sorted by start on read");
    CHECK (uns.queue == std::vector<int> ({ 1, 0 }), "queue kept as written (indexes the sorted list)");

    std::printf("[5] empty\n");
    CHECK (serializeUserBlocksState (UserBlocksState{}).isEmpty(), "no blocks -> empty payload");
    CHECK (deserializeUserBlocksState ("").blocks.empty(), "empty payload -> no state");
    CHECK (deserializeUserBlocksState ("garbage").blocks.empty(), "garbage -> no state");

    if (g_failures == 0) { std::printf("test_user_blocks_state: PASS\n"); return 0; }
    std::printf("test_user_blocks_state: %d FAILURE(S)\n", g_failures);
    return 1;
}
