// test_custom_kind_registry — sesja 100c (ADR-092) self-validation.
//
// Validates CustomKindRegistry CRUD + JSON round-trip + ID format invariants.
//
// Per ADR-065 + memory `feedback_python_no_longer_source_of_truth.md`:
// hand-computed invariants, no Python ground truth (custom kind registry is
// C++-canonical infrastructure).

#include "ui/CustomKindRegistry.h"

#include <juce_core/juce_core.h>
#include <juce_graphics/juce_graphics.h>

#include <cstdio>

namespace
{

#define CHECK(expr, msg) do {                                                  \
    if (! (expr)) {                                                            \
        std::printf ("FAIL: %s — %s\n", #expr, msg);                           \
        return false;                                                          \
    }                                                                          \
} while (0)

bool testAddAndLookup()
{
    reamix::ui::CustomKindRegistry r;
    r.setIdSeedForTesting (0x12345678);

    CHECK (r.empty(), "fresh registry should be empty");

    const auto id = r.add ("Pre-chorus 2", juce::Colour (0xFF884466));

    CHECK (id.isNotEmpty(),                      "add should return non-empty id");
    CHECK (id.startsWith ("ck_"),                "id must start with ck_ prefix");
    CHECK (id.length() == 11,                    "id must be ck_ + 8 hex chars = 11 chars");
    CHECK (r.size() == 1,                        "size after add should be 1");

    auto entry = r.lookup (id);
    CHECK (entry.has_value(),                    "lookup must succeed for added id");
    CHECK (entry->name == "Pre-chorus 2",        "name must round-trip");
    CHECK (entry->color.getARGB() == 0xFF884466, "color must round-trip");

    CHECK (! r.lookup ("ck_deadbeef").has_value(), "lookup of unknown id must miss");

    return true;
}

bool testRenameAndRecolor()
{
    reamix::ui::CustomKindRegistry r;
    r.setIdSeedForTesting (0xDEADBEEF);
    const auto id = r.add ("Bridge alt", juce::Colours::red);

    r.rename  (id, "Bridge alt v2");
    r.recolor (id, juce::Colour (0xFF112233));

    auto e = r.lookup (id);
    CHECK (e.has_value(),                        "entry must exist after rename/recolor");
    CHECK (e->name == "Bridge alt v2",           "rename must update name");
    CHECK (e->color.getARGB() == 0xFF112233,     "recolor must update color");

    // Mutating an unknown id must be silent no-op (not throw).
    r.rename  ("ck_unknown", "X");
    r.recolor ("ck_unknown", juce::Colours::black);
    CHECK (r.size() == 1,                        "no-op rename/recolor must not add entries");

    return true;
}

bool testRemove()
{
    reamix::ui::CustomKindRegistry r;
    r.setIdSeedForTesting (0xABCD1234);

    const auto id1 = r.add ("Outro stretch", juce::Colours::yellow);
    const auto id2 = r.add ("Bridge B",      juce::Colours::cyan);
    CHECK (r.size() == 2, "size after 2 adds");

    CHECK (r.remove (id1),                       "remove of present id returns true");
    CHECK (! r.remove (id1),                     "double-remove returns false");
    CHECK (r.size() == 1,                        "size after remove");
    CHECK (! r.lookup (id1).has_value(),         "removed id should not look up");
    CHECK (r.lookup (id2).has_value(),           "untouched id should still look up");

    return true;
}

bool testJsonRoundTrip()
{
    reamix::ui::CustomKindRegistry orig;
    orig.setIdSeedForTesting (0xCAFEF00D);
    const auto id1 = orig.add ("Pre-chorus 2", juce::Colour (0xFF884466));
    const auto id2 = orig.add ("Verse alt",    juce::Colour (0xFF335577));
    const auto id3 = orig.add ("Hook",         juce::Colour (0xFFAA9911));

    const juce::String json = orig.serialize();
    CHECK (json.contains ("Pre-chorus 2"), "json must contain entry name");
    CHECK (json.contains (id2),            "json must contain id");

    reamix::ui::CustomKindRegistry restored;
    restored.deserialize (json);

    CHECK (restored.size() == 3, "size after deserialize");

    auto e1 = restored.lookup (id1);
    auto e2 = restored.lookup (id2);
    auto e3 = restored.lookup (id3);
    CHECK (e1.has_value() && e1->name == "Pre-chorus 2", "id1 round-trip");
    CHECK (e2.has_value() && e2->name == "Verse alt",    "id2 round-trip");
    CHECK (e3.has_value() && e3->name == "Hook",         "id3 round-trip");
    CHECK (e1->color.getARGB() == 0xFF884466,            "id1 color round-trip");

    auto ordered = restored.all();
    CHECK (ordered.size() == 3,             "all() returns all entries");
    CHECK (ordered[0].first == id1,         "insertion order preserved (1)");
    CHECK (ordered[1].first == id2,         "insertion order preserved (2)");
    CHECK (ordered[2].first == id3,         "insertion order preserved (3)");

    return true;
}

bool testDeserializeRejectsMalformed()
{
    reamix::ui::CustomKindRegistry r;
    // Empty / non-array / malformed entries — must not throw, must result
    // in empty registry.
    r.deserialize ("");                                        CHECK (r.empty(), "empty json");
    r.deserialize ("{}");                                      CHECK (r.empty(), "non-array json");
    r.deserialize ("[1, 2, 3]");                               CHECK (r.empty(), "array of non-objects");
    r.deserialize ("[{\"id\": \"missing_prefix\", \"name\": \"X\", \"color\": \"FFFFFFFF\"}]");
    CHECK (r.empty(),                                          "id without ck_ prefix rejected");
    r.deserialize ("[{\"id\": \"ck_aaaa1111\", \"name\": \"\", \"color\": \"FFFFFFFF\"}]");
    CHECK (r.empty(),                                          "empty name rejected");

    // Valid entry mixed with malformed — valid one should be retained.
    r.deserialize ("[{\"id\": \"ck_aaaa1111\", \"name\": \"OK\", \"color\": \"FF112233\"},"
                   " {\"id\": \"bad\", \"name\": \"X\", \"color\": \"FFFFFFFF\"}]");
    CHECK (r.size() == 1,                                      "valid entry retained, bad one rejected");
    auto e = r.lookup ("ck_aaaa1111");
    CHECK (e.has_value() && e->name == "OK",                   "valid entry parsed");

    return true;
}

bool testCascadeDetectionShape()
{
    // Caller responsibility: when removing a custom kind, iterate UserBlocks
    // and clear customKindId. Registry itself only confirms the entry is gone
    // — it doesn't know about UserBlocks. This test asserts the lookup-miss
    // shape is what callers can detect for cascade.
    reamix::ui::CustomKindRegistry r;
    r.setIdSeedForTesting (0x55555555);
    const auto id = r.add ("To be deleted", juce::Colours::white);
    CHECK (r.lookup (id).has_value(), "pre-remove lookup");

    r.remove (id);
    CHECK (! r.lookup (id).has_value(), "post-remove lookup miss is the cascade signal");
    return true;
}

} // namespace

int main()
{
    bool ok = true;
    struct TestCase { const char* name; bool (*fn)(); };
    const TestCase cases [] = {
        { "addAndLookup",                testAddAndLookup },
        { "renameAndRecolor",            testRenameAndRecolor },
        { "remove",                      testRemove },
        { "jsonRoundTrip",               testJsonRoundTrip },
        { "deserializeRejectsMalformed", testDeserializeRejectsMalformed },
        { "cascadeDetectionShape",       testCascadeDetectionShape },
    };
    for (auto& tc : cases)
    {
        const bool pass = tc.fn();
        std::printf ("%s %s\n", pass ? "PASS" : "FAIL", tc.name);
        if (! pass) ok = false;
    }
    std::printf ("\n%s — custom kind registry\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
