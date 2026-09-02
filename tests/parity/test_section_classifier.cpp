// test_section_classifier — sesja 121 (DEV-098) self-validation of the LinkSeg
// section-classifier front-end + decoder (ADR-065: hand-computed invariants,
// no ORT). The model itself is checked end-to-end by the calibration harness
// `--dump-sections` against tools/dev/section_eval (Python) on real tracks.

#include "analysis/SectionClassifier.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

using reamix::analysis::SectionClassifier;
using reamix::analysis::SectionModelOutput;

static int failures = 0;
#define CHECK(cond, msg) do { if (! (cond)) { std::printf ("FAIL: %s\n", msg); ++failures; } } while (0)

static void testRequantise()
{
    // Duplicates collapse, frame 0 is always present, order is preserved.
    std::vector<double> beats { 0.5, 0.5001, 1.0, 1.0, 2.0 };
    auto q = SectionClassifier::requantiseBeats (beats);
    CHECK (q.size() == 4, "requantise: 0 + three unique frames");
    CHECK (q[0] == 0.0, "requantise: frame 0 first");
    for (std::size_t i = 1; i < q.size(); ++i) CHECK (q[i] > q[i - 1], "requantise: strictly increasing");
    // floor(0.5 * 22050 / 256) = 43 -> 43 * 256 / 22050
    CHECK (std::abs (q[1] - 43.0 * 256.0 / 22050.0) < 1e-12, "requantise: floor to 256-sample frame");

    // > 1500 beats halve until <= 1500.
    std::vector<double> many;
    for (int i = 0; i < 4000; ++i) many.push_back (0.05 * i);
    auto q2 = SectionClassifier::requantiseBeats (many);
    CHECK ((int) q2.size() <= SectionClassifier::kMaxBeats, "requantise: capped at 1500");
    CHECK ((int) q2.size() > SectionClassifier::kMaxBeats / 2, "requantise: halving, not truncation");
}

static void testMelWindows()
{
    // 1 kHz tone at 22050 Hz: the loudest mel band must sit at the band whose
    // centre is nearest 1 kHz (htk mel, 64 bands over 0..11025 Hz).
    const int n = 22050 * 3;
    std::vector<float> tone ((std::size_t) n);
    for (int i = 0; i < n; ++i) tone[(std::size_t) i] = 0.5f * (float) std::sin (2.0 * std::numbers::pi * 1000.0 * i / 22050.0);
    std::vector<double> beats { 1.0, 1.5, 2.0 };
    auto mel = SectionClassifier::melWindows (tone.data(), tone.size(), beats);
    CHECK (mel.size() == 3u * 64u * 64u, "mel: N x 64 x 64");
    auto hzToMel = [] (double hz) { return 2595.0 * std::log10 (1.0 + hz / 700.0); };
    auto melToHz = [] (double m) { return 700.0 * (std::pow (10.0, m / 2595.0) - 1.0); };
    const double mMax = hzToMel (11025.0);
    int expected = 0; double bestD = 1e9;
    for (int m = 0; m < 64; ++m)
    {
        const double centre = melToHz (mMax * (m + 1) / 65.0);
        if (std::abs (centre - 1000.0) < bestD) { bestD = std::abs (centre - 1000.0); expected = m; }
    }
    for (int w = 0; w < 3; ++w)
    {
        int arg = 0; float best = -1e9f;
        for (int m = 0; m < 64; ++m)
        {
            const float v = mel[(std::size_t) (w * 4096 + m * 64 + 32)];
            CHECK (std::isfinite (v), "mel: finite");
            if (v > best) { best = v; arg = m; }
        }
        CHECK (std::abs (arg - expected) <= 1, "mel: 1 kHz tone peaks in the expected band");
        CHECK (best > -20.0f && best < 60.0f, "mel: dB range plausible for a 0.5 amplitude tone");
    }
    // Silence -> exactly the floor 10*log10(1e-10) = -100 dB.
    std::vector<float> zeros ((std::size_t) 22050, 0.0f);
    auto melZ = SectionClassifier::melWindows (zeros.data(), zeros.size(), { 0.5 });
    CHECK (std::abs (melZ[0] + 100.0f) < 1e-3f, "mel: silence hits the 1e-10 floor");
}

static void testDecode()
{
    // 41 beats at 0.5 s (20 s), two clear boundary peaks at mid indices 10 and
    // 30 (10 s apart: a peak must be the maximum within +-8 s), labels:
    // intro (3) until beat 10, chorus (2) until 30, outro (4) after.
    SectionModelOutput out;
    for (int i = 0; i <= 40; ++i) out.beatTimes.push_back (0.5 * i);
    out.nClasses = 7;
    out.bound.assign (40, 0.05f);
    out.bound[10] = 0.9f;
    out.bound[30] = 0.8f;
    out.bound[31] = 0.6f;          // shoulder inside the 8 s window of peak 30: not a peak
    for (int n = 0; n <= 40; ++n)
    {
        std::vector<float> row (7, -5.0f);
        row[n < 10 ? 3 : (n < 30 ? 2 : 4)] = 4.0f;
        out.labelLogits.insert (out.labelLogits.end(), row.begin(), row.end());
    }
    auto segs = SectionClassifier::decode (out, 20.0);
    CHECK (segs.size() == 3, "decode: two peaks -> three segments");
    if (segs.size() == 3)
    {
        CHECK (segs[0].start == 0.0, "decode: first segment starts at 0");
        CHECK (segs[2].end == 20.0, "decode: last segment ends at duration");
        CHECK (std::abs (segs[0].end - 0.5 * (out.beatTimes[10] + out.beatTimes[11])) < 1e-9, "decode: boundary at the mid-beat time");
        CHECK (segs[0].label == "intro" && segs[1].label == "chorus" && segs[2].label == "outro", "decode: majority labels");
        CHECK (segs[1].confidence > 0.99, "decode: softmax confidence of a clear class");
        for (std::size_t i = 1; i < segs.size(); ++i) CHECK (segs[i].start == segs[i - 1].end, "decode: contiguous");
    }
    // Flat activations -> one segment; label tie -> smallest class (scipy mode).
    SectionModelOutput flat;
    for (int i = 0; i <= 8; ++i) flat.beatTimes.push_back (0.5 * i);
    flat.nClasses = 7;
    flat.bound.assign (8, 0.0f);
    for (int n = 0; n <= 8; ++n)
    {
        std::vector<float> row (7, 0.0f);
        row[n % 2 == 0 ? 6 : 1] = 1.0f;      // 5 x bridge(6) vs 4 x verse(1) over beats 0..7 -> bridge
        flat.labelLogits.insert (flat.labelLogits.end(), row.begin(), row.end());
    }
    auto one = SectionClassifier::decode (flat, 4.0);
    CHECK (one.size() == 1, "decode: flat -> single segment");
    CHECK (! one.empty() && one[0].label == "bridge", "decode: majority over the segment's beats");
    // 9-class label names.
    CHECK (std::string (SectionClassifier::labelName (7)) == "pre-chorus", "labels: 7 = pre-chorus");
    CHECK (std::string (SectionClassifier::labelName (8)) == "post-chorus", "labels: 8 = post-chorus");
}

int main()
{
    testRequantise();
    testMelWindows();
    testDecode();
    if (failures == 0) std::printf ("test_section_classifier: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
