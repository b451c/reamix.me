// Parity / correctness test for reamix::AudioLoader.
//
// Two asymmetric halves:
//
//   (1) resample() — BITWISE parity vs reabeat_ref::resample, a standalone
//       reference copy of REABeat BeatDetector::resampleTo22050 (generalized
//       from a hardcoded 22050 to an arbitrary dstRate). Both ports call the
//       same juce::LagrangeInterpolator on the same input buffer with the
//       same process() arguments, so max_abs_diff == 0.0 is expected.
//
//   (2) loadMono() — SEMANTIC correctness. REABeat has no corresponding
//       module (its production path reads audio via REAPER's PCM_Source
//       decoders, not JUCE). The bitwise-parity pattern therefore does not
//       apply. Instead: write a known float buffer to a temp WAV via JUCE's
//       own AudioFormatWriter, load it back via AudioLoader::loadMono, and
//       verify the round-trip error is bounded by the PCM16 quantization
//       step (~1/32768). This confirms our usage of JUCE's API is correct
//       (mono mixdown, sample-rate reporting, length, channel handling)
//       without manufacturing fictitious "parity" between two copies of the
//       same code.
//
// Exit 0 on pass, 1 on first failure; prints per-case max_abs_diff.

#include "io/AudioLoader.h"
#include "AudioLoaderReabeat.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace {

// Bit-exact float comparison via integer reinterpretation. Two floats are
// bit-equal iff memcpy'd to uint32_t they compare equal. This sidesteps the
// NaN != NaN trap and treats +0 and -0 as distinct (which is what we want
// for parity).
bool floatsBitEqual(float a, float b)
{
    std::uint32_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}

bool allBitEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!floatsBitEqual(a[i], b[i])) return false;
    return true;
}

// Deterministic sine-wave generator. Pure float arithmetic; no RNG.
std::vector<float> makeSine(int sampleRate, float durationSec, float freqHz, float amp = 0.7f)
{
    int n = static_cast<int>(static_cast<double>(sampleRate) * durationSec);
    std::vector<float> out(static_cast<size_t>(n));
    const float twoPiFOverSr = 2.0f * std::numbers::pi_v<float> * freqHz
                             / static_cast<float>(sampleRate);
    for (int i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] = amp * std::sin(twoPiFOverSr * static_cast<float>(i));
    return out;
}

struct ResampleCase
{
    const char* name;
    int srcRate;
    int dstRate;
    std::vector<float> input;
};

int runResampleCase(const ResampleCase& tc)
{
    auto got = reamix::AudioLoader::resample(tc.input, tc.srcRate, tc.dstRate);
    auto want = reabeat_ref::resample(tc.input, tc.srcRate, tc.dstRate);

    double m = maxAbsDiff(got, want);
    bool sizeEq = (got.size() == want.size());
    bool bitEq = allBitEqual(got, want);

    std::printf("resample/%s: src=%d dst=%d inN=%zu outN=%zu refN=%zu max_abs_diff=%.9g bit_equal=%s\n",
                tc.name, tc.srcRate, tc.dstRate, tc.input.size(), got.size(), want.size(),
                m, bitEq ? "yes" : "no");

    if (!sizeEq || !bitEq)
    {
        std::printf("  FAIL\n");
        return 1;
    }
    return 0;
}

int runLoadMonoRoundTrip()
{
    // Stereo input: L = sine(440, amp=0.5), R = sine(880, amp=0.5). Mixed
    // down to mono, the expected mono sample at index i is 0.5*(L[i]+R[i]).
    const int sr = 44100;
    const float durSec = 2.0f;
    const int numSamples = static_cast<int>(static_cast<double>(sr) * durSec);

    auto l = makeSine(sr, durSec, 440.0f, 0.5f);
    auto r = makeSine(sr, durSec, 880.0f, 0.5f);

    std::vector<float> expected(static_cast<size_t>(numSamples));
    for (int i = 0; i < numSamples; ++i)
        expected[static_cast<size_t>(i)] = 0.5f * (l[static_cast<size_t>(i)]
                                                 + r[static_cast<size_t>(i)]);

    // Write temp WAV via juce::WavAudioFormat, PCM 16-bit.
    auto tempFile = juce::File::createTempFile(".wav");

    {
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::FileOutputStream> out(tempFile.createOutputStream());
        if (!out)
        {
            std::printf("loadMono: FAIL — could not open temp output stream\n");
            return 1;
        }

        juce::StringPairArray metadata;
        std::unique_ptr<juce::AudioFormatWriter> writer(
            wav.createWriterFor(out.get(), static_cast<double>(sr), 2, 16, metadata, 0));
        if (!writer)
        {
            std::printf("loadMono: FAIL — could not create WAV writer\n");
            return 1;
        }
        (void)out.release(); // writer now owns the stream

        juce::AudioBuffer<float> buf(2, numSamples);
        std::copy(l.begin(), l.end(), buf.getWritePointer(0));
        std::copy(r.begin(), r.end(), buf.getWritePointer(1));

        const float* chans[2] = { buf.getReadPointer(0), buf.getReadPointer(1) };
        if (!writer->writeFromFloatArrays(chans, 2, numSamples))
        {
            std::printf("loadMono: FAIL — writeFromFloatArrays returned false\n");
            return 1;
        }
    } // writer/out destroyed — file flushed and closed

    int gotSr = 0;
    std::string err;
    auto loaded = reamix::AudioLoader::loadMono(tempFile, gotSr, err);
    tempFile.deleteFile();

    if (!loaded)
    {
        std::printf("loadMono: FAIL — %s\n", err.c_str());
        return 1;
    }
    if (gotSr != sr)
    {
        std::printf("loadMono: FAIL — sample rate mismatch: got %d want %d\n", gotSr, sr);
        return 1;
    }
    if (static_cast<int>(loaded->size()) != numSamples)
    {
        std::printf("loadMono: FAIL — length mismatch: got %zu want %d\n", loaded->size(), numSamples);
        return 1;
    }

    double m = maxAbsDiff(*loaded, expected);
    // PCM16 quantization step = 1/32768 ≈ 3.052e-5. Allow ~2 steps of slack
    // for JUCE's internal scaling (symmetric / asymmetric quantizer choices
    // vary; we're testing that our usage is sound, not the codec's exact
    // rounding). 1.0e-4 is well below any real signal feature.
    const double tolerance = 1.0e-4;
    std::printf("loadMono: sr=%d N=%d max_abs_diff=%.9g (tol=%.3g) %s\n",
                gotSr, numSamples, m, tolerance, (m <= tolerance ? "OK" : "FAIL"));

    if (m > tolerance) return 1;

    return 0;
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
{
    // Case 1: srcRate == dstRate → short-circuit return of input as-is.
    //         Expected: output == input, bit-equal (pointer may differ;
    //         value sequence must not).
    // Case 2: downsample 44100 → 22050 (ratio 0.5).
    // Case 3: upsample 22050 → 44100 (ratio 2.0).
    // Case 4: odd ratio 48000 → 22050 (ratio 0.459375).
    // Case 5: empty input (audio.size() == 0). outSize = 0, output empty.
    std::vector<ResampleCase> cases = {
        {"passthrough_22050",  22050, 22050, makeSine(22050, 1.0f, 440.0f)},
        {"down_44100_to_22050",44100, 22050, makeSine(44100, 1.0f, 440.0f)},
        {"up_22050_to_44100",  22050, 44100, makeSine(22050, 1.0f, 440.0f)},
        {"odd_48000_to_22050", 48000, 22050, makeSine(48000, 0.5f, 523.25f)},
        {"empty_input",        44100, 22050, std::vector<float>{}},
    };

    int fails = 0;
    for (const auto& tc : cases)
        fails += runResampleCase(tc);

    fails += runLoadMonoRoundTrip();

    if (fails == 0)
    {
        std::printf("audio_loader: all cases PASS\n");
        return 0;
    }
    std::printf("audio_loader: %d case(s) FAILED\n", fails);
    return 1;
}
