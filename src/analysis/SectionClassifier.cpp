#include "analysis/SectionClassifier.h"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <map>
#include <numbers>
#include <utility>

namespace reamix::analysis {

namespace {

constexpr int kNfft   = 1024;
constexpr int kNfreqs = kNfft / 2 + 1;   // 513
constexpr double kAmin = 1e-10;

// torchaudio.functional.melscale_fbanks(n_freqs=513, f_min=0, f_max=11025,
// n_mels=64, sample_rate=22050, norm=None, mel_scale="htk"), stored [mel][bin].
const std::vector<std::vector<float>>& melFilterbank()
{
    static const std::vector<std::vector<float>> fb = []
    {
        auto hzToMel = [] (double hz) { return 2595.0 * std::log10 (1.0 + hz / 700.0); };
        auto melToHz = [] (double mel) { return 700.0 * (std::pow (10.0, mel / 2595.0) - 1.0); };
        const double fMax = SectionClassifier::kSampleRate / 2.0;
        const int nMels = SectionClassifier::kMelBands;
        std::vector<double> allFreqs ((std::size_t) kNfreqs);
        for (int k = 0; k < kNfreqs; ++k) allFreqs[(std::size_t) k] = fMax * k / (kNfreqs - 1);
        const double mMax = hzToMel (fMax);
        std::vector<double> fPts ((std::size_t) nMels + 2);
        for (int m = 0; m < nMels + 2; ++m) fPts[(std::size_t) m] = melToHz (mMax * m / (nMels + 1));
        std::vector<std::vector<float>> out ((std::size_t) nMels, std::vector<float> ((std::size_t) kNfreqs, 0.0f));
        for (int m = 0; m < nMels; ++m)
        {
            const double lo = fPts[(std::size_t) m], mid = fPts[(std::size_t) m + 1], hi = fPts[(std::size_t) m + 2];
            for (int k = 0; k < kNfreqs; ++k)
            {
                const double f = allFreqs[(std::size_t) k];
                const double down = (f - lo) / (mid - lo);     // -slopes[:, :-2] / f_diff[:-1]
                const double up   = (hi - f) / (hi - mid);     //  slopes[:, 2:]  / f_diff[1:]
                out[(std::size_t) m][(std::size_t) k] = (float) std::max (0.0, std::min (down, up));
            }
        }
        return out;
    }();
    return fb;
}

const std::vector<float>& hannPeriodic()
{
    static const std::vector<float> w = []
    {
        std::vector<float> v ((std::size_t) kNfft);
        for (int n = 0; n < kNfft; ++n)
            v[(std::size_t) n] = (float) (0.5 * (1.0 - std::cos (2.0 * std::numbers::pi * n / kNfft)));
        return v;
    }();
    return w;
}

// One 16382-sample window -> 64 mel x 64 frames dB, written at dst[mel*64 + t].
void melWindow (const float* win, float* dst)
{
    constexpr int pad = kNfft / 2;
    std::vector<float> padded ((std::size_t) SectionClassifier::kWindow + 2 * pad);
    for (int k = 0; k < pad; ++k) padded[(std::size_t) k] = win[pad - k];                 // reflect left
    std::copy (win, win + SectionClassifier::kWindow, padded.begin() + pad);
    for (int k = 0; k < pad; ++k)
        padded[(std::size_t) (pad + SectionClassifier::kWindow + k)] = win[SectionClassifier::kWindow - 2 - k];

    const auto& hann = hannPeriodic();
    const auto& fb   = melFilterbank();
    pocketfft::shape_t  shape     = { (std::size_t) kNfft };
    pocketfft::stride_t strideIn  = { sizeof (float) };
    pocketfft::stride_t strideOut = { sizeof (std::complex<float>) };
    std::vector<float>               frame ((std::size_t) kNfft);
    std::vector<std::complex<float>> spec ((std::size_t) kNfreqs);
    std::vector<double>              power ((std::size_t) kNfreqs);

    for (int t = 0; t < SectionClassifier::kMelFrames; ++t)
    {
        const int start = t * SectionClassifier::kHop;
        for (int n = 0; n < kNfft; ++n) frame[(std::size_t) n] = padded[(std::size_t) (start + n)] * hann[(std::size_t) n];
        pocketfft::r2c (shape, strideIn, strideOut, 0, true, frame.data(), spec.data(), 1.0f);
        for (int k = 0; k < kNfreqs; ++k)
        {
            const double re = spec[(std::size_t) k].real(), im = spec[(std::size_t) k].imag();
            power[(std::size_t) k] = re * re + im * im;
        }
        for (int m = 0; m < SectionClassifier::kMelBands; ++m)
        {
            const auto& row = fb[(std::size_t) m];
            double acc = 0.0;
            for (int k = 0; k < kNfreqs; ++k) acc += (double) row[(std::size_t) k] * power[(std::size_t) k];
            dst[m * SectionClassifier::kMelFrames + t] = (float) (10.0 * std::log10 (std::max (acc, kAmin)));
        }
    }
}

constexpr const char* kLabels[9] = { "silence", "verse", "chorus", "intro", "outro",
                                     "inst", "bridge", "pre-chorus", "post-chorus" };

} // namespace

SectionClassifier::SectionClassifier() = default;
SectionClassifier::~SectionClassifier() = default;

const char* SectionClassifier::labelName (int cls)
{
    return (cls >= 0 && cls < 9) ? kLabels[cls] : "verse";
}

bool SectionClassifier::loadModel (const std::string& modelPath)
{
#if REAMIX_HAS_ONNX
    try
    {
        env_ = std::make_unique<Ort::Env> (ORT_LOGGING_LEVEL_WARNING, "reamix.sections");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads (4);
        opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wide (modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session> (*env_, wide.c_str(), opts);
#else
        session_ = std::make_unique<Ort::Session> (*env_, modelPath.c_str(), opts);
#endif
        loaded_ = true;
        return true;
    }
    catch (const Ort::Exception&)
    {
        session_.reset();
        loaded_ = false;
        return false;
    }
#else
    (void) modelPath;
    return false;
#endif
}

std::vector<double> SectionClassifier::requantiseBeats (const std::vector<double>& beatTimes)
{
    std::vector<long long> frames;
    frames.reserve (beatTimes.size() + 1);
    frames.push_back (0);
    for (double t : beatTimes)
    {
        const long long f = (long long) std::floor (t * kSampleRate / kHop);
        if (f >= 0) frames.push_back (f);
    }
    std::sort (frames.begin(), frames.end());
    frames.erase (std::unique (frames.begin(), frames.end()), frames.end());
    while ((int) frames.size() > kMaxBeats)
    {
        std::vector<long long> half;
        for (std::size_t i = 0; i < frames.size(); i += 2) half.push_back (frames[i]);
        frames.swap (half);
    }
    std::vector<double> out (frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) out[i] = (double) frames[i] * kHop / kSampleRate;
    return out;
}

std::vector<float> SectionClassifier::melWindows (const float* mono22k, std::size_t numSamples,
                                                  const std::vector<double>& beatTimes)
{
    const std::size_t N = beatTimes.size();
    std::vector<float> out (N * kMelBands * kMelFrames, 0.0f);
    if (numSamples == 0 || N == 0) return out;

    // np.pad(mode="edge") by kPad on both sides.
    std::vector<float> padded (numSamples + 2 * (std::size_t) kPad);
    std::fill (padded.begin(), padded.begin() + kPad, mono22k[0]);
    std::copy (mono22k, mono22k + numSamples, padded.begin() + kPad);
    std::fill (padded.begin() + kPad + (long) numSamples, padded.end(), mono22k[numSamples - 1]);

    std::vector<float> win ((std::size_t) kWindow);
    for (std::size_t i = 0; i < N; ++i)
    {
        const long long start = (long long) std::floor (beatTimes[i] * kSampleRate);
        for (int k = 0; k < kWindow; ++k)
        {
            const long long idx = start + k;
            win[(std::size_t) k] = (idx >= 0 && idx < (long long) padded.size()) ? padded[(std::size_t) idx]
                                                                                 : padded.back();
        }
        melWindow (win.data(), out.data() + i * kMelBands * kMelFrames);
    }
    return out;
}

SectionModelOutput SectionClassifier::run (const float* mono22k, std::size_t numSamples,
                                           const std::vector<double>& beatTimes, std::string* err)
{
    SectionModelOutput out;
    if (! loaded_) { if (err) *err = "section model not loaded"; return out; }
    if ((int) beatTimes.size() < kMinBeats || numSamples == 0)
    {
        if (err) *err = "too few beats for section analysis";
        return out;
    }
#if REAMIX_HAS_ONNX
    try
    {
        out.beatTimes = requantiseBeats (beatTimes);
        const std::size_t N = out.beatTimes.size();
        if ((int) N < kMinBeats) { if (err) *err = "too few beats for section analysis"; out.beatTimes.clear(); return out; }
        std::vector<float> mel = melWindows (mono22k, numSamples, out.beatTimes);

        const std::array<int64_t, 4> shape { (int64_t) N, 1, kMelBands, kMelFrames };
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float> (mem, mel.data(), mel.size(),
                                                            shape.data(), shape.size());
        const char* inputNames[]  = { "mel" };
        const char* outputNames[] = { "bound", "label" };
        auto results = session_->Run (Ort::RunOptions { nullptr }, inputNames, &input, 1, outputNames, 2);

        const auto bInfo = results[0].GetTensorTypeAndShapeInfo();
        const auto lInfo = results[1].GetTensorTypeAndShapeInfo();
        const auto lShape = lInfo.GetShape();
        out.nClasses = (lShape.size() == 2) ? (int) lShape[1] : 0;
        const float* b = results[0].GetTensorData<float>();
        const float* l = results[1].GetTensorData<float>();
        out.bound.assign (b, b + bInfo.GetElementCount());
        out.labelLogits.assign (l, l + lInfo.GetElementCount());
        if (out.bound.size() + 1 != N || out.labelLogits.size() != N * (std::size_t) out.nClasses)
        {
            if (err) *err = "section model returned unexpected shapes";
            out = {};
        }
    }
    catch (const Ort::Exception& e)
    {
        if (err) *err = std::string ("section model inference failed: ") + e.what();
        out = {};
    }
#else
    (void) mono22k; (void) numSamples;
    if (err) *err = "built without ONNX Runtime";
#endif
    return out;
}

namespace {

// LinkSeg post_processing.get_indices: beat range whose distance to `index`
// stays within the neighbourhood.
std::pair<int, int> neighbourhood (const std::vector<double>& bt, int index, double future, double past)
{
    int left = 0, right = (int) bt.size() - 1;
    for (int i = index; i > 0; --i)
        if (bt[(std::size_t) index] - bt[(std::size_t) i] > future) { left = i - 1; break; }
    for (int i = index; i < (int) bt.size(); ++i)
        if (bt[(std::size_t) i] - bt[(std::size_t) index] > past) { right = i - 1; break; }
    return { left, right };
}

} // namespace

std::vector<SectionSegment> SectionClassifier::decode (const SectionModelOutput& out, double durationSec,
                                                       double windowSec, double tau)
{
    std::vector<SectionSegment> segs;
    const std::size_t N = out.beatTimes.size();
    if (N < 2 || out.bound.size() + 1 != N || out.nClasses <= 0) return segs;

    std::vector<double> mid (N - 1);
    for (std::size_t i = 0; i + 1 < N; ++i) mid[i] = 0.5 * (out.beatTimes[i] + out.beatTimes[i + 1]);

    // Peak picking (empty-neighbourhood fallbacks 0 / 10 as in the reference).
    std::vector<int> peaks;
    const int M = (int) mid.size();
    for (int i = 1; i < M - 1; ++i)
    {
        const auto [ll, lr] = neighbourhood (mid, i, windowSec, windowSec);
        double maxLeft = -1e300, maxRight = -1e300;
        bool anyLeft = false, anyRight = false;
        for (int j = i - 1; j > ll; --j) { maxLeft = std::max (maxLeft, (double) out.bound[(std::size_t) j]); anyLeft = true; }
        for (int j = i + 1; j < lr; ++j) { maxRight = std::max (maxRight, (double) out.bound[(std::size_t) j]); anyRight = true; }
        if (! anyLeft)  maxLeft  = 0.0;
        if (! anyRight) maxRight = 10.0;
        const double v = out.bound[(std::size_t) i];
        if (maxLeft < v && v > maxRight && v > tau) peaks.push_back (i);
    }

    // Per-beat argmax + softmax probability of the argmax.
    const int C = out.nClasses;
    std::vector<int>    cls (N);
    std::vector<double> prob (N);
    for (std::size_t n = 0; n < N; ++n)
    {
        const float* row = out.labelLogits.data() + n * (std::size_t) C;
        int best = 0;
        for (int c = 1; c < C; ++c) if (row[c] > row[best]) best = c;
        double denom = 0.0;
        for (int c = 0; c < C; ++c) denom += std::exp ((double) row[c] - row[best]);
        cls[n]  = best;
        prob[n] = 1.0 / denom;
    }

    std::vector<int> idx;
    idx.push_back (0);
    idx.insert (idx.end(), peaks.begin(), peaks.end());
    idx.push_back (M - 1);
    for (std::size_t s = 0; s + 1 < idx.size(); ++s)
    {
        const int a = idx[s], b = idx[s + 1];
        std::map<int, int> votes;
        for (int r = a; r < b; ++r) votes[cls[(std::size_t) r]]++;
        int chosen = 0, bestCount = -1;
        for (const auto& [c, n] : votes)         // ascending keys: smallest class wins ties (scipy mode)
            if (n > bestCount) { bestCount = n; chosen = c; }
        double conf = 0.0; int cnt = 0;
        for (int r = a; r < b; ++r) if (cls[(std::size_t) r] == chosen) { conf += prob[(std::size_t) r]; ++cnt; }
        SectionSegment seg;
        seg.start      = (s == 0) ? 0.0 : mid[(std::size_t) a];
        seg.end        = (s + 2 == idx.size()) ? durationSec : mid[(std::size_t) b];
        seg.cls        = chosen;
        seg.label      = labelName (chosen);
        seg.confidence = cnt > 0 ? conf / cnt : 0.0;
        segs.push_back (std::move (seg));
    }
    return segs;
}

} // namespace reamix::analysis
