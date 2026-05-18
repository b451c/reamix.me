#include "analysis/BeatWindows.h"
#include <cstdint>

#include <cmath>

namespace reamix::analysis {

namespace {

// numpy.hanning(N): 0.5 * (1 - cos(2π·n / (N-1))) for n ∈ [0, N-1].
// Matches librosa's internal call `np.hanning(total).astype(float32)`;
// we synthesize directly as f32 via a f64 evaluation to keep every bit
// consistent across ports of this helper.
// PARITY: feature_extractor.py:439  window = np.hanning(total_samples).astype(np.float32)
inline void buildHannWindowF32(std::vector<float>& window, std::size_t total)
{
    window.resize(total);
    if (total == 0) return;
    if (total == 1) { window[0] = 0.0f; return; }
    constexpr double kTwoPi = 6.283185307179586476925286766559005768394;
    const double denom = static_cast<double>(total - 1);
    for (std::size_t n = 0; n < total; ++n) {
        const double w = 0.5 * (1.0 - std::cos(kTwoPi * static_cast<double>(n) / denom));
        window[n] = static_cast<float>(w);
    }
}

} // namespace

BeatWindows::EdgeFeatures
BeatWindows::extractEdgeFeatures(const float* features,
                                 std::size_t F,
                                 std::size_t T,
                                 const std::vector<int>& beatFrames,
                                 std::size_t nBeats,
                                 int nEdgeFrames)
{
    EdgeFeatures out;
    out.start.assign(nBeats * F, 0.0);
    out.end.assign(nBeats * F, 0.0);

    if (F == 0 || nBeats == 0) return out;

    const std::size_t nBeatFrames = beatFrames.size();
    const int Ti = static_cast<int>(T);

    for (std::size_t i = 0; i < nBeats; ++i) {
        // PARITY: feature_extractor.py:331-336  int(beat_frames[i]) / beat_frames[i+1]
        //         fallback frame_hi = n_frames; frame_hi = min(frame_hi, n_frames).
        // `int(...)` on a positive int64 is identity; negative inputs are
        // never produced here so no truncation issue.
        int frameLo = (i < nBeatFrames) ? beatFrames[i] : Ti;
        if (frameLo < 0) frameLo = 0;
        if (frameLo > Ti) frameLo = Ti;

        int frameHi;
        if (i + 1 < nBeatFrames) {
            frameHi = beatFrames[i + 1];
            if (frameHi < 0) frameHi = 0;
            if (frameHi > Ti) frameHi = Ti;
        } else {
            frameHi = Ti;
        }

        const int beatLen = frameHi - frameLo;
        if (beatLen <= 0) continue;

        // PARITY: feature_extractor.py:339  n_edge = min(n_edge_frames, max(1, beat_len // 2))
        const int halfFloor = beatLen / 2;
        const int halfAtLeastOne = (halfFloor < 1) ? 1 : halfFloor;
        const int nEdge = (nEdgeFrames < halfAtLeastOne) ? nEdgeFrames : halfAtLeastOne;

        const float invEdge = 1.0f / static_cast<float>(nEdge);
        const std::size_t startOffset = static_cast<std::size_t>(frameLo);
        const std::size_t endOffset   = static_cast<std::size_t>(frameHi - nEdge);

        double* rowStart = out.start.data() + i * F;
        double* rowEnd   = out.end.data()   + i * F;

        for (std::size_t f = 0; f < F; ++f) {
            const float* row = features + f * T;
            // np.mean over axis=1 on a float32 slice stays float32; for
            // n_edge ≤ 4 pairwise and naive-sequential are identical.
            float sumStart = 0.0f;
            float sumEnd   = 0.0f;
            for (int k = 0; k < nEdge; ++k) {
                sumStart += row[startOffset + static_cast<std::size_t>(k)];
                sumEnd   += row[endOffset   + static_cast<std::size_t>(k)];
            }
            // Assignment to float64 output upcasts the float32 mean.
            rowStart[f] = static_cast<double>(sumStart * invEdge);
            rowEnd[f]   = static_cast<double>(sumEnd   * invEdge);
        }
    }

    // PARITY: feature_extractor.py:346-349  per-row L2 normalize with
    //         `norms[norms == 0] = 1.0` (assign 1, NOT epsilon — preserves
    //         zero rows in output).
    auto l2NormalizeInPlace = [&](std::vector<double>& arr) {
        for (std::size_t i = 0; i < nBeats; ++i) {
            double* row = arr.data() + i * F;
            double sumSq = 0.0;
            for (std::size_t f = 0; f < F; ++f) sumSq += row[f] * row[f];
            double norm = std::sqrt(sumSq);
            if (norm == 0.0) norm = 1.0;
            const double inv = 1.0 / norm;
            for (std::size_t f = 0; f < F; ++f) row[f] *= inv;
        }
    };
    l2NormalizeInPlace(out.start);
    l2NormalizeInPlace(out.end);

    return out;
}

BeatWindows::EdgeRms
BeatWindows::extractEdgeRms(const float* y,
                            std::size_t nSamples,
                            int sr,
                            const std::vector<double>& beatTimes,
                            std::size_t nBeats,
                            double edgeMs)
{
    EdgeRms out;
    out.start.assign(nBeats, 0.0);
    out.end.assign(nBeats, 0.0);

    if (nBeats == 0) return out;

    // PARITY: feature_extractor.py:362  edge_samples = int(edge_ms * sr / 1000)
    //         truncation toward zero on a positive f64 product.
    const int edgeSamples =
        static_cast<int>(edgeMs * static_cast<double>(sr) / 1000.0);
    const std::int64_t nSamples64 = static_cast<std::int64_t>(nSamples);

    // PARITY: feature_extractor.py:364  beat_samples = (beat_times * sr).astype(int64)
    //         f64 multiply, truncation cast. Mirrored bit-for-bit.
    std::vector<std::int64_t> beatSamples(nBeats);
    for (std::size_t i = 0; i < nBeats; ++i)
        beatSamples[i] =
            static_cast<std::int64_t>(beatTimes[i] * static_cast<double>(sr));

    for (std::size_t i = 0; i < nBeats; ++i) {
        std::int64_t s = beatSamples[i];
        std::int64_t e = (i + 1 < nBeats)
                              ? beatSamples[i + 1]
                              : nSamples64;
        // PARITY: feature_extractor.py:375-376  e = min(e, n_samples);
        //         s = max(0, min(s, n_samples))
        if (e > nSamples64) e = nSamples64;
        if (s > nSamples64) s = nSamples64;
        if (s < 0) s = 0;

        const std::int64_t beatLen = e - s;
        if (beatLen <= 0) continue;

        const std::int64_t halfFloor = beatLen / 2;
        const std::int64_t halfAtLeastOne = (halfFloor < 1) ? 1 : halfFloor;
        std::int64_t nEdge = static_cast<std::int64_t>(edgeSamples);
        if (halfAtLeastOne < nEdge) nEdge = halfAtLeastOne;

        // PARITY: feature_extractor.py:384-385
        //   edge_rms_start[i] = np.sqrt(np.mean(start_chunk ** 2) + 1e-10)
        //   edge_rms_end[i]   = np.sqrt(np.mean(end_chunk   ** 2) + 1e-10)
        //
        // np.mean over a float32 array returns float32 (pairwise internally).
        // `float32 + 1e-10 (python float = f64)` upcasts the scalar to f64.
        // `np.sqrt(f64) → f64`. The output array is f64 already.
        //
        // We accumulate sum-of-squares in float32 (naive sequential) to
        // match numpy's float32 reduction dtype. For typical edge windows
        // (~1102 samples at 50 ms / 22050 Hz) naive vs pairwise differs by
        // ~log2(1102) ≈ 10 ULP of the sum, i.e. ~1e-6 on rms values of
        // order 0.1 — well under the 1e-3 gate.
        const std::int64_t sStart = s;
        const std::int64_t sEnd   = e - nEdge;

        float sumSqStart = 0.0f;
        float sumSqEnd   = 0.0f;
        for (std::int64_t k = 0; k < nEdge; ++k) {
            const float ys = y[sStart + k];
            const float ye = y[sEnd   + k];
            sumSqStart += ys * ys;
            sumSqEnd   += ye * ye;
        }
        const float meanSqStart_f32 = sumSqStart / static_cast<float>(nEdge);
        const float meanSqEnd_f32   = sumSqEnd   / static_cast<float>(nEdge);

        out.start[i] = std::sqrt(static_cast<double>(meanSqStart_f32) + 1.0e-10);
        out.end[i]   = std::sqrt(static_cast<double>(meanSqEnd_f32)   + 1.0e-10);
    }

    // PARITY: feature_extractor.py:388-390
    //   max_val = max(edge_rms_start.max(), edge_rms_end.max(), 1e-10)
    //   edge_rms_start /= max_val
    //   edge_rms_end   /= max_val
    double maxStart = 0.0, maxEnd = 0.0;
    for (double v : out.start) if (v > maxStart) maxStart = v;
    for (double v : out.end)   if (v > maxEnd)   maxEnd   = v;
    double maxVal = (maxStart > maxEnd) ? maxStart : maxEnd;
    if (maxVal < 1.0e-10) maxVal = 1.0e-10;
    const double invMax = 1.0 / maxVal;
    for (double& v : out.start) v *= invMax;
    for (double& v : out.end)   v *= invMax;

    return out;
}

std::vector<float>
BeatWindows::extractWaveformSnippets(const float* y,
                                     std::size_t nSamples,
                                     int sr,
                                     const std::vector<double>& beatTimes,
                                     std::size_t nBeats,
                                     double preMs,
                                     double postMs)
{
    // PARITY: feature_extractor.py:429-432
    //   pre_samples   = int(pre_ms  * sr / 1000.0)  — truncation (f64 mul).
    //   post_samples  = int(post_ms * sr / 1000.0)
    //   total_samples = max(8, pre_samples + post_samples)
    const int preSamples  = static_cast<int>(preMs  * static_cast<double>(sr) / 1000.0);
    const int postSamples = static_cast<int>(postMs * static_cast<double>(sr) / 1000.0);
    const int totalInt    = preSamples + postSamples;
    const std::size_t total = static_cast<std::size_t>((totalInt < 8) ? 8 : totalInt);

    std::vector<float> snippets(nBeats * total, 0.0f);
    if (nBeats == 0) return snippets;

    // Hann window in f32. `max(8, ...)` upstream ensures total ≥ 8, so
    // librosa's `if np.all(window == 0): window = np.ones(...)` branch
    // (which only fires for N=1) is unreachable. Not guarded.
    std::vector<float> window;
    buildHannWindowF32(window, total);

    // PARITY: feature_extractor.py:437  beat_samples = (beat_times * sr).astype(int64)
    std::vector<std::int64_t> beatSamples(nBeats);
    for (std::size_t i = 0; i < nBeats; ++i)
        beatSamples[i] =
            static_cast<std::int64_t>(beatTimes[i] * static_cast<double>(sr));

    const std::int64_t nAudio64 = static_cast<std::int64_t>(nSamples);

    for (std::size_t i = 0; i < nBeats; ++i) {
        // PARITY: feature_extractor.py:444-451
        const std::int64_t center = beatSamples[i];
        const std::int64_t start  = center - preSamples;
        const std::int64_t end    = center + postSamples;

        const std::int64_t clipStart = (start < 0) ? 0 : start;
        const std::int64_t clipEnd   = (end > nAudio64) ? nAudio64 : end;
        if (clipEnd <= clipStart) continue;

        const std::int64_t outStart = clipStart - start;

        float* snippet = snippets.data() + i * total;

        // Step 1 — copy y[clip_start..clip_end] into snippet at out_start.
        const std::int64_t copyLen = clipEnd - clipStart;
        for (std::int64_t k = 0; k < copyLen; ++k)
            snippet[outStart + k] = y[clipStart + k];

        // Step 2 — DC remove: snippet -= mean(snippet). numpy uses pairwise
        // summation on float32; we use naive sequential. For ~3417 (boundary)
        // or ~13230 (transition) samples, expected ULP drift on the mean is
        // ~log2(N) ULP ≈ 1e-7..1e-6 of the sum, giving <1e-9 in the mean.
        {
            float sum = 0.0f;
            for (std::size_t n = 0; n < total; ++n) sum += snippet[n];
            const float mean = sum / static_cast<float>(total);
            for (std::size_t n = 0; n < total; ++n) snippet[n] -= mean;
        }

        // Step 3 — apply window in-place (float32).
        for (std::size_t n = 0; n < total; ++n) snippet[n] *= window[n];

        // Step 4 — RMS: sqrt(float64(mean_f32(snippet²)) + 1e-8).
        // mean_f32(snippet²) matches numpy's `np.mean(float32_array)`
        // default dtype behaviour (float32 accumulator).
        float sumSq = 0.0f;
        for (std::size_t n = 0; n < total; ++n) sumSq += snippet[n] * snippet[n];
        const float meanSq_f32 = sumSq / static_cast<float>(total);
        const double rms_f64   = std::sqrt(static_cast<double>(meanSq_f32) + 1.0e-8);

        // Step 5 — snippet /= rms. numpy's in-place /= on a float32 array
        // with a float64 rhs casts the scalar to float32 at divide time
        // (modern numpy; older versions emitted a warning). Mirror that
        // explicitly so the port is version-invariant.
        const float rms = static_cast<float>(rms_f64);
        const float invRms = 1.0f / rms;
        for (std::size_t n = 0; n < total; ++n) snippet[n] *= invRms;
    }

    return snippets;
}

} // namespace reamix::analysis
