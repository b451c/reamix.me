// Splice — full SpliceMixin port
// (references/python-source/remix/splice.py, 514 LOC). Port discipline:
// Python-verbatim with -ffp-contract=off per ADR-028 (15th reuse — scalar
// accumulators dominate: np.mean / np.linalg.norm / np.dot / np.convolve
// uniform-kernel / np.diff).
//
// Session 34: 3 core primitives (L85-196) — windowOnsetIndex,
// stereoWindowSimilarity, scoreSplicePair. All at ≤ 1 ULP bit-exact parity
// (max_abs 8.674e-17 on real music).
//
// Session 35: findOnsetSample (L26-83) re-enabled + 4 composite methods —
// scoreAnchorAlignedPair (L198-245), searchAnchorTransitionGeometry
// (L247-330), transitionOverlapSamples (L332-355), refineTransitionSplice
// (L357-514). findOnsetSample primitive parity drifts ±1-4 samples per
// ADR-033; the composition-level test in searchAnchorTransitionGeometry
// determines whether the drift propagates to a different winner-beat-pair
// on real music (ADR-033 close criterion).
//
// Parity class target: int bitwise for discrete outputs (window onset
// indices, sample positions, beat indices); f64 max_abs ≤ 1e-9 for
// continuous outputs (similarity / score / seconds-units). Red-flag
// discipline per CLAUDE.md Principle 6: on uniform-sentinel output or
// fallback-always-fires pattern → bisect, do NOT widen gate.

#include "render/Splice.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace reamix::render {

namespace {

// 2-norm of f64 array via `sqrt(dot(x,x))`. Matches numpy's naive-path
// `np.linalg.norm` at the lengths session-34 exercises (≤ 3417 samples per
// case — well below BLAS dispatch thresholds). Same discipline as
// `src/render/PhaseAlign.cpp` l2norm (session 31 — 0.000e+00 bit-exact parity).
inline double l2norm(const double* x, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += x[i] * x[i];
    }
    return std::sqrt(s);
}

// Straight dot product matching numpy's `np.dot` on two 1-D f64 vectors at
// small-N lengths (naive path, not BLAS-dispatched).
inline double dotProduct(const double* a, const double* b, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

// Mean of a f64 vector, same accumulation order as `np.mean` (linear sum
// then divide). For our case sizes (≤ ~3417) numpy stays on the linear path
// (pairwise threshold is 8192 elements per axis).
inline double meanOf(const double* x, std::size_t n)
{
    if (n == 0) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i];
    return s / static_cast<double>(n);
}

inline double meanOfVec(const std::vector<double>& v)
{
    return meanOf(v.data(), v.size());
}

// np.hanning(M) = 0.5 - 0.5 * cos(2*pi*n / (M-1)) for n ∈ [0, M-1].
// numpy source: numpy/lib/function_base.py::hanning. Requires M ≥ 2; caller
// guards against M < 16 upstream (both callers: _stereo_window_similarity
// rejects n < 16 before calling _get_hanning; _window_onset_index doesn't
// use hanning at all).
inline void hanningWindow(std::size_t m, std::vector<double>& out)
{
    out.resize(m);
    if (m == 0) return;
    if (m == 1) { out[0] = 0.0; return; }
    const double k = 2.0 * 3.14159265358979323846 / static_cast<double>(m - 1);
    for (std::size_t n = 0; n < m; ++n) {
        out[n] = 0.5 - 0.5 * std::cos(k * static_cast<double>(n));
    }
}

// np.convolve(x, np.ones(K)/K, mode='same') — box-kernel moving average.
//
// Output length = N (same as input). Kernel is uniform `1/K`. For each
// output index i, the contributing input indices are:
//   m ∈ [max(0, i + offset - K + 1), min(N-1, i + offset)]
// where `offset = (K - 1) / 2` (integer division — numpy's 'same' centering
// convention; for even K this leans left by 0.5 — documented numpy behavior).
//
// numpy's internal loop (numpy/_core/src/multiarray/multiarraymodule.c
// PyArray_Correlate) multiplies `a[m] * v[k]` per-iteration before
// accumulating. For uniform kernel v[k] = 1/K this means each iteration
// does `a[m] * inv_K`, which we mirror below — NOT `sum(a[m]) * inv_K`
// applied once. The two differ by ~ULPs on each output sample (associativity
// of f64 addition); matching numpy's per-iteration pattern gives bit-exact
// parity at the `-ffp-contract=off` discipline.
inline void boxConvolveSame(const double* x, std::size_t n,
                            std::size_t kernelSize,
                            double* out)
{
    if (kernelSize == 0 || n == 0) return;
    const std::ptrdiff_t offset =
        static_cast<std::ptrdiff_t>(kernelSize - 1) / 2;
    const double invK = 1.0 / static_cast<double>(kernelSize);
    const std::ptrdiff_t nSigned = static_cast<std::ptrdiff_t>(n);
    const std::ptrdiff_t K = static_cast<std::ptrdiff_t>(kernelSize);

    for (std::ptrdiff_t i = 0; i < nSigned; ++i) {
        const std::ptrdiff_t loRaw = i + offset - K + 1;
        const std::ptrdiff_t hiRaw = i + offset;
        const std::ptrdiff_t lo = std::max<std::ptrdiff_t>(0, loRaw);
        const std::ptrdiff_t hi = std::min<std::ptrdiff_t>(nSigned - 1, hiRaw);
        double s = 0.0;
        for (std::ptrdiff_t m = lo; m <= hi; ++m) {
            s += x[m] * invK;
        }
        out[static_cast<std::size_t>(i)] = s;
    }
}

// np.argmax of a f64 vector. First occurrence wins (strict `>`; matches
// numpy's `np.argmax` first-occurrence semantic).
inline std::size_t argMax(const double* x, std::size_t n)
{
    if (n == 0) return 0;
    std::size_t best = 0;
    double bestVal = x[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i] > bestVal) {
            bestVal = x[i];
            best = i;
        }
    }
    return best;
}

} // namespace

// ---------------------------------------------------------------------------
// windowOnsetIndex — splice.py:85-98
// ---------------------------------------------------------------------------

int Splice::windowOnsetIndex(const double* window,
                             std::size_t   nCh,
                             std::size_t   n)
{
    // Python: if window.ndim != 2 or window.shape[-1] < 16:
    //             return max(0, window.shape[-1] // 2)
    // `n // 2` in Python is integer floor-division. For n ≥ 0 this is n/2.
    if (nCh == 0 || n < 16) {
        return static_cast<int>(n / 2u);
    }

    // mono = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=0))
    // For each sample j ∈ [0, n): mono[j] = sqrt((sum_c window[c,j]^2) / nCh)
    std::vector<double> mono(n);
    const double invNCh = 1.0 / static_cast<double>(nCh);
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t c = 0; c < nCh; ++c) {
            const double v = window[c * n + j];
            s += v * v;
        }
        mono[j] = std::sqrt(s * invNCh);
    }

    // kernel_size = min(64, max(4, mono.shape[-1] // 8))
    const std::size_t kernelSize =
        std::min<std::size_t>(64u, std::max<std::size_t>(4u, n / 8u));

    // envelope = np.convolve(mono, np.ones(kernel_size)/kernel_size, mode='same')
    std::vector<double> envelope(n);
    boxConvolveSame(mono.data(), n, kernelSize, envelope.data());

    // diff = np.diff(envelope, prepend=envelope[:1])  → length n
    //   diff[0] = envelope[0] - envelope[0] = 0
    //   diff[i] = envelope[i] - envelope[i-1] for i ≥ 1
    std::vector<double> diffArr(n);
    diffArr[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        diffArr[i] = envelope[i] - envelope[i - 1];
    }

    // diff.size == 0 guard: unreachable here since n ≥ 16 (outer branch).
    // Preserved implicitly by the above loop-layout — no explicit check.

    // diff = np.maximum(diff, 0.0)
    for (std::size_t i = 0; i < n; ++i) {
        if (diffArr[i] < 0.0) diffArr[i] = 0.0;
    }

    // peak_idx = int(np.argmax(diff)); return int(np.clip(peak_idx, 0, n-1))
    // The clip is redundant (argmax output is already in [0, n-1]).
    const std::size_t peak = argMax(diffArr.data(), n);
    return static_cast<int>(peak);
}

// ---------------------------------------------------------------------------
// stereoWindowSimilarity — splice.py:108-149
// ---------------------------------------------------------------------------

double Splice::stereoWindowSimilarity(const double* windowA,
                                      const double* windowB,
                                      std::size_t   nCh,
                                      std::size_t   n)
{
    // Python: if shape mismatch OR ndim != 2 OR shape[-1] < 16: return -1.0
    // In C++ we already require matching (nCh, n) via the single-shape
    // parameter pair. nCh == 0 → degenerate (matches ndim != 2).
    if (nCh == 0 || n < 16) return -1.0;

    // taper = self._get_hanning(window_a.shape[-1])
    std::vector<double> taper;
    hanningWindow(n, taper);

    // Scratch per-channel buffers reused across iterations.
    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> da;
    std::vector<double> db;
    if (n > 1) {
        da.resize(n - 1);
        db.resize(n - 1);
    }

    std::vector<double> similarities;
    std::vector<double> derivativeSimilarities;
    similarities.reserve(nCh);
    derivativeSimilarities.reserve(nCh);

    for (std::size_t c = 0; c < nCh; ++c) {
        const double* aCh = windowA + c * n;
        const double* bCh = windowB + c * n;

        // a = window_a[ch].astype(np.float64)  → identity for f64 input
        // a = (a - np.mean(a)) * taper
        const double meanA = meanOf(aCh, n);
        const double meanB = meanOf(bCh, n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = (aCh[i] - meanA) * taper[i];
            b[i] = (bCh[i] - meanB) * taper[i];
        }

        // norm_a = float(np.linalg.norm(a)); norm_b similarly
        const double normA = l2norm(a.data(), n);
        const double normB = l2norm(b.data(), n);

        // if norm_a <= 1e-8 or norm_b <= 1e-8: continue
        if (normA <= 1e-8 || normB <= 1e-8) continue;

        // similarities.append(float(np.dot(a, b) / (norm_a * norm_b)))
        similarities.push_back(dotProduct(a.data(), b.data(), n) / (normA * normB));

        // if len(a) > 1: compute derivative NCC
        if (n > 1) {
            for (std::size_t i = 0; i + 1 < n; ++i) {
                da[i] = a[i + 1] - a[i];
                db[i] = b[i + 1] - b[i];
            }
            const std::size_t nd = n - 1;
            const double normDa = l2norm(da.data(), nd);
            const double normDb = l2norm(db.data(), nd);

            // if norm_da > 1e-8 and norm_db > 1e-8: append derivative sim
            if (normDa > 1e-8 && normDb > 1e-8) {
                derivativeSimilarities.push_back(
                    dotProduct(da.data(), db.data(), nd) / (normDa * normDb));
            }
        }
    }

    // if not similarities: return -1.0
    if (similarities.empty()) return -1.0;

    const double simMean = meanOfVec(similarities);

    // derivative_score = mean(derivative_similarities) if non-empty
    //                    else mean(similarities)  [fallback]
    const double derivativeScore = derivativeSimilarities.empty()
        ? simMean
        : meanOfVec(derivativeSimilarities);

    // return float(np.clip(0.8 * mean(similarities) + 0.2 * derivative_score, -1.0, 1.0))
    const double raw = 0.8 * simMean + 0.2 * derivativeScore;
    return std::max(-1.0, std::min(1.0, raw));
}

// ---------------------------------------------------------------------------
// scoreSplicePair — splice.py:151-196
// ---------------------------------------------------------------------------

SplicePairScore Splice::scoreSplicePair(const double*        outgoing,
                                        const double*        incoming,
                                        std::size_t          nCh,
                                        std::size_t          n,
                                        int                  outgoingShift,
                                        int                  incomingShift,
                                        int                  maxShiftSamples,
                                        const SpliceConfig&  cfg)
{
    const double similarity = stereoWindowSimilarity(outgoing, incoming, nCh, n);

    // Python: if similarity <= -0.99: return -1.0, similarity
    if (similarity <= -0.99) {
        return SplicePairScore{ -1.0, similarity };
    }

    // rms_out = float(np.sqrt(np.mean(outgoing_window.astype(np.float64) ** 2) + 1e-10))
    // np.mean() with no axis flattens: sum of all nCh*n squares / (nCh*n).
    // The +1e-10 is INSIDE the sqrt (matches Python L163).
    const std::size_t total = nCh * n;
    double sumOut = 0.0;
    double sumIn  = 0.0;
    for (std::size_t k = 0; k < total; ++k) {
        sumOut += outgoing[k] * outgoing[k];
        sumIn  += incoming[k] * incoming[k];
    }
    const double rmsOut = std::sqrt(sumOut / static_cast<double>(total) + 1e-10);
    const double rmsIn  = std::sqrt(sumIn  / static_cast<double>(total) + 1e-10);

    // energy_penalty = 0.0; if rms_out > 1e-8 and rms_in > 1e-8:
    //   db_diff = abs(20.0 * np.log10(rms_out / rms_in))
    //   energy_penalty = min(1.0, db_diff / 12.0)
    double energyPenalty = 0.0;
    if (rmsOut > 1e-8 && rmsIn > 1e-8) {
        const double dbDiff = std::abs(20.0 * std::log10(rmsOut / rmsIn));
        energyPenalty = std::min(1.0, dbDiff / 12.0);
    }

    // if max_shift_samples > 0:
    //   position_penalty = min(1.0, (|out_s| + |in_s|) / (2 * max_shift))
    // else:
    //   position_penalty = 0.0
    double positionPenalty = 0.0;
    if (maxShiftSamples > 0) {
        const double absSum =
            std::abs(static_cast<double>(outgoingShift)) +
            std::abs(static_cast<double>(incomingShift));
        positionPenalty = std::min(1.0,
            absSum / (2.0 * static_cast<double>(maxShiftSamples)));
    }

    // overlap = outgoing_window.shape[-1]  (= n)
    // center_idx = 0.5 * max(overlap - 1, 0)
    // Python `max(int, int)` returns int; 0.5 * int → float.
    const std::size_t overlap = n;
    const double centerIdx = 0.5 * static_cast<double>(overlap > 0u ? overlap - 1u : 0u);

    // out_onset = float(self._window_onset_index(outgoing_window))
    // in_onset  = float(self._window_onset_index(incoming_window))
    const double outOnset = static_cast<double>(windowOnsetIndex(outgoing, nCh, n));
    const double inOnset  = static_cast<double>(windowOnsetIndex(incoming, nCh, n));

    // center_scale = max(1.0, overlap * 0.35)
    // center_penalty = min(1.0,
    //     (|out_onset - center_idx| + |in_onset - center_idx|) / (2 * center_scale))
    const double centerScale =
        std::max(1.0, static_cast<double>(overlap) * 0.35);
    const double centerPenalty = std::min(1.0,
        (std::abs(outOnset - centerIdx) + std::abs(inOnset - centerIdx))
        / (2.0 * centerScale));

    // alignment_penalty = min(1.0, |out_onset - in_onset| / max(1.0, overlap * 0.25))
    const double alignmentDenominator =
        std::max(1.0, static_cast<double>(overlap) * 0.25);
    const double alignmentPenalty = std::min(1.0,
        std::abs(outOnset - inOnset) / alignmentDenominator);

    // score = similarity
    //       - 0.10 * energy_penalty
    //       - 0.15 * position_penalty
    //       - transient_center_penalty_weight    * center_penalty
    //       - transient_alignment_penalty_weight * alignment_penalty
    const double score = similarity
        - 0.10 * energyPenalty
        - 0.15 * positionPenalty
        - cfg.transientCenterPenaltyWeight    * centerPenalty
        - cfg.transientAlignmentPenaltyWeight * alignmentPenalty;

    return SplicePairScore{ score, similarity };
}

// ---------------------------------------------------------------------------
// findOnsetSample — splice.py:26-83
// ---------------------------------------------------------------------------
//
// Per ADR-033: two-convolve path drifts ±1-4 samples vs numpy because
// np.convolve(K=32) dispatches through BLAS np.dot. The C++ `boxConvolveSame`
// matches numpy at K ≤ ~16; at K=32 the drift is 1-2 ULP per output sample.
// Session-35 tests composition — if winner-beat-pair matches Python despite
// primitive drift, ADR-033 closes positively. No cache is maintained in C++:
// cache is a perf optimization that does not affect output bit-exactness.
int Splice::findOnsetSample(const double*       monoEnergy,
                            std::size_t         nMonoEnergy,
                            int                 sr,
                            std::int64_t        beatSample,
                            double              lookbackMs,
                            double              lookaheadMs,
                            const SpliceConfig& cfg)
{
    // Python: default-ms sentinel branch (use_default_cache = lookback is None
    // and lookahead is None) — we collapse the cache-hit fast path (pure perf)
    // but still respect the explicit-vs-default distinction for ms conversion.
    if (lookbackMs  < 0.0) lookbackMs  = cfg.onsetSearchLookbackMs;
    if (lookaheadMs < 0.0) lookaheadMs = cfg.onsetSearchLookaheadMs;

    // Python: `int(lookback_ms * self._sr / 1000)` — int() truncates toward
    // zero. For positive values this is floor. The cast below matches.
    const std::int64_t lookback  =
        static_cast<std::int64_t>(lookbackMs  * static_cast<double>(sr) / 1000.0);
    const std::int64_t lookahead =
        static_cast<std::int64_t>(lookaheadMs * static_cast<double>(sr) / 1000.0);

    const std::int64_t start =
        std::max<std::int64_t>(0, beatSample - lookback);
    const std::int64_t end = std::min<std::int64_t>(
        static_cast<std::int64_t>(nMonoEnergy), beatSample + lookahead);

    // Python: `chunk = mono[start:end]`. If end ≤ start the slice is empty
    // (length 0) and the next `if len(chunk) < 128` fires → return beat_sample.
    const std::size_t chunkLen =
        (end > start) ? static_cast<std::size_t>(end - start) : 0u;
    if (chunkLen < 128) {
        return static_cast<int>(beatSample);
    }

    // Python: kernel_size = min(64, len(chunk) // 2); if < 4 → return beat_sample.
    const std::size_t kernelSize =
        std::min<std::size_t>(64u, chunkLen / 2u);
    if (kernelSize < 4) {
        return static_cast<int>(beatSample);
    }

    // envelope = np.convolve(chunk, ones(K)/K, mode='same')
    std::vector<double> envelope(chunkLen);
    boxConvolveSame(monoEnergy + start, chunkLen, kernelSize, envelope.data());

    // diff = np.diff(envelope) — length chunkLen - 1
    // Python's explicit `if len(diff) < 16: return beat_sample` fires on
    // chunkLen ∈ [128, 16] conceptually but chunkLen ≥ 128 guaranteed → this
    // gate is effectively unreachable here; preserve the guard for parity.
    if (chunkLen < 2) {
        return static_cast<int>(beatSample);
    }
    std::vector<double> diffArr(chunkLen - 1);
    for (std::size_t i = 0; i < chunkLen - 1; ++i) {
        diffArr[i] = envelope[i + 1] - envelope[i];
    }
    if (diffArr.size() < 16) {
        return static_cast<int>(beatSample);
    }

    // smooth_size = min(32, len(diff) // 2); if ≥ 2: diff = convolve(diff, ones(K)/K, 'same')
    const std::size_t smoothSize =
        std::min<std::size_t>(32u, diffArr.size() / 2u);
    if (smoothSize >= 2) {
        std::vector<double> smoothed(diffArr.size());
        boxConvolveSame(diffArr.data(), diffArr.size(), smoothSize,
                        smoothed.data());
        diffArr = std::move(smoothed);
    }

    // peak_idx = int(np.argmax(diff)); return start + peak_idx
    const std::size_t peak = argMax(diffArr.data(), diffArr.size());
    return static_cast<int>(start + static_cast<std::int64_t>(peak));
}

// ---------------------------------------------------------------------------
// scoreAnchorAlignedPair — splice.py:198-245
// ---------------------------------------------------------------------------

namespace {

// Copy a channel-major slice `src[:, startCol:endCol]` into a contiguous
// channel-major buffer `dst` of shape (nCh, sliceLen). Used for window
// extraction before calling stereoWindowSimilarity on a sub-range.
inline void copyChannelMajorSlice(const double* src,
                                  std::size_t   nCh,
                                  std::size_t   rowLen,
                                  std::size_t   startCol,
                                  std::size_t   sliceLen,
                                  double*       dst)
{
    for (std::size_t c = 0; c < nCh; ++c) {
        std::memcpy(dst + c * sliceLen,
                    src + c * rowLen + startCol,
                    sliceLen * sizeof(double));
    }
}

// RMS across all channels of a contiguous (nCh, n) buffer, with +1e-10
// sentinel inside the sqrt (matches Python splice.py:163-164 / 220-221).
// Used by scoreSplicePair / scoreAnchorAlignedPair / refineTransitionSplice
// RMS energy branches. `sumSquares` is the running `sum(x**2)` over the
// flattened (nCh*n) buffer.
inline double rmsWithSentinel(double sumSquares, std::size_t total)
{
    return std::sqrt(sumSquares / static_cast<double>(total) + 1e-10);
}

} // namespace

AnchorScore Splice::scoreAnchorAlignedPair(const double*        outgoing,
                                           const double*        incoming,
                                           std::size_t          nCh,
                                           std::size_t          n,
                                           int                  anchorIndex,
                                           double               vocalPresence,
                                           int                  sr,
                                           const SpliceConfig&  cfg)
{
    // similarity = self._stereo_window_similarity(outgoing, incoming)
    const double similarity = stereoWindowSimilarity(outgoing, incoming, nCh, n);

    // Python: if similarity <= -0.99: return 0.0, 0.0, 0.0
    if (similarity <= -0.99) {
        return AnchorScore{ 0.0, 0.0, 0.0 };
    }

    // local_radius = int(round(anchor_local_window_ms * sr / 2000.0))
    // Python's `round` uses banker's (half-to-even) for floats — match with
    // std::nearbyint under default FE_TONEAREST rounding mode. The 2000.0
    // (not 1000) is a HALF-window convention: the local window spans
    // 2 * local_radius centered on the anchor.
    const double localRadiusRaw =
        cfg.anchorLocalWindowMs * static_cast<double>(sr) / 2000.0;
    int localRadius = static_cast<int>(std::nearbyint(localRadiusRaw));
    localRadius = std::max(16, localRadius);

    const int overlapInt = static_cast<int>(n);
    const int lo = std::max(0, anchorIndex - localRadius);
    const int hi = std::min(overlapInt, anchorIndex + localRadius);

    double localSimilarity = similarity;
    if (hi - lo >= 16) {
        const std::size_t sliceN = static_cast<std::size_t>(hi - lo);
        std::vector<double> aSlice(nCh * sliceN);
        std::vector<double> bSlice(nCh * sliceN);
        copyChannelMajorSlice(outgoing, nCh, n,
                              static_cast<std::size_t>(lo), sliceN,
                              aSlice.data());
        copyChannelMajorSlice(incoming, nCh, n,
                              static_cast<std::size_t>(lo), sliceN,
                              bSlice.data());
        localSimilarity = stereoWindowSimilarity(
            aSlice.data(), bSlice.data(), nCh, sliceN);
    }

    // RMS on flattened (nCh, n): sum over all entries.
    const std::size_t total = nCh * n;
    double sumOut = 0.0;
    double sumIn  = 0.0;
    for (std::size_t k = 0; k < total; ++k) {
        sumOut += outgoing[k] * outgoing[k];
        sumIn  += incoming[k] * incoming[k];
    }
    const double rmsOut = rmsWithSentinel(sumOut, total);
    const double rmsIn  = rmsWithSentinel(sumIn,  total);

    double energyPenalty = 0.0;
    if (rmsOut > 1e-8 && rmsIn > 1e-8) {
        const double dbDiff = std::abs(20.0 * std::log10(rmsOut / rmsIn));
        energyPenalty = std::min(1.0, dbDiff / 12.0);
    }

    // anchor_margin = min(anchor_index, overlap - anchor_index)
    // edge_penalty = max(0.0, 1.0 - anchor_margin / max(1.0, overlap * 0.18))
    double edgePenalty = 0.0;
    if (overlapInt > 0) {
        const int anchorMargin =
            std::min(anchorIndex, overlapInt - anchorIndex);
        const double edgeDenom =
            std::max(1.0, static_cast<double>(overlapInt) * 0.18);
        edgePenalty = std::max(0.0,
            1.0 - static_cast<double>(anchorMargin) / edgeDenom);
    }

    // local_weight = min(0.70, 0.40 + 0.25 * clip(vocal_presence, 0, 1))
    const double vpClipped = std::max(0.0, std::min(1.0, vocalPresence));
    const double localWeight = std::min(0.70, 0.40 + 0.25 * vpClipped);
    const double overallWeight = 1.0 - localWeight;

    // Two independently-clipped 0..1 quality signals.
    const double sim01   = std::max(0.0, std::min(1.0, (similarity      + 1.0) * 0.5));
    const double local01 = std::max(0.0, std::min(1.0, (localSimilarity + 1.0) * 0.5));

    const double quality =
          overallWeight * sim01
        + localWeight   * local01
        - 0.08 * energyPenalty
        - 0.04 * edgePenalty;

    return AnchorScore{
        std::max(0.0, std::min(1.0, quality)),
        sim01,
        local01,
    };
}

// ---------------------------------------------------------------------------
// searchAnchorTransitionGeometry — splice.py:247-330
// ---------------------------------------------------------------------------

AnchorSearchResult Splice::searchAnchorTransitionGeometry(
    const SpliceContext&    ctx,
    int                     prevBeat,
    int                     currBeat,
    const TransitionMeta&   meta,
    const SpliceConfig&     cfg)
{
    AnchorSearchResult out;

    // search_beats = max(0, int(anchor_search_beats))
    const int searchBeats = std::max(0, cfg.anchorSearchBeats);
    if (searchBeats <= 0) return out;
    const int extensionBeats = std::max(1, cfg.anchorSearchMaxExtensionBeats);

    // min_context = int(round(anchor_min_context_ms * sr / 1000.0))
    const std::int64_t minContext = static_cast<std::int64_t>(std::nearbyint(
        cfg.anchorMinContextMs * static_cast<double>(ctx.sr) / 1000.0));

    const double vocalPresence = meta.vocalPresenceLevel;

    double bestScore = -1.0;
    const int nBeatsInt = static_cast<int>(ctx.nBeats);

    // Beat-range bounds (inclusive both ends, matching Python range(lo, hi+1)).
    const int outAnchorLo        = std::max(0,             prevBeat - searchBeats);
    const int outAnchorHi        = std::min(nBeatsInt - 1, prevBeat);
    const int inAnchorLo         = std::max(0,             currBeat);
    const int inAnchorHi         = std::min(nBeatsInt - 1, currBeat + searchBeats);
    const int incomingStartLo    = std::max(0,             currBeat - searchBeats);
    const int incomingStartHi    = std::min(nBeatsInt - 1, currBeat);
    const int outgoingBoundaryLo = std::max(1,             prevBeat + 1);
    const int outgoingBoundaryHi = std::min(nBeatsInt - 1, prevBeat + extensionBeats);

    // Scratch buffers (reused across iterations).
    std::vector<double> outWin;
    std::vector<double> inWin;

    for (int outAnchorBeat = outAnchorLo;
         outAnchorBeat <= outAnchorHi; ++outAnchorBeat)
    {
        const std::int64_t anchorOutSample = findOnsetSample(
            ctx.monoEnergy, ctx.nMonoEnergy, ctx.sr,
            ctx.beatSamples[outAnchorBeat], -1.0, -1.0, cfg);

        for (int inAnchorBeat = inAnchorLo;
             inAnchorBeat <= inAnchorHi; ++inAnchorBeat)
        {
            const std::int64_t anchorInSample = findOnsetSample(
                ctx.monoEnergy, ctx.nMonoEnergy, ctx.sr,
                ctx.beatSamples[inAnchorBeat], -1.0, -1.0, cfg);

            for (int incomingStartBeat = incomingStartLo;
                 incomingStartBeat <= incomingStartHi; ++incomingStartBeat)
            {
                const std::int64_t incomingStartSample =
                    ctx.beatSamples[incomingStartBeat];
                const std::int64_t preRoll =
                    anchorInSample - incomingStartSample;
                if (preRoll < minContext) continue;

                for (int outgoingBoundaryBeat = outgoingBoundaryLo;
                     outgoingBoundaryBeat <= outgoingBoundaryHi;
                     ++outgoingBoundaryBeat)
                {
                    const std::int64_t outgoingCutSample =
                        ctx.beatSamples[outgoingBoundaryBeat];
                    const std::int64_t postRoll =
                        outgoingCutSample - anchorOutSample;
                    if (postRoll < minContext) continue;

                    const std::int64_t overlap = preRoll + postRoll;
                    const std::int64_t outgoingStartSample =
                        anchorOutSample - preRoll;
                    const std::int64_t incomingEndSample =
                        anchorInSample + postRoll;

                    if (overlap < minContext * 2
                        || outgoingStartSample < 0
                        || incomingEndSample
                            > static_cast<std::int64_t>(ctx.nAudio))
                    {
                        continue;
                    }

                    const std::size_t olap =
                        static_cast<std::size_t>(overlap);

                    // outgoing_window = audio[..., outgoing_start:outgoing_cut]
                    // incoming_window = audio[..., incoming_start:incoming_end]
                    outWin.resize(ctx.nChAudio * olap);
                    inWin.resize(ctx.nChAudio * olap);
                    for (std::size_t c = 0; c < ctx.nChAudio; ++c) {
                        std::memcpy(
                            outWin.data() + c * olap,
                            ctx.audio + c * ctx.nAudio + outgoingStartSample,
                            olap * sizeof(double));
                        std::memcpy(
                            inWin.data() + c * olap,
                            ctx.audio + c * ctx.nAudio + incomingStartSample,
                            olap * sizeof(double));
                    }

                    // Python's shape-guard (L299-303) is tautological given
                    // our bounds checks, but preserving it is cheap.
                    // (We trust the bounds checks — the C++ memcpy above has
                    // already satisfied the shape invariant.)

                    const AnchorScore sc = scoreAnchorAlignedPair(
                        outWin.data(), inWin.data(),
                        ctx.nChAudio, olap,
                        static_cast<int>(preRoll), vocalPresence, ctx.sr, cfg);

                    // Python: `if score <= best_score: continue`
                    if (sc.quality <= bestScore) continue;

                    bestScore = sc.quality;
                    out.selected               = true;
                    out.score                  = sc.quality;
                    out.similarity01           = sc.similarity01;
                    out.localSimilarity01      = sc.localSimilarity01;
                    out.outAnchorBeat          = outAnchorBeat;
                    out.inAnchorBeat           = inAnchorBeat;
                    out.incomingStartBeat      = incomingStartBeat;
                    out.outgoingBoundaryBeat   = outgoingBoundaryBeat;
                    out.anchorOutSec =
                        static_cast<double>(anchorOutSample)
                        / static_cast<double>(ctx.sr);
                    out.anchorInSec =
                        static_cast<double>(anchorInSample)
                        / static_cast<double>(ctx.sr);
                    out.outgoingStartSample    = outgoingStartSample;
                    out.outgoingCutSample      = outgoingCutSample;
                    out.incomingCutSample      = incomingStartSample;
                    out.incomingEndSample      = incomingEndSample;
                    out.anchorOverlapSamples   = overlap;
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// transitionOverlapSamples — splice.py:332-355
// ---------------------------------------------------------------------------

int Splice::transitionOverlapSamples(int                    crossfadeSamples,
                                     int                    sr,
                                     const TransitionMeta&  meta,
                                     const SpliceConfig&    cfg)
{
    int baseOverlap = crossfadeSamples;
    if (meta.preferredOverlapSec > 0.0) {
        const int preferredOverlap = static_cast<int>(std::nearbyint(
            meta.preferredOverlapSec * static_cast<double>(sr)));
        baseOverlap = std::max(baseOverlap, preferredOverlap);
    }

    if (meta.vocalPresenceLevel < cfg.vocalActivityThreshold) {
        return baseOverlap;
    }

    double targetMs;
    if (meta.labelMatch < 0.5) {
        targetMs = cfg.vocalCrossfadeMs;
    } else if (std::min(meta.vocalEntrySupport, meta.vocalExitSupport) < 0.85) {
        targetMs = cfg.vocalSameLabelCrossfadeMs;
    } else {
        return baseOverlap;
    }

    const int targetOverlap = static_cast<int>(std::nearbyint(
        targetMs * static_cast<double>(sr) / 1000.0));
    return std::max(baseOverlap, targetOverlap);
}

// ---------------------------------------------------------------------------
// refineTransitionSplice — splice.py:357-514
// ---------------------------------------------------------------------------

RefineResult Splice::refineTransitionSplice(const SpliceContext&   ctx,
                                            std::int64_t           successorSample,
                                            std::int64_t           targetSample,
                                            std::int64_t           beatEnd,
                                            const TransitionMeta&  meta,
                                            int                    overlapSamplesOrNeg,
                                            const SpliceConfig&    cfg)
{
    RefineResult result;

    const std::int64_t overlap = (overlapSamplesOrNeg < 0)
        ? static_cast<std::int64_t>(ctx.crossfadeSamples)
        : static_cast<std::int64_t>(overlapSamplesOrNeg);
    const std::int64_t halfXfade = overlap / 2;
    if (overlap <= 16 || halfXfade <= 0) return result;

    // alignment_offset_samples = int(round(alignment_offset_sec * sr))
    const std::int64_t alignmentOffsetSamples =
        static_cast<std::int64_t>(std::nearbyint(
            meta.alignmentOffsetSec * static_cast<double>(ctx.sr)));

    // aligned_target_sample = int(np.clip(target + offset, 0, audio.shape[-1] - 1))
    const std::int64_t audioLen = static_cast<std::int64_t>(ctx.nAudio);
    const std::int64_t alignedTargetSample = std::max<std::int64_t>(0,
        std::min<std::int64_t>(audioLen - 1,
                                targetSample + alignmentOffsetSamples));

    const std::int64_t outgoingOnset = findOnsetSample(
        ctx.monoEnergy, ctx.nMonoEnergy, ctx.sr,
        successorSample, -1.0, -1.0, cfg);
    const std::int64_t incomingOnset = findOnsetSample(
        ctx.monoEnergy, ctx.nMonoEnergy, ctx.sr,
        alignedTargetSample, -1.0, -1.0, cfg);

    const std::int64_t nominalOutCut  = outgoingOnset + halfXfade;
    const std::int64_t nominalInStart = incomingOnset - halfXfade;

    // max_shift_samples = int(stereo_refine_max_shift_ms * sr / 1000.0)
    std::int64_t maxShiftSamples = static_cast<std::int64_t>(
        cfg.stereoRefineMaxShiftMs * static_cast<double>(ctx.sr) / 1000.0);
    maxShiftSamples = std::max<std::int64_t>(0,
        std::min<std::int64_t>(maxShiftSamples,
                                std::max<std::int64_t>(halfXfade - 1, 0)));
    if (maxShiftSamples <= 0) return result;

    const std::int64_t outMin = std::max<std::int64_t>(
        overlap, nominalOutCut - maxShiftSamples);
    const std::int64_t outMax = std::min<std::int64_t>(
        { audioLen, successorSample + overlap,
          nominalOutCut + maxShiftSamples });
    const std::int64_t inMin = std::max<std::int64_t>(
        0, nominalInStart - maxShiftSamples);
    const std::int64_t inMax = std::min<std::int64_t>(
        { audioLen - overlap, beatEnd - overlap,
          nominalInStart + maxShiftSamples });
    if (outMin >= outMax || inMin >= inMax) return result;

    // coarse_step = max(4, int(stereo_refine_coarse_step_samples))
    //             = min(coarse_step, max_shift_samples) if max_shift > 0
    std::int64_t coarseStep = std::max<std::int64_t>(4,
        static_cast<std::int64_t>(cfg.stereoRefineCoarseStepSamples));
    if (maxShiftSamples > 0) {
        coarseStep = std::min(coarseStep, maxShiftSamples);
    }

    std::int64_t bestOutCut = std::max<std::int64_t>(outMin,
        std::min<std::int64_t>(outMax, nominalOutCut));
    std::int64_t bestInStart = std::max<std::int64_t>(inMin,
        std::min<std::int64_t>(inMax, nominalInStart));
    double bestScore      = -1.0;
    double bestSimilarity = -1.0;

    // Hanning taper pre-computed once for the overlap size (splice.py:411).
    std::vector<double> taper;
    hanningWindow(static_cast<std::size_t>(overlap), taper);

    const std::size_t olap = static_cast<std::size_t>(overlap);

    // --- _fast_similarity(out_cut, in_start) ---------------------------------
    // Sum channels to mono, apply taper, compute plain NCC. Used ONLY in the
    // coarse stage; the fine + top-8 stages use the full stereoWindowSimilarity.
    auto fastSimilarity = [&](std::int64_t outCut, std::int64_t inStart) -> double {
        const std::int64_t aStart = outCut - overlap;
        if (aStart < 0 || outCut > audioLen
            || inStart < 0 || inStart + overlap > audioLen) {
            return -1.0;
        }
        // ma = a.sum(axis=0).astype(f64); mb = b.sum(axis=0).astype(f64)
        std::vector<double> ma(olap, 0.0);
        std::vector<double> mb(olap, 0.0);
        for (std::size_t c = 0; c < ctx.nChAudio; ++c) {
            const double* aRow = ctx.audio + c * ctx.nAudio + aStart;
            const double* bRow = ctx.audio + c * ctx.nAudio + inStart;
            for (std::size_t i = 0; i < olap; ++i) {
                ma[i] += aRow[i];
                mb[i] += bRow[i];
            }
        }
        const double mam = meanOf(ma.data(), olap);
        const double mbm = meanOf(mb.data(), olap);
        for (std::size_t i = 0; i < olap; ++i) {
            ma[i] = (ma[i] - mam) * taper[i];
            mb[i] = (mb[i] - mbm) * taper[i];
        }
        const double na = l2norm(ma.data(), olap);
        const double nb = l2norm(mb.data(), olap);
        if (na < 1e-8 || nb < 1e-8) return -1.0;
        return dotProduct(ma.data(), mb.data(), olap) / (na * nb);
    };

    // --- evaluate_pair_full(out_cut, in_start) -------------------------------
    std::unordered_map<std::int64_t, double> onsetCacheOut;
    std::unordered_map<std::int64_t, double> onsetCacheIn;
    std::vector<double> outWin(ctx.nChAudio * olap);
    std::vector<double> inWin(ctx.nChAudio * olap);

    auto evaluatePairFull = [&](std::int64_t outCut, std::int64_t inStart) {
        const std::int64_t aStart = outCut - overlap;
        if (aStart < 0 || outCut > audioLen
            || inStart < 0 || inStart + overlap > audioLen) {
            return;
        }
        for (std::size_t c = 0; c < ctx.nChAudio; ++c) {
            std::memcpy(outWin.data() + c * olap,
                        ctx.audio + c * ctx.nAudio + aStart,
                        olap * sizeof(double));
            std::memcpy(inWin.data() + c * olap,
                        ctx.audio + c * ctx.nAudio + inStart,
                        olap * sizeof(double));
        }
        const double similarity = stereoWindowSimilarity(
            outWin.data(), inWin.data(), ctx.nChAudio, olap);
        if (similarity <= -0.99) return;

        const std::size_t total = ctx.nChAudio * olap;
        double sumOut = 0.0;
        double sumIn  = 0.0;
        for (std::size_t k = 0; k < total; ++k) {
            sumOut += outWin[k] * outWin[k];
            sumIn  += inWin[k]  * inWin[k];
        }
        const double rmsOut = rmsWithSentinel(sumOut, total);
        const double rmsIn  = rmsWithSentinel(sumIn,  total);

        double energyPenalty = 0.0;
        if (rmsOut > 1e-8 && rmsIn > 1e-8) {
            const double dbDiff =
                std::abs(20.0 * std::log10(rmsOut / rmsIn));
            energyPenalty = std::min(1.0, dbDiff / 12.0);
        }

        double positionPenalty = 0.0;
        if (maxShiftSamples > 0) {
            const double absSum =
                std::abs(static_cast<double>(outCut  - nominalOutCut)) +
                std::abs(static_cast<double>(inStart - nominalInStart));
            positionPenalty = std::min(1.0,
                absSum / (2.0 * static_cast<double>(maxShiftSamples)));
        }

        auto itOut = onsetCacheOut.find(outCut);
        if (itOut == onsetCacheOut.end()) {
            const double v = static_cast<double>(
                windowOnsetIndex(outWin.data(), ctx.nChAudio, olap));
            itOut = onsetCacheOut.emplace(outCut, v).first;
        }
        auto itIn = onsetCacheIn.find(inStart);
        if (itIn == onsetCacheIn.end()) {
            const double v = static_cast<double>(
                windowOnsetIndex(inWin.data(), ctx.nChAudio, olap));
            itIn = onsetCacheIn.emplace(inStart, v).first;
        }
        const double outOnset = itOut->second;
        const double inOnset  = itIn->second;

        const double centerIdx =
            0.5 * static_cast<double>(overlap > 0 ? overlap - 1 : 0);
        const double centerScale =
            std::max(1.0, static_cast<double>(overlap) * 0.35);
        const double centerPenalty = std::min(1.0,
            (std::abs(outOnset - centerIdx) + std::abs(inOnset - centerIdx))
            / (2.0 * centerScale));
        const double alignmentDenom =
            std::max(1.0, static_cast<double>(overlap) * 0.25);
        const double alignmentPenalty = std::min(1.0,
            std::abs(outOnset - inOnset) / alignmentDenom);

        const double score = similarity
            - 0.10 * energyPenalty
            - 0.15 * positionPenalty
            - cfg.transientCenterPenaltyWeight    * centerPenalty
            - cfg.transientAlignmentPenaltyWeight * alignmentPenalty;

        if (score > bestScore) {
            bestScore      = score;
            bestSimilarity = similarity;
            bestOutCut     = outCut;
            bestInStart    = inStart;
        }
    };

    // --- Coarse grid: fast mono correlation ----------------------------------
    struct CoarseEntry {
        double score;
        std::int64_t outCut;
        std::int64_t inStart;
    };
    std::vector<CoarseEntry> coarseScores;

    // Python: `out_positions = range(out_min, out_max + 1, max(1, coarse_step))`
    const std::int64_t stepStride = std::max<std::int64_t>(1, coarseStep);
    for (std::int64_t outCut = outMin; outCut <= outMax; outCut += stepStride) {
        for (std::int64_t inStart = inMin; inStart <= inMax;
             inStart += stepStride)
        {
            const double s = fastSimilarity(outCut, inStart);
            if (s > -0.5) {
                coarseScores.push_back({ s, outCut, inStart });
            }
        }
    }

    // Python: `coarse_scores.sort(key=lambda x: -x[0])`. Timsort is stable —
    // for tied scores the original insertion order wins. std::stable_sort
    // matches (std::sort would be arbitrary on ties).
    std::stable_sort(coarseScores.begin(), coarseScores.end(),
        [](const CoarseEntry& a, const CoarseEntry& b) {
            return a.score > b.score;
        });

    // Full eval on top-8 coarse candidates.
    const std::size_t topK = std::min<std::size_t>(8u, coarseScores.size());
    for (std::size_t i = 0; i < topK; ++i) {
        evaluatePairFull(coarseScores[i].outCut, coarseScores[i].inStart);
    }

    // --- Fine radius exhaustive search ---------------------------------------
    const std::int64_t fineRadius = std::max<std::int64_t>(1, coarseStep / 8);
    const std::int64_t fineOutMin = std::max(outMin, bestOutCut  - fineRadius);
    const std::int64_t fineOutMax = std::min(outMax, bestOutCut  + fineRadius);
    const std::int64_t fineInMin  = std::max(inMin,  bestInStart - fineRadius);
    const std::int64_t fineInMax  = std::min(inMax,  bestInStart + fineRadius);
    for (std::int64_t outCut = fineOutMin; outCut <= fineOutMax; ++outCut) {
        for (std::int64_t inStart = fineInMin;
             inStart <= fineInMax; ++inStart)
        {
            evaluatePairFull(outCut, inStart);
        }
    }

    if (bestScore < -0.5) return result;

    result.found             = true;
    result.outgoingCutSample = bestOutCut;
    result.incomingCutSample = bestInStart;
    result.outgoingShiftSec  = static_cast<double>(bestOutCut  - nominalOutCut)
                               / static_cast<double>(ctx.sr);
    result.incomingShiftSec  = static_cast<double>(bestInStart - nominalInStart)
                               / static_cast<double>(ctx.sr);
    result.stereoSimilarity01 = std::max(0.0,
        std::min(1.0, (bestSimilarity + 1.0) * 0.5));
    result.spliceScore = bestScore;
    return result;
}

} // namespace reamix::render
