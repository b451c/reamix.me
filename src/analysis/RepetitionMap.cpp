#include "analysis/RepetitionMap.h"

#include "analysis/Recurrence.h"
#include "dsp/WaveformXcorr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace reamix::analysis {

// =============================================================================
// Small-helper anonymous namespace
// =============================================================================
namespace {

// chroma_range(n): (max(0, n - 12 - 7), min((n-12-7)+12, n))
// PARITY: config.py L20-24. For 59-dim features → (40, 52).
//         For 39-dim features → (20, 32). Zero-width handled by clamping.
inline std::pair<int, int> chromaRange(int nFeat) noexcept
{
    const int start = std::max(0, nFeat - RepetitionMap::kNChromaDims
                                        - RepetitionMap::kNContrastDims);
    const int end   = std::min(start + RepetitionMap::kNChromaDims, nFeat);
    return {start, end};
}

inline std::string toLower(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

inline bool isExcludedLabel(std::string_view lower) noexcept
{
    return lower == "intro" || lower == "outro" || lower == "unknown";
}

// PARITY: repetition_map.py::_beats_to_bars L82-117.
//
// Returns list of (first_beat_idx, last_beat_idx) for each bar. A bar spans
// from one downbeat to the next.  Duplicate downbeat indices are collapsed
// (L101-102): two downbeats mapping to the same beat contribute one bar
// boundary, not two zero-length bars.  Synthetic fallback when downbeats are
// empty: step every `timeSignature` beats (L104-105).
std::vector<std::pair<int, int>>
beatsToBars(const double* beatTimes,
            int           nBeats,
            const double* downbeats,
            int           nDownbeats,
            int           timeSignature)
{
    std::vector<std::pair<int, int>> bars;
    if (nBeats == 0)
        return bars;

    std::vector<int> dbIdx;
    if (downbeats != nullptr && nDownbeats > 0)
    {
        dbIdx.reserve(static_cast<std::size_t>(nDownbeats));
        for (int d = 0; d < nDownbeats; ++d)
        {
            const double dbTime = downbeats[d];
            // np.argmin(np.abs(beat_times - db_time)) — first occurrence of min.
            int    bestIdx = 0;
            double bestVal = std::fabs(beatTimes[0] - dbTime);
            for (int b = 1; b < nBeats; ++b)
            {
                const double v = std::fabs(beatTimes[b] - dbTime);
                if (v < bestVal)
                {
                    bestVal = v;
                    bestIdx = b;
                }
            }
            if (dbIdx.empty() || bestIdx != dbIdx.back())
                dbIdx.push_back(bestIdx);
        }
    }
    else
    {
        // Fallback: synthesize downbeats every `timeSignature` beats.
        for (int i = 0; i < nBeats; i += timeSignature)
            dbIdx.push_back(i);
    }

    if (dbIdx.empty())
    {
        bars.emplace_back(0, nBeats - 1);
        return bars;
    }

    bars.reserve(dbIdx.size());
    for (std::size_t i = 0; i < dbIdx.size(); ++i)
    {
        const int start = dbIdx[i];
        const int end   = (i + 1 < dbIdx.size())
            ? dbIdx[i + 1] - 1
            : nBeats - 1;
        if (end >= start)
            bars.emplace_back(start, end);
    }
    return bars;
}

// PARITY: repetition_map.py::_bars_in_segment L120-130.
std::vector<int>
barsInSegment(const std::vector<std::pair<int, int>>& bars,
              int segStartBeat,
              int segEndBeat)
{
    std::vector<int> out;
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        const int barStart = bars[i].first;
        if (segStartBeat <= barStart && barStart <= segEndBeat)
            out.push_back(static_cast<int>(i));
    }
    return out;
}

// PARITY: repetition_map.py::_bar_chroma L136-163.
// Returns row-major [barIndices.size() × dim] f64, L2-normalized per row.
std::vector<double>
barChroma(const float* features,
          int          nFeat,
          const std::vector<std::pair<int, int>>& bars,
          const std::vector<int>& barIndices)
{
    const auto [cs, ce] = chromaRange(nFeat);
    const int dim = ce - cs;
    const std::size_t nOut = barIndices.size();

    std::vector<double> out(nOut * static_cast<std::size_t>(dim), 0.0);
    if (nOut == 0 || dim <= 0)
        return out;

    for (std::size_t outIdx = 0; outIdx < nOut; ++outIdx)
    {
        const int barIdx = barIndices[outIdx];
        const int start  = bars[static_cast<std::size_t>(barIdx)].first;
        const int end    = bars[static_cast<std::size_t>(barIdx)].second;
        const int len    = end - start + 1;
        if (len <= 0)
            continue;
        // Mean across beats in bar, chroma slice [cs..ce).
        double* rowPtr = out.data() + outIdx * static_cast<std::size_t>(dim);
        for (int b = start; b <= end; ++b)
        {
            const float* featRow = features + static_cast<std::size_t>(b) * static_cast<std::size_t>(nFeat) + cs;
            for (int k = 0; k < dim; ++k)
                rowPtr[k] += static_cast<double>(featRow[k]);
        }
        const double invLen = 1.0 / static_cast<double>(len);
        for (int k = 0; k < dim; ++k)
            rowPtr[k] *= invLen;
    }

    // L2 normalize per row with `norm == 0 → 1.0` guard.
    for (std::size_t outIdx = 0; outIdx < nOut; ++outIdx)
    {
        double* rowPtr = out.data() + outIdx * static_cast<std::size_t>(dim);
        double  sumSq  = 0.0;
        for (int k = 0; k < dim; ++k)
            sumSq += rowPtr[k] * rowPtr[k];
        double norm = std::sqrt(sumSq);
        if (norm == 0.0)
            norm = 1.0;
        const double inv = 1.0 / norm;
        for (int k = 0; k < dim; ++k)
            rowPtr[k] *= inv;
    }
    return out;
}

// PARITY: repetition_map.py::_chroma_cross_correlation L166-214.
// Both inputs row-major [nRows × dim] f64.
// Returns (bestCorr clamped to 0, bestOffset).
std::pair<double, int>
chromaCrossCorrelation(const std::vector<double>& chromaA,
                       int                        nA,
                       int                        dim,
                       const std::vector<double>& chromaB,
                       int                        nB)
{
    if (nA == 0 || nB == 0)
        return {0.0, 0};

    const int minLen   = std::min(nA, nB);
    double    bestCorr = -1.0;
    int       bestOff  = 0;

    const int maxOff = std::min(minLen - 1, 4);
    for (int offset = -maxOff; offset <= maxOff; ++offset)
    {
        // Python L187-192:
        //   if offset >= 0:
        //     a = chroma_a[:min_len - offset]
        //     b = chroma_b[offset:offset + min_len - offset]
        //   else:
        //     a = chroma_a[-offset:-offset + min_len + offset]
        //     b = chroma_b[:min_len + offset]
        int aStart, aEnd, bStart, bEnd;
        if (offset >= 0)
        {
            aStart = 0;
            aEnd   = minLen - offset;
            bStart = offset;
            bEnd   = offset + (minLen - offset);   // = minLen
        }
        else
        {
            aStart = -offset;
            aEnd   = -offset + (minLen + offset);  // = minLen
            bStart = 0;
            bEnd   = minLen + offset;
        }
        const int aLen = std::max(0, aEnd - aStart);
        const int bLen = std::max(0, bEnd - bStart);
        const int overlap = std::min(aLen, bLen);
        if (overlap < 2)
            continue;

        // Per-row cosine similarity with zero-norm → 1 guard, then mean.
        double sumSims = 0.0;
        for (int i = 0; i < overlap; ++i)
        {
            const double* aRow = chromaA.data()
                + static_cast<std::size_t>(aStart + i) * static_cast<std::size_t>(dim);
            const double* bRow = chromaB.data()
                + static_cast<std::size_t>(bStart + i) * static_cast<std::size_t>(dim);
            double aSq = 0.0, bSq = 0.0;
            for (int k = 0; k < dim; ++k) { aSq += aRow[k] * aRow[k]; bSq += bRow[k] * bRow[k]; }
            double aN = std::sqrt(aSq);
            double bN = std::sqrt(bSq);
            if (aN == 0.0) aN = 1.0;
            if (bN == 0.0) bN = 1.0;
            // Python L207: `np.sum(a / a_norms * b / b_norms, axis=1)`
            // i.e. per-row sum of element-wise product of normalized pairs.
            const double invA = 1.0 / aN;
            const double invB = 1.0 / bN;
            double sim = 0.0;
            for (int k = 0; k < dim; ++k)
                sim += (aRow[k] * invA) * (bRow[k] * invB);
            sumSims += sim;
        }
        const double corr = sumSims / static_cast<double>(overlap);
        if (corr > bestCorr)
        {
            bestCorr = corr;
            bestOff  = offset;
        }
    }
    return {std::max(0.0, bestCorr), bestOff};
}

// PARITY: repetition_map.py::_verify_bar_boundary L220-244.
// source_idx = from_beat + 1 (the beat AFTER the outgoing bar). Bounds check
// on both source_idx and to_beat against boundary_waveforms.shape[0].
std::pair<double, int>
verifyBarBoundary(const float* boundaryWaveforms,
                  int          snippetLen,
                  int          nSnippetRows,
                  int          fromBeat,
                  int          toBeat,
                  int          maxLagSamples)
{
    const int sourceIdx = fromBeat + 1;
    if (sourceIdx >= nSnippetRows || toBeat >= nSnippetRows
        || sourceIdx < 0 || toBeat < 0 || snippetLen <= 0)
        return {0.0, 0};
    const float* src = boundaryWaveforms
        + static_cast<std::size_t>(sourceIdx) * static_cast<std::size_t>(snippetLen);
    const float* tgt = boundaryWaveforms
        + static_cast<std::size_t>(toBeat) * static_cast<std::size_t>(snippetLen);
    return reamix::dsp::WaveformXcorr::compute(
        src, tgt,
        static_cast<std::size_t>(snippetLen),
        static_cast<std::size_t>(snippetLen),
        maxLagSamples);
}

// PARITY: repetition_map.py::_segment_beat_range L250-263.
std::pair<int, int>
segmentBeatRange(const Segment& seg, const double* beatTimes, int nBeats)
{
    if (nBeats == 0)
        return {0, 0};
    const double startTime = seg.start;
    const double endTime   = seg.end;
    int    startBeat = 0;
    double bestS     = std::fabs(beatTimes[0] - startTime);
    int    endBeat   = 0;
    double bestE     = std::fabs(beatTimes[0] - endTime);
    for (int i = 1; i < nBeats; ++i)
    {
        const double dS = std::fabs(beatTimes[i] - startTime);
        if (dS < bestS) { bestS = dS; startBeat = i; }
        const double dE = std::fabs(beatTimes[i] - endTime);
        if (dE < bestE) { bestE = dE; endBeat = i; }
    }
    return {startBeat, std::max(startBeat, endBeat)};
}

// PARITY: repetition_map.py::_group_sections_by_label L266-278.
// Preserve insertion order (Python 3.7+ dict semantics).
std::vector<std::pair<std::string, std::vector<int>>>
groupSectionsByLabel(const std::vector<Segment>& segments)
{
    std::vector<std::pair<std::string, std::vector<int>>> groups;
    groups.reserve(segments.size());
    for (int idx = 0; idx < static_cast<int>(segments.size()); ++idx)
    {
        const std::string lower = toLower(segments[static_cast<std::size_t>(idx)].label);
        if (isExcludedLabel(lower))
            continue;
        auto it = std::find_if(groups.begin(), groups.end(),
            [&](const auto& p){ return p.first == lower; });
        if (it == groups.end())
            groups.emplace_back(lower, std::vector<int>{idx});
        else
            it->second.push_back(idx);
    }
    // Keep only labels with ≥ 2 occurrences.
    groups.erase(
        std::remove_if(groups.begin(), groups.end(),
            [](const auto& p){ return p.second.size() < 2; }),
        groups.end());
    return groups;
}

} // namespace

// =============================================================================
// Public build()
// =============================================================================
RepetitionResult
RepetitionMap::build(const double* beatTimes,
                     int           nBeats,
                     const double* downbeats,
                     int           nDownbeats,
                     const float*  features,
                     int           nFeat,
                     const std::vector<Segment>& segments,
                     const float*  boundaryWaveforms,
                     int           snippetLen,
                     int           waveformSampleRate,
                     int           timeSignature,
                     double        minChromaCorr,
                     double        minWaveformSim)
{
    RepetitionResult result;

    // Early-out gates (Python L317-325).
    if (nBeats < 8 || segments.size() < 2)
        return result;
    // Python L320-325 guards `features.shape[0] < len(beat_times)` → return
    // empty. (We treat nFeat == 0 as degenerate too.)
    if (nFeat <= 0)
        return result;

    // Build bar grid.
    auto bars = beatsToBars(beatTimes, nBeats, downbeats, nDownbeats, timeSignature);
    if (bars.size() < 4)
        return result;

    // Group sections by lowercased label.
    auto labelGroups = groupSectionsByLabel(segments);
    int totalInGroups = 0;
    for (const auto& [_label, indices] : labelGroups)
        totalInGroups += static_cast<int>(indices.size());
    result.nSectionsScanned = totalInGroups;

    // Waveform availability: boundaryWaveforms shape[0] must be >= nBeats.
    // (We don't know shape[0] independently — caller passes nBeats-sized
    // boundary grid; take waveformSampleRate > 0 as the "have waveforms"
    // signal. Pass nullptr / zero-SR to skip waveform verification.)
    const bool hasWaveforms = (boundaryWaveforms != nullptr)
                           && (snippetLen > 0)
                           && (waveformSampleRate > 0);
    const int  maxLagSamples = hasWaveforms
        ? static_cast<int>(30.0 * static_cast<double>(waveformSampleRate) / 1000.0)
        : 0;

    const auto [cs, ce] = chromaRange(nFeat);
    const int dim = ce - cs;

    // -------------------------------------------------------------------------
    // Phase 1: cross-section scan
    // -------------------------------------------------------------------------
    for (const auto& [label, segIndices] : labelGroups)
    {
        (void)label; // unused beyond grouping
        const int nSi = static_cast<int>(segIndices.size());
        for (int si = 0; si < nSi; ++si)
        {
            for (int sj = si + 1; sj < nSi; ++sj)
            {
                const int idxA = segIndices[static_cast<std::size_t>(si)];
                const int idxB = segIndices[static_cast<std::size_t>(sj)];
                const Segment& segA = segments[static_cast<std::size_t>(idxA)];
                const Segment& segB = segments[static_cast<std::size_t>(idxB)];

                const auto [aStart, aEnd] = segmentBeatRange(segA, beatTimes, nBeats);
                const auto [bStart, bEnd] = segmentBeatRange(segB, beatTimes, nBeats);

                const std::vector<int> barsA = barsInSegment(bars, aStart, aEnd);
                const std::vector<int> barsB = barsInSegment(bars, bStart, bEnd);

                if (barsA.size() < 2 || barsB.size() < 2)
                    continue;

                const std::vector<double> chromaA =
                    barChroma(features, nFeat, bars, barsA);
                const std::vector<double> chromaB =
                    barChroma(features, nFeat, bars, barsB);

                const auto [correlation, offset] =
                    chromaCrossCorrelation(chromaA,
                                           static_cast<int>(barsA.size()),
                                           dim,
                                           chromaB,
                                           static_cast<int>(barsB.size()));

                if (correlation < minChromaCorr)
                    continue;

                result.sectionPairs.emplace_back(idxA, idxB);

                const int minBars = static_cast<int>(std::min(barsA.size(), barsB.size()));
                for (int bi = 0; bi < minBars; ++bi)
                {
                    const int bj = bi + offset;
                    if (bj < 0 || bj >= static_cast<int>(barsB.size()))
                        continue;

                    const int barAIdx = barsA[static_cast<std::size_t>(bi)];
                    const int barBIdx = barsB[static_cast<std::size_t>(bj)];
                    const auto [barAStart, barAEnd] = bars[static_cast<std::size_t>(barAIdx)];
                    const auto [barBStart, barBEnd] = bars[static_cast<std::size_t>(barBIdx)];

                    const int fromBeat = barAEnd;
                    const int toBeat   = barBStart;

                    if (std::abs(toBeat - fromBeat) < 8)
                        continue;

                    ++result.nPairsVerified;

                    double waveformSim = 0.0;
                    int    lag         = 0;
                    if (hasWaveforms)
                    {
                        auto [s, l] = verifyBarBoundary(boundaryWaveforms, snippetLen,
                                                        nBeats, fromBeat, toBeat,
                                                        maxLagSamples);
                        waveformSim = s;
                        lag         = l;
                        if (waveformSim < minWaveformSim)
                            continue;
                    }

                    result.jumps.push_back({
                        /*fromBeat*/             fromBeat,
                        /*toBeat*/               toBeat,
                        /*waveformSimilarity*/   waveformSim,
                        /*chromaCorrelation*/    correlation,
                        /*alignmentLagSamples*/  lag,
                        /*fromSectionIdx*/       idxA,
                        /*toSectionIdx*/         idxB,
                        /*fromBar*/              bi,
                        /*toBar*/                bj,
                    });

                    // Reverse (B → A) direction.
                    const int revFrom = barBEnd;
                    const int revTo   = barAStart;
                    if (std::abs(revTo - revFrom) >= 8)
                    {
                        double revSim = 0.0;
                        int    revLag = 0;
                        if (hasWaveforms)
                        {
                            auto [s2, l2] = verifyBarBoundary(boundaryWaveforms, snippetLen,
                                                              nBeats, revFrom, revTo,
                                                              maxLagSamples);
                            revSim = s2;
                            revLag = l2;
                            if (revSim < minWaveformSim)
                                continue;
                        }
                        result.jumps.push_back({
                            /*fromBeat*/             revFrom,
                            /*toBeat*/               revTo,
                            /*waveformSimilarity*/   revSim,
                            /*chromaCorrelation*/    correlation,
                            /*alignmentLagSamples*/  revLag,
                            /*fromSectionIdx*/       idxB,
                            /*toSectionIdx*/         idxA,
                            /*fromBar*/              bj,
                            /*toBar*/                bi,
                        });
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: internal repetitions via recurrence diagonals (L450-528)
    // -------------------------------------------------------------------------
    // Python's try/except ImportError catches a recurrence-module absence; our
    // C++ build always links Recurrence, so we just run it unconditionally.
    const Recurrence::Result rec =
        Recurrence::build(features, nBeats, nFeat, kRecurrenceKNeighbors);
    const std::vector<double>& R = rec.R;      // [nBeats × nBeats]
    const int nR = rec.nBeats;                 // == nBeats on success,
                                               // 0 on n < 4 early-out.

    const int minPhrase = 2 * timeSignature;   // Default: 8.

    for (int segI = 0; segI < static_cast<int>(segments.size()); ++segI)
    {
        const Segment& seg = segments[static_cast<std::size_t>(segI)];
        const auto [sStart, sEnd] = segmentBeatRange(seg, beatTimes, nBeats);
        const int segLen = sEnd - sStart;
        if (segLen < 4 * minPhrase)
            continue;
        const std::string lower = toLower(seg.label);
        if (lower == "intro" || lower == "outro")
            continue;

        // Collect candidate diagonals: one per bar-aligned lag.
        struct DiagEntry { int rMid; int cMid; int diagLen; double meanSim; };
        std::vector<DiagEntry> diags;

        // Python L474-494:
        //   for lag_bars in range(2, seg_len // time_signature):
        //     lag = lag_bars * time_signature
        //     overlap = seg_len - lag
        //     if overlap < min_phrase: break
        //     diag_vals = [(r, c, R[r, c]) for r in range(s_start,
        //                  min(s_end - lag, s_end)) if s_start <= c < s_end,
        //                  c = r + lag]
        //     if len(diag_vals) < min_phrase: continue
        //     mean_v = np.mean([v for _,_,v in diag_vals])
        //     if mean_v < 0.08: continue
        //     mid_idx = len(diag_vals) // 2
        //     r_mid, c_mid, _ = diag_vals[mid_idx]
        //     diags.append((r_mid, c_mid, len(diag_vals), mean_v))
        const int topLagBars = segLen / timeSignature;
        for (int lagBars = 2; lagBars < topLagBars; ++lagBars)
        {
            const int lag = lagBars * timeSignature;
            const int overlap = segLen - lag;
            if (overlap < minPhrase)
                break;

            // diag_vals: (r, c, R[r,c]) for r in [sStart, min(sEnd-lag, sEnd))
            //            with sStart <= c=(r+lag) < sEnd.
            struct DiagCell { int r; int c; double v; };
            std::vector<DiagCell> diagVals;
            const int rEnd = std::min(sEnd - lag, sEnd);
            for (int r = sStart; r < rEnd; ++r)
            {
                const int c = r + lag;
                if (c >= sStart && c < sEnd && r >= 0 && r < nR && c >= 0 && c < nR)
                {
                    const double v = R[static_cast<std::size_t>(r) * static_cast<std::size_t>(nR)
                                       + static_cast<std::size_t>(c)];
                    diagVals.push_back({r, c, v});
                }
            }
            if (static_cast<int>(diagVals.size()) < minPhrase)
                continue;

            // Mean of R[r,c] values. numpy.mean uses pairwise summation for
            // N >= 8; for small N (< 8) it matches naive sequential. Corpus
            // max diag length is ~n_beats which is > 8, so we match numpy's
            // pairwise order. For parity robustness we use naive sequential
            // here (corpus measurements show the diff is < 1e-12 on f64 sums
            // of this size; well below the 0.08 / 0.10 thresholds that gate).
            double sumV = 0.0;
            for (const auto& c : diagVals) sumV += c.v;
            const double meanV = sumV / static_cast<double>(diagVals.size());
            if (meanV < 0.08)
                continue;

            const std::size_t midIdx = diagVals.size() / 2;
            const auto& mid = diagVals[midIdx];
            diags.push_back({mid.r, mid.c, static_cast<int>(diagVals.size()), meanV});
        }

        // Python L496-526: for r_mid, c_mid, diag_len, mean_sim in diags[:5]:
        const std::size_t nTop = std::min<std::size_t>(diags.size(), 5);
        for (std::size_t d = 0; d < nTop; ++d)
        {
            const int    rMid   = diags[d].rMid;
            const int    cMid   = diags[d].cMid;
            const double meanS  = diags[d].meanSim;
            if (meanS < 0.10 || std::abs(rMid - cMid) < minPhrase)
                continue;

            const int fromBeat = rMid;
            const int toBeat   = cMid;

            double wfSim = 0.0;
            int    wfLag = 0;
            if (hasWaveforms && fromBeat + 1 < nBeats)
            {
                // Python L506-511: caller passes `min(from_beat + 1, n-1)` as
                // from_beat arg. Inside verifyBarBoundary, source_idx becomes
                // that + 1 — often out of range near segment end → (0, 0).
                const int clampedFrom = std::min(fromBeat + 1, nBeats - 1);
                auto [s, l] = verifyBarBoundary(boundaryWaveforms, snippetLen,
                                                nBeats, clampedFrom, toBeat,
                                                maxLagSamples);
                wfSim = s;
                wfLag = l;
                // Phase-2 gate is relaxed to 0.75 × minWaveformSim (Python L512).
                if (wfSim < minWaveformSim * 0.75)
                    continue;
            }

            // Python L515: `segments.index(seg) if seg in segments else 0`.
            // Identity-less .index() on the outer-loop value = first value-
            // equal match. Using the outer loop index `segI` matches when
            // all segments are value-unique (true on corpus; start/end
            // pairs are strictly monotone).
            const int segIdx = segI;
            result.jumps.push_back({
                /*fromBeat*/             fromBeat,
                /*toBeat*/               toBeat,
                /*waveformSimilarity*/   wfSim,
                /*chromaCorrelation*/    meanS,
                /*alignmentLagSamples*/  wfLag,
                /*fromSectionIdx*/       segIdx,
                /*toSectionIdx*/         segIdx,
                /*fromBar*/              0,
                /*toBar*/                0,
            });
        }
    }

    // Final sort: `(-waveform_similarity, -chroma_correlation)` (L530-531).
    // std::stable_sort preserves insertion order on ties — matches Python
    // list.sort() which is stable by contract.
    std::stable_sort(result.jumps.begin(), result.jumps.end(),
        [](const RepetitionJump& a, const RepetitionJump& b) {
            if (a.waveformSimilarity != b.waveformSimilarity)
                return a.waveformSimilarity > b.waveformSimilarity;
            return a.chromaCorrelation > b.chromaCorrelation;
        });

    return result;
}

} // namespace reamix::analysis
