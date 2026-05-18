#include "analysis/VocalFeatures.h"

#include "analysis/BeatSync.h"
#include "dsp/RMS.h"
#include "dsp/SpectralFlatness.h"
#include "util/Percentile.h"

#include <algorithm>
#include <cstddef>

namespace reamix::analysis {

namespace {

// `librosa.fft_frequencies(sr, n_fft)` — f64 linear bin centers
// freq[k] = k * sr / n_fft, k ∈ [0, n_fft/2].
std::vector<double>
fftFrequencies(double sr, int nFft)
{
    const int nBins = nFft / 2 + 1;
    std::vector<double> out(static_cast<std::size_t>(nBins));
    for (int k = 0; k < nBins; ++k) {
        out[static_cast<std::size_t>(k)] =
            static_cast<double>(k) * sr / static_cast<double>(nFft);
    }
    return out;
}

// Per-frame mean over magnitude bins selected by `mask` (length = nBins).
// Returns f64 per frame; accumulator in f64 vs numpy's f32 mean, producing
// drift at f32 ULP (~1.7e-6) — well under VocalFeatures' 1e-3 gate.
std::vector<double>
maskedMean(const std::vector<std::vector<float>>& mag,
           const std::vector<bool>& mask)
{
    const std::size_t nFrames = mag.size();
    std::vector<double> out(nFrames, 0.0);
    if (nFrames == 0) return out;

    int nSelected = 0;
    for (bool m : mask) if (m) ++nSelected;
    if (nSelected == 0) return out;
    const double invN = 1.0 / static_cast<double>(nSelected);

    for (std::size_t t = 0; t < nFrames; ++t) {
        const auto& col = mag[t];
        double sum = 0.0;
        for (std::size_t k = 0; k < mask.size(); ++k) {
            if (mask[k]) sum += static_cast<double>(col[k]);
        }
        out[t] = sum * invN;
    }
    return out;
}

// PARITY: vocal_features.py L193-198 final `np.clip(x, 0, 1)` on each output.
void clip01(std::vector<double>& v)
{
    for (auto& x : v) {
        if (x < 0.0)      x = 0.0;
        else if (x > 1.0) x = 1.0;
    }
}

// PARITY: feature_extractor.py::_beat_sync_1d (L477-495). Wrap 1-D as
// [1 × T], call sync (mean aggregate), take row 0, truncate or pad with
// last value to nBeats.
std::vector<double>
beatSync1D(const std::vector<double>& values,
           const std::vector<int>& beatFrames,
           int nBeats)
{
    const std::size_t T = values.size();
    auto r = BeatSync::sync(values.data(), /*D=*/1, T, beatFrames);
    std::vector<double> out(static_cast<std::size_t>(nBeats), 0.0);

    const std::size_t synced = r.nSlices;
    const std::size_t take   = std::min<std::size_t>(synced, static_cast<std::size_t>(nBeats));
    for (std::size_t i = 0; i < take; ++i) out[i] = r.data[i];

    // PARITY: pad with last value (or 0.0 if synced empty) when synced < nBeats.
    if (synced < static_cast<std::size_t>(nBeats)) {
        const double pad = synced > 0 ? r.data[synced - 1] : 0.0;
        for (std::size_t i = synced; i < static_cast<std::size_t>(nBeats); ++i)
            out[i] = pad;
    }
    return out;
}

// Compute the per-beat slice [frame_lo, frame_hi) the same way Python's
// _extract_edge_1d does: beat i's end is beat_frames[i+1], or nFrames for
// the last beat; both ends clipped to [0, nFrames].
void beatSlice(const std::vector<int>& beatFrames,
               int i, int nFrames, int& frameLo, int& frameHi)
{
    const int nbf = static_cast<int>(beatFrames.size());
    int lo = beatFrames[static_cast<std::size_t>(i)];
    int hi = (i + 1 < nbf) ? beatFrames[static_cast<std::size_t>(i + 1)] : nFrames;
    lo = std::max(0, std::min(lo, nFrames));
    hi = std::max(lo, std::min(hi, nFrames));
    frameLo = lo;
    frameHi = hi;
}

// PARITY: vocal_features.py::_extract_edge_1d (L259-289). Per-beat first
// n_edge frames mean as start, last n_edge frames mean as end.
// n_edge = min(n_edge_frames, max(1, beat_len / 2)), beat_len = frame_hi - frame_lo.
// Skip (leave zero) when beat_len <= 0.
std::pair<std::vector<double>, std::vector<double>>
extractEdge1D(const std::vector<double>& values,
              const std::vector<int>& beatFrames,
              int nBeats, int nEdgeFrames)
{
    const int nFrames = static_cast<int>(values.size());
    std::vector<double> start(static_cast<std::size_t>(nBeats), 0.0);
    std::vector<double> end  (static_cast<std::size_t>(nBeats), 0.0);

    for (int i = 0; i < nBeats; ++i) {
        int lo, hi;
        beatSlice(beatFrames, i, nFrames, lo, hi);
        const int beatLen = hi - lo;
        if (beatLen <= 0) continue;
        const int nEdge = std::min(nEdgeFrames, std::max(1, beatLen / 2));

        double sumS = 0.0, sumE = 0.0;
        for (int k = 0; k < nEdge; ++k) {
            sumS += values[static_cast<std::size_t>(lo + k)];
            sumE += values[static_cast<std::size_t>(hi - nEdge + k)];
        }
        const double invN = 1.0 / static_cast<double>(nEdge);
        start[static_cast<std::size_t>(i)] = sumS * invN;
        end  [static_cast<std::size_t>(i)] = sumE * invN;
    }
    return {std::move(start), std::move(end)};
}

// PARITY: vocal_features.py::_extract_edge_1d_peak (L291-314). Same slicing
// as _extract_edge_1d but uses np.max instead of np.mean.
std::pair<std::vector<double>, std::vector<double>>
extractEdge1DPeak(const std::vector<double>& values,
                  const std::vector<int>& beatFrames,
                  int nBeats, int nEdgeFrames)
{
    const int nFrames = static_cast<int>(values.size());
    std::vector<double> start(static_cast<std::size_t>(nBeats), 0.0);
    std::vector<double> end  (static_cast<std::size_t>(nBeats), 0.0);

    for (int i = 0; i < nBeats; ++i) {
        int lo, hi;
        beatSlice(beatFrames, i, nFrames, lo, hi);
        const int beatLen = hi - lo;
        if (beatLen <= 0) continue;
        const int nEdge = std::min(nEdgeFrames, std::max(1, beatLen / 2));

        double maxS = values[static_cast<std::size_t>(lo)];
        double maxE = values[static_cast<std::size_t>(hi - nEdge)];
        for (int k = 1; k < nEdge; ++k) {
            const double s = values[static_cast<std::size_t>(lo + k)];
            const double e = values[static_cast<std::size_t>(hi - nEdge + k)];
            if (s > maxS) maxS = s;
            if (e > maxE) maxE = e;
        }
        start[static_cast<std::size_t>(i)] = maxS;
        end  [static_cast<std::size_t>(i)] = maxE;
    }
    return {std::move(start), std::move(end)};
}

} // namespace

VocalFeatures::Result
VocalFeatures::compute(const std::vector<float>& y,
                       const std::vector<float>& yHarmonic,
                       const std::vector<std::vector<float>>& magHarm,
                       double sr,
                       const std::vector<int>& beatFrames,
                       int nBeats)
{
    Result r;
    if (nBeats == 0) return r;
    const std::size_t nB = static_cast<std::size_t>(nBeats);
    r.vocalActivity         .assign(nB, 0.0);
    r.voicedRatio           .assign(nB, 0.0);  // DEFERRED (ADR-014 D4): zero-filled
    r.f0Hz                  .assign(nB, 0.0);  // DEFERRED (ADR-014 D2): zero-filled
    r.f0Confidence          .assign(nB, 0.0);  // DEFERRED (ADR-014 D3): zero-filled
    r.edgeVocalActivityStart.assign(nB, 0.0);
    r.edgeVocalActivityEnd  .assign(nB, 0.0);
    r.edgeVocalOnsetStart   .assign(nB, 0.0);
    r.edgeVocalReleaseEnd   .assign(nB, 0.0);

    // --- Harmonic ratio (vocal_features.py L75-79) ---
    const dsp::RMS rms;
    const auto harmonicRms = rms.compute(yHarmonic);  // f32
    const auto fullRms     = rms.compute(y);          // f32
    const std::size_t nFramesRms = std::min(harmonicRms.size(), fullRms.size());

    std::vector<double> harmonicRatio(nFramesRms, 0.0);
    for (std::size_t t = 0; t < nFramesRms; ++t) {
        const double fr = std::max(static_cast<double>(fullRms[t]), kRmsFloor);
        double v        = static_cast<double>(harmonicRms[t]) / fr / kHarmonicDivisor;
        if (v < 0.0)      v = 0.0;
        else if (v > 1.0) v = 1.0;
        harmonicRatio[t] = v;
    }

    // --- Spectral flatness + inv (vocal_features.py L81-88) ---
    const dsp::SpectralFlatness flat;
    const auto flatness = flat.compute(magHarm);  // f32 per frame

    // p95 of flatness (f32 input → f64 lerp, matches numpy percentile).
    double flatScale = kScaleFloor;
    if (!flatness.empty()) {
        const double p95 = util::percentile<float>(flatness, 95.0);
        flatScale = std::max(p95, kScaleFloor);
    }

    std::vector<double> flatnessInv(flatness.size(), 0.0);
    for (std::size_t t = 0; t < flatness.size(); ++t) {
        double v = 1.0 - static_cast<double>(flatness[t]) / flatScale;
        if (v < 0.0)      v = 0.0;
        else if (v > 1.0) v = 1.0;
        flatnessInv[t] = v;
    }

    // --- Voice-band ratio (vocal_features.py L90-103) ---
    const auto freqs = fftFrequencies(sr, kNfft);
    const std::size_t nBins = freqs.size();
    std::vector<bool> voiceMask(nBins, false);
    std::vector<bool> bodyMask (nBins, false);
    bool anyVoice = false, anyBody = false;
    for (std::size_t k = 0; k < nBins; ++k) {
        if (freqs[k] >= kVoiceLowHz && freqs[k] <= kVoiceHighHz) {
            voiceMask[k] = true; anyVoice = true;
        }
        if (freqs[k] >= kBodyLowHz && freqs[k] <= kBodyHighHz) {
            bodyMask[k] = true;  anyBody  = true;
        }
    }

    std::vector<double> voiceBandRatio(magHarm.size(), 0.0);
    if (anyVoice && anyBody && !magHarm.empty()) {
        const auto voiceE = maskedMean(magHarm, voiceMask);
        const auto bodyE  = maskedMean(magHarm, bodyMask);
        for (std::size_t t = 0; t < voiceE.size(); ++t) {
            voiceBandRatio[t] = voiceE[t] / std::max(bodyE[t], kBandFloor);
        }
        // p95 scale + clip.
        double ratioScale = kScaleFloor;
        if (!voiceBandRatio.empty()) {
            const double p95 = util::percentile<double>(voiceBandRatio, 95.0);
            ratioScale = std::max(p95, kScaleFloor);
        }
        for (auto& v : voiceBandRatio) {
            v /= ratioScale;
            if (v < 0.0)      v = 0.0;
            else if (v > 1.0) v = 1.0;
        }
    }
    // else: voiceBandRatio stays zero-filled, matching vocal_features.py L105-106.

    // --- Frame-count reconciliation (L127-132) ---
    const std::size_t frameCount = std::min({
        harmonicRatio.size(),
        flatnessInv.size(),
        voiceBandRatio.size(),
    });
    // use_pyin=False → f0_hz_frame / voiced_prob are zero-filled at shape of
    // harmonic_ratio; their truncation to frameCount is trivial (all zeros).
    // DEFERRED (ADR-014 D2/D3): no pYIN call; f0_hz_frame + voiced_prob stay
    // zero vectors of length frameCount. Re-enable when pYIN is ported
    // (provisional phase-2c-pyin). See phase-2b/spec.md § "Deferred".

    // --- Composite (vocal_features.py L148-155, use_pyin=False branch) ---
    std::vector<double> vocalActivityFrame(frameCount, 0.0);
    for (std::size_t t = 0; t < frameCount; ++t) {
        double v = kCompHarm * harmonicRatio[t]
                 + kCompVoiceBand * voiceBandRatio[t]
                 + kCompFlatnessInv * flatnessInv[t];
        if (v < 0.0)      v = 0.0;
        else if (v > 1.0) v = 1.0;
        vocalActivityFrame[t] = v;
    }

    // --- Rise/fall split (vocal_features.py L158-167) ---
    // diff with prepend=[first] → delta[0] = 0, delta[t>0] = va[t] - va[t-1].
    std::vector<double> vocalRise (frameCount, 0.0);
    std::vector<double> vocalFall (frameCount, 0.0);
    for (std::size_t t = 0; t < frameCount; ++t) {
        const double prev = (t == 0) ? vocalActivityFrame[0] : vocalActivityFrame[t - 1];
        const double d    = vocalActivityFrame[t] - prev;
        vocalRise[t] = d > 0.0 ? d : 0.0;
        vocalFall[t] = d < 0.0 ? -d : 0.0;
    }
    if (!vocalRise.empty()) {
        const double p95  = util::percentile<double>(vocalRise, 95.0);
        const double sc   = std::max(p95, kScaleFloor);
        for (auto& v : vocalRise) {
            v /= sc;
            if (v < 0.0)      v = 0.0;
            else if (v > 1.0) v = 1.0;
        }
    }
    if (!vocalFall.empty()) {
        const double p95  = util::percentile<double>(vocalFall, 95.0);
        const double sc   = std::max(p95, kScaleFloor);
        for (auto& v : vocalFall) {
            v /= sc;
            if (v < 0.0)      v = 0.0;
            else if (v > 1.0) v = 1.0;
        }
    }

    // --- Beat-sync + edges (vocal_features.py L169-182) ---
    r.vocalActivity = beatSync1D(vocalActivityFrame, beatFrames, nBeats);
    // DEFERRED (ADR-014 D4): voicedRatio = beat_sync_1d(voiced_frame=zeros) → zeros.
    //   Keep stored zero-fill above; no computation needed.
    // DEFERRED (ADR-014 D3): f0Confidence = beat_sync_1d(voiced_prob=zeros) → zeros.
    // DEFERRED (ADR-014 D2/D7): f0Hz = _beat_sync_voiced_f0(zero inputs) → zeros.
    //   The Python utility (vocal_features.py L224-252) would be called on zero
    //   inputs; mask `(f0>0) & (conf>=0.35)` never true, output is zeros.

    auto edgeVa = extractEdge1D(vocalActivityFrame, beatFrames, nBeats, kNEdgeFrames);
    r.edgeVocalActivityStart = std::move(edgeVa.first);
    r.edgeVocalActivityEnd   = std::move(edgeVa.second);

    auto edgeRise = extractEdge1DPeak(vocalRise, beatFrames, nBeats, kNEdgeFrames);
    r.edgeVocalOnsetStart    = std::move(edgeRise.first);
    // edgeRise.second discarded — matches vocal_features.py L179 "onset_start, _".

    auto edgeFall = extractEdge1DPeak(vocalFall, beatFrames, nBeats, kNEdgeFrames);
    // edgeFall.first discarded — matches vocal_features.py L183 "_, release_end".
    r.edgeVocalReleaseEnd    = std::move(edgeFall.second);

    // --- Final clips (vocal_features.py L191-198) ---
    clip01(r.vocalActivity);
    // voicedRatio / f0Confidence already zero. f0Hz clip below.
    clip01(r.edgeVocalActivityStart);
    clip01(r.edgeVocalActivityEnd);
    clip01(r.edgeVocalOnsetStart);
    clip01(r.edgeVocalReleaseEnd);

    // f0_hz clipped to [0, kVocalMaxHz * kF0ClipFactor]. DEFERRED (ADR-014 D8):
    // inactive path with zero input (zeros already in [0, 1200]); preserved for
    // symmetry with the future pYIN path.
    const double f0Hi = kVocalMaxHz * kF0ClipFactor;
    for (auto& v : r.f0Hz) {
        if (v < 0.0)     v = 0.0;
        else if (v > f0Hi) v = f0Hi;
    }

    return r;
}

} // namespace reamix::analysis
