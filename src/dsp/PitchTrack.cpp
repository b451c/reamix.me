#include "dsp/PitchTrack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace reamix::dsp {

namespace {

// Parabolic interpolation along the frequency axis. Mirrors librosa's
// numba stencil `_pi_stencil` (core/pitch.py L416-427):
//   a  = x[n+1] + x[n-1] - 2 x[n]
//   b  = (x[n+1] - x[n-1]) / 2
//   shift[n] = 0                 if |b| >= |a|  (optimum falls > 1 bin away)
//   shift[n] = -b / a            otherwise
// Edges (n=0 and n=N-1) are zeroed, per `_parabolic_interpolation` L469-471.
void parabolicInterpolationAxisFreq(const std::vector<float>& col,
                                    std::vector<float>& shift)
{
    const int n = static_cast<int>(col.size());
    shift.assign(static_cast<std::size_t>(n), 0.0f);
    for (int i = 1; i < n - 1; ++i) {
        const float xm = col[static_cast<std::size_t>(i - 1)];
        const float x0 = col[static_cast<std::size_t>(i)];
        const float xp = col[static_cast<std::size_t>(i + 1)];
        const float a  = xp + xm - 2.0f * x0;
        const float b  = 0.5f * (xp - xm);
        if (std::fabs(b) >= std::fabs(a))
            shift[static_cast<std::size_t>(i)] = 0.0f;
        else
            shift[static_cast<std::size_t>(i)] = -b / a;
    }
}

// np.gradient(col, axis=0) — first-order (default edge_order=1): central
// differences in the interior, one-sided at the edges.
void gradientAxisFreq(const std::vector<float>& col,
                      std::vector<float>& g)
{
    const int n = static_cast<int>(col.size());
    g.assign(static_cast<std::size_t>(n), 0.0f);
    if (n < 2) return;
    g[0] = col[1] - col[0];
    g[static_cast<std::size_t>(n - 1)] =
        col[static_cast<std::size_t>(n - 1)] - col[static_cast<std::size_t>(n - 2)];
    for (int i = 1; i < n - 1; ++i) {
        g[static_cast<std::size_t>(i)] = 0.5f
            * (col[static_cast<std::size_t>(i + 1)]
               - col[static_cast<std::size_t>(i - 1)]);
    }
}

// librosa.util.localmax along the last axis. Stencil:
//   lmax[n] = (x[n] > x[n-1]) & (x[n] >= x[n+1])
// With edges:
//   lmax[0]   = false               (numba stencil defaults to 0 at boundary)
//   lmax[N-1] = x[N-1] > x[N-2]     (explicit override in util.py L1125)
void localmaxAxisFreq(const std::vector<float>& col,
                      std::vector<unsigned char>& out)
{
    const int n = static_cast<int>(col.size());
    out.assign(static_cast<std::size_t>(n), 0);
    if (n < 2) return;
    for (int i = 1; i < n - 1; ++i) {
        const float xm = col[static_cast<std::size_t>(i - 1)];
        const float x0 = col[static_cast<std::size_t>(i)];
        const float xp = col[static_cast<std::size_t>(i + 1)];
        out[static_cast<std::size_t>(i)] = (x0 > xm && x0 >= xp) ? 1u : 0u;
    }
    out[static_cast<std::size_t>(n - 1)] =
        (col[static_cast<std::size_t>(n - 1)]
         > col[static_cast<std::size_t>(n - 2)]) ? 1u : 0u;
}

// Median of a buffer (non-const, modifies via std::nth_element). Matches
// numpy.median: for odd N, middle element; for even N, average of the two
// middle elements.
double medianInPlace(std::vector<float>& v)
{
    const std::size_t n = v.size();
    if (n == 0) return 0.0;
    const std::size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const double hi = static_cast<double>(v[mid]);
    if (n % 2 == 1) return hi;
    // Even N: need the element at mid-1 (the max of the lower half after
    // nth_element partitions at mid). std::max_element over [begin, mid).
    auto lowerMax = std::max_element(v.begin(),
                                     v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (hi + static_cast<double>(*lowerMax));
}

// pitch_tuning (librosa.core.pitch.pitch_tuning, L110-177). Returns the left
// edge of the winning bin of a 100-bin histogram over residuals.
double pitchTuning(const std::vector<double>& freqs,
                   double resolution,
                   int binsPerOctave)
{
    // Trim DC — freqs > 0.
    std::vector<double> residuals;
    residuals.reserve(freqs.size());
    // hz_to_octs(f, tuning=0, bpo=12) = log2(f / (A440 / 16)) = log2(f / 27.5).
    // PARITY: librosa/core/convert.py L1336-1372.
    constexpr double kA440Over16 = 27.5;
    const double bpo = static_cast<double>(binsPerOctave);
    for (double f : freqs) {
        if (!(f > 0.0)) continue;
        const double octs = std::log2(f / kA440Over16);
        double r = bpo * octs;
        // np.mod(x, 1.0) = x - floor(x) — always non-negative (differs from
        // C/C++ std::fmod for negative x). Input freqs > 150 Hz ⇒ octs > 0,
        // so fmod would suffice here, but use floor-based for safety.
        r -= std::floor(r);
        if (r >= 0.5) r -= 1.0;
        residuals.push_back(r);
    }
    if (residuals.empty()) return 0.0;

    // Histogram bins: linspace(-0.5, 0.5, ceil(1/resolution) + 1).
    // For resolution=0.01 → 101 edges, 100 bins, step = 1/100 = 0.01 exactly
    // in float64 (well, `0.01` isn't exact but numpy uses the same literal).
    const int nEdges = static_cast<int>(std::ceil(1.0 / resolution)) + 1;
    const int nBins  = nEdges - 1;
    const double step = 1.0 / static_cast<double>(nEdges - 1);

    std::vector<int> counts(static_cast<std::size_t>(nBins), 0);
    // numpy.histogram semantics: bin k covers [edges[k], edges[k+1]), except
    // the last bin [edges[-2], edges[-1]] is closed on both ends. We map each
    // residual via floor((r - edges[0]) / step), clamping the closed upper
    // edge into the last bin and dropping values outside [edges[0], edges[-1]].
    for (double r : residuals) {
        if (r < -0.5 || r > 0.5) continue;  // out of histogram range
        int idx = static_cast<int>(std::floor((r - (-0.5)) / step));
        if (idx < 0) idx = 0;
        if (idx >= nBins) idx = nBins - 1;
        ++counts[static_cast<std::size_t>(idx)];
    }

    // argmax. np.argmax returns the first index on ties.
    int best = 0;
    int bestCount = counts[0];
    for (int i = 1; i < nBins; ++i) {
        if (counts[static_cast<std::size_t>(i)] > bestCount) {
            bestCount = counts[static_cast<std::size_t>(i)];
            best = i;
        }
    }
    return -0.5 + static_cast<double>(best) * step;
}

} // namespace

float PitchTrack::estimateTuning(const std::vector<std::vector<float>>& S,
                                 float sr,
                                 int binsPerOctave,
                                 double resolution) const
{
    const std::size_t nFrames = S.size();
    if (nFrames == 0) return 0.0f;
    const std::size_t nBins = S[0].size();
    if (nBins < 2) return 0.0f;

    // PARITY: librosa piptrack infers n_fft from S.shape as 2*(n_bins-1).
    // See core/spectrum.py _spectrogram — for S-input, n_fft = 2*(d-1).
    const int nFft = 2 * (static_cast<int>(nBins) - 1);
    const double nFftD = static_cast<double>(nFft);
    const double srD   = static_cast<double>(sr);

    // fft_freqs[k] = sr * k / n_fft. PARITY: core/convert.py L1586-1607.
    std::vector<float> fftFreqs(nBins);
    for (std::size_t k = 0; k < nBins; ++k)
        fftFreqs[k] = static_cast<float>(srD * static_cast<double>(k) / nFftD);

    // fmin = max(fmin, 0); fmax = min(fmax, sr/2). PARITY: pitch.py L320-321.
    const float fminClip = std::max(kFmin, 0.0f);
    const float fmaxClip = std::min(kFmax, static_cast<float>(0.5 * srD));

    std::vector<unsigned char> freqMask(nBins, 0);
    for (std::size_t k = 0; k < nBins; ++k)
        freqMask[k] = (fminClip <= fftFreqs[k] && fftFreqs[k] < fmaxClip) ? 1u : 0u;

    // Pitches[t,b] we only store as residuals; keep a flat accumulator instead.
    std::vector<double> pickedPitches;
    std::vector<float>  pickedMags;
    pickedPitches.reserve(nFrames * 16);
    pickedMags.reserve(nFrames * 16);

    std::vector<float> grad(nBins), shift(nBins);
    std::vector<unsigned char> lmax(nBins);
    std::vector<float> masked(nBins);

    for (std::size_t t = 0; t < nFrames; ++t) {
        const auto& col = S[t];
        if (col.size() != nBins) continue;  // defensive

        // np.abs(S) — librosa.piptrack L317. S is already |STFT|² ≥ 0, no-op.

        // avg[b] = gradient(S, axis=-2)[b]
        gradientAxisFreq(col, grad);

        // shift[b] = parabolic_interpolation(S, axis=-2)[b]
        parabolicInterpolationAxisFreq(col, shift);

        // ref_value = threshold * max(S, axis=-2)  (scalar per frame)
        float maxS = 0.0f;
        for (std::size_t b = 0; b < nBins; ++b)
            if (col[b] > maxS) maxS = col[b];
        const float refValue = kRefThreshold * maxS;

        // localmax argument: S * (S > ref_value). Below-threshold bins
        // become 0 before the stencil runs (librosa L356).
        for (std::size_t b = 0; b < nBins; ++b)
            masked[b] = (col[b] > refValue) ? col[b] : 0.0f;

        localmaxAxisFreq(masked, lmax);

        // idx = nonzero(freq_mask & localmax(...)). Pitches/mags only at idx.
        for (std::size_t b = 0; b < nBins; ++b) {
            if (!(freqMask[b] && lmax[b])) continue;
            // pitches[b,t] = (b + shift[b,t]) * sr / n_fft
            const double pitchHz =
                (static_cast<double>(b) + static_cast<double>(shift[b]))
                * srD / nFftD;
            // mags[b,t] = S[b,t] + 0.5 * avg[b,t] * shift[b,t]   (dskew)
            // PARITY: pitch.py L331 (dskew) + L358.
            const float magVal = col[b] + 0.5f * grad[b] * shift[b];
            pickedPitches.push_back(pitchHz);
            pickedMags.push_back(magVal);
        }
    }

    // threshold = median(mag[pitch_mask]) — pitch_mask ≡ pitch > 0.
    // In librosa the full pitches/mags arrays are zero outside `idx`, and
    // pitch_mask filters back to the populated entries. We only stored the
    // populated ones, so pitch_mask is implicitly all-True here; still guard
    // against any non-positive pitches (shouldn't occur but cheap to check).
    std::vector<float> positiveMags;
    positiveMags.reserve(pickedMags.size());
    for (std::size_t i = 0; i < pickedPitches.size(); ++i) {
        if (pickedPitches[i] > 0.0) positiveMags.push_back(pickedMags[i]);
    }

    double thresholdMedian = 0.0;
    if (!positiveMags.empty())
        thresholdMedian = medianInPlace(positiveMags);

    // freqs = pitches[(mag >= threshold) & (pitch > 0)]
    std::vector<double> selectedFreqs;
    selectedFreqs.reserve(pickedPitches.size());
    for (std::size_t i = 0; i < pickedPitches.size(); ++i) {
        if (pickedPitches[i] > 0.0
            && static_cast<double>(pickedMags[i]) >= thresholdMedian) {
            selectedFreqs.push_back(pickedPitches[i]);
        }
    }

    return static_cast<float>(
        pitchTuning(selectedFreqs, resolution, binsPerOctave));
}

} // namespace reamix::dsp
