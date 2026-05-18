#include "dsp/ChromaSTFT.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace reamix::dsp {

namespace {

// numpy.remainder semantics for positive divisor: result ∈ [0, n).
// Differs from std::fmod on negative dividends (fmod preserves sign of x).
// PARITY: librosa/filters.py L388 uses `np.remainder(D + ..., n_chroma)`.
inline double modPositive(double x, double n)
{
    double m = std::fmod(x, n);
    if (m < 0.0) m += n;
    return m;
}

// Build `librosa.filters.chroma(sr, n_fft, tuning, n_chroma=12, ctroct=5.0,
// octwidth=2, norm=2, base_c=True, dtype=float32)`. Returns a (n_chroma, n_bins)
// row-major matrix where n_bins = 1 + n_fft/2. All internal math is float64
// (matches numpy's default array dtype inside librosa.filters.chroma until the
// final `.astype(dtype)` cast); output is float32.
//
// PARITY: librosa/filters.py L266-407 (full chroma() body).
std::vector<std::vector<float>>
buildChromaFilterbank(double sr, int nFft, double tuning, int nChroma)
{
    const int nBins = 1 + nFft / 2;
    const double bpo = static_cast<double>(nChroma);

    // A440 after tuning shift. PARITY: core/convert.py hz_to_octs L1369.
    const double a440Tuned = 440.0 * std::pow(2.0, tuning / bpo);
    const double refHz     = a440Tuned / 16.0;  // A0 = 27.5Hz at tuning=0

    // frqbins has length n_fft. frqbins[k+1] corresponds to FFT frequency
    // freq[k] = sr * (k+1) / n_fft for k=0..n_fft-2 (linspace(0,sr,n_fft,
    // endpoint=False)[1:]). frqbins[0] is a fictional 0Hz bin placed 1.5
    // octaves (= 1.5·n_chroma "chroma bins") below frqbins[1]. PARITY:
    // librosa/filters.py L369-377.
    std::vector<double> frqbins(static_cast<std::size_t>(nFft), 0.0);
    for (int k = 0; k < nFft - 1; ++k) {
        const double f    = sr * static_cast<double>(k + 1)
                               / static_cast<double>(nFft);
        const double octs = std::log2(f / refHz);
        frqbins[static_cast<std::size_t>(k + 1)] = bpo * octs;
    }
    frqbins[0] = frqbins[1] - 1.5 * bpo;

    // binwidthbins[k] = max(frqbins[k+1] - frqbins[k], 1.0) for k=0..n_fft-2,
    // and binwidthbins[n_fft-1] = 1.0 (dummy — librosa concatenates [1]).
    // PARITY: librosa/filters.py L379.
    std::vector<double> binwidth(static_cast<std::size_t>(nFft), 1.0);
    for (int k = 0; k < nFft - 1; ++k) {
        const double d = frqbins[static_cast<std::size_t>(k + 1)]
                       - frqbins[static_cast<std::size_t>(k)];
        binwidth[static_cast<std::size_t>(k)] = std::max(d, 1.0);
    }

    // D[c][f] = frqbins[f] - c, then wrapped via `remainder(D + n_chroma/2
    // + 10·n_chroma, n_chroma) - n_chroma/2` into [-n_chroma/2, +n_chroma/2).
    // PARITY: librosa/filters.py L381-388.
    const double half   = std::round(static_cast<double>(nChroma) / 2.0);  // 6.0 @ nChroma=12
    const double offset = half + 10.0 * bpo;

    // wts[c][f] = exp(-0.5 * (2·D_wrapped[c][f] / binwidth[f])²).
    // PARITY: librosa/filters.py L391.
    std::vector<std::vector<double>> wts(
        static_cast<std::size_t>(nChroma),
        std::vector<double>(static_cast<std::size_t>(nFft), 0.0));
    for (int c = 0; c < nChroma; ++c) {
        for (int f = 0; f < nFft; ++f) {
            const double raw = frqbins[static_cast<std::size_t>(f)]
                             - static_cast<double>(c);
            const double wrapped = modPositive(raw + offset, bpo) - half;
            const double arg     = 2.0 * wrapped / binwidth[static_cast<std::size_t>(f)];
            wts[static_cast<std::size_t>(c)][static_cast<std::size_t>(f)]
                = std::exp(-0.5 * arg * arg);
        }
    }

    // L2-normalize each column. PARITY: librosa/filters.py L394 →
    // util.normalize(wts, norm=2, axis=0). The threshold used is
    // `tiny(wts)` where wts is float64 at this point — np.finfo(f64).tiny
    // ≈ 2.225e-308, so the small-norm branch never fires on normal inputs.
    for (int f = 0; f < nFft; ++f) {
        double ss = 0.0;
        for (int c = 0; c < nChroma; ++c)
            ss += wts[static_cast<std::size_t>(c)][static_cast<std::size_t>(f)]
                * wts[static_cast<std::size_t>(c)][static_cast<std::size_t>(f)];
        const double l2 = std::sqrt(ss);
        if (l2 > std::numeric_limits<double>::min()) {
            const double inv = 1.0 / l2;
            for (int c = 0; c < nChroma; ++c)
                wts[static_cast<std::size_t>(c)][static_cast<std::size_t>(f)] *= inv;
        }
    }

    // Octave Gaussian dominance window (octwidth=2, ctroct=5.0). PARITY:
    // librosa/filters.py L397-401. Multiplies each column by
    //   exp(-0.5 * ((frqbins[f]/n_chroma - ctroct) / octwidth)²).
    const double ctroct   = static_cast<double>(ChromaSTFT::kCtroct);
    const double octwidth = static_cast<double>(ChromaSTFT::kOctwidth);
    for (int f = 0; f < nFft; ++f) {
        const double u = frqbins[static_cast<std::size_t>(f)] / bpo - ctroct;
        const double g = std::exp(-0.5 * (u / octwidth) * (u / octwidth));
        for (int c = 0; c < nChroma; ++c)
            wts[static_cast<std::size_t>(c)][static_cast<std::size_t>(f)] *= g;
    }

    // base_c=True: roll rows by -3·(n_chroma//12) so the filterbank starts at
    // 'C' instead of 'A'. PARITY: librosa/filters.py L403-404. For n_chroma=12
    // this is a simple 3-row upward roll: new[c] = old[(c+3) mod 12].
    const int shift = 3 * (nChroma / 12);
    std::vector<std::vector<double>> rolled(
        static_cast<std::size_t>(nChroma),
        std::vector<double>(static_cast<std::size_t>(nFft), 0.0));
    for (int c = 0; c < nChroma; ++c) {
        const int src = ((c + shift) % nChroma + nChroma) % nChroma;
        rolled[static_cast<std::size_t>(c)] = wts[static_cast<std::size_t>(src)];
    }

    // Slice to first nBins columns and cast float64 → float32.
    // PARITY: librosa/filters.py L407  wts[:, :1+n_fft/2].astype(float32).
    std::vector<std::vector<float>> out(
        static_cast<std::size_t>(nChroma),
        std::vector<float>(static_cast<std::size_t>(nBins), 0.0f));
    for (int c = 0; c < nChroma; ++c)
        for (int b = 0; b < nBins; ++b)
            out[static_cast<std::size_t>(c)][static_cast<std::size_t>(b)]
                = static_cast<float>(
                    rolled[static_cast<std::size_t>(c)][static_cast<std::size_t>(b)]);

    return out;
}

} // namespace

std::vector<std::vector<float>>
ChromaSTFT::compute(const std::vector<std::vector<float>>& S,
                    float tuning,
                    float sr) const
{
    const std::size_t nFrames = S.size();
    if (nFrames == 0)
        return {};

    const std::size_t nBins = S[0].size();
    if (nBins < 2)
        return std::vector<std::vector<float>>(
            nFrames, std::vector<float>(static_cast<std::size_t>(kNChroma), 0.0f));

    // PARITY: chromafb n_fft inferred from S via _spectrogram — for S-input
    // path n_fft = 2·(d - 1). Same convention as PitchTrack.
    const int nFft = 2 * (static_cast<int>(nBins) - 1);

    const auto chromafb = buildChromaFilterbank(
        static_cast<double>(sr),
        nFft,
        static_cast<double>(tuning),
        kNChroma);

    std::vector<std::vector<float>> out(
        nFrames, std::vector<float>(static_cast<std::size_t>(kNChroma), 0.0f));

    // PARITY: librosa/feature/spectral.py L1283  raw = einsum('cf,ft->ct').
    // chromafb is (n_chroma, n_bins) float32, S is (n_bins, n_frames) float32;
    // einsum with optimize=True accumulates in float32.
    //
    // Normalize per frame: PARITY: librosa/feature/spectral.py L1286 →
    // util.normalize(raw, norm=np.inf, axis=-2). axis=-2 on a 2-D array is
    // axis=0 (rows, i.e. chroma bins), so per-frame L∞. `mag` in normalize is
    // cast to float64 and the division happens in float64 before being stored
    // into a float32 output array (np.empty_like(raw)). We mirror by doing the
    // normalization in double and casting the final value to float.
    std::vector<float> raw(static_cast<std::size_t>(kNChroma), 0.0f);
    for (std::size_t t = 0; t < nFrames; ++t) {
        const auto& col = S[t];
        if (col.size() != nBins) continue;  // defensive

        for (int c = 0; c < kNChroma; ++c) {
            float acc = 0.0f;
            const auto& row = chromafb[static_cast<std::size_t>(c)];
            for (std::size_t b = 0; b < nBins; ++b)
                acc += row[b] * col[b];
            raw[static_cast<std::size_t>(c)] = acc;
        }

        // Per-frame L∞.
        double maxAbs = 0.0;
        for (int c = 0; c < kNChroma; ++c) {
            const double a = std::fabs(static_cast<double>(
                raw[static_cast<std::size_t>(c)]));
            if (a > maxAbs) maxAbs = a;
        }

        // tiny(float32) — below this, librosa leaves the slice un-normalized
        // (fill=None branch: length[small_idx] = 1.0). PARITY: util/utils.py
        // L999-1005. np.finfo(np.float32).tiny = 2**-126 ≈ 1.175e-38.
        constexpr double kTinyF32 = 1.1754943508222875e-38;
        const double scale = (maxAbs < kTinyF32) ? 1.0 : 1.0 / maxAbs;

        for (int c = 0; c < kNChroma; ++c)
            out[t][static_cast<std::size_t>(c)] = static_cast<float>(
                static_cast<double>(raw[static_cast<std::size_t>(c)]) * scale);
    }

    return out;
}

} // namespace reamix::dsp
