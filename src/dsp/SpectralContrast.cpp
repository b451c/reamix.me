#include "dsp/SpectralContrast.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace reamix::dsp {

namespace {

// Round half-to-even (banker's rounding), matching numpy's np.rint under
// default rounding mode. PARITY: librosa/feature/spectral.py L990
// `idx = np.rint(quantile * np.sum(current_band))`.
//
// std::nearbyint with the default rounding mode FE_TONEAREST implements
// IEEE 754 round-to-nearest-ties-to-even — matches np.rint exactly.
inline double rintToEven(double x)
{
    return std::nearbyint(x);
}

// power_to_db(M, ref=1.0, amin=1e-10, top_db=80.0) applied in place on a
// flattened row-major buffer of shape (kNRows, nFrames). Writes to `out`
// in float64, matching librosa's np.zeros((n_bands+1, n_frames)) default
// float64 allocation of `peak` / `valley`.
//
// PARITY: librosa/core/spectrum.py::power_to_db L1619-1632.
//   log_spec = 10 * log10(max(amin, M))
//   log_spec -= 10 * log10(max(amin, 1.0))     // == 0 for ref=1.0
//   if top_db is not None:
//       log_spec = max(log_spec, log_spec.max() - top_db)
// The clip is applied to the ENTIRE array (peak or valley), using that
// array's own max — NOT the max of (peak, valley) jointly. Subtle: valleys
// typically clip much earlier than peaks.
void applyPowerToDb(std::vector<double>& M, std::size_t rows, std::size_t cols,
                    double amin, double topDb)
{
    (void)rows; (void)cols;  // derived from M.size() == rows*cols in callers
    const double refLog = 10.0 * std::log10(std::max(amin, 1.0));  // = 0 here

    double logMax = -std::numeric_limits<double>::infinity();
    for (double& v : M) {
        v = 10.0 * std::log10(std::max(amin, v)) - refLog;
        if (v > logMax) logMax = v;
    }
    const double floor = logMax - topDb;
    for (double& v : M) {
        if (v < floor) v = floor;
    }
}

} // namespace

std::vector<std::vector<float>>
SpectralContrast::compute(const std::vector<std::vector<float>>& S,
                          float sr) const
{
    const std::size_t nFrames = S.size();
    if (nFrames == 0)
        return {};

    const std::size_t nBins = S[0].size();
    if (nBins < 2)
        return std::vector<std::vector<float>>(
            nFrames, std::vector<float>(static_cast<std::size_t>(kNRows), 0.0f));

    // PARITY: _spectrogram infers n_fft = 2·(d - 1) for S-input path.
    const int nFft = 2 * (static_cast<int>(nBins) - 1);

    // freq = fft_frequencies(sr, n_fft) = linspace(0, sr/2, 1 + n_fft/2).
    // PARITY: librosa/core/convert.py::fft_frequencies + np.fft.rfftfreq.
    std::vector<double> freq(nBins, 0.0);
    const double df = static_cast<double>(sr) / static_cast<double>(nFft);
    for (std::size_t b = 0; b < nBins; ++b)
        freq[b] = df * static_cast<double>(b);

    // octa = [0, fmin, 2·fmin, 4·fmin, ..., 2^n_bands·fmin] — length n_bands+2.
    // PARITY: librosa/feature/spectral.py L968-969.
    std::vector<double> octa(static_cast<std::size_t>(kNBands) + 2, 0.0);
    for (int i = 0; i <= kNBands; ++i)
        octa[static_cast<std::size_t>(i + 1)] =
            static_cast<double>(kFmin) * std::pow(2.0, static_cast<double>(i));

    // peak / valley matrices — row-major (kNRows, nFrames), float64.
    // PARITY: librosa/feature/spectral.py L979 `valley = np.zeros(shape)`
    // — default dtype is float64 regardless of S.dtype. The subsequent
    // `valley[k, :] = mean(...)` on float32 inputs upcasts to float64.
    std::vector<double> peak(static_cast<std::size_t>(kNRows) * nFrames, 0.0);
    std::vector<double> valley(static_cast<std::size_t>(kNRows) * nFrames, 0.0);

    // Per-band: build mask, flatnonzero, boundary-share, sub-band slice, sort,
    // mean of top/bottom quantile. PARITY: librosa/feature/spectral.py L982-995.
    std::vector<char> bandMask(nBins, 0);
    std::vector<float> column;  // reused per frame

    for (int k = 0; k <= kNBands; ++k) {
        const double fLow  = octa[static_cast<std::size_t>(k)];
        const double fHigh = octa[static_cast<std::size_t>(k + 1)];

        // current_band = (freq >= f_low) & (freq <= f_high) — inclusive both sides.
        // PARITY: L982.
        std::fill(bandMask.begin(), bandMask.end(), 0);
        int firstIdx = -1;
        int lastIdx  = -1;
        for (std::size_t b = 0; b < nBins; ++b) {
            if (freq[b] >= fLow && freq[b] <= fHigh) {
                bandMask[b] = 1;
                if (firstIdx < 0) firstIdx = static_cast<int>(b);
                lastIdx = static_cast<int>(b);
            }
        }

        // If no bin fell in the band, skip — shouldn't happen with fmin=200,
        // n_bands=6 on sr=22050, but guards against pathological inputs.
        if (firstIdx < 0) {
            // Leave peak/valley at zero for this band (current row). Matches
            // the downstream log10(max(amin, 0)) = log10(amin) = -100 dB clip
            // before top_db floor.
            continue;
        }

        // Boundary extensions:
        //   if k > 0:       current_band[idx[0] - 1] = True
        //   if k == n_bands: current_band[idx[-1] + 1 :] = True
        // PARITY: L985-989.
        if (k > 0 && firstIdx > 0)
            bandMask[static_cast<std::size_t>(firstIdx - 1)] = 1;
        if (k == kNBands) {
            for (std::size_t b = static_cast<std::size_t>(lastIdx + 1); b < nBins; ++b)
                bandMask[b] = 1;
        }

        // Collect selected bin indices (in ascending freq order).
        std::vector<std::size_t> selBins;
        selBins.reserve(nBins);
        for (std::size_t b = 0; b < nBins; ++b)
            if (bandMask[b]) selBins.push_back(b);

        // `n_sel = max(1, rint(quantile * sum(current_band)))` — computed from
        // the EXTENDED mask, BEFORE the last-bin drop. PARITY: L993-994.
        // kQuantile is double (0.02 exactly in float64 terms → 0.02·75 = 1.5
        // which banker's-rounds to 2). If kQuantile were float32 (0.02f ≈
        // 0.019999999552...), 0.02f·75 = 1.4999999664... would round to 1,
        // biasing nSel down by 1 for every band whose count hits a half-integer
        // quantile — the session-8 band-3 regression on billie_jean.
        const std::size_t maskSum = selBins.size();
        int nSel = static_cast<int>(
            rintToEven(kQuantile * static_cast<double>(maskSum)));
        if (nSel < 1) nSel = 1;

        // `if k < n_bands: sub_band = sub_band[..., :-1, :]` — drop the
        // highest-freq bin of this band (owned by the next band). PARITY: L991.
        std::size_t subRows = selBins.size();
        if (k < kNBands && subRows > 0) --subRows;

        // Clamp nSel to subRows if the extended count was tiny (edge case:
        // first band with very few bins). librosa's algorithm allows
        // nSel > subRows? Actually `sortedr[:nSel]` with nSel > len just
        // returns all rows — numpy slice is lenient. We mirror with clamp.
        int nSelEff = nSel;
        if (static_cast<std::size_t>(nSelEff) > subRows)
            nSelEff = static_cast<int>(subRows);
        if (nSelEff < 1) nSelEff = 1;

        // Per-frame: gather sub-band column, sort ascending, mean top/bottom nSel.
        column.resize(subRows);
        for (std::size_t t = 0; t < nFrames; ++t) {
            const auto& sCol = S[t];
            if (sCol.size() != nBins) continue;

            for (std::size_t r = 0; r < subRows; ++r)
                column[r] = sCol[selBins[r]];

            // np.sort(axis=-2) is ascending, default kind=quicksort (stable not
            // required — values, not indices, are the output). Use std::sort.
            std::sort(column.begin(), column.end());

            // Accumulate in float32 to match numpy's mean on float32 input
            // (np.sum uses input dtype as accumulator for float). Then
            // assignment into float64 peak/valley upcasts.
            float valleySum = 0.0f;
            for (int i = 0; i < nSelEff; ++i)
                valleySum += column[static_cast<std::size_t>(i)];
            float peakSum = 0.0f;
            for (int i = 0; i < nSelEff; ++i)
                peakSum += column[subRows - 1 - static_cast<std::size_t>(i)];

            const float invN = 1.0f / static_cast<float>(nSelEff);
            const std::size_t idxRow =
                static_cast<std::size_t>(k) * nFrames + t;
            valley[idxRow] = static_cast<double>(valleySum * invN);
            peak  [idxRow] = static_cast<double>(peakSum   * invN);
        }
    }

    // contrast = power_to_db(peak) - power_to_db(valley). PARITY: L1001-1004.
    applyPowerToDb(peak,   static_cast<std::size_t>(kNRows), nFrames, kAmin, kTopDb);
    applyPowerToDb(valley, static_cast<std::size_t>(kNRows), nFrames, kAmin, kTopDb);

    // Transpose to [frame][kNRows] float32 output matching our convention
    // (other phase-2 modules return std::vector<std::vector<float>> in that
    // orientation; parity test transposes Python's (rows, frames) dump).
    std::vector<std::vector<float>> out(
        nFrames, std::vector<float>(static_cast<std::size_t>(kNRows), 0.0f));
    for (int k = 0; k < kNRows; ++k) {
        for (std::size_t t = 0; t < nFrames; ++t) {
            const std::size_t idxRow = static_cast<std::size_t>(k) * nFrames + t;
            out[t][static_cast<std::size_t>(k)] =
                static_cast<float>(peak[idxRow] - valley[idxRow]);
        }
    }

    return out;
}

} // namespace reamix::dsp
