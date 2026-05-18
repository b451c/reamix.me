#include "dsp/MelSpectrogramLibrosa.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace reamix::dsp {

namespace {

// Slaney-style Hz ↔ mel conversion, matching
// librosa.convert.hz_to_mel / mel_to_hz (htk=False). All computation in
// double — librosa does the same via numpy scalars before the final
// cast-to-float32 for the filter weights.
//
// PARITY: librosa/convert.py (0.11.0) — hz_to_mel / mel_to_hz with htk=False.

constexpr double kFSp = 200.0 / 3.0;
constexpr double kMinLogHz = 1000.0;
constexpr double kMinLogMel = kMinLogHz / kFSp;  // 15.0
// logstep = np.log(6.4) / 27.0; numpy uses natural log — we match.
static const double kLogStep = std::log(6.4) / 27.0;

double hzToMel(double hz) noexcept
{
    if (hz >= kMinLogHz)
        return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
    return hz / kFSp;
}

double melToHz(double mel) noexcept
{
    if (mel >= kMinLogMel)
        return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
    return kFSp * mel;
}

} // namespace

MelSpectrogramLibrosa::MelSpectrogramLibrosa()
{
    // FFT bin center frequencies: np.linspace(0, sr/2, 1 + n_fft/2).
    std::vector<double> fftFreqs(kNumBins);
    const double nyquist = static_cast<double>(kSampleRate) / 2.0;
    for (int k = 0; k < kNumBins; ++k)
        fftFreqs[k] = nyquist * static_cast<double>(k) / static_cast<double>(kNumBins - 1);

    // Mel edge frequencies: linspace(min_mel, max_mel, n_mels+2) → mel_to_hz.
    // fmin=0, fmax=sr/2 (librosa melspectrogram defaults when fmax not given).
    const double melMin = hzToMel(0.0);
    const double melMax = hzToMel(nyquist);
    std::vector<double> melF(kNMels + 2);
    for (int i = 0; i < kNMels + 2; ++i)
    {
        const double mel = melMin + (melMax - melMin)
                                    * static_cast<double>(i)
                                    / static_cast<double>(kNMels + 1);
        melF[i] = melToHz(mel);
    }

    // Step 1: triangular weights, computed in float64, stored float32.
    // Matches numpy: weights[i] = np.maximum(0, np.minimum(lower, upper))
    // where the assignment to a float32 row casts each element down.
    melBasis_.assign(kNMels, std::vector<float>(kNumBins, 0.0f));
    for (int m = 0; m < kNMels; ++m)
    {
        const double lo = melF[m];
        const double ctr = melF[m + 1];
        const double hi = melF[m + 2];
        for (int k = 0; k < kNumBins; ++k)
        {
            const double lower = (fftFreqs[k] - lo) / (ctr - lo);
            const double upper = (hi - fftFreqs[k]) / (hi - ctr);
            const double w = std::max(0.0, std::min(lower, upper));
            melBasis_[m][k] = static_cast<float>(w);
        }
    }

    // Step 2: Slaney norm, applied in-place as float32 *= float64 (librosa's
    // `weights *= enorm[:, np.newaxis]` where weights is float32 and enorm
    // is float64). Each product rounds to float32 individually.
    // PARITY: librosa/filters.py `mel(... norm='slaney')` path.
    for (int m = 0; m < kNMels; ++m)
    {
        const double enorm = 2.0 / (melF[m + 2] - melF[m]);
        for (int k = 0; k < kNumBins; ++k)
        {
            const double scaled = static_cast<double>(melBasis_[m][k]) * enorm;
            melBasis_[m][k] = static_cast<float>(scaled);
        }
    }
}

std::vector<std::vector<float>>
MelSpectrogramLibrosa::power(const std::vector<float>& y) const
{
    // librosa.feature.melspectrogram(power=2.0):
    //   S = |stft(y)|              (float32 magnitude)
    //   S **= power  (= 2.0)       (float32 power)
    //   return einsum("ft,mf->mt", S, mel_basis, optimize=True)   (float32)
    //
    // We keep the layout [frame][mel] internally; the parity test transposes
    // on read.
    auto mag = stft_.magnitude(y);  // [frame][bin], float
    const int nFrames = static_cast<int>(mag.size());

    std::vector<std::vector<float>> out(nFrames, std::vector<float>(kNMels, 0.0f));

    std::vector<float> powerSpec(kNumBins);
    for (int f = 0; f < nFrames; ++f)
    {
        // Square in float32 (numpy S **= 2.0 on float32 array).
        for (int k = 0; k < kNumBins; ++k)
        {
            const float v = mag[f][k];
            powerSpec[k] = v * v;
        }

        // mel_basis @ power: naive dot, float32 accumulator. Accumulator
        // ordering differs from a BLAS dot product by a few ULP; acceptable
        // at the float32 floor — see validation.md ULP-floor note.
        for (int m = 0; m < kNMels; ++m)
        {
            const auto& row = melBasis_[m];
            float acc = 0.0f;
            for (int k = 0; k < kNumBins; ++k)
                acc += row[k] * powerSpec[k];
            out[f][m] = acc;
        }
    }

    return out;
}

std::vector<std::vector<float>>
MelSpectrogramLibrosa::powerToDb(const std::vector<std::vector<float>>& melPower,
                                 float ref) const
{
    // librosa.power_to_db(S, ref=<ref>, amin=1e-10, top_db=80):
    //   log_spec = 10 * log10(max(amin, S)) - 10 * log10(max(amin, abs(ref)))
    //   log_spec = max(log_spec, log_spec.max() - top_db)
    //
    // All ops in float32 (input is float32 mel_power; numpy keeps dtype).

    // ref ≤ 0 → "use max" (librosa ref=np.max). fall back to amin for the
    // all-silence edge case (librosa clamps via np.maximum(amin, ref_value)).
    float refValue;
    if (ref <= 0.0f) {
        float refMax = 0.0f;
        for (const auto& frame : melPower)
            for (float v : frame)
                if (v > refMax) refMax = v;
        refValue = refMax;
    } else {
        refValue = ref;
    }

    const float refClamped = std::max(kAmin, refValue);
    const float refLog = 10.0f * std::log10(refClamped);

    const std::size_t nFrames = melPower.size();
    std::vector<std::vector<float>> out(nFrames, std::vector<float>(kNMels, 0.0f));

    float logSpecMax = -std::numeric_limits<float>::infinity();
    for (std::size_t f = 0; f < nFrames; ++f)
    {
        for (int m = 0; m < kNMels; ++m)
        {
            const float clamped = std::max(kAmin, melPower[f][m]);
            const float lv = 10.0f * std::log10(clamped) - refLog;
            out[f][m] = lv;
            if (lv > logSpecMax) logSpecMax = lv;
        }
    }

    // Clip at log_spec.max() - top_db, matching librosa's two-step
    // (max-after-subtraction, then floor).
    const float floor = logSpecMax - kTopDbFloor;
    for (auto& frame : out)
        for (auto& v : frame)
            if (v < floor) v = floor;

    return out;
}

std::vector<std::vector<float>>
MelSpectrogramLibrosa::logMelDb(const std::vector<float>& y) const
{
    return powerToDb(power(y));
}

} // namespace reamix::dsp
