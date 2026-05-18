// Crossfade — multi-band adaptive crossfade (port of crossfade.py:219-332 with
// ADR-032 Option B.myfix on the phase-align pad-back branch) + simple_crossfade
// single-band fallback (port of crossfade.py:335-366, session 33 SB-9 closure).
//
// Parity class target: per-sample max_abs ≤ 1e-9 vs Python goldens.
// adaptive: fixed-Python (tools/dump_phase5_crossfade.py monkey-patches the same
//           fix locally — ADR-032). simple: unmodified Python (no bug).

#include "render/Crossfade.h"

#include "dsp/Butterworth.h"
#include "render/PhaseAlign.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace reamix::render {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Numerics inventory (crossfade.py):
//   0.35   L294, L360 — lower clamp (-9 dB)          [adaptive + simple]
//   2.83   L294, L360 — upper clamp (+9 dB)          [adaptive + simple]
//   1e-10  L291-292, L354,L357 — RMS sentinel        [adaptive + simple]
//   1e-6   L293, L359 — gain-apply threshold         [adaptive + simple]
//   75.0   L31 — DEFAULT_CROSSFADE_MS                [simple only — kDefaultCrossfadeMs in .h]
//   4      L362 — simple_crossfade ramp_len multiplier (n_overlap * 4) [simple only]
//
// All numerics are verbatim.

constexpr double kRmsClampLo        = 0.35;
constexpr double kRmsClampHi        = 2.83;
constexpr double kRmsSentinel       = 1e-10;
constexpr double kGainThreshold     = 1e-6;
constexpr std::size_t kRampMultiplier = 4;  // simple_crossfade ramp_len = min(n_overlap * 4, inLen)

// ---------------------------------------------------------------------------
// Band filter design dispatch.
// Returns nSections for the chosen type. Matches crossfade.py::_bandpass L58-71.
// If both lowHz<=0 AND highHz<0 (Python's `low_hz<=0 and high_hz is None`),
// caller handles "identity band" (copy-through); this function is not called.
// ---------------------------------------------------------------------------
std::size_t designBandpassSos(int         lowHz,
                              int         highHz,
                              int         sr,
                              double*     sosOut /* at least kMaxSections*6 */)
{
    const double nyq = 0.5 * static_cast<double>(sr);
    std::size_t nSections = 0;
    if (lowHz <= 0) {
        // Lowpass at highHz.
        reamix::dsp::Butterworth::designLowpass(
            4, static_cast<double>(highHz) / nyq,
            sosOut, &nSections);
    } else if (highHz < 0 || static_cast<double>(highHz) >= nyq) {
        // Highpass at lowHz (Python: high_hz is None OR high_hz >= nyq).
        reamix::dsp::Butterworth::designHighpass(
            4, static_cast<double>(lowHz) / nyq,
            sosOut, &nSections);
    } else {
        // Bandpass [lowHz, highHz].
        reamix::dsp::Butterworth::designBandpass(
            4,
            static_cast<double>(lowHz) / nyq,
            static_cast<double>(highHz) / nyq,
            sosOut, &nSections);
    }
    return nSections;
}

// Per-channel sosfiltfilt — matches the Python loop at L77-80 of _bandpass.
// Input and output both (nCh × nSamp) row-major. No aliasing required.
void bandpassMulti(const double* sos, std::size_t nSections,
                   const double* in_,
                   std::size_t   nCh,
                   std::size_t   nSamp,
                   double*       out)
{
    for (std::size_t ch = 0; ch < nCh; ++ch) {
        reamix::dsp::Butterworth::sosFiltFilt(
            sos, nSections,
            in_ + ch * nSamp, nSamp,
            out + ch * nSamp);
    }
}

// Equal-power fade curves (crossfade.py:159-166). t = linspace(0, pi/2, n).
// fade_out[i] = cos(t[i]); fade_in[i] = sin(t[i]).
// np.linspace(0, pi/2, n) semantics:
//   n == 0  → empty
//   n == 1  → [0.0]
//   n >= 2  → [0, step, 2*step, ..., pi/2]  with step = (pi/2) / (n-1)
void equalPowerFade(std::size_t n, double* fadeOut, double* fadeIn)
{
    if (n == 0) return;
    if (n == 1) {
        fadeOut[0] = 1.0;  // cos(0) = 1
        fadeIn[0]  = 0.0;  // sin(0) = 0
        return;
    }
    const double step = (kPi * 0.5) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = step * static_cast<double>(i);
        fadeOut[i] = std::cos(t);
        fadeIn[i]  = std::sin(t);
    }
}

// Apply integer shift to a multi-channel buffer IN PLACE (ADR-032 Option B.myfix).
//
// Semantics mirror crossfade.py::_phase_align shift-apply branch (L139-151) but
// applied to the full `in_band` buffer (length max_xf per channel) rather than
// to a xf_samples slice. Post-condition: inBand contains the shifted copy,
// with the freed region (|shift| samples) zeroed.
//
//   shift > 0 : left-shift; inBand[:-shift] = inBand[shift:]; inBand[-shift:] = 0
//   shift < 0 : right-shift; inBand[-shift:] = inBand[:shift]; inBand[:-shift] = 0
//   shift = 0 : no-op.
//
// Uses a temporary buffer per channel (stack-spillover ok at max_xf <= 8820 f64).
// Could be done with memmove if we were careful about direction, but memmove
// semantics with overlapping regions + zero-fill is easier to reason about
// through a scratch copy.
void applyShiftFull(double*     inBand,
                    std::size_t nCh,
                    std::size_t nSamp,
                    int         shift)
{
    if (shift == 0 || nSamp == 0) return;
    if (shift >= static_cast<int>(nSamp) || -shift >= static_cast<int>(nSamp)) {
        // Shift larger than buffer: everything becomes zero. Matches numpy
        // slicing (out-of-range left/right slice leaves result fully zero).
        std::memset(inBand, 0, nCh * nSamp * sizeof(double));
        return;
    }
    std::vector<double> scratch(nSamp);
    if (shift > 0) {
        const std::size_t s = static_cast<std::size_t>(shift);
        for (std::size_t ch = 0; ch < nCh; ++ch) {
            double* row = inBand + ch * nSamp;
            std::memcpy(scratch.data(), row + s, (nSamp - s) * sizeof(double));
            std::memcpy(row, scratch.data(), (nSamp - s) * sizeof(double));
            std::memset(row + (nSamp - s), 0, s * sizeof(double));
        }
    } else {
        const std::size_t s = static_cast<std::size_t>(-shift);
        for (std::size_t ch = 0; ch < nCh; ++ch) {
            double* row = inBand + ch * nSamp;
            std::memcpy(scratch.data(), row, (nSamp - s) * sizeof(double));
            std::memset(row, 0, s * sizeof(double));
            std::memcpy(row + s, scratch.data(), (nSamp - s) * sizeof(double));
        }
    }
}

// Compute max_xf_samples per crossfade.py:252-253.
std::size_t maxXfSamples(const Crossfade::Band* bands, std::size_t nBands,
                         int sr,
                         std::size_t outLen, std::size_t inLen)
{
    int mxfSamp = 0;
    for (std::size_t b = 0; b < nBands; ++b) {
        const int s = static_cast<int>(bands[b].xfMs * sr / 1000.0);
        if (s > mxfSamp) mxfSamp = s;
    }
    std::size_t m = static_cast<std::size_t>(mxfSamp > 0 ? mxfSamp : 0);
    m = std::min(m, outLen);
    m = std::min(m, inLen);
    return m;
}

// RMS over an entire multi-channel buffer (crossfade.py:291-292 semantics):
//   sqrt(mean(x^2) + 1e-10)
// numpy np.mean on 2D array averages over ALL elements (flatten-mean).
double rmsMultiChannel(const double* x, std::size_t nCh, std::size_t nSamp)
{
    const std::size_t n = nCh * nSamp;
    double sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) sumsq += x[i] * x[i];
    const double mean = (n > 0) ? (sumsq / static_cast<double>(n)) : 0.0;
    return std::sqrt(mean + kRmsSentinel);
}

// Scale a multi-channel buffer by a scalar (in-place).
void scaleInPlace(double* x, std::size_t n, double gain)
{
    for (std::size_t i = 0; i < n; ++i) x[i] *= gain;
}

// RMS over the LEADING `sliceLen` samples of each row (matches Python
// `np.sqrt(np.mean(x[..., :sliceLen] ** 2) + 1e-10)` used in simple_crossfade).
// `rowLen` is the full stride per channel; the trailing samples are ignored.
double rmsLeading(const double* x, std::size_t nCh, std::size_t rowLen, std::size_t sliceLen)
{
    double sumsq = 0.0;
    for (std::size_t ch = 0; ch < nCh; ++ch) {
        const double* p = x + ch * rowLen;
        for (std::size_t i = 0; i < sliceLen; ++i) sumsq += p[i] * p[i];
    }
    const std::size_t n = nCh * sliceLen;
    const double mean = (n > 0) ? (sumsq / static_cast<double>(n)) : 0.0;
    return std::sqrt(mean + kRmsSentinel);
}

// RMS over the TRAILING `sliceLen` samples of each row (matches Python
// `np.sqrt(np.mean(x[..., -sliceLen:] ** 2) + 1e-10)`).
double rmsTrailing(const double* x, std::size_t nCh, std::size_t rowLen, std::size_t sliceLen)
{
    double sumsq = 0.0;
    for (std::size_t ch = 0; ch < nCh; ++ch) {
        const double* p = x + ch * rowLen + (rowLen - sliceLen);
        for (std::size_t i = 0; i < sliceLen; ++i) sumsq += p[i] * p[i];
    }
    const std::size_t n = nCh * sliceLen;
    const double mean = (n > 0) ? (sumsq / static_cast<double>(n)) : 0.0;
    return std::sqrt(mean + kRmsSentinel);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::size_t Crossfade::computeResultLen(std::size_t outLen,
                                        std::size_t inLen,
                                        int         sr,
                                        const Band* bands,
                                        std::size_t nBands)
{
    const Band* b = bands ? bands : kDefaultBands;
    const std::size_t nb = bands ? nBands : 3;
    const std::size_t mxf = maxXfSamples(b, nb, sr, outLen, inLen);
    if (mxf == 0) return outLen + inLen;  // matches np.concatenate fallback
    return outLen + inLen - mxf;
}

std::size_t Crossfade::computeSimpleResultLen(std::size_t outLen,
                                              std::size_t inLen,
                                              int         sr,
                                              double      crossfadeMs)
{
    // n_overlap = min(int(crossfade_ms * sr / 1000.0), outLen, inLen)  (L346-350)
    const int requested = static_cast<int>(crossfadeMs * sr / 1000.0);
    std::size_t nOverlap = (requested > 0) ? static_cast<std::size_t>(requested) : 0;
    nOverlap = std::min(nOverlap, outLen);
    nOverlap = std::min(nOverlap, inLen);
    // _crossfade_segment L184-185: overlap <= 0 → np.concatenate fallback.
    if (nOverlap == 0) return outLen + inLen;
    // _crossfade_segment L193: result_len = outLen + inLen - overlap.
    return outLen + inLen - nOverlap;
}

std::size_t Crossfade::adaptiveCrossfade(
    const double* outgoing,  std::size_t nChannels, std::size_t outLen,
    const double* incoming,  std::size_t inLen,
    int           sr,
    const Band*   bands,     std::size_t nBands,
    bool          energyCompensate,
    bool          phaseAlign,
    double        maxPhaseShiftMs,
    double*       resultOut,
    std::size_t   resultCapacityPerChannel)
{
    const Band* bArr = bands ? bands : kDefaultBands;
    const std::size_t nb = bands ? nBands : 3;

    const int maxPhaseShiftSamples =
        static_cast<int>(maxPhaseShiftMs * sr / 1000.0);

    const std::size_t mxf = maxXfSamples(bArr, nb, sr, outLen, inLen);
    const std::size_t resultLen =
        (mxf == 0) ? (outLen + inLen) : (outLen + inLen - mxf);

    assert(resultCapacityPerChannel >= resultLen &&
           "resultCapacityPerChannel < computeResultLen()");
    (void) resultCapacityPerChannel;

    // Zero result first (Python: np.zeros(shape, dtype=np.float64)).
    std::memset(resultOut, 0, nChannels * resultLen * sizeof(double));

    // Fallback path: no overlap → simple concat (L255-256).
    if (mxf == 0) {
        for (std::size_t ch = 0; ch < nChannels; ++ch) {
            std::memcpy(resultOut + ch * resultLen,
                        outgoing + ch * outLen,
                        outLen * sizeof(double));
            std::memcpy(resultOut + ch * resultLen + outLen,
                        incoming + ch * inLen,
                        inLen * sizeof(double));
        }
        return resultLen;
    }

    // Non-overlap regions of result (L263-272).
    const std::size_t nonOverlapOut = outLen - mxf;
    for (std::size_t ch = 0; ch < nChannels; ++ch) {
        // result[..., :non_overlap_out] = outgoing[..., :non_overlap_out]
        if (nonOverlapOut > 0) {
            std::memcpy(resultOut + ch * resultLen,
                        outgoing + ch * outLen,
                        nonOverlapOut * sizeof(double));
        }
        // result[..., non_overlap_out + max_xf:] = incoming[..., max_xf:]
        const std::size_t remainingIn = (inLen > mxf) ? (inLen - mxf) : 0;
        if (remainingIn > 0) {
            std::memcpy(resultOut + ch * resultLen + nonOverlapOut + mxf,
                        incoming + ch * inLen + mxf,
                        remainingIn * sizeof(double));
        }
    }

    // out_overlap = outgoing[..., -max_xf:].astype(f64)   (L275)
    // in_overlap  = incoming[..., :max_xf].astype(f64)    (L276)
    std::vector<double> outOverlap(nChannels * mxf);
    std::vector<double> inOverlap (nChannels * mxf);
    for (std::size_t ch = 0; ch < nChannels; ++ch) {
        std::memcpy(outOverlap.data() + ch * mxf,
                    outgoing + ch * outLen + (outLen - mxf),
                    mxf * sizeof(double));
        std::memcpy(inOverlap.data() + ch * mxf,
                    incoming + ch * inLen,
                    mxf * sizeof(double));
    }

    // overlap_result = zeros_like(out_overlap)   (L278)
    std::vector<double> overlapResult(nChannels * mxf, 0.0);

    // Per-band loop (L280-329).
    std::vector<double> outBand(nChannels * mxf);
    std::vector<double> inBand (nChannels * mxf);
    std::vector<double> sos(reamix::dsp::Butterworth::kMaxSections * 6, 0.0);
    std::vector<double> fadeOut, fadeIn;
    fadeOut.reserve(mxf);
    fadeIn.reserve(mxf);

    for (std::size_t b = 0; b < nb; ++b) {
        const Band& band = bArr[b];

        // xf_samples = min(int(xf_ms * sr / 1000.0), max_xf_samples)   (L281)
        std::size_t xf = static_cast<std::size_t>(
            static_cast<int>(band.xfMs * sr / 1000.0));
        xf = std::min(xf, mxf);
        if (xf == 0) continue;

        // Identity-band short-circuit (Python _bandpass L58-59 — lowHz<=0 AND
        // highHz is None). Passes audio unchanged. Not reachable with
        // kDefaultBands; kept for caller-specified band safety.
        const bool identityBand = (band.lowHz <= 0 && band.highHz < 0);

        if (identityBand) {
            std::memcpy(outBand.data(), outOverlap.data(),
                        nChannels * mxf * sizeof(double));
            std::memcpy(inBand.data(), inOverlap.data(),
                        nChannels * mxf * sizeof(double));
        } else {
            const std::size_t nSections = designBandpassSos(
                band.lowHz, band.highHz, sr, sos.data());
            bandpassMulti(sos.data(), nSections,
                          outOverlap.data(), nChannels, mxf,
                          outBand.data());
            bandpassMulti(sos.data(), nSections,
                          inOverlap.data(), nChannels, mxf,
                          inBand.data());
        }

        // Energy compensation (L289-295).
        if (energyCompensate) {
            const double rmsOut = rmsMultiChannel(outBand.data(), nChannels, mxf);
            const double rmsIn  = rmsMultiChannel(inBand.data(),  nChannels, mxf);
            if (rmsOut > kGainThreshold && rmsIn > kGainThreshold) {
                double gain = rmsOut / rmsIn;
                if      (gain < kRmsClampLo) gain = kRmsClampLo;
                else if (gain > kRmsClampHi) gain = kRmsClampHi;
                scaleInPlace(inBand.data(), nChannels * mxf, gain);
            }
        }

        // ADR-032 Option B.myfix — compute shift on fade-zone pair,
        // apply to FULL in_band (not slice+pad-back).
        if (phaseAlign && maxPhaseShiftSamples > 0) {
            // Shift input: out_band[..., -xf:] vs in_band[..., :xf], first channel
            // used for NCC (matches Python _phase_align multi-channel branch L98-101).
            const double* outFade = outBand.data() + 0 * mxf + (mxf - xf);  // ch0 last xf
            const double* inFade  = inBand.data()  + 0 * mxf;               // ch0 first xf
            int shift = 0;
            // alignedOut arg of alignMono is required; we discard the aligned
            // buffer (we only want the shift scalar) but must pass a valid
            // destination. Use a stack-size-safe scratch buffer.
            std::vector<double> alignedScratch(xf);
            reamix::render::PhaseAlign::alignMono(
                outFade, xf,
                inFade,  xf,
                maxPhaseShiftSamples,
                &shift,
                alignedScratch.data());
            // Apply shift to full-length in_band (all channels).
            applyShiftFull(inBand.data(), nChannels, mxf, shift);
        }

        // Envelope build (L312-327). pad_before = (max_xf - xf) // 2.
        const std::size_t padBefore = (mxf - xf) / 2;
        fadeOut.resize(xf);
        fadeIn.resize(xf);
        equalPowerFade(xf, fadeOut.data(), fadeIn.data());

        // env_out / env_in live as float envelopes of length mxf. Rather than
        // materialize them, we inline the multiplies per region.
        //
        //   region A [0 : pad_before]         : env_out = 1.0  env_in = 0.0
        //   region B [pad_before : pad_before+xf] : env_out = fade_out  env_in = fade_in
        //   region C [pad_before+xf : mxf]   : env_out = 0.0  env_in = 1.0
        //
        // overlap_result += out_band * env_out + in_band * env_in.

        for (std::size_t ch = 0; ch < nChannels; ++ch) {
            double* ovRow = overlapResult.data() + ch * mxf;
            const double* outRow = outBand.data() + ch * mxf;
            const double* inRow  = inBand.data()  + ch * mxf;

            // region A
            for (std::size_t i = 0; i < padBefore; ++i) {
                ovRow[i] += outRow[i];
            }
            // region B
            for (std::size_t i = 0; i < xf; ++i) {
                const std::size_t idx = padBefore + i;
                ovRow[idx] += outRow[idx] * fadeOut[i]
                            + inRow[idx]  * fadeIn[i];
            }
            // region C
            const std::size_t cStart = padBefore + xf;
            for (std::size_t i = cStart; i < mxf; ++i) {
                ovRow[i] += inRow[i];
            }
        }
    }

    // Write overlap_result into result[..., non_overlap_out : non_overlap_out + mxf]
    for (std::size_t ch = 0; ch < nChannels; ++ch) {
        std::memcpy(resultOut + ch * resultLen + nonOverlapOut,
                    overlapResult.data() + ch * mxf,
                    mxf * sizeof(double));
    }

    return resultLen;
}

// ---------------------------------------------------------------------------
// simpleCrossfade — port of crossfade.py:335-366 (single-band fallback).
//
// Sequence matches Python exactly:
//   1. n_overlap = min(int(crossfade_ms * sr / 1000), outLen, inLen)    (L346-350)
//   2. if energy_compensate and n_overlap > 0:                          (L352)
//        rms_out = sqrt(mean(outgoing[-n_overlap:]^2) + 1e-10)          (L353-355)
//        rms_in  = sqrt(mean(incoming[:n_overlap]^2)  + 1e-10)          (L356-358)
//        if rms_out > 1e-6 and rms_in > 1e-6:                           (L359)
//            gain = clip(rms_out / rms_in, 0.35, 2.83)                  (L360)
//            incoming = incoming.copy()                                 (L361)
//            ramp_len = min(n_overlap * 4, inLen)                       (L362)
//            ramp = linspace(gain, 1.0, ramp_len)                       (L363)
//            incoming[:ramp_len] *= ramp (broadcast across channels)    (L364)
//   3. return _crossfade_segment(outgoing, incoming, n_overlap)         (L366)
//
// _crossfade_segment (L169-213):
//   if overlap <= 0: return concatenate(out, in)
//   fade_out, fade_in = equal_power_fade(overlap)
//   result_len = outLen + inLen - overlap
//   non_overlap = outLen - overlap   (head of outgoing, unmodified)
//   fade-zone:  result[xf_start : xf_start+overlap] = out[-overlap:]*fade_out
//                                                    + in[:overlap]*fade_in
//   tail:       result[xf_start+overlap :] = in[overlap:]
// ---------------------------------------------------------------------------
std::size_t Crossfade::simpleCrossfade(
    const double* outgoing, std::size_t nChannels, std::size_t outLen,
    const double* incoming, std::size_t inLen,
    int           sr,
    double        crossfadeMs,
    bool          energyCompensate,
    double*       resultOut,
    std::size_t   resultCapacityPerChannel)
{
    // Step 1 — n_overlap computation (L346-350).
    const int requested = static_cast<int>(crossfadeMs * sr / 1000.0);
    std::size_t nOverlap = (requested > 0) ? static_cast<std::size_t>(requested) : 0;
    nOverlap = std::min(nOverlap, outLen);
    nOverlap = std::min(nOverlap, inLen);

    // Step 2 — optional RMS compensation + linear ramp on incoming (L352-364).
    // `inBuf` aliases `incoming` unless we need to modify it (gain ramp applied).
    const double* inBuf = incoming;
    std::vector<double> inCopy;

    if (energyCompensate && nOverlap > 0) {
        // rms_out = sqrt(mean(outgoing[-n_overlap:]^2) + 1e-10)    (L353-355)
        const double rmsOut = rmsTrailing(outgoing, nChannels, outLen, nOverlap);
        // rms_in  = sqrt(mean(incoming[:n_overlap]^2)  + 1e-10)    (L356-358)
        const double rmsIn  = rmsLeading (incoming, nChannels, inLen,  nOverlap);

        if (rmsOut > kGainThreshold && rmsIn > kGainThreshold) {
            // gain = clip(rms_out / rms_in, 0.35, 2.83)            (L360)
            double gain = rmsOut / rmsIn;
            if      (gain < kRmsClampLo) gain = kRmsClampLo;
            else if (gain > kRmsClampHi) gain = kRmsClampHi;

            // incoming = incoming.copy()                           (L361)
            inCopy.assign(incoming, incoming + nChannels * inLen);
            inBuf = inCopy.data();

            // ramp_len = min(n_overlap * 4, inLen)                 (L362)
            const std::size_t rampLen = std::min(nOverlap * kRampMultiplier, inLen);

            // ramp = np.linspace(gain, 1.0, ramp_len)              (L363)
            // numpy linspace ops: step = (stop-start)/(num-1); y[i] = start + step*i;
            //                     y[-1] forced to stop (endpoint=True).
            //   rampLen == 0 → empty; no-op.
            //   rampLen == 1 → [gain].
            //   rampLen >= 2 → [gain, gain+step, ..., 1.0].
            //
            // incoming[:ramp_len] *= ramp (broadcasts across channels — all channels
            //                              get the same factor at each sample index). (L364)
            if (rampLen == 1) {
                for (std::size_t ch = 0; ch < nChannels; ++ch) {
                    inCopy[ch * inLen + 0] *= gain;
                }
            } else if (rampLen >= 2) {
                const double step = (1.0 - gain) / static_cast<double>(rampLen - 1);
                for (std::size_t i = 0; i < rampLen; ++i) {
                    const double r = (i == rampLen - 1)
                                   ? 1.0                             // endpoint exact
                                   : gain + step * static_cast<double>(i);
                    for (std::size_t ch = 0; ch < nChannels; ++ch) {
                        inCopy[ch * inLen + i] *= r;
                    }
                }
            }
        }
    }

    // Step 3 — _crossfade_segment(outgoing, inBuf, nOverlap) (L169-213).

    // Fallback path: overlap <= 0 → np.concatenate (L184-185 + L188-189).
    // nOverlap is already clamped to min(..., outLen, inLen), so the inner
    // clamp at L187 is a no-op here.
    if (nOverlap == 0) {
        const std::size_t resultLen = outLen + inLen;
        assert(resultCapacityPerChannel >= resultLen &&
               "resultCapacityPerChannel < computeSimpleResultLen()");
        (void) resultCapacityPerChannel;
        for (std::size_t ch = 0; ch < nChannels; ++ch) {
            std::memcpy(resultOut + ch * resultLen,
                        outgoing + ch * outLen,
                        outLen * sizeof(double));
            std::memcpy(resultOut + ch * resultLen + outLen,
                        inBuf + ch * inLen,
                        inLen * sizeof(double));
        }
        return resultLen;
    }

    // result_len = outLen + inLen - overlap   (L193)
    const std::size_t resultLen = outLen + inLen - nOverlap;
    assert(resultCapacityPerChannel >= resultLen &&
           "resultCapacityPerChannel < computeSimpleResultLen()");
    (void) resultCapacityPerChannel;

    // Zero result first (np.zeros dtype=outgoing.dtype at L195; we're f64 → np.float64).
    std::memset(resultOut, 0, nChannels * resultLen * sizeof(double));

    // fade_out, fade_in = _equal_power_fade(overlap)   (L191)
    std::vector<double> fadeOut(nOverlap);
    std::vector<double> fadeIn (nOverlap);
    equalPowerFade(nOverlap, fadeOut.data(), fadeIn.data());

    const std::size_t nonOverlap = outLen - nOverlap;  // L198
    const std::size_t xfStart    = nonOverlap;         // L203

    for (std::size_t ch = 0; ch < nChannels; ++ch) {
        // Non-overlapping outgoing: result[:non_overlap] = outgoing[:non_overlap]  (L200)
        if (nonOverlap > 0) {
            std::memcpy(resultOut + ch * resultLen,
                        outgoing + ch * outLen,
                        nonOverlap * sizeof(double));
        }

        // Crossfade zone: result[xf_start : xf_start+overlap] =
        //   outgoing[-overlap:] * fade_out + incoming[:overlap] * fade_in   (L204-206)
        const double* outTail = outgoing + ch * outLen + nonOverlap;
        const double* inHead  = inBuf    + ch * inLen;
        double*       dst     = resultOut + ch * resultLen + xfStart;
        for (std::size_t i = 0; i < nOverlap; ++i) {
            dst[i] = outTail[i] * fadeOut[i] + inHead[i] * fadeIn[i];
        }

        // Non-overlapping incoming: result[xf_start+overlap:] = incoming[overlap:]  (L209-211)
        if (inLen > nOverlap) {
            std::memcpy(resultOut + ch * resultLen + xfStart + nOverlap,
                        inHead + nOverlap,
                        (inLen - nOverlap) * sizeof(double));
        }
    }

    return resultLen;
}

} // namespace reamix::render
