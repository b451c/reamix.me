#pragma once

#include "dsp/DCT.h"
#include "dsp/MelSpectrogramLibrosa.h"

#include <vector>

namespace reamix::analysis {

// Mel-Frequency Cepstral Coefficients, matching
// `librosa.feature.mfcc(y=y, sr=22050, n_mfcc=N, n_fft=2048, hop_length=512)`
// with default parameters on librosa 0.11.0 — the exact call site at
// python-source/analysis/feature_extractor.py:307-315.
//
// Pipeline per frame:
//   y → MelSpectrogramLibrosa::power  (mel power, float32 [frame][mel])
//     → powerToDb(ref=1.0)            (log-mel dB, float32 [frame][mel])
//     → DCT-II ortho, slice [:n_mfcc] (cepstrum,  float32 [frame][mfcc])
//
// NOTE: librosa.feature.mfcc internally uses `power_to_db` with default
// ref=1.0 (NOT ref=np.max). Uniform shifts of log-mel only move DCT[0],
// so MFCC[k>0] is invariant to the choice; the port deliberately picks
// ref=1.0 to match librosa's call path bit-for-bit on every coefficient.
//
// Output layout: [frame][mfcc]. Python dumps shape (n_mfcc, n_frames) are
// transposed at the parity test.

class Mfcc
{
public:
    // n_mfcc: number of cepstral coefficients to keep. Standard vector is
    // 40, fast vector is 20 (per phases/phase-2-features/spec.md).
    explicit Mfcc(int nMfcc);

    int nMfcc() const noexcept { return nMfcc_; }

    // Compute MFCC from raw audio. Output [frame][mfcc].
    std::vector<std::vector<float>> compute(const std::vector<float>& y) const;

private:
    int nMfcc_;
    dsp::MelSpectrogramLibrosa mel_;
    dsp::DCT dct_;
};

} // namespace reamix::analysis
