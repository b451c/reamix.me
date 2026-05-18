#pragma once

#include <cstddef>
#include <vector>

namespace reamix::dsp {

// Harmonic-Percussive Source Separation matching `librosa.effects.hpss(y)`
// with librosa 0.11.0 default args:
//   kernel_size=31, power=2.0, margin=1.0, mask=False,
//   n_fft=2048, hop_length=512, win_length=n_fft,
//   window="hann", center=True, pad_mode="constant".
//
// PARITY: librosa/effects.py::hpss (forward STFT → decompose.hpss → inverse
// STFT with length=y.shape[-1]) + librosa/decompose.py::hpss + librosa/util/utils.py::softmask.
// Hard Rule #8 (source-over-docstring): verified against librosa source, not
// docstrings. Two landmines vs the docs:
//   1. `split_zeros = (margin_harm == 1 AND margin_perc == 1)` — with default
//      margin=1.0 this is TRUE (docstring is silent on the default). Zero-energy
//      cells get mask=0.5, NOT 0.0. (Earlier plan in phase-2b/spec.md had
//      this wrong.)
//   2. Median filter is applied to the *magnitude* |S| from `magphase`, NOT
//      to |S|² (power). The soft-mask then raises to `power=2`.
//
// Output is the time-domain harmonic signal y_harmonic of the same length as
// the input y. The percussive path is not exposed — phase-2b vocal pipeline
// only needs harmonic; adding y_perc would be one more ISTFT + softmask and
// cost ~0.3-0.5 s on a 180-s clip (Principle 2: don't pay for speculation).
class HPSS {
public:
    // Compute y_harmonic for input mono audio y (any length >= 1).
    // Returns a vector of the same size as y.
    static std::vector<float> harmonic(const std::vector<float>& y);

    // Kernel size for median filters (time axis [1,k] for harm; freq axis
    // [k,1] for perc). PARITY: librosa.effects.hpss default.
    static constexpr int kKernelSize = 31;
};

} // namespace reamix::dsp
