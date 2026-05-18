#pragma once

#include <vector>

namespace reamix::dsp {

// Type-II DCT with 'ortho' normalization, matching
// `scipy.fft.dct(x, type=2, norm='ortho')` on librosa 0.11.0's call site
// inside `librosa.feature.mfcc` (axis=-2 along the n_mels axis).
//
//   y[0]   = (1 / sqrt(N)) * sum_{n=0}^{N-1} x[n]
//   y[k>0] = sqrt(2 / N)   * sum_{n=0}^{N-1} x[n] * cos(pi (2n+1) k / (2 N))
//
// Direct cosine-matrix multiply, O(N·M) per transform. For the standard
// MFCC path N=128, M=40 → ~5 k fmadds per frame — negligible next to STFT.
// Cosine matrix is built in double precision then cast-stored to float32
// once at construction, matching scipy.fft.dct's behavior on float32 input
// (trig table in double, product accumulated in input dtype).
//
// Used by analysis::Mfcc downstream of MelSpectrogramLibrosa::powerToDb.
// PARITY: scipy.fft.dct(type=2, norm='ortho') called from
// librosa/feature/spectral.py `mfcc()`.

class DCT
{
public:
    // n = input length (e.g. 128 = n_mels). m = number of output coefficients
    // kept (e.g. 40 = n_mfcc standard). Requires 0 < m ≤ n.
    DCT(int n, int m);

    int inputSize()  const noexcept { return n_; }
    int outputSize() const noexcept { return m_; }

    // Apply DCT to a single vector of length n_. Output length m_.
    std::vector<float> apply(const std::vector<float>& x) const;

    // Per-frame batch: [frame][n_] → [frame][m_]. librosa's mfcc applies the
    // DCT along axis=-2 (the n_mels axis) of a (n_mels, n_frames) matrix,
    // which is per-frame in our [frame][mel] layout.
    std::vector<std::vector<float>>
    applyFrames(const std::vector<std::vector<float>>& frames) const;

    // Cosine matrix, [m_][n_], float32. Exposed for parity debugging.
    const std::vector<std::vector<float>>& basis() const noexcept { return basis_; }

private:
    int n_;
    int m_;
    std::vector<std::vector<float>> basis_;  // [m_][n_]
};

} // namespace reamix::dsp
