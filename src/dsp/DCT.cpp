#include "dsp/DCT.h"

#include <cmath>
#include <stdexcept>

namespace reamix::dsp {

namespace {
constexpr double kPi = 3.14159265358979323846;
} // namespace

DCT::DCT(int n, int m)
    : n_(n), m_(m)
{
    if (n <= 0 || m <= 0 || m > n)
        throw std::invalid_argument("DCT: require 0 < m <= n");

    basis_.assign(static_cast<std::size_t>(m_),
                  std::vector<float>(static_cast<std::size_t>(n_), 0.0f));

    // Scales chosen to match scipy.fft.dct(type=2, norm='ortho'):
    //   y[0]   = (1/sqrt(N)) * sum x[n]
    //   y[k>0] = sqrt(2/N)   * sum x[n] * cos(pi (2n+1) k / (2N))
    const double invSqrtN    = 1.0 / std::sqrt(static_cast<double>(n_));
    const double sqrt2OverN  = std::sqrt(2.0 / static_cast<double>(n_));
    const double halfPiOverN = kPi / (2.0 * static_cast<double>(n_));

    for (int k = 0; k < m_; ++k)
    {
        const double scale = (k == 0) ? invSqrtN : sqrt2OverN;
        for (int n = 0; n < n_; ++n)
        {
            const double c = std::cos(halfPiOverN
                                      * static_cast<double>(2 * n + 1)
                                      * static_cast<double>(k));
            basis_[static_cast<std::size_t>(k)][static_cast<std::size_t>(n)]
                = static_cast<float>(scale * c);
        }
    }
}

std::vector<float> DCT::apply(const std::vector<float>& x) const
{
    if (static_cast<int>(x.size()) != n_)
        throw std::invalid_argument("DCT::apply: input size mismatch");

    std::vector<float> y(static_cast<std::size_t>(m_), 0.0f);
    for (int k = 0; k < m_; ++k)
    {
        const auto& row = basis_[static_cast<std::size_t>(k)];
        // Naive float32 accumulator — matches scipy.fft.dct's float32 path
        // within a few ULP. BLAS / Kahan / pairwise sum would diverge by
        // the amount they claim to fix (see MelSpectrogramLibrosa note).
        float acc = 0.0f;
        for (int n = 0; n < n_; ++n)
            acc += row[static_cast<std::size_t>(n)] * x[static_cast<std::size_t>(n)];
        y[static_cast<std::size_t>(k)] = acc;
    }
    return y;
}

std::vector<std::vector<float>>
DCT::applyFrames(const std::vector<std::vector<float>>& frames) const
{
    std::vector<std::vector<float>> out;
    out.reserve(frames.size());
    for (const auto& f : frames)
        out.push_back(apply(f));
    return out;
}

} // namespace reamix::dsp
