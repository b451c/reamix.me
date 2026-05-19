#include "dsp/Butterworth.h"

#include "dsp/SosFilt.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace reamix::dsp {

namespace {

using cplx = std::complex<double>;

// Keep the value in a scalar variable to match the one `double`-precision
// constant scipy uses (`numpy.pi` is also an IEEE-754 double, derived from
// a C `M_PI`-equivalent literal). std::numbers::pi_v<double> is identical
// to numpy.pi bit-for-bit on all IEEE-754 conforming platforms.
constexpr double kPi = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// Analog Butterworth prototype (`scipy.signal.buttap`):
//   m = -N+1, -N+3, ..., N-1   (length N)
//   p = -exp(j * pi * m / (2N))
//   z = empty;  k = 1
// Since m is symmetric about 0 and N is even in our use-case, every pole
// comes with a conjugate. We return all N poles (ordering: by ascending m).
// ---------------------------------------------------------------------------
std::vector<cplx> buttap(int N)
{
    std::vector<cplx> p;
    p.reserve(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        const int m = -N + 1 + 2 * k;
        const double angle = kPi * static_cast<double>(m) / (2.0 * static_cast<double>(N));
        // -exp(j*angle) = (-cos(angle)) + j*(-sin(angle))
        p.emplace_back(-std::cos(angle), -std::sin(angle));
    }
    return p;
}

// ---------------------------------------------------------------------------
// Analog prototype transforms (`scipy.signal._filter_design._zpklp2*`).
// All operate on (z, p, k) in place; `degree = len(p) - len(z)` is the
// relative degree used by gain/zero-balancing steps.
// ---------------------------------------------------------------------------
int relativeDegree(const std::vector<cplx>& z, const std::vector<cplx>& p)
{
    return static_cast<int>(p.size()) - static_cast<int>(z.size());
}

// lp2lp: scale poles & zeros radially; gain compensates for scaling.
void lp2lp(std::vector<cplx>& z, std::vector<cplx>& p, double& k, double wo)
{
    const int degree = relativeDegree(z, p);
    for (auto& zi : z) zi *= wo;
    for (auto& pi : p) pi *= wo;
    k *= std::pow(wo, static_cast<double>(degree));
}

// lp2hp: invert radially (p -> wo/p, z -> wo/z); add `degree` zeros at
// the origin; gain multiplier = real(prod(-z_orig) / prod(-p_orig)).
void lp2hp(std::vector<cplx>& z, std::vector<cplx>& p, double& k, double wo)
{
    const int  degree       = relativeDegree(z, p);
    cplx       neg_z_prod(1.0, 0.0);
    cplx       neg_p_prod(1.0, 0.0);
    for (auto& zi : z) { neg_z_prod *= -zi; }
    for (auto& pi : p) { neg_p_prod *= -pi; }
    for (auto& zi : z) zi = wo / zi;
    for (auto& pi : p) pi = wo / pi;
    for (int i = 0; i < degree; ++i) z.emplace_back(0.0, 0.0);
    const cplx ratio = neg_z_prod / neg_p_prod;
    k *= std::real(ratio);
}

// lp2bp: for each pole/zero x, produce digital pair
//   x_lp = x * bw/2
//   x_bp,± = x_lp ± sqrt(x_lp^2 - wo^2)
// Then add `degree` zeros at origin; gain *= bw^degree.
// scipy concatenates as `[+ sqrt block, - sqrt block]` (all `+` first,
// then all `-`); we match the order so downstream _cplxreal sorting is
// over the same input sequence.
void lp2bp(std::vector<cplx>& z, std::vector<cplx>& p, double& k, double wo, double bw)
{
    const int  degree  = relativeDegree(z, p);
    const cplx wo2(wo * wo, 0.0);

    auto transform = [&](const std::vector<cplx>& src) -> std::vector<cplx> {
        std::vector<cplx> out;
        out.reserve(2 * src.size());
        for (const auto& x : src) {
            const cplx x_lp = x * (bw / 2.0);
            const cplx discr = std::sqrt(x_lp * x_lp - wo2);
            out.push_back(x_lp + discr);
        }
        for (const auto& x : src) {
            const cplx x_lp = x * (bw / 2.0);
            const cplx discr = std::sqrt(x_lp * x_lp - wo2);
            out.push_back(x_lp - discr);
        }
        return out;
    };

    std::vector<cplx> z_bp = transform(z);
    std::vector<cplx> p_bp = transform(p);

    for (int i = 0; i < degree; ++i) z_bp.emplace_back(0.0, 0.0);

    z = std::move(z_bp);
    p = std::move(p_bp);
    k *= std::pow(bw, static_cast<double>(degree));
}

// ---------------------------------------------------------------------------
// Bilinear transform (`scipy.signal.bilinear_zpk` with internal fs=2):
//   fs2 = 2*fs = 4
//   z_d = (fs2 + z_a) / (fs2 - z_a), likewise for poles
//   Add `degree` zeros at z=-1 (Nyquist, from analog infinity zeros)
//   k *= real(prod(fs2 - z_a_original) / prod(fs2 - p_a_original))
// ---------------------------------------------------------------------------
void bilinearZpk(std::vector<cplx>& z, std::vector<cplx>& p, double& k)
{
    constexpr double fs2 = 4.0; // 2 * fs, scipy uses fs=2 internally
    const int        degree = relativeDegree(z, p);

    cplx z_corr(1.0, 0.0);
    cplx p_corr(1.0, 0.0);
    for (const auto& zi : z) z_corr *= (fs2 - zi);
    for (const auto& pi : p) p_corr *= (fs2 - pi);

    for (auto& zi : z) zi = (fs2 + zi) / (fs2 - zi);
    for (auto& pi : p) pi = (fs2 + pi) / (fs2 - pi);
    for (int i = 0; i < degree; ++i) z.emplace_back(-1.0, 0.0);

    k *= std::real(z_corr / p_corr);
}

// ---------------------------------------------------------------------------
// _cplxreal: split into complex-conjugate representatives (positive imag)
// and real parts. Ports `scipy.signal._filter_design._cplxreal` with the
// assumptions that hold for Butterworth-via-bilinear outputs:
//   - Exactly conjugate-paired complex values (modulo ULP averaging).
//   - Real values only where exactly real (tol = 100*eps).
// For z and p we call this separately; returned vectors are concatenated
// as `concat(zc, zr)` to feed zpk2sos' main loop.
// ---------------------------------------------------------------------------
struct CplxReal {
    std::vector<cplx>   zc;  // complex, one per conj pair (positive imag)
    std::vector<double> zr;  // real parts, sorted ascending
};

CplxReal cplxReal(const std::vector<cplx>& z)
{
    CplxReal out;
    if (z.empty()) return out;

    const double tol = 100.0 * std::numeric_limits<double>::epsilon();

    // np.lexsort((abs(z.imag), z.real)) — primary key real, secondary |imag|.
    std::vector<cplx> z_sorted = z;
    std::stable_sort(z_sorted.begin(), z_sorted.end(),
        [](const cplx& a, const cplx& b) {
            if (a.real() != b.real()) return a.real() < b.real();
            return std::abs(a.imag()) < std::abs(b.imag());
        });

    std::vector<cplx> complex_only;
    for (const auto& zi : z_sorted) {
        const double amag = std::hypot(zi.real(), zi.imag());
        if (std::abs(zi.imag()) <= tol * amag) {
            out.zr.push_back(zi.real());
        } else {
            complex_only.push_back(zi);
        }
    }
    // zr is already sorted by z_sorted ordering; scipy returns zr sorted
    // ascending without re-sort (the (real, |imag|) ordering collapses to
    // real ordering for reals). Stable-sort once more to be safe against
    // floating-point edge cases.
    std::sort(out.zr.begin(), out.zr.end());

    if (complex_only.empty()) return out;

    std::vector<cplx> zp, zn;
    zp.reserve(complex_only.size() / 2);
    zn.reserve(complex_only.size() / 2);
    for (const auto& zi : complex_only) {
        if (zi.imag() > 0.0) zp.push_back(zi);
        else                 zn.push_back(zi);
    }

    // Both zp and zn inherit (real, |imag|) ascending order. scipy additionally
    // re-sorts runs of equal-real by |imag|; that's a no-op for our order.
    // Average: zc[i] = (zp[i] + conj(zn[i])) / 2.
    out.zc.reserve(zp.size());
    for (std::size_t i = 0; i < zp.size(); ++i) {
        out.zc.push_back((zp[i] + std::conj(zn[i])) / 2.0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// zpk2sos with pairing='nearest' — specialized for Butterworth where all
// poles are complex (even order) and zeros are all real. Matches the
// scipy algorithm exactly for this case:
//
//   1. Compute p_list = zc(poles) — positive-imag representatives.
//      z_list = zr(zeros) — sorted real zero values.
//   2. For si in (n_sections-1 .. 0):
//        p1 = p_list.pop(argmin |1 - |p||)
//        p2 = conj(p1)
//        z1 = z_list.pop(argmin |z - p1|)
//        z2 = z_list.pop(argmin |z - p1|)   (on remaining)
//        biquad b = (1, -(z1+z2), z1*z2);  a = (1, -2*Re(p1), |p1|^2)
//   3. Consolidate gain into first section: sos[0][:3] *= k.
// ---------------------------------------------------------------------------
std::vector<std::array<double, 6>>
zpk2sosButterworth(const std::vector<cplx>& z,
                   const std::vector<cplx>& p,
                   double                   k)
{
    CplxReal cr_p = cplxReal(p);
    CplxReal cr_z = cplxReal(z);

    // Assumption: Butterworth even-order designs always produce zero real
    // poles. Keep an assert for safety against future extension.
    assert(cr_p.zr.empty() && "Butterworth even-order design should have no real poles");

    // p_list (complex, one entry per conj-pair), z_list (real, sorted).
    // For Butterworth LP/HP/BP cr_z.zc is empty (all zeros real).
    std::vector<cplx>   p_list = cr_p.zc;
    std::vector<double> z_list = cr_z.zr;

    // Pad zeros to match total pole count (pairing != 'minimal' branch of scipy):
    //   p_total = 2 * p_list.size() (each conj-pair counts as 2)
    //   z_total must also be p_total; if there are fewer, scipy zero-pads.
    // For our case (Butterworth): LP/HP -> z_total = 2*p_list (matches);
    // BP -> same. So no padding needed.

    const std::size_t n_sections = p_list.size();
    std::vector<std::array<double, 6>> sos(n_sections);

    for (int si = static_cast<int>(n_sections) - 1; si >= 0; --si) {
        // Pick worst pole (closest to unit circle).
        std::size_t iw = 0;
        double      best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < p_list.size(); ++i) {
            const double d = std::abs(1.0 - std::abs(p_list[i]));
            if (d < best) { best = d; iw = i; }
        }
        const cplx p1 = p_list[iw];
        p_list.erase(p_list.begin() + static_cast<std::ptrdiff_t>(iw));
        // p2 = conj(p1) implicit in denominator coefficients below.

        // Find closest zero to p1 (all real in our case → "any" is real).
        auto argminDist = [&](const std::vector<double>& arr, const cplx& target) {
            std::size_t best_i = 0;
            double      best_d = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < arr.size(); ++i) {
                const double d = std::abs(cplx(arr[i], 0.0) - target);
                if (d < best_d) { best_d = d; best_i = i; }
            }
            return best_i;
        };

        const std::size_t iz1 = argminDist(z_list, p1);
        const double      z1  = z_list[iz1];
        z_list.erase(z_list.begin() + static_cast<std::ptrdiff_t>(iz1));

        const std::size_t iz2 = argminDist(z_list, p1);
        const double      z2  = z_list[iz2];
        z_list.erase(z_list.begin() + static_cast<std::ptrdiff_t>(iz2));

        // biquad numerator: (z - z1)(z - z2) = z^2 - (z1+z2)z + z1*z2
        const double b0 = 1.0;
        const double b1 = -(z1 + z2);
        const double b2 = z1 * z2;
        // biquad denominator: (z - p1)(z - p2) = z^2 - 2*Re(p1)*z + |p1|^2
        const double a0 = 1.0;
        const double a1 = -2.0 * p1.real();
        const double a2 = p1.real() * p1.real() + p1.imag() * p1.imag();

        // Explicit construction: MSVC does not implicitly convert a braced
        // init-list to std::array for operator=, even though Clang/GCC do.
        sos[si] = std::array<double, 6>{b0, b1, b2, a0, a1, a2};
    }

    // Consolidate gain into first section.
    sos[0][0] *= k;
    sos[0][1] *= k;
    sos[0][2] *= k;

    return sos;
}

// ---------------------------------------------------------------------------
// Shared design driver: analog proto → btype transform → bilinear → zpk2sos.
// Writes row-major `(n_sections, 6)` f64 to sosOut.
// ---------------------------------------------------------------------------
enum class BType { Low, High, Band };

void design(BType btype, int order, double Wn1, double Wn2,
            double* sosOut, std::size_t* nSectionsOut)
{
    // Step 1. Analog Butterworth prototype.
    std::vector<cplx> z;               // empty
    std::vector<cplx> p = buttap(order);
    double            k = 1.0;

    // Step 2. Pre-warp. scipy internal fs=2, so warped = 4 * tan(pi * Wn / 2).
    //   warped_hz = 2*fs*tan(pi*Wn/fs) with fs=2 → 4*tan(pi*Wn/2)
    constexpr double fs_internal = 2.0;
    const double     warped1 = 2.0 * fs_internal * std::tan(kPi * Wn1 / fs_internal);
    const double     warped2 = (btype == BType::Band)
                             ? 2.0 * fs_internal * std::tan(kPi * Wn2 / fs_internal)
                             : 0.0;

    // Step 3. btype transform.
    if (btype == BType::Low) {
        lp2lp(z, p, k, warped1);
    } else if (btype == BType::High) {
        lp2hp(z, p, k, warped1);
    } else { // Band
        const double bw = warped2 - warped1;
        const double wo = std::sqrt(warped1 * warped2);
        lp2bp(z, p, k, wo, bw);
    }

    // Step 4. Bilinear (internal fs=2).
    bilinearZpk(z, p, k);

    // Step 5. zpk2sos pairing='nearest' for Butterworth.
    const auto sos = zpk2sosButterworth(z, p, k);

    *nSectionsOut = sos.size();
    for (std::size_t i = 0; i < sos.size(); ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            sosOut[i * 6 + j] = sos[i][j];
        }
    }
}

} // anonymous namespace

void Butterworth::designLowpass(int order, double Wn, double* sosOut,
                                std::size_t* nSectionsOut)
{
    design(BType::Low, order, Wn, 0.0, sosOut, nSectionsOut);
}

void Butterworth::designHighpass(int order, double Wn, double* sosOut,
                                 std::size_t* nSectionsOut)
{
    design(BType::High, order, Wn, 0.0, sosOut, nSectionsOut);
}

void Butterworth::designBandpass(int order, double WnLow, double WnHigh,
                                 double* sosOut, std::size_t* nSectionsOut)
{
    design(BType::Band, order, WnLow, WnHigh, sosOut, nSectionsOut);
}

// ---------------------------------------------------------------------------
// sosfilt_zi: per-section steady-state initial conditions for step response.
// Explicit closed-form for 2nd-order sections — avoids the matrix solve in
// scipy's lfilter_zi (which uses linalg.solve on `I - A^T`). Verified
// bit-exact to ~1.4e-14 vs scipy on Butterworth order-4 LP @ 200/(22050/2)
// (session 30 audit).
//   B0 = b1 - a1*b0
//   B1 = b2 - a2*b0
//   zi_section[0] = (B0 + B1) / (1 + a1 + a2)
//   zi_section[1] = (1 + a1) * zi_section[0] - B0
// Cumulative scale: zi[k] = scale_k * zi_section_k, where
//   scale_0 = 1,  scale_{k+1} = scale_k * (b0+b1+b2)/(1+a1+a2).
// ---------------------------------------------------------------------------
void Butterworth::sosfiltZi(const double* sos, std::size_t n_sections,
                            double* ziOut)
{
    double scale = 1.0;
    for (std::size_t k = 0; k < n_sections; ++k) {
        const double* s  = sos + k * 6;
        const double  b0 = s[0], b1 = s[1], b2 = s[2];
        const double  a1 = s[4], a2 = s[5];

        const double B0   = b1 - a1 * b0;
        const double B1   = b2 - a2 * b0;
        const double den  = 1.0 + a1 + a2;
        const double z0   = (B0 + B1) / den;
        const double z1   = (1.0 + a1) * z0 - B0;

        ziOut[2 * k + 0] = scale * z0;
        ziOut[2 * k + 1] = scale * z1;

        const double sum_b = b0 + b1 + b2;
        const double sum_a = 1.0 + a1 + a2;
        scale *= sum_b / sum_a;
    }
}

// ---------------------------------------------------------------------------
// sosFiltFilt: port of scipy.signal.sosfiltfilt with default padtype='odd',
// padlen=3*ntaps. Steps:
//
//   1. ntaps = 2*n_sections + 1 - min(count(sos[:,2]==0), count(sos[:,5]==0)).
//   2. edge = 3 * ntaps.
//   3. Odd-extend input on both sides by `edge` samples:
//        left_ext [i] = 2*x[0]  - x[edge - i]         for i=0..edge-1
//        right_ext[i] = 2*x[-1] - x[-2 - i]           for i=0..edge-1
//      (Matches numpy `odd_ext`.)
//   4. zi = sosfiltZi(sos); state_init = zi * ext[0].
//   5. y_fwd = sosfilt(ext, zi=state_init).
//   6. state_init2 = zi * y_fwd[-1]; y_rev_fwd = sosfilt(reverse(y_fwd), zi=state_init2).
//   7. output = reverse(y_rev_fwd)[edge : edge + n_samples].
// ---------------------------------------------------------------------------
void Butterworth::sosFiltFilt(const double* sos, std::size_t n_sections,
                              const double* input, std::size_t n_samples,
                              double* output)
{
    if (n_sections == 0) {
        for (std::size_t n = 0; n < n_samples; ++n) output[n] = input[n];
        return;
    }

    // Step 1-2. ntaps / edge.
    std::size_t zero_b2 = 0, zero_a2 = 0;
    for (std::size_t k = 0; k < n_sections; ++k) {
        if (sos[k * 6 + 2] == 0.0) ++zero_b2;
        if (sos[k * 6 + 5] == 0.0) ++zero_a2;
    }
    const std::size_t ntaps = 2 * n_sections + 1 - std::min(zero_b2, zero_a2);
    const std::size_t edge  = 3 * ntaps;

    // scipy raises ValueError with the same precondition; we surface a
    // runtime exception so release builds (NDEBUG) don't silently read out of
    // bounds. Phase-5 session 32 Gap 2 closure.
    if (n_samples <= edge) {
        throw std::invalid_argument(
            "Butterworth::sosFiltFilt: n_samples (" + std::to_string(n_samples) +
            ") must exceed padlen (" + std::to_string(edge) + ")");
    }

    // Step 3. Build odd-extended buffer (length n_samples + 2*edge).
    std::vector<double> ext(n_samples + 2 * edge);
    const double        x_left  = input[0];
    const double        x_right = input[n_samples - 1];
    for (std::size_t i = 0; i < edge; ++i) {
        // left_ext[i] mirrors `2*x[0] - x[edge-i]` (i=0 gives x[edge], i=edge-1 gives x[1]).
        ext[i] = 2.0 * x_left - input[edge - i];
    }
    for (std::size_t i = 0; i < n_samples; ++i) {
        ext[edge + i] = input[i];
    }
    for (std::size_t i = 0; i < edge; ++i) {
        // right_ext[i] mirrors `2*x[-1] - x[-2 - i]` → input[n_samples - 2 - i].
        ext[edge + n_samples + i] = 2.0 * x_right - input[n_samples - 2 - i];
    }

    // Step 4. Compute zi, scale by ext[0].
    std::vector<double> zi(2 * n_sections);
    sosfiltZi(sos, n_sections, zi.data());
    std::vector<double> state(2 * n_sections);
    const double        ext0 = ext[0];
    for (std::size_t k = 0; k < n_sections; ++k) {
        state[2 * k + 0] = zi[2 * k + 0] * ext0;
        state[2 * k + 1] = zi[2 * k + 1] * ext0;
    }

    // Step 5. Forward pass.
    std::vector<double> y_fwd(ext.size());
    SosFilt::applyWithZi(sos, n_sections, state.data(),
                         ext.data(), ext.size(), y_fwd.data(), nullptr);

    // Step 6. Reverse + second forward pass with zi * y_fwd[-1].
    const double y_last = y_fwd.back();
    for (std::size_t k = 0; k < n_sections; ++k) {
        state[2 * k + 0] = zi[2 * k + 0] * y_last;
        state[2 * k + 1] = zi[2 * k + 1] * y_last;
    }
    // Reverse in a second buffer (can't alias input+output of applyWithZi
    // directly to a reversed view).
    std::vector<double> y_rev(ext.size());
    for (std::size_t i = 0; i < ext.size(); ++i) {
        y_rev[i] = y_fwd[ext.size() - 1 - i];
    }
    std::vector<double> y2(ext.size());
    SosFilt::applyWithZi(sos, n_sections, state.data(),
                         y_rev.data(), y_rev.size(), y2.data(), nullptr);

    // Step 7. Reverse y2 and strip padding `edge` samples each side.
    for (std::size_t i = 0; i < n_samples; ++i) {
        // Reversed index into y2: (ext.size()-1) - (edge + i) = n_samples + edge - 1 - i
        output[i] = y2[n_samples + edge - 1 - i];
    }
}

} // namespace reamix::dsp
