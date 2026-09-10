#include "SignalNorm.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace reamix::remix {

void DistanceBaseline::build(std::vector<double> samples)
{
    sorted_.clear();
    sorted_.reserve(samples.size());
    for (double s : samples)
        if (std::isfinite(s) && s >= 0.0) sorted_.push_back(s);
    std::sort(sorted_.begin(), sorted_.end());
    // Guard: a flat track (all steps 0) must not divide by zero; a tiny
    // positive scale then makes any non-zero step cost heavily, which is
    // the right reading of "the track never moves between beats".
    scale_  = std::max(percentile(kScalePercentile), 1e-6);
    reject_ = percentile(kRejectPercentile);
}

double DistanceBaseline::quality(double d) const noexcept
{
    if (! valid()) return 1.0;
    if (! std::isfinite(d)) return 0.0;
    return std::exp(-std::max(d, 0.0) / scale_);
}

bool DistanceBaseline::reject(double d) const noexcept
{
    if (! valid()) return false;
    if (! std::isfinite(d)) return true;
    return d > reject_;
}

double DistanceBaseline::percentile(double p) const noexcept
{
    if (sorted_.empty()) return 0.0;
    const double pc = std::clamp(p, 0.0, 1.0);
    const auto idx = static_cast<std::size_t>(
        std::llround(pc * static_cast<double>(sorted_.size() - 1)));
    return sorted_[std::min(idx, sorted_.size() - 1)];
}

std::vector<double> sequentialAbsDiff(const double* v, int n, bool log_domain, double floor)
{
    std::vector<double> out;
    if (v == nullptr || n < 2) return out;
    out.reserve(static_cast<std::size_t>(n - 1));
    auto xf = [&](double x) { return log_domain ? std::log(std::max(x, floor)) : x; };
    for (int i = 0; i + 1 < n; ++i) {
        const double a = v[i], b = v[i + 1];
        if (! std::isfinite(a) || ! std::isfinite(b)) continue;
        out.push_back(std::abs(xf(b) - xf(a)));
    }
    return out;
}

std::vector<double> sequentialPairDiff(const double* end_of_i, const double* start_of_j, int n)
{
    std::vector<double> out;
    if (end_of_i == nullptr || start_of_j == nullptr || n < 2) return out;
    out.reserve(static_cast<std::size_t>(n - 1));
    for (int i = 0; i + 1 < n; ++i) {
        const double a = end_of_i[i], b = start_of_j[i + 1];
        if (! std::isfinite(a) || ! std::isfinite(b)) continue;
        out.push_back(std::abs(b - a));
    }
    return out;
}

// ---- Sesja 129: edge continuity -------------------------------------------

double EdgeContinuityScale::quality(double d) const noexcept
{
    if (! std::isfinite(d)) return 0.0;
    if (! valid()) return 1.0;
    if (d <= 0.0) return 1.0;
    return std::exp(-d / scale);
}

double edgeMelDistance(const float* edge_mel, int n_mel, int a, int b) noexcept
{
    if (edge_mel == nullptr || n_mel <= 0) return 0.0;
    const float* ra = edge_mel + static_cast<std::size_t>(a) * static_cast<std::size_t>(n_mel);
    const float* rb = edge_mel + static_cast<std::size_t>(b) * static_cast<std::size_t>(n_mel);
    double acc = 0.0;
    for (int m = 0; m < n_mel; ++m) {
        const double d = static_cast<double>(ra[m]) - static_cast<double>(rb[m]);
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(n_mel));
}

EdgeContinuityScale buildEdgeContinuityScale(const float* edge_mel_end, int n_beats,
                                             int n_mel, int lag_beats,
                                             int fallback_lag_beats)
{
    EdgeContinuityScale s;
    if (edge_mel_end == nullptr || n_beats <= 0 || n_mel <= 0) return s;
    for (const int lag : {lag_beats, fallback_lag_beats}) {
        if (lag <= 0 || lag >= n_beats) continue;
        std::vector<double> d;
        d.reserve(static_cast<std::size_t>(n_beats - lag));
        for (int b = 0; b + lag < n_beats; ++b) {
            const double v = edgeMelDistance(edge_mel_end, n_mel, b, b + lag);
            if (std::isfinite(v)) d.push_back(v);
        }
        if (static_cast<int>(d.size()) < DistanceBaseline::kMinSamples) continue;
        std::sort(d.begin(), d.end());
        const std::size_t n = d.size();
        s.scale   = (n % 2 == 1) ? d[n / 2] : 0.5 * (d[n / 2 - 1] + d[n / 2]);
        s.samples = static_cast<int>(n);
        return s;
    }
    return s;
}

EdgeContinuityValue edgeContinuityV2(const SignalBaselines& b, const float* edge_mel_end,
                                     int n_mel, int n_beats, int i, int j) noexcept
{
    EdgeContinuityValue v;
    if (! b.edge_continuity.valid() || edge_mel_end == nullptr || n_mel <= 0) return v;
    if (i < 0 || j <= 0 || i >= n_beats || j >= n_beats) return v;
    const double d = edgeMelDistance(edge_mel_end, n_mel, i, j - 1);
    v.distance  = d / b.edge_continuity.scale;
    v.quality   = b.edge_continuity.quality(d);
    v.available = true;
    return v;
}

SignalBaselines buildSignalBaselines(const double* rms_energy,
                                     const double* spectral_centroid,
                                     const double* onset_strength,
                                     const double* edge_db_end,
                                     const double* edge_db_start,
                                     int n_beats,
                                     const float* edge_mel_end,
                                     int n_edge_mel,
                                     int bar_beats)
{
    SignalBaselines b;
    b.energy.build(sequentialAbsDiff(rms_energy, n_beats, /*log*/ true));
    b.centroid.build(sequentialAbsDiff(spectral_centroid, n_beats, /*log*/ true));
    b.onset.build(sequentialAbsDiff(onset_strength, n_beats, /*log*/ false));
    b.edge_energy.build(sequentialPairDiff(edge_db_end, edge_db_start, n_beats));
    if (edge_mel_end != nullptr && n_edge_mel > 0 && bar_beats > 0)
        b.edge_continuity = buildEdgeContinuityScale(edge_mel_end, n_beats, n_edge_mel,
                                                     8 * bar_beats, bar_beats);
    return b;
}

namespace {
constexpr double kLogFloor = 1e-9;
inline double logStep(double a, double b)
{
    return std::abs(std::log(std::max(b, kLogFloor)) - std::log(std::max(a, kLogFloor)));
}
} // namespace

double energyQualityV2(const SignalBaselines& b, double rms_i, double rms_j, double legacy)
{
    return b.energy.valid() ? b.energy.quality(logStep(rms_i, rms_j)) : legacy;
}

double centroidQualityV2(const SignalBaselines& b, double c_i, double c_j, double legacy)
{
    return b.centroid.valid() ? b.centroid.quality(logStep(c_i, c_j)) : legacy;
}

double onsetQualityV2(const SignalBaselines& b, double o_i, double o_j, double legacy)
{
    return b.onset.valid() ? b.onset.quality(std::abs(o_j - o_i)) : legacy;
}

double edgeEnergyQualityV2(const SignalBaselines& b, double energy_diff_db, double legacy)
{
    return b.edge_energy.valid() ? b.edge_energy.quality(energy_diff_db) : legacy;
}

bool loudnessRejectV2(const SignalBaselines& b, double rms_i, double rms_j,
                      double energy_diff_db)
{
    return b.energy.reject(logStep(rms_i, rms_j))
        || b.edge_energy.reject(energy_diff_db);
}

} // namespace reamix::remix
