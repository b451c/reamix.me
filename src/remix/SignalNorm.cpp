#include "SignalNorm.h"

#include <algorithm>
#include <cmath>

namespace reamix::remix {

void DistanceBaseline::build(std::vector<double> samples)
{
    sorted_.clear();
    sorted_.reserve(samples.size());
    for (double s : samples)
        if (std::isfinite(s) && s >= 0.0) sorted_.push_back(s);
    std::sort(sorted_.begin(), sorted_.end());
}

double DistanceBaseline::quality(double d) const noexcept
{
    if (! valid()) return 1.0;
    if (! std::isfinite(d)) return 0.0;
    // number of reference samples strictly greater than d
    const auto it = std::upper_bound(sorted_.begin(), sorted_.end(), d);
    const auto worse = static_cast<double>(sorted_.end() - it);
    return worse / static_cast<double>(sorted_.size());
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

SignalBaselines buildSignalBaselines(const double* rms_energy,
                                     const double* spectral_centroid,
                                     const double* onset_strength,
                                     const double* edge_db_end,
                                     const double* edge_db_start,
                                     int n_beats)
{
    SignalBaselines b;
    b.energy.build(sequentialAbsDiff(rms_energy, n_beats, /*log*/ true));
    b.centroid.build(sequentialAbsDiff(spectral_centroid, n_beats, /*log*/ true));
    b.onset.build(sequentialAbsDiff(onset_strength, n_beats, /*log*/ false));
    b.edge_energy.build(sequentialPairDiff(edge_db_end, edge_db_start, n_beats));
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

} // namespace reamix::remix
