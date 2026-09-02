#include "RepetitionPrior.h"

#include "analysis/Recurrence.h"

#include <algorithm>

namespace reamix::remix {

RepetitionPrior RepetitionPrior::fromRecurrence(const double* R, int n,
                                                const std::set<int>& pre_db_set,
                                                const std::set<int>& db_set,
                                                int time_signature,
                                                int min_run,
                                                double threshold)
{
    RepetitionPrior p;
    p.n_beats = n;
    if (R == nullptr || n < 4 || pre_db_set.empty() || db_set.empty()) return p;

    const int ts = time_signature > 0 ? time_signature : 4;
    const bool user_run = min_run > 0;
    if (! user_run) min_run = std::max(1, static_cast<int>(kMinRunBars * ts + 0.5));

    int n_sources = 0;
    auto fill = [&](int run) {
        p.mask.assign(static_cast<std::size_t>(n) * n, 0);
        p.n_allowed = 0;
        n_sources = 0;
        for (int i : pre_db_set) {
            if (i < 0 || i + 1 >= n) continue;
            bool any = false;
            for (int j : db_set) {
                if (j < 0 || j >= n || j == i + 1) continue;
                // Successor view: beat j stands in for beat i+1. Count
                // recurrence hits along the diagonal through (i+1, j) over
                // the bar before and the bar after the junction.
                int hits = 0;
                for (int k = -ts; k < ts; ++k) {
                    const int r = i + 1 + k, c = j + k;
                    if (r < 0 || c < 0 || r >= n || c >= n) continue;
                    if (R[static_cast<std::size_t>(r) * n + c] > threshold) ++hits;
                }
                if (hits >= run) {
                    p.mask[static_cast<std::size_t>(i) * n + j] = 1;
                    ++p.n_allowed;
                    any = true;
                }
            }
            if (any) ++n_sources;
        }
        p.min_run_used = run;
    };
    fill(min_run);
    const double per_source = static_cast<double>(p.n_allowed)
                            / static_cast<double>(pre_db_set.size());
    const int relaxed = std::max(1, static_cast<int>(kRelaxedMinRunBars * ts + 0.5));
    if (! user_run && per_source < kMinAllowedPerSource && relaxed < min_run)
        fill(relaxed);
    p.n_sources = n_sources;
    const double coverage = static_cast<double>(n_sources)
                          / static_cast<double>(pre_db_set.size());
    p.active = p.n_allowed > 0 && coverage >= kMinSourceCoverage;
    if (! p.active) p.mask.clear();
    return p;
}

RepetitionPrior RepetitionPrior::build(const float* features, int n, int n_feat,
                                       const std::set<int>& pre_db_set,
                                       const std::set<int>& db_set,
                                       int time_signature)
{
    if (features == nullptr || n < 4 || n_feat <= 0) return RepetitionPrior{};
    const auto rec = analysis::Recurrence::build(features, n, n_feat, kRecurrenceK);
    return fromRecurrence(rec.R.data(), n, pre_db_set, db_set, time_signature);
}

} // namespace reamix::remix
