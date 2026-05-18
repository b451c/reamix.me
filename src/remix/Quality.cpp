#include "remix/Quality.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace reamix::remix {

double computeQualityScore(const QualityInputs& inputs, const QualityWeights& weights)
{
    // Accumulate in exactly the same order as quality.py L84-116:
    //   1. Optional signals (waveform, edge_splice) — each either contributes
    //      to `weighted_sum` and adds to `available_weight`, OR adds to
    //      `missing_weight` so redistribution can cover it later.
    //   2. The 8 always-available signals in the order of Python's
    //      `always_pairs` tuple (successor, context, label, section,
    //      bar_align, energy, edge_energy, centroid).
    // This order matters for bitwise parity: float summation is not
    // associative and any reorder would shift the last 1-2 ULPs.
    //
    // The `weights` parameter defaults to `kDefaultQualityWeights` whose
    // fields are bit-exact copies of the W_* constants — when called without
    // an explicit override, `weights.waveform == W_WAVEFORM` etc., preserving
    // parity tests at the byte level.

    double available_weight = 0.0;
    double weighted_sum     = 0.0;
    double missing_weight   = 0.0;

    // D2 harmonic mean (sesja 81 ADR-068 D2). When `weights.use_harmonic_mean`
    // is true, the function returns `available_weight / Σ(wᵢ / max(qᵢ, ε))`
    // instead of the arithmetic weighted mean. We accumulate `harmonic_inv_sum`
    // in parallel only when the flag is set — for the legacy path this stays
    // 0.0 and is never read, so bit-exact arithmetic parity is preserved.
    const bool   harmonic        = weights.use_harmonic_mean;
    double       harmonic_inv_sum = 0.0;

    // Sesja 71b — fix-in-port (ADR-060, see dev-028-lessons.md): every
    // contribution gates on `weights.X > 0.0` so calibration weight vectors
    // can set any signal to zero without IEEE-754 NaN propagating through
    // `0.0 * NaN`. Real signal-level NaNs occur (cosine of zero edge-feature
    // vectors → 0/0; sesja-71b smoke confirmed billie_jean cand(35,104)
    // edge_splice_sim was NaN). Without this guard, a zero weight on a NaN
    // signal cascaded through `weighted_sum`, then `std::min(1.0, NaN)`
    // saturated to 1.0 (IEEE comparison returns false → first arg wins),
    // and the final clamp emitted 1.0 — i.e. EVERY zero-weight calibration
    // run hit the same uniform sentinel quality regardless of which signal
    // dominated. Behavior with all-non-zero `kDefaultQualityWeights` is
    // unchanged (guard always fires for default → bit-exact parity).

    // --- Optional signals -----------------------------------------------
    if (inputs.waveform_sim.has_value()) {
        if (weights.waveform > 0.0) {
            const double q = inputs.waveform_sim.value();
            available_weight += weights.waveform;
            weighted_sum     += weights.waveform * q;
            if (harmonic) harmonic_inv_sum += weights.waveform / std::max(q, kHarmonicMeanEpsilon);
        }
    } else {
        missing_weight   += weights.waveform;
    }

    if (inputs.edge_splice_sim.has_value()) {
        if (weights.edge_splice > 0.0) {
            const double q = inputs.edge_splice_sim.value();
            available_weight += weights.edge_splice;
            weighted_sum     += weights.edge_splice * q;
            if (harmonic) harmonic_inv_sum += weights.edge_splice / std::max(q, kHarmonicMeanEpsilon);
        }
    } else {
        missing_weight   += weights.edge_splice;
    }

    // --- Always-available signals (Python L103-112 tuple order) ---------
    // (W_SUCCESSOR, successor_sim),
    if (weights.successor > 0.0) {
        available_weight += weights.successor;
        weighted_sum     += weights.successor * inputs.successor_sim;
        if (harmonic) harmonic_inv_sum += weights.successor / std::max(inputs.successor_sim, kHarmonicMeanEpsilon);
    }
    // (W_CONTEXT, context_sim),
    if (weights.context > 0.0) {
        available_weight += weights.context;
        weighted_sum     += weights.context * inputs.context_sim;
        if (harmonic) harmonic_inv_sum += weights.context / std::max(inputs.context_sim, kHarmonicMeanEpsilon);
    }
    // (W_LABEL, label_match),
    if (weights.label > 0.0) {
        available_weight += weights.label;
        weighted_sum     += weights.label * inputs.label_match;
        if (harmonic) harmonic_inv_sum += weights.label / std::max(inputs.label_match, kHarmonicMeanEpsilon);
    }
    // (W_SECTION, section_sim),
    if (weights.section > 0.0) {
        available_weight += weights.section;
        weighted_sum     += weights.section * inputs.section_sim;
        if (harmonic) harmonic_inv_sum += weights.section / std::max(inputs.section_sim, kHarmonicMeanEpsilon);
    }
    // (W_BAR_ALIGN, bar_aligned),
    if (weights.bar_align > 0.0) {
        available_weight += weights.bar_align;
        weighted_sum     += weights.bar_align * inputs.bar_aligned;
        if (harmonic) harmonic_inv_sum += weights.bar_align / std::max(inputs.bar_aligned, kHarmonicMeanEpsilon);
    }
    // (W_ENERGY, energy_match),
    if (weights.energy > 0.0) {
        available_weight += weights.energy;
        weighted_sum     += weights.energy * inputs.energy_match;
        if (harmonic) harmonic_inv_sum += weights.energy / std::max(inputs.energy_match, kHarmonicMeanEpsilon);
    }
    // (W_EDGE_ENERGY, edge_energy_match),
    if (weights.edge_energy > 0.0) {
        available_weight += weights.edge_energy;
        weighted_sum     += weights.edge_energy * inputs.edge_energy_match;
        if (harmonic) harmonic_inv_sum += weights.edge_energy / std::max(inputs.edge_energy_match, kHarmonicMeanEpsilon);
    }
    // (W_CENTROID, centroid_match),
    if (weights.centroid > 0.0) {
        available_weight += weights.centroid;
        weighted_sum     += weights.centroid * inputs.centroid_match;
        if (harmonic) harmonic_inv_sum += weights.centroid / std::max(inputs.centroid_match, kHarmonicMeanEpsilon);
    }

    // ---- 11th slot — transient continuity (sesja 75 ADR-064) ----------
    // Optional both-sides: default weight 0.0 + nullopt value → no
    // contribution → bit-exact parity preserved against pre-sesja-75 baseline.
    // Mirrors waveform/edge_splice missing-signal semantics (treats nullopt
    // as a missing-signal contribution to `missing_weight` so redistribution
    // covers it exactly like the other two optional inputs). ADR-060
    // null-guard `weights.X > 0.0` retained.
    if (inputs.transient_continuity.has_value()) {
        if (weights.transient_continuity > 0.0) {
            const double q = inputs.transient_continuity.value();
            available_weight += weights.transient_continuity;
            weighted_sum     += weights.transient_continuity * q;
            if (harmonic) harmonic_inv_sum += weights.transient_continuity / std::max(q, kHarmonicMeanEpsilon);
        }
    } else {
        missing_weight   += weights.transient_continuity;
    }

    // ---- 12th slot — MFCC spectral continuity (sesja 77 ADR-066) ------
    // Optional both-sides: default weight 0.0 + nullopt value → no
    // contribution → bit-exact parity preserved against pre-sesja-77 baseline.
    if (inputs.mfcc_continuity.has_value()) {
        if (weights.mfcc_continuity > 0.0) {
            const double q = inputs.mfcc_continuity.value();
            available_weight += weights.mfcc_continuity;
            weighted_sum     += weights.mfcc_continuity * q;
            if (harmonic) harmonic_inv_sum += weights.mfcc_continuity / std::max(q, kHarmonicMeanEpsilon);
        }
    } else {
        missing_weight   += weights.mfcc_continuity;
    }

    // ---- DEV-028 sesja 74 expressivity scratch slot (extra1) ----------
    // Optional both-sides: default weight 0.0 + nullopt value → no
    // contribution → bit-exact parity preserved.
    if (inputs.extra1_value.has_value()) {
        if (weights.extra1 > 0.0) {
            const double q = inputs.extra1_value.value();
            available_weight += weights.extra1;
            weighted_sum     += weights.extra1 * q;
            if (harmonic) harmonic_inv_sum += weights.extra1 / std::max(q, kHarmonicMeanEpsilon);
        }
    } else {
        missing_weight   += weights.extra1;
    }

    // ---- ADR-088 sesja 98 vocal phrase continuity --------------------
    // Optional both-sides: default weight 0.0 + nullopt value → no
    // contribution → bit-exact parity preserved. Mirrors transient_continuity
    // (ADR-064) shape exactly.
    if (inputs.vocal_continuity.has_value()) {
        if (weights.vocal_continuity > 0.0) {
            const double q = inputs.vocal_continuity.value();
            available_weight += weights.vocal_continuity;
            weighted_sum     += weights.vocal_continuity * q;
            if (harmonic) harmonic_inv_sum += weights.vocal_continuity / std::max(q, kHarmonicMeanEpsilon);
        }
    } else {
        missing_weight   += weights.vocal_continuity;
    }

    // ---- D4 collapse: sequential_continuity (sesja 81 ADR-068 D4) -----
    // Single weight replacing the redundant successor + context +
    // mfcc_continuity triplet (ρ=0.67-0.71 measured sesja-80 D1). The
    // sequential dimension is the mean of the three constituents, with
    // mfcc_continuity excluded when nullopt (the optional input is
    // populated only when MFCC features were available upstream — same
    // semantics as the per-component path). When all three constituents
    // are present, seq = (successor + context + mfcc_continuity) / 3.
    //
    // This branch is the ONLY path that consumes the `sequential_continuity`
    // weight — the constituent fields (`successor`, `context`,
    // `mfcc_continuity`) are still consumed by their per-component branches
    // ABOVE. In the new production simplex (kProductionQualityWeights, ships
    // post-D3 commit), the per-component weights are 0.0 and only this
    // collapsed weight contributes; in legacy weights (kLegacyQualityWeights,
    // = current kDefaultQualityWeights at this commit), this weight is 0.0
    // and only per-component branches contribute → bit-exact parity preserved.
    //
    // CANONICAL DEFINITION (no Python reference): per ADR-068 D4 (sesja 81),
    // collapse evidence sesja-80 D1 correlation matrix. Self-validated by
    // tests/parity/test_sequential_continuity.cpp.
    if (weights.sequential_continuity > 0.0) {
        // Mean over the available constituents. successor_sim + context_sim
        // are always present (per the QualityInputs contract — they are
        // required, not optional); mfcc_continuity is optional.
        double sum    = inputs.successor_sim + inputs.context_sim;
        double n_term = 2.0;
        if (inputs.mfcc_continuity.has_value()) {
            sum    += inputs.mfcc_continuity.value();
            n_term += 1.0;
        }
        const double seq = sum / n_term;
        available_weight += weights.sequential_continuity;
        weighted_sum     += weights.sequential_continuity * seq;
        if (harmonic) harmonic_inv_sum += weights.sequential_continuity / std::max(seq, kHarmonicMeanEpsilon);
    }

    // --- Compute timbre block score -------------------------------------
    // Single-return refactor (sesja 92, ADR-080 RESCOPE) so the Tone slider
    // blend below has a single composition site instead of repeating across
    // three return paths. When `harmonic_vs_timbre == 0.0` the blend block
    // is bypassed and the function returns the same timbre score as the
    // pre-sesja-92 implementation (bit-exact baseline preservation).
    double timbre_score;

    // D2 Harmonic-mean branch (sesja 81 ADR-068 D2):
    //   Q = available_weight / Σᵢ ( wᵢ / max(qᵢ, ε) )
    // Missing-signal redistribution is implicit (available_weight already
    // excludes missing signals' weights). harmonic_inv_sum == 0 (all
    // weights zero or all present q ≥ ε saturated) returns 0.0 to mirror
    // arithmetic edge case.
    if (harmonic) {
        if (harmonic_inv_sum > 0.0) {
            timbre_score = available_weight / harmonic_inv_sum;
        } else {
            timbre_score = 0.0;
        }
    }
    // Redistribute missing weight proportionally (Python L118-120).
    else if (available_weight > 0.0 && missing_weight > 0.0) {
        const double scale = 1.0 / available_weight;
        timbre_score = weighted_sum * scale;
    }
    // Common case: all signals available (Python L123).
    else {
        timbre_score = weighted_sum;
    }

    timbre_score = std::max(0.0, std::min(1.0, timbre_score));

    // --- ADR-080 Tone slider blend (sesja 92, RESCOPE STATUS UPDATE 1; --
    //     ADR-084 sesja 93 formula change to fix mid-α inertness)
    // When `weights.harmonic_vs_timbre > 0.0` AND chroma input is available,
    // blend timbre block score with full-mix chroma similarity per:
    //   timbre_coef  = 1 - 0.7 · α
    //   chroma_coef  = √α
    //   final_score  = clamp(timbre_coef · timbre_score + chroma_coef · chroma, 0, 1)
    //
    // Why damped+sqrt instead of convex (1-α)·t + α·c:
    // sesja 93 user smoke on Despacito (harmonically monotone EDM remix)
    // showed mid-α positions inaudible. Research-scout HIGH-confidence
    // verdict: convex blend is mathematically near-inert at mid when the
    // chroma similarity matrix is near-uniform across beat-pairs (verses
    // and choruses share the same chord progression → chroma_sim std ≪
    // timbre_score std → blend at α=0.5 still ranked by timbre signal).
    // Damped+sqrt curve keeps both signals alive at every α:
    //   - timbre_coef floors at 0.3 at α=1.0 (vs 0.0 in convex blend) so
    //     timbre signal never disappears even at "pure harmonic" extreme.
    //   - chroma_coef = √α gives 0.5 contribution already at α=0.25
    //     (vs 0.25 in convex blend) → chroma signal accelerates onset.
    // Final value clamped to [0,1] because coefficients can sum > 1 at
    // high α (e.g. α=1: timbre_coef=0.3 + chroma_coef=1.0 → 1.3 max).
    //
    // Default `harmonic_vs_timbre = 0.0` → blend bypassed → bit-exact
    // baseline (1·timbre + 0·chroma = timbre). Slider thumb starts at
    // 0.0 (Timbre 100%) per ADR-080; user explicitly drags right to
    // enable harmonic blend. nullopt chroma input also bypasses blend
    // (e.g. when upstream skipped chroma slice population).
    //
    // Self-validated by tests/parity/test_harmonic_vs_timbre_blend.cpp.
    if (weights.harmonic_vs_timbre > 0.0 && inputs.full_mix_chroma_continuity.has_value()) {
        const double tone   = std::clamp(weights.harmonic_vs_timbre, 0.0, 1.0);
        const double chroma = std::clamp(inputs.full_mix_chroma_continuity.value(), 0.0, 1.0);
        const double timbre_coef = 1.0 - 0.7 * tone;
        const double chroma_coef = std::sqrt(tone);
        timbre_score = std::clamp(timbre_coef * timbre_score + chroma_coef * chroma,
                                  0.0, 1.0);
    }

    return timbre_score;
}

double computeVocalPenalty(double va_source,
                           double va_dest,
                           std::optional<double> edge_va_end,
                           std::optional<double> edge_va_start)
{
    double penalty = 0.0;

    // (1) Continuity: penalize abrupt vocal-level changes.
    // Python L145: `vocal_continuity = max(0.0, 1.0 - abs(va_source - va_dest) * 2.0)`
    const double vocal_continuity =
        std::max(0.0, 1.0 - std::abs(va_source - va_dest) * 2.0);
    if (vocal_continuity < 1.0) {
        penalty += (1.0 - vocal_continuity) * VOCAL_CONTINUITY_PENALTY;
    }

    // (2) Splice-through: penalize cutting through active vocals.
    // Python L151-154: use edge-resolution features when both are available,
    // otherwise fall back to beat-level vocal activity.
    double splice_activity;
    if (edge_va_end.has_value() && edge_va_start.has_value()) {
        splice_activity = std::min(edge_va_end.value(), edge_va_start.value());
    } else {
        splice_activity = std::min(va_source, va_dest);
    }

    if (splice_activity > VOCAL_SPLICE_THRESHOLD) {
        penalty += (splice_activity - VOCAL_SPLICE_THRESHOLD) * VOCAL_SPLICE_PENALTY;
    }

    return penalty;
}

std::vector<double> computeOnsetNorm(const double* onset_strength, int n_beats)
{
    // Min-max normalise onset_strength into [0, 1]. Returns n_beats-length
    // vector for inline composition `1 - |onset_norm[i] - onset_norm[j]|` by
    // TC/RC/BA per-pair scoring loops (sesja 75 ADR-064).
    //
    // Edge cases:
    //   - n_beats <= 0 or onset_strength == nullptr → empty vector.
    //   - max == min (uniform onset) → denominator clamped to 1e-9 so each
    //     element evaluates to ~0; transient_continuity ~1.0 across all
    //     pairs (no-information signal). Avoids divide-by-zero / NaN.
    if (onset_strength == nullptr || n_beats <= 0) {
        return {};
    }
    double mn =  std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < n_beats; ++k) {
        const double v = onset_strength[k];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const double rng = std::max(1e-9, mx - mn);
    std::vector<double> out(static_cast<std::size_t>(n_beats));
    for (int k = 0; k < n_beats; ++k) {
        out[static_cast<std::size_t>(k)] =
            (onset_strength[k] - mn) / rng;
    }
    return out;
}

std::vector<double> computeMfccContinuityMatrix(const float* features,
                                                int          n_beats,
                                                int          n_features)
{
    // n×n MFCC + delta-MFCC L2 similarity matrix per ADR-066 (sesja 77).
    // Mirrors calibration_harness::computeExtra1Matrix signal=mfcc_continuity
    // bit-for-bit so expressivity check + production scoring stay aligned for
    // the full-band `mfcc_continuity` axis.
    if (features == nullptr || n_beats <= 0 || n_features <= 0) {
        return {};
    }
    const int K = std::min(MFCC_CONTINUITY_N_BANDS_MAX, n_features);
    if (K <= 0) {
        return {};
    }

    const double normRef    = std::sqrt(static_cast<double>(K));
    const double saturation = MFCC_CONTINUITY_SATURATION;
    const double rs         = MFCC_CONTINUITY_STATIC_RATIO;
    const double rd         = MFCC_CONTINUITY_DYNAMIC_RATIO;

    std::vector<double> m(static_cast<std::size_t>(n_beats) *
                          static_cast<std::size_t>(n_beats));

    for (int i = 0; i < n_beats; ++i) {
        const int ip = std::max(0, i - 1);
        for (int j = 0; j < n_beats; ++j) {
            const int jn = std::min(n_beats - 1, j + 1);

            double s2_static = 0.0;
            double s2_dyn    = 0.0;
            for (int k = 0; k < K; ++k) {
                const double fi  = static_cast<double>(
                    features[(std::size_t) i  * (std::size_t) n_features + k]);
                const double fj  = static_cast<double>(
                    features[(std::size_t) j  * (std::size_t) n_features + k]);
                const double fip = static_cast<double>(
                    features[(std::size_t) ip * (std::size_t) n_features + k]);
                const double fjn = static_cast<double>(
                    features[(std::size_t) jn * (std::size_t) n_features + k]);

                const double d_static = fi - fj;
                s2_static += d_static * d_static;

                const double dl = fi  - fip;
                const double dr = fjn - fj;
                const double dd = dl - dr;
                s2_dyn += dd * dd;
            }

            const double d_static  = std::sqrt(s2_static) / normRef;
            const double d_dynamic = std::sqrt(s2_dyn)    / normRef;

            double raw = rs * d_static + rd * d_dynamic;
            raw        = std::min(1.0, raw / saturation);

            m[(std::size_t) i * (std::size_t) n_beats + (std::size_t) j] =
                1.0 - raw;
        }
    }
    return m;
}

std::vector<double> computeChromaContinuityMatrix(const float* features,
                                                  int          n_beats,
                                                  int          n_chroma)
{
    // n×n chroma cosine similarity matrix. Used as the chroma kernel
    // underlying ADR-080 PROPOSED Audition Timbre/Harmonic slider. Synthetic
    // chord sequences exercise expected high similarity within a chord +
    // low similarity across chord changes.
    if (features == nullptr || n_beats <= 0 || n_chroma <= 0) {
        return {};
    }

    constexpr double kEps = 1e-12;

    std::vector<double> norms(static_cast<std::size_t>(n_beats), 0.0);
    for (int i = 0; i < n_beats; ++i) {
        double s = 0.0;
        for (int k = 0; k < n_chroma; ++k) {
            const double v = static_cast<double>(
                features[(std::size_t) i * (std::size_t) n_chroma + k]);
            s += v * v;
        }
        norms[static_cast<std::size_t>(i)] = std::sqrt(s);
    }

    std::vector<double> m(static_cast<std::size_t>(n_beats) *
                          static_cast<std::size_t>(n_beats));

    for (int i = 0; i < n_beats; ++i) {
        for (int j = 0; j < n_beats; ++j) {
            double dot = 0.0;
            for (int k = 0; k < n_chroma; ++k) {
                const double a = static_cast<double>(
                    features[(std::size_t) i * (std::size_t) n_chroma + k]);
                const double b = static_cast<double>(
                    features[(std::size_t) j * (std::size_t) n_chroma + k]);
                dot += a * b;
            }
            const double denom = norms[(std::size_t) i] *
                                 norms[(std::size_t) j];
            const double sim = (denom > kEps) ? (dot / denom) : 0.0;
            m[(std::size_t) i * (std::size_t) n_beats + (std::size_t) j] =
                std::max(0.0, std::min(1.0, sim));
        }
    }
    return m;
}

double computeOnsetPenalty(std::optional<double> onset_dest)
{
    // Python L174-175: None destination = no penalty.
    if (!onset_dest.has_value()) {
        return 0.0;
    }
    const double v = onset_dest.value();

    // Python L176-177: strong onset = attack masks the cut, no penalty.
    if (v >= ONSET_SUSTAIN_THRESHOLD) {
        return 0.0;
    }

    // Python L179-181: linear ramp below threshold.
    //   (THRESHOLD - onset_dest) * (PENALTY / THRESHOLD)
    // Parenthesization matches Python exactly; the division is evaluated at
    // every call (0.12 / 0.25 = 0.48 is not bit-exactly representable, and
    // precomputing it as a constant would introduce a bit-level drift).
    return (ONSET_SUSTAIN_THRESHOLD - v) *
           (ONSET_SUSTAIN_PENALTY / ONSET_SUSTAIN_THRESHOLD);
}

std::string_view qualityLabel(double score)
{
    // Python L186-190: strict-greater for Great, >= for Good, else Weak.
    if (score > QUALITY_GREAT)  return "Great";
    if (score >= QUALITY_GOOD)  return "Good";
    return "Weak";
}

std::string_view qualityColorKey(std::optional<double> score)
{
    // Python L195-201: None -> "unknown", then same > / >= partition.
    if (!score.has_value())        return "unknown";
    const double s = score.value();
    if (s > QUALITY_GREAT)         return "good";
    if (s >= QUALITY_GOOD)         return "medium";
    return "bad";
}

} // namespace reamix::remix
