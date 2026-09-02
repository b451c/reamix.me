#include "remix/BlockAssembly.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>    // dev-only REAMIX_BLOCKS_DEBUG dump (sesja 119)
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dsp/WaveformXcorr.h"
#include "remix/Quality.h"
#include "remix/PairScorer.h"   // sesja 119 (DEV-096) shared pair scorer
#include "remix/RegionCost.h"   // REGION_CHROMA_PREFILTER (Region C2 gate, shared sesja 119)
#include "remix/SignalNorm.h"  // ADR-115 v2 scoring
#include "remix/TransitionCost.h"  // chromaRange + EDGE_ENERGY_SATURATION_DB + N_CHROMA_DIMS

namespace reamix::remix {

namespace {

// PARITY: standard library utility — argmin |arr - target| over n elements.
// Matches `int(np.argmin(np.abs(beat_times - t)))` semantics.
int argminAbs(const double* arr, int n, double target)
{
    int    best_i = 0;
    double best_v = std::abs(arr[0] - target);
    for (int i = 1; i < n; ++i) {
        const double v = std::abs(arr[i] - target);
        if (v < best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

// PARITY: capitalize first char, lower the rest. Matches Python `str.capitalize()`
// for the ASCII subset present in our segment labels. If label starts with a
// non-ASCII byte, we leave it untouched (no label currently observed uses UTF-8
// multibyte beyond ASCII range).
std::string capitalizeFirst(const std::string& s)
{
    if (s.empty()) return s;
    std::string out = s;
    // Python capitalize() uppercases the first char AND lowercases the rest.
    // All production labels are lowercase ASCII ("intro", "verse", "chorus",
    // "bridge", "outro", "unknown"), so `toupper` on [0] suffices; preserving
    // case beyond [0] would mismatch on labels like "VERSE" which don't
    // exist anyway.
    for (std::size_t i = 0; i < out.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(out[i]);
        if (i == 0) {
            out[i] = static_cast<char>(std::toupper(c));
        } else {
            out[i] = static_cast<char>(std::tolower(c));
        }
    }
    return out;
}

// PARITY: block_assembly.py:279-392 _score_junction.
// Full 10-signal composition + graduated energy penalty + vocal penalty.
// Returns quality ∈ [0, 1]; callers skip candidates where quality ≤ 0.
struct JunctionContext
{
    int            n_beats;
    int            n_features;
    const float*   features;
    // L2-normalized features (per-row) and chroma sub-vector (per-row).
    // Flat row-major [n × n_features] / [n × n_chroma].
    const double*  feats_normed;
    const double*  chroma_normed;
    int            n_chroma;
    const float*   boundary_waveforms;
    bool           has_wf;
    int            max_lag;
    int            n_samples_per_bnd;
    const float*   edge_features_start;
    const float*   edge_features_end;
    int            n_edge_features;
    const double*  edge_db_end;
    const double*  edge_db_start;
    bool           has_edge_db;
    const double*  rms_energy;
    const double*  spectral_centroid;
    const double*  vocal_activity;
    // FIX-IN-PORT (sesja 71, ADR-059): onset-sustain penalty source array.
    // Null = legacy parity path (matches Python BA which lacks onset penalty).
    const double*  onset_strength;
    // ADR-064 (sesja 75): pre-normalised onset for per-pair transient
    // continuity composition. Empty when onset_strength was not provided.
    const double*  onset_norm;
    int            onset_norm_n;
    // ADR-066 (sesja 77): pre-computed n×n MFCC continuity matrix for per-
    // pair `mfcc_continuity` composition. Null when features were not
    // provided → qi.mfcc_continuity stays nullopt → parity preserved.
    const double*  mfcc_continuity_matrix;
    int            mfcc_continuity_n;
    // ADR-080 RESCOPE + ADR-083 (sesja 92): pre-computed n×n full-mix chroma
    // similarity matrix for per-pair `full_mix_chroma_continuity` composition.
    // Consumed by Tone slider blend in computeQualityScore. Null when features
    // were not provided → qi.full_mix_chroma_continuity stays nullopt → blend
    // bypassed → bit-exact baseline preserved.
    const double*  chroma_continuity_matrix;
    int            chroma_continuity_n;
    // ADR-088 sesja 98 — vocal phrase boundary signals. Null = parity path.
    const double*  edge_vocal_onset_start;
    const double*  edge_vocal_release_end;
    // ADR-115 (sesja 114) — v2 scoring: baselines + default weight set
    const SignalBaselines* baselines = nullptr;
    bool           v2 = false;
    // QualityWeights override (sesja 71, ADR-058). Null = default.
    const QualityWeights* quality_weights;
    const std::set<int>* db_set;
    const std::set<int>* pre_db_set;
};

// PARITY: block_assembly.py:279-392.
double scoreJunction(int bi, int bj,
                     const std::string& label_a,
                     const std::string& label_b,
                     const JunctionContext& ctx)
{
    // source_boundary = min(bi + 1, n_beats - 1) — block_assembly.py:307.
    const int source_boundary = std::min(bi + 1, ctx.n_beats - 1);

    // PARITY: block_assembly.py:310-316 — graduated energy penalty.
    double energy_diff    = 0.0;
    double energy_penalty = 0.0;
    if (ctx.has_edge_db) {
        energy_diff = std::abs(ctx.edge_db_end[bi] - ctx.edge_db_start[bj]);
        if (energy_diff > BLOCK_ENERGY_PENALTY_THRESHOLD_DB) {
            energy_penalty = std::min(
                BLOCK_ENERGY_PENALTY_CAP,
                (energy_diff - BLOCK_ENERGY_PENALTY_THRESHOLD_DB) * BLOCK_ENERGY_PENALTY_SLOPE);
        }
    }

    // PARITY: block_assembly.py:319-323 — vocal activity readouts with OOB guard.
    double va_i = 0.0;
    double va_j = 0.0;
    if (ctx.vocal_activity != nullptr) {
        va_i = (bi < ctx.n_beats) ? ctx.vocal_activity[bi] : 0.0;
        va_j = (bj < ctx.n_beats) ? ctx.vocal_activity[bj] : 0.0;
    }

    // PARITY: block_assembly.py:325-332 — waveform xcorr.
    std::optional<double> waveform_sim;
    if (ctx.has_wf && source_boundary < ctx.n_beats) {
        const float* src_wave =
            ctx.boundary_waveforms +
            static_cast<std::size_t>(source_boundary) * ctx.n_samples_per_bnd;
        const float* tgt_wave =
            ctx.boundary_waveforms +
            static_cast<std::size_t>(bj) * ctx.n_samples_per_bnd;
        auto [sim, _lag] = dsp::WaveformXcorr::compute(
            src_wave, tgt_wave,
            static_cast<std::size_t>(ctx.n_samples_per_bnd),
            static_cast<std::size_t>(ctx.n_samples_per_bnd),
            ctx.max_lag);
        (void)_lag;
        waveform_sim = sim;
    }

    // PARITY: block_assembly.py:334-335 — successor similarity (row-shifted).
    const double* fn_src =
        ctx.feats_normed + static_cast<std::size_t>(source_boundary) * ctx.n_features;
    const double* fn_tgt =
        ctx.feats_normed + static_cast<std::size_t>(bj) * ctx.n_features;
    double successor_sim = 0.0;
    for (int k = 0; k < ctx.n_features; ++k) successor_sim += fn_src[k] * fn_tgt[k];

    // PARITY: block_assembly.py:337-345 — edge splice similarity (raw features,
    // NOT pre-normalized; inline norm guard + clip to [-1, 1]).
    std::optional<double> edge_splice_sim;
    if (ctx.edge_features_start != nullptr && ctx.edge_features_end != nullptr) {
        const float* ee =
            ctx.edge_features_end +
            static_cast<std::size_t>(bi) * ctx.n_edge_features;
        const float* es =
            ctx.edge_features_start +
            static_cast<std::size_t>(bj) * ctx.n_edge_features;
        double ne_sq = 0.0, ns_sq = 0.0, dot = 0.0;
        for (int k = 0; k < ctx.n_edge_features; ++k) {
            const double e = static_cast<double>(ee[k]);
            const double s = static_cast<double>(es[k]);
            ne_sq += e * e;
            ns_sq += s * s;
            dot   += e * s;
        }
        const double ne = std::sqrt(ne_sq);
        const double ns = std::sqrt(ns_sq);
        if (ne > BLOCK_NORM_FLOOR && ns > BLOCK_NORM_FLOOR) {
            double v = dot / (ne * ns);
            if (v < -1.0) v = -1.0;
            if (v >  1.0) v =  1.0;
            edge_splice_sim = v;
        }
    }

    // PARITY: block_assembly.py:347-353 — context similarity. Asymmetric:
    //   lo_i = max(0, bi - 2)        → ctx_a = features[lo_i : bi+1].mean(axis=0)
    //   hi_j = min(n_beats, bj + 3)  → ctx_b = features[bj : hi_j].mean(axis=0)
    // Python uses f32 `features.mean(axis=0)` which preserves f32 accumulator;
    // port upcast to f64 for numerical stability — f32-class drift (≤ 3e-8).
    const int lo_i = std::max(0, bi + BLOCK_CONTEXT_BEFORE_LO_OFFSET);
    const int hi_j = std::min(ctx.n_beats, bj + BLOCK_CONTEXT_AFTER_HI_OFFSET);

    std::vector<double> ctx_a(ctx.n_features, 0.0);
    const int count_a = (bi + 1) - lo_i;
    if (count_a > 0) {
        for (int r = lo_i; r < bi + 1; ++r) {
            const float* row = ctx.features + static_cast<std::size_t>(r) * ctx.n_features;
            for (int k = 0; k < ctx.n_features; ++k) ctx_a[k] += static_cast<double>(row[k]);
        }
        const double inv = 1.0 / static_cast<double>(count_a);
        for (int k = 0; k < ctx.n_features; ++k) ctx_a[k] *= inv;
    }

    std::vector<double> ctx_b(ctx.n_features, 0.0);
    // PARITY: block_assembly.py:351 — `features[bj:hi_j].mean(axis=0) if hi_j > bj
    // else features[bj]`. If hi_j == bj (shouldn't happen given +3 offset and
    // bj < n_beats), Python returns the raw single row. Port branches explicitly.
    if (hi_j > bj) {
        const int count_b = hi_j - bj;
        for (int r = bj; r < hi_j; ++r) {
            const float* row = ctx.features + static_cast<std::size_t>(r) * ctx.n_features;
            for (int k = 0; k < ctx.n_features; ++k) ctx_b[k] += static_cast<double>(row[k]);
        }
        const double inv = 1.0 / static_cast<double>(count_b);
        for (int k = 0; k < ctx.n_features; ++k) ctx_b[k] *= inv;
    } else if (bj < ctx.n_beats) {
        const float* row = ctx.features + static_cast<std::size_t>(bj) * ctx.n_features;
        for (int k = 0; k < ctx.n_features; ++k) ctx_b[k] = static_cast<double>(row[k]);
    }

    double na_sq = 0.0, nb_sq = 0.0, ab_dot = 0.0;
    for (int k = 0; k < ctx.n_features; ++k) {
        na_sq  += ctx_a[k] * ctx_a[k];
        nb_sq  += ctx_b[k] * ctx_b[k];
        ab_dot += ctx_a[k] * ctx_b[k];
    }
    const double na = std::sqrt(na_sq);
    const double nb = std::sqrt(nb_sq);
    double context_sim = 0.0;
    if (na > BLOCK_NORM_FLOOR && nb > BLOCK_NORM_FLOOR) {
        context_sim = ab_dot / (na * nb);
    }

    // PARITY: block_assembly.py:355-356 — chroma similarity on pre-normalized chroma.
    const double* cn_src =
        ctx.chroma_normed + static_cast<std::size_t>(source_boundary) * ctx.n_chroma;
    const double* cn_tgt =
        ctx.chroma_normed + static_cast<std::size_t>(bj) * ctx.n_chroma;
    // chroma_sim is computed in Python but never passed to compute_quality_score.
    // Python dumps it into an unused local (block_assembly.py:356). We skip it.
    (void)cn_src;
    (void)cn_tgt;

    // PARITY: block_assembly.py:358-360 — label match + bar align + section sim.
    const double label_match =
        (label_a == label_b && label_a != BLOCK_LABEL_UNKNOWN) ? 1.0 : 0.0;
    const double bar_aligned =
        (ctx.pre_db_set->count(bi) > 0 && ctx.db_set->count(bj) > 0) ? 1.0 : 0.0;
    const double section_sim = label_match * BLOCK_SECTION_SIM_SCALE + BLOCK_SECTION_SIM_BIAS;

    // PARITY: block_assembly.py:362-372 — energy/edge_energy/centroid matches.
    double energy_match = 1.0;
    if (ctx.rms_energy != nullptr) {
        energy_match = std::max(
            0.0,
            1.0 - std::abs(ctx.rms_energy[bi] - ctx.rms_energy[bj]) * BLOCK_RMS_DIFF_SCALE);
    }

    double edge_energy_match = 1.0;
    if (ctx.has_edge_db) {
        // 3rd consumer of EDGE_ENERGY_SATURATION_DB (session-18 TransitionCost
        // + session-24 RegionCost retrofit + session-24 BlockAssembly).
        edge_energy_match = std::max(
            0.0,
            1.0 - std::min(energy_diff, EDGE_ENERGY_SATURATION_DB) / EDGE_ENERGY_SATURATION_DB);
    }

    double centroid_match = 1.0;
    if (ctx.spectral_centroid != nullptr) {
        centroid_match = std::max(
            0.0,
            1.0 - std::abs(ctx.spectral_centroid[bi] - ctx.spectral_centroid[bj])
                  * BLOCK_CENTROID_DIFF_SCALE);
    }
    if (ctx.v2 && ctx.baselines != nullptr) {   // ADR-115 E1
        if (ctx.rms_energy != nullptr)
            energy_match = energyQualityV2(*ctx.baselines, ctx.rms_energy[bi], ctx.rms_energy[bj], energy_match);
        if (ctx.has_edge_db)
            edge_energy_match = edgeEnergyQualityV2(*ctx.baselines, energy_diff, edge_energy_match);
        if (ctx.spectral_centroid != nullptr)
            centroid_match = centroidQualityV2(*ctx.baselines, ctx.spectral_centroid[bi], ctx.spectral_centroid[bj], centroid_match);
    }

    // PARITY: block_assembly.py:374-385 — compute_quality_score composition.
    QualityInputs qi{};
    qi.waveform_sim        = waveform_sim;
    qi.successor_sim       = successor_sim;
    qi.edge_splice_sim     = edge_splice_sim;
    qi.context_sim         = context_sim;
    qi.label_match         = label_match;
    qi.section_sim         = section_sim;
    qi.bar_aligned         = bar_aligned;
    qi.energy_match        = energy_match;
    qi.edge_energy_match   = edge_energy_match;
    qi.centroid_match      = centroid_match;
    // ADR-064 sesja 75 — transient continuity from pre-normalised onset.
    if (ctx.onset_norm != nullptr
        && bi < ctx.onset_norm_n && bj < ctx.onset_norm_n) {
        qi.transient_continuity =
            1.0 - std::abs(ctx.onset_norm[(std::size_t) bi]
                         - ctx.onset_norm[(std::size_t) bj]);
        if (ctx.v2 && ctx.baselines != nullptr && ctx.onset_strength != nullptr)
            qi.transient_continuity = onsetQualityV2(
                *ctx.baselines, ctx.onset_strength[bi], ctx.onset_strength[bj],
                *qi.transient_continuity);
    }
    // ADR-066 sesja 77 — MFCC continuity from pre-computed n×n matrix.
    if (ctx.mfcc_continuity_matrix != nullptr
        && bi < ctx.mfcc_continuity_n && bj < ctx.mfcc_continuity_n) {
        qi.mfcc_continuity = ctx.mfcc_continuity_matrix[
            (std::size_t) bi * (std::size_t) ctx.mfcc_continuity_n
          + (std::size_t) bj];
    }
    // ADR-080 RESCOPE + ADR-083 sesja 92 — full-mix chroma continuity from
    // pre-computed n×n matrix. Consumed by Tone slider blend in
    // computeQualityScore. nullopt → blend bypassed.
    if (ctx.chroma_continuity_matrix != nullptr
        && bi < ctx.chroma_continuity_n && bj < ctx.chroma_continuity_n) {
        qi.full_mix_chroma_continuity = ctx.chroma_continuity_matrix[
            (std::size_t) bi * (std::size_t) ctx.chroma_continuity_n
          + (std::size_t) bj];
    }
    // ADR-088 sesja 98 STATUS UPDATE 1 — vocal phrase continuity, fixed
    // formula (see TransitionCost.cpp for rationale).
    //   HIGH boundary signals → HIGH quality (reward alignment).
    //   Silence-gate: vocal_density<0.1 → q=1.0 (instrumental clean).
    //   Soft floor 0.5: prevents harmonic mean composite crash.
    if (ctx.edge_vocal_onset_start != nullptr
        && ctx.edge_vocal_release_end != nullptr
        && bi < ctx.n_beats && bj < ctx.n_beats) {
        const double rel_i = ctx.edge_vocal_release_end[(std::size_t) bi];
        const double on_j  = ctx.edge_vocal_onset_start[(std::size_t) bj];
        const double boundary = std::max(rel_i, on_j);

        double vocal_density = 0.0;
        if (ctx.vocal_activity != nullptr) {
            const double va_i_v = ctx.vocal_activity[(std::size_t) bi];
            const double va_j_v = ctx.vocal_activity[(std::size_t) bj];
            vocal_density = std::max(va_i_v, va_j_v);
        }

        constexpr double kSilenceThreshold = 0.1;
        if (vocal_density < kSilenceThreshold) {
            qi.vocal_continuity = 1.0;
        } else {
            qi.vocal_continuity = 0.5 + 0.5 * boundary;
        }
    }

    double quality = computeQualityScore(
        qi,
        ctx.quality_weights != nullptr ? *ctx.quality_weights
                                       : (ctx.v2 ? kV2QualityWeights : kDefaultQualityWeights));

    // PARITY: block_assembly.py:387-392 — additive penalties, then clamp ≥ 0.
    // energy_penalty first, then vocal penalty (order matters for f64 sums).
    quality -= energy_penalty;
    if (ctx.vocal_activity != nullptr) {
        quality -= computeVocalPenalty(va_i, va_j);
    }
    // FIX-IN-PORT (sesja 71, ADR-059): onset-sustain penalty matches
    // TransitionCost.cpp:792-796 + RegionCost.cpp:546-549 pattern. Python
    // BA does NOT have this — see ADR-059 § Root-cause. Null guard preserves
    // parity when caller (parity tests) doesn't pass onset_strength.
    if (ctx.onset_strength != nullptr) {
        std::optional<double> os_j;
        if (bj < ctx.n_beats) os_j = ctx.onset_strength[bj];
        quality -= computeOnsetPenalty(os_j);
    }

    if (quality < 0.0) quality = 0.0;
    return quality;
}

// PARITY: block_assembly.py:398-409 _get_splice_k.
// Falls back from requested k down to 0 if the kth slot's quality ≤ 0.
std::pair<int, int>
getSpliceK(const BlockCompatResult& compat, int i, int j, int k)
{
    const int K = BLOCK_TOP_K;
    if (k >= K) k = K - 1;
    const std::size_t base = (static_cast<std::size_t>(i) * compat.n + j) * K;
    while (k > 0 && compat.top_k_quality[base + k] <= 0.0) {
        k -= 1;
    }
    return {static_cast<int>(compat.top_k_from[base + k]),
            static_cast<int>(compat.top_k_to  [base + k])};
}


// Sesja 119 (DEV-094): joint junction resolution for the β path.
//
// The legacy assembler takes block k's entry from junction (k-1, k) and its
// exit from junction (k, k+1) as two unrelated top-K lookups, then clamps
// `exit = max(entry, exit)`: with 2-bar outside windows the exit can land
// before the entry and the block silently collapses to one beat, or an all-
// rejected junction reads the zero-initialised slot and splices to beat 0.
// Here every junction keeps its full sorted candidate pool and a small DP
// over (junction, candidate) picks the cheapest combination in which every
// block keeps at least `min_keep_beats` beats (one measured bar, or the
// whole block when shorter). Cost per junction = scale x (1 - quality), the
// quality already carrying the drift / fragment penalties, so whenever the
// independent picks are feasible the DP returns exactly them.
//
// "Try different splice" (variation / junction_variations) forces the
// chosen diverse top-K slot at that junction; the other junctions re-solve
// around it. An empty pool becomes the flagged user boundary
// (`junction_fallback` = 1, quality 0). If no combination is feasible even
// with a one-beat minimum, the independent picks are used with the legacy
// clamp so the remix still renders.
struct JointCandidate
{
    BlockJunctionCandidate c;
    bool fallback = false;
};

RemixPath assembleBlocksJoint(const std::vector<int>&        seq,
                              const std::vector<BlockInfo>&  blocks,
                              int                            n_beats,
                              const BlockCompatResult&       compat,
                              int                            variation,
                              const std::map<int, int>*      junction_variations,
                              double                         edit_length_jump_scale,
                              int                            min_keep_beats)
{
    RemixPath path;
    const int m = static_cast<int>(seq.size());
    const int J = m - 1;

    // Per-junction pools (forced slot, full pool, or flagged fallback).
    std::vector<std::vector<JointCandidate>> P(static_cast<std::size_t>(std::max(0, J)));
    for (int k = 0; k < J; ++k) {
        const int i = seq[static_cast<std::size_t>(k)];
        const int j = seq[static_cast<std::size_t>(k + 1)];
        const auto& pool = compat.pools[static_cast<std::size_t>(i) * compat.n + j];
        auto& Pk = P[static_cast<std::size_t>(k)];

        int kk = variation;
        if (junction_variations != nullptr) {
            auto it = junction_variations->find(k);
            if (it != junction_variations->end()) kk = it->second;
        }
        if (kk > 0 && !pool.empty()) {
            if (kk >= BLOCK_TOP_K) kk = BLOCK_TOP_K - 1;
            const std::size_t base = (static_cast<std::size_t>(i) * compat.n + j) * BLOCK_TOP_K;
            while (kk > 0 && compat.top_k_quality[base + kk] <= 0.0) --kk;
            if (kk > 0) {
                const int ff = static_cast<int>(compat.top_k_from[base + kk]);
                const int tt = static_cast<int>(compat.top_k_to  [base + kk]);
                for (const auto& c : pool)
                    if (c.from_beat == ff && c.to_beat == tt) { Pk.push_back({c, false}); break; }
                if (Pk.empty()) {   // diverse slot beyond the pool cap
                    BlockJunctionCandidate c;
                    c.from_beat = ff; c.to_beat = tt; c.quality = compat.top_k_quality[base + kk];
                    Pk.push_back({c, false});
                }
            }
        }
        if (Pk.empty()) {
            for (const auto& c : pool) Pk.push_back({c, false});
        }
        if (Pk.empty()) {
            BlockJunctionCandidate c;
            c.from_beat = blocks[static_cast<std::size_t>(i)].end_beat - 1;
            c.to_beat   = blocks[static_cast<std::size_t>(j)].start_beat;
            c.quality   = 0.0;
            Pk.push_back({c, true});
        }
    }

    auto entryOf = [&](int b, int choice_prev) {
        return b == 0 ? blocks[static_cast<std::size_t>(seq[0])].start_beat
                      : P[static_cast<std::size_t>(b - 1)][static_cast<std::size_t>(choice_prev)].c.to_beat;
    };
    auto exitOf = [&](int b, int choice) {
        return b == J ? blocks[static_cast<std::size_t>(seq[static_cast<std::size_t>(J)])].end_beat - 1
                      : P[static_cast<std::size_t>(b)][static_cast<std::size_t>(choice)].c.from_beat;
    };

    std::vector<int> choice(static_cast<std::size_t>(std::max(0, J)), 0);
    bool feasible = (J == 0);

    for (int attempt = 0; attempt < 2 && !feasible && J > 0; ++attempt) {
        auto minKeep = [&](int b) {
            const int len = blocks[static_cast<std::size_t>(seq[static_cast<std::size_t>(b)])].n_beats;
            return attempt == 0 ? std::max(1, std::min(len, min_keep_beats)) : 1;
        };
        auto ok = [&](int b, int entry, int exit) { return exit - entry + 1 >= minKeep(b); };

        constexpr double kInf = 1e300;
        std::vector<std::vector<double>> best(static_cast<std::size_t>(J));
        std::vector<std::vector<int>>    from(static_cast<std::size_t>(J));
        for (int k = 0; k < J; ++k) {
            const auto& Pk = P[static_cast<std::size_t>(k)];
            best[static_cast<std::size_t>(k)].assign(Pk.size(), kInf);
            from[static_cast<std::size_t>(k)].assign(Pk.size(), -1);
            for (std::size_t c = 0; c < Pk.size(); ++c) {
                const double cost = edit_length_jump_scale * (1.0 - Pk[c].c.quality);
                if (k == 0) {
                    if (ok(0, entryOf(0, 0), exitOf(0, static_cast<int>(c))))
                        best[0][c] = cost;
                    continue;
                }
                const auto& Pp = P[static_cast<std::size_t>(k - 1)];
                for (std::size_t cp = 0; cp < Pp.size(); ++cp) {
                    const double prev = best[static_cast<std::size_t>(k - 1)][cp];
                    if (prev >= kInf) continue;
                    if (!ok(k, entryOf(k, static_cast<int>(cp)), exitOf(k, static_cast<int>(c)))) continue;
                    if (prev + cost < best[static_cast<std::size_t>(k)][c]) {
                        best[static_cast<std::size_t>(k)][c] = prev + cost;
                        from[static_cast<std::size_t>(k)][c] = static_cast<int>(cp);
                    }
                }
            }
        }
        // Last block: entry from the last junction, exit at its authored end.
        int best_c = -1;
        double best_v = kInf;
        const auto& Pl = P[static_cast<std::size_t>(J - 1)];
        for (std::size_t c = 0; c < Pl.size(); ++c) {
            const double v = best[static_cast<std::size_t>(J - 1)][c];
            if (v >= kInf) continue;
            if (!ok(J, entryOf(J, static_cast<int>(c)), exitOf(J, 0))) continue;
            if (v < best_v) { best_v = v; best_c = static_cast<int>(c); }
        }
        if (best_c < 0) continue;
        feasible = true;
        for (int k = J - 1; k >= 0; --k) {
            choice[static_cast<std::size_t>(k)] = best_c;
            best_c = from[static_cast<std::size_t>(k)][static_cast<std::size_t>(best_c)];
        }
    }
    // Infeasible even at one beat per block: independent best picks (choice
    // stays 0 = highest quality) with the legacy clamp below.

    std::vector<int> all_beats;
    double total_cost = 0.0;
    for (int b = 0; b < m; ++b) {
        const int block_idx = seq[static_cast<std::size_t>(b)];
        int entry_beat = entryOf(b, b > 0 ? choice[static_cast<std::size_t>(b - 1)] : 0);
        int exit_beat  = exitOf(b, b < J ? choice[static_cast<std::size_t>(b)] : 0);
        entry_beat = std::max(0, std::min(entry_beat, n_beats - 1));
        exit_beat  = std::max(feasible ? 0 : entry_beat, std::min(exit_beat, n_beats - 1));
        if (exit_beat < entry_beat) exit_beat = entry_beat;   // defensive (feasible DP never triggers)

        if (b > 0 && !all_beats.empty()) {
            const int last_beat  = all_beats.back();
            const int first_beat = entry_beat;
            if (first_beat != last_beat + 1) {
                const auto& jc = P[static_cast<std::size_t>(b - 1)][static_cast<std::size_t>(choice[static_cast<std::size_t>(b - 1)])];
                path.transitions.emplace_back(last_beat, first_beat);
                path.transition_junctions.push_back(b - 1);   // DEV-109
                total_cost += edit_length_jump_scale * (1.0 - jc.c.quality);
                auto& meta = path.transition_metadata[{last_beat, first_beat}];
                meta["quality_score"]       = jc.c.quality;
                meta["block_transition"]    = BLOCK_TRANSITION_MARKER;
                meta["energy_diff_db"]      = jc.c.energy_diff_db;
                meta["waveform_similarity"] = jc.c.waveform_sim;
                meta["chroma_distance"]     = jc.c.chroma_distance;
                meta["total_cost"]          = 1.0 - jc.c.quality;
                meta["junction_idx"]        = static_cast<double>(b - 1);
                meta["junction_fallback"]   = jc.fallback ? 1.0 : 0.0;
            }
        }
        for (int bb = entry_beat; bb <= exit_beat; ++bb) all_beats.push_back(bb);
        (void) block_idx;
    }

    path.beat_indices   = std::move(all_beats);
    path.total_cost     = total_cost;
    path.duration_beats = static_cast<int>(path.beat_indices.size());
    return path;
}

} // namespace

// --- Public API implementations -----------------------------------------

std::vector<BlockInfo>
buildBlocks(const analysis::Segment* segments,
            int                       n_segments,
            const double*             beat_times,
            int                       n_beats)
{
    std::vector<BlockInfo> blocks;
    // PARITY: block_assembly.py:88-89 — early empty-return.
    if (segments == nullptr || n_segments <= 0 || n_beats < 2) return blocks;

    // PARITY: block_assembly.py:92-104 — first pass: build with raw labels.
    std::map<std::string, int> label_counts;
    blocks.reserve(static_cast<std::size_t>(n_segments));
    for (int idx = 0; idx < n_segments; ++idx) {
        const auto& seg = segments[idx];
        std::string label = seg.label.empty() ? BLOCK_LABEL_UNKNOWN : seg.label;
        const double start_sec  = seg.start;
        const double end_sec    = seg.end;
        const int    cluster_id = seg.cluster_id;

        int start_beat = argminAbs(beat_times, n_beats, start_sec);
        int end_beat   = argminAbs(beat_times, n_beats, end_sec);
        // PARITY: block_assembly.py:103 — `max(start+1, min(end, n_beats))`.
        end_beat = std::max(start_beat + 1, std::min(end_beat, n_beats));

        label_counts[label] = label_counts[label] + 1;

        BlockInfo bi;
        bi.segment_idx  = idx;
        bi.label        = label;
        bi.display_name = capitalizeFirst(label) + " " + std::to_string(label_counts[label]);
        bi.start_beat   = start_beat;
        bi.end_beat     = end_beat;
        bi.start_sec    = start_sec;
        bi.end_sec      = end_sec;
        bi.n_beats      = end_beat - start_beat;
        bi.duration_sec = end_sec - start_sec;
        bi.cluster_id   = cluster_id;
        blocks.push_back(std::move(bi));
    }

    // PARITY: block_assembly.py:123-128 — second pass: if only one instance of
    // a label total, strip the "1" suffix so "Intro 1" → "Intro".
    std::map<std::string, int> final_counts;
    for (const auto& b : blocks) final_counts[b.label] += 1;
    for (auto& b : blocks) {
        if (final_counts[b.label] == 1) b.display_name = capitalizeFirst(b.label);
    }

    return blocks;
}

BlockCompatResult
computeBlockCompatibility(const BlockCompatInputs& in)
{
    BlockCompatResult out;
    out.n = in.n_blocks;
    const int n = in.n_blocks;
    if (n == 0) {
        // PARITY: block_assembly.py:177-183 — empty matrices.
        return out;
    }

    // PARITY: block_assembly.py:185-186 — allocate n × n primary + n × n × K top-K.
    const std::size_t nn  = static_cast<std::size_t>(n) * n;
    const std::size_t nnk = nn * BLOCK_TOP_K;
    out.quality       .assign(nn,  0.0);
    out.splice_from   .assign(nn,  0);
    out.splice_to     .assign(nn,  0);
    out.top_k_quality .assign(nnk, 0.0);
    out.top_k_from    .assign(nnk, 0);
    out.top_k_to      .assign(nnk, 0);

    const int n_beats = in.n_beats;

    // PARITY: block_assembly.py:188-197 — pre-compute normalized feature rows.
    // Chroma slice + L2-normalize (f64 for numerical stability — f32-class drift).
    const auto [cs, ce] = chromaRange(in.n_features);
    const int n_chroma  = ce - cs;
    std::vector<double> chroma_normed(static_cast<std::size_t>(n_beats) * n_chroma, 0.0);
    for (int i = 0; i < n_beats; ++i) {
        const float* src = in.features + static_cast<std::size_t>(i) * in.n_features + cs;
        double norm_sq = 0.0;
        for (int k = 0; k < n_chroma; ++k) {
            const double v = static_cast<double>(src[k]);
            chroma_normed[static_cast<std::size_t>(i) * n_chroma + k] = v;
            norm_sq += v * v;
        }
        double norm = std::sqrt(norm_sq);
        // PARITY: block_assembly.py:192 — `c_norms[c_norms == 0] = 1.0`.
        if (norm == 0.0) norm = 1.0;
        for (int k = 0; k < n_chroma; ++k)
            chroma_normed[static_cast<std::size_t>(i) * n_chroma + k] /= norm;
    }

    std::vector<double> feats_normed(static_cast<std::size_t>(n_beats) * in.n_features, 0.0);
    for (int i = 0; i < n_beats; ++i) {
        const float* src = in.features + static_cast<std::size_t>(i) * in.n_features;
        double norm_sq = 0.0;
        for (int k = 0; k < in.n_features; ++k) {
            const double v = static_cast<double>(src[k]);
            feats_normed[static_cast<std::size_t>(i) * in.n_features + k] = v;
            norm_sq += v * v;
        }
        double norm = std::sqrt(norm_sq);
        if (norm == 0.0) norm = 1.0;
        for (int k = 0; k < in.n_features; ++k)
            feats_normed[static_cast<std::size_t>(i) * in.n_features + k] /= norm;
    }

    // PARITY: block_assembly.py:199-205 — downbeat sets for bar_aligned signal.
    std::set<int> db_set;
    std::set<int> pre_db_set;
    if (in.downbeats != nullptr && in.n_downbeats > 0) {
        for (int k = 0; k < in.n_downbeats; ++k) {
            const int idx = argminAbs(in.beat_times, n_beats, in.downbeats[k]);
            db_set.insert(idx);
        }
        for (int d : db_set) {
            if (d > 0) pre_db_set.insert(d - 1);
        }
    }

    // PARITY: block_assembly.py:207-213 — waveform setup.
    // Python default `waveform_alignment_max_shift_ms=30.0` from config.py:165.
    const bool wf_ptr_ok =
        in.boundary_waveforms != nullptr && in.waveform_sample_rate > 0;
    int max_lag = 0;
    bool has_wf = false;
    if (wf_ptr_ok) {
        const double max_lag_ms = 30.0;  // config.py waveform_alignment_max_shift_ms
        max_lag = static_cast<int>(
            max_lag_ms * static_cast<double>(in.waveform_sample_rate) / 1000.0);
        has_wf = in.n_boundary_waveforms >= n_beats && max_lag > 0;
    }

    // PARITY: block_assembly.py:215-220 — edge dB conversion with 1e-6 floor.
    std::vector<double> edge_db_start_vec;
    std::vector<double> edge_db_end_vec;
    const bool has_edge_db =
        in.edge_rms_start != nullptr && in.edge_rms_end != nullptr;
    if (has_edge_db) {
        edge_db_start_vec.resize(static_cast<std::size_t>(n_beats));
        edge_db_end_vec  .resize(static_cast<std::size_t>(n_beats));
        for (int i = 0; i < n_beats; ++i) {
            edge_db_end_vec  [i] = 20.0 * std::log10(std::max(in.edge_rms_end  [i], BLOCK_DB_FLOOR));
            edge_db_start_vec[i] = 20.0 * std::log10(std::max(in.edge_rms_start[i], BLOCK_DB_FLOOR));
        }
    }

    // Build junction context for _score_junction calls.
    JunctionContext ctx{};
    ctx.n_beats             = n_beats;
    ctx.n_features          = in.n_features;
    ctx.features            = in.features;
    ctx.feats_normed        = feats_normed.data();
    ctx.chroma_normed       = chroma_normed.data();
    ctx.n_chroma            = n_chroma;
    ctx.boundary_waveforms  = in.boundary_waveforms;
    ctx.has_wf              = has_wf;
    ctx.max_lag             = max_lag;
    ctx.n_samples_per_bnd   = in.n_samples_per_bnd;
    ctx.edge_features_start = in.edge_features_start;
    ctx.edge_features_end   = in.edge_features_end;
    ctx.n_edge_features     = in.n_edge_features;
    ctx.edge_db_end         = has_edge_db ? edge_db_end_vec.data()   : nullptr;
    ctx.edge_db_start       = has_edge_db ? edge_db_start_vec.data() : nullptr;
    ctx.has_edge_db         = has_edge_db;
    ctx.rms_energy          = in.rms_energy;
    ctx.spectral_centroid   = in.spectral_centroid;
    ctx.vocal_activity      = in.vocal_activity;
    ctx.onset_strength      = in.onset_strength;  // FIX-IN-PORT sesja 71 (ADR-059)
    // ADR-064 sesja 75 — pre-normalise onset once for transient_continuity.
    const std::vector<double> onset_norm =
        computeOnsetNorm(in.onset_strength, n_beats);
    ctx.onset_norm          = onset_norm.empty() ? nullptr : onset_norm.data();
    ctx.onset_norm_n        = static_cast<int>(onset_norm.size());
    // ADR-066 sesja 77 — pre-compute n×n MFCC continuity matrix once.
    const std::vector<double> mfcc_continuity_matrix =
        computeMfccContinuityMatrix(in.features, n_beats, in.n_features);
    ctx.mfcc_continuity_matrix = mfcc_continuity_matrix.empty()
                               ? nullptr : mfcc_continuity_matrix.data();
    ctx.mfcc_continuity_n      = mfcc_continuity_matrix.empty() ? 0 : n_beats;
    // ADR-080 RESCOPE + ADR-083 sesja 92 — pre-compute n×n full-mix chroma
    // similarity matrix once. Tightly-packed chroma slice extracted from
    // 59-dim features columns [chroma_start, chroma_end) per chromaRange().
    const auto [block_chroma_start, block_chroma_end] = chromaRange(in.n_features);
    const int  block_n_chroma = block_chroma_end - block_chroma_start;
    std::vector<float>  block_chroma_slice;
    std::vector<double> chroma_continuity_matrix;
    if (block_n_chroma > 0 && in.features != nullptr) {
        block_chroma_slice.resize(static_cast<std::size_t>(n_beats) *
                                  static_cast<std::size_t>(block_n_chroma));
        for (int i_b = 0; i_b < n_beats; ++i_b) {
            const float* src = in.features
                             + static_cast<std::size_t>(i_b) * in.n_features
                             + block_chroma_start;
            float* dst = block_chroma_slice.data()
                       + static_cast<std::size_t>(i_b) * block_n_chroma;
            for (int k = 0; k < block_n_chroma; ++k) dst[k] = src[k];
        }
        chroma_continuity_matrix =
            computeChromaContinuityMatrix(block_chroma_slice.data(), n_beats, block_n_chroma);
    }
    ctx.chroma_continuity_matrix = chroma_continuity_matrix.empty()
                                 ? nullptr : chroma_continuity_matrix.data();
    ctx.chroma_continuity_n      = chroma_continuity_matrix.empty() ? 0 : n_beats;
    // ADR-088 sesja 98 — vocal phrase boundary signals (passthrough nullptr-OK).
    ctx.edge_vocal_onset_start   = in.edge_vocal_onset_start;
    ctx.edge_vocal_release_end   = in.edge_vocal_release_end;
    ctx.quality_weights     = in.quality_weights;  // ADR-058 calibration override
    ctx.db_set              = &db_set;
    ctx.pre_db_set          = &pre_db_set;
    // ADR-115 E1 (sesja 114) — v2 sequential baselines over the whole track.
    const SignalBaselines v2_baselines = in.v2_scoring
        ? buildSignalBaselines(in.rms_energy, in.spectral_centroid, in.onset_strength,
                               has_edge_db ? edge_db_end_vec.data()   : nullptr,
                               has_edge_db ? edge_db_start_vec.data() : nullptr, n_beats)
        : SignalBaselines{};
    ctx.baselines           = in.v2_scoring ? &v2_baselines : nullptr;
    ctx.v2                  = in.v2_scoring;

    // ADR-051 — runtime-configurable window radius (UI Splice flexibility
    // slider). Falls back to the hardcoded constant when caller didn't set
    // the field (parity goldens use the constant default).
    const int W = std::max(1, in.search_window_beats);
    const double drift_w = in.drift_penalty_weight;

    // ADR-081 (sesja 96) — β-model candidate-space expansion. When
    // block_assembly_beta == true, replace per-pair ±W candidate search
    // with full-block-interior + outside_window search, fragment penalty,
    // short-block bypass, top-K spatial diversity, min-jump filter,
    // downbeat-only filter, and (when block_sequence_lazy == true) lazy
    // compute restricted to (seq[k], seq[k+1]) junctions.
    //
    // Sesja 119 (DEV-094 / DEV-096): the β path scores every candidate with
    // the pair scorer shared with Region (remix/PairScorer.h) - v2 successor-
    // view energy gate, p98 loudness reject, chroma prefilter, track-level
    // vocal gate - and keeps the full sorted candidate pool per junction
    // (`pools`, capped at BLOCK_POOL_CAP) for the joint resolution in
    // assembleBlocks. The legacy (non-v2) β path keeps the graduated energy
    // penalty instead of the hard gate so the harness "legacy engine" rows
    // keep their meaning.
    //
    // Default block_assembly_beta=false → legacy ±W path runs unchanged
    // (bit-exact baseline + Python parity preserved).
    if (in.block_assembly_beta) {
        const int W_out          = std::max(0, in.outside_window_beats);
        const int short_thresh   = std::max(0, in.short_block_threshold_beats);
        const int min_sep        = std::max(0, in.top_k_min_separation_beats);
        const int min_jump       = std::max(0, in.min_jump_beats);
        const double frag_w      = in.fragment_penalty_weight;
        const bool downbeat_only =
            in.downbeat_only_splices && !db_set.empty() && !pre_db_set.empty();

        // Track-level vocal detection (Region rule, region_cost.py:98-102).
        bool track_has_vocals = false;
        if (in.vocal_activity != nullptr) {
            double max_va = 0.0;
            for (int b = 0; b < n_beats; ++b) max_va = std::max(max_va, in.vocal_activity[b]);
            track_has_vocals = max_va >= TRACK_VOCAL_THRESHOLD;
        }

        PairScorerTrack track{};
        track.n_total             = n_beats;
        track.n_features          = in.n_features;
        track.features            = in.features;
        track.boundary_waveforms  = in.boundary_waveforms;
        track.n_samples_per_bnd   = in.n_samples_per_bnd;
        track.max_lag             = max_lag;
        track.has_waveforms       = has_wf;
        track.edge_db_end         = ctx.edge_db_end;
        track.edge_db_start       = ctx.edge_db_start;
        track.edge_features_start = in.edge_features_start;
        track.edge_features_end   = in.edge_features_end;
        track.n_edge_features     = in.n_edge_features;
        track.rms_energy          = in.rms_energy;
        track.spectral_centroid   = in.spectral_centroid;
        track.onset_strength      = in.onset_strength;
        track.vocal_activity      = in.vocal_activity;
        track.edge_vocal_activity_start = in.edge_vocal_activity_start;
        track.edge_vocal_activity_end   = in.edge_vocal_activity_end;
        track.edge_vocal_onset_start    = in.edge_vocal_onset_start;
        track.edge_vocal_release_end    = in.edge_vocal_release_end;
        track.onset_norm          = ctx.onset_norm;
        track.onset_norm_n        = ctx.onset_norm_n;
        track.mfcc_continuity_matrix   = ctx.mfcc_continuity_matrix;
        track.chroma_continuity_matrix = ctx.chroma_continuity_matrix;
        track.ctx_lo              = 0;
        track.ctx_hi              = n_beats;
        track.v2                  = in.v2_scoring;
        track.baselines           = ctx.baselines;
        track.weights             = ctx.quality_weights != nullptr ? ctx.quality_weights
                                  : (in.v2_scoring ? &kV2QualityWeights : &kDefaultQualityWeights);
        track.track_has_vocals    = track_has_vocals;
        track.gate                = (in.v2_scoring && in.block_energy_gate) ? PairGate::SuccessorView
                                                                             : PairGate::None;
        track.graduated_energy_penalty = ! in.v2_scoring;

        out.pools.assign(nn, {});

        // Build the (i, j) pair list. Lazy mode = only the junctions in the
        // user-arranged block sequence; full mode = n × n.
        std::vector<std::pair<int, int>> pair_list;
        if (in.block_sequence_lazy
            && in.block_sequence != nullptr
            && in.n_block_sequence >= 2) {
            pair_list.reserve(static_cast<std::size_t>(in.n_block_sequence - 1));
            for (int k = 0; k < in.n_block_sequence - 1; ++k) {
                const int a = in.block_sequence[k];
                const int b = in.block_sequence[k + 1];
                if (a < 0 || a >= n || b < 0 || b >= n) continue;
                pair_list.emplace_back(a, b);
            }
        } else {
            pair_list.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    pair_list.emplace_back(i, j);
        }

        for (const auto [i, j] : pair_list) {
            const BlockInfo& bi_block = in.blocks[i];
            const BlockInfo& bj_block = in.blocks[j];

            const int a_end   = bi_block.end_beat;
            const int b_start = bj_block.start_beat;
            const int core_exit  = a_end - 1;
            const int core_entry = b_start;

            // Short-block bypass: either side ≤ threshold → quality-only,
            // no fragment penalty, no min-jump filter. Mirrors sesja-69
            // captured design (short blocks don't have meaningful interior
            // to penalise truncation against).
            const bool bypass_short =
                bi_block.n_beats <= short_thresh
             || bj_block.n_beats <= short_thresh;

            // Adjacent block pair (block_i.end == block_j.start):
            // sequential continuation, min-jump filter doesn't apply.
            const bool adjacent_pair = (a_end == b_start);

            // Search ranges per sesja-69 captured design + Patch 1.
            const int bi_lo = std::max(bi_block.start_beat - W_out, 0);
            const int bi_hi = std::min(a_end + W_out, n_beats - 1);
            const int bj_lo = std::max(b_start - W_out, 0);
            const int bj_hi = std::min(bj_block.end_beat + W_out, n_beats);

            const double label_match =
                (bi_block.label == bj_block.label && bi_block.label != BLOCK_LABEL_UNKNOWN) ? 1.0 : 0.0;
            const double section_sim = label_match * BLOCK_SECTION_SIM_SCALE + BLOCK_SECTION_SIM_BIAS;

            std::vector<BlockJunctionCandidate> candidates;
            const int est = std::max(0, (bi_hi - bi_lo)) * std::max(0, (bj_hi - bj_lo));
            candidates.reserve(static_cast<std::size_t>(std::min(est, 4096)));

            for (int bi = bi_lo; bi < bi_hi; ++bi) {
                if (downbeat_only && pre_db_set.count(bi) == 0) continue;

                for (int bj = bj_lo; bj < bj_hi; ++bj) {
                    if (bi == bj) continue;
                    if (downbeat_only && db_set.count(bj) == 0) continue;

                    // Min-jump filter: skip when |bj - bi| < min_jump for
                    // non-adjacent block pairs (keeps adjacent-block
                    // legitimate sequential continuations).
                    if (!adjacent_pair && !bypass_short) {
                        if (std::abs(bj - bi) < min_jump) continue;
                    }

                    // Chroma prefilter (Region C2, REGION_CHROMA_PREFILTER):
                    // row-shifted chroma cosine distance between the beat
                    // after the cut and the landing beat.
                    const int sb = std::min(bi + 1, n_beats - 1);
                    const double* cn_src = chroma_normed.data() + static_cast<std::size_t>(sb) * n_chroma;
                    const double* cn_tgt = chroma_normed.data() + static_cast<std::size_t>(bj) * n_chroma;
                    double cdot = 0.0;
                    for (int k = 0; k < n_chroma; ++k) cdot += cn_src[k] * cn_tgt[k];
                    const double chroma_distance = 1.0 - std::clamp(cdot, -1.0, 1.0);
                    if (chroma_distance > REGION_CHROMA_PREFILTER) continue;

                    PairScorerRequest req{};
                    req.abs_i       = bi;
                    req.abs_j       = bj;
                    req.label_match = label_match;
                    req.section_sim = section_sim;
                    req.bar_aligned =
                        (pre_db_set.count(bi) > 0 && db_set.count(bj) > 0) ? 1.0 : 0.0;
                    const PairScore score = scorePair(track, req);
                    if (score.rejected) continue;
                    double q = score.quality;

                    // Drift penalty (ADR-051) preserved in β-mode for
                    // continuity with legacy path semantics. drift_w == 0
                    // by default (parity); Block UI sets > 0.
                    if (drift_w > 0.0) {
                        const double drift =
                            (double) (std::abs(bi - core_exit)
                                    + std::abs(bj - core_entry))
                            / (double) std::max(1, W);
                        q -= drift_w * drift;
                    }

                    // Fragment penalty (sesja-69 design, fixed sesja 119 /
                    // DEV-094): the sesja-96 formula measured the DROPPED
                    // fraction as "kept", so the authored boundary paid the
                    // full penalty and truncation paid none. Now dev_i = how
                    // far the cut sits from the authored end of block i, as a
                    // fraction of the block length (truncation and extension
                    // alike - an extra bar changes a 2-bar block, not a 16-bar
                    // one), dev_j the same for the authored start of block j.
                    // The authored boundary (bi = end_i - 1, bj = start_j) is
                    // free. Bypassed for short blocks (no meaningful interior).
                    if (!bypass_short && frag_w > 0.0) {
                        const double len_i = std::max(1, bi_block.n_beats);
                        const double len_j = std::max(1, bj_block.n_beats);
                        const double dev_i = std::min(1.0, std::abs(bi + 1 - a_end) / len_i);
                        const double dev_j = std::min(1.0, std::abs(bj - b_start) / len_j);
                        q -= frag_w * (dev_i + dev_j);
                    }

                    if (q > 0.0) {
                        BlockJunctionCandidate c;
                        c.from_beat       = bi;
                        c.to_beat         = bj;
                        c.quality         = q;
                        c.energy_diff_db  = score.energy_diff_db;
                        c.waveform_sim    = score.waveform_sim;
                        c.chroma_distance = chroma_distance;
                        candidates.push_back(c);
                    }
                }
            }

            // Sort by quality DESC (stable, matches Python Timsort).
            std::stable_sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a.quality > b.quality; });

            // Top-K spatial diversity: greedy fill ensures ≥ min_sep beats
            // separation between selected (bi, bj) tuples. Preserves
            // highest-quality candidate first.
            std::vector<BlockJunctionCandidate> selected;
            selected.reserve(BLOCK_TOP_K);
            for (const auto& cand : candidates) {
                if ((int) selected.size() >= BLOCK_TOP_K) break;
                bool too_close = false;
                if (min_sep > 0) {
                    for (const auto& sel : selected) {
                        const int dbi = std::abs(cand.from_beat - sel.from_beat);
                        const int dbj = std::abs(cand.to_beat - sel.to_beat);
                        if (dbi < min_sep && dbj < min_sep) {
                            too_close = true;
                            break;
                        }
                    }
                }
                if (!too_close) selected.push_back(cand);
            }

            const std::size_t base_k =
                (static_cast<std::size_t>(i) * n + j) * BLOCK_TOP_K;
            for (std::size_t k = 0; k < selected.size(); ++k) {
                out.top_k_quality[base_k + k] = selected[k].quality;
                out.top_k_from   [base_k + k] = static_cast<int64_t>(selected[k].from_beat);
                out.top_k_to     [base_k + k] = static_cast<int64_t>(selected[k].to_beat);
            }

            const std::size_t idx2 = static_cast<std::size_t>(i) * n + j;

            // Dev-only junction diagnostic (sesja 119, DEV-094 / DEV-096):
            // REAMIX_BLOCKS_DEBUG=<path> appends, per block pair, the score
            // of the authored boundary (or the gate that rejected it), the
            // pool size and the five best candidates.
            if (const char* dbg = std::getenv("REAMIX_BLOCKS_DEBUG")) {
                if (FILE* f = std::fopen(dbg, "a")) {
                    PairScorerRequest areq{};
                    areq.abs_i = core_exit; areq.abs_j = core_entry;
                    areq.label_match = label_match; areq.section_sim = section_sim;
                    areq.bar_aligned = (pre_db_set.count(core_exit) > 0 && db_set.count(core_entry) > 0) ? 1.0 : 0.0;
                    const PairScore as = scorePair(track, areq);
                    std::fprintf(f, "# pair %d(%s %d..%d) -> %d(%s %d..%d): authored (%d,%d) %s q=%.4f edb=%.2f bar=%d | pool=%d\n",
                                 i, bi_block.label.c_str(), bi_block.start_beat, a_end,
                                 j, bj_block.label.c_str(), b_start, bj_block.end_beat,
                                 core_exit, core_entry,
                                 as.rejected ? (as.gate == 1 ? "GATE:energy" : "GATE:loudness") : "ok",
                                 as.quality, as.energy_diff_db, (int) areq.bar_aligned,
                                 static_cast<int>(candidates.size()));
                    for (std::size_t k = 0; k < candidates.size() && k < 5; ++k)
                        std::fprintf(f, "#   %d -> %d  q=%.4f  edb=%.2f  cd=%.3f  drift=%d\n",
                                     candidates[k].from_beat, candidates[k].to_beat, candidates[k].quality,
                                     candidates[k].energy_diff_db, candidates[k].chroma_distance,
                                     std::abs(candidates[k].from_beat - core_exit) + std::abs(candidates[k].to_beat - core_entry));
                    std::fclose(f);
                }
            }

            if (candidates.size() > static_cast<std::size_t>(BLOCK_POOL_CAP))
                candidates.resize(static_cast<std::size_t>(BLOCK_POOL_CAP));
            out.pools[idx2] = std::move(candidates);
            if (!selected.empty()) {
                out.quality    [idx2] = selected[0].quality;
                out.splice_from[idx2] = static_cast<int64_t>(selected[0].from_beat);
                out.splice_to  [idx2] = static_cast<int64_t>(selected[0].to_beat);
            } else {
                // No survivor: the user boundary is the fallback. assembleBlocks
                // synthesises the flagged candidate from the empty pool.
                out.splice_from[idx2] = static_cast<int64_t>(a_end - 1);
                out.splice_to  [idx2] = static_cast<int64_t>(b_start);
            }
        }

        return out;
    }

    // PARITY: block_assembly.py:229-272 — outer (i, j) loop over block pairs.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const BlockInfo& bi_block = in.blocks[i];
            const BlockInfo& bj_block = in.blocks[j];

            const int a_end   = bi_block.end_beat;
            const int b_start = bj_block.start_beat;

            // ADR-051 — user-authored boundaries (the "core" the splice
            // should sit near). Drift = |bi - core_exit| + |bj - core_entry|;
            // normalized by W, scaled by drift_penalty_weight, subtracted
            // from quality. Zero penalty when drift_w == 0 (parity path).
            const int core_exit  = a_end - 1;
            const int core_entry = b_start;

            // Candidates tuple list: (quality, from_beat, to_beat).
            std::vector<std::array<double, 3>> candidates;
            // Worst case ~ (2W+1)^2 = 289 per pair; pre-reserve halfway to that.
            candidates.reserve(64);

            // PARITY: block_assembly.py:234-243 — ±W window in both dimensions.
            for (int di = -W; di <= W; ++di) {
                const int bi = a_end - 1 + di;
                // Python L236-237: bi in [block_i.start_beat, min(a_end + W, n_beats - 1)).
                if (bi < bi_block.start_beat) continue;
                if (bi >= std::min(a_end + W, n_beats - 1)) continue;

                for (int dj = -W; dj <= W; ++dj) {
                    const int bj = b_start + dj;
                    // Python L240-241: bj in [max(b_start - W, 0), min(block_j.end_beat, n_beats)).
                    if (bj < std::max(b_start - W, 0)) continue;
                    if (bj >= std::min(bj_block.end_beat, n_beats)) continue;
                    if (bi == bj) continue;

                    double q = scoreJunction(
                        bi, bj,
                        bi_block.label, bj_block.label,
                        ctx);

                    // ADR-051 — graduated penalty for splice drift from user
                    // intent. drift_w == 0 → exact Python parity; > 0 →
                    // splices closer to user-authored boundaries score higher
                    // (algorithm still picks the cleaner splice when the
                    // drift cost is dominated by transition-quality gain).
                    if (drift_w > 0.0) {
                        const double drift =
                            (double) (std::abs(bi - core_exit)
                                    + std::abs(bj - core_entry))
                            / (double) W;
                        q -= drift_w * drift;
                    }

                    if (q > 0.0) {
                        candidates.push_back({q,
                                              static_cast<double>(bi),
                                              static_cast<double>(bj)});
                    }
                }
            }

            // PARITY: block_assembly.py:258 — sort by quality DESC.
            // Python uses `list.sort(key=lambda c: -c[0])` which is stable
            // (Timsort). std::sort is NOT stable; std::stable_sort matches
            // Python semantics for tied quality values.
            std::stable_sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a[0] > b[0]; });

            const std::size_t K_cap = std::min<std::size_t>(BLOCK_TOP_K, candidates.size());
            const std::size_t base_k = (static_cast<std::size_t>(i) * n + j) * BLOCK_TOP_K;
            for (std::size_t k = 0; k < K_cap; ++k) {
                out.top_k_quality[base_k + k] = candidates[k][0];
                out.top_k_from   [base_k + k] = static_cast<int64_t>(candidates[k][1]);
                out.top_k_to     [base_k + k] = static_cast<int64_t>(candidates[k][2]);
            }

            // PARITY: block_assembly.py:265-271 — primary = rank 0 or fallback.
            const std::size_t idx2 = static_cast<std::size_t>(i) * n + j;
            if (!candidates.empty()) {
                out.quality    [idx2] = candidates[0][0];
                out.splice_from[idx2] = static_cast<int64_t>(candidates[0][1]);
                out.splice_to  [idx2] = static_cast<int64_t>(candidates[0][2]);
            } else {
                // PARITY: block_assembly.py:270-271 — fallback to boundary.
                out.splice_from[idx2] = static_cast<int64_t>(a_end - 1);
                out.splice_to  [idx2] = static_cast<int64_t>(b_start);
            }
        }
    }

    return out;
}

RemixPath
assembleBlocks(const std::vector<int>&              block_sequence,
               const std::vector<BlockInfo>&        blocks,
               const double*                        beat_times,
               int                                  n_beats,
               const BlockCompatResult&             compat,
               int                                  variation,
               const std::map<int, int>*            junction_variations,
               double                               edit_length_jump_scale,
               bool                                 allow_outside_window,
               int                                  min_keep_beats)
{
    RemixPath path;

    // PARITY: block_assembly.py:439-443 — empty-in → empty-out.
    if (block_sequence.empty() || blocks.empty()) return path;

    // Sesja 119 (DEV-094): β compat carries per-junction pools → joint path.
    if (!compat.pools.empty()) {
        std::vector<int> seq;
        seq.reserve(block_sequence.size());
        for (int b : block_sequence)
            if (b >= 0 && b < static_cast<int>(blocks.size())) seq.push_back(b);
        if (seq.empty()) return path;
        return assembleBlocksJoint(seq, blocks, n_beats, compat, variation,
                                   junction_variations, edit_length_jump_scale,
                                   std::max(1, min_keep_beats));
    }

    const int n_seq = static_cast<int>(block_sequence.size());
    std::vector<int> all_beats;
    double total_cost = 0.0;

    // PARITY: block_assembly.py:451-504 — main loop.
    for (int seq_idx = 0; seq_idx < n_seq; ++seq_idx) {
        const int block_idx = block_sequence[seq_idx];
        if (block_idx < 0 || block_idx >= static_cast<int>(blocks.size())) continue;
        const BlockInfo& block = blocks[block_idx];

        int entry_beat;
        int exit_beat;

        // PARITY: block_assembly.py:458-465 — entry point resolution.
        if (seq_idx > 0) {
            const int prev_idx = block_sequence[seq_idx - 1];
            const int junc_idx = seq_idx - 1;
            int k = variation;
            if (junction_variations != nullptr) {
                auto it = junction_variations->find(junc_idx);
                if (it != junction_variations->end()) k = it->second;
            }
            auto [_from, to] = getSpliceK(compat, prev_idx, block_idx, k);
            (void)_from;
            entry_beat = to;
        } else {
            entry_beat = block.start_beat;
        }

        // PARITY: block_assembly.py:468-475 — exit point resolution.
        if (seq_idx < n_seq - 1) {
            const int next_idx = block_sequence[seq_idx + 1];
            const int junc_idx = seq_idx;
            int k = variation;
            if (junction_variations != nullptr) {
                auto it = junction_variations->find(junc_idx);
                if (it != junction_variations->end()) k = it->second;
            }
            auto [from, _to] = getSpliceK(compat, block_idx, next_idx, k);
            (void)_to;
            exit_beat = from;
        } else {
            exit_beat = block.end_beat - 1;
        }

        // PARITY: block_assembly.py:477-479 — clamp to block interior.
        // ADR-081 (sesja 96, DEV-043): when allow_outside_window == true,
        // β-model candidates outside [block.start, block.end - 1] land in
        // final audio (clamp loosened to [0, n_beats - 1]). Default false
        // → bit-exact baseline preserved (Python parity + legacy callers).
        if (allow_outside_window) {
            entry_beat = std::max(0,          std::min(entry_beat, n_beats - 1));
            exit_beat  = std::max(entry_beat, std::min(exit_beat,  n_beats - 1));
        } else {
            entry_beat = std::max(block.start_beat, std::min(entry_beat, block.end_beat - 1));
            exit_beat  = std::max(entry_beat,       std::min(exit_beat,  block.end_beat - 1));
        }

        // PARITY: block_assembly.py:482 — inclusive range [entry, exit].
        std::vector<int> block_beats;
        block_beats.reserve(static_cast<std::size_t>(std::max(0, exit_beat - entry_beat + 1)));
        for (int b = entry_beat; b <= exit_beat; ++b) block_beats.push_back(b);

        // PARITY: block_assembly.py:485-502 — record transition from previous block.
        if (seq_idx > 0 && !all_beats.empty()) {
            const int prev_idx  = block_sequence[seq_idx - 1];
            const int last_beat = all_beats.back();
            const int first_beat = entry_beat;

            if (first_beat != last_beat + 1) {
                path.transitions.emplace_back(last_beat, first_beat);
                path.transition_junctions.push_back(seq_idx - 1);   // DEV-109

                int junc_k = variation;
                if (junction_variations != nullptr) {
                    auto it = junction_variations->find(seq_idx - 1);
                    if (it != junction_variations->end()) junc_k = it->second;
                }
                if (junc_k >= BLOCK_TOP_K) junc_k = BLOCK_TOP_K - 1;

                const std::size_t base_k =
                    (static_cast<std::size_t>(prev_idx) * compat.n + block_idx) * BLOCK_TOP_K;
                double quality = compat.top_k_quality[base_k + junc_k];
                if (quality <= 0.0) {
                    quality = compat.quality[static_cast<std::size_t>(prev_idx) * compat.n + block_idx];
                }
                // ADR-084 sesja 93 — Edit Length MULTIPLICATIVE per-junction
                // scale (supersedes sesja-92 ADR-083 additive). Default 1.0
                // → bit-exact baseline. Slider=100 → 4× junction cost (DP
                // biases toward higher-quality junctions). Slider=0 →
                // 0.25× (accepts lower-quality more easily).
                total_cost += edit_length_jump_scale * (1.0 - quality);

                auto& meta = path.transition_metadata[{last_beat, first_beat}];
                meta["quality_score"]     = quality;
                meta["block_transition"]  = BLOCK_TRANSITION_MARKER;
                meta["crossfade_ms"]      = crossfadeForQuality(quality);
            }
        }

        for (int b : block_beats) all_beats.push_back(b);
    }

    path.beat_indices   = std::move(all_beats);
    path.total_cost     = total_cost;
    path.duration_beats = static_cast<int>(path.beat_indices.size());
    (void)beat_times;
    (void)n_beats;
    return path;
}

double crossfadeForQuality(double quality)
{
    // PARITY: block_assembly.py:519-523.
    if (quality >= BLOCK_QUALITY_GREEN)  return BLOCK_CROSSFADE_STANDARD_MS;
    if (quality >= BLOCK_QUALITY_YELLOW) return BLOCK_CROSSFADE_EXTENDED_MS;
    return BLOCK_CROSSFADE_LONG_MS;
}

} // namespace reamix::remix
