// SignalNorm - per-track, sequential-baseline normalisation of splice signals.
//
// ADR-115 E1 (sesja 114). Every distance-type splice signal (energy step in
// dB, spectral-centroid step, onset-strength step, edge-energy step) is mapped
// to a quality q in [0, 1] by ranking the candidate's distance against the
// distribution of the SAME distance over the track's own consecutive beat
// pairs (i -> i+1). Consecutive beats are, by construction, inaudible
// transitions, so the track calibrates its own tolerance: q = share of the
// track's consecutive steps that are WORSE than the candidate. This replaces
// the hand scales (5.0, 12 dB, max-normalisation) that made the legacy q
// values genre- and mastering-dependent (audit: meta/research/
// sesja-114-engine-audit.md section 3.8).
//
// C++-canonical (ADR-065): no Python reference; validated by hand-computed
// cases in tests/parity/test_signal_norm.cpp.
#pragma once

#include <cstddef>
#include <vector>

namespace reamix::remix {

// Empirical distribution of a non-negative "distance" (lower = better).
class DistanceBaseline
{
public:
    // Below this many consecutive pairs a p90 / p98 scale is meaningless;
    // the caller falls back to the legacy formula (research sesja 115).
    static constexpr int    kMinSamples = 32;
    static constexpr double kScalePercentile  = 0.90;
    static constexpr double kRejectPercentile = 0.98;

    DistanceBaseline() = default;
    explicit DistanceBaseline(std::vector<double> samples) { build(std::move(samples)); }

    void build(std::vector<double> samples);

    bool valid() const noexcept { return sorted_.size() >= static_cast<std::size_t>(kMinSamples); }
    std::size_t size() const noexcept { return sorted_.size(); }

    // Quality of a candidate distance: exp(-d / scale()), monotone with no
    // saturation (d = 0 -> 1, d = scale -> e^-1, d = 2 scale -> e^-2 ...).
    // Returns 1.0 when the baseline is not valid so a missing baseline never
    // penalises (caller decides whether to use it); 0.0 for non-finite d.
    double quality(double d) const noexcept;

    // True when d exceeds the track's own p98 consecutive step (ADR-115 E2
    // per-signal reject). False when the baseline is not valid.
    bool reject(double d) const noexcept;

    double scale() const noexcept { return scale_; }         // p90, > 0
    double rejectAbove() const noexcept { return reject_; }  // p98

    // Value at percentile p in [0, 1] (nearest-rank, clamped).
    double percentile(double p) const noexcept;

private:
    std::vector<double> sorted_;
    double scale_  { 1.0 };
    double reject_ { 0.0 };
};

// |v[i+1] - v[i]| over i, optionally in the log domain (guarded by `floor`),
// skipping non-finite samples. Used for rms (log -> dB-like), centroid (log)
// and onset strength (linear).
std::vector<double> sequentialAbsDiff(const double* v, int n,
                                      bool log_domain, double floor = 1e-9);

// |a[i] - b[i+1]| over i for two aligned per-beat arrays (edge end-dB of
// beat i vs edge start-dB of beat i+1).
std::vector<double> sequentialPairDiff(const double* end_of_i,
                                       const double* start_of_j, int n);

// Baselines for the side-channel signals shared by Duration, Region and Block
// scoring. Any pointer may be null; the matching baseline is then invalid and
// the legacy formula for that signal is used instead.
// Sesja 129 (ADR-116 step 2, meta/research/sesja-129-edge-continuity.md):
// per-track scale of the edge-continuity distance. The seam replaces what
// preceded the landing beat j in the original (the end edge of beat j-1) by
// the end edge of the outgoing beat i; d = RMS dB difference across the
// voice-band mel rows of those two edges. Consecutive pairs give d = 0 by
// construction, so the track's own tolerance is the median distance between
// end edges one PHRASE apart (8 bars; one bar when the track is too short):
// the same phrase position a phrase later is what a clean repetition cut
// lands on. Probe: every ok cut <= 0.62 of that scale, every bad one >= 0.81.
struct EdgeContinuityScale
{
    double scale   = 0.0;   // median phrase-lag distance (dB)
    int    samples = 0;

    bool valid() const noexcept
    {
        return samples >= DistanceBaseline::kMinSamples && scale > 0.0;
    }
    // exp(-d / scale); 1.0 when the scale is not valid, 0.0 for non-finite d.
    double quality(double d) const noexcept;
};

// RMS dB difference across the n_mel bands of rows a and b of a row-major
// (n_beats x n_mel) edge matrix.
double edgeMelDistance(const float* edge_mel, int n_mel, int a, int b) noexcept;

// Median of edgeMelDistance(end, b, b + lag) over b; when fewer than
// DistanceBaseline::kMinSamples pairs exist at `lag_beats` the shorter
// `fallback_lag_beats` is tried. Null / empty input -> invalid scale.
EdgeContinuityScale buildEdgeContinuityScale(const float* edge_mel_end, int n_beats,
                                             int n_mel, int lag_beats,
                                             int fallback_lag_beats);

struct SignalBaselines
{
    DistanceBaseline energy;       // log-rms step
    DistanceBaseline centroid;     // log-centroid step
    DistanceBaseline onset;        // onset-strength step
    DistanceBaseline edge_energy;  // edge dB step (end_i vs start_{i+1})
    EdgeContinuityScale edge_continuity;   // sesja 129

    bool any() const noexcept
    {
        return energy.valid() || centroid.valid() || onset.valid() || edge_energy.valid();
    }
};

// Sesja 129: the trailing edge-mel arguments build the edge-continuity scale
// (lag = 8 x bar_beats, fallback one bar); null / zero leaves it invalid.
SignalBaselines buildSignalBaselines(const double* rms_energy,
                                     const double* spectral_centroid,
                                     const double* onset_strength,
                                     const double* edge_db_end,
                                     const double* edge_db_start,
                                     int n_beats,
                                     const float* edge_mel_end = nullptr,
                                     int n_edge_mel = 0,
                                     int bar_beats = 0);

// Sesja 129: edge continuity of the pair (i -> j) = quality of the distance
// between the end edge of beat i and the end edge of beat j-1. Returns
// {quality, d / scale}; quality 1.0 / distance -1.0 when the scale is not
// valid, the arrays are missing, or j == 0 (no original context).
struct EdgeContinuityValue
{
    double quality  = 1.0;
    double distance = -1.0;   // normalised (d / scale), -1 = not available
    bool   available = false;
};
EdgeContinuityValue edgeContinuityV2(const SignalBaselines& b, const float* edge_mel_end,
                                     int n_mel, int n_beats, int i, int j) noexcept;

// V2 signal qualities for a candidate pair (i -> j) - each returns the
// legacy value when its baseline is invalid so callers can substitute freely.
double energyQualityV2(const SignalBaselines& b, double rms_i, double rms_j,
                       double legacy);
double centroidQualityV2(const SignalBaselines& b, double c_i, double c_j,
                         double legacy);
double onsetQualityV2(const SignalBaselines& b, double o_i, double o_j,
                      double legacy);
double edgeEnergyQualityV2(const SignalBaselines& b, double energy_diff_db,
                           double legacy);

// ADR-115 E2 per-signal reject on the two loudness signals (whole-beat RMS
// step, edge dB step): the candidate is worse than the track's own p98
// consecutive step. Never fires when the baseline is invalid.
bool loudnessRejectV2(const SignalBaselines& b, double rms_i, double rms_j,
                      double energy_diff_db);

// Sesja 135 (DEV-121): brightness collapse of an open cut-in, in units of
// the track's own p90 consecutive log-centroid step. Contexts of
// `context_beats`: O = the beats the ear just heard (.. i], L = the landing
// [j ..), P = what preceded the landing in the original [.. j). The remix
// step log(mean L / mean O) minus the song's own DROP into the landing
// min(log(mean L / mean P), 0): a landing the song itself darkens into is
// not charged, a landing the song continues flat into is charged the whole
// step. Negative = the landing is darker than the ear was set up for; a
// rise is never charged (the value is only meaningful when negative). On the
// 31 rated open seams of the sesja-134 rounds the four Daft Punk verse ->
// filtered-outro cut-ins rated "dynamika" sit at -1.95 .. -2.52, every
// rated-clean seam at >= -1.00 (tools/dev/shape_eval/cutin_probe.py).
// 0.0 when the centroid baseline is invalid or the indices are out of range.
double centroidCollapseV2(const SignalBaselines& b, const double* spectral_centroid,
                          int n_beats, int i, int j, int context_beats) noexcept;

} // namespace reamix::remix
