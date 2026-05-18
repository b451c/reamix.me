#include "remix/Optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "analysis/RepetitionMap.h"
#include "analysis/StructureResult.h"
#include "remix/SegmentData.h"
#include "remix/TransitionCost.h"
#include "remix/ViterbiDP.h"

namespace reamix::remix {

namespace {

// ---------------------------------------------------------------------------
// Helpers (file-local)
// ---------------------------------------------------------------------------

// Python 3 `round(x)` returns int using banker's rounding (round-half-to-
// even). `std::nearbyint` with the default IEEE 754 rounding mode
// FE_TONEAREST matches this exactly. PARITY: same pattern as session-6
// `SpectralContrast.cpp::rintToEven` + `CBMSegmenter.cpp` precedents.
//
// Used for `int(round(target_duration / avg_beat_duration))` at
// optimizer.py:199 + `int(round(adaptive_tolerance / avg_beat_duration))`
// at optimizer.py:208-211. Session-21 48-case corpus validated bit-exact
// `dp_params` via this rounding — the session-20 dump tool's
// `compute_dp_params` Python helper also uses `int(round(...))`.
inline int pyIntRound(double x)
{
    return static_cast<int>(std::nearbyint(x));
}

// Case-insensitive equality of ASCII strings. Python `.lower() == X` where
// X is already lowercase. We fold the input to lowercase and compare.
bool asciiLowerEquals(const std::string& s, const char* lowerTarget)
{
    const std::size_t n = s.size();
    if (n != std::char_traits<char>::length(lowerTarget)) return false;
    for (std::size_t k = 0; k < n; ++k) {
        char c = s[k];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
        if (c != lowerTarget[k]) return false;
    }
    return true;
}

// `argmin(|beat_times - target_time|)`. Ties: np.argmin picks FIRST index.
// C++ std::min_element has the same first-tie semantic (returns first iter
// satisfying strict-less comparator). std::abs on f64.
int argMinAbsDelta(const double* beat_times, int n_beats, double target_time)
{
    int    best_idx  = 0;
    double best_diff = std::abs(beat_times[0] - target_time);
    for (int k = 1; k < n_beats; ++k) {
        const double diff = std::abs(beat_times[k] - target_time);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx  = k;
        }
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// UNJUSTIFIED constants from optimizer.py main-path (ADR-026 inline audit)
// ---------------------------------------------------------------------------
//
// All literals below have a `optimizer.py:LINE (2026-04-22)` citation. The
// `K_` prefix distinguishes them from header-exposed constants (which
// follow the `kFoo` naming). Placed in anonymous namespace — module-private.

// B2: avg_beat_duration fallback when n_beats <= 1. Implicit 120 BPM.
// UNJUSTIFIED. `optimizer.py:129 (2026-04-22)`.
constexpr double K_FALLBACK_AVG_BEAT_DURATION = 0.5;

// C2: `max_outro = 3 * time_signature` — 3 bars maximum outro reservation.
// UNJUSTIFIED (no citation in Python). `optimizer.py:182 (2026-04-22)`.
constexpr int K_MAX_OUTRO_BARS = 3;

// D1: target_beats floor. DEFENSIVE. `optimizer.py:199 (2026-04-22)`.
constexpr int K_MIN_TARGET_BEATS = 2;

// D2: adaptive_tolerance floor in seconds. UNJUSTIFIED.
// `optimizer.py:207 (2026-04-22)`.
constexpr double K_MIN_ADAPTIVE_TOLERANCE_SEC = 2.0;

// D3: target_ratio clip bounds for adaptive tolerance. UNJUSTIFIED.
// `optimizer.py:207 (2026-04-22)`.
constexpr double K_TOL_RATIO_FLOOR = 0.4;
constexpr double K_TOL_RATIO_CEIL  = 1.0;

// D4: tolerance_beats floor. DEFENSIVE. `optimizer.py:208-211 (2026-04-22)`.
constexpr int K_MIN_TOLERANCE_BEATS = 2;

// D5: max_beats cap multiplier (target_beats + tolerance, capped at N×
// n_beats). UNJUSTIFIED. `optimizer.py:213 (2026-04-22)`.
constexpr int K_MAX_BEATS_MULTIPLIER = 3;

// D6: target_ratio threshold below which intro/outro get scaled down.
// UNJUSTIFIED. `optimizer.py:223 (2026-04-22)`.
constexpr double K_ADAPTIVE_SCALING_THRESHOLD = 0.8;

// D7: intro_scale floor + min intro beats after scaling. UNJUSTIFIED.
// `optimizer.py:224-225 (2026-04-22)`.
constexpr double K_INTRO_SCALE_FLOOR = 0.25;
constexpr int    K_MIN_INTRO_BEATS   = 4;

// D8: outro_scale floor + min outro beats after scaling. UNJUSTIFIED.
// `optimizer.py:226-227 (2026-04-22)`.
constexpr double K_OUTRO_SCALE_FLOOR = 0.3;
constexpr int    K_MIN_OUTRO_BEATS   = 2;

// E1: min_jumps gate — target_ratio threshold below which we force ≥ 1
// non-sequential jump. UNJUSTIFIED. `optimizer.py:247 (2026-04-22)`.
constexpr double K_MIN_JUMPS_RATIO_THRESHOLD = 0.45;

// E2: adaptive cooldown coefficient.
// `max(1, int(COOLDOWN_BARS * target_ratio * 2))` at optimizer.py:256.
// UNJUSTIFIED. Ported verbatim — session-21 48-case corpus (16 × r0.3 =
// 16 cases) validated bit-exact with this integer-coerce formula.
constexpr double K_ADAPTIVE_COOLDOWN_COEFF = 2.0;

// E2 threshold: target_ratio below which adaptive cooldown fires.
// `optimizer.py:255 (2026-04-22)`.
constexpr double K_ADAPTIVE_COOLDOWN_THRESHOLD = 0.5;

// E5: post-DP outro extension cap — `max(4, int(len(path) * 0.2))` at
// optimizer.py:297. UNJUSTIFIED.
// ZERO CURRENT PARITY COVERAGE: ViterbiDP parity (sessions 20+21) checked
// only raw `dp_path` — the 20% extension branch is FIRST tested in session
// -22 `test_optimizer`. If smoke fails with remix_path mismatch while
// dp_path matches, the bug is here.
constexpr int    K_MIN_EXTENSION_BEATS       = 4;
constexpr double K_MAX_EXTENSION_PATH_RATIO  = 0.2;

// F2: `is_repetition_jump = 1.0` metadata marker.
// `optimizer.py:336 (2026-04-22)`.
constexpr double K_REPETITION_JUMP_MARKER    = 1.0;

// Metadata keys — mirror optimizer.py:321-338 Python dict keys.
constexpr const char* K_META_QUALITY_SCORE        = "quality_score";
constexpr const char* K_META_WAVEFORM_SIM         = "waveform_similarity";
constexpr const char* K_META_SUCCESSOR_SIM        = "successor_similarity";
constexpr const char* K_META_EDGE_SPLICE_SIM      = "edge_splice_similarity";
constexpr const char* K_META_CHROMA_DISTANCE      = "chroma_distance";
constexpr const char* K_META_ENERGY_DIFF_DB       = "energy_diff_db";
constexpr const char* K_META_ALIGNMENT_OFFSET_SEC = "alignment_offset_sec";
constexpr const char* K_META_TOTAL_COST           = "total_cost";
constexpr const char* K_META_IS_REPETITION_JUMP   = "is_repetition_jump";
constexpr const char* K_META_CHROMA_CORRELATION   = "chroma_correlation";

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor — port of optimizer.py::CleanOptimizer.__init__ (L83-161)
// ---------------------------------------------------------------------------
CleanOptimizer::CleanOptimizer(const CleanOptimizerInputs& in)
    : W_(in.W),
      candidates_(in.candidates),
      n_beats_(in.n_beats),
      jumps_(in.jumps),
      n_jumps_(in.n_jumps),
      // B1: `max(1, time_signature)` defensive lower bound (Python L106).
      time_signature_(std::max(1, in.time_signature)),
      preserve_intro_(in.preserve_intro_beats),
      preserve_outro_(in.preserve_outro_beats),
      max_transitions_(in.max_transitions),
      duration_tolerance_sec_(in.duration_tolerance_sec),
      sample_rate_(in.sample_rate)
{
    // Defensive input validation. Python relies on numpy broadcast errors for
    // mismatched shapes; C++ needs explicit guards for raw-pointer inputs.
    if (n_beats_ < 0 || in.beat_times == nullptr) {
        throw std::invalid_argument("CleanOptimizer: invalid beat_times / n_beats");
    }
    if (W_ == nullptr && n_beats_ > 0) {
        throw std::invalid_argument("CleanOptimizer: W cannot be null when n_beats > 0");
    }

    // Python L104: `self._beat_times = beat_times.astype(np.float64)`.
    // The caller may pass a different dtype view; we always hold f64 copy.
    beat_times_.assign(in.beat_times, in.beat_times + n_beats_);

    // Python L109-110: phrase protection adapts to time signature (4/4 → 16,
    // 3/4 → 12, 6/8 → 24). Session-20 ADR-026 cite chain: COOLDOWN_BARS
    // at `viterbi_dp.py:34 (2026-04-21)` × `time_signature` caller.
    //
    // NOTE: these ATTRIBUTES are later passed to `_viterbi_dp` as
    // `min_seq_after_jump` / `min_forward_jump` in the ratio ≥ 0.5 branch
    // of `_run_dp_and_build_path` (Python L260-261). The `_build_neighbors`
    // call (Python L152) uses a DIFFERENT value (`time_signature` alone).
    // Session-20 CRITICAL CALLER OVERRIDE: do NOT conflate the two.
    using reamix::remix::COOLDOWN_BARS;
    // ADR-083 sesja 92 — Min cut UI slider override. Default 0 → legacy
    // compute COOLDOWN_BARS × time_signature (= 16 for 4/4) → bit-exact
    // baseline. UI passes user slider value 4-32 beats when slider
    // dragged off default 16.
    if (in.min_seq_after_jump_override > 0) {
        min_seq_after_jump_ = in.min_seq_after_jump_override;
        min_forward_jump_   = in.min_seq_after_jump_override;
        min_seq_after_jump_user_override_ = true;  // skip adaptive scaling per ADR-083
    } else {
        min_seq_after_jump_ = COOLDOWN_BARS * time_signature_;
        min_forward_jump_   = COOLDOWN_BARS * time_signature_;
    }
    // ADR-084 sesja 93 — Edit Length multiplicative jump-cost scale
    // propagated to ViterbiDP. Supersedes sesja-92 ADR-083 additive design.
    edit_length_jump_scale_ = in.edit_length_jump_scale;

    // Python L111: `self._segments = list(segments or [])`. Own a copy so
    // the caller can free its buffer. `analysis::Segment` is a plain struct
    // (POD-ish), copy is cheap.
    if (in.segments != nullptr && in.n_segments > 0) {
        segments_.assign(in.segments, in.segments + in.n_segments);
    }

    // Python L119-130: beat durations + cumulative timeline.
    if (n_beats_ > 1) {
        // avg_beat_duration = (last - first) / (n_beats - 1). Not the mean
        // of diff(beat_times); this is Python's chosen estimator even when
        // instantaneous tempo varies.
        // Python L120-122.
        avg_beat_duration_ =
            (beat_times_[n_beats_ - 1] - beat_times_[0])
            / static_cast<double>(n_beats_ - 1);

        // durations = diff(beat_times) + append(avg_beat_duration)
        // cumulative = [0.0] + cumsum(durations)
        // Python L123-127. Output shape: (n_beats + 1,).
        cumulative_.resize(static_cast<std::size_t>(n_beats_) + 1);
        cumulative_[0] = 0.0;
        double acc = 0.0;
        for (int k = 0; k < n_beats_ - 1; ++k) {
            acc += beat_times_[k + 1] - beat_times_[k];
            cumulative_[static_cast<std::size_t>(k) + 1] = acc;
        }
        // Last element: cumulative[n_beats] = cumulative[n_beats-1] + avg.
        acc += avg_beat_duration_;
        cumulative_[static_cast<std::size_t>(n_beats_)] = acc;
    } else {
        // B2/B3: n_beats <= 1 fallback. `avg=0.5 sec`, `cumulative=[0.0, 0.5]`.
        // Python L128-130.
        avg_beat_duration_ = K_FALLBACK_AVG_BEAT_DURATION;
        cumulative_        = {0.0, K_FALLBACK_AVG_BEAT_DURATION};
    }

    // Python L132-135: segment_data via shared SegmentData module (session 19).
    // Python passes `segments` directly (may be empty list); C++ passes
    // pointer + size with nullptr-when-empty signal.
    const analysis::Segment* seg_ptr =
        segments_.empty() ? nullptr : segments_.data();
    const int seg_n = static_cast<int>(segments_.size());
    segment_data_ = computeSegmentData(n_beats_,
                                       seg_ptr,
                                       seg_n,
                                       beat_times_.data(),
                                       in.features,
                                       in.n_features);

    // Python L137-142: downbeat arrays via ViterbiDP helper (session 20).
    // `downbeat_constraint=true` matches config.py:108 production setting;
    // the `false` branch is not threaded through the facade (no consumer).
    downbeat_arrays_ = computeDownbeatArrays(beat_times_.data(),
                                             n_beats_,
                                             in.downbeats,
                                             in.n_downbeats,
                                             time_signature_,
                                             /*downbeat_constraint=*/true);

    // Python L144-146: segment-label-aware intro/outro.
    effective_intro_ = findIntroEnd();
    effective_outro_ = findOutroBeats();

    // Python L148-161: neighbor CSR.
    //
    // CRITICAL SESSION-20 CALLER OVERRIDE (re-confirmed session-21 corpus):
    // `min_forward_jump = time_signature` (= 4 for 4/4), NOT the declared
    // default 16. The comment at Python L148-151 explains why: neighbors
    // must include short-range candidates so the DP can use them when the
    // cooldown is reduced for aggressive shortening. The actual cooldown
    // is enforced IN THE DP (adaptive per target_ratio); neighbors keep
    // the full 1-bar-and-above spectrum.
    const int min_neighbor_jump = time_signature_;  // optimizer.py:152

    // buildNeighbors takes a nullable DownbeatArrays pointer. Session-20
    // impl checks `.valid` + non-empty sets internally.
    const DownbeatArrays* db_ptr =
        downbeat_arrays_.valid ? &downbeat_arrays_ : nullptr;

    neighbors_ = buildNeighbors(n_beats_,
                                W_,
                                jumps_,
                                n_jumps_,
                                db_ptr,
                                segment_data_.boundary_beats,
                                kMaxNeighborsDefault,
                                min_neighbor_jump);
}

// ---------------------------------------------------------------------------
// _find_intro_end — port of optimizer.py L163-172
// ---------------------------------------------------------------------------
//
// Scan segments for a "intro"-labeled one (case-insensitive exact match).
// If found, map segment.end (seconds) to nearest beat index and return
// max(default, beat_idx). Otherwise return default.
int CleanOptimizer::findIntroEnd() const
{
    const int default_val = preserve_intro_;
    if (n_beats_ <= 0) return default_val;

    for (const auto& seg : segments_) {
        // Python L167-168: `label = seg.get("label", "").lower()`.
        if (asciiLowerEquals(seg.label, "intro")) {
            // Python L169-170: `end_time = float(seg.get("end", 0.0))`;
            // `beat_idx = int(argmin(|beat_times - end_time|))`.
            const double end_time = seg.end;
            const int    beat_idx = argMinAbsDelta(beat_times_.data(),
                                                   n_beats_,
                                                   end_time);
            // Python L171: `return max(default, beat_idx)`.
            return std::max(default_val, beat_idx);
        }
    }
    return default_val;
}

// ---------------------------------------------------------------------------
// _find_outro_beats — port of optimizer.py L174-192
// ---------------------------------------------------------------------------
//
// Similar to _find_intro_end but with a 3-bar max-cap and nested min/max
// for tail size. Critical: port the `max(default, min(full_outro,
// max_outro))` idiom VERBATIM — do NOT simplify via DRY.
int CleanOptimizer::findOutroBeats() const
{
    // C2: `max_outro = 3 * time_signature` — 3 bars maximum.
    // `optimizer.py:182 (2026-04-22)` UNJUSTIFIED.
    const int max_outro = K_MAX_OUTRO_BARS * time_signature_;
    // Python L183: `default = min(preserve_outro, max_outro)`.
    const int default_val = std::min(preserve_outro_, max_outro);

    if (n_beats_ <= 0) return default_val;

    for (const auto& seg : segments_) {
        if (asciiLowerEquals(seg.label, "outro")) {
            // Python L187-189.
            const double start_time = seg.start;
            const int    beat_idx   = argMinAbsDelta(beat_times_.data(),
                                                     n_beats_,
                                                     start_time);
            const int full_outro    = n_beats_ - beat_idx;
            // Python L191: `return max(default, min(full_outro, max_outro))`.
            // Nested min/max idiom — 3-bar cap applies first, then we
            // ensure we do not shrink below default. Port VERBATIM.
            return std::max(default_val, std::min(full_outro, max_outro));
        }
    }
    return default_val;
}

// ---------------------------------------------------------------------------
// _dp_params — port of optimizer.py L194-236
// ---------------------------------------------------------------------------
//
// Computes the DP parameters for a given target duration. 9 numeric steps
// classified in weights-audit.md session 22 (D1-D9).
//
// D9 BENIGN-DUPLICATION: Python computes `target_ratio = target_duration /
// max(1.0, original_duration)` TWICE (L206 + L222). Port VERBATIM. Session
// -21 48-case corpus validated the dump tool's `compute_dp_params` (which
// also duplicates) bit-exact against Python.
CleanOptimizer::DpParams
CleanOptimizer::computeDpParams(double target_duration) const
{
    // Python L198: `original_duration = float(self._cumulative[self._n_beats])`.
    // `_cumulative` has size n_beats + 1 with last element = total track
    // duration + 1 avg-beat-duration (per the L123-127 append).
    const double original_duration = cumulative_[static_cast<std::size_t>(n_beats_)];

    // Python L199: `target_beats = max(2, int(round(target_duration /
    //                                               avg_beat_duration)))`.
    // D1 K_MIN_TARGET_BEATS=2, pyIntRound = banker's rounding (Python int(round())).
    const int target_beats = std::max(
        K_MIN_TARGET_BEATS,
        pyIntRound(target_duration / avg_beat_duration_));

    // Python L200: `is_shortening = target_duration < original_duration`.
    const bool is_shortening = target_duration < original_duration;

    // D9 #1 — FIRST target_ratio computation. Drives adaptive_tolerance.
    // Python L206: `target_ratio = target_duration / max(1.0, original_duration)`.
    const double target_ratio_tol =
        target_duration / std::max(1.0, original_duration);

    // D2/D3: adaptive tolerance + clip.
    // Python L207: `adaptive_tolerance = max(2.0, duration_tolerance_sec *
    //                                       max(0.4, min(1.0, target_ratio)))`.
    // Clip target_ratio to [0.4, 1.0] before multiplying; then floor the
    // product at 2.0 seconds.
    //
    // ADR-084 sesja 93 — Allow ± slider floor relaxation. When user drags
    // slider below the 2.0-sec floor (UI surface 0..15 sec), respect their
    // explicit intent down to 0.5 sec instead of silently clamping to 2.0.
    // sesja-92 ADR-083 wired the slider but `max(K_MIN_ADAPTIVE_TOLERANCE_SEC,
    // ...)` was eating slider values 0/1/2 — they all collapsed to 2 sec
    // effective. Per scout sesja-93 verdict: user-explicit override
    // authoritative below default floor.
    const double clipped_tol_ratio = std::max(
        K_TOL_RATIO_FLOOR,
        std::min(K_TOL_RATIO_CEIL, target_ratio_tol));
    const double effective_tol_floor =
        (duration_tolerance_sec_ < K_MIN_ADAPTIVE_TOLERANCE_SEC)
            ? std::max(0.5, duration_tolerance_sec_)   // user-explicit <2 → relaxed
            : K_MIN_ADAPTIVE_TOLERANCE_SEC;            // default behavior preserved
    const double adaptive_tolerance = std::max(
        effective_tol_floor,
        duration_tolerance_sec_ * clipped_tol_ratio);

    // D4: tolerance_beats floor at 2 (relaxed to 1 when explicit <2 sec).
    // Python L208-211: `max(2, int(round(adaptive_tolerance / avg)))`.
    const int beats_floor =
        (duration_tolerance_sec_ < K_MIN_ADAPTIVE_TOLERANCE_SEC)
            ? 1                       // ADR-084 sesja 93 — user-explicit relaxed
            : K_MIN_TOLERANCE_BEATS;  // default 2-beat floor preserved
    const int tolerance_beats = std::max(
        beats_floor,
        pyIntRound(adaptive_tolerance / avg_beat_duration_));

    // Python L212: `min_beats = max(2, target_beats - tolerance_beats)`.
    const int min_beats = std::max(K_MIN_TARGET_BEATS,
                                   target_beats - tolerance_beats);

    // D5: max_beats cap at n_beats × 3.
    // Python L213: `max_beats = min(n_beats * 3, target_beats + tolerance_beats)`.
    const int max_beats = std::min(n_beats_ * K_MAX_BEATS_MULTIPLIER,
                                   target_beats + tolerance_beats);

    // Python L215-216: copy effective intro/outro (may be mutated by
    // adaptive scaling below — must match Python's mutation semantics).
    int outro_beats = effective_outro_;
    int intro_beats = effective_intro_;

    // D9 #2 — SECOND target_ratio computation. DUPLICATE of #1 (same
    // expression). Python computes it twice (L206 + L222). Port VERBATIM,
    // do NOT DRY — keeping the duplicate preserves line-for-line parity
    // and guards against accidental semantic change during future
    // refactor. f64 arithmetic is deterministic so the two values are
    // bit-identical; the duplicate is purely a Python code-style choice.
    const double target_ratio_scale =
        target_duration / std::max(1.0, original_duration);

    // D6/D7/D8: adaptive intro/outro scaling (prevents intro+outro from
    // eating the entire budget on aggressive shortening).
    // Python L223-227.
    if (target_ratio_scale < K_ADAPTIVE_SCALING_THRESHOLD) {
        // D7: `intro_scale = max(0.25, target_ratio)`; then `intro_beats =
        //      max(4, int(intro_beats * intro_scale))`.
        // Python L224-225.
        const double intro_scale = std::max(K_INTRO_SCALE_FLOOR, target_ratio_scale);
        intro_beats = std::max(
            K_MIN_INTRO_BEATS,
            static_cast<int>(static_cast<double>(intro_beats) * intro_scale));

        // D8: `outro_scale = max(0.3, target_ratio)`; then `outro_beats =
        //      max(2, int(outro_beats * outro_scale))`.
        // Python L226-227.
        const double outro_scale = std::max(K_OUTRO_SCALE_FLOOR, target_ratio_scale);
        outro_beats = std::max(
            K_MIN_OUTRO_BEATS,
            static_cast<int>(static_cast<double>(outro_beats) * outro_scale));
    }

    DpParams params;
    params.target_beats  = target_beats;
    params.is_shortening = is_shortening;
    params.intro_beats   = intro_beats;
    params.outro_beats   = outro_beats;
    params.effective_max = max_beats;
    params.effective_min = std::max(K_MIN_TARGET_BEATS, min_beats);  // Python L235
    return params;
}

// ---------------------------------------------------------------------------
// _run_dp_and_build_path — port of optimizer.py L238-303
// ---------------------------------------------------------------------------
//
// Core pipeline: adaptive cooldown branch → viterbiDP call → empty-path
// fallback → post-DP outro extension → extractRemixPath.
//
// `W` taken as non-const pointer because `remix()` may have mutated it
// via blocked_transitions; the pointer flows unchanged through this
// function (DP kernel does not mutate W).
RemixPath
CleanOptimizer::runDpAndBuildPath(double* W, const DpParams& params) const
{
    // E1: min_jumps gate for aggressive shortening.
    // Python L246-247: `target_ratio = target_beats / max(1, n_beats)`;
    //                  `min_jumps = 1 if ratio<0.45 and is_shortening else 0`.
    const double target_ratio =
        static_cast<double>(params.target_beats)
        / static_cast<double>(std::max(1, n_beats_));
    const int min_jumps =
        (target_ratio < K_MIN_JUMPS_RATIO_THRESHOLD && params.is_shortening)
            ? 1
            : 0;

    // E2: adaptive cooldown for aggressive shortening.
    // Python L255-261: if ratio<0.5, cooldown_bars = max(1, int(COOLDOWN_BARS
    //                  * target_ratio * 2)); cooldown = cooldown_bars × TS.
    //                  else: use the phrase-safe default min_seq_after_jump_
    //                  / min_forward_jump_.
    //
    // Session-21 48-case corpus (16 r0.3 cases) validated bit-exact with
    // `int(COOLDOWN_BARS * target_ratio * 2)` via Python-style integer
    // truncation toward zero (positive operands only).
    int adaptive_cooldown;
    int adaptive_forward;
    // ADR-083 sesja 92 — UI Min cut slider override takes precedence over
    // algorithm's adaptive cooldown scaling. Without this guard, dragging
    // Min cut on an aggressive-shortening track (target_ratio < 0.5) would
    // be silently ignored — user reported "Min cut nic" sesja 92 close.
    if (min_seq_after_jump_user_override_) {
        adaptive_cooldown = min_seq_after_jump_;
        adaptive_forward  = min_forward_jump_;
    }
    else if (target_ratio < K_ADAPTIVE_COOLDOWN_THRESHOLD) {
        using reamix::remix::COOLDOWN_BARS;
        const double scaled = static_cast<double>(COOLDOWN_BARS)
                              * target_ratio
                              * K_ADAPTIVE_COOLDOWN_COEFF;
        // Python `int(x)` for positive x = truncation toward zero =
        // static_cast<int>(x) in C++ (no rounding).
        const int adaptive_cooldown_bars =
            std::max(1, static_cast<int>(scaled));
        adaptive_cooldown = adaptive_cooldown_bars * time_signature_;
        adaptive_forward  = adaptive_cooldown;
    } else {
        adaptive_cooldown = min_seq_after_jump_;
        adaptive_forward  = min_forward_jump_;
    }

    // -- DP call ------------------------------------------------------------
    // Python L263-281: _viterbi_dp(...).
    ViterbiDPInputs dp_in{};
    dp_in.W                   = W;
    dp_in.n_beats             = n_beats_;
    dp_in.target_length       = params.effective_max;
    dp_in.min_target_length   = params.effective_min;
    dp_in.intro_beats         = params.intro_beats;
    dp_in.outro_beats         = params.outro_beats;
    dp_in.is_shortening       = params.is_shortening;

    dp_in.neighbor_indices      = neighbors_.indices.data();
    dp_in.n_neighbor_indices    = static_cast<int>(neighbors_.indices.size());
    dp_in.neighbor_offsets      = neighbors_.offsets.data();

    dp_in.beat_to_segment       = segment_data_.beat_to_segment.data();
    dp_in.seg_sim_matrix        =
        segment_data_.seg_sim.empty() ? nullptr : segment_data_.seg_sim.data();
    dp_in.n_segs                = segment_data_.n_segs;

    // Python `has_bar_info = pre_downbeat_arr is not None and downbeat_arr
    //                        is not None` at viterbi_dp.py:175.
    // C++ ViterbiDP already handles `nullptr` pre/downbeat arrays via its
    // own `has_bar_info` check (session-20 impl L175-177 of ViterbiDP.cpp).
    const bool has_bar_info = downbeat_arrays_.valid
                              && !downbeat_arrays_.pre_downbeat_arr.empty()
                              && !downbeat_arrays_.downbeat_arr.empty();
    dp_in.pre_downbeat_arr = has_bar_info ? downbeat_arrays_.pre_downbeat_arr.data()
                                          : nullptr;
    dp_in.downbeat_arr     = has_bar_info ? downbeat_arrays_.downbeat_arr.data()
                                          : nullptr;

    dp_in.max_transitions    = max_transitions_;
    dp_in.min_jumps          = min_jumps;
    dp_in.min_seq_after_jump = adaptive_cooldown;
    dp_in.min_forward_jump   = adaptive_forward;
    // min_segment_beats: leave default (kMinSegmentBeatsDefault=16, matches
    // config.py:105). Python `_viterbi_dp` reads `DEFAULT_CONFIG.remix
    // .min_segment_beats` at runtime (viterbi_dp.py:237) — same value.
    // ADR-084 sesja 93 — Edit Length multiplicative jump-cost scale propagated.
    dp_in.edit_length_jump_scale = edit_length_jump_scale_;

    ViterbiPath dp_result = viterbiDP(dp_in);
    std::vector<std::int64_t> path = std::move(dp_result.path);
    double                    cost = dp_result.total_cost;

    // E4: empty-path fallback.
    // Python L283-287: `if len(path) == 0: path = arange(min(target_beats,
    //                  n_beats)); cost = 0.0`.
    if (path.empty()) {
        const int fill_count = std::min(params.target_beats, n_beats_);
        path.resize(static_cast<std::size_t>(std::max(0, fill_count)));
        for (int k = 0; k < fill_count; ++k) {
            path[static_cast<std::size_t>(k)] = static_cast<std::int64_t>(k);
        }
        cost = 0.0;
    }

    // E5: post-DP outro extension (20% cap, min 4 beats).
    // Python L289-301. Fires only when DP's last beat is in outro region
    // AND there's track remaining. Extends path.
    //
    // ZERO CURRENT PARITY COVERAGE — session-22 `test_optimizer` is the
    // first gate on this branch. If smoke produces drift here with
    // dp_path bit-exact, this is the bisection target.
    if (params.outro_beats > 0 && !path.empty()) {
        const int last_beat   = static_cast<int>(path.back());
        const int outro_start = n_beats_ - params.outro_beats;
        const int remaining   = n_beats_ - 1 - last_beat;
        if (last_beat >= outro_start && remaining > 0) {
            // Python L297: `max_extend = max(4, int(len(path) * 0.2))`.
            // `int(path * 0.2)` = truncation (positive).
            const int max_extend = std::max(
                K_MIN_EXTENSION_BEATS,
                static_cast<int>(static_cast<double>(path.size())
                                 * K_MAX_EXTENSION_PATH_RATIO));
            // Python L298: `extend_to = min(n_beats, last_beat + 1 + max_extend)`.
            const int extend_to = std::min(n_beats_, last_beat + 1 + max_extend);
            // Python L299: `extend = arange(last_beat + 1, extend_to, dtype=int64)`.
            // Python L300-301: `if len(extend) > 0: path = concat([path, extend])`.
            for (int b = last_beat + 1; b < extend_to; ++b) {
                path.push_back(static_cast<std::int64_t>(b));
            }
        }
    }

    // Python L303.
    return extractRemixPath(path, cost);
}

// ---------------------------------------------------------------------------
// _extract_remix_path — port of optimizer.py L305-348
// ---------------------------------------------------------------------------
//
// Converts a raw beat-index path + total cost into a RemixPath with
// transitions + per-transition metadata. Iterates consecutive path pairs,
// emits non-sequential (j != i+1) pairs as transitions, populates metadata
// from `candidates` map + optional `repetition_map.get_jumps_from(i)` scan.
RemixPath
CleanOptimizer::extractRemixPath(const std::vector<std::int64_t>& path,
                                 double                            cost) const
{
    RemixPath out;
    out.total_cost     = cost;
    out.duration_beats = static_cast<int>(path.size());

    // beat_indices: promote int64 path → int (Path.h convention). Session
    // -16 Path.h comment L21-22 documents the int-vs-int64 convention at
    // the serialization boundary — matches StructureResult.cluster_id
    // pattern. Range check: int64 beat index ≤ ~1B in any realistic track.
    out.beat_indices.reserve(path.size());
    for (std::int64_t idx : path) {
        out.beat_indices.push_back(static_cast<int>(idx));
    }

    if (path.size() < 2) {
        // Python: the `for idx in range(len(path) - 1)` loop at L312 does
        // not execute when path is empty or size-1. Return with empty
        // transitions + metadata (matching Python dataclass defaults).
        return out;
    }

    // Python L312-341: per-transition metadata build.
    for (std::size_t idx = 0; idx + 1 < path.size(); ++idx) {
        const int i = static_cast<int>(path[idx]);
        const int j = static_cast<int>(path[idx + 1]);
        if (j == i + 1) continue;  // sequential step, not a transition

        out.transitions.emplace_back(i, j);

        // Python L317: `candidate = self._candidates.get((i, j))`. Returns
        // None when absent; C++ equivalent is map::find.
        std::map<std::string, double> meta;
        if (candidates_ != nullptr) {
            auto it = candidates_->find(std::make_pair(i, j));
            if (it != candidates_->end()) {
                const TransitionCandidate& c = it->second;
                // Python L320-332: 8 keys from candidate fields.
                meta[K_META_QUALITY_SCORE]   = c.quality_score;
                meta[K_META_WAVEFORM_SIM]    = c.waveform_similarity;
                meta[K_META_SUCCESSOR_SIM]   = c.successor_similarity;
                meta[K_META_EDGE_SPLICE_SIM] = c.edge_splice_similarity;
                meta[K_META_CHROMA_DISTANCE] = c.chroma_distance;
                meta[K_META_ENERGY_DIFF_DB]  = c.energy_diff_db;
                // F1: `alignment_offset_sec = alignment_lag_samples /
                //      max(1, sample_rate)`. Python L327-330.
                meta[K_META_ALIGNMENT_OFFSET_SEC] =
                    static_cast<double>(c.alignment_lag_samples)
                    / static_cast<double>(std::max(1, sample_rate_));
                meta[K_META_TOTAL_COST] = c.total_cost;
            }
        }

        // Python L333-338: repetition jump detection. Iterate jumps-from-i,
        // find first with to_beat == j; set marker + chroma_correlation;
        // break on first match.
        if (jumps_ != nullptr && n_jumps_ > 0) {
            for (int k = 0; k < n_jumps_; ++k) {
                const analysis::RepetitionJump& jp = jumps_[k];
                if (jp.fromBeat == i) {
                    if (jp.toBeat == j) {
                        // F2: `is_repetition_jump = 1.0`.
                        meta[K_META_IS_REPETITION_JUMP] = K_REPETITION_JUMP_MARKER;
                        meta[K_META_CHROMA_CORRELATION] = jp.chromaCorrelation;
                        break;
                    }
                }
            }
        }

        out.transition_metadata[std::make_pair(i, j)] = std::move(meta);
    }

    return out;
}

// ---------------------------------------------------------------------------
// remix — port of optimizer.py::remix (L350-387)
// ---------------------------------------------------------------------------
RemixPath
CleanOptimizer::remix(double                                        target_duration,
                      const std::set<std::pair<int, int>>*          blocked_transitions)
{
    // Python L364-370: `n_beats < 2` short-circuit. Return sequential path,
    // cost=0, empty transitions + metadata.
    if (n_beats_ < 2) {
        RemixPath out;
        out.total_cost     = 0.0;
        out.duration_beats = n_beats_;
        out.beat_indices.reserve(static_cast<std::size_t>(std::max(0, n_beats_)));
        for (int k = 0; k < n_beats_; ++k) {
            out.beat_indices.push_back(k);
        }
        return out;
    }

    // Python L372: _dp_params(target_duration).
    const DpParams params = computeDpParams(target_duration);

    // Python L373: `W = self._W` — reference assignment, NOT copy. We hold
    // the same caller-owned buffer; mutate + restore below.
    double* W = W_;

    // Python L376-381: apply blocked_transitions.
    // Save old values for restoration. Guard bounds (Python uses
    // `0 <= from_b < W.shape[0]`).
    std::vector<std::tuple<int, int, double>> restored;
    if (blocked_transitions != nullptr && !blocked_transitions->empty()) {
        restored.reserve(blocked_transitions->size());
        for (const auto& pr : *blocked_transitions) {
            const int from_b = pr.first;
            const int to_b   = pr.second;
            if (from_b >= 0 && from_b < n_beats_
                && to_b   >= 0 && to_b   < n_beats_) {
                const std::size_t offset =
                    static_cast<std::size_t>(from_b) * static_cast<std::size_t>(n_beats_)
                    + static_cast<std::size_t>(to_b);
                restored.emplace_back(from_b, to_b, W[offset]);
                W[offset] = INF;  // reamix::remix::INF from TransitionCost.h
            }
        }
    }

    // Python L383-387: try/finally — restore on exit (normal OR exception).
    // C++ RAII: scope-guard via struct with destructor.
    struct Restore
    {
        double* W;
        int     n_beats;
        const std::vector<std::tuple<int, int, double>>& saved;
        ~Restore()
        {
            // Best-effort restore; dtor cannot throw. W may be nullptr in
            // pathological cases (n_beats=0 skip above), guard defensively.
            if (W == nullptr) return;
            const std::size_t stride = static_cast<std::size_t>(n_beats);
            for (const auto& [f, t, val] : saved) {
                W[static_cast<std::size_t>(f) * stride
                  + static_cast<std::size_t>(t)] = val;
            }
        }
    };
    Restore guard{W, n_beats_, restored};

    // Python L384: `return self._run_dp_and_build_path(W, params)`.
    return runDpAndBuildPath(W, params);
}

// ---------------------------------------------------------------------------
// remix_k_best — port of optimizer.py:389-458 (session 58, ADR-048 / DEV-027)
// ---------------------------------------------------------------------------
//
// Algorithm: get best path → for each transition (fb, tb) in best.transitions,
// build augmented_blocked = user_blocked ∪ {(fb,tb)}, re-call remix() with
// augmented set, dedup by beat_indices, accumulate up to k unique paths.
//
// Per ADR-048 Option B: re-uses `remix(target, augmented)` rather than
// exposing `runDpAndBuildPath` privately. Existing RAII guard inside
// `remix()` body handles W mutate + restore on each iteration.
std::vector<RemixPath>
CleanOptimizer::remix_k_best(double                                       target_duration,
                             int                                          k,
                             const std::set<std::pair<int, int>>*         blocked_transitions)
{
    // Python L410-411: n_beats < 2 short-circuit. PORT FIX: pass
    // blocked_transitions through (Python silently drops; C++ preserves
    // parity with `remix()` semantic for caller predictability).
    if (n_beats_ < 2) {
        return std::vector<RemixPath>{ remix(target_duration, blocked_transitions) };
    }

    // Python L425: get base path with user-blocked applied.
    RemixPath best = remix(target_duration, blocked_transitions);

    // Python L431-432: no transitions OR k<=1 short-circuit.
    if (best.transitions.empty() || k <= 1) {
        return std::vector<RemixPath>{ std::move(best) };
    }

    std::vector<RemixPath> paths;
    paths.reserve(static_cast<std::size_t>(k));
    paths.push_back(std::move(best));

    // Python L435: dedup by tuple(beat_indices). C++ uses std::vector<int>
    // directly as set key (lex-ordered, equality-comparable).
    std::set<std::vector<int>> seen;
    seen.insert(paths.front().beat_indices);

    // Capture transitions BEFORE the loop because we mutate `paths` (push_back)
    // inside, and using `paths.front().transitions` directly would be fragile
    // if a future refactor reallocs the vector.
    const std::vector<std::pair<int, int>> base_transitions = paths.front().transitions;

    // Python L437: iterate best.transitions in path order.
    for (const auto& tr : base_transitions) {
        if (paths.size() >= static_cast<std::size_t>(k)) break;

        // Python L442-456: build augmented blocked set + run DP + dedup.
        // C++ implementation: re-call existing remix() with augmented set
        // (per ADR-048 Option B — RAII reuse, no W mutation here).
        std::set<std::pair<int, int>> augmented;
        if (blocked_transitions != nullptr) {
            augmented = *blocked_transitions;
        }
        augmented.insert(tr);

        RemixPath alt = remix(target_duration, &augmented);

        // Python L450: dedup by beat_indices, drop empty alternatives.
        if (!alt.beat_indices.empty() && seen.find(alt.beat_indices) == seen.end()) {
            seen.insert(alt.beat_indices);
            paths.push_back(std::move(alt));
        }
    }

    return paths;
}

// ---------------------------------------------------------------------------
// remix_variation — convenience replicating server `_remix.py:140-147`
// ---------------------------------------------------------------------------
RemixPath
CleanOptimizer::remix_variation(double                                       target_duration,
                                int                                          variation_idx,
                                const std::set<std::pair<int, int>>*         blocked_transitions)
{
    const int v = std::max(0, variation_idx);
    const int k = std::max(2, v + 1);   // Python `_remix.py:143`
    auto paths = remix_k_best(target_duration, k, blocked_transitions);
    if (paths.empty()) {
        // Defensive: remix_k_best invariant guarantees at least [best] for
        // any well-formed instance. Reach here only on pathological input.
        return RemixPath{};
    }
    const int idx = std::min(v, static_cast<int>(paths.size()) - 1);  // Python `_remix.py:146`
    return paths[static_cast<std::size_t>(idx)];
}

} // namespace reamix::remix
