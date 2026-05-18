// test_block_assembly_beta — sesja 96 (ADR-081 STATUS UPDATE 4) self-validation.
//
// Self-validation parity test for the Block Assembly β-model candidate-space
// expansion at junctions (DEV-040 § A) + outside-window unclamp in
// assembleBlocks (DEV-043). Per ADR-065 + memory
// `feedback_python_no_longer_source_of_truth.md`: parity tests for new
// C++-canonical extensions validate against hand-computed values + invariants,
// not vs Python ground truth.
//
// CANONICAL SOURCE: `BlockAssembly.cpp::computeBlockCompatibility` β-branch
// (active when `block_assembly_beta == true`) + `BlockAssembly.cpp::assembleBlocks`
// outside-window unclamp (active when `allow_outside_window == true`).
//
// Test invariants:
//   1. `block_assembly_beta` field default = false (bit-exact baseline
//      preserved → existing block_assembly_parity 16-case corpus passes).
//   2. Legacy path (β=false) completes with well-formed output (n × n quality
//      matrix populated, top_k slots filled per BLOCK_TOP_K).
//   3. Beta path (β=true) on identical fixture produces output distinct from
//      legacy at junctions where the β-search range exceeds ±W (verifies the
//      β-branch is wired through and exercised).
//   4. Fragment penalty monotonicity: with same fixture, β=true + frag_w > 0
//      yields quality ≤ β=true + frag_w = 0 at off-boundary candidates
//      (penalty subtracts from quality).
//   5. Min-jump filter: with min_jump_beats = K, no selected candidate at
//      a non-adjacent block pair has |bj - bi| < K.
//   6. Lazy compute: when block_sequence_lazy=true with queue [a, b, c],
//      only (a, b) and (b, c) cells have non-zero quality; off-sequence
//      cells (e.g. (c, a)) remain at fallback (zero quality).
//   7. assembleBlocks outside-window unclamp: when compat carries splice
//      points outside [block.start, block.end - 1] AND allow_outside_window
//      = true, the resulting path beats include the unclamped values
//      (DEV-043 fix verifies outside-window candidates land in final audio).

#include "remix/BlockAssembly.h"
#include "remix/Path.h"
#include "analysis/StructureResult.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

using reamix::remix::BlockCompatInputs;
using reamix::remix::BlockCompatResult;
using reamix::remix::BlockInfo;
using reamix::remix::RemixPath;
using reamix::remix::assembleBlocks;
using reamix::remix::computeBlockCompatibility;

namespace {

// Synthetic 4-block fixture: 32 beats split into 4 contiguous blocks of 8
// beats each. Features = 4 random-ish but deterministic floats per beat
// (cosine similarity ~0.7-0.95 between distinct beats; not uniform).
//
// Beats 0..8   = block 0 ("intro" label)
// Beats 8..16  = block 1 ("verse" label)
// Beats 16..24 = block 2 ("chorus" label, repeated label sets section_sim base)
// Beats 24..32 = block 3 ("verse" label = same as block 1)

constexpr int kBeats     = 32;
constexpr int kNFeat     = 12;
constexpr int kNBlocks   = 4;
constexpr int kBlockLen  = 8;

struct Fixture {
    std::vector<BlockInfo>  blocks;
    std::vector<double>     beat_times;
    std::vector<float>      features;       // (n_beats, n_feat) row-major
    std::vector<double>     downbeats;      // every 4 beats
    std::vector<double>     edge_rms_start; // (n_beats,)
    std::vector<double>     edge_rms_end;
    std::vector<double>     rms_energy;
    std::vector<double>     spectral_centroid;
};

Fixture buildFixture()
{
    Fixture f;
    f.beat_times.resize(kBeats);
    f.edge_rms_start.resize(kBeats);
    f.edge_rms_end.resize(kBeats);
    f.rms_energy.resize(kBeats);
    f.spectral_centroid.resize(kBeats);
    f.features.assign(static_cast<size_t>(kBeats) * kNFeat, 0.0f);

    // Beat times at 0.5 sec stride, downbeats every 4 beats.
    for (int i = 0; i < kBeats; ++i) {
        f.beat_times[i] = i * 0.5;
        // Synthetic features: mostly stable cluster per block + small drift
        // beat-to-beat. Use pseudo-deterministic xorshift for reproducibility.
        for (int k = 0; k < kNFeat; ++k) {
            double v = std::sin(0.7 * i + 0.3 * k);
            // Block-specific bias so adjacent blocks aren't identical.
            v += 0.3 * (i / kBlockLen);
            f.features[static_cast<size_t>(i) * kNFeat + k] = static_cast<float>(v);
        }
        f.edge_rms_start[i] = 0.05 + 0.001 * (i % 8);
        f.edge_rms_end[i]   = 0.05 + 0.001 * ((i + 1) % 8);
        f.rms_energy[i]     = 0.10 + 0.01 * std::sin(0.5 * i);
        f.spectral_centroid[i] = 0.40 + 0.02 * std::cos(0.4 * i);
    }
    for (int i = 0; i < kBeats; i += 4) f.downbeats.push_back(i * 0.5);

    const std::vector<std::string> labels = { "intro", "verse", "chorus", "verse" };
    for (int b = 0; b < kNBlocks; ++b) {
        BlockInfo info;
        info.segment_idx = b;
        info.label       = labels[b];
        info.display_name= labels[b];
        info.start_beat  = b * kBlockLen;
        info.end_beat    = (b + 1) * kBlockLen;
        info.start_sec   = info.start_beat * 0.5;
        info.end_sec     = info.end_beat   * 0.5;
        info.n_beats     = kBlockLen;
        info.duration_sec= info.end_sec - info.start_sec;
        info.cluster_id  = b;
        f.blocks.push_back(info);
    }
    return f;
}

void wireInputs(BlockCompatInputs& bin, const Fixture& f)
{
    bin.blocks    = f.blocks.data();
    bin.n_blocks  = static_cast<int>(f.blocks.size());
    bin.beat_times= f.beat_times.data();
    bin.n_beats   = static_cast<int>(f.beat_times.size());
    bin.features  = f.features.data();
    bin.n_features= kNFeat;
    bin.edge_rms_start = f.edge_rms_start.data();
    bin.edge_rms_end   = f.edge_rms_end.data();
    bin.rms_energy     = f.rms_energy.data();
    bin.spectral_centroid = f.spectral_centroid.data();
    bin.downbeats   = f.downbeats.data();
    bin.n_downbeats = static_cast<int>(f.downbeats.size());
    bin.time_signature = 4;
    // Default search_window_beats = 8 (legacy ±W path).
}

bool kCheckPassed = true;

void check(bool ok, const char* msg)
{
    std::printf("  %s %s\n", ok ? "[PASS]" : "[FAIL]", msg);
    if (!ok) kCheckPassed = false;
}

// ─── Test 1: default_field_value ──────────────────────────────────────────
void test_default_field_value()
{
    std::printf("[1] default_field_value\n");
    BlockCompatInputs bin{};
    check(bin.block_assembly_beta == false,
          "BlockCompatInputs{}.block_assembly_beta defaults to false");
    check(bin.fragment_penalty_weight == 0.03,
          "BlockCompatInputs{}.fragment_penalty_weight defaults to 0.03");
    check(bin.outside_window_beats == 8,
          "BlockCompatInputs{}.outside_window_beats defaults to 8");
    check(bin.min_jump_beats == 4,
          "BlockCompatInputs{}.min_jump_beats defaults to 4");
    check(bin.downbeat_only_splices == true,
          "BlockCompatInputs{}.downbeat_only_splices defaults to true");
    check(bin.block_sequence_lazy == true,
          "BlockCompatInputs{}.block_sequence_lazy defaults to true");
}

// ─── Test 2: legacy_completes ─────────────────────────────────────────────
void test_legacy_completes()
{
    std::printf("[2] legacy_completes (β=false, ±W=8)\n");
    auto f = buildFixture();
    BlockCompatInputs bin{};
    wireInputs(bin, f);
    bin.block_assembly_beta = false;

    auto compat = computeBlockCompatibility(bin);
    check(compat.n == kNBlocks, "compat.n == n_blocks");
    check(static_cast<int>(compat.quality.size()) == kNBlocks * kNBlocks,
          "compat.quality has n × n entries");
    check(static_cast<int>(compat.top_k_quality.size()) ==
            kNBlocks * kNBlocks * reamix::remix::BLOCK_TOP_K,
          "compat.top_k_quality has n × n × K entries");

    // At least one adjacent-block junction should have non-zero quality.
    bool any_nonzero = false;
    for (int i = 0; i < kNBlocks - 1; ++i) {
        const size_t idx = static_cast<size_t>(i) * kNBlocks + (i + 1);
        if (compat.quality[idx] > 0.0) { any_nonzero = true; break; }
    }
    check(any_nonzero, "Legacy path produces non-zero adjacent-junction quality");
}

// ─── Test 3: beta_distinct_from_legacy ────────────────────────────────────
void test_beta_distinct_from_legacy()
{
    std::printf("[3] beta_distinct_from_legacy\n");
    auto f = buildFixture();
    std::vector<int> queue = { 0, 1, 2, 3 };

    BlockCompatInputs bin_legacy{};
    wireInputs(bin_legacy, f);
    bin_legacy.block_assembly_beta = false;
    auto compat_legacy = computeBlockCompatibility(bin_legacy);

    BlockCompatInputs bin_beta{};
    wireInputs(bin_beta, f);
    bin_beta.block_assembly_beta    = true;
    bin_beta.block_sequence         = queue.data();
    bin_beta.n_block_sequence       = static_cast<int>(queue.size());
    bin_beta.fragment_penalty_weight= 0.03;
    bin_beta.short_block_threshold_beats = 0;  // disable bypass for this test
    bin_beta.top_k_min_separation_beats = 0;   // disable diversity for direct compare
    bin_beta.outside_window_beats   = 8;
    bin_beta.min_jump_beats         = 4;
    bin_beta.downbeat_only_splices  = false;   // disable filter for direct compare
    auto compat_beta = computeBlockCompatibility(bin_beta);

    // At least one queue-junction (0→1, 1→2, 2→3) should differ between
    // legacy and β paths. β candidate range exceeds legacy ±8 around the
    // boundary because the β-branch can also penalise via fragment cost
    // and select different optimum.
    bool any_distinct = false;
    for (int k = 0; k < static_cast<int>(queue.size()) - 1; ++k) {
        const int a = queue[k];
        const int b = queue[k + 1];
        const size_t idx = static_cast<size_t>(a) * kNBlocks + b;
        if (std::abs(compat_legacy.quality[idx] - compat_beta.quality[idx]) > 1e-9) {
            any_distinct = true;
            break;
        }
        if (compat_legacy.splice_from[idx] != compat_beta.splice_from[idx]) {
            any_distinct = true;
            break;
        }
        if (compat_legacy.splice_to[idx]   != compat_beta.splice_to[idx]) {
            any_distinct = true;
            break;
        }
    }
    check(any_distinct, "β-path yields distinct splice / quality vs legacy on at least one queue junction");
}

// ─── Test 4: fragment_penalty_monotonicity ────────────────────────────────
void test_fragment_penalty_monotonicity()
{
    std::printf("[4] fragment_penalty_monotonicity\n");
    auto f = buildFixture();
    std::vector<int> queue = { 0, 1, 2, 3 };

    auto run = [&](double frag_w) {
        BlockCompatInputs bin{};
        wireInputs(bin, f);
        bin.block_assembly_beta = true;
        bin.block_sequence      = queue.data();
        bin.n_block_sequence    = static_cast<int>(queue.size());
        bin.fragment_penalty_weight = frag_w;
        bin.short_block_threshold_beats = 0;     // no bypass
        bin.top_k_min_separation_beats  = 0;     // no diversity filter
        bin.outside_window_beats        = 0;     // confine to block interior
        bin.min_jump_beats              = 0;     // no min-jump filter
        bin.downbeat_only_splices       = false; // no downbeat filter
        return computeBlockCompatibility(bin);
    };

    auto compat_zero = run(0.0);
    auto compat_pos  = run(0.10);

    // For every queue-junction cell, quality with positive fragment penalty
    // must be ≤ quality with zero penalty. (Equality occurs only when the
    // selected splice happens to be at exact block boundary on both sides.)
    bool monotone = true;
    bool strict_lt_at_least_once = false;
    for (int k = 0; k < static_cast<int>(queue.size()) - 1; ++k) {
        const int a = queue[k];
        const int b = queue[k + 1];
        const size_t idx = static_cast<size_t>(a) * kNBlocks + b;
        const double q0 = compat_zero.quality[idx];
        const double qp = compat_pos.quality[idx];
        if (qp > q0 + 1e-9) { monotone = false; }
        if (qp < q0 - 1e-9) { strict_lt_at_least_once = true; }
    }
    check(monotone, "fragment_penalty_weight > 0 never RAISES quality");
    check(strict_lt_at_least_once,
          "fragment_penalty_weight > 0 strictly LOWERS quality on ≥1 junction");
}

// ─── Test 5: min_jump_filter ──────────────────────────────────────────────
void test_min_jump_filter()
{
    std::printf("[5] min_jump_filter\n");
    auto f = buildFixture();
    // Use non-adjacent queue (0 → 2) so min-jump filter applies.
    std::vector<int> queue = { 0, 2 };

    BlockCompatInputs bin{};
    wireInputs(bin, f);
    bin.block_assembly_beta = true;
    bin.block_sequence      = queue.data();
    bin.n_block_sequence    = static_cast<int>(queue.size());
    bin.fragment_penalty_weight     = 0.03;
    bin.short_block_threshold_beats = 0;
    bin.top_k_min_separation_beats  = 0;
    bin.outside_window_beats        = 0;
    bin.min_jump_beats              = 6;     // require |bj-bi| ≥ 6
    bin.downbeat_only_splices       = false;

    auto compat = computeBlockCompatibility(bin);
    const size_t base_k =
        (static_cast<size_t>(queue[0]) * kNBlocks + queue[1])
            * reamix::remix::BLOCK_TOP_K;
    bool ok = true;
    int  populated = 0;
    for (int k = 0; k < reamix::remix::BLOCK_TOP_K; ++k) {
        if (compat.top_k_quality[base_k + k] <= 0.0) continue;
        ++populated;
        const int bi = static_cast<int>(compat.top_k_from[base_k + k]);
        const int bj = static_cast<int>(compat.top_k_to  [base_k + k]);
        if (std::abs(bj - bi) < 6) { ok = false; break; }
    }
    check(ok, "All selected candidates satisfy |bj - bi| ≥ min_jump_beats");
    check(populated > 0, "min_jump filter still leaves at least one candidate");
}

// ─── Test 6: lazy_compute ─────────────────────────────────────────────────
void test_lazy_compute()
{
    std::printf("[6] lazy_compute\n");
    auto f = buildFixture();
    std::vector<int> queue = { 0, 1, 2 };  // 3 not in queue

    BlockCompatInputs bin{};
    wireInputs(bin, f);
    bin.block_assembly_beta = true;
    bin.block_sequence      = queue.data();
    bin.n_block_sequence    = static_cast<int>(queue.size());
    bin.block_sequence_lazy = true;
    bin.fragment_penalty_weight     = 0.03;
    bin.short_block_threshold_beats = 0;
    bin.top_k_min_separation_beats  = 0;
    bin.outside_window_beats        = 4;
    bin.min_jump_beats              = 0;
    bin.downbeat_only_splices       = false;

    auto compat = computeBlockCompatibility(bin);

    // On-sequence cells (0,1) and (1,2) should be computed (non-zero).
    const size_t idx_01 = 0 * kNBlocks + 1;
    const size_t idx_12 = 1 * kNBlocks + 2;
    check(compat.quality[idx_01] > 0.0 || compat.splice_from[idx_01] != 0
                                       || compat.splice_to[idx_01] != 0,
          "Lazy: queue-junction (0→1) populated");
    check(compat.quality[idx_12] > 0.0 || compat.splice_from[idx_12] != 0
                                       || compat.splice_to[idx_12] != 0,
          "Lazy: queue-junction (1→2) populated");

    // Off-sequence cell (3, 0) — block 3 not in queue. Must remain at
    // zero-init (no compute). Use top_k_quality slot 0 as canary.
    const size_t base_30 = (static_cast<size_t>(3) * kNBlocks + 0)
                            * reamix::remix::BLOCK_TOP_K;
    bool off_seq_zero = true;
    for (int k = 0; k < reamix::remix::BLOCK_TOP_K; ++k) {
        if (compat.top_k_quality[base_30 + k] != 0.0) {
            off_seq_zero = false;
            break;
        }
    }
    check(off_seq_zero, "Lazy: off-sequence cell (3→0) skipped (zero-init)");
}

// ─── Test 7: assembleBlocks_outside_window_unclamp (DEV-043) ──────────────
void test_assemble_outside_window_unclamp()
{
    std::printf("[7] assembleBlocks_outside_window_unclamp (DEV-043)\n");
    auto f = buildFixture();

    // Build a synthetic compat where splice points lie OUTSIDE block
    // interior on purpose: (0→1) splice_from beat 9 (inside block 1, NOT
    // block 0), splice_to beat 6 (inside block 0, NOT block 1). These are
    // exactly the kind of outside-window candidates the β-model emits.
    BlockCompatResult compat;
    compat.n = kNBlocks;
    const size_t nn  = static_cast<size_t>(kNBlocks) * kNBlocks;
    const size_t nnk = nn * reamix::remix::BLOCK_TOP_K;
    compat.quality.assign(nn, 0.0);
    compat.splice_from.assign(nn, 0);
    compat.splice_to.assign(nn, 0);
    compat.top_k_quality.assign(nnk, 0.0);
    compat.top_k_from.assign(nnk, 0);
    compat.top_k_to.assign(nnk, 0);

    // Junction (0→1): outside-window splice points.
    const size_t idx_01 = 0 * kNBlocks + 1;
    compat.quality[idx_01] = 0.85;
    compat.splice_from[idx_01] = 9;   // beat 9 = INSIDE block 1, OUTSIDE block 0
    compat.splice_to[idx_01]   = 6;   // beat 6 = INSIDE block 0, OUTSIDE block 1
    const size_t base_01 = idx_01 * reamix::remix::BLOCK_TOP_K;
    compat.top_k_quality[base_01] = 0.85;
    compat.top_k_from[base_01] = 9;
    compat.top_k_to[base_01]   = 6;

    std::vector<int> queue = { 0, 1 };

    // Default allow_outside_window = false: clamp engaged, splice_to=6 in
    // block 1 gets clamped to block_1.start_beat=8.
    auto path_clamped = assembleBlocks(queue, f.blocks,
                                       f.beat_times.data(), kBeats,
                                       compat, 0, nullptr, 1.0,
                                       /*allow_outside_window=*/false);
    bool clamp_ok = !path_clamped.beat_indices.empty()
                  && path_clamped.beat_indices.front() >= 0
                  && path_clamped.beat_indices.back()  <= kBeats - 1;
    // Clamped path should NOT contain beat 6 in block 1's range (block 1 is
    // [8..16), so 6 would be OOB; clamped to 8).
    bool path_missing_6_in_block1 = true;
    for (int i = 1; i < static_cast<int>(path_clamped.beat_indices.size()); ++i) {
        // Beats in second block segment must be ≥ 8 after clamp.
        if (path_clamped.beat_indices[i] < 8 && path_clamped.beat_indices[i - 1] >= 8) {
            path_missing_6_in_block1 = false;
            break;
        }
    }
    check(clamp_ok, "Clamped path (allow_outside_window=false) is well-formed");
    check(path_missing_6_in_block1,
          "Clamped path drops outside-window beat 6 from block 1 segment");

    // allow_outside_window = true: unclamped, splice_to=6 should appear as
    // entry beat for block 1 segment.
    auto path_unclamp = assembleBlocks(queue, f.blocks,
                                       f.beat_times.data(), kBeats,
                                       compat, 0, nullptr, 1.0,
                                       /*allow_outside_window=*/true);
    bool found_outside_beat = false;
    for (int b : path_unclamp.beat_indices) {
        if (b == 6) { found_outside_beat = true; break; }
    }
    check(found_outside_beat,
          "Unclamped path (allow_outside_window=true) includes outside-window beat 6");
}

}  // namespace

int main()
{
    std::printf("test_block_assembly_beta — sesja 96 (ADR-081 STATUS UPDATE 4)\n");
    test_default_field_value();
    test_legacy_completes();
    test_beta_distinct_from_legacy();
    test_fragment_penalty_monotonicity();
    test_min_jump_filter();
    test_lazy_compute();
    test_assemble_outside_window_unclamp();
    if (!kCheckPassed) {
        std::printf("test_block_assembly_beta: FAILED\n");
        return 1;
    }
    std::printf("test_block_assembly_beta: ALL PASS\n");
    return 0;
}
