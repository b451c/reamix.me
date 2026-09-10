// SeamJudge — sesja 131 (ADR-116 step 3c / ADR-117 shape-first planner).
//
// A whole-track BOUNDARY seam judge: scores "leave at the end of beat i,
// land on beat j" as a boundary cut (substitution view, kV2BoundaryQuality-
// Weights, the loudness gates on end(i) vs end(j-1) and rms(i) vs rms(j-1))
// through the shared pair scorer (PairScorer.h, PairScorerRequest::boundary).
// It owns every derived array the scorer needs (edge dB, onset norm, MFCC /
// chroma continuity matrices, the v2 signal baselines), built once from the
// same BlockCompatInputs the Blocks path fills (ui/BlockCompatWiring.h), so a
// caller that wants the quality of ONE arbitrary pair - the shape planner's
// section-to-section seams - does not have to go through a mode pool.
//
// The context build mirrors BlockAssembly::computeBlockCompatibility (β path)
// line for line; the two are candidates for one shared builder under ADR-116
// step 4 (acceptance unification) - not refactored here so the Blocks
// junction pools stay bit-exact.
#pragma once

#include "remix/BlockAssembly.h"
#include "remix/PairScorer.h"
#include "remix/SignalNorm.h"

#include <optional>
#include <vector>

namespace reamix::remix
{

class BoundarySeamJudge
{
public:
    explicit BoundarySeamJudge(const BlockCompatInputs& in);

    // Full pair score (rejected + gate id when a hard gate fired).
    // `relaxed` = without the per-track p98 loudness reject (the 8 dB hard
    // block stays) - the shape planner's tiers D / E.
    PairScore score(int i, int j, bool relaxed = false) const;

    // Convenience: quality when no gate fired, nullopt otherwise.
    std::optional<double> quality(int i, int j, bool relaxed = false) const;

    bool valid() const noexcept { return valid_; }
    int  nBeats() const noexcept { return track_.n_total; }
    const PairScorerTrack& track() const noexcept { return track_; }

private:
    bool                valid_ { false };
    std::vector<double> edge_db_start_;
    std::vector<double> edge_db_end_;
    std::vector<double> onset_norm_;
    std::vector<double> mfcc_continuity_;
    std::vector<float>  chroma_slice_;
    std::vector<double> chroma_continuity_;
    SignalBaselines     baselines_;
    PairScorerTrack     track_ {};
};

} // namespace reamix::remix
