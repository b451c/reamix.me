#include "SegmentConsolidation.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace reamix::analysis {

namespace {

using Segment = SegmentConsolidation::Segment;

// PARITY: segment_consolidation.py:30-50 _merge_same_label_segments.
// Sequential: walk left-to-right; when consecutive segments share label,
// replace the accumulator's tail with a fused record whose end/confidence
// come from the newer segment and start/label/cluster_id are inherited
// from the accumulator. Confidence is f64 pair-mean via `(a + b) * 0.5`
// (Python L45). cluster_id inherits from `prev` (first-wins).
std::vector<Segment> mergeSameLabel(const std::vector<Segment>& segments)
{
    if (segments.empty())
        return {};

    std::vector<Segment> merged;
    merged.reserve(segments.size());
    merged.push_back(segments.front());

    for (std::size_t i = 1; i < segments.size(); ++i)
    {
        const Segment& seg  = segments[i];
        Segment&       prev = merged.back();
        if (prev.label == seg.label)
        {
            prev = Segment{
                prev.start,
                seg.end,
                (prev.confidence + seg.confidence) * 0.5,
                prev.cluster_id,
                prev.label,
            };
        }
        else
        {
            merged.push_back(seg);
        }
    }
    return merged;
}

// PARITY: segment_consolidation.py:52-67 _merge_triplet.
// Confidence: `np.mean([a, b, c])`. numpy reduces length-3 with pairwise
// sum < 8 naive branch = `((a + b) + c)`, then divides by 3.0. C++ mirrors
// `((a + b) + c) / 3.0`.
// cluster_id/label: if left and right share a label the fused segment
// inherits from `left`; otherwise the middle dominates.
Segment mergeTriplet(const Segment& left,
                     const Segment& middle,
                     const Segment& right)
{
    const double confidence =
        ((left.confidence + middle.confidence) + right.confidence) / 3.0;
    const bool   sideMatch = (left.label == right.label);
    const int    clusterId = sideMatch ? left.cluster_id : middle.cluster_id;
    std::string  label     = sideMatch ? left.label       : middle.label;
    return Segment{
        left.start,
        right.end,
        confidence,
        clusterId,
        std::move(label),
    };
}

// PARITY: segment_consolidation.py:69-81 _merge_pair.
// Label override: nullopt → second.label (Python `label or second.label`).
// cluster_id: `second.cluster_id` if the resulting label matches
// `second.label`, else `first.cluster_id` — so an override that matches
// `first`'s label propagates `first.cluster_id`.
Segment mergePair(const Segment&                   first,
                  const Segment&                   second,
                  const std::optional<std::string>& labelOverride)
{
    std::string mergedLabel = labelOverride.has_value()
                                  ? *labelOverride
                                  : second.label;
    const int   clusterId   = (mergedLabel == second.label)
                                  ? second.cluster_id
                                  : first.cluster_id;
    return Segment{
        first.start,
        second.end,
        (first.confidence + second.confidence) * 0.5,
        clusterId,
        std::move(mergedLabel),
    };
}

bool isIntroOutro(const std::string& label) noexcept
{
    return label == "intro" || label == "outro";
}

bool isChoreVerseBridge(const std::string& label) noexcept
{
    return label == "bridge" || label == "verse" || label == "chorus";
}

} // namespace


std::vector<Segment>
SegmentConsolidation::consolidate(const std::vector<Segment>& segments,
                                  double                      duration)
{
    if (segments.empty())
        return {};

    // Step 1 + outer-loop entry: initial same-label merge (L90).
    std::vector<Segment> macro = mergeSameLabel(segments);

    constexpr double minDuration     = kMacroMinDurationSec;   // 10.0
    constexpr double bridgeMergeMax  = kBridgeMergeMaxSec;     //  8.0
    constexpr int    maxSegments     = kMaxSegments;           //   12

    // Step 2: `while changed` predicate loop (L95-144). Re-runs until a full
    // sweep fires no merge.
    bool changed = true;
    while (changed)
    {
        changed = false;
        std::vector<Segment> next;
        next.reserve(macro.size());

        std::size_t i = 0;
        while (i < macro.size())
        {
            const Segment& current         = macro[i];
            const double   currentDuration = current.end - current.start;
            const Segment* prev = next.empty() ? nullptr : &next.back();
            const Segment* nxt  = (i + 1 < macro.size()) ? &macro[i + 1]
                                                         : nullptr;

            // Predicate 1 (L106-117): short "bridge" between two same-label
            // non-intro/outro neighbours.
            if (prev != nullptr
                && nxt != nullptr
                && currentDuration <= bridgeMergeMax
                && current.label == "bridge"
                && prev->label == nxt->label
                && !isIntroOutro(prev->label))
            {
                next.back() = mergeTriplet(*prev, current, *nxt);
                i += 2;
                changed = true;
                continue;
            }

            // Predicate 2 (L119-128): short middle (< 0.75 × min_duration)
            // between same-label neighbours.
            if (prev != nullptr
                && nxt != nullptr
                && currentDuration < minDuration * 0.75
                && prev->label == nxt->label)
            {
                next.back() = mergeTriplet(*prev, current, *nxt);
                i += 2;
                changed = true;
                continue;
            }

            // Predicate 3 (L130-139): very-short chorus/verse/bridge absorbed
            // into same-label predecessor.
            if (currentDuration < minDuration * 0.6
                && isChoreVerseBridge(current.label)
                && prev != nullptr
                && prev->label == current.label)
            {
                next.back() = mergePair(*prev, current, prev->label);
                i += 1;
                changed = true;
                continue;
            }

            next.push_back(current);
            i += 1;
        }

        macro = mergeSameLabel(next);
    }

    // Step 3: cap total count at max_segments (L146-174). Greedy: repeatedly
    // fuse the shortest interior (non-first, non-last) segment with its
    // context. Priority order:
    //   (a) triplet-merge when flanking segs share label;
    //   (b) pair-merge with the longer neighbour otherwise (ties → right
    //       via `>=` at Python L162).
    while (static_cast<int>(macro.size()) > maxSegments)
    {
        // interior = [(idx, duration)] over macro[1:-1]. Python L147-150.
        int    idx          = -1;
        double shortestDur  = 0.0;
        for (std::size_t j = 1; j + 1 < macro.size(); ++j)
        {
            const double d = macro[j].end - macro[j].start;
            if (idx == -1 || d < shortestDur)
            {
                idx         = static_cast<int>(j);
                shortestDur = d;
            }
        }
        if (idx == -1)  // interior empty (≤ 2 segments total) — Python L151-152.
            break;

        const Segment  left    = macro[idx - 1];
        const Segment  current = macro[idx];
        const Segment* right   = (static_cast<std::size_t>(idx) + 1 < macro.size())
                                      ? &macro[idx + 1]
                                      : nullptr;

        std::vector<Segment> rebuilt;
        rebuilt.reserve(macro.size());

        if (right != nullptr && left.label == right->label)
        {
            // (a) triplet-merge (L159-161): macro[:idx-1] + [triplet] + macro[idx+2:].
            for (int k = 0; k < idx - 1; ++k)
                rebuilt.push_back(macro[k]);
            rebuilt.push_back(mergeTriplet(left, current, *right));
            for (std::size_t k = static_cast<std::size_t>(idx) + 2; k < macro.size(); ++k)
                rebuilt.push_back(macro[k]);
        }
        else if (right != nullptr
                 && (right->end - right->start) >= (left.end - left.start))
        {
            // (b1) pair-merge current + right, label = right.label
            // (L162-167): macro[:idx] + [pair] + macro[idx+2:].
            for (int k = 0; k < idx; ++k)
                rebuilt.push_back(macro[k]);
            rebuilt.push_back(mergePair(current, *right, right->label));
            for (std::size_t k = static_cast<std::size_t>(idx) + 2; k < macro.size(); ++k)
                rebuilt.push_back(macro[k]);
        }
        else
        {
            // (b2) pair-merge left + current, label = left.label
            // (L168-172): macro[:idx-1] + [pair] + macro[idx+1:].
            for (int k = 0; k < idx - 1; ++k)
                rebuilt.push_back(macro[k]);
            rebuilt.push_back(mergePair(left, current, left.label));
            for (std::size_t k = static_cast<std::size_t>(idx) + 1; k < macro.size(); ++k)
                rebuilt.push_back(macro[k]);
        }

        macro = mergeSameLabel(rebuilt);
    }

    // Step 4: position-based intro/outro overrides + boundary clamp to
    // [0, duration] (L176-191). Python's `macro[0] = Segment(start=0.0,
    // end=macro[0].end, label="intro" if macro[0].start < ..., ...)` reads
    // every `macro[0].*` as the OLD value in RHS, then replaces. We capture
    // an explicit `const` snapshot to keep that contract visible.
    if (!macro.empty())
    {
        const Segment oldFirst = macro.front();
        macro.front() = Segment{
            0.0,
            oldFirst.end,
            oldFirst.confidence,
            oldFirst.cluster_id,
            (oldFirst.start < duration * 0.15) ? std::string("intro")
                                                : oldFirst.label,
        };

        const Segment oldLast = macro.back();
        macro.back() = Segment{
            oldLast.start,
            duration,
            oldLast.confidence,
            oldLast.cluster_id,
            (oldLast.end > duration * 0.85) ? std::string("outro")
                                            : oldLast.label,
        };
    }

    return macro;
}

} // namespace reamix::analysis
