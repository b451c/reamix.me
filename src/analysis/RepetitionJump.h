#pragma once

namespace reamix::analysis {

// One verified bar-boundary repetition: "the bar ending at `fromBeat` may be
// followed by the bar starting at `toBeat`". Optional candidate input of the
// legacy optimizer path (`OptimizerInputs::jumps`, `ViterbiDP::run`), where
// each jump becomes a transition carrying the `is_repetition_jump = 1.0`
// metadata marker (optimizer.py F2). Production passes no jumps since
// ADR-044; the phase-4 parity tests feed the Python goldens through this
// type. The producer (RepetitionMap, phase-3 step 8) was deleted in sesja
// 122 (ADR-115 E6); the struct stays as the data contract.
struct RepetitionJump
{
    int     fromBeat;                // Last beat of outgoing bar (pre-downbeat)
    int     toBeat;                  // First beat of incoming bar (downbeat)
    double  waveformSimilarity;      // 0-1, from waveform xcorr at bar boundary
    double  chromaCorrelation;       // 0-1, bar-level chroma correlation
    int     alignmentLagSamples;     // Micro-alignment offset in samples
    int     fromSectionIdx;
    int     toSectionIdx;
    int     fromBar;                 // Bar index within section
    int     toBar;
};

} // namespace reamix::analysis
