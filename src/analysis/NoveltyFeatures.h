#pragma once

#include <cstddef>
#include <vector>

namespace reamix::analysis {

// 25-dim per-frame feature matrix for novelty-path SSM. Port of
// `StructureAnalyzer._extract_features`
// (python-source/analysis/structure_analyzer.py L193-210) on librosa 0.11.0:
//
//   mfcc   = librosa.feature.mfcc(y, sr=22050, n_mfcc=13, hop_length=512)
//   chroma = librosa.feature.chroma_stft(y, sr=22050, hop_length=512, n_fft=2048)
//   mfcc   = (mfcc   - mfcc.mean(axis=1, keepdims=True))   / (mfcc.std(axis=1,   keepdims=True) + 1e-8)
//   chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (chroma.std(axis=1, keepdims=True) + 1e-8)
//   features = np.vstack([mfcc * 1.5, chroma])     # shape (25, T), float32
//
// ADR-019: pure composition of existing phase-2 primitives (Mfcc,
// PitchTrack::estimateTuning, ChromaSTFT). Zero new DSP modules. Does NOT
// touch the phase-2 `FeatureExtractor::Result` struct — novelty features
// are phase-3-internal.
//
// Output layout: [frame][kNFeat] row-major float32, T frames where
// T = STFT::numFrames(nSamples). Python dumps have shape (kNFeat, T); the
// parity test transposes at read time (matches phase-2 precedent).
class NoveltyFeatures
{
public:
    // PARITY: structure_analyzer.py:195, :198, :202-207, :209.
    static constexpr int   kNMfcc      = 13;
    static constexpr int   kNChroma    = 12;
    static constexpr int   kNFeat      = kNMfcc + kNChroma;   // 25
    static constexpr float kStdEps     = 1e-8f;               // Python `+ 1e-8` guard
    static constexpr float kMfccWeight = 1.5f;                // `mfcc * 1.5` in vstack

    // Compute 25-dim per-frame novelty features from mono float32 audio.
    // `sr` defaults to 22050 (hardcoded in structure_analyzer.py L71);
    // passing sr explicitly lets tests exercise alternate rates if ever
    // needed. Returns an empty matrix for empty input.
    static std::vector<std::vector<float>>
    extract(const float* y, std::size_t nSamples, int sr = 22050);
};

} // namespace reamix::analysis
