#pragma once

#include <cmath>
#include <complex>
#include <vector>

namespace reamix {

// Mel-scale spectrogram matching beat-this / torchaudio:
// SR=22050, n_fft=1024, hop=441, n_mels=128, f_min=30, f_max=11000,
// Hann window, reflect padding, Slaney mel scale, amplitude spectrum
// normalized by 1/sqrt(n_fft)=1/32 (matches torchaudio normalized="frame_length").
// Output is log1p(1000 * melEnergy). Direct port of REABeat reference.
class MelSpectrogram
{
public:
    MelSpectrogram();

    // Compute mel spectrogram from mono audio at 22050 Hz.
    // Returns [frames][128] log-mel values.
    std::vector<std::vector<float>> compute(const std::vector<float>& audio);

    // Frame count for a given audio length (center padding assumed).
    int getFrameCount(int audioLength) const;

private:
    static constexpr int kSampleRate = 22050;
    static constexpr int kNfft = 1024;
    static constexpr int kWinLength = 1024;
    static constexpr int kHopLength = 441;
    static constexpr int kNmels = 128;
    static constexpr int kFmin = 30;
    static constexpr int kFmax = 11000;
    static constexpr float kLogMultiplier = 1000.0f;
    static constexpr double kAmin = 1e-10;

    std::vector<std::vector<double>> filterbank_;  // [n_fft/2+1][n_mels]
    std::vector<float> window_;

    void createFilterbank();
    void createWindow();
    static float hzToMel(float hz);
    static float melToHz(float mel);
};

} // namespace reamix
