// Copy of references/reabeat-template/src/MelSpectrogram.h with class renamed
// to MelSpectrogramReabeat so it can be compiled alongside reamix::MelSpectrogram
// in the same parity-test binary. Do not edit — if REABeat's reference changes,
// re-copy the file and re-apply the rename.
#pragma once
#include <vector>
#include <complex>
#include <cmath>

class MelSpectrogramReabeat
{
public:
    MelSpectrogramReabeat();

    std::vector<std::vector<float>> compute(const std::vector<float>& audio);
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

    std::vector<std::vector<double>> filterbank_;
    std::vector<float> window_;

    void createFilterbank();
    void createWindow();
    static float hzToMel(float hz);
    static float melToHz(float mel);
};
