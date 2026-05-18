// Copy of references/reabeat-template/src/TimeSigDetector.h with class renamed
// to TimeSigDetectorReabeat so it can be compiled alongside reamix::TimeSigDetector
// in the same parity-test binary. Do not edit — if REABeat's reference changes,
// re-copy the file and re-apply the rename.
#pragma once
#include <vector>

// Detect time signature numerator from beat/downbeat positions.
// Port of detector.py _time_sig_from_downbeats()

class TimeSigDetectorReabeat
{
public:
    // Count beats between downbeats, return most common count.
    // Falls back to 4 if insufficient data.
    static int detect(const std::vector<float>& beats,
                      const std::vector<float>& downbeats);

private:
    static constexpr float kTolerance = 0.03f;  // 30ms tolerance for beat counting
};
