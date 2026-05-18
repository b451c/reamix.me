#include "analysis/Mfcc.h"

namespace reamix::analysis {

Mfcc::Mfcc(int nMfcc)
    : nMfcc_(nMfcc)
    , mel_()
    , dct_(dsp::MelSpectrogramLibrosa::kNMels, nMfcc)
{
}

std::vector<std::vector<float>>
Mfcc::compute(const std::vector<float>& y) const
{
    // librosa.feature.mfcc default path uses power_to_db(ref=1.0), not
    // ref=np.max. See dsp::MelSpectrogramLibrosa::powerToDb docs.
    auto melPower = mel_.power(y);                   // [frame][mel]
    auto logMel   = mel_.powerToDb(melPower, 1.0f);  // [frame][mel]
    return dct_.applyFrames(logMel);                 // [frame][mfcc]
}

} // namespace reamix::analysis
