// Parity: reamix::util::medianFilter2DReflect vs
// `scipy.ndimage.median_filter(a, size=(kR,kC), mode='reflect')` (scipy 1.15.3).
//
// Bitwise target — median of a finite set is unique, regardless of which
// selection algorithm each side uses internally.
//
// Fixtures under tests/parity/reference/data/median_filter/:
//   small_input.npy  (6, 10) f32   + small_harm3.npy  / small_perc3.npy
//   stft_input.npy   (1025, 200) f32 + stft_harm31.npy / stft_perc31.npy
//
// The small case exercises reflect-boundary every row/col (kernel reaches
// beyond the edge everywhere for (1,3) and (3,1) on 6×10). The large case
// is STFT-shaped with the actual HPSS kernel sizes (1,31)/(31,1).

#include "util/MedianFilter.h"
#include "NpyIO.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct FailCount { int n = 0; };

bool bitwiseEqualF32(float a, float b)
{
    std::uint32_t ai, bi;
    std::memcpy(&ai, &a, sizeof(ai));
    std::memcpy(&bi, &b, sizeof(bi));
    return ai == bi;
}

void runCase(const char* name,
             const fs::path& dir,
             const std::string& inName,
             const std::string& refName,
             std::size_t kR, std::size_t kC,
             FailCount& fc)
{
    const fs::path inPath  = dir / (inName + ".npy");
    const fs::path refPath = dir / (refName + ".npy");
    auto in  = reamix::test::loadNpy2DFloat32(inPath.string());
    auto ref = reamix::test::loadNpy2DFloat32(refPath.string());

    if (in.rows != ref.rows || in.cols != ref.cols) {
        std::printf("[FAIL] %s: shape mismatch in (%zu,%zu) vs ref (%zu,%zu)\n",
                    name, in.rows, in.cols, ref.rows, ref.cols);
        ++fc.n;
        return;
    }

    std::vector<float> out(in.rows * in.cols);
    reamix::util::medianFilter2DReflect<float>(
        in.data.data(), in.rows, in.cols, kR, kC, out.data()
    );

    std::size_t bitwiseMismatches = 0;
    double maxAbs = 0.0;
    double sumAbs = 0.0;
    const std::size_t nTotal = in.rows * in.cols;
    for (std::size_t k = 0; k < nTotal; ++k) {
        const float got = out[k];
        const float want = ref.data[k];
        if (!bitwiseEqualF32(got, want)) ++bitwiseMismatches;
        const double d = std::abs(static_cast<double>(got) - static_cast<double>(want));
        if (d > maxAbs) maxAbs = d;
        sumAbs += d;
    }
    const double meanAbs = nTotal ? sumAbs / static_cast<double>(nTotal) : 0.0;
    const bool pass = (bitwiseMismatches == 0);
    std::printf("%-22s  shape=(%4zu,%4zu)  kernel=(%2zu,%2zu)  "
                "bitwise_mismatches=%zu/%zu  max_abs=%.2e  mean_abs=%.2e  %s\n",
                name, in.rows, in.cols, kR, kC,
                bitwiseMismatches, nTotal, maxAbs, meanAbs,
                pass ? "PASS" : "FAIL");
    if (!pass) ++fc.n;
}

} // namespace

int main(int argc, char** argv)
{
    const fs::path root = (argc > 1)
        ? fs::path(argv[1])
        : fs::path("tests/parity/reference/data/median_filter");

    if (!fs::exists(root) || !fs::is_directory(root)) {
        std::fprintf(stderr, "data directory not found: %s\n"
                             "run `python tools/dump_median_filter_tests.py` first\n",
                     root.string().c_str());
        return 2;
    }

    FailCount fc;

    runCase("small_harm3 (1,3)",  root, "small_input", "small_harm3",  1, 3,  fc);
    runCase("small_perc3 (3,1)",  root, "small_input", "small_perc3",  3, 1,  fc);
    runCase("stft_harm31 (1,31)", root, "stft_input",  "stft_harm31",  1, 31, fc);
    runCase("stft_perc31 (31,1)", root, "stft_input",  "stft_perc31",  31, 1, fc);

    std::printf("\nfailures: %d\n", fc.n);
    return fc.n == 0 ? 0 : 1;
}
