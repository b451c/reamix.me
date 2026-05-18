// test_crossfade — phase-5 parity:
//   adaptive: reamix::render::Crossfade::adaptiveCrossfade vs Python
//             adaptive_crossfade_fixed (ADR-032 Option B.myfix — shift-full-in_band
//             instead of slice+pad-back at crossfade.py:305-308). Monkey-patched in
//             tools/dump_phase5_crossfade.py; the `references/` symlink is untouched.
//   simple:   reamix::render::Crossfade::simpleCrossfade vs unmodified Python
//             simple_crossfade (crossfade.py:335-366). No monkey-patch — simple path
//             has no pad/envelope bug.
//
// Gate: per-sample max_abs ≤ 1e-9 across all cases. Accumulation class:
// adaptive:
//   2× Butterworth::sosFiltFilt (per band, out + in)  ULP ~5e-14
// + PhaseAlign NCC / shift argmax (int-exact, no f64 contribution)
// + shift-apply on full in_band (pure memcpy + memset, int-indexed)
// + equal-power fade (cos/sin on linspace, f64 accuracy ~1e-16)
// + energy-compensate clamp (scalar multiply, ULP ~2e-16)
// + per-band sum into overlap_result (accumulator, n_bands=3 terms)
// → worst-case end-to-end ~5e-13 per sample (Butterworth dominates;
//   other components are additive at ULP scale). 1e-9 is comfortable.
// simple:
//   RMS (sum-of-squares, mean, sqrt) — ULP ~4e-16 per RMS
// + linear ramp linspace (step precomputed, endpoint exact) — ULP ~2e-16 per sample
// + equal-power fade (cos/sin) — ULP ~1e-16 per sample
// + ramp * incoming multiply + fade-zone sum — ULP ~2e-16
// → worst-case ~5e-15 per sample. 1e-9 is comfortable (6 orders over).
//
// Cases (tools/dump_phase5_crossfade.py):
//   adaptive path:
//   case_01 — billie_jean boundary stereo, PA=True  EC=True  (prod default path)
//   case_02 — billie_jean boundary stereo, PA=False EC=True  (no-align control)
//   case_03 — synthetic bass+treble stereo, PA=True  EC=False (ADR-032 fix visible)
//   case_04 — synthetic stereo + 20 dB RMS mismatch, PA=True EC=True (clamp path)
//   simple path:
//   case_05 — billie_jean boundary stereo, cf=75.0 ms, EC=True  (prod real music)
//   case_06 — synthetic stereo -20 dB mismatch, cf=75.0 ms, EC=True (clamp + full ramp)
//   case_07 — synthetic stereo, cf=75.0 ms, EC=False  (no-ramp control)

#include "render/Crossfade.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "NpyIO.h"

namespace {

constexpr double TOL_ABS = 1e-9;

enum class Kind { Adaptive, Simple };

struct Case {
    const char* dir;
    const char* label;
    Kind        kind;
};

constexpr Case kCases[] = {
    {"case_01_billie_jean_prod_default",     "BJ_prod    (adaptive PA=T EC=T)", Kind::Adaptive},
    {"case_02_billie_jean_no_align",         "BJ_no_pa   (adaptive PA=F EC=T)", Kind::Adaptive},
    {"case_03_synthetic_bass_treble_pa_on",  "synth_ba_tr (adaptive PA=T EC=F)", Kind::Adaptive},
    {"case_04_synthetic_ec_clamp",           "synth_ec   (adaptive PA=T EC=T)", Kind::Adaptive},
    {"case_05_simple_billie_jean_prod",      "BJ_simple  (simple cf=75 EC=T)",  Kind::Simple},
    {"case_06_simple_ec_clamp",              "synth_simple_ec (simple cf=75 EC=T)", Kind::Simple},
    {"case_07_simple_no_ec",                 "synth_simple_no_ec (simple cf=75 EC=F)", Kind::Simple},
};

std::string dataRoot(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') return std::string(argv[i]);
    }
    return std::string("references/golden/phase-5/crossfade");
}

// Minimal flags.json reader — handles boolean + numeric keys we care about.
// Matches the tight schema written by tools/dump_phase5_crossfade.py (no nested
// objects, no arrays except out_shape/in_shape/result_shape which we ignore).
struct CaseFlags {
    bool   phaseAlign       = true;
    bool   energyCompensate = true;
    double maxPhaseShiftMs  = 3.0;
    double crossfadeMs      = 75.0;
};

CaseFlags loadFlags(const std::string& path)
{
    CaseFlags f;
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string txt = ss.str();

    auto find_bool = [&](const char* key, bool def) {
        const std::string needle = std::string("\"") + key + "\"";
        auto p = txt.find(needle);
        if (p == std::string::npos) return def;
        auto q = txt.find(':', p);
        if (q == std::string::npos) return def;
        // Skip whitespace
        ++q;
        while (q < txt.size() && (txt[q] == ' ' || txt[q] == '\t')) ++q;
        if (txt.compare(q, 4, "true") == 0) return true;
        if (txt.compare(q, 5, "false") == 0) return false;
        return def;
    };
    auto find_number = [&](const char* key, double def) {
        const std::string needle = std::string("\"") + key + "\"";
        auto p = txt.find(needle);
        if (p == std::string::npos) return def;
        auto q = txt.find(':', p);
        if (q == std::string::npos) return def;
        ++q;
        while (q < txt.size() && (txt[q] == ' ' || txt[q] == '\t')) ++q;
        return std::atof(txt.c_str() + q);
    };

    f.phaseAlign       = find_bool("phase_align",       true);
    f.energyCompensate = find_bool("energy_compensate", true);
    f.maxPhaseShiftMs  = find_number("max_phase_shift_ms", 3.0);
    f.crossfadeMs      = find_number("crossfade_ms", 75.0);
    return f;
}

bool runCase(const std::string& root, const Case& c, bool verbose,
             double& gmax, std::size_t& gover, std::size_t& gtotal,
             int& gfailCases)
{
    const std::string caseRoot = root + "/" + c.dir;
    const auto srVec = reamix::test::loadNpy1DInt64(caseRoot + "/sr.npy");
    const int  sr    = static_cast<int>(srVec.at(0));
    const auto out_m = reamix::test::loadNpy2DFloat64(caseRoot + "/out.npy");
    const auto in_m  = reamix::test::loadNpy2DFloat64(caseRoot + "/in.npy");
    const auto res_m = reamix::test::loadNpy2DFloat64(caseRoot + "/result.npy");
    const CaseFlags flags = loadFlags(caseRoot + "/flags.json");

    const std::size_t nCh    = out_m.rows;
    const std::size_t outLen = out_m.cols;
    const std::size_t inLen  = in_m.cols;
    const std::size_t resLen = res_m.cols;

    if (res_m.rows != nCh || in_m.rows != nCh) {
        std::fprintf(stderr, "  %s: channel count mismatch out=%zu in=%zu res=%zu\n",
                     c.dir, out_m.rows, in_m.rows, res_m.rows);
        ++gfailCases;
        return false;
    }

    const std::size_t expectedResLen =
        (c.kind == Kind::Adaptive)
            ? reamix::render::Crossfade::computeResultLen(outLen, inLen, sr, nullptr, 0)
            : reamix::render::Crossfade::computeSimpleResultLen(outLen, inLen, sr,
                                                                flags.crossfadeMs);
    if (expectedResLen != resLen) {
        std::fprintf(stderr,
            "  %s: computeResultLen=%zu != golden resLen=%zu\n",
            c.dir, expectedResLen, resLen);
        ++gfailCases;
        return false;
    }

    std::vector<double> cppRes(nCh * resLen, 0.0);
    std::size_t written = 0;
    if (c.kind == Kind::Adaptive) {
        written = reamix::render::Crossfade::adaptiveCrossfade(
            out_m.data.data(), nCh, outLen,
            in_m.data.data(),  inLen,
            sr,
            nullptr, 0,                      // default BANDS
            flags.energyCompensate,
            flags.phaseAlign,
            flags.maxPhaseShiftMs,
            cppRes.data(),
            resLen);
    } else {
        written = reamix::render::Crossfade::simpleCrossfade(
            out_m.data.data(), nCh, outLen,
            in_m.data.data(),  inLen,
            sr,
            flags.crossfadeMs,
            flags.energyCompensate,
            cppRes.data(),
            resLen);
    }

    if (written != resLen) {
        std::fprintf(stderr,
            "  %s: %s returned %zu, expected %zu\n",
            c.dir,
            (c.kind == Kind::Adaptive) ? "adaptiveCrossfade" : "simpleCrossfade",
            written, resLen);
        ++gfailCases;
        return false;
    }

    double case_max = 0.0;
    std::size_t case_over = 0;
    for (std::size_t i = 0; i < nCh * resLen; ++i) {
        const double d = std::abs(cppRes[i] - res_m.data[i]);
        if (d > case_max) case_max = d;
        if (d > TOL_ABS) ++case_over;
        ++gtotal;
    }
    if (case_max > gmax) gmax = case_max;
    gover += case_over;

    const bool ok = (case_max <= TOL_ABS);
    if (!ok) ++gfailCases;
    if (verbose || !ok) {
        std::printf(
            "  %-40s sr=%-5d nCh=%zu res=(%zu,%zu) "
            "case_max=%.3e over=%zu/%zu  %s\n",
            c.label, sr, nCh, nCh, resLen,
            case_max, case_over, nCh * resLen,
            ok ? "PASS" : "FAIL");
    }
    return ok;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    const std::string root = dataRoot(argc, argv);
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--verbose" || a == "-v") verbose = true;
    }

    std::printf("=== test_crossfade: adaptive + simple parity (tol=%.1e) ===\n",
                TOL_ABS);

    double gmax = 0.0;
    std::size_t gover = 0, gtotal = 0;
    int gfailCases = 0;
    int casesOk = 0;
    for (const auto& c : kCases) {
        if (runCase(root, c, verbose, gmax, gover, gtotal, gfailCases))
            ++casesOk;
    }
    std::printf(
        "  cases=%d/%zu PASS  total samples=%zu  over_tol=%zu  max_abs=%.3e  %s\n",
        casesOk, sizeof(kCases) / sizeof(kCases[0]),
        gtotal, gover, gmax,
        (gfailCases == 0) ? "PASS" : "FAIL");

    return (gfailCases == 0) ? 0 : 1;
}
