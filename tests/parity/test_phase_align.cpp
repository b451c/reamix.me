// test_phase_align — phase-5 session 31 parity:
//   reamix::render::PhaseAlign vs references/python-source/remix/crossfade.py::_phase_align
//
// Two gates per ADR-031 SB-4:
//
//   Gate 1 — best_shift index bit-exact:
//     integer equality between C++ `shiftOut` and Python-dumped `shift_out`.
//     Any mismatch is an algorithmic bug (argmax tie-break or loop bounds),
//     NOT a numerical gate to widen.
//
//   Gate 2 — aligned-output max_abs ≤ 1e-12:
//     f64 ULP class. No sosfiltfilt-style reverse-and-re-apply accumulation,
//     so gate is tighter than Butterworth Fixture B (1e-9). Shift-apply is
//     pure memcpy of zero-padded slice, so Gate 2 is usually bit-exact (0.0)
//     on non-aliased output — the 1e-12 budget covers potential f32 round-trip
//     wrapping if a future caller passes f32 goldens.
//
// 7 cases dumped by tools/dump_phase5_phase_align.py:
//   case_01 — mono, shift -17 @ sr=44100, max_shift=132 (positive-shift apply)
//   case_02 — mono, shift +23 @ sr=44100, max_shift=132 (negative-shift apply)
//   case_03 — mono, zero-shift passthrough
//   case_04 — stereo 2D, shift -17; test drives each channel independently
//             (first-channel NCC gives the shift; all channels use that shift)
//   case_05 — norm-guard trigger (outgoing all zero)
//   case_06 — short buffer n<32 (min-length gate early return)
//   case_07 — noise, API-default max_shift=64, shift +5

#include "render/PhaseAlign.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "NpyIO.h"

namespace {

constexpr double TOL_ALIGNED_ABS = 1e-12;

struct Case {
    const char* dir;
    // `isStereo`: true → 2-D (2, N) f64 goldens; run alignMono on each
    // channel independently, verify shift matches on channel 0 and apply to
    // channel 1 broadcasts consistently. Python's _phase_align picks the
    // first channel only for NCC; this test enforces that discipline.
    bool        isStereo;
};

constexpr Case kCases[] = {
    {"case_01_mono_shift_plus17_sr44100",  false},
    {"case_02_mono_shift_minus23_sr44100", false},
    {"case_03_mono_zero_shift_sr44100",    false},
    {"case_04_stereo_shift_plus17_sr44100", true},
    {"case_05_norm_guard_zero_out",        false},
    {"case_06_short_buffer_n31",           false},
    {"case_07_mono_noise_max_shift_64",    false},
};

std::string dataRoot(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') return std::string(argv[i]);
    }
    return std::string("references/golden/phase-5/phase_align");
}

// Run Gate 1 (shift-index bit-exact) and Gate 2 (aligned-output tolerance)
// for a single mono case. Returns true iff both gates pass.
bool runMono(const std::string& caseRoot, const char* label,
             int refShift, int refMaxShift,
             const std::vector<double>& outVec,
             const std::vector<double>& inVec,
             const std::vector<double>& alignedRef,
             bool verbose)
{
    int shift = -12345;
    std::vector<double> alignedCpp(inVec.size(), 0.0);

    reamix::render::PhaseAlign::alignMono(
        outVec.data(), outVec.size(),
        inVec.data(),  inVec.size(),
        refMaxShift,
        &shift,
        alignedCpp.data());

    // Gate 1: shift-index bit-exact
    const bool shift_ok = (shift == refShift);

    // Gate 2: aligned-output max_abs tolerance
    double case_max = 0.0;
    std::size_t case_over = 0;
    for (std::size_t i = 0; i < alignedCpp.size(); ++i) {
        const double d = std::abs(alignedCpp[i] - alignedRef[i]);
        if (d > case_max) case_max = d;
        if (d > TOL_ALIGNED_ABS) ++case_over;
    }
    const bool align_ok = (case_max <= TOL_ALIGNED_ABS);

    const bool ok = shift_ok && align_ok;
    if (verbose || !ok) {
        std::printf(
            "  %-48s %-6s max_shift=%3d  shift_cpp=%+4d ref=%+4d %s  "
            "aligned_max_abs=%.3e over=%zu  %s\n",
            caseRoot.c_str(), label, refMaxShift, shift, refShift,
            shift_ok ? "OK" : "MISMATCH",
            case_max, case_over,
            ok ? "PASS" : "FAIL");
    }
    return ok;
}

bool runCase(const std::string& root, const Case& c, bool verbose,
             double& gmax, std::size_t& gover, int& shiftMismatches)
{
    const std::string caseRoot = root + "/" + c.dir;
    const auto max_shift_vec = reamix::test::loadNpy1DInt64(caseRoot + "/max_shift.npy");
    const auto shift_out_vec = reamix::test::loadNpy1DInt64(caseRoot + "/shift_out.npy");
    const int refMaxShift = static_cast<int>(max_shift_vec.at(0));
    const int refShift    = static_cast<int>(shift_out_vec.at(0));

    bool ok = true;

    if (!c.isStereo) {
        const auto out_v     = reamix::test::loadNpy1DFloat64(caseRoot + "/out.npy");
        const auto in_v      = reamix::test::loadNpy1DFloat64(caseRoot + "/in.npy");
        const auto aligned_v = reamix::test::loadNpy1DFloat64(caseRoot + "/aligned_out.npy");
        ok = runMono(c.dir, "mono", refShift, refMaxShift,
                     out_v, in_v, aligned_v, verbose);
    } else {
        // 2-D (2, N) goldens. Python picks channel 0 for NCC; our mono-only
        // API mirrors that. The same shift applies to channel 1 (broadcast
        // in Python L145/L151). We drive alignMono per-channel and verify
        // both channels parity-match their row of aligned_out.
        const auto out_m     = reamix::test::loadNpy2DFloat64(caseRoot + "/out.npy");
        const auto in_m      = reamix::test::loadNpy2DFloat64(caseRoot + "/in.npy");
        const auto aligned_m = reamix::test::loadNpy2DFloat64(caseRoot + "/aligned_out.npy");
        const std::size_t n = out_m.cols;

        // Channel 0 — drives the shift decision (matches Python's choice).
        std::vector<double> out0(out_m.data.begin(), out_m.data.begin() + n);
        std::vector<double> in0(in_m.data.begin(),  in_m.data.begin()  + n);
        std::vector<double> aligned0(aligned_m.data.begin(),
                                     aligned_m.data.begin() + n);
        // Channel 1 — same shift, independent corr math but the shift-apply
        // step is deterministic so alignedOut matches the golden row.
        std::vector<double> out1(out_m.data.begin() + n,
                                 out_m.data.begin() + 2 * n);
        std::vector<double> in1(in_m.data.begin() + n,
                                in_m.data.begin() + 2 * n);
        std::vector<double> aligned1(aligned_m.data.begin() + n,
                                     aligned_m.data.begin() + 2 * n);

        const bool ch0_ok = runMono(c.dir, "ch0", refShift, refMaxShift,
                                    out0, in0, aligned0, verbose);
        // Channel 1: NCC on channel-1 content may disagree with channel-0 NCC
        // if signals differ, but for our golden (L / 0.9×L) they're
        // perfectly correlated and produce the same argmax. The gate is:
        // ch1 shift-apply must match the golden row (which used ch0's shift).
        const bool ch1_ok = runMono(c.dir, "ch1", refShift, refMaxShift,
                                    out1, in1, aligned1, verbose);
        ok = ch0_ok && ch1_ok;
    }

    // Recompute aggregate metrics (cheap; case counts are small).
    // We already printed per-case metrics inside runMono; here we only
    // accumulate the global counters. Implementation: rerun mono gate 2
    // quickly, accumulating. For stereo we accumulate per-channel.
    auto accumulate = [&](const std::vector<double>& ref,
                          std::size_t nShiftArg, const std::vector<double>& outV,
                          const std::vector<double>& inV) {
        int shift = 0;
        std::vector<double> alignedCpp(inV.size(), 0.0);
        reamix::render::PhaseAlign::alignMono(
            outV.data(), outV.size(),
            inV.data(), inV.size(),
            static_cast<int>(nShiftArg),
            &shift,
            alignedCpp.data());
        if (shift != refShift) ++shiftMismatches;
        for (std::size_t i = 0; i < alignedCpp.size(); ++i) {
            const double d = std::abs(alignedCpp[i] - ref[i]);
            if (d > gmax) gmax = d;
            if (d > TOL_ALIGNED_ABS) ++gover;
        }
    };

    if (!c.isStereo) {
        const auto out_v     = reamix::test::loadNpy1DFloat64(caseRoot + "/out.npy");
        const auto in_v      = reamix::test::loadNpy1DFloat64(caseRoot + "/in.npy");
        const auto aligned_v = reamix::test::loadNpy1DFloat64(caseRoot + "/aligned_out.npy");
        accumulate(aligned_v, static_cast<std::size_t>(refMaxShift), out_v, in_v);
    } else {
        const auto out_m     = reamix::test::loadNpy2DFloat64(caseRoot + "/out.npy");
        const auto in_m      = reamix::test::loadNpy2DFloat64(caseRoot + "/in.npy");
        const auto aligned_m = reamix::test::loadNpy2DFloat64(caseRoot + "/aligned_out.npy");
        const std::size_t n = out_m.cols;
        std::vector<double> out0(out_m.data.begin(), out_m.data.begin() + n);
        std::vector<double> in0(in_m.data.begin(),   in_m.data.begin() + n);
        std::vector<double> aligned0(aligned_m.data.begin(), aligned_m.data.begin() + n);
        std::vector<double> out1(out_m.data.begin() + n, out_m.data.begin() + 2 * n);
        std::vector<double> in1(in_m.data.begin() + n,   in_m.data.begin()  + 2 * n);
        std::vector<double> aligned1(aligned_m.data.begin() + n,
                                     aligned_m.data.begin() + 2 * n);
        accumulate(aligned0, static_cast<std::size_t>(refMaxShift), out0, in0);
        accumulate(aligned1, static_cast<std::size_t>(refMaxShift), out1, in1);
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

    std::printf("\n=== PhaseAlign parity (Gate 1: shift bit-exact / "
                "Gate 2: aligned max_abs ≤ %.1e) ===\n", TOL_ALIGNED_ABS);

    int cases_ok = 0;
    int cases_tot = 0;
    double gmax = 0.0;
    std::size_t gover = 0;
    int shiftMismatches = 0;
    bool all_ok = true;

    for (const auto& c : kCases) {
        ++cases_tot;
        if (runCase(root, c, verbose, gmax, gover, shiftMismatches)) {
            ++cases_ok;
        } else {
            all_ok = false;
        }
    }

    std::printf("\n  cases=%d/%d PASS  shift_mismatches=%d  aligned_over_tol=%zu  "
                "aligned_max_abs=%.3e  %s\n",
                cases_ok, cases_tot, shiftMismatches, gover, gmax,
                all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
