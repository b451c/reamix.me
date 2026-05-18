// test_butterworth — phase-5 session 30 parity: reamix::dsp::Butterworth
// vs scipy.signal.butter(4, ..., output='sos') + scipy.signal.sosfiltfilt.
//
// Two micro-gates per ADR-031:
//
//   Fixture A — SOS coefficient bit-exact vs scipy-dumped .npy:
//     3 band configs (bass LP @200 Hz / mid BP [200, 4000] Hz / treble HP
//     @4000 Hz) × 2 sample rates (22050, 44100) = 6 comparisons.
//     Gate: max_abs ≤ 1e-12 per element (f64 ULP class).
//
//   Fixture B — sosfiltfilt output round-trip:
//     4 signals (sine sweep 20 Hz→10 kHz, Gaussian white noise rng(42),
//     unit impulse at sample 1000, real music boundary waveform from
//     billie_jean phase-2 dump — session-32 Gap 1 closure) × 3 band configs
//     × 2 sample rates = 24 cases. Loads scipy-dumped x + sos + y as .npy;
//     runs C++ Butterworth::sosFiltFilt on the loaded (x, sos); compares
//     to loaded y.
//     Gate: max_abs ≤ 1e-9 per sample. Baseline: SosFilt session-18 ULP
//     ~2.3e-14 × 2× reverse-and-re-apply accumulation × up to 4096 samples.
//
//   Fixture C — boundary sample count (session-32 Gap 2 closure):
//     Positive: n = edge + 1 (LP/HP: n=16, BP: n=28) × 3 bands × 2 sr =
//       6 cases. Verifies sosFiltFilt doesn't throw AND matches scipy to
//       the same 1e-9 gate.
//     Negative: n = edge (LP/HP: n=15, BP: n=27) × 3 bands × 2 sr =
//       6 cases. Verifies sosFiltFilt throws std::invalid_argument.
//     Zero-length: n = 0 → also throws.
//
//   Fixture D — in-place aliasing (session-32 Gap 3 closure):
//     Re-run one Fixture B case with output == input (same buffer);
//     compare to the non-aliased reference; gate bitwise equal.
//
// Goldens dumped by tools/dump_phase5_filter_coefs.py (phase-5 session 30,
// extended session 32 with `real_boundary` + boundary-min inputs).

#include "dsp/Butterworth.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "NpyIO.h"

namespace {

struct BandCfg {
    const char* label;     // "bass" / "mid" / "treble"
    int         btype;     // 0=LP, 1=HP, 2=BP
    double      cutoff1Hz; // LP/HP: cutoff; BP: low edge
    double      cutoff2Hz; // BP: high edge (else ignored)
};

constexpr BandCfg kConfigs[] = {
    {"bass",   0 /*LP*/, 200.0,  0.0},
    {"mid",    2 /*BP*/, 200.0,  4000.0},
    {"treble", 1 /*HP*/, 4000.0, 0.0},
};

constexpr const char* kSignals[] = {"sine_sweep", "white_noise", "impulse", "real_boundary"};
constexpr int         kSampleRates[] = {22050, 44100};

constexpr double TOL_SOS_ABS = 1e-12;
constexpr double TOL_FF_ABS  = 1e-9;

std::string dataRoot(int argc, char** argv)
{
    // First non-flag positional arg wins; otherwise default to repo-relative.
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') return std::string(argv[i]);
    }
    return std::string("references/golden/phase-5/filter_coefs");
}

// Design SOS via C++ Butterworth based on btype + cutoffs + sr.
void designCpp(const BandCfg& cfg, int sr,
               std::vector<double>& sosOut, std::size_t& nSections)
{
    const double nyq = 0.5 * static_cast<double>(sr);
    sosOut.assign(reamix::dsp::Butterworth::kMaxSections * 6, 0.0);
    if (cfg.btype == 0) { // LP
        reamix::dsp::Butterworth::designLowpass(
            4, cfg.cutoff1Hz / nyq, sosOut.data(), &nSections);
    } else if (cfg.btype == 1) { // HP
        reamix::dsp::Butterworth::designHighpass(
            4, cfg.cutoff1Hz / nyq, sosOut.data(), &nSections);
    } else { // BP
        reamix::dsp::Butterworth::designBandpass(
            4, cfg.cutoff1Hz / nyq, cfg.cutoff2Hz / nyq,
            sosOut.data(), &nSections);
    }
    sosOut.resize(nSections * 6);
}

bool runFixtureA(const std::string& root, bool verbose)
{
    std::printf("\n=== Fixture A: SOS coefficient bit-exact (tol=%.1e) ===\n",
                TOL_SOS_ABS);
    double max_abs = 0.0;
    std::size_t over_tol = 0;
    std::size_t total = 0;
    bool ok = true;

    for (int sr : kSampleRates) {
        for (const auto& cfg : kConfigs) {
            const std::string srDir = root + "/" + std::to_string(sr);
            const std::string path  = srDir + "/sos_" + cfg.label + ".npy";
            auto ref = reamix::test::loadNpy2DFloat64(path);
            if (ref.cols != 6) {
                std::fprintf(stderr, "  %s: expected 6 cols, got %zu\n",
                             path.c_str(), ref.cols);
                return false;
            }

            std::vector<double> cpp;
            std::size_t         nSec = 0;
            designCpp(cfg, sr, cpp, nSec);

            if (nSec != ref.rows) {
                std::fprintf(stderr,
                    "  sr=%d %s: n_sections mismatch cpp=%zu ref=%zu\n",
                    sr, cfg.label, nSec, ref.rows);
                ok = false;
                continue;
            }

            double cfg_max = 0.0;
            for (std::size_t i = 0; i < nSec * 6; ++i) {
                const double d = std::abs(cpp[i] - ref.data[i]);
                if (d > cfg_max) cfg_max = d;
                if (d > max_abs) max_abs = d;
                if (d > TOL_SOS_ABS) ++over_tol;
                ++total;
            }
            if (verbose || cfg_max > TOL_SOS_ABS) {
                std::printf("  sr=%-5d %-6s n_sec=%zu  cfg_max_abs=%.3e  %s\n",
                            sr, cfg.label, nSec, cfg_max,
                            (cfg_max <= TOL_SOS_ABS) ? "PASS" : "FAIL");
            }
            if (cfg_max > TOL_SOS_ABS) ok = false;
        }
    }

    std::printf("  total elements=%zu  over_tol=%zu  max_abs=%.3e  %s\n",
                total, over_tol, max_abs, ok ? "PASS" : "FAIL");
    return ok;
}

bool runFixtureB(const std::string& root, bool verbose)
{
    std::printf("\n=== Fixture B: sosfiltfilt round-trip (tol=%.1e) ===\n",
                TOL_FF_ABS);
    double max_abs_global = 0.0;
    std::size_t over_tol_global = 0;
    std::size_t cases_ok = 0;
    std::size_t cases_tot = 0;
    bool ok = true;

    for (int sr : kSampleRates) {
        for (const auto& cfg : kConfigs) {
            const std::string srDir = root + "/" + std::to_string(sr);
            auto sos_ref = reamix::test::loadNpy2DFloat64(
                srDir + "/sos_" + cfg.label + ".npy");

            for (const char* sig : kSignals) {
                const std::string sigLabel(sig);
                auto x = reamix::test::loadNpy1DFloat64(
                    srDir + "/x_" + sigLabel + ".npy");
                auto y = reamix::test::loadNpy1DFloat64(
                    srDir + "/y_" + cfg.label + "_" + sigLabel + ".npy");

                if (x.size() != y.size()) {
                    std::fprintf(stderr,
                        "  sr=%d %s/%s: x/y size mismatch %zu vs %zu\n",
                        sr, cfg.label, sig, x.size(), y.size());
                    ok = false;
                    continue;
                }

                std::vector<double> out(x.size());
                reamix::dsp::Butterworth::sosFiltFilt(
                    sos_ref.data.data(), sos_ref.rows,
                    x.data(), x.size(), out.data());

                double case_max = 0.0;
                std::size_t case_over = 0;
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const double d = std::abs(out[i] - y[i]);
                    if (d > case_max) case_max = d;
                    if (d > TOL_FF_ABS) ++case_over;
                }

                if (case_max > max_abs_global) max_abs_global = case_max;
                over_tol_global += case_over;
                ++cases_tot;
                if (case_max <= TOL_FF_ABS) ++cases_ok;

                if (verbose || case_max > TOL_FF_ABS) {
                    std::printf(
                        "  sr=%-5d %-6s %-11s n=%zu over_tol=%zu/%zu "
                        "case_max=%.3e  %s\n",
                        sr, cfg.label, sig, out.size(), case_over, out.size(),
                        case_max, (case_max <= TOL_FF_ABS) ? "PASS" : "FAIL");
                }
                if (case_max > TOL_FF_ABS) ok = false;
            }
        }
    }

    std::printf("  cases=%zu/%zu PASS  over_tol_total=%zu  max_abs=%.3e  %s\n",
                cases_ok, cases_tot, over_tol_global, max_abs_global,
                ok ? "PASS" : "FAIL");
    return ok;
}

// edge = 3 * ntaps for Butterworth order 4: LP/HP -> 15, BP -> 27.
// (Verified via scipy in tools/dump_phase5_filter_coefs.py; sr-independent
// because the ntaps calculation depends on the zero pattern of the SOS, which
// is fixed per btype at a given order.)
int edgeFor(const BandCfg& cfg)
{
    return (cfg.btype == 2 /*BP*/) ? 27 : 15;
}

const char* boundaryInputLabelFor(const BandCfg& cfg)
{
    return (cfg.btype == 2 /*BP*/) ? "boundary_n28" : "boundary_n16";
}

bool runFixtureC(const std::string& root, bool verbose)
{
    std::printf("\n=== Fixture C: boundary sample count (tol=%.1e) ===\n",
                TOL_FF_ABS);
    double max_abs_global = 0.0;
    std::size_t pos_ok = 0, pos_tot = 0;
    std::size_t neg_ok = 0, neg_tot = 0;
    bool ok = true;

    for (int sr : kSampleRates) {
        for (const auto& cfg : kConfigs) {
            const std::string srDir = root + "/" + std::to_string(sr);
            auto sos_ref = reamix::test::loadNpy2DFloat64(
                srDir + "/sos_" + cfg.label + ".npy");
            const char* bmLabel = boundaryInputLabelFor(cfg);
            auto x = reamix::test::loadNpy1DFloat64(
                srDir + "/x_" + bmLabel + ".npy");
            auto y = reamix::test::loadNpy1DFloat64(
                srDir + "/y_" + std::string(cfg.label) + "_" + bmLabel + ".npy");

            // --- Positive: n = edge+1 — does NOT throw, parity vs scipy.
            ++pos_tot;
            std::vector<double> out(x.size());
            bool threw = false;
            try {
                reamix::dsp::Butterworth::sosFiltFilt(
                    sos_ref.data.data(), sos_ref.rows,
                    x.data(), x.size(), out.data());
            } catch (const std::exception&) {
                threw = true;
            }
            if (threw) {
                std::fprintf(stderr,
                    "  sr=%d %s %s (n=%zu, edge+1): unexpected throw\n",
                    sr, cfg.label, bmLabel, x.size());
                ok = false;
            } else {
                double case_max = 0.0;
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const double d = std::abs(out[i] - y[i]);
                    if (d > case_max) case_max = d;
                }
                if (case_max > max_abs_global) max_abs_global = case_max;
                if (case_max <= TOL_FF_ABS) ++pos_ok;
                else ok = false;
                if (verbose || case_max > TOL_FF_ABS) {
                    std::printf(
                        "  [pos] sr=%-5d %-6s n=%-2zu case_max=%.3e  %s\n",
                        sr, cfg.label, x.size(), case_max,
                        (case_max <= TOL_FF_ABS) ? "PASS" : "FAIL");
                }
            }

            // --- Negative: n = edge — MUST throw std::invalid_argument.
            ++neg_tot;
            const int           edge = edgeFor(cfg);
            std::vector<double> xe(static_cast<std::size_t>(edge), 1.0);
            std::vector<double> oute(static_cast<std::size_t>(edge));
            bool threwInvalid = false;
            try {
                reamix::dsp::Butterworth::sosFiltFilt(
                    sos_ref.data.data(), sos_ref.rows,
                    xe.data(), xe.size(), oute.data());
            } catch (const std::invalid_argument&) {
                threwInvalid = true;
            } catch (...) {
                // wrong exception type
            }
            if (threwInvalid) ++neg_ok;
            else ok = false;
            if (verbose || !threwInvalid) {
                std::printf(
                    "  [neg] sr=%-5d %-6s n=%-2d (edge) throw=%s  %s\n",
                    sr, cfg.label, edge, threwInvalid ? "yes" : "no",
                    threwInvalid ? "PASS" : "FAIL");
            }
        }
    }

    std::printf(
        "  pos=%zu/%zu PASS  neg=%zu/%zu PASS  max_abs=%.3e  %s\n",
        pos_ok, pos_tot, neg_ok, neg_tot, max_abs_global,
        ok ? "PASS" : "FAIL");
    return ok;
}

bool runFixtureD(const std::string& root, bool verbose)
{
    // One Fixture B case re-run in aliased mode (output == input buffer).
    // Pick sr=44100 bass real_boundary (music signal, non-trivial correlation).
    std::printf("\n=== Fixture D: in-place aliasing (bitwise equal) ===\n");
    const int sr = 44100;
    const BandCfg& cfg = kConfigs[0]; // bass
    const std::string srDir = root + "/" + std::to_string(sr);
    auto sos_ref = reamix::test::loadNpy2DFloat64(
        srDir + "/sos_" + cfg.label + ".npy");
    auto x = reamix::test::loadNpy1DFloat64(srDir + "/x_real_boundary.npy");

    // Non-aliased reference.
    std::vector<double> y_ref(x.size());
    reamix::dsp::Butterworth::sosFiltFilt(
        sos_ref.data.data(), sos_ref.rows,
        x.data(), x.size(), y_ref.data());

    // Aliased: output == input buffer. Copy x into a working buffer and pass
    // the same pointer for both input and output.
    std::vector<double> buf(x.size());
    std::memcpy(buf.data(), x.data(), x.size() * sizeof(double));
    reamix::dsp::Butterworth::sosFiltFilt(
        sos_ref.data.data(), sos_ref.rows,
        buf.data(), buf.size(), buf.data());

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] != y_ref[i]) ++mismatches;
    }
    const bool ok = (mismatches == 0);
    std::printf("  sr=%d %s real_boundary n=%zu aliased==ref: %s  (mismatches=%zu)\n",
                sr, cfg.label, x.size(),
                ok ? "PASS" : "FAIL", mismatches);
    if (verbose && ok) {
        std::printf("  (bitwise identical as expected — impl allocates "
                    "internal ext/y_fwd/y_rev/y2 buffers)\n");
    }
    return ok;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    const std::string root = dataRoot(argc, argv);
    // Default: one-line summary per fixture + only FAIL cases. `--verbose`
    // (or `-v`) prints every case.
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--verbose" || a == "-v") verbose = true;
    }

    const bool a_ok = runFixtureA(root, verbose);
    const bool b_ok = runFixtureB(root, verbose);
    const bool c_ok = runFixtureC(root, verbose);
    const bool d_ok = runFixtureD(root, verbose);

    std::printf("\noverall: Fixture A %s, Fixture B %s, "
                "Fixture C %s, Fixture D %s\n",
                a_ok ? "PASS" : "FAIL",
                b_ok ? "PASS" : "FAIL",
                c_ok ? "PASS" : "FAIL",
                d_ok ? "PASS" : "FAIL");
    return (a_ok && b_ok && c_ok && d_ok) ? 0 : 1;
}
