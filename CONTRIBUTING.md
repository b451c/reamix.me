# Contributing to reamix.me

Thank you for considering contributing to reamix.me. This document describes the standards and workflow for making changes.

## Quality bar

reamix.me ships **perceptually inaudible** edits. Every change that touches the DSP pipeline must preserve this bar. Practically:

- **Numerical parity is a gate.** DSP modules ported from Python (MFCC, chroma, spectral contrast, crossfade filters, cross-correlation) must match the Python reference within the tolerances enforced by `tests/parity/`. CI runs parity tests; failures block merges.
- **No silent threshold widening.** If a parity test fails, root-cause-investigate before adjusting the tolerance.
- **Listen before you ship.** Audio-producing changes must produce listenable WAVs and be subjected to A/B listening before merge.

This is a higher bar than typical OSS audio projects. It exists because remix-quality drift is *audible* — a click, an off-downbeat splice, a mid-vocal cut. The quality bar is the product.

## Getting started

### Prerequisites
- C++20 compiler (Clang 14+, GCC 12+, MSVC 2022+)
- CMake 3.22+
- Python 3.11+ (for parity tests against the Python reference)
- Platform deps documented in [README.md § Build from source](README.md#build-from-source).

### Build + test
```sh
git clone --recurse-submodules https://github.com/b451c/reamix.me.git
cd reamix.me

# Download ONNX Runtime — see .github/workflows/build.yml for per-platform URLs

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 8

# Run all tests (production + dev parity). Expect ~50 sec on M1.
ctest --test-dir build -j 8
```

The compiled binary is at `build/reaper_reamix.{dylib,dll,so}`. Copy it to REAPER's `UserPlugins/` folder and restart REAPER to test.

## Repository layout

| Path | Role |
|------|------|
| `src/` | C++ source. |
| `tests/{unit,parity,e2e}/` | Tests. The `parity/` directory blocks merges on numerical drift. |
| `tools/` | Build helpers (CLIs, benchmarks, golden generators). |
| `vendor/` | Submodules (JUCE, PocketFFT, WDL, reaper-sdk). ONNX Runtime is downloaded at build time per `.github/workflows/build.yml`. |
| `.github/` | CI workflows + issue templates. |
| `assets/` | Bundled fonts (Inter, JetBrains Mono — SIL OFL 1.1). |

## Pull request checklist

1. **Fork** the repository and create a feature branch (`git checkout -b feature/short-name`).
2. **Build + test locally**: `cmake --build build -j` + `ctest --test-dir build`. All tests must pass.
3. **For DSP / cost-matrix / rendering changes**: run an A/B listening session on at least 2 tracks of distinct genre. Document the comparison in the PR description.
4. **For UI changes**: include a screenshot or short screen recording in the PR. UI is the product — visual review matters.
5. **For new cost components or threshold changes**: document context / chosen direction / counter-arguments in the PR description.
6. **Commit messages** should describe the **why**, not the **what**. The diff shows the what.
7. Open a PR. CI must pass on all four platforms (macOS arm64 + macOS x86_64 + Windows x64 + Linux x86_64) before merge.

## Code style

- **C++20**, no exceptions in hot paths.
- **JUCE coding conventions** for UI components (lowerCamelCase methods, 4-space indent, leading-underscore for member-of-class qualification is **not** used — trailing underscore is: `weights_`, `progress_`).
- **All REAPER API mutations** wrapped in `Undo_BeginBlock` / `Undo_EndBlock` pairs.
- **No blocking dialogs** during compute — the plugin window must stay responsive.
- `-Wall -Wextra` clean (zero warnings).
- **Comments explain WHY, not WHAT.** Identifier names should explain the what. Comments are reserved for non-obvious invariants, references to ADRs, or sesja-N history pointers.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Please include:

- **REAPER version** + **OS + architecture** (e.g. macOS 14.5 arm64).
- **reamix.me version** (visible in plugin window's bottom-right corner).
- **Steps to reproduce** — including the audio file's BPM, duration, and genre if quality-related.
- **Expected vs actual** behavior.
- **Audio file** if possible (or a publicly accessible sample of the same character).

## Requesting features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml). Describe the **use case** — not just the feature. We need to know what audible/visual outcome you can't achieve today; that informs whether the right answer is a new control, a new mode, or a cost-component tweak.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) (with the JUCE AGPLv3 binary obligation as documented in the README).
