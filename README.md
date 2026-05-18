# reamix.me

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-arm64%20%7C%20Intel-blue)]()
[![Windows](https://img.shields.io/badge/Windows-x64-blue)]()
[![Linux](https://img.shields.io/badge/Linux-x86__64%20%7C%20aarch64-blue)]()

Music-track retargeting for [REAPER](https://www.reaper.fm/). Pick any song, set a target length, and reamix produces a version at that length that sounds like a natural edit of the song — not a time-stretched version.

The concept is **Adobe Audition Remix**, re-implemented as a native REAPER extension. The technique is **beat-aligned splicing**, not time-stretching: the algorithm finds pairs of beats where splicing from one to the other is perceptually inaudible — matched phase, matched chroma, matched energy, matched structural role — and stitches those splices into a path that hits the target duration.

**Quality bar: a listener unaware of the edit cannot tell it happened.**

## Three modes

| Mode | What it does |
|------|--------------|
| **Target Duration** | Set a duration; the algorithm picks the optimal splice path across the whole track. |
| **Region Remix** | Retarget only the REAPER time-selection; rest of the track untouched. |
| **Block Assembly** | Manually order sections (Intro → Chorus → Verse → Chorus → Outro); the algorithm optimizes only the splice at each junction. |

## Install via ReaPack (recommended)

[ReaPack](https://reapack.com/) is REAPER's package manager.

1. Install ReaPack if you don't have it already.
2. In REAPER: **Extensions → ReaPack → Import repositories…** and paste:
   ```
   https://github.com/b451c/reamix.me/raw/main/index.xml
   ```
3. **Extensions → ReaPack → Browse packages…** → search `reamix` → install.
4. Restart REAPER. The plugin is now under **Extensions → reamix.me: Show/Hide Window**.

On first analysis the beat-detection model (~79 MB) downloads to `~/.reamix/models/`. One-time only.

## Install manually (direct download)

1. Download the binary for your platform from [Releases](https://github.com/b451c/reamix.me/releases/latest).
2. Place it in REAPER's `UserPlugins/` folder:
   - macOS: `~/Library/Application Support/REAPER/UserPlugins/`
   - Windows: `%APPDATA%\REAPER\UserPlugins\`
   - Linux: `~/.config/REAPER/UserPlugins/`
3. On macOS, if the dylib was downloaded via a browser, the system marks it quarantined. Clear that flag:
   ```sh
   xattr -d com.apple.quarantine ~/Library/Application\ Support/REAPER/UserPlugins/reaper_reamix.dylib
   ```
4. Restart REAPER → **Extensions → reamix.me: Show/Hide Window**.

ReaPack-installed binaries bypass the quarantine flag — the manual step is only needed for browser downloads.

## Pipeline (how it works)

```
audio in  →  beat detection (beat-this ONNX)
          →  structure segmentation (CBM, bar-level)
          →  per-beat 59-dim feature vector (MFCC + chroma + spectral contrast)
          →  transition cost matrix (10 perceptual signals + hard gates)
          →  Viterbi DP finds min-cost splice path under duration / cooldown / intro-outro constraints
          →  multi-band crossfade renders the splices
          →  WAV out
```

The whole pipeline is **in-process C++**. No Python runtime, no TCP server, no external interpreter. Single `.dylib` / `.dll` / `.so` drop-in.

## Build from source

### Prerequisites
- C++20 compiler (Clang 14+, GCC 12+, MSVC 2022+)
- CMake 3.22+
- Git
- Platform deps:
  - **macOS**: Xcode command-line tools
  - **Linux**: `freetype`, `libx11`, `libxrandr`, `libxcursor`, `libxinerama`, `alsa`, `webkit2gtk-4.0`, `libcurl4-openssl-dev`
  - **Windows**: Visual Studio 2022 with C++ workload

### Build
```sh
git clone --recurse-submodules https://github.com/b451c/reamix.me.git
cd reamix.me

# Download ONNX Runtime for your platform into vendor/onnxruntime/
# (URLs in .github/workflows/build.yml step "Download ONNX Runtime")

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8

# Run tests
ctest --test-dir build -j 8
```

The compiled binary is at `build/reaper_reamix.{dylib,dll,so}`. Copy it to REAPER's `UserPlugins/` folder.

## Architecture

The repo is organized as a **port** of an earlier Python+Lua implementation. Each pipeline stage corresponds to a closed **phase**:

| Phase | Module |
|-------|--------|
| 1 | Beat detection (beat-this ONNX) |
| 2 | Feature extraction (MFCC + chroma + spectral contrast) |
| 3 | Structure segmentation (CBM, bar-level) |
| 4 | Transition cost matrix (10 perceptual signals) |
| 5 | Optimization + rendering (Viterbi DP + multi-band crossfade) |
| 6 | JUCE UI (this is where the user-facing plugin lives) |
| 7 | Packaging + OSS release |

Each pipeline stage has parity tests against the Python reference. CI runs the full parity suite on every PR; failures block merges.

For contributors, the canonical entry point is **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## License

reamix.me's own source code is **MIT** licensed (see [LICENSE](LICENSE)).

The plugin links against [JUCE](https://github.com/juce-framework/JUCE), which is licensed under **AGPLv3** on the free track. The distributed binary therefore inherits AGPLv3 obligations when combined with JUCE: when redistributing the precompiled `.dylib` / `.dll` / `.so`, you must provide source under AGPLv3-compatible terms (the MIT source meets this) and preserve the JUCE copyright notice.

In practice this means:
- You can **use** reamix.me freely, including in commercial REAPER sessions.
- You can **modify and redistribute the source** under MIT.
- If you **redistribute the binary**, your distribution carries AGPLv3 terms for the combined work. Linking your own MIT source against JUCE is allowed.

### Third-party components

| Component | License | Used for |
|-----------|---------|----------|
| [JUCE](https://github.com/juce-framework/JUCE) | AGPLv3 (free track) | UI framework |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Neural inference |
| [beat-this](https://github.com/CPJKU/beat_this) (CPJKU, ISMIR 2024) | CC-BY-4.0 | Beat detection model (downloaded at runtime) |
| [reaper-sdk](https://github.com/justinfrankel/reaper-sdk) | LGPL-2.1 / custom | REAPER plugin headers |
| [WDL](https://www.cockos.com/wdl/) (SWELL) | zlib | macOS / Linux dialog layer |
| [PocketFFT](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) | BSD-3-Clause | FFT |
| [Inter](https://github.com/rsms/inter) | SIL OFL 1.1 | UI font |
| [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) | SIL OFL 1.1 | Monospace font |

Bundled font licenses are preserved under `assets/fonts/*-LICENSE.txt`.

## Acknowledgments

- **CPJKU** for the [beat-this](https://github.com/CPJKU/beat_this) model (ISMIR 2024) — best published F1 scores for beat + downbeat tracking.
- **Adobe Audition** for popularizing the Remix workflow that motivated this project.
- **[REABeat](https://github.com/b451c/ReaBeat)** — sibling REAPER extension by the same author; serves as the architectural template for native C++/JUCE/REAPER integration.
- **[SneakPeak / EditView](https://github.com/b451c/EditView)** — UI patterns (waveform, minimap, fade handles).
- The **REAPER community** for [ReaPack](https://reapack.com/) (cfillion), [SWS extension](https://www.sws-extension.org/) (numerous contributors), and the wider plugin ecosystem.

## Support development

If reamix.me saves you time, consider supporting development:

- [Ko-fi](https://ko-fi.com/quickmd)
- [Buy Me a Coffee](https://buymeacoffee.com/bsroczynskh)
- [PayPal](https://www.paypal.com/paypalme/b451c)

Donations fund the donate-funded OSS model for this project — no commercial licensing, no paywall, no telemetry.

## Contact + issues

- **Bugs and feature requests**: [GitHub Issues](https://github.com/b451c/reamix.me/issues)
- **Discussion**: [REAPER Forum thread](https://forum.cockos.com/) (coming with v1.0.0 release)
- **Security**: see [SECURITY.md](SECURITY.md)
