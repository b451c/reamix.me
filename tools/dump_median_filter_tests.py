#!/usr/bin/env python3
"""
dump_median_filter_tests.py — scipy.ndimage.median_filter reference dumps for
the `reamix::util::MedianFilter` C++ parity test.

Emits four .npy files under `tests/parity/reference/data/median_filter/`:

  small_input.npy         # f32 (6, 10), handcrafted numbers to trigger every
                            reflect-boundary branch. Includes a column of
                            duplicates and a strict-monotonic row.
  small_harm3.npy         # f32, scipy.ndimage.median_filter(small, (1,3), mode='reflect')
  small_perc3.npy         # f32, scipy.ndimage.median_filter(small, (3,1), mode='reflect')
  stft_input.npy          # f32 (1025, 200), |STFT| slice from a deterministic
                            pink-noise audio buffer (numpy Generator seed=42).
                            Size chosen to match HPSS's real input shape
                            (1025 freq bins × n_frames) while staying small
                            enough to keep the .npy under ~1 MB compressed.
  stft_harm31.npy         # f32, median_filter(stft, (1,31), mode='reflect') — HPSS harmonic path.
  stft_perc31.npy         # f32, median_filter(stft, (31,1), mode='reflect') — HPSS percussive path.

Pins are the same as the main dump tool (scipy 1.15.3, numpy 1.26.x).

Usage:
    python tools/dump_median_filter_tests.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "parity" / "reference" / "data" / "median_filter"


def _assert_pins() -> None:
    # Conservative pin — matches tools/dump_python_features.py.
    if scipy.__version__ != "1.15.3":
        raise SystemExit(
            f"scipy {scipy.__version__} != pinned 1.15.3 — .npy SHAs will drift"
        )
    if np.__version__.startswith("2."):
        raise SystemExit(f"numpy {np.__version__} outside pinned range (<2.0)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(name: str, arr: np.ndarray) -> dict:
    assert arr.dtype == np.float32, f"{name}: expected f32, got {arr.dtype}"
    path = OUT_DIR / f"{name}.npy"
    np.save(path, arr, allow_pickle=False)
    return {
        "file":   f"{name}.npy",
        "shape":  list(arr.shape),
        "dtype":  str(arr.dtype),
        "sha256": _sha256(path),
        "bytes":  path.stat().st_size,
    }


def main() -> int:
    _assert_pins()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- small handcrafted ---------------------------------------------------
    # Crafted so: row 0 has a single spike (boundary-adjacent), row 5 is
    # strictly increasing, col 3 has a repeated value pattern, negatives
    # appear to sanity-check sort order.
    small = np.array([
        [ 1.0,  2.0,  3.0, -5.0,  4.0,  2.0,  1.0,  0.0, -1.0, -2.0],
        [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        [-3.0,  1.0,  4.0,  1.0,  5.0,  9.0,  2.0,  6.0,  5.0,  3.0],
        [ 2.0,  7.0,  1.0,  8.0,  2.0,  8.0,  1.0,  8.0,  2.0,  8.0],
        [ 0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0],
        [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0],
    ], dtype=np.float32)
    small_harm3 = ndimage.median_filter(small, size=(1, 3), mode="reflect").astype(np.float32, copy=False)
    small_perc3 = ndimage.median_filter(small, size=(3, 1), mode="reflect").astype(np.float32, copy=False)

    # --- realistic STFT-shaped ----------------------------------------------
    rng = np.random.Generator(np.random.PCG64(42))
    # Pink-noise-ish magnitude spectrum: positive, heavy-tailed with 1/f slope.
    # Shape (1025, 200) mirrors librosa.stft output (1 + n_fft/2 rows) at ~4 s
    # of audio at sr=22050, hop=512.
    noise = rng.standard_normal((1025, 200)).astype(np.float32)
    freq_idx = np.arange(1025, dtype=np.float32)[:, None]
    magnitude = np.abs(noise) / (1.0 + freq_idx / 50.0)
    stft = magnitude.astype(np.float32)

    stft_harm31 = ndimage.median_filter(stft, size=(1, 31), mode="reflect").astype(np.float32, copy=False)
    stft_perc31 = ndimage.median_filter(stft, size=(31, 1), mode="reflect").astype(np.float32, copy=False)

    manifest = {
        "pins": {"scipy": scipy.__version__, "numpy": np.__version__},
        "dumps": {
            "small_input":  _save("small_input", small),
            "small_harm3":  _save("small_harm3", small_harm3),
            "small_perc3":  _save("small_perc3", small_perc3),
            "stft_input":   _save("stft_input",  stft),
            "stft_harm31":  _save("stft_harm31", stft_harm31),
            "stft_perc31":  _save("stft_perc31", stft_perc31),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[dump] wrote {OUT_DIR}/manifest.json")
    for k, v in manifest["dumps"].items():
        print(f"  {k:14s} {str(v['shape']):14s} sha={v['sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
