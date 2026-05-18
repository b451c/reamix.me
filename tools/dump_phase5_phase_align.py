#!/usr/bin/env python3
"""Phase-5 parity dump tool — _phase_align goldens (time-domain NCC).

Dumps golden .npy fixtures for:
  - references/python-source/remix/crossfade.py::_phase_align (L86-153)

_phase_align semantics, verbatim from Python:
  - Input: `outgoing`, `incoming` (ndarrays, ..., samples) + `max_shift_samples: int`
  - Uses first channel (if multi-D) for NCC lookup; applies shift to all channels.
  - Brute loop over `shift in range(-max_shift, max_shift+1)`:
      o = out[shift:]                  (shift >= 0)
      i = in[:n-shift]
      o = out[:n+shift]                (shift < 0)
      i = in[-shift:]
      skip if len(o) < min_overlap = max(16, n//2)
      skip if ||o|| < 1e-8 or ||i|| < 1e-8
      corr = dot(o, i) / (||o|| * ||i||)
      update best if `corr > best_corr` (init -1.0) — FIRST-occurrence tie-break.
  - If best_shift == 0: return (incoming, 0) — original reference, no copy.
  - Else: result = np.zeros_like(incoming); shifted-slice assign.

Production caller: adaptive_crossfade (crossfade.py:298-303) passes
`max_phase_shift_samples = int(3.0 * sr / 1000.0)` (→ 132 @ sr=44100, 66 @ sr=22050).
The `_phase_align` signature default `max_shift_samples=64` is NEVER reached from
the production path; included only as an API-default case for port-time completeness.

Cases dumped (each under references/golden/phase-5/phase_align/case_<N>/):

  case_01 — 1D f64 mono, synthetic sinusoid, known shift +17, max_shift=132.
            Verifies positive-shift branch + precise shift-index detection.
  case_02 — 1D f64 mono, synthetic sinusoid, known shift -23, max_shift=132.
            Verifies negative-shift branch.
  case_03 — 1D f64 mono, zero-shift case (incoming == outgoing), max_shift=132.
            Verifies best_shift==0 passthrough branch (output == input bit-exact).
  case_04 — 2D f64 stereo, same sinusoid content as case_01, shift +17, max_shift=132.
            Verifies stereo shift-broadcast (first-channel NCC lookup, all-channel apply).
  case_05 — 1D f64 mono, norm-guard trigger: outgoing all-zero, max_shift=32.
            Verifies both-norm-guards skip path → best_shift stays 0 → passthrough.
  case_06 — 1D f64 mono, short buffer n=31 (< 32 min-length gate), max_shift=16.
            Verifies early-return passthrough.
  case_07 — 1D f64 mono, synthetic pink-ish noise, shift +5, max_shift=64.
            Verifies API-default max_shift path + near-zero shift + noise (non-sinusoidal).

Per-case .npy files:
  out.npy          outgoing array      (f64 1D or f64 2D)
  in.npy           incoming array      (f64 1D or f64 2D)
  max_shift.npy    ()                  int64 scalar
  shift_out.npy    ()                  int64 scalar — Python `_phase_align` return[1]
  aligned_out.npy  shape of `in`       f64            — Python `_phase_align` return[0]

Usage:
    python3 tools/dump_phase5_phase_align.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "references" / "golden" / "phase-5" / "phase_align"

# Import Python _phase_align verbatim from the reference tree. Mirrors the
# sys.path pattern used by tools/dump_phase4_tests.py (parent-of-remix_tool).
_PY_SRC_SYMLINK = ROOT / "references" / "python-source"
_REMIX_TOOL_PKG_PARENT = _PY_SRC_SYMLINK.resolve().parent
if _REMIX_TOOL_PKG_PARENT.is_dir() and str(_REMIX_TOOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_REMIX_TOOL_PKG_PARENT))
from remix_tool.remix.crossfade import _phase_align  # noqa: E402


def _shift_clean(x: np.ndarray, k: int) -> np.ndarray:
    """Reference shift-apply helper (matches _phase_align shift-apply branch).
    Returns zeros_like(x) with x shifted by k: positive k → left-shift, trailing
    zeros; negative k → right-shift, leading zeros. Used only to construct
    known-answer synthetic fixtures (ground-truth *incoming* arrays that, once
    aligned by -k, should produce a zeroed original)."""
    r = np.zeros_like(x)
    if k == 0:
        return x.copy()
    if k > 0:
        if x.ndim == 1:
            r[:-k] = x[k:]
        else:
            r[:, :-k] = x[:, k:]
    else:
        s = -k
        if x.ndim == 1:
            r[s:] = x[:-s]
        else:
            r[:, s:] = x[:, :-s]
    return r


def _write_case(case_dir: Path, out: np.ndarray, inp: np.ndarray,
                max_shift: int, notes: dict) -> None:
    aligned, shift = _phase_align(out, inp, max_shift)
    case_dir.mkdir(parents=True, exist_ok=True)

    # f64 everywhere (production path casts inputs to f64 upstream; bit-exact
    # parity easiest if the goldens are f64 and the C++ reads f64).
    np.save(case_dir / "out.npy", out.astype(np.float64))
    np.save(case_dir / "in.npy", inp.astype(np.float64))
    # NpyIO supports only 1-D/2-D; wrap scalars as shape (1,) per the
    # tuning_est.npy precedent (NpyIO.h L114 comment).
    np.save(case_dir / "max_shift.npy", np.array([max_shift], dtype=np.int64))
    np.save(case_dir / "shift_out.npy", np.array([shift], dtype=np.int64))
    # `_phase_align` returns `incoming` reference when shift==0 (same dtype).
    # Force output to f64 so the C++ side always reads f64.
    np.save(case_dir / "aligned_out.npy", np.asarray(aligned, dtype=np.float64))

    with (case_dir / "notes.json").open("w") as f:
        json.dump({**notes, "shift_out": int(shift)}, f, indent=2)

    print(f"  {case_dir.name}: max_shift={max_shift:3d} "
          f"n_out={out.shape[-1]:5d} n_in={inp.shape[-1]:5d} "
          f"ndim={out.ndim} -> shift={shift:+4d}")


def _sinusoid(n: int, f_hz: float, sr: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / float(sr)
    return np.sin(2.0 * np.pi * f_hz * t + phase).astype(np.float64)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"dumping phase_align goldens under {OUT_ROOT}")

    # --- case_01: mono sinusoid, known shift +17 @ sr=44100, max_shift=132 --
    # Build outgoing = sine, incoming = sine shifted by k (i.e., incoming[i] =
    # sine[i - k] extended with leading zeros). _phase_align should return
    # best_shift = -k (so that aligning incoming by -k recovers outgoing).
    #
    # Actually: Python _phase_align's *interpretation* of shift:
    #   shift >= 0: o = out[shift:],   i = in[:n - shift]
    #   so it tests "incoming leads outgoing by `shift` samples"
    # If we set incoming = out shifted LATE by k positive samples (so incoming
    # has outgoing-content starting at sample k, with leading zeros), then the
    # best alignment is shift = -k: o = out[:n-k], i = in[k:] — same content.
    # Loop range (-max, max+1) ascending; at shift=-k we find the first match.
    sr = 44100
    n = 2048
    out = _sinusoid(n, 440.0, sr)
    k = 17
    inp = np.zeros_like(out)
    inp[k:] = out[:-k]          # incoming lags outgoing by +k samples
    # _phase_align expected to return shift = -k (incoming starts "later").
    _write_case(
        OUT_ROOT / "case_01_mono_shift_plus17_sr44100",
        out, inp, 132,
        notes={
            "n": n, "sr": sr, "known_shift": -k,
            "signal": "sine 440 Hz",
            "why": "positive-shift branch (incoming lags; best_shift negative)",
        },
    )

    # --- case_02: mono sinusoid, shift that forces best_shift = +23 ---
    # Symmetric construction: incoming = out with LEADING content zeroed past k.
    # inp[:n-k] = out[k:] means "incoming contains out shifted EARLY by k".
    # _phase_align expected to return shift = +k (o = out[k:], i = in[:n-k]).
    out = _sinusoid(n, 880.0, sr)
    k = 23
    inp = np.zeros_like(out)
    inp[:n-k] = out[k:]         # incoming leads outgoing by +k samples
    _write_case(
        OUT_ROOT / "case_02_mono_shift_minus23_sr44100",
        out, inp, 132,
        notes={
            "n": n, "sr": sr, "known_shift": +k,
            "signal": "sine 880 Hz",
            "why": "negative-shift branch (incoming leads; best_shift positive)",
        },
    )

    # --- case_03: mono, zero-shift passthrough (incoming == outgoing) --------
    out = _sinusoid(n, 523.25, sr)  # C5
    inp = out.copy()
    _write_case(
        OUT_ROOT / "case_03_mono_zero_shift_sr44100",
        out, inp, 132,
        notes={
            "n": n, "sr": sr, "known_shift": 0,
            "signal": "sine 523.25 Hz (C5) — identical in/out",
            "why": "best_shift==0 passthrough branch (output == input bit-exact)",
        },
    )

    # --- case_04: stereo 2D f64, same sinusoid content, shift +17 ----------
    sr = 44100
    n = 2048
    base = _sinusoid(n, 440.0, sr)
    # Stereo: left = base, right = same base ×0.9 (distinct RMS but correlated)
    out_st = np.stack([base, 0.9 * base])
    k = 17
    inp_st = np.zeros_like(out_st)
    inp_st[:, k:] = out_st[:, :-k]
    _write_case(
        OUT_ROOT / "case_04_stereo_shift_plus17_sr44100",
        out_st, inp_st, 132,
        notes={
            "n": n, "sr": sr, "known_shift": -k, "ndim": 2,
            "signal": "stereo sine 440 Hz (L=1.0×, R=0.9×)",
            "why": "stereo NCC-on-first-channel + all-channel shift-apply",
        },
    )

    # --- case_05: norm-guard trigger (outgoing all zero) -------------------
    # When ||outgoing||<1e-8 for every shift, ALL iterations skip. best_corr
    # stays -1.0, best_shift stays 0. Early-return (incoming, 0) passthrough.
    out = np.zeros(1024, dtype=np.float64)
    inp = _sinusoid(1024, 440.0, sr)
    _write_case(
        OUT_ROOT / "case_05_norm_guard_zero_out",
        out, inp, 32,
        notes={
            "n": 1024, "sr": sr, "known_shift": 0,
            "signal": "out=all_zero, in=sine 440 Hz",
            "why": "norm-guard 1e-8 on both; every iter skips; passthrough",
        },
    )

    # --- case_06: short buffer (n < 32 min-length gate) ---------------------
    out = _sinusoid(31, 440.0, sr)
    inp = _sinusoid(31, 440.0, sr, phase=0.1)
    _write_case(
        OUT_ROOT / "case_06_short_buffer_n31",
        out, inp, 16,
        notes={
            "n": 31, "sr": sr, "known_shift": 0,
            "signal": "sine 440 Hz, n=31",
            "why": "n<32 min-length gate → early return (incoming, 0)",
        },
    )

    # --- case_07: API-default max_shift=64 + noise ------------------------
    rng = np.random.default_rng(42)
    out = rng.standard_normal(2048).astype(np.float64)
    k = 5
    inp = np.zeros_like(out)
    inp[:2048-k] = out[k:]     # incoming leads by +5
    _write_case(
        OUT_ROOT / "case_07_mono_noise_max_shift_64",
        out, inp, 64,
        notes={
            "n": 2048, "sr": sr, "known_shift": +k,
            "signal": "white noise rng(42)",
            "why": "API-default max_shift=64 + noise (non-sinusoidal corr)",
        },
    )

    print(f"done. goldens at {OUT_ROOT}")


if __name__ == "__main__":
    main()
