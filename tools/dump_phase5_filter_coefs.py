#!/usr/bin/env python3
"""Phase-5 parity dump tool — Butterworth SOS design + sosfiltfilt round-trip.

Dumps golden .npy fixtures for:
  - scipy.signal.butter(4, Wn, btype, output='sos')  SOS coefficients
  - scipy.signal.sosfiltfilt(sos, x)  zero-phase forward-backward output

Band configurations mirror `references/python-source/remix/crossfade.py`:
  - bass lowpass  (btype='low',  Wn=200 Hz)           -> 2 sections
  - mid bandpass  (btype='band', Wn=[200, 4000] Hz)   -> 4 sections
  - treble highpass (btype='high', Wn=4000 Hz)         -> 2 sections

Each config is dumped at two sample rates (22050, 44100). Three synthetic
signals (length 4096) feed the round-trip micro-gate:
  - sine_sweep : exponential chirp 20 Hz -> 10 kHz (linear freq ramp in f64)
  - white_noise: Gaussian N(0,1), np.random.default_rng(42)
  - impulse    : unit impulse at sample 1000

Session-32 addition — one real-audio case (`real_boundary`, length 3417) closes
Gap 1 of the session-30 Butterworth self-audit: boundary waveform #0 from the
`billie_jean` phase-2 dump (music fragment at sr=22050, cast f32 -> f64), to
exercise real-music correlation structure + dynamic range that synthetics don't
produce. Used at BOTH sample rates for parity coverage (not a musical claim —
filter parity is sr-agnostic; 44100 just runs C++ sosfiltfilt on a different
SOS matrix than 22050 with the same input).

Session-32 addition — minimum-length boundary cases close Gap 2. scipy
sosfiltfilt requires `len(x) > padlen`; our padlen is `edge = 3 * ntaps`. For
Butterworth order 4 with sos-from-butter: LP/HP -> edge=15, BP -> edge=27.
We dump `x_boundary_n16.npy` (used by LP/HP configs, n = 16 = edge+1) and
`x_boundary_n28.npy` (used by BP, n = 28 = edge+1), plus corresponding y
goldens per band. Edge values are sr-independent so a single x per length is
shared across the two sample-rate dirs.

Output tree: references/golden/phase-5/filter_coefs/<sr>/
  sos_bass.npy           (n_sections, 6)  f64
  sos_mid.npy            (n_sections, 6)  f64
  sos_treble.npy         (n_sections, 6)  f64
  x_sine_sweep.npy       (4096,)          f64
  x_white_noise.npy      (4096,)          f64
  x_impulse.npy          (4096,)          f64
  y_bass_sine_sweep.npy  (4096,)          f64    # sosfiltfilt(sos_bass, x_sine_sweep)
  ... (9 combinations per sr: 3 configs × 3 signals)

The C++ side (tests/parity/test_butterworth.cpp) generates the SAME synthetic
inputs from the documented formulas; the .npy x_*.npy files are recorded for
audit + exact-numeric cross-check.

Usage:
    python3 tools/dump_phase5_filter_coefs.py           # dumps everything
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "references" / "golden" / "phase-5" / "filter_coefs"

# billie_jean boundary waveform #0 — phase-2 fixture, already available locally.
# Phase-4 session-18 `test_sosfilt` consumed this same file for its parity dump.
REAL_BOUNDARY_SRC = (
    ROOT / "references" / "golden" / "phase-2" / "dumps" /
    "billie_jean" / "boundary_waveforms.npy"
)

SAMPLE_RATES = [22050, 44100]
SIGNAL_LEN = 4096

# (label, btype, Wn_hz) — Wn_hz is scalar for LP/HP, 2-tuple for BP.
# Cutoffs below are the ones hardcoded in crossfade.py::BANDS.
BAND_CONFIGS = [
    ("bass",   "low",  200.0),
    ("mid",    "band", (200.0, 4000.0)),
    ("treble", "high", 4000.0),
]


def _design_sos(btype: str, wn_hz, sr: int) -> np.ndarray:
    nyq = sr / 2.0
    if isinstance(wn_hz, tuple):
        wn = [wn_hz[0] / nyq, wn_hz[1] / nyq]
    else:
        wn = wn_hz / nyq
    return butter(4, wn, btype=btype, output="sos")


def _load_real_boundary() -> np.ndarray:
    """boundary_waveforms[0] as f64. RMS-normalized music fragment ~155 ms @ 22050."""
    wf = np.load(REAL_BOUNDARY_SRC)
    return wf[0].astype(np.float64, copy=False)


def _make_signals(sr: int) -> dict[str, np.ndarray]:
    n = SIGNAL_LEN
    t = np.arange(n, dtype=np.float64) / float(sr)

    # Exponential chirp 20 Hz -> 10 kHz over the 4096-sample window. We compute
    # instantaneous phase by integrating the exponential frequency ramp, which
    # gives a cleaner spectral sweep than scipy.signal.chirp's linear variant
    # and avoids the library-version-sensitive chirp() dependency.
    f0, f1 = 20.0, 10_000.0
    T = float(n - 1) / float(sr)
    k = np.log(f1 / f0) / T
    phase = 2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    sine_sweep = np.sin(phase)

    rng = np.random.default_rng(42)
    white_noise = rng.standard_normal(n)

    impulse = np.zeros(n, dtype=np.float64)
    impulse[1000] = 1.0

    return {
        "sine_sweep":    sine_sweep,
        "white_noise":   white_noise,
        "impulse":       impulse,
        "real_boundary": _load_real_boundary(),
    }


def _make_boundary_min() -> dict[str, np.ndarray]:
    """Minimum-length inputs for Gap-2 boundary parity: n = edge + 1.

    Signal shape: small deterministic ramp (values 1..n) is enough — the point
    is to exercise the n ≈ edge+1 branch of sosfiltfilt, not a musical property.
    """
    n16 = np.arange(1, 17, dtype=np.float64)  # for LP/HP (edge=15 -> n=16)
    n28 = np.arange(1, 29, dtype=np.float64)  # for BP    (edge=27 -> n=28)
    return {
        "boundary_n16": n16,
        "boundary_n28": n28,
    }


# Map band label -> which boundary input applies. Keyed by the ntaps-class
# the C++ Butterworth computes, which depends on the btype at order 4.
BOUNDARY_INPUT_FOR_BAND = {
    "bass":   "boundary_n16",  # LP, edge=15
    "mid":    "boundary_n28",  # BP, edge=27
    "treble": "boundary_n16",  # HP, edge=15
}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    boundary_min = _make_boundary_min()

    for sr in SAMPLE_RATES:
        sr_dir = OUT_ROOT / f"{sr}"
        sr_dir.mkdir(parents=True, exist_ok=True)

        signals = _make_signals(sr)
        for sig_label, x in signals.items():
            np.save(sr_dir / f"x_{sig_label}.npy", x.astype(np.float64, copy=False))

        # Gap-2 boundary inputs — same content each sr (inputs are sr-independent;
        # only the SOS matrix differs).
        for sig_label, x in boundary_min.items():
            np.save(sr_dir / f"x_{sig_label}.npy", x)

        for cfg_label, btype, wn_hz in BAND_CONFIGS:
            sos = _design_sos(btype, wn_hz, sr).astype(np.float64, copy=False)
            np.save(sr_dir / f"sos_{cfg_label}.npy", sos)

            for sig_label, x in signals.items():
                y = sosfiltfilt(sos, x).astype(np.float64, copy=False)
                np.save(sr_dir / f"y_{cfg_label}_{sig_label}.npy", y)

            # Gap-2: boundary-min y per band (uses the band's edge-length input).
            bm_label = BOUNDARY_INPUT_FOR_BAND[cfg_label]
            y_bm = sosfiltfilt(sos, boundary_min[bm_label]).astype(
                np.float64, copy=False)
            np.save(sr_dir / f"y_{cfg_label}_{bm_label}.npy", y_bm)

        n_bm_y = len(BAND_CONFIGS)
        print(f"[phase-5 dump] sr={sr}: wrote "
              f"{len(BAND_CONFIGS)} SOS + {len(signals)} signals + "
              f"{len(boundary_min)} boundary inputs + "
              f"{len(BAND_CONFIGS) * len(signals)} outputs + "
              f"{n_bm_y} boundary outputs -> {sr_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
