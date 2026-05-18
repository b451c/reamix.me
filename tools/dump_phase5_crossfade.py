#!/usr/bin/env python3
"""Phase-5 parity dump tool — crossfade goldens (adaptive FIXED + simple).

Adaptive path (cases 01-04):
Dumps golden .npy fixtures for `adaptive_crossfade` from
`references/python-source/remix/crossfade.py` with **one algorithmic deviation**
from the unmodified Python (ADR-032 Option B.myfix):

  - Instead of slicing `in_band[..., :xf_samples]` → `_phase_align` → pad-back
    at `[0:xf_samples]` (lines 299-308 of crossfade.py), the fixed variant
    computes the shift on the same fade-zone pair but applies the integer
    shift to the FULL-length `in_band` (max_xf_samples samples). This matches
    the envelope placement in lines 312-327 and recovers post-fade incoming
    content that the unmodified Python zeros out.
  - All other logic (bandpass cascade via Butterworth+sosfiltfilt, energy
    compensation RMS clamp 0.35..2.83 at ±9 dB, equal-power cos/sin envelope,
    per-band sum into overlap_result) is ported verbatim from Python.

Simple path (cases 05-07) — session-33 SB-9 closure:
Dumps golden fixtures for `simple_crossfade` (crossfade.py:335-366) — the
single-band legacy path. Imported directly from the unmodified reference
(no bug on this path, no monkey-patch needed). Exercises:

  - DEFAULT_CROSSFADE_MS = 75.0 default duration.
  - Linear gain ramp `linspace(gain, 1.0, min(n_overlap*4, inLen))` applied
    to incoming (extends BEYOND overlap zone up to 4× n_overlap into the
    non-overlap incoming — smooths the RMS gain from `gain` back to 1.0).
  - Shared constants with adaptive: 0.35/2.83 clamp, 1e-10 RMS sentinel,
    1e-6 gain-apply threshold.

ADR-032 records the full adaptive root cause, empirical data, and revisit
triggers. Per `CLAUDE.md` "references are read-only", the unmodified Python
symlink is not touched; the adaptive fix is re-implemented locally in this
dump tool and in the C++ port (src/render/Crossfade.{h,cpp}).

Cases (each under references/golden/phase-5/crossfade/case_<NN>/):

  case_01 — billie_jean boundary stereo, adaptive, PA=True, EC=True
            (production default path). Real music, all 3 bands, PHASE-ALIGN BUG FIX
            visible in the mid + treble post-fade regions vs. unmodified Python.
  case_02 — billie_jean boundary stereo, adaptive, PA=False, EC=True
            (no-align control). Bit-exact with unmodified Python (no divergence on
            this branch).
  case_03 — synthetic bass(80 Hz)+treble(8 kHz) stereo with phase offset,
            adaptive, PA=True, EC=False. Isolates PhaseAlign + envelope
            without the RMS clamp — cleanest view of the ADR-032 fix effect.
  case_04 — synthetic sines with large RMS mismatch (-20 dB), adaptive, PA=True,
            EC=True. Exercises the clip(0.35, 2.83) clamp.

  case_05 — billie_jean boundary stereo, simple, crossfade_ms=75.0, EC=True.
            Production-default simple path on real music — exercises ramp on
            real content, ramp saturates `n_overlap*4` cap by inLen (inLen<4*n_overlap).
  case_06 — synthetic stereo with -20 dB incoming mismatch, simple,
            crossfade_ms=75.0, EC=True. Exercises RMS clamp upper bound + full
            ramp length (inLen >> 4*n_overlap).
  case_07 — synthetic stereo, simple, crossfade_ms=75.0, EC=False. Control
            case — no RMS compensation, no ramp, pure equal-power cos/sin fade.

Per-case files:
  out.npy      outgoing audio  (f64 2D nCh × nSamp)
  in.npy       incoming audio  (f64 2D nCh × nSamp)
  sr.npy       sample rate     (int64 shape (1,))
  flags.json   kind + flags (adaptive: phase_align/energy_compensate/max_phase_shift_ms;
               simple: crossfade_ms/energy_compensate) + notes
  result.npy   output (f64 2D nCh × nResult)

Usage:
    python3 tools/dump_phase5_crossfade.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "references" / "golden" / "phase-5" / "crossfade"

_PY_SRC_SYMLINK = ROOT / "references" / "python-source"
_REMIX_TOOL_PKG_PARENT = _PY_SRC_SYMLINK.resolve().parent
if _REMIX_TOOL_PKG_PARENT.is_dir() and str(_REMIX_TOOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_REMIX_TOOL_PKG_PARENT))

# Import primitives — these are NOT the bug site and need no modification.
# simple_crossfade has no bug; import directly for cases 05-07.
from remix_tool.remix.crossfade import (  # noqa: E402
    BANDS,
    DEFAULT_CROSSFADE_MS,
    _bandpass,
    _equal_power_fade,
    _phase_align,
    simple_crossfade,
)


def _apply_shift_full(in_band: np.ndarray, shift: int) -> np.ndarray:
    """Shift a full-length buffer by `shift` samples, zero-pad the freed end.

    Semantics match Python `_phase_align` shift-apply branch (crossfade.py:140-151),
    but applied to `in_band` of length max_xf_samples instead of to a xf_samples slice.
    """
    if shift == 0:
        return in_band
    shifted = np.zeros_like(in_band)
    if shift > 0:
        shifted[..., :-shift] = in_band[..., shift:]
    else:
        s = -shift
        shifted[..., s:] = in_band[..., :-s]
    return shifted


def adaptive_crossfade_fixed(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    *,
    bands: list | None = None,
    energy_compensate: bool = True,
    phase_align: bool = True,
    max_phase_shift_ms: float = 3.0,
) -> np.ndarray:
    """ADR-032 Option B.myfix — adaptive_crossfade with shift-full-in_band.

    Identical to `references/python-source/remix/crossfade.py::adaptive_crossfade`
    EXCEPT the phase-align block replaces the slice+pad-back with a shift-full
    operation. See ADR-032 for rationale.
    """
    if bands is None:
        bands = BANDS

    max_phase_shift_samples = int(max_phase_shift_ms * sr / 1000.0)

    max_xf_samples = max(int(xf_ms * sr / 1000.0) for _, _, xf_ms in bands)
    max_xf_samples = min(max_xf_samples, outgoing.shape[-1], incoming.shape[-1])

    if max_xf_samples <= 0:
        return np.concatenate([outgoing, incoming], axis=-1)

    result_len = outgoing.shape[-1] + incoming.shape[-1] - max_xf_samples
    shape = (*outgoing.shape[:-1], result_len)
    result = np.zeros(shape, dtype=np.float64)

    non_overlap_out = outgoing.shape[-1] - max_xf_samples
    if non_overlap_out > 0:
        result[..., :non_overlap_out] = outgoing[..., :non_overlap_out]
    non_overlap_in_start = non_overlap_out + max_xf_samples
    remaining_in = incoming.shape[-1] - max_xf_samples
    if remaining_in > 0:
        result[..., non_overlap_in_start:] = incoming[..., max_xf_samples:]

    out_overlap = outgoing[..., -max_xf_samples:].astype(np.float64)
    in_overlap = incoming[..., :max_xf_samples].astype(np.float64)
    overlap_result = np.zeros_like(out_overlap)

    for low_hz, high_hz, xf_ms in bands:
        xf_samples = min(int(xf_ms * sr / 1000.0), max_xf_samples)
        if xf_samples <= 0:
            continue

        out_band = _bandpass(out_overlap, low_hz, high_hz, sr)
        in_band = _bandpass(in_overlap, low_hz, high_hz, sr)

        if energy_compensate:
            rms_out = np.sqrt(np.mean(out_band ** 2) + 1e-10)
            rms_in = np.sqrt(np.mean(in_band ** 2) + 1e-10)
            if rms_out > 1e-6 and rms_in > 1e-6:
                gain = np.clip(rms_out / rms_in, 0.35, 2.83)
                in_band = in_band * gain

        # --- ADR-032 deviation: compute shift on fade-zone pair, apply to full in_band ---
        if phase_align and max_phase_shift_samples > 0:
            _, shift = _phase_align(
                out_band[..., -xf_samples:],
                in_band[..., :xf_samples],
                max_phase_shift_samples,
            )
            in_band = _apply_shift_full(in_band, shift)
        # --- end ADR-032 deviation ---

        pad_before = (max_xf_samples - xf_samples) // 2
        fade_out, fade_in = _equal_power_fade(xf_samples)

        env_out = np.zeros(max_xf_samples, dtype=np.float64)
        env_in = np.zeros(max_xf_samples, dtype=np.float64)
        env_out[:pad_before] = 1.0
        env_out[pad_before:pad_before + xf_samples] = fade_out
        env_in[pad_before + xf_samples:] = 1.0
        env_in[pad_before:pad_before + xf_samples] = fade_in

        overlap_result += out_band * env_out + in_band * env_in

    result[..., non_overlap_out:non_overlap_out + max_xf_samples] = overlap_result
    return result.astype(outgoing.dtype)


def _save_case(
    case_dir: Path,
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    result: np.ndarray,
    flags: dict,
    notes: dict,
    label: str,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    # f64 everywhere. 2D shape (nCh, nSamp).
    out64 = outgoing.astype(np.float64)
    in64 = incoming.astype(np.float64)
    if out64.ndim == 1:
        out64 = out64[np.newaxis, :]
    if in64.ndim == 1:
        in64 = in64[np.newaxis, :]
    if result.ndim == 1:
        result = result[np.newaxis, :]

    np.save(case_dir / "out.npy", out64)
    np.save(case_dir / "in.npy", in64)
    np.save(case_dir / "sr.npy", np.array([sr], dtype=np.int64))
    np.save(case_dir / "result.npy", result.astype(np.float64))

    with (case_dir / "flags.json").open("w") as f:
        json.dump({**flags, **notes,
                   "out_shape": list(out64.shape),
                   "in_shape":  list(in64.shape),
                   "result_shape": list(result.shape)}, f, indent=2)

    print(f"  {case_dir.name}: sr={sr} "
          f"out={tuple(out64.shape)} in={tuple(in64.shape)} "
          f"res={tuple(result.shape)} "
          f"{label}")


def _write_case(
    case_dir: Path,
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    flags: dict,
    notes: dict,
) -> None:
    """Adaptive-crossfade case (uses ADR-032 fixed variant)."""
    result = adaptive_crossfade_fixed(
        outgoing.astype(np.float64), incoming.astype(np.float64), sr,
        phase_align=flags["phase_align"],
        energy_compensate=flags["energy_compensate"],
        max_phase_shift_ms=flags.get("max_phase_shift_ms", 3.0),
    )
    label = (f"kind=adaptive PA={flags['phase_align']} "
             f"EC={flags['energy_compensate']}")
    _save_case(case_dir, outgoing, incoming, sr, result,
               {**flags, "kind": "adaptive"}, notes, label)


def _write_simple_case(
    case_dir: Path,
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    flags: dict,
    notes: dict,
) -> None:
    """Simple-crossfade case (imported unmodified from reference)."""
    result = simple_crossfade(
        outgoing.astype(np.float64), incoming.astype(np.float64), sr,
        crossfade_ms=flags.get("crossfade_ms", DEFAULT_CROSSFADE_MS),
        energy_compensate=flags["energy_compensate"],
    )
    label = (f"kind=simple cf_ms={flags.get('crossfade_ms', DEFAULT_CROSSFADE_MS)} "
             f"EC={flags['energy_compensate']}")
    _save_case(case_dir, outgoing, incoming, sr, result,
               {**flags, "kind": "simple"}, notes, label)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"dumping crossfade goldens under {OUT_ROOT}")

    # Real boundary waveforms from billie_jean phase-2 fixture (sr=22050).
    wf = np.load(
        ROOT / "references" / "golden" / "phase-2" / "dumps" /
        "billie_jean" / "boundary_waveforms.npy"
    ).astype(np.float64)

    # --- case_01: real music, production default path (PA=True, EC=True) --
    # Rows [0] -> outgoing, [1] -> incoming; duplicated to stereo for nCh>1 coverage.
    out_r = np.stack([wf[0], wf[0]])
    in_r = np.stack([wf[1], wf[1]])
    _write_case(
        OUT_ROOT / "case_01_billie_jean_prod_default",
        out_r, in_r, 22050,
        flags={"phase_align": True, "energy_compensate": True,
               "max_phase_shift_ms": 3.0},
        notes={
            "signal": "billie_jean/boundary_waveforms.npy rows [0] and [1], "
                      "stereo-duplicated",
            "why": "production default path — PA=True + EC=True — exercises "
                   "ADR-032 phase-align fix in mid + treble, bass unaffected "
                   "(xf==max_xf)",
        },
    )

    # --- case_02: phase_align=False control --
    _write_case(
        OUT_ROOT / "case_02_billie_jean_no_align",
        out_r, in_r, 22050,
        flags={"phase_align": False, "energy_compensate": True,
               "max_phase_shift_ms": 3.0},
        notes={
            "signal": "billie_jean rows [0]/[1] stereo, PA=False",
            "why": "no-align control — bit-exact with unmodified Python "
                   "adaptive_crossfade (no divergence on this branch)",
        },
    )

    # --- case_03: synthetic bass+treble phase-offset, PA=True, EC=False --
    sr_s = 44100
    n_s = int(0.5 * sr_s)
    t = np.arange(n_s, dtype=np.float64) / float(sr_s)
    out_sig = np.sin(2 * np.pi * 80 * t) + 0.5 * np.sin(2 * np.pi * 8000 * t)
    in_sig = np.sin(2 * np.pi * 80 * t + 0.3) + 0.5 * np.sin(2 * np.pi * 8000 * t + 1.0)
    out_s = np.stack([out_sig, out_sig])
    in_s = np.stack([in_sig, in_sig])
    _write_case(
        OUT_ROOT / "case_03_synthetic_bass_treble_pa_on",
        out_s, in_s, sr_s,
        flags={"phase_align": True, "energy_compensate": False,
               "max_phase_shift_ms": 3.0},
        notes={
            "signal": "synthetic stereo 80 Hz + 8 kHz sines with phase offsets",
            "why": "isolates PhaseAlign + envelope — cleanest view of ADR-032 fix "
                   "(post-fade treble recovered from ~0.007 to ~0.35)",
        },
    )

    # --- case_04: energy-compensate clamp path --
    # Incoming is attenuated by 20 dB → rms_out/rms_in ≈ 10 → clipped to 2.83.
    out_ec = np.stack([out_sig, out_sig])
    in_ec = np.stack([in_sig, in_sig]) * 0.1  # -20 dB
    _write_case(
        OUT_ROOT / "case_04_synthetic_ec_clamp",
        out_ec, in_ec, sr_s,
        flags={"phase_align": True, "energy_compensate": True,
               "max_phase_shift_ms": 3.0},
        notes={
            "signal": "case_03 with incoming × 0.1 (-20 dB)",
            "why": "exercises RMS clip(0.35, 2.83) saturation — gain per band "
                   "saturates at 2.83 (upper cap)",
        },
    )

    # --- case_05: simple_crossfade real music production default (SB-9) --
    # billie_jean boundary stereo @ sr=22050, crossfade_ms=75.0, EC=True.
    # n_overlap = int(75 * 22.05) = 1653; ramp_len = min(1653*4=6612, inLen=3417) = 3417
    # → ramp spans entire incoming (saturated by inLen cap).
    _write_simple_case(
        OUT_ROOT / "case_05_simple_billie_jean_prod",
        out_r, in_r, 22050,
        flags={"crossfade_ms": DEFAULT_CROSSFADE_MS, "energy_compensate": True},
        notes={
            "signal": "billie_jean boundary rows [0]/[1] stereo, cf=75.0 ms, EC=True",
            "why": "simple_crossfade production default on real music — exercises "
                   "RMS clamp on natural content + ramp saturated by inLen cap "
                   "(3417 < 4*n_overlap=6612)",
        },
    )

    # --- case_06: simple_crossfade -20 dB clamp + full ramp --
    # Longer incoming (0.5 s @ sr=44100 = 22050 samples) so inLen > 4*n_overlap.
    # n_overlap = int(75 * 44.1) = 3307; ramp_len = min(13228, 22050) = 13228.
    _write_simple_case(
        OUT_ROOT / "case_06_simple_ec_clamp",
        out_ec, in_ec, sr_s,
        flags={"crossfade_ms": DEFAULT_CROSSFADE_MS, "energy_compensate": True},
        notes={
            "signal": "case_04 inputs (synthetic 80 Hz + 8 kHz, incoming -20 dB), "
                      "simple path cf=75.0 ms EC=True",
            "why": "exercises RMS clamp saturation (gain=2.83) + ramp spanning "
                   "13228 samples (4*n_overlap), not capped by inLen",
        },
    )

    # --- case_07: simple_crossfade EC=False control --
    # Same synthetic inputs as case_03, EC=False → no RMS calc, no ramp,
    # pure equal-power cos/sin fade.
    _write_simple_case(
        OUT_ROOT / "case_07_simple_no_ec",
        out_s, in_s, sr_s,
        flags={"crossfade_ms": DEFAULT_CROSSFADE_MS, "energy_compensate": False},
        notes={
            "signal": "case_03 inputs (synthetic 80 Hz + 8 kHz phase-offset), "
                      "simple path cf=75.0 ms EC=False",
            "why": "control — no RMS compensation, no ramp, pure equal-power "
                   "cos/sin crossfade over 3307 samples (no dependency on ramp "
                   "arithmetic ordering)",
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
