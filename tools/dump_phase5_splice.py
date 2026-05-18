#!/usr/bin/env python3
"""Phase-5 parity dump tool — Splice goldens (session-34 + session-35 scope).

Session-34 (closed) covers 3 primitives from
references/python-source/remix/splice.py:
  - _window_onset_index        (L85-98)
  - _get_hanning               (L100-106) — trivial; exercised implicitly
  - _stereo_window_similarity  (L108-149)
  - _score_splice_pair         (L151-196)

Session-35 (this session) adds:
  - _find_onset_sample                     (L26-83, re-enabled per ADR-033)
  - _score_anchor_aligned_pair             (L198-245)
  - _search_anchor_transition_geometry     (L247-330)  ← ADR-033 close test
  - _transition_overlap_samples            (L332-355)
  - _refine_transition_splice              (L357-514)

ADR-033 context (findOnsetSample re-enable): session 34 bisected a
numpy-vs-C++ accumulator drift in np.convolve at K=32 smoothing kernel
(±1-4 sample argmax shift). The primitive-level gate was deferred; session
35 tests at composition level via _search_anchor_transition_geometry — if
the selected winner-anchor-beat-pair matches Python despite primitive drift,
ADR-033 closes positively. We re-dump findOnsetSample goldens here purely
for diagnostic / regression purposes (primitive-level ±N tolerance).

Per-case files (schema dispatched by `kind` in flags.json):

  kind="find_onset":
    input.npy      f64 1D      mono_energy array
    expected.npy   int64 1D    shape (1,) — result sample index
    flags.json     sr, beat_sample, lookback_ms (nullable), lookahead_ms

  kind="window_onset":
    input.npy      f64 2D      window (nCh, n)
    expected.npy   int64 1D    shape (1,)

  kind="similarity":
    a.npy          f64 2D      window_a (nCh, n)
    b.npy          f64 2D      window_b (nCh, n)
    expected.npy   f64 1D      shape (1,) — similarity

  kind="score_splice":
    a.npy          f64 2D
    b.npy          f64 2D
    expected.npy   f64 1D      shape (2,) — [score, similarity]
    flags.json     outgoing_shift, incoming_shift, max_shift_samples, ...

  kind="score_anchor":  (session 35)
    a.npy          f64 2D      outgoing_window (nCh, n)
    b.npy          f64 2D      incoming_window (nCh, n)
    expected.npy   f64 1D      shape (3,) — [quality, similarity01, local_similarity01]
    flags.json     sr, anchor_index, vocal_presence, anchor_local_window_ms

  kind="transition_overlap":  (session 35)
    (no .npy inputs — all scalars in flags.json)
    expected.npy   int64 1D    shape (1,) — overlap_samples
    flags.json     crossfade_samples, sr, vocal_presence_level,
                   preferred_overlap_sec, label_match, vocal_entry_support,
                   vocal_exit_support, vocal_activity_threshold,
                   vocal_crossfade_ms, vocal_same_label_crossfade_ms

  kind="search_anchor":  (session 35 — CRITICAL ADR-033 close test)
    audio.npy          f64 2D     (nCh, N_audio)
    mono_energy.npy    f64 1D     (N_audio,)
    beat_samples.npy   int64 1D   (N_beats,)
    expected.json      dict       emits Python returned dict's 17 render_* keys
                                  OR `{"selected": false}` when no candidate wins
    flags.json         sr, prev_beat, curr_beat, vocal_presence_level,
                       anchor_search_beats, anchor_search_max_extension_beats,
                       anchor_min_context_ms, anchor_local_window_ms,
                       onset_search_lookback_ms, onset_search_lookahead_ms,
                       transient_center_penalty_weight, transient_alignment_penalty_weight

  kind="refine":  (session 35 — 2-stage coarse→fine refinement)
    audio.npy          f64 2D     (nCh, N_audio)
    mono_energy.npy    f64 1D     (N_audio,)
    expected.json      dict       emits Python's 6 render_* keys OR `{"found": false}`
    flags.json         sr, crossfade_samples, successor_sample, target_sample,
                       beat_end, alignment_offset_sec,
                       overlap_samples (nullable), stereo_refine_max_shift_ms,
                       stereo_refine_coarse_step_samples, transient_* weights,
                       onset_search_lookback_ms, onset_search_lookahead_ms

Per session-34 precedent all cases use the UNMODIFIED reference function
(no monkey-patch; session-35 scope has zero identified reference bugs —
_refine's coarse→fine ordering, _search's 4-level loop, and the 4 other
methods are all clean ports).

Usage:
    python3 tools/dump_phase5_splice.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "references" / "golden" / "phase-5" / "splice"

_PY_SRC_SYMLINK = ROOT / "references" / "python-source"
_REMIX_TOOL_PKG_PARENT = _PY_SRC_SYMLINK.resolve().parent
if _REMIX_TOOL_PKG_PARENT.is_dir() and str(_REMIX_TOOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_REMIX_TOOL_PKG_PARENT))

from remix_tool.remix.splice import SpliceMixin  # noqa: E402
from remix_tool.config import DEFAULT_CONFIG  # noqa: E402


# --- Stub host for SpliceMixin methods -----------------------------------
#
# Session-35 adds 3 more attributes beyond session-34 to cover the composite
# methods: _audio, _n_beats, _beat_samples, _crossfade_samples. Explicit
# setters keep per-case construction trivial.
class _SpliceStub(SpliceMixin):
    def __init__(self,
                 sr: int = 44100,
                 audio: Optional[np.ndarray] = None,
                 mono_energy: Optional[np.ndarray] = None,
                 beat_samples: Optional[np.ndarray] = None,
                 crossfade_samples: int = 0) -> None:
        self._sr = sr
        self._audio = audio if audio is not None else np.zeros((1, 0), dtype=np.float64)
        if self._audio.ndim == 1:
            self._audio = self._audio[np.newaxis, :]
        self._mono_energy = (
            mono_energy.astype(np.float64)
            if mono_energy is not None else
            np.sqrt(np.mean(self._audio.astype(np.float64) ** 2, axis=0))
        )
        self._beat_samples = (
            beat_samples.astype(np.int64)
            if beat_samples is not None else
            np.zeros(0, dtype=np.int64)
        )
        self._n_beats = int(len(self._beat_samples))
        self._crossfade_samples = int(crossfade_samples)
        self._default_onset_cache: Dict[int, int] = {}


# --- Per-case writers ----------------------------------------------------

def _save_scalar_int64(path: Path, value: int) -> None:
    np.save(path, np.array([int(value)], dtype=np.int64))


def _save_scalar_f64(path: Path, value: float) -> None:
    np.save(path, np.array([float(value)], dtype=np.float64))


def _save_f64_1d(path: Path, arr: np.ndarray) -> None:
    np.save(path, np.asarray(arr, dtype=np.float64))


def _save_f64_2d(path: Path, arr: np.ndarray) -> None:
    assert arr.ndim == 2, f"expected 2D, got {arr.ndim} at {path}"
    np.save(path, np.asarray(arr, dtype=np.float64))


def _save_i64_1d(path: Path, arr: np.ndarray) -> None:
    np.save(path, np.asarray(arr, dtype=np.int64))


_DEFAULT_WEIGHTS = {
    "onset_search_lookback_ms":            float(DEFAULT_CONFIG.remix.onset_search_lookback_ms),
    "onset_search_lookahead_ms":           float(DEFAULT_CONFIG.remix.onset_search_lookahead_ms),
    "transient_center_penalty_weight":     float(DEFAULT_CONFIG.remix.transient_center_penalty_weight),
    "transient_alignment_penalty_weight":  float(DEFAULT_CONFIG.remix.transient_alignment_penalty_weight),
    "anchor_local_window_ms":              float(DEFAULT_CONFIG.remix.anchor_local_window_ms),
    "anchor_search_beats":                 int(DEFAULT_CONFIG.remix.anchor_search_beats),
    "anchor_search_max_extension_beats":   int(DEFAULT_CONFIG.remix.anchor_search_max_extension_beats),
    "anchor_min_context_ms":               float(DEFAULT_CONFIG.remix.anchor_min_context_ms),
    "vocal_activity_threshold":            float(DEFAULT_CONFIG.remix.vocal_activity_threshold),
    "vocal_crossfade_ms":                  float(DEFAULT_CONFIG.remix.vocal_crossfade_ms),
    "vocal_same_label_crossfade_ms":       float(DEFAULT_CONFIG.remix.vocal_same_label_crossfade_ms),
    "stereo_refine_max_shift_ms":          float(DEFAULT_CONFIG.remix.stereo_refine_max_shift_ms),
    "stereo_refine_coarse_step_samples":   int(DEFAULT_CONFIG.remix.stereo_refine_coarse_step_samples),
}


def _write_flags(case_dir: Path, flags: Dict[str, Any]) -> None:
    merged = {**_DEFAULT_WEIGHTS, **flags}
    with (case_dir / "flags.json").open("w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)


def _write_expected_json(case_dir: Path, payload: Dict[str, Any]) -> None:
    with (case_dir / "expected.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# --- Session-34 writers (unchanged) --------------------------------------

def _write_find_onset_case(case_dir: Path, sr: int, mono_energy: np.ndarray,
                           beat_sample: int, lookback_ms: Optional[float],
                           lookahead_ms: Optional[float], notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub(sr=sr, mono_energy=mono_energy.astype(np.float64))
    result = stub._find_onset_sample(int(beat_sample), lookback_ms, lookahead_ms)
    _save_f64_1d(case_dir / "input.npy", mono_energy)
    _save_scalar_int64(case_dir / "expected.npy", int(result))
    _write_flags(case_dir, {
        "kind": "find_onset",
        "sr": int(sr),
        "beat_sample": int(beat_sample),
        "lookback_ms": (None if lookback_ms is None else float(lookback_ms)),
        "lookahead_ms": (None if lookahead_ms is None else float(lookahead_ms)),
        "n_mono_energy": int(mono_energy.shape[0]),
        **notes,
    })
    lb_s = "None" if lookback_ms is None else f"{lookback_ms:.1f}"
    la_s = "None" if lookahead_ms is None else f"{lookahead_ms:.1f}"
    print(f"  {case_dir.name}: find_onset "
          f"sr={sr} n={mono_energy.shape[0]:5d} beat={beat_sample:5d} "
          f"lb={lb_s:>6s} la={la_s:>6s} -> result={result}")


def _write_window_onset_case(case_dir: Path, window: np.ndarray,
                             notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub()
    result = stub._window_onset_index(window.astype(np.float64))
    _save_f64_2d(case_dir / "input.npy", window)
    _save_scalar_int64(case_dir / "expected.npy", int(result))
    _write_flags(case_dir, {
        "kind": "window_onset",
        "n_channels": int(window.shape[0]),
        "n_samples": int(window.shape[-1]),
        **notes,
    })
    print(f"  {case_dir.name}: window_onset "
          f"shape=({window.shape[0]},{window.shape[1]}) -> idx={result}")


def _write_similarity_case(case_dir: Path, window_a: np.ndarray, window_b: np.ndarray,
                           notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub()
    sim = stub._stereo_window_similarity(
        window_a.astype(np.float64), window_b.astype(np.float64)
    )
    _save_f64_2d(case_dir / "a.npy", window_a)
    _save_f64_2d(case_dir / "b.npy", window_b)
    _save_scalar_f64(case_dir / "expected.npy", float(sim))
    _write_flags(case_dir, {
        "kind": "similarity",
        "n_channels": int(window_a.shape[0]),
        "n_samples": int(window_a.shape[-1]),
        **notes,
    })
    print(f"  {case_dir.name}: similarity "
          f"shape=({window_a.shape[0]},{window_a.shape[1]}) -> sim={sim:+.6e}")


def _write_score_splice_case(case_dir: Path, outgoing: np.ndarray, incoming: np.ndarray,
                             outgoing_shift: int, incoming_shift: int,
                             max_shift_samples: int, notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub()
    score, sim = stub._score_splice_pair(
        outgoing.astype(np.float64), incoming.astype(np.float64),
        int(outgoing_shift), int(incoming_shift), int(max_shift_samples),
    )
    _save_f64_2d(case_dir / "a.npy", outgoing)
    _save_f64_2d(case_dir / "b.npy", incoming)
    np.save(case_dir / "expected.npy",
            np.array([float(score), float(sim)], dtype=np.float64))
    _write_flags(case_dir, {
        "kind": "score_splice",
        "outgoing_shift": int(outgoing_shift),
        "incoming_shift": int(incoming_shift),
        "max_shift_samples": int(max_shift_samples),
        "n_channels": int(outgoing.shape[0]),
        "n_samples": int(outgoing.shape[-1]),
        **notes,
    })
    print(f"  {case_dir.name}: score_splice "
          f"shape=({outgoing.shape[0]},{outgoing.shape[1]}) "
          f"shifts=({outgoing_shift:+d},{incoming_shift:+d}) max={max_shift_samples} "
          f"-> score={score:+.6e} sim={sim:+.6e}")


# --- Session-35 writers ---------------------------------------------------

def _write_score_anchor_case(case_dir: Path, outgoing: np.ndarray, incoming: np.ndarray,
                             anchor_index: int, vocal_presence: float, sr: int,
                             notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub(sr=sr)
    quality, sim01, local01 = stub._score_anchor_aligned_pair(
        outgoing.astype(np.float64), incoming.astype(np.float64),
        int(anchor_index), float(vocal_presence),
    )
    _save_f64_2d(case_dir / "a.npy", outgoing)
    _save_f64_2d(case_dir / "b.npy", incoming)
    np.save(case_dir / "expected.npy",
            np.array([float(quality), float(sim01), float(local01)], dtype=np.float64))
    _write_flags(case_dir, {
        "kind": "score_anchor",
        "sr": int(sr),
        "anchor_index": int(anchor_index),
        "vocal_presence": float(vocal_presence),
        "n_channels": int(outgoing.shape[0]),
        "n_samples": int(outgoing.shape[-1]),
        **notes,
    })
    print(f"  {case_dir.name}: score_anchor "
          f"shape=({outgoing.shape[0]},{outgoing.shape[1]}) "
          f"anchor={anchor_index} vp={vocal_presence:.2f} "
          f"-> q={quality:.6f} s01={sim01:.6f} l01={local01:.6f}")


def _write_transition_overlap_case(case_dir: Path, crossfade_samples: int, sr: int,
                                   vocal_presence_level: float,
                                   preferred_overlap_sec: float,
                                   label_match: float,
                                   vocal_entry_support: float,
                                   vocal_exit_support: float,
                                   notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub(sr=sr, crossfade_samples=crossfade_samples)
    meta = {
        "vocal_presence_level":  float(vocal_presence_level),
        "preferred_overlap_sec": float(preferred_overlap_sec),
        "label_match":           float(label_match),
        "vocal_entry_support":   float(vocal_entry_support),
        "vocal_exit_support":    float(vocal_exit_support),
    }
    result = stub._transition_overlap_samples(meta)
    _save_scalar_int64(case_dir / "expected.npy", int(result))
    _write_flags(case_dir, {
        "kind": "transition_overlap",
        "sr": int(sr),
        "crossfade_samples": int(crossfade_samples),
        **meta,
        **notes,
    })
    print(f"  {case_dir.name}: transition_overlap "
          f"cf={crossfade_samples:6d} vp={vocal_presence_level:.2f} "
          f"lm={label_match:.2f} entry={vocal_entry_support:.2f} exit={vocal_exit_support:.2f} "
          f"-> {result}")


def _write_search_anchor_case(case_dir: Path,
                              audio_2d: np.ndarray,
                              mono_energy: np.ndarray,
                              beat_samples: np.ndarray,
                              sr: int,
                              prev_beat: int, curr_beat: int,
                              vocal_presence_level: float,
                              notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub(sr=sr, audio=audio_2d, mono_energy=mono_energy,
                       beat_samples=beat_samples)
    meta = {"vocal_presence_level": float(vocal_presence_level)}
    result = stub._search_anchor_transition_geometry(
        int(prev_beat), int(curr_beat), meta,
    )
    _save_f64_2d(case_dir / "audio.npy", audio_2d.astype(np.float64))
    _save_f64_1d(case_dir / "mono_energy.npy", mono_energy.astype(np.float64))
    _save_i64_1d(case_dir / "beat_samples.npy", beat_samples.astype(np.int64))
    # Expected as dict. Empty dict = "not selected" (Python returned {}).
    if not result:
        _write_expected_json(case_dir, {"selected": False})
    else:
        payload = {"selected": True, **{k: float(v) for k, v in result.items()}}
        _write_expected_json(case_dir, payload)
    _write_flags(case_dir, {
        "kind": "search_anchor",
        "sr": int(sr),
        "prev_beat": int(prev_beat),
        "curr_beat": int(curr_beat),
        "n_channels": int(audio_2d.shape[0]),
        "n_samples": int(audio_2d.shape[-1]),
        "n_beats": int(beat_samples.shape[0]),
        **meta,
        **notes,
    })
    sel = "SELECTED" if result else "no-cand"
    if result:
        print(f"  {case_dir.name}: search_anchor prev={prev_beat} curr={curr_beat} "
              f"vp={vocal_presence_level:.2f} -> {sel} out_beat={int(result['render_anchor_out_beat'])} "
              f"in_beat={int(result['render_anchor_in_beat'])} "
              f"score={result['render_anchor_splice_score']:.6f}")
    else:
        print(f"  {case_dir.name}: search_anchor prev={prev_beat} curr={curr_beat} "
              f"vp={vocal_presence_level:.2f} -> {sel} (best_score unchanged from -1.0)")


def _write_refine_case(case_dir: Path,
                       audio_2d: np.ndarray,
                       mono_energy: np.ndarray,
                       sr: int,
                       crossfade_samples: int,
                       successor_sample: int,
                       target_sample: int,
                       beat_end: int,
                       alignment_offset_sec: float,
                       overlap_samples: Optional[int],
                       notes: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stub = _SpliceStub(sr=sr, audio=audio_2d, mono_energy=mono_energy,
                       crossfade_samples=crossfade_samples)
    meta = {"alignment_offset_sec": float(alignment_offset_sec)}
    result = stub._refine_transition_splice(
        int(successor_sample), int(target_sample), int(beat_end),
        meta, overlap_samples,
    )
    _save_f64_2d(case_dir / "audio.npy", audio_2d.astype(np.float64))
    _save_f64_1d(case_dir / "mono_energy.npy", mono_energy.astype(np.float64))
    if not result:
        _write_expected_json(case_dir, {"found": False})
    else:
        payload = {"found": True, **{k: float(v) for k, v in result.items()}}
        _write_expected_json(case_dir, payload)
    _write_flags(case_dir, {
        "kind": "refine",
        "sr": int(sr),
        "crossfade_samples": int(crossfade_samples),
        "successor_sample": int(successor_sample),
        "target_sample": int(target_sample),
        "beat_end": int(beat_end),
        "alignment_offset_sec": float(alignment_offset_sec),
        "overlap_samples": (None if overlap_samples is None else int(overlap_samples)),
        "n_channels": int(audio_2d.shape[0]),
        "n_samples": int(audio_2d.shape[-1]),
        **notes,
    })
    if result:
        print(f"  {case_dir.name}: refine succ={successor_sample} tgt={target_sample} "
              f"end={beat_end} -> FOUND out_cut={int(result['render_outgoing_cut_sample'])} "
              f"in_start={int(result['render_incoming_cut_sample'])} "
              f"score={result['render_splice_score']:.6f}")
    else:
        print(f"  {case_dir.name}: refine succ={successor_sample} tgt={target_sample} "
              f"end={beat_end} -> NOT FOUND")


# --- Synthetic signal helpers --------------------------------------------

def _transient_mono_energy(n: int, sr: int, transient_sample: int,
                           rms_floor: float = 0.01, rms_peak: float = 0.9,
                           decay_samples: int = 4000) -> np.ndarray:
    """Synthetic mono-energy array with a single clear transient."""
    e = np.full(n, rms_floor, dtype=np.float64)
    if 0 <= transient_sample < n:
        rise_len = min(100, n - transient_sample)
        e[transient_sample:transient_sample + rise_len] = np.linspace(
            rms_floor, rms_peak, rise_len
        )
        tail_start = transient_sample + rise_len
        tail_len = min(decay_samples, n - tail_start)
        if tail_len > 0:
            decay = rms_peak * np.exp(-np.arange(tail_len, dtype=np.float64) /
                                       (decay_samples / 4.0))
            e[tail_start:tail_start + tail_len] = np.maximum(decay, rms_floor)
    return e


def _sinusoid(n: int, f_hz: float, sr: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / float(sr)
    return np.sin(2.0 * np.pi * f_hz * t + phase).astype(np.float64)


def _load_billie_boundary_pair() -> tuple[np.ndarray, np.ndarray]:
    wf = np.load(
        ROOT / "references" / "golden" / "phase-2" / "dumps" /
        "billie_jean" / "boundary_waveforms.npy"
    ).astype(np.float64)
    out_stereo = np.stack([wf[0], wf[0]])
    in_stereo = np.stack([wf[1], wf[1]])
    return out_stereo, in_stereo


def _load_billie_track() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load full billie_jean track: (audio_2d, mono_energy, beat_samples, sr).

    Source audio is mono f32 @ sr=22050 (1323000 samples, ~60 s). Reshape to
    (1, N) channel-major for RemixRenderer-compatible shape. Duplicate to
    stereo (2, N) for composite tests that exercise both channels. For this
    dump tool session-35 we use STEREO (nCh=2) — matches production where
    librosa.load(mono=False) would produce stereo for stereo source files.
    """
    y_mono = np.load(
        ROOT / "references" / "golden" / "phase-2" / "dumps" /
        "billie_jean" / "y_audio.npy"
    ).astype(np.float64)
    # Duplicate to stereo with tiny channel imbalance for NCC sensitivity.
    y_stereo = np.stack([y_mono, 0.92 * y_mono])
    mono_energy = np.sqrt(np.mean(y_stereo ** 2, axis=0))
    bt = np.load(
        ROOT / "references" / "golden" / "phase-2" / "dumps" /
        "billie_jean" / "beat_times.npy"
    ).astype(np.float64)
    sr = 22050
    beat_samples = (bt * sr).astype(np.int64)
    return y_stereo, mono_energy, beat_samples, sr


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"dumping splice goldens under {OUT_ROOT}")

    sr = 44100

    # =====================================================================
    # _find_onset_sample cases — RE-ENABLED session 35 per ADR-033
    # =====================================================================

    # case_01: synthetic mono_energy with a clear transient at 5000 samples.
    # beat_sample=4800 → lookback/lookahead defaults → onset captured ~5000.
    mono_n = 20000
    mono_e = _transient_mono_energy(mono_n, sr, transient_sample=5000)
    _write_find_onset_case(
        OUT_ROOT / "case_01_find_onset_transient",
        sr=sr, mono_energy=mono_e,
        beat_sample=4800, lookback_ms=None, lookahead_ms=None,
        notes={
            "signal": "synthetic mono_energy with RMS transient at sample 5000",
            "why": "default path — asymmetric lookback=70ms + lookahead=18ms "
                   "around beat=4800 → chunk envelope + diff + smooth + argmax",
        },
    )

    # case_02: short-chunk early return (n<128). Force by using a tiny
    # mono_energy array.
    mono_short = np.linspace(0.1, 0.5, 50, dtype=np.float64)
    _write_find_onset_case(
        OUT_ROOT / "case_02_find_onset_shortchunk",
        sr=sr, mono_energy=mono_short,
        beat_sample=25, lookback_ms=None, lookahead_ms=None,
        notes={
            "signal": "linear ramp mono_energy, length=50 (< 128 samples)",
            "why": "len(chunk)<128 early-return → result = beat_sample (25)",
        },
    )

    # case_03: explicit ms args (non-default cache-bypass path).
    _write_find_onset_case(
        OUT_ROOT / "case_03_find_onset_explicit_ms",
        sr=sr, mono_energy=mono_e,
        beat_sample=6000, lookback_ms=50.0, lookahead_ms=30.0,
        notes={
            "signal": "same mono_energy as case_01 with different beat_sample",
            "why": "explicit ms args → cache is bypassed; different lookback/"
                   "lookahead produces different chunk range",
        },
    )

    # =====================================================================
    # _window_onset_index cases (unchanged from session 34)
    # =====================================================================

    n_win = 2048
    base = _sinusoid(n_win, 220.0, sr) * 0.1
    env_boost = np.zeros_like(base)
    boost_start = 1024
    boost_len = 100
    env_boost[boost_start:boost_start + boost_len] = np.linspace(0.0, 0.7, boost_len)
    decay_tail = np.exp(-np.arange(n_win - boost_start - boost_len) / 300.0) * 0.7
    env_boost[boost_start + boost_len:] = decay_tail
    signal = (base + env_boost) * np.sign(np.sin(2.0 * np.pi * 880.0 * np.arange(n_win) / sr))
    window_a = np.stack([signal, 0.9 * signal])
    _write_window_onset_case(
        OUT_ROOT / "case_04_window_onset_stereo_mid",
        window=window_a,
        notes={
            "signal": "stereo carrier 220/880 Hz with transient ramp at sample 1024",
            "why": "default path — RMS-mono → box-convolve envelope → diff-prepend → "
                   "max(diff, 0) → argmax near sample 1024",
        },
    )

    window_short = np.zeros((2, 10), dtype=np.float64)
    window_short[0, 5] = 1.0
    _write_window_onset_case(
        OUT_ROOT / "case_05_window_onset_short",
        window=window_short,
        notes={
            "signal": "stereo zeros with single spike (shape (2,10))",
            "why": "n<16 early-return branch → result = max(0, n//2) = 5",
        },
    )

    # =====================================================================
    # _stereo_window_similarity cases (unchanged from session 34)
    # =====================================================================

    ident = _sinusoid(1024, 440.0, sr)
    window_ident = np.stack([ident, 0.7 * ident + 0.01 * _sinusoid(1024, 1320.0, sr)])
    _write_similarity_case(
        OUT_ROOT / "case_06_similarity_identical",
        window_a=window_ident, window_b=window_ident.copy(),
        notes={"signal": "identical stereo windows", "why": "sim clipped to +1.0"},
    )

    bj_out, bj_in = _load_billie_boundary_pair()
    _write_similarity_case(
        OUT_ROOT / "case_07_similarity_billie",
        window_a=bj_out, window_b=bj_in,
        notes={"signal": "billie_jean boundary rows [0]/[1]",
               "why": "real-music stereo similarity"},
    )

    anti = _sinusoid(1024, 440.0, sr)
    window_pos = np.stack([anti, 0.7 * anti])
    window_neg = np.stack([-anti, -0.7 * anti])
    _write_similarity_case(
        OUT_ROOT / "case_08_similarity_anticorrelated",
        window_a=window_pos, window_b=window_neg,
        notes={"signal": "stereo sine vs stereo -sine",
               "why": "anti-correlation → sim clipped to -1.0"},
    )

    # =====================================================================
    # _score_splice_pair cases (unchanged from session 34)
    # =====================================================================

    _write_score_splice_case(
        OUT_ROOT / "case_09_score_splice_billie",
        outgoing=bj_out, incoming=bj_in,
        outgoing_shift=17, incoming_shift=-5, max_shift_samples=132,
        notes={"signal": "billie_jean real boundary",
               "why": "composite score, all 5 penalty terms"},
    )
    _write_score_splice_case(
        OUT_ROOT / "case_10_score_splice_reject",
        outgoing=window_pos, incoming=window_neg,
        outgoing_shift=0, incoming_shift=0, max_shift_samples=132,
        notes={"signal": "anti-correlated",
               "why": "similarity ≤ -0.99 reject branch"},
    )
    _write_score_splice_case(
        OUT_ROOT / "case_11_score_splice_no_shift",
        outgoing=bj_out, incoming=bj_in,
        outgoing_shift=0, incoming_shift=0, max_shift_samples=0,
        notes={"signal": "billie_jean real boundary",
               "why": "max_shift=0 → position_penalty zeroed"},
    )

    # =====================================================================
    # _score_anchor_aligned_pair cases (session 35 NEW)
    # =====================================================================

    # case_12: BJ real stereo + anchor at mid-window + vocal_presence=0.5
    # Exercises composite: similarity + local_similarity (non-trivial local
    # window centered on anchor) + vocal-scaled local_weight + energy + edge.
    bj_out_f, bj_in_f = _load_billie_boundary_pair()
    _write_score_anchor_case(
        OUT_ROOT / "case_12_score_anchor_billie_mid",
        outgoing=bj_out_f, incoming=bj_in_f,
        anchor_index=bj_out_f.shape[-1] // 2,
        vocal_presence=0.5, sr=22050,
        notes={"signal": "billie_jean stereo boundary",
               "why": "composite local + static similarity with local weight 0.525"},
    )

    # case_13: same windows but anchor near the EDGE to exercise edge_penalty.
    _write_score_anchor_case(
        OUT_ROOT / "case_13_score_anchor_billie_edge",
        outgoing=bj_out_f, incoming=bj_in_f,
        anchor_index=max(8, int(bj_out_f.shape[-1] * 0.05)),
        vocal_presence=0.2, sr=22050,
        notes={"signal": "billie_jean stereo boundary",
               "why": "anchor at ~5% from edge → edge_penalty non-zero"},
    )

    # case_14: anti-correlated → early return (0, 0, 0).
    _write_score_anchor_case(
        OUT_ROOT / "case_14_score_anchor_reject",
        outgoing=window_pos, incoming=window_neg,
        anchor_index=512, vocal_presence=0.5, sr=sr,
        notes={"signal": "anti-correlated",
               "why": "similarity ≤ -0.99 reject branch → (0,0,0)"},
    )

    # case_15: high vocal_presence → local_weight hits the 0.70 cap.
    _write_score_anchor_case(
        OUT_ROOT / "case_15_score_anchor_high_vocal",
        outgoing=bj_out_f, incoming=bj_in_f,
        anchor_index=bj_out_f.shape[-1] // 2,
        vocal_presence=2.0,  # clipped to 1.0; local_weight = min(0.70, 0.65) = 0.65
        sr=22050,
        notes={"signal": "billie_jean stereo boundary",
               "why": "vocal_presence > 1.0 → clip(_, 0, 1)=1.0 → local_weight=0.65"},
    )

    # =====================================================================
    # _transition_overlap_samples cases (session 35 NEW)
    # =====================================================================

    # Branch matrix: vocal below threshold / label_match<0.5 / support<0.85 /
    # else branch. Also exercise preferred_overlap_sec override.
    cf75 = int(round(75.0 * 22050 / 1000))  # 1653 samples default

    # case_16: vocal < 0.28 → return baseOverlap (possibly boosted by preferred).
    _write_transition_overlap_case(
        OUT_ROOT / "case_16_overlap_low_vocal",
        crossfade_samples=cf75, sr=22050,
        vocal_presence_level=0.1,
        preferred_overlap_sec=0.0, label_match=1.0,
        vocal_entry_support=1.0, vocal_exit_support=1.0,
        notes={"why": "vocal < vocalActivityThreshold → baseOverlap branch"},
    )

    # case_17: vocal high + label_match < 0.5 → vocal_crossfade_ms (300ms)
    _write_transition_overlap_case(
        OUT_ROOT / "case_17_overlap_label_change",
        crossfade_samples=cf75, sr=22050,
        vocal_presence_level=0.6,
        preferred_overlap_sec=0.0, label_match=0.3,
        vocal_entry_support=1.0, vocal_exit_support=1.0,
        notes={"why": "vocal above + label_match<0.5 → 300ms vocal_crossfade"},
    )

    # case_18: vocal high + same label + entry_support<0.85 → 180ms.
    _write_transition_overlap_case(
        OUT_ROOT / "case_18_overlap_same_label_weak_entry",
        crossfade_samples=cf75, sr=22050,
        vocal_presence_level=0.6,
        preferred_overlap_sec=0.0, label_match=1.0,
        vocal_entry_support=0.4, vocal_exit_support=1.0,
        notes={"why": "same label + entry<0.85 → 180ms vocal_same_label_crossfade"},
    )

    # case_19: vocal high + same label + all supports ≥ 0.85 → baseOverlap
    _write_transition_overlap_case(
        OUT_ROOT / "case_19_overlap_strong_support",
        crossfade_samples=cf75, sr=22050,
        vocal_presence_level=0.6,
        preferred_overlap_sec=0.0, label_match=1.0,
        vocal_entry_support=0.95, vocal_exit_support=0.95,
        notes={"why": "all supports ≥ 0.85 → baseOverlap branch"},
    )

    # case_20: preferred_overlap_sec override raises baseOverlap above default.
    _write_transition_overlap_case(
        OUT_ROOT / "case_20_overlap_preferred_override",
        crossfade_samples=cf75, sr=22050,
        vocal_presence_level=0.1,
        preferred_overlap_sec=0.2,  # 0.2 * 22050 = 4410 > cf75=1653
        label_match=1.0,
        vocal_entry_support=1.0, vocal_exit_support=1.0,
        notes={"why": "preferred_overlap_sec > 0 boosts baseOverlap, "
               "vocal<threshold so return baseOverlap"},
    )

    # =====================================================================
    # _search_anchor_transition_geometry cases (session 35 — CRITICAL)
    # =====================================================================

    bj_audio_2d, bj_mono_e, bj_beat_samples, bj_sr = _load_billie_track()

    # case_21: mid-track pair — realistic distance between prev and curr beats.
    # Pick two beats far apart so the 4-level search has room.
    # Beat 30 (~13.9 s) to beat 60 (~27.8 s).
    _write_search_anchor_case(
        OUT_ROOT / "case_21_search_anchor_bj_mid",
        audio_2d=bj_audio_2d, mono_energy=bj_mono_e,
        beat_samples=bj_beat_samples, sr=bj_sr,
        prev_beat=30, curr_beat=60,
        vocal_presence_level=0.3,
        notes={"signal": "billie_jean mid-track beat pair",
               "why": "4-level anchor search with onset alignment at both ends"},
    )

    # case_22: vocal-heavy assumption (vp=0.9) — same beats, different vocal weight.
    _write_search_anchor_case(
        OUT_ROOT / "case_22_search_anchor_bj_vocal",
        audio_2d=bj_audio_2d, mono_energy=bj_mono_e,
        beat_samples=bj_beat_samples, sr=bj_sr,
        prev_beat=40, curr_beat=80,
        vocal_presence_level=0.9,
        notes={"signal": "billie_jean mid-track beat pair",
               "why": "high vocal_presence → local_weight hits cap → "
               "local_similarity dominates quality"},
    )

    # case_23: corner — prev_beat=0 forces out_anchor_lo=0, limits search range.
    # curr_beat near end so in_anchor has minimal room.
    _write_search_anchor_case(
        OUT_ROOT / "case_23_search_anchor_bj_edge",
        audio_2d=bj_audio_2d, mono_energy=bj_mono_e,
        beat_samples=bj_beat_samples, sr=bj_sr,
        prev_beat=1, curr_beat=125,
        vocal_presence_level=0.0,
        notes={"signal": "billie_jean near-edge beat pair",
               "why": "prev near 0, curr near n_beats-1 → ranges clipped"},
    )

    # =====================================================================
    # _refine_transition_splice cases (session 35 NEW)
    # =====================================================================

    cf_bj = int(75.0 * bj_sr / 1000)  # 1653 samples

    # case_24: middle-of-track refine — plausible successor/target/beat_end.
    # Successor = beat[30] ≈ 13.9s; target = beat[60]; beat_end = beat[61].
    _write_refine_case(
        OUT_ROOT / "case_24_refine_bj_mid",
        audio_2d=bj_audio_2d, mono_energy=bj_mono_e,
        sr=bj_sr, crossfade_samples=cf_bj,
        successor_sample=int(bj_beat_samples[30]),
        target_sample=int(bj_beat_samples[60]),
        beat_end=int(bj_beat_samples[61]),
        alignment_offset_sec=0.0,
        overlap_samples=None,  # None → use _crossfade_samples
        notes={"signal": "billie_jean mid-track refine",
               "why": "2-stage coarse→fine grid search with default overlap"},
    )

    # case_25: explicit overlap_samples argument (overrides ctx default).
    _write_refine_case(
        OUT_ROOT / "case_25_refine_bj_explicit_overlap",
        audio_2d=bj_audio_2d, mono_energy=bj_mono_e,
        sr=bj_sr, crossfade_samples=cf_bj,
        successor_sample=int(bj_beat_samples[40]),
        target_sample=int(bj_beat_samples[80]),
        beat_end=int(bj_beat_samples[81]),
        alignment_offset_sec=0.0,
        overlap_samples=2200,  # explicit override, different from default cf_bj
        notes={"signal": "billie_jean mid-track refine",
               "why": "explicit overlap_samples overrides ctx.crossfade_samples"},
    )

    # case_26: with non-zero alignment_offset_sec (transient_meta field).
    _write_refine_case(
        OUT_ROOT / "case_26_refine_bj_offset",
        audio_2d=bj_audio_2d, mono_energy=bj_mono_e,
        sr=bj_sr, crossfade_samples=cf_bj,
        successor_sample=int(bj_beat_samples[50]),
        target_sample=int(bj_beat_samples[100]),
        beat_end=int(bj_beat_samples[101]),
        alignment_offset_sec=0.005,  # 5 ms offset
        overlap_samples=None,
        notes={"signal": "billie_jean with transient_meta offset",
               "why": "non-zero alignment_offset_sec shifts aligned_target_sample"},
    )

    print(f"done. goldens at {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
