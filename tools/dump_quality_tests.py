#!/usr/bin/env python3
"""
dump_quality_tests.py — Python reference dumps for the `reamix::remix::Quality`
C++ parity test. Bitwise target since quality.py is pure scalar arithmetic
with deterministic evaluation order (waveform, edge_splice, then 8
always-available signals in `always_pairs` order).

Emits under tests/parity/reference/data/quality/:

  quality_score_inputs.npy    (N, 10) f64  — columns 0..9 in Python signature
                                              order (waveform, successor,
                                              edge_splice, context, label,
                                              section, bar_aligned, energy,
                                              edge_energy, centroid).
  quality_score_flags.npy     (N, 2)  i64  — [waveform_missing,
                                              edge_splice_missing] booleans
                                              (stored as int64 for
                                              NpyIO.h compatibility).
  quality_score_expected.npy  (N,)    f64  — compute_quality_score(...)
                                              with None substituted for
                                              missing columns per flags.

  vocal_penalty_inputs.npy    (M, 4)  f64  — [va_source, va_dest,
                                              edge_va_end, edge_va_start].
  vocal_penalty_flags.npy     (M, 2)  i64  — [edge_end_missing,
                                              edge_start_missing].
  vocal_penalty_expected.npy  (M,)    f64

  onset_penalty_inputs.npy    (K,)    f64  — dest values (NaN where missing).
  onset_penalty_flags.npy     (K,)    i64  — dest_missing boolean.
  onset_penalty_expected.npy  (K,)    f64

N=200, M=100, K=50 + handcrafted edge cases prepended to each block:
  score: all-zero, all-one, all-half, each optional missing (both paths),
         both missing, clamp-negative, clamp-gt-one.
  vocal: all-zero, abrupt change, identical, edge-overrides-beat, threshold.
  onset: None, exactly threshold, zero, full sustain, mid-sustain.

Usage:
    python tools/dump_quality_tests.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import quality.py directly via importlib — the parent `remix/__init__.py`
# re-exports from `remix_tool.*` which only exists inside the production
# RemixTool install tree, not in our repo checkout.
import importlib.util  # noqa: E402
_QUALITY_PATH = REPO_ROOT / "references" / "python-source" / "remix" / "quality.py"
_spec = importlib.util.spec_from_file_location("_quality_ref", _QUALITY_PATH)
assert _spec and _spec.loader, f"failed to locate {_QUALITY_PATH}"
_quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_quality)

compute_quality_score = _quality.compute_quality_score
compute_vocal_penalty = _quality.compute_vocal_penalty
compute_onset_penalty = _quality.compute_onset_penalty

OUT_DIR = REPO_ROOT / "tests" / "parity" / "reference" / "data" / "quality"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(name: str, arr: np.ndarray) -> dict:
    path = OUT_DIR / f"{name}.npy"
    np.save(path, arr, allow_pickle=False)
    return {
        "file":   f"{name}.npy",
        "shape":  list(arr.shape),
        "dtype":  str(arr.dtype),
        "sha256": _sha256(path),
        "bytes":  path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------

def _dump_quality_score() -> dict:
    # Handcrafted edge cases (inputs, flags) ---------------------------
    edge_rows: list[tuple[np.ndarray, tuple[int, int]]] = []

    # Each row is 10 signals in Python signature order:
    # [waveform, successor, edge_splice, context, label, section,
    #  bar_aligned, energy, edge_energy, centroid]
    all_zero = np.zeros(10, dtype=np.float64)
    all_one  = np.ones(10, dtype=np.float64)
    all_half = np.full(10, 0.5, dtype=np.float64)

    edge_rows.append((all_zero,                    (0, 0)))  # score = 0
    edge_rows.append((all_one,                     (0, 0)))  # score = 1
    edge_rows.append((all_half,                    (0, 0)))  # score = 0.5
    edge_rows.append((all_half,                    (1, 0)))  # waveform missing
    edge_rows.append((all_half,                    (0, 1)))  # edge_splice missing
    edge_rows.append((all_half,                    (1, 1)))  # both missing

    # Clamp cases — put out-of-range in non-optional slots, so the
    # redistribution path is NOT taken and the clamp is exercised directly.
    neg_row = np.array(
        [0.5, -0.3, 0.5, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3],
        dtype=np.float64,
    )
    big_row = np.array(
        [0.5,  1.7, 0.5,  1.7,  1.7,  1.7,  1.7,  1.7,  1.7,  1.7],
        dtype=np.float64,
    )
    edge_rows.append((neg_row, (0, 0)))  # weighted_sum < 0 -> clamped to 0
    edge_rows.append((big_row, (0, 0)))  # weighted_sum > 1 -> clamped to 1

    # Redistribution correctness — inputs that make the redistributed math
    # differ visibly from the non-redistributed math.
    redist_row = np.array(
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    edge_rows.append((redist_row, (0, 0)))  # waveform=0 present
    edge_rows.append((redist_row, (1, 0)))  # waveform missing -> score=1.0

    # --- Random cases --------------------------------------------------
    rng = np.random.Generator(np.random.PCG64(42))
    N_RANDOM = 200
    random_inputs = rng.uniform(0.0, 1.0, size=(N_RANDOM, 10)).astype(np.float64)
    random_flags  = rng.integers(0, 2, size=(N_RANDOM, 2), dtype=np.int64)

    # Pack edge + random
    edge_inputs = np.stack([r[0] for r in edge_rows], axis=0)
    edge_flags  = np.array([r[1]  for r in edge_rows], dtype=np.int64)

    inputs = np.concatenate([edge_inputs, random_inputs], axis=0)
    flags  = np.concatenate([edge_flags,  random_flags],  axis=0)

    # --- Compute expected via Python reference -------------------------
    N = inputs.shape[0]
    expected = np.zeros(N, dtype=np.float64)
    for i in range(N):
        waveform    = None if flags[i, 0] else inputs[i, 0]
        edge_splice = None if flags[i, 1] else inputs[i, 2]
        expected[i] = compute_quality_score(
            waveform_sim      = waveform,
            successor_sim     = inputs[i, 1],
            edge_splice_sim   = edge_splice,
            context_sim       = inputs[i, 3],
            label_match       = inputs[i, 4],
            section_sim       = inputs[i, 5],
            bar_aligned       = inputs[i, 6],
            energy_match      = inputs[i, 7],
            edge_energy_match = inputs[i, 8],
            centroid_match    = inputs[i, 9],
        )

    return {
        "n_edge":  len(edge_rows),
        "n_random": N_RANDOM,
        "n_total": N,
        "inputs":   _save("quality_score_inputs",   inputs),
        "flags":    _save("quality_score_flags",    flags),
        "expected": _save("quality_score_expected", expected),
    }


# ---------------------------------------------------------------------------
# compute_vocal_penalty
# ---------------------------------------------------------------------------

def _dump_vocal_penalty() -> dict:
    edge_rows: list[tuple[np.ndarray, tuple[int, int]]] = []

    # [va_source, va_dest, edge_end, edge_start], flags = [edge_end_missing, edge_start_missing]
    edge_rows.append((np.array([0.0, 0.0, 0.0, 0.0], np.float64), (1, 1)))  # trivial
    edge_rows.append((np.array([1.0, 0.0, 0.0, 0.0], np.float64), (1, 1)))  # abrupt
    edge_rows.append((np.array([0.5, 0.5, 0.0, 0.0], np.float64), (1, 1)))  # identical
    edge_rows.append((np.array([0.6, 0.6, 0.6, 0.6], np.float64), (0, 0)))  # edge above thresh
    edge_rows.append((np.array([0.9, 0.9, 0.1, 0.1], np.float64), (0, 0)))  # edge overrides high beat
    edge_rows.append((np.array([0.40, 0.40, 0.0, 0.0], np.float64), (1, 1)))  # at threshold
    edge_rows.append((np.array([0.41, 0.41, 0.0, 0.0], np.float64), (1, 1)))  # just above threshold
    edge_rows.append((np.array([0.39, 0.39, 0.0, 0.0], np.float64), (1, 1)))  # just below threshold
    edge_rows.append((np.array([0.5, 0.5, 0.0, 0.0], np.float64), (1, 0)))  # partial edge (falls back)
    edge_rows.append((np.array([0.5, 0.5, 0.0, 0.0], np.float64), (0, 1)))  # partial edge (falls back)

    # Random
    rng = np.random.Generator(np.random.PCG64(7))
    M_RANDOM = 100
    random_inputs = rng.uniform(0.0, 1.0, size=(M_RANDOM, 4)).astype(np.float64)
    random_flags  = rng.integers(0, 2, size=(M_RANDOM, 2), dtype=np.int64)

    edge_inputs = np.stack([r[0] for r in edge_rows], axis=0)
    edge_flags  = np.array([r[1] for r in edge_rows], dtype=np.int64)

    inputs = np.concatenate([edge_inputs, random_inputs], axis=0)
    flags  = np.concatenate([edge_flags,  random_flags],  axis=0)

    M = inputs.shape[0]
    expected = np.zeros(M, dtype=np.float64)
    for i in range(M):
        edge_end   = None if flags[i, 0] else inputs[i, 2]
        edge_start = None if flags[i, 1] else inputs[i, 3]
        expected[i] = compute_vocal_penalty(
            va_source    = inputs[i, 0],
            va_dest      = inputs[i, 1],
            edge_va_end  = edge_end,
            edge_va_start = edge_start,
        )

    return {
        "n_edge":   len(edge_rows),
        "n_random": M_RANDOM,
        "n_total":  M,
        "inputs":   _save("vocal_penalty_inputs",   inputs),
        "flags":    _save("vocal_penalty_flags",    flags),
        "expected": _save("vocal_penalty_expected", expected),
    }


# ---------------------------------------------------------------------------
# compute_onset_penalty
# ---------------------------------------------------------------------------

def _dump_onset_penalty() -> dict:
    # [dest], flags = [missing]
    edge_pairs: list[tuple[float, int]] = [
        (0.0,  1),  # missing
        (0.25, 0),  # exactly threshold -> 0
        (0.0,  0),  # full sustain -> max penalty
        (0.125, 0), # mid-sustain
        (1.0,  0),  # strong onset -> 0
        (0.249, 0), # just below threshold
        (0.251, 0), # just above threshold
    ]
    rng = np.random.Generator(np.random.PCG64(11))
    K_RANDOM = 50
    random_inputs = rng.uniform(0.0, 1.0, size=K_RANDOM).astype(np.float64)
    random_flags  = rng.integers(0, 2, size=K_RANDOM, dtype=np.int64)

    edge_inputs = np.array([p[0] for p in edge_pairs], dtype=np.float64)
    edge_flags  = np.array([p[1] for p in edge_pairs], dtype=np.int64)

    inputs = np.concatenate([edge_inputs, random_inputs], axis=0)
    flags  = np.concatenate([edge_flags,  random_flags],  axis=0)

    K = inputs.shape[0]
    expected = np.zeros(K, dtype=np.float64)
    for i in range(K):
        dest = None if flags[i] else inputs[i]
        expected[i] = compute_onset_penalty(onset_dest=dest)

    return {
        "n_edge":   len(edge_pairs),
        "n_random": K_RANDOM,
        "n_total":  K,
        "inputs":   _save("onset_penalty_inputs",   inputs),
        "flags":    _save("onset_penalty_flags",    flags),
        "expected": _save("onset_penalty_expected", expected),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "pins":  {"numpy": np.__version__, "python": sys.version.split()[0]},
        "score": _dump_quality_score(),
        "vocal": _dump_vocal_penalty(),
        "onset": _dump_onset_penalty(),
    }

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[dump] wrote {OUT_DIR}/manifest.json")
    for section in ("score", "vocal", "onset"):
        s = manifest[section]
        print(f"  {section}: n_edge={s['n_edge']} n_random={s['n_random']} n_total={s['n_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
