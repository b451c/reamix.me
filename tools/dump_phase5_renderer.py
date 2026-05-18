#!/usr/bin/env python3
"""Phase-5 renderer parity dump tool — full 48-case corpus.

Per ADR-034 + phase-5 spec L25, the phase-5 close gate is the FULL
16-track × 3-ratio = 48-case corpus. Iterates TRACK_LIST (from
dump_phase4_tests.py) × {0.3, 0.5, 0.7} — all shortening targets.

Session-36 smoke seed (billie_jean r=0.5, alice_in_chains_nutshell r=0.3,
meshuggah_bleed r=0.7) is a subset of this corpus and was dumped first to
de-risk; remaining 45 cases extend the gate to the ADR-034 mandate.

Pipeline per case:
  1. Load 60-180 s clip audio + beats_cpp grid + features from
     references/golden/phase-2/dumps/<track>/.
  2. Build RepetitionMap + TransitionCost + CleanOptimizer.remix(target) →
     RemixPath (mirrors tools/dump_phase4_tests.py:dump_optimizer_e2e_smoke).
  3. Monkey-patch `librosa.load` so RemixRenderer.__init__ skips disk I/O
     and reuses the in-memory y_clip + sr.
  4. Construct RemixRenderer + .render(remix_path) → RenderResult with
     audio + transition_times + edit_plan.
  5. Dump under references/golden/phase-5/renderer/<track>/r<R>/.

Per-case files:
  audio_in.npy           f32 (nCh, nSamples) — Renderer input audio.
  beat_times.npy         f64 (nBeats,)       — Renderer beat grid.
  remix_path.npy         i64 (nPath,)        — RemixPath.beat_indices.
  remix_pairs.npy        i64 (2*nTrans,)     — sorted (from,to) pairs flat.
  remix_meta_keys.json   list[str]           — meta key names captured below.
  remix_meta_<key>.npy   f64 (nTrans,)       — per-transition value; NaN absent.
  render_audio.npy       f32 (nCh, nSamples) — RenderResult.audio.
  transition_times.npy   f64 (nTrans,)       — timeline-relative.
  edit_plan.json         list[dict]          — per-clip 14-field EditClip dump.
  manifest.json          sr / n_beats / n_clips / duration / sha256s.

Reference-import pattern: mirror dump_phase4_tests.py — sys.path.insert on
the parent of the python-source/ symlink (resolves to .../RemixTool/src
where remix_tool/ package lives).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "references" / "golden" / "phase-5" / "renderer"
PHASE2_DUMPS = ROOT / "references" / "golden" / "phase-2" / "dumps"

_PY_SRC_SYMLINK = ROOT / "references" / "python-source"
_REMIX_TOOL_PKG_PARENT = _PY_SRC_SYMLINK.resolve().parent
if _REMIX_TOOL_PKG_PARENT.is_dir() and str(_REMIX_TOOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_REMIX_TOOL_PKG_PARENT))

# Re-use phase-4 dump pipeline helpers verbatim (already debugged session 26+).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_phase4_tests import (  # noqa: E402
    TRACK_LIST,
    load_track_inputs_beatthis,
    compute_segment_boundaries,
    _resolve_audio_source,
)

# ADR-032 fix-in-port: the C++ Renderer composes Crossfade::adaptiveCrossfade
# which already implements "Option B.myfix" (no pad-back; shift applied to
# full in_band buffer). Python source `crossfade.py:298-308` has the
# pad-back BUG — incoming post-treble-fade region is zeroed because
# `_phase_align` returns a slice of length xf_samples and the pad-back
# leaves [xf_samples:max_xf_samples] zeroed. To produce goldens that match
# C++ Renderer behavior (= the perceptually correct rendition per ADR-032),
# we monkey-patch `adaptive_crossfade` in the renderer module to call
# `adaptive_crossfade_fixed` from dump_phase5_crossfade.py for the duration
# of the renderer.render() call. The same precedent as phase-5 session 32.
from dump_phase5_crossfade import adaptive_crossfade_fixed  # noqa: E402

import librosa  # noqa: E402

from remix_tool.remix import transition_cost as _tc  # noqa: E402
from remix_tool.remix.optimizer import CleanOptimizer  # noqa: E402
from remix_tool.remix.renderer import RemixRenderer  # noqa: E402
from remix_tool.analysis.repetition_map import build_repetition_map  # noqa: E402


# Full 48-case corpus per ADR-034 phase-close gate: 16 tracks × 3 ratios.
# All ratios < 1.0 = shortening targets (the production use case).
RATIOS: Tuple[float, ...] = (0.3, 0.5, 0.7)
CORPUS: List[Tuple[str, float]] = [(track, r) for track in TRACK_LIST for r in RATIOS]


def _ratio_subdir(r: float) -> str:
    return f"r{int(round(r * 10)):02d}"


def _build_remix_path(track: str, target_ratio: float) -> Tuple[Any, Dict[str, Any], np.ndarray, int]:
    """Return (remix_path, optimizer_diag, beats_cpp, sr).

    Verbatim slice of dump_phase4_tests.py:dump_optimizer_smoke L864-932 +
    a returned remix_path object (instead of dumping its parts).
    """
    inputs = load_track_inputs_beatthis(track)
    sr = int(inputs["sr"])

    segments    = inputs["segments"]
    beat_times  = inputs["beat_times"]
    downbeats   = inputs["downbeats"]
    features    = inputs["features"].astype(np.float32)
    boundary_wf = inputs["boundary_waveforms"]

    rep_map = build_repetition_map(
        beat_times=beat_times,
        downbeats=downbeats if downbeats.size > 0 else None,
        features=features,
        segments=segments,
        boundary_waveforms=boundary_wf,
        waveform_sample_rate=sr,
        time_signature=4,
    )

    seg_boundaries = compute_segment_boundaries(segments, beat_times)
    tc_result = _tc.compute_transition_costs(
        features=features,
        boundary_waveforms=boundary_wf,
        waveform_sample_rate=sr,
        segments=segments,
        beat_times=beat_times,
        segment_boundaries=seg_boundaries if seg_boundaries.size > 0 else None,
        edge_rms_start=inputs["edge_rms_start"],
        edge_rms_end=inputs["edge_rms_end"],
        edge_features_start=inputs["edge_features_start"].astype(np.float32),
        edge_features_end=inputs["edge_features_end"].astype(np.float32),
        rms_energy=inputs["rms_energy"],
        onset_strength=inputs["onset_strength"],
        spectral_centroid=inputs["spectral_centroid"],
        vocal_activity=inputs["vocal_activity"],
        edge_vocal_activity_start=inputs["edge_vocal_activity_start"],
        edge_vocal_activity_end=inputs["edge_vocal_activity_end"],
        downbeats=downbeats if downbeats.size > 0 else None,
        time_signature=4,
    )

    n_beats = len(beat_times)
    original_duration = (
        float(beat_times[-1] - beat_times[0]) * (n_beats / max(1, n_beats - 1))
        if n_beats > 1 else 1.0
    )
    target_duration = original_duration * target_ratio

    optimizer = CleanOptimizer(
        cost_result=tc_result,
        beat_times=beat_times,
        downbeats=downbeats if downbeats.size > 0 else None,
        time_signature=4,
        segments=segments,
        features=features,
        repetition_map=rep_map,
        preserve_intro_beats=8,
        preserve_outro_beats=4,
        max_transitions=6,
        duration_tolerance_sec=5.0,
        boundary_waveforms=boundary_wf,
        waveform_sample_rate=sr,
    )
    remix_path = optimizer.remix(target_duration)

    diag = {
        "n_beats":         n_beats,
        "target_duration": target_duration,
        "remix_path_len":  len(remix_path.beat_indices),
        "n_transitions":   len(remix_path.transitions),
        "total_cost":      float(remix_path.total_cost),
    }
    return remix_path, diag, beat_times, sr


def _run_renderer(track: str,
                  remix_path: Any,
                  beat_times: np.ndarray,
                  sr: int) -> Tuple[Any, np.ndarray, int]:
    """Run RemixRenderer.render(path); return (RenderResult, audio_in, audio_in_nCh).

    Monkey-patches librosa.load so the renderer uses the in-memory clip from
    PHASE2_DUMPS/<track>/y_audio.npy (which is the clip fed into phase-1
    BeatDetector, so beat_times align with this audio exactly).
    """
    y_audio = np.load(PHASE2_DUMPS / track / "y_audio.npy")
    # librosa returns (samples,) for mono / (channels, samples) for stereo.
    # y_audio.npy is mono in our phase-2 corpus.
    audio_in_for_dump = y_audio if y_audio.ndim == 2 else y_audio[np.newaxis, :]
    audio_in_for_dump = audio_in_for_dump.astype(np.float32, copy=False)
    n_channels_in = int(audio_in_for_dump.shape[0])

    audio_path = _resolve_audio_source(track)

    orig_load = librosa.load

    def patched_load(path, *args, **kwargs):
        # Renderer passes positional sr=sample_rate via kwargs; we ignore and
        # return the pre-loaded clip + its true sr.
        return y_audio, sr

    # Hook adaptive_crossfade to record per-call (outgoing, incoming, blended).
    # These slices are the EXACT inputs/outputs the C++ Renderer must replicate.
    from remix_tool.remix import renderer as _renderer_mod
    orig_xfade = _renderer_mod.adaptive_crossfade
    crossfade_calls: List[Dict[str, np.ndarray]] = []

    def patched_xfade(outgoing, incoming, sr_arg, *args, **kwargs):
        # Substitute ADR-032 fixed implementation (NOT the buged Python
        # source). C++ Crossfade replicates this fixed semantic.
        blended = adaptive_crossfade_fixed(outgoing, incoming, sr_arg, *args, **kwargs)
        crossfade_calls.append({
            "outgoing": np.array(outgoing, dtype=np.float64, copy=True),
            "incoming": np.array(incoming, dtype=np.float64, copy=True),
            "blended":  np.array(blended,  dtype=np.float64, copy=True),
            "sr":       int(sr_arg),
        })
        return blended

    _renderer_mod.adaptive_crossfade = patched_xfade
    librosa.load = patched_load
    try:
        renderer = RemixRenderer(
            audio_path=str(audio_path),
            beat_times=beat_times,
            sample_rate=sr,
        )
        result = renderer.render(remix_path)
    finally:
        librosa.load = orig_load
        _renderer_mod.adaptive_crossfade = orig_xfade

    # Stash the captured calls on the result for the dumper to write out.
    result._crossfade_calls = crossfade_calls  # type: ignore[attr-defined]
    return result, audio_in_for_dump, n_channels_in


# All metadata keys the renderer might READ from transition_metadata
# (renderer.py L126, L150-152, L189-196, L215-216) PLUS the candidate +
# repetition keys phase-4 emits. We dump every key actually present in any
# transition's dict.
def _collect_meta_keys(remix_path: Any) -> List[str]:
    keys = set()
    for meta in remix_path.transition_metadata.values():
        for k in meta.keys():
            keys.add(str(k))
    return sorted(keys)


def _serialize_clips(plan: Any) -> List[Dict[str, Any]]:
    return [
        {
            "clip_index":         int(c.clip_index),
            "source_path":        str(c.source_path),
            "start_beat":         int(c.start_beat),
            "end_beat":           int(c.end_beat),
            "source_start_sec":   float(c.source_start_sec),
            "source_end_sec":     float(c.source_end_sec),
            "timeline_start_sec": float(c.timeline_start_sec),
            "timeline_end_sec":   float(c.timeline_end_sec),
            "duration_sec":       float(c.duration_sec),
            "fade_in_sec":        float(c.fade_in_sec),
            "fade_out_sec":       float(c.fade_out_sec),
            "overlap_before_sec": float(c.overlap_before_sec),
            "overlap_after_sec":  float(c.overlap_after_sec),
            "gap_after_sec":      float(c.gap_after_sec),
        }
        for c in plan.clips
    ]


def _sha256_npy(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dump_renderer_case(track: str, target_ratio: float, *, force: bool = False) -> None:
    out_dir = OUT_ROOT / track / _ratio_subdir(target_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force and (out_dir / "manifest.json").exists():
        print(f"[renderer/{track}/r{target_ratio:.1f}] EXISTS — skip (use --force to re-dump)")
        return

    remix_path, opt_diag, beat_times, sr = _build_remix_path(track, target_ratio)
    result, audio_in, n_channels_in = _run_renderer(track, remix_path, beat_times, sr)

    # Audio: result.audio shape is (n,) for mono squeezed, (nCh, n) for stereo
    # (per renderer.py:468-469). Re-expand to (1, n) if mono so the dump
    # convention matches C++ Renderer's channel-major output.
    audio_out = result.audio
    if audio_out.ndim == 1:
        audio_out = audio_out[np.newaxis, :]
    audio_out = audio_out.astype(np.float32, copy=False)

    # --- Save core arrays ------------------------------------------------
    np.save(out_dir / "audio_in.npy",      audio_in)
    np.save(out_dir / "beat_times.npy",    np.asarray(beat_times, dtype=np.float64))
    np.save(out_dir / "remix_path.npy",    np.asarray(remix_path.beat_indices, dtype=np.int64))

    # Pairs sorted (from, to) — match phase-4 dump convention.
    sorted_pairs = sorted(
        ((int(a), int(b)) for (a, b) in remix_path.transitions),
        key=lambda p: (p[0], p[1]),
    )
    pairs_flat: List[int] = []
    for a, b in sorted_pairs:
        pairs_flat.append(a)
        pairs_flat.append(b)
    np.save(out_dir / "remix_pairs.npy", np.asarray(pairs_flat, dtype=np.int64))

    # Per-key meta vectors (NaN where absent). Include EVERY key seen.
    sorted_meta = sorted(
        ((int(k[0]), int(k[1]), v) for k, v in remix_path.transition_metadata.items()),
        key=lambda t: (t[0], t[1]),
    )
    meta_keys = _collect_meta_keys(remix_path)
    (out_dir / "remix_meta_keys.json").write_text(json.dumps(meta_keys, indent=2))
    n_meta = len(sorted_meta)
    for key in meta_keys:
        vec = np.full(n_meta, np.nan, dtype=np.float64)
        for idx, (_, _, meta) in enumerate(sorted_meta):
            if key in meta:
                vec[idx] = float(meta[key])
        # Sanitize key (paths). Keys are dict-string already; reject any with
        # filesystem separators just in case.
        safe = key.replace("/", "_").replace("\\", "_")
        np.save(out_dir / f"remix_meta_{safe}.npy", vec)

    # Render outputs.
    np.save(out_dir / "render_audio.npy", audio_out)
    np.save(out_dir / "transition_times.npy",
            np.asarray(result.transition_times, dtype=np.float64))
    (out_dir / "edit_plan.json").write_text(
        json.dumps(_serialize_clips(result.edit_plan), indent=2)
    )

    # Per-crossfade-call dump for isolation tests (C++ vs Python on the same
    # outgoing/incoming slice). Bisection aid for the audio drift bug.
    crossfade_calls = getattr(result, "_crossfade_calls", [])
    for i, call in enumerate(crossfade_calls):
        np.save(out_dir / f"crossfade_call_{i:02d}_outgoing.npy", call["outgoing"])
        np.save(out_dir / f"crossfade_call_{i:02d}_incoming.npy", call["incoming"])
        np.save(out_dir / f"crossfade_call_{i:02d}_blended.npy",  call["blended"])

    # Manifest with diagnostics + sha256s.
    manifest = {
        "track":           track,
        "target_ratio":    target_ratio,
        "sample_rate":     sr,
        "n_beats":         int(len(beat_times)),
        "audio_in_shape":  list(audio_in.shape),
        "audio_in_dtype":  str(audio_in.dtype),
        "audio_out_shape": list(audio_out.shape),
        "audio_out_dtype": str(audio_out.dtype),
        "duration_sec":    float(result.duration),
        "n_transitions":   int(result.n_transitions),
        "n_clips":         int(len(result.edit_plan.clips)),
        "n_meta_keys":     len(meta_keys),
        "meta_keys":       meta_keys,
        "optimizer_diag":  opt_diag,
        "sha256_truncated": {
            "audio_in":         _sha256_npy(out_dir / "audio_in.npy"),
            "render_audio":     _sha256_npy(out_dir / "render_audio.npy"),
            "remix_path":       _sha256_npy(out_dir / "remix_path.npy"),
            "transition_times": _sha256_npy(out_dir / "transition_times.npy"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[renderer/{track}/r{target_ratio:.1f}] "
        f"sr={sr} nBeats={len(beat_times)} "
        f"path_len={opt_diag['remix_path_len']} "
        f"n_trans={opt_diag['n_transitions']} "
        f"n_clips={len(result.edit_plan.clips)} "
        f"dur={result.duration:.3f}s "
        f"audio_out={audio_out.shape} "
        f"meta_keys={len(meta_keys)} "
        f"-> {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing dumps")
    parser.add_argument("--case", action="append", default=None,
                        help="Subset to dump as 'track:ratio' (e.g. billie_jean:0.5). Repeatable.")
    args = parser.parse_args()

    cases: Iterable[Tuple[str, float]]
    if args.case:
        cases = [(c.split(":")[0], float(c.split(":")[1])) for c in args.case]
    else:
        cases = CORPUS

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for track, ratio in cases:
        dump_renderer_case(track, ratio, force=args.force)


if __name__ == "__main__":
    main()
