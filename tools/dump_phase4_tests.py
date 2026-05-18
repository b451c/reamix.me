#!/usr/bin/env python3
"""Phase-4 parity dump tool.

Dumps golden inputs + outputs for:
  - scipy.signal.sosfilt vs reamix::dsp::SosFilt on the vocal-band filter
    (scipy butter(4, [250, 3400], bandpass, fs=sr, output="sos")).
  - references/python-source/remix/transition_cost.py::compute_transition_costs
    vs reamix::remix::computeTransitionCosts.

History:
  - Session 18 (smoke): 1 track (billie_jean), 129-beat librosa path,
    `segments=None`, `downbeats=None`. Label / section / bar_align branches
    stayed at defaults.
  - Session 19 (this file): extended to 16-track corpus + segments enabled
    (phase-3 `dispatched_segments_*`, cross-grid-compatible since segments
    are in seconds) + downbeats enabled (from `downbeats_cpp.npy`, also in
    seconds). `segment_boundaries` derived per `_analyze.py:80-87` pattern
    (nearest-beat index for each segment start + end). 129-beat librosa
    path retained per HANDOVER-18 L127 rec; ADR-027 117-beat C++-BeatDetector
    path switches at phase-4 close.

Reference-import pattern: `sys.path.insert` on the parent of the
`python-source/` symlink (resolves to `.../RemixTool/src` where the
`remix_tool/` package lives). Same pattern as session-11 RepetitionMap
dump + session-18 smoke. .venv-phase2 already has the full remix_tool
dependency tree from phase-1/2/3.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
_PY_SRC_SYMLINK = ROOT / "references" / "python-source"
_REMIX_TOOL_PKG_PARENT = _PY_SRC_SYMLINK.resolve().parent
if _REMIX_TOOL_PKG_PARENT.is_dir() and str(_REMIX_TOOL_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_REMIX_TOOL_PKG_PARENT))

from remix_tool.remix import transition_cost as _tc  # noqa: E402
from remix_tool.remix import quality as _quality     # noqa: E402
from remix_tool.remix import viterbi_dp as _vdp      # noqa: E402
from remix_tool.remix import region_cost as _rc      # noqa: E402
from remix_tool.remix.optimizer import CleanOptimizer  # noqa: E402
from remix_tool.remix.region_cost import compute_region_costs  # noqa: E402
from remix_tool.remix.block_assembly import (  # noqa: E402
    build_blocks,
    compute_block_compatibility,
    assemble_blocks,
    TOP_K as _BLOCK_TOP_K,
)
from remix_tool.remix.transition_prescreen import prescreen_transitions  # noqa: E402
from remix_tool.analysis.repetition_map import build_repetition_map  # noqa: E402

from scipy.signal import butter, sosfilt             # noqa: E402


PHASE2_DUMPS = ROOT / "references" / "golden" / "phase-2" / "dumps"
PHASE4_OUT   = ROOT / "tests" / "parity" / "reference" / "data" / "phase4"

# ADR-027 integration parity on beat-this distribution (session 26+):
# parallel output tree so E2E (beats_cpp grid) goldens cannot collide with
# session-22 Optimizer goldens (librosa-beats grid).
PHASE4_E2E_OUT = ROOT / "tests" / "parity" / "reference" / "data" / "phase4_e2e"

# ADR-044 no-structure goldens (session 53+): parallel tree so the status-quo
# `phase4/` corpus stays runnable side-by-side with the no-structure gate.
# Same per-track shape as `phase4/` but with empty segments + Python pipeline
# patched to match the C++ noStructure semantic (section_sim=0, label_match=0,
# SPAN_PENALTY_*=0). See `no_structure_patches()` below + ADR-044 §
# Implementation plan steps (c)(d).
PHASE4_NS_OUT = ROOT / "tests" / "parity" / "reference" / "data" / "phase4_no_structure"

# 16-track corpus (session-15 gap-A/B expansion; billie_jean_real is
# Fixture C real-beats, not part of the structure corpus).
TRACK_LIST = [
    "alice_in_chains_nutshell",
    "billie_jean",
    "bob_dylan_lay_lady_lay",
    "daft_punk_aerodynamic",
    "dance_monkey",
    "eminem_without_me",
    "eno_music_for_airports_1_1",
    "goldberg_var_15_andante",
    "meshuggah_bleed",
    "miles_davis_so_what",
    "shostakovich_jazz_waltz",
    "smells_like_teen_spirit",
    "tiesto_the_motto",
    "vocal_solo_with_fx",
    "wardruna_voluspa",
    "woodkid_iron_acoustic",
]


def load_track_inputs(track: str) -> dict:
    """Load all per-beat inputs for the 129-beat librosa path.

    Returns a dict of np.ndarrays + scalar sr. Segments + downbeats come from
    the 117-beat phase-3 dispatcher + C++ BeatDetector dumps; since segments
    and downbeats are expressed in SECONDS (not beat indices), they are
    grid-agnostic and feed cleanly into the 129-beat librosa-based
    compute_transition_costs call.
    """
    d = PHASE2_DUMPS / track
    if not d.exists():
        raise FileNotFoundError(f"No phase-2 dump for {track} at {d}")

    def npy(name):
        return np.load(d / f"{name}.npy")

    manifest = json.loads((d / "manifest.json").read_text())

    # Segments: per-segment dicts with start/end in seconds + label.
    # `dispatched_segments_fields.npy` is f64 (n_seg, 4) [start, end,
    # confidence, cluster_id]. `dispatched_segments_labels.txt` has one
    # label per line.
    seg_fields = npy("dispatched_segments_fields").astype(np.float64)
    seg_labels_path = d / "dispatched_segments_labels.txt"
    seg_labels = seg_labels_path.read_text().splitlines()
    if seg_fields.shape[0] != len(seg_labels):
        raise RuntimeError(
            f"{track}: seg fields rows ({seg_fields.shape[0]}) != labels "
            f"count ({len(seg_labels)})"
        )
    segments = [
        {
            "start":      float(seg_fields[i, 0]),
            "end":        float(seg_fields[i, 1]),
            "confidence": float(seg_fields[i, 2]),
            "cluster_id": int(seg_fields[i, 3]),
            "label":      str(seg_labels[i]),
        }
        for i in range(seg_fields.shape[0])
    ]

    downbeats = npy("downbeats_cpp").astype(np.float64)

    return dict(
        beat_times=npy("beat_times").astype(np.float64),
        features=npy("orch_std_feature_matrix").astype(np.float64),         # (n, 59)
        boundary_waveforms=npy("boundary_waveforms").astype(np.float32),    # (n, 3417)
        edge_rms_start=npy("edge_rms_start").astype(np.float64),
        edge_rms_end=npy("edge_rms_end").astype(np.float64),
        edge_features_start=npy("orch_std_edge_features_start").astype(np.float64),
        edge_features_end=npy("orch_std_edge_features_end").astype(np.float64),
        rms_energy=npy("orch_std_rms_energy").astype(np.float64),
        onset_strength=npy("orch_std_onset_strength").astype(np.float64),
        spectral_centroid=npy("orch_std_spectral_centroid").astype(np.float64),
        vocal_activity=npy("orch_std_vocal_activity").astype(np.float64),
        edge_vocal_activity_start=npy("edge_vocal_activity_start").astype(np.float64),
        edge_vocal_activity_end=npy("edge_vocal_activity_end").astype(np.float64),
        segments=segments,
        seg_fields=seg_fields,
        seg_labels=seg_labels,
        downbeats=downbeats,
        sr=int(manifest["sr"]),
    )


# ---------------------------------------------------------------------------
# Session 26 — ADR-027 integration parity on beat-this distribution
# ---------------------------------------------------------------------------
#
# Sessions 18-24 validated per-module parity under a MIXED grid:
#   beat_times (librosa 129-beat for billie_jean) + features/edges on the same
#   librosa grid + downbeats_cpp (C++-BeatDetector) + dispatched_segments
#   (C++-grid via ADR-020). Segments + downbeats are in seconds → grid-
#   agnostic, so the mix is algebraically consistent.
#
# ADR-027 asks: run the FULL pipeline on the C++ BeatDetector grid instead of
# the librosa grid — catches cross-module distribution-sensitivity bugs that
# per-module tests cannot surface (e.g., _compute_downbeat_arrays argmin
# flipping on different beat spacings).
#
# Implementation option β (session-25 Q7, user-approved): call Python
# `FeatureExtractor.extract()` in memory on the `beats_cpp` grid — zero new
# phase-2 golden keys, same production code path as server/_analyze.py
# L184-190. Per-track wall ~2-5 s (librosa MFCC + chroma + contrast + vocal).

def _resolve_audio_source(track: str) -> Path:
    """Resolve a track's real audio file path via phase-2 manifest.

    FeatureExtractor.extract() validates `audio_path` exists on disk (L140);
    we point it at the full-track symlink under `references/golden/phase-1/`
    and override with `audio_buffer=y_clip, sample_rate=sr` so the on-disk
    file is existence-checked only, not reloaded.
    """
    manifest_path = PHASE2_DUMPS / track / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audio_source = manifest["audio_source"]  # relative to ROOT
    audio_path = (ROOT / audio_source).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio source for {track} not found: {audio_path} "
            f"(manifest['audio_source']={audio_source})"
        )
    return audio_path


def load_track_inputs_beatthis(track: str) -> dict:
    """Load per-beat inputs aligned to the C++ BeatDetector grid (beats_cpp).

    Same output shape as `load_track_inputs` but runs Python
    `FeatureExtractor.extract()` on `beats_cpp` instead of reading
    pre-dumped librosa-grid `orch_std_*` keys. Production code path (mirrors
    `server/handlers/_analyze.py` L184-190).

    Session 26 (ADR-027 integration parity).
    """
    # Late import to avoid pulling FeatureExtractor's chain unless this loader
    # is actually exercised. Phase-2 .venv verified clean (no torch pull).
    from remix_tool.analysis.feature_extractor import FeatureExtractor

    d = PHASE2_DUMPS / track
    if not d.exists():
        raise FileNotFoundError(f"No phase-2 dump for {track} at {d}")

    manifest = json.loads((d / "manifest.json").read_text())
    sr = int(manifest["sr"])

    beats_cpp = np.load(d / "beats_cpp.npy").astype(np.float64)
    if beats_cpp.size == 0:
        raise RuntimeError(f"{track}: beats_cpp.npy is empty (phase-2 dump incomplete?)")
    downbeats_cpp = np.load(d / "downbeats_cpp.npy").astype(np.float64)
    y = np.load(d / "y_audio.npy")

    audio_path = _resolve_audio_source(track)

    extractor = FeatureExtractor()
    result = extractor.extract(
        audio_path=str(audio_path),
        beat_times=beats_cpp,
        audio_buffer=y,
        sample_rate=sr,
        analysis_mode="standard",
    )

    # Segments + downbeats from phase-3 ADR-020 path — already on C++-beat-
    # compatible grid (seconds-valued, grid-agnostic). Reuse same fields as
    # `load_track_inputs`.
    seg_fields = np.load(d / "dispatched_segments_fields.npy").astype(np.float64)
    seg_labels_path = d / "dispatched_segments_labels.txt"
    seg_labels = seg_labels_path.read_text().splitlines()
    if seg_fields.shape[0] != len(seg_labels):
        raise RuntimeError(
            f"{track}: seg fields rows ({seg_fields.shape[0]}) != labels "
            f"count ({len(seg_labels)})"
        )
    segments = [
        {
            "start":      float(seg_fields[i, 0]),
            "end":        float(seg_fields[i, 1]),
            "confidence": float(seg_fields[i, 2]),
            "cluster_id": int(seg_fields[i, 3]),
            "label":      str(seg_labels[i]),
        }
        for i in range(seg_fields.shape[0])
    ]

    # Guard: FeatureResult fields may be None when skip_vocal_features fires
    # (fast mode). We request "standard" mode so they must be present; fail
    # loudly if not — parity tests cannot compare against None.
    # ALSO: FeatureExtractor returns some Fortran-order arrays (librosa chain
    # produces column-major); C++ NpyIO loader is C-order only, so force
    # `np.ascontiguousarray` on everything we return.
    def _require_c(name: str, arr, dtype):
        if arr is None:
            raise RuntimeError(
                f"{track}: FeatureExtractor returned None for {name!r} — "
                f"vocal features required for phase-4 integration parity."
            )
        return np.ascontiguousarray(arr, dtype=dtype)

    return dict(
        beat_times=np.ascontiguousarray(beats_cpp, dtype=np.float64),
        features=_require_c("features", result.features, np.float64),
        boundary_waveforms=_require_c("boundary_waveforms", result.boundary_waveforms, np.float32),
        edge_rms_start=_require_c("edge_rms_start", result.edge_rms_start, np.float64),
        edge_rms_end=_require_c("edge_rms_end", result.edge_rms_end, np.float64),
        edge_features_start=_require_c("edge_features_start", result.edge_features_start, np.float64),
        edge_features_end=_require_c("edge_features_end", result.edge_features_end, np.float64),
        rms_energy=_require_c("rms_energy", result.rms_energy, np.float64),
        onset_strength=_require_c("onset_strength", result.onset_strength, np.float64),
        spectral_centroid=_require_c("spectral_centroid", result.spectral_centroid, np.float64),
        vocal_activity=_require_c("vocal_activity", result.vocal_activity, np.float64),
        edge_vocal_activity_start=_require_c("edge_vocal_activity_start", result.edge_vocal_activity_start, np.float64),
        edge_vocal_activity_end=_require_c("edge_vocal_activity_end", result.edge_vocal_activity_end, np.float64),
        segments=segments,
        seg_fields=seg_fields,
        seg_labels=seg_labels,
        downbeats=np.ascontiguousarray(downbeats_cpp, dtype=np.float64),
        sr=sr,
    )


def compute_segment_boundaries(segments: list, beat_times: np.ndarray) -> np.ndarray:
    """Replicates server/handlers/_analyze.py:80-87 derivation.

    For each segment, finds the nearest beat index to start + end; the unique
    sorted list of these indices is `segment_boundaries`. Passed to
    `compute_transition_costs` via the `segment_boundaries=` kwarg and used by
    `_fill_sequential_costs` to apply a harsher sequential coeff at boundary
    beats. Empty-in → empty-out (None on production path; here we return an
    empty int array for uniform dump).
    """
    if not segments or beat_times.size == 0:
        return np.zeros(0, dtype=np.int64)
    idxs = []
    for seg in segments:
        idxs.append(int(np.argmin(np.abs(beat_times - float(seg["start"])))))
        idxs.append(int(np.argmin(np.abs(beat_times - float(seg["end"])))))
    return np.unique(np.asarray(idxs, dtype=np.int64))


def precompute_vocal_band(waveforms: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Butter 4th-order bandpass [250, 3400] Hz, SOS form, sample-rate `sr`."""
    sos = butter(4, [250, 3400], btype="bandpass", fs=sr, output="sos")
    sos = sos.astype(np.float64, copy=False)
    # Mirror transition_cost.py L353: f64 filter, then back to f32.
    vb = np.array([sosfilt(sos, w.astype(np.float64)).astype(np.float32)
                   for w in waveforms])
    return sos, vb


# ---------------------------------------------------------------------------
# ADR-044 no-structure monkey-patch (session 53)
# ---------------------------------------------------------------------------
#
# Mirrors the C++ TransitionCost.cpp + RegionCost.cpp gating: when the input
# pipeline carries empty segments (auto path post-ADR-044), the cost matrix
# treats `section_sim=0`, `label_match=0`, `SPAN_PENALTY_CROSS_SECTION=0`,
# `SPAN_PENALTY_SAME_SECTION=0`. Python's natural `segments=None`/`[]` path
# differs:
#   - `_compute_segment_data(segments=[])` returns `seg_sim_matrix=[[1.0]]`
#     (all-same-segment fallback) — leaks max-similarity into section_sim.
#   - `region_cost.py:368` computes `section_sim = label_match * 0.9 + 0.1`
#     inline — yields 0.1 when label_match=0, not 0.
#   - `SPAN_PENALTY_CROSS_SECTION` still fires when `label_match < 0.5`.
#
# To produce goldens that match the C++ ADR-044 semantic exactly, we patch
# four sites for the duration of the dump call:
#
#   1. `_compute_segment_data` (in BOTH `_tc` and `_vdp` — the former because
#      `transition_cost.py:32` does `from ... import _compute_segment_data`
#      which captures a module-local binding; the latter because viterbi_dp's
#      own internal calls go through its module dict).
#   2. `compute_quality_score` (in `_quality`, `_tc`, `_rc`) → forces
#      section_sim=0 + label_match=0 regardless of caller.
#   3. `SPAN_PENALTY_CROSS_SECTION` and `SPAN_PENALTY_SAME_SECTION` → 0.0
#      in `_tc`; `SPAN_PENALTY_CROSS_SECTION` → 0.0 in `_rc` (region uses
#      only the cross variant per `region_cost.py:387` comment).
#
# Per ADR-044 § Implementation plan (c). Pattern adapted from
# `tools/listening_adr044.py:_build_remix_path_mode` — that tool patched only
# `_vdp._compute_segment_data` and was sufficient for audible A/B but NOT
# strictly bit-equivalent to C++ on the `compute_transition_costs` output
# path (transition_cost.py:304 still saw the original `_compute_segment_data`
# via its `from ... import` binding). The parity gate requires the stronger
# patch set.

@contextlib.contextmanager
def no_structure_patches():
    """Apply ADR-044 no-structure semantic to the Python pipeline.

    Use as a context manager around `compute_transition_costs` and/or
    `compute_region_costs` calls. All patches are restored on exit.
    """
    orig_compute_seg_data = _vdp._compute_segment_data

    def patched_compute_seg_data(n, segments, *args, **kwargs):
        beat_to_segment, seg_sim, used_segments = orig_compute_seg_data(
            n, segments, *args, **kwargs)
        if segments is None or len(segments) == 0:
            seg_sim = np.zeros_like(seg_sim)
        return beat_to_segment, seg_sim, used_segments

    orig_quality = _quality.compute_quality_score

    def patched_quality(*, label_match, section_sim, **kwargs):  # noqa: ARG001
        return orig_quality(label_match=0.0, section_sim=0.0, **kwargs)

    orig_tc_cross = _tc.SPAN_PENALTY_CROSS_SECTION
    orig_tc_same  = _tc.SPAN_PENALTY_SAME_SECTION
    orig_rc_cross = _rc.SPAN_PENALTY_CROSS_SECTION

    try:
        # _compute_segment_data — patch in both modules that look it up.
        _vdp._compute_segment_data = patched_compute_seg_data
        _tc._compute_segment_data  = patched_compute_seg_data
        # compute_quality_score — patch in the source module + every
        # consumer's import-bound name.
        _quality.compute_quality_score = patched_quality
        _tc.compute_quality_score      = patched_quality
        _rc.compute_quality_score      = patched_quality
        # SPAN_PENALTY constants (each consumer has its own import-bound copy).
        _tc.SPAN_PENALTY_CROSS_SECTION = 0.0
        _tc.SPAN_PENALTY_SAME_SECTION  = 0.0
        _rc.SPAN_PENALTY_CROSS_SECTION = 0.0
        yield
    finally:
        _vdp._compute_segment_data = orig_compute_seg_data
        _tc._compute_segment_data  = orig_compute_seg_data
        _quality.compute_quality_score = orig_quality
        _tc.compute_quality_score      = orig_quality
        _rc.compute_quality_score      = orig_quality
        _tc.SPAN_PENALTY_CROSS_SECTION = orig_tc_cross
        _tc.SPAN_PENALTY_SAME_SECTION  = orig_tc_same
        _rc.SPAN_PENALTY_CROSS_SECTION = orig_rc_cross


def candidates_to_flat(candidates: dict) -> dict:
    """Flatten sparse candidate map to 1D arrays, sorted by (from, to).

    Matches C++ std::map<std::pair<int,int>> iteration order.
    """
    items = sorted(candidates.items(), key=lambda kv: kv[0])
    n = len(items)
    out = {
        "from":                   np.zeros(n, dtype=np.int64),
        "to":                     np.zeros(n, dtype=np.int64),
        "quality_score":          np.zeros(n, dtype=np.float64),
        "waveform_similarity":    np.zeros(n, dtype=np.float64),
        "successor_similarity":   np.zeros(n, dtype=np.float64),
        "edge_splice_similarity": np.zeros(n, dtype=np.float64),
        "chroma_distance":        np.zeros(n, dtype=np.float64),
        "energy_diff_db":         np.zeros(n, dtype=np.float64),
        "alignment_lag_samples":  np.zeros(n, dtype=np.int64),
        "total_cost":             np.zeros(n, dtype=np.float64),
    }
    for idx, ((fi, ti), cand) in enumerate(items):
        out["from"][idx]                   = fi
        out["to"][idx]                     = ti
        out["quality_score"][idx]          = cand.quality_score
        out["waveform_similarity"][idx]    = cand.waveform_similarity
        out["successor_similarity"][idx]   = cand.successor_similarity
        out["edge_splice_similarity"][idx] = cand.edge_splice_similarity
        out["chroma_distance"][idx]        = cand.chroma_distance
        out["energy_diff_db"][idx]         = cand.energy_diff_db
        out["alignment_lag_samples"][idx]  = cand.alignment_lag_samples
        out["total_cost"][idx]             = cand.total_cost
    return out


def dump_track(track: str, *, inputs_override: dict | None = None,
               out_root: Path | None = None,
               no_structure: bool = False) -> dict:
    inputs = inputs_override if inputs_override is not None else load_track_inputs(track)
    sr = inputs["sr"]

    sos, vb_waveforms = precompute_vocal_band(inputs["boundary_waveforms"], sr)

    # test_sosfilt standalone case (f64 throughout, bypass round-trip noise).
    sosfilt_input_f64  = inputs["boundary_waveforms"][0].astype(np.float64)
    sosfilt_output_f64 = sosfilt(sos, sosfilt_input_f64).astype(np.float64)

    if no_structure:
        # ADR-044 auto path: segments empty; downbeats stay (beat-this still
        # provides them — only StructureAnalyzer is skipped). Mirror the C++
        # AnalyzeWorker.cpp default-constructed StructureResult.
        segments       = []
        seg_fields_dump = np.zeros((0, 4), dtype=np.float64)
        seg_labels_dump = []
    else:
        segments        = inputs["segments"]
        seg_fields_dump = inputs["seg_fields"]
        seg_labels_dump = inputs["seg_labels"]
    beat_times    = inputs["beat_times"]
    downbeats     = inputs["downbeats"]
    seg_boundaries = compute_segment_boundaries(segments, beat_times)

    n = len(beat_times)
    cm = no_structure_patches() if no_structure else contextlib.nullcontext()
    with cm:
        result = _tc.compute_transition_costs(
            features=inputs["features"].astype(np.float32),
            boundary_waveforms=inputs["boundary_waveforms"],
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

    cand_flat = candidates_to_flat(result.candidates)

    out_dir = (out_root or PHASE4_OUT) / track
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", arr)

    # Inputs (echoed so C++ test doesn't need to traverse two dump trees).
    save("beat_times",                  inputs["beat_times"])
    save("features",                    inputs["features"].astype(np.float32))
    save("boundary_waveforms",          inputs["boundary_waveforms"])
    save("vocal_band_waveforms",        vb_waveforms)
    save("edge_rms_start",              inputs["edge_rms_start"])
    save("edge_rms_end",                inputs["edge_rms_end"])
    save("edge_features_start",         inputs["edge_features_start"].astype(np.float32))
    save("edge_features_end",           inputs["edge_features_end"].astype(np.float32))
    save("rms_energy",                  inputs["rms_energy"])
    save("onset_strength",              inputs["onset_strength"])
    save("spectral_centroid",           inputs["spectral_centroid"])
    save("vocal_activity",              inputs["vocal_activity"])
    save("edge_vocal_activity_start",   inputs["edge_vocal_activity_start"])
    save("edge_vocal_activity_end",     inputs["edge_vocal_activity_end"])

    # Segment context (session 19+; ADR-044 no-structure mode emits empty
    # arrays — the C++ test reads `seg_start.size() == 0` → `n_segments=0`
    # and triggers the noStructure gating in TransitionCost.cpp).
    if seg_fields_dump.shape[0] == 0:
        save("seg_start",      np.zeros(0, dtype=np.float64))
        save("seg_end",        np.zeros(0, dtype=np.float64))
        save("seg_confidence", np.zeros(0, dtype=np.float64))
        save("seg_cluster_id", np.zeros(0, dtype=np.int64))
        # Empty file — readLines() returns []. Note: writing "\n" instead
        # would yield [""] on the C++ side and trip the size-mismatch check
        # in test_transition_cost.cpp::loadTrackBundle.
        (out_dir / "seg_labels.txt").write_text("")
    else:
        save("seg_start",      seg_fields_dump[:, 0].astype(np.float64))
        save("seg_end",        seg_fields_dump[:, 1].astype(np.float64))
        save("seg_confidence", seg_fields_dump[:, 2].astype(np.float64))
        save("seg_cluster_id", seg_fields_dump[:, 3].astype(np.int64))
        (out_dir / "seg_labels.txt").write_text("\n".join(seg_labels_dump) + "\n")

    save("segment_boundaries", seg_boundaries)
    save("downbeats",          downbeats)

    # SOS + sosfilt reference case (for test_sosfilt).
    save("sos_coeffs",                  sos)
    save("sosfilt_input_f64",           sosfilt_input_f64)
    save("sosfilt_output_f64",          sosfilt_output_f64)

    # compute_transition_costs outputs.
    save("W",                           result.W)
    # chroma_D comes out as f32 from Python (numpy keeps dtype in
    # `1.0 - np.clip(f32_matmul, ...)`). We upcast to f64 for the dump so
    # the NpyIO loader on the C++ side has a single path. The upcast is
    # value-preserving, and the C++ port computes its final clip+subtract
    # in f32 before widening, so the comparison stays bit-exact.
    save("chroma_D",                    result.chroma_D.astype(np.float64))
    save("importance",                  result.importance)

    save("cand_from",                   cand_flat["from"])
    save("cand_to",                     cand_flat["to"])
    save("cand_quality_score",          cand_flat["quality_score"])
    save("cand_waveform_similarity",    cand_flat["waveform_similarity"])
    save("cand_successor_similarity",   cand_flat["successor_similarity"])
    save("cand_edge_splice_similarity", cand_flat["edge_splice_similarity"])
    save("cand_chroma_distance",        cand_flat["chroma_distance"])
    save("cand_energy_diff_db",         cand_flat["energy_diff_db"])
    save("cand_alignment_lag_samples",  cand_flat["alignment_lag_samples"])
    save("cand_total_cost",             cand_flat["total_cost"])

    W = result.W
    finite_count = int(np.sum(np.isfinite(W) & (W < _tc.INF * 0.9)))
    track_has_vocals = bool(
        inputs["vocal_activity"].size > 0
        and float(np.max(inputs["vocal_activity"])) >= _tc.TRACK_VOCAL_THRESHOLD
    )

    manifest = {
        "track": track,
        "source_phase2_dump": f"references/golden/phase-2/dumps/{track}/",
        "upstream_path": "librosa-129-beat (HANDOVER-18 L127 rec; ADR-027 117-beat C++-BeatDetector at phase-4 close)",
        "sr": sr,
        "n_beats": n,
        "n_boundary_waveforms": int(inputs["boundary_waveforms"].shape[0]),
        "n_samples_per_bnd":    int(inputs["boundary_waveforms"].shape[1]),
        "n_edge_features":      int(inputs["edge_features_start"].shape[1]),
        "n_features":           int(inputs["features"].shape[1]),
        "sos_shape":            list(sos.shape),
        "n_candidates":         len(cand_flat["from"]),
        "n_segments":           len(segments),
        "n_segment_boundaries": int(seg_boundaries.size),
        "n_downbeats":          int(downbeats.size),
        "segments_provided":    not no_structure,
        "no_structure":         bool(no_structure),
        "downbeats_provided":   bool(downbeats.size > 0),
        "track_has_vocals":     track_has_vocals,
        "time_signature":       4,
        "quality_floor":        float(_quality.QUALITY_HARD_FLOOR),
        "chroma_prefilter":     float(_tc.CHROMA_PREFILTER_THRESHOLD),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "track":          track,
        "n_beats":        n,
        "n_candidates":   len(cand_flat["from"]),
        "finite_W":       finite_count,
        "n_segments":     len(segments),
        "n_downbeats":    int(downbeats.size),
        "has_vocals":     track_has_vocals,
    }
    print(
        f"[{track}] n_beats={n} n_cands={len(cand_flat['from'])} "
        f"finite_W={finite_count} n_seg={len(segments)} "
        f"n_db={int(downbeats.size)} has_vocals={track_has_vocals} "
        f"dir={out_dir}"
    )
    return summary


def compute_dp_params(
    n_beats: int,
    beat_times: np.ndarray,
    target_duration: float,
    *,
    duration_tolerance_sec: float = 5.0,
    preserve_intro_beats: int = 8,
    preserve_outro_beats: int = 4,
    time_signature: int = 4,
) -> dict:
    """Mirror optimizer.py::_dp_params + adaptive_cooldown + min_jumps.

    Returns the full set of `_viterbi_dp` kwargs needed for session-20 smoke.
    Verbatim replica of optimizer.py:194-261 for the single-pass non-region
    remix path.
    """
    # Avg beat duration + original duration (optimizer.py:119-124).
    if n_beats > 1:
        avg_beat_duration = float((beat_times[-1] - beat_times[0]) / (n_beats - 1))
    else:
        avg_beat_duration = 1.0
    cumulative_total = avg_beat_duration * n_beats  # approximates optimizer._cumulative[n_beats]

    # optimizer.py:198-213 — target_beats + tolerance.
    target_beats   = max(2, int(round(target_duration / avg_beat_duration)))
    is_shortening  = target_duration < cumulative_total
    target_ratio   = target_duration / max(1.0, cumulative_total)
    adaptive_tol   = max(2.0, duration_tolerance_sec * max(0.4, min(1.0, target_ratio)))
    tolerance_beats = max(2, int(round(adaptive_tol / avg_beat_duration)))
    min_beats = max(2, target_beats - tolerance_beats)
    max_beats = min(n_beats * 3, target_beats + tolerance_beats)

    # optimizer.py:215-227 — adaptive intro / outro.
    intro_beats = preserve_intro_beats
    outro_beats = preserve_outro_beats
    if target_ratio < 0.8:
        intro_scale = max(0.25, target_ratio)
        intro_beats = max(4, int(preserve_intro_beats * intro_scale))
        outro_scale = max(0.3, target_ratio)
        outro_beats = max(2, int(preserve_outro_beats * outro_scale))

    # optimizer.py:247 — min_jumps gate.
    target_ratio_beats = target_beats / max(1, n_beats)
    min_jumps = 1 if (target_ratio_beats < 0.45 and is_shortening) else 0

    # optimizer.py:249-261 — adaptive cooldown.
    # COOLDOWN_BARS = 4 from viterbi_dp.py:26.
    cooldown_bars = 4
    if target_ratio_beats < 0.5:
        adaptive_cooldown_bars = max(1, int(cooldown_bars * target_ratio_beats * 2))
        adaptive_cooldown = adaptive_cooldown_bars * time_signature
        adaptive_forward  = adaptive_cooldown
    else:
        adaptive_cooldown = cooldown_bars * time_signature
        adaptive_forward  = cooldown_bars * time_signature

    return dict(
        target_length      = int(max_beats),       # effective_max -> viterbi_dp target_length
        min_target_length  = int(max(2, min_beats)),
        intro_beats        = int(intro_beats),
        outro_beats        = int(outro_beats),
        is_shortening      = bool(is_shortening),
        max_transitions    = 6,
        min_jumps          = int(min_jumps),
        min_seq_after_jump = int(adaptive_cooldown),
        min_forward_jump   = int(adaptive_forward),
        target_duration    = float(target_duration),
        target_beats       = int(target_beats),
        target_ratio       = float(target_ratio),
    )


def _ratio_subdir(target_ratio: float) -> str:
    """Canonical per-ratio subdir name.

    Session-20 dumped flat under `<track>/viterbi/`. Session-21 extends to
    48 cases (16 tracks × 3 ratios), so each ratio lives in its own subdir.
    Uses one-decimal suffix (`r0.3`, `r0.5`, `r0.7`) — short, readable,
    collision-free for the ratios in the corpus.
    """
    return f"r{target_ratio:.1f}"


def dump_viterbi_smoke(track: str, target_ratio: float = 0.5, *,
                       inputs_override: dict | None = None,
                       out_root: Path | None = None) -> None:
    """ViterbiDP golden dump for one (track, target_ratio) pair.

    Writes to `tests/parity/reference/data/phase4/<track>/viterbi/r<R>/`.
    Session-21 extends the session-20 single-track smoke to a 48-case
    corpus (16 tracks × 3 ratios); each case lives under its own `r<R>/`
    subdir so bisection stays per-ratio.

    Session 26 (ADR-027 integration parity): accepts `inputs_override` +
    `out_root` so the caller can feed C++-BeatDetector-grid inputs and
    write to `phase4_e2e/` instead of the default librosa-grid path.

    Calls Python `_compute_downbeat_arrays`, `_build_neighbors`, and
    `_viterbi_dp` directly via verbatim imports from
    `remix_tool.remix.viterbi_dp`. RepetitionMap is built via the canonical
    `build_repetition_map` from `remix_tool.analysis.repetition_map`.
    """
    inputs = inputs_override if inputs_override is not None else load_track_inputs(track)
    sr = inputs["sr"]

    segments   = inputs["segments"]
    beat_times = inputs["beat_times"]
    downbeats  = inputs["downbeats"]
    features   = inputs["features"].astype(np.float32)
    boundary_wf = inputs["boundary_waveforms"]

    # --- Stage 1: rebuild RepetitionMap (session-11 production pattern) ---
    rep_map = build_repetition_map(
        beat_times=beat_times,
        downbeats=downbeats if downbeats.size > 0 else None,
        features=features,
        segments=segments,
        boundary_waveforms=boundary_wf,
        waveform_sample_rate=sr,
        time_signature=4,
    )

    # --- Stage 2: compute transition costs (reuse 16-track dump path) ----
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
    W = tc_result.W

    # --- Stage 3: _compute_downbeat_arrays -------------------------------
    pre_db_arr, db_arr, db_indices = _vdp._compute_downbeat_arrays(
        beat_times=beat_times,
        downbeats=downbeats if downbeats.size > 0 else None,
        time_signature=4,
        n_beats=len(beat_times),
    )

    # --- Stage 4: _compute_segment_data ---------------------------------
    beat_to_segment, seg_sim_matrix, boundary_beats = _vdp._compute_segment_data(
        n_beats=len(beat_times),
        segments=segments,
        beat_times=beat_times,
        features=features,
    )

    # --- Stage 5: _build_neighbors --------------------------------------
    pre_db_indices = {db - 1 for db in db_indices if db > 0} if db_indices is not None else None

    # CRITICAL: optimizer.py L152 uses `min_neighbor_jump = time_signature`
    # (= 4 for 4/4), NOT the viterbi_dp.py:28 default of 16. The actual
    # cooldown is enforced inside `_viterbi_dp` (adaptive per target);
    # neighbors must include short-range candidates so DP can use them when
    # the cooldown shrinks for aggressive shortening. Missing this subtlety
    # produces a RESTRICTED neighbor set and wrong path selection.
    min_neighbor_jump = 4  # time_signature for 4/4 corpus
    indices, offsets = _vdp._build_neighbors(
        n_beats=len(beat_times),
        W=W,
        repetition_map=rep_map,
        downbeat_indices=db_indices,
        pre_downbeat_indices=pre_db_indices,
        segment_boundaries=boundary_beats,
        max_neighbors=32,
        min_forward_jump=min_neighbor_jump,
    )

    # --- Stage 6: DP params + _viterbi_dp -------------------------------
    n_beats = len(beat_times)
    original_duration = (
        float(beat_times[-1] - beat_times[0]) * (n_beats / max(1, n_beats - 1))
        if n_beats > 1 else 1.0
    )
    target_duration = original_duration * target_ratio
    dp_params = compute_dp_params(
        n_beats=n_beats,
        beat_times=beat_times,
        target_duration=target_duration,
        time_signature=4,
    )

    # RepetitionMap was consumed in _build_neighbors; for sub-function
    # parity the C++ test reads the same `indices`/`offsets` golden.
    path, total_cost = _vdp._viterbi_dp(
        W=W,
        target_length=dp_params["target_length"],
        min_target_length=dp_params["min_target_length"],
        intro_beats=dp_params["intro_beats"],
        outro_beats=dp_params["outro_beats"],
        neighbor_indices=indices,
        neighbor_offsets=offsets,
        beat_to_segment=beat_to_segment,
        seg_sim_matrix=seg_sim_matrix,
        pre_downbeat_arr=pre_db_arr,
        downbeat_arr=db_arr,
        n_beats=n_beats,
        is_shortening=dp_params["is_shortening"],
        max_transitions=dp_params["max_transitions"],
        min_jumps=dp_params["min_jumps"],
        min_seq_after_jump=dp_params["min_seq_after_jump"],
        min_forward_jump=dp_params["min_forward_jump"],
    )

    # --- Stage 7: dump --------------------------------------------------
    out_dir = (out_root or PHASE4_OUT) / track / "viterbi" / _ratio_subdir(target_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", arr)

    # Downbeat arrays (pre_db/db arrays can be None on `downbeat_constraint=False`;
    # production uses True so arrays are present for this smoke). Saved as
    # int64 (not int8 despite Python dtype) because NpyIO.h's reader surface
    # is f64/f32/i64 only; any downstream C++ consumer cast back to int8
    # at load time.
    save("pre_downbeat_arr", np.asarray(pre_db_arr, dtype=np.int64))
    save("downbeat_arr",     np.asarray(db_arr,     dtype=np.int64))
    save("downbeat_indices", np.asarray(sorted(db_indices), dtype=np.int64))
    save("pre_downbeat_indices",
         np.asarray(sorted(pre_db_indices) if pre_db_indices else [], dtype=np.int64))

    # Segment data (echoed — redundant with session-19 seg_sim would be but
    # Python `_compute_segment_data` returns flat f64 matrix; keep separate
    # to isolate the sub-function parity gate from SegmentData extraction
    # at session 19).
    save("beat_to_segment", np.asarray(beat_to_segment, dtype=np.int64))
    save("seg_sim_matrix",  np.asarray(seg_sim_matrix,  dtype=np.float64))
    save("boundary_beats",  np.asarray(sorted(boundary_beats), dtype=np.int64))

    # RepetitionMap jumps — flat arrays (9 fields per RepetitionJump dataclass).
    rep_n = len(rep_map.jumps)
    save("rep_from_beat",              np.asarray([j.from_beat for j in rep_map.jumps], dtype=np.int64))
    save("rep_to_beat",                np.asarray([j.to_beat   for j in rep_map.jumps], dtype=np.int64))
    save("rep_waveform_similarity",    np.asarray([j.waveform_similarity for j in rep_map.jumps], dtype=np.float64))
    save("rep_chroma_correlation",     np.asarray([j.chroma_correlation  for j in rep_map.jumps], dtype=np.float64))
    save("rep_alignment_lag_samples",  np.asarray([j.alignment_lag_samples for j in rep_map.jumps], dtype=np.int64))
    save("rep_from_section_idx",       np.asarray([j.from_section_idx for j in rep_map.jumps], dtype=np.int64))
    save("rep_to_section_idx",         np.asarray([j.to_section_idx   for j in rep_map.jumps], dtype=np.int64))
    save("rep_from_bar",               np.asarray([j.from_bar for j in rep_map.jumps], dtype=np.int64))
    save("rep_to_bar",                 np.asarray([j.to_bar   for j in rep_map.jumps], dtype=np.int64))

    # Neighbor CSR.
    save("neighbor_indices", np.asarray(indices, dtype=np.int64))
    save("neighbor_offsets", np.asarray(offsets, dtype=np.int64))

    # DP result.
    save("dp_path",       np.asarray(path, dtype=np.int64))
    save("dp_total_cost", np.asarray([float(total_cost)], dtype=np.float64))

    # W is already dumped at the TransitionCost level; echo here for isolated
    # test bisection (reload single dir).
    save("W", W)

    # Manifest.
    manifest = {
        "track":                 track,
        "target_ratio":          target_ratio,
        "target_duration":       float(target_duration),
        "original_duration":     float(original_duration),
        "n_beats":               n_beats,
        "n_segments":            len(segments),
        "n_downbeats":           int(downbeats.size),
        "n_rep_jumps":           rep_n,
        "dp_params":             dp_params,
        "path_length":           int(len(path)),
        "dp_total_cost":         float(total_cost),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[viterbi/{track}] n_beats={n_beats} target={dp_params['target_beats']} "
        f"(ratio={dp_params['target_ratio']:.3f}) "
        f"path_len={len(path)} n_jumps_in_path={_count_jumps(path)} "
        f"rep_jumps={rep_n} "
        f"min_seq={dp_params['min_seq_after_jump']} "
        f"cost={total_cost:.4f} dir={out_dir}"
    )


def _count_jumps(path) -> int:
    """Count non-sequential transitions in the beat-index path."""
    if len(path) < 2:
        return 0
    return int(sum(1 for k in range(1, len(path)) if path[k] != path[k - 1] + 1))


# ---------------------------------------------------------------------------
# Session 22 — CleanOptimizer facade smoke dump
# ---------------------------------------------------------------------------
#
# Layered on top of `dump_viterbi_smoke` — reuses the same per-track
# TransitionCost inputs + RepetitionMap + beat_times + segments + downbeats.
# Writes under `<track>/optimizer/r<R>/` (separate subdir from ViterbiDP
# per HANDOVER-21 L141 rec). ViterbiDP `dp_path` dumped in `viterbi/r<R>/`
# remains the session-20/21 artifact; the optimizer dump captures the
# facade's POST-DP output (dp_path + outro-extension + transition_metadata).
#
# Relationship:
#   - remix_path == dp_path + optional end-of-song extension at optimizer
#     .py L289-301. When DP does NOT terminate in the outro region, or
#     outro_beats==0, remix_path == dp_path exactly.
#   - remix_total_cost == dp_total_cost (no extension cost is added).
#
# Metadata representation (for parity test convenience):
#   meta_pairs.npy  int64 flat (2 * n_transitions,): sorted-by-(from,to)
#                   pairs — both Python and C++ sort by pair order so the
#                   representation is deterministic.
#   meta_<key>.npy  f64 (n_transitions,): per-transition value; NaN =
#                   key absent. Python test harness emits NaN when the
#                   Python dict does not contain the key; C++ test reads
#                   NaN as "C++ should have no such key in its dict".
# Ten possible keys per optimizer.py L321-338:
#   candidate-sourced (8): quality_score, waveform_similarity,
#     successor_similarity, edge_splice_similarity, chroma_distance,
#     energy_diff_db, alignment_offset_sec, total_cost.
#   repetition-sourced (2): is_repetition_jump, chroma_correlation.
_META_CANDIDATE_KEYS = (
    "quality_score",
    "waveform_similarity",
    "successor_similarity",
    "edge_splice_similarity",
    "chroma_distance",
    "energy_diff_db",
    "alignment_offset_sec",
    "total_cost",
)
_META_REPETITION_KEYS = (
    "is_repetition_jump",
    "chroma_correlation",
)
_META_ALL_KEYS = _META_CANDIDATE_KEYS + _META_REPETITION_KEYS


def dump_optimizer_smoke(track: str, target_ratio: float = 0.5, *,
                         inputs_override: dict | None = None,
                         out_root: Path | None = None) -> None:
    """CleanOptimizer facade golden dump for one (track, target_ratio) pair.

    Layers on top of `dump_viterbi_smoke` — exercises the same DP path plus
    the post-DP outro extension + transition_metadata build (optimizer.py
    L289-348). Writes under `<track>/optimizer/r<R>/`.

    Session 26 (ADR-027): accepts `inputs_override` + `out_root` for the
    E2E beat-this-distribution path.
    """
    inputs = inputs_override if inputs_override is not None else load_track_inputs(track)
    sr = inputs["sr"]

    segments   = inputs["segments"]
    beat_times = inputs["beat_times"]
    downbeats  = inputs["downbeats"]
    features   = inputs["features"].astype(np.float32)
    boundary_wf = inputs["boundary_waveforms"]

    # --- Build the TransitionCostResult + RepetitionMap exactly as the -----
    # --- production CleanOptimizer __init__ path consumes them -------------
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

    # --- Instantiate CleanOptimizer ---------------------------------------
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

    # --- Call .remix() ----------------------------------------------------
    remix_path = optimizer.remix(target_duration)

    # --- Build the sorted-by-(from,to) metadata representation ------------
    # Python's transition_metadata is dict with insertion order (path
    # order). C++ uses std::map (lex order). Both sides sort by (from, to)
    # at dump/read time so comparison is deterministic.
    sorted_meta_items = sorted(
        remix_path.transition_metadata.items(),
        key=lambda kv: (int(kv[0][0]), int(kv[0][1])),
    )
    n_transitions = len(sorted_meta_items)

    meta_pairs_flat: list[int] = []
    for (i, j), _ in sorted_meta_items:
        meta_pairs_flat.append(int(i))
        meta_pairs_flat.append(int(j))

    def extract_key_values(key: str) -> np.ndarray:
        """Per-transition vector of values; NaN where key absent."""
        out = np.full(n_transitions, np.nan, dtype=np.float64)
        for idx, (_, meta) in enumerate(sorted_meta_items):
            if key in meta:
                out[idx] = float(meta[key])
        return out

    # --- Dump -------------------------------------------------------------
    out_dir = (out_root or PHASE4_OUT) / track / "optimizer" / _ratio_subdir(target_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", arr)

    # Primary facade outputs.
    save("remix_path",        np.asarray(remix_path.beat_indices, dtype=np.int64))
    save("remix_total_cost",  np.asarray([float(remix_path.total_cost)], dtype=np.float64))
    save("meta_pairs",        np.asarray(meta_pairs_flat, dtype=np.int64))
    for key in _META_ALL_KEYS:
        save(f"meta_{key}", extract_key_values(key))

    # Diagnostic echo of the computed DP parameters for manifest.
    dp_params = compute_dp_params(
        n_beats=n_beats,
        beat_times=beat_times,
        target_duration=target_duration,
        time_signature=4,
    )

    # Assert: Python `CleanOptimizer` uses the same `_find_intro_end` /
    # `_find_outro_beats` logic we dump. Capture the result so the C++
    # test can cross-check without re-deriving segment labels.
    manifest = {
        "track":                 track,
        "target_ratio":          target_ratio,
        "target_duration":       float(target_duration),
        "original_duration":     float(original_duration),
        "n_beats":               n_beats,
        "n_segments":            len(segments),
        "n_downbeats":           int(downbeats.size),
        "n_transitions":         n_transitions,
        "dp_params":             dp_params,
        "effective_intro":       int(optimizer._effective_intro),
        "effective_outro":       int(optimizer._effective_outro),
        "avg_beat_duration":     float(optimizer._avg_beat_duration),
        "remix_path_length":     int(len(remix_path.beat_indices)),
        "remix_total_cost":      float(remix_path.total_cost),
        "remix_duration_beats":  int(remix_path.duration_beats),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[optimizer/{track}] n_beats={n_beats} "
        f"ratio={target_ratio:.1f} "
        f"path_len={len(remix_path.beat_indices)} "
        f"n_trans={n_transitions} "
        f"cost={remix_path.total_cost:.4f} "
        f"intro={optimizer._effective_intro} outro={optimizer._effective_outro} "
        f"dir={out_dir}"
    )


def _dump_remix_path_to_dir(out_dir: Path, remix_path) -> dict:
    """Helper — write one RemixPath to a goldens directory with the standard
    optimizer dump layout. Returns the per-path summary dict for manifest.

    Used by `dump_optimizer_smoke` (single best path) and `dump_optimizer_
    variations_smoke` (one dir per variation). Refactored out at session 58.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_meta_items = sorted(
        remix_path.transition_metadata.items(),
        key=lambda kv: (int(kv[0][0]), int(kv[0][1])),
    )
    n_transitions = len(sorted_meta_items)

    meta_pairs_flat: list[int] = []
    for (i, j), _ in sorted_meta_items:
        meta_pairs_flat.append(int(i))
        meta_pairs_flat.append(int(j))

    def extract_key_values(key: str) -> np.ndarray:
        out = np.full(n_transitions, np.nan, dtype=np.float64)
        for idx, (_, meta) in enumerate(sorted_meta_items):
            if key in meta:
                out[idx] = float(meta[key])
        return out

    np.save(out_dir / "remix_path.npy",       np.asarray(remix_path.beat_indices, dtype=np.int64))
    np.save(out_dir / "remix_total_cost.npy", np.asarray([float(remix_path.total_cost)], dtype=np.float64))
    np.save(out_dir / "meta_pairs.npy",       np.asarray(meta_pairs_flat, dtype=np.int64))
    for key in _META_ALL_KEYS:
        np.save(out_dir / f"meta_{key}.npy", extract_key_values(key))

    return {
        "n_transitions":     n_transitions,
        "remix_path_length": int(len(remix_path.beat_indices)),
        "remix_total_cost":  float(remix_path.total_cost),
        "remix_duration_beats": int(remix_path.duration_beats),
    }


def dump_optimizer_variations_smoke(track: str, target_ratio: float = 0.5, *,
                                     k: int = 4,
                                     inputs_override: dict | None = None,
                                     out_root: Path | None = None) -> None:
    """CleanOptimizer.remix_k_best parity dump for one (track, target_ratio).

    Layered on top of `dump_optimizer_smoke` setup — loads inputs, builds
    optimizer, calls `optimizer.remix_k_best(target, k=k)`, and dumps each
    returned path under `<track>/optimizer/r<R>/variations/v<idx>/` mirroring
    the existing `<track>/optimizer/r<R>/` layout (so C++ `loadCase` can
    point to either by passing the leaf directory).

    Session 58 / ADR-048 / DEV-027.
    """
    inputs = inputs_override if inputs_override is not None else load_track_inputs(track)
    sr = inputs["sr"]

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

    paths = optimizer.remix_k_best(target_duration, k=k)
    n_paths = len(paths)

    variations_root = (
        (out_root or PHASE4_OUT) / track / "optimizer" / _ratio_subdir(target_ratio) / "variations"
    )
    variations_root.mkdir(parents=True, exist_ok=True)

    per_path_summary = []
    for v_idx, path in enumerate(paths):
        v_dir = variations_root / f"v{v_idx}"
        info = _dump_remix_path_to_dir(v_dir, path)
        info["variation"] = v_idx
        per_path_summary.append(info)

    manifest = {
        "track":               track,
        "target_ratio":        target_ratio,
        "target_duration":     float(target_duration),
        "original_duration":   float(original_duration),
        "n_beats":             n_beats,
        "k_requested":         k,
        "n_paths_returned":    n_paths,
        "variation_strategy":  "remix_k_best_session58_ADR048",
        "paths":               per_path_summary,
    }
    (variations_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    costs_str = ", ".join(f"{p['remix_total_cost']:.4f}" for p in per_path_summary)
    print(
        f"[optimizer-variations/{track}] ratio={target_ratio:.1f} "
        f"k_req={k} n_paths={n_paths} costs=[{costs_str}] dir={variations_root}"
    )


def dump_optimizer_e2e_smoke(track: str, target_ratio: float = 0.5,
                              *, inputs: dict | None = None) -> None:
    """ADR-027 E2E integration parity smoke — one (track, ratio) pair on the
    C++-BeatDetector grid.

    Composes the full pipeline on `beats_cpp` instead of the librosa grid:
      1. `load_track_inputs_beatthis(track)` — calls Python FeatureExtractor
         on the clip audio + beats_cpp; produces the full per-beat bundle
         (features, edges, RMS, onset, centroid, vocals) aligned to beats_cpp.
      2. `dump_track(track, inputs_override=..., out_root=PHASE4_E2E_OUT)` —
         runs `compute_transition_costs` on the C++-grid inputs; writes
         beat_times/features/W/candidates/etc. under `phase4_e2e/<track>/`.
      3. `dump_viterbi_smoke(track, ratio, inputs_override=..., out_root=...)`
         — RepetitionMap + neighbors + DP on C++-grid inputs; writes under
         `phase4_e2e/<track>/viterbi/r<R>/`.
      4. `dump_optimizer_smoke(track, ratio, inputs_override=..., out_root=...)`
         — CleanOptimizer facade (post-DP outro extension + transition_metadata)
         on C++-grid inputs; writes under `phase4_e2e/<track>/optimizer/r<R>/`.

    Session 26 (ADR-027 phase-4 close gate prep). Session 27 added `inputs`
    kwarg so the corpus dispatcher can load once per track and pass across
    ratios (saves ~5-10 min wall on 48-case corpus).
    """
    if inputs is None:
        inputs = load_track_inputs_beatthis(track)
    dump_track(track, inputs_override=inputs, out_root=PHASE4_E2E_OUT)
    dump_viterbi_smoke(track, target_ratio=target_ratio,
                       inputs_override=inputs, out_root=PHASE4_E2E_OUT)
    dump_optimizer_smoke(track, target_ratio=target_ratio,
                         inputs_override=inputs, out_root=PHASE4_E2E_OUT)


# ---------------------------------------------------------------------------
# RegionOptimizer + RegionCost dump — session 23
# ---------------------------------------------------------------------------
#
# Region corpus parameterization (per session-23 Q5 user-approved):
#   - Canonical region = middle 50% of track (0.25·T → 0.75·T).
#   - Target ratios relative to REGION duration: {0.5, 0.8, 1.2}
#     → 0.5/0.8 = shortening, 1.2 = extending (exercises is_extending branch
#       with the +0.5 backward-jump penalty at region_optimizer.py:158).
#   - 16 tracks × 3 ratios = 48 cases.
#
# Region metadata keys (subset of CleanOptimizer's 10): RegionRemixMixin
# emits 8 keys max per transition (region_optimizer.py:342-354) — no
# `is_repetition_jump` or `chroma_correlation` markers.
_REGION_META_KEYS = (
    "quality_score",
    "waveform_similarity",
    "successor_similarity",
    "edge_splice_similarity",
    "chroma_distance",
    "energy_diff_db",
    "alignment_offset_sec",
    "total_cost",
)


def dump_region_smoke(track: str, target_ratio: float = 0.8,
                      *, no_structure: bool = False,
                      out_root: Path | None = None) -> None:
    """RegionOptimizer facade golden dump for one (track, target_ratio) pair.

    Middle-50% canonical region. target_ratio applies to REGION duration
    (0.5 = shortening, 1.2 = extending). Reuses the session-22 CleanOptimizer
    instance as the host for RegionRemixMixin.remix_region (Python inheritance
    composition). Writes under `<track>/region/r<R>/`.

    `no_structure=True` (ADR-044): segments=[] propagated through the full
    region pipeline (TransitionCost + RegionCost + RegionOptimizer) under the
    `no_structure_patches()` context manager, mirroring the C++ ADR-044
    semantic.
    """
    inputs = load_track_inputs(track)
    sr = inputs["sr"]

    segments   = [] if no_structure else inputs["segments"]
    beat_times = inputs["beat_times"]
    downbeats  = inputs["downbeats"]
    features   = inputs["features"].astype(np.float32)
    boundary_wf = inputs["boundary_waveforms"]

    cm = no_structure_patches() if no_structure else contextlib.nullcontext()
    with cm:
        # Reuse the session-22 TransitionCost + CleanOptimizer construction.
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

        # Canonical region: middle 50% by beat_times.
        if n_beats < 2:
            raise SystemExit(f"Track {track} has n_beats<2, cannot define region.")
        t_start = float(beat_times[0])
        t_end   = float(beat_times[-1])
        region_start_sec = t_start + 0.25 * (t_end - t_start)
        region_end_sec   = t_start + 0.75 * (t_end - t_start)
        region_duration  = region_end_sec - region_start_sec
        target_duration  = region_duration * target_ratio

        # Compute region-specific cost matrix via RegionCost module.
        # entry_beat / exit_beat semantics match region_optimizer.py:73-75.
        entry_b = int(np.argmin(np.abs(beat_times - region_start_sec)))
        exit_b  = int(np.argmin(np.abs(beat_times - region_end_sec)))
        exit_b  = max(entry_b + 1, min(exit_b, n_beats - 1))

        rgn_W, rgn_cands = compute_region_costs(
            entry_beat=entry_b,
            exit_beat=exit_b,
            features=features,
            beat_times=beat_times,
            boundary_waveforms=boundary_wf,
            waveform_sample_rate=sr,
            segments=segments,
            edge_features_start=inputs["edge_features_start"].astype(np.float32),
            edge_features_end=inputs["edge_features_end"].astype(np.float32),
            edge_rms_start=inputs["edge_rms_start"],
            edge_rms_end=inputs["edge_rms_end"],
            rms_energy=inputs["rms_energy"],
            onset_strength=inputs["onset_strength"],
            spectral_centroid=inputs["spectral_centroid"],
            vocal_activity=inputs["vocal_activity"],
            edge_vocal_activity_start=inputs["edge_vocal_activity_start"],
            edge_vocal_activity_end=inputs["edge_vocal_activity_end"],
            downbeats=downbeats if downbeats.size > 0 else None,
            time_signature=4,
        )

        # Call RegionRemixMixin.remix_region (inherited into CleanOptimizer).
        remix_path = optimizer.remix_region(
            target_duration=target_duration,
            region_start_sec=region_start_sec,
            region_end_sec=region_end_sec,
            region_W=rgn_W,
            region_candidates=rgn_cands,
        )

    n_region = exit_b - entry_b

    # --- Dump --------------------------------------------------------------
    out_dir = (out_root or PHASE4_OUT) / track / "region" / _ratio_subdir(target_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", arr)

    # RegionCost outputs (so RegionCost parity can be tested separately from
    # RegionOptimizer DP output if the pipeline fails end-to-end).
    # region_W stored flat (n_region × n_region), row-major f64.
    save("region_W", np.asarray(rgn_W, dtype=np.float64).reshape(-1))
    # Candidate arrays — (from_beat, to_beat) → TransitionCandidate.
    # Serialized as sorted-by-(from,to) flat arrays, mirroring session-18 TC dump.
    sorted_cand_items = sorted(rgn_cands.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1])))
    n_cands = len(sorted_cand_items)
    cand_pairs: list[int] = []
    cand_scalars: dict[str, list[float]] = {k: [] for k in _REGION_META_KEYS + ("alignment_lag_samples",)}
    for (i, j), cand in sorted_cand_items:
        cand_pairs.append(int(i))
        cand_pairs.append(int(j))
        cand_scalars["quality_score"].append(float(cand.quality_score))
        cand_scalars["waveform_similarity"].append(float(cand.waveform_similarity))
        cand_scalars["successor_similarity"].append(float(cand.successor_similarity))
        cand_scalars["edge_splice_similarity"].append(float(cand.edge_splice_similarity))
        cand_scalars["chroma_distance"].append(float(cand.chroma_distance))
        cand_scalars["energy_diff_db"].append(float(cand.energy_diff_db))
        # alignment_offset_sec is per-candidate derived field; store for ref.
        cand_scalars["alignment_offset_sec"].append(
            float(cand.alignment_lag_samples) / float(max(1, sr))
        )
        cand_scalars["total_cost"].append(float(cand.total_cost))
        cand_scalars["alignment_lag_samples"].append(float(cand.alignment_lag_samples))
    save("cand_pairs", np.asarray(cand_pairs, dtype=np.int64))
    for k, v in cand_scalars.items():
        save(f"cand_{k}", np.asarray(v, dtype=np.float64))

    # Primary RegionOptimizer facade outputs.
    save("region_path",       np.asarray(remix_path.beat_indices, dtype=np.int64))
    save("region_total_cost", np.asarray([float(remix_path.total_cost)], dtype=np.float64))

    # Metadata — sorted-by-(from,to) per session-22 convention.
    sorted_meta_items = sorted(
        remix_path.transition_metadata.items(),
        key=lambda kv: (int(kv[0][0]), int(kv[0][1])),
    )
    n_transitions = len(sorted_meta_items)
    meta_pairs_flat: list[int] = []
    for (i, j), _ in sorted_meta_items:
        meta_pairs_flat.append(int(i))
        meta_pairs_flat.append(int(j))
    save("meta_pairs", np.asarray(meta_pairs_flat, dtype=np.int64))

    def extract_key_values(key: str) -> np.ndarray:
        out = np.full(n_transitions, np.nan, dtype=np.float64)
        for idx, (_, meta) in enumerate(sorted_meta_items):
            if key in meta:
                out[idx] = float(meta[key])
        return out

    for key in _REGION_META_KEYS:
        save(f"meta_{key}", extract_key_values(key))

    manifest = {
        "track":                track,
        "target_ratio":         target_ratio,
        "target_duration":      float(target_duration),
        "region_start_sec":     float(region_start_sec),
        "region_end_sec":       float(region_end_sec),
        "region_duration":      float(region_duration),
        "entry_beat":           int(entry_b),
        "exit_beat":            int(exit_b),
        "n_region":             int(n_region),
        "n_beats":              int(n_beats),
        "n_candidates":         int(n_cands),
        "n_transitions":        int(n_transitions),
        "avg_beat_duration":    float(optimizer._avg_beat_duration),
        "duration_tolerance_sec": 5.0,
        "is_extending":         bool(target_duration > region_duration),
        "region_path_length":   int(len(remix_path.beat_indices)),
        "region_total_cost":    float(remix_path.total_cost),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[region/{track}] entry={entry_b} exit={exit_b} n_region={n_region} "
        f"ratio={target_ratio:.1f} path_len={len(remix_path.beat_indices)} "
        f"n_cands={n_cands} n_trans={n_transitions} "
        f"extending={target_duration > region_duration} "
        f"cost={remix_path.total_cost:.4f} dir={out_dir}"
    )


# ---------------------------------------------------------------------------
# Session 24 — TransitionPrescreen + BlockAssembly dumps
# ---------------------------------------------------------------------------

def dump_prescreen_smoke(track: str) -> None:
    """Prescreen golden dump for one track.

    Session-24. Prescreen is ratio-independent — one pass per track.
    """
    inputs = load_track_inputs(track)
    sr = inputs["sr"]

    segments   = inputs["segments"]
    beat_times = inputs["beat_times"]
    downbeats  = inputs["downbeats"]
    features   = inputs["features"].astype(np.float32)
    boundary_wf = inputs["boundary_waveforms"]

    results = prescreen_transitions(
        features=features,
        beat_times=beat_times,
        segments=segments,
        downbeats=downbeats if downbeats.size > 0 else None,
        time_signature=4,
        boundary_waveforms=boundary_wf,
        waveform_sample_rate=sr,
    )

    out_dir = PHASE4_OUT / track / "prescreen"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", arr)

    n = len(results)
    save("prescreen_from",      np.asarray([t.from_beat        for t in results], dtype=np.int64))
    save("prescreen_to",        np.asarray([t.to_beat          for t in results], dtype=np.int64))
    save("prescreen_diag_len",  np.asarray([t.diagonal_length  for t in results], dtype=np.int64))
    save("prescreen_rec_score", np.asarray([t.recurrence_score for t in results], dtype=np.float64))
    save("prescreen_wf_sim",    np.asarray([t.waveform_similarity for t in results], dtype=np.float64))

    manifest = {
        "track":             track,
        "n_prescreened":     int(n),
        "time_signature":    4,
        "k_neighbors":       12,  # CRITICAL CALLER OVERRIDE
        "has_waveform":      bool(boundary_wf is not None and sr > 0),
        "waveform_sample_rate": int(sr),
        "min_waveform_similarity": 0.30,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[prescreen/{track}] n_results={n} "
        f"max_diag_len={max((t.diagonal_length for t in results), default=0)} "
        f"dir={out_dir}"
    )


# Block assembly metadata keys emitted by assemble_blocks (block_assembly.py:498-502).
_BLOCK_META_KEYS = (
    "quality_score",
    "block_transition",
    "crossfade_ms",
)


def dump_block_assembly_smoke(track: str) -> None:
    """BlockAssembly golden dump for one track.

    Canonical test: block_sequence = [0, 1, 2, ..., n_blocks - 1]
    (segments in source order, no repetition, no reordering). variation = 0
    (best splice each junction). junction_variations = None.

    Session-24. BlockAssembly is ratio-independent — one pass per track.
    """
    inputs = load_track_inputs(track)
    sr = inputs["sr"]

    segments   = inputs["segments"]
    beat_times = inputs["beat_times"]
    downbeats  = inputs["downbeats"]
    features   = inputs["features"].astype(np.float32)
    boundary_wf = inputs["boundary_waveforms"]

    # Stage 1: build_blocks.
    blocks = build_blocks(segments, beat_times)

    # Stage 2: compute_block_compatibility.
    compat = compute_block_compatibility(
        blocks=blocks,
        beat_times=beat_times,
        features=features,
        boundary_waveforms=boundary_wf,
        waveform_sample_rate=sr,
        edge_features_start=inputs["edge_features_start"].astype(np.float32),
        edge_features_end=inputs["edge_features_end"].astype(np.float32),
        edge_rms_start=inputs["edge_rms_start"],
        edge_rms_end=inputs["edge_rms_end"],
        rms_energy=inputs["rms_energy"],
        spectral_centroid=inputs["spectral_centroid"],
        vocal_activity=inputs["vocal_activity"],
        downbeats=downbeats if downbeats.size > 0 else None,
        time_signature=4,
    )

    # Stage 3: assemble_blocks with canonical sequence.
    n_blocks = len(blocks)
    block_sequence = list(range(n_blocks))
    path = assemble_blocks(
        block_sequence=block_sequence,
        blocks=blocks,
        beat_times=beat_times,
        compat=compat,
        variation=0,
        junction_variations=None,
    )

    out_dir = PHASE4_OUT / track / "block_assembly"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", arr)

    # -- Block info ---------------------------------------------------------
    save("block_segment_idx", np.asarray([b.segment_idx for b in blocks], dtype=np.int64))
    save("block_start_beat",  np.asarray([b.start_beat  for b in blocks], dtype=np.int64))
    save("block_end_beat",    np.asarray([b.end_beat    for b in blocks], dtype=np.int64))
    save("block_n_beats",     np.asarray([b.n_beats     for b in blocks], dtype=np.int64))
    save("block_cluster_id",  np.asarray([b.cluster_id  for b in blocks], dtype=np.int64))
    save("block_start_sec",   np.asarray([b.start_sec   for b in blocks], dtype=np.float64))
    save("block_end_sec",     np.asarray([b.end_sec     for b in blocks], dtype=np.float64))
    save("block_duration_sec", np.asarray([b.duration_sec for b in blocks], dtype=np.float64))

    (out_dir / "block_labels.txt").write_text("\n".join(b.label for b in blocks) + "\n")
    (out_dir / "block_display_names.txt").write_text(
        "\n".join(b.display_name for b in blocks) + "\n")

    # -- Compatibility matrix ---------------------------------------------
    save("compat_quality",     compat.quality.astype(np.float64).reshape(-1))
    save("compat_splice_from", compat.splice_from.astype(np.int64).reshape(-1))
    save("compat_splice_to",   compat.splice_to.astype(np.int64).reshape(-1))
    save("compat_top_k_quality", compat.top_k_quality.astype(np.float64).reshape(-1))
    save("compat_top_k_from",    compat.top_k_from.astype(np.int64).reshape(-1))
    save("compat_top_k_to",      compat.top_k_to.astype(np.int64).reshape(-1))

    # -- Assembled path ----------------------------------------------------
    save("block_sequence",    np.asarray(block_sequence, dtype=np.int64))
    save("assembled_beat_indices", np.asarray(path.beat_indices, dtype=np.int64))
    save("assembled_total_cost",   np.asarray([float(path.total_cost)], dtype=np.float64))

    sorted_trans = sorted(path.transition_metadata.items(),
                          key=lambda kv: (int(kv[0][0]), int(kv[0][1])))
    trans_pairs_flat: list[int] = []
    for (fb, tb), _ in sorted_trans:
        trans_pairs_flat.append(int(fb))
        trans_pairs_flat.append(int(tb))
    save("assembled_trans_pairs", np.asarray(trans_pairs_flat, dtype=np.int64))

    n_trans = len(sorted_trans)
    for key in _BLOCK_META_KEYS:
        arr = np.full(n_trans, np.nan, dtype=np.float64)
        for idx, (_, meta) in enumerate(sorted_trans):
            if key in meta:
                arr[idx] = float(meta[key])
        save(f"assembled_meta_{key}", arr)

    # Manifest.
    manifest = {
        "track":           track,
        "n_blocks":        int(n_blocks),
        "n_beats":         int(len(beat_times)),
        "n_features":      int(features.shape[1]),
        "n_edge_features": int(inputs["edge_features_start"].shape[1]),
        "n_downbeats":     int(downbeats.size),
        "block_sequence":  block_sequence,
        "variation":       0,
        "top_k":           int(_BLOCK_TOP_K),
        "path_len":        int(len(path.beat_indices)),
        "n_transitions":   int(n_trans),
        "total_cost":      float(path.total_cost),
        "time_signature":  4,
        "waveform_sample_rate": int(sr),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[block_assembly/{track}] n_blocks={n_blocks} "
        f"path_len={len(path.beat_indices)} n_trans={n_trans} "
        f"total_cost={path.total_cost:.4f} dir={out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track",
        default=None,
        help="Track name; default = iterate over all 16 TRACK_LIST entries.",
    )
    parser.add_argument(
        "--viterbi-smoke",
        action="store_true",
        help=("Also dump ViterbiDP goldens for the selected track at "
              "--target-ratio (default 0.5). Requires --track."),
    )
    parser.add_argument(
        "--viterbi-corpus",
        action="store_true",
        help=("Dump ViterbiDP goldens for all 16 TRACK_LIST entries at every "
              "ratio in --target-ratios. Produces 48 cases by default."),
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.5,
        help="Target duration ratio vs original (single-track smoke, default 0.5).",
    )
    parser.add_argument(
        "--target-ratios",
        default="0.3,0.5,0.7",
        help=("Comma-separated target ratios for --viterbi-corpus mode "
              "(default 0.3,0.5,0.7 per phase-4 acceptance criterion (a))."),
    )
    parser.add_argument(
        "--optimizer-smoke",
        action="store_true",
        help=("Also dump CleanOptimizer facade goldens for the selected "
              "track at --target-ratio (default 0.5). Requires --track. "
              "Layers on top of ViterbiDP dump (session 22)."),
    )
    parser.add_argument(
        "--optimizer-corpus",
        action="store_true",
        help=("Dump CleanOptimizer facade goldens for all 16 TRACK_LIST "
              "entries at every ratio in --target-ratios. 48 cases by "
              "default. Layers on top of ViterbiDP dump (session 22)."),
    )
    parser.add_argument(
        "--region-smoke",
        action="store_true",
        help=("Also dump RegionOptimizer + RegionCost goldens for the "
              "selected track at --target-ratio (default 0.8). Requires "
              "--track. Canonical region = middle 50% of track. Session 23."),
    )
    parser.add_argument(
        "--region-corpus",
        action="store_true",
        help=("Dump RegionOptimizer + RegionCost goldens for all 16 "
              "TRACK_LIST entries at every ratio in --region-target-ratios "
              "(default 0.5,0.8,1.2 — covers shortening + extending). "
              "48 cases by default. Session 23."),
    )
    parser.add_argument(
        "--region-target-ratios",
        default="0.5,0.8,1.2",
        help=("Comma-separated target ratios for --region-corpus mode "
              "applied to REGION duration (default 0.5,0.8,1.2 — 0.5/0.8 "
              "shortening, 1.2 extending)."),
    )
    parser.add_argument(
        "--prescreen-smoke",
        action="store_true",
        help=("Dump TransitionPrescreen goldens for the selected track "
              "(ratio-independent, one pass per track). Requires --track. "
              "Session 24."),
    )
    parser.add_argument(
        "--prescreen-corpus",
        action="store_true",
        help=("Dump TransitionPrescreen goldens for all 16 TRACK_LIST "
              "entries. 16 cases. Session 24."),
    )
    parser.add_argument(
        "--blockassembly-smoke",
        action="store_true",
        help=("Dump BlockAssembly goldens (build + compat + assemble) for "
              "the selected track with canonical block_sequence "
              "[0, 1, ..., n_blocks-1]. Requires --track. Session 24."),
    )
    parser.add_argument(
        "--blockassembly-corpus",
        action="store_true",
        help=("Dump BlockAssembly goldens for all 16 TRACK_LIST entries "
              "with canonical block_sequence. 16 cases. Session 24."),
    )
    parser.add_argument(
        "--optimizer-e2e-smoke",
        action="store_true",
        help=("ADR-027 integration parity smoke — runs the full pipeline on "
              "the C++ BeatDetector grid (beats_cpp) for one (track, ratio) "
              "pair. Writes to tests/parity/reference/data/phase4_e2e/. "
              "Requires --track. Session 26."),
    )
    parser.add_argument(
        "--optimizer-e2e-corpus",
        action="store_true",
        help=("ADR-027 integration parity corpus — runs the full pipeline on "
              "the C++ BeatDetector grid for all 16 TRACK_LIST entries at "
              "every ratio in --target-ratios (48 cases by default). Caches "
              "FeatureExtractor output once per track across ratios. Writes "
              "to tests/parity/reference/data/phase4_e2e/. Session 27."),
    )
    parser.add_argument(
        "--variations-smoke",
        action="store_true",
        help=("ADR-048 / DEV-027: dump CleanOptimizer.remix_k_best goldens "
              "for the selected track at --target-ratio with --variations-k "
              "paths (default 4). Writes to "
              "<phase4>/<track>/optimizer/r<R>/variations/v<N>/. Requires "
              "--track. Session 58."),
    )
    parser.add_argument(
        "--variations-corpus",
        action="store_true",
        help=("ADR-048 / DEV-027: dump CleanOptimizer.remix_k_best goldens "
              "for all tracks in --variations-tracks (default = sesja-58 "
              "in-session corpus billie_jean + woodkid_iron_acoustic) at "
              "every ratio in --target-ratios with --variations-k paths. "
              "Full 16-track gate deferred to phase-6 close per ADR-034. "
              "Session 58."),
    )
    parser.add_argument(
        "--variations-k",
        type=int,
        default=4,
        help=("k for --variations-{smoke,corpus} dumps (default 4 = sesja-58 "
              "parity test scope: best + 3 alternatives)."),
    )
    parser.add_argument(
        "--variations-tracks",
        default="billie_jean,woodkid_iron_acoustic",
        help=("Comma-separated track list for --variations-corpus mode "
              "(default sesja-58 in-session 2-track gate)."),
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help=("ADR-044 close-out: dump TransitionCost goldens for all 16 "
              "tracks PLUS RegionCost goldens for 16 tracks × "
              "--region-target-ratios (default 0.5,0.8,1.2 = 48 cases) under "
              "the no-structure semantic (segments=[], section_sim=0, "
              "label_match=0, SPAN_PENALTY_*=0 — mirrors C++ AnalyzeWorker "
              "skip-StructureAnalyzer + TransitionCost.cpp/RegionCost.cpp "
              "noStructure gating). Writes to "
              "tests/parity/reference/data/phase4_no_structure/. Session 53."),
    )
    args = parser.parse_args()

    if args.viterbi_smoke:
        if args.track is None:
            raise SystemExit("--viterbi-smoke requires --track <name>.")
        dump_track(args.track)              # also refresh TransitionCost inputs
        dump_viterbi_smoke(args.track, target_ratio=args.target_ratio)
        return

    if args.viterbi_corpus:
        ratios = [float(x) for x in args.target_ratios.split(",") if x.strip()]
        if not ratios:
            raise SystemExit("--target-ratios produced empty list.")
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(
            f"=== ViterbiDP corpus dump: {len(tracks)} tracks × "
            f"{len(ratios)} ratios = {len(tracks) * len(ratios)} cases ==="
        )
        for t in tracks:
            dump_track(t)                   # refresh TransitionCost inputs once per track
            for r in ratios:
                dump_viterbi_smoke(t, target_ratio=r)
        return

    if args.optimizer_smoke:
        if args.track is None:
            raise SystemExit("--optimizer-smoke requires --track <name>.")
        dump_track(args.track)              # refresh TransitionCost inputs
        dump_viterbi_smoke(args.track, target_ratio=args.target_ratio)
        dump_optimizer_smoke(args.track, target_ratio=args.target_ratio)
        return

    if args.optimizer_corpus:
        ratios = [float(x) for x in args.target_ratios.split(",") if x.strip()]
        if not ratios:
            raise SystemExit("--target-ratios produced empty list.")
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(
            f"=== Optimizer corpus dump: {len(tracks)} tracks × "
            f"{len(ratios)} ratios = {len(tracks) * len(ratios)} cases ==="
        )
        for t in tracks:
            dump_track(t)                   # refresh TransitionCost inputs once per track
            for r in ratios:
                dump_viterbi_smoke(t, target_ratio=r)
                dump_optimizer_smoke(t, target_ratio=r)
        return

    if args.variations_smoke:
        if args.track is None:
            raise SystemExit("--variations-smoke requires --track <name>.")
        dump_track(args.track)              # refresh TransitionCost inputs
        dump_viterbi_smoke(args.track, target_ratio=args.target_ratio)
        dump_optimizer_smoke(args.track, target_ratio=args.target_ratio)
        dump_optimizer_variations_smoke(
            args.track, target_ratio=args.target_ratio, k=args.variations_k,
        )
        return

    if args.variations_corpus:
        ratios = [float(x) for x in args.target_ratios.split(",") if x.strip()]
        if not ratios:
            raise SystemExit("--target-ratios produced empty list.")
        tracks = [t.strip() for t in args.variations_tracks.split(",") if t.strip()]
        if not tracks:
            raise SystemExit("--variations-tracks produced empty list.")
        print(
            f"=== Variations corpus dump: {len(tracks)} tracks × "
            f"{len(ratios)} ratios × k={args.variations_k} ==="
        )
        for t in tracks:
            dump_track(t)
            for r in ratios:
                dump_viterbi_smoke(t, target_ratio=r)
                dump_optimizer_smoke(t, target_ratio=r)
                dump_optimizer_variations_smoke(t, target_ratio=r, k=args.variations_k)
        return

    if args.region_smoke:
        if args.track is None:
            raise SystemExit("--region-smoke requires --track <name>.")
        dump_track(args.track)
        # Use --target-ratio (default 0.5 globally, but 0.8 is the region smoke
        # canonical — caller can override via --target-ratio).
        region_ratio = args.target_ratio if args.target_ratio != 0.5 else 0.8
        dump_region_smoke(args.track, target_ratio=region_ratio)
        return

    if args.region_corpus:
        ratios = [float(x) for x in args.region_target_ratios.split(",") if x.strip()]
        if not ratios:
            raise SystemExit("--region-target-ratios produced empty list.")
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(
            f"=== Region corpus dump: {len(tracks)} tracks × "
            f"{len(ratios)} ratios = {len(tracks) * len(ratios)} cases ==="
        )
        for t in tracks:
            dump_track(t)                   # refresh TransitionCost inputs once per track
            for r in ratios:
                dump_region_smoke(t, target_ratio=r)
        return

    if args.prescreen_smoke:
        if args.track is None:
            raise SystemExit("--prescreen-smoke requires --track <name>.")
        dump_track(args.track)
        dump_prescreen_smoke(args.track)
        return

    if args.prescreen_corpus:
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(f"=== Prescreen corpus dump: {len(tracks)} tracks ===")
        for t in tracks:
            dump_track(t)
            dump_prescreen_smoke(t)
        return

    if args.blockassembly_smoke:
        if args.track is None:
            raise SystemExit("--blockassembly-smoke requires --track <name>.")
        dump_track(args.track)
        dump_block_assembly_smoke(args.track)
        return

    if args.blockassembly_corpus:
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(f"=== BlockAssembly corpus dump: {len(tracks)} tracks ===")
        for t in tracks:
            dump_track(t)
            dump_block_assembly_smoke(t)
        return

    if args.optimizer_e2e_smoke:
        if args.track is None:
            raise SystemExit("--optimizer-e2e-smoke requires --track <name>.")
        print(
            f"=== ADR-027 E2E smoke: {args.track} @ ratio={args.target_ratio} "
            f"(out: {PHASE4_E2E_OUT}) ==="
        )
        dump_optimizer_e2e_smoke(args.track, target_ratio=args.target_ratio)
        return

    if args.optimizer_e2e_corpus:
        ratios = [float(x) for x in args.target_ratios.split(",") if x.strip()]
        if not ratios:
            raise SystemExit("--target-ratios produced empty list.")
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(
            f"=== ADR-027 E2E corpus dump: {len(tracks)} tracks × "
            f"{len(ratios)} ratios = {len(tracks) * len(ratios)} cases "
            f"(out: {PHASE4_E2E_OUT}) ==="
        )
        for t in tracks:
            # Cache once per track — FeatureExtractor is the expensive step
            # (~10-20 s/track) and inputs are ratio-independent.
            inputs = load_track_inputs_beatthis(t)
            for r in ratios:
                dump_optimizer_e2e_smoke(t, target_ratio=r, inputs=inputs)
        return

    if args.no_structure:
        # ADR-044 close-out: status-quo `phase4/` corpus stays untouched;
        # this dumps a parallel `phase4_no_structure/` tree with empty
        # segments + Python pipeline patched to match the C++ ADR-044
        # semantic. TransitionCost (1 dump per track) + RegionCost (3 ratios
        # per track from --region-target-ratios).
        region_ratios = [float(x) for x in args.region_target_ratios.split(",") if x.strip()]
        if not region_ratios:
            raise SystemExit("--region-target-ratios produced empty list.")
        tracks = [args.track] if args.track is not None else TRACK_LIST
        print(
            f"=== ADR-044 no-structure dump: {len(tracks)} tracks → "
            f"TransitionCost + RegionCost × {len(region_ratios)} ratios "
            f"(out: {PHASE4_NS_OUT}) ==="
        )
        ns_summaries = []
        for t in tracks:
            ns_summaries.append(
                dump_track(t, no_structure=True, out_root=PHASE4_NS_OUT)
            )
            for r in region_ratios:
                dump_region_smoke(t, target_ratio=r,
                                  no_structure=True, out_root=PHASE4_NS_OUT)
        print("")
        print("=== No-structure corpus summary ===")
        for s in ns_summaries:
            print(
                f"  {s['track']:<32} n_beats={s['n_beats']:>4}  "
                f"n_cands={s['n_candidates']:>5}  finite_W={s['finite_W']:>5}  "
                f"n_seg={s['n_segments']:>3}  n_db={s['n_downbeats']:>3}  "
                f"voc={int(s['has_vocals'])}"
            )
        return

    if args.track is not None:
        dump_track(args.track)
        return

    summaries = []
    for t in TRACK_LIST:
        summaries.append(dump_track(t))

    # Corpus summary
    print("")
    print("=== Corpus summary ===")
    for s in summaries:
        print(
            f"  {s['track']:<32} n_beats={s['n_beats']:>4}  "
            f"n_cands={s['n_candidates']:>5}  finite_W={s['finite_W']:>5}  "
            f"n_seg={s['n_segments']:>3}  n_db={s['n_downbeats']:>3}  "
            f"voc={int(s['has_vocals'])}"
        )


if __name__ == "__main__":
    main()
