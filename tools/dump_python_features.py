#!/usr/bin/env python3
"""
dump_python_features.py — Python librosa reference dumps for phase-2 parity tests.

Emits .npy per intermediate feature on one audio file (or looped over a manifest).
Each run writes a per-track manifest recording the SHA-256 of every .npy plus the
library version pins, so a future regeneration on a different librosa/scipy/numpy
version produces a visible diff in manifest.json at parity-test time.

Pinned versions (mandatory, see tools/requirements-phase2.txt):
  librosa==0.11.0
  scipy==1.15.3
  numpy>=1.24,<2.0

Audio is loaded mono at sr=22050 (phase-2 default per config.AnalysisConfig).
Intermediate features:
  stft_magnitude          # |librosa.stft(y, n_fft=2048, hop=512)|
  mel_power               # librosa.feature.melspectrogram power=2.0
  log_mel                 # librosa.power_to_db(mel_power, ref=np.max)
  log_mel_ref1            # librosa.power_to_db(mel_power, ref=1.0) — matches
                          # librosa.feature.mfcc's internal call; feeds DCT
                          # parity test to isolate the DCT step.
  mfcc_40                 # librosa.feature.mfcc n_mfcc=40 (standard vector)
  mfcc_20                 # librosa.feature.mfcc n_mfcc=20 (fast vector)
  chroma_stft             # librosa.feature.chroma_stft (default tuning=None!)
  tuning_est              # librosa.estimate_tuning(S=|STFT|², sr, bpo=12) — the
                          # exact scalar chroma_stft(tuning=None) computes
                          # internally. Shape (1,) float64 for .npy round-trip.
  spectral_contrast       # librosa.feature.spectral_contrast (7 rows: bands 1..n_bands)
  onset_strength          # librosa.onset.onset_strength
  rms                     # librosa.feature.rms
  spectral_centroid       # librosa.feature.spectral_centroid
  beat_frames             # synthetic beat frames np.arange(10, T, 20) (int64),
                          # test fixture only — NOT real beats. Stored as
                          # <i8> .npy; chosen to exercise BeatSync's boundary
                          # padding + dedupe at a realistic ~130 BPM density
                          # (hop=512 sr=22050 → 20 frames ≈ 0.465 s).
  beat_times              # beat_frames * hop / sr (float64). The inverse of
                          # librosa.time_to_frames on integer beat_frames
                          # that sit at bin centres, so re-applying
                          # time_to_frames round-trips exactly. Used by the
                          # step-9 helpers (edge_rms, waveform snippets) to
                          # exercise the `(beat_times * sr).astype(int64)`
                          # truncation convention.
  beat_sync_mel_power     # librosa.util.sync(mel_power, beat_frames,
                          # aggregate=np.mean, pad=True, axis=-1) — 2-D ref
                          # for BeatSync parity test.
  edge_features_start     # step-9 _extract_edge_features on mel_power cast
  edge_features_end       # to float32, with n_edge_frames=4. Output f64
                          # per-beat L2-normalized (n_beats, n_mels=128).
                          # Uses mel_power (not the orchestrator's 59-dim
                          # stacked matrix) as the fixture feature matrix —
                          # phase 2 has no orchestrator yet, and the helper
                          # is generic over the feature dimension.
  edge_rms_start          # step-9 _extract_edge_rms on y + beat_times,
  edge_rms_end            # edge_ms=50.0. Output f64, jointly normalized
                          # to [0, 1] by the shared max of both arrays.
  boundary_waveforms      # step-9 _extract_waveform_snippets, pre_ms=35,
                          # post_ms=120. Output f32 (n_beats, ~3417 samples
                          # at sr=22050).
  transition_waveforms    # step-9 _extract_waveform_snippets, pre_ms=280,
                          # post_ms=320. Output f32 (n_beats, ~13230 samples
                          # at sr=22050).

Step-10 orchestrator reference arrays (added session 14; inline mirror of
feature_extractor.py::FeatureExtractor._extract_internal L189-254 operating
on the synthetic beats above):

  orch_std_feature_matrix      # (n_beats, 59) float64 — L2-normalized per row.
                                 Standard mode: vstack(mfcc_40, chroma, contrast).
  orch_std_edge_features_start # (n_beats, 59) float64 — edges on the 59-dim
  orch_std_edge_features_end     stacked matrix (DISTINCT from the existing
                                 edge_features_{start,end}.npy which are on
                                 mel_power as a step-9-era fixture).
  orch_std_rms_energy          # (n_beats,) float64 — per-track max-normalized
                                 to [0, 1] with 1e-10 floor.
  orch_std_onset_strength      # same layout and normalization.
  orch_std_spectral_centroid   # same.
  orch_fast_feature_matrix     # (n_beats, 39) float64 — fast mode (mfcc_20 +
                                 chroma + contrast). transitionWaveforms
                                 intentionally OMITTED per
                                 config.AnalysisConfig.fast_skip_transition_waveforms.
  orch_fast_edge_features_start# (n_beats, 39) float64.
  orch_fast_edge_features_end

Phase-2b vocal reference arrays (added session 2; default-mode only per ADR-014;
synthetic-beats fixture only — the `--real-beats` path does NOT emit vocal refs):

  hpss_y_harmonic              # f32 time-domain. librosa.effects.hpss(y)[0].
                                 Length matches len(y) via default ISTFT
                                 length-reconciliation.

Phase-3 structure-analysis reference arrays (added session 2; ADR-019 novelty
feature pipeline + optional cheap-bonus SSM for future NoveltySegmenter
bisection). Per session-2 scope discipline the R_matrix / segments.json /
boundaries dumps live in later sessions (see phases/phase-3-structure/log.md):

  novelty_features_25d         # f32 (25, T). Port target for
                                 src/analysis/NoveltyFeatures — mirror of
                                 StructureAnalyzer._extract_features
                                 (structure_analyzer.py L193-210): MFCC(13)
                                 + ChromaSTFT(12), per-row z-score with
                                 ddof=0 std + 1e-8 guard, vstack([mfcc*1.5,
                                 chroma]). Produced on SAME sr=22050 /
                                 hop=512 / n_fft=2048 as phase-2 keys —
                                 reuses existing STFT/mel/DCT/chroma pins.
  recurrence_beat_times        # f64 (n_rec,). Beats from librosa.beat.beat_track(y)
                                 for the phase-3 Recurrence parity fixture. Per
                                 ADR-018 consequences: module parity uses these
                                 beats on BOTH the Python dump and the C++ test
                                 (fed from the same .npy goldens) — production
                                 beat-this path is validated separately in
                                 phase-4 integration testing. Beat source in
                                 this dump is module-parity-irrelevant; only
                                 the distribution (realistic non-uniform
                                 spacing, phase-2 Fixture C precedent) matters.
  recurrence_feature_matrix    # f64 (n_rec, 59). Feature matrix from
                                 _run_orchestrator on recurrence_beat_times —
                                 L2-normalized per row, 59-dim standard. Same
                                 numerical path as orch_std_feature_matrix, but
                                 on real beats rather than synthetic
                                 arange(10, T, 20). Stored as f64 for npy
                                 round-trip; bit-exact f32 cast on the
                                 consumer side.
  R_matrix                     # f64 (n_rec, n_rec). Combined recurrence
                                 matrix R = 0.15·R_feat + 0.35·R_chroma +
                                 0.50·R_homo per recurrence.py:65. Primary
                                 parity target for C++ Recurrence::build.
  R_feat                       # f64 (n_rec, n_rec). Mutual-kNN recurrence on
                                 full feature vectors. Bisection aid only.
  R_chroma                     # f64 (n_rec, n_rec). Mutual-kNN recurrence on
                                 chroma sub-vector (cols 40..52). Bisection.
  R_homo                       # f64 (n_rec, n_rec). Gaussian kernel on
                                 consecutive-beat distances (|i-j|=1 only).
                                 Bisection aid.
  novelty_ssm_stride4          # f32 (T_ds, T_ds) where T_ds = ceil(T/4).
                                 Standard-mode self-similarity matrix from
                                 StructureAnalyzer._compute_ssm (L212-225):
                                 norms = L2(features_ds, axis=0) + 1e-8,
                                 ssm = clip(features_norm.T @ features_norm,
                                 0, 1). Bonus dump — not consumed by
                                 session-2 NoveltyFeatures port, but the
                                 matmul is numerically trivial (naive f32
                                 inner product) so regenerating later is
                                 both cheap AND bisection-useful when
                                 NoveltySegmenter lands in a future session.
  novelty_curve                # f64 (T_ds,). Session-7 NoveltySegmenter
                                 pre-clustering dump per ADR-021. Replay of
                                 StructureAnalyzer._compute_novelty (L227-251):
                                 Foote checkerboard kernel O(n²) + uniform_filter1d
                                 size=5 (scipy.ndimage default mode='reflect') +
                                 normalize by max when max>0. Parity gate
                                 ≤ 1e-9 L∞ (f64 arithmetic, stride-4 region ~
                                 32×32 = 1024 f64 mults per frame).
  novelty_peaks_raw            # i64 (n_peaks,). Raw output of
                                 scipy.signal.find_peaks(novelty, distance,
                                 prominence=0.20, height=percentile(novelty, 60))
                                 BEFORE margin clip. Bisection aid for C++
                                 find_peaks port. Integer-exact gate.
  novelty_boundaries           # f64 (n_segments+1,). Post-margin-clip
                                 boundaries [0.0, t1, ..., duration]. Margin
                                 = 2.0 s (Python L271). This is the final
                                 segmentation the novelty path hands to the
                                 segment-embeddings stage. Parity gate bitwise.
  novelty_embeddings           # f32 (n_segments, 25). Per-segment mean of
                                 25-dim novelty features sliced by boundaries
                                 converted to frames via sr/hop_length.
                                 StructureAnalyzer._compute_segment_embeddings
                                 (L277-296). Session-8 clustering consumes
                                 this. Parity gate bitwise.
  novelty_affinity             # f32 (n_segs, n_segs). clip(norm_embed @ norm_embed.T,
                                 0, 1) — the clipped-cosine similarity matrix
                                 consumed by SpectralClustering(affinity="precomputed")
                                 and silhouette_score(metric="precomputed").
                                 Session-8 ADR-021 bisection aid: C++ affinity
                                 L∞ ≤ 1e-6 expected (inherits embeddings bitwise
                                 gate + f32 matmul ULP). Bit-exactness NOT
                                 required by main gate.
  novelty_silhouette_scores    # f64 (max_k-2,) = (len(range(2, min(7, n_segs))),).
                                 Silhouette score per k, computed on DISTANCE
                                 matrix `1 - affinity` per ADR-022 Option B
                                 (corrects Python L332 which passed the
                                 similarity matrix directly, raising ValueError
                                 on every k under the pinned sklearn). -1.0
                                 sentinel for k values where clustering
                                 produced <2 unique labels. Empty when
                                 n_segs ≤ 2. Bisection aid for parity.
  novelty_best_k               # i64 0-d scalar. The k picked by silhouette
                                 argmax on `1 - affinity` distance (ADR-022
                                 fix). Values observed on the 10-track corpus:
                                 k=2 on 3 tracks (daft_punk, miles_davis,
                                 eminem), k=3 on 5 tracks (billie_jean,
                                 smells_like, shostakovich, tiesto, meshuggah),
                                 k=4 on 2 tracks (dance_monkey, vocal_solo) —
                                 adaptive to musical structure. Bisection aid.
  novelty_cluster_ids          # i64 (n_segs,). Final cluster assignment from
                                 _cluster_segments. Values in [0, best_k). NOT
                                 bitwise-comparable across implementations —
                                 the C++ parity gate is Hungarian-matched
                                 agreement ≥ 90 % (ADR-021 § Parity gate).
  novelty_segments_fields      # f64 (n_segs, 4). Per-segment [start, end,
                                 confidence, cluster_id] from _create_segments
                                 (L369-475). start/end from novelty_boundaries
                                 (bitwise), confidence computed from novelty peak
                                 height at boundary_frame (i==0: 0.8 literal;
                                 i>0: min(1.0, 0.5 + 0.5 * novelty[b_frame])).
                                 Gate: start/end bitwise, confidence L∞ ≤ 1e-6.
  novelty_segments_labels      # List[str] → .txt. One label per segment: intro,
                                 outro (position-override 15%/85%) or chorus /
                                 verse / bridge (energy-rank on repeating
                                 clusters, Counter.most_common fallback if no
                                 cluster repeats). Gate: exact string match ≥ 90 %
                                 (label depends on cluster_ids which is not
                                 bitwise, so 1 segment per track may flip).
  cbm_consolidated_segments_fields     # f64 (n', 4). SegmentConsolidation
                                         applied to CBM-path segments, mirrors
                                         dispatcher L151 (structure_analyzer.py).
                                         Port target: src/analysis/SegmentConsolidation.
                                         Gate: strict bitwise (deterministic
                                         sequential list processing; no RNG).
  cbm_consolidated_segments_labels     # List[str] → .txt. Same, label column.
  novelty_consolidated_segments_fields # f64 (n'', 4). SegmentConsolidation
                                         applied to novelty-path segments,
                                         mirrors dispatcher L172. Gate: strict
                                         bitwise.
  novelty_consolidated_segments_labels # List[str] → .txt. Same, label column.
  dispatched_path              # List[str] len=1 → .txt. Single line "cbm" or
                                 "novelty" — which branch of structure_analyzer.py
                                 L127-182 `_analyze` fires on this track. CBM
                                 fires when `len(downbeats) >= 4 AND beat_features
                                 present AND cbm_analyze returns >= 5 segments`;
                                 else novelty fallback. Session-10 port target:
                                 src/analysis/StructureAnalyzer::analyze.
  dispatched_segments_fields   # f64 (n, 4). Post-consolidation segments from
                                 whichever branch dispatched (cbm/novelty). Same
                                 layout as cbm_/novelty_consolidated_segments_fields
                                 but selected by predicate. Gate: strict bitwise
                                 on CBM-dispatched tracks (session-6 precedent);
                                 ADR-022 Option B ≥ 70 % Hungarian cluster +
                                 ≥ 70 % label on novelty-dispatched tracks
                                 (session-8 precedent).
  dispatched_segments_labels   # List[str] → .txt. Same, label column.
  dispatched_boundaries        # f64 (n+1,). Rebuilt per L152-155 / L174-177
                                 as `[segments[0].start] + [seg.end for seg in
                                 segments]` after consolidation. Bitwise gate
                                 (deterministic function of dispatched segments).
  spectral_flatness            # f32 per frame. librosa.feature.spectral_flatness(
                                 y=y_harmonic, n_fft=2048, hop=512)[0].
  harmonic_ratio               # f64 per frame. clip(rms(y_harm)/max(rms(y),1e-8)
                                 / 1.15, 0, 1).
  flatness_inv                 # f64 per frame. 1 - flatness/p95(flatness)
                                 (percentile method='linear'), clipped [0,1].
  voice_band_ratio             # f64 per frame. mean(|STFT[250..3400Hz]|) /
                                 mean(|STFT[80..7000Hz]|), p95-scaled, clipped.
  vocal_activity_frame         # f64 per frame. 0.40·harm + 0.33·vb + 0.27·flat,
                                 clipped [0,1]. Default-mode composite proxy.
  vocal_rise_frame             # f64 per frame. clip(diff(va, prepend=va[:1]),0,∞)
                                 normalized by p95, clipped [0,1].
  vocal_fall_frame             # f64 per frame. clip(-diff(...), 0, ∞), p95.
  vocal_activity               # f64 per beat. _beat_sync_1d(va_frame).
  voiced_ratio                 # f64 per beat. Zero-filled (ADR-014 D4).
  f0_hz                        # f64 per beat. Zero-filled (ADR-014 D2).
  f0_confidence                # f64 per beat. Zero-filled (ADR-014 D3).
  edge_vocal_activity_start    # f64 per beat. Mean-edge first/last.
  edge_vocal_activity_end
  edge_vocal_onset_start       # f64 per beat. Peak-edge of vocal_rise_frame.
  edge_vocal_release_end       # f64 per beat. Peak-edge of vocal_fall_frame.

Real-beats fixture (Fixture B per session-14 plan — CLI `--real-beats`):

  Runs in a distinct output dir (e.g. dumps/billie_jean_real/). Generates
  beat_times via `librosa.beat.beat_track(y)` — which produces realistic
  float-drift timings (e.g. 12.4783 s) that exercise the
  `librosa.time_to_frames` round-half-to-even edge cases that synthetic
  beats (arange × hop / sr) round-trip past without exercising. The source
  of beats is NOT the C++ phase-1 BeatDetector: the orchestrator is
  beat-source-agnostic, and using the same beat_times on both sides of the
  parity test isolates orchestrator drift from beat-detection drift. Real-
  beats dumps have names identical to the synthetic-beats orchestrator
  dumps above but sit under a separate per-track directory.

Usage:
    # One track, explicit paths (synthetic-beats fixture):
    python tools/dump_python_features.py \\
        --audio references/golden/phase-1/billie_jean.mp3 \\
        --name billie_jean \\
        --clip-start 60 --clip-end 120 \\
        --out references/golden/phase-2/dumps/billie_jean

    # Loop over every resolvable track in manifest.json:
    python tools/dump_python_features.py \\
        --from-manifest references/golden/phase-2/manifest.json

    # Real-beats fixture (billie_jean only; writes under a distinct dir):
    python tools/dump_python_features.py \\
        --audio references/golden/phase-1/billie_jean.mp3 \\
        --name billie_jean_real \\
        --clip-start 60 --clip-end 120 \\
        --real-beats \\
        --out references/golden/phase-2/dumps/billie_jean_real
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from collections import Counter
from dataclasses import dataclass

import librosa
import numpy as np
import scipy
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# --- version guard -----------------------------------------------------------
REQUIRED = {
    "librosa": "0.11.0",
    "scipy":   "1.15.3",
    # numpy window is >=1.24,<2.0; we only assert upper bound matters
}


def _assert_pins() -> None:
    """Abort if pins are violated. Pins protect .npy SHA stability."""
    got = {"librosa": librosa.__version__, "scipy": scipy.__version__}
    mismatches = [k for k, v in REQUIRED.items() if got[k] != v]
    if mismatches:
        msg = ["VERSION PIN MISMATCH (would silently drift .npy SHAs):"]
        for k in mismatches:
            msg.append(f"  {k}: expected {REQUIRED[k]}, got {got[k]}")
        msg.append("Install exact pins: pip install -r tools/requirements-phase2.txt")
        raise SystemExit("\n".join(msg))
    if np.__version__.startswith("2."):
        raise SystemExit(f"numpy {np.__version__} outside pinned range (need <2.0)")


# --- config (mirrors references/python-source/config.py AnalysisConfig) ------
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128           # librosa default for melspectrogram
N_MFCC_STD = 40
N_MFCC_FAST = 20
N_CONTRAST_BANDS = 6   # librosa default; returns bands 1..n_bands, so 7 rows total

# Step-9 constants mirror references/python-source/config.py:62-65 +
# feature_extractor.py default args. Inlined here (rather than imported)
# to keep the dump tool independent of the Python reference package's
# heavier imports (audio_cache, vocal_features).
BOUNDARY_PRE_MS  = 35.0
BOUNDARY_POST_MS = 120.0
TRANSITION_PRE_MS  = 280.0
TRANSITION_POST_MS = 320.0
N_EDGE_FRAMES = 4      # feature_extractor.py:322 default
EDGE_MS = 50.0         # feature_extractor.py:359 default

# --- ADR-020: C++ beat_detector_cli integration for CBM downbeat source -----
# Paths resolved once at module-load; overridable via env vars so CI or
# alternate checkouts can point elsewhere without editing this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CPP_CLI_PATH = Path(os.environ.get(
    "REAMIX_CPP_CLI",
    str(_REPO_ROOT / "build" / "beat_detector_cli"),
))
_BEAT_MODEL_PATH = Path(os.environ.get(
    "REAMIX_BEAT_MODEL",
    str(Path.home() / "Library" / "Application Support" / "reamix"
        / "models" / "beat_this_final0.onnx"),
))
_CBM_MODULE_PATH = (
    _REPO_ROOT / "references" / "python-source" / "analysis" / "cbm_segmenter.py"
)
_cbm_module_cache: object | None = None

# --- Phase-3 session-11 RepetitionMap reference import ------------------------
# `repetition_map.py` has top-level `from remix_tool.config import chroma_range`
# plus lazy imports of `remix_tool.analysis.recurrence` and
# `remix_tool.remix.transition_cost` inside try-blocks. Stubbing four package
# shims is more LOC than a single sys.path insertion; .venv-phase2 has the
# full remix_tool dependency tree (torch transitively) already available from
# the phase-1 beat_detector_cli orchestration, so direct import is free.
# Resolved parent of the `python-source` symlink is `.../RemixTool/src` which
# contains the `remix_tool/` package directory — adding it to sys.path makes
# `import remix_tool.*` importable without disturbing the ORPHAN pattern used
# for cbm_segmenter (which still avoids pulling remix_tool.__init__).
_PY_SRC_SYMLINK = _REPO_ROOT / "references" / "python-source"
try:
    _REMIX_TOOL_PKG_PARENT = _PY_SRC_SYMLINK.resolve().parent
    if _REMIX_TOOL_PKG_PARENT.is_dir() and str(_REMIX_TOOL_PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(_REMIX_TOOL_PKG_PARENT))
except OSError:
    pass  # Symlink absent — dump-tool runs without RepetitionMap replay.


def _load_cbm_reference_module():
    """Load references/python-source/analysis/cbm_segmenter.py as an ORPHAN
    module (single-file exec, bypasses analysis/__init__.py which pulls
    remix_tool.analysis.beat_detector → torch). cbm_segmenter itself only
    depends on numpy + stdlib, so no venv growth per ADR-020.
    """
    global _cbm_module_cache
    if _cbm_module_cache is not None:
        return _cbm_module_cache
    if not _CBM_MODULE_PATH.is_file():
        raise SystemExit(
            f"cbm_segmenter.py not found at {_CBM_MODULE_PATH}\n"
            f"  check that references/python-source symlink is valid."
        )
    spec = importlib.util.spec_from_file_location(
        "_ref_cbm_segmenter", _CBM_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cbm_module_cache = module
    return module


def _export_beats_and_downbeats_via_cpp(
    audio_path: Path, clip_range_sec: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """PARITY: ADR-020 option (b). Run build/beat_detector_cli on the FULL
    audio file (not clip), parse JSON output, filter BOTH beats and
    downbeats into [clip_start, clip_end), shift by -clip_start so
    returned times are clip-local seconds.

    Returns (beats_clip, downbeats_clip) f64 arrays from a SINGLE CLI
    invocation — CBM and Viterbi in production receive beats + downbeats
    from the same BeatDetector output, so both goldens must come from the
    same run (not two subprocess calls at 9 s each).

    Rationale: BeatDetector's internal tempo-estimation + consistency-filter
    uses full-track context; running on a clipped file would produce
    different (worse) output. This matches the production path — users
    load full songs in REAPER, not pre-clipped segments.
    """
    if not _CPP_CLI_PATH.is_file():
        raise SystemExit(
            f"beat_detector_cli binary not found at {_CPP_CLI_PATH}\n"
            f"  build it: cd build && cmake --build . --target beat_detector_cli\n"
            f"  or set REAMIX_CPP_CLI=<path> to override."
        )
    if not _BEAT_MODEL_PATH.is_file():
        raise SystemExit(
            f"beat-this ONNX model not found at {_BEAT_MODEL_PATH}\n"
            f"  download via the plugin's first-run model-download path, or\n"
            f"  set REAMIX_BEAT_MODEL=<path> to override."
        )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                str(_CPP_CLI_PATH),
                "--audio",  str(audio_path),
                "--model",  str(_BEAT_MODEL_PATH),
                "--output", str(tmp_path),
            ],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"beat_detector_cli failed (rc={result.returncode}) on {audio_path}\n"
                f"stderr: {result.stderr}"
            )
        data = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)

    raw_beats     = np.asarray(data.get("beats",     []), dtype=np.float64)
    raw_downbeats = np.asarray(data.get("downbeats", []), dtype=np.float64)

    if clip_range_sec is None:
        return raw_beats, raw_downbeats

    clip_start, clip_end = float(clip_range_sec[0]), float(clip_range_sec[1])
    beat_mask = (raw_beats >= clip_start) & (raw_beats < clip_end)
    db_mask   = (raw_downbeats >= clip_start) & (raw_downbeats < clip_end)
    beats     = (raw_beats[beat_mask]     - clip_start).astype(np.float64, copy=False)
    downbeats = (raw_downbeats[db_mask]   - clip_start).astype(np.float64, copy=False)
    return beats, downbeats


# --- utils -------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_resolved(symlink_or_file: Path) -> str:
    """Hash the dereferenced target (follows symlinks)."""
    return _sha256(symlink_or_file.resolve())


# ADR-023 session-13 bug-scan byproduct: weird-value detector.
# Narrow-whitelist calibrated on 10-track corpus. ABORT on NaN/Inf (any
# numerical array). WARN on uniform non-zero values across ≥ 5 elements —
# catches silhouette-class sentinel patterns (ADR-022 dumped
# `novelty_silhouette_scores = [-1.0, -1.0, ...]` k-range-size = 5 under the
# Python reference's precomputed-affinity-as-distance bug). Uniform-zero is
# silent because ADR-014 legitimately stubs `f0_hz` / `f0_confidence` /
# `voiced_ratio` to zeros in default mode; same-value repetition_* tuples on
# single-section tracks (e.g. tiesto 5×from_section_idx=1) are known-OK
# false-positives and emit WARN — operator verifies against dump log on regen.
_WEIRD_VALUE_MIN_ELEMENTS = 5


def _scan_weird_values(arr: np.ndarray, key: str) -> None:
    if arr.size < _WEIRD_VALUE_MIN_ELEMENTS:
        return
    if not np.issubdtype(arr.dtype, np.number):
        return
    flat = arr.ravel()
    if np.issubdtype(flat.dtype, np.floating) and not np.all(np.isfinite(flat)):
        n_bad = int(np.sum(~np.isfinite(flat)))
        raise ValueError(
            f"ABORT weird-value: {key} has {n_bad}/{flat.size} non-finite "
            f"(NaN/Inf) entries — suspected reference bug, investigate before regen"
        )
    mn = float(flat.min())
    mx = float(flat.max())
    if mn == mx and mn != 0.0:
        print(
            f"  WARN weird-value: {key} uniform {mn:.4g} across {flat.size} "
            f"elements — verify not sentinel (ADR-023)",
            flush=True,
        )


def _save_npy(arr: np.ndarray, out_dir: Path, name: str) -> tuple[str, int]:
    _scan_weird_values(arr, name)
    path = out_dir / f"{name}.npy"
    np.save(path, arr, allow_pickle=False)
    return _sha256(path), path.stat().st_size


def _save_json(obj: object, out_dir: Path, name: str) -> tuple[str, int]:
    """ADR-020: CBM segments output is List[dict] — not an ndarray. Writes
    a sibling .json next to the .npy tree, returns (sha256, bytes). Uses
    sort_keys=True + fixed indent for deterministic SHA across regens.
    """
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return _sha256(path), path.stat().st_size


def _save_text(lines: list[str], out_dir: Path, name: str) -> tuple[str, int]:
    """Session-6 decision-2: dump string lists (cbm_segments_labels) as
    plain UTF-8 text, one entry per line. C++ test reads with std::getline —
    zero JSON parser, zero pickled-numpy surface. Each line must be free of
    newline characters (asserted upstream).
    """
    for line in lines:
        if "\n" in line or "\r" in line:
            raise ValueError(
                f"_save_text: line {line!r} contains newline/cr — unsupported"
            )
    path = out_dir / f"{name}.txt"
    # Trailing newline after the last entry (POSIX text file convention) so
    # empty lists emit a file of length 0 rather than a single blank line.
    body = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(body, encoding="utf-8")
    return _sha256(path), path.stat().st_size


# --- step 9 helpers (verbatim from feature_extractor.py L317-463) ------------
# Inlined rather than imported so the dump tool stays independent of the
# Python reference's heavier imports. The C++ port (src/analysis/BeatWindows)
# is validated against *these* copies via the usual .npy dump pipeline;
# if a future change to feature_extractor.py touches one of these helpers,
# the dumps' .npy SHAs will drift visibly in the per-track manifest.

def _extract_edge_features(
    features: np.ndarray, beat_frames: np.ndarray, n_beats: int,
    n_edge_frames: int = N_EDGE_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """PARITY: feature_extractor.py:317 _extract_edge_features."""
    n_features = features.shape[0]
    n_frames = features.shape[1]
    edge_start = np.zeros((n_beats, n_features), dtype=np.float64)
    edge_end = np.zeros((n_beats, n_features), dtype=np.float64)

    for i in range(n_beats):
        frame_lo = int(beat_frames[i])
        if i + 1 < len(beat_frames):
            frame_hi = int(beat_frames[i + 1])
        else:
            frame_hi = n_frames
        frame_hi = min(frame_hi, n_frames)

        beat_len = frame_hi - frame_lo
        n_edge = min(n_edge_frames, max(1, beat_len // 2))

        if beat_len > 0:
            edge_start[i] = features[:, frame_lo:frame_lo + n_edge].mean(axis=1)
            edge_end[i] = features[:, frame_hi - n_edge:frame_hi].mean(axis=1)

    for arr in [edge_start, edge_end]:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr /= norms

    return edge_start, edge_end


def _extract_edge_rms(
    y: np.ndarray, sr: int, beat_times: np.ndarray, n_beats: int,
    edge_ms: float = EDGE_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """PARITY: feature_extractor.py:353 _extract_edge_rms."""
    edge_samples = int(edge_ms * sr / 1000)
    n_samples = len(y)
    beat_samples = (beat_times * sr).astype(np.int64)

    edge_rms_start = np.zeros(n_beats, dtype=np.float64)
    edge_rms_end = np.zeros(n_beats, dtype=np.float64)

    for i in range(n_beats):
        s = int(beat_samples[i])
        if i + 1 < n_beats:
            e = int(beat_samples[i + 1])
        else:
            e = n_samples
        e = min(e, n_samples)
        s = max(0, min(s, n_samples))

        beat_len = e - s
        n_edge = min(edge_samples, max(1, beat_len // 2))

        if beat_len > 0:
            start_chunk = y[s:s + n_edge]
            end_chunk = y[e - n_edge:e]
            edge_rms_start[i] = np.sqrt(np.mean(start_chunk ** 2) + 1e-10)
            edge_rms_end[i] = np.sqrt(np.mean(end_chunk ** 2) + 1e-10)

    max_val = max(edge_rms_start.max(), edge_rms_end.max(), 1e-10)
    edge_rms_start /= max_val
    edge_rms_end /= max_val

    return edge_rms_start, edge_rms_end


def _run_orchestrator(
    y: np.ndarray,
    sr: int,
    mfcc_f32: np.ndarray,     # (n_mfcc, T) float32
    chroma_f32: np.ndarray,   # (12, T)   float32
    contrast_f32: np.ndarray, # (7, T)    float32
    rms_f32: np.ndarray,      # (T,)      float32
    onset_f32: np.ndarray,    # (T,)      float32
    centroid_f32: np.ndarray, # (T,)      float32
    beat_times: np.ndarray,   # (n_beats,) float64 (any source — synthetic or real)
    *,
    skip_transition_waveforms: bool,
    skip_vocal_features: bool = True,
) -> dict[str, np.ndarray]:
    """Inline mirror of FeatureExtractor._extract_internal L189-254.

    Takes pre-computed f32 feature matrices (as emitted by librosa in the
    real pipeline; our dump cast them to f64 for storage, callers re-cast
    to f32 before handing off here). Returns a dict of reference arrays the
    C++ orchestrator parity test will load.

    PARITY: feature_extractor.py:189-254.
    """
    n_beats = int(len(beat_times))

    # PARITY: L210 — vstack in librosa-native order (MFCC → chroma → contrast).
    stacked_f32 = np.vstack([mfcc_f32, chroma_f32, contrast_f32]).astype(np.float32, copy=False)

    # PARITY: L211-213 — librosa.time_to_frames on float64 beat_times with
    # round-half-to-even via np.round. On synthetic beats derived from
    # `beat_frames * HOP / SR` this round-trips exactly; on real beats it
    # exercises the round-half-to-even edge case.
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP_LENGTH)

    # PARITY: L214 → _beat_sync_features (L465-475). Truncate if too many
    # slices, pad with last-row tile if too few.
    synced_f32 = librosa.util.sync(stacked_f32, beat_frames, aggregate=np.mean).T
    if synced_f32.shape[0] > n_beats:
        synced_f32 = synced_f32[:n_beats]
    elif synced_f32.shape[0] < n_beats:
        pad_count = n_beats - synced_f32.shape[0]
        synced_f32 = np.vstack([synced_f32, np.tile(synced_f32[-1:], (pad_count, 1))])
    beat_features_f32 = synced_f32.astype(np.float32, copy=False)

    # PARITY: L217-219 — edge features on the PRE-L2 stacked matrix.
    edge_start, edge_end = _extract_edge_features(stacked_f32, beat_frames, n_beats)

    # PARITY: L233-235 — L2 norm per beat; zero-row → norm := 1 (NOT tiny).
    norms = np.linalg.norm(beat_features_f32, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    beat_features_f32 = beat_features_f32 / norms

    assert beat_features_f32.shape[0] == n_beats, (
        f"Feature dimension mismatch: {beat_features_f32.shape[0]} vs {n_beats}"
    )

    # PARITY: L244-254 — side channels, f32 sync → f64 side-channel output
    # via division by Python-float max (promotes f32/f64 → f64).
    rms_sync_f32 = librosa.util.sync(
        rms_f32[np.newaxis, :], beat_frames, aggregate=np.mean
    )[0][:n_beats]
    rms_energy = rms_sync_f32 / max(rms_sync_f32.max(), 1e-10)

    onset_sync_f32 = librosa.util.sync(
        onset_f32[np.newaxis, :], beat_frames, aggregate=np.mean
    )[0][:n_beats]
    onset_strength_n = onset_sync_f32 / max(onset_sync_f32.max(), 1e-10)

    centroid_sync_f32 = librosa.util.sync(
        centroid_f32[np.newaxis, :], beat_frames, aggregate=np.mean
    )[0][:n_beats]
    spectral_centroid_n = centroid_sync_f32 / max(centroid_sync_f32.max(), 1e-10)

    out = {
        "feature_matrix":      np.ascontiguousarray(beat_features_f32, dtype=np.float64),
        "edge_features_start": np.ascontiguousarray(edge_start,       dtype=np.float64),
        "edge_features_end":   np.ascontiguousarray(edge_end,         dtype=np.float64),
        "rms_energy":          np.ascontiguousarray(rms_energy,       dtype=np.float64),
        "onset_strength":      np.ascontiguousarray(onset_strength_n, dtype=np.float64),
        "spectral_centroid":   np.ascontiguousarray(spectral_centroid_n, dtype=np.float64),
    }

    # PARITY: phase-2b step 6 — vocal features on round-tripped beat_frames.
    # feature_extractor.py:264-281 calls _extract_vocal_features with the SAME
    # beat_frames computed at L211-213 (via librosa.time_to_frames). Parity
    # test_feature_extractor Fixture A therefore compares C++ against these
    # round-tripped vocal outputs, NOT against the raw-arange vocal dumps
    # used by test_vocal_features (which has a different beat-frame source).
    if not skip_vocal_features:
        vocal = _extract_vocal_features_dump(
            y, sr, HOP_LENGTH, N_FFT, beat_frames, n_beats,
        )
        out["vocal_activity"]             = vocal["vocal_activity"]
        out["voiced_ratio"]               = vocal["voiced_ratio"]
        out["f0_hz"]                      = vocal["f0_hz"]
        out["f0_confidence"]              = vocal["f0_confidence"]
        out["edge_vocal_activity_start"]  = vocal["edge_vocal_activity_start"]
        out["edge_vocal_activity_end"]    = vocal["edge_vocal_activity_end"]
        out["edge_vocal_onset_start"]     = vocal["edge_vocal_onset_start"]
        out["edge_vocal_release_end"]     = vocal["edge_vocal_release_end"]

    # PARITY: L226-230 — transitionWaveforms omitted in fast mode.
    # boundary/edge_rms/transition on y+beat_times are identical math to
    # step-9's existing dumps; for synthetic beats we reuse those .npy
    # files. For real beats, we re-emit via _extract_* against the real
    # beat_times here (caller decides whether to serialize these via a
    # different entry point).
    return out


# --- phase-2b helpers (verbatim from feature_extractor.py + vocal_features.py)
# Inlined rather than imported so the dump tool stays independent of the
# Python reference package's heavier imports. The C++ ports (src/dsp/HPSS,
# src/dsp/SpectralFlatness, src/util/Percentile, src/util/MedianFilter,
# src/analysis/VocalFeatures) are validated against *these* copies via the
# usual .npy dump pipeline.
#
# ADR-014: phase-2b ships the default-mode (vocal_use_pyin=False) composite
# proxy only. pYIN branches are NOT ported. The inline below therefore
# reproduces vocal_features.py L66-218 with use_pyin=False hardcoded.

_VOCAL_EDGE_FRAMES = 4  # config.py:70 vocal_edge_frames


def _beat_sync_1d(
    values: np.ndarray, beat_frames: np.ndarray, n_beats: int,
    aggregate=np.mean,
) -> np.ndarray:
    """PARITY: feature_extractor.py:477 _beat_sync_1d."""
    synced = librosa.util.sync(
        np.asarray(values, dtype=np.float64)[np.newaxis, :],
        beat_frames,
        aggregate=aggregate,
    )[0]
    if synced.shape[0] > n_beats:
        synced = synced[:n_beats]
    elif synced.shape[0] < n_beats:
        pad_value = synced[-1] if synced.size else 0.0
        synced = np.pad(synced, (0, n_beats - synced.shape[0]), constant_values=pad_value)
    return np.asarray(synced, dtype=np.float64)


def _beat_sync_voiced_f0(
    f0_hz: np.ndarray, voiced_prob: np.ndarray,
    beat_frames: np.ndarray, n_beats: int,
) -> np.ndarray:
    """PARITY: vocal_features.py:224 _beat_sync_voiced_f0."""
    f0_sync = np.zeros(n_beats, dtype=np.float64)
    n_frames = len(f0_hz)

    for i in range(n_beats):
        frame_lo = int(beat_frames[i])
        if i + 1 < len(beat_frames):
            frame_hi = int(beat_frames[i + 1])
        else:
            frame_hi = n_frames
        frame_lo = max(0, min(frame_lo, n_frames))
        frame_hi = max(frame_lo, min(frame_hi, n_frames))
        if frame_hi <= frame_lo:
            continue

        beat_f0 = f0_hz[frame_lo:frame_hi]
        beat_conf = voiced_prob[frame_lo:frame_hi]
        mask = np.isfinite(beat_f0) & (beat_f0 > 0.0) & (beat_conf >= 0.35)
        if np.any(mask):
            f0_sync[i] = float(np.median(beat_f0[mask]))

    return f0_sync


def _extract_edge_1d(
    values: np.ndarray, beat_frames: np.ndarray, n_beats: int,
    n_edge_frames: int = _VOCAL_EDGE_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """PARITY: vocal_features.py:258 _extract_edge_1d (mean-edge)."""
    n_frames = len(values)
    edge_start = np.zeros(n_beats, dtype=np.float64)
    edge_end = np.zeros(n_beats, dtype=np.float64)

    for i in range(n_beats):
        frame_lo = int(beat_frames[i])
        if i + 1 < len(beat_frames):
            frame_hi = int(beat_frames[i + 1])
        else:
            frame_hi = n_frames
        frame_lo = max(0, min(frame_lo, n_frames))
        frame_hi = max(frame_lo, min(frame_hi, n_frames))
        beat_len = frame_hi - frame_lo
        n_edge = min(n_edge_frames, max(1, beat_len // 2))
        if beat_len <= 0:
            continue
        edge_start[i] = float(np.mean(values[frame_lo:frame_lo + n_edge]))
        edge_end[i] = float(np.mean(values[frame_hi - n_edge:frame_hi]))

    return edge_start, edge_end


def _extract_edge_1d_peak(
    values: np.ndarray, beat_frames: np.ndarray, n_beats: int,
    n_edge_frames: int = _VOCAL_EDGE_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """PARITY: vocal_features.py:287 _extract_edge_1d_peak (max-edge)."""
    n_frames = len(values)
    edge_start = np.zeros(n_beats, dtype=np.float64)
    edge_end = np.zeros(n_beats, dtype=np.float64)

    for i in range(n_beats):
        frame_lo = int(beat_frames[i])
        if i + 1 < len(beat_frames):
            frame_hi = int(beat_frames[i + 1])
        else:
            frame_hi = n_frames
        frame_lo = max(0, min(frame_lo, n_frames))
        frame_hi = max(frame_lo, min(frame_hi, n_frames))
        beat_len = frame_hi - frame_lo
        n_edge = min(n_edge_frames, max(1, beat_len // 2))
        if beat_len <= 0:
            continue
        edge_start[i] = float(np.max(values[frame_lo:frame_lo + n_edge]))
        edge_end[i] = float(np.max(values[frame_hi - n_edge:frame_hi]))

    return edge_start, edge_end


def _extract_vocal_features_dump(
    y: np.ndarray, sr: int, hop_length: int, n_fft: int,
    beat_frames: np.ndarray, n_beats: int,
) -> dict[str, np.ndarray]:
    """PARITY: vocal_features.py:40 _extract_vocal_features, hardcoded
    use_pyin=False (ADR-014). Returns a dict with BOTH intermediates
    (per-frame) and final beat-sync outputs — the intermediates give
    the C++ parity tests an in-pipeline breakpoint per module
    (HPSS → flatness → composite → beat-sync).
    """
    if n_beats == 0:
        zeros = np.zeros(0, dtype=np.float64)
        return {
            "hpss_y_harmonic":          np.zeros(0, dtype=np.float32),
            "spectral_flatness":        np.zeros(0, dtype=np.float32),
            "harmonic_ratio":           zeros,
            "flatness_inv":             zeros,
            "voice_band_ratio":         zeros,
            "vocal_activity_frame":     zeros,
            "vocal_rise_frame":         zeros,
            "vocal_fall_frame":         zeros,
            "vocal_activity":           zeros,
            "voiced_ratio":             zeros,
            "f0_hz":                    zeros,
            "f0_confidence":            zeros,
            "edge_vocal_activity_start": zeros,
            "edge_vocal_activity_end":   zeros,
            "edge_vocal_onset_start":    zeros,
            "edge_vocal_release_end":    zeros,
        }

    # --- Lightweight HPSS features (vocal_features.py L72-104) ---
    y_harmonic, _ = librosa.effects.hpss(y)

    harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    full_rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    harmonic_ratio = harmonic_rms / np.maximum(full_rms, 1e-8)
    harmonic_ratio = np.clip(harmonic_ratio / 1.15, 0.0, 1.0)

    flatness = librosa.feature.spectral_flatness(
        y=y_harmonic, hop_length=hop_length, n_fft=n_fft,
    )[0]
    flatness_scale = np.percentile(flatness, 95) if flatness.size else 1.0
    flatness_scale = max(float(flatness_scale), 1e-6)
    flatness_inv = np.clip(1.0 - flatness / flatness_scale, 0.0, 1.0)

    spectrum = np.abs(
        librosa.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    voice_mask = (freqs >= 250.0) & (freqs <= 3400.0)
    body_mask = (freqs >= 80.0) & (freqs <= 7000.0)
    if np.any(voice_mask) and np.any(body_mask):
        voice_energy = np.mean(spectrum[voice_mask], axis=0)
        body_energy = np.mean(spectrum[body_mask], axis=0)
        voice_band_ratio = voice_energy / np.maximum(body_energy, 1e-8)
        ratio_scale = np.percentile(voice_band_ratio, 95) if voice_band_ratio.size else 1.0
        ratio_scale = max(float(ratio_scale), 1e-6)
        voice_band_ratio = np.clip(voice_band_ratio / ratio_scale, 0.0, 1.0)
    else:
        voice_band_ratio = np.zeros(harmonic_ratio.shape[0], dtype=np.float64)

    # use_pyin=False (ADR-014): f0_hz_frame and voiced_prob are zero-filled
    # at the frame shape of harmonic_ratio. This matches vocal_features.py
    # L123-124 exactly.
    f0_hz_frame = np.zeros_like(harmonic_ratio)
    voiced_prob = np.zeros_like(harmonic_ratio)

    frame_count = min(
        len(harmonic_ratio), len(flatness_inv), len(voice_band_ratio),
        len(f0_hz_frame), len(voiced_prob),
    )
    harmonic_ratio   = harmonic_ratio[:frame_count]
    flatness_inv     = flatness_inv[:frame_count]
    voice_band_ratio = voice_band_ratio[:frame_count]
    f0_hz_frame      = f0_hz_frame[:frame_count]
    voiced_prob      = voiced_prob[:frame_count]

    # Composite (vocal_features.py L148-155, use_pyin=False branch).
    vocal_activity_frame = np.clip(
        0.40 * harmonic_ratio
        + 0.33 * voice_band_ratio
        + 0.27 * flatness_inv,
        0.0, 1.0,
    )

    # use_pyin=False → voiced_frame all zeros (L157 else branch).
    voiced_frame = np.zeros(frame_count)
    vocal_delta = np.diff(vocal_activity_frame, prepend=vocal_activity_frame[:1])
    vocal_rise_frame = np.clip(vocal_delta, 0.0, None)
    vocal_fall_frame = np.clip(-vocal_delta, 0.0, None)
    if vocal_rise_frame.size:
        rise_scale = max(float(np.percentile(vocal_rise_frame, 95)), 1e-6)
        vocal_rise_frame = np.clip(vocal_rise_frame / rise_scale, 0.0, 1.0)
    if vocal_fall_frame.size:
        fall_scale = max(float(np.percentile(vocal_fall_frame, 95)), 1e-6)
        vocal_fall_frame = np.clip(vocal_fall_frame / fall_scale, 0.0, 1.0)

    vocal_activity = _beat_sync_1d(vocal_activity_frame, beat_frames, n_beats)
    voiced_ratio   = _beat_sync_1d(voiced_frame, beat_frames, n_beats)
    f0_confidence  = _beat_sync_1d(voiced_prob, beat_frames, n_beats)
    f0_hz          = _beat_sync_voiced_f0(f0_hz_frame, voiced_prob, beat_frames, n_beats)
    edge_vocal_start, edge_vocal_end = _extract_edge_1d(
        vocal_activity_frame, beat_frames, n_beats,
    )
    edge_vocal_onset_start, _ = _extract_edge_1d_peak(
        vocal_rise_frame, beat_frames, n_beats,
    )
    _, edge_vocal_release_end = _extract_edge_1d_peak(
        vocal_fall_frame, beat_frames, n_beats,
    )

    # Final clips (vocal_features.py L191-198).
    vocal_activity         = np.clip(vocal_activity, 0.0, 1.0)
    voiced_ratio           = np.clip(voiced_ratio, 0.0, 1.0)
    f0_confidence          = np.clip(f0_confidence, 0.0, 1.0)
    edge_vocal_start       = np.clip(edge_vocal_start, 0.0, 1.0)
    edge_vocal_end         = np.clip(edge_vocal_end, 0.0, 1.0)
    edge_vocal_onset_start = np.clip(edge_vocal_onset_start, 0.0, 1.0)
    edge_vocal_release_end = np.clip(edge_vocal_release_end, 0.0, 1.0)
    # vocal_max_hz=1000.0 (config.py:69) * 1.2 = 1200.0 (inactive under
    # use_pyin=False since f0_hz is all zeros, but replicate path).
    f0_hz = np.clip(f0_hz, 0.0, 1000.0 * 1.2)

    return {
        # --- phase-2b intermediates (per-frame) ---
        "hpss_y_harmonic":          y_harmonic.astype(np.float32, copy=False),
        "stft_magnitude_harmonic":  np.ascontiguousarray(spectrum, dtype=np.float32),
        "spectral_flatness":        flatness.astype(np.float32, copy=False),
        "harmonic_ratio":           np.ascontiguousarray(harmonic_ratio, dtype=np.float64),
        "flatness_inv":             np.ascontiguousarray(flatness_inv, dtype=np.float64),
        "voice_band_ratio":         np.ascontiguousarray(voice_band_ratio, dtype=np.float64),
        "vocal_activity_frame":     np.ascontiguousarray(vocal_activity_frame, dtype=np.float64),
        "vocal_rise_frame":         np.ascontiguousarray(vocal_rise_frame, dtype=np.float64),
        "vocal_fall_frame":         np.ascontiguousarray(vocal_fall_frame, dtype=np.float64),
        # --- phase-2b beat-sync outputs (per-beat; 3 zero-filled per ADR-014) ---
        "vocal_activity":           np.ascontiguousarray(vocal_activity, dtype=np.float64),
        "voiced_ratio":             np.ascontiguousarray(voiced_ratio, dtype=np.float64),
        "f0_hz":                    np.ascontiguousarray(f0_hz, dtype=np.float64),
        "f0_confidence":            np.ascontiguousarray(f0_confidence, dtype=np.float64),
        "edge_vocal_activity_start": np.ascontiguousarray(edge_vocal_start, dtype=np.float64),
        "edge_vocal_activity_end":   np.ascontiguousarray(edge_vocal_end, dtype=np.float64),
        "edge_vocal_onset_start":    np.ascontiguousarray(edge_vocal_onset_start, dtype=np.float64),
        "edge_vocal_release_end":    np.ascontiguousarray(edge_vocal_release_end, dtype=np.float64),
    }


# --- phase-3 helpers (verbatim from structure_analyzer.py L193-225) ----------
# Inlined per the same pattern as phase-2b helpers above: the dump tool stays
# independent of the Python reference package's heavier imports (CBM segmenter,
# audio_cache, sklearn SpectralClustering). The C++ port
# (src/analysis/NoveltyFeatures) is validated against *this* copy.
#
# ADR-019: novelty feature pipeline ports the 17 lines of
# StructureAnalyzer._extract_features verbatim via composition of existing
# phase-2 C++ primitives (STFT + MelSpectrogramLibrosa + DCT + PitchTrack +
# ChromaSTFT). Zero new DSP. Parity gate 1e-3 absolute / 5e-7 ratio (ADR-011
# escape precedent).

def _extract_novelty_features_25d(y: np.ndarray, sr: int) -> np.ndarray:
    """PARITY: structure_analyzer.py:193-210 StructureAnalyzer._extract_features.

    sr=22050 and hop_length=512 are hardcoded in structure_analyzer at L71-72.
    We pass sr explicitly so the dump tool's SR constant drives both phase-2
    and phase-3 keys from one source of truth.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH,
    )
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT,
    )
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (
        mfcc.std(axis=1, keepdims=True) + 1e-8
    )
    chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / (
        chroma.std(axis=1, keepdims=True) + 1e-8
    )
    features = np.vstack([mfcc * 1.5, chroma])
    return np.ascontiguousarray(features, dtype=np.float32)


def _compute_novelty_ssm(features: np.ndarray, stride: int = 4) -> np.ndarray:
    """PARITY: structure_analyzer.py:212-225 StructureAnalyzer._compute_ssm.

    stride=4 is the standard-mode value (fast_mode branch hits
    DEFAULT_CONFIG.analysis.fast_structure_frame_stride). We dump the
    standard-mode SSM only — fast-mode SSM falls out of the same helper
    with a different stride when needed.
    """
    stride = max(1, int(stride))
    features_ds = features[:, ::stride]
    norms = np.linalg.norm(features_ds, axis=0, keepdims=True) + 1e-8
    features_norm = features_ds / norms
    ssm = features_norm.T @ features_norm
    ssm = np.clip(ssm, 0, 1)
    return np.ascontiguousarray(ssm, dtype=np.float32)


# --- phase-3 NoveltySegmenter (pre-clustering) reference helpers -------------
# ADR-021 session 7 scope: port SSM → novelty curve → find_peaks → boundaries
# → segment embeddings. Clustering deferred to session 8.
#
# These helpers replay structure_analyzer.py::_compute_novelty /
# _find_boundaries / _compute_segment_embeddings on inputs that are already
# dumped (features + SSM). They are additive to the dump schema — zero SHA
# drift on existing keys verified at regen.

def _compute_novelty_curve(ssm: np.ndarray) -> np.ndarray:
    """PARITY: structure_analyzer.py:227-251 StructureAnalyzer._compute_novelty.

    Returns the f64 novelty curve (same dtype numpy uses internally —
    `np.zeros(n)` defaults to f64, and `np.sum(region * kernel)` where
    region is f32 and kernel is f64 upcasts the multiply to f64).
    Normalized to [0, 1] by max at the end when max > 0.
    """
    n = int(ssm.shape[0])
    kernel_size = min(32, n // 4)
    if kernel_size < 4:
        kernel_size = 4
    kernel_size = kernel_size - (kernel_size % 2)  # ensure even

    half = kernel_size // 2
    kernel = np.ones((kernel_size, kernel_size))  # default dtype f64
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    novelty = np.zeros(n)  # f64
    for i in range(half, n - half):
        region = ssm[i - half: i + half, i - half: i + half]
        novelty[i] = np.abs(np.sum(region * kernel))

    novelty = uniform_filter1d(novelty, size=5)  # scipy.ndimage default mode='reflect'

    if novelty.max() > 0:
        novelty = novelty / novelty.max()

    return np.ascontiguousarray(novelty, dtype=np.float64)


def _find_novelty_peaks_and_boundaries(
    novelty: np.ndarray, duration: float, min_segment_duration: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """PARITY: structure_analyzer.py:253-275 StructureAnalyzer._find_boundaries.

    Returns (raw_peaks_i64, boundaries_f64). raw_peaks are the scipy.signal
    find_peaks output BEFORE margin clip (bisection aid for C++ port);
    boundaries are the final f64 [0, t1, ..., duration] array the
    segmenter outputs. `margin = 2.0` hardcoded in Python L271.
    """
    n_frames = int(len(novelty))
    frame_duration = duration / n_frames

    min_distance = int(min_segment_duration / frame_duration)
    min_distance = max(min_distance, 1)

    # height = percentile(novelty, 60) — np.percentile default 'linear' interp.
    height = float(np.percentile(novelty, 60))

    peaks, _ = find_peaks(
        novelty,
        distance=min_distance,
        prominence=0.20,
        height=height,
    )

    times = peaks * frame_duration
    margin = 2.0
    times = times[(times > margin) & (times < duration - margin)]

    boundaries = np.concatenate([[0.0], times, [duration]]).astype(np.float64)
    return (
        np.ascontiguousarray(peaks, dtype=np.int64),
        np.ascontiguousarray(boundaries, dtype=np.float64),
    )


def _compute_novelty_segment_embeddings(
    features: np.ndarray, boundaries: np.ndarray, sr: int, hop_length: int,
) -> np.ndarray:
    """PARITY: structure_analyzer.py:277-296 StructureAnalyzer._compute_segment_embeddings.

    features is f32 (25, T); embeddings list-of-1D then np.array stacks
    to (n_segments, 25) preserving f32 dtype (all rows same dtype).
    """
    n_frames = int(features.shape[1])
    n_segments = int(len(boundaries)) - 1
    embeddings: list[np.ndarray] = []

    for i in range(n_segments):
        start_frame = int(boundaries[i] * sr / hop_length)
        end_frame = int(boundaries[i + 1] * sr / hop_length)

        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, n_frames))

        segment_features = features[:, start_frame:end_frame]
        embedding = segment_features.mean(axis=1)
        embeddings.append(embedding)

    if not embeddings:
        return np.zeros((0, int(features.shape[0])), dtype=features.dtype)
    return np.ascontiguousarray(np.array(embeddings), dtype=features.dtype)


# --- phase-3 NoveltySegmenter (clustering + labeling) reference helpers ------
# ADR-021 session 8 scope: port StructureAnalyzer._cluster_segments (L298-348)
# + _run_clustering (L350-367) + _create_segments (L369-475). Silhouette-
# selected spectral clustering on clipped-cosine affinity (sklearn, f32 →
# f64 internals), followed by energy-aware, repetition-based labeling.
#
# **ADR-022 Option B — silhouette-call fix**: `structure_analyzer.py:332`
# passes `affinity` (similarity matrix, diagonal = 1.0) to
# `silhouette_score(..., metric="precomputed")` which expects a DISTANCE
# matrix (diagonal = 0.0). This raises `ValueError` on every k in the
# pinned sklearn version, which `except Exception: continue` silently
# catches — so Python's silhouette search is dead code and the fallback
# `_run_clustering(embeddings, min(4, n_segs))` always fires. This helper
# passes `1 - affinity` (the correct distance matrix) so the silhouette
# loop actually runs and picks an adaptive k per track per the function's
# documented intent. Empirical audit on our 10-track corpus (session-8
# redo, 2026-04-21) showed the fixed version picks k=2 or k=3 on 8/10
# tracks (adaptive to musical structure) vs the buggy fallback's static
# k=4, and produces musically more accurate labels on 3-4 tracks
# (miles_davis theme grouping, billie_jean chorus identification,
# smells_like repeating-chorus structure, eminem verse-chorus split).
#
# Bitwise parity unattainable (eigenvector signs + k-means++ RNG) — C++
# parity gate is Hungarian-matched cluster_id agreement ≥ 90 % + exact
# label ≥ 90 %. These helpers dump affinity / silhouette-per-k / best_k /
# cluster_ids / segments_fields / labels directly so the C++ side can
# bisect each stage independently if the gate flips.

def _cluster_novelty_segments(
    embeddings: np.ndarray, fast_mode: bool = False,
) -> dict:
    """PARITY: structure_analyzer.py:298-367 _cluster_segments + _run_clustering.

    Returns dict with:
      - affinity         f32 (n_segs, n_segs)  — clipped cosine on L2-norm embeddings
      - silhouette_scores f64 (len(ks),)       — silhouette per k; -1.0 sentinel for
                                                 skipped/failed k (matches the C++
                                                 "< 2 unique labels" short-circuit)
      - best_k           i64 scalar            — the k picked (silhouette argmax
                                                 or fallback min(4, n_segs)); 0 for n≤1
      - cluster_ids      i64 (n_segs,)         — final labels fed to _create_segments

    Empty-ks case (n_segs ≤ 2): silhouette_scores = []. n_segs == 2 uses
    _run_clustering(..., 2) directly per Python L305-307. n_segs ≤ 1 returns
    zeros + best_k = 0 (sentinel).
    """
    n_segments = int(embeddings.shape[0])

    if n_segments == 0:
        return {
            "affinity": np.zeros((0, 0), dtype=np.float32),
            "silhouette_scores": np.array([], dtype=np.float64),
            "best_k": np.asarray(0, dtype=np.int64),
            "cluster_ids": np.zeros(0, dtype=np.int64),
        }

    # Affinity computed once (outside the k-loop — matches Python L310-313 which
    # computes it in the outer try before the for loop and reuses it).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_norm = embeddings / norms
    affinity = embeddings_norm @ embeddings_norm.T
    affinity = np.clip(affinity, 0, 1)
    affinity_dump = np.ascontiguousarray(affinity, dtype=np.float32)

    if n_segments == 1:
        return {
            "affinity": affinity_dump,
            "silhouette_scores": np.array([], dtype=np.float64),
            "best_k": np.asarray(0, dtype=np.int64),
            "cluster_ids": np.zeros(1, dtype=np.int64),
        }

    if n_segments == 2:
        cluster_ids = _run_novelty_spectral_clustering(embeddings, 2).astype(np.int64)
        return {
            "affinity": affinity_dump,
            "silhouette_scores": np.array([], dtype=np.float64),
            "best_k": np.asarray(2, dtype=np.int64),
            "cluster_ids": cluster_ids,
        }

    max_k = min(5 if fast_mode else 7, n_segments)
    ks = list(range(2, max_k))  # Python L320 — exclusive upper

    best_k, best_score = 2, -1.0
    best_labels: np.ndarray | None = None
    silhouette_scores_list: list[float] = []

    for k in ks:
        try:
            clustering = SpectralClustering(
                n_clusters=k, affinity="precomputed",
                random_state=42, assign_labels="kmeans",
            )
            labels = clustering.fit_predict(affinity)
            if len(set(labels)) < 2:
                silhouette_scores_list.append(-1.0)
                continue
            # ADR-022 Option B — fix: pass 1 - affinity (distance matrix)
            # instead of affinity (similarity matrix). sklearn's precomputed
            # silhouette expects distances with diagonal = 0; our clipped
            # cosine affinity has diagonal = 1 so the distance form is
            # `1 - affinity`. This is what Python's docstring intended
            # ("auto-selected k via silhouette score") but the original
            # code passed the wrong matrix.
            score = silhouette_score(1 - affinity, labels, metric="precomputed")
            silhouette_scores_list.append(float(score))
            if score > best_score:
                best_k, best_score = k, float(score)
                best_labels = labels
        except Exception:
            silhouette_scores_list.append(-1.0)
            continue

    if best_labels is not None:
        cluster_ids = best_labels.astype(np.int64)
        used_k = int(best_k)
    else:
        # Python L344 fallback: _run_clustering(min(3 or 4, n_segments))
        fallback_k = min(3 if fast_mode else 4, n_segments)
        cluster_ids = _run_novelty_spectral_clustering(
            embeddings, fallback_k,
        ).astype(np.int64)
        used_k = int(fallback_k)

    return {
        "affinity": affinity_dump,
        "silhouette_scores": np.asarray(silhouette_scores_list, dtype=np.float64),
        "best_k": np.asarray(used_k, dtype=np.int64),
        "cluster_ids": cluster_ids,
    }


def _run_novelty_spectral_clustering(
    embeddings: np.ndarray, n_clusters: int,
) -> np.ndarray:
    """PARITY: structure_analyzer.py:350-367 _run_clustering.

    Used for the n_segments == 2 direct path and the all-k-failed fallback
    path (n_clusters = min(4, n_segs) in standard mode). Outer-exception
    fallback mirrors Python L366-367 modulo-sequential.
    """
    try:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_norm = embeddings / norms
        affinity = embeddings_norm @ embeddings_norm.T
        affinity = np.clip(affinity, 0, 1)
        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed",
            random_state=42, assign_labels="kmeans",
        )
        return clustering.fit_predict(affinity)
    except Exception:
        return np.arange(len(embeddings)) % n_clusters


def _create_novelty_segments(
    boundaries: np.ndarray, cluster_ids: np.ndarray, duration: float,
    novelty: np.ndarray, y: np.ndarray, sr: int,
) -> tuple[np.ndarray, list[str]]:
    """PARITY: structure_analyzer.py:369-475 _create_segments.

    Returns (fields, labels). fields is f64 (n_segments, 4) laid out as
    [start, end, confidence, cluster_id] — cluster_id stored as f64 (will be
    read as int by the C++ side; a float64 integer loses no information for
    values < 2^53). labels is list[str], one entry per segment, aligned to
    fields rows.

    Position overrides: first segment with start < 0.15 * duration → "intro";
    last segment with end > 0.85 * duration → "outro" (NOT 20/80 like CBM).
    Confidence for i==0 is literal 0.8; for i>0 it's min(1.0, 0.5 + 0.5 *
    novelty[boundary_frame]) with boundary_frame = int(start / frame_dur)
    clamped to n_frames - 1. No rounding — raw floats.
    """
    n_segments = int(len(boundaries)) - 1
    if n_segments <= 0:
        return np.zeros((0, 4), dtype=np.float64), []

    # Per-segment RMS energy (L392-401 — uses self._sr which is 22050).
    seg_energy = np.zeros(n_segments)
    for i in range(n_segments):
        start_sample = int(boundaries[i] * sr)
        end_sample = int(boundaries[i + 1] * sr)
        start_sample = max(0, min(start_sample, len(y) - 1))
        end_sample = max(start_sample + 1, min(end_sample, len(y)))
        segment_audio = y[start_sample:end_sample]
        seg_energy[i] = np.sqrt(np.mean(segment_audio ** 2))

    cluster_counts = Counter(int(c) for c in cluster_ids)
    repeating = {c for c, n in cluster_counts.items() if n >= 2}

    cluster_energy: dict[int, float] = {}
    for c in repeating:
        indices = [i for i, cid in enumerate(cluster_ids) if int(cid) == c]
        cluster_energy[c] = float(np.mean(seg_energy[indices]))

    sorted_by_energy = sorted(cluster_energy, key=cluster_energy.get, reverse=True)
    cluster_to_label: dict[int, str] = {}
    for rank, c in enumerate(sorted_by_energy):
        if rank == 0:
            cluster_to_label[c] = "chorus"
        elif rank == 1:
            cluster_to_label[c] = "verse"
        else:
            cluster_to_label[c] = "bridge"

    # Non-repeating clusters get "bridge" (L424-427).
    for c in cluster_counts:
        if c not in cluster_to_label:
            cluster_to_label[c] = "bridge"

    # If NO repeating clusters, fall back to frequency-based (L429-438).
    if not repeating:
        sorted_clusters = cluster_counts.most_common()
        for rank, (cid, _count) in enumerate(sorted_clusters):
            if rank == 0:
                cluster_to_label[cid] = "chorus"
            elif rank == 1:
                cluster_to_label[cid] = "verse"
            else:
                cluster_to_label[cid] = "bridge"

    n_frames = int(len(novelty))
    frame_duration = duration / n_frames if n_frames > 0 else 1.0

    fields = np.zeros((n_segments, 4), dtype=np.float64)
    labels: list[str] = []
    for i in range(n_segments):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])
        cluster_id = int(cluster_ids[i])

        if i == 0 and start < duration * 0.15:
            label = "intro"
        elif i == n_segments - 1 and end > duration * 0.85:
            label = "outro"
        else:
            label = cluster_to_label.get(cluster_id, "verse")

        if i > 0 and n_frames > 0:
            boundary_frame = int(start / frame_duration)
            boundary_frame = min(boundary_frame, n_frames - 1)
            confidence = min(1.0, 0.5 + 0.5 * float(novelty[boundary_frame]))
        else:
            confidence = 0.8

        fields[i, 0] = start
        fields[i, 1] = end
        fields[i, 2] = float(confidence)
        fields[i, 3] = float(cluster_id)
        labels.append(label)

    return fields, labels


# --- phase-3 SegmentConsolidation replay (session 9) -------------------------
# Verbatim replay of references/python-source/analysis/segment_consolidation.py
# L30-193 (the mixin methods actually invoked from StructureAnalyzer._analyze,
# not refine_boundaries / _snap_to_onsets which are out of scope for the
# dispatcher's close loop). Free functions rather than a mixin — none of these
# methods touch `self.*` attributes, only `self.` for method dispatch.
#
# Rationale for inline replay (same pattern as session-7 novelty helpers +
# session-8 cluster/create-segments helpers): orphan-loading
# structure_analyzer.py would require sys.modules monkey-patching for the
# `from remix_tool.analysis.structure_analyzer import Segment` lazy imports
# in segment_consolidation.py. Copy-port is simpler and the 268 Python LOC
# is all deterministic sequential list processing.
#
# Config values inlined from references/python-source/config.py:56-59. The
# port target (src/analysis/SegmentConsolidation.{h,cpp}) will hardcode
# the same constants at the same class scope — reproducibility contract
# is "these three numbers match Python source on both sides".


@dataclass
class _StructSegment:
    """Replay of structure_analyzer.py:34-42 Segment dataclass (for
    consolidation input/output; field order mirrors Python source exactly).
    """
    start: float
    end: float
    label: str
    confidence: float
    cluster_id: int = 0


def _consolidation_merge_same_label(
    segments: list[_StructSegment],
) -> list[_StructSegment]:
    """PARITY: segment_consolidation.py:30-50 _merge_same_label_segments.

    Confidence is f64 average via `(a + b) * 0.5` per Python L45. Cluster_id
    inherits from `prev` (first-wins) per L46 — preserved across merge chain.
    """
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if prev.label == seg.label:
            merged[-1] = _StructSegment(
                start=prev.start,
                end=seg.end,
                label=prev.label,
                confidence=float((prev.confidence + seg.confidence) * 0.5),
                cluster_id=prev.cluster_id,
            )
        else:
            merged.append(seg)
    return merged


def _consolidation_merge_triplet(
    left: _StructSegment, middle: _StructSegment, right: _StructSegment,
) -> _StructSegment:
    """PARITY: segment_consolidation.py:52-67 _merge_triplet.

    Confidence is numpy.mean of 3 floats → sum in pairwise fashion then
    divide by 3.0 in f64. For n=3 numpy's pairwise-sum branch is naive
    `((a + b) + c)` (length < 8). C++ port must match.
    """
    confidence = float(
        np.mean([left.confidence, middle.confidence, right.confidence])
    )
    cluster_id = left.cluster_id if left.label == right.label else middle.cluster_id
    label = left.label if left.label == right.label else middle.label
    return _StructSegment(
        start=left.start,
        end=right.end,
        label=label,
        confidence=confidence,
        cluster_id=cluster_id,
    )


def _consolidation_merge_pair(
    first: _StructSegment,
    second: _StructSegment,
    label: str | None = None,
) -> _StructSegment:
    """PARITY: segment_consolidation.py:69-81 _merge_pair.

    Label defaults to `second.label` when caller passes None. Cluster_id
    picks `second`'s when the resulting label matches `second`'s label,
    else `first`'s — so an explicit label override that matches `first`
    propagates `first.cluster_id`.
    """
    merged_label = label if label is not None else second.label
    cluster_id = (
        second.cluster_id if merged_label == second.label else first.cluster_id
    )
    return _StructSegment(
        start=first.start,
        end=second.end,
        label=merged_label,
        confidence=float((first.confidence + second.confidence) * 0.5),
        cluster_id=cluster_id,
    )


# Mirrors references/python-source/config.py:56-59 AnalysisConfig. Inlined
# to keep the dump tool independent of remix_tool.config (which pulls
# audio_cache → torch via __init__ chain).
_STRUCT_MACRO_MIN_DURATION_SEC = 10.0
_STRUCT_BRIDGE_MERGE_MAX_SEC   = 8.0
_STRUCT_MAX_SEGMENTS           = 12


def _consolidation_apply(
    segments: list[_StructSegment], duration: float,
) -> list[_StructSegment]:
    """PARITY: segment_consolidation.py:83-193 _consolidate_segments.

    Deterministic: no RNG, no set/dict iteration order, only sequential
    list processing + integer indexing. Bit-exact C++ port achievable
    under IEEE 754 f64 arithmetic with identical predicate order +
    identical short-circuit semantics. The `while changed` outer loop
    terminates because every merge strictly decreases len(macro) OR sets
    changed=False when no predicate fires. The len > max_segments cap
    loop terminates because each iteration either decreases len(macro)
    by ≥ 1 or breaks (when interior is empty).
    """
    if not segments:
        return []

    macro = _consolidation_merge_same_label(segments)
    min_duration = _STRUCT_MACRO_MIN_DURATION_SEC
    bridge_merge_max = _STRUCT_BRIDGE_MERGE_MAX_SEC
    max_segments = _STRUCT_MAX_SEGMENTS

    changed = True
    while changed:
        changed = False
        new_segments: list[_StructSegment] = []
        i = 0
        while i < len(macro):
            current = macro[i]
            current_duration = current.end - current.start
            prev = new_segments[-1] if new_segments else None
            nxt = macro[i + 1] if i + 1 < len(macro) else None

            # Predicate 1 (L106-117): short "bridge" flanked by same-label
            # non-intro/outro neighbours → fuse triplet.
            if (
                prev is not None
                and nxt is not None
                and current_duration <= bridge_merge_max
                and current.label == "bridge"
                and prev.label == nxt.label
                and prev.label not in {"intro", "outro"}
            ):
                new_segments[-1] = _consolidation_merge_triplet(prev, current, nxt)
                i += 2
                changed = True
                continue

            # Predicate 2 (L119-128): short middle (<0.75 × min_duration)
            # flanked by same-label neighbours → fuse triplet.
            if (
                prev is not None
                and nxt is not None
                and current_duration < min_duration * 0.75
                and prev.label == nxt.label
            ):
                new_segments[-1] = _consolidation_merge_triplet(prev, current, nxt)
                i += 2
                changed = True
                continue

            # Predicate 3 (L130-139): very-short (<0.6 × min_duration) chorus /
            # verse / bridge absorbed into same-label predecessor.
            if (
                current_duration < min_duration * 0.6
                and current.label in {"bridge", "verse", "chorus"}
                and prev is not None
                and prev.label == current.label
            ):
                new_segments[-1] = _consolidation_merge_pair(
                    prev, current, prev.label,
                )
                i += 1
                changed = True
                continue

            new_segments.append(current)
            i += 1

        macro = _consolidation_merge_same_label(new_segments)

    # Cap total segment count at max_segments (L146-174): greedily fuse
    # the shortest interior segment with its context. Priority order:
    #   (a) triplet-merge when flanking segs share label;
    #   (b) pair-merge with the longer neighbour otherwise (ties → right).
    while len(macro) > max_segments:
        interior = [
            (idx, seg.end - seg.start)
            for idx, seg in enumerate(macro[1:-1], start=1)
        ]
        if not interior:
            break
        idx = min(interior, key=lambda item: item[1])[0]

        left = macro[idx - 1]
        current = macro[idx]
        right = macro[idx + 1] if idx + 1 < len(macro) else None

        if right is not None and left.label == right.label:
            merged = _consolidation_merge_triplet(left, current, right)
            macro = macro[: idx - 1] + [merged] + macro[idx + 2:]
        elif right is not None and (right.end - right.start) >= (left.end - left.start):
            macro = (
                macro[:idx]
                + [_consolidation_merge_pair(current, right, right.label)]
                + macro[idx + 2:]
            )
        else:
            macro = (
                macro[: idx - 1]
                + [_consolidation_merge_pair(left, current, left.label)]
                + macro[idx + 1:]
            )
        macro = _consolidation_merge_same_label(macro)

    # Position overrides + clamp to [0, duration] (L176-191).
    if macro:
        macro[0] = _StructSegment(
            start=0.0,
            end=macro[0].end,
            label=(
                "intro" if macro[0].start < duration * 0.15 else macro[0].label
            ),
            confidence=macro[0].confidence,
            cluster_id=macro[0].cluster_id,
        )
        last = macro[-1]
        macro[-1] = _StructSegment(
            start=last.start,
            end=duration,
            label="outro" if last.end > duration * 0.85 else last.label,
            confidence=last.confidence,
            cluster_id=last.cluster_id,
        )

    return macro


def _consolidation_dump(
    fields_in: np.ndarray, labels_in: list[str], duration: float,
) -> tuple[np.ndarray, list[str]]:
    """Package _consolidation_apply for dump-tool use.

    Input field layout mirrors session-6 (cbm_segments_fields) and session-8
    (novelty_segments_fields): f64 (n, 4) = [start, end, confidence,
    cluster_id]. Output has the same layout with possibly fewer rows after
    merges + relabels.
    """
    n = int(fields_in.shape[0]) if fields_in.ndim == 2 else 0
    if n == 0 or len(labels_in) == 0:
        return np.zeros((0, 4), dtype=np.float64), []
    if len(labels_in) != n:
        raise ValueError(
            f"_consolidation_dump: fields has {n} rows but labels has "
            f"{len(labels_in)} entries — input mismatch"
        )

    segments = [
        _StructSegment(
            start=float(fields_in[i, 0]),
            end=float(fields_in[i, 1]),
            label=str(labels_in[i]),
            confidence=float(fields_in[i, 2]),
            cluster_id=int(fields_in[i, 3]),
        )
        for i in range(n)
    ]

    consolidated = _consolidation_apply(segments, duration)

    out_n = len(consolidated)
    out_fields = np.zeros((out_n, 4), dtype=np.float64)
    out_labels: list[str] = []
    for i, seg in enumerate(consolidated):
        out_fields[i, 0] = seg.start
        out_fields[i, 1] = seg.end
        out_fields[i, 2] = seg.confidence
        out_fields[i, 3] = float(seg.cluster_id)
        out_labels.append(seg.label)

    return out_fields, out_labels


# --- phase-3 Recurrence reference helpers ------------------------------------
# Inlined verbatim from references/python-source/analysis/recurrence.py L32-146
# and config.py chroma_range(). Kept local to the dump tool so phase-3 keys
# regenerate without importing remix_tool.analysis (which pulls in audio_cache,
# CBM, sklearn-heavy paths that drift .npy SHAs). Same pattern as session-2
# NoveltyFeatures (_extract_novelty_features_25d).
#
# ADR-018 consequences: module parity uses librosa.beat.beat_track beats per
# phase-2 Fixture C precedent; production beat-this path validated separately
# in phase-4 integration. Input precision to build_recurrence_matrix is f32
# (matches production path: FeatureExtractor returns f32 features; numpy
# broadcasts to f64 inside NearestNeighbors / np.exp). Output R is f64.

# config.py:16-24 — chroma slice for N-dim feature vector (std=59 → 40..52).
_N_CHROMA_DIMS = 12
_N_CONTRAST_DIMS = 7

# recurrence.py:27-29 — combination weights.
_W_FEATURES = 0.15
_W_CHROMA = 0.35
_W_HOMOGENEITY = 0.50


def _chroma_range(n_features: int) -> tuple[int, int]:
    """PARITY: config.py:20-24."""
    start = max(0, n_features - _N_CHROMA_DIMS - _N_CONTRAST_DIMS)
    end = min(start + _N_CHROMA_DIMS, n_features)
    return start, end


def _recurrence_l2_normalize(X: np.ndarray) -> np.ndarray:
    """PARITY: recurrence.py:69-73 _l2_normalize."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _recurrence_mutual_knn(X_norm: np.ndarray, k: int) -> np.ndarray:
    """PARITY: recurrence.py:76-114 _mutual_knn_recurrence."""
    n = X_norm.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return np.eye(n, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    mu = float(np.median(distances[:, 1])) if n > 1 else 1.0
    mu = max(mu, 1e-8)

    knn_set = [set(int(j) for j in indices[i, 1:]) for i in range(n)]

    R = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in knn_set[i]:
            if R[i, j] > 0:
                continue
            dist_sq = float(np.sum((X_norm[i] - X_norm[j]) ** 2))
            val = np.exp(-dist_sq / mu)
            R[i, j] = val
            R[j, i] = val
    return R


def _recurrence_homogeneity(X_norm: np.ndarray) -> np.ndarray:
    """PARITY: recurrence.py:117-146 _homogeneity_matrix.

    Source audit (Hard Rule #8): the Python signature takes beat_times but
    the function body NEVER uses it (L134-138 operate purely on consecutive
    row diffs of X_norm). Port drops the argument accordingly.
    """
    n = X_norm.shape[0]
    R = np.zeros((n, n), dtype=np.float64)
    if n < 2:
        return R

    diffs = X_norm[1:] - X_norm[:-1]
    dist_sq = np.sum(diffs ** 2, axis=1)
    sigma_sq = float(np.median(dist_sq))
    sigma_sq = max(sigma_sq, 1e-8)

    for i in range(n - 1):
        val = np.exp(-dist_sq[i] / sigma_sq)
        R[i, i + 1] = val
        R[i + 1, i] = val
    return R


def _build_recurrence_reference(
    features_f32: np.ndarray,
    k_neighbors: int = 10,
) -> dict[str, np.ndarray]:
    """PARITY: recurrence.py:32-66 build_recurrence_matrix.

    Input: (n_beats, n_features) float32 — matches production path where
    FeatureExtractor.extract returns f32. Do NOT cast to f64 here: numpy
    broadcasts up to f64 inside NearestNeighbors / np.exp automatically;
    keeping the f32 surface on L2-normalize + consecutive-diff preserves
    the production arithmetic exactly.

    Returns dict of 4 matrices for bisection: R, R_feat, R_chroma, R_homo.
    R = 0.15·R_feat + 0.35·R_chroma + 0.50·R_homo (L65).
    """
    n = features_f32.shape[0]
    if n < 4:
        # recurrence.py:48-49 — n<4 short-circuit returns identity.
        eye = np.eye(n, dtype=np.float64)
        return {
            "R": eye,
            "R_feat":   np.zeros((n, n), dtype=np.float64),
            "R_chroma": np.zeros((n, n), dtype=np.float64),
            "R_homo":   np.zeros((n, n), dtype=np.float64),
        }

    cs, ce = _chroma_range(features_f32.shape[1])
    chroma = features_f32[:, cs:ce].copy()

    feat_norm   = _recurrence_l2_normalize(features_f32)
    chroma_norm = _recurrence_l2_normalize(chroma)

    R_feat   = _recurrence_mutual_knn(feat_norm,   k_neighbors)
    R_chroma = _recurrence_mutual_knn(chroma_norm, k_neighbors)
    R_homo   = _recurrence_homogeneity(feat_norm)

    R = _W_FEATURES * R_feat + _W_CHROMA * R_chroma + _W_HOMOGENEITY * R_homo
    return {"R": R, "R_feat": R_feat, "R_chroma": R_chroma, "R_homo": R_homo}


def _extract_waveform_snippets(
    y: np.ndarray, sr: int, beat_times: np.ndarray, n_beats: int,
    *, pre_ms: float, post_ms: float,
) -> np.ndarray:
    """PARITY: feature_extractor.py:424 _extract_waveform_snippets."""
    pre_samples = int(pre_ms * sr / 1000.0)
    post_samples = int(post_ms * sr / 1000.0)
    total_samples = max(8, pre_samples + post_samples)
    snippets = np.zeros((n_beats, total_samples), dtype=np.float32)

    if n_beats == 0:
        return snippets

    beat_samples = (beat_times * sr).astype(np.int64)
    n_audio_samples = len(y)
    window = np.hanning(total_samples).astype(np.float32)
    if np.all(window == 0):
        window = np.ones(total_samples, dtype=np.float32)

    for i in range(n_beats):
        center = int(beat_samples[i])
        start = center - pre_samples
        end = center + post_samples

        clip_start = max(0, start)
        clip_end = min(n_audio_samples, end)
        if clip_end <= clip_start:
            continue

        out_start = clip_start - start
        out_end = out_start + (clip_end - clip_start)
        snippet = snippets[i]
        snippet[out_start:out_end] = y[clip_start:clip_end]

        snippet -= np.mean(snippet)
        snippet *= window
        rms = np.sqrt(np.mean(snippet ** 2) + 1e-8)
        snippet /= rms

    return snippets


# --- core pipeline -----------------------------------------------------------
def compute_features(
    y: np.ndarray,
    sr: int,
    *,
    audio_path: Path | None = None,
    clip_range_sec: tuple[float, float] | None = None,
) -> dict[str, object]:
    """All phase-2 intermediate reference outputs, mirroring feature_extractor.py call sites.

    The `audio_path` + `clip_range_sec` kwargs are ADR-020 additions — they
    route phase-3 CBM goldens through `build/beat_detector_cli` on the full
    audio file (production-path downbeats). If `audio_path` is None the
    phase-3 CBM block is skipped with zero-length arrays (useful for unit
    testing the rest of the pipeline in isolation; --from-manifest always
    supplies a real path).
    """
    # STFT magnitude (amplitude, no normalization — librosa default).
    # Force C-order: librosa's rfft(axis=0) leaves the output F-contiguous,
    # which np.save serializes with fortran_order=True and trips our minimal
    # C++ npy reader. Asking for order='C' costs one copy at dump time and
    # makes the on-disk layout predictable.
    stft_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_magnitude = np.ascontiguousarray(np.abs(stft_complex), dtype=np.float64)

    # Mel power + log-mel (feeds MFCC path internally)
    mel_power = np.ascontiguousarray(librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    ), dtype=np.float64)
    log_mel = np.ascontiguousarray(librosa.power_to_db(mel_power, ref=np.max), dtype=np.float64)
    # ref=1.0 variant matches what librosa.feature.mfcc uses internally
    # (mfcc default path never sets ref=np.max). Used by the DCT parity
    # test to isolate the DCT step; differs from log_mel only by a uniform
    # shift of 10·log10(mel_power.max()), which moves only DCT coeff 0.
    log_mel_ref1 = np.ascontiguousarray(librosa.power_to_db(mel_power, ref=1.0), dtype=np.float64)

    # MFCC (two variants — standard 40-dim + fast 20-dim)
    mfcc_40 = np.ascontiguousarray(librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC_STD, hop_length=HOP_LENGTH, n_fft=N_FFT,
    ), dtype=np.float64)
    mfcc_20 = np.ascontiguousarray(librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC_FAST, hop_length=HOP_LENGTH, n_fft=N_FFT,
    ), dtype=np.float64)

    # Chroma — NOTE: tuning=None (librosa default) triggers estimate_tuning
    # (S-path), a data-dependent nonlinearity. See meta/RESEARCH.md 2026-04-20
    # phase-2 entry and ADR-010 (closed — port to C++).
    chroma_stft = np.ascontiguousarray(librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT,
    ), dtype=np.float64)

    # tuning_est — the exact scalar chroma_stft(tuning=None) passes to
    # filters.chroma. Mirrors librosa/feature/spectral.py L1274-1275:
    #   tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)
    # where S comes out of _spectrogram(..., power=2) as float32 |STFT|².
    # We reconstruct that exact path from the already-computed stft_complex
    # (complex64 → |·| float32 → ·² float32) so the scalar is bit-for-bit
    # what chroma_stft would compute. Stored as (1,) float64 to reuse the
    # existing npy pipeline; C++ parity test reads element [0].
    stft_power_f32 = np.abs(stft_complex).astype(np.float32) ** 2
    tuning_est = np.array(
        [float(librosa.estimate_tuning(S=stft_power_f32, sr=sr, bins_per_octave=12))],
        dtype=np.float64,
    )

    spectral_contrast = np.ascontiguousarray(librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT, n_bands=N_CONTRAST_BANDS,
    ), dtype=np.float64)  # shape (n_bands+1, T) = (7, T)

    onset_strength = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH,
    ).astype(np.float64)  # shape (T,)

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0].astype(np.float64)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH,
    )[0].astype(np.float64)

    # BeatSync parity fixture. Real beat frames come from phase-1 beat
    # detection, but parity for librosa.util.sync doesn't require musically
    # real beats — only a boundary list that exercises fix_frames + mean
    # aggregation. Synthetic arange(10, T, 20) gives ~129 beats on a 60 s
    # clip (hop=512, sr=22050) at ~130 BPM density, which is typical music.
    # Stride 20 avoids the degenerate "duplicate beat frames" path (which
    # np.unique would collapse silently) and exercises realistic slice
    # widths.
    T = mel_power.shape[-1]
    beat_frames = np.arange(10, T, 20, dtype=np.int64)
    # PARITY: librosa.util.sync defaults — pad=True, axis=-1, aggregate=np.mean.
    beat_sync_mel_power = np.ascontiguousarray(
        librosa.util.sync(mel_power, beat_frames, aggregate=np.mean),
        dtype=np.float64,
    )  # shape (n_mels, n_beats+1) after pad-dedup

    # Beat times derived from the synthetic beat_frames so the step-9
    # fixtures (which take beat_times) round-trip exactly through
    # librosa.time_to_frames — beat_frames ∈ np.arange(10, T, 20) sit at
    # bin centres, so `np.round(beat_times * sr / hop) = beat_frames`.
    beat_times = beat_frames.astype(np.float64) * HOP_LENGTH / SR

    # Step-9 reference outputs. Edge features take the feature matrix;
    # mel_power is used as the fixture matrix (cast to float32 to match
    # the orchestrator's stacked_features dtype — MFCC/chroma/contrast
    # are all float32 in the real pipeline).
    n_beats_synth = len(beat_frames)
    edge_features_start, edge_features_end = _extract_edge_features(
        mel_power.astype(np.float32, copy=False),
        beat_frames,
        n_beats_synth,
    )
    edge_rms_start, edge_rms_end = _extract_edge_rms(
        y, sr, beat_times, n_beats_synth,
    )
    boundary_waveforms = _extract_waveform_snippets(
        y, sr, beat_times, n_beats_synth,
        pre_ms=BOUNDARY_PRE_MS, post_ms=BOUNDARY_POST_MS,
    )
    transition_waveforms = _extract_waveform_snippets(
        y, sr, beat_times, n_beats_synth,
        pre_ms=TRANSITION_PRE_MS, post_ms=TRANSITION_POST_MS,
    )

    # Step-10 orchestrator outputs on the synthetic beats. Two modes emitted
    # in one pass because their shared pipeline (STFT → mel → MFCC/chroma/
    # contrast) is already computed above; only MFCC variant + transition
    # skip differ. Fast-mode mfcc width is 20 (vs 40), chroma + contrast
    # are identical to standard, transition_waveforms is omitted.
    # We cast each f64 dump back to f32 because the real pipeline computes
    # and stacks in f32 (librosa's natural dtype on float32 input). The
    # round-trip f32 → f64 → f32 is bit-exact (f64 stores the f32 exactly),
    # so this is identical to running the pipeline entirely in f32.
    mfcc_40_f32    = mfcc_40.astype(np.float32,    copy=False)
    mfcc_20_f32    = mfcc_20.astype(np.float32,    copy=False)
    chroma_f32     = chroma_stft.astype(np.float32, copy=False)
    contrast_f32   = spectral_contrast.astype(np.float32, copy=False)
    rms_f32        = rms.astype(np.float32,       copy=False)
    onset_f32      = onset_strength.astype(np.float32, copy=False)
    centroid_f32   = spectral_centroid.astype(np.float32, copy=False)

    orch_std = _run_orchestrator(
        y, sr, mfcc_40_f32, chroma_f32, contrast_f32,
        rms_f32, onset_f32, centroid_f32,
        beat_times, skip_transition_waveforms=False,
        skip_vocal_features=False,  # std mode runs vocal per ADR-014 + ADR-015
    )
    orch_fast = _run_orchestrator(
        y, sr, mfcc_20_f32, chroma_f32, contrast_f32,
        rms_f32, onset_f32, centroid_f32,
        beat_times, skip_transition_waveforms=True,
        skip_vocal_features=True,   # fast mode skips vocal (Result fields stay empty)
    )

    # Phase-2b vocal features. Default-mode only (ADR-014). Uses the
    # synthetic beats already computed above — the C++ VocalFeatures
    # parity test feeds the same beats into the C++ port, so any
    # frame/beat drift between the two sides isolates to modules under
    # test, not to beat source.
    vocal = _extract_vocal_features_dump(
        y, sr, HOP_LENGTH, N_FFT, beat_frames, n_beats_synth,
    )

    # Phase-3 novelty features (25-dim) per ADR-019. Bonus: standard-mode
    # SSM (stride=4) as a future-proofing dump for NoveltySegmenter; the
    # matmul is float32 naive inner-product on the 25-dim z-scored rows
    # so regeneration cost is trivial. Session-2 scope discipline: only
    # novelty_features_25d is ACTIVELY tested; ssm rides along.
    novelty_features_25d = _extract_novelty_features_25d(y, sr)
    novelty_ssm_stride4  = _compute_novelty_ssm(novelty_features_25d, stride=4)

    # ADR-021 session-7 NoveltySegmenter pre-clustering dumps. SSM stride is
    # the stride-4 already dumped above — novelty+boundaries+embeddings are
    # deterministic functions of (features, ssm, duration) and do not change
    # any existing key. `min_segment_duration=8.0` matches StructureAnalyzer
    # default (__init__ L66) + production call site. `duration` is derived
    # from y.shape[0] / sr (same as structure_analyzer.py:119).
    _ssm_t_frames = int(novelty_features_25d.shape[1])  # for frame_duration
    _duration = float(len(y)) / float(sr)
    # IMPORTANT: _compute_novelty is keyed on SSM length (T_ds after stride-4),
    # but _find_boundaries uses `duration / len(novelty)` as the frame
    # duration — i.e., the novelty curve spans the clip in n = T_ds samples.
    # We MUST use the same stride-4 SSM that Python produces.
    novelty_curve      = _compute_novelty_curve(novelty_ssm_stride4)
    novelty_peaks_raw, novelty_boundaries = _find_novelty_peaks_and_boundaries(
        novelty_curve, _duration, min_segment_duration=8.0,
    )
    novelty_embeddings = _compute_novelty_segment_embeddings(
        novelty_features_25d, novelty_boundaries, sr, HOP_LENGTH,
    )

    # ADR-021 session-8 clustering + labeling dumps. Inputs to the C++ parity
    # test: novelty_embeddings (from session 7) + novelty_boundaries +
    # novelty_curve + y. Outputs: (affinity, silhouette_scores, best_k,
    # cluster_ids, segments_fields, segments_labels). Gate design per ADR-021:
    # Hungarian-matched cluster_id agreement ≥ 90 % + exact label ≥ 90 % (k-means++
    # RNG + eigenvector signs make bitwise unattainable regardless of C++ impl).
    # `fast_mode=False` matches standard-mode StructureAnalyzer default.
    _cluster_result = _cluster_novelty_segments(novelty_embeddings, fast_mode=False)
    novelty_affinity          = _cluster_result["affinity"]            # f32 (n_segs, n_segs)
    novelty_silhouette_scores = _cluster_result["silhouette_scores"]   # f64 (len(ks),)
    novelty_best_k            = _cluster_result["best_k"]              # i64 scalar
    novelty_cluster_ids       = _cluster_result["cluster_ids"]         # i64 (n_segs,)
    novelty_segments_fields, novelty_segments_labels = _create_novelty_segments(
        novelty_boundaries, novelty_cluster_ids, _duration,
        novelty_curve, y, sr,
    )

    # --- Phase-3 Recurrence references (session 3, ADR-018) -----------------
    # Real beats via librosa.beat.beat_track (phase-2 Fixture C precedent).
    # ADR-018 consequences: module parity uses beat-agnostic inputs — both
    # Python dump and C++ port consume the SAME beat_times + feature_matrix
    # from these goldens, so beat-source choice in the dump tool is arbitrary
    # for module parity. Production beat-this pipeline is validated in
    # phase-4 integration testing (end-to-end), not here.
    _, beat_frames_rec = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_times_rec = librosa.frames_to_time(
        beat_frames_rec, sr=sr, hop_length=HOP_LENGTH,
    ).astype(np.float64)
    n_rec = int(len(beat_times_rec))
    if n_rec == 0:
        raise SystemExit(
            "librosa.beat.beat_track produced zero beats on this clip — "
            "choose a different clip_range_sec"
        )

    # Reuse the existing f32 feature grids already computed above (mfcc_40,
    # chroma_stft, spectral_contrast, rms, onset, spectral_centroid). These
    # are shared with the synthetic-beats orch_std dump — they're frame-level
    # feature grids, independent of beat_times. Only the subsequent
    # beat-sync / L2-norm stage differs between the two dumps.
    orch_rec = _run_orchestrator(
        y, sr, mfcc_40_f32, chroma_f32, contrast_f32,
        rms_f32, onset_f32, centroid_f32,
        beat_times_rec,
        skip_transition_waveforms=True,  # unused by Recurrence
        skip_vocal_features=True,        # unused by Recurrence
    )
    # feature_matrix is stored as f64 in orch_rec (cast from f32 inside
    # _run_orchestrator at L402). Round-trip back to f32 is bit-exact
    # (f64 exactly represents every f32 value) and matches the production
    # path that passes f32 features into build_recurrence_matrix.
    recurrence_feature_matrix_f64 = orch_rec["feature_matrix"]
    recurrence_feature_matrix_f32 = recurrence_feature_matrix_f64.astype(
        np.float32, copy=False,
    )
    rec = _build_recurrence_reference(recurrence_feature_matrix_f32, k_neighbors=10)

    # --- Phase-3 CBM references (session 4, ADR-020) -----------------------
    # Beat + downbeat source = C++ BeatDetector via beat_detector_cli
    # (production path). Run on the FULL audio file (not clip), then filter
    # to the clip range and shift to clip-local seconds. Production CBM
    # consumes beats + downbeats from the SAME BeatDetector.detect() call
    # (aligned by construction: every downbeat is a beat); we preserve
    # that invariant by taking both from one subprocess invocation.
    #
    # feature_matrix_cbm is a SEPARATE beat-sync pass on these C++ beats
    # (not reusable from recurrence_feature_matrix, which is aligned to
    # librosa.beat.beat_track beats per ADR-018). Rationale: CBM requires
    # beat_times and feature_matrix aligned to the SAME beat source as
    # downbeats — mismatched sources cause "Map each downbeat to nearest
    # beat index" (cbm_segmenter.py:55-60) to misalign, producing wrong
    # bar boundaries. Quality-first: extra _run_orchestrator pass costs
    # ~2 s/track; acceptable for production-path parity.
    if audio_path is not None:
        beats_cpp, downbeats_cpp = _export_beats_and_downbeats_via_cpp(
            audio_path, clip_range_sec,
        )
        if len(beats_cpp) == 0:
            raise SystemExit(
                f"beat_detector_cli produced zero beats in clip range "
                f"{clip_range_sec} on {audio_path} — adjust clip_range_sec."
            )
        # Beat-sync the shared f32 feature grids onto beats_cpp via the
        # same orchestrator used for recurrence/phase-2. skip_transition +
        # skip_vocal_features: CBM doesn't consume these, saves a few seconds.
        orch_cbm = _run_orchestrator(
            y, sr, mfcc_40_f32, chroma_f32, contrast_f32,
            rms_f32, onset_f32, centroid_f32,
            beats_cpp,
            skip_transition_waveforms=True,
            skip_vocal_features=True,
        )
        feature_matrix_cbm_f64 = orch_cbm["feature_matrix"]
        feature_matrix_cbm_f32 = feature_matrix_cbm_f64.astype(
            np.float32, copy=False,
        )
        _cbm = _load_cbm_reference_module()

        # Session 5 (port-order step 4 C++) bisection aids: capture DP-pipeline
        # intermediates (bar_features, bar_times, autosim, bar_segments) that
        # cbm_analyze consumes internally but never returns. These feed the
        # CBMSegmenter parity test which gates on bar_segments (exact integer
        # match) and uses bar_features + autosim as bisection diagnostics if
        # the gate fails. Replay steps 1-3 of cbm_analyze (same function calls,
        # same inputs) — tanio (matmul on ≤~60 bars) and does NOT change the
        # cbm_analyze output since Python is deterministic on identical inputs.
        cbm_bar_features_f32, cbm_bar_times = _cbm.compute_bar_features(
            feature_matrix_cbm_f32, beats_cpp, downbeats_cpp,
        )
        if cbm_bar_features_f32.shape[0] >= 2:
            cbm_autosim_f32 = _cbm.compute_bar_autosimilarity(cbm_bar_features_f32)
            cbm_bar_segments_py = _cbm.cbm_segment(cbm_autosim_f32)
        else:
            cbm_autosim_f32 = np.zeros((0, 0), dtype=np.float32)
            cbm_bar_segments_py = []
        if cbm_bar_segments_py:
            cbm_bar_segments_arr = np.asarray(
                cbm_bar_segments_py, dtype=np.int64,
            ).reshape(-1, 2)
        else:
            cbm_bar_segments_arr = np.zeros((0, 2), dtype=np.int64)

        cbm_segments_py = _cbm.cbm_analyze(
            features=feature_matrix_cbm_f32,
            beat_times=beats_cpp,
            downbeats=downbeats_cpp,
            audio_mono=y,
            sample_rate=sr,
        )
        # cbm_analyze returns Optional[List[dict]] — None when bars<4 or
        # segments<3 (degenerate). Preserve that distinction on the wire
        # (empty list vs legitimate segments) so C++ parity can match
        # the fallback behavior faithfully.
        if cbm_segments_py is None:
            cbm_segments_list: list[dict] = []
            cbm_boundaries = np.zeros(0, dtype=np.float64)
        else:
            cbm_segments_list = [
                {
                    "start":      float(seg["start"]),
                    "end":        float(seg["end"]),
                    "label":      str(seg["label"]),
                    "confidence": float(seg["confidence"]),
                    "cluster_id": int(seg["cluster_id"]),
                }
                for seg in cbm_segments_py
            ]
            boundaries = [cbm_segments_list[0]["start"]] + [
                seg["end"] for seg in cbm_segments_list
            ]
            cbm_boundaries = np.asarray(boundaries, dtype=np.float64)

        # --- Session 6 (port-order step 4 C++ continuation) bisection aids
        # + structured-binary golden for labelSegments + cbmAnalyze parity.
        # These are emitted BEFORE `labelSegments` is called; they replay
        # its internal intermediates (L278-295 of cbm_segmenter.py) +
        # `cbm_analyze` step 4 (L422-433, bar_energy RMS) on the same
        # inputs the C++ test feeds in. Determinism: Python is stateless
        # on identical inputs, so replaying does NOT change cbm_segments
        # output. Zero SHA drift on existing 730 keys verified at regen.
        #
        # Rationale per session-6 decision 2: C++ test consumes structured
        # binary/text (not JSON) — cbm_segments_fields + cbm_segments_labels
        # replace a JSON parser bug-class entirely. cbm_seg_sim +
        # cbm_seg_normed_centroids + cbm_bar_energy are bisection aids for
        # the two ULP-sensitive boundaries in labelSegments (clustering
        # threshold 0.80 + energy-rank tiebreak).
        n_bars_ref = int(cbm_bar_features_f32.shape[0])
        if cbm_segments_py is None or n_bars_ref < 4:
            # Degenerate: zero-sized aids. Preserved so the dump schema
            # stays well-typed across tracks.
            cbm_seg_normed_centroids = np.zeros(
                (0, cbm_bar_features_f32.shape[1] if cbm_bar_features_f32.ndim == 2 else 0),
                dtype=np.float32,
            )
            cbm_seg_sim = np.zeros((0, 0), dtype=np.float32)
            cbm_bar_energy = np.zeros(n_bars_ref, dtype=np.float64)
            cbm_segments_fields = np.zeros((0, 4), dtype=np.float64)
            cbm_segments_labels: list[str] = []
        else:
            # --- bar_energy (PARITY: cbm_segmenter.py:422-433) -----------
            # Replay `cbm_analyze` step 4 on the same (y, sr, cbm_bar_times)
            # that `_cbm.cbm_analyze(...)` consumed internally. numpy mean
            # on f32 uses f32 accumulator with pairwise reduction; we call
            # np.mean directly so the C++ test's `cbm_bar_energy.npy`
            # golden IS bit-for-bit what cbm_analyze produced internally.
            cbm_bar_energy = np.zeros(n_bars_ref, dtype=np.float64)
            for _i in range(n_bars_ref):
                _t0 = float(cbm_bar_times[_i])
                _t1 = float(cbm_bar_times[min(_i + 1, len(cbm_bar_times) - 1)])
                _s0 = int(_t0 * sr)
                _s1 = int(_t1 * sr)
                _s0 = max(0, min(_s0, len(y) - 1))
                _s1 = max(_s0 + 1, min(_s1, len(y)))
                cbm_bar_energy[_i] = float(
                    np.sqrt(np.mean(y[_s0:_s1] ** 2))
                )

            # --- seg centroids + L2 norm + sim (PARITY: L278-295) --------
            # Replay `label_segments` internals on bar_features sliced by
            # the segmentation cbm_analyze used. `bar_segments` from the
            # session-5 intermediate is the same list cbm_analyze passed.
            bar_segments_tuples = [
                (int(r[0]), int(r[1])) for r in cbm_bar_segments_arr.tolist()
            ]
            _seg_centroids: list[np.ndarray] = []
            _n_feat = int(cbm_bar_features_f32.shape[1])
            for _start_bar, _end_bar in bar_segments_tuples:
                _end_bar = min(_end_bar, n_bars_ref)
                if _end_bar > _start_bar:
                    _seg_centroids.append(
                        cbm_bar_features_f32[_start_bar:_end_bar].mean(axis=0)
                    )
                else:
                    _seg_centroids.append(
                        np.zeros(_n_feat, dtype=cbm_bar_features_f32.dtype)
                    )
            # np.array on list of f32 rows stays f32 (all elements same dtype).
            _centroids = np.asarray(_seg_centroids, dtype=np.float32)
            _norms = np.linalg.norm(_centroids, axis=1, keepdims=True)
            _norms = np.where(_norms < 1e-8, 1.0, _norms).astype(np.float32)
            cbm_seg_normed_centroids = (_centroids / _norms).astype(
                np.float32, copy=False,
            )
            cbm_seg_sim = (
                cbm_seg_normed_centroids @ cbm_seg_normed_centroids.T
            ).astype(np.float32, copy=False)

            # --- segments_fields + labels as structured goldens ----------
            # C++ test consumes these; no JSON parsing required.
            cbm_segments_fields = np.asarray(
                [
                    [
                        float(seg["start"]),
                        float(seg["end"]),
                        float(seg["confidence"]),
                        float(int(seg["cluster_id"])),
                    ]
                    for seg in cbm_segments_list
                ],
                dtype=np.float64,
            )
            cbm_segments_labels = [str(seg["label"]) for seg in cbm_segments_list]
            # Cross-check: structured goldens must be equivalent to
            # cbm_segments.json (same atomic source: cbm_segments_list).
            # Assertion is cheap (n_segs ≤ 12 on our corpus) and catches
            # dump-tool bugs BEFORE a parity run fails downstream.
            assert len(cbm_segments_labels) == cbm_segments_fields.shape[0]
            for _i, _seg in enumerate(cbm_segments_list):
                assert cbm_segments_fields[_i, 0] == float(_seg["start"])
                assert cbm_segments_fields[_i, 1] == float(_seg["end"])
                assert cbm_segments_fields[_i, 2] == float(_seg["confidence"])
                assert int(cbm_segments_fields[_i, 3]) == int(_seg["cluster_id"])
                assert cbm_segments_labels[_i] == str(_seg["label"])
    else:
        # Unit-test / --audio path without clip_range_sec: emit empty
        # phase-3-CBM outputs so the rest of the return dict stays
        # well-typed. Not used by --from-manifest.
        beats_cpp = np.zeros(0, dtype=np.float64)
        downbeats_cpp = np.zeros(0, dtype=np.float64)
        feature_matrix_cbm_f64 = np.zeros((0, 59), dtype=np.float64)
        cbm_bar_features_f32 = np.zeros((0, 59), dtype=np.float32)
        cbm_bar_times = np.zeros(0, dtype=np.float64)
        cbm_autosim_f32 = np.zeros((0, 0), dtype=np.float32)
        cbm_bar_segments_arr = np.zeros((0, 2), dtype=np.int64)
        cbm_segments_list = []
        cbm_boundaries = np.zeros(0, dtype=np.float64)
        cbm_seg_normed_centroids = np.zeros((0, 59), dtype=np.float32)
        cbm_seg_sim = np.zeros((0, 0), dtype=np.float32)
        cbm_bar_energy = np.zeros(0, dtype=np.float64)
        cbm_segments_fields = np.zeros((0, 4), dtype=np.float64)
        cbm_segments_labels = []

    # --- Phase-3 session-9 SegmentConsolidation dumps -----------------------
    # Replay _consolidate_segments(segments, duration) on both the CBM-path
    # and novelty-path segment lists. `duration` matches dispatcher L119
    # (`len(y) / sr` in f64). The dispatcher calls consolidation INSIDE the
    # CBM/novelty branches, so the outputs below are what the full
    # StructureAnalyzer.analyze() pipeline would emit BEFORE the dispatcher's
    # boundaries-rebuild step (L152-155 / L173-177). Session-9 parity test
    # feeds these fields+labels into C++ SegmentConsolidation::consolidate
    # and gates strict bitwise.
    #
    # Both sides are dumped even though in production only one path fires per
    # track — the module-level parity test wants coverage of both input
    # distributions (CBM-produced segments have different label distributions
    # and durations than novelty-produced ones).
    cbm_consolidated_segments_fields, cbm_consolidated_segments_labels = (
        _consolidation_dump(
            cbm_segments_fields, cbm_segments_labels, _duration,
        )
    )
    novelty_consolidated_segments_fields, novelty_consolidated_segments_labels = (
        _consolidation_dump(
            novelty_segments_fields, novelty_segments_labels, _duration,
        )
    )

    # --- Phase-3 session-10 StructureAnalyzer dispatcher dumps --------------
    # PARITY: structure_analyzer.py:127-182 `_analyze` dispatcher. Replays the
    # predicate + branch composition on SAME inputs a production pipeline
    # would feed (beats_cpp, downbeats_cpp, feature_matrix_cbm). Result is the
    # `(path, segments_fields, segments_labels, boundaries)` tuple AFTER
    # consolidation — what the full StructureAnalyzer.analyze() would return.
    #
    # Predicate (L127-142): CBM branch fires when
    #   `not fast_mode AND beat_times is not None AND downbeats is not None
    #    AND len(downbeats) >= 4 AND beat_features is not None`
    # AND the subsequent output-count gate `cbm_segments AND len(cbm_segments)
    #   >= 5` holds. All 10-track corpus runs fast_mode=False with all three
    # CBM inputs present; the `>= 5` gate on cbm_analyze output is what splits
    # the corpus (session-4 counts: shostakovich 7, vocal_solo 7, meshuggah 12
    # → CBM; the other 7 tracks → novelty fallback).
    #
    # Consolidation ran above for BOTH branches; we pick whichever matches the
    # dispatched path. Boundary rebuild mirrors Python L152-155 / L174-177:
    # `[segments[0].start] + [seg.end for seg in segments]` as f64.
    #
    # Empty-segment edge case: CBM branch (L155) returns `[0.0, duration]`;
    # novelty branch (L173-177) leaves the pre-consolidation _find_boundaries
    # output. No corpus track hits this case (min 1 consolidated segment);
    # we mirror the CBM branch's `[0.0, duration]` for both for consistency.
    _cbm_eligible = bool(
        len(downbeats_cpp) >= 4
        and feature_matrix_cbm_f64.size > 0
        and cbm_segments_list
        and len(cbm_segments_list) >= 5
    )
    if _cbm_eligible:
        _dispatched_path_val       = "cbm"
        dispatched_segments_fields = cbm_consolidated_segments_fields
        dispatched_segments_labels = list(cbm_consolidated_segments_labels)
    else:
        _dispatched_path_val       = "novelty"
        dispatched_segments_fields = novelty_consolidated_segments_fields
        dispatched_segments_labels = list(novelty_consolidated_segments_labels)

    if dispatched_segments_fields.shape[0] > 0:
        dispatched_boundaries = np.concatenate(
            [
                np.asarray([float(dispatched_segments_fields[0, 0])], dtype=np.float64),
                dispatched_segments_fields[:, 1].astype(np.float64, copy=False),
            ]
        ).astype(np.float64, copy=False)
    else:
        dispatched_boundaries = np.array([0.0, _duration], dtype=np.float64)

    # Wrap single-string in a 1-element list so the dump-loop's list-of-str
    # dispatch emits .txt (not .json). C++ reads via std::getline and takes
    # the first line.
    dispatched_path = [_dispatched_path_val]

    # --- Phase-3 session-11 RepetitionMap replay ----------------------------
    # PARITY: repetition_map.py:284-532 `build_repetition_map`. Replays the
    # full 2-phase algorithm (cross-section chroma xcorr + internal recurrence
    # diagonals + waveform verification + final sort) on the SAME inputs the
    # production pipeline feeds — `beats_cpp` + `downbeats_cpp` (beat-this via
    # beat_detector_cli, ADR-020), `feature_matrix_cbm_f64` (beat-synced 59-dim
    # on beats_cpp, session-4 convention), `dispatched_*` segments (session-10
    # StructureAnalyzer output), `_extract_waveform_snippets(pre=35, post=120)`
    # on beats_cpp. Result is the full `RepetitionMap` dataclass the C++
    # port must reproduce; we serialize it as nine per-field .npy arrays (all
    # i64/f64 — C++ reads without dtype inference) + section_pairs i64 [n, 2]
    # + a 2-line counts.txt. Gate: ≥ 90 % agreement on pairs with
    # waveform_similarity > 0.6 (phase-3 spec L35).
    #
    # Import note: `build_repetition_map` is imported from the live
    # remix_tool package via the sys.path shim set up at module-load (see
    # `_REMIX_TOOL_PKG_PARENT`). Unlike cbm_segmenter (ORPHAN module to avoid
    # pulling torch), repetition_map is allowed to transit through
    # `remix_tool.__init__` because .venv-phase2 already has torch available
    # and the alternative (stubbing four package-level shims) would add more
    # LOC than the shim import itself.
    if len(beats_cpp) > 0 and feature_matrix_cbm_f64.size > 0:
        _rep_segments_list = [
            {
                "start":      float(dispatched_segments_fields[i, 0]),
                "end":        float(dispatched_segments_fields[i, 1]),
                "confidence": float(dispatched_segments_fields[i, 2]),
                "cluster_id": int(dispatched_segments_fields[i, 3]),
                "label":      str(dispatched_segments_labels[i]),
            }
            for i in range(dispatched_segments_fields.shape[0])
        ]
        # Boundary waveforms on the PRODUCTION beats (beats_cpp). Distinct
        # from the existing `boundary_waveforms` key which is on synthetic
        # beats (Fixture A). We do NOT emit this as a golden — C++ recomputes
        # via `BeatWindows::extractWaveformSnippets` which is phase-2 validated
        # bitwise against `_extract_waveform_snippets` (see phase-2 VALIDATION).
        _rep_boundary_waveforms = _extract_waveform_snippets(
            y, SR, beats_cpp, int(len(beats_cpp)),
            pre_ms=BOUNDARY_PRE_MS, post_ms=BOUNDARY_POST_MS,
        )
        from remix_tool.analysis.repetition_map import build_repetition_map  # lazy
        _rep_map = build_repetition_map(
            beat_times=beats_cpp,
            downbeats=downbeats_cpp,
            features=feature_matrix_cbm_f64,
            segments=_rep_segments_list,
            boundary_waveforms=_rep_boundary_waveforms,
            waveform_sample_rate=int(SR),
        )
        _jumps = _rep_map.jumps
        _n_jumps = len(_jumps)
        repetition_from_beat              = np.asarray([j.from_beat for j in _jumps],              dtype=np.int64).reshape(-1)
        repetition_to_beat                = np.asarray([j.to_beat for j in _jumps],                dtype=np.int64).reshape(-1)
        repetition_waveform_similarity    = np.asarray([j.waveform_similarity for j in _jumps],    dtype=np.float64).reshape(-1)
        repetition_chroma_correlation     = np.asarray([j.chroma_correlation for j in _jumps],     dtype=np.float64).reshape(-1)
        repetition_alignment_lag_samples  = np.asarray([j.alignment_lag_samples for j in _jumps],  dtype=np.int64).reshape(-1)
        repetition_from_section_idx       = np.asarray([j.from_section_idx for j in _jumps],       dtype=np.int64).reshape(-1)
        repetition_to_section_idx         = np.asarray([j.to_section_idx for j in _jumps],         dtype=np.int64).reshape(-1)
        repetition_from_bar               = np.asarray([j.from_bar for j in _jumps],               dtype=np.int64).reshape(-1)
        repetition_to_bar                 = np.asarray([j.to_bar for j in _jumps],                 dtype=np.int64).reshape(-1)
        if _rep_map.section_pairs:
            repetition_section_pairs = np.asarray(
                _rep_map.section_pairs, dtype=np.int64,
            ).reshape(-1, 2)
        else:
            repetition_section_pairs = np.zeros((0, 2), dtype=np.int64)
        repetition_counts = [
            f"n_sections_scanned={_rep_map.n_sections_scanned}",
            f"n_pairs_verified={_rep_map.n_pairs_verified}",
        ]
    else:
        # Zero-beat or zero-feature fallback: emit empty arrays + zero counts.
        # Matches Python's `len(beat_times) < 8 or len(segments) < 2` early-out
        # behavior (L317-318) which returns an empty RepetitionMap.
        repetition_from_beat              = np.zeros(0, dtype=np.int64)
        repetition_to_beat                = np.zeros(0, dtype=np.int64)
        repetition_waveform_similarity    = np.zeros(0, dtype=np.float64)
        repetition_chroma_correlation     = np.zeros(0, dtype=np.float64)
        repetition_alignment_lag_samples  = np.zeros(0, dtype=np.int64)
        repetition_from_section_idx       = np.zeros(0, dtype=np.int64)
        repetition_to_section_idx         = np.zeros(0, dtype=np.int64)
        repetition_from_bar               = np.zeros(0, dtype=np.int64)
        repetition_to_bar                 = np.zeros(0, dtype=np.int64)
        repetition_section_pairs          = np.zeros((0, 2), dtype=np.int64)
        repetition_counts = [
            "n_sections_scanned=0",
            "n_pairs_verified=0",
        ]

    return {
        "stft_magnitude":    stft_magnitude,
        "mel_power":         mel_power,
        "log_mel":           log_mel,
        "log_mel_ref1":      log_mel_ref1,
        "mfcc_40":           mfcc_40,
        "mfcc_20":           mfcc_20,
        "chroma_stft":       chroma_stft,
        "tuning_est":        tuning_est,
        "spectral_contrast": spectral_contrast,
        "onset_strength":    onset_strength,
        "rms":               rms,
        "spectral_centroid": spectral_centroid,
        "beat_frames":           beat_frames,          # int64
        "beat_times":            beat_times,           # float64
        "beat_sync_mel_power":   beat_sync_mel_power,  # float64
        "edge_features_start":   edge_features_start,  # float64 (n_beats, n_mels)
        "edge_features_end":     edge_features_end,    # float64
        "edge_rms_start":        edge_rms_start,       # float64 (n_beats,)
        "edge_rms_end":          edge_rms_end,         # float64
        "boundary_waveforms":    boundary_waveforms,   # float32 (n_beats, total)
        "transition_waveforms":  transition_waveforms, # float32
        # --- step-10 orchestrator references (synthetic beats) ------------
        "orch_std_feature_matrix":       orch_std["feature_matrix"],        # f64 (n_beats, 59)
        "orch_std_edge_features_start":  orch_std["edge_features_start"],   # f64 (n_beats, 59)
        "orch_std_edge_features_end":    orch_std["edge_features_end"],     # f64 (n_beats, 59)
        "orch_std_rms_energy":           orch_std["rms_energy"],            # f64 (n_beats,)
        "orch_std_onset_strength":       orch_std["onset_strength"],        # f64 (n_beats,)
        "orch_std_spectral_centroid":    orch_std["spectral_centroid"],     # f64 (n_beats,)
        "orch_fast_feature_matrix":      orch_fast["feature_matrix"],       # f64 (n_beats, 39)
        "orch_fast_edge_features_start": orch_fast["edge_features_start"],  # f64 (n_beats, 39)
        "orch_fast_edge_features_end":   orch_fast["edge_features_end"],    # f64 (n_beats, 39)
        # --- step-10 orchestrator vocal outputs (std mode only; ADR-014 + ADR-015) ---
        # Round-tripped beat_frames (= librosa.time_to_frames(beat_times)) so
        # these match the flow used by C++ FeatureExtractor::extract. Differs
        # from the raw-arange "vocal_activity.npy" above: on some synthetic
        # frames (e.g. beat_frame=30 @ sr=22050) time_to_frames collapses to
        # frame-1 via the two-step int-cast landmine (ADR-013), moving slice
        # boundaries. Test_vocal_features uses raw-arange; test_feature_extractor
        # Fixture A uses these round-tripped dumps.
        "orch_std_vocal_activity":            orch_std["vocal_activity"],             # f64 (n_beats,)
        "orch_std_voiced_ratio":              orch_std["voiced_ratio"],               # f64 (n_beats,) zeros
        "orch_std_f0_hz":                     orch_std["f0_hz"],                      # f64 (n_beats,) zeros
        "orch_std_f0_confidence":             orch_std["f0_confidence"],              # f64 (n_beats,) zeros
        "orch_std_edge_vocal_activity_start": orch_std["edge_vocal_activity_start"],  # f64 (n_beats,)
        "orch_std_edge_vocal_activity_end":   orch_std["edge_vocal_activity_end"],    # f64 (n_beats,)
        "orch_std_edge_vocal_onset_start":    orch_std["edge_vocal_onset_start"],     # f64 (n_beats,)
        "orch_std_edge_vocal_release_end":    orch_std["edge_vocal_release_end"],     # f64 (n_beats,)
        # --- phase-2b vocal references (synthetic beats, default-mode only) ---
        "hpss_y_harmonic":           vocal["hpss_y_harmonic"],          # f32 time-domain
        "stft_magnitude_harmonic":   vocal["stft_magnitude_harmonic"],  # f32 (bins, frames) on y_harmonic
        "spectral_flatness":         vocal["spectral_flatness"],        # f32 per frame
        "harmonic_ratio":            vocal["harmonic_ratio"],           # f64 per frame
        "flatness_inv":              vocal["flatness_inv"],             # f64 per frame
        "voice_band_ratio":          vocal["voice_band_ratio"],         # f64 per frame
        "vocal_activity_frame":      vocal["vocal_activity_frame"],     # f64 per frame
        "vocal_rise_frame":          vocal["vocal_rise_frame"],         # f64 per frame
        "vocal_fall_frame":          vocal["vocal_fall_frame"],         # f64 per frame
        "vocal_activity":            vocal["vocal_activity"],           # f64 per beat
        "voiced_ratio":              vocal["voiced_ratio"],             # f64 per beat (zeros; ADR-014)
        "f0_hz":                     vocal["f0_hz"],                    # f64 per beat (zeros; ADR-014)
        "f0_confidence":             vocal["f0_confidence"],            # f64 per beat (zeros; ADR-014)
        "edge_vocal_activity_start": vocal["edge_vocal_activity_start"],# f64 per beat
        "edge_vocal_activity_end":   vocal["edge_vocal_activity_end"],  # f64 per beat
        "edge_vocal_onset_start":    vocal["edge_vocal_onset_start"],   # f64 per beat
        "edge_vocal_release_end":    vocal["edge_vocal_release_end"],   # f64 per beat
        # --- phase-3 structure-analysis references (ADR-019 session-2 scope) ---
        "novelty_features_25d":      novelty_features_25d,              # f32 (25, T)
        "novelty_ssm_stride4":       novelty_ssm_stride4,               # f32 (T_ds, T_ds)
        # ADR-021 session-7 NoveltySegmenter pre-clustering goldens.
        # Gates: novelty_curve L∞ ≤ 1e-9 (f64 checkerboard + uniform_filter1d),
        # novelty_peaks_raw integer exact, novelty_boundaries bitwise f64,
        # novelty_embeddings bitwise f32 (pre-clustering, strict).
        "novelty_curve":             novelty_curve,                     # f64 (T_ds,)
        "novelty_peaks_raw":         novelty_peaks_raw,                 # i64 (n_peaks_pre_margin,)
        "novelty_boundaries":        novelty_boundaries,                # f64 (n_segments+1,)
        "novelty_embeddings":        novelty_embeddings,                # f32 (n_segments, 25)
        # ADR-021 session-8 NoveltySegmenter clustering + labeling goldens.
        # Gate design: Hungarian-matched cluster_id agreement ≥ 90 % (affinity
        # + silhouette bisection aids), exact label ≥ 90 %, confidence/start/end
        # deterministic given bit-exact novelty + boundaries (inherited from
        # session-7 bitwise gate). See ADR-021 § Parity gate design.
        "novelty_affinity":          novelty_affinity,                  # f32 (n_segs, n_segs)
        "novelty_silhouette_scores": novelty_silhouette_scores,         # f64 (len(ks),)
        "novelty_best_k":            novelty_best_k,                    # i64 scalar (0 if n≤1)
        "novelty_cluster_ids":       novelty_cluster_ids,               # i64 (n_segs,)
        "novelty_segments_fields":   novelty_segments_fields,           # f64 (n_segs, 4)
        "novelty_segments_labels":   novelty_segments_labels,           # List[str] → .txt
        # --- phase-3 Recurrence references (session 3, ADR-018) -----------
        # Input lanes (consumed by C++ Recurrence parity test):
        "recurrence_beat_times":       beat_times_rec,                  # f64 (n_rec,)
        "recurrence_feature_matrix":   recurrence_feature_matrix_f64,   # f64 (n_rec, 59)
        # Output lanes: R is the primary parity target; 3 components are
        # bisection aids if L∞ drift spikes on a single track.
        "R_matrix":                    rec["R"],                        # f64 (n_rec, n_rec)
        "R_feat":                      rec["R_feat"],                   # f64 (n_rec, n_rec)
        "R_chroma":                    rec["R_chroma"],                 # f64 (n_rec, n_rec)
        "R_homo":                      rec["R_homo"],                   # f64 (n_rec, n_rec)
        # --- phase-3 CBM references (session 4, ADR-020) -----------------
        # Inputs: clip-local beat + downbeat times from C++ BeatDetector
        # (production path) + feature_matrix beat-synced on those beats
        # (DISTINCT from recurrence_feature_matrix which uses librosa beats).
        # Output: pythonowy cbm_analyze segments + derived boundaries.
        # cbm_segments is List[dict] not ndarray — dispatched to _save_json
        # in dump_one's save loop.
        "beats_cpp":                   beats_cpp,                       # f64 (n_beats_cpp,)
        "downbeats_cpp":               downbeats_cpp,                   # f64 (n_downbeats,)
        "feature_matrix_cbm":          feature_matrix_cbm_f64,          # f64 (n_beats_cpp, 59)
        # Session 5 DP-pipeline intermediates (CBMSegmenter parity gate on
        # cbm_bar_segments; bar_features + bar_times + autosim as bisection
        # aids). f32 dtypes mirror Python's librosa-native pipeline where
        # beat_features hit cbm_analyze as f32 (production path).
        "cbm_bar_features":            cbm_bar_features_f32,            # f32 (n_bars, 59)
        "cbm_bar_times":               cbm_bar_times,                   # f64 (n_bars+1,)
        "cbm_autosim":                 cbm_autosim_f32,                 # f32 (n_bars, n_bars)
        "cbm_bar_segments":            cbm_bar_segments_arr,            # i64 (n_segs, 2)
        "cbm_segments":                cbm_segments_list,               # List[dict] → .json
        "cbm_boundaries":              cbm_boundaries,                  # f64 (n_segments+1,)
        # Session 6 (port-order step 4 C++ continuation) — aids + structured
        # goldens for labelSegments + cbmAnalyze parity. Per session-6
        # decision 2, cbm_segments_fields + cbm_segments_labels replace a
        # JSON parser in the C++ test path. seg_sim + normed_centroids +
        # bar_energy are bisection aids if the strict gate flips.
        "cbm_seg_normed_centroids":    cbm_seg_normed_centroids,        # f32 (n_segs, n_feat)
        "cbm_seg_sim":                 cbm_seg_sim,                     # f32 (n_segs, n_segs)
        "cbm_bar_energy":              cbm_bar_energy,                  # f64 (n_bars,)
        "cbm_segments_fields":         cbm_segments_fields,             # f64 (n_segs, 4)
        "cbm_segments_labels":         cbm_segments_labels,             # List[str] → .txt
        # Session 9 (port-order step 7) — SegmentConsolidation goldens.
        # PARITY: segment_consolidation.py:83-193 _consolidate_segments
        # applied to both the CBM-path and novelty-path segment lists. The
        # dispatcher (L151, L172) runs consolidation on whichever path it
        # dispatches; we dump both to give the module-level parity test
        # coverage of both input distributions. Duration matches dispatcher
        # L119: `float(len(y)) / float(sr)` in f64. Gate: strict bitwise
        # (algorithm is deterministic, no RNG, no spectral divergence).
        "cbm_consolidated_segments_fields":     cbm_consolidated_segments_fields,     # f64 (n', 4)
        "cbm_consolidated_segments_labels":     cbm_consolidated_segments_labels,     # List[str] → .txt
        "novelty_consolidated_segments_fields": novelty_consolidated_segments_fields, # f64 (n'', 4)
        "novelty_consolidated_segments_labels": novelty_consolidated_segments_labels, # List[str] → .txt
        # Session 10 (port-order step 6) — StructureAnalyzer dispatcher goldens.
        # PARITY: structure_analyzer.py:127-182 `_analyze`. Final post-consolidation
        # `(path, segments_fields, segments_labels, boundaries)` the C++
        # StructureAnalyzer::analyze must reproduce. path.txt is single-line
        # "cbm" or "novelty"; segments_fields/labels/boundaries are the same
        # layouts as the CBM/novelty per-path goldens, selected by the
        # dispatched branch.
        "dispatched_path":             dispatched_path,              # List[str] len=1 → .txt
        "dispatched_segments_fields":  dispatched_segments_fields,   # f64 (n, 4)
        "dispatched_segments_labels":  dispatched_segments_labels,   # List[str] → .txt
        "dispatched_boundaries":       dispatched_boundaries,        # f64 (n+1,)
        # Session 11 (port-order step 8) — RepetitionMap goldens.
        # PARITY: repetition_map.py:284-532 `build_repetition_map` on
        # production beats (beats_cpp) + dispatcher-consolidated segments
        # (session-10 dispatched_*) + `_extract_waveform_snippets` on same
        # beats (pre=35 ms, post=120 ms). Each jump is serialized as nine
        # parallel arrays (primitives only — C++ reads i64/f64 .npy without
        # dtype inference) matching RepetitionJump's dataclass fields in
        # declaration order (from_beat, to_beat, waveform_similarity,
        # chroma_correlation, alignment_lag_samples, from_section_idx,
        # to_section_idx, from_bar, to_bar). Sort order is Python's
        # `sort(key=lambda j: (-waveform_similarity, -chroma_correlation))`
        # (L530-531) — C++ port must reproduce the exact ordering (Python
        # sort is stable on ties, so ties inherit emission order from the
        # nested label-group loops).
        #
        # Gate (phase-3 spec L35): ≥ 90 % agreement on pairs with
        # waveform_similarity > 0.6. Counts + section_pairs are universal
        # checks (bitwise). Per-track ADR-022 eminem cascade is expected
        # if segment cluster_ids diverge upstream; extend the exception set
        # in test_repetition_map.cpp the same way session-10 did for
        # StructureAnalyzer.
        "repetition_from_beat":             repetition_from_beat,             # i64 (n_jumps,)
        "repetition_to_beat":               repetition_to_beat,               # i64 (n_jumps,)
        "repetition_waveform_similarity":   repetition_waveform_similarity,   # f64 (n_jumps,)
        "repetition_chroma_correlation":    repetition_chroma_correlation,    # f64 (n_jumps,)
        "repetition_alignment_lag_samples": repetition_alignment_lag_samples, # i64 (n_jumps,)
        "repetition_from_section_idx":      repetition_from_section_idx,      # i64 (n_jumps,)
        "repetition_to_section_idx":        repetition_to_section_idx,        # i64 (n_jumps,)
        "repetition_from_bar":              repetition_from_bar,              # i64 (n_jumps,)
        "repetition_to_bar":                repetition_to_bar,                # i64 (n_jumps,)
        "repetition_section_pairs":         repetition_section_pairs,         # i64 (n_pairs, 2)
        "repetition_counts":                repetition_counts,                # List[str] → .txt
    }


def compute_real_beats_features(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Fixture B: orchestrator references with realistic (non-synthetic) beat_times.

    Beat source: `librosa.beat.beat_track(y)` — produces float64 beat times
    with typical-music float drift (e.g. 12.4783 s) that exercise
    `librosa.time_to_frames` round-half-to-even edge cases synthetic
    beats don't reach. The orchestrator is beat-source-agnostic, so the
    fact that these beats aren't from phase-1 BeatDetector doesn't weaken
    the test — both sides of the parity use the same beat_times.
    """
    # librosa.beat.beat_track returns (tempo, beat_frames). Convert to
    # beat_times in float64 for orchestrator input.
    _, beat_frames_real = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beat_times_real = librosa.frames_to_time(beat_frames_real, sr=sr, hop_length=HOP_LENGTH).astype(np.float64)
    n_beats_real = int(len(beat_times_real))
    if n_beats_real == 0:
        raise SystemExit("librosa.beat.beat_track produced zero beats — choose a different clip")

    # Recompute minimum-necessary intermediates (f32 throughout to mirror
    # the real pipeline's dtype).
    mfcc_40_f32 = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC_STD, hop_length=HOP_LENGTH, n_fft=N_FFT,
    ).astype(np.float32, copy=False)
    chroma_f32 = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT,
    ).astype(np.float32, copy=False)
    contrast_f32 = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT, n_bands=N_CONTRAST_BANDS,
    ).astype(np.float32, copy=False)
    rms_f32 = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0].astype(np.float32, copy=False)
    onset_f32 = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH,
    ).astype(np.float32, copy=False)
    centroid_f32 = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH,
    )[0].astype(np.float32, copy=False)

    orch = _run_orchestrator(
        y, sr, mfcc_40_f32, chroma_f32, contrast_f32,
        rms_f32, onset_f32, centroid_f32,
        beat_times_real, skip_transition_waveforms=False,
    )

    # y+beat_times step-9 outputs need their own real-beats version
    # (boundary/transition/edge_rms depend on beat_times, not on the
    # stacked matrix).
    edge_rms_start, edge_rms_end = _extract_edge_rms(y, sr, beat_times_real, n_beats_real)
    boundary_waveforms = _extract_waveform_snippets(
        y, sr, beat_times_real, n_beats_real,
        pre_ms=BOUNDARY_PRE_MS, post_ms=BOUNDARY_POST_MS,
    )
    transition_waveforms = _extract_waveform_snippets(
        y, sr, beat_times_real, n_beats_real,
        pre_ms=TRANSITION_PRE_MS, post_ms=TRANSITION_POST_MS,
    )

    return {
        "beat_times_real":    beat_times_real,                               # f64 (n_beats,)
        "feature_matrix":     orch["feature_matrix"],                        # f64 (n_beats, 59)
        "edge_features_start": orch["edge_features_start"],
        "edge_features_end":   orch["edge_features_end"],
        "edge_rms_start":     edge_rms_start,
        "edge_rms_end":       edge_rms_end,
        "boundary_waveforms": boundary_waveforms,
        "transition_waveforms": transition_waveforms,
        "rms_energy":         orch["rms_energy"],
        "onset_strength":     orch["onset_strength"],
        "spectral_centroid":  orch["spectral_centroid"],
    }


def dump_one(
    audio_path: Path,
    out_dir: Path,
    name: str,
    clip_range_sec: tuple[float, float] | None,
    expected_audio_sha: str | None,
    *,
    real_beats: bool = False,
) -> dict:
    """Dump one track. Returns the per-track manifest dict."""
    audio_sha = _sha256_resolved(audio_path)
    if expected_audio_sha and audio_sha != expected_audio_sha:
        raise SystemExit(
            f"Audio SHA mismatch for {name}:\n"
            f"  expected {expected_audio_sha}\n"
            f"  got      {audio_sha}\n"
            f"  path     {audio_path} -> {audio_path.resolve()}"
        )

    # Honor clip_range_sec via librosa.load offset+duration (saves decode time on long tracks)
    offset = 0.0
    duration = None
    if clip_range_sec:
        start, end = clip_range_sec
        offset = float(start)
        duration = float(end - start)

    y, loaded_sr = librosa.load(str(audio_path), sr=SR, mono=True, offset=offset, duration=duration)
    assert loaded_sr == SR, f"librosa.load returned sr={loaded_sr}, expected {SR}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Also dump the loaded+resampled audio itself so C++ parity tests use byte-identical
    # input and are not exposed to mp3 decoder / resampler variance. `y` is float32
    # (librosa.load default), matching the precision used internally by librosa.stft.
    y_audio = y.astype(np.float32, copy=False)

    if real_beats:
        features = compute_real_beats_features(y, SR)
    else:
        # ADR-020: pass audio_path + clip_range_sec so compute_features can
        # call beat_detector_cli on the full file for CBM downbeats.
        features = compute_features(
            y, SR,
            audio_path=audio_path,
            clip_range_sec=clip_range_sec,
        )
    features_with_audio: dict[str, object] = {"y_audio": y_audio, **features}

    dumps: dict[str, dict] = {}
    for feat_name, value in features_with_audio.items():
        if isinstance(value, np.ndarray):
            sha, size = _save_npy(value, out_dir, feat_name)
            dumps[feat_name] = {
                "file":   f"{feat_name}.npy",
                "shape":  list(value.shape),
                "dtype":  str(value.dtype),
                "sha256": sha,
                "bytes":  size,
            }
        elif isinstance(value, list):
            # Session-6 decision-2: dispatch list-of-str to .txt (used by
            # cbm_segments_labels), list-of-dict / other to .json (ADR-020
            # cbm_segments). Non-empty lists use runtime-type check. Empty
            # lists would be ambiguous (`all(...)` on [] is vacuously True),
            # so key-name convention picks: keys ending in `_labels` save
            # as `.txt` (session-6 decision-2 intent), everything else as
            # `.json`. Session-15 (ADR-024 sub-fix A) fixed a prior
            # short-circuit that pushed ALL empty lists to `.json`, crashing
            # segment_consolidation_parity for alice_in_chains_nutshell
            # (cbm_analyze→None when bar_segments<3 per L2229-2232 → empty
            # `cbm_segments_labels` expected as `.txt` by the C++ test).
            is_str_list = (
                all(isinstance(x, str) for x in value)
                if value
                else feat_name.endswith("_labels")
            )
            if is_str_list:
                sha, size = _save_text(value, out_dir, feat_name)
                dumps[feat_name] = {
                    "file":   f"{feat_name}.txt",
                    "shape":  [len(value)],
                    "dtype":  "text",
                    "sha256": sha,
                    "bytes":  size,
                }
            else:
                sha, size = _save_json(value, out_dir, feat_name)
                dumps[feat_name] = {
                    "file":   f"{feat_name}.json",
                    "shape":  [len(value)],
                    "dtype":  "json",
                    "sha256": sha,
                    "bytes":  size,
                }
        else:
            raise TypeError(
                f"unsupported dump value type for {feat_name}: {type(value)}"
            )

    manifest = {
        "track":            name,
        "audio_source":     str(audio_path),
        "audio_sha256":     audio_sha,
        "clip_range_sec":   list(clip_range_sec) if clip_range_sec else None,
        "y_samples":        int(y.shape[0]),
        "sr":               SR,
        "real_beats":       bool(real_beats),
        "pins": {
            "librosa": librosa.__version__,
            "scipy":   scipy.__version__,
            "numpy":   np.__version__,
        },
        "config": {
            "n_fft":      N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels":     N_MELS,
            "n_mfcc_std": N_MFCC_STD,
            "n_mfcc_fast": N_MFCC_FAST,
            "n_contrast_bands": N_CONTRAST_BANDS,
        },
        "dumps":            dumps,
        "generated_utc":    datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --- CLI ---------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=Path, help="Single audio file to dump")
    group.add_argument("--from-manifest", type=Path, help="Iterate all resolvable tracks in manifest.json")
    parser.add_argument("--name", type=str, help="Track name (required with --audio)")
    parser.add_argument("--out", type=Path, help="Output directory (required with --audio)")
    parser.add_argument("--clip-start", type=float, default=None)
    parser.add_argument("--clip-end",   type=float, default=None)
    parser.add_argument("--expected-sha", type=str, default=None)
    parser.add_argument(
        "--real-beats", action="store_true",
        help="Use librosa.beat.beat_track(y) for orchestrator fixture B — exercises "
             "time_to_frames round-half-to-even on realistic float-drift beat_times. "
             "Writes a minimal orchestrator reference set (no synthetic-beats dumps).",
    )
    args = parser.parse_args()

    _assert_pins()

    if args.audio:
        if not args.name or not args.out:
            parser.error("--audio requires --name and --out")
        clip = None
        if args.clip_start is not None and args.clip_end is not None:
            clip = (args.clip_start, args.clip_end)
        mode_tag = "real-beats" if args.real_beats else "synthetic"
        print(f"[dump] {args.name}  ({args.audio})  [{mode_tag}]")
        dump_one(
            args.audio, args.out, args.name, clip, args.expected_sha,
            real_beats=args.real_beats,
        )
        print(f"[dump] wrote {args.out}/manifest.json")
        return 0

    if args.real_beats:
        parser.error("--real-beats requires --audio (not --from-manifest)")

    # --from-manifest mode
    manifest_path = args.from_manifest
    manifest = json.loads(manifest_path.read_text())
    dumps_root = manifest_path.parent / "dumps"
    skipped = []
    for slot in manifest["tracks"]:
        if slot.get("filename") is None:
            skipped.append((slot["slot"], slot["name"], "no filename (TBD slot)"))
            continue
        audio_path = (manifest_path.parent.parent / "phase-1" / slot["filename"])
        # Slots 5-10 may be placed under phase-2/ instead of phase-1/ — try both
        if not audio_path.exists():
            audio_path = manifest_path.parent / slot["filename"]
        if not audio_path.exists():
            skipped.append((slot["slot"], slot["name"], f"audio not found: {slot['filename']}"))
            continue

        clip_range = slot.get("clip_range_sec")
        clip = tuple(clip_range) if clip_range else None
        out_dir = dumps_root / slot["name"]
        print(f"[dump] slot {slot['slot']} {slot['name']}  ({audio_path})")
        try:
            dump_one(audio_path, out_dir, slot["name"], clip, slot.get("sha256"))
        except SystemExit as e:
            skipped.append((slot["slot"], slot["name"], str(e)))
            continue

    if skipped:
        print("\n[dump] skipped:")
        for slot_num, name, reason in skipped:
            print(f"  slot {slot_num} {name}: {reason}")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(_cli())
