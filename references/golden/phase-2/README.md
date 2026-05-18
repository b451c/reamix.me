# Phase 2 golden corpus

Parity test corpus for phase-2 feature extraction (MFCC + chroma + spectral contrast).

## What is tracked here

- `manifest.json` — the **single source of truth** for the 10-track corpus: file, genre, source, SHA-256 of audio, clip range. Committed to git.
- `dumps/<track>/*.npy` — Python librosa reference outputs (per intermediate feature). **Not** committed (gitignored) — regenerate via `tools/dump_python_features.py`.
- `dumps/<track>/manifest.json` — per-track dump manifest: per-`.npy` SHA-256 + librosa/scipy/numpy versions + generation date. **Committed** (acts as a tamper-evident seal: if someone regenerates dumps on a newer librosa, the SHA hashes won't match and the mismatch surfaces at parity-test time).

## What is NOT tracked

- The audio files themselves (`*.wav`, `*.flac`, `*.mp3`). Copyright-sensitive. Either symlinked into the local user's music library (phase-1 pattern, slots 1–4 do this) or obtained from CC / public-domain sources and placed locally.
- The `.npy` dumps themselves — they're 100s of MB and regenerable from audio + pinned librosa version.

## Version pins

**Mandatory** for any `.npy` regeneration:

- `librosa==0.11.0`
- `scipy==1.15.3`
- `numpy>=1.24,<2.0`

Pinned in `tools/requirements-phase2.txt`. The Python reference project (`references/python-source/`) resolves to these versions via its `uv.lock`; we track that resolution exactly.

## Track list and status (2026-04-20)

| # | Genre | Status | Source |
|---|-------|--------|--------|
| 1 | pop | ✅ symlink | local: Billie Jean |
| 2 | rock | ✅ symlink | local: Smells Like Teen Spirit |
| 3 | classical 3/4 | ✅ symlink | local: Shostakovich Jazz Waltz |
| 4 | classical sparse 3/4 | ✅ symlink | local: Pan Tadeusz Polonez |
| 5 | acoustic | ⏳ TBD | FMA or DAW bounce |
| 6 | electronic | ⏳ TBD | FMA CC-BY / CC-BY-SA |
| 7 | hip-hop | ⏳ TBD | FMA or DAW bounce |
| 8 | jazz | ⏳ TBD | FMA or pre-1929 PD |
| 9 | speech-like | ⏳ TBD | LibriVox PD or CommonVoice CC0 |
| 10 | dense mix | ⏳ TBD | FMA metal/noise |

Slots 5–10 are acquisition tasks for the next session (or user). See `manifest.json` for per-slot suggestions.

## Regeneration workflow

```bash
# 1. Ensure correct Python environment (pinned librosa/scipy/numpy)
pip install -r tools/requirements-phase2.txt

# 2. Generate dumps for one track
python tools/dump_python_features.py \
  --audio references/golden/phase-1/billie_jean.mp3 \
  --out references/golden/phase-2/dumps/billie_jean/

# 3. Or loop over the whole manifest
python tools/dump_python_features.py --from-manifest references/golden/phase-2/manifest.json
```

Each run of `dump_python_features.py`:

1. Reads the manifest (or one `--audio` path).
2. Honors `clip_range_sec` to cut the 30–60 s excerpt.
3. Emits `.npy` per intermediate feature (STFT magnitude, mel power, log-mel, MFCC, chroma_stft, spectral_contrast, onset_strength, rms, spectral_centroid).
4. Writes `dumps/<track>/manifest.json` with per-`.npy` SHA-256 + version pins + date.
5. Verifies audio SHA-256 matches `manifest.json` — aborts loudly on mismatch.

## Why hash the dumps

librosa and scipy update frequently. A silent regenerate on a newer version would shift all parity baselines without any diff visible in git (we don't track `.npy`). By hashing the dumps and committing the hash manifest, a regeneration produces a git diff in `dumps/<track>/manifest.json` that surfaces the drift immediately.

## See also

- `phases/phase-2-features/spec.md`
- `meta/RESEARCH.md` § phase-2 feature-extraction kickoff (2026-04-20)
- `meta/DECISIONS.md` ADR-009 (split)
