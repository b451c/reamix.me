"""ADR-078 sesja 88 CI guardrail — listener_ui slot randomization.

Verifies that ListenerState.next_pair() randomizes the (anchor, candidate)
→ (slot A, slot B) display mapping. Without this, BoTorch GP/EUBO returns
deterministic (highest_mean, second_highest_mean) → anchor stuck in slot A
for entire session → side bias indistinguishable from quality signal.

Sesja 88 audit precedent: Drake 14-D N=77 had r01024 in slot A 41/41 times,
r00294 37/38 times → 79% choice 'a' could not be disambiguated from true
quality (anchor recognized after a few pairs). Sesja 85 cross-track Δ matrix
(Drake +31.6 pp / Avicii 0 pp / JJ +4 pp) had same structural bug.

Test: force ListenerState.next_pair() to receive same (anchor, candidate)
1000× via patched session.next_pair_from_pool(); count occurrences of each
ID in slot A. Pass: each ID appears 400-600× (50% ± noise).
Fail: deviation > ±10% → slot bias still present.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Inject tools/dev/calibration so listener_ui imports preference_gp resolve.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "dev" / "calibration"))


def main() -> int:
    from listener_ui import ListenerState  # noqa: E402

    # Realistic pool: load Drake 14-D stage1.jsonl from sesja 88. If absent
    # (clean checkout), build 2-record synthetic pool with same field shape.
    stage1_path = (REPO_ROOT / "references" / "listening" / "dev-028"
                   / "sesja-88" / "drake_hotline_bling_14d" / "stage1.jsonl")
    if stage1_path.exists():
        records = [json.loads(L) for L in stage1_path.read_text().splitlines()
                   if L.strip()]
    else:
        records = [
            {"id": "synthA", "wav_path": "/tmp/a.wav", "csv_path": "/tmp/a.csv",
             "weights": [1.0] + [0.0]*13, "n_splices": 1, "mean_proxy_loss": 0.1,
             "per_splice": [], "wav_sha256": "0"*64,
             "axis_bucket": "waveform", "track_id": "synth", "mode": "duration"},
            {"id": "synthB", "wav_path": "/tmp/b.wav", "csv_path": "/tmp/b.csv",
             "weights": [0.5, 0.5] + [0.0]*12, "n_splices": 1,
             "mean_proxy_loss": 0.2, "per_splice": [], "wav_sha256": "1"*64,
             "axis_bucket": "successor", "track_id": "synth", "mode": "duration"},
        ]
    assert len(records) >= 2, "test pool must have ≥2 records"

    tmp_dir = Path(tempfile.mkdtemp(prefix="listener_slot_test_"))
    state = ListenerState(records=records,
                          cell_id="slot_randomization_smoke",
                          gp_state_path=tmp_dir / "gp.json",
                          prefs_path=tmp_dir / "prefs.jsonl",
                          slot_seed=42)

    anchor_id = records[0]["id"]
    candidate_id = records[1]["id"]
    anchor_w = np.asarray(records[0]["weights"])
    candidate_w = np.asarray(records[1]["weights"])

    # Force GP to always return same (anchor, candidate) pair so we test
    # ONLY the display-slot mapping logic.
    state.session.next_pair_from_pool = lambda: (anchor_w, candidate_w)

    N = 1000
    slot_a_count = {anchor_id: 0, candidate_id: 0}
    for _ in range(N):
        state.seen_pair_keys = set()  # reset so same anchor pair re-found
        ok = state.next_pair()
        assert ok, "next_pair returned False on non-empty pool"
        slot_a_count[state.pair_a["id"]] += 1

    print(f"slot A distribution over N={N} pairs (anchor + candidate forced):")
    for rid, c in slot_a_count.items():
        print(f"  {rid}: {c} ({100*c/N:.1f}%)")

    anchor_in_a = slot_a_count[anchor_id]
    deviation = abs(anchor_in_a - N // 2)
    tolerance = N // 10  # ±10% = ±100 of 1000

    if not (N // 2 - tolerance) <= anchor_in_a <= (N // 2 + tolerance):
        print(f"FAIL: anchor in slot A {anchor_in_a}/{N} "
              f"(expected {N//2}±{tolerance}). "
              f"Slot bias still present — ADR-078 fix regressed.")
        return 1

    print(f"PASS: deviation={deviation} (tolerance ±{tolerance}). "
          f"Slot randomization working per ADR-078.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
