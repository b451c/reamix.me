#!/usr/bin/env python3
"""ADR-115 v2-default smoke, macOS leg (ReaProof isolated REAPER).

Sesja 115: v2 scoring (exp sequential-baseline mapping, repetition prior,
cleaned beat grid) became the production default in AnalyzePipeline /
RemixPipeline. This leg proves the shipped dylib still analyses a real item
inside a real REAPER: insert media, open reamix, click Analyze when the
waveform is absent, wait for the waveform to appear and settle, and confirm
the REAPER process + reamix window are alive afterwards. Evidence PNG +
JSON under tests/reaproof/evidence/v2-analyze-mac/.

Run from the repo root (ReaProof copy at ./reaproof):
  PYTHONPATH=reaproof/src LC_ALL=en_US.UTF-8 LC_NUMERIC=C TZ=UTC \
    python3 tests/reaproof/v2_analyze_mac.py --dylib build/reaper_reamix.dylib --tag v2 [--runs 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reaproof" / "src"))
sys.path.insert(0, str(REPO / "tests" / "reaproof"))

from reaproof.runner.session import ReaperSession  # noqa: E402
from reaproof.observe.input import _MacMouse  # noqa: E402
import dev086_mac as base  # noqa: E402  (window / capture / click helpers)

OUT = REPO / "tests" / "reaproof" / "evidence" / "v2-analyze-mac"


def run(dylib: Path, tag: str, runs: int) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(runs):
        rtag = f"{tag}_run{i + 1}"
        r: dict = {"tag": rtag, "ok": False}
        with ReaperSession("reamix-v2", extensions=[Path(dylib), base.ORT]) as s:
            pid = s.handle.pid
            r["item_len"] = s.eval(f'''
reaper.Main_OnCommand(40023, 0)
reaper.InsertTrackAtIndex(0, false)
local tr = reaper.GetTrack(0, 0)
reaper.SetOnlyTrackSelected(tr)
reaper.SetEditCurPos(0, false, false)
reaper.InsertMedia("{base.MEDIA}", 0)
local it = reaper.GetTrackMediaItem(tr, 0)
reaper.SetMediaItemSelected(it, true)
reaper.UpdateArrange()
return reaper.GetMediaItemInfo_Value(it, "D_LENGTH")''', timeout=120, hang_timeout=60)
            s.eval('reaper.SetExtState("reamix.me", "welcome_shown", "1", true) return true')
            cmd = s.eval('return reaper.NamedCommandLookup("_reamix_ShowWindow")')
            for _ in range(4):
                if s.eval(f'return reaper.GetToggleCommandState({cmd})') == 1:
                    break
                s.eval(f'reaper.Main_OnCommand({cmd}, 0) return true', hang_timeout=30)
                time.sleep(2.0)
            win = None
            for _ in range(30):
                base.front(pid)
                win = next((w for w in base.windows_of(pid) if w["name"] == "reamix.me"), None)
                if win:
                    break
                time.sleep(0.5)
            if not win:
                r["error"] = "reamix window not found"; results.append(r); continue
            time.sleep(3.0)
            wpng = OUT / f"{rtag}_window.png"
            base.shot(win["id"], wpng)
            ox, oy, scale = base.client_origin(win, wpng)
            wave_band = (oy - win["y"] + 300, oy - win["y"] + 420)
            already = base.waveform_present(wpng, scale, *wave_band) > 400
            r["cache_hit"] = already
            mouse = _MacMouse()
            base.front(pid)
            if not already:
                r["analyze_click"] = base.click_until_changed(
                    mouse, win["id"], ox + 672, oy + 74, wpng, OUT / f"{rtag}_after_analyze_click.png")
            # wait for the waveform + a stable picture (progress bar gone)
            t0 = time.time()
            stable = 0
            prev = None
            while time.time() - t0 < 240:
                time.sleep(2.0)
                cur = OUT / f"{rtag}_poll.png"
                base.shot(win["id"], cur)
                present = base.waveform_present(cur, scale, *wave_band) > 400
                same = prev is not None and base.compare(prev, cur) == 0
                if present and same:
                    stable += 1
                    if stable >= 2:
                        break
                else:
                    stable = 0
                prev_path = OUT / f"{rtag}_prev.png"
                cur.replace(prev_path)
                prev = prev_path
            r["analyze_seconds"] = round(time.time() - t0, 1)
            final = OUT / f"{rtag}_final.png"
            base.shot(win["id"], final)
            r["waveform_present"] = base.waveform_present(final, scale, *wave_band) > 400
            r["window_alive"] = any(w["name"] == "reamix.me" for w in base.windows_of(pid))
            r["reaper_alive"] = s.eval('return 1') == 1
            r["ok"] = bool(r["waveform_present"] and r["window_alive"] and r["reaper_alive"])
            results.append(r)
    (OUT / f"{tag}_result.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dylib", required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    res = run(Path(a.dylib), a.tag, a.runs)
    for r in res:
        print(json.dumps(r))
    ok = all(r.get("ok") for r in res) and len(res) == a.runs
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
