#!/usr/bin/env python3
"""Sesja 122 DEV-093 gate: a REAPER time-selection on the selected item (auto-Region) must
show the selection scrim on the plugin waveform, and a drag on that scrim must take the
region over and write it back into REAPER's time-selection.

Usage: rm_region_scrim.py --so <reaper_reamix .so> --media <audio> [--expect green|red]

Oracle 1 (scrim): window capture before / after the time-selection; in the waveform canvas
rows the changed columns must cover the selection span, and the span must carry the Info
tint (mean blue - red inside the span >= 8 above the columns outside it; Region mode dims
the whole canvas, so "nothing changed outside" is NOT the oracle - sesja 122 run 1).
Oracle 2 (takeover): drag from the selection's right edge 60 px to the right; REAPER's
time-selection end must follow (item at position 0, so item time == project time).
Mutation: a pre-sesja-122 .so paints no scrim and ignores the drag in auto mode -> red.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rm_open  # noqa: E402
import xtools  # noqa: E402
import rm_blocks  # noqa: E402

SEL = (60.0, 90.0)          # seconds, item-relative == project time
ROWS = (300, 600)           # pure waveform rows in the 720 px window (ruler ~250, segBar ~640)
DRAG_PX = 60


def changed_columns(a, b, rows):
    """Columns whose pixels differ (any channel > 8) in >= 30 % of the given rows."""
    w = min(a.width, b.width)
    pa, pb = a.convert("RGB").load(), b.convert("RGB").load()
    cols = []
    n = rows[1] - rows[0]
    for x in range(w):
        c = 0
        for y in range(rows[0], rows[1]):
            ra, ga, ba = pa[x, y]
            rb, gb, bb = pb[x, y]
            if abs(ra - rb) > 8 or abs(ga - gb) > 8 or abs(ba - bb) > 8:
                c += 1
        if c >= 0.3 * n:
            cols.append(x)
    return cols, w


def tint(img, xs, rows):
    """Mean (blue - red) over the given columns / rows: the Info scrim raises it."""
    p = img.convert("RGB").load()
    s = n = 0
    for x in xs:
        for y in range(rows[0], rows[1], 3):
            s += p[x, y][2] - p[x, y][0]; n += 1
    return s / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--media", required=True)
    ap.add_argument("--expect", default="green")
    a = ap.parse_args()
    rm_blocks.install(a.so)
    tag = Path(a.so).stem + "_scrim"
    if rm_open.alive():
        raise RuntimeError("REAPER already running")
    rm_open.remove_bridge(); rm_open.deploy_bridge()
    ok, out = rm_open.launch(["-new", "-nosplash"])
    if not ok:
        raise RuntimeError("launch failed: " + out)
    res = {"tag": tag, "green": False}
    try:
        b = rm_open.client(); b.wait_ready(120)
        ids, length = rm_open.open_reamix(b, a.media)
        res["media_len"] = length
        rm = None
        for _ in range(20):
            rm = xtools.find("reamix.me")
            if rm:
                break
            time.sleep(0.5)
        if not rm:
            raise RuntimeError("no reamix window")
        xtools.focus(rm["win"]); time.sleep(2.0)
        before = rm_open.OUT / f"{tag}_before.png"
        xtools.capture(rm["win"], str(before))

        b.eval(f"reaper.GetSet_LoopTimeRange(true, false, {SEL[0]}, {SEL[1]}, false) "
               "reaper.UpdateTimeline() return true", hang_timeout=15)
        time.sleep(3.0)                      # 100 ms poll -> auto Region -> repaint
        after = rm_open.OUT / f"{tag}_after.png"
        xtools.capture(rm["win"], str(after))

        ia, ib = Image.open(before), Image.open(after)
        cols, w = changed_columns(ia, ib, ROWS)
        pps = w / length
        x0, x1 = int(SEL[0] * pps), int(SEL[1] * pps)
        inside = [x for x in cols if x0 <= x <= x1]
        outside = [x for x in cols if x < x0 - 4 or x > x1 + 4]
        cover = len(inside) / max(1, x1 - x0 + 1)
        tint_in = tint(ib, range(x0 + 4, x1 - 4), ROWS)
        tint_out = tint(ib, list(range(20, x0 - 16)) + list(range(x1 + 16, w - 20)), ROWS)
        res.update({"px_per_sec": pps, "span_px": [x0, x1], "changed_cols": len(cols),
                    "inside_cover": cover, "outside_cols": len(outside),
                    "tint_inside": tint_in, "tint_outside": tint_out})
        scrim_ok = cover >= 0.8 and (tint_in - tint_out) >= 8.0
        res["scrim_ok"] = scrim_ok

        # Oracle 2: drag from the right edge of the scrim 60 px to the right.
        y = rm["y"] + (ROWS[0] + ROWS[1]) // 2
        xtools.drag(rm["x"] + x1, y, rm["x"] + x1 + DRAG_PX, y)
        time.sleep(2.0)
        ts = b.eval("local s, e = reaper.GetSet_LoopTimeRange(false, false, 0, 0, false) "
                    "return {s = s, e = e}", hang_timeout=15)
        res["reaper_ts_after_drag"] = ts
        exp_end = SEL[1] + DRAG_PX / pps
        end_ok = abs(ts["e"] - exp_end) <= 4.0
        start_ok = abs(ts["s"] - SEL[0]) <= 4.0 or abs(ts["s"] - SEL[1]) <= 4.0
        res["expected_end"] = exp_end
        res["drag_ok"] = bool(end_ok and start_ok)
        xtools.capture(rm["win"], str(rm_open.OUT / f"{tag}_drag.png"))
        res["green"] = bool(scrim_ok and res["drag_ok"])
    finally:
        try:
            rm_open.quit_reaper(rm_open.client())
        except Exception as e:  # noqa: BLE001
            print("quit:", repr(e))
        rm_open.remove_bridge()
    res["reaper_alive_after"] = rm_open.alive()
    (rm_open.OUT / f"{tag}_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    want = a.expect == "green"
    print("VERDICT", "GREEN" if res["green"] else "RED", "(as expected)" if res["green"] == want else "(UNEXPECTED)")
    return 0 if res["green"] == want else 1


if __name__ == "__main__":
    sys.exit(main())
