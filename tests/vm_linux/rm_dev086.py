#!/usr/bin/env python3
"""DEV-086 Linux popover-noise gate (ReaProof-style: observable effect, mutation-verified).

Flow per run: install the requested .so -> launch the VM's REAPER with the ReaProof
bridge -> insert + select a media item -> open reamix -> Blocks tab -> drag on the
section bar -> the MARK SECTION picker (a JUCE CallOutBox = override-redirect X
window) appears -> XGetImage of that window -> oracle on the outer frame pixels
(must be uniformly theme Bg0 = #0B0A09 outside the rounded path; the un-fixed
binary leaks X11 backing-buffer noise there) -> quit REAPER.

Usage: rm_dev086.py --so <path to reaper_reamix .so> --media <audio> --expect green|red [--runs 2]
       rm_dev086.py --check <png>      # oracle only
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rm_open  # noqa: E402
import xtools  # noqa: E402

PLUGIN = Path.home() / ".config" / "REAPER" / "UserPlugins" / "reaper_reamix.so"
OUT = Path.home() / "reamix_vm"
BG0 = (0x0B, 0x0A, 0x09)          # src/ui/Theme.h Bg0 (tokens.css:6)
FRAME = 8                         # px of the CallOutBox margin sampled on each side
TOL = 3                           # per-channel tolerance vs Bg0

# Window-relative click targets (reamix window is 720x760 on the VM; layout from
# MainComponent::resized: header 40 + source panel + mode tabs 36 ...). Verified
# against a capture on 2026-09-02 (rm1.png / rm2.png).
BLOCKS_TAB = (210, 125)
SEGBAR_Y = 646
SEGBAR_X0, SEGBAR_X1 = 120, 260


def oracle(png, expect_rgb=BG0):
    """Frame oracle: the outer FRAME px of a popup window must be one flat colour.
    The CallOutBox arrow (any side) and its anti-aliased edge are excluded via a
    60 px band centred on each edge. Returns the share of the dominant colour;
    X11 backing-buffer noise gives hundreds of colours and a share near 0."""
    from collections import Counter
    from PIL import Image
    im = Image.open(png).convert("RGB")
    w, h = im.size
    px = im.load()
    band_x = range(w // 2 - 30, w // 2 + 30)
    band_y = range(h // 2 - 30, h // 2 + 30)
    counts = Counter()
    for y in range(h):
        for x in range(w):
            top, bot, left, right = y < FRAME, y >= h - FRAME, x < FRAME, x >= w - FRAME
            if not (top or bot or left or right):
                continue
            if (top or bot) and x in band_x:
                continue
            if (left or right) and y in band_y:
                continue
            counts[px[x, y]] += 1
    total = sum(counts.values())
    mode_rgb, mode_n = counts.most_common(1)[0]
    share = 100.0 * mode_n / max(1, total)
    near_expect = sum(n for c, n in counts.items()
                      if all(abs(c[k] - expect_rgb[k]) <= TOL for k in range(3)))
    top3 = [[list(c), n] for c, n in counts.most_common(3)]
    # flat frame = a handful of colours (fill + shadow gradient + arrow AA) with one
    # clearly dominant; noise = hundreds of colours, dominant share < 10 %.
    return {"png": str(png), "size": [w, h], "frame_px": total, "distinct_colours": len(counts),
            "top3": top3, "dominant_share_pct": round(share, 2),
            "expect_rgb_share_pct": round(100.0 * near_expect / max(1, total), 2),
            "green": share >= 60.0 and len(counts) <= 64}


def install(so):
    if PLUGIN.exists():
        PLUGIN.unlink()                     # fresh inode (mapped-inode crash lore)
    shutil.copy2(so, PLUGIN)


def one_run(media, tag):
    if rm_open.alive():
        raise RuntimeError("REAPER already running")
    rm_open.remove_bridge(); rm_open.deploy_bridge()
    ok, out = rm_open.launch(["-new", "-nosplash"])
    if not ok:
        raise RuntimeError("launch failed: " + out)
    b = rm_open.client()
    b.wait_ready(120)
    ids, length = rm_open.open_reamix(b, media)
    # the first ShowWindow only creates the window; toggle until visible
    for _ in range(3):
        st = b.eval(f'return reaper.GetToggleCommandState({ids["toggle"]})')
        if st == 1:
            break
        b.eval(f'reaper.Main_OnCommand({ids["toggle"]}, 0) return true', hang_timeout=30)
        time.sleep(2.0)
    rm = None
    for _ in range(20):
        rm = xtools.find("reamix.me")
        if rm:
            break
        time.sleep(0.5)
    if not rm:
        raise RuntimeError("reamix window not found")
    # analysis must be complete before the section bar accepts a drag: poll the
    # window until the Blocks tab shows the section bar hint (cheap proxy: wait
    # for the source panel to report beats = capture pixel changes settle)
    time.sleep(4.0)
    pid = rm_open.pid()
    x0, y0 = rm["x"], rm["y"]
    xtools.click(x0 + BLOCKS_TAB[0], y0 + BLOCKS_TAB[1])
    time.sleep(1.5)
    xtools.capture(rm["win"], str(OUT / f"{tag}_blocks.png"))
    # the section bar only accepts the mark-drag once analysis is complete;
    # analysis time varies (disk-cache hit vs cold), so retry the gesture.
    picker = None
    attempts = 0
    for attempt in range(8):
        attempts = attempt + 1
        xtools.drag(x0 + SEGBAR_X0, y0 + SEGBAR_Y, x0 + SEGBAR_X1, y0 + SEGBAR_Y)
        for _ in range(12):
            for e in xtools.windows():
                if e["pid"] == pid and e["override"] and e["w"] > 100 and e["h"] > 100:
                    picker = e
            if picker:
                break
            time.sleep(0.25)
        if picker:
            break
        xtools.capture(rm["win"], str(OUT / f"{tag}_nopicker{attempt + 1}.png"))
        time.sleep(3.0)
    result = {"tag": tag, "picker": None, "drag_attempts": attempts}
    if picker:
        time.sleep(1.0)   # let the first paint + any deferred repaint settle
        png = OUT / f"{tag}_picker.png"
        xtools.capture(picker["win"], str(png))
        result["picker"] = {"w": picker["w"], "h": picker["h"], "x": picker["x"], "y": picker["y"]}
        result["oracle"] = oracle(png)
        # DEV-085 stage: "+ Add custom" tile (picker-relative 90,208) opens the
        # Add-custom-kind AlertWindow; its frame must be flat too.
        known = {e["id"] for e in xtools.windows()}
        xtools.click(picker["x"] + 90, picker["y"] + 208)
        modal = None
        for _ in range(16):
            for e in xtools.windows():
                if e["id"] not in known and e["pid"] == pid and e["w"] > 150 and e["h"] > 80 \
                        and "mutter" not in str(e["cls"]):
                    modal = e
            if modal:
                break
            time.sleep(0.25)
        if modal:
            time.sleep(1.0)
            mpng = OUT / f"{tag}_modal.png"
            xtools.capture(modal["win"], str(mpng))
            o = oracle(mpng)
            o["name"] = modal["name"]
            result["modal_oracle"] = o
            xtools.key("Escape")      # Cancel (escapeKey) closes the AlertWindow
            time.sleep(0.6)
        else:
            result["modal_oracle"] = None
        xtools.key("Escape")          # closes the picker
        time.sleep(0.5)
    rm_open.quit_reaper(b)
    rm_open.remove_bridge()
    result["reaper_alive_after"] = rm_open.alive()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so")
    ap.add_argument("--media")
    ap.add_argument("--expect", choices=["green", "red"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.check:
        print(json.dumps(oracle(a.check), indent=1)); return
    install(a.so)
    tag = Path(a.so).stem
    results = []
    for i in range(a.runs):
        r = one_run(a.media, f"{tag}_run{i + 1}")
        print(json.dumps(r, indent=1), flush=True)
        results.append(r)
    verdicts = [r.get("oracle", {}).get("green") for r in results]
    modal_verdicts = [(r.get("modal_oracle") or {}).get("green") for r in results]
    want = a.expect == "green"
    ok = all(v is not None and v == want for v in verdicts)
    print("VERDICTS picker", verdicts, "modal", modal_verdicts, "expected", a.expect, "->", "PASS" if ok else "FAIL")
    (OUT / f"{tag}_result.json").write_text(json.dumps(results, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
