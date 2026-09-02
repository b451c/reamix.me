#!/usr/bin/env python3
"""Sesja 121 visual check: open reamix on a media item, click Analyze, wait for the
analysis, capture the window on the Duration / Region / Blocks tabs.

Usage: rm_tabs.py --so <reaper_reamix .so> --media <audio> [--wait 60]
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rm_open  # noqa: E402
import xtools  # noqa: E402
import rm_blocks  # noqa: E402

TABS = {"duration": rm_blocks.DURATION_TAB, "region": (112, 125), "blocks": rm_blocks.BLOCKS_TAB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--media", required=True)
    ap.add_argument("--wait", type=float, default=60.0)
    a = ap.parse_args()
    rm_blocks.install(a.so)
    tag = Path(a.so).stem + "_tabs"
    if rm_open.alive():
        raise RuntimeError("REAPER already running")
    rm_open.remove_bridge(); rm_open.deploy_bridge()
    ok, out = rm_open.launch(["-new", "-nosplash"])
    if not ok:
        raise RuntimeError("launch failed: " + out)
    b = rm_open.client(); b.wait_ready(120)
    ids, _ = rm_open.open_reamix(b, a.media)
    rm = None
    for _ in range(20):
        rm = xtools.find("reamix.me")
        if rm:
            break
        time.sleep(0.5)
    if not rm:
        raise RuntimeError("no reamix window")
    x0, y0 = rm["x"], rm["y"]
    xtools.focus(rm["win"])
    xtools.click(x0 + rm_blocks.ANALYZE_BTN[0], y0 + rm_blocks.ANALYZE_BTN[1])
    t0 = time.time()
    while time.time() - t0 < a.wait:      # wait for chips on the Blocks tab = analysis done
        xtools.click(x0 + TABS["blocks"][0], y0 + TABS["blocks"][1]); time.sleep(3.0)
        png = rm_open.OUT / f"{tag}_poll.png"
        xtools.capture(rm["win"], str(png))
        _, rs = rm_blocks.chips(png)
        if rs:
            break
    for name, (tx, ty) in TABS.items():
        xtools.click(x0 + tx, y0 + ty); time.sleep(1.5)
        xtools.capture(rm["win"], str(rm_open.OUT / f"{tag}_{name}.png"))
        print("captured", name)
    rm_open.quit_reaper(b); rm_open.remove_bridge()


if __name__ == "__main__":
    main()
