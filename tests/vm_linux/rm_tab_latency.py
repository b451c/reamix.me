#!/usr/bin/env python3
"""Sesja 122 (DEV-102): after the chips appear on the Blocks tab, click the Duration tab once a
second and report when the click takes (how long the plugin stays busy after analysis).

Usage (on the VM): python3 rm_tab_latency.py <reaper_reamix .so> <audio>
Measured 2026-09-02 on Sia (cached analysis): s121 build +1.7 s, s122 build +3.9 s.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_open, xtools, rm_blocks
from PIL import Image
so, media = sys.argv[1], sys.argv[2]
def is_duration(png):
    p = Image.open(png).convert("RGB").load()
    # Duration tab underline: orange pixels near y=143 under x 12..80; Blocks underline under x 180..276
    def orange(x0, x1):
        return sum(1 for x in range(x0, x1) for y in range(140, 147) if p[x, y][0] > 150 and p[x, y][2] < 110 and p[x, y][0] - p[x, y][2] > 60)
    return orange(12, 80), orange(180, 276)
rm_blocks.install(so)
rm_open.remove_bridge(); rm_open.deploy_bridge()
ok, out = rm_open.launch(["-new", "-nosplash"]); assert ok, out
b = rm_open.client(); b.wait_ready(120)
ids, length = rm_open.open_reamix(b, media)
rm = None
for _ in range(20):
    rm = xtools.find("reamix.me")
    if rm: break
    time.sleep(0.5)
x0, y0 = rm["x"], rm["y"]; xtools.focus(rm["win"])
xtools.click(x0 + rm_blocks.ANALYZE_BTN[0], y0 + rm_blocks.ANALYZE_BTN[1])
t0 = time.time()
while time.time() - t0 < 120:
    xtools.click(x0 + rm_blocks.BLOCKS_TAB[0], y0 + rm_blocks.BLOCKS_TAB[1]); time.sleep(3.0)
    png = rm_open.OUT / "probe2_poll.png"; xtools.capture(rm["win"], str(png))
    _, rs = rm_blocks.chips(png)
    if rs: break
tc = time.time()
for i in range(12):
    xtools.click(x0 + rm_blocks.DURATION_TAB[0], y0 + rm_blocks.DURATION_TAB[1]); time.sleep(0.4)
    png = rm_open.OUT / f"probe2_try{i}.png"; xtools.capture(rm["win"], str(png))
    d, bl = is_duration(str(png))
    print(f"try {i} at +{time.time()-tc:.1f}s: duration_px={d} blocks_px={bl}", flush=True)
    if d > 20 and bl < 5: break
    time.sleep(0.6)
rm_open.quit_reaper(b); rm_open.remove_bridge()
