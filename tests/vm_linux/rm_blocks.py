#!/usr/bin/env python3
"""Sesja 120 (DEV-097) Linux gate: Block Assembly suggestion chips + arrangement persistence.

Flow per run: install the .so -> launch the VM's REAPER with the ReaProof bridge ->
insert + select a media item -> open reamix -> Blocks tab -> wait until suggestion
chips (tinted, chip-hued runs) appear in the section bar (= analysis complete and the
whole-track pool proposed blocks) -> click the first chip -> Lua reads the item's
P_EXT:reamix_blocks -> must be the v2 object with one block and queue [0] -> click a
second chip that does not overlap the first block -> queue [0, 1] -> capture the
window (seam pill "~NN%") -> quit REAPER.

Mutation: a pre-sesja-120 .so shows no chips in Blocks mode; its pulsing section-bar
hint ("DRAG HERE TO MARK YOUR FIRST BLOCK") still passes the tinted-run detector as
three short word runs, the click on it creates nothing and P_EXT stays empty -> the
verdict is the P_EXT oracle (v2 object + queue), never the chip detector alone.

Usage: rm_blocks.py --so <reaper_reamix .so> --media <audio> --expect green|red [--runs 2]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rm_open  # noqa: E402
import xtools  # noqa: E402

PLUGIN = Path.home() / ".config" / "REAPER" / "UserPlugins" / "reaper_reamix.so"
OUT = Path.home() / "reamix_vm"

# Window-relative targets (reamix window 720x760 on the VM; see rm_dev086.py).
BLOCKS_TAB = (210, 125)
DURATION_TAB = (45, 125)
ANALYZE_BTN = (672, 74)      # source panel "Analyze" button (analysis does not auto-start)
SEGBAR_Y = 646
SEGBAR_X0, SEGBAR_X1 = 40, 700
CHIP_WAIT_S = 240          # cold analysis of a 3-4 min song on the aarch64 VM


def tinted(p):
    r, g, b = p
    return max(r, g, b) - min(r, g, b) >= 12


def runs_in_row(px, y, x0, x1, gap=3, min_w=30):
    runs, start, last = [], None, None
    for x in range(x0, x1):
        if tinted(px[x, y]):
            if start is None:
                start = x
            elif x - last > gap:
                if last - start + 1 >= min_w:
                    runs.append((start, last))
                start = x
            last = x
    if start is not None and last - start + 1 >= min_w:
        runs.append((start, last))
    return runs


def chips(png):
    """Best row in a +-10 px band around the section bar: the runs of tinted pixels
    >= 30 px wide (a chip is fill tint + 1 px border + coloured label, one run).
    Returns (y, [(x0, x1), ...]) window-relative."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    best = (SEGBAR_Y, [])
    best_score = (0, 0)
    # Sesja 121: rows by distance from the bar centre, ties keep the nearer
    # row - the kind-tinted waveform above the bar (model sections) is as
    # tinted as the bar itself, and a click there only seeks.
    # Sesja 123: score = (number of runs, tinted pixels). The bottom 2 px of
    # the bar are pure section cells (one 660 px run, no chip gaps) and won
    # the old "most tinted pixels" pick when the window sat a pixel lower;
    # the centre of that run is the gap between two chips (s123b red run).
    for y in sorted(range(SEGBAR_Y - 6, SEGBAR_Y + 7), key=lambda v: abs(v - SEGBAR_Y)):
        if y < 0 or y >= im.size[1]:
            continue
        rs = runs_in_row(px, y, SEGBAR_X0, min(SEGBAR_X1, im.size[0]))
        score = (len(rs), sum(b - a for a, b in rs))
        if score > best_score:
            best_score, best = score, (y, rs)
    return best


def read_pext(b):
    return b.eval('''
local it = reaper.GetSelectedMediaItem(0, 0)
if not it then return "NOITEM" end
local ok, s = reaper.GetSetMediaItemInfo_String(it, "P_EXT:reamix_blocks", "", false)
return s or ""
''')


def parse_state(raw):
    try:
        v = json.loads(raw) if raw else None
    except Exception:
        return None
    if isinstance(v, list):
        return {"v": 1, "blocks": v, "queue": None}
    if isinstance(v, dict):
        return v
    return None


def install(so):
    if PLUGIN.exists():
        PLUGIN.unlink()
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
    time.sleep(3.0)
    x0, y0 = rm["x"], rm["y"]
    result = {"tag": tag, "media_len": length, "chips1": [], "chips2": [], "pext1": None, "pext2": None}
    xtools.click(x0 + ANALYZE_BTN[0], y0 + ANALYZE_BTN[1])
    time.sleep(2.0)

    # Blocks tab, then poll for chips (the tab click is repeated: before analysis
    # completes the tab is disabled and the click is a no-op).
    t0 = time.time()
    y_row, rs = SEGBAR_Y, []
    while time.time() - t0 < CHIP_WAIT_S:
        # Duration -> Blocks: the auto Duration remix that follows analysis can
        # land while Blocks is already active and leave the Remix variant on
        # screen (DEV-101); a tab switch resets the material view to Source.
        xtools.click(x0 + DURATION_TAB[0], y0 + DURATION_TAB[1])
        time.sleep(0.8)
        xtools.click(x0 + BLOCKS_TAB[0], y0 + BLOCKS_TAB[1])
        time.sleep(2.5)
        png = OUT / f"{tag}_poll.png"
        xtools.capture(rm["win"], str(png))
        y_row, rs = chips(png)
        if rs:
            break
    result["chip_wait_s"] = round(time.time() - t0, 1)
    result["chips1"] = rs
    xtools.capture(rm["win"], str(OUT / f"{tag}_chips.png"))
    if not rs:
        rm_open.quit_reaper(b); rm_open.remove_bridge()
        result["reaper_alive_after"] = rm_open.alive()
        return result

    # Click the first chip -> block + queue [0] in P_EXT.
    c1 = rs[0]
    cx1 = (c1[0] + c1[1]) // 2
    xtools.click(x0 + cx1, y0 + y_row)
    time.sleep(1.5)
    raw1 = read_pext(b)
    result["pext1_raw"] = raw1[:300] if isinstance(raw1, str) else raw1
    st1 = parse_state(raw1 if isinstance(raw1, str) else "")
    result["pext1"] = st1
    xtools.capture(rm["win"], str(OUT / f"{tag}_after1.png"))

    # Second chip: a run that does not overlap the first block's x-range.
    y2, rs2 = chips(OUT / f"{tag}_after1.png")
    cands = [r for r in rs2 if r[1] < c1[0] - 4 or r[0] > c1[1] + 4]
    result["chips2"] = rs2
    if cands:
        c2 = max(cands, key=lambda r: r[1] - r[0])
        cx2 = (c2[0] + c2[1]) // 2
        xtools.click(x0 + cx2, y0 + y2)
        time.sleep(2.5)                  # live junction preview lands async
        raw2 = read_pext(b)
        result["pext2_raw"] = raw2[:300] if isinstance(raw2, str) else raw2
        result["pext2"] = parse_state(raw2 if isinstance(raw2, str) else "")
        xtools.capture(rm["win"], str(OUT / f"{tag}_after2.png"))

    rm_open.quit_reaper(b)
    rm_open.remove_bridge()
    result["reaper_alive_after"] = rm_open.alive()
    return result


def verdict(r):
    """green = chips appeared, the first click wrote a v2 payload with one block and
    queue [0]; when a second chip was clickable, queue [0, 1] with two blocks."""
    st1 = r.get("pext1")
    if not r.get("chips1") or not st1 or st1.get("v") != 2:
        return False
    if len(st1.get("blocks") or []) != 1 or st1.get("queue") != [0]:
        return False
    st2 = r.get("pext2")
    if r.get("chips2") and st2 is not None:
        if len(st2.get("blocks") or []) != 2 or sorted(st2.get("queue") or []) != [0, 1] \
                or len(st2.get("queue") or []) != 2:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--media", required=True)
    ap.add_argument("--expect", choices=["green", "red"], required=True)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    install(a.so)
    tag = Path(a.so).stem
    results = []
    for i in range(a.runs):
        r = one_run(a.media, f"{tag}_blocks_run{i + 1}")
        r["green"] = verdict(r)
        print(json.dumps(r, indent=1), flush=True)
        results.append(r)
    verdicts = [r["green"] for r in results]
    want = a.expect == "green"
    ok = all(v == want for v in verdicts)
    print("VERDICTS", verdicts, "expected", a.expect, "->", "PASS" if ok else "FAIL")
    (OUT / f"{tag}_blocks_result.json").write_text(json.dumps(results, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
