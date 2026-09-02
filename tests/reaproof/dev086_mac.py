#!/usr/bin/env python3
"""DEV-086 macOS leg (ReaProof isolated REAPER, screencapture per CGWindowID).

Drives the built dylib inside ReaProof's pinned, isolated REAPER 7.75: insert a
media item, open reamix, Blocks tab, drag on the section bar, capture the MARK
SECTION picker (a JUCE CallOutBox = separate NSWindow) and the reamix window.
The macOS oracle is a regression check: the picker rendered by the fixed dylib
must be pixel-identical (within AA tolerance) to the baseline dylib's picker.

Run from the repo root (ReaProof copy at ./reaproof):
  PYTHONPATH=reaproof/src LC_ALL=en_US.UTF-8 LC_NUMERIC=C TZ=UTC \
    python3 tests/reaproof/dev086_mac.py --dylib build/reaper_reamix.dylib --tag after
  ... --dylib build/reaper_reamix_before.dylib --tag before
  python3 tests/reaproof/dev086_mac.py --compare <before.png> <after.png>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reaproof" / "src"))

from reaproof.runner.session import ReaperSession  # noqa: E402
from reaproof.observe.input import _MacMouse  # noqa: E402

ORT = REPO / "vendor" / "onnxruntime" / "lib" / "libonnxruntime.1.24.4.dylib"
MEDIA = REPO / "references" / "golden" / "phase-2" / "billie_jean.mp3"
OUT = REPO / "tests" / "reaproof" / "evidence" / "dev086-mac"
BG0 = (0x0B, 0x0A, 0x09)

# client-relative targets (same layout as the Linux/Windows legs; verified on
# the 720x760 client via capture before clicking)
BLOCKS_TAB = (210, 125)
SEGBAR_Y = 646
SEGBAR_X0, SEGBAR_X1 = 120, 260


def windows_of(pid):
    import Quartz
    out = []
    for w in Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") != pid:
            continue
        b = w["kCGWindowBounds"]
        out.append({"id": w["kCGWindowNumber"], "name": w.get("kCGWindowName") or "",
                    "x": int(b["X"]), "y": int(b["Y"]), "w": int(b["Width"]), "h": int(b["Height"]),
                    "layer": w.get("kCGWindowLayer", 0)})
    return out


def shot(wid, path):
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(path)], capture_output=True)
    return Path(path).exists()


def slow_click(mouse, x, y):
    """move, settle, hold: a 30 ms down/up is dropped by the JUCE window when the
    app was not already active (probed 2026-09-02); 150 ms hold registers."""
    mouse.move(x, y); time.sleep(0.4)
    mouse.down(x, y); time.sleep(0.15)
    mouse.up(x, y); time.sleep(0.3)


def click_until_changed(mouse, wid, x, y, ref_png, tmp_png, tries=6):
    """Click (settled) and re-capture until the window pixels differ from ref_png;
    the first click on a not-yet-key JUCE window is swallowed by activation."""
    import filecmp
    for _ in range(tries):
        slow_click(mouse, x, y); time.sleep(0.8)
        shot(wid, tmp_png)
        if not filecmp.cmp(str(ref_png), str(tmp_png), shallow=False):
            return True
        time.sleep(0.7)
    return False


def front(pid):
    subprocess.run(["osascript", "-e",
                    'tell application "System Events" to set frontmost of '
                    f'(first process whose unix id is {pid}) to true'], capture_output=True)


def client_origin(win, png):
    """Return (screen_x, screen_y, scale) of the JUCE client origin: the native
    title bar is light; the first dark row (sum < 90) at x = 4 starts the client."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    scale = im.size[0] / max(1, win["w"])
    px = im.load()
    oy = 0
    seen_light = False          # rounded-corner pixels above the title bar are black
    for y in range(im.size[1]):
        s_ = sum(px[4, y])
        if s_ > 400:
            seen_light = True
        elif seen_light and s_ < 90:
            oy = y
            break
    return win["x"], win["y"] + oy / scale, scale


def find_orange_row(png, scale, y_from, y_to):
    """Image row (client px) with the most orange pixels (the section-bar hint
    'DRAG HERE TO MARK YOUR FIRST BLOCK' / block tiles) between y_from..y_to."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    # first orange-dominant row scanning DOWN from the ruler: the section-bar
    # hint sits right under the waveform, above the Edit-tuning sliders (whose
    # orange tracks are wider but lower on the page).
    for y in range(int(y_from * scale), min(im.size[1], int(y_to * scale))):
        n = 0
        for x in range(min(im.size[0], int(600 * scale))):   # skip the right-aligned volume badge
            r, g, b = px[x, y]
            if r > 110 and r > g + 20 and g > b + 12:      # warm text (hint is ~(127,94,59))
                n += 1
        if n >= 5:           # thin caption text; slider tracks come later
            return y / scale, n
    return None, 0


def waveform_present(png, scale, y_from, y_to):
    """Analysis done proxy: the waveform band has many mid-grey pixels."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    n = 0
    for y in range(int(y_from * scale), min(im.size[1], int(y_to * scale)), 4):
        for x in range(0, im.size[0], 4):
            r, g, b = px[x, y]
            if 70 < r < 200 and abs(r - g) < 30 and abs(g - b) < 40:
                n += 1
    return n


def run(dylib, tag, runs):
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(runs):
        rtag = f"{tag}_run{i + 1}"
        with ReaperSession("reamix-dev086", extensions=[Path(dylib), ORT]) as s:
            pid = s.handle.pid
            r = {"tag": rtag, "picker": None}
            length = s.eval(f'''
reaper.Main_OnCommand(40023, 0)
reaper.InsertTrackAtIndex(0, false)
local tr = reaper.GetTrack(0, 0)
reaper.SetOnlyTrackSelected(tr)
reaper.SetEditCurPos(0, false, false)
reaper.InsertMedia("{MEDIA}", 0)
local it = reaper.GetTrackMediaItem(tr, 0)
reaper.SetMediaItemSelected(it, true)
reaper.UpdateArrange()
return reaper.GetMediaItemInfo_Value(it, "D_LENGTH")''', timeout=120, hang_timeout=60)
            r["item_len"] = length
            # the isolated profile is pristine: pre-mark the first-run welcome as
            # seen, otherwise the "Welcome to reamix.me" window covers the UI
            s.eval('reaper.SetExtState("reamix.me", "welcome_shown", "1", true) return true')
            cmd = s.eval('return reaper.NamedCommandLookup("_reamix_ShowWindow")')
            for _ in range(4):
                if s.eval(f'return reaper.GetToggleCommandState({cmd})') == 1:
                    break
                s.eval(f'reaper.Main_OnCommand({cmd}, 0) return true', hang_timeout=30)
                time.sleep(2.0)
            win = None
            for _ in range(30):
                front(pid)
                win = next((w for w in windows_of(pid) if w["name"] == "reamix.me"), None)
                if win:
                    break
                time.sleep(0.5)
            if not win:
                r["error"] = "reamix window not found"; results.append(r); continue
            time.sleep(3.0)
            wpng = OUT / f"{rtag}_window.png"
            shot(win["id"], wpng)
            ox, oy, scale = client_origin(win, wpng)
            r["window"] = win; r["client_origin"] = [ox, oy, scale]
            ch = win["h"] - (oy - win["y"])          # client height (title bar excluded)
            mouse = _MacMouse()
            front(pid)
            # activation click on a neutral spot (source panel text), then the
            # Analyze button at the right of the source panel (client y ~74)
            wave_band = (oy - win["y"] + 300, oy - win["y"] + 420)
            # analysis may already be in the shared disk cache (auto-loaded on
            # item select): click Analyze only when the waveform is absent
            already = waveform_present(wpng, scale, *wave_band) > 400
            r["cache_hit"] = already
            if not already:
                r["analyze_click"] = click_until_changed(mouse, win["id"], ox + 672, oy + 74, wpng,
                                                         OUT / f"{rtag}_after_analyze_click.png")
            # wait until the waveform is present AND the picture is stable
            # (no progress-bar animation, no layout shift)
            import filecmp
            analyzed = False
            prev = None
            for _ in range(80):
                cur = OUT / f"{rtag}_poll.png"
                shot(win["id"], cur)
                if waveform_present(cur, scale, *wave_band) > 400 and prev is not None \
                        and filecmp.cmp(str(prev), str(cur), shallow=False):
                    analyzed = True; break
                prev2 = OUT / f"{rtag}_poll_prev.png"
                import shutil; shutil.copy(cur, prev2); prev = prev2
                time.sleep(2.0)
            r["analyzed"] = analyzed
            front(pid)
            bpng = OUT / f"{rtag}_blocks.png"
            r["blocks_click"] = click_until_changed(mouse, win["id"], ox + BLOCKS_TAB[0], oy + BLOCKS_TAB[1],
                                                    OUT / f"{rtag}_poll.png", bpng)
            time.sleep(1.0); shot(win["id"], bpng)
            seg_y, seg_n = find_orange_row(bpng, scale, oy - win["y"] + 300, oy - win["y"] + ch - 60)
            r["segbar_hint"] = [seg_y, seg_n]
            seg_y = (seg_y - (oy - win["y"])) if seg_y is not None else SEGBAR_Y
            picker = None
            for attempt in range(6):
                known = {w["id"] for w in windows_of(pid)}
                x0, y = ox + SEGBAR_X0, oy + seg_y
                x1 = ox + SEGBAR_X1
                mouse.move(x0, y); time.sleep(0.4)
                mouse.down(x0, y); time.sleep(0.15)
                for k in range(1, 13):
                    mouse.drag_step(x0 + (x1 - x0) * k / 12, y); time.sleep(0.04)
                time.sleep(0.15)
                mouse.up(x1, y)
                for _ in range(12):
                    cands = [w for w in windows_of(pid) if w["id"] not in known and w["w"] > 100 and w["h"] > 100]
                    if cands:
                        picker = cands[0]; break
                    time.sleep(0.25)
                if picker:
                    break
                time.sleep(3.0)
            r["drag_attempts"] = attempt + 1
            if picker:
                time.sleep(1.0)
                ppng = OUT / f"{rtag}_picker.png"
                shot(picker["id"], ppng)
                r["picker"] = picker; r["picker_png"] = str(ppng)
                # Escape closes the picker (CallOutBox modal)
                import Quartz
                for down in (True, False):
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, Quartz.CGEventCreateKeyboardEvent(None, 53, down))
                    time.sleep(0.05)
            results.append(r)
            print(json.dumps(r, indent=1, default=str), flush=True)
    (OUT / f"{tag}_result.json").write_text(json.dumps(results, indent=1, default=str))
    return results


def compare(a, b, tol=8):
    from PIL import Image, ImageChops
    ia = Image.open(a).convert("RGBA"); ib = Image.open(b).convert("RGBA")
    if ia.size != ib.size:
        return {"same_size": False, "a": ia.size, "b": ib.size}
    d = ImageChops.difference(ia, ib)
    px = list(d.getdata())
    n = sum(1 for p in px if max(p) > tol)
    return {"same_size": True, "size": ia.size, "pixels_over_tol": n, "total": len(px),
            "pct": round(100.0 * n / len(px), 3), "bbox": d.getbbox()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dylib"); ap.add_argument("--tag", default="run")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=1)); return
    run(a.dylib, a.tag, a.runs)


if __name__ == "__main__":
    main()
