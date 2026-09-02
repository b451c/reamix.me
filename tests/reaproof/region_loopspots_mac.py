#!/usr/bin/env python3
"""ADR-115 E11 loop-spot suggestions, macOS leg (ReaProof isolated REAPER).

Sesja 117. Region mode shows suggested loop spots as quality-coloured chips
in the section bar (whole track without a selection, inside the selection
with one) and a verdict pill on the selection scrim. Two runs on Billie Jean:

  whole : analyse, click the Region tab with NO time selection -> chips
          present in the section bar; click the first chip -> REAPER's
          time-selection (Lua read-back) becomes that span (>= 6 s, inside
          the item).
  verse : analyse, Lua time-selection 120-136 s (vocal verse, sesja 116:
          no clean loop) -> auto-Region, red "NO CLEAN LOOP" pill, no
          chips; then 60-90 s (beat intro) -> green/amber pill + chips.

Evidence PNG + JSON under tests/reaproof/evidence/region-loopspots-mac/.

Run from the repo root (ReaProof copy at ./reaproof, user's REAPER closed):
  PYTHONPATH=reaproof/src LC_ALL=en_US.UTF-8 LC_NUMERIC=C TZ=UTC \
    python3 tests/reaproof/region_loopspots_mac.py --dylib build/reaper_reamix.dylib --tag e11
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

OUT = REPO / "tests" / "reaproof" / "evidence" / "region-loopspots-mac"

TAB_ROW_Y = 125            # client px: mode tabs row (dev086 BLOCKS_TAB)
CHIP_BAND = (300, 560)     # client px: rows searched for chips (canvas + section bar, above the sliders)
SEGBAR_X_MAX = 640         # client px: exclude the orange volume badge on the right
CANVAS_TOP_BAND = (250, 300)  # client px: verdict pill rows (canvas top + 2 px)
GOOD = (108, 194, 138)     # theme Good
WARN = (224, 178, 74)      # theme Warn
BAD = (213, 101, 76)       # theme Bad


def is_good(p):
    r, g, b = p
    return g >= 90 and g > r + 25 and g > b + 20   # dim 1 px border captures as (68,103,74)


def is_warn(p):
    # screencapture shifts the theme colours (Warn text lands near (200,160,72),
    # its border near (176,141,64)); classify by hue so the Accent slider
    # tracks (232,161,90): g/r 0.69) stay out.
    r, g, b = p
    return r >= 140 and g >= 0.75 * r and b < 0.5 * r


def is_bad(p):
    r, g, b = p
    return r >= 150 and 0.35 * r <= g <= 0.6 * r and b <= 0.5 * r


CLASSIFIER = {GOOD: is_good, WARN: is_warn, BAD: is_bad}


def near(px, rgb, tol=None):
    return CLASSIFIER[rgb](px)


def tol_of(rgb):
    return None


def colour_runs(png, scale, y_from, y_to, colours, min_gap=24, x_max=SEGBAR_X_MAX):
    """x-runs (client px) of chip-coloured pixels (chip text / border) inside
    the band: the section bar sits at a window-height dependent y, so the band
    is wide and the colours tight (nothing else in the canvas / section bar
    is pure Good or pure Warn; the volume badge on the right is excluded)."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    cols = set()
    for y in range(int(y_from * scale), min(im.size[1], int(y_to * scale))):
        for x in range(min(im.size[0], int(x_max * scale))):
            p = px[x, y]
            if any(near(p, c, tol_of(c)) for c in colours):
                cols.add(x)
    xs = sorted(cols)
    runs = []
    for x in xs:
        if runs and x - runs[-1][1] <= min_gap * scale:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    # a narrow chip shows only its two 1 px borders: keep 1 px runs, merge
    # within a chip width (chips never overlap, adjacent ones merge harmlessly)
    return [(a / scale, b / scale) for a, b in runs]


def count_colour(png, scale, y_from, y_to, rgb):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    n = 0
    for y in range(int(y_from * scale), min(im.size[1], int(y_to * scale))):
        for x in range(im.size[0]):
            if near(px[x, y], rgb, tol_of(rgb)):
                n += 1
    return n


def chip_row(png, scale, y_from, y_to):
    """Image row (client px) with the most chip-coloured pixels in the band."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    best, best_n = (y_from + y_to) / 2, 0
    for y in range(int(y_from * scale), min(im.size[1], int(y_to * scale))):
        n = sum(1 for x in range(min(im.size[0], int(SEGBAR_X_MAX * scale)))
                if is_good(px[x, y]) or is_warn(px[x, y]))
        if n > best_n:
            best, best_n = y / scale, n
    return best


def tab_centres(png, scale, y):
    """Light-text clusters along the tabs row: Duration / Region(+BETA) / Blocks(+BETA)."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    px = im.load()
    row = int(y * scale)
    xs = []
    for dy in (-3, 0, 3):
        yy = min(im.size[1] - 1, max(0, row + int(dy * scale)))
        for x in range(im.size[0]):
            if sum(px[x, yy]) > 300:      # tab labels (grey / accent) on the dark bar
                xs.append(x)
    xs = sorted(set(xs))
    clusters = []
    for x in xs:
        if clusters and x - clusters[-1][1] <= 14 * scale:
            clusters[-1][1] = x
        else:
            clusters.append([x, x])
    return [((a + b) / 2 / scale, (b - a) / scale) for a, b in clusters if b - a > 6 * scale]


def open_reamix(s, pid, rtag, r):
    """Insert the media, open the window, wait for the analysed waveform."""
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
        r["error"] = "reamix window not found"
        return None
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
    t0 = time.time()
    present_since = None
    while time.time() - t0 < 240:
        time.sleep(2.0)
        cur = OUT / f"{rtag}_poll.png"
        w2 = next((w for w in base.windows_of(pid) if w["name"] == "reamix.me"), None)
        if w2 is None:
            r["window_lost_at"] = round(time.time() - t0, 1)
            return None
        win = w2
        base.shot(win["id"], cur)
        if not cur.exists():
            continue
        if base.waveform_present(cur, scale, *wave_band) > 400:
            # the preview playhead may keep the picture changing; two polls
            # with the waveform present are enough (DEV-092: the harness
            # quits REAPER after ~2-3 min, keep the leg short)
            present_since = present_since or time.time()
            if time.time() - present_since >= 3.0:
                break
        else:
            present_since = None
    r["analyze_seconds"] = round(time.time() - t0, 1)
    ctx = {"win": win, "ox": ox, "oy": oy, "scale": scale, "mouse": mouse,
           "cy": oy - win["y"]}   # cy = client origin y inside the capture (client px)
    return ctx


def capture(ctx, pid, path):
    win = next((w for w in base.windows_of(pid) if w["name"] == "reamix.me"), None)
    if win is None:
        return False
    ctx["win"] = win
    base.shot(win["id"], path)
    return path.exists()


def band(ctx, client_band):
    return (ctx["cy"] + client_band[0], ctx["cy"] + client_band[1])


def run_whole(s, pid, rtag, r):
    ctx = open_reamix(s, pid, rtag, r)
    if ctx is None:
        return
    ox, oy, scale, mouse = ctx["ox"], ctx["oy"], ctx["scale"], ctx["mouse"]
    # start from a clean state: no REAPER time-selection (an auto-Region
    # left over from the project template would otherwise own the mode)
    s.eval('reaper.GetSet_LoopTimeRange(true, false, 0, 0, false) return true')
    time.sleep(1.0)
    before = OUT / f"{rtag}_before_tab.png"
    capture(ctx, pid, before)
    tabs = tab_centres(before, scale, ctx["cy"] + TAB_ROW_Y)
    r["tabs"] = tabs
    if len(tabs) < 2:
        r["error"] = "mode tabs not found"
        return
    region_x = tabs[1][0]
    r["region_tab_x"] = region_x
    base.front(pid)
    after = OUT / f"{rtag}_region_tab.png"
    base.slow_click(mouse, ox + region_x, oy + TAB_ROW_Y)
    time.sleep(1.5)
    capture(ctx, pid, after)
    chips = colour_runs(after, scale, *band(ctx, CHIP_BAND), [GOOD, WARN])
    r["chips"] = chips
    r["chips_present"] = len(chips) >= 1
    if not chips:
        return
    cx = (chips[0][0] + chips[0][1]) / 2
    cy = chip_row(after, scale, *band(ctx, CHIP_BAND)) - ctx["cy"]
    r["chip_row_client_y"] = cy
    base.front(pid)
    base.slow_click(mouse, ox + cx, oy + cy)
    time.sleep(1.5)
    clicked = OUT / f"{rtag}_chip_click.png"
    capture(ctx, pid, clicked)
    sel = s.eval('local a, b = reaper.GetSet_LoopTimeRange(false, false, 0, 0, false) '
                 'return string.format("%.3f;%.3f", a, b)')
    r["time_selection_after_click"] = sel
    try:
        a, b = (float(v) for v in str(sel).split(";"))
    except Exception:  # noqa: BLE001
        a, b = 0.0, 0.0
    r["selection_ok"] = (b - a) >= 6.0 and a >= 0.0 and b <= float(r["item_len"]) + 0.01
    r["verdict_px"] = {
        "good": count_colour(clicked, scale, *band(ctx, CANVAS_TOP_BAND), GOOD),
        "warn": count_colour(clicked, scale, *band(ctx, CANVAS_TOP_BAND), WARN),
        "bad": count_colour(clicked, scale, *band(ctx, CANVAS_TOP_BAND), BAD),
    }
    r["verdict_ok"] = (r["verdict_px"]["good"] + r["verdict_px"]["warn"]) > 200
    r["reaper_alive"] = s.eval('return 1') == 1
    r["ok"] = bool(r["chips_present"] and r["selection_ok"] and r["verdict_ok"] and r["reaper_alive"])


def run_verse(s, pid, rtag, r):
    ctx = open_reamix(s, pid, rtag, r)
    if ctx is None:
        return
    scale = ctx["scale"]
    pos = s.eval('local it = reaper.GetTrackMediaItem(reaper.GetTrack(0, 0), 0) '
                 'return reaper.GetMediaItemInfo_Value(it, "D_POSITION")')
    pos = float(pos)
    results = {}
    for name, (a, b), expect_loop in (("verse", (120.0, 136.0), False), ("intro", (60.0, 90.0), True)):
        s.eval(f'reaper.GetSet_LoopTimeRange(true, false, {pos + a}, {pos + b}, false) '
               'reaper.UpdateTimeline() return true')
        time.sleep(2.0)
        png = OUT / f"{rtag}_{name}.png"
        capture(ctx, pid, png)
        px = {
            "good": count_colour(png, scale, *band(ctx, CANVAS_TOP_BAND), GOOD),
            "warn": count_colour(png, scale, *band(ctx, CANVAS_TOP_BAND), WARN),
            "bad": count_colour(png, scale, *band(ctx, CANVAS_TOP_BAND), BAD),
        }
        chips = colour_runs(png, scale, *band(ctx, CHIP_BAND), [GOOD, WARN])
        ok = (px["good"] + px["warn"] > 200 and len(chips) >= 1) if expect_loop \
            else (px["bad"] > 200 and len(chips) == 0)
        results[name] = {"verdict_px": px, "chips": chips, "ok": ok}
    r["cases"] = results
    r["reaper_alive"] = s.eval('return 1') == 1
    r["ok"] = bool(all(c["ok"] for c in results.values()) and r["reaper_alive"])


def run(dylib: Path, tag: str, runs: int) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(runs):
        for name, fn in (("whole", run_whole), ("verse", run_verse)):
            rtag = f"{tag}_{name}_run{i + 1}"
            r: dict = {"tag": rtag, "ok": False}
            with ReaperSession("reamix-e11", extensions=[Path(dylib), base.ORT]) as s:
                fn(s, s.handle.pid, rtag, r)
            results.append(r)
            print(json.dumps(r, default=str), flush=True)
    (OUT / f"{tag}_result.json").write_text(json.dumps(results, indent=1, default=str))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dylib", required=True)
    ap.add_argument("--tag", default="e11")
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    res = run(Path(a.dylib), a.tag, a.runs)
    ok = all(r.get("ok") for r in res) and len(res) == 2 * a.runs
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
