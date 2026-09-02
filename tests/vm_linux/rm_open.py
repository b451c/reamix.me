#!/usr/bin/env python3
"""reamix Linux VM driver: launch the VM's own REAPER with the ReaProof bridge as
Scripts/__startup.lua, insert a track + media item, select it, open the reamix
window, and dump an X capture of the window. Leaves REAPER running for the
interactive steps (rm_step.py) unless --quit.

Usage: rm_open.py <media path> [--quit]
Requires: ~/reaproof (ReaProof copy), ~/mp_launch_linux.sh, python-xlib, PIL.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "reaproof" / "src"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reaproof.control.bridge_client import BridgeClient  # noqa: E402
import xtools  # noqa: E402

RES = Path.home() / ".config" / "REAPER"
RUN = RES / "_reaproof"
STARTUP = RES / "Scripts" / "__startup.lua"
BRIDGE = Path.home() / "reaproof" / "bridge" / "reaproof_bridge.lua"
OUT = Path.home() / "reamix_vm"
OUT.mkdir(exist_ok=True)


def alive():
    return subprocess.run(["pgrep", "-x", "reaper"], capture_output=True).returncode == 0


def pid():
    r = subprocess.run(["pgrep", "-x", "reaper"], capture_output=True, text=True).stdout.split()
    return int(r[0]) if r else None


def deploy_bridge():
    STARTUP.parent.mkdir(parents=True, exist_ok=True)
    STARTUP.write_text(BRIDGE.read_text())


def remove_bridge():
    if STARTUP.exists():
        STARTUP.unlink()
    subprocess.run(["rm", "-rf", str(RUN)])


def launch(args):
    out = subprocess.run(["bash", str(Path.home() / "mp_launch_linux.sh")] + args,
                         capture_output=True, text=True).stdout
    return "REAPER-STARTED" in out, out


def client():
    return BridgeClient(RUN, is_alive=alive)


def open_reamix(b, media):
    ids = b.eval('return {toggle=reaper.NamedCommandLookup("_reamix_ShowWindow"), '
                 'ver=reaper.GetAppVersion()}')
    assert ids["toggle"], "reamix_ShowWindow not registered: %r" % (ids,)
    length = b.eval(f'''
reaper.Main_OnCommand(40023, 0)
reaper.InsertTrackAtIndex(0, false)
local tr = reaper.GetTrack(0, 0)
reaper.SetOnlyTrackSelected(tr)
reaper.SetEditCurPos(0, false, false)
reaper.InsertMedia("{media}", 0)
local it = reaper.GetTrackMediaItem(tr, 0)
reaper.SetMediaItemSelected(it, true)
reaper.UpdateArrange()
return reaper.GetMediaItemInfo_Value(it, "D_LENGTH")
''', timeout=120, hang_timeout=60)
    st = b.eval(f'return reaper.GetToggleCommandState({ids["toggle"]})')
    if st != 1:
        b.eval(f'reaper.Main_OnCommand({ids["toggle"]}, 0) return true', hang_timeout=30)
        time.sleep(2.0)
    return ids, length


def quit_reaper(b):
    try:
        b.eval('local n=0 local function q() n=n+1 if n>=15 then reaper.Main_OnCommand(40004,0) '
               'else reaper.defer(q) end end reaper.defer(q) return true')
    except Exception as e:
        print("quit eval:", repr(e))
    t0 = time.monotonic()
    while alive() and time.monotonic() - t0 < 15:
        time.sleep(1)
        q = xtools.find("REAPER Query")
        if q:
            xtools.focus(q["win"]); xtools.key("n"); time.sleep(2)
    if alive():
        os.kill(pid(), signal.SIGTERM)
        t1 = time.monotonic()
        while alive() and time.monotonic() - t1 < 10:
            time.sleep(0.5)


def main():
    media = sys.argv[1]
    do_quit = "--quit" in sys.argv
    if alive():
        print("REAPER already running - abort"); sys.exit(2)
    remove_bridge(); deploy_bridge()
    ok, out = launch(["-new", "-nosplash"])
    print("launch:", ok, out.strip().splitlines()[-1] if out else "")
    b = client()
    env = b.wait_ready(120)
    print("bridge ready:", json.dumps(env))
    ids, length = open_reamix(b, media)
    print("item length:", length, "ids:", ids)
    time.sleep(3.0)
    wins = [e for e in xtools.windows() if e["pid"] == pid() or "reamix" in e["name"].lower()]
    for e in wins:
        print("win:", repr(e["name"]), f"{e['w']}x{e['h']}+{e['x']}+{e['y']}", e["cls"], "ovr=%d" % e["override"])
    rm = xtools.find("reamix")
    if rm:
        xtools.capture(rm["win"], str(OUT / "reamix_window.png"))
        print("captured", OUT / "reamix_window.png", f"{rm['w']}x{rm['h']}+{rm['x']}+{rm['y']}")
    xtools.capture_root(str(OUT / "root.png"))
    if do_quit:
        quit_reaper(b); remove_bridge(); print("quit; alive =", alive())
    else:
        print("REAPER left running (bridge deployed; run rm_open.py --cleanup later)")


if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        b = client()
        if alive():
            quit_reaper(b)
        remove_bridge(); print("cleaned; alive =", alive())
    else:
        main()
