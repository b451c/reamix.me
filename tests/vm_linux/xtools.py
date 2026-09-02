#!/usr/bin/env python3
"""X11 helpers for the Linux VM leg (Xwayland under GNOME, no screenshot tools,
no sudo): window listing, per-window pixel capture (XGetImage), XTEST input.

Pattern lifted from EditView/cpp/tests/vm_linux/{xwin,xfocus}.py.
"""
import os
import sys
import time
from pathlib import Path

xa = sorted(Path("/run/user/1000").glob(".mutter-Xwaylandauth.*"))
if xa:
    os.environ["XAUTHORITY"] = str(xa[0])
os.environ.setdefault("DISPLAY", ":0")

from Xlib import X, XK, display, protocol  # noqa: E402
from Xlib.ext import xtest  # noqa: E402

d = display.Display()
root = d.screen().root
NET_NAME = d.intern_atom("_NET_WM_NAME")
NET_PID = d.intern_atom("_NET_WM_PID")
NET_ACTIVE = d.intern_atom("_NET_ACTIVE_WINDOW")


def wname(w):
    try:
        p = w.get_full_property(NET_NAME, 0)
        if p and p.value:
            return p.value.decode(errors="replace")
        n = w.get_wm_name()
        return n if isinstance(n, str) else (n.decode(errors="replace") if n else "")
    except Exception:
        return ""


def _abs_geom(w):
    g = w.get_geometry()
    t = root.translate_coords(w, 0, 0) if False else w.translate_coords(root, 0, 0)
    # translate_coords(root, 0, 0) gives root origin in w coords -> negate
    return (-t.x, -t.y, g.width, g.height)


def windows(depth_max=2):
    """Yield dicts for every viewable X window (depth<=2) with abs geometry."""
    out = []

    def walk(w, depth):
        try:
            kids = w.query_tree().children
        except Exception:
            return
        for k in kids:
            try:
                a = k.get_attributes()
                if a.map_state == X.IsViewable:
                    x, y, cw, ch = _abs_geom(k)
                    pid = k.get_full_property(NET_PID, 0)
                    cls = k.get_wm_class()
                    out.append({"win": k, "id": k.id, "name": wname(k), "x": x, "y": y,
                                "w": cw, "h": ch, "pid": pid.value[0] if pid else None,
                                "cls": cls, "override": bool(a.override_redirect)})
            except Exception:
                pass
            if depth < depth_max:
                walk(k, depth + 1)

    walk(root, 0)
    return out


def find(sub, cls_sub=None):
    for e in windows():
        if sub in e["name"] and (cls_sub is None or (e["cls"] and cls_sub in str(e["cls"]))):
            if e["cls"] and e["cls"][0] == "mutter-x11-frames":
                continue
            return e
    return None


def capture(win, path):
    """Save the window's current pixels (XGetImage on the window itself) as PNG."""
    from PIL import Image
    g = win.get_geometry()
    raw = win.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
    img = Image.frombytes("RGBX", (g.width, g.height), raw.data, "raw", "BGRX")
    img = img.convert("RGB")
    img.save(path)
    return img


def capture_root(path, x=None, y=None, w=None, h=None):
    from PIL import Image
    g = root.get_geometry()
    x = 0 if x is None else x
    y = 0 if y is None else y
    w = g.width - x if w is None else w
    h = g.height - y if h is None else h
    raw = root.get_image(x, y, w, h, X.ZPixmap, 0xFFFFFFFF)
    img = Image.frombytes("RGBX", (w, h), raw.data, "raw", "BGRX").convert("RGB")
    img.save(path)
    return img


def focus(win):
    ev = protocol.event.ClientMessage(window=win, client_type=NET_ACTIVE,
                                      data=(32, [2, X.CurrentTime, 0, 0, 0]))
    root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
    d.sync()
    time.sleep(0.3)
    win.set_input_focus(X.RevertToParent, X.CurrentTime)
    d.sync()
    time.sleep(0.2)


def move(x, y):
    xtest.fake_input(d, X.MotionNotify, x=x, y=y)
    d.sync()


def click(x, y, button=1, hold=0.05):
    move(x, y)
    time.sleep(0.05)
    xtest.fake_input(d, X.ButtonPress, button)
    d.sync()
    time.sleep(hold)
    xtest.fake_input(d, X.ButtonRelease, button)
    d.sync()


def drag(x1, y1, x2, y2, button=1, steps=12, dt=0.03):
    move(x1, y1)
    time.sleep(0.1)
    xtest.fake_input(d, X.ButtonPress, button)
    d.sync()
    time.sleep(0.1)
    for i in range(1, steps + 1):
        move(int(x1 + (x2 - x1) * i / steps), int(y1 + (y2 - y1) * i / steps))
        time.sleep(dt)
    time.sleep(0.1)
    xtest.fake_input(d, X.ButtonRelease, button)
    d.sync()


def key(name):
    kc = d.keysym_to_keycode(XK.string_to_keysym(name))
    xtest.fake_input(d, X.KeyPress, kc)
    xtest.fake_input(d, X.KeyRelease, kc)
    d.sync()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for e in windows():
            print(repr(e["name"]), f"{e['w']}x{e['h']}+{e['x']}+{e['y']}", "pid=%s" % e["pid"],
                  "cls=%s" % (e["cls"],), "ovr=%d" % e["override"], "id=0x%x" % e["id"])
    elif cmd == "cap":
        e = find(sys.argv[2])
        assert e, "window not found"
        capture(e["win"], sys.argv[3])
        print("saved", sys.argv[3], f"{e['w']}x{e['h']}+{e['x']}+{e['y']}")
    elif cmd == "caproot":
        capture_root(sys.argv[2])
        print("saved", sys.argv[2])
    elif cmd == "click":
        click(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "drag":
        drag(*map(int, sys.argv[2:6]))
    elif cmd == "focus":
        e = find(sys.argv[2])
        assert e, "window not found"
        focus(e["win"])
        print("focused", repr(e["name"]))
    elif cmd == "key":
        key(sys.argv[2])
    elif cmd == "capid":
        w = d.create_resource_object("window", int(sys.argv[2], 16))
        capture(w, sys.argv[3]); print("saved", sys.argv[3])
    elif cmd == "capall":
        # capture every viewable window of a pid except the REAPER main window
        pid_ = int(sys.argv[2]); prefix = sys.argv[3]; n = 0
        for e in windows():
            if e["pid"] == pid_ and "mutter" not in str(e["cls"]) and "REAPER v" not in e["name"]:
                n += 1; out = f"{prefix}{n}.png"; capture(e["win"], out)
                print(repr(e["name"]), f"{e['w']}x{e['h']}+{e['x']}+{e['y']}", "ovr=%d" % e["override"], "->", out)
