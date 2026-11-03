"""Tunables for the phone-as-junction-camera demo.

The reading behaviour itself is deliberately NOT redefined here - the floors,
the vote and the aim box come from livecam.settings, which was tuned by
measurement. A second copy of those numbers would drift from the first.
"""
from __future__ import annotations

import os
from pathlib import Path

from livecam.settings import _flag, _float, _int          # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
CERT_DIR = ROOT / "phonecam" / "certs"

# --- serving ------------------------------------------------------------
HOST = os.getenv("GODSEYE_PHONE_HOST", "0.0.0.0")
PORT = _int("GODSEYE_PHONE_PORT", 8504)

# getUserMedia is refused on any origin that is not a secure context, and a LAN
# address is not one - so a phone browser simply will not hand over its camera
# over plain http. A self-signed certificate covering this machine's LAN
# addresses is what makes the phone page work at all; the browser shows one
# warning that has to be accepted once per phone.
HTTPS = _flag("GODSEYE_PHONE_HTTPS", True)
CERT_DAYS = _int("GODSEYE_PHONE_CERT_DAYS", 30)

# A second listener, plain http, bound to loopback only.
#
# The phones need https or they get no camera. The Streamlit control room needs
# the opposite: it is served over http, and an <img> pointing at an https origin
# carrying a certificate the browser has never accepted is blocked or warned
# about on every load. Polling stills through Streamlit instead is a slideshow -
# measured at 57-149 ms per frame per junction, before Streamlit's own media
# round-trip.
#
# So the same app also answers on 127.0.0.1 over http, and the dashboard embeds
# the MJPEG stream directly. The browser then holds one connection per tile and
# frames arrive as they are produced. Loopback only, deliberately: this listener
# exists for a process on this machine, and nothing on the LAN needs it.
LOCAL_PORT = _int("GODSEYE_PHONE_LOCAL_PORT", PORT + 1)
LOCAL_HOST = "127.0.0.1"

# --- what the phone sends -----------------------------------------------
# Frames are capped client-side and gated on an ack, so a slow link cannot
# build a backlog. 8/s is far more than the vote needs to lock in well under a
# second, and keeps three phones inside a sensible bandwidth budget.
PHONE_FPS = _int("GODSEYE_PHONE_FPS", 8)
PHONE_WIDTH = _int("GODSEYE_PHONE_WIDTH", 960)
PHONE_JPEG_Q = _float("GODSEYE_PHONE_JPEG_Q", 0.72)

# How many frames may be unacknowledged at once.
#
# 1 is stop-and-wait, which is what this started as, and it caps the frame rate
# at 1/round-trip. On a LAN that is invisible - the round trip is a millisecond
# or two. Through a Cloudflare tunnel it is not: measured at ~1.4 s per round
# trip, stop-and-wait collapsed a phone to 0.4 fps, which is too slow for the
# vote to lock in reasonable time.
#
# A small window decouples the frame rate from the latency while still bounding
# how far behind a slow link may fall - the server keeps only the newest frame
# anyway, so the cost of a few in flight is bandwidth, not staleness.
PHONE_MAX_INFLIGHT = _int("GODSEYE_PHONE_INFLIGHT", 4)

# --- the operator wall --------------------------------------------------
WALL_FPS = _int("GODSEYE_PHONE_WALL_FPS", 10)
WALL_WIDTH = _int("GODSEYE_PHONE_WALL_WIDTH", 640)
WALL_JPEG_Q = _int("GODSEYE_PHONE_WALL_Q", 72)

# --- writing sightings --------------------------------------------------
# On by default here, unlike livecam. A junction camera that reads a plate and
# does not record it has not demonstrated anything the platform does; the whole
# point of several junctions is the trajectory between them.
ENROL = _flag("GODSEYE_PHONE_ENROL", True)
OPERATOR = os.getenv("GODSEYE_PHONE_OPERATOR", "phone-cam").strip()

# Timestamps. Plausible by default: core/enrol.suggest_ts() spaces the
# sightings by the real road distance at an ordinary city speed, so two phones
# on one table still produce a journey the platform reads as a journey.
#
# REALTIME stamps the actual clock instead, which for junctions kilometres
# apart implies an impossible speed - and is therefore the setting to use when
# the thing being demonstrated is the clone detector catching it.
REALTIME = _flag("GODSEYE_PHONE_REALTIME", False)

# Ignore a re-lock of the same plate at the same junction within this window,
# so a board left in view does not write a sighting every few seconds.
#
# Derived from core/enrol.py's own duplicate window rather than set
# independently, because the two disagreeing is a silent bug: at 25s against
# enrol's 90s, every re-lock between those two numbers passed this guard and
# was then refused by enrol as a double submission. The tile went green, the
# table stayed empty, and the reason only ever reached this process's stderr.
# Tying them together means this guard never proposes a write enrol will refuse.
def _relock_default() -> float:
    try:
        import config
        return float(config.ENROL_DUPLICATE_WINDOW_S)
    except Exception:
        return 90.0


RELOCK_S = _float("GODSEYE_PHONE_RELOCK_S", _relock_default())

# --- junctions ----------------------------------------------------------
# Esplanade -> Park Street -> Rashbehari, overridable with --junctions. Picked
# so one setup demonstrates both things:
#
#   CAM-20 <-> CAM-21   1.4 km, adjacent  - a clean journey along a real road
#   CAM-21 <-> CAM-28   4.7 km, adjacent  - clears config.CLONE_MIN_KM (3 km),
#                                           so --realtime makes the clone rule
#                                           fire on a board carried between them
#   CAM-20 <-> CAM-28   6.1 km, NOT adjacent - the route is inferred, and the
#                                           trajectory says so
DEFAULT_JUNCTIONS = os.getenv("GODSEYE_PHONE_JUNCTIONS", "CAM-20,CAM-21,CAM-28")
