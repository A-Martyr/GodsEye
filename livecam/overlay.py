"""Drawing the annotated frame. Shared by both front-ends so they cannot drift.

Ordering matters here: the frame is mirrored first and the box coordinates are
mirrored with it, then everything is drawn on top. Flipping a composed image
instead would mirror the text as well, which is the obvious implementation and
produces an unreadable registration.
"""
from __future__ import annotations

import cv2
import numpy as np

from livecam import settings

# BGR. Borrowed from the dashboard's palette so the demo looks like the rest
# of the platform on a projector.
INK = (248, 242, 237)
DIM = (180, 161, 147)
MUTED = (126, 107, 92)
GREEN = (67, 160, 46)
AMBER = (36, 165, 245)
RED = (77, 72, 229)
PANEL = (22, 18, 15)

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_DUPLEX


def _panel(img, x1, y1, x2, y2, alpha=0.72, colour=PANEL):
    """A translucent slab, clipped to the frame."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    slab = np.full_like(roi, colour, dtype=np.uint8)
    cv2.addWeighted(slab, alpha, roi, 1 - alpha, 0, roi)


def _brackets(img, box, colour, thick=3, run=0.18):
    """Corner brackets rather than a full rectangle — they frame the board
    without putting a line through the characters being read."""
    x, y, w, h = box
    L = max(12, int(min(w, h) * run))
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                             (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), colour, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), colour, thick, cv2.LINE_AA)


def _text(img, s, org, scale=0.6, colour=INK, thick=1, font=FONT):
    cv2.putText(img, s, org, font, scale, colour, thick, cv2.LINE_AA)


def _pretty(plate: str) -> str:
    try:
        from anpr.plates import pretty
        return pretty(plate)
    except Exception:
        return plate


def draw(frame: np.ndarray, state) -> np.ndarray:
    """The annotated frame, ready to show or encode."""
    if frame is None:
        return _placeholder()

    mirror = settings.MIRROR_PREVIEW
    img = cv2.flip(frame, 1) if mirror else frame.copy()
    H, W = img.shape[:2]

    def mx(box):
        """Box coordinates, mirrored to match the image if the preview is."""
        x, y, w, h = box
        return (W - x - w, y, w, h) if mirror else (x, y, w, h)

    v = state.vote
    locked = v.locked and bool(v.text)
    aim_colour = GREEN if locked else (AMBER if v.votes else MUTED)

    # regions the detector proposed
    for box in (state.boxes or []):
        x, y, w, h = mx(box)
        cv2.rectangle(img, (x, y), (x + w, y + h), AMBER, 2, cv2.LINE_AA)

    if state.aim_box is not None:
        _brackets(img, mx(state.aim_box), aim_colour, 3)

    _header(img, W, state)
    _result(img, W, H, state, locked)
    return img


def _header(img, W, state):
    """One status line: which reader, how fast, and any degradation."""
    _panel(img, 0, 0, W, 34, 0.62)
    backend = state.backend or "?"
    bits = [f"reader {backend}"]
    if state.infer_fps:
        bits.append(f"{state.infer_fps:4.1f} reads/s")
    if state.infer_ms:
        bits.append(f"{state.infer_ms:5.0f} ms")
    if state.camera_fps:
        bits.append(f"cam {state.camera_fps:4.1f} fps")
    _text(img, "   ·   ".join(bits), (12, 23), 0.52, DIM)

    if state.paused:
        _text(img, "PAUSED", (W - 90, 23), 0.6, AMBER, 2)

    # A reader that quietly degraded to the CRNN would read a real board at
    # about 1% and look like a broken demo, so it is said loudly.
    if state.backend_note:
        _panel(img, 0, 34, W, 62, 0.8, (40, 30, 90))
        _text(img, state.backend_note[:110], (12, 53), 0.46, (150, 200, 255))


def _result(img, W, H, state, locked):
    """The bottom slab: the registration, its confidence, and the vote."""
    v = state.vote
    top = H - 132
    _panel(img, 0, top, W, H, 0.78)

    if locked or v.text:
        colour = GREEN if locked else DIM
        _text(img, _pretty(v.text), (24, top + 62), 1.9, colour, 4, MONO)
        label = "CONFIRMED" if locked else "reading…"
        _text(img, label, (24, top + 92), 0.6, colour, 2)
        _text(img, f"{v.confidence:.0%} confidence   ·   {v.votes} frames agree",
              (24, top + 118), 0.55, DIM)
    else:
        _text(img, "hold the plate inside the brackets", (24, top + 62), 0.95,
              MUTED, 2)
        # Three genuinely different situations, and they need opposite fixes on
        # the day: a decode that fell under the floor, a good decode that has
        # simply aged out of the vote window, and nothing seen at all. Reporting
        # the first message for all three produced the contradiction "seeing
        # WB24AB1234 at 100% - under the 95% floor".
        if state.raw_text and state.raw_conf < settings.LIVE_MIN_CONFIDENCE:
            _text(img, f"seeing  {state.raw_text}  at {state.raw_conf:.0%} — under the "
                       f"{settings.LIVE_MIN_CONFIDENCE:.0%} floor",
                  (24, top + 100), 0.5, AMBER)
        elif state.raw_text:
            _text(img, f"last read  {state.raw_text}  at {state.raw_conf:.0%} — "
                       "waiting for a fresh frame", (24, top + 100), 0.5, MUTED)
        else:
            _text(img, "the reader is running — nothing has cleared the confidence "
                       "floor yet", (24, top + 100), 0.5, MUTED)

    _vote_bar(img, W, top, v, locked)

    if state.enrol_enabled:
        _text(img, f"enrolling at {state.enrol_camera}", (W - 300, top + 24),
              0.5, AMBER)
    if state.last_enrol:
        _text(img, state.last_enrol[:70], (W - 300, top + 48), 0.45, DIM)


def _vote_bar(img, W, top, v, locked):
    """Progress towards a lock — the cue that says 'hold it still a moment'."""
    x1, x2 = W - 300, W - 40
    y = top + 86
    cv2.rectangle(img, (x1, y), (x2, y + 10), (60, 52, 46), -1, cv2.LINE_AA)
    fill = int((x2 - x1) * max(0.0, min(1.0, v.progress)))
    if fill > 0:
        cv2.rectangle(img, (x1, y), (x1 + fill, y + 10),
                      GREEN if locked else AMBER, -1, cv2.LINE_AA)
    _text(img, "agreement", (x1, y - 8), 0.44, MUTED)

    # What the losing frames said. The disagreement is the evidence: a vote
    # nobody contested is a single read with extra steps.
    others = [f"{t} x{n}" for t, n in (v.runners or [])[1:3]]
    if others:
        _text(img, "also seen: " + "   ".join(others), (x1, y + 30), 0.42, MUTED)


def _placeholder(w: int = 960, h: int = 540) -> np.ndarray:
    img = np.full((h, w, 3), 18, dtype=np.uint8)
    _text(img, "waiting for the camera…", (40, h // 2), 0.9, MUTED, 2)
    return img
