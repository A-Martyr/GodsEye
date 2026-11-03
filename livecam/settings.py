"""Tunables for the live demo, in their own GODSEYE_LIVE_* namespace.

Deliberately a separate file from the platform's config.py. Every knob the
live path wants to move - detector input size, how many boxes are worth
reading, how long a vote lasts - is one the still-photograph path was tuned
against, and sharing them would mean tuning the demo retunes the benchmark.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


# --- camera -------------------------------------------------------------
CAMERA_INDEX = _int("GODSEYE_LIVE_CAM", 0)
FRAME_WIDTH = _int("GODSEYE_LIVE_W", 1280)
FRAME_HEIGHT = _int("GODSEYE_LIVE_H", 720)
CAMERA_FPS = _int("GODSEYE_LIVE_FPS", 30)

# MJPEG off the webcam rather than raw YUY2. A USB 2.0 webcam cannot deliver
# 1280x720 uncompressed at 30fps - it negotiates down to 5-10fps instead, and
# the resulting stutter reads as "the reader is slow" when the reader has not
# been reached yet.
CAMERA_MJPEG = _flag("GODSEYE_LIVE_MJPEG", True)

# Mirror the preview. OFF by default, which is the opposite of the usual
# webcam default and deliberate: mirroring helps someone aim their own face,
# but here the subject IS text. Mirrored, the board reads backwards on the
# projector while the recognised plate beside it reads forwards, and an
# audience cannot check one against the other - which is the entire point of
# showing the video at all. The frame handed to the reader is never mirrored
# either way.
MIRROR_PREVIEW = _flag("GODSEYE_LIVE_MIRROR", False)


# --- recogniser ---------------------------------------------------------
# anpr_ocr is the only backend with real-world accuracy (91.0% against the
# CRNN's 0.9% on real photographs), and a board in front of a webcam is a real
# photograph. Without paddle in this interpreter anpr.ocr falls back to the
# CRNN and says so - which the UI surfaces rather than hides.
BACKEND = os.getenv("GODSEYE_LIVE_BACKEND", "anpr_ocr").strip().lower()

# --- detection ----------------------------------------------------------
# imgsz is the real cost knob: ultralytics letterboxes whatever it is given to
# imgsz before the forward pass, so handing it a smaller frame saves one cheap
# resize and nothing else - while costing the OBB rectified crop, whose
# polygon side-channel is keyed to the coordinates the detector actually ran
# in. Off by default for that reason; set it only for a very high-resolution
# camera, where the resize itself starts to matter.
DETECT_WIDTH = _int("GODSEYE_LIVE_DETECT_W", 0)
DETECT_IMGSZ = _int("GODSEYE_LIVE_IMGSZ", 640)

# Only the best few regions are worth reading in a stream. read_frame() sweeps
# up to twelve because a still photograph gets one chance; here the next frame
# is 30ms away.
MAX_BOXES = _int("GODSEYE_LIVE_MAX_BOXES", 2)

# Detect every Nth inference pass and reuse the previous boxes in between. 1
# disables it. A held board barely moves, so 2 nearly halves detector cost.
DETECT_EVERY = max(1, _int("GODSEYE_LIVE_DETECT_EVERY", 1))


# --- the aim box --------------------------------------------------------
# A drawn rectangle the presenter holds the board inside. Its crop is read
# alongside whatever the detector found, as one more entry in the same batch,
# so a board the detector misses is still read. This is the reliability
# backstop for a hand-held board, which is not what the detector was
# trained on.
AIM_BOX = _flag("GODSEYE_LIVE_AIM", True)
AIM_WIDTH_FRAC = _float("GODSEYE_LIVE_AIM_W", 0.58)
AIM_ASPECT = _float("GODSEYE_LIVE_AIM_ASPECT", 4.5)     # plate proportions

# Skip the detector entirely and read only the aim box. The lowest-latency
# mode there is, and the one to fall back on if YOLO proves unreliable on a
# hand-held board on the day.
AIM_ONLY = _flag("GODSEYE_LIVE_AIM_ONLY", False)


# --- what counts as a plate at all --------------------------------------
# The engine deliberately returns an ungrammatical read when nothing legal is
# on the beam, so an unusual real-world format degrades instead of vanishing.
# That is right for the platform and wrong here: a webcam pointed at a room
# sees posters, screens and whiteboards, and measured on this repo the line
# "Q3 ROADMAP 2026" decoded to R04DMAP202 at 0.98 - comfortably above the 0.80
# floor. The demo's board is printed in Bharat format by livecam/makeboard.py,
# so requiring that format costs nothing and removes the whole class.
STRICT_GRAMMAR = _flag("GODSEYE_LIVE_STRICT", True)
REQUIRE_STATE_CODE = _flag("GODSEYE_LIVE_STATE_CODE", True)

# An extra floor on top of the backend's own, for this deployment only. Chosen
# by measurement, not taste: swept over size, blur, angle, brightness, sensor
# noise and JPEG quality, every *correct* read of a printed board scored 1.00,
# while the room-text decode scored 0.88 and the two genuine misreads 0.88 and
# 0.65. Anywhere in (0.88, 1.00] separates them; 0.95 leaves margin on both
# sides. It also filters near-miss misreads, not only room text.
#
# If reads are being refused on the day, this is the first knob to lower -
# real optics are harsher than the simulated degradations it was fitted on.
LIVE_MIN_CONFIDENCE = _float("GODSEYE_LIVE_MIN_CONF", 0.95)


# --- temporal vote ------------------------------------------------------
# The same idea as the platform's burst fusion, applied to a stream: frames
# fail independently, so agreement across them is evidence no single frame can
# provide. In a video the extra frames are free.
VOTE_WINDOW_S = _float("GODSEYE_LIVE_VOTE_WINDOW", 1.6)
VOTE_MIN_AGREE = _int("GODSEYE_LIVE_VOTE_AGREE", 3)
VOTE_MIN_MARGIN = _float("GODSEYE_LIVE_VOTE_MARGIN", 1.15)

# How long a locked read stays on screen after the board is taken away, so an
# audience has time to read it.
LOCK_HOLD_S = _float("GODSEYE_LIVE_LOCK_HOLD", 3.5)


# --- enrolment (off unless asked for) -----------------------------------
# When on, a locked read is written to data/godseye.db as an operator-attested
# sighting through core/enrol.py - the module built for exactly this, "an
# operator standing in for a camera that is not there".
ENROL_ENABLED = _flag("GODSEYE_LIVE_ENROL", False)
ENROL_CAMERA = os.getenv("GODSEYE_LIVE_ENROL_CAM", "").strip().upper()
ENROL_OPERATOR = os.getenv("GODSEYE_LIVE_ENROL_BY", "live-demo").strip()


# --- web front-end ------------------------------------------------------
WEB_HOST = os.getenv("GODSEYE_LIVE_HOST", "0.0.0.0")
WEB_PORT = _int("GODSEYE_LIVE_PORT", 8503)
JPEG_QUALITY = _int("GODSEYE_LIVE_JPEG_Q", 80)
STREAM_FPS = _int("GODSEYE_LIVE_STREAM_FPS", 25)
