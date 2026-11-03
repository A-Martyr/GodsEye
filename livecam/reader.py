"""One frame in, one plate read out — the lean path through the real engine.

Deliberately NOT ANPREngine.read_frame(). That function is tuned for a single
uploaded photograph, where one frame is the only chance there will ever be, and
two of its behaviours are actively wrong in a stream:

  * when nothing decodes it escalates, re-running the detector at 960, 1280 and
    1600px looking harder. In a video most frames contain no board at all, so
    the escalation would fire on nearly every one of them - paying the most
    expensive path the engine has, permanently, to confirm an empty room.
  * it sweeps up to twelve regions with a 20-second budget. Here the next frame
    is 33ms away, so a region not worth reading immediately is not worth
    reading.

Everything else is the platform's own code: the same detector, the same crop
variants, the same recogniser, the same per-backend confidence floor. Only the
search policy differs, and it lives here rather than in anpr/ so the tuned
still-photograph path keeps the behaviour it was measured with.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anpr import ocr as ocr_mod                                  # noqa: E402
from anpr.plates import STATE_CODES                              # noqa: E402
from livecam import settings                                     # noqa: E402


def looks_like_a_plate(text: str) -> bool:
    """Is this a registration, or is it text that happened to decode?

    The gate the confidence floor cannot be: a poster or a whiteboard produces
    a high-confidence string that is simply not plate-shaped.
    """
    if not text:
        return False
    if not settings.STRICT_GRAMMAR:
        return True
    if ocr_mod.is_bharat_series(text):        # 22BH1234AA carries no state code
        return True
    if not ocr_mod.matches_plate_grammar(text):
        return False
    return not settings.REQUIRE_STATE_CODE or text[:2] in STATE_CODES


def live_accepted(read) -> bool:
    """Would the live demo stand behind this read?

    The engine's own per-backend floor, plus this deployment's higher one. Kept
    separate from PlateRead.accepted so the platform's calibrated floor keeps
    exactly the meaning it was measured with - this is an operating point for
    a webcam in a room, not a correction to the engine.
    """
    return bool(read is not None and getattr(read, "accepted", False)
                and read.confidence >= settings.LIVE_MIN_CONFIDENCE
                and looks_like_a_plate(read.text))


@dataclass
class FrameResult:
    """What one inference pass produced."""
    read: object | None = None                  # best PlateRead, or None
    accepted: bool = False                      # cleared the live demo's floor
    boxes: list = field(default_factory=list)   # detector boxes, full-res coords
    aim_box: tuple | None = None
    detect_ms: float = 0.0
    read_ms: float = 0.0
    total_ms: float = 0.0
    crops_tried: int = 0


def aim_box_for(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """The centred rectangle the presenter holds the board inside."""
    w = int(frame_w * settings.AIM_WIDTH_FRAC)
    h = max(24, int(w / max(settings.AIM_ASPECT, 1.2)))
    if h > frame_h * 0.6:
        h = int(frame_h * 0.6)
        w = int(h * settings.AIM_ASPECT)
    return ((frame_w - w) // 2, (frame_h - h) // 2, w, h)


def _simple_variants(frame: np.ndarray, box) -> list:
    """Box crop plus a slightly tightened one.

    Used only when the frame was downscaled for detection, which invalidates
    anpr.ocr's OBB polygon side-channel (it is keyed by the box tuple from the
    most recent detect call, in that call's coordinate space).
    """
    x, y, w, h = box
    out = [frame[y:y + h, x:x + w]]
    ix, iy = int(0.06 * w), int(0.12 * h)
    if w - 2 * ix > 24 and h - 2 * iy > 10:
        out.append(frame[y + iy:y + h - iy, x + ix:x + w - ix])
    return [c for c in out if c is not None and c.size]


class LiveReader:
    """Holds the engine and reads frames with a stream-appropriate policy."""

    MAX_CROPS = 8            # bounds one batched forward pass

    def __init__(self, backend: str | None = None):
        self.backend = (backend or settings.BACKEND).strip().lower()
        self.engine = ocr_mod.engine(self.backend)
        # Empty when the requested backend is the one actually running;
        # otherwise a plain-language reason it is not. Surfaced in both UIs -
        # a reader that has silently degraded to the CRNN would read a real
        # board at roughly 1% and look like a broken demo.
        self.status = ocr_mod.engine_status(self.backend)
        self.actual = ("anpr_ocr" if self.engine.anpr_ocr is not None
                       else "crnn" if self.engine.crnn is not None else "classical")
        self._boxes_cache: list = []
        self._pass = 0

    def warm(self) -> float:
        """Pay every first-call cost before the window opens."""
        return ocr_mod.warm(self.backend, verbose=True)

    # --- the pass -------------------------------------------------------
    def read(self, frame: np.ndarray) -> FrameResult:
        """Two stages, cheapest first.

        Measured on a printed board through a simulated 720p webcam: reading
        the aim box alone is 29ms and gets the plate exactly right; running the
        detector and sweeping its crop variants is 162ms for the same answer.
        In the demo's normal case - a board deliberately held inside the
        brackets - the detector is 130ms spent confirming what one crop already
        knew, so it is only reached when that crop does not clear its floor.

        This is the same principle as read_frame()'s early exit, moved one
        level earlier: do not pay for a search once the answer is in hand.
        """
        started = time.perf_counter()
        h, w = frame.shape[:2]
        aim = aim_box_for(w, h) if (settings.AIM_BOX or settings.AIM_ONLY) else None

        # --- stage 1: the aim box on its own ---------------------------
        first = None
        read_ms = 0.0
        crops_tried = 0
        if aim is not None:
            ax, ay, aw, ah = aim
            crop = frame[ay:ay + ah, ax:ax + aw]
            if crop.size:
                first, read_ms = self._read_crops([crop])
                crops_tried = 1
                # Short-circuit only on a read the demo would actually stand
                # behind. A weak one falls through and lets the detector try.
                if live_accepted(first) or settings.AIM_ONLY:
                    return FrameResult(read=first, accepted=live_accepted(first),
                                       boxes=[], aim_box=aim, read_ms=read_ms,
                                       crops_tried=crops_tried,
                                       total_ms=(time.perf_counter() - started) * 1000.0)

        # --- stage 2: locate the plate and sweep its crops -------------
        boxes, detect_ms, downscaled = self._detect(frame)
        crops: list = []
        for box in boxes[:settings.MAX_BOXES]:
            if len(crops) >= self.MAX_CROPS:
                break
            variants = (_simple_variants(frame, box) if downscaled
                        else [c for c in ocr_mod._crop_variants(frame, box) if c.size])
            crops.extend(variants[:self.MAX_CROPS - len(crops)])

        second, ms2 = self._read_crops(crops)
        read = self._better(first, second)
        total = (time.perf_counter() - started) * 1000.0
        return FrameResult(read=read, accepted=live_accepted(read), boxes=boxes,
                           aim_box=aim, detect_ms=detect_ms,
                           read_ms=read_ms + ms2, total_ms=total,
                           crops_tried=crops_tried + len(crops))

    @staticmethod
    def _better(a, b):
        """Accepted beats refused; then higher confidence wins."""
        if a is None or not getattr(a, "text", ""):
            return b
        if b is None or not getattr(b, "text", ""):
            return a
        return b if (bool(b.accepted), b.confidence) > (bool(a.accepted), a.confidence) else a

    def _detect(self, frame: np.ndarray) -> tuple[list, float, bool]:
        """(boxes in frame coords, ms spent, whether the frame was downscaled)."""
        if settings.AIM_ONLY:
            return [], 0.0, False

        # Reuse the previous boxes on the frames in between. A held board moves
        # a few pixels between passes, and the aim box is read every pass
        # regardless, so this cannot lose the plate - only a box's exact edges.
        self._pass += 1
        if settings.DETECT_EVERY > 1 and self._pass % settings.DETECT_EVERY != 1:
            return list(self._boxes_cache), 0.0, False

        t0 = time.perf_counter()
        target = frame
        scale = 1.0
        downscaled = False
        if 0 < settings.DETECT_WIDTH < frame.shape[1]:
            scale = settings.DETECT_WIDTH / float(frame.shape[1])
            target = cv2.resize(frame, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            downscaled = True

        boxes = ocr_mod.detect_plate_candidates(
            target, max_candidates=settings.MAX_BOXES,
            scales=(settings.DETECT_IMGSZ,))

        if downscaled and boxes:
            inv = 1.0 / scale
            H, W = frame.shape[:2]
            boxes = [(max(0, int(x * inv)), max(0, int(y * inv)),
                      min(W, int(w * inv)), min(H, int(h * inv)))
                     for (x, y, w, h) in boxes]

        self._boxes_cache = list(boxes)
        return boxes, (time.perf_counter() - t0) * 1000.0, downscaled

    def _read_crops(self, crops: list) -> tuple[object | None, float]:
        """Decode every crop in one batch where the backend allows it."""
        if not crops:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            reads = self.engine._read_many(crops)
        except Exception as exc:
            # A backend that throws mid-stream must not take the demo down.
            # anpr_ocr is the one that gets here: paddle and torch clash over
            # compiled symbols, so it can load, answer, then fail.
            print(f"[livecam] read failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None, (time.perf_counter() - t0) * 1000.0
        ms = (time.perf_counter() - t0) * 1000.0

        # Accepted first, then confidence - the same order ANPREngine.
        # read_sequence() uses. A refused read at 0.99 is still refused.
        best = None
        for r in reads:
            if not r or not r.text or not looks_like_a_plate(r.text):
                continue
            if best is None:
                best = r
                continue
            if (bool(r.accepted), r.confidence) > (bool(best.accepted), best.confidence):
                best = r
        return best, ms
