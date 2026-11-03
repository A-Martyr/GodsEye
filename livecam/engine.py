"""The headless live engine: capture -> read -> vote, on its own thread.

The one architectural decision that makes this feel instant: **the display
never waits for the recogniser.** Rendering runs at camera speed and always
draws the newest frame with the most recent result available; inference runs
on its own thread at whatever rate it can manage. A read taking 90ms does not
make the video 11fps, it makes the *plate text* update at 11Hz over video that
is still smooth. Coupling the two is the single most common way a live vision
demo ends up looking broken.

Both front-ends (livecam/window.py, livecam/server.py) are thin layers over
this class and share every pixel of the overlay, so they cannot drift apart.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from livecam import settings                                     # noqa: E402
from livecam.capture import Webcam                               # noqa: E402
from livecam.reader import FrameResult, LiveReader               # noqa: E402
from livecam.vote import StreamVote, VoteState                   # noqa: E402


@dataclass
class LiveState:
    """Everything a front-end needs to draw one frame. Plain data, no locks."""
    vote: VoteState = field(default_factory=VoteState)
    boxes: list = field(default_factory=list)
    aim_box: tuple | None = None
    detect_ms: float = 0.0
    read_ms: float = 0.0
    infer_ms: float = 0.0
    infer_fps: float = 0.0
    camera_fps: float = 0.0
    crops_tried: int = 0
    backend: str = ""
    backend_note: str = ""
    # The last raw decode whatever its confidence, so the header can show that
    # the reader IS seeing something when nothing has cleared the floor.
    raw_text: str = ""
    raw_conf: float = 0.0
    frames_dropped: int = 0
    enrol_enabled: bool = False
    enrol_camera: str = ""
    last_enrol: str = ""
    paused: bool = False


class LiveEngine:
    def __init__(self, backend: str | None = None, camera_index: int | None = None,
                 enrol_camera: str | None = None):
        self.cam = Webcam(camera_index)
        self.reader = LiveReader(backend)
        self.vote = StreamVote(settings.VOTE_WINDOW_S, settings.VOTE_MIN_AGREE,
                               settings.VOTE_MIN_MARGIN, settings.LOCK_HOLD_S)
        self.enrol_camera = (enrol_camera or settings.ENROL_CAMERA or "").upper()
        self.enrol_enabled = bool(settings.ENROL_ENABLED and self.enrol_camera)
        self.last_enrol = ""
        self.paused = False
        self.error = ""

        self._state = LiveState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._infer_times: deque[float] = deque(maxlen=30)

    # --- lifecycle ------------------------------------------------------
    def start(self, warm: bool = True) -> bool:
        if warm:
            # ASCII only: a Windows console in a non-UTF-8 code page mangles
            # anything else, and a demo that starts with mojibake looks broken
            # before it has done anything.
            print("[livecam] warming the reader...", file=sys.stderr)
            self.reader.warm()
        if self.reader.status:
            print(f"[livecam] {self.reader.status}", file=sys.stderr)
        if not self.cam.open():
            self.error = self.cam.error
            return False
        print(f"[livecam] camera {self.cam.resolution[0]}x{self.cam.resolution[1]} "
              f"at {self.cam.measured_fps:.0f} fps", file=sys.stderr)
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, name="livecam-infer",
                                        daemon=True)
        self._worker.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self.cam.close()

    # --- the inference thread -------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            frame, _, _ = self.cam.take()
            if frame is None:
                time.sleep(0.003)
                continue
            if self.paused:
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()
            result = self.reader.read(frame)
            self.vote.add(result.read, accepted=result.accepted)
            self._infer_times.append(time.perf_counter())
            infer_ms = (time.perf_counter() - t0) * 1000.0

            self._maybe_enrol()
            self._publish(result, infer_ms)

    def _publish(self, result: FrameResult, infer_ms: float) -> None:
        state = LiveState(
            vote=self.vote.state(), boxes=result.boxes, aim_box=result.aim_box,
            detect_ms=result.detect_ms, read_ms=result.read_ms, infer_ms=infer_ms,
            infer_fps=_rate(self._infer_times), camera_fps=self.cam.fps,
            crops_tried=result.crops_tried, backend=self.reader.actual,
            backend_note=self.reader.status, frames_dropped=self.cam.dropped,
            enrol_enabled=self.enrol_enabled, enrol_camera=self.enrol_camera,
            last_enrol=self.last_enrol, paused=self.paused,
            raw_text=getattr(result.read, "text", "") or "",
            raw_conf=float(getattr(result.read, "confidence", 0.0) or 0.0))
        with self._lock:
            self._state = state

    # --- access for the front-ends --------------------------------------
    def snapshot(self):
        """(newest frame, latest state). The frame is fresh even mid-inference."""
        frame, _, _ = self.cam.latest()
        with self._lock:
            state = self._state
        # The vote decays on the wall clock, so ask it again at draw time -
        # otherwise a lock would appear to persist until the next inference
        # pass rather than for exactly LOCK_HOLD_S.
        state.vote = self.vote.state()
        state.last_enrol = self.last_enrol
        state.paused = self.paused
        return frame, state

    def reset_vote(self) -> None:
        self.vote.reset()

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    # --- enrolment (only when explicitly switched on) -------------------
    def _maybe_enrol(self) -> None:
        if not self.enrol_enabled:
            return
        read = self.vote.take_fresh_lock()
        if read is not None:
            self.enrol(read)

    def enrol(self, read=None) -> str:
        """Write one operator-attested sighting. Returns a message for the UI.

        Everything that could go wrong is core/enrol.py's business, not this
        module's: the confidence floor, the bound on correcting a misread, and
        the timestamp arithmetic that stops the platform accusing its own demo
        vehicle of being a cloned plate. Imported lazily so a read-only run
        never opens the platform's database at all.
        """
        if not self.enrol_camera:
            return "no junction chosen — pass --enrol CAM-XX"
        if read is None:
            read = self.vote.state().best_read
        if read is None:
            return "nothing locked to enrol"
        try:
            from core import enrol as enrol_mod
            out = enrol_mod.enrol(read, self.enrol_camera,
                                  operator=settings.ENROL_OPERATOR,
                                  note="read live from a board on the webcam demo")
        except ValueError as exc:
            self.last_enrol = f"refused: {exc}"
            return self.last_enrol
        except Exception as exc:
            self.last_enrol = f"error: {type(exc).__name__}: {exc}"
            return self.last_enrol
        alerts = out.get("alerts") or []
        note = f"enrolled {out['plate']} at {out['camera_name']}"
        if alerts:
            note += f" — raised {', '.join(a['kind'] for a in alerts)}"
        self.last_enrol = note
        return note


def _rate(times: deque) -> float:
    """Events per second over the samples held."""
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0
