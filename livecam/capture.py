"""Webcam capture that never lets the picture fall behind the world.

The failure this exists to prevent: cv2.VideoCapture buffers frames, so a
consumer slower than the camera does not skip - it reads progressively older
frames, and the lag grows without bound. A read taking 80ms against a 30fps
camera falls a second behind inside fifteen seconds, and the demo looks
broken in a way that has nothing to do with the recogniser.

So the grab loop runs on its own thread at the camera's pace and keeps
exactly one frame. Everything downstream gets the newest frame or none;
frames produced while the reader was busy are dropped on purpose, and counted
so the diagnostics can say how many.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque

import cv2

from livecam import settings


class Webcam:
    """A camera whose latest frame is always available and never stale."""

    def __init__(self, index: int | None = None):
        self.index = settings.CAMERA_INDEX if index is None else index
        self.cap: cv2.VideoCapture | None = None
        self._frame = None
        self._seq = 0
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.grabbed = 0
        self.measured_fps = 0.0
        self.error = ""
        self._grabs: deque = deque(maxlen=40)

    # --- lifecycle ------------------------------------------------------
    def open(self) -> bool:
        """Open the device and start grabbing. False (with .error set) on failure.

        The backend is chosen by measuring it, not by assuming. On this
        project's demo laptop the two Windows backends differ threefold at
        720p - DSHOW negotiates the resolution and then delivers 10fps, MSMF
        delivers 30 - and which one wins is a property of the driver, not
        something that can be predicted. Half a second of probing is cheap
        against a demo that looks like it is running at ten frames a second.
        """
        best = None                       # (fps, cap, frame)
        for api in self._backends():
            cap = cv2.VideoCapture(self.index, api) if api else cv2.VideoCapture(self.index)
            if not cap.isOpened():
                cap.release()
                continue
            self._configure(cap)
            fps, frame = self._probe(cap)
            if frame is None:
                cap.release()
                continue
            if fps >= self.GOOD_FPS:
                best = self._replace(best, (fps, cap, frame))
                break
            best = self._replace(best, (fps, cap, frame))

        if best is None:
            self.error = (f"could not open camera {self.index}. Close anything else "
                          "using the webcam (Teams, Zoom, the Camera app), or pass a "
                          "different index with --camera 1.")
            return False

        self.measured_fps, self.cap, frame = best
        with self._lock:
            self._frame, self._seq, self._stamp = frame, 1, time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="livecam-grab",
                                        daemon=True)
        self._thread.start()
        return True

    GOOD_FPS = 20.0          # stop probing once a backend clears this

    @staticmethod
    def _replace(best, candidate):
        """Keep the faster of two probed backends, releasing the loser."""
        if best is None:
            return candidate
        if candidate[0] > best[0]:
            best[1].release()
            return candidate
        candidate[1].release()
        return best

    @staticmethod
    def _probe(cap, frames: int = 12, timeout: float = 1.0):
        """(frames per second, one frame) for an already-configured capture."""
        for _ in range(3):                # let exposure and the pipeline settle
            cap.read()
        last = None
        n = 0
        t0 = time.perf_counter()
        while n < frames and time.perf_counter() - t0 < timeout:
            ok, frame = cap.read()
            if ok and frame is not None:
                last, n = frame, n + 1
        dt = time.perf_counter() - t0
        return (n / dt if dt > 0 else 0.0), last

    @staticmethod
    def _backends() -> list:
        # MSMF first on Windows: measured 30fps at 720p on the demo laptop
        # against DSHOW's 10. DSHOW stays as the fallback because it opens on
        # some devices MSMF refuses entirely.
        if sys.platform == "win32":
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]
        return [0]

    def _configure(self, cap: cv2.VideoCapture) -> None:
        if settings.CAMERA_MJPEG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
        # Ask the driver for a one-frame buffer too. Honoured by some backends
        # and ignored by others, which is why the grab thread above does not
        # rely on it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            with self._lock:
                if self._frame is not None:
                    self.dropped += 1
                self._frame, self._stamp = frame, now
                self._seq += 1
                self.grabbed += 1
                self._grabs.append(now)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # --- access ---------------------------------------------------------
    def latest(self):
        """(frame, seq, captured_at) — the newest frame, or (None, 0, 0.0)."""
        with self._lock:
            if self._frame is None:
                return None, 0, 0.0
            return self._frame, self._seq, self._stamp

    def take(self):
        """latest(), but clears the slot so the same frame is not read twice."""
        with self._lock:
            if self._frame is None:
                return None, 0, 0.0
            frame, seq, stamp = self._frame, self._seq, self._stamp
            self._frame = None
            return frame, seq, stamp

    @property
    def fps(self) -> float:
        """Frames the camera is actually producing, not the number consumed.

        Measured here rather than downstream on purpose: the consumer's rate is
        the recogniser's, and reporting that as the camera's hides which of the
        two is the bottleneck - the one thing this number is for.
        """
        with self._lock:
            stamps = list(self._grabs)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cap is None:
            return (0, 0)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
