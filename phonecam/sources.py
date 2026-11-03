"""One junction's video, however it arrives.

Two ways in, behind one interface:

  push  a phone browser opens /cam/<id> and sends JPEG frames over a websocket.
        No app to install, and it uses the phone's real camera - which is a
        better sensor than any laptop webcam in the room.
  pull  anything OpenCV can open: an IP-Webcam app's MJPEG URL, an RTSP
        stream, or a video file. The fallback when a phone browser will not
        cooperate, and the way to rehearse the whole demo with no phones at
        all.

Both keep exactly one frame. A junction whose frames arrive faster than the
shared recogniser can read them must drop the old ones, not queue them -
queueing is how a live view ends up showing something that happened a minute
ago.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


class JunctionSource:
    """The newest frame from one junction, and how it has been behaving."""

    def __init__(self, camera_id: str, name: str = "", url: str | None = None):
        self.camera_id = camera_id
        self.name = name or camera_id
        self.url = url
        self.mode = "pull" if url else "push"

        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._arrivals: deque = deque(maxlen=40)

        self.received = 0
        self.dropped = 0
        self.decode_errors = 0
        self.last_seen = 0.0
        self.client = ""            # user agent, for the wall
        # Switched off means off: a pushed frame is dropped before it is
        # decoded, a pulled source releases its capture, and peek() reports
        # nothing - so the junction costs no CPU, no GPU and, for a webcam or
        # an RTSP camera, does not hold the device open either.
        self.enabled = True

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- push ------------------------------------------------------------
    def push_jpeg(self, blob: bytes) -> bool:
        """Store one JPEG from a phone. False if dropped or undecodable.

        The enabled check comes before imdecode on purpose: decoding a frame
        only to discard it is the cost this switch exists to avoid.
        """
        if not self.enabled:
            return False
        buf = np.frombuffer(blob, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.decode_errors += 1
            return False
        self._store(frame)
        return True

    def _store(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._frame is not None:
                self.dropped += 1
            self._frame, self._stamp = frame, now
            self.received += 1
            self._arrivals.append(now)
        self.last_seen = time.time()

    # --- pull ------------------------------------------------------------
    def start_pull(self) -> bool:
        if not self.url:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._pull_loop,
                                        name=f"pull-{self.camera_id}", daemon=True)
        self._thread.start()
        return True

    def _open_capture(self):
        """A capture for this source's URL, or None.

        `webcam:N` opens a local camera, which makes the laptop's own webcam
        just another junction rather than a second demo with its own service -
        one live-camera process for the dashboard to integrate with instead of
        two.
        """
        if self.url.startswith("webcam:"):
            tail = self.url.split(":", 1)[1].strip()
            index = int(tail) if tail.isdigit() else 0
            apis = ([cv2.CAP_MSMF, cv2.CAP_DSHOW, 0] if sys.platform == "win32"
                    else [0])
            for api in apis:
                cap = (cv2.VideoCapture(index, api) if api
                       else cv2.VideoCapture(index))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
                cap.release()
            return None
        return cv2.VideoCapture(self.url)

    def _pull_loop(self) -> None:
        while not self._stop.is_set():
            # Switched off: hold the device closed rather than reading and
            # discarding. A webcam or an RTSP camera is then genuinely free for
            # something else.
            if not self.enabled:
                if self._stop.wait(0.4):
                    return
                continue
            cap = self._open_capture()
            if cap is None or not cap.isOpened():
                print(f"[phonecam] {self.camera_id}: cannot open {self.url}",
                      file=sys.stderr)
                if cap is not None:
                    cap.release()
                if self._stop.wait(3.0):
                    return
                continue
            while not self._stop.is_set() and self.enabled:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break                      # stream ended; loop re-opens it
                self._store(frame)
            cap.release()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    # --- access ----------------------------------------------------------
    def take(self):
        """The newest frame, clearing the slot so it is read once."""
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def peek(self):
        """The newest frame without consuming it — for the wall's stream.

        Nothing while switched off, so the tile shows the off state rather than
        the last frame from before, which would read as a live camera.
        """
        if not self.enabled:
            return None
        with self._lock:
            return self._frame if self._frame is not None else self._last_shown

    _last_shown: np.ndarray | None = None

    def remember(self, frame: np.ndarray) -> None:
        """Hold the last frame the worker saw, so the wall keeps showing it
        between arrivals instead of flickering to nothing."""
        self._last_shown = frame

    @property
    def fps(self) -> float:
        # Zero while off, rather than the rate from before. A switched-off
        # junction still reporting 30 fps is the kind of stale number that
        # makes an operator doubt the switch worked.
        if not self.enabled:
            return 0.0
        with self._lock:
            stamps = list(self._arrivals)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    @property
    def online(self) -> bool:
        return bool(self.last_seen) and (time.time() - self.last_seen) < 5.0

    @property
    def silent_for(self) -> float:
        return (time.time() - self.last_seen) if self.last_seen else -1.0
