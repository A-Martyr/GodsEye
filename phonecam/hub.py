"""Several junctions, one recogniser.

There is one GPU and one paddle context on this machine, so there is exactly
one inference thread. It cycles the junctions, reading whichever has a frame
waiting. That shares the card fairly without any locking, bounds the total load
however many phones connect, and means a junction nobody is pointing at costs
nothing.

Each junction keeps its own reader and its own vote, because a vote is a claim
about one camera's view. What they share is the model underneath - anpr.ocr
caches its engine per backend, so N readers are N thin wrappers over one set of
weights.

The payoff this file exists for is at the bottom: when the same registration
locks at a second junction, that is a leg, and core/trajectory.py can
reconstruct the journey. Reading a plate is a feature; reconstructing where the
vehicle went is the product.
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

from livecam import settings as live_settings                  # noqa: E402
from livecam.reader import LiveReader                          # noqa: E402
from livecam.vote import StreamVote                            # noqa: E402
from phonecam import settings                                  # noqa: E402
from phonecam.sources import JunctionSource                    # noqa: E402


@dataclass
class Junction:
    """One camera site: its video, its reader, its vote and its last result."""
    camera_id: str
    name: str
    source: JunctionSource
    reader: LiveReader
    vote: StreamVote
    sector: str = ""
    boxes: list = field(default_factory=list)
    aim_box: tuple | None = None
    infer_ms: float = 0.0
    reads: int = 0
    locks: int = 0
    last_plate: str = ""
    last_lock_ts: float = 0.0
    raw_text: str = ""
    raw_conf: float = 0.0
    # What happened to the LAST lock: recorded | refused | not_recording | error.
    # Separate from the vote, because locking and recording are separate events
    # and only one of them was ever shown.
    last_outcome: str = ""
    last_outcome_note: str = ""
    last_outcome_ts: float = 0.0

    def set_outcome(self, outcome: str, note: str = "") -> None:
        self.last_outcome = outcome
        self.last_outcome_note = note
        self.last_outcome_ts = time.time()


class JunctionHub:
    def __init__(self, assignments: list[tuple[str, str | None]],
                 backend: str | None = None):
        from core.network import network

        self.net = network()
        self.backend = backend or live_settings.BACKEND
        self.junctions: dict[str, Junction] = {}
        self.events: deque = deque(maxlen=120)
        self.seen: dict[str, list[tuple[float, str]]] = {}   # plate -> [(ts, cam)]
        # Alert row ids this run's live reads actually caused. The control room
        # sounds its alarm for these and nothing else.
        self.caused_alerts: list[int] = []
        self._enrolled_at: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.enrol_enabled = settings.ENROL
        self.realtime = settings.REALTIME

        for camera_id, url in assignments:
            cam = self.net.cameras.get(camera_id)
            if cam is None:
                raise SystemExit(f"{camera_id} is not a camera in data/cameras.json")
            src = JunctionSource(camera_id, cam["name"], url)
            self.junctions[camera_id] = Junction(
                camera_id=camera_id, name=cam["name"], sector=cam.get("sector") or "",
                source=src, reader=LiveReader(self.backend),
                vote=StreamVote(live_settings.VOTE_WINDOW_S,
                                live_settings.VOTE_MIN_AGREE,
                                live_settings.VOTE_MIN_MARGIN,
                                live_settings.LOCK_HOLD_S))

    # --- lifecycle ------------------------------------------------------
    def start(self, warm: bool = True) -> None:
        if warm:
            print("[phonecam] warming the reader...", file=sys.stderr)
            any_reader = next(iter(self.junctions.values())).reader
            any_reader.warm()
            if any_reader.status:
                print(f"[phonecam] {any_reader.status}", file=sys.stderr)
        for j in self.junctions.values():
            if j.source.mode == "pull":
                j.source.start_pull()
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, name="phonecam-infer",
                                        daemon=True)
        self._worker.start()
        self.log("system", "", f"watching {len(self.junctions)} junctions: "
                 + ", ".join(f"{j.camera_id} {j.name}" for j in self.junctions.values()))

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        for j in self.junctions.values():
            j.source.stop()

    # --- the shared inference thread ------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            worked = False
            for j in list(self.junctions.values()):
                if self._stop.is_set():
                    break
                if not j.source.enabled:
                    continue          # no inference, so no share of the GPU
                frame = j.source.take()
                if frame is None:
                    continue
                worked = True
                j.source.remember(frame)
                t0 = time.perf_counter()
                result = j.reader.read(frame)
                j.infer_ms = (time.perf_counter() - t0) * 1000.0
                j.reads += 1
                j.boxes = result.boxes
                j.aim_box = result.aim_box
                j.raw_text = getattr(result.read, "text", "") or ""
                j.raw_conf = float(getattr(result.read, "confidence", 0.0) or 0.0)
                j.vote.add(result.read, accepted=result.accepted)
                locked = j.vote.take_fresh_lock()
                if locked is not None:
                    self._on_lock(j, locked)
            if not worked:
                time.sleep(0.004)

    # --- a registration has settled at a junction -----------------------
    def _on_lock(self, j: Junction, read) -> None:
        plate = read.text
        now = time.time()
        key = (j.camera_id, plate)
        # A board left in view relocks every few seconds. One pass through a
        # junction is one sighting.
        if now - self._enrolled_at.get(key, 0.0) < settings.RELOCK_S:
            return
        self._enrolled_at[key] = now

        j.locks += 1
        j.last_plate = plate
        j.last_lock_ts = now
        self.log("read", j.camera_id,
                 f"{_pretty(plate)} at {j.name} ({read.confidence:.0%})", plate)

        # The journey is announced against the timestamp actually stored, not
        # against the wall clock. They differ on purpose - enrol.suggest_ts()
        # spaces sightings by the real road distance - and reporting the wall
        # clock here would contradict the trajectory panel directly below it.
        stored_ts = now
        if not self.enrol_enabled:
            # Locking and recording are different events, and the tile used to
            # show only the first. A green CONFIRMED while recording is off
            # tells an operator the read was stored when it was not.
            j.set_outcome("not_recording", "read only - recording is off")
        else:
            out = self._enrol(j, read, plate)
            if out is not None:
                stored_ts = float(out.get("ts", now))
        with self._lock:
            self.seen.setdefault(plate, []).append((stored_ts, j.camera_id))
        self._announce_journey(plate)

    def _enrol(self, j: Junction, read, plate: str) -> dict | None:
        """Write the sighting through the platform's own enrolment path."""
        try:
            from core import enrol as enrol_mod
        except Exception as exc:
            self.log("error", j.camera_id, f"enrolment unavailable: {exc}")
            j.set_outcome("error", f"enrolment unavailable: {exc}")
            return None
        # Realtime stamps the clock, which between junctions kilometres apart
        # implies an impossible speed - so it has to be allowed through
        # explicitly, and it is exactly what makes the clone rule fire.
        ts = time.time() if self.realtime else None
        try:
            out = enrol_mod.enrol(read, j.camera_id, ts=ts,
                                  operator=settings.OPERATOR,
                                  note=f"read live from a phone at {j.name}",
                                  allow_implausible=self.realtime,
                                  net=self.net)
        except ValueError as exc:
            # A refusal is core/enrol.py declining to fabricate, and it is the
            # single most confusing thing this demo can do silently: the tile
            # goes green, nothing is stored, and the reason only ever reached
            # this process's stderr. It is now carried to the tile.
            self.log("refused", j.camera_id, f"{_pretty(plate)}: {exc}", plate)
            j.set_outcome("refused", str(exc))
            return None
        except Exception as exc:
            self.log("error", j.camera_id, f"{type(exc).__name__}: {exc}", plate)
            j.set_outcome("error", f"{type(exc).__name__}: {exc}")
            return None

        self.log("enrol", j.camera_id,
                 f"recorded {_pretty(plate)} at {out['camera_name']}", plate)
        j.set_outcome("recorded", f"stored at {out['camera_name']}")

        # The alert ids this read actually caused. enrol() runs the same
        # core.alerts.evaluate() every ingest path runs, and _emit() hands back
        # the row id - so the live camera knows EXACTLY which alerts it is
        # responsible for. That is the difference between sounding for a
        # watchlisted car entering this feed and sounding for every alert the
        # simulator happens to raise on a plate this feed once saw.
        for a in (out.get("alerts") or []):
            self.log("alert", j.camera_id,
                     f"{a['kind'].upper()} - {a['message']}", plate)
            aid = a.get("id")
            if aid is not None:
                with self._lock:
                    self.caused_alerts.append(int(aid))
                    del self.caused_alerts[:-200]
        return out

    def _announce_journey(self, plate: str) -> None:
        """Say so the moment a plate has been seen at more than one junction."""
        with self._lock:
            hits = self.seen.get(plate, [])
        cams = list(dict.fromkeys(c for _ts, c in hits))
        if len(cams) < 2:
            return
        a, b = cams[-2], cams[-1]
        km = self.net.route_km(a, b)
        gap = hits[-1][0] - next(ts for ts, c in reversed(hits[:-1]) if c == a)
        speed = (km / (gap / 3600.0)) if gap > 0 else 0.0
        self.log("journey", b,
                 f"{_pretty(plate)} travelled {self.net.name(a)} -> "
                 f"{self.net.name(b)}, {km:.1f} km in {gap/60:.1f} min "
                 f"({speed:.0f} km/h implied)", plate)

    # --- switching cameras on and off ------------------------------------
    def set_camera(self, camera_id: str, enabled: bool) -> bool:
        """Switch one junction on or off. Returns the new state.

        Off is off at the source: pushed frames are dropped before they are
        decoded and a pulled camera is released. The vote is cleared too, so a
        lock from before does not sit on the tile looking current.
        """
        j = self.junctions.get(camera_id.upper())
        if j is None:
            raise KeyError(camera_id)
        if j.source.enabled == enabled:
            return enabled
        j.source.enabled = enabled
        if not enabled:
            j.vote.reset()
            j.raw_text, j.raw_conf = "", 0.0
            j.boxes = []
        self.log("system", j.camera_id,
                 f"{j.name}: camera switched {'on' if enabled else 'off'}")
        return enabled

    def set_all_cameras(self, enabled: bool) -> int:
        """Switch every junction at once. Returns how many changed."""
        changed = 0
        for cid, j in self.junctions.items():
            if j.source.enabled != enabled:
                self.set_camera(cid, enabled)
                changed += 1
        return changed

    def set_recording(self, enabled: bool) -> bool:
        """Whether a confirmed read is written into the platform.

        Separate from switching a camera off, because they answer different
        questions: off stops the platform seeing, recording off lets it keep
        reading and showing while writing nothing to the database.
        """
        if self.enrol_enabled != enabled:
            self.enrol_enabled = enabled
            self.log("system", "",
                     f"recording {'on - reads become sightings' if enabled else 'off - nothing is written'}")
        return self.enrol_enabled

    def forget_history(self) -> dict:
        """Drop what this run remembers: journeys, events, per-junction reads.

        In-memory only. The sightings themselves live in the platform's
        database and are the control room's to remove - this just stops the
        wall showing journeys whose rows have gone, which would otherwise read
        as the clear having failed.
        """
        with self._lock:
            journeys = len([p for p, hits in self.seen.items()
                            if len({c for _t, c in hits}) >= 2])
            self.seen.clear()
            self.events.clear()
            self.caused_alerts.clear()
        self._enrolled_at.clear()
        for j in self.junctions.values():
            j.vote.reset()
            j.locks = 0
            j.last_plate = ""
            j.last_lock_ts = 0.0
            j.raw_text, j.raw_conf = "", 0.0
            j.set_outcome("", "")
        self.log("system", "", "history cleared - journeys and recent reads forgotten")
        return {"journeys_forgotten": journeys}

    # --- reporting -------------------------------------------------------
    def log(self, kind: str, camera_id: str, message: str, plate: str = "") -> None:
        entry = {"ts": time.time(), "kind": kind, "camera_id": camera_id,
                 "plate": plate, "message": message}
        with self._lock:
            self.events.appendleft(entry)
        print(f"[phonecam] {kind}: {message}", file=sys.stderr)

    def state(self) -> dict:
        out = []
        for j in self.junctions.values():
            v = j.vote.state()
            out.append({
                "camera_id": j.camera_id, "name": j.name, "sector": j.sector,
                "mode": j.source.mode, "enabled": j.source.enabled,
                "online": j.source.enabled and j.source.online,
                "silent_for": round(j.source.silent_for, 1),
                "fps": round(j.source.fps, 1), "frames": j.source.received,
                "dropped": j.source.dropped, "reads": j.reads, "locks": j.locks,
                "infer_ms": round(j.infer_ms, 1),
                "plate": v.text, "pretty": _pretty(v.text) if v.text else "",
                "locked": v.locked, "confidence": round(v.confidence, 4),
                "votes": v.votes, "progress": round(v.progress, 3),
                "raw_text": j.raw_text, "raw_conf": round(j.raw_conf, 3),
                "last_plate": j.last_plate,
                "last_outcome": j.last_outcome,
                "last_outcome_note": j.last_outcome_note,
                "last_outcome_ts": j.last_outcome_ts,
            })
        with self._lock:
            events = list(self.events)[:40]
            journeys = [{"plate": p, "pretty": _pretty(p),
                         "cameras": list(dict.fromkeys(c for _t, c in hits)),
                         "hits": len(hits)}
                        for p, hits in self.seen.items()
                        if len({c for _t, c in hits}) >= 2]
        with self._lock:
            caused = list(self.caused_alerts)
        return {"junctions": out, "events": events, "journeys": journeys,
                "caused_alerts": caused,
                "backend": next(iter(self.junctions.values())).reader.actual,
                "enrol": self.enrol_enabled, "realtime": self.realtime}

    def trajectory(self, plate: str) -> dict:
        """The platform's own reconstruction, for the wall's journey panel."""
        from core import trajectory as traj_mod

        t = traj_mod.reconstruct(plate, since=traj_mod.lookback(24.0))
        if not t.sightings:
            return {"known": False, "plate": plate}
        return {
            "known": True, "plate": plate, "pretty": _pretty(plate),
            "summary": t.summary,
            "legs": [{"from": l.from_name, "to": l.to_name,
                      "minutes": round(l.minutes, 1), "km": round(l.road_km, 2),
                      "kmph": round(l.implied_kmph, 1), "direction": l.direction,
                      "plausible": l.plausible, "note": l.note} for l in t.legs],
        }


def _pretty(plate: str) -> str:
    if not plate:
        return ""
    try:
        from anpr.plates import pretty
        return pretty(plate)
    except Exception:
        return plate
