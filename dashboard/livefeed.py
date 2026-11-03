"""The control room's view of the live cameras.

Two sources of truth, deliberately kept apart:

  the database   is where live reads actually live. A phone read is a
                 `sightings` row with source='enrolment', so the trajectory
                 engine, the alert rules and every analytics query already act
                 on it without knowing a phone was involved. That is the real
                 integration, and it works whether or not the live service is
                 running.

  phonecam       is only asked for the things a database cannot hold: is a
                 junction online right now, what is its frame rate, and what
                 does its camera see this second.

So a stopped live service degrades to "no video and no status", never to
"the live reads disappeared".

Video is streamed by the BROWSER, straight from phonecam's loopback http
listener, not polled through here. That distinction is the difference between a
live feed and a slideshow: a fetched still costs 57-149 ms per junction before
Streamlit's own media round-trip, so three tiles could never be smooth, whereas
an <img> pointed at MJPEG holds one connection and receives frames as they are
produced. The loopback listener exists because the phones' listener is https
with a self-signed certificate, which a browser will not embed quietly.

This module therefore fetches only JSON status - small, cheap, and the one
thing that genuinely has to come through Python.
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

import config
from core import db, trajectory

# The loopback, plain-http listener - never the https one the phones use.
PHONECAM_URL = os.getenv("GODSEYE_PHONECAM_URL", "http://127.0.0.1:8505").rstrip("/")


# --- the live service ---------------------------------------------------
@st.cache_data(ttl=1, show_spinner=False)
def status() -> dict | None:
    """phonecam's state, or None when it is not running."""
    try:
        r = requests.get(f"{PHONECAM_URL}/api/state", timeout=1.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def stream_url(camera_id: str) -> str:
    """The MJPEG URL for a junction, for the browser to open directly."""
    return f"{PHONECAM_URL}/stream/{camera_id}.mjpg"


@st.cache_data(ttl=1, show_spinner=False)
def snapshot(camera_id: str, width: int = 520) -> bytes | None:
    """One annotated frame. Kept for callers that need bytes in Python."""
    try:
        r = requests.get(f"{PHONECAM_URL}/snapshot/{camera_id}.jpg",
                         params={"width": width}, timeout=2.0)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


# --- controls -----------------------------------------------------------
def _post(path: str) -> dict | None:
    """Send a control to the live service and invalidate the cached status.

    Clearing the cache matters more than it looks: without it a switch flipped
    here would keep reporting its old position for up to a second, which reads
    as a button that did not work and invites a second press.
    """
    try:
        r = requests.post(f"{PHONECAM_URL}{path}", timeout=3.0)
        r.raise_for_status()
        out = r.json()
    except Exception:
        return None
    status.clear()
    return out


def set_camera(camera_id: str, on: bool) -> dict | None:
    """Switch one junction's camera on or off."""
    return _post(f"/api/cameras/{camera_id}/{'on' if on else 'off'}")


def set_all_cameras(on: bool) -> dict | None:
    """Switch every junction's camera on or off."""
    return _post(f"/api/cameras/{'on' if on else 'off'}")


def set_recording(on: bool) -> dict | None:
    """Whether confirmed reads are written into the platform's database."""
    return _post(f"/api/recording/{'on' if on else 'off'}")


def shutdown() -> dict | None:
    """Stop the live camera service entirely."""
    return _post("/api/shutdown")


@st.cache_data(ttl=2, show_spinner=False)
def connections() -> dict | None:
    """Every device that has reached the live service, with its address.

    The one thing that settles "a phone will not connect": if the phone is not
    in this list, its packets never arrived and no change here can help. If it
    is, the problem is on the phone - the certificate, or the camera permission.
    """
    try:
        r = requests.get(f"{PHONECAM_URL}/api/connections", timeout=1.5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def clear_live_records(confirm: bool = False) -> dict:
    """Remove what the live cameras have written. Dry run unless told twice.

    Modelled on core/privacy.retention_purge(), which this codebase already
    decided is the right shape for irreversible deletion: report first, delete
    only on a second, explicit instruction.

    Two guards, and the second is the one that matters:

      Only enrolment rows. The WHERE clause names sightings.source, so a
      camera's own capture cannot be caught by this however the caller asks.

      Alerts are removed only for a plate the cameras never saw themselves.
      Live reads share the sightings table with the simulator, so a plate the
      simulator also drives would otherwise have its genuine alert history
      deleted along with the demo's.
    """
    marks = ",".join("?" * len(config.ENROL_SOURCES))
    rows = db.rows(f"SELECT plate, camera_id, ts FROM sightings "
                   f"WHERE source IN ({marks})", config.ENROL_SOURCES)
    plates = sorted({r["plate"] for r in rows})

    doomed_alerts, kept_alerts = [], []
    for p in plates:
        others = db.rows("SELECT COUNT(*) n FROM sightings WHERE plate = ?"
                         f" AND source NOT IN ({marks})",
                         (p, *config.ENROL_SOURCES))[0]["n"]
        found = db.rows("SELECT id FROM alerts WHERE plate = ?", (p,))
        (kept_alerts if others else doomed_alerts).extend(a["id"] for a in found)

    preview = {
        "sightings": len(rows), "plates": plates,
        "alerts": len(doomed_alerts),
        "alerts_kept": len(kept_alerts),
        "cleared": False,
    }
    if not confirm or not rows:
        return preview

    conn = db.connect()
    conn.execute(f"DELETE FROM sightings WHERE source IN ({marks})",
                 config.ENROL_SOURCES)
    for aid in doomed_alerts:
        conn.execute("DELETE FROM alerts WHERE id = ?", (aid,))
    conn.commit()

    # The live service keeps its own in-memory history, so clearing only the
    # database would leave the junction wall still showing journeys whose
    # sightings no longer exist.
    _post("/api/records/clear")

    preview["cleared"] = True
    for fn in (live_sightings, live_plates, live_journeys, alerts_from_live):
        fn.clear()
    return preview


def live_camera_ids(state: dict | None) -> set[str]:
    """Junctions that have a live camera attached right now."""
    if not state:
        return set()
    return {j["camera_id"] for j in state.get("junctions", []) if j.get("online")}


def watched_camera_ids(state: dict | None) -> set[str]:
    """Junctions assigned to the live service, online or not."""
    if not state:
        return set()
    return {j["camera_id"] for j in state.get("junctions", [])}


# --- what the live cameras have put into the platform -------------------
def _window(hours: float) -> float:
    # Anchored to the newest sighting, not the wall clock, for the same reason
    # every other window in this dashboard is: a seeded store's "now" drifts,
    # and enrolments are stamped on that same network clock.
    return trajectory.lookback(hours)


@st.cache_data(ttl=4, show_spinner=False)
def live_sightings(hours: float = 24.0, limit: int = 250) -> list[dict]:
    """Sightings a camera operator - not a simulated node - put there."""
    marks = ",".join("?" * len(config.ENROL_SOURCES))
    return db.rows(
        f"SELECT * FROM sightings WHERE source IN ({marks}) AND ts >= ?"
        f" ORDER BY ts DESC LIMIT ?",
        (*config.ENROL_SOURCES, _window(hours), limit))


@st.cache_data(ttl=4, show_spinner=False)
def live_plates(hours: float = 24.0) -> tuple[str, ...]:
    """Registrations seen by a live camera in the window."""
    return tuple(dict.fromkeys(s["plate"] for s in live_sightings(hours)))


@st.cache_data(ttl=4, show_spinner=False)
def live_journeys(hours: float = 24.0) -> list[dict]:
    """Plates a live camera saw at two or more junctions — the payoff.

    'Sufficient data' for a trajectory is exactly this: one registration, two
    places, in order. Everything after it is core/trajectory.py's job.
    """
    by_plate: dict[str, list] = {}
    for s in reversed(live_sightings(hours)):          # oldest first
        by_plate.setdefault(s["plate"], []).append(s)
    out = []
    for plate, rows in by_plate.items():
        cams = list(dict.fromkeys(r["camera_id"] for r in rows))
        if len(cams) < 2:
            continue
        out.append({"plate": plate, "cameras": cams, "sightings": len(rows),
                    "first_ts": rows[0]["ts"], "last_ts": rows[-1]["ts"]})
    out.sort(key=lambda d: -d["last_ts"])
    return out


@st.cache_data(ttl=2, show_spinner=False)
def audible_alert_ids(hours: float = 6.0) -> set:
    """The alert ids the control room should SOUND for, and only those.

    Two sources, both provenance rather than guesswork:

      the live cameras   phonecam keeps the alert ids its own reads caused -
                         core/alerts.evaluate() runs inside core/enrol.py and
                         hands the row id straight back, so the live service
                         knows exactly what it is responsible for.

      the judge console  api/routes.py stamps each injection into the platform's
                         `meta` table, and any alert raised on that plate after
                         that moment is the injection being detected.

    Everything else - which is nearly everything, the simulator raises alerts
    continuously - is deliberately silent. Matching on the PLATE alone was the
    bug: a registration the simulator also drives would sound the alarm for the
    simulator's own alerts about it.
    """
    ids: set = set()
    since = time.time() - hours * 3600.0

    # Live-camera-caused alerts, straight from the database. core/alerts.py's
    # _emit() writes the alert with ts = the SIGHTING's ts, so plate + camera +
    # timestamp is an exact join back to the row that caused it, not a fuzzy
    # correlation. Doing it here rather than trusting phonecam's in-memory list
    # means it still works when the live service has been restarted, or when
    # the dashboard is opened after the fact.
    marks = ",".join("?" * len(config.ENROL_SOURCES))
    for r in db.rows(
            f"SELECT a.id FROM alerts a JOIN sightings s"
            f" ON s.plate = a.plate AND s.camera_id = a.camera_id"
            f" AND ABS(s.ts - a.ts) < 0.01"
            f" WHERE s.source IN ({marks}) AND a.ts >= ?",
            (*config.ENROL_SOURCES, since)):
        ids.add(int(r["id"]))

    # The live service's own record, when it is running. Redundant with the
    # join above and kept as a belt-and-braces for anything it cannot see.
    state_now = status()
    if state_now:
        ids.update(int(i) for i in (state_now.get("caused_alerts") or []))

    for mark in judge_injections():
        since = float(mark.get("ts", 0)) - 2.0        # clock skew headroom
        for a in db.rows(
                "SELECT id FROM alerts WHERE plate = ? AND ts >= ?"
                " ORDER BY ts DESC LIMIT 20", (mark.get("plate", ""), since)):
            ids.add(int(a["id"]))
    return ids


@st.cache_data(ttl=5, show_spinner=False)
def judge_injections(hours: float = 6.0) -> list[dict]:
    """What the judge console has released recently."""
    import json
    rows = db.rows("SELECT value FROM meta WHERE key = 'judge_injections'")
    if not rows:
        return []
    try:
        hist = json.loads(rows[0]["value"])
    except (ValueError, TypeError):
        return []
    if not isinstance(hist, list):
        return []
    cutoff = time.time() - hours * 3600.0
    return [h for h in hist if isinstance(h, dict) and h.get("ts", 0) >= cutoff]


@st.cache_data(ttl=4, show_spinner=False)
def alerts_from_live(hours: float = 24.0, limit: int = 60) -> list[dict]:
    """Alerts raised on a registration a live camera reported.

    The alert rules never knew a phone was involved - core/enrol.py runs the
    same core.alerts.evaluate() every ingest path runs. This only joins the two
    back together so the queue can say which alerts the demo just caused.
    """
    plates = live_plates(hours)
    if not plates:
        return []
    marks = ",".join("?" * len(plates))
    return db.rows(
        f"SELECT * FROM alerts WHERE plate IN ({marks}) AND ts >= ?"
        f" ORDER BY ts DESC LIMIT ?",
        (*plates, _window(hours), limit))


def start_command() -> str:
    return "phonecam.bat"
