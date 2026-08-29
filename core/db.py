"""SQLite store for sightings, cameras, watchlist and alerts.

SQLite in WAL mode is enough for a city-scale PoC: the ingest process writes
while the API and dashboard read, from separate processes, without locking each
other out. The schema and every query below are plain SQL, so the same code
points at PostgreSQL/TimescaleDB in deployment by swapping the connection.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    lat      REAL NOT NULL,
    lon      REAL NOT NULL,
    sector   TEXT,
    road     TEXT,
    lanes    INTEGER,
    type     TEXT
);

CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    camera_id   TEXT NOT NULL,
    plate       TEXT NOT NULL,          -- what the OCR engine read
    confidence  REAL NOT NULL,
    speed_kmph  REAL,
    lane        INTEGER,
    heading     REAL,                   -- compass bearing of travel
    direction   TEXT,                   -- N / NE / E ... derived from heading
    next_camera TEXT,                   -- camera the vehicle is heading towards
    vehicle_class TEXT,
    condition   TEXT,                   -- capture condition (simulator only)
    ocr_variant TEXT,
    true_plate  TEXT,                   -- ground truth, simulator only; NULL live
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);
CREATE INDEX IF NOT EXISTS idx_sight_plate ON sightings(plate, ts);
CREATE INDEX IF NOT EXISTS idx_sight_ts    ON sightings(ts);
CREATE INDEX IF NOT EXISTS idx_sight_cam   ON sightings(camera_id, ts);

CREATE TABLE IF NOT EXISTS watchlist (
    plate     TEXT PRIMARY KEY,
    reason    TEXT NOT NULL,
    severity  TEXT NOT NULL DEFAULT 'high',
    added_ts  REAL NOT NULL,
    added_by  TEXT DEFAULT 'control-room'
);

CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    kind      TEXT NOT NULL,
    severity  TEXT NOT NULL,
    plate     TEXT,
    camera_id TEXT,
    message   TEXT NOT NULL,
    detail    TEXT,                     -- JSON blob
    acked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(ts);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class Sighting:
    ts: float
    camera_id: str
    plate: str
    confidence: float
    speed_kmph: float | None = None
    lane: int | None = None
    heading: float | None = None
    direction: str | None = None
    next_camera: str | None = None
    vehicle_class: str | None = None
    condition: str | None = None
    ocr_variant: str | None = None
    true_plate: str | None = None
    id: int | None = field(default=None)

    def as_dict(self) -> dict:
        return asdict(self)


def connect(path: Path | None = None) -> sqlite3.Connection:
    """One connection per thread, WAL enabled."""
    path = Path(path or config.DB_PATH)
    key = f"conn::{path}"
    conn = getattr(_local, key, None)
    if conn is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=8000")
        setattr(_local, key, conn)
    return conn


def init(path: Path | None = None, cameras: list[dict] | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    if cameras:
        conn.executemany(
            "INSERT OR REPLACE INTO cameras (id, name, lat, lon, sector, road, lanes, type)"
            " VALUES (:id, :name, :lat, :lon, :sector, :road, :lanes, :type)", cameras)
    conn.commit()
    return conn


def reset(path: Path | None = None, keep_watchlist: bool = False) -> None:
    """Drop all observation data. The watchlist goes too unless asked otherwise:
    after a re-seed its plates refer to vehicles that no longer exist."""
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM sightings")
    conn.execute("DELETE FROM alerts")
    if not keep_watchlist:
        conn.execute("DELETE FROM watchlist")
    conn.commit()


# --- writes -------------------------------------------------------------
_INSERT = (
    "INSERT INTO sightings (ts, camera_id, plate, confidence, speed_kmph, lane, heading,"
    " direction, next_camera, vehicle_class, condition, ocr_variant, true_plate)"
    " VALUES (:ts, :camera_id, :plate, :confidence, :speed_kmph, :lane, :heading,"
    " :direction, :next_camera, :vehicle_class, :condition, :ocr_variant, :true_plate)"
)


def add_sighting(s: Sighting, conn: sqlite3.Connection | None = None) -> int:
    conn = conn or connect()
    row = s.as_dict()
    row.pop("id", None)
    cur = conn.execute(_INSERT, row)
    conn.commit()
    return int(cur.lastrowid)


def add_sightings(rows: list[Sighting], conn: sqlite3.Connection | None = None) -> int:
    conn = conn or connect()
    payload = []
    for s in rows:
        d = s.as_dict()
        d.pop("id", None)
        payload.append(d)
    conn.executemany(_INSERT, payload)
    conn.commit()
    return len(payload)


def add_alert(kind: str, severity: str, message: str, *, plate: str | None = None,
              camera_id: str | None = None, detail: dict | None = None,
              ts: float | None = None, conn: sqlite3.Connection | None = None) -> int:
    conn = conn or connect()
    cur = conn.execute(
        "INSERT INTO alerts (ts, kind, severity, plate, camera_id, message, detail)"
        " VALUES (?,?,?,?,?,?,?)",
        (ts or time.time(), kind, severity, plate, camera_id, message,
         json.dumps(detail or {})))
    conn.commit()
    return int(cur.lastrowid)


def ack_alert(alert_id: int, conn: sqlite3.Connection | None = None) -> None:
    conn = conn or connect()
    conn.execute("UPDATE alerts SET acked = 1 WHERE id = ?", (alert_id,))
    conn.commit()


def add_watch(plate: str, reason: str, severity: str = "high",
              added_by: str = "control-room", conn: sqlite3.Connection | None = None) -> None:
    conn = conn or connect()
    conn.execute(
        "INSERT OR REPLACE INTO watchlist (plate, reason, severity, added_ts, added_by)"
        " VALUES (?,?,?,?,?)", (plate.upper().replace(" ", ""), reason, severity,
                                time.time(), added_by))
    conn.commit()


def remove_watch(plate: str, conn: sqlite3.Connection | None = None) -> int:
    conn = conn or connect()
    cur = conn.execute("DELETE FROM watchlist WHERE plate = ?", (plate.upper().replace(" ", ""),))
    conn.commit()
    return cur.rowcount


# --- reads --------------------------------------------------------------
def rows(sql: str, params: tuple | dict = (), conn: sqlite3.Connection | None = None) -> list[dict]:
    conn = conn or connect()
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def cameras(conn: sqlite3.Connection | None = None) -> list[dict]:
    return rows("SELECT * FROM cameras ORDER BY id", conn=conn)


def watchlist(conn: sqlite3.Connection | None = None) -> list[dict]:
    return rows("SELECT * FROM watchlist ORDER BY added_ts DESC", conn=conn)


def is_watched(plate: str, conn: sqlite3.Connection | None = None) -> dict | None:
    r = rows("SELECT * FROM watchlist WHERE plate = ?", (plate,), conn=conn)
    return r[0] if r else None


def recent_sightings(limit: int = 100, camera_id: str | None = None,
                     conn: sqlite3.Connection | None = None) -> list[dict]:
    if camera_id:
        return rows("SELECT * FROM sightings WHERE camera_id = ? ORDER BY ts DESC LIMIT ?",
                    (camera_id, limit), conn=conn)
    return rows("SELECT * FROM sightings ORDER BY ts DESC LIMIT ?", (limit,), conn=conn)


def sightings_for_plate(plate: str, since: float | None = None, until: float | None = None,
                        conn: sqlite3.Connection | None = None) -> list[dict]:
    sql = "SELECT * FROM sightings WHERE plate = ?"
    params: list = [plate]
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    if until is not None:
        sql += " AND ts <= ?"
        params.append(until)
    return rows(sql + " ORDER BY ts ASC", tuple(params), conn=conn)


def sightings_between(since: float, until: float | None = None,
                      conn: sqlite3.Connection | None = None) -> list[dict]:
    return rows("SELECT * FROM sightings WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
                (since, until if until is not None else time.time() + 1), conn=conn)


def distinct_plates(since: float | None = None, conn: sqlite3.Connection | None = None) -> list[str]:
    if since is None:
        return [r["plate"] for r in rows("SELECT DISTINCT plate FROM sightings", conn=conn)]
    return [r["plate"] for r in rows(
        "SELECT DISTINCT plate FROM sightings WHERE ts >= ?", (since,), conn=conn)]


def alerts(limit: int = 100, include_acked: bool = True, since: float | None = None,
           conn: sqlite3.Connection | None = None) -> list[dict]:
    sql = "SELECT * FROM alerts WHERE 1=1"
    params: list = []
    if not include_acked:
        sql += " AND acked = 0"
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    out = rows(sql + " ORDER BY ts DESC LIMIT ?", tuple(params + [limit]), conn=conn)
    for a in out:
        try:
            a["detail"] = json.loads(a.get("detail") or "{}")
        except json.JSONDecodeError:
            a["detail"] = {}
    return out


def stats(conn: sqlite3.Connection | None = None) -> dict:
    conn = conn or connect()
    one = lambda sql: conn.execute(sql).fetchone()[0]      # noqa: E731
    span = conn.execute("SELECT MIN(ts), MAX(ts) FROM sightings").fetchone()
    return {
        "sightings": one("SELECT COUNT(*) FROM sightings"),
        "unique_plates": one("SELECT COUNT(DISTINCT plate) FROM sightings"),
        "cameras": one("SELECT COUNT(*) FROM cameras"),
        "alerts": one("SELECT COUNT(*) FROM alerts"),
        "open_alerts": one("SELECT COUNT(*) FROM alerts WHERE acked = 0"),
        "watchlist": one("SELECT COUNT(*) FROM watchlist"),
        "first_ts": span[0], "last_ts": span[1],
        "mean_confidence": one("SELECT COALESCE(AVG(confidence), 0) FROM sightings"),
    }
