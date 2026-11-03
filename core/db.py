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
    -- Who attested that this vehicle was at this place at this time. 'camera'
    -- for everything a node or the simulator wrote; an enrolment value when an
    -- operator supplied the place and time from a photograph the engine read.
    -- The plate itself is the engine's read either way - this column is about
    -- provenance of the *observation*, not of the string.
    source        TEXT NOT NULL DEFAULT 'camera',
    enrolled_by   TEXT,                 -- operator identity, enrolments only
    enrolled_note TEXT,                 -- why, and the original read if corrected
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

-- Captures the engine could not stand behind. A read below the confidence
-- floor is not stored as a sighting - it never was, and inventing one is how a
-- platform starts fabricating vehicles. But the *evidence* is worth keeping for
-- a while: the CTC lattice still holds the plate, and once the neighbouring
-- cameras have reported, a candidate set exists that did not exist at capture
-- time. `evidence` is dropped by prune_evidence() once that window has passed.
CREATE TABLE IF NOT EXISTS unresolved (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    camera_id   TEXT NOT NULL,
    best_text   TEXT,                   -- what the engine decoded, below floor
    confidence  REAL NOT NULL,
    evidence    BLOB,                   -- CTC lattices, float16, compressed
    -- The plate crop, JPEG, ~2 KB. Retained for exactly as long as the lattice
    -- beside it and dropped by the same pruner, because it exists for the same
    -- reason: anpr/salvage.py locates glare and blur in the PIXELS, which is
    -- the one place the recogniser never looks, and it cannot do that from a
    -- lattice. The plate region only - never the vehicle, never the cabin.
    crop        BLOB,
    frames      INTEGER NOT NULL DEFAULT 1,
    condition   TEXT,
    true_plate  TEXT,                   -- ground truth, simulator only
    resolved    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_unres_open ON unresolved(resolved, ts);

-- Registrations the platform *inferred* rather than read. Deliberately not in
-- `sightings`: an inference is a hypothesis with an evidence chain, not an
-- observation, and the two must never be confused by a query that ends up in
-- front of a magistrate. Keeping them apart also enforces the no-chaining rule
-- structurally - candidate sets are built from `sightings`, so an inference can
-- never become evidence for another inference.
CREATE TABLE IF NOT EXISTS inferences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    unresolved_id INTEGER NOT NULL,
    ts            REAL NOT NULL,
    camera_id     TEXT NOT NULL,
    plate         TEXT NOT NULL,
    score         REAL NOT NULL,        -- per-character CTC log-likelihood
    margin        REAL NOT NULL,        -- how far it beat the runner-up
    candidates    INTEGER NOT NULL,     -- how many plates it was chosen from
    detail        TEXT,                 -- JSON: the neighbour sightings that supported it
    true_plate    TEXT,
    FOREIGN KEY (unresolved_id) REFERENCES unresolved(id)
);
CREATE INDEX IF NOT EXISTS idx_infer_plate ON inferences(plate, ts);
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
    # These three sit before `id` deliberately: add_sighting() pops `id` and
    # feeds asdict() straight into a named-parameter INSERT, so a field added
    # after it would still work - but every existing construction site passes
    # keywords, and keeping `id` last keeps the dataclass readable as
    # "the row, then its rowid".
    source: str = config.CAMERA_SOURCE
    enrolled_by: str | None = None
    enrolled_note: str | None = None
    id: int | None = field(default=None)

    def as_dict(self) -> dict:
        return asdict(self)


# Columns added after the first databases were seeded. `CREATE TABLE IF NOT
# EXISTS` cannot add a column to a table that already exists, so a database
# built before manual enrolment existed needs these bolted on explicitly.
_ADDED_COLUMNS = {
    "sightings": [
        ("source", f"TEXT NOT NULL DEFAULT '{config.CAMERA_SOURCE}'"),
        ("enrolled_by", "TEXT"),
        ("enrolled_note", "TEXT"),
    ],
}


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Bring an existing database up to the current column set. Idempotent.

    Takes the connection rather than a path, because connect() calls it and a
    path argument would recurse. Returns the columns it actually added, which
    is what the enrolment tests assert on.

    Deliberately tolerant of a table that is not there yet: on a fresh database
    connect() runs this before the schema has been created, and the right
    answer then is to do nothing and let executescript() build the table with
    the columns already in it.
    """
    added: list[str] = []
    for table, columns in _ADDED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not have:
            continue                    # table does not exist yet - fresh database
        for name, decl in columns:
            if name in have:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            added.append(f"{table}.{name}")
    # Created here rather than in SCHEMA on purpose: executescript(SCHEMA) runs
    # at API startup against whatever database is already on disk, and a
    # CREATE INDEX naming a column an old table lacks is a hard error, not a
    # no-op - it would kill the API before the dashboard ever opened.
    if conn.execute("PRAGMA table_info(sightings)").fetchall():
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sight_source ON sightings(source, ts)")
    if added:
        conn.commit()
    return added


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
        # Here, not in init()/ensure_schema(): the Streamlit dashboard reaches
        # the database through connect() alone and calls neither, and the
        # dashboard is where an enrolment is written.
        migrate(conn)
    return conn


def init(path: Path | None = None, cameras: list[dict] | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    migrate(conn)          # fresh database: the table exists only now
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
    conn.execute("DELETE FROM inferences")
    conn.execute("DELETE FROM unresolved")
    if not keep_watchlist:
        conn.execute("DELETE FROM watchlist")
    conn.commit()


# --- writes -------------------------------------------------------------
_INSERT = (
    "INSERT INTO sightings (ts, camera_id, plate, confidence, speed_kmph, lane, heading,"
    " direction, next_camera, vehicle_class, condition, ocr_variant, true_plate,"
    " source, enrolled_by, enrolled_note)"
    " VALUES (:ts, :camera_id, :plate, :confidence, :speed_kmph, :lane, :heading,"
    " :direction, :next_camera, :vehicle_class, :condition, :ocr_variant, :true_plate,"
    " :source, :enrolled_by, :enrolled_note)"
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


def ensure_schema(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    """Create any missing tables. Cheap - every statement is IF NOT EXISTS - and
    it lets the repair layer run against a database seeded before it existed."""
    conn = conn or connect()
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    # a column added after a database was seeded needs saying explicitly. Kept
    # as a list rather than a version number because every migration here is
    # additive and idempotent - ALTER fails with "duplicate column" when it has
    # already run, which is the signal that there is nothing to do.
    for table, column, decl in (("unresolved", "crop", "BLOB"),):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError:
            pass                        # already there
    return conn


def add_unresolved(ts: float, camera_id: str, best_text: str, confidence: float,
                   evidence: bytes | None, frames: int = 1, condition: str | None = None,
                   true_plate: str | None = None, crop: bytes | None = None,
                   conn: sqlite3.Connection | None = None) -> int:
    conn = conn or connect()
    cur = conn.execute(
        "INSERT INTO unresolved (ts, camera_id, best_text, confidence, evidence,"
        " frames, condition, true_plate, crop) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, camera_id, best_text, confidence, evidence, frames, condition,
         true_plate, crop))
    return int(cur.lastrowid)


def open_unresolved(since: float | None = None, until: float | None = None,
                    limit: int = 500, conn: sqlite3.Connection | None = None) -> list[dict]:
    """Captures still awaiting a verdict, oldest first."""
    sql = "SELECT * FROM unresolved WHERE resolved = 0 AND evidence IS NOT NULL"
    params: list = []
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    if until is not None:
        sql += " AND ts <= ?"
        params.append(until)
    sql += " ORDER BY ts LIMIT ?"
    params.append(limit)
    return rows(sql, tuple(params), conn=conn)


def resolve_unresolved(ids: list[int], conn: sqlite3.Connection | None = None) -> None:
    if not ids:
        return
    conn = conn or connect()
    conn.executemany("UPDATE unresolved SET resolved = 1 WHERE id = ?",
                     [(int(i),) for i in ids])
    conn.commit()


def add_inference(unresolved_id: int, ts: float, camera_id: str, plate: str,
                  score: float, margin: float, candidates: int, detail: dict,
                  true_plate: str | None = None,
                  conn: sqlite3.Connection | None = None) -> int:
    conn = conn or connect()
    cur = conn.execute(
        "INSERT INTO inferences (unresolved_id, ts, camera_id, plate, score, margin,"
        " candidates, detail, true_plate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (unresolved_id, ts, camera_id, plate, score, margin, candidates,
         json.dumps(detail), true_plate))
    return int(cur.lastrowid)


def inferences(plate: str | None = None, since: float | None = None, limit: int = 200,
               conn: sqlite3.Connection | None = None) -> list[dict]:
    sql = "SELECT * FROM inferences WHERE 1 = 1"
    params: list = []
    if plate:
        sql += " AND plate = ?"
        params.append(plate)
    if since is not None:
        sql += " AND ts >= ?"
        params.append(since)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    out = rows(sql, tuple(params), conn=conn)
    for r in out:
        r["detail"] = json.loads(r["detail"]) if r.get("detail") else {}
    return out


def prune_evidence(older_than_s: float, conn: sqlite3.Connection | None = None) -> int:
    """Drop retained CTC lattices past the reconciliation window.

    The evidence is only useful until the neighbouring cameras have reported and
    the repair pass has run. Keeping it beyond that turns a bounded buffer into
    an unbounded store of raw recognition data, which is both a storage problem
    and a data-protection one.
    """
    conn = conn or connect()
    # The crop goes with the lattice, on the same clock and in the same
    # statement. Separating them would let a retained image outlive the
    # evidence it was kept to explain, which is precisely the unbounded store
    # of raw recognition data this function exists to prevent.
    cur = conn.execute("UPDATE unresolved SET evidence = NULL, crop = NULL"
                       " WHERE (evidence IS NOT NULL OR crop IS NOT NULL)"
                       " AND ts < ?", (older_than_s,))
    conn.commit()
    return cur.rowcount


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
                      conn: sqlite3.Connection | None = None,
                      sources: tuple[str, ...] | None = None) -> list[dict]:
    """Sightings in a time range. `sources` restricts by provenance.

    Default is every source, because most callers want "what was seen". Pass
    `sources=(config.CAMERA_SOURCE,)` to measure the camera network itself -
    core/analytics.py does, so an operator enrolment can never move a traffic
    statistic. Note the `until` default: it bounds at wall clock, so a
    future-dated row is invisible here until the clock catches up.
    """
    until = until if until is not None else time.time() + 1
    if not sources:
        return rows("SELECT * FROM sightings WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
                    (since, until), conn=conn)
    marks = ",".join("?" * len(sources))
    return rows(f"SELECT * FROM sightings WHERE ts >= ? AND ts <= ?"
                f" AND source IN ({marks}) ORDER BY ts ASC",
                (since, until, *sources), conn=conn)


def latest_ts(sources: tuple[str, ...] | None = None,
              conn: sqlite3.Connection | None = None) -> float:
    """The newest sighting's timestamp - the store's own idea of "now".

    A seeded database stops advancing the moment seeding finishes, so its
    present drifts away from wall clock by however long ago it was built. Code
    that windows on `time.time()` is silently asserting the data is live;
    everything that asks "the last N hours" should ask it of this instead.

    Falls back to wall clock on an empty table so a fresh install still works.
    `sources` restricts by provenance - analytics passes the camera source so an
    operator enrolment cannot move the clock the traffic statistics hang off.
    """
    if sources:
        marks = ",".join("?" * len(sources))
        r = rows(f"SELECT MAX(ts) t FROM sightings WHERE source IN ({marks})",
                 tuple(sources), conn=conn)
    else:
        r = rows("SELECT MAX(ts) t FROM sightings", conn=conn)
    latest = r[0]["t"] if r else None
    return float(latest) if latest is not None else time.time()


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
