"""The registry itself: schema, generation, and lookup(plate).

`lookup()` is the seam. Its SHAPE is what a real VAHAN integration would
return, so replacing this file with one that calls a real endpoint changes
nothing above it. Everything it returns today is invented by this module and
every record says so in `synthetic`.

Two properties are deliberate:

  Deterministic. A record is derived from a hash of the plate, so the same
  registration always yields the same vehicle, owner and history - across
  processes, across re-seeds, across machines. A registry that answered
  differently on each call would be useless to test against, and a demo where
  the same plate is a stolen truck one minute and a clean scooter the next
  would be worse.

  Read-only about the platform. The one place this package touches
  data/godseye.db it opens with mode=ro, so a coding mistake here cannot write
  to the store the rest of the project depends on. That is enforced by SQLite,
  not by care.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from registry import catalogue as cat
from registry import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    plate            TEXT PRIMARY KEY,
    vehicle_class    TEXT NOT NULL,     -- registered class, per this registry
    make             TEXT NOT NULL,
    model            TEXT NOT NULL,
    fuel             TEXT NOT NULL,
    colour           TEXT NOT NULL,
    manufactured     INTEGER NOT NULL,
    engine_no        TEXT NOT NULL,
    chassis_no       TEXT NOT NULL,
    rto_code         TEXT NOT NULL,
    rto_office       TEXT NOT NULL,
    rto_state        TEXT NOT NULL,
    registered_on    REAL NOT NULL,
    -- Deliberately NO owner. This registry describes vehicles, not people.
    -- Invented names would still have been invented PEOPLE on a wall display
    -- in front of judges, and nothing downstream needs one: every rule in
    -- registry/sentinel.py acts on the vehicle and routes to an authority.
    -- The column is absent rather than blanked so it cannot creep back.
    -- validity, as unix seconds. NULL where the document does not apply.
    insurance_to     REAL,
    insurer          TEXT,
    puc_to           REAL,
    fitness_to       REAL,              -- commercial only
    permit_to        REAL,              -- commercial only
    permit_type      TEXT,
    tax_to           REAL,
    status           TEXT NOT NULL DEFAULT 'active',
    status_note      TEXT,
    status_since     REAL
);

CREATE TABLE IF NOT EXISTS cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    plate        TEXT NOT NULL,
    kind         TEXT NOT NULL,        -- challan | fir | summons
    section      TEXT NOT NULL,
    description  TEXT NOT NULL,
    amount       REAL NOT NULL DEFAULT 0,
    raised_on    REAL NOT NULL,
    station      TEXT NOT NULL,
    authority    TEXT NOT NULL,        -- who owns it: traffic/police/transport/pollution
    settled      INTEGER NOT NULL DEFAULT 0,
    reference    TEXT NOT NULL,
    FOREIGN KEY (plate) REFERENCES vehicles(plate)
);
CREATE INDEX IF NOT EXISTS idx_cases_plate ON cases(plate, settled);

-- Who asked about whom. A registry that cannot answer that is exactly what
-- core/privacy.py argues against, so this one can.
CREATE TABLE IF NOT EXISTS lookups (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    plate    TEXT NOT NULL,
    actor    TEXT NOT NULL,
    purpose  TEXT NOT NULL,
    found    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lookups_plate ON lookups(plate, ts);

-- A finding somebody decided to act on. Stored BEFORE anyone is notified, and
-- never removed by a failed notification: an incident that vanishes because a
-- downstream system was unreachable is the one outcome this table exists to
-- prevent. Delivery state is a column on the incident, not the incident's
-- reason for existing.
CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ref             TEXT UNIQUE NOT NULL,      -- INC-0042, the human handle
    ts              REAL NOT NULL,
    plate           TEXT NOT NULL,
    kind            TEXT NOT NULL,             -- incident_type
    severity        TEXT NOT NULL,
    camera_id       TEXT,
    camera_name     TEXT,
    headline        TEXT NOT NULL,
    detail          TEXT,
    caveat          TEXT,                      -- what this is NOT
    recommended     TEXT,
    authority       TEXT NOT NULL,             -- who it was routed to
    registry_status TEXT NOT NULL,             -- FOUND | NOT_FOUND | UNKNOWN
    confidence      REAL,
    evidence        TEXT,                      -- JSON list
    -- delivery, entirely separate from the incident's own truth
    delivery_status TEXT NOT NULL DEFAULT 'PENDING',
    attempts        INTEGER NOT NULL DEFAULT 0,
    authority_ref   TEXT,                      -- AUTH-8821, once acknowledged
    last_attempt_ts REAL,
    last_error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_inc_ts ON incidents(ts);
CREATE INDEX IF NOT EXISTS idx_inc_delivery ON incidents(delivery_status, ts);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


# --- connections --------------------------------------------------------
def connect(path: Path | None = None) -> sqlite3.Connection:
    """The registry's own database, created on first use."""
    p = Path(path or settings.DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def platform_ro() -> sqlite3.Connection | None:
    """The platform's store, opened READ-ONLY. None if it is not there.

    mode=ro is the whole point: this package must be unable to alter the
    database the rest of the project runs on, and an enforced guarantee beats
    a promise not to write.
    """
    p = Path(settings.PLATFORM_DB)
    if not p.exists():
        return None
    uri = f"file:{p.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# --- deterministic generation ------------------------------------------
def _rng(plate: str) -> random.Random:
    """A generator bound to this plate, stable across runs and machines.

    Seeded from a hash rather than hash(): Python randomises str hashing per
    process, so hash() would give a different vehicle on every restart.
    """
    digest = hashlib.sha256(plate.upper().encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _state_from_plate(plate: str) -> str:
    return plate[:2].upper() if len(plate) >= 2 else "WB"


def _pick_rto(rng: random.Random, plate: str):
    """The office that issued this registration.

    Always in the state the plate's own code names. A named office is used
    where the catalogue has one for that district or state; otherwise the
    office is synthesised from the plate itself rather than borrowed from
    another state. Borrowing was a real bug: a KA plate came back registered
    at a Kolkata RTO, which would have made "registered out of state" - a
    signal worth having - indistinguishable from noise.
    """
    head = plate[:4].upper()
    exact = [r for r in cat.RTO_OFFICES if r[0] == head]
    if exact:
        return exact[0]
    state = _state_from_plate(plate)
    same = [r for r in cat.RTO_OFFICES if r[0].startswith(state)]
    if same:
        return rng.choice(same)
    return (head, f"{cat.STATE_NAMES.get(state, state)} RTO {head[2:4]}",
            cat.STATE_NAMES.get(state, state))


def _days(now: float, lo: int, hi: int, rng: random.Random) -> float:
    return now + rng.uniform(lo, hi) * 86400.0


def build_record(plate: str, observed_class: str | None, now: float) -> dict:
    """Invent this plate's registration. Deterministic for a given plate.

    `observed_class` is what the cameras have been seeing. It is used only to
    make the registry AGREE with reality most of the time - a registry that
    disagreed at random would make the class-mismatch rule meaningless noise
    instead of a fraud signal.
    """
    plate = plate.upper().replace(" ", "")
    rng = _rng(plate)
    s = settings

    klass = observed_class or rng.choice(list(cat.MODELS))
    if klass not in cat.MODELS:
        klass = "car"
    # A rare few are registered as something else entirely. That is the point
    # of the rule downstream: a plate on the wrong kind of vehicle.
    mismatched = rng.random() < s.SHARE_CLASS_MISMATCH
    reg_class = klass
    if mismatched:
        others = [k for k in cat.MODELS if k != klass]
        reg_class = rng.choice(others)

    make, model, fuels = rng.choice(cat.MODELS[reg_class])
    rto_code, rto_office, rto_state = _pick_rto(rng, plate)
    commercial = reg_class in cat.COMMERCIAL

    age_years = rng.randint(0, 17)
    registered_on = now - age_years * 365.25 * 86400 - rng.uniform(0, 300) * 86400
    manufactured = 2026 - age_years

    # Validity. Most documents are in date; the shares in settings decide how
    # many are not, and an expired one is expired by a plausible margin rather
    # than by a day.
    ins_expired = rng.random() < s.SHARE_INSURANCE_EXPIRED
    puc_expired = rng.random() < s.SHARE_PUC_EXPIRED
    tax_due = rng.random() < s.SHARE_TAX_DUE
    insurance_to = (_days(now, -420, -3, rng) if ins_expired
                    else _days(now, 20, 400, rng))
    puc_to = _days(now, -260, -2, rng) if puc_expired else _days(now, 10, 180, rng)
    tax_to = _days(now, -300, -5, rng) if tax_due else _days(now, 30, 500, rng)

    fitness_to = permit_to = permit_type = None
    if commercial:
        fit_expired = rng.random() < s.SHARE_FITNESS_EXPIRED
        per_expired = rng.random() < s.SHARE_PERMIT_EXPIRED
        fitness_to = (_days(now, -300, -2, rng) if fit_expired
                      else _days(now, 15, 330, rng))
        permit_to = (_days(now, -240, -2, rng) if per_expired
                     else _days(now, 25, 900, rng))
        permit_type = rng.choice(
            ["Contract Carriage", "Stage Carriage", "Goods - National",
             "Goods - State", "City Permit"])

    # Status. Ordered so the most serious wins when two coincide.
    status, note, since = "active", None, None
    roll = rng.random()
    if roll < s.SHARE_STOLEN:
        status = "stolen"
        note = (f"Reported stolen, FIR {rng.randint(100, 999)}/2026, "
                f"{rng.choice(cat.POLICE_STATIONS)}")
        since = _days(now, -400, -6, rng)
    elif roll < s.SHARE_STOLEN + s.SHARE_SCRAPPED:
        status = "scrapped"
        note = (f"Certificate of deposit issued at an RVSF on "
                f"{time.strftime('%d %b %Y', time.localtime(_days(now, -900, -60, rng)))}")
        since = _days(now, -900, -60, rng)
    elif roll < s.SHARE_STOLEN + s.SHARE_SCRAPPED + s.SHARE_BLACKLISTED:
        status = "blacklisted"
        note = rng.choice([
            "Blacklisted by RTO - repeated permit violations",
            "Blacklisted - tax default beyond 12 months",
            "Registration suspended pending NGT emission review"])
        since = _days(now, -500, -20, rng)

    return {
        "plate": plate,
        "vehicle_class": reg_class,
        "make": make, "model": model,
        "fuel": rng.choice(fuels),
        "colour": rng.choice(cat.COLOURS),
        "manufactured": manufactured,
        "engine_no": f"{rng.choice('ABCDEFGHJKLMN')}{rng.randint(10**9, 10**10 - 1)}",
        "chassis_no": f"MA{rng.choice('123456789')}{rng.randint(10**12, 10**13 - 1)}",
        "rto_code": rto_code, "rto_office": rto_office, "rto_state": rto_state,
        "registered_on": registered_on,
        "insurance_to": insurance_to,
        "insurer": rng.choice(cat.INSURERS),
        "puc_to": puc_to,
        "fitness_to": fitness_to,
        "permit_to": permit_to,
        "permit_type": permit_type,
        "tax_to": tax_to,
        "status": status, "status_note": note, "status_since": since,
        "_observed_class": klass,
        "_mismatch": mismatched,
    }


def build_cases(plate: str, record: dict, now: float) -> list[dict]:
    """Pending cases and challans against this plate. Deterministic."""
    rng = _rng(plate + "|cases")
    out: list[dict] = []

    # A stolen vehicle always carries its theft FIR; that is the record that
    # made it stolen, and a status with no case behind it would be unevidenced.
    if record["status"] == "stolen":
        out.append({
            "kind": "fir", "section": "IPC 379", "description": "Theft of vehicle",
            "amount": 0.0, "raised_on": record["status_since"] or now,
            "station": rng.choice(cat.POLICE_STATIONS), "authority": "police",
            "reference": f"FIR/{rng.randint(100, 999)}/2026",
        })

    if rng.random() < settings.SHARE_HAS_CASES:
        weights = [c[5] for c in cat.CASE_TYPES]
        for _ in range(rng.randint(1, settings.MAX_CASES_PER_VEHICLE)):
            kind, section, desc, (lo, hi), auth, _w = rng.choices(
                cat.CASE_TYPES, weights=weights)[0]
            # One criminal case is a situation; three on one scooter is a
            # corpus artefact. Later FIR draws fall back to a traffic offence.
            if kind == "fir" and any(c["kind"] == "fir" for c in out):
                kind, section, desc, (lo, hi), auth, _w = cat.CASE_TYPES[0]
            out.append({
                "kind": kind, "section": section, "description": desc,
                "amount": float(rng.randrange(lo, hi + 1, 100)) if hi else 0.0,
                "raised_on": _days(now, -700, -2, rng),
                "station": rng.choice(cat.POLICE_STATIONS),
                "authority": auth,
                "reference": (f"{'CH' if kind == 'challan' else kind.upper()}/"
                              f"{rng.randint(10000, 99999)}/2026"),
            })
    return out


# --- the public interface ----------------------------------------------
@dataclass
class Registration:
    """What a registry answers with. The shape a real one would return."""
    plate: str
    found: bool
    # The outcome of the LOOKUP: FOUND | NOT_FOUND | UNKNOWN.
    #
    # Distinct from `status` below, which is the VEHICLE's standing (active,
    # stolen, scrapped). Conflating them is the dangerous bug: if the registry
    # is unreachable and that is reported as NOT_FOUND, then every ordinary car
    # on the road becomes "no registration exists - possibly a fabricated
    # plate" and gets routed to the police. An outage would manufacture
    # accusations. UNKNOWN means "we could not ask", and no rule may act on it.
    registry_status: str = "UNKNOWN"
    synthetic: bool = True
    vehicle: dict = field(default_factory=dict)
    registration: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)
    status: str = "unknown"
    status_note: str | None = None
    status_since: float | None = None
    cases: list = field(default_factory=list)

    def __post_init__(self) -> None:
        # found=True with registry_status=UNKNOWN is a contradiction: we cannot
        # both hold a record and have failed to ask. UNKNOWN stays the default
        # because it is the safe one - a caller that forgets to say gets the
        # answer nothing may be concluded from - but a record that was actually
        # returned settles it.
        if self.found and self.registry_status == "UNKNOWN":
            self.registry_status = "FOUND"

    @property
    def pending_cases(self) -> list:
        return [c for c in self.cases if not c.get("settled")]

    @property
    def outstanding_amount(self) -> float:
        return round(sum(c.get("amount") or 0.0 for c in self.pending_cases), 2)

    def expired(self, now: float | None = None) -> list[str]:
        """Which documents are out of date right now."""
        now = time.time() if now is None else now
        out = []
        for key, label in (("insurance_to", "insurance"), ("puc_to", "PUC"),
                           ("fitness_to", "fitness"), ("permit_to", "permit"),
                           ("tax_to", "road tax")):
            v = self.validity.get(key)
            if v is not None and v < now:
                out.append(label)
        return out

    def as_dict(self) -> dict:
        d = asdict(self)
        d["pending_cases"] = self.pending_cases
        d["outstanding_amount"] = self.outstanding_amount
        d["expired"] = self.expired()
        return d


def _row_to_registration(row: sqlite3.Row, cases: list) -> Registration:
    return Registration(
        plate=row["plate"], found=True, registry_status="FOUND",
        vehicle={
            "class": row["vehicle_class"], "make": row["make"], "model": row["model"],
            "fuel": row["fuel"], "colour": row["colour"],
            "manufactured": row["manufactured"],
            # Identifiers are masked. A registry hands these to a workshop, not
            # to a wall display, and the console this feeds is a wall display.
            "engine_no": row["engine_no"][:3] + "*****",
            "chassis_no": row["chassis_no"][:4] + "*********",
        },
        registration={
            "rto_code": row["rto_code"], "rto_office": row["rto_office"],
            "state": row["rto_state"], "registered_on": row["registered_on"],
        },
        validity={
            "insurance_to": row["insurance_to"], "insurer": row["insurer"],
            "puc_to": row["puc_to"], "fitness_to": row["fitness_to"],
            "permit_to": row["permit_to"], "permit_type": row["permit_type"],
            "tax_to": row["tax_to"],
        },
        status=row["status"], status_note=row["status_note"],
        status_since=row["status_since"], cases=cases,
    )


def lookup(plate: str, *, actor: str = "sentinel", purpose: str = "",
           conn: sqlite3.Connection | None = None) -> Registration:
    """Everything the registry holds on one plate.

    Records the lookup. A registry that cannot say who asked about whom is the
    thing core/privacy.py exists to argue against, and this one is asked about
    private citizens by an automated console.
    """
    plate = plate.upper().replace(" ", "")

    # Demo control: force the registry to look unavailable or to know nothing,
    # so the resilience of the pipeline can be SHOWN rather than asserted.
    mode = (settings.FORCE_MODE or "").upper()
    if mode == "UNAVAILABLE":
        return unreachable(plate, "registry forced unavailable (demo control)")
    if mode == "NOT_FOUND":
        return Registration(plate=plate, found=False, registry_status="NOT_FOUND",
                            status="not_registered")

    own = conn is None
    try:
        conn = conn or connect()
    except Exception as exc:
        # Could not even open the registry. This is the case that must NOT be
        # reported as "no such vehicle".
        return unreachable(plate, f"registry unreachable: {type(exc).__name__}")
    try:
        row = conn.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,)).fetchone()
        cases = []
        if row is not None:
            cases = [dict(c) for c in conn.execute(
                "SELECT * FROM cases WHERE plate = ? ORDER BY raised_on DESC",
                (plate,)).fetchall()]
        if settings.AUDIT_LOOKUPS:
            conn.execute(
                "INSERT INTO lookups (ts, plate, actor, purpose, found)"
                " VALUES (?,?,?,?,?)",
                (time.time(), plate, actor, purpose, 1 if row is not None else 0))
            conn.commit()
        if row is None:
            return Registration(plate=plate, found=False,
                                registry_status="NOT_FOUND",
                                status="not_registered")
        return _row_to_registration(row, cases)
    except sqlite3.Error as exc:
        # The query itself failed - a locked, corrupt or vanished database.
        # Still not an answer about the vehicle.
        return unreachable(plate, f"registry query failed: {exc}")
    finally:
        if own and conn is not None:
            conn.close()


# --- cache --------------------------------------------------------------
# A plate crossing a junction is read many times a second and its registration
# does not change between frames. Keyed by plate, with a TTL.
#
# UNKNOWN is deliberately NOT cached: it is a statement about the registry's
# availability, not about the vehicle, and caching it would keep answering
# "we could not ask" for two minutes after the registry came back.
_cache: dict = {}


def cached_lookup(plate: str, *, actor: str = "sentinel", purpose: str = "",
                  ttl: float | None = None) -> Registration:
    """lookup(), reusing a recent answer for the same plate."""
    plate = plate.upper().replace(" ", "")
    ttl = settings.CACHE_TTL_S if ttl is None else ttl
    now = time.time()
    hit = _cache.get(plate)
    if hit and now - hit[0] < ttl:
        return hit[1]
    reg = lookup(plate, actor=actor, purpose=purpose)
    if reg.registry_status != "UNKNOWN":
        if len(_cache) >= settings.CACHE_MAX:
            _cache.clear()
        _cache[plate] = (now, reg)
    return reg


def cache_clear() -> None:
    _cache.clear()


def cache_stats() -> dict:
    return {"entries": len(_cache), "ttl_s": settings.CACHE_TTL_S,
            "max": settings.CACHE_MAX}


def unreachable(plate: str, reason: str) -> Registration:
    """The registry could not be asked. Says nothing about the vehicle."""
    return Registration(plate=plate.upper().replace(" ", ""), found=False,
                        registry_status="UNKNOWN", status="unknown",
                        status_note=reason)


def bulk_lookup(plates, *, actor: str = "sentinel", purpose: str = "",
                conn: sqlite3.Connection | None = None) -> dict:
    """lookup() over many plates on one connection. {plate: Registration}."""
    own = conn is None
    conn = conn or connect()
    try:
        return {p: lookup(p, actor=actor, purpose=purpose, conn=conn)
                for p in dict.fromkeys(x.upper().replace(" ", "") for x in plates)}
    finally:
        if own:
            conn.close()


def stats(conn: sqlite3.Connection | None = None) -> dict:
    own = conn is None
    conn = conn or connect()
    try:
        def one(sql, args=()):
            r = conn.execute(sql, args).fetchone()
            return r[0] if r else 0
        now = time.time()
        return {
            "vehicles": one("SELECT COUNT(*) FROM vehicles"),
            "by_status": {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) n FROM vehicles GROUP BY status")},
            "cases_open": one("SELECT COUNT(*) FROM cases WHERE settled = 0"),
            "outstanding": round(one(
                "SELECT COALESCE(SUM(amount),0) FROM cases WHERE settled = 0"), 2),
            "insurance_expired": one(
                "SELECT COUNT(*) FROM vehicles WHERE insurance_to < ?", (now,)),
            "puc_expired": one(
                "SELECT COUNT(*) FROM vehicles WHERE puc_to < ?", (now,)),
            "fitness_expired": one(
                "SELECT COUNT(*) FROM vehicles WHERE fitness_to IS NOT NULL"
                " AND fitness_to < ?", (now,)),
            "lookups": one("SELECT COUNT(*) FROM lookups"),
            "seeded_at": one("SELECT value FROM meta WHERE key = 'seeded_at'") or None,
        }
    finally:
        if own:
            conn.close()


# --- incidents ----------------------------------------------------------
def record_incident(finding: dict, *, registry_status: str = "FOUND",
                    confidence: float | None = None,
                    conn: sqlite3.Connection | None = None) -> dict:
    """Store a finding as an incident. Returns the stored row.

    Called BEFORE any attempt to notify anyone. That ordering is the whole
    point: the incident exists whether or not the authority can be reached, so
    a failed referral is a delivery problem and never a lost record.
    """
    own = conn is None
    conn = conn or connect()
    try:
        # The reference comes from the AUTOINCREMENT id, never from COUNT(*).
        # A count is reused the moment any row is deleted, which would mint a
        # second INC-0005 - a reference an authority has already been given for
        # a different incident, and the one thing a reference may never be. It
        # would also raise: `ref` is UNIQUE, so the collision surfaces as a
        # crash in whichever button happened to press it.
        #
        # Written in two steps because the id does not exist until the insert
        # has happened. The placeholder is unique per nanosecond and never
        # survives the transaction.
        tmp = f"pending-{time.time_ns()}"
        cur = conn.execute(
            "INSERT INTO incidents (ref, ts, plate, kind, severity, camera_id,"
            " camera_name, headline, detail, caveat, recommended, authority,"
            " registry_status, confidence, evidence)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tmp, time.time(), finding.get("plate", ""), finding.get("kind", ""),
             finding.get("severity", "low"), finding.get("camera_id"),
             finding.get("camera_name"), finding.get("headline", ""),
             finding.get("detail", ""), finding.get("caveat", ""),
             finding.get("recommended", ""),
             finding.get("authority", "police"), registry_status, confidence,
             json.dumps(finding.get("evidence") or [])))
        ref = f"INC-{cur.lastrowid:04d}"
        conn.execute("UPDATE incidents SET ref = ? WHERE id = ?",
                     (ref, cur.lastrowid))
        conn.commit()
        row = conn.execute("SELECT * FROM incidents WHERE id = ?",
                           (cur.lastrowid,)).fetchone()
        return dict(row)
    finally:
        if own:
            conn.close()


def set_delivery(incident_id: int, status: str, *, authority_ref: str | None = None,
                 error: str | None = None, bump_attempt: bool = True,
                 conn: sqlite3.Connection | None = None) -> None:
    """Update only how delivery went. The incident itself is never touched."""
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute(
            "UPDATE incidents SET delivery_status = ?,"
            " attempts = attempts + ?, authority_ref = COALESCE(?, authority_ref),"
            " last_attempt_ts = ?, last_error = ? WHERE id = ?",
            (status, 1 if bump_attempt else 0, authority_ref, time.time(),
             error, incident_id))
        conn.commit()
    finally:
        if own:
            conn.close()


def incidents(limit: int = 100, delivery_status: str | None = None,
              conn: sqlite3.Connection | None = None) -> list[dict]:
    own = conn is None
    conn = conn or connect()
    try:
        if delivery_status:
            rows = conn.execute(
                "SELECT * FROM incidents WHERE delivery_status = ?"
                " ORDER BY ts DESC LIMIT ?", (delivery_status, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM incidents ORDER BY ts DESC LIMIT ?",
                                (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own:
            conn.close()


def incidents_for(plate: str, kind: str | None = None, limit: int = 10,
                  conn: sqlite3.Connection | None = None) -> list[dict]:
    """Incidents already raised for one registration, newest first.

    The console asks this before offering to refer a finding: an operator who
    cannot see that the same thing was referred an hour ago will refer it
    again, and the authority on the other end learns to ignore the channel.
    """
    own = conn is None
    conn = conn or connect()
    try:
        plate = plate.upper().replace(" ", "")
        if kind:
            rows = conn.execute(
                "SELECT * FROM incidents WHERE plate = ? AND kind = ?"
                " ORDER BY ts DESC LIMIT ?", (plate, kind, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM incidents WHERE plate = ? ORDER BY ts DESC"
                " LIMIT ?", (plate, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own:
            conn.close()


def incident_by_ref(ref: str, conn: sqlite3.Connection | None = None) -> dict | None:
    own = conn is None
    conn = conn or connect()
    try:
        row = conn.execute("SELECT * FROM incidents WHERE ref = ?", (ref,)).fetchone()
        return dict(row) if row else None
    finally:
        if own:
            conn.close()


def who_looked_up(plate: str, limit: int = 50,
                  conn: sqlite3.Connection | None = None) -> list[dict]:
    """The audit answer: who asked about this registration, and why."""
    own = conn is None
    conn = conn or connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM lookups WHERE plate = ? ORDER BY ts DESC LIMIT ?",
            (plate.upper().replace(" ", ""), limit)).fetchall()]
    finally:
        if own:
            conn.close()
