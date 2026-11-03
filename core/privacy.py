"""Limits the platform enforces on itself, rather than limits it documents.

Every commercial plate-reader product ships a privacy policy, a data-management
FAQ and a compliance guide. Those are documents. They describe what an operator
is supposed to do with a store of several million records of where private
citizens drove, and nothing in the software makes any of it happen. The gap
between the policy and the binary is where mass plate surveillance actually
lives.

This module is the same set of promises written as code that runs:

  retention   what is past the horizon, per table, with the exact rows named,
              a dry run by default and a purge you have to ask for twice
  audit       a lookup of a citizen's movements writes a row saying who looked,
              at what, and why, before it returns anything
  aggregate   traffic statistics with every group smaller than k suppressed, so
              the analytics can be handed out without handing out a movement log
  posture     one call that says what the system is holding and on what grounds

Three rules run through all of it.

Say only what you did. Nothing here claims conformance with CJIS, the DPDP Act
or anything else with a name and a certification body - conformance is a legal
finding about an organisation, not a property of a function. The purge reports
that SQLite leaves deleted pages inside the database file until VACUUM, because
that is true and calling it erasure would be a lie.

The dry run must be the purge. Preview and delete build one temporary table of
held registrations and then share the identical predicate, so a preview cannot
promise a deletion the purge would not make, and cannot miss one it would.

Refuse rather than degrade. record() rejects a lookup with no stated purpose;
aggregate_traffic() has no plate parameter to filter on, because a k-anonymous
query with a plate filter is just a lookup wearing a hat. A control that can be
skipped by passing an empty string is documentation again.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from collections import Counter

import config
from core import db
from core.network import CameraNetwork, network

# How long observation data is kept before it is eligible for deletion. Thirty
# days is a placeholder, not a recommendation - the number belongs in a policy
# a human signed, and this module's job is to apply whatever that number is.
RETENTION_DAYS = float(getattr(config, "PRIVACY_RETENTION_DAYS", 30.0))

# The audit log outlives the data it describes on purpose. Expiring both on the
# same clock would mean that the moment a record ages out, the question "who
# looked at it while we had it" becomes permanently unanswerable - which would
# make the audit trail worthless in exactly the case it exists for.
AUDIT_RETENTION_DAYS = float(getattr(config, "PRIVACY_AUDIT_RETENTION_DAYS", 365.0))

# Minimum distinct vehicles in a published group. Below this the group is
# suppressed. Counting vehicles and not sightings is the whole point: a
# junction with 400 reads of one car is a movement log for that car.
K_ANONYMITY = int(getattr(config, "PRIVACY_K_ANONYMITY", 5))

# A lookup with no stated reason is refused. This is the one control here that
# an operator will find annoying, which is roughly the evidence that it works.
REQUIRE_PURPOSE = bool(getattr(config, "PRIVACY_REQUIRE_PURPOSE", True))

# A purge that would take out more than this share of the store stops and asks
# again. This guards against a fat-fingered horizon (`days=0` on a live
# database), not against policy - a first legitimate run on an old store really
# can be most of it, and force=True is the way to say so.
MAX_PURGE_FRACTION = float(getattr(config, "PRIVACY_MAX_PURGE_FRACTION", 0.5))

# Severity of open alert that pins a registration against automatic deletion.
# Default is every open alert, which is the cautious reading of "evidence in an
# open case". See holds() for why that default is worth looking at twice.
HOLD_SEVERITIES = tuple(getattr(config, "PRIVACY_HOLD_SEVERITIES",
                                ("critical", "high", "medium", "low")))

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS access_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    actor    TEXT NOT NULL,           -- who ran it, as the system knows them
    action   TEXT NOT NULL,           -- search / trajectory / forecast / export ...
    plate    TEXT,                    -- NULL for a query that named no individual
    purpose  TEXT NOT NULL,           -- why, in the operator's own words
    case_ref TEXT,                    -- FIR or case number, when there is one
    scope    TEXT,                    -- JSON: window, cameras, parameters
    results  INTEGER,                 -- how many records came back
    channel  TEXT NOT NULL DEFAULT 'api'
);
CREATE INDEX IF NOT EXISTS idx_access_ts    ON access_log(ts);
CREATE INDEX IF NOT EXISTS idx_access_plate ON access_log(plate, ts);
CREATE INDEX IF NOT EXISTS idx_access_actor ON access_log(actor, ts);
"""


def _ensure_audit(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    """Create the access log if it is not there. Additive and idempotent.

    Run on every entry point rather than once at import, because the dashboard,
    the API and the CLI each reach the database through their own thread-local
    connection and none of them calls an init hook this module controls. The
    statements are all IF NOT EXISTS, so the cost of being wrong about whether
    it already ran is a parse, and the cost of skipping it is a crash in front
    of an audience.
    """
    conn = conn or db.connect()
    conn.executescript(AUDIT_SCHEMA)
    return conn


# --- audit --------------------------------------------------------------
def record(actor: str, action: str, *, plate: str | None = None, purpose: str = "",
           case_ref: str | None = None, scope: dict | None = None,
           results: int | None = None, channel: str = "api",
           ts: float | None = None, conn: sqlite3.Connection | None = None) -> int:
    """Write the trace a lookup leaves behind. Returns the audit row id.

    Call this *before* returning results, not after, and do not wrap it in a
    try/except that swallows the failure. A search that cannot be logged should
    fail rather than proceed: a system that quietly keeps answering when its
    audit trail is broken is a surveillance apparatus with the receipts turned
    off, and that state is indistinguishable from the one an abusive operator
    would engineer on purpose.

    Raises ValueError on an empty actor or purpose. That refusal is the control
    - an audit log with 4,000 rows reading "actor=system, purpose=" documents
    nothing, and is worse than none because it looks like accountability.
    """
    actor = (actor or "").strip()
    purpose = (purpose or "").strip()
    if not actor:
        raise ValueError("access log needs an actor - an unattributed lookup is not auditable")
    if REQUIRE_PURPOSE and not purpose:
        raise ValueError(f"access log needs a stated purpose for {action} "
                         f"{'on ' + plate if plate else ''}".strip())
    conn = _ensure_audit(conn)
    cur = conn.execute(
        "INSERT INTO access_log (ts, actor, action, plate, purpose, case_ref, scope,"
        " results, channel) VALUES (?,?,?,?,?,?,?,?,?)",
        (float(ts if ts is not None else time.time()), actor, action,
         plate.upper().replace(" ", "") if plate else None,
         purpose, case_ref, json.dumps(scope or {}), results, channel))
    conn.commit()
    return int(cur.lastrowid)


def access_history(plate: str | None = None, actor: str | None = None,
                   action: str | None = None, since: float | None = None,
                   days: float | None = None, limit: int = 200,
                   conn: sqlite3.Connection | None = None) -> list[dict]:
    """The raw log, newest first, filtered however the reviewer needs it.

    Anchored on wall clock, unlike the analytics windows: an access happened
    when a person did it, and that has nothing to do with how fresh the
    sighting data is.
    """
    conn = _ensure_audit(conn)
    if since is None and days is not None:
        since = time.time() - float(days) * 86400.0
    sql = "SELECT * FROM access_log WHERE 1 = 1"
    params: list = []
    if plate:
        sql += " AND plate = ?"
        params.append(plate.upper().replace(" ", ""))
    if actor:
        sql += " AND actor = ?"
        params.append(actor)
    if action:
        sql += " AND action = ?"
        params.append(action)
    if since is not None:
        sql += " AND ts >= ?"
        params.append(float(since))
    out = db.rows(sql + " ORDER BY ts DESC LIMIT ?", tuple(params + [int(limit)]), conn=conn)
    for r in out:
        try:
            r["scope"] = json.loads(r.get("scope") or "{}")
        except json.JSONDecodeError:
            r["scope"] = {}
    return out


def who_searched(plate: str, days: float | None = None,
                 conn: sqlite3.Connection | None = None) -> dict:
    """Everyone who has looked up this registration, and what they said it was for.

    This is the subject-access question - the one a citizen, an oversight board
    or a defence lawyer asks - and it is answerable here only because record()
    refuses to write a row without an actor and a purpose.
    """
    plate = plate.upper().replace(" ", "")
    hits = access_history(plate=plate, days=days, limit=10_000, conn=conn)
    by_actor: dict[str, dict] = {}
    for h in hits:
        a = by_actor.setdefault(h["actor"], {"actor": h["actor"], "lookups": 0,
                                             "first_ts": h["ts"], "last_ts": h["ts"],
                                             "actions": Counter(), "purposes": [],
                                             "case_refs": set()})
        a["lookups"] += 1
        a["first_ts"] = min(a["first_ts"], h["ts"])
        a["last_ts"] = max(a["last_ts"], h["ts"])
        a["actions"][h["action"]] += 1
        if h["purpose"] and h["purpose"] not in a["purposes"]:
            a["purposes"].append(h["purpose"])
        if h.get("case_ref"):
            a["case_refs"].add(h["case_ref"])
    actors = []
    for a in by_actor.values():
        a["actions"] = dict(a["actions"])
        a["case_refs"] = sorted(a["case_refs"])
        actors.append(a)
    actors.sort(key=lambda a: -a["lookups"])
    return {
        "plate": plate,
        "lookups": len(hits),
        "actors": actors,
        "unattributed_lookups": sum(1 for h in hits if not h.get("case_ref")),
        "window_days": days,
        # An empty result means nobody looked *through an audited path*. It is
        # not proof nobody looked - see the module limitations.
        "audited_paths_only": True,
    }


def access_summary(days: float = 7.0, conn: sqlite3.Connection | None = None) -> dict:
    """Who has been searching, how much, and how much of it names a case.

    The number that matters here is `no_case_ref`. Targeted lookups against a
    named investigation are the legitimate use; a high volume of plate lookups
    with no case attached is what misuse of a plate-reader system looks like
    from the inside, and it is visible only in aggregate over an actor.

    This is a signal, not a finding. Plenty of legitimate work has no case
    number yet, so the report counts and ranks and stops there.
    """
    conn = _ensure_audit(conn)
    since = time.time() - float(days) * 86400.0
    hits = access_history(since=since, limit=100_000, conn=conn)
    by_actor: dict[str, dict] = {}
    for h in hits:
        a = by_actor.setdefault(h["actor"], {"actor": h["actor"], "lookups": 0,
                                             "distinct_plates": set(), "no_case_ref": 0,
                                             "actions": Counter()})
        a["lookups"] += 1
        if h["plate"]:
            a["distinct_plates"].add(h["plate"])
        if not h.get("case_ref"):
            a["no_case_ref"] += 1
        a["actions"][h["action"]] += 1
    actors = []
    for a in by_actor.values():
        actors.append({"actor": a["actor"], "lookups": a["lookups"],
                       "distinct_plates": len(a["distinct_plates"]),
                       "no_case_ref": a["no_case_ref"],
                       "actions": dict(a["actions"])})
    actors.sort(key=lambda a: -a["lookups"])
    plates = Counter(h["plate"] for h in hits if h["plate"])
    return {
        "window_days": days,
        "lookups": len(hits),
        "identified_lookups": sum(1 for h in hits if h["plate"]),
        "distinct_plates": len(plates),
        "distinct_actors": len(actors),
        "no_case_ref": sum(1 for h in hits if not h.get("case_ref")),
        "actors": actors,
        "most_searched": [{"plate": p, "lookups": n} for p, n in plates.most_common(10)],
    }


# --- retention holds ----------------------------------------------------
def holds(conn: sqlite3.Connection | None = None,
          severities: tuple[str, ...] = HOLD_SEVERITIES) -> dict:
    """Registrations that automatic deletion must not touch, and on what grounds.

    Two grounds. A plate on the watchlist is the subject of a live instruction
    from the control room. A plate attached to an unacknowledged alert is
    evidence in something nobody has closed yet. Neither is routine traffic
    data and neither should quietly disappear on a timer.

    The breakdown by alert kind is here because it exposes a failure mode this
    design creates. On the seeded database 220 of 679 registrations are held,
    and 194 of those are held by an unacknowledged `odd_hour` alert - a
    low-severity rule that fires on anyone driving between 01:00 and 05:00. An
    alert queue nobody clears silently becomes a retention policy nobody chose,
    and a third of the city gets kept indefinitely because of it. The fix is
    operational (work the queue) or configured (`PRIVACY_HOLD_SEVERITIES`); the
    code's job is to make it impossible not to notice.
    """
    conn = _ensure_audit(conn)
    watch = {r["plate"]: r for r in db.rows("SELECT * FROM watchlist", conn=conn)}
    marks = ",".join("?" * len(severities)) if severities else "''"
    alerted = db.rows(
        f"SELECT plate, kind, severity, COUNT(*) n FROM alerts"
        f" WHERE acked = 0 AND plate IS NOT NULL AND severity IN ({marks})"
        f" GROUP BY plate, kind, severity", tuple(severities), conn=conn)

    by_plate: dict[str, dict] = {}
    for p, w in watch.items():
        by_plate[p] = {"plate": p, "watchlist": True, "reason": w["reason"],
                       "severity": w["severity"], "open_alerts": 0, "alert_kinds": []}
    by_kind: Counter = Counter()
    for r in alerted:
        e = by_plate.setdefault(r["plate"], {"plate": r["plate"], "watchlist": False,
                                             "reason": None, "severity": None,
                                             "open_alerts": 0, "alert_kinds": []})
        e["open_alerts"] += r["n"]
        e["alert_kinds"].append(r["kind"])
        if _SEVERITY_RANK.get(r["severity"], 0) > _SEVERITY_RANK.get(e["severity"] or "", -1):
            e["severity"] = r["severity"]
        by_kind[r["kind"]] += 1

    watch_only = [p for p, e in by_plate.items() if e["watchlist"] and not e["open_alerts"]]
    return {
        "plates": sorted(by_plate),
        "count": len(by_plate),
        "watchlisted": len(watch),
        "watchlist_only": len(watch_only),
        "open_alert_plates": sum(1 for e in by_plate.values() if e["open_alerts"]),
        "by_alert_kind": dict(by_kind.most_common()),
        "hold_severities": list(severities),
        "detail": sorted(by_plate.values(), key=lambda e: (-e["open_alerts"], e["plate"])),
    }


def _stage_holds(conn: sqlite3.Connection, plates: list[str]) -> None:
    """Put the held registrations in a temp table the SQL can join against.

    Not a Python-side filter and not an inline IN (...) list, for one reason:
    the preview and the purge must run byte-identical predicates. A staged
    table makes that structural rather than a thing two functions are supposed
    to keep in step, which is how a dry run ends up lying about what a delete
    will do. It also sidesteps SQLite's bound-parameter ceiling once a hold set
    grows past a few hundred plates.
    """
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _privacy_hold (plate TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _privacy_hold")
    conn.executemany("INSERT OR IGNORE INTO _privacy_hold (plate) VALUES (?)",
                     [(p,) for p in plates])


# Each entry is (table, time column, deletable predicate, held-back predicate).
# `_HELD` is the shared subquery, so a row is either counted as deletable or
# counted as held and never both, and the two predicates of a table always
# partition everything past the cutoff.
_HELD = "SELECT plate FROM _privacy_hold"
_TABLES: list[tuple[str, str, str, str]] = [
    ("sightings", "ts", f"plate NOT IN ({_HELD})", f"plate IN ({_HELD})"),
    # An open alert is never routine data whatever its age, so acked = 0 counts
    # as held here rather than as deletable-but-skipped.
    ("alerts", "ts", f"acked = 1 AND (plate IS NULL OR plate NOT IN ({_HELD}))",
     f"acked = 0 OR plate IN ({_HELD})"),
    # `unresolved.best_text` is a below-threshold read, not an established
    # registration, but if it happens to name a held plate it is treated as
    # held - the cautious direction when the string is unreliable in the first
    # place.
    ("unresolved", "ts", f"best_text IS NULL OR best_text NOT IN ({_HELD})",
     f"best_text IN ({_HELD})"),
    ("inferences", "ts", f"plate NOT IN ({_HELD})", f"plate IN ({_HELD})"),
]


def retention_preview(days: float | None = None, now: float | None = None,
                      severities: tuple[str, ...] = HOLD_SEVERITIES,
                      conn: sqlite3.Connection | None = None) -> dict:
    """What a purge would delete, per table, with counts and date ranges.

    The cutoff is measured against wall clock, deliberately, and not against
    `db.latest_ts()` the way every analytics window in this platform is. The
    analytics anchor exists so a seeded database still shows a busy last thirty
    minutes. Retention is the opposite question: a store whose cameras went
    dark a year ago is a store holding year-old movement records, and anchoring
    the horizon on the newest row would mean data stops ageing the moment
    ingest stops. The report carries `clock_drift_hours` so an operator can see
    when the two disagree.
    """
    conn = _ensure_audit(conn)
    days = float(RETENTION_DAYS if days is None else days)
    now = float(now if now is not None else time.time())
    cutoff = now - days * 86400.0

    held = holds(conn=conn, severities=severities)
    _stage_holds(conn, held["plates"])

    tables = []
    total_rows = 0
    total_delete = 0
    for table, tscol, deletable, kept in _TABLES:
        agg = db.rows(
            f"SELECT COUNT(*) n, MIN({tscol}) a, MAX({tscol}) b FROM {table}"
            f" WHERE {tscol} < ? AND ({deletable})", (cutoff,), conn=conn)[0]
        held_n = db.rows(f"SELECT COUNT(*) n FROM {table} WHERE {tscol} < ? AND ({kept})",
                         (cutoff,), conn=conn)[0]["n"]
        total = db.rows(f"SELECT COUNT(*) n, MIN({tscol}) a, MAX({tscol}) b"
                        f" FROM {table}", conn=conn)[0]
        total_rows += total["n"]
        total_delete += agg["n"]
        tables.append({
            "table": table,
            "rows": total["n"],
            "oldest_ts": total["a"], "newest_ts": total["b"],
            "past_horizon": agg["n"] + held_n,
            "deletable": agg["n"],
            "held_back": held_n,
            "delete_from_ts": agg["a"], "delete_to_ts": agg["b"],
            "delete_from": _stamp(agg["a"]), "delete_to": _stamp(agg["b"]),
        })

    newest = db.rows("SELECT MAX(ts) t FROM sightings", conn=conn)[0]["t"]
    audit_cutoff = now - AUDIT_RETENTION_DAYS * 86400.0
    audit_due = db.rows("SELECT COUNT(*) n FROM access_log WHERE ts < ?",
                        (audit_cutoff,), conn=conn)[0]["n"]
    return {
        "retention_days": days,
        "cutoff_ts": cutoff,
        "cutoff": _stamp(cutoff),
        "now_ts": now,
        "anchor": "wall clock",
        # Positive means the newest sighting is older than the clock - a seeded
        # or replayed database. Large drift makes every number here suspect and
        # the operator should know before acting on them.
        "clock_drift_hours": (round((now - newest) / 3600.0, 2)
                              if newest is not None else None),
        "tables": tables,
        "deletable_rows": total_delete,
        "store_rows": total_rows,
        "deletable_fraction": round(total_delete / total_rows, 4) if total_rows else 0.0,
        "holds": {k: held[k] for k in ("count", "watchlisted", "watchlist_only",
                                       "open_alert_plates", "by_alert_kind",
                                       "hold_severities")},
        "audit_log": {"retention_days": AUDIT_RETENTION_DAYS,
                      "past_horizon": audit_due,
                      "note": "the access log is kept longer than the data it describes"},
        "dry_run": True,
    }


def retention_purge(days: float | None = None, *, confirm: bool = False,
                    actor: str = "", purpose: str = "scheduled retention purge",
                    force: bool = False, now: float | None = None,
                    severities: tuple[str, ...] = HOLD_SEVERITIES,
                    conn: sqlite3.Connection | None = None) -> dict:
    """Delete what retention_preview() says is deletable. Dry run unless told twice.

    `confirm=False` is the default and returns the preview untouched, so the
    unthinking call and the call from a scheduler that lost its arguments both
    do nothing. An actor is required when confirm is set, because the purge
    writes itself into the access log - a deletion of evidence that leaves no
    trace is the one operation in this module that could be used to cover
    something up.

    What this does not do: erase. SQLite marks the pages free and reuses them
    later, so the bytes of a deleted sighting stay in the database file until
    VACUUM rewrites it, and stay in the WAL and in any filesystem snapshot
    regardless. `vacuum=True` is not offered here because VACUUM takes an
    exclusive lock and this database has a live API and two dashboards on it.
    The returned `secure_erase: False` is the honest label.
    """
    plan = retention_preview(days=days, now=now, severities=severities, conn=conn)
    if not confirm:
        plan["deleted"] = {}
        plan["reason"] = "dry run - pass confirm=True to delete"
        return plan

    actor = (actor or "").strip()
    if not actor:
        raise ValueError("a purge needs a named actor - see the docstring")
    if plan["deletable_fraction"] > MAX_PURGE_FRACTION and not force:
        plan["dry_run"] = True
        plan["deleted"] = {}
        plan["refused"] = True
        plan["reason"] = (
            f"would delete {plan['deletable_fraction']:.1%} of the store "
            f"({plan['deletable_rows']} rows) at a {plan['retention_days']:g}-day "
            f"horizon, above the {MAX_PURGE_FRACTION:.0%} safety limit - "
            f"pass force=True if that is genuinely the policy")
        return plan

    conn = _ensure_audit(conn)
    cutoff = plan["cutoff_ts"]
    deleted: dict[str, int] = {}
    try:
        # Inferences first: each one points at an `unresolved` row, and
        # deleting the parent before the child would leave a dangling
        # evidence chain behind rather than a clean absence.
        order = {t[0]: i for i, t in enumerate(_TABLES)}
        for table, tscol, deletable, _kept in sorted(
                _TABLES, key=lambda t: -order[t[0]]):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE {tscol} < ? AND ({deletable})", (cutoff,))
            deleted[table] = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    plan["dry_run"] = False
    plan["deleted"] = deleted
    plan["deleted_rows"] = sum(deleted.values())
    plan["secure_erase"] = False
    plan["erase_note"] = ("rows are gone from every query; SQLite keeps the freed pages"
                          " inside the database file until VACUUM, and the WAL and any"
                          " filesystem snapshot are outside this module's reach")
    plan["audit_id"] = record(actor, "retention_purge", purpose=purpose,
                              scope={"retention_days": plan["retention_days"],
                                     "cutoff": plan["cutoff"], "deleted": deleted},
                              results=plan["deleted_rows"], channel="privacy",
                              conn=conn)
    return plan


def purge_audit_log(days: float | None = None, *, confirm: bool = False,
                    actor: str = "", now: float | None = None,
                    conn: sqlite3.Connection | None = None) -> dict:
    """Age out the access log on its own, longer horizon. Dry run by default.

    Kept separate from retention_purge() rather than folded into it, so that
    trimming the record of who searched is always a deliberate act with its own
    audit row, and can never happen as a side effect of routine housekeeping.
    """
    conn = _ensure_audit(conn)
    days = float(AUDIT_RETENTION_DAYS if days is None else days)
    now = float(now if now is not None else time.time())
    cutoff = now - days * 86400.0
    agg = db.rows("SELECT COUNT(*) n, MIN(ts) a, MAX(ts) b FROM access_log WHERE ts < ?",
                  (cutoff,), conn=conn)[0]
    out = {"retention_days": days, "cutoff_ts": cutoff, "cutoff": _stamp(cutoff),
           "deletable": agg["n"], "delete_from": _stamp(agg["a"]),
           "delete_to": _stamp(agg["b"]), "dry_run": True, "deleted": 0}
    if not confirm:
        out["reason"] = "dry run - pass confirm=True to delete"
        return out
    if not (actor or "").strip():
        raise ValueError("trimming the audit log needs a named actor")
    audit_id = record(actor, "audit_log_purge", purpose=f"audit retention {days:g} days",
                      scope={"cutoff": out["cutoff"], "eligible": agg["n"]},
                      results=agg["n"], channel="privacy", conn=conn)
    cur = conn.execute("DELETE FROM access_log WHERE ts < ? AND id != ?", (cutoff, audit_id))
    conn.commit()
    out.update(dry_run=False, deleted=cur.rowcount, audit_id=audit_id)
    return out


# --- aggregate access ---------------------------------------------------
# Only these groupings can be asked for. A whitelist rather than a column name
# passed through to SQL, for two reasons: an arbitrary expression here is an
# injection, and a grouping fine enough (plate, sighting id, second-resolution
# time) re-identifies individuals no matter what k is set to.
_DIMENSIONS: dict[str, str] = {
    "camera": "s.camera_id",
    "sector": "c.sector",
    "road": "c.road",
    "vehicle_class": "s.vehicle_class",
    "direction": "s.direction",
    "hour": "CAST(strftime('%H', s.ts, 'unixepoch', 'localtime') AS INTEGER)",
    "weekday": "CAST(strftime('%w', s.ts, 'unixepoch', 'localtime') AS INTEGER)",
    "date": "strftime('%Y-%m-%d', s.ts, 'unixepoch', 'localtime')",
}


def aggregate_traffic(group_by: tuple[str, ...] | list[str] = ("camera",),
                      minutes: float | None = None, k: int | None = None,
                      since: float | None = None, until: float | None = None,
                      camera_source_only: bool = True,
                      net: CameraNetwork | None = None,
                      conn: sqlite3.Connection | None = None) -> dict:
    """Traffic statistics with any group of fewer than k vehicles suppressed.

    This is the function to hand a transport department, a researcher or a
    press office. It has no plate parameter and cannot be given one - a
    k-anonymous query with an identity filter is a lookup, and offering the
    argument at all would guarantee someone uses it and calls the result
    anonymised.

    k counts distinct vehicles, not sightings. The distinction is the whole
    control: one car passing a quiet junction forty times in a day is forty
    rows and one person, and a threshold on row count would publish that
    person's evening as a statistic.

    Complementary suppression is applied. If exactly one group falls below k,
    its size is recoverable by subtracting every published group from the
    total, so the next smallest group is suppressed alongside it. That costs a
    real cell of data and is the price of the released total meaning anything.

    What it does not defend against: a caller who runs it repeatedly over
    sliding windows and differences the results. Nothing at this layer can -
    that needs query budgeting or noise, and either would be a bigger claim
    than this code can support. The access log is the mitigation that exists,
    which is why aggregate calls should be recorded too.
    """
    net = net or network()
    conn = _ensure_audit(conn)
    k = int(K_ANONYMITY if k is None else k)
    if k < 2:
        raise ValueError("k below 2 suppresses nothing - that is not anonymity")

    dims = [d for d in group_by]
    unknown = [d for d in dims if d not in _DIMENSIONS]
    if not dims or unknown:
        raise ValueError(f"group_by must be a non-empty subset of "
                         f"{sorted(_DIMENSIONS)}; got {unknown or dims}")

    anchor = db.latest_ts((config.CAMERA_SOURCE,), conn=conn)
    if until is None:
        until = anchor
    if since is None:
        since = until - float(minutes if minutes is not None else 24 * 60) * 60.0

    select = ", ".join(f"{_DIMENSIONS[d]} AS {d}" for d in dims)
    group = ", ".join(_DIMENSIONS[d] for d in dims)
    where = "s.ts >= ? AND s.ts <= ?"
    params: list = [float(since), float(until)]
    if camera_source_only:
        # Same reasoning as core/analytics.py: an operator enrolment is a
        # person filing a photograph, not the road being measured, and one of
        # them can set a thin group's entire statistic.
        where += " AND s.source = ?"
        params.append(config.CAMERA_SOURCE)

    raw = db.rows(
        f"SELECT {select}, COUNT(*) AS sightings, COUNT(DISTINCT s.plate) AS vehicles"
        f" FROM sightings s LEFT JOIN cameras c ON c.id = s.camera_id"
        f" WHERE {where} GROUP BY {group}", tuple(params), conn=conn)

    released = [r for r in raw if r["vehicles"] >= k]
    suppressed = [r for r in raw if r["vehicles"] < k]
    complementary = 0
    if len(suppressed) == 1 and released:
        released.sort(key=lambda r: (r["vehicles"], r["sightings"]))
        suppressed.append(released.pop(0))
        complementary = 1

    for r in released:
        if "camera" in dims:
            r["camera_name"] = net.name(r["camera"])
            lat, lon = (net.coords(r["camera"]) if r["camera"] in net.cameras
                        else (None, None))
            r["lat"], r["lon"] = lat, lon
    released.sort(key=lambda r: -r["sightings"])

    supp_sightings = sum(r["sightings"] for r in suppressed)
    total_sightings = sum(r["sightings"] for r in raw)
    return {
        "group_by": dims,
        "k": k,
        "since_ts": since, "until_ts": until,
        "since": _stamp(since), "until": _stamp(until),
        "groups": released,
        "groups_released": len(released),
        "groups_suppressed": len(suppressed),
        "complementary_suppressed": complementary,
        "released_sightings": total_sightings - supp_sightings,
        "suppressed_sightings": supp_sightings,
        "suppressed_share": (round(supp_sightings / total_sightings, 4)
                             if total_sightings else 0.0),
        # False when suppression bit hard enough that the released groups are
        # no longer a fair picture of the window - the caller should widen the
        # window or coarsen the grouping rather than publish this.
        "representative": (total_sightings == 0
                           or (total_sightings - supp_sightings) / total_sightings >= 0.9),
        "contains_identifiers": False,
        "basis": (f"{len(raw)} groups, {len(suppressed)} below k={k} distinct vehicles"
                  + (" including one suppressed to protect the total"
                     if complementary else "")),
    }


def pseudonym(plate: str, conn: sqlite3.Connection | None = None) -> str:
    """A stable keyed digest of a registration, for linking rows without naming one.

    Use when an export needs to say "the same vehicle appeared in both groups"
    without saying which vehicle. Keyed with a salt generated once per database
    and stored in `meta`, so digests are comparable inside one deployment and
    meaningless outside it.

    This is pseudonymisation and not anonymisation, and the difference is not
    pedantry. Indian registration numbers are a small enumerable space - anyone
    holding the salt can hash every plausible plate in minutes and invert the
    whole table. It protects a leaked export; it does not protect against
    someone who leaked the database that holds the salt.
    """
    conn = _ensure_audit(conn)
    salt = (getattr(config, "PRIVACY_PSEUDONYM_SALT", "")
            or os.getenv("GODSEYE_PSEUDONYM_SALT", "")
            or _db_salt(conn))
    plate = plate.upper().replace(" ", "")
    return hmac.new(salt.encode(), plate.encode(), hashlib.sha256).hexdigest()[:16]


def _db_salt(conn: sqlite3.Connection) -> str:
    """Per-database salt, created on first use and kept in the `meta` table."""
    row = db.rows("SELECT value FROM meta WHERE key = 'privacy.pseudonym_salt'", conn=conn)
    if row:
        return row[0]["value"]
    salt = secrets.token_hex(16)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 ("privacy.pseudonym_salt", salt))
    conn.commit()
    return salt


# --- posture ------------------------------------------------------------
def posture(days: float = 7.0, retention_days: float | None = None,
            conn: sqlite3.Connection | None = None,
            net: CameraNetwork | None = None) -> dict:
    """One call that says what this system is holding, and on what authority.

    Built to be read by someone who does not trust the platform: an oversight
    board, a judge, a journalist. Every figure is a count from the database
    rather than a policy statement, and where a control is weak the report says
    so in `caveats` instead of leaving it for them to find.
    """
    conn = _ensure_audit(conn)
    net = net or network()
    plan = retention_preview(days=retention_days, conn=conn)
    access = access_summary(days=days, conn=conn)
    held = holds(conn=conn)
    st = db.stats(conn=conn)

    watch = db.rows("SELECT plate, reason, severity, added_ts, added_by FROM watchlist"
                    " ORDER BY added_ts DESC", conn=conn)
    for w in watch:
        w["added"] = _stamp(w["added_ts"])
        w["days_listed"] = round((time.time() - w["added_ts"]) / 86400.0, 1)

    span_days = ((st["last_ts"] - st["first_ts"]) / 86400.0
                 if st["first_ts"] and st["last_ts"] else 0.0)
    caveats = []
    if plan["clock_drift_hours"] is not None and plan["clock_drift_hours"] > 24:
        caveats.append(f"the newest sighting is {plan['clock_drift_hours']:.0f}h old - "
                       f"this database is seeded or replayed, not live")
    if held["count"]:
        top = next(iter(held["by_alert_kind"].items()), None)
        if top and top[1] >= 0.25 * max(st["unique_plates"], 1):
            caveats.append(f"{top[1]} registrations are held from deletion by an "
                           f"unacknowledged '{top[0]}' alert - an unworked alert queue "
                           f"is acting as a retention policy")
    if access["lookups"] == 0:
        caveats.append("no audited lookups in the window - either nobody searched, or "
                       "a search path is reaching the data without calling record()")
    if not REQUIRE_PURPOSE:
        caveats.append("PRIVACY_REQUIRE_PURPOSE is off - lookups can be logged with no "
                       "stated reason, which makes the audit trail decorative")
    if span_days < plan["retention_days"]:
        caveats.append(f"the store spans {span_days:.1f} days against a "
                       f"{plan['retention_days']:g}-day horizon, so nothing is due for "
                       f"deletion yet - the purge is untested against this data")

    return {
        "generated": _stamp(time.time()),
        "store": {
            "sightings": st["sightings"], "unique_plates": st["unique_plates"],
            "cameras": st["cameras"], "alerts": st["alerts"],
            "open_alerts": st["open_alerts"],
            "oldest": _stamp(st["first_ts"]), "newest": _stamp(st["last_ts"]),
            "span_days": round(span_days, 2),
        },
        "retention": {
            "horizon_days": plan["retention_days"],
            "cutoff": plan["cutoff"],
            "rows_past_horizon": sum(t["past_horizon"] for t in plan["tables"]),
            "rows_deletable": plan["deletable_rows"],
            "rows_held_back": sum(t["held_back"] for t in plan["tables"]),
            "fraction_deletable": plan["deletable_fraction"],
            "per_table": plan["tables"],
            "audit_log_horizon_days": AUDIT_RETENTION_DAYS,
        },
        "holds": {
            "plates_held": held["count"],
            "watchlisted": held["watchlisted"],
            "by_open_alert": held["open_alert_plates"],
            "by_alert_kind": held["by_alert_kind"],
        },
        "access": {
            "window_days": days,
            "lookups": access["lookups"],
            "identified_lookups": access["identified_lookups"],
            "distinct_plates_searched": access["distinct_plates"],
            "distinct_actors": access["distinct_actors"],
            "lookups_without_case_ref": access["no_case_ref"],
            "top_actors": access["actors"][:5],
        },
        "watchlist": {"count": len(watch), "entries": watch,
                      "by_severity": dict(Counter(w["severity"] for w in watch))},
        "controls": {
            "purpose_required_on_lookup": REQUIRE_PURPOSE,
            "k_anonymity": K_ANONYMITY,
            "purge_requires_confirmation": True,
            "purge_safety_limit": MAX_PURGE_FRACTION,
            "hold_severities": list(HOLD_SEVERITIES),
            "audit_outlives_data": AUDIT_RETENTION_DAYS > RETENTION_DAYS,
        },
        "caveats": caveats,
        # Said plainly because the alternative is that a screenshot of this
        # report gets read as a certification.
        "not_claimed": [
            "no conformance with CJIS, the DPDP Act or any other named standard is "
            "asserted - those are findings about an organisation, not about code",
            "deletion frees database pages; it does not overwrite them, and backups, "
            "the WAL and filesystem snapshots are outside this module",
            "the audit trail covers callers that invoke record(); direct SQL against "
            "the database file leaves no trace here",
        ],
    }


def _stamp(ts: float | None) -> str | None:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else None


if __name__ == "__main__":     # pragma: no cover - operator console
    import argparse

    ap = argparse.ArgumentParser(description="GodsEye privacy and retention controls")
    ap.add_argument("--posture", action="store_true", help="what the system is holding")
    ap.add_argument("--preview", action="store_true", help="what a purge would delete")
    ap.add_argument("--purge", action="store_true", help="actually delete it")
    ap.add_argument("--days", type=float, default=None, help="retention horizon in days")
    ap.add_argument("--actor", default="", help="who is running the purge")
    ap.add_argument("--force", action="store_true", help="past the safety limit")
    args = ap.parse_args()

    if args.purge:
        out = retention_purge(days=args.days, confirm=True, actor=args.actor,
                              force=args.force)
    elif args.preview:
        out = retention_preview(days=args.days)
    else:
        out = posture(retention_days=args.days)
    print(json.dumps(out, indent=2, default=str))
