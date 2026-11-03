"""Turning a registration and a movement history into something worth acting on.

The platform already detects things about MOVEMENT: a clone, a loiterer, a
detour, an impossible leg. This module detects things about the VEHICLE - that
it is reported stolen, that its papers lapsed, that the registry says it is a
scooter while every camera saw a truck - and, more usefully, things that are
only visible when the two are put together.

Three rules that need both halves, and are the reason this file exists rather
than a column on the registry page:

  A scrapped vehicle cannot drive. If a registration was deregistered at a
  scrapping facility and a camera photographs it moving, either the scrappage
  was fraudulent or the plate is on another vehicle. Neither is visible to the
  registry alone, which only knows the paperwork, nor to the cameras alone,
  which only know something drove past.

  A stolen vehicle in motion has a direction and a next junction. "Stolen" is
  a database flag until you can say where it was ten minutes ago and where the
  road leads - which is core/forecast.py's job, not the registry's.

  A plate no registry has ever issued, seen repeatedly at real junctions, is
  either a misread the platform should stop trusting or a fabricated plate.

EVERY finding carries what it is NOT. A confidence, the evidence behind it,
and - where the platform's own accuracy could explain the finding instead - the
caveat that says so. That is the same discipline core/clones.py applies when it
refuses to call a misread a clone, and it matters more here, because these
findings name an owner and route to a police station.

Read-only. This module opens the platform's database through
store.platform_ro() and cannot write to it.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registry import settings, store                          # noqa: E402

# Who a finding is routed to. A console that sends everything to one queue is
# a console nobody reads.
AUTHORITIES = {
    "police": "State Police — control room",
    "traffic": "Traffic Police — enforcement cell",
    "transport": "Transport Department (RTO) — enforcement",
    "pollution": "Pollution Control Board",
    "platform": "GodsEye operations — data quality",
}

SEVERITY_ORDER = ["critical", "high", "medium", "low"]


@dataclass
class Finding:
    """One actionable insight about one registration."""
    kind: str
    severity: str
    plate: str
    headline: str
    detail: str
    authority: str
    evidence: list = field(default_factory=list)
    caveat: str = ""
    camera_id: str | None = None
    camera_name: str | None = None
    last_seen: float | None = None
    recommended: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["authority_name"] = AUTHORITIES.get(self.authority, self.authority)
        return d


# --- movement, read only ------------------------------------------------
def movement(plates, hours: float = 24.0) -> dict:
    """{plate: {sightings, cameras, last_ts, last_camera, classes}} — read-only.

    One query for all the plates rather than one per plate: the console runs
    this over every registration seen in a window, and a per-plate query would
    make the page cost proportional to traffic.
    """
    conn = store.platform_ro()
    if conn is None:
        return {}
    plates = [p.upper().replace(" ", "") for p in dict.fromkeys(plates)]
    if not plates:
        return {}
    try:
        newest = conn.execute("SELECT MAX(ts) t FROM sightings").fetchone()["t"]
        if newest is None:
            return {}
        since = newest - hours * 3600.0
        out: dict[str, dict] = {}
        CHUNK = 400                       # SQLite caps variables per statement
        for i in range(0, len(plates), CHUNK):
            batch = plates[i:i + CHUNK]
            marks = ",".join("?" * len(batch))
            for r in conn.execute(
                f"SELECT plate, COUNT(*) n, COUNT(DISTINCT camera_id) cams,"
                f" MAX(ts) last_ts FROM sightings"
                f" WHERE plate IN ({marks}) AND ts >= ? GROUP BY plate",
                    (*batch, since)):
                out[r["plate"]] = {"sightings": r["n"], "cameras": r["cams"],
                                   "last_ts": r["last_ts"], "last_camera": None,
                                   "classes": {}}
            for r in conn.execute(
                f"SELECT plate, camera_id, vehicle_class, ts FROM sightings"
                f" WHERE plate IN ({marks}) AND ts >= ? ORDER BY ts",
                    (*batch, since)):
                slot = out.get(r["plate"])
                if slot is None:
                    continue
                slot["last_camera"] = r["camera_id"]
                if r["vehicle_class"]:
                    slot["classes"][r["vehicle_class"]] = \
                        slot["classes"].get(r["vehicle_class"], 0) + 1
        return out
    finally:
        conn.close()


def seen_plates(hours: float = 24.0, limit: int = 4000) -> list[str]:
    """Registrations the cameras recorded in the window. Read-only."""
    conn = store.platform_ro()
    if conn is None:
        return []
    try:
        newest = conn.execute("SELECT MAX(ts) t FROM sightings").fetchone()["t"]
        if newest is None:
            return []
        rows = conn.execute(
            "SELECT plate, COUNT(*) n FROM sightings WHERE ts >= ?"
            " GROUP BY plate ORDER BY n DESC LIMIT ?",
            (newest - hours * 3600.0, limit)).fetchall()
        return [r["plate"] for r in rows]
    finally:
        conn.close()


def _camera_name(camera_id: str | None) -> str | None:
    if not camera_id:
        return None
    try:
        from core.network import network
        return network().name(camera_id)
    except Exception:
        return camera_id


# --- the rules ----------------------------------------------------------
def assess(plate: str, reg, move: dict | None, now: float | None = None) -> list:
    """Every finding for one registration. `reg` is a store.Registration."""
    now = time.time() if now is None else now
    move = move or {}
    seen = move.get("sightings", 0)
    cam = move.get("last_camera")
    cam_name = _camera_name(cam)
    last = move.get("last_ts")
    out: list[Finding] = []

    def add(kind, severity, headline, detail, authority, evidence,
            caveat="", recommended=""):
        out.append(Finding(
            kind=kind, severity=severity, plate=plate, headline=headline,
            detail=detail, authority=authority, evidence=evidence, caveat=caveat,
            camera_id=cam, camera_name=cam_name, last_seen=last,
            recommended=recommended))

    # --- the registry could not be asked --------------------------------
    # Nothing may be concluded from silence. Every rule below reasons from the
    # RECORD, and without one the only honest output is none: reporting "no
    # registration exists" here would turn an outage into an accusation against
    # every ordinary vehicle on the road, routed to the police.
    if getattr(reg, "registry_status", "FOUND") == "UNKNOWN":
        return out

    # --- not registered at all -----------------------------------------
    if not reg.found:
        if seen >= settings.UNREGISTERED_MIN_SIGHTINGS:
            add("unregistered", "high",
                f"{plate} is not on any registry",
                f"Read {seen} times across {move.get('cameras', 0)} junctions, but no "
                f"registration exists. Either the plate is fabricated, or the reader "
                f"is making the same mistake repeatedly.",
                "police",
                [f"{seen} sightings in the window",
                 f"{move.get('cameras', 0)} distinct junctions",
                 "no record in the vehicle registry"],
                caveat="A plate read wrongly the same way at several cameras "
                       "produces exactly this pattern. Check the crops before "
                       "treating it as a fabricated plate.",
                recommended="Verify against the stored crops, then refer for "
                            "physical check if the read is sound.")
        return out

    # --- status ---------------------------------------------------------
    if reg.status == "stolen":
        add("stolen", "critical",
            f"{plate} is reported stolen and is moving",
            (f"{reg.vehicle.get('make')} {reg.vehicle.get('model')}, "
             f"{reg.vehicle.get('colour')}. {reg.status_note or ''}").strip(),
            "police",
            [reg.status_note or "flagged stolen on the registry"]
            + ([f"last read at {cam_name}"] if cam_name else [])
            + ([f"{seen} sightings in the window"] if seen else []),
            recommended="Intercept. Alert the station holding the FIR and the "
                        "nearest patrol to the last junction.")

    elif reg.status == "scrapped" and seen:
        add("scrapped_but_moving", "critical",
            f"{plate} was scrapped, yet a camera photographed it",
            f"The registration was deregistered at a scrapping facility. "
            f"{reg.status_note or ''} A scrapped vehicle cannot lawfully be on the "
            f"road, so either the scrappage record is false or this plate is now "
            f"on a different vehicle.",
            "transport",
            [reg.status_note or "deregistered at an RVSF",
             f"{seen} sightings after the scrappage date"]
            + ([f"last read at {cam_name}"] if cam_name else []),
            caveat="A misread of a similar registration would look the same. "
                   "Confirm the read before acting.",
            recommended="Refer to the RTO that issued the certificate of deposit.")

    elif reg.status == "blacklisted" and seen:
        add("blacklisted", "high",
            f"{plate} is blacklisted and still in traffic",
            reg.status_note or "Registration blacklisted by the RTO.",
            "transport",
            [reg.status_note or "blacklisted", f"{seen} sightings in the window"],
            recommended="Enforcement check at the next junction on its route.")

    # --- papers ----------------------------------------------------------
    expired = reg.expired(now)
    if expired and seen:
        # Insurance is the one with a victim attached, so it is graded above
        # the rest rather than folded into a single "papers" finding.
        if "insurance" in expired:
            days = int((now - (reg.validity.get("insurance_to") or now)) / 86400)
            add("uninsured", "high",
                f"{plate} has been uninsured for {days} days",
                f"Third-party cover lapsed {days} days ago "
                f"({reg.validity.get('insurer') or 'insurer unknown'}). Driving "
                f"uninsured is an offence under MVA s.196 and leaves any person it "
                f"injures with no recourse.",
                "traffic",
                [f"insurance expired {days} days ago",
                 f"{seen} sightings since"],
                recommended="Challan under MVA s.196 at the next enforcement point.")
        rest = [e for e in expired if e != "insurance"]
        if rest:
            add("papers_lapsed", "medium",
                f"{plate} is driving on lapsed papers",
                "Out of date: " + ", ".join(rest) + ".",
                "transport" if ("fitness" in rest or "permit" in rest) else "pollution",
                [f"{e} expired" for e in rest] + [f"{seen} sightings in the window"],
                recommended="Notice to the registered owner; roadside check if "
                            "fitness or permit is involved.")

    # --- registry against reality ---------------------------------------
    classes = move.get("classes") or {}
    if classes and reg.vehicle.get("class"):
        observed = max(classes, key=classes.get)
        n = classes[observed]
        if (observed != reg.vehicle["class"]
                and n >= settings.MISMATCH_MIN_SIGHTINGS):
            add("class_mismatch", "high",
                f"{plate} is registered as a {reg.vehicle['class'].replace('_', ' ')} "
                f"but reads as a {observed.replace('_', ' ')}",
                f"The registry holds a {reg.vehicle.get('make')} "
                f"{reg.vehicle.get('model')} ({reg.vehicle['class'].replace('_', ' ')}). "
                f"The cameras classified this plate as {observed.replace('_', ' ')} "
                f"on {n} of {sum(classes.values())} sightings. A plate on the wrong "
                f"class of vehicle is the ordinary signature of a cloned or "
                f"transplanted plate.",
                "police",
                [f"registered: {reg.vehicle['class']}",
                 f"observed: {observed} on {n}/{sum(classes.values())} sightings"],
                caveat="Vehicle class here is assigned by the simulator, not by a "
                       "classifier reading the image, so on this build the "
                       "disagreement is only as good as that label.",
                recommended="Cross-check against the clone verdict for this plate.")

    # --- money -----------------------------------------------------------
    pending = reg.pending_cases
    if pending:
        firs = [c for c in pending if c["kind"] == "fir"]
        if firs and reg.status != "stolen":
            from registry.catalogue import FIR_CRITICAL
            # Graded by the offence, not by the fact that an FIR exists. A
            # narcotics or theft case on a vehicle currently in traffic is
            # worth interrupting someone for; a forgery case from last year is
            # not, and grading them alike is how a critical queue becomes
            # wallpaper nobody reads.
            urgent = [c for c in firs if c["section"] in FIR_CRITICAL]
            add("open_fir", "critical" if urgent else "high",
                f"{plate} has {len(firs)} open criminal case(s)",
                "; ".join(f"{c['section']} — {c['description']}" for c in firs[:3]),
                "police",
                [f"{c['reference']}: {c['section']} {c['description']} "
                 f"({c['station']})" for c in firs[:4]],
                recommended=("Notify the investigating station that the vehicle "
                             "is in traffic." if urgent else
                             "Flag to the investigating station at the next "
                             "routine update."))
        if reg.outstanding_amount >= settings.CHALLAN_ALERT_AMOUNT:
            add("challans_outstanding", "medium",
                f"{plate} owes Rs {reg.outstanding_amount:,.0f} in unpaid challans",
                f"{len(pending)} pending item(s), oldest from "
                f"{time.strftime('%d %b %Y', time.localtime(min(c['raised_on'] for c in pending)))}.",
                "traffic",
                [f"{c['reference']}: {c['description']} Rs {c['amount']:,.0f}"
                 for c in sorted(pending, key=lambda c: -c["amount"])[:4]],
                recommended="Recovery notice; consider detention if it stays unpaid.")

    return out


def sweep(hours: float = 24.0, limit: int = 1500, now: float | None = None) -> dict:
    """Every finding across every registration the cameras saw in the window.

    This is what the console renders. One pass, one registry connection, one
    read-only platform connection.
    """
    now = time.time() if now is None else now
    plates = seen_plates(hours=hours, limit=limit)
    if not plates:
        return {"findings": [], "checked": 0, "hours": hours,
                "note": "no sightings in the window"}
    moves = movement(plates, hours=hours)
    conn = store.connect()
    try:
        regs = store.bulk_lookup(plates, actor="sentinel",
                                 purpose=f"routine sweep, {hours:g}h", conn=conn)
    finally:
        conn.close()

    findings: list[Finding] = []
    for p in plates:
        findings.extend(assess(p, regs[p], moves.get(p), now=now))

    findings.sort(key=lambda f: (SEVERITY_ORDER.index(f.severity)
                                 if f.severity in SEVERITY_ORDER else 9,
                                 -(f.last_seen or 0)))
    by_authority: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for f in findings:
        by_authority[f.authority] = by_authority.get(f.authority, 0) + 1
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
    # How many of those lookups came back UNKNOWN. Without this the console
    # cannot tell an empty queue from a blind one - and those look identical
    # on screen while meaning opposite things. A sweep during an outage finds
    # nothing precisely because it learned nothing, and saying "nothing
    # outstanding" there would be the single most misleading thing this page
    # could print.
    unknown = sum(1 for p in plates
                  if getattr(regs[p], "registry_status", "FOUND") == "UNKNOWN")
    return {
        "findings": [f.as_dict() for f in findings],
        "checked": len(plates), "hours": hours,
        "unknown": unknown, "blind": unknown == len(plates) and bool(plates),
        "by_authority": by_authority, "by_kind": by_kind,
        "by_severity": {s: sum(1 for f in findings if f.severity == s)
                        for s in SEVERITY_ORDER},
        "generated_at": now,
    }


def brief(plate: str, hours: float = 24.0, now: float | None = None) -> dict:
    """Everything about one registration: the record, its movement, its findings."""
    plate = plate.upper().replace(" ", "")
    reg = store.lookup(plate, actor="console", purpose="single plate brief")
    move = movement([plate], hours=hours).get(plate, {})
    return {
        "plate": plate,
        "registration": reg.as_dict(),
        "movement": move,
        "findings": [f.as_dict() for f in assess(plate, reg, move, now=now)],
    }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Registry sentinel — actionable insights")
    p.add_argument("plate", nargs="?", help="brief one registration")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--limit", type=int, default=1500)
    a = p.parse_args(argv)

    if a.plate:
        b = brief(a.plate, hours=a.hours)
        r = b["registration"]
        print(f"\n{a.plate}  —  {'found' if r['found'] else 'NOT REGISTERED'}")
        if r["found"]:
            v, o = r["vehicle"], r["owner"]
            print(f"  {v['make']} {v['model']}  {v['colour']}  {v['fuel']}  {v['manufactured']}")
            print(f"  owner {o['name']} (#{o['serial']}, {o['locality']})")
            print(f"  status {r['status']}   expired: {', '.join(r['expired']) or 'nothing'}")
            print(f"  pending cases {len(r['pending_cases'])}, Rs {r['outstanding_amount']:,.0f}")
        print(f"  movement: {b['movement'].get('sightings', 0)} sightings")
        for f in b["findings"]:
            print(f"\n  [{f['severity'].upper()}] {f['headline']}")
            print(f"     -> {f['authority_name']}")
            if f["caveat"]:
                print(f"     caveat: {f['caveat']}")
        return 0

    s = sweep(hours=a.hours, limit=a.limit)
    print(f"\nchecked {s['checked']} registrations over {s['hours']:g}h")
    print(f"  by severity: {s.get('by_severity')}")
    print(f"  by authority: {s.get('by_authority')}")
    print(f"  by kind: {s.get('by_kind')}")
    for f in s["findings"][:12]:
        print(f"\n  [{f['severity'].upper():8s}] {f['headline']}")
        print(f"     -> {f['authority_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
