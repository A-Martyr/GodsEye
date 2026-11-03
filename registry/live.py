"""The live chain: a camera read becomes a registry answer, and maybe an incident.

    phone camera -> plate -> registry lookup -> assessment -> incident -> authority

The dependency runs ONE WAY, deliberately. Nothing in anpr/, core/, api/, sim/
or dashboard/ imports this package - registry/selftest.py asserts it - so this
module reaches into the platform rather than being called by it, and always
read-only. That is what keeps "delete registry/ and the demo is exactly as it
was" true, and it is why enrichment happens here instead of inside the camera
pipeline where a bug could stall a read.

Observed and registered information are kept apart in everything this module
returns. They are different kinds of claim: one is what a camera saw a moment
ago, the other is what an authority recorded years ago, and a panel that blends
them invites an operator to trust the weaker of the two.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config                                                    # noqa: E402
from registry import authority, sentinel, store                  # noqa: E402

# A plate already referred for the same reason is not referred again inside
# this window. Without it a board held up for a minute produces a queue of
# identical incidents and the console stops being readable.
REFER_DEDUP_S = 300.0

# Only these reach an authority automatically. Everything else is still stored
# and visible; it simply does not page anyone. A console that refers a Rs 500
# challan to a police control room teaches them to ignore it.
AUTO_REFER_SEVERITIES = ("critical", "high")


def live_sightings(hours: float = 6.0, limit: int = 60) -> list[dict]:
    """Recent LIVE-CAMERA reads from the platform. Read-only.

    Live reads are `sightings` rows whose source is one of the enrolment
    values - what core/enrol.py writes when a phone at a junction reads a
    plate. Simulated captures are excluded: this view is about the cameras in
    the room.
    """
    conn = store.platform_ro()
    if conn is None:
        return []
    try:
        newest = conn.execute("SELECT MAX(ts) t FROM sightings").fetchone()["t"]
        if newest is None:
            return []
        marks = ",".join("?" * len(config.ENROL_SOURCES))
        rows = conn.execute(
            f"SELECT ts, plate, camera_id, confidence, source, vehicle_class"
            f" FROM sightings WHERE source IN ({marks}) AND ts >= ?"
            f" ORDER BY ts DESC LIMIT ?",
            (*config.ENROL_SOURCES, newest - hours * 3600.0, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def enrich(plate: str, *, camera_id: str | None = None,
           observed: dict | None = None, hours: float = 24.0) -> dict:
    """One plate: what the camera saw, what the registry holds, what follows.

    The three stay in separate keys. `observed` is the camera's, `registry` is
    the authority's, `findings` is what only the two together imply.
    """
    plate = plate.upper().replace(" ", "")
    reg = store.cached_lookup(plate, actor="live",
                              purpose="live camera enrichment")
    move = sentinel.movement([plate], hours=hours).get(plate, {})
    if camera_id and not move.get("last_camera"):
        move["last_camera"] = camera_id

    findings = [f.as_dict() for f in sentinel.assess(plate, reg, move)]
    return {
        "plate": plate,
        "observed": observed or {},
        "movement": move,
        "registry": reg.as_dict(),
        "registry_status": reg.registry_status,
        "findings": findings,
    }


def _recently_referred(plate: str, kind: str, conn) -> bool:
    row = conn.execute(
        "SELECT ts FROM incidents WHERE plate = ? AND kind = ?"
        " ORDER BY ts DESC LIMIT 1", (plate, kind)).fetchone()
    return bool(row) and (time.time() - row["ts"]) < REFER_DEDUP_S


def process(plate: str, *, camera_id: str | None = None,
            camera_name: str | None = None, confidence: float | None = None,
            observed: dict | None = None, notify: bool = True,
            severities=AUTO_REFER_SEVERITIES) -> dict:
    """The whole chain for one read. Returns what happened at every step.

    Registry failure cannot stop this: enrich() returns UNKNOWN, assess()
    declines to conclude anything from it, and the function returns a result
    saying exactly that. Nothing raises into a caller.
    """
    out = enrich(plate, camera_id=camera_id, observed=observed)
    out["incidents"] = []

    if out["registry_status"] == "UNKNOWN":
        out["note"] = ("registry unreachable - the read stands, but nothing is "
                       "concluded about the vehicle")
        return out

    conn = store.connect()
    try:
        for f in out["findings"]:
            if f["severity"] not in severities:
                continue
            if _recently_referred(f["plate"], f["kind"], conn):
                continue
            f = dict(f)
            f.setdefault("camera_id", camera_id)
            f.setdefault("camera_name", camera_name)
            if camera_id:
                f["camera_id"] = f.get("camera_id") or camera_id
            if camera_name:
                f["camera_name"] = f.get("camera_name") or camera_name
            res = authority.refer(f, registry_status=out["registry_status"],
                                  confidence=confidence, notify=notify)
            out["incidents"].append({
                "ref": res["incident"]["ref"],
                "kind": res["incident"]["kind"],
                "severity": res["incident"]["severity"],
                "authority": res["incident"]["authority"],
                "authority_name": authority.authority_name(
                    res["incident"]["authority"]),
                "delivery": res["delivery"]["status"],
                "authority_ref": res["delivery"].get("authority_ref"),
                "attempts": res["delivery"].get("attempts"),
            })
    finally:
        conn.close()
    return out


def sweep_live(hours: float = 6.0, limit: int = 40, notify: bool = True) -> dict:
    """Run the chain over every live read in the window.

    This is what the console calls. One pass, deduplicated, so watching it does
    not manufacture a new incident every few seconds.
    """
    reads = live_sightings(hours=hours, limit=limit)
    seen: set = set()
    results = []
    for r in reads:
        key = (r["plate"], r["camera_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append(process(
            r["plate"], camera_id=r["camera_id"],
            camera_name=sentinel._camera_name(r["camera_id"]),
            confidence=r.get("confidence"),
            observed={"ts": r["ts"], "camera_id": r["camera_id"],
                      "confidence": r.get("confidence"),
                      "class_seen": r.get("vehicle_class"),
                      "source": r.get("source")},
            notify=notify))
    raised = [i for res in results for i in res["incidents"]]
    return {
        "reads": len(reads), "vehicles": len(results),
        "incidents": raised,
        "unknown": sum(1 for r in results if r["registry_status"] == "UNKNOWN"),
        "not_found": sum(1 for r in results if r["registry_status"] == "NOT_FOUND"),
        "results": results,
    }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Live camera -> registry -> authority")
    p.add_argument("plate", nargs="?", help="run the chain for one plate")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--no-notify", action="store_true")
    a = p.parse_args(argv)

    if a.plate:
        out = process(a.plate, notify=not a.no_notify)
        r = out["registry"]
        print(f"\n{out['plate']}  registry={out['registry_status']}")
        if r.get("found"):
            v = r["vehicle"]
            print(f"  {v['make']} {v['model']}  {v['colour']}  status={r['status']}")
        print(f"  findings: {len(out['findings'])}")
        for f in out["findings"]:
            print(f"    [{f['severity']}] {f['headline']}")
        for i in out["incidents"]:
            print(f"  -> {i['ref']} to {i['authority_name']}: {i['delivery']}"
                  + (f" ({i['authority_ref']})" if i["authority_ref"] else ""))
        return 0

    s = sweep_live(hours=a.hours, notify=not a.no_notify)
    print(f"\n{s['reads']} live reads, {s['vehicles']} vehicles, "
          f"{len(s['incidents'])} incident(s) raised")
    print(f"  registry unknown: {s['unknown']}   not found: {s['not_found']}")
    for i in s["incidents"]:
        print(f"  {i['ref']}  [{i['severity']}] {i['kind']} -> "
              f"{i['authority_name']}: {i['delivery']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
