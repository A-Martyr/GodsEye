"""End-to-end tests against the RUNNING services, before integrating anything.

    python -m registry.integrationtest

registry/selftest.py tests the logic in isolation and needs nothing running.
This is the other half: it drives the real HTTP surface of registry/service.py,
flips the demo switches, and checks what actually comes back - because the
properties that matter here are about failure, and failure is exactly what a
unit test with a stubbed client will tell you works fine.

The fifth test is the one the whole exercise is for. Every other test breaks
something on purpose; this one asserts that the cameras kept running while it
happened. A registry outage that stops plate reads is not an outage, it is an
outage plus a new failure the platform invented for itself.

Nothing here writes to data/godseye.db. Incidents land in data/registry.db,
which is this package's own file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests                                                   # noqa: E402

from registry import authority, settings, store                   # noqa: E402

BASE = settings.SERVICE_URL
PASS, FAIL = [], []

# The camera side of the demo. Checked between every step, not just at the end:
# "still up afterwards" is a much weaker claim than "never went down".
CAMERA_PROBES = [
    ("phonecam hub", "http://127.0.0.1:8505/api/state"),
    ("phonecam connections", "http://127.0.0.1:8505/api/connections"),
]


def check(label: str, ok: bool, note: str = "") -> bool:
    (PASS if ok else FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({note})" if note else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


def get(path: str, **kw):
    return requests.get(f"{BASE}{path}", timeout=5, **kw)


def post(path: str, **kw):
    return requests.post(f"{BASE}{path}", timeout=8, **kw)


def registry_mode(mode: str) -> None:
    """Flip the mode in BOTH processes.

    The POST only reaches registry/service.py. `live.process()` runs in THIS
    process and reads its own settings, so a test that only posted would check
    the HTTP surface against a working local registry and quietly prove
    nothing. Setting both is what makes "this process cannot reach the
    registry" actually true here.
    """
    post(f"/api/v1/registry/mode/{mode}")
    settings.FORCE_MODE = "" if mode == "NORMAL" else mode
    store.cache_clear()          # a cached answer would outlive the switch


def authority_mode(mode: str) -> None:
    post(f"/api/v1/authority/mode/{mode}")


def cameras_alive(when: str) -> bool:
    """Are the cameras still serving? Called at every stage of every test."""
    alive = []
    for name, url in CAMERA_PROBES:
        try:
            alive.append(requests.get(url, timeout=4).status_code == 200)
        except Exception:
            alive.append(False)
    return check(f"cameras still serving {when}", all(alive),
                 f"{sum(alive)}/{len(alive)} probes")


def a_registered_plate() -> str:
    conn = store.connect()
    try:
        row = conn.execute(
            "SELECT plate FROM vehicles WHERE status = 'active' LIMIT 1").fetchone()
        return row["plate"] if row else ""
    finally:
        conn.close()


def a_stolen_plate() -> str:
    conn = store.connect()
    try:
        row = conn.execute(
            "SELECT plate FROM vehicles WHERE status = 'stolen' LIMIT 1").fetchone()
        return row["plate"] if row else ""
    finally:
        conn.close()


# --- 1: a lookup that works ---------------------------------------------
def test_lookup_succeeds() -> None:
    section("1. a successful lookup")
    plate = a_registered_plate()
    r = get(f"/api/v1/vehicles/{plate}")
    b = r.json()
    check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    check("says FOUND", b.get("registry_status") == "FOUND")
    check("carries the vehicle", bool(b.get("vehicle", {}).get("make")),
          f"{b.get('vehicle', {}).get('make')} "
          f"{b.get('vehicle', {}).get('model')}")
    check("carries registration and validity",
          bool(b.get("registration")) and "validity" in b)
    check("declares itself synthetic",
          b.get("synthetic") is True or "synthetic" in str(b).lower())
    check("same plate gives the same record twice",
          get(f"/api/v1/vehicles/{plate}").json().get("vehicle")
          == b.get("vehicle"), "deterministic from the plate hash")
    cameras_alive("after a normal lookup")


# --- 2: the registry cannot be reached -----------------------------------
def test_registry_unavailable() -> None:
    section("2. the registry is unavailable")
    plate = a_stolen_plate()
    registry_mode("UNAVAILABLE")
    try:
        r = get(f"/api/v1/vehicles/{plate}")
        b = r.json()
        check("HTTP 503, not 404", r.status_code == 503, f"got {r.status_code}")
        check("says UNKNOWN", b.get("registry_status") == "UNKNOWN")
        check("does NOT say the vehicle is unregistered", not b.get("found"),
              "found=False, but registry_status distinguishes it from NOT_FOUND")

        # The property the design exists for: an outage must not become a
        # finding. This plate IS stolen; with the registry down we must still
        # conclude nothing, because we did not actually learn that.
        from registry import live
        out = live.process(plate, notify=False)
        check("an outage produces no findings", out["findings"] == [],
              f"{len(out['findings'])} finding(s)")
        check("and no incident is referred", out["incidents"] == [])
        check("and it says why", "unreachable" in out.get("note", ""))
        # A cached UNKNOWN would outlive the outage: the registry would come
        # back and this process would keep answering "unknown" for the TTL.
        store.cache_clear()
        store.cached_lookup(plate, actor="test", purpose="cache check")
        check("an UNKNOWN answer is not cached",
              store.cache_stats().get("entries", 0) == 0,
              f"{store.cache_stats().get('entries', 0)} entries after an "
              f"UNKNOWN lookup")
        cameras_alive("while the registry is down")
    finally:
        registry_mode("NORMAL")
    r = get(f"/api/v1/vehicles/{plate}")
    check("recovers as soon as the registry returns",
          r.status_code == 200 and r.json().get("registry_status") == "FOUND")
    cameras_alive("after the registry recovers")


# --- 3: there is no such vehicle -----------------------------------------
def test_vehicle_not_found() -> None:
    section("3. the vehicle is not on the registry")
    r = get("/api/v1/vehicles/WB99ZZ9999")
    b = r.json()
    check("HTTP 404, not 503", r.status_code == 404, f"got {r.status_code}")
    check("says NOT_FOUND", b.get("registry_status") == "NOT_FOUND")
    check("found is False", b.get("found") is False)
    check("404 and 503 are different answers",
          b.get("registry_status") != "UNKNOWN",
          "'no such vehicle' is a fact; 'could not ask' is not")

    # One sighting of an unknown plate is a misread far more often than a
    # forgery, and the threshold is what stops the console accusing anyone.
    from registry import live
    out = live.process("WB99ZZ9999", notify=False)
    check("a plate absent from the registry is not automatically an incident",
          out["incidents"] == [],
          f"needs {settings.UNREGISTERED_MIN_SIGHTINGS} sightings first")
    cameras_alive("after a not-found lookup")


# --- 4: the authority cannot be reached ----------------------------------
def test_authority_unavailable() -> None:
    section("4. the authority is unavailable, then comes back")
    plate = a_stolen_plate()
    finding = {
        "plate": plate, "kind": "integration_test", "severity": "critical",
        "headline": f"{plate} — integration test referral",
        "detail": "raised by registry/integrationtest.py",
        "caveat": "This is a test incident, not a real finding.",
        "recommended": "No action. Test data.",
        "authority": "police", "camera_id": "CAM-20",
        "camera_name": "Esplanade", "last_seen": time.time(), "evidence": [],
    }

    authority_mode("OFFLINE")
    t0 = time.time()
    res = authority.refer(dict(finding), registry_status="FOUND")
    took = time.time() - t0
    ref, d = res["incident"]["ref"], res["delivery"]

    check("delivery fails", d["status"] == authority.FAILED, d["status"])
    check("after retrying", d["attempts"] == settings.DELIVERY_ATTEMPTS,
          f"{d['attempts']} attempts in {took:.1f}s")
    check("and the error is recorded", bool(d.get("error")), str(d.get("error"))[:60])

    # The whole reason the module records before it notifies.
    stored = store.incident_by_ref(ref)
    check("the incident SURVIVES the failed delivery", stored is not None)
    check("with its headline intact",
          bool(stored) and stored["headline"] == finding["headline"])
    check("marked as a DELIVERY problem, not a lost finding",
          bool(stored) and stored["delivery_status"] == authority.FAILED)
    cameras_alive("while the authority is unreachable")

    authority_mode("ONLINE")
    out = authority.retry_failed(limit=50)
    mine = [o for o in out if o["ref"] == ref]
    check("retry delivers it once the authority is back",
          bool(mine) and mine[0]["status"] == authority.ACKNOWLEDGED,
          mine[0].get("authority_ref") if mine else "not retried")
    again = store.incident_by_ref(ref)
    check("and the stored incident now carries their reference",
          bool(again) and bool(again["authority_ref"]),
          again["authority_ref"] if again else "")
    check("the authority confirms it received it",
          any(i["referral"]["incident_id"] == ref
              for i in get("/api/v1/authority/incidents").json()["incidents"]))
    cameras_alive("after the authority recovers")


# --- 5: the cameras never stopped ----------------------------------------
def test_cameras_unaffected() -> None:
    section("5. the camera pipeline was never affected")
    for name, url in CAMERA_PROBES:
        try:
            r = requests.get(url, timeout=4)
            check(f"{name} responding", r.status_code == 200, f"{r.status_code}")
        except Exception as exc:
            check(f"{name} responding", False, type(exc).__name__)

    try:
        state = requests.get("http://127.0.0.1:8505/api/state", timeout=4).json()
        cams = state.get("cameras") or state.get("junctions") or []
        check("junction cameras are configured", len(cams) >= 3,
              f"{len(cams)} junction(s)")
    except Exception as exc:
        check("junction cameras are configured", False, type(exc).__name__)

    # The isolation claim, restated as a test. The ENGINE is what matters here:
    # nothing on the path from a camera to a stored sighting imports this
    # package, so none of the failures above could have reached the cameras
    # even if it had wanted to. Nothing else imports it either (checked by
    # registry/selftest.py): the control room shows core/registry.py's mock.
    #
    # Matched in Python, not by shelling out. The grep form of this check was
    # silently vacuous: `\s` does not survive subprocess quoting to
    # Git-for-Windows grep, and a plain "import registry" substring also
    # matches any name that starts with it.
    import re
    pat = re.compile(r"^\s*(?:from\s+registry(?:\.\w+)*\s+import"
                     r"|import\s+registry(?:\.\w+)*\s*(?:as\b|#|$))", re.M)
    engine = [p.relative_to(Path(ROOT)).as_posix()
              for folder in ("anpr", "core", "api", "sim")
              for p in sorted((Path(ROOT) / folder).rglob("*.py"))
              if pat.search(p.read_text(encoding="utf-8", errors="replace"))]
    check("no engine module imports registry/", not engine,
          ", ".join(engine) or "anpr/ core/ api/ sim/ are clean")

    # And the platform's own store was never written.
    conn = store.platform_ro()
    check("platform database opens read-only", conn is not None)
    if conn:
        try:
            conn.execute("CREATE TABLE _x (a INT)")
            check("platform database REFUSES a write", False, "write succeeded")
        except Exception as exc:
            check("platform database REFUSES a write", True,
                  type(exc).__name__)
        finally:
            conn.close()


def main() -> int:
    print(f"\nGodsEye registry — integration tests against {BASE}")
    try:
        h = get("/api/v1/health").json()
        print(f"service up · {h['registry']['vehicles']:,} vehicles · "
              f"registry {h['force_mode']} · authority {h['authority_mode']}")
    except Exception as exc:
        print(f"\n  the registry service is not running on {BASE}\n"
              f"  start it with:  python -m registry.service\n  ({exc})")
        return 2

    for t in (test_lookup_succeeds, test_registry_unavailable,
              test_vehicle_not_found, test_authority_unavailable,
              test_cameras_unaffected):
        try:
            t()
        except Exception as exc:
            check(f"{t.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    # Leave the switches where the demo expects them, whatever happened above.
    registry_mode("NORMAL")
    authority_mode("ONLINE")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
