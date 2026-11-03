"""Checks for the registry package — behaviour, and the isolation promises.

    python -m registry.selftest

The isolation checks matter more than the behaviour ones. This package was
added on the condition that the working demo is not disturbed, and a promise
nobody re-checks is not a guarantee. These assert it: that no existing module
imports this one, that the platform's database is genuinely unwritable from
here, and that the registry keeps its own storage and port.

Uses a scratch database, never data/registry.db, so running it does not
disturb a seeded registry either.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_passed = 0
_failed = 0


def check(name: str, ok: bool, note: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  PASS  {name}" + (f"  ({note})" if note else ""))
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  ({note})" if note else ""))


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    from registry import catalogue as cat
    from registry import sentinel, settings, store
    from registry.seed import registrable

    now = time.time()
    scratch = Path(tempfile.gettempdir()) / "godseye_registry_selftest.db"
    if scratch.exists():
        scratch.unlink()
    conn = store.connect(scratch)

    # ---------------------------------------------------------------
    section("isolation — the promises this package was added under")

    import re
    import subprocess

    # Matched in Python rather than by shelling out to grep, after the shell
    # version was found to be silently vacuous: `\s` does not survive Python's
    # subprocess quoting to Git-for-Windows grep, so the pattern matched
    # nothing and the check passed no matter what the files contained. A test
    # that cannot fail is worse than no test, because it is also a claim.
    #
    # The trailing group is what stops `import registry_link` - the dashboard's
    # own adapter - from counting as a direct import of the package.
    IMPORTS_REGISTRY = re.compile(
        r"^\s*(?:from\s+registry(?:\.\w+)*\s+import"
        r"|import\s+registry(?:\.\w+)*\s*(?:as\b|#|$))", re.M)

    def importers(*folders) -> list[str]:
        out = []
        for folder in folders:
            for path in sorted((ROOT / folder).rglob("*.py")):
                try:
                    if IMPORTS_REGISTRY.search(path.read_text(encoding="utf-8",
                                                              errors="replace")):
                        out.append(path.relative_to(ROOT).as_posix())
                except OSError:
                    pass
        return out

    # The pattern is checked against a known positive before it is trusted, so
    # a future edit that breaks it fails loudly here instead of turning both
    # checks below back into decoration.
    check("the isolation check can actually detect an import",
          bool(IMPORTS_REGISTRY.search("    from registry import store"))
          and bool(IMPORTS_REGISTRY.search("import registry.store as s"))
          and not IMPORTS_REGISTRY.search("import registry_link"),
          "matches real imports, ignores registry_link")

    # The ENGINE stays hard-isolated. Nothing on the path a plate takes from a
    # camera to a stored sighting may know this package exists - that is what
    # keeps a slow or broken registry structurally incapable of costing a read
    # a millisecond.
    engine = importers("anpr", "core", "api", "sim")
    check("no engine module imports this package", not engine,
          ", ".join(engine) or "anpr/ core/ api/ sim/ are clean")

    # The CONTROL ROOM is allowed to, because the registry is meant to be
    # visible to an operator - but through exactly one adapter, so the coupling
    # stays something you can delete in one line.
    direct = importers("dashboard")
    check("the dashboard reaches the registry through one adapter only",
          direct == ["dashboard/registry_link.py"],
          ", ".join(direct) or "nothing imports it — the adapter is missing")

    # And the promise that actually matters, executed rather than asserted:
    # with `registry` unimportable, the adapter must load, report itself
    # unavailable, and render nothing - not raise. "No import statement exists"
    # was always a proxy for this; now that the dashboard does import it, the
    # proxy is gone and the real property has to be tested directly.
    probe = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            class Block:
                def find_module(self, name, path=None):
                    return self if name.split('.')[0] == 'registry' else None
                def find_spec(self, name, path=None, target=None):
                    if name.split('.')[0] == 'registry':
                        raise ImportError('registry removed (simulated)')
                    return None
            sys.meta_path.insert(0, Block())
            sys.path.insert(0, 'dashboard')
            import registry_link as r
            assert r.available() is False, 'reported available with no registry'
            assert r.lookup('WB24AB1234') is None
            assert r.lookup_many(('WB24AB1234',)) == {}
            assert r.findings('WB24AB1234') == []
            assert r.incidents_for('WB24AB1234') == []
            assert r.badge_html(None) == ''
            assert r.refer({}) is None
            r.clear()
            print('OK')
        """)], cwd=ROOT, capture_output=True, text=True)
    check("the dashboard survives registry/ being deleted",
          "OK" in probe.stdout,
          (probe.stderr.strip().splitlines() or ["no output"])[-1][:70])

    ro = store.platform_ro()
    if ro is None:
        check("platform database opens read-only", True, "no platform db present")
    else:
        try:
            ro.execute("INSERT INTO meta (key, value) VALUES ('selftest','x')")
            check("platform database is NOT writable from here", False,
                  "the insert succeeded - isolation is broken")
        except sqlite3.OperationalError as exc:
            check("platform database is NOT writable from here", True, str(exc)[:44])
        finally:
            ro.close()

    check("registry keeps its own database",
          settings.DB_PATH != settings.PLATFORM_DB,
          f"{settings.DB_PATH.name} vs {settings.PLATFORM_DB.name}")
    check("console port is clear of the platform's",
          settings.CONSOLE_PORT not in (8000, 8501, 8502, 8503, 8504, 8505),
          f"port {settings.CONSOLE_PORT}")

    # ---------------------------------------------------------------
    section("the registry")

    rec = store.build_record("WB24AB1234", "car", now)
    store_rec = dict(rec)
    for k in ("_observed_class", "_mismatch"):
        store_rec.pop(k, None)
    conn.execute(
        f"INSERT OR REPLACE INTO vehicles ({','.join(store_rec)}) "
        f"VALUES ({','.join('?' * len(store_rec))})", tuple(store_rec.values()))
    conn.commit()

    a = store.build_record("WB24AB1234", "car", now)
    b = store.build_record("WB24AB1234", "car", now)
    check("the same plate always yields the same vehicle",
          (a["make"], a["model"], a["engine_no"]) ==
          (b["make"], b["model"], b["engine_no"]),
          f"{a['make']} {a['model']}")

    c = store.build_record("WB24AB1235", "car", now)
    check("a different plate yields a different record",
          (a["engine_no"], a["make"]) != (c["engine_no"], c["make"]))

    check("no owner is held at all",
          not any("owner" in k for k in a),
          "this registry describes vehicles, not people")

    r = store.lookup("WB24AB1234", conn=conn)
    check("lookup returns a found registration", r.found and r.plate == "WB24AB1234")
    check("a found lookup says FOUND", r.registry_status == "FOUND")
    check("identifiers are masked in what lookup returns",
          "*" in r.vehicle["engine_no"] and "*" in r.vehicle["chassis_no"],
          f"{r.vehicle['engine_no']} / {r.vehicle['chassis_no']}")
    miss = store.lookup("ZZ99ZZ9999", conn=conn)
    check("a plate with no record answers not_registered", miss.found is False)
    check("and it says NOT_FOUND, not UNKNOWN",
          miss.registry_status == "NOT_FOUND")

    # The distinction the whole design turns on. An unreachable registry says
    # nothing about the vehicle, and must not be reported as though it had.
    section("an outage is not a verdict")
    settings.FORCE_MODE = "UNAVAILABLE"
    try:
        down = store.lookup("WB24AB1234")
        check("an unreachable registry answers UNKNOWN",
              down.registry_status == "UNKNOWN", down.status_note or "")
        check("UNKNOWN is not reported as 'not registered'",
              down.status != "not_registered")
        quiet = sentinel.assess("WB24AB1234", down,
                                {"sightings": 12, "cameras": 3})
        check("no finding is raised while the registry is down",
              quiet == [],
              "otherwise an outage accuses every ordinary vehicle on the road")
    finally:
        settings.FORCE_MODE = ""

    loud = sentinel.assess("ZZ99ZZ9999", store.lookup("ZZ99ZZ9999", conn=conn),
                           {"sightings": 12, "cameras": 3})
    check("but a genuinely unregistered plate still is",
          any(f.kind == "unregistered" for f in loud))
    check("every lookup is recorded",
          len(store.who_looked_up("WB24AB1234", conn=conn)) >= 1)

    check("a plate is registered in the state its code names",
          store.build_record("KA05MJ1234", "car", now)["rto_state"] not in ("West Bengal",),
          store.build_record("KA05MJ1234", "car", now)["rto_office"])

    check("an OCR misread is not registrable", not registrable("6J45N1463"))
    check("a real registration is registrable", registrable("WB24AB1234"))
    check("a Bharat-series plate is registrable", registrable("22BH1234AA"))

    # ---------------------------------------------------------------
    section("an incident reference is never reused")

    # An authority is given this reference and will quote it back. If deleting
    # a row can mint the same one again, two different incidents answer to one
    # reference - and because `ref` is UNIQUE, the collision arrives as a crash
    # rather than as a subtle mix-up. This was real: the reference used to be
    # COUNT(*) + 1.
    finding = {"plate": "WB24AB1234", "kind": "selftest", "severity": "low",
               "headline": "selftest incident", "authority": "police"}
    a = store.record_incident(finding, conn=conn)
    b = store.record_incident(finding, conn=conn)
    check("two incidents get different references", a["ref"] != b["ref"],
          f"{a['ref']} vs {b['ref']}")

    conn.execute("DELETE FROM incidents WHERE id = ?", (b["id"],))
    conn.commit()
    c_inc = store.record_incident(finding, conn=conn)
    check("a deleted incident does not free its reference for reuse",
          c_inc["ref"] not in (a["ref"], b["ref"]),
          f"after deleting {b['ref']}, the next was {c_inc['ref']}")
    check("no incident is left holding a placeholder reference",
          not conn.execute("SELECT 1 FROM incidents WHERE ref LIKE 'pending-%'"
                           ).fetchone(),
          "the temporary ref never survives the insert")

    # ---------------------------------------------------------------
    section("the sentinel")

    stolen = store.build_record("WB24AB1234", "car", now)
    stolen.update(status="stolen", status_note="Reported stolen, FIR 1/2026")
    reg = store.Registration(
        plate="WB24AB1234", found=True, vehicle={"class": "car", "make": "Tata",
                                                 "model": "Nexon", "colour": "White"},
        registration={},
        validity={"insurance_to": now + 86400, "puc_to": now + 86400},
        status="stolen", status_note="Reported stolen, FIR 1/2026", cases=[])
    found = sentinel.assess("WB24AB1234", reg, {"sightings": 5, "cameras": 3,
                                                "last_ts": now, "classes": {}})
    kinds = {f.kind for f in found}
    check("a stolen vehicle in motion is flagged", "stolen" in kinds)
    check("it is routed to the police",
          any(f.authority == "police" for f in found if f.kind == "stolen"))
    check("it is critical", any(f.severity == "critical"
                                for f in found if f.kind == "stolen"))

    lapsed = store.Registration(
        plate="WB24AB1111", found=True,
        vehicle={"class": "car"}, registration={},
        validity={"insurance_to": now - 30 * 86400, "puc_to": now - 5 * 86400},
        status="active", cases=[])
    found = sentinel.assess("WB24AB1111", lapsed, {"sightings": 3, "cameras": 2,
                                                   "last_ts": now, "classes": {}})
    check("an uninsured vehicle is flagged separately from other papers",
          {"uninsured", "papers_lapsed"} <= {f.kind for f in found})

    clean = store.Registration(
        plate="WB24AB2222", found=True, vehicle={"class": "car"},
        registration={}, validity={"insurance_to": now + 86400,
                                   "puc_to": now + 86400},
        status="active", cases=[])
    check("a clean registration produces nothing",
          sentinel.assess("WB24AB2222", clean,
                          {"sightings": 4, "cameras": 2, "last_ts": now,
                           "classes": {"car": 4}}) == [])

    # NOT_FOUND stated explicitly: the registry was asked and had no record.
    # Leaving it off defaults to UNKNOWN - "we could not ask" - and the sentinel
    # correctly refuses to conclude anything from that, which is what the
    # "an outage is not a verdict" section above asserts.
    missing = store.Registration(plate="ZZ99ZZ9999", found=False,
                                 registry_status="NOT_FOUND",
                                 status="not_registered")
    few = sentinel.assess("ZZ99ZZ9999", missing, {"sightings": 1, "cameras": 1,
                                                  "last_ts": now, "classes": {}})
    many = sentinel.assess("ZZ99ZZ9999", missing,
                           {"sightings": settings.UNREGISTERED_MIN_SIGHTINGS + 2,
                            "cameras": 3, "last_ts": now, "classes": {}})
    check("one sighting of an unknown plate is not a finding", few == [],
          "a single misread must not accuse anyone")
    check("repeated sightings of an unknown plate are", len(many) == 1)
    check("and it carries the misread caveat",
          bool(many and many[0].caveat), "the platform's own error rate explains it")

    mismatch = store.Registration(
        plate="WB24AB3333", found=True, vehicle={"class": "two_wheeler"},
        registration={},
        validity={"insurance_to": now + 86400, "puc_to": now + 86400},
        status="active", cases=[])
    found = sentinel.assess("WB24AB3333", mismatch,
                            {"sightings": 6, "cameras": 3, "last_ts": now,
                             "classes": {"truck": 6}})
    check("a class mismatch is flagged", "class_mismatch" in {f.kind for f in found})
    check("and it says what else could explain it",
          all(f.caveat for f in found if f.kind == "class_mismatch"))

    # ---------------------------------------------------------------
    section("severity is meaningful")

    weights = {c[0]: 0 for c in cat.CASE_TYPES}
    for c in cat.CASE_TYPES:
        weights[c[0]] += c[5]
    total = sum(weights.values())
    check("criminal cases are rarer than traffic offences",
          weights["fir"] / total < 0.10,
          f"FIR weight {weights['fir']}/{total} = {weights['fir']/total:.0%}")
    check("only serious offences make an FIR critical",
          cat.FIR_CRITICAL and "IPC 420" not in cat.FIR_CRITICAL,
          f"critical: {sorted(cat.FIR_CRITICAL)}")

    conn.close()
    scratch.unlink(missing_ok=True)

    print(f"\n{_passed} passed, {_failed} failed\n")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
