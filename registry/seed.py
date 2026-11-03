"""Populate the mock registry.

    python -m registry.seed              # register every plate the cameras know
    python -m registry.seed --extra 500  # plus unseen vehicles, so the registry
                                         # is bigger than the corpus, as a real
                                         # one would be
    python -m registry.seed --reset

Plates come from the platform's store, opened READ-ONLY, plus optional
invented extras. Registering exactly the plates the cameras have seen is what
lets the console join the two together; the extras exist so the registry is not
suspiciously a perfect mirror of the sightings table.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registry import settings, store                          # noqa: E402


def observed_classes() -> dict[str, str]:
    """{plate: the class the cameras most often recorded}. Read-only.

    The registry agrees with the cameras for all but a deliberate few, so the
    class-mismatch rule downstream means something when it does fire.
    """
    conn = store.platform_ro()
    if conn is None:
        print(f"[registry] no platform database at {settings.PLATFORM_DB} - "
              "seeding extras only", file=sys.stderr)
        return {}
    try:
        rows = conn.execute(
            "SELECT plate, vehicle_class, COUNT(*) n FROM sightings"
            " WHERE vehicle_class IS NOT NULL"
            " GROUP BY plate, vehicle_class ORDER BY plate, n DESC").fetchall()
    finally:
        conn.close()
    best: dict[str, str] = {}
    for r in rows:
        if registrable(r["plate"]):
            best.setdefault(r["plate"], r["vehicle_class"])
    return best


def registrable(plate: str) -> bool:
    """Could a real authority have issued this registration?

    The sightings table holds what the OCR engine READ, misreads included - so
    seeding straight from it registers strings like 6J45N1463 as though they
    were vehicles, and gives them an invented "6J RTO 45" office in a state
    that does not exist.

    That is not a cosmetic problem. "This plate is not in the registry" is one
    of the more useful things the console can say - it means a misread or a
    fabricated plate - and the signal is worthless if the registry has quietly
    issued papers to every misread the cameras ever produced.
    """
    from anpr.ocr import is_bharat_series, matches_plate_grammar
    from anpr.plates import STATE_CODES

    plate = (plate or "").upper().replace(" ", "")
    if not plate:
        return False
    if is_bharat_series(plate):          # 22BH1234AA carries no state code
        return True
    return matches_plate_grammar(plate) and plate[:2] in STATE_CODES


def _invented_plates(n: int, known: set[str]) -> list[str]:
    """Registrations no camera has seen, so the registry is bigger than the corpus."""
    from anpr.plates import random_plate
    rng = random.Random(20260911)
    out: list[str] = []
    while len(out) < n:
        p = random_plate(rng)
        if p not in known:
            known.add(p)
            out.append(p)
    return out


def seed(extra: int = 400, reset: bool = False, verbose: bool = True) -> dict:
    now = time.time()
    conn = store.connect()
    if reset:
        conn.executescript("DELETE FROM cases; DELETE FROM vehicles; DELETE FROM meta;")
        conn.commit()

    seen = observed_classes()
    plates = dict(seen)
    for p in _invented_plates(extra, set(plates)):
        plates[p] = None

    already = {r[0] for r in conn.execute("SELECT plate FROM vehicles").fetchall()}
    todo = [(p, k) for p, k in plates.items() if p not in already]
    if verbose:
        print(f"[registry] {len(seen)} plates from the cameras, {extra} invented, "
              f"{len(todo)} to register ({len(already)} already there)")

    made = 0
    for plate, klass in todo:
        rec = store.build_record(plate, klass, now)
        cases = store.build_cases(plate, rec, now)
        conn.execute(
            "INSERT OR REPLACE INTO vehicles (plate, vehicle_class, make, model,"
            " fuel, colour, manufactured, engine_no, chassis_no, rto_code,"
            " rto_office, rto_state, registered_on, insurance_to, insurer,"
            " puc_to, fitness_to, permit_to, permit_type, tax_to, status,"
            " status_note, status_since)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec["plate"], rec["vehicle_class"], rec["make"], rec["model"],
             rec["fuel"], rec["colour"], rec["manufactured"], rec["engine_no"],
             rec["chassis_no"], rec["rto_code"], rec["rto_office"],
             rec["rto_state"], rec["registered_on"], rec["insurance_to"],
             rec["insurer"], rec["puc_to"], rec["fitness_to"], rec["permit_to"],
             rec["permit_type"], rec["tax_to"], rec["status"],
             rec["status_note"], rec["status_since"]))
        for c in cases:
            conn.execute(
                "INSERT INTO cases (plate, kind, section, description, amount,"
                " raised_on, station, authority, settled, reference)"
                " VALUES (?,?,?,?,?,?,?,?,0,?)",
                (rec["plate"], c["kind"], c["section"], c["description"],
                 c["amount"], c["raised_on"], c["station"], c["authority"],
                 c["reference"]))
        made += 1
        if verbose and made % 400 == 0:
            print(f"   {made}/{len(todo)}")

    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seeded_at', ?)",
                 (str(int(now)),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('synthetic', '1')")
    conn.commit()
    s = store.stats(conn=conn)
    conn.close()
    if verbose:
        print(f"[registry] {s['vehicles']:,} vehicles, {s['cases_open']:,} open cases, "
              f"Rs {s['outstanding']:,.0f} outstanding")
        print(f"[registry] status mix: {s['by_status']}")
        print(f"[registry] expired: insurance {s['insurance_expired']}, "
              f"PUC {s['puc_expired']}, fitness {s['fitness_expired']}")
        print(f"[registry] written to {settings.DB_PATH}")
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Seed the mock vehicle registry")
    p.add_argument("--extra", type=int, default=400,
                   help="invented registrations no camera has seen (default 400)")
    p.add_argument("--reset", action="store_true", help="empty it first")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    seed(extra=a.extra, reset=a.reset, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
