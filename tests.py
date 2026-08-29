"""End-to-end self-check.

    python tests.py

Walks the whole platform in one pass — reads plates, writes sightings,
reconstructs a trajectory through a deliberately misread plate, computes
analytics, fires every alert rule — against a scratch database, so it never
touches the demo data. Run it before a demo: if this passes, every layer works.
"""
from __future__ import annotations

import random
import sys
import tempfile
import time
from pathlib import Path

import config

FAILED: list[str] = []
PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    # Point the store at a scratch file before anything opens a connection.
    tmp = Path(tempfile.mkdtemp(prefix="godseye-test-")) / "test.db"
    config.DB_PATH = tmp

    from anpr import ocr, plates, segment
    from core import alerts as alert_rules
    from core import analytics, db, trajectory
    from core.network import haversine_km, network

    print("GodsEye self-check\n")
    net = network()

    print(f"network ({net.city})")
    first = next(iter(net.cameras))
    check("cameras loaded", len(net) >= 20, f"{len(net)} cameras")
    check("graph is connected",
          all(net.route(first, c) for c in net.cameras if c != first))
    south = min(net.cameras.values(), key=lambda c: c["lat"])["id"]
    north = max(net.cameras.values(), key=lambda c: c["lat"])["id"]
    check("routing returns road distance", net.route_km(south, north) > 10,
          f"{net.name(south)} to {net.name(north)}: {net.route_km(south, north):.1f} km")
    check("bearings point the right way", "N" in net.direction(south, north),
          f"{net.direction(south, north)} northbound")
    # A link cannot be shorter than the straight line between its cameras, and a
    # city road is rarely more than ~1.9x it.
    bad = []
    for u, v, d in net.graph.edges(data=True):
        (la, lo), (lb, lob) = net.coords(u), net.coords(v)
        straight = haversine_km(la, lo, lb, lob)
        ratio = d["km"] / max(straight, 0.01)
        if ratio < 0.95 or ratio > 1.95:
            bad.append(f"{u}-{v} {ratio:.2f}x")
    check("link lengths are geometrically possible", not bad, ", ".join(bad[:4]) or
          f"{net.graph.number_of_edges()} links checked")
    shaped = [d for *_, d in net.graph.edges(data=True) if d.get("shape")]
    check("roads carry shape geometry", len(shaped) >= 5,
          f"{len(shaped)} links bend through intermediate points")
    check("routed polyline follows that geometry",
          len(net.polyline(south, north)) > len(net.route(south, north)))

    print("\nANPR engine")
    eng = ocr.engine()
    rng = random.Random(7)
    clean = plates.capture("KA05MJ1234", "clean", rng, severity=0.3)
    read = eng.read(clean.image)
    check("reads a clean plate", read.text == "KA05MJ1234", read.text)
    check("confidence is high on a clean plate", read.confidence > 0.8,
          f"{read.confidence:.2f}")
    check("plate formatting", plates.pretty("KA05MJ1234") == "KA 05 MJ 1234")

    hard = [plates.capture(None, c, rng) for c in plates.CONDITIONS for _ in range(4)]
    reads = [eng.read(h.image) for h in hard]
    hits = sum(r.text == h.text for r, h in zip(reads, hard))
    check("reads degraded plates", hits >= 0.75 * len(hard),
          f"{hits}/{len(hard)} across all conditions")
    stored = [(r, h) for r, h in zip(reads, hard) if r.confidence >= config.MIN_PLATE_CONFIDENCE]
    kept_ok = sum(r.text == h.text for r, h in stored)
    check("confidence floor filters bad reads", kept_ok / max(len(stored), 1) >= hits / len(hard),
          f"{kept_ok}/{len(stored)} of accepted reads correct")
    check("segmentation finds characters",
          len(segment.segment(segment.binarize(segment.normalize(clean.image)))) >= 8)
    check("confusion distance discounts real OCR errors",
          ocr.confusion_distance("KA05MJ1234", "KA05NJ1234") < 1.0        # M/N confusable
          and ocr.confusion_distance("KA05MJ1234", "KA05WJ1234") == 1.0)  # M/W is not

    print("\nstore + simulator")
    from sim.city import CitySimulator
    db.init(cameras=[c for c in net.cameras.values()])
    sim = CitySimulator(net, seed=3, fleet_size=60)
    now = time.time()
    n = sim.history(hours=3.0, end_ts=now, ocr_fraction=0.0, verbose=False)
    check("simulator produced sightings", n > 200, f"{n} rows")
    stats = db.stats()
    check("sightings are stored", stats["sightings"] == n)
    check("many distinct plates seen", stats["unique_plates"] > 40,
          f"{stats['unique_plates']}")

    busiest = db.rows("SELECT plate, COUNT(*) c FROM sightings GROUP BY plate"
                      " ORDER BY c DESC LIMIT 1")[0]
    plate = busiest["plate"]

    print("\ntrajectory")
    traj = trajectory.reconstruct(plate)
    check("trajectory reconstructed", traj.summary["found"] and len(traj.legs) >= 1,
          f"{plate}: {traj.summary['sightings']} sightings, {len(traj.legs)} legs")
    check("legs carry distance, time and heading",
          all(l.road_km >= 0 and l.minutes > 0 and l.direction for l in traj.legs))
    check("legs follow the road graph",
          all(len(l.polyline) >= 2 for l in traj.legs))
    misread = plate[:4] + ("O" if plate[4] == "0" else "0") + plate[5:]
    found = trajectory.search(misread, limit=5)
    check("fuzzy search finds a misread plate",
          any(m["plate"] == plate for m in found), f"queried {misread}")

    print("\nanalytics")
    d = analytics.camera_density(180)
    check("camera density computed", not d.empty and d["count"].sum() > 0,
          f"{int(d['count'].sum())} reads over {int((d['count'] > 0).sum())} cameras")
    lf = analytics.link_flows(180)
    check("link flows computed", not lf.empty, f"{len(lf)} links")
    check("link flows only use real road links",
          all(net.graph.has_edge(a, b) for a, b in zip(lf["from_camera"], lf["to_camera"])))
    check("congestion levels assigned", set(lf["level"]) <= {"free", "moderate", "heavy", "severe"})
    bn = analytics.bottlenecks(180)
    check("bottlenecks ranked by delay", bn.empty or bn["delay_min_total"].is_monotonic_decreasing)
    odm = analytics.od_matrix(180)
    check("origin-destination matrix built", not odm.empty, f"{len(odm)} pairs")
    hm = analytics.heatmap(180)
    check("heatmap points generated", not hm.empty, f"{len(hm)} points")
    summ = analytics.city_summary(180)
    check("city summary complete",
          summ["sightings"] > 0 and summ["total_cameras"] == len(net))

    print("\nalerts")
    db.add_watch(plate, "self-check target", "critical")
    last = db.sightings_for_plate(plate)[-1]
    fired = alert_rules.evaluate({**last, "ts": time.time()}, net=net)
    check("watchlist hit fires", any(a["kind"] == "watchlist" for a in fired),
          ", ".join(a["kind"] for a in fired) or "nothing fired")
    cloned = sim.inject_clone()
    check("cloned plate detected",
          any(a["kind"] == "clone" for a in db.alerts(limit=50) if a["plate"] == cloned), cloned)
    loiter = sim.inject_loiterer()
    check("loitering detected",
          any(a["kind"] == "loitering" for a in db.alerts(limit=50) if a["plate"] == loiter),
          loiter)
    check("alerts are de-duplicated",
          len(alert_rules.evaluate({**last, "ts": time.time()}, net=net)) == 0)

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
