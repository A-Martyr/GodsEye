"""Query-time clone detection: is this registration on more than one vehicle?

Why this is not just "flag the impossible leg"
----------------------------------------------
`trajectory.reconstruct` already marks a leg implausible when the implied speed
exceeds `config.IMPOSSIBLE_SPEED_KMPH`, and the dashboard used to assert from
that alone that the plate "is probably running on more than one vehicle". On a
platform whose OCR is right about 73% of the time that claim is wrong a large
share of the time, and wrong in the worst direction: a misread merges two
different vehicles onto one registration and produces exactly the same
signature as a clone. Shipping that puts an accusation in front of an officer
built on nothing but the platform's own recognition error.

So the verdict here has to survive three alternative explanations before it
blames anyone.

1. *It was a misread.* If some other plate within OCR confusion distance was
   legitimately near that camera at that time, the sighting is better explained
   as that vehicle misread than as a clone of this one. This is the guard that
   matters most, and the platform already has the machinery for it -
   `confusion_distance` is what makes fuzzy plate search work.

2. *It was a weak read.* Evidence from a sighting the engine barely stood behind
   is not evidence. Both ends of a conflicting pair must clear
   `config.CLONE_MIN_CONFIDENCE`.

3. *It was a coincidence of geometry.* Two cameras metres apart, or a pair whose
   road distance is trivial, produce large implied speeds from ordinary timing
   jitter and a wrong clock. Pairs closer than `config.CLONE_MIN_KM` are ignored.

Only a plate that still has conflicting pairs after all three is reported, and
even then the verdict is graded: one surviving pair is `suspected`, several
independent ones are `confident`. An operator is never shown a bare accusation.

What it deliberately does not use
---------------------------------
Inferences. `core.repair` writes recovered plates to the `inferences` table and
never to `sightings`, and every query here reads `sightings`. A registration the
platform *guessed* can therefore never be the evidence that accuses someone. The
repair pass has the mirror-image rule - it skips plates already flagged as
clones - because it works by assuming plausible travel and would happily
reconstruct one coherent path from two real vehicles.

A fourth, additive signal: the vehicle registry
------------------------------------------------
`core/registry.py` is a mock, VAHAN-shaped lookup (synthetic - see its module
docstring). Once a conflict has survived the three guards above, this module
asks one further question: does the registry's `vehicle_class` for this plate
disagree with what *both* conflicting sightings recorded? `vehicle_class` is
the only vehicle attribute this platform ever observes at all (and only on the
simulated path - a real photograph read through `core/enrol.py` never sets it),
so it is the only registry field used as evidence here; `make`/`model`/`colour`
have nothing to be checked against and are not used.

The asymmetry matters: agreement between the registry and the cameras proves
nothing (two clones can easily share a vehicle class) and must never *lower* a
verdict, so it is never even inspected. Disagreement on both ends of an
otherwise-`suspected` conflict is independent corroborating evidence and
promotes that conflict's verdict to `confident` on its own, without needing a
second geographic leg. This is a deliberately narrower feature than "detect a
clone from a single sighting with no geometry at all" (`CHATGPT_BRIEF.md` §5
point 1) - it only strengthens a conflict `_pairs()` had already flagged as
geometrically impossible, which is the conservative scope call for a signal
that feeds an alert queue.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import config
from anpr.ocr import confusion_distance
from core import db
from core import registry
from core.network import CameraNetwork, network


@dataclass
class Conflict:
    """Two sightings of one plate that no single vehicle could have produced."""
    from_camera: str
    to_camera: str
    from_name: str
    to_name: str
    from_ts: float
    to_ts: float
    minutes: float
    road_km: float
    implied_kmph: float
    from_confidence: float
    to_confidence: float
    explained_by_misread: str = ""     # the plate it is more likely to have been
    from_vehicle_class: str | None = None
    to_vehicle_class: str | None = None
    attribute_conflict: bool = False   # both ends disagree with the mock registry

    @property
    def counts(self) -> bool:
        return not self.explained_by_misread


@dataclass
class CloneReport:
    plate: str
    verdict: str = "none"              # none | suspected | confident
    conflicts: list[Conflict] = field(default_factory=list)
    dismissed: list[Conflict] = field(default_factory=list)
    min_vehicles: int = 1
    reason: str = ""

    @property
    def is_clone(self) -> bool:
        return self.verdict in ("suspected", "confident")

    def as_dict(self) -> dict:
        return {"plate": self.plate, "verdict": self.verdict,
                "is_clone": self.is_clone, "min_vehicles": self.min_vehicles,
                "reason": self.reason,
                "conflicts": [asdict(c) for c in self.conflicts],
                "dismissed": [asdict(c) for c in self.dismissed]}


def _pairs(sightings: list[dict], net: CameraNetwork) -> list[Conflict]:
    """Sighting pairs that no single vehicle could have produced.

    Consecutive pairs only. A vehicle that is genuinely at A then B then C makes
    A->C look impossible whenever A->B is, so testing every pair would count one
    conflict many times and inflate a single anomaly into a confident verdict.
    """
    out: list[Conflict] = []
    for a, b in zip(sightings, sightings[1:]):
        if a["camera_id"] == b["camera_id"]:
            continue
        dt_min = (b["ts"] - a["ts"]) / 60.0
        if dt_min <= 0:
            continue
        km = net.route_km(a["camera_id"], b["camera_id"])
        if km < config.CLONE_MIN_KM:
            # Two cameras a few hundred metres apart - or two lanes of one
            # junction - turn ordinary clock skew into a huge implied speed.
            continue
        kmph = km / (dt_min / 60.0)
        if kmph <= config.IMPOSSIBLE_SPEED_KMPH:
            continue
        out.append(Conflict(
            from_camera=a["camera_id"], to_camera=b["camera_id"],
            from_name=net.name(a["camera_id"]), to_name=net.name(b["camera_id"]),
            from_ts=a["ts"], to_ts=b["ts"], minutes=dt_min, road_km=km,
            implied_kmph=kmph,
            from_confidence=float(a.get("confidence") or 0.0),
            to_confidence=float(b.get("confidence") or 0.0),
            from_vehicle_class=a.get("vehicle_class"),
            to_vehicle_class=b.get("vehicle_class")))
    return out


def _misread_alternative(plate: str, camera_id: str, ts: float, net: CameraNetwork,
                         conn) -> str:
    """A plate this sighting was more likely to have been, or "".

    The question is not "does a similar plate exist" - on a city network one
    always does. It is whether a similar plate was *plausibly at this camera at
    this time*, judged by its own sightings either side. If it was, then reading
    it as `plate` is a far more ordinary event than a cloned registration.
    """
    window = config.CLONE_MISREAD_WINDOW_S
    near = db.rows(
        "SELECT DISTINCT plate FROM sightings WHERE ts BETWEEN ? AND ? AND plate != ?",
        (ts - window, ts + window, plate), conn=conn)
    best, best_d = "", 99.0
    for row in near:
        other = row["plate"]
        d = confusion_distance(plate, other)
        if d > config.CLONE_MISREAD_MAX_DISTANCE or d >= best_d:
            continue
        # could that vehicle have been at this camera at this moment?
        its = db.rows(
            "SELECT camera_id, ts FROM sightings WHERE plate = ?"
            " AND ts BETWEEN ? AND ? ORDER BY ts", (other, ts - window, ts + window),
            conn=conn)
        for s in its:
            if s["camera_id"] == camera_id:
                best, best_d = other, d
                break
            gap_min = abs(s["ts"] - ts) / 60.0
            km = net.route_km(s["camera_id"], camera_id)
            if gap_min <= 0:
                continue
            if km / (gap_min / 60.0) <= config.IMPOSSIBLE_SPEED_KMPH:
                best, best_d = other, d
                break
    return best


def _min_vehicles(sightings: list[dict], net: CameraNetwork) -> int:
    """Fewest vehicles needed to explain these sightings.

    Greedy: walk the sightings in time order and drop each onto the first track
    it could feasibly continue, opening a new track when none fits. Greedy is not
    guaranteed minimal, so this is reported as a lower bound - which is the
    honest direction for a number that will be read as "at least N vehicles".
    """
    tracks: list[dict] = []
    for s in sightings:
        placed = False
        for t in tracks:
            dt_min = (s["ts"] - t["ts"]) / 60.0
            if dt_min < 0:
                continue
            km = net.route_km(t["camera_id"], s["camera_id"])
            if dt_min <= 0:
                if s["camera_id"] == t["camera_id"]:
                    placed = True
                    t.update(ts=s["ts"], camera_id=s["camera_id"])
                    break
                continue
            if km / (dt_min / 60.0) <= config.IMPOSSIBLE_SPEED_KMPH:
                t.update(ts=s["ts"], camera_id=s["camera_id"])
                placed = True
                break
        if not placed:
            tracks.append({"ts": s["ts"], "camera_id": s["camera_id"]})
    return max(1, len(tracks))


def check(plate: str, since: float | None = None, until: float | None = None,
          net: CameraNetwork | None = None, conn=None,
          test_misreads: bool = True) -> CloneReport:
    """Is this registration running on more than one vehicle?

    Cheap when the answer is no, which it almost always is: the only work for a
    clean plate is one indexed query and a walk over consecutive pairs. The
    expensive part - testing every conflict against the confusable plates that
    were nearby - runs only once a conflict has actually been found.
    """
    net = net or network()
    conn = conn or db.connect()
    plate = plate.upper().replace(" ", "")
    sightings = db.sightings_for_plate(plate, since, until, conn=conn)
    report = CloneReport(plate=plate)
    if len(sightings) < 2:
        report.reason = "too few sightings to say anything"
        return report

    raw = _pairs(sightings, net)
    if not raw:
        report.reason = "every leg is physically possible for one vehicle"
        return report

    for c in raw:
        weak = (c.from_confidence < config.CLONE_MIN_CONFIDENCE
                or c.to_confidence < config.CLONE_MIN_CONFIDENCE)
        if weak:
            c.explained_by_misread = "low-confidence read"
        elif test_misreads:
            alt = (_misread_alternative(plate, c.to_camera, c.to_ts, net, conn)
                   or _misread_alternative(plate, c.from_camera, c.from_ts, net, conn))
            if alt:
                c.explained_by_misread = alt
    report.conflicts = [c for c in raw if c.counts]
    report.dismissed = [c for c in raw if not c.counts]

    if not report.conflicts:
        why = report.dismissed[0].explained_by_misread if report.dismissed else ""
        report.reason = (
            f"{len(report.dismissed)} impossible leg(s), but each is better explained "
            f"as a misread{f' of {why}' if why and why != 'low-confidence read' else ''}"
            " than as a clone")
        return report

    reg = None
    if config.CLONE_ATTRIBUTE_CHECK_ENABLED:
        reg = registry.lookup(plate, conn=conn)
        if reg.found:
            for c in report.conflicts:
                if (c.from_vehicle_class and c.to_vehicle_class
                        and c.from_vehicle_class != reg.vehicle_class
                        and c.to_vehicle_class != reg.vehicle_class):
                    # Both ends, not either - agreement is never inspected (it
                    # would prove nothing and must never lower a verdict), and
                    # requiring both roughly halves how often the registry's
                    # own deliberate mismatch rate (config.REGISTRY_MISMATCH_RATE)
                    # promotes a verdict by coincidence rather than by evidence.
                    c.attribute_conflict = True
        # else: NOT_FOUND / UNKNOWN carries no evidence either way - skip
        # silently rather than guessing, same as a real photo's vehicle_class
        # being None on either side above.

    report.min_vehicles = _min_vehicles(sightings, net)
    n = len(report.conflicts)
    attribute_backed = any(c.attribute_conflict for c in report.conflicts)
    worst = max(report.conflicts, key=lambda c: c.implied_kmph)
    if n >= config.CLONE_CONFIRM_PAIRS or attribute_backed:
        report.verdict = "confident"
        if n >= config.CLONE_CONFIRM_PAIRS:
            report.reason = (
                f"{n} independent impossible legs survive the misread test - at least "
                f"{report.min_vehicles} vehicles are carrying this registration. Worst: "
                f"{worst.implied_kmph:.0f} km/h over {worst.road_km:.1f} km between "
                f"{worst.from_name} and {worst.to_name}.")
        else:
            reg_class = reg.vehicle_class if reg is not None else None
            report.reason = (
                f"one impossible leg - {worst.implied_kmph:.0f} km/h over "
                f"{worst.road_km:.1f} km between {worst.from_name} and {worst.to_name} - "
                f"and the vehicle registry (mock, see core/registry.py) independently "
                f"disagrees with what both cameras recorded: registered as {reg_class}, "
                f"observed as {worst.from_vehicle_class}. Two independent signals from a "
                f"single leg are being treated as sufficient.")
    else:
        report.verdict = "suspected"
        report.reason = (
            f"one impossible leg survives the misread test - {worst.implied_kmph:.0f} "
            f"km/h over {worst.road_km:.1f} km between {worst.from_name} and "
            f"{worst.to_name}. A single conflict can still be a clock error at one "
            f"camera; treat as a lead, not a finding.")
    return report


def screen(plates: list[str], since: float | None = None, net: CameraNetwork | None = None,
           conn=None) -> dict[str, CloneReport]:
    """Run the check over a set of candidates - what a search does automatically.

    Search results are the moment an operator meets a plate, and it is the wrong
    moment to let a duplicate registration go unmentioned: whatever they do next
    with that plate assumes it belongs to one vehicle.
    """
    net = net or network()
    conn = conn or db.connect()
    return {p: check(p, since=since, net=net, conn=conn) for p in plates}


def raise_alerts(report: CloneReport, conn=None) -> int | None:
    """Record a confident verdict in the alert queue, once.

    Only `confident`. A suspicion belongs on the operator's screen where it is
    shown with its caveat, not in the queue where it becomes an item of record
    that somebody later acts on without the caveat.
    """
    if report.verdict != "confident" or not report.conflicts:
        return None
    conn = conn or db.connect()
    now = time.time()
    recent = db.rows(
        "SELECT 1 FROM alerts WHERE kind = 'clone' AND plate = ? AND ts >= ? LIMIT 1",
        (report.plate, now - config.CLONE_ALERT_DEDUP_S), conn=conn)
    if recent:
        return None
    worst = max(report.conflicts, key=lambda c: c.implied_kmph)
    return db.add_alert(
        "clone", "critical",
        f"{report.plate} - {report.reason}", plate=report.plate,
        camera_id=worst.to_camera,
        detail={"source": "query-time scan", "min_vehicles": report.min_vehicles,
                "conflicts": [asdict(c) for c in report.conflicts]},
        ts=now, conn=conn)


# --- live response ------------------------------------------------------
def live_alerts(window_s: float | None = None, net: CameraNetwork | None = None,
                conn=None) -> list[dict]:
    """Clone alerts recent enough for the control room to still be reacting to.

    Normalises the two shapes a clone alert can arrive in, because the map and
    the alarm should not care which detector fired:

      - the ingest-time rule (core/alerts.py) writes one pair, flat, as
        {"from", "to", "km", "minutes", "implied_kmph"};
      - the query-time scan (raise_alerts above) writes every conflict it found,
        nested under "conflicts" as Conflict dicts.

    Both reduce to the same question an operator is actually asking: which two
    junctions, how far apart, and how fast would the vehicle have had to be
    going. Returns newest first, each with the coordinates already resolved so
    the caller can draw it without touching the network again.
    """
    net = net or network()
    window_s = config.CLONE_LIVE_WINDOW_S if window_s is None else window_s
    # Anchored to the store's own clock, not the wall clock. An alert inherits
    # the timestamp of the sighting that raised it, so on a database that was
    # seeded last night every alert in it is hours "old" by wall clock and a
    # ten-minute wall-clock window contains nothing - including an incident
    # injected thirty seconds ago. With the live feed running the two clocks
    # agree and this changes nothing.
    #
    # The *camera* clock specifically, matching core/analytics: an alert is
    # about what the network saw, and an operator enrolment can be timestamped
    # anywhere, so letting one move this anchor would push genuine alerts out
    # of the window.
    now = db.latest_ts((config.CAMERA_SOURCE,), conn=conn)
    rows_ = db.alerts(limit=50, include_acked=False, since=now - window_s, conn=conn)

    out = []
    for a in rows_:
        if a.get("kind") != "clone":
            continue
        d = a.get("detail") or {}
        pairs = []
        if d.get("conflicts"):
            for c in d["conflicts"]:
                pairs.append((c.get("from_camera"), c.get("to_camera"),
                              c.get("road_km"), c.get("minutes"), c.get("implied_kmph")))
        elif d.get("from") and d.get("to"):
            pairs.append((d["from"], d["to"], d.get("km"),
                          d.get("minutes"), d.get("implied_kmph")))
        if not pairs:
            continue
        # The worst pair is the one worth lighting up: it is the leg the
        # operator will be asked to justify.
        frm, to, km, minutes, kmph = max(pairs, key=lambda p: (p[4] or 0.0))
        if frm not in net.cameras or to not in net.cameras:
            continue
        a_lat, a_lon = net.coords(frm)
        b_lat, b_lon = net.coords(to)
        out.append({
            "alert_id": a["id"], "plate": a.get("plate"), "ts": a["ts"],
            "age_s": now - a["ts"], "message": a.get("message", ""),
            "from_camera": frm, "to_camera": to,
            "from_name": net.name(frm), "to_name": net.name(to),
            "from_lat": a_lat, "from_lon": a_lon, "to_lat": b_lat, "to_lon": b_lon,
            "road_km": km, "minutes": minutes, "implied_kmph": kmph,
            "min_vehicles": d.get("min_vehicles", 2),
            "detector": "query-time scan" if d.get("conflicts") else "ingest rule",
        })
    return out
