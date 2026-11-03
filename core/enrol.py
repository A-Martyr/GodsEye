"""Manual enrolment — an operator standing in for a camera that is not there.

Why this exists
---------------
The platform reads a plate from a real photograph perfectly well, and then has
nothing to say about it. `trajectory.reconstruct()` joins a registration to the
camera network, and a vehicle photographed on someone's phone has never passed
one of these 35 junctions, so the honest answer is "no record of it" and the
demonstration stops there.

Enrolment closes that gap without lying about anything. A photograph carries the
registration — the engine reads it, at its own confidence, against its own floor
— but it does not carry *which junction* or *at what instant*. Those are exactly
the two facts a camera contributes and a phone does not, and they are exactly
what an operator standing at the junction can supply. Enrol the same plate at a
second junction and the trajectory engine has a leg to draw, which is the whole
feature.

Where this sits in the platform's taxonomy
------------------------------------------
In `sightings`, not `inferences`, and the distinction matters because this
codebase is careful about it (see the schema comments in core/db.py):

  - an `inferences` row is a *hypothesis*: nobody could read the plate, and the
    platform named one by ranking neighbouring cameras against a retained CTC
    lattice. It is quarantined so a guess can never become evidence.
  - an enrolment is an *observation*. The plate was read, by the same engine
    that reads every camera capture here, above the same floor. What a human
    added is the place and the time — the camera's contribution, not the
    reader's.

So it is a sighting, and `sightings.source` records who attested it. Everything
that measures the *network* filters that column (core/analytics.py); everything
that tracks a *vehicle* deliberately does not (core/trajectory.py,
core/clones.py), because there the enrolment is precisely the evidence being
asked for.

Two rules are enforced structurally rather than by convention
-------------------------------------------------------------
1. **A below-floor read can never be enrolled.** `PlateRead.accepted` is the
   gate, unchanged and per-backend. The platform's own rule is that a read the
   engine will not stand behind is not stored — "inventing one is how a platform
   starts fabricating vehicles" — and an operator with a mouse is not an
   exemption from it. A refused capture already has a path: `unresolved` plus
   core/repair.py.

2. **Confidence is never invented.** The stored confidence is always the number
   the engine produced. An operator may correct a misread string (they are
   looking at the plate crop, which is better evidence than a 0.97 argmax), but
   correcting the text does not manufacture certainty, and the correction is
   bounded by OCR confusion distance so "correct" cannot quietly mean "type a
   different vehicle".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import config
from anpr.ocr import confusion_distance, matches_plate_grammar
from core import alerts as alert_rules
from core import db
from core.network import CameraNetwork, network


@dataclass
class Plausibility:
    """What the network thinks of putting this plate at this place at this time.

    Both neighbours matter, not just the previous one. A backdated enrolment
    lands *between* two existing sightings, and checking only backwards would
    happily write a row whose forward leg is impossible — which
    trajectory.reconstruct() would then draw in red and label a clone.
    """
    ok: bool
    reason: str
    prev: dict | None = None
    nxt: dict | None = None
    back_kmph: float | None = None
    fwd_kmph: float | None = None
    earliest_ok_ts: float | None = None


def network_now(conn=None) -> float:
    """The newest thing a camera saw — this network's idea of 'now'.

    Wall clock is the wrong default. A seeded database is routinely hours old by
    the time anyone demonstrates it, so an enrolment stamped `time.time()` sits
    alone outside every analytics window and every "last 30 minutes" table, and
    the operator sees their vehicle vanish from the very views they just used.
    Anchoring to the corpus puts the enrolment where the rest of the data is.
    """
    rows = db.rows("SELECT MAX(ts) t FROM sightings WHERE source = ?",
                   (config.CAMERA_SOURCE,), conn=conn)
    latest = rows[0]["t"] if rows else None
    if latest is None or not config.ENROL_ANCHOR_TO_NETWORK_CLOCK:
        return time.time()
    return float(latest)


def neighbours(plate: str, ts: float, conn=None) -> tuple[dict | None, dict | None]:
    """The sightings of `plate` immediately before and after `ts`.

    Deliberately not trajectory.last_known(), which is unconditionally the
    newest row for the plate. With a backdated timestamp the newest row is the
    wrong predecessor, and the arithmetic against it comes out negative — a
    leg that travels backwards in time reads as comfortably under any speed
    limit and sails through a naive check.
    """
    plate = plate.upper().replace(" ", "")
    prev = db.rows("SELECT * FROM sightings WHERE plate = ? AND ts <= ?"
                   " ORDER BY ts DESC LIMIT 1", (plate, ts), conn=conn)
    nxt = db.rows("SELECT * FROM sightings WHERE plate = ? AND ts > ?"
                  " ORDER BY ts ASC LIMIT 1", (plate, ts), conn=conn)
    return (prev[0] if prev else None), (nxt[0] if nxt else None)


def _implied_kmph(net: CameraNetwork, from_cam: str, to_cam: str,
                  seconds: float) -> tuple[float, float]:
    """(road_km, implied km/h) for a leg. Same camera or no time => 0 km/h."""
    if from_cam == to_cam:
        return 0.0, 0.0
    km = net.route_km(from_cam, to_cam)
    if seconds <= 0:
        return km, float("inf")
    return km, km / (seconds / 3600.0)


def suggest_ts(plate: str, camera_id: str, conn=None,
               net: CameraNetwork | None = None) -> tuple[float, str]:
    """A timestamp that makes this enrolment physically possible. -> (ts, why)

    This is the single most important piece of the feature, because the obvious
    implementation — stamp `now` — walks straight into the platform's own clone
    detector. Two enrolments a minute apart at junctions 8 km apart imply
    480 km/h, and trajectory.reconstruct() labels that leg "physically
    impossible, same plate probably running on two vehicles". The demonstration
    would accuse itself of a clone at the exact moment it was supposed to draw
    a route.

    So the gap is derived from the road, not guessed: the distance the network
    actually reports between the two junctions, divided by an ordinary city
    speed, floored at a few minutes so the alert rules stay quiet too.
    """
    net = net or network()
    last = db.rows("SELECT * FROM sightings WHERE plate = ? ORDER BY ts DESC LIMIT 1",
                   (plate.upper().replace(" ", ""),), conn=conn)
    if not last:
        ts = network_now(conn=conn)
        return ts, ("first sighting of this plate - stamped at the network's own "
                    "clock so it lands inside the live analytics window")
    prev = last[0]
    if prev["camera_id"] == camera_id:
        ts = prev["ts"] + config.ENROL_MIN_GAP_S
        return ts, (f"same junction as the previous sighting - "
                    f"{config.ENROL_MIN_GAP_S / 60:.0f} min later")
    km = net.route_km(prev["camera_id"], camera_id)
    needed = (km / config.ENROL_TARGET_KMPH) * 3600.0
    gap = max(needed, config.ENROL_MIN_GAP_S)
    return prev["ts"] + gap, (
        f"{km:.1f} km by road from {prev['camera_id']}; at "
        f"{config.ENROL_TARGET_KMPH:.0f} km/h that is {gap / 60:.0f} min")


def check(plate: str, camera_id: str, ts: float, conn=None,
          net: CameraNetwork | None = None) -> Plausibility:
    """Would this enrolment produce a leg the platform itself calls impossible?"""
    net = net or network()
    if camera_id not in net.cameras:
        return Plausibility(False, f"{camera_id} is not a camera on this network")

    prev, nxt = neighbours(plate, ts, conn=conn)
    back_kmph = fwd_kmph = None
    earliest = None

    # A second click on the enrol button is not a second pass through the
    # junction. Same plate, same camera, seconds apart is a double submission -
    # harmless to the route (reconstruct() skips same-camera pairs) but it puts
    # a phantom row in the vehicle's history and in the audit trail.
    if prev is not None and prev["camera_id"] == camera_id:
        gap = abs(ts - prev["ts"])
        if gap < config.ENROL_DUPLICATE_WINDOW_S:
            return Plausibility(
                False,
                f"this plate was already enrolled at {camera_id} {gap:.0f}s ago. "
                "Enrol it at the next junction to extend the route, or wait if "
                "the vehicle genuinely passed twice.",
                prev, nxt, earliest_ok_ts=prev["ts"] + config.ENROL_DUPLICATE_WINDOW_S)

    if prev is not None:
        km, back_kmph = _implied_kmph(net, prev["camera_id"], camera_id, ts - prev["ts"])
        if back_kmph > config.ENROL_MAX_IMPLIED_KMPH:
            needed = (km / config.ENROL_TARGET_KMPH) * 3600.0
            earliest = prev["ts"] + max(needed, config.ENROL_MIN_GAP_S)
            return Plausibility(
                False,
                f"{km:.1f} km from {prev['camera_id']} in "
                f"{(ts - prev['ts']) / 60:.1f} min implies {back_kmph:.0f} km/h. The "
                f"platform would flag that leg as a cloned plate rather than a route.",
                prev, nxt, back_kmph, fwd_kmph, earliest)

    if nxt is not None:
        km, fwd_kmph = _implied_kmph(net, camera_id, nxt["camera_id"], nxt["ts"] - ts)
        if fwd_kmph > config.ENROL_MAX_IMPLIED_KMPH:
            return Plausibility(
                False,
                f"this plate is already recorded at {nxt['camera_id']} "
                f"{(nxt['ts'] - ts) / 60:.1f} min later, {km:.1f} km away - "
                f"implying {fwd_kmph:.0f} km/h onward. Choose an earlier time.",
                prev, nxt, back_kmph, fwd_kmph)

    return Plausibility(True, "consistent with this vehicle's other sightings",
                        prev, nxt, back_kmph, fwd_kmph)


def validate_correction(read_text: str, corrected: str) -> tuple[bool, str]:
    """May the operator replace `read_text` with `corrected`?

    Yes for the realistic case — the engine dropped or swapped a character and
    the operator can see the right one on the crop. This is not a theoretical
    concern: a monsoon capture in this repo reads one character wrong at 0.97,
    comfortably above the 0.96 floor, and without a correction path the
    second enrolment simply never joins the first and the route never appears.

    No for typing an unrelated registration. The bound is OCR confusion
    distance, the same measure trajectory search uses, so a correction has to
    be the kind of mistake this engine actually makes.
    """
    corrected = (corrected or "").upper().replace(" ", "")
    if not corrected:
        return False, "empty"
    if corrected == read_text:
        return True, ""
    if not matches_plate_grammar(corrected):
        return False, (f"{corrected} is not a registration shape this platform "
                       "recognises - two state letters, district digits, series, "
                       "four digits")
    d = confusion_distance(read_text, corrected)
    if d > config.ENROL_CORRECTION_MAX_DISTANCE:
        return False, (f"{corrected} is too far from what the engine read "
                       f"({read_text}) - confusion distance {d:.1f}, limit "
                       f"{config.ENROL_CORRECTION_MAX_DISTANCE:.1f}. Correcting a "
                       "misread is supported; entering a different vehicle is not.")
    return True, ""


def enrol(read, camera_id: str, *, ts: float | None = None,
          plate: str | None = None, operator: str | None = None,
          note: str | None = None, allow_implausible: bool = False,
          conn=None, net: CameraNetwork | None = None) -> dict:
    """Write one operator-attested sighting. Raises ValueError with a reason.

    `read` is the PlateRead the engine produced — passed whole rather than as a
    string so the confidence and the accept/refuse decision come from the engine
    and cannot be supplied by the caller.
    """
    net = net or network()
    conn = conn or db.connect()

    if not getattr(read, "text", ""):
        raise ValueError("nothing was read from this image")
    if not read.accepted:
        raise ValueError(
            f"{read.pretty} was read at {read.confidence:.0%}, below the "
            f"{read.floor:.0%} floor for the {read.backend} reader. A read the "
            "engine will not stand behind is not stored as a sighting - try the "
            "anpr_ocr backend, or a clearer photograph.")
    if camera_id not in net.cameras:
        raise ValueError(f"{camera_id} is not a camera on this network")

    engine_text = read.text.upper().replace(" ", "")
    final = (plate or engine_text).upper().replace(" ", "")
    corrected = final != engine_text
    if corrected:
        ok, why = validate_correction(engine_text, final)
        if not ok:
            raise ValueError(why)

    if ts is None:
        ts, _ = suggest_ts(final, camera_id, conn=conn, net=net)

    verdict = check(final, camera_id, ts, conn=conn, net=net)
    if not verdict.ok and not allow_implausible:
        raise ValueError(verdict.reason)

    # Heading and direction describe the approach into this junction, which is
    # knowable only if the vehicle was somewhere before it. Speed is left NULL
    # on purpose: nobody measured one, and trajectory.reconstruct() derives leg
    # speeds itself from the timestamps.
    heading = direction = None
    prev = verdict.prev
    if prev is not None and prev["camera_id"] != camera_id:
        heading = net.heading(prev["camera_id"], camera_id)
        direction = net.direction(prev["camera_id"], camera_id)

    detail = note or ""
    if corrected:
        stamp = f"engine read {engine_text}, operator corrected to {final}"
        detail = f"{stamp}. {detail}".strip()

    s = db.Sighting(
        ts=float(ts), camera_id=camera_id, plate=final,
        confidence=float(read.confidence),      # the engine's number, never invented
        speed_kmph=None, lane=None,
        heading=heading, direction=direction, next_camera=None,
        vehicle_class=None,
        condition=None,          # 'capture condition (simulator only)' - not ours to set
        ocr_variant=f"enrol:{read.backend}",
        true_plate=None,         # 'ground truth, simulator only' - never fabricate one
        source=config.ENROL_SOURCE_CORRECTED if corrected else config.ENROL_SOURCE,
        enrolled_by=operator or "control-room",
        enrolled_note=detail or None,
    )
    s.id = db.add_sighting(s, conn=conn)

    # Same two-step every other ingest path uses (sim/city.py:356-359), so an
    # enrolled vehicle on the watchlist raises the same alert a camera would.
    fired = alert_rules.evaluate(s.as_dict(), conn=conn, net=net) or []

    return {
        "sighting_id": s.id,
        "plate": final,
        "engine_read": engine_text,
        "corrected": corrected,
        "camera_id": camera_id,
        "camera_name": net.name(camera_id),
        "ts": float(ts),
        "confidence": float(read.confidence),
        "source": s.source,
        "alerts": fired,
        "plausible": verdict.ok,
        "note": verdict.reason,
        "total_sightings": len(db.sightings_for_plate(final, conn=conn)),
    }


def enrolled(plate: str | None = None, limit: int = 200, conn=None) -> list[dict]:
    """Every operator-attested sighting, newest first. For an audit panel."""
    marks = ",".join("?" * len(config.ENROL_SOURCES))
    if plate:
        return db.rows(
            f"SELECT * FROM sightings WHERE source IN ({marks}) AND plate = ?"
            f" ORDER BY ts DESC LIMIT ?",
            (*config.ENROL_SOURCES, plate.upper().replace(" ", ""), limit), conn=conn)
    return db.rows(
        f"SELECT * FROM sightings WHERE source IN ({marks}) ORDER BY ts DESC LIMIT ?",
        (*config.ENROL_SOURCES, limit), conn=conn)
