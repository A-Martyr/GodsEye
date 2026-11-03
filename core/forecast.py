"""Where a vehicle is likely to go next, and when it is usually where.

The two questions an operator asks that a sighting log cannot answer on its
own. Commercial plate-reader platforms sell both - "determine where it may be
located in the future", "know the best time to intercept" - and they answer
them from point detections alone, because a camera that reads plates by the
roadside knows nothing about the road.

This platform does. data/cameras.json carries 35 junctions and 52 links with
real geometry, so a prediction here is not "this plate was often seen at
CAM-12 next". It is a junction, reached by a named road of known length, at a
time derived from how fast traffic has actually been moving on that link -
and the same machinery says which of those is worth an operator's attention.

Two rules run through everything below.

Blend, do not choose. A plate seen three times carries almost no personal
signal and a plate seen fifty times carries a lot; either extreme is easy and
the middle is where the honest answer lives. Every distribution here is the
vehicle's own history shrunk towards what traffic in general does at that
junction, weighted by how much history there actually is.

Say when you do not know. A ranked list always looks like knowledge. Each
prediction carries the number of observations behind it, and `confident` is
false when that number is too small to mean anything - which the dashboard
shows rather than hides, because an interception sent to the wrong junction
costs more than one not sent.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import config
from core import db
from core.network import CameraNetwork, network

# A leg is two consecutive sightings of one plate. Beyond this the vehicle
# almost certainly stopped, parked or left the network and came back, and
# treating that as a transition would teach the model roads nobody drove.
MAX_LEG_MINUTES = float(getattr(config, "FORECAST_MAX_LEG_MIN", 45.0))

# How hard the network prior pulls. In the blend below this is, in effect, the
# number of imaginary journeys every vehicle is credited with having made the
# way traffic in general does: at k=4 a plate needs four observed departures
# from a junction before its own habit outweighs the crowd's.
PRIOR_STRENGTH = float(getattr(config, "FORECAST_PRIOR_STRENGTH", 4.0))

# Below this many observed departures a prediction is reported but not called
# confident.
MIN_OBSERVATIONS = int(getattr(config, "FORECAST_MIN_OBSERVATIONS", 3))


def _legs(since: float | None = None, plate: str | None = None,
          conn=None) -> list[tuple[str, str, float]]:
    """Observed (from_camera, to_camera, minutes) hops, newest data first.

    Built from consecutive sightings of the same plate rather than from the
    `next_camera` column, which records where the simulator intended a vehicle
    to go. What was intended and what a camera actually saw are different
    things, and only the second one is evidence.
    """
    where, params = [], []
    if since is not None:
        where.append("ts >= ?")
        params.append(since)
    if plate:
        where.append("plate = ?")
        params.append(plate.upper().replace(" ", ""))
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.rows(
        f"SELECT plate, camera_id, ts FROM sightings{clause} ORDER BY plate, ts",
        tuple(params), conn=conn)

    out: list[tuple[str, str, float]] = []
    prev = None
    for r in rows:
        if prev is not None and prev["plate"] == r["plate"]:
            gap_min = (r["ts"] - prev["ts"]) / 60.0
            if 0 < gap_min <= MAX_LEG_MINUTES and prev["camera_id"] != r["camera_id"]:
                out.append((prev["camera_id"], r["camera_id"], gap_min))
        prev = r
    return out


class TransitionModel:
    """Who goes where from each junction, and how long it takes them.

    Counts, not probabilities, so a plate's own history and the network's can
    be combined by adding evidence rather than averaging two numbers that were
    computed from wildly different amounts of it.
    """

    def __init__(self, legs: list[tuple[str, str, float]]):
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.times: dict[tuple[str, str], list[float]] = defaultdict(list)
        for a, b, mins in legs:
            self.counts[a][b] += 1
            self.times[(a, b)].append(mins)

    def departures(self, camera_id: str) -> int:
        return sum(self.counts.get(camera_id, {}).values())

    def median_minutes(self, a: str, b: str) -> float | None:
        """Typical observed time on this hop. The median, not the mean: a
        vehicle that stopped for fuel mid-leg is in this data and would drag
        an average somewhere no journey actually took."""
        seen = sorted(self.times.get((a, b), []))
        if not seen:
            return None
        mid = len(seen) // 2
        return seen[mid] if len(seen) % 2 else 0.5 * (seen[mid - 1] + seen[mid])


def _blended(plate_model: TransitionModel, prior: TransitionModel,
             camera_id: str, net: CameraNetwork) -> dict[str, float]:
    """P(next junction) for this vehicle, shrunk towards what everyone does.

    The shrinkage is the whole point. Taking the vehicle's own history alone
    means one observed turn reads as a 100% certainty, which is how a
    prediction system ends up sending someone to the wrong junction with great
    confidence. Taking only the crowd's means a vehicle with a daily commute is
    predicted as though it were a stranger.
    """
    own = plate_model.counts.get(camera_id, {})
    crowd = prior.counts.get(camera_id, {})
    crowd_total = sum(crowd.values())

    # Every junction physically reachable is a candidate, even one nobody has
    # been observed taking - an empty road is still a road.
    candidates = set(own) | set(crowd) | set(net.neighbours(camera_id))
    if not candidates:
        return {}

    weights: dict[str, float] = {}
    for nxt in candidates:
        crowd_p = (crowd.get(nxt, 0) / crowd_total) if crowd_total else (1.0 / len(candidates))
        weights[nxt] = own.get(nxt, 0) + PRIOR_STRENGTH * crowd_p
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def predict(plate: str, top: int = 5, net: CameraNetwork | None = None,
            since: float | None = None, conn=None) -> dict:
    """Where this vehicle is likely to be seen next, and roughly when.

    Returns the last known sighting, ranked candidate junctions with an ETA
    window on each, and an honest account of how much evidence is behind it.
    """
    net = net or network()
    plate = plate.upper().replace(" ", "")
    conn = conn or db.connect()

    last = db.rows("SELECT * FROM sightings WHERE plate = ? ORDER BY ts DESC LIMIT 1",
                   (plate,), conn=conn)
    if not last:
        return {"plate": plate, "known": False,
                "reason": "no sighting of this registration on the network"}
    last = dict(last[0])

    prior = TransitionModel(_legs(since=since, conn=conn))
    own = TransitionModel(_legs(since=since, plate=plate, conn=conn))
    here = last["camera_id"]

    dist = _blended(own, prior, here, net)
    if not dist:
        return {"plate": plate, "known": True, "last": last, "candidates": [],
                "reason": f"{net.name(here)} has no onward link in the network model"}

    elapsed_min = (time.time() - last["ts"]) / 60.0
    out = []
    for cam, p in sorted(dist.items(), key=lambda kv: -kv[1])[:top]:
        # Prefer what traffic actually did on this link; fall back to the
        # road's free-flow time when nobody has been observed making the hop.
        observed = own.median_minutes(here, cam) or prior.median_minutes(here, cam)
        modelled = net.free_flow_minutes(here, cam)
        eta_min = observed if observed is not None else modelled
        out.append({
            "camera_id": cam,
            "camera_name": net.name(cam),
            "probability": round(p, 4),
            "observed_hops": own.counts.get(here, {}).get(cam, 0),
            "network_hops": prior.counts.get(here, {}).get(cam, 0),
            "km": round(net.link_km(here, cam) or net.route_km(here, cam) or 0.0, 2),
            "direction": net.direction(here, cam),
            "typical_minutes": round(eta_min, 1) if eta_min else None,
            "from_observation": observed is not None,
            # Negative means the window has already passed - the vehicle
            # should have arrived and did not, which is information too.
            "due_in_minutes": (round(eta_min - elapsed_min, 1)
                               if eta_min is not None else None),
            "lat": net.coords(cam)[0], "lon": net.coords(cam)[1],
        })

    departures = own.departures(here)
    return {
        "plate": plate,
        "known": True,
        "last": last,
        "last_camera_name": net.name(here),
        "minutes_since_seen": round(elapsed_min, 1),
        "candidates": out,
        "observations": departures,
        "confident": departures >= MIN_OBSERVATIONS,
        "basis": ("this vehicle's own history at this junction"
                  if departures >= MIN_OBSERVATIONS
                  else "mostly how traffic in general leaves this junction - "
                       f"only {departures} departure(s) by this vehicle are on record"),
    }


def pattern_of_life(plate: str, net: CameraNetwork | None = None,
                    conn=None) -> dict:
    """When and where this vehicle is usually seen.

    The other half of the interception question. Knowing a vehicle passes
    Shyambazar is worth little; knowing it passes Shyambazar on weekday
    mornings is what lets someone be standing there. Built from the sighting
    history alone, so it says nothing a camera did not see.
    """
    net = net or network()
    plate = plate.upper().replace(" ", "")
    rows = db.rows("SELECT camera_id, ts FROM sightings WHERE plate = ? ORDER BY ts",
                   (plate,), conn=conn)
    if not rows:
        return {"plate": plate, "known": False, "sightings": 0}

    by_hour = defaultdict(int)
    by_camera = defaultdict(int)
    by_hour_camera: dict[tuple[int, str], int] = defaultdict(int)
    for r in rows:
        hour = time.localtime(r["ts"]).tm_hour
        by_hour[hour] += 1
        by_camera[r["camera_id"]] += 1
        by_hour_camera[(hour, r["camera_id"])] += 1

    n = len(rows)
    best_hour, best_hour_n = max(by_hour.items(), key=lambda kv: kv[1])
    best_cam, best_cam_n = max(by_camera.items(), key=lambda kv: kv[1])
    (peak_hour, peak_cam), peak_n = max(by_hour_camera.items(), key=lambda kv: kv[1])

    # The city's own baseline, which is what "usual" has to be measured against.
    # A vehicle seen most often at Shyambazar has told you nothing if Shyambazar
    # is where everyone is seen most often.
    #
    # Both baselines are aggregated in SQL rather than in Python. The earlier
    # arrangement here read `SELECT ts FROM sightings` with no window at all,
    # then called time.localtime() once per row - every timestamp in the store,
    # on every dashboard rerun, to arrive at 24 counters. That is a table scan
    # plus tens of thousands of allocations for a number SQLite can produce
    # itself, and it showed as a stall on the tab an operator uses most.
    totals = db.rows(
        "SELECT (SELECT COUNT(*) FROM sightings) AS all_n,"
        " (SELECT COUNT(*) FROM sightings WHERE camera_id = ?) AS cam_n,"
        " (SELECT COUNT(*) FROM sightings WHERE CAST(strftime('%H', ts, 'unixepoch',"
        " 'localtime') AS INTEGER) = ?) AS hour_n",
        # best_cam, NOT peak_cam. The test below is on best_cam_n, so its
        # baseline has to be the share of traffic at that same junction - an
        # earlier version drew the count from one camera and the baseline from
        # another, and silently compared them whenever the two differed.
        (best_cam, best_hour), conn=conn)[0]

    tot_st = totals["all_n"] or 0
    p_cam = (totals["cam_n"] / tot_st) if tot_st else 1.0
    p_hour = (totals["hour_n"] / tot_st) if tot_st else 1.0

    def binom_tail(k: int, N: int, p: float) -> float:
        """P(X >= k) for X ~ Binomial(N, p). Summed exactly - N is one
        vehicle's sighting count, rarely past a few hundred, and a normal
        approximation is worst exactly here: small N, small p, k in the tail."""
        if k > N or k <= 0:
            return 0.0
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total = 0.0
        for j in range(k, N + 1):
            total += math.exp(math.lgamma(N + 1) - math.lgamma(j + 1)
                              - math.lgamma(N - j + 1)
                              + j * math.log(p) + (N - j) * math.log1p(-p))
        return min(total, 1.0)

    # Each flag tests a MARGINAL maximum against its own family of cells, which
    # is why the corrections are 35 and 24 rather than 840: best_cam_n is the
    # largest of 35 per-camera counts, best_hour_n the largest of 24 per-hour
    # counts. Selecting the maximum and then testing it is exactly the bias
    # Bonferroni is for, and without it the winning cell clears any threshold
    # you like on noise alone - which is what the old volume-only gate did.
    #
    # best_intercept below is the argmax over the 24x35 JOINT cells and is NOT
    # what these flags test. It is reported as the most-populated cell, which
    # is a description of this vehicle's history, not a claim that the pairing
    # beats chance. Read the two flags for that.
    cam_pval = binom_tail(best_cam_n, n, p_cam)
    hour_pval = binom_tail(best_hour_n, n, p_hour)

    # Six sightings is the floor. Below it the tail can still return a small
    # number, and a routine asserted from five observations is not a routine.
    s_rel = n >= 6 and (cam_pval < 0.05 / 35.0)
    t_rel = n >= 6 and (hour_pval < 0.05 / 24.0)

    return {
        "plate": plate,
        "known": True,
        "sightings": n,
        "hours": {h: by_hour.get(h, 0) for h in range(24)},
        "cameras": [{"camera_id": c, "camera_name": net.name(c), "sightings": k,
                     "share": round(k / n, 3)}
                    for c, k in sorted(by_camera.items(), key=lambda kv: -kv[1])[:8]],
        "busiest_hour": best_hour,
        "busiest_hour_share": round(best_hour_n / n, 3),
        "usual_camera": best_cam,
        "usual_camera_name": net.name(best_cam),
        "best_intercept": {
            "camera_id": peak_cam,
            "camera_name": net.name(peak_cam),
            "hour": peak_hour,
            "sightings": peak_n,
            "share": round(peak_n / n, 3),
        },
        # Two flags, not one boolean. A vehicle can genuinely favour a junction
        # while keeping no hours at all, and the reverse - collapsing those into
        # one "reliable" threw away the half an operator planning an
        # interception actually needs.
        "spatial_reliable": s_rel,
        "temporal_reliable": t_rel,
        "reliable": s_rel and t_rel,
        # The evidence behind the flags, so a caller can show its working rather
        # than asking anyone to trust a boolean.
        "spatial_p": cam_pval,
        "temporal_p": hour_pval,
        "city_share_at_usual_camera": round(p_cam, 4),
        "city_share_at_busiest_hour": round(p_hour, 4),
    }
