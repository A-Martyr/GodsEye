"""Geographic zones, and what crossing one is actually evidence of.

Commercial plate-reader platforms sell a "geographic boundary" filter: draw a
shape on a map, get the vehicles inside it. Drawing the shape is the easy half
and it is the half those products ship. The hard half is that a zone drawn on a
map is a claim about the *world*, while everything this platform knows comes
from 35 cameras, and the gap between the two is where a control room gets an
answer that is confidently wrong.

Three things follow from that gap, and they shape every function below.

A zone is only as good as its doors. A boundary that no camera sits on cannot
observe a crossing - vehicles simply materialise inside it. A zone containing
one camera cannot tell "inside the area" from "at that junction", so its dwell
times are revisit intervals wearing a costume. `observability()` reports this
before anything else does, and reports it from the data as well as from the
geometry: the share of entries where the platform actually saw the vehicle
approach from outside is a measured number, not a property of the polygon.

"Inside" is an inference, not an observation. A camera sees a vehicle at a
junction. It never sees a vehicle *in an area*. Everything here is built from
runs of consecutive sightings at cameras that happen to fall inside the shape,
which is a weaker thing than it looks, and the weakness is carried in the
output rather than smoothed away.

"Still inside" is the question that will be got wrong. A vehicle last seen at a
zone camera may be parked in it, or may have left twenty minutes ago through a
side road no camera covers. Asserting the first is how a cordon gets set around
an empty car park. So `visits()` does not assert it: it estimates it against
this zone's own history of how long vehicles stayed unseen before turning up
outside again, discounts it when the last sighting was pointed out of the zone,
and says how many completed visits that estimate rests on - the same discipline
core/forecast.py applies to a predicted junction and core/clones.py applies to
an accusation.

Dwell time gets two numbers for the same reason. The span between the first and
last sighting inside is a lower bound; the span between the bracketing
sightings outside is an upper bound; the truth is between them and this module
declines to pick a point in that interval and call it the answer.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from core import db
from core.network import COMPASS, CameraNetwork, haversine_km, network

# Zones live in their own file rather than in the database. They are operator
# configuration, not observations, and core/db.reset() exists to wipe the
# observations before a demo - a zone an officer drew should survive that.
ZONE_FILE = Path(getattr(config, "GEOFENCE_FILE", config.DATA_DIR / "zones.json"))

# How many completed visits a zone needs before its own exit-lag distribution
# is trusted to answer "is it still in there". Below this the estimate falls
# back to a generic decay and says so.
MIN_LAG_SAMPLES = int(getattr(config, "GEOFENCE_MIN_LAG_SAMPLES", 8))

# The decay constant for that fallback, in minutes. Deliberately short: with no
# evidence about a zone, the safer error is to stop claiming a vehicle is still
# inside rather than to keep claiming it for an hour.
FALLBACK_DWELL_MIN = float(getattr(config, "GEOFENCE_FALLBACK_DWELL_MIN", 20.0))

# How far a vehicle's heading may differ from the bearing towards an outside
# neighbour and still count as "pointed out of the zone". 60 degrees is a
# junction's worth of slack - a vehicle in the correct lane for an exit road
# rarely reads more than that off the link bearing.
OUTBOUND_TOLERANCE_DEG = float(getattr(config, "GEOFENCE_OUTBOUND_DEG", 60.0))

# Do not raise the same zone entry for the same plate twice inside this. Mirrors
# the dedup in core/alerts.py, which is why it is expressed the same way.
DEDUP_MINUTES = float(getattr(config, "GEOFENCE_DEDUP_MIN", 20.0))

DEFAULT_WINDOW_HOURS = float(getattr(config, "GEOFENCE_WINDOW_HOURS", 6.0))

# The `kind` written to the alerts table. Kept as a constant because the
# orchestrator wiring, the dedup query here and any dashboard filter all have
# to agree on the same string.
ALERT_KIND = "geofence"

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# --- geometry -----------------------------------------------------------
def point_in_polygon(lat: float, lon: float, points: list) -> bool:
    """Ray casting in raw degrees, treating lon as x and lat as y.

    No projection, which would be wrong for a country and is irrelevant for a
    city: over the 20 km Kolkata spans, the error from ignoring the convergence
    of meridians is smaller than the width of the junction a camera stands in.
    The distance test used by circular zones is a proper haversine, because
    there a systematic error would show up as a radius that is not the radius.

    A vertex exactly on the ray is handled by the strict/non-strict pair in the
    crossing test, so a camera sitting precisely on a boundary latitude is
    counted once rather than zero or twice.
    """
    n = len(points)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        y_i, x_i = float(points[i][0]), float(points[i][1])
        y_j, x_j = float(points[j][0]), float(points[j][1])
        if (y_i > lat) != (y_j > lat):
            # The guard above guarantees y_j != y_i, so this cannot divide by
            # zero - a horizontal edge is simply never a crossing.
            x_cross = x_i + (lat - y_i) * (x_j - x_i) / (y_j - y_i)
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def _angle_gap(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _heading_of(sighting: dict) -> float | None:
    """The bearing a sighting was travelling on, or None if it is not recorded.

    Prefers the numeric `heading` and falls back to decoding the `direction`
    string, which is quantised to 22.5 degrees and carries "-" when the node
    could not work out which way the vehicle was pointing. Deliberately does
    not touch `next_camera`: as core/forecast.py notes, that column records
    where a vehicle was *intended* to go, and intent is not observation.
    """
    h = sighting.get("heading")
    if h is not None:
        try:
            return float(h) % 360.0
        except (TypeError, ValueError):
            pass
    d = (sighting.get("direction") or "").strip().upper()
    if d in COMPASS:
        return COMPASS.index(d) * 22.5
    return None


# --- zone ---------------------------------------------------------------
@dataclass
class Zone:
    """A named area, plus what the control room wants done when it is entered.

    The camera membership is deliberately NOT stored on the zone. It is derived
    from the shape and data/cameras.json, and a stored copy would go stale the
    moment a camera moved or was added - producing a zone that silently stopped
    watching a junction it appears on the map to cover. `cameras_in()`
    recomputes it, which costs 35 haversines.
    """
    id: str
    name: str
    kind: str = "circle"                 # circle | polygon
    lat: float | None = None             # circle centre
    lon: float | None = None
    radius_km: float | None = None
    points: list = field(default_factory=list)   # polygon, [[lat, lon], ...]
    note: str = ""
    # What an entry is worth raising. "watchlist" is the VehicleManager
    # behaviour - only vehicles already of interest. "any" is the restricted
    # area case, where the entry itself is the offence. "none" makes the zone a
    # pure query filter with no alerting at all.
    alert_on: str = "watchlist"          # watchlist | any | none
    # Restrict alerting to these vehicle classes - a lorry ban in a city
    # centre zone, say. Empty means every class.
    alert_classes: list = field(default_factory=list)
    severity: str = ""                   # blank: inherit from the watchlist row
    created_ts: float = 0.0

    def contains(self, lat: float, lon: float) -> bool:
        if self.kind == "circle":
            if self.lat is None or self.lon is None or not self.radius_km:
                return False
            return haversine_km(self.lat, self.lon, float(lat), float(lon)) <= float(self.radius_km)
        return point_in_polygon(float(lat), float(lon), self.points)

    def centre(self) -> tuple[float, float]:
        """A point to hang a map label on. The vertex centroid for a polygon,
        which is not the area centroid and is not meant to be - it only has to
        land somewhere sensible inside a convex city block."""
        if self.kind == "circle" and self.lat is not None:
            return float(self.lat), float(self.lon or 0.0)
        if not self.points:
            return config.CITY_CENTER
        return (sum(float(p[0]) for p in self.points) / len(self.points),
                sum(float(p[1]) for p in self.points) / len(self.points))

    def outline(self, steps: int = 48) -> list:
        """The boundary as [[lat, lon], ...], closed, ready for a map layer.

        A circle becomes a polygon here rather than in the dashboard, so the
        thing drawn on the map is the same shape `contains()` tests against.
        The longitude step is divided by cos(latitude) because a degree of
        longitude is shorter than a degree of latitude everywhere but the
        equator, and skipping that draws an ellipse and calls it a radius.
        """
        if self.kind == "polygon":
            pts = [[float(p[0]), float(p[1])] for p in self.points]
            return pts + pts[:1] if pts else []
        if self.lat is None or not self.radius_km:
            return []
        lat, lon, r = float(self.lat), float(self.lon or 0.0), float(self.radius_km)
        d_lat = r / 110.574
        d_lon = r / (111.320 * max(math.cos(math.radians(lat)), 1e-6))
        ring = [[lat + d_lat * math.cos(2 * math.pi * i / steps),
                 lon + d_lon * math.sin(2 * math.pi * i / steps)]
                for i in range(steps)]
        return ring + ring[:1]

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or f"zone-{int(time.time())}"


# --- persistence --------------------------------------------------------
# A zone file is a handful of kilobytes read on every dashboard rerun, so it is
# cached against its own mtime: cheap enough to reload when an operator edits
# it, cheap enough not to reload sixty times a minute when nobody has.
_cache: tuple[float, list[Zone]] | None = None

_DEFAULT_ZONES = [
    # Chosen to be immediately meaningful on this network rather than tidy.
    # The first two are well covered; the third is deliberately a zone with one
    # camera in it, because a platform that only ships examples it handles well
    # teaches an operator nothing about where it stops working.
    {"id": "maidan-esplanade", "name": "Maidan and Esplanade core", "kind": "circle",
     "lat": 22.5560, "lon": 88.3500, "radius_km": 1.6,
     "note": "VIP movement and rally ground - the cordon drawn for an event",
     "alert_on": "watchlist"},
    {"id": "salt-lake-sector-v", "name": "Salt Lake and Sector V", "kind": "polygon",
     "points": [[22.6125, 88.3950], [22.6125, 88.4850],
                [22.5550, 88.4850], [22.5550, 88.3950]],
     "note": "IT corridor - the origin-destination question the transport cell asks",
     "alert_on": "watchlist"},
    {"id": "airport-approach", "name": "NSCBI Airport approach", "kind": "circle",
     "lat": 22.6480, "lon": 88.4420, "radius_km": 1.5,
     "note": "One camera only - crossings cannot be observed, see observability()",
     "alert_on": "watchlist"},
]


def load_zones(path: Path | None = None) -> list[Zone]:
    """Every defined zone, seeding a starter set the first time.

    The seed exists so the feature is not an empty screen on a fresh checkout.
    It only fires when the file is *absent*: an operator who deletes every zone
    writes an empty list, the file exists, and their decision stands rather than
    being quietly undone on the next page load.
    """
    global _cache
    p = Path(path or ZONE_FILE)
    if not p.exists():
        seeded = [Zone.from_dict({**d, "created_ts": time.time()}) for d in _DEFAULT_ZONES]
        save_zones(seeded, p)
        return seeded
    stamp = p.stat().st_mtime
    if _cache is not None and _cache[0] == stamp and path is None:
        return list(_cache[1])
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A half-written or hand-edited file must not take the dashboard down
        # with it. No zones is a survivable state; a traceback on page load is
        # not, and the operator can see the file is broken from the empty list.
        return []
    out = [Zone.from_dict(d) for d in raw.get("zones", [])]
    if path is None:
        _cache = (stamp, list(out))
    return out


def save_zones(zones: list[Zone], path: Path | None = None) -> Path:
    global _cache
    p = Path(path or ZONE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note": "GodsEye geofence zones - core/geofence.py",
               "zones": [z.as_dict() for z in zones]}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _cache = None
    return p


def get_zone(zone_id: str, path: Path | None = None) -> Zone | None:
    for z in load_zones(path):
        if z.id == zone_id:
            return z
    return None


def resolve(zone: "Zone | str") -> Zone:
    """Take either a Zone or its id, the way the rest of the module is called.

    Raises rather than returning None on an unknown id: every caller below is
    about to compute an answer *about* a zone, and a None here would surface as
    an empty result set that looks like "no vehicles" instead of "no zone".
    """
    if isinstance(zone, Zone):
        return zone
    z = get_zone(str(zone))
    if z is None:
        raise KeyError(f"no zone with id {zone!r} - see geofence.load_zones()")
    return z


def define_circle(name: str, lat: float, lon: float, radius_km: float, *,
                  zone_id: str | None = None, note: str = "",
                  alert_on: str = "watchlist", alert_classes: list | None = None,
                  severity: str = "", path: Path | None = None) -> Zone:
    """Create or replace a circular zone and persist it."""
    z = Zone(id=zone_id or _slug(name), name=name, kind="circle", lat=float(lat),
             lon=float(lon), radius_km=float(radius_km), note=note, alert_on=alert_on,
             alert_classes=list(alert_classes or []), severity=severity,
             created_ts=time.time())
    return _upsert(z, path)


def define_polygon(name: str, points: list, *, zone_id: str | None = None,
                   note: str = "", alert_on: str = "watchlist",
                   alert_classes: list | None = None, severity: str = "",
                   path: Path | None = None) -> Zone:
    """Create or replace a polygon zone from [[lat, lon], ...] and persist it.

    Three vertices are the minimum a boundary can have; fewer is a line, and a
    line contains nothing, so it is refused here rather than silently producing
    a zone that never matches a camera.
    """
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 3:
        raise ValueError("a polygon zone needs at least 3 points")
    z = Zone(id=zone_id or _slug(name), name=name, kind="polygon", points=pts,
             note=note, alert_on=alert_on, alert_classes=list(alert_classes or []),
             severity=severity, created_ts=time.time())
    return _upsert(z, path)


def _upsert(zone: Zone, path: Path | None = None) -> Zone:
    zones = [z for z in load_zones(path) if z.id != zone.id]
    zones.append(zone)
    save_zones(zones, path)
    return zone


def delete_zone(zone_id: str, path: Path | None = None) -> bool:
    zones = load_zones(path)
    kept = [z for z in zones if z.id != zone_id]
    if len(kept) == len(zones):
        return False
    save_zones(kept, path)
    return True


# --- membership ---------------------------------------------------------
def cameras_in(zone: "Zone | str", net: CameraNetwork | None = None) -> list[str]:
    """Camera ids whose coordinates fall inside the shape, in id order."""
    z = resolve(zone)
    net = net or network()
    return sorted(cid for cid, cam in net.cameras.items()
                  if z.contains(cam["lat"], cam["lon"]))


def zones_at(camera_id: str, net: CameraNetwork | None = None,
             path: Path | None = None) -> list[Zone]:
    """Every zone this camera sits inside - what the ingest path needs to know
    to decide whether a fresh sighting is a zone entry at all."""
    net = net or network()
    cam = net.cameras.get(camera_id)
    if not cam:
        return []
    return [z for z in load_zones(path) if z.contains(cam["lat"], cam["lon"])]


def boundary(zone: "Zone | str", net: CameraNetwork | None = None) -> dict:
    """The doors: which links cross the boundary, and which cameras guard them.

    A "portal" is a camera inside the zone with at least one road link to a
    camera outside it. Portals are the only junctions at which this platform can
    witness a crossing, so they are the structural ceiling on everything else
    the module claims.
    """
    z = resolve(zone)
    net = net or network()
    inside = set(cameras_in(z, net))
    portals: dict[str, list[str]] = {}
    crossings: list[tuple[str, str]] = []
    for cid in sorted(inside):
        outward = sorted(n for n in net.neighbours(cid) if n not in inside)
        if outward:
            portals[cid] = outward
            crossings.extend((cid, o) for o in outward)
    interior = sorted(cid for cid in inside if cid not in portals)
    return {"inside": sorted(inside), "portals": portals, "interior": interior,
            "crossing_links": crossings,
            "gateways_inside": sorted(c for c in inside
                                      if net.cameras[c].get("type") == "gateway")}


# --- visits -------------------------------------------------------------
@dataclass
class Visit:
    """One continuous run of sightings inside a zone, and its honest edges.

    `entry_confirmed` and `exit_confirmed` are what separate a witnessed
    crossing from a coincidence of the query window. A visit whose first
    sighting is the first sighting of the window may have started hours before
    it; a visit with no outside sighting after it may be a vehicle still in the
    zone or a vehicle that left through a road with no camera on it.
    """
    plate: str
    zone_id: str
    entered_ts: float
    last_inside_ts: float
    entry_camera: str
    last_inside_camera: str
    cameras: list
    inside_sightings: int
    vehicle_class: str | None
    entry_confirmed: bool
    entry_from_camera: str | None
    exit_confirmed: bool
    exit_ts: float | None
    exit_to_camera: str | None
    # Lower bound: the span the vehicle was demonstrably inside. Upper bound:
    # the span between the sightings that bracket the visit from outside, which
    # is None when either bracket is missing rather than guessed at.
    dwell_observed_min: float
    dwell_upper_min: float | None
    still_inside: bool
    silence_min: float
    inside_probability: float
    verdict: str
    pointed_outward: bool
    basis: str
    basis_samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def _window(hours: float, end: float | None, conn=None) -> tuple[float, float]:
    """A look-back window anchored to the data's own present, not the wall clock.

    Same reasoning as core/trajectory.lookback(): a seeded database stops
    advancing when seeding finishes, so a wall-clock window drifts entirely past
    the last sighting overnight and every zone reports zero vehicles - which
    reads as a quiet city rather than as a stale query.
    """
    end = end if end is not None else db.latest_ts(conn=conn)
    return end - hours * 3600.0, end


def _survival(lags: list, silence_min: float) -> tuple[float, int]:
    """P(not yet seen outside | unseen for this long), from the zone's own data.

    `lags` are the minutes between a vehicle's last sighting inside the zone and
    its first sighting outside it, over every visit in this window that was seen
    to end - the zone's own answer to "how long does a vehicle go unseen before
    it turns up outside again". Worth naming its bias: a vehicle that left
    through an uncovered road and was never seen again contributes nothing,
    which pulls the distribution short. Short lags make this function *less*
    willing to say a vehicle is still inside, which is the direction the error
    should point when the alternative is a cordon around an empty car park.

    The empirical survival function rather than a fitted exponential: dwell in a
    city zone is nothing like memoryless - a delivery van at a loading bay and a
    car crossing Esplanade at 60 km/h are two populations, and a single rate
    parameter splits the difference between them and describes neither.
    """
    if len(lags) < MIN_LAG_SAMPLES:
        return math.exp(-silence_min / max(FALLBACK_DWELL_MIN, 1e-6)), len(lags)
    longer = sum(1 for l in lags if l > silence_min)
    # Laplace smoothing, so a silence longer than every lag on record comes back
    # as "very unlikely" rather than as a flat zero that no evidence supports.
    return (longer + 0.5) / (len(lags) + 1.0), len(lags)


def _verdict(p: float) -> str:
    if p >= 0.60:
        return "likely inside"
    if p >= 0.25:
        return "possibly inside"
    return "probably left unseen"


def _pointed_out(sighting: dict, camera_id: str, inside: set,
                 net: CameraNetwork) -> bool:
    """Was the vehicle's last observed heading aimed at a road out of the zone?

    Positive evidence of departure, and the only kind available: nothing
    observes a vehicle leaving through an uncovered junction, but a vehicle
    photographed at a portal facing the exit road usually took it. Used to
    discount a "still inside" claim, never to assert a departure on its own.
    """
    h = _heading_of(sighting)
    if h is None:
        return False
    for nbr in net.neighbours(camera_id):
        if nbr in inside:
            continue
        if _angle_gap(h, net.heading(camera_id, nbr)) <= OUTBOUND_TOLERANCE_DEG:
            return True
    return False


def visits(zone: "Zone | str", hours: float = DEFAULT_WINDOW_HOURS,
           end: float | None = None, net: CameraNetwork | None = None,
           conn=None, sources: tuple | None = None,
           min_confidence: float = 0.0) -> list[Visit]:
    """Every zone visit in the window, one per continuous run of inside sightings.

    A vehicle that goes in, out and in again produces two visits, because that
    is two crossings and a control room asked about crossings. Sightings are not
    restricted to camera provenance by default: an operator enrolment is a human
    attesting that a vehicle was at a junction, which is exactly the observation
    a geofence needs. Pass `sources=(config.CAMERA_SOURCE,)` to measure the
    camera network alone, the way core/analytics.py does.

    The passes are ordered so the expensive part happens once - the exit-lag
    distribution that grades "still inside" is built from the completed visits
    of this same window, so the zone is calibrated against its own traffic at
    the time of day being asked about, not against a global average.
    """
    z = resolve(zone)
    net = net or network()
    conn = conn or db.connect()
    since, until = _window(hours, end, conn=conn)

    inside_cams = set(cameras_in(z, net))
    if not inside_cams:
        return []

    sql = "SELECT * FROM sightings WHERE ts >= ? AND ts <= ?"
    params: list = [since, until]
    if sources:
        sql += f" AND source IN ({','.join('?' * len(sources))})"
        params.extend(sources)
    if min_confidence > 0:
        sql += " AND confidence >= ?"
        params.append(min_confidence)
    rows = db.rows(sql + " ORDER BY plate, ts", tuple(params), conn=conn)

    # Pass one: cut each plate's timeline into runs of consecutive inside
    # sightings, carrying the outside sightings that bracket each run.
    by_plate: dict[str, list[dict]] = {}
    for r in rows:
        by_plate.setdefault(r["plate"], []).append(r)

    raw_runs: list[tuple[str, list[dict], dict | None, dict | None]] = []
    for plate, seq in by_plate.items():
        run: list[dict] = []
        before: dict | None = None
        last_outside: dict | None = None
        for s in seq:
            if s["camera_id"] in inside_cams:
                if not run:
                    before = last_outside
                run.append(s)
            else:
                if run:
                    raw_runs.append((plate, run, before, s))
                    run = []
                last_outside = s
        if run:
            raw_runs.append((plate, run, before, None))

    lags = [((r[3]["ts"] - r[1][-1]["ts"]) / 60.0) for r in raw_runs if r[3] is not None]

    out: list[Visit] = []
    for plate, run, before, after in raw_runs:
        first, last = run[0], run[-1]
        silence = max(0.0, (until - last["ts"]) / 60.0)
        pointed = (_pointed_out(last, last["camera_id"], inside_cams, net)
                   if after is None else False)
        if after is None:
            p, n = _survival(lags, silence)
            if pointed:
                # Halve rather than zero it. A vehicle facing the exit road
                # usually leaves, but a delivery van indicating towards the
                # main road and then parking is an ordinary event, so this is
                # a discount on the claim, not a refutation of it.
                p *= 0.5
            basis = (f"{n} completed visit(s) in this window set the scale"
                     if n >= MIN_LAG_SAMPLES
                     else f"only {n} completed visit(s) here - graded against a "
                          f"generic {FALLBACK_DWELL_MIN:.0f} min decay, not this zone")
            if pointed:
                basis += "; last seen pointed at a road out of the zone"
        else:
            p, n, basis = 0.0, len(lags), "seen outside the zone after this visit"

        upper = ((after["ts"] - before["ts"]) / 60.0
                 if (after is not None and before is not None) else None)

        out.append(Visit(
            plate=plate, zone_id=z.id,
            entered_ts=first["ts"], last_inside_ts=last["ts"],
            entry_camera=first["camera_id"], last_inside_camera=last["camera_id"],
            cameras=sorted({s["camera_id"] for s in run}),
            inside_sightings=len(run),
            vehicle_class=last.get("vehicle_class"),
            entry_confirmed=before is not None,
            entry_from_camera=before["camera_id"] if before else None,
            exit_confirmed=after is not None,
            exit_ts=after["ts"] if after else None,
            exit_to_camera=after["camera_id"] if after else None,
            dwell_observed_min=round((last["ts"] - first["ts"]) / 60.0, 1),
            dwell_upper_min=round(upper, 1) if upper is not None else None,
            still_inside=after is None,
            silence_min=round(silence, 1),
            inside_probability=round(p, 3),
            verdict=_verdict(p) if after is None else "left",
            pointed_outward=pointed,
            basis=basis, basis_samples=n,
        ))
    out.sort(key=lambda v: v.entered_ts)
    return out


# --- observability ------------------------------------------------------
def observability(zone: "Zone | str", hours: float = DEFAULT_WINDOW_HOURS,
                  end: float | None = None, net: CameraNetwork | None = None,
                  conn=None) -> dict:
    """Can this zone see what it claims to see - structurally, and in practice.

    Two halves, and the second is the one that matters. The structural half is
    geometry: how many cameras are inside, which of them sit on a boundary link,
    whether a gateway to the wider road network is enclosed. The measured half
    asks the data what actually happened: of the vehicles that appeared inside
    this zone, what share were seen approaching from outside first. A zone can
    look well guarded on a map and still score badly here, because the roads
    people really use to get in are not the roads with cameras on them.

    `usable` is the flag a dashboard should gate on. It is false for a zone with
    no cameras (it observes nothing), for a zone with one camera (it cannot
    distinguish being in the area from being at that junction, so its dwell
    times are revisit intervals), and for a zone with no portal (nothing can
    ever be witnessed crossing the boundary).
    """
    z = resolve(zone)
    net = net or network()
    b = boundary(z, net)
    inside, portals = b["inside"], b["portals"]

    vs = visits(z, hours=hours, end=end, net=net, conn=conn) if inside else []
    entries = len(vs)
    confirmed = sum(1 for v in vs if v.entry_confirmed)
    at_portal = sum(1 for v in vs if v.entry_camera in portals)
    exits = sum(1 for v in vs if v.exit_confirmed)
    used_portals = sorted({v.entry_camera for v in vs if v.entry_camera in portals}
                          | {v.last_inside_camera for v in vs if v.exit_confirmed
                             and v.last_inside_camera in portals})

    if not inside:
        verdict = "blind - no camera falls inside this boundary"
    elif len(inside) == 1:
        verdict = ("single camera - this zone cannot tell 'inside the area' from "
                   "'at that junction', and its dwell times are revisit intervals")
    elif not portals:
        verdict = "sealed - no link leaves this zone, so no crossing can be witnessed"
    elif entries and confirmed / entries < 0.5:
        verdict = (f"porous - only {confirmed} of {entries} vehicles were seen "
                   "approaching from outside, so most get in unobserved")
    else:
        verdict = "observable"

    return {
        "zone": z.id, "name": z.name,
        "cameras": [{"id": c, "name": net.name(c),
                     "portal": c in portals,
                     "gateway": c in b["gateways_inside"]} for c in inside],
        "camera_count": len(inside),
        "portal_count": len(portals),
        "portals": {c: [{"id": o, "name": net.name(o)} for o in outs]
                    for c, outs in portals.items()},
        "interior_cameras": b["interior"],
        "crossing_links": len(b["crossing_links"]),
        # A gateway inside the zone is a hole that geometry cannot close: a
        # vehicle can leave the modelled network entirely from in here, so its
        # silence is not evidence that it stayed.
        "gateways_inside": b["gateways_inside"],
        "window_hours": hours,
        "entries_observed": entries,
        "entries_confirmed_crossing": confirmed,
        "crossing_capture": round(confirmed / entries, 3) if entries else None,
        "entries_at_a_portal": at_portal,
        "exits_confirmed": exits,
        "exit_capture": round(exits / entries, 3) if entries else None,
        "portals_actually_used": used_portals,
        "portals_never_used": sorted(set(portals) - set(used_portals)),
        "usable": bool(len(inside) >= 2 and portals),
        "verdict": verdict,
    }


# --- the control room's questions ---------------------------------------
def activity(zone: "Zone | str", hours: float = DEFAULT_WINDOW_HOURS,
             end: float | None = None, net: CameraNetwork | None = None,
             conn=None, sources: tuple | None = None) -> dict:
    """Entered, left, still inside, and how long they stayed - in one pass.

    The four questions arrive together in a control room and share all of their
    work, so they are answered together rather than by four functions that each
    re-read the window. `still_inside` is ranked by how well the claim is
    supported, so the top of that list is the part of it worth acting on.

    The `observability` block is attached rather than left to the caller: the
    same numbers mean different things in a zone with six covered doors and in
    a zone with one, and a dashboard that shows the vehicle list without the
    coverage caveat is the failure this module exists to prevent.
    """
    z = resolve(zone)
    net = net or network()
    conn = conn or db.connect()
    since, until = _window(hours, end, conn=conn)
    vs = visits(z, hours=hours, end=until, net=net, conn=conn, sources=sources)

    entered = [v for v in vs if v.entry_confirmed]
    left = [v for v in vs if v.exit_confirmed]
    still = sorted((v for v in vs if v.still_inside),
                   key=lambda v: (-v.inside_probability, v.silence_min))
    completed = [v.dwell_observed_min for v in left]
    completed.sort()

    def _median(xs: list) -> float | None:
        if not xs:
            return None
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    by_class: dict[str, int] = {}
    for v in vs:
        by_class[v.vehicle_class or "unknown"] = by_class.get(v.vehicle_class or "unknown", 0) + 1

    return {
        "zone": z.id, "name": z.name, "note": z.note,
        "window": {"since": since, "until": until, "hours": hours},
        "cameras": cameras_in(z, net),
        "visits": len(vs),
        "unique_vehicles": len({v.plate for v in vs}),
        "entered": [v.as_dict() for v in entered],
        "entered_count": len(entered),
        # Vehicles whose first sighting of the window was already inside. Not
        # entries - the window simply opened while they were there - and
        # counting them as entries is how a zone reports a rush hour it never saw.
        "already_inside_at_window_start": len(vs) - len(entered),
        "left": [v.as_dict() for v in left],
        "left_count": len(left),
        "still_inside": [v.as_dict() for v in still],
        "still_inside_count": len(still),
        "still_inside_supported": sum(1 for v in still if v.inside_probability >= 0.60),
        "dwell": {
            "completed_visits": len(completed),
            "median_minutes": _median(completed),
            "p90_minutes": (completed[min(len(completed) - 1, int(0.9 * len(completed)))]
                            if completed else None),
            "longest_minutes": completed[-1] if completed else None,
            "note": ("dwell is bounded, not measured - dwell_observed_min is the span "
                     "the vehicle was demonstrably inside and dwell_upper_min the span "
                     "between the sightings bracketing the visit from outside"),
        },
        "by_vehicle_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "observability": observability(z, hours=hours, end=until, net=net, conn=conn),
    }


def dwell_table(zone: "Zone | str", hours: float = DEFAULT_WINDOW_HOURS,
                end: float | None = None, net: CameraNetwork | None = None,
                conn=None, top: int = 25) -> list[dict]:
    """Per-vehicle time in the zone, longest first - the loitering question.

    Aggregates across visits, because a vehicle that circled out and back is one
    vehicle spending a long time in an area, which is what an operator watching
    a cordon means. Sorting on the *lower* bound is deliberate: a long upper
    bound can be manufactured by two sightings twelve hours apart, and ranking
    on it would put the least-observed vehicles at the top of a list that is
    read as the most suspicious.
    """
    z = resolve(zone)
    agg: dict[str, dict] = {}
    for v in visits(z, hours=hours, end=end, net=net, conn=conn):
        a = agg.setdefault(v.plate, {
            "plate": v.plate, "zone": z.id, "visits": 0, "sightings": 0,
            "observed_minutes": 0.0, "upper_minutes": 0.0, "upper_bounded": True,
            "cameras": set(), "first_ts": v.entered_ts, "last_ts": v.last_inside_ts,
            "still_inside": False, "inside_probability": 0.0,
            "vehicle_class": v.vehicle_class,
        })
        a["visits"] += 1
        a["sightings"] += v.inside_sightings
        a["observed_minutes"] += v.dwell_observed_min
        if v.dwell_upper_min is None:
            a["upper_bounded"] = False
        else:
            a["upper_minutes"] += v.dwell_upper_min
        a["cameras"].update(v.cameras)
        a["first_ts"] = min(a["first_ts"], v.entered_ts)
        a["last_ts"] = max(a["last_ts"], v.last_inside_ts)
        if v.still_inside:
            a["still_inside"] = True
            a["inside_probability"] = max(a["inside_probability"], v.inside_probability)

    out = []
    for a in agg.values():
        a["cameras"] = sorted(a["cameras"])
        a["observed_minutes"] = round(a["observed_minutes"], 1)
        a["upper_minutes"] = round(a["upper_minutes"], 1) if a["upper_bounded"] else None
        # One sighting inside is a pass-through, and its zero-minute dwell is
        # not a short stay - it is no measurement at all.
        a["measured"] = a["sightings"] >= 2
        out.append(a)
    out.sort(key=lambda a: (-a["observed_minutes"], -a["sightings"]))
    return out[:top]


# --- screening for alerts -----------------------------------------------
def _recently_alerted(plate: str, zone_id: str, now: float, conn) -> bool:
    """Has this plate already been raised for this zone recently.

    Read-only on purpose. This module never writes an alert - the orchestrator
    wires emission - but it must not hand back an event that the queue already
    carries, or every ingest tick would re-report the same entry.
    """
    rows = db.rows(
        "SELECT detail FROM alerts WHERE kind = ? AND plate = ? AND ts >= ?",
        (ALERT_KIND, plate, now - DEDUP_MINUTES * 60.0), conn=conn)
    for r in rows:
        try:
            if json.loads(r["detail"] or "{}").get("zone") == zone_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def screen(zone: "Zone | str | None" = None, hours: float = 1.0,
           end: float | None = None, net: CameraNetwork | None = None,
           conn=None, dedup: bool = True, confirmed_only: bool = False,
           path: Path | None = None) -> list[dict]:
    """Zone entries that deserve an alert, shaped like core/alerts.py's output.

    Returns events; it does not emit them. core/alerts.py owns the alert table
    and the dedup policy for the rules it implements, and a second module
    writing into that queue from the side would make the two disagree about
    what has already been raised. So this hands back the same dict shape
    `alerts._emit` returns, minus the `id` a write would have produced, and the
    caller decides whether to persist it.

    Three policies, set per zone. `watchlist` is the commercial behaviour: a
    plate already of interest crossed into an area. `any` is the restricted
    area, where the entry is itself the event. `alert_classes` narrows either to
    particular vehicle types, which is the lorry-ban and the bus-lane case and
    is the one thing here that a plate-only product cannot express.

    `confirmed_only` is the honest switch. An entry is only a *crossing* when
    the vehicle was seen outside the zone immediately before it; otherwise the
    window merely opened while the vehicle was already there. Left False by
    default because a watchlisted vehicle sitting inside a cordon is worth
    knowing about either way, but the flag rides on every event so an operator
    is never shown an inference dressed as a witnessed crossing.
    """
    net = net or network()
    conn = conn or db.connect()
    # Pinned once. The database is written to while this runs, so leaving `end`
    # as None would let each zone below resolve its own slightly later network
    # clock and report a set of entries no single moment ever contained.
    end = end if end is not None else db.latest_ts(conn=conn)
    targets = [resolve(zone)] if zone is not None else load_zones(path)
    watched = {w["plate"]: w for w in db.watchlist(conn=conn)}

    events: list[dict] = []
    for z in targets:
        if z.alert_on == "none":
            continue
        cams = cameras_in(z, net)
        if not cams:
            continue
        cover = observability(z, hours=max(hours, DEFAULT_WINDOW_HOURS),
                              end=end, net=net, conn=conn)
        for v in visits(z, hours=hours, end=end, net=net, conn=conn):
            if confirmed_only and not v.entry_confirmed:
                continue
            if z.alert_classes and (v.vehicle_class or "") not in z.alert_classes:
                continue

            w = watched.get(v.plate)
            if z.alert_on == "watchlist":
                if not w:
                    continue
                reason = w["reason"]
                severity = z.severity or w["severity"]
            else:
                reason = z.note or f"entered restricted zone {z.name}"
                severity = z.severity or "medium"
                if w and _SEVERITY_ORDER.get(w["severity"], 0) > _SEVERITY_ORDER.get(severity, 0):
                    severity = w["severity"]

            if dedup and _recently_alerted(v.plate, z.id, v.entered_ts, conn):
                continue

            how = ("crossed into" if v.entry_confirmed
                   else "was first seen inside")
            events.append({
                "ts": v.entered_ts,
                "kind": ALERT_KIND,
                "severity": severity,
                "plate": v.plate,
                "camera_id": v.entry_camera,
                "message": (f"GEOFENCE {v.plate} {how} {z.name} at "
                            f"{net.name(v.entry_camera)} - {reason}"),
                "detail": {
                    "zone": z.id, "zone_name": z.name, "reason": reason,
                    "policy": z.alert_on,
                    "camera": net.name(v.entry_camera),
                    "vehicle_class": v.vehicle_class,
                    "entry_confirmed": v.entry_confirmed,
                    "arrived_from": v.entry_from_camera,
                    "still_inside": v.still_inside,
                    "inside_probability": v.inside_probability,
                    "dwell_observed_min": v.dwell_observed_min,
                    "watchlisted": bool(w),
                    # Carried into the alert so an officer reading it later can
                    # see how much the zone could actually observe at the time,
                    # rather than having to go and ask.
                    "zone_usable": cover["usable"],
                    "zone_verdict": cover["verdict"],
                },
                "acked": 0,
                # Not an alerts-table column. It rides alongside so a dashboard
                # can sort witnessed crossings above window artefacts without
                # re-deriving them.
                "confirmed_crossing": v.entry_confirmed,
            })

    events.sort(key=lambda e: (-_SEVERITY_ORDER.get(e["severity"], 0), -e["ts"]))
    return events


def summary(hours: float = DEFAULT_WINDOW_HOURS, end: float | None = None,
            net: CameraNetwork | None = None, conn=None,
            path: Path | None = None) -> list[dict]:
    """One row per zone for an overview table - counts plus the coverage caveat.

    Kept deliberately thin. Anything that needs the vehicle lists calls
    `activity()` for the one zone the operator opened, because building full
    visit lists for every zone on every dashboard rerun is work nobody is
    looking at.
    """
    net = net or network()
    conn = conn or db.connect()
    # Same reason as screen(): one clock for the whole table, so two rows of it
    # are never counting over windows that ended a few seconds apart.
    end = end if end is not None else db.latest_ts(conn=conn)
    out = []
    for z in load_zones(path):
        a = activity(z, hours=hours, end=end, net=net, conn=conn)
        out.append({
            "zone": z.id, "name": z.name, "kind": z.kind, "note": z.note,
            "alert_on": z.alert_on,
            "cameras": len(a["cameras"]),
            "visits": a["visits"], "unique_vehicles": a["unique_vehicles"],
            "entered": a["entered_count"], "left": a["left_count"],
            "still_inside": a["still_inside_count"],
            "still_inside_supported": a["still_inside_supported"],
            "median_dwell_min": a["dwell"]["median_minutes"],
            "usable": a["observability"]["usable"],
            "crossing_capture": a["observability"]["crossing_capture"],
            "verdict": a["observability"]["verdict"],
            "centre": z.centre(),
        })
    return out
