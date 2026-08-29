"""City traffic simulator — the stand-in for a live ANPR camera network.

A fleet of vehicles is routed across the camera graph on realistic origin-
destination demand, with a time-of-day profile and standing congestion on the
corridors that are actually congested in Kolkata. Every time a vehicle passes
a camera the simulator produces a *plate image*, degrades it for the conditions
at that camera at that hour, and pushes it through the real OCR engine. What
lands in the database is therefore what the engine read - errors included -
which is the point: the trajectory and analytics layers have to cope with
imperfect reads exactly as they would in deployment.

Running every capture through the engine costs ~75 ms, so live mode caps how
many are decoded per tick and falls back to a calibrated error model for the
rest; `ocr_mode` on each sighting records which path produced it.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import config
from anpr import plates
from core import alerts as alert_rules
from core import db
from core.network import CameraNetwork, network

VEHICLE_CLASSES = [
    ("car", 0.52), ("two_wheeler", 0.24), ("auto", 0.09),
    ("taxi", 0.07), ("bus", 0.03), ("truck", 0.05),
]

# Corridors that are genuinely congested in Kolkata; multiplies free-flow time.
STANDING_CONGESTION = {
    ("CAM-07", "CAM-15"): 2.6,   # Ultadanga - Chingrighata, the city's worst
    ("CAM-24", "CAM-25"): 2.5,   # Howrah Bridge
    ("CAM-04", "CAM-07"): 2.2,   # Shyambazar - Kankurgachi
    ("CAM-22", "CAM-27"): 2.2,   # Park Circus - Gariahat
    ("CAM-20", "CAM-24"): 2.0,   # Strand Road into the bridge
    ("CAM-18", "CAM-19"): 2.0,   # Sealdah approach
    ("CAM-27", "CAM-28"): 1.9,   # Rashbehari Avenue
    ("CAM-01", "CAM-04"): 1.8,   # BT Road
    ("CAM-15", "CAM-16"): 1.7,   # EM Bypass mid
    ("CAM-08", "CAM-07"): 1.7,   # VIP Road into Ultadanga
}

# Hour-of-day demand multiplier (index 0 = midnight).
HOUR_PROFILE = [0.18, 0.12, 0.10, 0.11, 0.16, 0.32, 0.58, 0.86, 1.00, 0.94,
                0.78, 0.72, 0.74, 0.76, 0.74, 0.79, 0.90, 1.00, 0.98, 0.84,
                0.62, 0.48, 0.36, 0.25]

# Plate-read accuracy per capture condition, measured by anpr.benchmark. Used
# only for captures the live loop cannot afford to decode for real.
CONDITION_ACCURACY = {
    "clean": 1.00, "night": 1.00, "glare": 0.98, "rain": 1.00, "motion_blur": 0.95,
    "angled": 0.86, "dirty": 0.71, "damaged": 0.77, "low_res": 1.00, "mixed": 0.85,
}


@dataclass
class Vehicle:
    plate: str
    vehicle_class: str
    commercial: bool
    route: list[str]
    leg: int = 0
    next_ts: float = 0.0
    pace: float = 1.0                    # <1 drives faster than the crowd
    speeder: bool = False                # a small share of drivers really do
    trips: int = 0

    @property
    def camera(self) -> str:
        return self.route[self.leg]

    @property
    def next_camera(self) -> str | None:
        return self.route[self.leg + 1] if self.leg + 1 < len(self.route) else None


@dataclass
class Capture:
    """One camera crossing before OCR."""
    ts: float
    camera_id: str
    true_plate: str
    vehicle_class: str
    commercial: bool
    speed_kmph: float
    lane: int
    heading: float
    direction: str
    next_camera: str | None
    condition: str


class CitySimulator:
    def __init__(self, net: CameraNetwork | None = None, seed: int = 11,
                 fleet_size: int = config.FLEET_SIZE, engine=None, weather: str = "auto"):
        self.net = net or network()
        self.rng = random.Random(seed)
        self.weather = weather
        self.engine = engine
        self.fleet: list[Vehicle] = []
        self.cameras = list(self.net.cameras)
        self.gateways = self.net.gateways or self.cameras
        self.stats = {"captures": 0, "reads_ocr": 0, "reads_modelled": 0,
                      "dropped_low_confidence": 0, "misreads": 0}
        for _ in range(fleet_size):
            self.fleet.append(self._spawn(time.time()))

    # --- fleet ---------------------------------------------------------
    def _spawn(self, now: float, plate: str | None = None) -> Vehicle:
        vclass = self.rng.choices([c for c, _ in VEHICLE_CLASSES],
                                  [w for _, w in VEHICLE_CLASSES])[0]
        commercial = vclass in ("bus", "truck", "taxi", "auto")
        route = self._route(now)
        return Vehicle(plate=plate or plates.random_plate(self.rng), vehicle_class=vclass,
                       commercial=commercial, route=route,
                       next_ts=now + self.rng.uniform(0, 900),
                       pace=self.rng.uniform(0.85, 1.3),
                       speeder=self.rng.random() < 0.05)

    def _route(self, now: float, origin: str | None = None) -> list[str]:
        """Pick a destination and route to it.

        A vehicle's next trip starts where its last one ended - anything else
        teleports the plate across the city and manufactures exactly the
        impossible-speed pattern the alert rules exist to catch. Trips start or
        end at a gateway most of the time: city traffic is largely through
        traffic and commuting from the edges.
        """
        for _ in range(12):
            a = origin or (self.rng.choice(self.gateways) if self.rng.random() < 0.6
                           else self.rng.choice(self.cameras))
            b = (self.rng.choice(self.gateways) if origin and self.rng.random() < 0.45
                 else self.rng.choice(self.cameras))
            if a == b:
                continue
            route = self.net.route(a, b)
            if len(route) >= 3:
                return list(route)
        return list(self.net.route(origin or self.cameras[0], self.cameras[-1]))

    # --- movement ------------------------------------------------------
    def _demand(self, ts: float) -> float:
        hour = time.localtime(ts).tm_hour
        return HOUR_PROFILE[hour]

    def _congestion(self, a: str, b: str, ts: float) -> float:
        """Travel-time multiplier on a link: standing congestion x rush hour."""
        base = STANDING_CONGESTION.get((a, b)) or STANDING_CONGESTION.get((b, a)) or 1.0
        rush = 0.75 + 0.85 * self._demand(ts)
        return base * rush * self.rng.uniform(0.85, 1.25)

    def _condition(self, ts: float) -> str:
        """Capture condition for this crossing: darkness and weather dominate."""
        hour = time.localtime(ts).tm_hour
        night = hour >= 19 or hour <= 5
        wet = self.weather == "rain" or (self.weather == "auto" and self.rng.random() < 0.12)
        r = self.rng.random()
        if wet and r < 0.55:
            return "rain"
        if night and r < 0.55:
            return "night"
        if r < 0.34:
            return "clean"
        return self.rng.choices(
            ["motion_blur", "angled", "glare", "low_res", "dirty", "damaged", "mixed"],
            [0.24, 0.20, 0.14, 0.16, 0.10, 0.06, 0.10])[0]

    def advance(self, until_ts: float, max_captures: int = 5000) -> list[Capture]:
        """Move every vehicle up to `until_ts`, returning the camera crossings."""
        out: list[Capture] = []
        for v in self.fleet:
            while v.next_ts <= until_ts and len(out) < max_captures:
                nxt = v.next_camera
                ts = v.next_ts
                km = self.net.link_km(v.camera, nxt) if nxt else 0.0
                free = self.net.link_free_kmph(v.camera, nxt) if nxt else 40.0
                if nxt:
                    factor = self._congestion(v.camera, nxt, ts) * v.pace
                    # Nobody outruns the road by more than a third, however
                    # empty it is - except the ~5% who genuinely speed, which is
                    # what gives the enforcement rule something real to catch.
                    ceiling = free * (2.1 if v.speeder else 1.35)
                    kmph = min(max(6.0, free / factor), ceiling)
                    if v.speeder:
                        kmph = max(kmph, free * self.rng.uniform(1.0, 1.9))
                    heading = self.net.heading(v.camera, nxt)
                    direction = self.net.direction(v.camera, nxt)
                    travel_s = km / kmph * 3600.0
                else:
                    kmph, heading, direction, travel_s = free * 0.6, 0.0, "-", 0.0

                out.append(Capture(
                    ts=ts, camera_id=v.camera, true_plate=v.plate,
                    vehicle_class=v.vehicle_class, commercial=v.commercial,
                    speed_kmph=round(kmph * self.rng.uniform(0.9, 1.1), 1),
                    lane=self.rng.randint(1, max(1, self.net.camera(v.camera)["lanes"])),
                    heading=heading, direction=direction, next_camera=nxt,
                    condition=self._condition(ts)))

                if nxt is None:                       # trip finished - new journey
                    v.trips += 1
                    if self.rng.random() < 0.78:
                        # drives on from where it stopped after a short halt
                        v.route = self._route(ts, origin=v.camera)
                        idle = self.rng.expovariate(1 / 900.0) / max(self._demand(ts), 0.15)
                        v.next_ts = ts + min(idle, 5400)
                    else:
                        # parked off-network / left the city: reappears elsewhere,
                        # but only after long enough that it is physically possible
                        v.route = self._route(ts)
                        v.next_ts = ts + self.rng.uniform(45 * 60, 180 * 60)
                    v.leg = 0
                else:
                    v.leg += 1
                    v.next_ts = ts + travel_s
        out.sort(key=lambda c: c.ts)
        return out

    # --- OCR -----------------------------------------------------------
    def _engine(self):
        if self.engine is None:
            from anpr.ocr import engine
            self.engine = engine()
        return self.engine

    def read(self, cap: Capture, use_ocr: bool) -> tuple[str, float, str, str]:
        """-> (plate as read, confidence, ocr mode, winning binarisation variant)"""
        if use_ocr:
            img = plates.capture(cap.true_plate, cap.condition, self.rng)
            read = self._engine().read(img.image)
            self.stats["reads_ocr"] += 1
            if read.text and read.text != cap.true_plate:
                self.stats["misreads"] += 1
            return read.text, read.confidence, "engine", read.variant
        # calibrated stand-in: same error *rate* as the engine, cheap
        acc = CONDITION_ACCURACY.get(cap.condition, 0.9)
        self.stats["reads_modelled"] += 1
        if self.rng.random() < acc:
            return cap.true_plate, self.rng.uniform(0.86, 1.0), "modelled", ""
        # Wrong reads carry lower confidence, as the engine's do, so the same
        # share of them falls below the floor and never reaches the database.
        return (_corrupt(cap.true_plate, self.rng),
                self.rng.uniform(0.35, 0.88), "modelled", "")

    def to_sighting(self, cap: Capture, use_ocr: bool) -> db.Sighting | None:
        plate, conf, mode, variant = self.read(cap, use_ocr)
        self.stats["captures"] += 1
        if not plate or conf < config.MIN_PLATE_CONFIDENCE:
            # A real ANPR node discards reads it cannot stand behind. So do we —
            # which is why a trajectory can have gaps the operator must bridge.
            self.stats["dropped_low_confidence"] += 1
            return None
        return db.Sighting(
            ts=cap.ts, camera_id=cap.camera_id, plate=plate, confidence=round(conf, 4),
            speed_kmph=cap.speed_kmph, lane=cap.lane, heading=round(cap.heading, 1),
            direction=cap.direction, next_camera=cap.next_camera,
            vehicle_class=cap.vehicle_class, condition=cap.condition,
            ocr_variant=f"{mode}:{variant}" if variant else mode,
            true_plate=cap.true_plate)

    # --- driving the store ---------------------------------------------
    def step(self, until_ts: float | None = None, ocr_budget: int | None = None,
             conn=None, run_alerts: bool = True) -> tuple[list[db.Sighting], list[dict]]:
        """Advance to `until_ts`, store the sightings, return them with any alerts."""
        conn = conn or db.connect()
        until_ts = until_ts if until_ts is not None else time.time()
        budget = config.INLINE_OCR_MAX_PER_TICK if ocr_budget is None else ocr_budget
        if not config.INLINE_OCR:
            budget = 0
        caps = self.advance(until_ts)
        sightings, fired = [], []
        for i, cap in enumerate(caps):
            s = self.to_sighting(cap, use_ocr=i < budget)
            if s is None:
                continue
            s.id = db.add_sighting(s, conn=conn)
            sightings.append(s)
            if run_alerts:
                fired.extend(alert_rules.evaluate(s.as_dict(), conn=conn, net=self.net))
        return sightings, fired

    def history(self, hours: float = 6.0, end_ts: float | None = None,
                ocr_fraction: float = 0.05, conn=None, verbose: bool = True) -> int:
        """Backfill `hours` of traffic ending now, in one batch write."""
        conn = conn or db.connect()
        end_ts = end_ts or time.time()
        start = end_ts - hours * 3600.0
        for v in self.fleet:                     # rewind the fleet to the window start
            v.next_ts = start + self.rng.uniform(0, 600)
        caps = self.advance(end_ts, max_captures=400_000)
        rows = []
        for cap in caps:
            use = self.rng.random() < ocr_fraction
            s = self.to_sighting(cap, use_ocr=use)
            if s is not None:
                rows.append(s)
        db.add_sightings(rows, conn=conn)
        if verbose:
            print(f"  {len(caps)} camera crossings -> {len(rows)} stored sightings "
                  f"({self.stats['reads_ocr']} decoded by the engine, "
                  f"{self.stats['dropped_low_confidence']} dropped below confidence floor)")
        return len(rows)

    # --- scripted incidents --------------------------------------------
    def inject_clone(self, conn=None, gap_minutes: float = 3.0) -> str:
        """Same plate at opposite ends of the city minutes apart."""
        conn = conn or db.connect()
        v = self.rng.choice(self.fleet)
        a, b = "CAM-35", "CAM-12"                     # Kona toll and New Town, ~22 km apart
        now = time.time()
        for cam, ts in ((a, now - gap_minutes * 60), (b, now)):
            s = db.Sighting(ts=ts, camera_id=cam, plate=v.plate, confidence=0.93,
                            speed_kmph=48.0, lane=1, heading=90.0, direction="E",
                            next_camera=None, vehicle_class=v.vehicle_class,
                            condition="clean", ocr_variant="scripted", true_plate=v.plate)
            s.id = db.add_sighting(s, conn=conn)
            alert_rules.evaluate(s.as_dict(), conn=conn, net=self.net)
        return v.plate

    def inject_loiterer(self, camera: str = "CAM-20", passes: int = 5, conn=None) -> str:
        """A vehicle circling one junction — the classic reconnaissance pattern."""
        conn = conn or db.connect()
        v = self.rng.choice(self.fleet)
        now = time.time()
        for i in range(passes):
            ts = now - (passes - i) * 7 * 60
            s = db.Sighting(ts=ts, camera_id=camera, plate=v.plate, confidence=0.91,
                            speed_kmph=22.0, lane=2, heading=180.0, direction="S",
                            next_camera=None, vehicle_class=v.vehicle_class,
                            condition="clean", ocr_variant="scripted", true_plate=v.plate)
            s.id = db.add_sighting(s, conn=conn)
            alert_rules.evaluate(s.as_dict(), conn=conn, net=self.net)
        return v.plate

    def sample_plates(self, n: int = 6) -> list[str]:
        return [v.plate for v in self.rng.sample(self.fleet, min(n, len(self.fleet)))]


def _corrupt(plate: str, rng: random.Random) -> str:
    """Apply one plausible OCR confusion, mirroring the engine's real failures."""
    from anpr.ocr import _CONF

    idx = [i for i, c in enumerate(plate) if c in _CONF]
    if not idx:
        return plate
    i = rng.choice(idx)
    return plate[:i] + rng.choice(sorted(_CONF[plate[i]])) + plate[i + 1:]
