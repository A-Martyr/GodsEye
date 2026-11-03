"""Convoy detection that survives a busy road.

`trajectory.co_travellers` answers "who else was at that camera around then"
and ranks by how often. On a quiet road that is the same question as "who is
travelling with this vehicle". On a trunk corridor at 09:00 it is not: a camera
that passes forty vehicles a minute puts hundreds of unrelated plates inside
any two-minute window, and the top of a raw co-occurrence count is simply
whoever drives the busiest corridor most often. Measured on this repo's own
database, three of the four highest-ranked plates by that count are also among
the eight most-sighted vehicles on the network - the ranking is largely
reporting mileage.

That matters more here than in most analytics. Convoy analysis is sold for
trafficking and organised-crime work, so a false convoy does not produce a bad
chart, it points an investigation at a commuter who happens to leave home when
the target does. The bar has to be higher than "appeared together often".

Two questions, asked separately
-------------------------------
1. *Did they meet more often than chance?* For every sighting of A, count how
   many vehicles that camera actually passed in the same window and multiply
   by B's own share of that camera's traffic. That product is the number of
   B-sightings the window would be expected to contain if B's journeys had
   nothing to do with A's, and it is computed from the real local density, so
   a rush-hour junction sets its own baseline rather than being scored against
   the city average. Summed over A's sightings it gives an expected count; the
   observed count is tested against it as a Poisson upper tail.

2. *Given that they met, did they keep station?* This is the question the
   naive version cannot ask at all, and it is the one that separates a convoy
   from a shared commute. Take a leg A drove between two junctions and check
   whether B drove the same leg between the same two junctions as its own
   consecutive movement. If B trails A by 9 s entering the link and 11 s
   leaving it, the two vehicles held formation across it. If B is 90 s behind
   at one end and 20 s ahead at the other, they were on the same road, which
   is not the same thing. Under the null the entry and exit gaps are
   independent draws from the co-occurrence window, so the chance of a drift
   that small has a closed form, and the count of station-keeping legs is
   tested as a binomial upper tail.

The two tests are combined by adding their log-evidence, which is what `score`
is. They are not strictly independent - the second conditions on the first - so
`score` is a ranking quantity, not a calibrated probability, and both
components are reported separately so an analyst who trusts only one of them
can rank on that one alone.

What this deliberately does not do
----------------------------------
It does not call a shared commute a convoy. A pair that co-occurs far above
chance but never shares a single leg in formation gets the verdict
"shared route" and a sentence saying so, because that is a real and common
pattern and hiding it inside the convoy list is how the false positive gets
made. It also does not treat repeat visits to one junction as accumulating
evidence of travel together: those land in `junction_repeats`, reported and
excluded from the station-keeping test, since two neighbours who both pass the
same junction at 09:00 every day generate dozens of them.

Honesty about the assumptions. The Poisson null assumes a vehicle's crossings
of a given camera are exchangeable within that camera's traffic, which
understates chance co-occurrence for two vehicles that share a *timetable* - so
the co-occurrence p-value alone will over-call commuter pairs, and that is
exactly why no verdict rests on it alone. The binomial null assumes the entry
and exit gaps are independent given co-occurrence, which is true of unrelated
vehicles and false of vehicles that were following each other, which is the
direction that makes the test conservative. Neither p-value is corrected for
the number of pairs tested: a whole-network sweep examines thousands, so read
the sweep's tail as a ranking, and the named-plate query as a test.
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import networkx as nx

import config
from core import db
from core.network import CameraNetwork, network

# How far either side of a sighting another plate counts as "at the same place
# at the same time". Two minutes is generous for a convoy - vehicles in
# formation are seconds apart - and it is deliberately loose, because this
# window only decides what gets *considered*. The station-keeping test is what
# decides what gets believed, and a loose window makes that test stronger by
# giving an unrelated vehicle more room to drift.
WINDOW_S = float(getattr(config, "CONVOY_WINDOW_S", 120.0))

# How much the gap between two vehicles may change across one link before they
# stop counting as holding formation. Fifteen seconds absorbs a signal phase
# and a lane change without admitting two vehicles that merely used the same
# road inside the window.
LOCK_DRIFT_S = float(getattr(config, "CONVOY_LOCK_DRIFT_S", 15.0))

# Beyond this a pair of consecutive sightings is not one journey leg - the
# vehicle stopped, parked, or left the network and came back. Same reasoning
# and same default as core/forecast.MAX_LEG_MINUTES.
MAX_LEG_MINUTES = float(getattr(config, "CONVOY_MAX_LEG_MIN", 45.0))

# A pair with fewer co-occurrences than this is not scored at all. Not because
# two meetings cannot be a convoy, but because the tails below are sharp enough
# that k=2 against a small expectation produces an impressive-looking number
# out of almost nothing.
MIN_PASSES = int(getattr(config, "CONVOY_MIN_PASSES", 3))

# Station-keeping legs needed before the word "convoy" is used.
MIN_LOCKED_HOPS = int(getattr(config, "CONVOY_MIN_LOCKED_HOPS", 3))

# Significance thresholds. P_STRONG is what a verdict rests on; P_WEAK is what
# is worth putting in front of an analyst at all.
P_STRONG = float(getattr(config, "CONVOY_P_STRONG", 1e-3))
P_WEAK = float(getattr(config, "CONVOY_P_WEAK", 0.05))

# Evidence from reads the engine barely stood behind is not evidence. Mirrors
# config.CLONE_MIN_CONFIDENCE, for the same reason.
MIN_CONFIDENCE = float(getattr(config, "CONVOY_MIN_CONFIDENCE",
                               getattr(config, "CLONE_MIN_CONFIDENCE", 0.70)))

# Two registrations this close in OCR confusion distance are the kind of pair
# this engine manufactures out of one vehicle. Not grounds for dismissal - a
# real convoy can contain two similar plates - but grounds for saying so.
CONFUSABLE_DISTANCE = float(getattr(config, "CONVOY_CONFUSABLE_DISTANCE", 1.0))

_LOG10 = math.log(10.0)
_FLOOR_LOG10 = -300.0            # past this the tails are numerical noise


# --- tails ---------------------------------------------------------------
def _log10_poisson_tail(k: int, lam: float) -> float:
    """log10 P(K >= k) for K ~ Poisson(lam), computed in log space.

    The naive `1 - cdf` underflows to zero for exactly the cases that matter -
    a pair seen together fifty times against an expectation of eight - and a
    p-value pinned at the smallest representable float gives every strong pair
    the same score, which destroys the ranking precisely at the top of it.
    Summing outward from the k-th term with the leading factor kept as a
    logarithm keeps the answer meaningful well past 1e-300.
    """
    if k <= 0:
        return 0.0
    lam = max(lam, 1e-12)
    if k <= lam:
        # Not a surprise at all; the tail is O(1) and direct summation is exact
        # enough to say so.
        below, term = 0.0, math.exp(-lam)
        for j in range(k):
            if j:
                term *= lam / j
            below += term
        return math.log10(max(1.0 - below, 1e-300))
    log_term = -lam + k * math.log(lam) - math.lgamma(k + 1)
    ratio, total, j = 1.0, 1.0, k + 1
    while ratio > 1e-16 * total and j < k + 10000:
        ratio *= lam / j
        total += ratio
        j += 1
    return max((log_term + math.log(total)) / _LOG10, _FLOOR_LOG10)


def _log10_binom_tail(k: int, n: int, q: float) -> float:
    """log10 P(X >= k) for X ~ Binomial(n, q).

    Summed exactly rather than approximated: n here is a count of journey legs,
    rarely past a few dozen, and a normal approximation is worst exactly where
    this is used - small n, small q, k in the far tail.
    """
    if k <= 0:
        return 0.0
    if k > n:
        return _FLOOR_LOG10
    q = min(max(q, 1e-12), 1.0 - 1e-12)
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
            + j * math.log(q) + (n - j) * math.log1p(-q) for j in range(k, n + 1)]
    top = max(logs)
    return max((top + math.log(sum(math.exp(x - top) for x in logs))) / _LOG10,
               _FLOOR_LOG10)


def _lock_probability(window_s: float = WINDOW_S,
                      drift_s: float = LOCK_DRIFT_S) -> float:
    """Chance that two unrelated vehicles appear to hold formation over one leg.

    Both the entry gap and the exit gap are, under the null, independent draws
    from the co-occurrence window - anything outside it would not have been
    paired in the first place. Their difference is therefore triangular on
    [-2w, 2w], and P(|difference| <= d) has the closed form below. At the
    defaults (w=120 s, d=15 s) that is 12.1%, and it is the number the binomial
    test runs against: a pair gets no credit for holding station until it does
    so on noticeably more than one leg in eight.
    """
    span = 2.0 * window_s
    if drift_s >= span:
        return 1.0
    return 1.0 - (1.0 - drift_s / span) ** 2


# --- evidence ------------------------------------------------------------
@dataclass
class SharedPass:
    """One camera that saw both vehicles inside the window."""
    camera_id: str
    camera_name: str
    ts: float
    gap_s: float                 # positive: the partner arrived after the subject
    local_traffic: int           # vehicles that camera passed in the same window
    expected: float              # partner-sightings chance would have put there


@dataclass
class SharedHop:
    """One link both vehicles crossed as their own consecutive movement."""
    from_camera: str
    to_camera: str
    from_name: str
    to_name: str
    ts: float
    minutes: float
    km: float
    direction: str
    entry_gap_s: float           # partner's lead or lag entering the link
    exit_gap_s: float            # ... and leaving it
    drift_s: float               # how much that changed - small means formation
    locked: bool
    direct_link: bool            # a real road link, not a routed multi-hop path


@dataclass
class ConvoyPair:
    """One pair of registrations, both tests, and everything behind them.

    Deliberately wide. A ranked list with a score and nothing else cannot be
    disagreed with, and this output is the sort that ends up justifying a day
    of someone's time: `expected_passes` against `passes` is the whole
    co-occurrence argument in two numbers, `locked_hops` against `hops` is the
    whole station-keeping argument, and `hop_evidence` names the junctions and
    the seconds so a reader can check the claim against the sighting log.
    """
    plate_a: str
    plate_b: str
    verdict: str = "coincidence"
    confidence: str = "low"
    confident: bool = False
    score: float = 0.0
    passes: int = 0
    expected_passes: float = 0.0
    lift: float = 0.0
    co_log10_p: float = 0.0
    hops: int = 0
    locked_hops: int = 0
    chain_links: int = 0          # adjacent shared legs - a run of L gives L-1
    longest_chain: int = 0
    station_log10_p: float = 0.0
    chance_lock_rate: float = 0.0  # how often chance alone fakes a held formation
    junction_repeats: int = 0     # co-occurrences at a junction they never left together
    median_gap_s: float = 0.0
    median_drift_s: float = 0.0
    cameras: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    first_ts: float = 0.0
    last_ts: float = 0.0
    vehicle_classes: list[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    weak_reads: int = 0
    confusable_plates: bool = False
    reason: str = ""
    pass_evidence: list[SharedPass] = field(default_factory=list)
    hop_evidence: list[SharedHop] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --- the index -----------------------------------------------------------
class SightingIndex:
    """Everything the scoring needs, laid out once.

    Both entry points below need per-camera traffic, per-camera plate shares
    and per-plate ordered journeys, and a whole-network sweep needs them for
    every plate at once. Rebuilding them per pair would make the sweep
    quadratic in database work for no reason, so the index is a first-class
    object a caller can build once and hand to several queries.
    """

    def __init__(self, rows: list[dict], window_s: float = WINDOW_S):
        self.window_s = float(window_s)
        self.rows = rows
        self.by_plate: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self.by_camera: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self.meta: dict[str, dict] = {}          # plate -> class, mean confidence
        conf_sum: dict[str, float] = defaultdict(float)
        conf_n: dict[str, int] = defaultdict(int)
        for r in rows:
            p, c, t = r["plate"], r["camera_id"], float(r["ts"])
            self.by_plate[p].append((t, c))
            self.by_camera[c].append((t, p))
            conf_sum[p] += float(r.get("confidence") or 0.0)
            conf_n[p] += 1
            if r.get("vehicle_class"):
                self.meta.setdefault(p, {})["vehicle_class"] = r["vehicle_class"]
        for p in self.by_plate:
            self.by_plate[p].sort()
            self.meta.setdefault(p, {})["mean_confidence"] = (
                conf_sum[p] / conf_n[p] if conf_n[p] else 0.0)

        self.cam_ts: dict[str, list[float]] = {}
        self.cam_total: dict[str, int] = {}
        self.cam_share: dict[str, dict[str, float]] = {}
        for c, lst in self.by_camera.items():
            lst.sort()
            self.cam_ts[c] = [t for t, _ in lst]
            self.cam_total[c] = len(lst)
            counts: dict[str, int] = defaultdict(int)
            for _, p in lst:
                counts[p] += 1
            self.cam_share[c] = {p: n / len(lst) for p, n in counts.items()}

        # plate -> camera -> sorted (ts, position in that plate's journey). The
        # position is what lets a shared leg be checked as "B's *next* sighting
        # was the far end", rather than the much weaker "B was also there later".
        self.plate_cam: dict[str, dict[str, list[tuple[float, int]]]] = {}
        for p, journey in self.by_plate.items():
            per: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for i, (t, c) in enumerate(journey):
                per[c].append((t, i))
            self.plate_cam[p] = per

    def local_traffic(self, camera_id: str, ts: float) -> int:
        """Vehicles that camera passed within the window, excluding this one."""
        arr = self.cam_ts.get(camera_id)
        if not arr:
            return 0
        w = self.window_s
        return max(bisect.bisect_right(arr, ts + w)
                   - bisect.bisect_left(arr, ts - w) - 1, 0)

    def share(self, camera_id: str, plate: str) -> float:
        """This plate's share of that camera's traffic over the whole window."""
        return self.cam_share.get(camera_id, {}).get(plate, 0.0)

    def co_occurrences(self, min_passes: int = MIN_PASSES) -> dict[tuple[str, str], int]:
        """Every plate pair seen at a common camera inside the window, counted.

        The candidate generator for the whole-network sweep, and the only place
        the naive count is computed - everything after this is about deciding
        which of these counts mean anything.
        """
        counts: dict[tuple[str, str], int] = defaultdict(int)
        w = self.window_s
        for c, lst in self.by_camera.items():
            arr = self.cam_ts[c]
            for i, (t, p) in enumerate(lst):
                for j in range(i + 1, bisect.bisect_right(arr, t + w, lo=i + 1)):
                    q = lst[j][1]
                    if q != p:
                        counts[(p, q) if p < q else (q, p)] += 1
        return {k: v for k, v in counts.items() if v >= min_passes}


def build_index(since: float | None = None, until: float | None = None,
                hours: float | None = None, window_s: float = WINDOW_S,
                sources: tuple[str, ...] | None = None, conn=None) -> SightingIndex:
    """Load the sightings a convoy query will run over.

    `hours` is measured back from the newest sighting rather than from wall
    clock, for the reason trajectory.lookback() documents: a seeded database
    stops advancing when seeding finishes, so a wall-clock window sits entirely
    after the last row by the following morning and every query comes back
    empty.

    Enrolments are excluded by default. An enrolment's *place* is attested by
    an operator and its *time* is chosen by a heuristic (config.ENROL_TARGET_
    KMPH picks a plausible gap), so two enrolments can be near-simultaneous by
    construction. That is precisely the signature this module reads as
    evidence, and it is not an observation of two vehicles being in one place.
    Pass `sources=()` to include everything anyway.
    """
    conn = conn or db.connect()
    sources = sources if sources is not None else (config.CAMERA_SOURCE,)
    if hours is not None and since is None:
        since = db.latest_ts(sources or None, conn=conn) - hours * 3600.0
    where, params = [], []
    if since is not None:
        where.append("ts >= ?")
        params.append(since)
    if until is not None:
        where.append("ts <= ?")
        params.append(until)
    if sources:
        where.append("source IN (%s)" % ",".join("?" * len(sources)))
        params.extend(sources)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.rows(
        "SELECT plate, camera_id, ts, confidence, vehicle_class FROM sightings"
        f"{clause} ORDER BY ts", tuple(params), conn=conn)
    return SightingIndex(rows, window_s=window_s)


# --- scoring one pair ----------------------------------------------------
def _passes(a: str, b: str, index: SightingIndex,
            net: CameraNetwork) -> list[SharedPass]:
    """Co-occurrences anchored on A, each carrying its own chance expectation."""
    out: list[SharedPass] = []
    w = index.window_s
    for t, c in index.by_plate.get(a, []):
        local = index.local_traffic(c, t)
        for bt, _ in index.plate_cam.get(b, {}).get(c, []):
            if abs(bt - t) <= w:
                out.append(SharedPass(
                    camera_id=c, camera_name=net.name(c), ts=t, gap_s=bt - t,
                    local_traffic=local,
                    expected=round(local * index.share(c, b), 3)))
    return out


def _expected_passes(a: str, b: str, index: SightingIndex) -> float:
    """Co-occurrences chance alone would produce, taking the kinder direction.

    Anchoring on A and anchoring on B give different expectations - they
    condition on different journeys - and there is no principled way to pick
    one. Taking the larger is the conservative choice: it is whichever way of
    looking at the pair makes their meetings least surprising, so a pair only
    clears the threshold when it clears it from both directions.
    """
    def anchored(x: str, y: str) -> float:
        return sum(index.local_traffic(c, t) * index.share(c, y)
                   for t, c in index.by_plate.get(x, []))
    return max(anchored(a, b), anchored(b, a))


def _hops(a: str, b: str, index: SightingIndex,
          net: CameraNetwork) -> tuple[list[SharedHop], list[int]]:
    """Legs A drove that B drove too, as B's own consecutive movement.

    The "consecutive for B as well" requirement is what makes this a shared
    journey rather than a shared destination. Without it a vehicle that
    wandered off to three other junctions in between would still be credited
    with the leg, because it was seen at both ends of it.
    """
    journey = index.by_plate.get(a, [])
    b_cams = index.plate_cam.get(b, {})
    w, max_leg = index.window_s, MAX_LEG_MINUTES * 60.0
    hops: list[SharedHop] = []
    leg_index: list[int] = []

    def nearest(camera_id: str, ts: float) -> tuple[float, int] | None:
        seen = b_cams.get(camera_id)
        if not seen:
            return None
        best = None
        j = bisect.bisect_left(seen, (ts - w, -1))
        while j < len(seen) and seen[j][0] <= ts + w:
            if best is None or abs(seen[j][0] - ts) < abs(best[0] - ts):
                best = seen[j]
            j += 1
        return best

    for i in range(len(journey) - 1):
        (t1, c1), (t2, c2) = journey[i], journey[i + 1]
        if c1 == c2 or not (0.0 < t2 - t1 <= max_leg):
            continue
        first, second = nearest(c1, t1), nearest(c2, t2)
        if first is None or second is None:
            continue
        if second[1] != first[1] + 1:
            continue
        entry, exit_gap = first[0] - t1, second[0] - t2
        drift = exit_gap - entry
        hops.append(SharedHop(
            from_camera=c1, to_camera=c2,
            from_name=net.name(c1), to_name=net.name(c2),
            ts=t1, minutes=round((t2 - t1) / 60.0, 2),
            km=round(net.link_km(c1, c2) or net.route_km(c1, c2) or 0.0, 2),
            direction=net.direction(c1, c2),
            entry_gap_s=round(entry, 1), exit_gap_s=round(exit_gap, 1),
            drift_s=round(drift, 1), locked=abs(drift) <= LOCK_DRIFT_S,
            direct_link=net.graph.has_edge(c1, c2)))
        leg_index.append(i)
    return hops, leg_index


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _confusable(a: str, b: str) -> bool:
    """Are these two registrations one OCR slip apart?

    Imported lazily. anpr.ocr pulls in cv2 and the whole reader stack, which a
    dashboard tab asking a database question should not pay for, and a checkout
    without the reader installed should still be able to run convoy analysis. A
    pair that cannot be checked is reported as not confusable rather than as
    unknown, which is safe because the flag only ever downgrades a verdict.
    """
    try:
        from anpr.ocr import confusion_distance
    except Exception:
        return False
    return confusion_distance(a, b) <= CONFUSABLE_DISTANCE


def score_pair(a: str, b: str, index: SightingIndex,
               net: CameraNetwork | None = None) -> ConvoyPair:
    """Both tests, the evidence behind them, and a verdict someone can argue with."""
    net = net or network()
    passes = _passes(a, b, index, net)
    hops, leg_index = _hops(a, b, index, net)
    pair = ConvoyPair(plate_a=a, plate_b=b, passes=len(passes), hops=len(hops))
    if not passes:
        pair.verdict, pair.reason = "coincidence", (
            "these two registrations were never at one camera together")
        return pair

    k = len(passes)
    lam = _expected_passes(a, b, index)
    pair.expected_passes = round(lam, 2)
    pair.lift = round(k / lam, 2) if lam > 0 else float(k)
    pair.co_log10_p = round(_log10_poisson_tail(k, lam), 2)

    locked = [h for h in hops if h.locked]
    pair.locked_hops = len(locked)
    pair.chance_lock_rate = round(_lock_probability(index.window_s, LOCK_DRIFT_S), 4)
    pair.station_log10_p = round(
        _log10_binom_tail(len(locked), len(hops), pair.chance_lock_rate), 2)

    # Consecutive legs of A's journey that were both shared. Two vehicles
    # crossing three junctions in a row together is one sustained event, not
    # three independent ones, and it is far stronger evidence than three shared
    # legs scattered across a day - which is the distinction the naive count
    # cannot make.
    runs, run = [], 1
    for x, y in zip(leg_index, leg_index[1:]):
        if y == x + 1:
            run += 1
        else:
            runs.append(run)
            run = 1
    if leg_index:
        runs.append(run)
    pair.longest_chain = max(runs) if runs else 0
    pair.chain_links = sum(r - 1 for r in runs)

    hop_cameras = {h.from_camera for h in hops} | {h.to_camera for h in hops}
    pair.junction_repeats = sum(1 for p in passes if p.camera_id not in hop_cameras)
    pair.cameras = sorted({p.camera_id for p in passes})
    pair.links = sorted({f"{h.from_camera}>{h.to_camera}" for h in hops})
    pair.first_ts = min(p.ts for p in passes)
    pair.last_ts = max(p.ts for p in passes)
    pair.median_gap_s = round(_median([abs(h.entry_gap_s) for h in locked]
                                      or [abs(p.gap_s) for p in passes]), 1)
    pair.median_drift_s = round(_median([abs(h.drift_s) for h in hops]), 1)

    meta_a, meta_b = index.meta.get(a, {}), index.meta.get(b, {})
    pair.vehicle_classes = [meta_a.get("vehicle_class") or "unknown",
                            meta_b.get("vehicle_class") or "unknown"]
    confs = [float(meta_a.get("mean_confidence", 0.0)),
             float(meta_b.get("mean_confidence", 0.0))]
    pair.mean_confidence = round(sum(confs) / 2.0, 3)
    pair.weak_reads = sum(1 for c in confs if c < MIN_CONFIDENCE)
    pair.confusable_plates = _confusable(a, b)

    pair.score = round(-(pair.co_log10_p + pair.station_log10_p), 2)
    pair.pass_evidence = sorted(passes, key=lambda p: abs(p.gap_s))[:12]
    pair.hop_evidence = sorted(hops, key=lambda h: abs(h.drift_s))[:12]
    _verdict(pair)
    return pair


def _verdict(pair: ConvoyPair) -> None:
    """Turn the two tests into a word and a sentence.

    The ladder is deliberately blunt. An operator reading this is deciding
    whether to spend a day on a vehicle, and a continuous score with no
    threshold means every pair looks slightly suspicious. Each rung names the
    evidence it stands on, so a reader can disagree with the rung rather than
    with an unexplained number.

    The order matters and is not the obvious one. Station-keeping is asked
    first and co-occurrence second, so a pair that met far above chance but
    never held formation drops to "shared route" rather than being promoted by
    the strength of the co-occurrence alone. Only a pair that fails *both*
    tests is called a coincidence - an earlier arrangement here called a pair
    seen 34 times against an expectation of 6.7 "inside what the traffic alone
    produces", which was flatly contradicted by the numbers printed beside it.
    """
    co_p, station_p = 10.0 ** pair.co_log10_p, 10.0 ** pair.station_log10_p
    a, b = pair.plate_a, pair.plate_b
    lock_rate = pair.chance_lock_rate or _lock_probability()

    if pair.passes < MIN_PASSES:
        pair.verdict, pair.confidence = "insufficient evidence", "low"
        pair.reason = (f"only {pair.passes} shared camera pass(es) - too few to separate "
                       "from chance in either direction")
        return

    if pair.locked_hops >= MIN_LOCKED_HOPS and pair.chain_links >= 1 and station_p <= P_STRONG:
        pair.verdict = "convoy"
        pair.confidence = ("high" if pair.locked_hops >= 5 and pair.longest_chain >= 3
                           and not pair.weak_reads else "medium")
        pair.reason = (
            f"{b} crossed {pair.locked_hops} junction-to-junction links with {a} while "
            f"holding station to within {LOCK_DRIFT_S:.0f}s, up to {pair.longest_chain} of "
            f"them back to back, typically {pair.median_gap_s:.0f}s apart. Chance would "
            f"put {pair.expected_passes:.1f} of their sightings in a shared window; "
            f"{pair.passes} are")
        if pair.median_gap_s > 30.0:
            # Worth saying out loud: this is not two vehicles nose to tail. A
            # separation that stays constant across many links is the signature
            # of a deliberate tail, and an operator who pictures a tight pair
            # will look for the wrong thing on the CCTV.
            pair.reason += (f". They are not nose to tail - the {pair.median_gap_s:.0f}s "
                            "separation is held rather than closed, which is what a "
                            "following vehicle looks like")
    elif pair.locked_hops >= 1 and station_p <= P_WEAK:
        pair.verdict = "travelling together"
        pair.confidence = "medium" if pair.locked_hops >= 2 else "low"
        pair.reason = (
            f"{pair.locked_hops} of {pair.hops} shared link crossing(s) held formation, "
            f"more than the {lock_rate:.0%} rate chance produces, but not enough of them - "
            "or not consecutively enough - to call it a convoy")
    elif co_p <= P_STRONG:
        pair.verdict = "shared route"
        pair.confidence = "low"
        held = ("not once did they cross a link in formation" if not pair.locked_hops
                else f"only {pair.locked_hops} of {pair.hops} crossings held formation, "
                     f"which is inside the {lock_rate:.0%} chance rate")
        pair.reason = (
            f"seen together {pair.passes} times against a chance expectation of "
            f"{pair.expected_passes:.1f} ({pair.lift:.1f}x), so they are genuinely on the "
            f"road at the same times - but {held} (median gap change "
            f"{pair.median_drift_s:.0f}s). This is what a shared commute or a common "
            "corridor looks like, not a convoy")
    else:
        pair.verdict, pair.confidence = "coincidence", "low"
        pair.reason = (
            f"{pair.passes} shared passes against {pair.expected_passes:.1f} expected on "
            f"these cameras at these times - inside what the traffic alone produces")

    if pair.junction_repeats and pair.verdict in ("shared route", "coincidence"):
        pair.reason += (f"; {pair.junction_repeats} of the passes were at junctions they "
                        "never left together")
    if pair.weak_reads:
        if pair.confidence == "high":
            pair.confidence = "medium"
        elif pair.confidence == "medium":
            pair.confidence = "low"
        pair.reason += (f"; {pair.weak_reads} of the two plates averages below "
                        f"{MIN_CONFIDENCE:.2f} OCR confidence, so the identities themselves "
                        "are soft")
    if pair.confusable_plates:
        pair.confidence = "low"
        pair.reason += ("; the two registrations are within one OCR substitution of each "
                        "other, so check they are not one vehicle read two ways")
    pair.confident = pair.verdict == "convoy" and pair.confidence == "high"


# --- entry points --------------------------------------------------------
# The ladder as an ordering. "insufficient evidence" sits level with
# "coincidence" on purpose: neither is a finding, and sorting one above the
# other would put thin data ahead of a measured negative.
_RANK = {"coincidence": 0, "insufficient evidence": 0, "shared route": 1,
         "travelling together": 2, "convoy": 3}


def partners(plate: str, hours: float | None = None, min_passes: int = MIN_PASSES,
             top: int = 10, index: SightingIndex | None = None,
             net: CameraNetwork | None = None, conn=None) -> dict:
    """Who is travelling with this vehicle, scored against chance.

    The replacement for trajectory.co_travellers(). Same question, but every
    candidate carries what chance would have produced, whether the pair ever
    held formation across a link, and a verdict that separates a convoy from a
    shared commute instead of ranking both by raw count.
    """
    net = net or network()
    plate = plate.upper().replace(" ", "")
    index = index or build_index(hours=hours, conn=conn)
    journey = index.by_plate.get(plate)
    if not journey:
        return {"plate": plate, "known": False, "partners": [], "reliable": False,
                "reason": "no sighting of this registration in the window"}

    # Candidates first, scoring second: only plates that actually shared a
    # camera window can score, and there are far fewer of those than plates.
    seen: dict[str, int] = defaultdict(int)
    w = index.window_s
    for t, c in journey:
        for bt, other in index.by_camera.get(c, []):
            if other != plate and abs(bt - t) <= w:
                seen[other] += 1

    scored = [score_pair(plate, other, index, net)
              for other, n in seen.items() if n >= min_passes]
    scored.sort(key=lambda p: (-_RANK.get(p.verdict, 0), -p.locked_hops, -p.score))
    ranked = [p for p in scored if p.verdict != "coincidence"] or scored
    convoys = sum(1 for p in scored if p.verdict == "convoy")

    return {
        "plate": plate,
        "known": True,
        "sightings": len(journey),
        "window_s": w,
        "candidates": len(seen),
        "scored": len(scored),
        "convoys": convoys,
        "partners": [p.as_dict() for p in ranked[:top]],
        # A vehicle seen a handful of times cannot be shown to be travelling
        # with anyone. The tails will still produce a number for it, which is
        # exactly why this flag is here.
        "reliable": len(journey) >= 8,
        "basis": (f"{len(seen)} plates shared a camera window with {plate}; "
                  f"{len(scored)} met the {min_passes}-pass floor and "
                  f"{convoys} scored as a convoy"),
    }


def sweep(hours: float | None = None, min_passes: int = MIN_PASSES,
          min_verdict: str = "travelling together", top: int = 40,
          index: SightingIndex | None = None, net: CameraNetwork | None = None,
          conn=None) -> list[dict]:
    """Every pair on the network that looks like it is moving in formation.

    The question an analyst starts with - "show me who is moving in groups" -
    rather than the one they can only ask once they already have a suspect.
    Runs both tests over every plate pair that shared a camera window at all,
    then keeps the ones that clear `min_verdict`. Note that a sweep tests
    thousands of pairs and the p-values are uncorrected, so read this output as
    a ranked queue of leads and re-test the ones that matter with partners().
    """
    net = net or network()
    index = index or build_index(hours=hours, conn=conn)
    floor = _RANK.get(min_verdict, 2)
    out = [pair for pair in
           (score_pair(a, b, index, net) for a, b in index.co_occurrences(min_passes))
           if _RANK.get(pair.verdict, 0) >= floor]
    out.sort(key=lambda p: (-_RANK.get(p.verdict, 0), -p.locked_hops, -p.score))
    return [p.as_dict() for p in out[:top]]


def groups(hours: float | None = None, min_size: int = 2, top: int = 15,
           min_verdict: str = "travelling together",
           index: SightingIndex | None = None, net: CameraNetwork | None = None,
           conn=None) -> list[dict]:
    """Scored pairs joined into the groups of vehicles they imply.

    A group is a set of vehicles in which *every* pair independently scored,
    not a set that happens to be reachable through scored pairs. The
    difference is not academic: run against this repo's database, taking
    connected components of the pair graph returns one blob of 46 vehicles
    held together by 102 of its 1,035 possible pairs. That is not a
    46-vehicle fleet, it is the north-corridor traffic, and reporting it as a
    group would be the same failure as the naive co-occurrence count, one
    level up.

    So this takes maximal cliques instead, which is exactly the "every pair
    confirmed" condition, and every group is therefore mutually confirmed by
    construction. The honest cost is that cliques overlap and a real group
    with one weak internal pair is reported as its strongest complete subset -
    a five-vehicle convoy where two members never crossed a link together
    comes back as two overlapping fours. Two fields keep that visible instead
    of hiding it: `component_size` says how large the loosely connected
    neighbourhood around the group is, and `variants` counts the overlapping
    cliques folded into this row because they introduced no vehicle it does
    not already name. Every row returned therefore brings at least one
    registration the rows above it did not.
    """
    net = net or network()
    index = index or build_index(hours=hours, conn=conn)
    floor = _RANK.get(min_verdict, 2)
    scored: dict[tuple[str, str], ConvoyPair] = {}
    graph = nx.Graph()
    for a, b in index.co_occurrences(MIN_PASSES):
        pair = score_pair(a, b, index, net)
        if _RANK.get(pair.verdict, 0) >= floor:
            scored[(a, b)] = pair
            graph.add_edge(a, b)

    component_of = {}
    for component in nx.connected_components(graph):
        for member in component:
            component_of[member] = len(component)

    # find_cliques is output-sensitive and this graph is sparse, but a denser
    # one is theoretically exponential and this runs behind an HTTP request.
    # The cap turns a pathological graph into a truncated answer rather than a
    # hung dashboard.
    cliques = []
    for i, clique in enumerate(nx.find_cliques(graph) if graph else []):
        if i >= 20000:
            break
        cliques.append(sorted(clique))

    candidates = []
    for members in cliques:
        if len(members) < min_size:
            continue
        inner = [scored[(a, b)] for i, a in enumerate(members)
                 for b in members[i + 1:] if (a, b) in scored]
        strong = [p for p in inner if p.verdict == "convoy"]
        cameras = sorted({c for p in inner for c in p.cameras})
        # The group is only as good as its weakest internal pair - that pair is
        # the one an opposing analyst will pull on, so it is reported rather
        # than averaged away by the sum below.
        weakest = min(inner, key=lambda p: (_RANK.get(p.verdict, 0), p.score))
        candidates.append({
            "members": members,
            "size": len(members),
            "pairs": len(inner),
            "convoy_pairs": len(strong),
            "component_size": component_of.get(members[0], len(members)),
            "variants": 0,
            "score": round(sum(p.score for p in inner), 2),
            "weakest_pair": [weakest.plate_a, weakest.plate_b],
            "weakest_verdict": weakest.verdict,
            "weakest_score": weakest.score,
            "locked_hops": sum(p.locked_hops for p in inner),
            "cameras": cameras,
            "camera_names": [net.name(c) for c in cameras[:8]],
            "first_ts": min((p.first_ts for p in inner), default=0.0),
            "last_ts": max((p.last_ts for p in inner), default=0.0),
            "vehicle_classes": sorted({v for p in inner for v in p.vehicle_classes}),
            # Every internal pair has to be a convoy in its own right. A group
            # where one leg is only "travelling together" is a lead.
            "confident": len(strong) == len(inner),
            "reason": (
                f"{len(members)} vehicles, all {len(inner)} internal pairs scored as "
                f"travelling together ({len(strong)} of them as convoys); weakest link is "
                f"{weakest.plate_a}+{weakest.plate_b} at {weakest.verdict}"
                + ("" if component_of.get(members[0], 0) <= len(members) + 2 else
                   f". They sit inside a loosely connected neighbourhood of "
                   f"{component_of[members[0]]} vehicles, which usually means a shared "
                   "corridor - check the group is not just its busiest stretch")),
            "detail": [p.as_dict() for p in sorted(inner, key=lambda p: -p.score)[:6]],
        })

    candidates.sort(key=lambda g: (-g["size"], -g["convoy_pairs"], -g["score"]))
    out: list[dict] = []
    covered: set[str] = set()
    for group in candidates:
        novel = [m for m in group["members"] if m not in covered]
        if not novel and out:
            # Nothing here that a stronger group above does not already name.
            # Fold it in rather than dropping it silently - an analyst counting
            # groups should be able to see that this one existed.
            nearest = max(out, key=lambda g: len(set(g["members"]) & set(group["members"])))
            nearest["variants"] += 1
            continue
        group["novel_members"] = novel
        out.append(group)
        covered.update(group["members"])
        if len(out) >= top:
            break
    for group in out:
        if group["variants"]:
            group["reason"] += (f". {group['variants']} overlapping clique(s) of the same "
                                "vehicles were folded into this row")
    return out
