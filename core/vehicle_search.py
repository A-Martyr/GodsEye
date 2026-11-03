"""Find a vehicle when nobody caught the plate.

A plate-reader platform that can only be queried by registration is useless in
the one case that matters most - the witness saw the vehicle and did not get
the number. Commercial products sell the way out of this as two features
("describe the vehicle of interest in natural language", "find images without a
license plate"); both are really the same thing, which is a search over the
attributes a camera records *besides* the plate.

What this network actually records about a vehicle is: what kind it is, which
way it was pointing, how fast it was going, which junction saw it, when, and
how far the reader stood behind the read. That is the whole vocabulary. It is
smaller than a witness statement and much smaller than what a person will type,
and the entire design of this module follows from that gap.

Three rules run through it.

Say what you did not use. An operator who types "red truck" and is handed every
truck in the city will believe the reds were filtered - that is the failure
mode, and it is worse than returning nothing, because it converts a wide net
into a false narrowing that someone will act on. Colour, make, model and
occupancy are recognised terms here precisely so they can be reported as
recognised-but-not-recorded, term by term, rather than silently dropped.

A rule-based parser can enumerate itself. There is no model here and no new
dependency; the vocabulary is built from data/cameras.json and the vehicle
classes actually present in the store, so `vocabulary()` can print exactly what
the parser understands and `parse()` can list every token it did not. A
language model cannot do either, and for a tool whose output points officers at
members of the public, being able to say "I did not understand the word
'maroon'" is worth more than understanding it approximately.

A ranked list looks like knowledge. Everything below reports how much it
narrowed the field: how many sightings matched against how many were in the
window at all, how many separate vehicles are in the answer, and how many
independent attributes the description actually constrained. A description that
fits one sighting in three is not a description, and `discriminating` is false
when that is what happened.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field

import config
from core import db
from core.network import CameraNetwork, haversine_km, network

# --- tunables -----------------------------------------------------------
# "Speeding" means the same thing here as it does in core/alerts.py - above
# 1.6x the free-flow speed of the road - so an operator searching for speeding
# vehicles and the ingest rule that raises speeding alerts cannot disagree
# about what the word means. The difference is only the measurement: the alert
# rule uses the speed implied between two cameras, this uses the spot speed the
# camera itself reported.
SPEEDING_FACTOR = float(getattr(config, "SEARCH_SPEEDING_FACTOR", 1.6))
# ... and its mirror. Below this share of free flow the vehicle was crawling,
# which is what someone means by "stopped in traffic" or "moving slowly".
SLOW_FACTOR = float(getattr(config, "SEARCH_SLOW_FACTOR", 0.45))
# How far "near <place>" reaches beyond the junctions whose name, road or
# sector actually matched. Straight-line, because a witness saying "near Salt
# Lake" is describing a neighbourhood, not a route.
NEAR_KM = float(getattr(config, "SEARCH_NEAR_KM", 2.0))
# Ranking half-life. A description search is nearly always asked in order to
# find the vehicle *now*, so an equally good match from four hours ago ranks
# below one from twenty minutes ago.
RECENCY_HALFLIFE_H = float(getattr(config, "SEARCH_RECENCY_HALFLIFE_H", 6.0))
# Above this many candidate vehicles the answer is a list, not a lead: nobody
# works a shortlist of two hundred registrations.
MAX_ACTIONABLE = int(getattr(config, "SEARCH_MAX_ACTIONABLE", 40))
# A description matching more than this share of the traffic in its own window
# has not described anything.
BROAD_SHARE = float(getattr(config, "SEARCH_BROAD_SHARE", 0.10))
# Two independent attributes is the floor for calling a description a
# description. "trucks" is a category.
MIN_CONSTRAINTS = int(getattr(config, "SEARCH_MIN_CONSTRAINTS", 2))
# Reused rather than reinvented: core/clones.py already had to decide what a
# read too weak to build on looks like, and a description search should not
# hold a second opinion about the same number.
WEAK_READ = float(getattr(config, "CLONE_MIN_CONFIDENCE", 0.70))
# Ceiling on rows pulled into memory for ranking. The whole store is ~20k rows
# so this never fires here, but an unconstrained query against a year of a real
# city must not try to sort the city.
MAX_SCAN = int(getattr(config, "SEARCH_MAX_SCAN", 60000))

DEFAULT_LIMIT = 50

# --- vocabulary ---------------------------------------------------------
# The synonyms a person actually types, mapped onto the six values the store
# holds in sightings.vehicle_class. Multi-word entries are matched as phrases
# before single words, so "auto rickshaw" never resolves through "auto" alone
# and "yellow taxi" consumes its own colour word instead of reporting it as an
# attribute that was ignored.
VEHICLE_CLASS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "car": ("car", "cars", "sedan", "sedans", "hatchback", "hatch", "suv", "suvs",
            "private car", "motorcar", "four wheeler", "saloon"),
    "two_wheeler": ("two wheeler", "two wheelers", "twowheeler", "bike", "bikes",
                    "motorbike", "motorbikes", "motorcycle", "motorcycles", "scooter",
                    "scooters", "scooty", "moped", "pillion rider"),
    "auto": ("auto", "autos", "auto rickshaw", "autorickshaw", "rickshaw", "rickshaws",
             "tuk tuk", "tuktuk", "three wheeler", "toto", "e rickshaw"),
    "truck": ("truck", "trucks", "lorry", "lorries", "hgv", "goods vehicle", "tipper",
              "container", "trailer", "dumper", "tempo"),
    "taxi": ("taxi", "taxis", "taxicab", "cab", "cabs", "yellow taxi", "yellow cab"),
    "bus": ("bus", "buses", "busses", "minibus", "coach", "double decker"),
}

# Compass words to their exact code. The parser also accepts the codes
# themselves when they are two or three letters ("nne", "wsw"); single letters
# are refused because "n" and "s" turn up inside ordinary typing.
DIRECTION_WORDS: dict[str, str] = {
    "north": "N", "south": "S", "east": "E", "west": "W",
    "northeast": "NE", "north east": "NE", "northwest": "NW", "north west": "NW",
    "southeast": "SE", "south east": "SE", "southwest": "SW", "south west": "SW",
}
# Words that mark the compass term as a heading rather than a part of the city.
DIRECTION_CUES = frozenset({
    "heading", "headed", "head", "going", "goes", "travelling", "traveling",
    "travelled", "traveled", "moving", "driving", "drove", "bound", "towards",
    "toward", "direction", "facing", "outbound", "inbound", "away",
})
# ... and the ones that mark it as a part of the city.
PLACE_CUES = frozenset({
    "in", "at", "near", "around", "on", "by", "from", "within", "across",
    "side", "sector", "part", "area", "zone", "close",
})
PROXIMITY_CUES = frozenset({"near", "around", "close", "beside", "by", "outside"})

COMPASS_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

# Hour-of-day bands. Local hours, inclusive, as a set rather than a range so
# "night" can wrap midnight without any special case downstream.
TIME_BANDS: dict[str, tuple[int, ...]] = {
    "morning": (5, 6, 7, 8, 9, 10, 11),
    "afternoon": (12, 13, 14, 15, 16),
    "evening": (17, 18, 19, 20),
    "night": (21, 22, 23, 0, 1, 2, 3, 4),
    "midday": (11, 12, 13),
    "noon": (11, 12, 13),
    "dawn": (4, 5, 6, 7),
    "rush hour": (8, 9, 10, 17, 18, 19, 20),
    "peak hour": (8, 9, 10, 17, 18, 19, 20),
    "odd hours": (1, 2, 3, 4, 5),
}

# Terms this platform recognises and cannot act on. Keeping them in the
# vocabulary is the point: an unknown word is a typo, but "maroon" is a real
# description of a real vehicle that this network simply does not see, and the
# operator has to be told which of the two happened.
NOT_RECORDED: dict[str, dict] = {
    "colour": {
        "terms": ("red", "blue", "black", "white", "silver", "grey", "gray", "green",
                  "yellow", "orange", "brown", "maroon", "golden", "gold", "beige",
                  "purple", "pink", "cream", "navy", "dark", "light coloured"),
        "why": ("no camera on this network reports body colour - the reader works on "
                "a plate crop, not the vehicle - so the results below contain every "
                "colour"),
    },
    "make": {
        "terms": ("maruti", "suzuki", "hyundai", "tata", "mahindra", "honda", "toyota",
                  "kia", "ford", "bmw", "audi", "mercedes", "benz", "hero", "bajaj",
                  "tvs", "royal enfield", "bullet", "ashok leyland", "eicher", "volvo",
                  "skoda", "nissan", "renault", "jeep", "yamaha", "ktm"),
        "why": ("make recognition needs a vision model on the vehicle body; this "
                "platform classifies vehicles only into car / two_wheeler / auto / "
                "truck / taxi / bus"),
    },
    "model": {
        "terms": ("swift", "wagonr", "wagon r", "alto", "i10", "i20", "creta", "nexon",
                  "scorpio", "thar", "innova", "fortuner", "dzire", "baleno", "activa",
                  "pulsar", "splendor", "ertiga", "xuv", "bolero", "santro"),
        "why": ("model recognition needs a vision model on the vehicle body, which "
                "this platform does not run"),
    },
    "occupancy": {
        "terms": ("driver", "passenger", "passengers", "occupants", "man", "woman",
                  "men", "women", "helmet", "helmetless", "no helmet", "triple riding",
                  "overloaded", "empty"),
        "why": ("nothing about the occupants is recorded - the platform stores a "
                "registration, a place and a time, and deliberately stores no image "
                "of anyone"),
    },
    "condition": {
        "terms": ("damaged", "dented", "scratched", "rusty", "tinted", "sticker",
                  "stickers", "roof rack", "modified", "loud", "new", "old", "shiny",
                  "broken headlight", "no headlight"),
        "why": ("body condition and fittings are not recorded; only the plate read, "
                "its confidence, and the vehicle class are"),
    },
}

# Single words that name a *kind* of place rather than a place, so they must
# not become gazetteer keys of their own. Without this, "gate" matches three
# junctions and "road" matches most of the city, and a query that mentions
# either quietly acquires a location filter the operator never asked for.
GENERIC_PLACE_WORDS = frozenset({
    "crossing", "junction", "gate", "station", "road", "street", "avenue", "point",
    "bridge", "toll", "sector", "approach", "forecourt", "tower", "clock", "temple",
    "setu", "expressway", "five", "seven", "second", "first", "new", "the", "of",
    "and", "nh", "pwd", "gt", "sn", "apc", "ajc", "sp", "em", "v", "extension",
    "north", "south", "east", "west", "central", "major", "arterial", "broadway",
    # "city" is in here for a reason worth keeping: without it, "entering the
    # city" resolved through Science City and quietly turned a question about
    # the cordon into a question about one junction on the bypass.
    "city",
})

# Words with no filtering meaning that should not be reported back as
# "not understood" - doing so would bury the terms that genuinely were not.
STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "with",
    "any", "all", "some", "was", "were", "is", "are", "be", "been", "seen", "saw",
    "spotted", "sighted", "show", "find", "search", "look", "looking", "me", "my",
    "please", "vehicle", "vehicles", "that", "this", "it", "which", "who", "there",
    "near", "around", "close", "by", "from", "within", "across", "side", "part",
    "area", "zone", "heading", "headed", "going", "travelling", "traveling",
    "moving", "driving", "towards", "toward", "direction", "facing", "bound",
    "then", "last", "past", "over", "under", "about", "between",
    "plate", "number", "registration", "reg", "no", "colour", "color",
})

_CLOCK_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$")
_PLATE_RE = re.compile(r"^[a-z]{2}\d{1,2}[a-z]{0,3}\d{0,4}$")


# --- parsed query -------------------------------------------------------
@dataclass
class Ignored:
    """A term the parser understood and the platform cannot act on."""
    term: str
    attribute: str
    note: str


@dataclass
class Place:
    """One location phrase, resolved to junctions."""
    term: str
    kind: str                       # camera | camera_id | road | sector
    label: str
    cameras: list[str] = field(default_factory=list)
    specificity: float = 1.0        # how narrowly the phrase pinned the map
    expanded_km: float = 0.0        # "near X" reach, 0 when not a proximity phrase


@dataclass
class Query:
    """A plain-English phrase turned into something the store can be asked.

    Carries the negative space as well as the filter. `ignored`, `unrecognised`
    and `ambiguities` are not diagnostics - they are half the answer, and any
    caller that renders the results without rendering these is reproducing the
    exact failure this module exists to prevent.
    """
    text: str = ""
    anchor_ts: float = 0.0
    vehicle_classes: list[str] = field(default_factory=list)
    class_terms: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)        # every code accepted
    primary_directions: list[str] = field(default_factory=list)  # the exact ones
    direction_terms: list[str] = field(default_factory=list)
    places: list[Place] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    since: float | None = None
    until: float | None = None
    hours: list[int] = field(default_factory=list)
    time_terms: list[str] = field(default_factory=list)
    min_speed: float | None = None
    max_speed: float | None = None
    speeding: bool = False
    crawling: bool = False
    min_confidence: float | None = None
    plate_fragment: str = ""
    plate_anchor: str = "contains"      # contains | prefix | suffix
    gateways_only: bool = False
    # Set when two understood terms contradict each other, so the filter has no
    # rows by construction. Kept as a flag rather than an empty hour list
    # because "no hours" and "every hour" are the same absent clause in SQL, and
    # the difference between them is the whole answer.
    impossible: bool = False
    understood: list[str] = field(default_factory=list)
    ignored: list[Ignored] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    @property
    def constraints(self) -> int:
        """How many independent attributes this description actually pins down.

        The single most useful number in the whole object. One attribute is a
        category ("trucks"); the confidence of everything downstream turns on
        this being at least two.
        """
        return sum([
            bool(self.vehicle_classes), bool(self.directions), bool(self.cameras),
            self.since is not None or self.until is not None or bool(self.hours),
            self.min_speed is not None or self.max_speed is not None
            or self.speeding or self.crawling,
            self.min_confidence is not None, bool(self.plate_fragment),
            self.gateways_only,
        ])

    @property
    def readback(self) -> str:
        """The filter in a sentence an operator can check against what they meant."""
        if not self.understood:
            return "nothing in that phrase constrained the search"
        return "; ".join(self.understood)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["constraints"] = self.constraints
        d["readback"] = self.readback
        return d


# --- gazetteer ----------------------------------------------------------
_GAZETTEER: dict[int, list[dict]] = {}


def _norm(text: str) -> str:
    """Lowercase, punctuation to spaces, hyphens split.

    Hyphens become spaces on purpose: "two-wheeler", "CAM-13", "tuk-tuk" and
    "NH-19" all have to survive as ordinary token sequences, and every one of
    them is written both ways by real people.
    """
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return _norm(text).split()


def _gazetteer(net: CameraNetwork) -> list[dict]:
    """Every place phrase this network can be asked about.

    Built by taking each junction's name and road apart into contiguous word
    runs and resolving every run back against the whole network, so a phrase
    lands on all the junctions it plausibly names rather than only the one it
    came from: "salt lake" is the name of CAM-13 and the road of CAM-14, and an
    operator who says it means both.

    Keys are cached per network object - the network is a process-wide
    singleton, so this is built once.
    """
    cached = _GAZETTEER.get(id(net))
    if cached is not None:
        return cached

    cams = list(net.cameras.values())
    text_of = {c["id"]: (_norm(c["name"]), _norm(c.get("road") or ""),
                         _norm(c.get("sector") or "")) for c in cams}

    keys: dict[str, str] = {}       # phrase -> kind, longest/most specific wins
    for c in cams:
        keys.setdefault(_norm(c["id"]), "camera_id")
        for source, kind in ((c["name"], "camera"), (c.get("road") or "", "road")):
            words = _norm(source).split()
            for n in range(1, min(len(words), 4) + 1):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    if n == 1 and (phrase in GENERIC_PLACE_WORDS or phrase.isdigit()):
                        continue
                    keys.setdefault(phrase, kind)
        sector = _norm(c.get("sector") or "")
        if sector:
            keys.setdefault(sector, "sector")

    out: list[dict] = []
    for phrase, kind in keys.items():
        hits = []
        for c in cams:
            name, road, sector = text_of[c["id"]]
            if kind == "camera_id":
                if phrase == _norm(c["id"]):
                    hits.append(c["id"])
            elif phrase in name or phrase in road or phrase == sector:
                hits.append(c["id"])
        if not hits:
            continue
        # A phrase that lands on most of the city is not a location. Full road
        # and sector names are exempt: covering several junctions is what a
        # road is, and the operator said the word deliberately.
        if kind == "camera" and len(hits) > max(4, len(cams) // 3):
            continue
        if kind == "camera_id":
            spec = 1.0
        elif kind == "camera":
            spec = 1.0 if len(hits) == 1 else 0.85
        elif kind == "road":
            spec = 0.7
        else:
            spec = 0.5
        out.append({"phrase": phrase, "tokens": phrase.split(), "kind": kind,
                    "cameras": hits, "specificity": spec,
                    # A compass-only sector name is ambiguous with a heading and
                    # may only be read as a place when the sentence says so.
                    "needs_place_cue": kind == "sector" and phrase in DIRECTION_WORDS})
    out.sort(key=lambda k: (-len(k["tokens"]), -k["specificity"]))
    _GAZETTEER[id(net)] = out
    return out


def _camera_free_kmph(net: CameraNetwork, cam_id: str) -> float:
    """Free-flow speed of the roads meeting at a junction.

    The reference a spot speed is judged against. Mean over the incident links
    rather than the fastest, because a junction where a 60 km/h bypass meets a
    22 km/h lane is not a 60 km/h place for the vehicle that was turning.
    """
    speeds = [net.link_free_kmph(cam_id, n) for n in net.neighbours(cam_id)]
    speeds = [s for s in speeds if s > 0]
    return sum(speeds) / len(speeds) if speeds else 0.0


# --- time -------------------------------------------------------------
def _now(conn=None) -> float:
    """The data's own present, not the wall clock.

    Same anchor and same reason as core/trajectory.lookback: a seeded database
    stops advancing the moment seeding ends, so "yesterday evening" measured
    from wall clock can sit entirely after the last sighting and return nothing
    for every vehicle in the store. Unlike core/analytics this does not filter
    to camera-sourced rows - a manual enrolment is exactly the kind of evidence
    a descriptive search is looking for, and it should not be invisible to the
    clock that decides the window.
    """
    return db.latest_ts(conn=conn)


def _midnight(ts: float, days_back: int = 0) -> tuple[float, float]:
    """Local-midnight bounds of the day `days_back` before `ts`."""
    lt = time.localtime(ts - days_back * 86400.0)
    start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    return start, start + 86400.0


def _fmt(ts: float | None) -> str:
    return time.strftime("%a %d %b %H:%M", time.localtime(ts)) if ts else "-"


def _read_clock(tokens: list[str], i: int) -> tuple[float, int] | None:
    """Hour-of-day at tokens[i], and how many tokens it took.

    Accepts "6pm", "6 pm", "18:00", "6:30pm", "18". A bare number is only ever
    read as a clock time by the caller, which knows it followed "after" or
    "before" - on its own, "6" in a sentence is far more likely to be part of a
    plate or a count.
    """
    if i >= len(tokens):
        return None
    m = _CLOCK_RE.match(tokens[i])
    if not m:
        return None
    hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    used = 1
    if mer is None and i + 1 < len(tokens) and tokens[i + 1] in ("am", "pm"):
        mer, used = tokens[i + 1], 2
    if mer == "pm" and hour < 12:
        hour += 12
    elif mer == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour + minute / 60.0, used


def _clock_ts(anchor: float, hour_of_day: float, days_back: int = 0) -> float:
    start, _ = _midnight(anchor, days_back)
    return start + hour_of_day * 3600.0


# --- parser -----------------------------------------------------------
def parse(text: str, net: CameraNetwork | None = None, conn=None,
          anchor_ts: float | None = None) -> Query:
    """Turn a plain-English phrase into a filter, and say what was left over.

    Deliberately greedy in one direction only: a term is consumed by the most
    specific rule that can take it (a four-word junction name before a two-word
    road before a single word), and everything unconsumed is reported. A parser
    that guesses at leftovers produces filters nobody asked for; one that
    silently drops them produces confidence nobody earned.
    """
    net = net or network()
    q = Query(text=text, anchor_ts=anchor_ts if anchor_ts is not None else _now(conn))
    tokens = _tokens(text)
    if not tokens:
        q.ambiguities.append("empty query")
        return q
    used = [False] * len(tokens)

    def free(i: int, n: int = 1) -> bool:
        return i + n <= len(tokens) and not any(used[i:i + n])

    def take(i: int, n: int = 1) -> None:
        for k in range(i, min(i + n, len(tokens))):
            used[k] = True

    def cue_before(i: int, cues: frozenset) -> bool:
        """Is one of these cue words within two tokens to the left?

        Two, not one, so "heading due north" and "in the north east" both read
        the way they were meant.
        """
        return any(tokens[j] in cues for j in range(max(0, i - 2), i))

    # Gateways before places: "entering the city" is a phrase about the cordon,
    # and the place matcher would otherwise take the word "city" out of it.
    _parse_gateways(tokens, used, free, take, q, net)
    _parse_places(tokens, used, free, take, cue_before, q, net)
    _parse_classes(tokens, used, free, take, q)
    _parse_directions(tokens, used, free, take, cue_before, q)
    _parse_time(tokens, used, free, take, q)
    _parse_speed(tokens, used, free, take, q)
    _parse_confidence(tokens, used, free, take, q)
    _parse_plate_fragment(tokens, used, free, take, q)
    _parse_not_recorded(tokens, used, free, take, q)

    q.unrecognised = [t for i, t in enumerate(tokens)
                      if not used[i] and t not in STOPWORDS and not t.isdigit()]
    if q.unrecognised:
        q.ambiguities.append(
            f"not understood, and therefore not used: {', '.join(repr(t) for t in q.unrecognised)}")
    return q


def _parse_places(tokens, used, free, take, cue_before, q: Query,
                  net: CameraNetwork) -> None:
    """Longest gazetteer phrase first, so a junction name beats its own words."""
    for entry in _gazetteer(net):
        n = len(entry["tokens"])
        for i in range(len(tokens) - n + 1):
            if not free(i, n) or tokens[i:i + n] != entry["tokens"]:
                continue
            if entry["needs_place_cue"] and not cue_before(i, PLACE_CUES):
                continue
            cams = list(entry["cameras"])
            expanded = 0.0
            if cue_before(i, PROXIMITY_CUES):
                # "near Salt Lake" is a neighbourhood, not a junction. Reach out
                # by road-network geography rather than pretending the operator
                # meant only the cameras whose text happened to match.
                expanded = NEAR_KM
                seeds = [net.coords(c) for c in cams]
                for cam in net.cameras:
                    if cam in cams:
                        continue
                    lat, lon = net.coords(cam)
                    if any(haversine_km(lat, lon, sl, sn) <= NEAR_KM for sl, sn in seeds):
                        cams.append(cam)
            label = ", ".join(net.name(c) for c in cams[:4]) + ("..." if len(cams) > 4 else "")
            q.places.append(Place(term=entry["phrase"], kind=entry["kind"], label=label,
                                  cameras=cams, specificity=entry["specificity"],
                                  expanded_km=expanded))
            take(i, n)
            near = f", plus junctions within {expanded:.1f} km" if expanded else ""
            q.understood.append(
                f"location {entry['phrase']!r} ({entry['kind']}) -> "
                f"{len(cams)} junction(s){near}: {label}")
    seen: list[str] = []
    for p in q.places:
        for c in p.cameras:
            if c not in seen:
                seen.append(c)
    q.cameras = seen


def _parse_classes(tokens, used, free, take, q: Query) -> None:
    """Vehicle class, longest synonym phrase first."""
    pairs = sorted(((cls, syn) for cls, syns in VEHICLE_CLASS_SYNONYMS.items()
                    for syn in syns), key=lambda p: -len(p[1].split()))
    for cls, syn in pairs:
        parts = syn.split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if not free(i, n) or tokens[i:i + n] != parts:
                continue
            take(i, n)
            if cls not in q.vehicle_classes:
                q.vehicle_classes.append(cls)
                q.class_terms.append(syn)
                said = f" (from {syn!r})" if syn != cls else ""
                q.understood.append(f"vehicle class: {cls}{said}")


def _parse_directions(tokens, used, free, take, cue_before, q: Query) -> None:
    """Heading, expanded to the neighbouring compass points.

    A camera's recorded direction is the bearing of the road at that junction
    rounded to sixteen points, so a vehicle a witness would call "northbound"
    is stored as NNE as often as N. Filtering on N alone silently drops most of
    the vehicles the operator meant; the neighbours are accepted and then
    ranked below the exact point, which is where the difference belongs.
    """
    words = sorted(DIRECTION_WORDS, key=lambda w: -len(w.split()))
    for word in words:
        parts = word.split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if not free(i, n) or tokens[i:i + n] != parts:
                continue
            bare = not cue_before(i, DIRECTION_CUES)
            suffixed = False
            if bare:
                # "northbound" and "north bound" are unambiguous even with no cue.
                if i + n < len(tokens) and tokens[i + n] == "bound" and free(i + n):
                    take(i + n)
                    suffixed = True
                elif cue_before(i, PLACE_CUES):
                    continue        # "in the north" - a part of the city, not a heading
            _add_direction(q, DIRECTION_WORDS[word], word)
            take(i, n)
            if bare and not suffixed:
                q.ambiguities.append(
                    f"{word!r} was read as direction of travel; say 'in the {word}' "
                    f"to filter by part of the city instead")

    for i, tok in enumerate(tokens):
        if not free(i):
            continue
        if tok.endswith("bound") and tok[:-5] in DIRECTION_WORDS:
            _add_direction(q, DIRECTION_WORDS[tok[:-5]], tok)
            take(i)
        elif tok.upper() in COMPASS_16 and len(tok) >= 2:
            _add_direction(q, tok.upper(), tok)
            take(i)


def _add_direction(q: Query, code: str, term: str) -> None:
    if code in q.primary_directions:
        return
    idx = COMPASS_16.index(code)
    accepted = [code, COMPASS_16[(idx - 1) % 16], COMPASS_16[(idx + 1) % 16]]
    q.primary_directions.append(code)
    q.direction_terms.append(term)
    for c in accepted:
        if c not in q.directions:
            q.directions.append(c)
    q.understood.append(
        f"heading {term} - accepting {', '.join(accepted)} "
        f"(a junction's bearing is rounded to 16 points, so the neighbours "
        f"of {code} are the same road)")


def _set_hours(q: Query, hours, source: str) -> None:
    """Add an hour-of-day constraint.

    A band widens ("morning or evening"), a clock bound narrows ("mornings,
    after 9"). Two bands are alternatives; a band and a clock bound are two
    statements about the same window, and a filter that widened on the second
    one would return hours the operator had just excluded.
    """
    hours = sorted(set(int(h) for h in hours))
    if q.hours and source == "clock":
        q.hours = sorted(set(q.hours) & set(hours))
    else:
        q.hours = sorted(set(q.hours) | set(hours))


def _parse_time(tokens, used, free, take, q: Query) -> None:
    """Day words, hour-of-day bands, clock bounds and rolling windows.

    A day and a band compose: "yesterday evening" is yesterday's midnight
    bounds *and* the evening hours, applied as two conditions rather than one
    range, so a band that crosses midnight needs no special case. "last night"
    and "tonight" are the exceptions and get explicit windows, because nobody
    means "yesterday, hours 21-04" by them - they mean one continuous night.

    A clock bound with no day named is an hour of day, not an instant, and this
    is the correction that matters most here. Read as an instant against the
    store's clock, "buses after 6pm" asked at half past midnight resolves to
    18:00 *today*, which is in the future: the query returns nothing, the
    operator reads that as "no buses" and it is really "no time has passed
    yet". As an hour of day it means 18:00 to 23:59 on every day held, which is
    both what they meant and never empty for the wrong reason. Naming a day
    puts it back to being an instant on that day, which is then well defined.
    """
    anchor = q.anchor_ts
    day_back: int | None = None
    touched: list[str] = []          # hour terms seen, to tell "none" from "empty"

    def set_hours(hours, source: str) -> None:
        touched.append(source)
        _set_hours(q, hours, source)

    # Day first, and in its own pass, so "after 6pm yesterday" and "yesterday
    # after 6pm" resolve identically rather than depending on word order.
    for i, tok in enumerate(tokens):
        if not free(i):
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok == "last" and nxt == "night" and free(i, 2):
            start, _ = _midnight(anchor, 1)
            q.since, q.until = start + 21 * 3600.0, start + 29 * 3600.0
            q.time_terms.append("last night")
            q.understood.append(f"time: last night, {_fmt(q.since)} to {_fmt(q.until)}")
            take(i, 2)
        elif tok == "tonight" and free(i):
            start, _ = _midnight(anchor, 0)
            q.since, q.until = start + 21 * 3600.0, anchor
            q.time_terms.append("tonight")
            q.understood.append(f"time: tonight, {_fmt(q.since)} to {_fmt(q.until)}")
            take(i)
        elif tok in ("yesterday", "today") and free(i):
            day_back = 1 if tok == "yesterday" else 0
            q.since, q.until = _midnight(anchor, day_back)
            q.time_terms.append(tok)
            q.understood.append(f"day: {tok}, {_fmt(q.since)} to {_fmt(q.until)}")
            take(i)

    # Then the named bands, which are always hours of day.
    for band, hours in sorted(TIME_BANDS.items(), key=lambda kv: -len(kv[0].split())):
        parts = band.split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if not free(i, n) or tokens[i:i + n] != parts:
                continue
            take(i, n)
            set_hours(hours, "band")
            q.time_terms.append(band)
            span = "21:00-04:59" if band == "night" else \
                f"{min(hours):02d}:00-{max(hours):02d}:59"
            undated = (" - on every day held, because no day was given"
                       if q.since is None and q.until is None else "")
            q.understood.append(f"time of day: {band} ({span}){undated}")

    # Then rolling windows and clock bounds.
    for i, tok in enumerate(tokens):
        if not free(i):
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""

        if tok in ("last", "past", "previous"):
            amount, unit_at = 1.0, i + 1
            m = _CLOCK_RE.match(nxt) if nxt else None
            if m and m.group(2) is None and m.group(3) is None:
                amount, unit_at = float(m.group(1)), i + 2
            unit = tokens[unit_at] if unit_at < len(tokens) else ""
            secs = {"minute": 60.0, "minutes": 60.0, "min": 60.0, "mins": 60.0,
                    "hour": 3600.0, "hours": 3600.0, "hr": 3600.0, "hrs": 3600.0,
                    "day": 86400.0, "days": 86400.0}.get(unit)
            if secs:
                q.since, q.until = anchor - amount * secs, None
                q.time_terms.append(f"last {amount:g} {unit}")
                q.understood.append(
                    f"time: the last {amount:g} {unit} of traffic, since {_fmt(q.since)} "
                    f"(measured from the newest sighting held, not the wall clock)")
                take(i, unit_at - i + 1)
                continue

        if tok in ("after", "since", "from"):
            got = _read_clock(tokens, i + 1)
            if got:
                hour, n = got
                take(i, n + 1)
                q.time_terms.append(f"after {hour:g}h")
                if day_back is not None:
                    q.since = _clock_ts(anchor, hour, day_back)
                    q.understood.append(f"time: after {_fmt(q.since)}")
                else:
                    # Whole hours, rounded outwards on both bounds: a search
                    # must never drop a match to a rounding decision the
                    # operator did not make. "after 6:30pm" therefore includes
                    # 18:00-18:29, which the readback says.
                    set_hours(range(int(hour), 24), "clock")
                    q.understood.append(
                        f"time of day: from {int(hour):02d}:00 onwards, on every day "
                        f"held - name a day to make it an instant")
                continue

        if tok in ("before", "until", "till", "upto"):
            got = _read_clock(tokens, i + 1)
            if got:
                hour, n = got
                take(i, n + 1)
                q.time_terms.append(f"before {hour:g}h")
                if day_back is not None:
                    q.until = _clock_ts(anchor, hour, day_back)
                    q.understood.append(f"time: before {_fmt(q.until)}")
                else:
                    last = max(0, math.ceil(hour) - 1)
                    set_hours(range(0, last + 1), "clock")
                    q.understood.append(
                        f"time of day: up to {last:02d}:59, on every day held - "
                        f"name a day to make it an instant")
                continue

        if tok == "between":
            a = _read_clock(tokens, i + 1)
            if a and i + 1 + a[1] < len(tokens) and tokens[i + 1 + a[1]] == "and":
                b = _read_clock(tokens, i + 2 + a[1])
                if b:
                    take(i, 2 + a[1] + b[1])
                    q.time_terms.append(f"between {a[0]:g}h and {b[0]:g}h")
                    if day_back is not None:
                        q.since = _clock_ts(anchor, a[0], day_back)
                        q.until = _clock_ts(anchor, b[0], day_back)
                        q.understood.append(f"time: {_fmt(q.since)} to {_fmt(q.until)}")
                    else:
                        # Wraps through midnight when the second hour is the
                        # smaller one - "between 10pm and 4am" is one night.
                        # The upper bound is exclusive: 2pm to 5pm is hours 14,
                        # 15 and 16, because 17:xx is after 5pm.
                        lo, end = int(a[0]), math.ceil(b[0])
                        span = (list(range(lo, end)) if lo < end
                                else list(range(lo, 24)) + list(range(0, end)))
                        set_hours(span, "clock")
                        last = span[-1] if span else lo
                        q.understood.append(
                            f"time of day: {lo:02d}:00 to {last:02d}:59, on every day "
                            f"held - name a day to make it one afternoon")
                    continue

    if touched and not q.hours:
        # An intersection can empty the set - "mornings after 6pm" is a real
        # thing to type and has no answer. An empty hour list and no hour list
        # produce the same SQL, so this has to be flagged, or the query that
        # can match nothing would quietly match everything.
        q.impossible = True
        q.ambiguities.append(
            f"{' and '.join(repr(t) for t in q.time_terms)} have no hours in common, "
            f"so nothing can match; check them against each other")


def _parse_speed(tokens, used, free, take, q: Query) -> None:
    """Explicit thresholds, and the two words people use instead of a number."""
    for i, tok in enumerate(tokens):
        if not free(i):
            continue
        if tok in ("speeding", "overspeeding", "speed", "fast", "racing", "rash"):
            if tok == "speed":
                continue                # "speed" alone is a noun, not a filter
            q.speeding = True
            take(i)
            q.understood.append(
                f"speed: above {SPEEDING_FACTOR:g}x the free-flow speed of the roads "
                f"meeting that junction - the same threshold the speeding alert rule "
                f"uses, measured on the spot speed the camera reported")
            continue
        if tok in ("crawling", "stopped", "stationary", "slow", "slowly", "jammed"):
            q.crawling = True
            take(i)
            q.understood.append(
                f"speed: below {SLOW_FACTOR:g}x the free-flow speed of the roads "
                f"meeting that junction")
            continue
        if tok in ("over", "above", "faster", "exceeding", "more") and free(i):
            got = _numeric(tokens, i + 1)
            if got:
                q.min_speed, n = got
                take(i, n + 1)
                q.understood.append(f"speed: over {q.min_speed:g} km/h")
                continue
        if tok in ("under", "below", "slower", "less") and free(i):
            got = _numeric(tokens, i + 1)
            if got:
                q.max_speed, n = got
                take(i, n + 1)
                q.understood.append(f"speed: under {q.max_speed:g} km/h")


def _numeric(tokens: list[str], i: int) -> tuple[float, int] | None:
    """A speed reading at tokens[i], skipping the filler words around it.

    Handles "over 60", "faster than 60", "above 60 kmph", "more than 60 km/h" -
    which _norm has already flattened to "km h", hence the two-token unit.
    """
    n = 0
    while i < len(tokens) and tokens[i] in ("than", "the", "a"):
        i, n = i + 1, n + 1
    if i >= len(tokens) or not re.fullmatch(r"\d{1,3}", tokens[i]):
        return None
    value, n = float(tokens[i]), n + 1
    for unit in (("kmph",), ("kph",), ("km", "h"), ("km", "hr"), ("kmh",)):
        if tokens[i + 1:i + 1 + len(unit)] == list(unit):
            n += len(unit)
            break
    return value, n


def _parse_confidence(tokens, used, free, take, q: Query) -> None:
    """Only reads the engine stood behind."""
    for i, tok in enumerate(tokens):
        if not free(i):
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in ("confident", "reliable", "strong", "certain") or \
                (tok == "high" and nxt in ("confidence", "quality")):
            take(i, 2 if nxt in ("confidence", "quality", "reads", "only") else 1)
            q.min_confidence = WEAK_READ
            q.understood.append(
                f"reads only: plate read confidence at or above {WEAK_READ:g}")
            return


def _parse_plate_fragment(tokens, used, free, take, q: Query) -> None:
    """The part of the registration a witness did get.

    Only from an explicit cue, or from a token that is shaped like an Indian
    registration. A bare four-character token is far more often a word than a
    plate fragment, and inventing a plate filter is the one mistake here that
    would return nothing and look like an answer.
    """
    anchors = {"starting": "prefix", "starts": "prefix", "beginning": "prefix",
               "begins": "prefix", "ending": "suffix", "ends": "suffix",
               "containing": "contains", "contains": "contains", "like": "contains"}
    for i, tok in enumerate(tokens):
        if not free(i):
            continue
        if tok in anchors and i + 1 < len(tokens) and free(i + 1):
            frag = tokens[i + 1].upper()
            if len(frag) >= 2 and frag.isalnum():
                q.plate_fragment, q.plate_anchor = frag, anchors[tok]
                take(i, 2)
                q.understood.append(f"plate {q.plate_anchor} {frag!r}")
                return
    for i, tok in enumerate(tokens):
        if free(i) and len(tok) >= 4 and _PLATE_RE.match(tok):
            q.plate_fragment, q.plate_anchor = tok.upper(), "prefix"
            take(i)
            q.understood.append(
                f"plate starting {q.plate_fragment!r} - read as a partial "
                f"registration because of its shape")
            return


def _parse_gateways(tokens, used, free, take, q: Query, net: CameraNetwork) -> None:
    """Entering or leaving the city, which is a property of the junction."""
    phrases = ("entering the city", "leaving the city", "city entry", "city exit",
               "gateway", "gateways", "entry point", "border")
    for phrase in sorted(phrases, key=lambda p: -len(p.split())):
        parts = phrase.split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if not free(i, n) or tokens[i:i + n] != parts:
                continue
            take(i, n)
            q.gateways_only = True
            gws = net.gateways
            q.understood.append(
                f"at a city gateway - {len(gws)} junction(s): "
                + ", ".join(net.name(g) for g in gws))
            return


def _parse_not_recorded(tokens, used, free, take, q: Query) -> None:
    """Terms this platform understands the meaning of and cannot filter on.

    Runs last, so a colour word that was part of a place name or of "yellow
    taxi" has already been consumed by the rule that owns it and is not
    reported here as an ignored attribute it never was.
    """
    for attribute, spec in NOT_RECORDED.items():
        for term in sorted(spec["terms"], key=lambda t: -len(t.split())):
            parts = term.split()
            n = len(parts)
            for i in range(len(tokens) - n + 1):
                if not free(i, n) or tokens[i:i + n] != parts:
                    continue
                take(i, n)
                if any(g.term == term for g in q.ignored):
                    continue
                q.ignored.append(Ignored(
                    term=term, attribute=attribute,
                    note=f"{attribute} {term!r} was NOT used to filter: {spec['why']}"))


# --- search -----------------------------------------------------------
def _where(q: Query) -> tuple[str, list]:
    """The part of the filter SQLite can do."""
    clauses, params = ["1 = 1"], []
    if q.vehicle_classes:
        clauses.append("vehicle_class IN (%s)" % ",".join("?" * len(q.vehicle_classes)))
        params += q.vehicle_classes
    if q.cameras:
        clauses.append("camera_id IN (%s)" % ",".join("?" * len(q.cameras)))
        params += q.cameras
    if q.directions:
        clauses.append("direction IN (%s)" % ",".join("?" * len(q.directions)))
        params += q.directions
    if q.since is not None:
        clauses.append("ts >= ?")
        params.append(q.since)
    if q.until is not None:
        clauses.append("ts <= ?")
        params.append(q.until)
    if q.hours:
        clauses.append("CAST(strftime('%%H', ts, 'unixepoch', 'localtime') AS INTEGER)"
                       " IN (%s)" % ",".join("?" * len(q.hours)))
        params += q.hours
    if q.min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(q.min_confidence)
    if q.min_speed is not None:
        clauses.append("speed_kmph >= ?")
        params.append(q.min_speed)
    if q.max_speed is not None:
        clauses.append("speed_kmph <= ?")
        params.append(q.max_speed)
    if q.plate_fragment:
        pattern = {"prefix": q.plate_fragment + "%", "suffix": "%" + q.plate_fragment,
                   "contains": "%" + q.plate_fragment + "%"}[q.plate_anchor]
        clauses.append("plate LIKE ?")
        params.append(pattern)
    return " AND ".join(clauses), params


def _score(row: dict, q: Query, net: CameraNetwork, spec_by_cam: dict[str, float],
           now: float) -> tuple[float, dict]:
    """How well one sighting fits, for ordering only.

    Explicitly not a probability that this is the vehicle. Every row here
    already passed the filter, so the ordering is about which of several equally
    legal matches an operator should look at first, and it is built from four
    things that genuinely differ between them: whether the recorded heading is
    the exact compass point asked for or one of its neighbours, how narrowly the
    location phrase pinned that junction, how much the reader stood behind the
    plate it read, and how recently it happened. A search for a vehicle is
    nearly always a search for where it is now, which is why recency carries as
    much weight as the read.
    """
    direction_fit = 1.0
    if q.primary_directions:
        direction_fit = 1.0 if row.get("direction") in q.primary_directions else 0.75
    place_fit = spec_by_cam.get(row["camera_id"], 1.0) if q.cameras else 1.0
    read = float(row.get("confidence") or 0.0)
    age_h = max(0.0, (now - float(row["ts"]))) / 3600.0
    recency = 0.5 ** (age_h / RECENCY_HALFLIFE_H) if RECENCY_HALFLIFE_H > 0 else 1.0
    score = 0.30 * direction_fit + 0.20 * place_fit + 0.25 * read + 0.25 * recency
    return score, {"direction_fit": round(direction_fit, 2),
                   "place_fit": round(place_fit, 2),
                   "read_confidence": round(read, 3),
                   "recency": round(recency, 3)}


def search(text: str | Query, limit: int = DEFAULT_LIMIT,
           net: CameraNetwork | None = None, conn=None,
           since: float | None = None, until: float | None = None,
           allow_unconstrained: bool = False) -> dict:
    """Search the sighting log by description rather than by plate.

    Returns the parsed filter, the matching sightings, the distinct vehicles
    behind them, and an account of how much the description actually narrowed
    the field. Callers must render the `ignored` and `caveats` lists alongside
    the results: they are what stops a wide net being read as a narrow one.

    `since`/`until` narrow whatever the phrase itself said - they are the
    caller's bounds, not an override, so a dashboard that scopes the whole page
    to a shift cannot be widened by what someone types into the box.
    """
    net = net or network()
    conn = conn or db.connect()
    q = text if isinstance(text, Query) else parse(text, net=net, conn=conn)
    if since is not None:
        q.since = since if q.since is None else max(q.since, since)
    if until is not None:
        q.until = until if q.until is None else min(q.until, until)

    now = q.anchor_ts or _now(conn)
    result = {
        "query": q.text,
        "parsed": q.as_dict(),
        "readback": q.readback,
        "ignored": [asdict(g) for g in q.ignored],
        "unrecognised": q.unrecognised,
        "ambiguities": q.ambiguities,
        "anchor_ts": now,
        "matched": 0, "scanned": 0, "share": 0.0,
        "sightings": [], "candidates": [], "truncated": False,
        "confidence": {}, "caveats": [],
    }

    if q.constraints == 0 and not allow_unconstrained:
        result["refused"] = True
        result["confidence"] = {"discriminating": False, "actionable": False,
                                "constraints": 0,
                                "reason": "nothing in that phrase constrained the "
                                          "search, so running it would return the "
                                          "whole store rather than an answer"}
        result["caveats"].append(
            "No filter was understood. " + _suggestion(q))
        return result

    if q.impossible:
        # Refused rather than run. The filter contradicts itself, and an empty
        # result from a query that was never askable reads exactly like an
        # empty result from a query that was.
        result["refused"] = True
        result["confidence"] = {"discriminating": False, "actionable": False,
                                "constraints": q.constraints,
                                "reason": "the phrase contradicts itself, so it was "
                                          "not run - an empty answer here would look "
                                          "like 'no such vehicle'"}
        result["caveats"] = [a + "." for a in q.ambiguities]
        return result

    where, params = _where(q)
    rows = db.rows(f"SELECT * FROM sightings WHERE {where} ORDER BY ts DESC LIMIT ?",
                   tuple(params + [MAX_SCAN + 1]), conn=conn)
    if len(rows) > MAX_SCAN:
        rows = rows[:MAX_SCAN]
        result["truncated"] = True

    # The speed comparison is per junction, so it cannot be a constant in the
    # WHERE clause - a 40 km/h spot speed is speeding on a 22 km/h lane and
    # ordinary on the bypass.
    if q.speeding or q.crawling:
        kept = []
        for r in rows:
            ref = _camera_free_kmph(net, r["camera_id"])
            sp = r.get("speed_kmph")
            if ref <= 0 or sp is None:
                continue
            if q.speeding and sp > SPEEDING_FACTOR * ref:
                kept.append(r)
            elif q.crawling and sp < SLOW_FACTOR * ref:
                kept.append(r)
        rows = kept

    spec_by_cam: dict[str, float] = {}
    for p in q.places:
        for c in p.cameras:
            spec_by_cam[c] = max(spec_by_cam.get(c, 0.0), p.specificity)
    if q.gateways_only:
        gws = set(net.gateways)
        rows = [r for r in rows if r["camera_id"] in gws]

    # The base rate. Without it "37 matches" is unreadable: 37 out of 90 is a
    # description of the traffic, 37 out of 19,000 is a description of a vehicle.
    time_where, time_params = [], []
    if q.since is not None:
        time_where.append("ts >= ?")
        time_params.append(q.since)
    if q.until is not None:
        time_where.append("ts <= ?")
        time_params.append(q.until)
    clause = " AND ".join(time_where) if time_where else "1 = 1"
    scanned = db.rows(f"SELECT COUNT(*) n FROM sightings WHERE {clause}",
                      tuple(time_params), conn=conn)[0]["n"]

    scored = []
    for r in rows:
        s, parts = _score(r, q, net, spec_by_cam, now)
        cam = net.cameras.get(r["camera_id"], {})
        scored.append({**r,
                       "camera_name": net.name(r["camera_id"]),
                       "sector": cam.get("sector"), "road": cam.get("road"),
                       "lat": cam.get("lat"), "lon": cam.get("lon"),
                       "age_minutes": round((now - float(r["ts"])) / 60.0, 1),
                       "match_score": round(s, 4), "match_parts": parts})
    scored.sort(key=lambda r: -r["match_score"])

    result["matched"] = len(scored)
    result["scanned"] = scanned
    result["share"] = round(len(scored) / scanned, 5) if scanned else 0.0
    result["sightings"] = scored[:limit]
    result["candidates"] = _candidates(scored, net, now, conn)
    result["confidence"] = _confidence(q, scored, result["candidates"], scanned)
    result["caveats"] = _caveats(q, scored, result["candidates"], scanned, result)
    return result


def _candidates(scored: list[dict], net: CameraNetwork, now: float,
                conn) -> list[dict]:
    """The distinct vehicles behind the matching sightings, best lead first.

    This, not the sighting list, is what the operator is actually after - they
    are looking for a vehicle, and a vehicle seen four times fitting the
    description is a far better lead than four vehicles seen once each. That is
    the only evidence a structured search can offer towards identity, so it is
    the thing the ranking is built on: the mean fit of a plate's matching
    sightings, raised by how many of them there were. `corroborated` marks the
    ones that fitted more than once, and it is deliberately a separate flag
    rather than being buried in the score, because one strong match and three
    weak ones are different situations that can end up with the same number.
    """
    by_plate: dict[str, list[dict]] = {}
    for r in scored:
        by_plate.setdefault(r["plate"], []).append(r)

    out = []
    for plate, rs in by_plate.items():
        scores = [r["match_score"] for r in rs]
        mean = sum(scores) / len(scores)
        newest = max(rs, key=lambda r: r["ts"])
        cams = []
        for r in rs:
            if r["camera_id"] not in cams:
                cams.append(r["camera_id"])
        out.append({
            "plate": plate,
            "matches": len(rs),
            "corroborated": len(rs) >= 2,
            # log, not linear: the tenth matching sighting says much less than
            # the second, and a vehicle that simply uses a monitored junction
            # every day should not outrank a good fit on volume alone.
            "score": round(mean * (1.0 + 0.35 * math.log(len(rs))), 4),
            "best_score": round(max(scores), 4),
            "mean_confidence": round(sum(float(r["confidence"] or 0) for r in rs) / len(rs), 3),
            "first_ts": min(r["ts"] for r in rs),
            "last_ts": newest["ts"],
            "last_camera": newest["camera_id"],
            "last_camera_name": net.name(newest["camera_id"]),
            "last_seen_minutes_ago": round((now - float(newest["ts"])) / 60.0, 1),
            "cameras": cams,
            "camera_names": [net.name(c) for c in cams],
            "classes": sorted({r.get("vehicle_class") for r in rs if r.get("vehicle_class")}),
            "directions": sorted({r.get("direction") for r in rs if r.get("direction")}),
            # An operator meeting a plate by description is at exactly the
            # moment core/trajectory.search worries about: whatever they do next
            # assumes this registration means one vehicle, and a watchlist hit
            # must not wait for them to look it up separately.
            "watched": bool(db.is_watched(plate, conn=conn)),
        })
    out.sort(key=lambda c: -c["score"])
    return out


def _confidence(q: Query, scored: list[dict], candidates: list[dict],
                scanned: int) -> dict:
    """Whether this result is a lead or just a slice of the traffic."""
    share = (len(scored) / scanned) if scanned else 0.0
    discriminating = q.constraints >= MIN_CONSTRAINTS and 0 < share <= BROAD_SHARE
    actionable = 0 < len(candidates) <= MAX_ACTIONABLE
    if not scored:
        reason = ("nothing matched. The filter is sound - it simply had no rows, "
                  "which is a real answer, not a failure")
    elif not discriminating and q.constraints < MIN_CONSTRAINTS:
        reason = (f"only {q.constraints} attribute was constrained, so this is a "
                  f"category, not a description")
    elif not discriminating:
        reason = (f"this description fits {share:.0%} of everything seen in the same "
                  f"window - it does not narrow the field enough to point at a vehicle")
    elif not actionable:
        reason = (f"{len(candidates)} distinct registrations fit; the description "
                  f"narrows the traffic but not to a workable shortlist")
    else:
        reason = (f"{len(candidates)} registration(s) out of {scanned} sightings in "
                  f"the window fit {q.constraints} independent attributes")
    return {"discriminating": discriminating, "actionable": actionable,
            "constraints": q.constraints, "share": round(share, 5),
            "candidates": len(candidates),
            "corroborated": sum(1 for c in candidates if c["corroborated"]),
            "reason": reason}


def _caveats(q: Query, scored: list[dict], candidates: list[dict], scanned: int,
             result: dict) -> list[str]:
    """Everything the operator has to know before believing this list.

    Assembled as prose rather than flags because it is meant to be read out.
    The colour caveat is first on purpose: it is the one that changes what the
    result means.
    """
    out: list[str] = []
    for g in q.ignored:
        out.append(g.note + ".")
    if q.ambiguities:
        out.extend(a + "." for a in q.ambiguities)
    if q.constraints < MIN_CONSTRAINTS:
        out.append(f"Only {q.constraints} attribute was understood as a filter. "
                   + _suggestion(q))
    if q.hours and q.since is None and q.until is None:
        terms = " and ".join(repr(t) for t in q.time_terms) or "the time given"
        out.append(f"{terms} was applied as an hour of day across every day held, not "
                   f"one particular occasion - so a match may be from any day in the "
                   f"store. Add 'yesterday' or 'today' to pin it to a date.")
    if q.since is None and q.until is None and not q.hours:
        out.append("No time was given, so this searched the whole retained history.")
    if scored:
        weak = sum(1 for r in scored if float(r.get("confidence") or 0) < WEAK_READ)
        if weak:
            out.append(
                f"{weak} of {len(scored)} matching reads are below {WEAK_READ:g} "
                f"confidence. The vehicle was there; which registration it carried is "
                f"less certain on those rows.")
    if result.get("truncated"):
        out.append(f"More than {MAX_SCAN} sightings matched; only the newest "
                   f"{MAX_SCAN} were ranked, so the counts are a floor.")
    if len(candidates) > MAX_ACTIONABLE:
        out.append(f"{len(candidates)} vehicles fit this description. It is a filter, "
                   f"not an identification - narrow it before acting on any one of them.")
    if candidates and not any(c["corroborated"] for c in candidates):
        out.append("No vehicle fits the description more than once, so nothing here "
                   "is corroborated by a second sighting.")
    return out


def _suggestion(q: Query) -> str:
    """What the operator could add, phrased as the vocabulary that exists."""
    missing = []
    if not q.vehicle_classes:
        missing.append("a vehicle type (car, two-wheeler, auto, truck, taxi, bus)")
    if not q.cameras:
        missing.append("a place (a junction, a road, or a sector of the city)")
    if q.since is None and q.until is None and not q.hours:
        missing.append("a time (yesterday, this morning, after 6pm, last 2 hours)")
    if not q.directions:
        missing.append("a direction of travel (northbound, heading south)")
    return "Add " + ", or ".join(missing[:3]) + "." if missing else ""


# --- self-description ---------------------------------------------------
def vocabulary(net: CameraNetwork | None = None) -> dict:
    """Everything this parser understands, enumerated.

    The compensating advantage of a rule-based parser, and the reason one was
    chosen over a model: it can be printed. An operator who has this in front
    of them knows before they type whether the platform can answer the question
    they are about to ask, and a term that is missing from this list is missing
    from the parser, with no third possibility.
    """
    net = net or network()
    return {
        "vehicle_classes": {cls: list(syn) for cls, syn in VEHICLE_CLASS_SYNONYMS.items()},
        "directions": {"terms": sorted(DIRECTION_WORDS), "codes": COMPASS_16,
                       "note": "a compass term also accepts its two neighbouring "
                               "points, because a junction's bearing is rounded to 16"},
        "time": {"bands": {b: list(h) for b, h in TIME_BANDS.items()},
                 "days": ["today", "yesterday", "tonight", "last night"],
                 "clock": ["after 6pm", "before 9am", "between 2pm and 5pm"],
                 "rolling": ["last 2 hours", "past 30 minutes", "last 3 days"]},
        "speed": ["speeding", "crawling", "over 60", "under 20",
                  f"speeding means above {SPEEDING_FACTOR:g}x the free-flow speed of "
                  f"the roads at that junction"],
        "confidence": ["confident reads", "reliable reads", "high confidence"],
        "plate": ["plate starting WB08", "plate ending 4321", "WB08AB1234"],
        "places": {"sectors": sorted({c.get("sector") for c in net.cameras.values() if c.get("sector")}),
                   "roads": sorted({c.get("road") for c in net.cameras.values() if c.get("road")}),
                   "junctions": [c["name"] for c in net.cameras.values()],
                   "note": f"a place phrase matches junction names, roads and sectors "
                           f"by substring; 'near <place>' also reaches "
                           f"{NEAR_KM:g} km around it"},
        "gateways": ["entering the city", "leaving the city", "gateway"],
        "not_recorded": {a: {"terms": list(s["terms"]), "why": s["why"]}
                         for a, s in NOT_RECORDED.items()},
    }


def explain(text: str, net: CameraNetwork | None = None, conn=None) -> dict:
    """Parse a phrase and report the filter without running it.

    Exists so a search box can show the operator what it is about to do before
    it does it. Same parse, no database scan.
    """
    q = parse(text, net=net, conn=conn)
    return {"query": text, "readback": q.readback, "constraints": q.constraints,
            "understood": q.understood, "ignored": [asdict(g) for g in q.ignored],
            "unrecognised": q.unrecognised, "ambiguities": q.ambiguities,
            "filter": q.as_dict()}
