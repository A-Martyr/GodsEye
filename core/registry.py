"""A mock vehicle-registry lookup, VAHAN-shaped, for the engine's own use.

Synthetic. Not VAHAN.
----------------------
VAHAN (vahan.parivahan.gov.in) is the Government of India's national vehicle
registration database. Public programmatic access to it is not open - the
citizen-facing lookup is a web form behind a CAPTCHA, and bulk/API access runs
through agreements with the Ministry of Road Transport and Highways. Nothing
here talks to VAHAN, imitates it, or asserts access to it. Every record
`lookup()` returns is invented, deterministically, from the plate string
itself (and, for one field, from this platform's own sighting history - see
below). `RegistryRecord.synthetic` is `True` on every record for exactly this
reason, and it is carried into the API response rather than only living in a
comment.

Why this exists instead of just `vahan_db_implementation/`
------------------------------------------------------------
That folder was an unreviewed, unintegrated sketch: a standalone SQLite
schema incompatible with `core/db.py`, and a FastAPI router calling a
recogniser module (`anpr_engine_v2`) that does not exist anywhere in this
repo. It has been deleted. This module is the thing it was reaching for,
built to actually plug into `core/clones.py` and `core/repair.py`.

Why there are TWO non-integrated mock registries in this repository
----------------------------------------------------------------------
The top-level `registry/` package (a "Security Console": stolen/blacklisted-
vehicle checks, authority referral) is a *separate*, already-shipped mock
with a tested isolation invariant: nothing in `anpr/`, `core/`, `api/`,
`sim/` - "the engine" - imports it, so a control-room feature can never slow
or break the path from a camera to a stored sighting (see
`registry/selftest.py`). This module is the opposite kind of thing: an
engine-internal signal that `core/clones.py` and `core/repair.py` are meant
to import directly. Merging the two would either force the engine to depend
on an optional demo package, or force the Security Console's isolation
invariant to be broken. Keeping them separate is deliberate, not
duplication-by-accident. If a future consolidation makes sense, it has to
preserve that isolation invariant.

What is actually used as evidence, and what is not
------------------------------------------------------
Only `vehicle_class` is used as grounded cross-check evidence in
`core/clones.py` and `core/repair.py`. That is the *one* vehicle attribute
this platform's `sightings` table carries at all - and even then, only for
simulated captures (`sim/city.py` writes the simulator's own ground truth
into every sighting); a real photograph read through `core/enrol.py` stores
`vehicle_class=None` ("not ours to set" - there is no vision step that
infers it). `core/vehicle_search.py`, `core/convoy.py` and `core/geofence.py`
already treat this same field as "what the camera recorded", so using it
here is consistent with existing precedent, not a new claim.

`make`, `model`, `colour`, `fuel` and the validity dates are generated and
returned for API completeness and for future work (Goal B points 3 and 4 in
`CHATGPT_BRIEF.md` - search by description, partial-plate resolution) but are
**not** wired into any detection logic today. The engine has no camera-side
signal to compare them against, and pretending otherwise would be exactly the
kind of unlabelled inference this codebase's other modules go out of their
way to avoid (see `core/vehicle_search.py`'s `NOT_RECORDED` terms, or
`core/geofence.py`'s refusal to assert "still inside").

Grounding `vehicle_class`, so a mismatch means something
-------------------------------------------------------------
A registry whose `vehicle_class` were generated from a plate-seeded RNG with
no reference to what the cameras actually saw would disagree with the
sightings at random, and a "class mismatch" cross-check against random noise
is not a signal. `registry/seed.py::observed_classes()` and
`registry/store.py::build_record()` already establish and document this
exact lesson for the separate package described above ("make the registry
AGREE with reality most of the time - a registry that disagreed at random
would make the class-mismatch rule meaningless noise"). This module
replicates the *algorithm* - majority vehicle_class over the plate's own
sighting history, agree by default, disagree at a small configured rate -
rather than importing that package, to preserve the isolation invariant.

One consequence worth stating plainly: because the "true" class is read back
from the plate's own sightings, `lookup()`'s `vehicle_class` is deterministic
only for a fixed sightings snapshot, not permanently the way
`registry/store.py`'s records are (that package seeds once, offline, and
never re-derives from live data). A plate with no sighting history yet - the
only way `core/clones.py`/`core/repair.py` would ever see this, since both
only look up plates that already have sightings - falls back to a random
class from a small fixed vocabulary.

Partial-plate resolution - `resolve_partial()`
------------------------------------------------
`CHATGPT_BRIEF.md` §5 point 4: given a masked pattern like `WB01??9492`,
narrow the combinatorial space to a shortlist. This answers a different
question from `core/trajectory.py::search()`'s existing partial-plate
handling - that function only finds plates the platform has already *seen*
in `sightings`, by contiguous substring. `resolve_partial()` is for a plate a
witness partially caught in person, that may never have crossed a camera.

Two tiers, kept structurally separate on purpose:

- `seen` - candidates matching the mask that already have rows in
  `sightings`, found by a direct SQL `LIKE`, independent of plate grammar.
  Real evidence should never be grammar-filtered away - this platform's own
  test plates aren't grammar-valid either, and neither is an OCR misread
  sitting in the store, which is exactly the kind of thing this tier exists
  to surface.
- `unseen_registered` - candidates that are grammar-valid (built against
  `anpr.ocr`'s `PATTERNS`/`BH_PATTERNS`/`STATE_CODES`), were never sighted,
  and this mock registry reports `FOUND` for. Ranked by
  `anpr.ocr.plate_plausibility()` - a real, already-tested, NON-synthetic
  signal (actual Indian RTO district-code ceilings) - not by this module's
  own arbitrary `NOT_FOUND` rate alone. This is the weaker tier: no camera
  ever corroborated it, capped harder, and always labelled as such.

An optional `vehicle_class` filter never removes a candidate from either
tier - it only reorders and flags `matches_vehicle_class` - and it is
compared against a different thing per tier, reported as
`vehicle_class_source`: `"observed"` (the plate's own majority-recorded
class, for `seen`) versus `"registry"` (this mock's per-candidate invented
value, for `unseen_registered`, since nothing about that plate was ever
observed). The two must never be visually interchangeable.

`colour`/`make` filtering is deliberately not offered. No camera on this
network records either at all (`core/vehicle_search.py`'s `NOT_RECORDED`
dict already establishes this platform-wide) - letting a witness's "silver"
touch the `seen` tier would smuggle a fabricated attribute onto real camera
evidence, which is worse than anything the weaker tier could do alone. A
grounded colour/make signal would need a vision channel that reads the
vehicle crop (`CHATGPT_BRIEF.md` §6 point 5), which does not exist yet.
"""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass

import config
from anpr.ocr import (BH_PATTERNS, DIGITS, LETTERS, PATTERNS, STATE_CODES,
                      matches_plate_grammar, plate_plausibility)
from core import db

# Duplicated deliberately rather than imported from sim/city.py or
# registry/catalogue.py - core/ importing sim/ would run the architecture
# diagram backwards, and importing registry/ would break its isolation
# invariant. Keep in sync with sim/city.py's vehicle class vocabulary by eye.
_VEHICLE_CLASSES = ("car", "two_wheeler", "auto", "taxi", "bus", "truck")
_MAKES_BY_CLASS = {
    "car": (("Maruti Suzuki", "Swift"), ("Hyundai", "i20"), ("Tata", "Tiago")),
    "two_wheeler": (("Hero", "Splendor"), ("Honda", "Activa"), ("Bajaj", "Pulsar")),
    "auto": (("Bajaj", "RE"), ("Piaggio", "Ape")),
    "taxi": (("Maruti Suzuki", "Dzire"), ("Hyundai", "Xcent")),
    "bus": (("Tata", "Starbus"), ("Ashok Leyland", "Viking")),
    "truck": (("Tata", "407"), ("Ashok Leyland", "Dost")),
}
_COLOURS = ("white", "silver", "black", "grey", "red", "blue")
_FUELS = ("petrol", "diesel", "cng", "electric")
_STATUSES = ("active", "active", "active", "active", "active",
             "active", "active", "active", "active", "stolen", "blacklisted")

FOUND = "FOUND"
NOT_FOUND = "NOT_FOUND"
UNKNOWN = "UNKNOWN"


@dataclass
class RegistryRecord:
    plate: str
    registry_status: str = UNKNOWN     # FOUND | NOT_FOUND | UNKNOWN
    vehicle_class: str | None = None
    make: str | None = None
    model: str | None = None
    colour: str | None = None
    fuel: str | None = None
    status: str | None = None          # active | stolen | blacklisted, when FOUND
    registered_on: float | None = None
    insurance_to: float | None = None
    synthetic: bool = True

    @property
    def found(self) -> bool:
        return self.registry_status == FOUND

    def as_dict(self) -> dict:
        return {
            "plate": self.plate, "registry_status": self.registry_status,
            "vehicle_class": self.vehicle_class, "make": self.make,
            "model": self.model, "colour": self.colour, "fuel": self.fuel,
            "status": self.status, "registered_on": self.registered_on,
            "insurance_to": self.insurance_to, "synthetic": self.synthetic,
            "disclaimer": ("mock data, generated locally from the plate string - "
                           "not a VAHAN lookup, no government system was contacted"),
        }


_CACHE: dict[str, tuple[float, RegistryRecord]] = {}


def _rng(plate: str) -> random.Random:
    return random.Random(f"registry::{plate}")


def _majority_vehicle_class(plate: str, conn=None) -> str | None:
    """The class the cameras most often recorded for this plate, or None.

    Same shape as `registry/seed.py::observed_classes()`, scoped to one plate
    instead of the whole store, since this runs per-lookup rather than as an
    offline batch seed.
    """
    rows_ = db.rows(
        "SELECT vehicle_class, COUNT(*) n FROM sightings"
        " WHERE plate = ? AND vehicle_class IS NOT NULL"
        " GROUP BY vehicle_class ORDER BY n DESC LIMIT 1", (plate,), conn=conn)
    return rows_[0]["vehicle_class"] if rows_ else None


def _build(plate: str, conn=None) -> RegistryRecord:
    rng = _rng(plate)
    if rng.random() < config.REGISTRY_NOT_FOUND_RATE:
        return RegistryRecord(plate=plate, registry_status=NOT_FOUND)

    observed = _majority_vehicle_class(plate, conn=conn)
    truth = observed if observed in _VEHICLE_CLASSES else rng.choice(_VEHICLE_CLASSES)
    if rng.random() < config.REGISTRY_MISMATCH_RATE:
        klass = rng.choice([c for c in _VEHICLE_CLASSES if c != truth])
    else:
        klass = truth

    make, model = rng.choice(_MAKES_BY_CLASS[klass])
    now = time.time()
    age_years = rng.randint(0, 15)
    return RegistryRecord(
        plate=plate, registry_status=FOUND, vehicle_class=klass,
        make=make, model=model, colour=rng.choice(_COLOURS), fuel=rng.choice(_FUELS),
        status=rng.choice(_STATUSES),
        registered_on=now - age_years * 365.25 * 86400.0,
        insurance_to=now + rng.uniform(-200, 400) * 86400.0)


def lookup(plate: str, conn=None) -> RegistryRecord:
    """A synthetic registry record for `plate`. See the module docstring.

    Deterministic for a fixed sightings snapshot (not permanently - the
    vehicle_class output can change if more sightings for this plate arrive
    between calls). Cached for `config.REGISTRY_CACHE_TTL_S`; UNKNOWN is never
    cached, the same rule the registry/ package's own cache uses, since a
    forced or transient outage should not be remembered as one.
    """
    plate = plate.upper().replace(" ", "")
    if config.REGISTRY_FORCE_MODE == "UNKNOWN":
        return RegistryRecord(plate=plate, registry_status=UNKNOWN)

    cached = _CACHE.get(plate)
    if cached and time.time() - cached[0] < config.REGISTRY_CACHE_TTL_S:
        return cached[1]

    record = _build(plate, conn=conn)
    _CACHE[plate] = (time.time(), record)
    return record


# --- partial-plate resolution -------------------------------------------
def _compatible_shapes(mask: str) -> list[str]:
    """PATTERNS/BH_PATTERNS shapes whose length and fixed-position class agree
    with `mask`. Coarse pass only - see `_shape_options` for the state-code and
    literal-BH narrowing the raw letter/digit shape string doesn't encode."""
    wc = config.REGISTRY_MASK_WILDCARD
    out = []
    for shape in PATTERNS + BH_PATTERNS:
        if len(shape) != len(mask):
            continue
        ok = True
        for m, s in zip(mask, shape):
            if m == wc:
                continue
            m_class = "L" if m in LETTERS else "D"
            if m_class != s:
                ok = False
                break
        if ok:
            out.append(shape)
    return out


def _shape_options(mask: str, shape: str) -> list[list[str]] | None:
    """Segment-wise admissible strings for one shape.

    The state-code pair (an ordinary series' first two characters) and the
    literal "BH" (a Bharat series' third and fourth) are each returned as one
    two-character segment rather than two independent single-character ones -
    STATE_CODES does not factor into independent per-position choices (not
    every combination of a valid first letter and a valid second letter is a
    real code), and treating them as a joined block is what keeps enumeration
    exact rather than over-generous. `itertools.product` over the returned
    segments, joined, produces exactly the valid candidates for this shape -
    nothing this function admits should ever fail `matches_plate_grammar()`.
    Returns None if a fixed character rules the whole shape out once these
    extra rules are applied (only possible at the state-code/BH positions).
    """
    wc = config.REGISTRY_MASK_WILDCARD
    is_bh = shape in BH_PATTERNS
    segments: list[list[str]] = []
    i, n = 0, len(shape)
    while i < n:
        if not is_bh and i == 0:
            pair = mask[0:2]
            if wc not in pair:
                if pair not in STATE_CODES:
                    return None
                segments.append([pair])
            else:
                codes = [c for c in STATE_CODES
                        if all(pc == wc or pc == cc for pc, cc in zip(pair, c))]
                if not codes:
                    return None
                segments.append(codes)
            i += 2
            continue
        if is_bh and i == 2:
            if mask[2] not in (wc, "B") or mask[3] not in (wc, "H"):
                return None
            segments.append(["BH"])
            i += 2
            continue
        m, s = mask[i], shape[i]
        segments.append([m] if m != wc else (list(LETTERS) if s == "L" else list(DIGITS)))
        i += 1
    return segments


def _candidate_space(mask: str) -> tuple[list[tuple[str, list[list[str]]]], int]:
    """Every (shape, segments) this mask admits, plus the exact candidate
    count - computed analytically, without generating a single string, so the
    cap in `resolve_partial()` can be enforced before any real work happens."""
    usable = []
    total = 0
    for shape in _compatible_shapes(mask):
        segs = _shape_options(mask, shape)
        if segs is None:
            continue
        count = 1
        for seg in segs:
            count *= len(seg)
        usable.append((shape, segs))
        total += count
    return usable, total


def _enumerate(usable: list[tuple[str, list[list[str]]]]) -> list[str]:
    """Materialise every candidate `_candidate_space` counted.

    Different compatible shapes are provably disjoint - a concrete string's
    own letter/digit layout matches at most one shape's class sequence, since
    a character is unambiguously either a letter or a digit - so no candidate
    is ever generated twice across shapes.
    """
    out = []
    for _shape, segs in usable:
        for combo in itertools.product(*segs):
            out.append("".join(combo))
    return out


def _seen_matches(mask: str, conn=None) -> list[dict]:
    """Plates already sighted that match `mask`, by direct SQL LIKE.

    Independent of plate grammar on purpose - see the module docstring's
    "Partial-plate resolution" section for why real evidence must never be
    grammar-filtered away.
    """
    wc = config.REGISTRY_MASK_WILDCARD
    like = mask.replace(wc, "_")
    rows_ = db.rows(
        "SELECT plate, COUNT(*) n, MIN(ts) first_ts, MAX(ts) last_ts,"
        " AVG(confidence) conf FROM sightings WHERE plate LIKE ? GROUP BY plate",
        (like,), conn=conn)
    out = []
    for r in rows_:
        plate = r["plate"]
        last = db.rows("SELECT camera_id FROM sightings WHERE plate = ?"
                       " ORDER BY ts DESC LIMIT 1", (plate,), conn=conn)
        out.append({
            "plate": plate, "sightings": r["n"], "first_ts": r["first_ts"],
            "last_ts": r["last_ts"], "mean_confidence": r["conf"],
            "last_camera": last[0]["camera_id"] if last else None,
            "vehicle_class": _majority_vehicle_class(plate, conn=conn),
            "vehicle_class_source": "observed",
        })
    return out


def resolve_partial(mask: str, vehicle_class: str | None = None, conn=None) -> dict:
    """Given a masked pattern like WB01??9492, a shortlist to chase - not a
    verdict. See the module docstring's "Partial-plate resolution" section.
    """
    wc = config.REGISTRY_MASK_WILDCARD
    mask = mask.upper().replace(" ", "").replace("-", "")
    valid_chars = LETTERS | DIGITS | {wc}
    base = {"mask": mask, "wildcard": wc, "refused": True,
           "seen": [], "unseen_registered": [], "disclaimer": ""}
    if not mask or any(c not in valid_chars for c in mask):
        return {**base, "reason": "mask must contain only letters, digits and "
                                  f"the wildcard character '{wc}'"}
    if wc not in mask:
        return {**base, "reason": "no wildcard characters - use "
                                  "/plates/{plate}/registry for a single known plate"}
    if not (7 <= len(mask) <= 11):
        return {**base, "reason": f"length {len(mask)} matches no plate format "
                                  "(7-11 characters)"}

    conn = conn or db.connect()
    disclaimer = (
        "A lead list for an operator to verify, not a verdict. 'seen' entries are "
        "real camera sightings. 'unseen_registered' entries have never been sighted "
        "by any camera and are matched only against a mock, non-VAHAN registry "
        "(core/registry.py) - confirm independently before acting.")

    seen = _seen_matches(mask, conn=conn)
    seen_plates = {e["plate"] for e in seen}
    if vehicle_class:
        for e in seen:
            e["matches_vehicle_class"] = e["vehicle_class"] == vehicle_class

    usable, space = _candidate_space(mask)
    result = {
        "mask": mask, "wildcard": wc, "length": len(mask),
        "compatible_shapes": [s for s, _ in usable],
        "refused": False, "reason": "",
        "candidate_space": space, "witness_vehicle_class": vehicle_class,
        "seen": seen, "unseen_registered": [], "unseen_registered_truncated": False,
        "disclaimer": disclaimer,
    }
    if not usable:
        result["tier2_status"] = "no_matching_shape"
        return result
    if space > config.REGISTRY_PARTIAL_MAX_CANDIDATES:
        result["tier2_status"] = "over_cap"
        return result

    result["tier2_status"] = "ok"
    candidates = [p for p in _enumerate(usable)
                 if p not in seen_plates and matches_plate_grammar(p)]
    tier2 = []
    for p in candidates:
        rec = lookup(p, conn=conn)
        if rec.registry_status != FOUND:
            continue
        entry = {"plate": p, "plausibility": plate_plausibility(p),
                 "vehicle_class": rec.vehicle_class, "vehicle_class_source": "registry",
                 "make": rec.make, "colour": rec.colour, "synthetic": True}
        if vehicle_class:
            entry["matches_vehicle_class"] = rec.vehicle_class == vehicle_class
        tier2.append(entry)
    tier2.sort(key=lambda e: (0 if e.get("matches_vehicle_class") else 1,
                              -e["plausibility"]))
    result["unseen_registered_truncated"] = len(tier2) > config.REGISTRY_PARTIAL_MAX_UNSEEN
    result["unseen_registered"] = tier2[:config.REGISTRY_PARTIAL_MAX_UNSEEN]
    return result
