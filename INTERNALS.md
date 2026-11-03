# The four capability modules, in depth

`FEATURES.md` says what these do. This says **how**, at the level of the data
structures, the arithmetic, the thresholds and the failure modes — the detail
you need to modify them, defend them to a judge, or decide you disagree with a
design decision.

Everything here landed in one commit, `92125bc`, on 6 September 2026:

| Module | Lines | Answers |
|---|---:|---|
| `core/vehicle_search.py` | 1,356 | Find a vehicle when nobody caught the plate |
| `core/geofence.py` | 985 | What crossing a zone is actually evidence of |
| `core/convoy.py` | 861 | Who is travelling *with* this vehicle, against chance |
| `core/privacy.py` | 838 | Limits the platform enforces on itself |
| | **4,040** | |

Plus 52 lines of `config.py` (30 tunables), 76 lines of `api/routes.py` (five
routes), and `data/zones.json`. **No existing file was modified** — the four
were written by parallel agents with one file each, because this repo has no
worktree isolation and two agents editing `core/db.py` would have collided.

Every number quoted below was measured against `data/godseye.db` as it stood on
6 September 2026 (23,786 sightings, 771 plates, 35 cameras, spanning 3.3 days).
Where the original commit message quotes different figures, it measured an
earlier state of the same database; the *shape* of the result reproduces, the
digits do not. Re-running the probes in this document is the way to check.

---

## Table of contents

1. [The common design rules](#0-the-common-design-rules)
2. [`vehicle_search.py` — description instead of registration](#1-vehicle_searchpy)
3. [`geofence.py` — zones, and what a zone can honestly see](#2-geofencepy)
4. [`convoy.py` — travelling together, scored against chance](#3-convoypy)
5. [`privacy.py` — controls that run rather than controls that are documented](#4-privacypy)
6. [Wiring: config, routes, and one defect](#5-wiring)
7. [How to check every claim here yourself](#6-reproducing-all-of-it)

---

## 0. The common design rules

The four modules were written independently and converge on the same three
rules, because they all face the same problem: a control room will act on
whatever the screen says, and a plate-reader platform's outputs are used to
point police at members of the public.

**Say what you did not use.** An operator who types "red truck" and is handed
every truck will believe the reds were filtered. That is worse than returning
nothing — it converts a wide net into a false narrowing that someone will act
on. So `vehicle_search` reports colour as *recognised-but-not-recorded* rather
than dropping the word; `geofence` reports how much of the boundary it can
actually see; `convoy` reports the chance expectation next to the observed
count; `privacy` reports that deletion frees pages rather than overwriting them.

**Separate observation from inference, in the output.** A camera sees a vehicle
at a junction. It never sees a vehicle *in an area*, never sees a vehicle
*leave* through an uncovered road, and never sees two vehicles *travelling
together*. Each of those is an inference, and each is labelled as one in the
returned object rather than smoothed into a single confident number.

**Refuse rather than degrade.** `vehicle_search.search()` refuses a query that
constrains nothing. `privacy.record()` refuses a lookup with no stated purpose.
`privacy.retention_purge()` is a dry run unless told twice. A control that can
be skipped by passing an empty string is documentation again.

---

## 1. `vehicle_search.py`

### 1.1 The problem

The platform can be queried by registration. The case that matters most is the
one where the witness saw the vehicle and did **not** get the number.

Commercial products sell the way out as two features — "describe the vehicle in
natural language" and "find images without a license plate". Both are the same
thing: a search over the attributes a camera records *besides* the plate.

What this network actually records about a vehicle is:

```
vehicle_class   car | two_wheeler | auto | truck | taxi | bus
direction       one of 16 compass points
speed_kmph      the spot speed at the junction
camera_id       which junction
ts              when
confidence      how far the reader stood behind the read
```

That is the entire vocabulary. It is smaller than a witness statement and much
smaller than what a person will type, and **the gap between the two is what the
whole module is built around**.

### 1.2 Why a rule-based parser, not a language model

A rule-based parser can enumerate itself. `vocabulary()` prints exactly what the
parser understands; `parse()` lists every token it did not consume. A language
model can do neither.

For a tool whose output points officers at members of the public, being able to
say *"I did not understand the word 'maroon'"* is worth more than understanding
it approximately. It also adds no dependency and no inference latency.

### 1.3 The gazetteer

Built once per `CameraNetwork` object (a process-wide singleton), cached by
`id(net)` in `_GAZETTEER`.

For each of the 35 cameras, `_gazetteer()` takes the junction **name** and the
**road** apart into every contiguous word run of length 1–4, and resolves each
run back against the *whole* network. That last part matters: "salt lake" is the
name of CAM-13 and the road of CAM-14, and an operator who says it means both.

Two filters keep the gazetteer from swallowing the city:

- **`GENERIC_PLACE_WORDS`** — a 40-word frozenset of terms that name a *kind* of
  place rather than a place: `crossing`, `junction`, `gate`, `road`, `sector`,
  `bridge`, `toll`, `expressway`, … Single-word keys matching one of these are
  skipped. Without it, "gate" matches three junctions and "road" matches most of
  the city, and a query mentioning either quietly acquires a location filter
  nobody asked for.

  `city` is in that set for a specific reason: without it, "entering the city"
  resolved through **Science City** and turned a question about the cordon into
  a question about one junction on the bypass.

- **Breadth cap** — a `camera`-kind phrase landing on more than
  `max(4, len(cameras) // 3)` junctions (12 on this network) is discarded
  entirely. Road and sector names are exempt: covering several junctions is what
  a road *is*, and the operator said the word deliberately.

Each surviving key carries a **specificity**, which later becomes a ranking term:

| Kind | Specificity | Meaning |
|---|---:|---|
| `camera_id` | 1.00 | "CAM-13" — exact |
| `camera` (unique hit) | 1.00 | a junction name matching one junction |
| `camera` (several hits) | 0.85 | a junction name shared by a few |
| `road` | 0.70 | every junction on a road |
| `sector` | 0.50 | a whole part of the city |

Keys are sorted **longest phrase first, then most specific**, so a four-word
junction name is consumed before the two-word road inside it, which is consumed
before any single word.

### 1.4 The parse

`parse()` tokenises via `_norm()` — lowercase, `&` → `and`, punctuation to
spaces, then collapse. Hyphens become spaces deliberately: "two-wheeler",
"CAM-13", "tuk-tuk" and "NH-19" are all written both ways by real people.

A `used[]` bitmap tracks consumed tokens. Sub-parsers run **in a fixed order**,
each taking only unconsumed tokens:

```
1. _parse_gateways      before places, so "entering the city" survives intact
2. _parse_places        longest gazetteer phrase first
3. _parse_classes       longest synonym phrase first
4. _parse_directions
5. _parse_time
6. _parse_speed
7. _parse_confidence
8. _parse_plate_fragment
9. _parse_not_recorded  LAST, so a colour inside a place name is already gone
```

The ordering is load-bearing at both ends. Gateways run first so the place
matcher cannot take `city` out of "entering the city". `_parse_not_recorded`
runs last so that "yellow taxi" consumes its own colour word through the class
synonym, and `yellow` is **not** then reported as an ignored colour attribute it
never was.

Anything still unconsumed, minus `STOPWORDS` and bare digits, becomes
`unrecognised` and generates an ambiguity line.

#### Vehicle class

`VEHICLE_CLASS_SYNONYMS` maps ~60 terms people actually type onto the six stored
values — `scooty`, `tuk tuk`, `toto`, `lorry`, `tempo`, `double decker`. Phrases
are matched before single words, so "auto rickshaw" never resolves through
"auto" alone.

#### Direction — the neighbour expansion

A camera's recorded direction is the bearing of the road at that junction
**rounded to 16 compass points**. A vehicle a witness calls "northbound" is
stored as `NNE` as often as `N`.

So `_add_direction()` accepts the named point *and its two neighbours*:

```python
idx = COMPASS_16.index(code)
accepted = [code, COMPASS_16[(idx - 1) % 16], COMPASS_16[(idx + 1) % 16]]
```

Filtering on `N` alone silently drops most of the vehicles the operator meant.
The exact point is kept separately in `primary_directions` and used to rank the
neighbours *below* it in `_score()` — which is where the difference belongs.

Disambiguating heading from place: `north` is both. The rule is cue words within
two tokens to the left.

- `DIRECTION_CUES` (`heading`, `bound`, `travelling`, `towards`, …) → heading
- `PLACE_CUES` (`in`, `at`, `near`, `sector`, `side`, …) → part of the city
- `northbound` / `north bound` → heading, unambiguously, with no cue needed
- Bare `north` with no cue → **read as a heading, and an ambiguity is recorded**
  telling the operator to say "in the north" if they meant the place

#### Time — the correction that matters most

A clock bound with **no day named** is an hour of day, not an instant.

Read as an instant against the store's clock, `"buses after 6pm"` asked at half
past midnight resolves to 18:00 *today*, which is in the future. The query
returns nothing, the operator reads that as "no buses", and it really means "no
time has passed yet". As an hour of day it means 18:00–23:59 on every day held —
both what they meant, and never empty for the wrong reason. Naming a day puts it
back to being a well-defined instant.

Day words are parsed in their own prior pass so `"after 6pm yesterday"` and
`"yesterday after 6pm"` resolve identically rather than depending on word order.

`_set_hours()` composes two kinds of clause differently:

- **band ∪ band** — "morning or evening" are alternatives, so they *widen*
- **band ∩ clock** — "mornings, after 9" are two statements about one window, so
  the clock bound *narrows*

An intersection can empty the set. `"mornings after 6pm"` is a real thing to
type and has no answer. An empty hour list and no hour list produce **identical
SQL**, so the difference is carried in a separate `impossible` flag, and
`search()` refuses to run rather than returning an empty result that reads like
"no such vehicle".

Bounds round **outwards** on both sides — "after 6:30pm" includes 18:00–18:29 —
because a search must never drop a match to a rounding decision the operator did
not make. The readback says so.

`_now()` returns `db.latest_ts()`, the data's own present, not the wall clock.
A seeded database stops advancing when seeding ends, so "yesterday evening"
measured from wall clock can sit entirely after the last sighting and return
nothing for every vehicle in the store.

#### Speed

"Speeding" means the same thing here as in `core/alerts.py` — above
`SPEEDING_FACTOR` (1.6) × the free-flow speed — so an operator searching for
speeding vehicles and the ingest rule that raises speeding alerts cannot
disagree about the word. The measurement differs: the alert rule uses the speed
implied *between* two cameras, this uses the spot speed the camera reported.

The reference is `_camera_free_kmph()`: the **mean** free-flow speed over the
links meeting that junction, not the fastest. A junction where a 60 km/h bypass
meets a 22 km/h lane is not a 60 km/h place for the vehicle that was turning.

This cannot be a constant in the `WHERE` clause — 40 km/h is speeding on a lane
and ordinary on the bypass — so it is applied as a post-filter in Python.

#### Plate fragment

Only from an explicit cue (`starting`, `ending`, `containing`, `like`) or from a
token matching `^[a-z]{2}\d{1,2}[a-z]{0,3}\d{0,4}$`. A bare four-character token
is far more often a word than a plate fragment, and **inventing a plate filter
is the one mistake here that would return nothing and look like an answer**.

#### Not-recorded terms

`NOT_RECORDED` holds five attributes the platform understands the meaning of and
cannot act on — colour (20 terms), make (27), model (21), occupancy (13),
condition (15) — each with a `why` string naming the actual reason:

> `colour` — *"no camera on this network reports body colour — the reader works
> on a plate crop, not the vehicle — so the results below contain every colour"*

Keeping them in the vocabulary **is the point**. An unknown word is a typo, but
"maroon" is a real description of a real vehicle this network simply does not
see, and the operator has to be told which of the two happened.

### 1.5 Scoring and confidence

`_where()` builds the part SQLite can do. Rows come back newest-first, capped at
`MAX_SCAN` (60,000) with a `truncated` flag.

Each row is scored **for ordering only** — explicitly not a probability that
this is the vehicle, since every row already passed the filter:

```
score = 0.30 · direction_fit    exact compass point = 1.0, neighbour = 0.75
      + 0.20 · place_fit        the gazetteer specificity of that junction
      + 0.25 · read_confidence  what the OCR engine stood behind
      + 0.25 · recency          0.5 ^ (age_hours / 6)
```

Recency carries as much weight as the read because a description search is
nearly always a search for where the vehicle is **now**.

`_candidates()` then collapses sightings to distinct registrations — the thing
the operator is actually after. A vehicle seen four times fitting the
description is a far better lead than four vehicles seen once each, and that is
the only evidence a structured search can offer towards identity:

```python
score = mean(match_scores) * (1.0 + 0.35 * log(n_matches))
```

Logarithmic, not linear: the tenth matching sighting says much less than the
second, and a vehicle that simply uses a monitored junction daily should not
outrank a good fit on volume alone. `corroborated` (≥2 matches) is a **separate
flag** rather than being buried in the score, because one strong match and three
weak ones can end up with the same number.

Each candidate carries `watched` from `db.is_watched()` — an operator meeting a
plate by description is at exactly the moment where whatever they do next
assumes this registration means one vehicle, and a watchlist hit must not wait
for them to look it up separately.

Two verdict flags:

```python
discriminating = constraints >= 2 and 0 < share <= 0.10
actionable     = 0 < len(candidates) <= 40
```

`share` is matched ÷ everything in the same window. **Without the base rate,
"37 matches" is unreadable**: 37 out of 90 is a description of the traffic, 37
out of 19,000 is a description of a vehicle.

### 1.6 Measured, 6 September 2026

| Query | Constraints | Matched / scanned | Share | Candidates | Verdict |
|---|---:|---:|---:|---:|---|
| `red truck heading north near Salt Lake yesterday evening` | 4 | 2 / 7,428 | 0.03% | 2 | discriminating, actionable |
| `speeding two wheeler in the north after 6pm` | 4 | 2 / 23,784 | 0.01% | 1 | discriminating, actionable |
| `trucks` | 1 | 1,496 / 23,784 | 6.29% | 55 | *"only 1 attribute was constrained, so this is a category, not a description"* |
| `mornings after 6pm` | 1 | 4,853 / 23,784 | 20.4% | 347 | not discriminating |

The first query also returned:

```
IGNORED [colour] 'red'
```

Two sightings out of 7,428 — and the operator is told, in the same response,
that the word "red" did no work.

**A gap worth knowing about.** `mornings` came back as *unrecognised*.
`TIME_BANDS` holds `morning`, not the plural, so the plural falls through to the
unrecognised list. That is the honest failure — the operator is told the word was
not used — but it is still a word the parser should understand, and the fix is
one entry in `TIME_BANDS`.

### 1.7 Self-description

- `vocabulary(net)` — every class synonym, compass term, time band, speed and
  confidence phrase, plate form, sector, road, junction name, gateway phrase and
  not-recorded term, with the `why` strings. An operator holding this knows
  *before they type* whether the platform can answer the question. **A term
  missing from this list is missing from the parser, with no third possibility.**
- `explain(text)` — parse and report the filter **without running it**, so a
  search box can show what it is about to do before it does it.

---

## 2. `geofence.py`

### 2.1 The problem

Commercial platforms sell a "geographic boundary" filter: draw a shape, get the
vehicles inside it. Drawing the shape is the easy half and it is the half those
products ship.

The hard half: **a zone drawn on a map is a claim about the world, while
everything this platform knows comes from 35 cameras.** The gap between the two
is where a control room gets an answer that is confidently wrong.

### 2.2 Geometry

`point_in_polygon()` is ray casting in **raw degrees**, treating longitude as x
and latitude as y — no projection. Over the 20 km Kolkata spans, the error from
ignoring meridian convergence is smaller than the width of the junction a camera
stands in. Circular zones use a proper `haversine_km`, because there a
systematic error would show up as a radius that is not the radius.

The strict/non-strict pair in the crossing test — `(y_i > lat) != (y_j > lat)` —
handles a vertex exactly on the ray, so a camera sitting precisely on a boundary
latitude is counted once rather than zero or twice. The same guard makes the
division safe: a horizontal edge is never a crossing.

`outline()` converts a circle to a 48-point polygon **in this module**, so the
shape drawn on the map is the same shape `contains()` tests against:

```python
d_lat = r / 110.574
d_lon = r / (111.320 * max(cos(radians(lat)), 1e-6))
```

Dividing the longitude step by `cos(latitude)` is not optional — skipping it
draws an ellipse and calls it a radius.

### 2.3 The Zone

Persisted in `data/zones.json`, **not** in the database, for a specific reason:
zones are operator configuration, not observations, and `core/db.reset()` exists
to wipe the observations before a demo. A zone an officer drew should survive
that.

**Camera membership is deliberately not stored on the zone.** It is recomputed
by `cameras_in()` from the shape and `data/cameras.json` — 35 haversines. A
stored copy would go stale the moment a camera moved or was added, producing a
zone that silently stopped watching a junction it appears on the map to cover.

Three alerting policies:

| `alert_on` | Behaviour |
|---|---|
| `watchlist` | only vehicles already of interest — the commercial behaviour |
| `any` | the entry itself is the offence — restricted area |
| `none` | a pure query filter, no alerting |

`alert_classes` narrows either to particular vehicle types — the lorry-ban and
bus-lane case, **the one thing here a plate-only product cannot express**.

The file is cached against its own mtime, so an operator editing it is picked up
but sixty reruns a minute are not. A malformed file returns `[]` rather than
raising: no zones is a survivable state, a traceback on page load is not.

The seed set fires **only when the file is absent**. An operator who deletes
every zone writes an empty list, the file exists, and their decision stands.

The third seeded zone (`airport-approach`) is deliberately a zone with **one
camera in it**, because a platform that only ships examples it handles well
teaches an operator nothing about where it stops working.

### 2.4 Portals — the doors

`boundary()` finds the **portals**: cameras inside the zone with at least one
road link to a camera outside it. Portals are the only junctions at which this
platform can witness a crossing, so they are **the structural ceiling on
everything else the module claims**.

It also reports `gateways_inside` — a gateway to the wider road network enclosed
by the zone is a hole geometry cannot close. A vehicle can leave the modelled
network entirely from in there, so its silence is not evidence that it stayed.

### 2.5 Visits

`visits()` cuts each plate's timeline into **runs of consecutive inside
sightings**, carrying the outside sightings that bracket each run:

```
outside … [inside inside inside] … outside
   ^before        one visit           ^after
```

A vehicle that goes in, out and in again produces **two** visits, because that
is two crossings and a control room asked about crossings.

Two flags separate a witnessed crossing from a coincidence of the query window:

- `entry_confirmed` — there was an outside sighting immediately before. Without
  it, the window merely opened while the vehicle was already there.
- `exit_confirmed` — there was an outside sighting after.

Dwell gets **two numbers, and no point estimate between them**:

- `dwell_observed_min` — first to last sighting inside. A **lower bound**.
- `dwell_upper_min` — the span between the bracketing outside sightings. An
  **upper bound**, and `None` when either bracket is missing rather than guessed.

### 2.6 "Still inside" — the question that will be got wrong

A vehicle last seen at a zone camera may be parked in it, or may have left
twenty minutes ago through a side road no camera covers. Asserting the first is
**how a cordon gets set around an empty car park**.

So `visits()` does not assert it. It estimates against the zone's own history.

`lags` = minutes between a vehicle's last sighting inside and its first sighting
outside, over every visit **in this same window** that was seen to end. That is
the zone's own answer to "how long does a vehicle go unseen before it turns up
outside again", calibrated against the traffic at the time of day being asked
about rather than a global average.

```python
def _survival(lags, silence_min):
    if len(lags) < MIN_LAG_SAMPLES:              # 8
        return exp(-silence_min / FALLBACK_DWELL_MIN), len(lags)   # 20 min
    longer = sum(1 for l in lags if l > silence_min)
    return (longer + 0.5) / (len(lags) + 1.0), len(lags)           # Laplace
```

Three decisions in nine lines:

- **Empirical survival function, not a fitted exponential.** Dwell in a city zone
  is nothing like memoryless — a delivery van at a loading bay and a car crossing
  Esplanade at 60 km/h are two populations, and a single rate parameter splits
  the difference and describes neither.
- **Laplace smoothing**, so a silence longer than every lag on record comes back
  as "very unlikely" rather than a flat zero no evidence supports.
- **Short fallback (20 min)** below 8 samples. With no evidence about a zone, the
  safer error is to stop claiming a vehicle is inside rather than keep claiming
  it for an hour. The `basis` string says which branch ran.

**A named bias, stated in the source.** A vehicle that left through an uncovered
road and was never seen again contributes nothing to `lags`, which pulls the
distribution short. Short lags make the function *less* willing to say a vehicle
is still inside — which is the direction the error should point.

`_pointed_out()` adds the only positive evidence of departure available: was the
vehicle's last observed heading aimed at a road out of the zone, within
`OUTBOUND_TOLERANCE_DEG` (60°, a junction's worth of slack)? If so the
probability is **halved, not zeroed**. A vehicle facing the exit road usually
leaves, but a delivery van indicating towards the main road and then parking is
an ordinary event — this is a discount on the claim, not a refutation.

Verdicts: `≥0.60` likely inside · `≥0.25` possibly inside · else probably left
unseen.

### 2.7 Observability — the gate a dashboard should check

`observability()` has two halves, **and the second is the one that matters**.

The structural half is geometry: cameras inside, portals, enclosed gateways.

The measured half asks the data what actually happened: of the vehicles that
appeared inside this zone, what share were seen approaching from outside first?
**A zone can look well guarded on a map and still score badly here, because the
roads people really use to get in are not the roads with cameras on them.**

`usable` is false for three structural reasons:

- no cameras — it observes nothing
- one camera — it cannot distinguish "in the area" from "at that junction", so
  its dwell times are revisit intervals wearing a costume
- no portal — nothing can ever be witnessed crossing

A fourth verdict, `porous`, fires when fewer than half the entries were
confirmed crossings.

### 2.8 Measured, 6 September 2026 (6-hour window)

| Zone | Kind | Cameras | Portals | Entries | Confirmed | Capture | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `maidan-esplanade` | circle 1.6 km | 3 | 3 | 141 | 125 | **88.7%** | observable |
| `salt-lake-sector-v` | polygon | 7 | 5 | 281 | 207 | **73.7%** | observable |
| `airport-approach` | circle 1.5 km | 1 | 1 | 72 | 49 | 68.1% | **not usable — single camera** |

Activity and dwell:

| Zone | Visits | Unique | Entered | Left | Still inside | *Supported* (p ≥ 0.60) | Median dwell | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `maidan-esplanade` | 141 | 114 | 125 | 123 | 18 | **3** | 5.0 min | 27.2 min |
| `salt-lake-sector-v` | 281 | 195 | 207 | 200 | 81 | **47** | 20.4 min | 118.7 min |
| `airport-approach` | 72 | 39 | 49 | 57 | 15 | **0** | 0.0 min | 98.5 min |

Read the "still inside" columns together. Maidan reports 18 vehicles unseen —
but only **3** are supported at p ≥ 0.60. A dashboard that showed 18 would be
overstating by six-fold.

The airport zone is the instructive one. Every one of its 15 "still inside"
vehicles came back at **p = 0.496 — "possibly inside"**, none supported, with:

```
basis: 57 completed visit(s) in this window set the scale;
       last seen pointed at a road out of the zone
```

That is the arithmetic in the open: the survival function gave ≈0.992 (a silence
of 0.8 min is shorter than nearly every recorded lag), and `_pointed_out()`
halved it to 0.496 — straight past the 0.60 "likely inside" threshold. Its
median dwell of **0.0 min** is the single-camera failure showing itself: with one
camera, most visits are one sighting, and a zero-minute dwell is not a short stay,
it is no measurement at all. Hence `measured: sightings >= 2` in `dwell_table()`.

### 2.9 Alerting

`screen()` **returns events; it does not emit them.** `core/alerts.py` owns the
alert table and the dedup policy, and a second module writing into that queue
from the side would make the two disagree about what has already been raised. So
this hands back the same dict shape `alerts._emit` returns, minus the `id` a
write would have produced, and the caller decides.

`end` is pinned **once** before the loop. The database is written to while this
runs, so leaving it `None` would let each zone resolve its own slightly later
clock and report a set of entries no single moment ever contained.

`_recently_alerted()` is read-only and checks `kind='geofence'` alerts for this
plate within `DEDUP_MINUTES` (20), matching on the zone id inside the JSON
`detail` column.

Every event carries `zone_usable` and `zone_verdict` **into the alert detail**,
so an officer reading it three weeks later can see how much the zone could
actually observe at the time, rather than having to go and ask.

`confirmed_only` is the honest switch — default `False`, because a watchlisted
vehicle sitting inside a cordon is worth knowing about either way, but the flag
rides on every event so an operator is never shown an inference dressed as a
witnessed crossing.

---

## 3. `convoy.py`

### 3.1 Why the existing function was not enough

`trajectory.co_travellers()` answers "who else was at that camera around then"
and ranks by how often. On a quiet road that is the same question as "who is
travelling with this vehicle". On a trunk corridor at 09:00 **it is not**: a
camera passing forty vehicles a minute puts hundreds of unrelated plates inside
any two-minute window, and the top of a raw co-occurrence count is simply
whoever drives the busiest corridor most often.

That matters more here than in most analytics. Convoy analysis is sold for
trafficking and organised-crime work, so **a false convoy does not produce a bad
chart — it points an investigation at a commuter who happens to leave home when
the target does.**

Measured today on `TR29AK4933`, the naive top five and what scoring says about
them:

| Naive rank | Plate | Shared | Cameras | Total sightings | **Mileage rank** | Scored verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `TN07K5264` | 20 | 6 | 50 | 23 | coincidence (lift 1.78, locked 0/3) |
| 2 | `WB27YN7910` | 45 | 5 | 87 | **1** | shared route (lift 4.52, locked 2/12) |
| 3 | `TR25SR3051` | 21 | 5 | 56 | 10 | coincidence (lift 2.00, locked 1/5) |
| 4 | `RJ17NW0291` | 14 | 5 | 49 | 26 | coincidence (lift 1.29, locked 0/1) |
| 5 | `WB46HBK0511` | 9 | 5 | 47 | 42 | coincidence (lift 2.08, locked 0/0) |

**Four of the naive top five are coincidences**, and its number two is the single
most-sighted vehicle on the entire network. The ranking is largely reporting
mileage. That is the whole case for this module.

### 3.2 Two questions, asked separately

**1. Did they meet more often than chance?**

For every sighting of A at camera *c* at time *t*, count how many vehicles that
camera actually passed in the window `[t-w, t+w]`, and multiply by B's own share
of that camera's traffic:

```
expected_c,t = local_traffic(c, t) × share(c, B)
```

That product is the number of B-sightings the window would be expected to
contain if B's journeys had nothing to do with A's. It is computed from the
**real local density**, so a rush-hour junction sets its own baseline rather
than being scored against the city average.

Summed over A's sightings it gives λ. The observed count *k* is tested as a
**Poisson upper tail**.

`_expected_passes()` computes it anchored on A *and* anchored on B and **takes
the larger**. The two condition on different journeys and there is no principled
way to pick one; taking the larger is the conservative choice — whichever way of
looking at the pair makes their meetings least surprising — so a pair only
clears the threshold when it clears it from both directions.

**2. Given that they met, did they keep station?**

This is the question the naive version cannot ask at all, and it is what
separates a convoy from a shared commute.

Take a leg A drove between two junctions. Did B drive the same leg between the
same two junctions **as its own consecutive movement**?

```
A:  c1 ──────── t2-t1 ────────▶ c2
B:  c1 ──────── (B's next) ───▶ c2
    entry_gap = B_at_c1 − A_at_c1
    exit_gap  = B_at_c2 − A_at_c2
    drift     = exit_gap − entry_gap
```

If B trails A by 9 s entering the link and 11 s leaving it, the two held
formation. If B is 90 s behind at one end and 20 s ahead at the other, they were
on the same road — which is not the same thing.

**The "consecutive for B as well" requirement is what makes this a shared journey
rather than a shared destination.** Without it, a vehicle that wandered off to
three other junctions in between would still be credited with the leg, because
it was seen at both ends. In code this is the single line:

```python
if second[1] != first[1] + 1:    # B's journey positions must be adjacent
    continue
```

### 3.3 The chance lock rate — a closed form

Under the null, the entry gap and the exit gap are **independent draws from the
co-occurrence window** (anything outside it would not have been paired at all).
Their difference is therefore triangular on `[-2w, 2w]`, and:

```
P(|drift| ≤ d) = 1 − (1 − d / 2w)²
```

At the defaults (w = 120 s, d = 15 s):

```
1 − (1 − 15/240)² = 1 − 0.9375² = 0.12109
```

Verified live: `0.1211`. So a pair gets **no credit for holding station until it
does so on noticeably more than one leg in eight**, and the locked-hop count is
tested as a binomial upper tail against q = 0.1211.

`WINDOW_S` is deliberately loose. It only decides what gets *considered*; the
station-keeping test decides what gets *believed*, and a looser window makes
that test **stronger** by giving an unrelated vehicle more room to drift.

### 3.4 Numerical care in the tails

`_log10_poisson_tail()` computes in log space throughout. The naive `1 - cdf`
underflows to zero for exactly the cases that matter — a pair seen fifty times
against an expectation of eight — and a p-value pinned at the smallest
representable float gives every strong pair the same score, **destroying the
ranking precisely at the top of it**. Summing outward from the k-th term with
the leading factor kept as a logarithm stays meaningful well past 1e-300.

For `k ≤ λ` it sums the lower tail directly and subtracts — not a surprise at
all, and direct summation is exact enough to say so.

`_log10_binom_tail()` is summed **exactly** rather than approximated. *n* here is
a count of journey legs, rarely past a few dozen, and a normal approximation is
worst exactly where this is used: small *n*, small *q*, *k* in the far tail. The
log-sum-exp trick keeps it stable.

### 3.5 The index

`SightingIndex` is a first-class object a caller builds **once** and hands to
several queries — a whole-network sweep needs per-camera traffic, per-camera
plate shares and per-plate ordered journeys for every plate at once, and
rebuilding them per pair would make the sweep quadratic in database work for no
reason.

| Structure | Shape | Purpose |
|---|---|---|
| `by_plate` | plate → sorted [(ts, camera)] | the journey |
| `by_camera` | camera → sorted [(ts, plate)] | who passed |
| `cam_ts` | camera → sorted [ts] | `bisect` for local traffic |
| `cam_share` | camera → plate → fraction | the chance baseline |
| `plate_cam` | plate → camera → [(ts, **journey position**)] | adjacency test |
| `meta` | plate → class, mean confidence | provenance |

`local_traffic()` is two `bisect` calls minus one (excluding the vehicle itself).
`build_index(hours=24)` over 10,588 rows / 495 plates / 35 cameras builds in
**under 0.1 s**, and `partners()` over 103 candidates likewise.

**Enrolments are excluded by default** (`sources=(config.CAMERA_SOURCE,)`). An
enrolment's *place* is attested by an operator but its *time* is chosen by a
heuristic (`config.ENROL_TARGET_KMPH` picks a plausible gap), so two enrolments
can be near-simultaneous **by construction**. That is precisely the signature
this module reads as evidence, and it is not an observation of two vehicles
being in one place.

### 3.6 The verdict ladder

```
score = −(co_log10_p + station_log10_p)
```

Adding log-evidence. The two tests are **not strictly independent** — the second
conditions on the first — so `score` is a **ranking quantity, not a calibrated
probability**, and both components are reported separately so an analyst who
trusts only one can rank on that alone.

The ladder, and **the order is not the obvious one**:

| Rung | Condition | Meaning |
|---|---|---|
| `insufficient evidence` | passes < 3 | too few to separate from chance either way |
| **`convoy`** | locked ≥ 3 **and** chain_links ≥ 1 **and** station_p ≤ 1e-3 | held formation, repeatedly, consecutively |
| `travelling together` | locked ≥ 1 **and** station_p ≤ 0.05 | some formation, not enough |
| `shared route` | co_p ≤ 1e-3 | met far above chance, **never held formation** |
| `coincidence` | neither | inside what the traffic produces |

**Station-keeping is asked first and co-occurrence second.** A pair that met far
above chance but never held formation drops to `shared route` rather than being
promoted by the strength of the co-occurrence alone. Only a pair failing *both*
tests is called a coincidence — an earlier arrangement called a pair seen 34
times against an expectation of 6.7 *"inside what the traffic alone produces"*,
which was flatly contradicted by the numbers printed beside it.

Three downgrades apply after the ladder:

- **weak reads** — either plate averaging below 0.70 OCR confidence drops the
  confidence one rung: *"the identities themselves are soft"*
- **confusable plates** — within one OCR substitution of each other forces
  confidence to `low` with *"check they are not one vehicle read two ways"*.
  Imported lazily from `anpr.ocr`, so a dashboard tab asking a database question
  does not pay for cv2 and the reader stack; a pair that cannot be checked is
  reported as not confusable, which is safe because the flag only ever downgrades.
- **junction repeats** — co-occurrences at junctions they never left together are
  counted, reported, and **excluded from the station-keeping test**, since two
  neighbours who both pass the same junction at 09:00 daily generate dozens.

`chain_links` and `longest_chain` measure *consecutive* shared legs of A's
journey. Two vehicles crossing three junctions in a row together is **one
sustained event, not three independent ones**, and far stronger evidence than
three shared legs scattered across a day — a distinction the naive count cannot
make.

There is even a note for the operator's mental picture: if the median gap
exceeds 30 s the reason adds *"they are not nose to tail — the separation is held
rather than closed, which is what a following vehicle looks like"*, because an
operator picturing a tight pair will look for the wrong thing on the CCTV.

### 3.7 Measured, 6 September 2026

`convoy.partners("TR29AK4933", hours=24)` — 84 sightings, 103 candidates shared
a camera window, 47 met the 3-pass floor:

| Partner | Verdict | Passes | Expected | Lift | Locked / hops | log₁₀ p (co) | log₁₀ p (station) | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `WB50CB4531` | **travelling together / medium** | 23 | 4.95 | 4.65× | **6 / 14** | −8.49 | −2.40 | 10.89 |
| `WB27YN7910` | shared route / low | 22 | 4.86 | 4.52× | 2 / 12 | −7.95 | −0.36 | 8.31 |

`WB50CB4531` **does not appear in the naive top five at all.** Its evidence:

```
hop Salt Lake Sector V   -> Chingrighata Crossing  entry= 30.4s exit= 27.3s drift= -3.1s  locked
hop Chingrighata Crossing -> Science City          entry=-91.9s exit=-88.1s drift=  3.7s  locked
hop Chingrighata Crossing -> Science City          entry= 27.3s exit= 22.0s drift= -5.3s  locked
```

Note the second row: B is **91.9 s ahead** entering and 88.1 s ahead leaving. A
gap of a minute and a half — but held to within 3.7 s across the link. That is
station-keeping, and no co-occurrence count would ever find it.

`WB27YN7910` is the demotion that justifies the module. It met 22 times against
4.86 expected — genuinely on the road at the same times, 4.5× lift — but only
2 of 12 crossings held formation, **inside the 12% chance rate**, with a median
gap change of 54 s. Verdict: *"This is what a shared commute or a common corridor
looks like, not a convoy."*

Two honest notes on today's numbers:

- The naive count says 45 shared passes for `WB27YN7910`; the scorer says 22.
  `trajectory.co_travellers()` has **no time-window parameter at all** — it scans
  the plate's entire history — while the index was built over 24 hours.
- No pair on this database currently reaches the `convoy` rung. `WB50CB4531`
  clears the locked-hop floor (6 ≥ 3) and the chain requirement (longest chain 3)
  but its station p-value, −2.40, misses `P_STRONG` (−3). **The ladder is doing
  its job: it is not calling something a convoy that the arithmetic does not
  support.** The commit message's stronger figures for this same pair were
  measured against an earlier state of the database.

### 3.8 Groups — cliques, not components

```python
groups(hours=2, min_size=2, min_verdict="travelling together")
```

A group is a set in which **every pair independently scored** — not a set that
happens to be reachable through scored pairs.

The difference is not academic. Run against this repo's database, taking
connected components of the pair graph returns **one blob of 46 vehicles held
together by 102 of its 1,035 possible pairs**. That is not a 46-vehicle fleet,
it is the north-corridor traffic — and reporting it as a group would be the same
failure as the naive co-occurrence count, one level up.

So `nx.find_cliques()` is used instead, which is exactly the "every pair
confirmed" condition. `find_cliques` is output-sensitive and this graph is
sparse, but a denser one is theoretically exponential and this runs behind an
HTTP request, so it is **capped at 20,000 cliques** — a pathological graph
becomes a truncated answer rather than a hung dashboard.

The honest cost: cliques overlap, so a five-vehicle convoy with one weak
internal pair comes back as two overlapping fours. Two fields keep that visible
rather than hiding it — `component_size` (how large the loose neighbourhood
around the group is) and `variants` (overlapping cliques folded into this row
because they introduced no new vehicle). Every row returned therefore brings at
least one registration the rows above did not.

`weakest_pair` is reported rather than averaged away, **because that pair is the
one an opposing analyst will pull on**.

Measured at `hours=2`: four groups, all size 2, all `travelling together`, none
`confident` — which requires every internal pair to be a full `convoy`.

### 3.9 Stated limitations

Written into the module docstring, not discovered later:

- The Poisson null assumes a vehicle's crossings of a camera are exchangeable
  within that camera's traffic. This **understates chance co-occurrence for two
  vehicles sharing a timetable**, so the co-occurrence p-value alone over-calls
  commuter pairs — which is exactly why no verdict rests on it alone.
- The binomial null assumes entry and exit gaps are independent given
  co-occurrence. True of unrelated vehicles, false of vehicles that were
  following each other — the direction that makes the test **conservative**.
- **Neither p-value is corrected for the number of pairs tested.** A whole-network
  sweep examines thousands, so read the sweep as a *ranking* and the named-plate
  query as a *test*.

---

## 4. `privacy.py`

### 4.1 The problem

Every commercial plate-reader product ships a privacy policy, a data-management
FAQ and a compliance guide. **Those are documents.** They describe what an
operator is supposed to do with a store of several million records of where
private citizens drove, and nothing in the software makes any of it happen.

**The gap between the policy and the binary is where mass plate surveillance
actually lives.** This module is the same set of promises written as code that
runs.

### 4.2 Audit — `record()`

Creates `access_log` (id, ts, actor, action, plate, purpose, case_ref, scope,
results, channel) with three indices. `_ensure_audit()` runs on every entry
point rather than once at import, because the dashboard, the API and the CLI
each reach the database through their own thread-local connection and none calls
an init hook this module controls. The statements are all `IF NOT EXISTS`, so
being wrong costs a parse; skipping it costs a crash in front of an audience.

Two refusals, and they are the control:

```python
if not actor:
    raise ValueError("access log needs an actor - an unattributed lookup is not auditable")
if REQUIRE_PURPOSE and not purpose:
    raise ValueError(f"access log needs a stated purpose for {action} ...")
```

**An audit log with 4,000 rows reading `actor=system, purpose=` documents
nothing, and is worse than none because it looks like accountability.**

The instruction in the docstring is explicit: call `record()` *before* returning
results, and **do not wrap it in a try/except that swallows the failure**. A
search that cannot be logged should fail rather than proceed — a system that
quietly keeps answering when its audit trail is broken is a surveillance
apparatus with the receipts turned off, and that state is indistinguishable from
the one an abusive operator would engineer on purpose.

Three readers:

- `who_searched(plate)` — the **subject-access question**, the one a citizen, an
  oversight board or a defence lawyer asks. Answerable only because `record()`
  refuses to write a row without an actor and a purpose. Flagged
  `audited_paths_only: True` — an empty result means nobody looked *through an
  audited path*, which is not proof nobody looked.
- `access_summary(days)` — per-actor volume. **The number that matters is
  `no_case_ref`.** Targeted lookups against a named investigation are the
  legitimate use; a high volume of plate lookups with no case attached is what
  misuse looks like from the inside, and it is visible only in aggregate over an
  actor. Deliberately a **signal, not a finding** — plenty of legitimate work has
  no case number yet, so it counts and ranks and stops.
- `access_history(...)` — the raw log, **anchored on wall clock** unlike every
  analytics window in this platform, because an access happened when a person did
  it and that has nothing to do with how fresh the sighting data is.

### 4.3 Retention holds

Two grounds for pinning a registration against automatic deletion: it is on the
**watchlist** (a live instruction from the control room), or it is attached to an
**unacknowledged alert** (evidence in something nobody has closed).

The breakdown by alert kind exists because it **exposes a failure mode this
design creates**. Measured today:

```
holds: 255 plates of 771 (33%)
  watchlisted:       8      (all 8 also carry open alerts)
  by open alert:   255
  by_alert_kind:  odd_hour 255 · detour 69 · clone 29 · speeding 13
                  · watchlist 8 · loitering 3
```

**A third of the city is held from deletion, and every one of those 255 plates is
held by an unacknowledged `odd_hour` alert** — a low-severity rule that fires on
anyone driving between 01:00 and 05:00.

An alert queue nobody clears **silently becomes a retention policy nobody chose.**
The fix is operational (work the queue) or configured
(`PRIVACY_HOLD_SEVERITIES`, which defaults to all four severities — the cautious
reading of "evidence in an open case", and worth looking at twice). The code's
job is to make it impossible not to notice, which is why `posture()` raises it as
a caveat automatically.

### 4.4 Retention — the dry run *is* the purge

The single most important structural decision in this module:

```python
_HELD = "SELECT plate FROM _privacy_hold"
_TABLES = [
    ("sightings",  "ts", f"plate NOT IN ({_HELD})",  f"plate IN ({_HELD})"),
    ("alerts",     "ts", f"acked = 1 AND (plate IS NULL OR plate NOT IN ({_HELD}))",
                         f"acked = 0 OR plate IN ({_HELD})"),
    ("unresolved", "ts", f"best_text IS NULL OR best_text NOT IN ({_HELD})",
                         f"best_text IN ({_HELD})"),
    ("inferences", "ts", f"plate NOT IN ({_HELD})",  f"plate IN ({_HELD})"),
]
```

`_stage_holds()` writes the held registrations into a **temp table** — not a
Python-side filter, not an inline `IN (...)` list. Preview and purge then run
**byte-identical predicates**. That makes correctness *structural* rather than a
thing two functions are supposed to keep in step, which is how a dry run ends up
lying about what a delete will do. It also sidesteps SQLite's bound-parameter
ceiling once a hold set grows past a few hundred plates.

Each table's two predicates **partition** everything past the cutoff, so a row is
counted as deletable or held and never both. Note the asymmetries:

- an **open alert is never routine data whatever its age**, so `acked = 0` counts
  as *held*, not as deletable-but-skipped
- `unresolved.best_text` is a below-threshold read, not an established
  registration — but if it happens to name a held plate it is treated as held,
  the cautious direction when the string is unreliable anyway

**The cutoff is measured against wall clock, deliberately**, and not against
`db.latest_ts()` the way every analytics window is. The analytics anchor exists
so a seeded database still shows a busy last thirty minutes. Retention is the
opposite question: a store whose cameras went dark a year ago is a store holding
year-old movement records, and anchoring the horizon on the newest row would
mean **data stops ageing the moment ingest stops**. `clock_drift_hours` is
reported so an operator can see when the two disagree.

### 4.5 The purge

`retention_purge()` is a dry run unless told twice:

1. `confirm=False` is the **default** — the unthinking call and the call from a
   scheduler that lost its arguments both do nothing
2. `actor` is **required** when confirm is set, because the purge writes itself
   into the access log. A deletion of evidence that leaves no trace is the one
   operation here that could be used to cover something up.
3. `MAX_PURGE_FRACTION` (0.5) — a purge taking out more than half the store stops
   and asks again. This guards against a fat-fingered horizon (`days=0` on a live
   database), **not against policy**; a first legitimate run on an old store
   really can be most of it, and `force=True` says so.

Deletion order is **reversed** — `inferences` first, then `unresolved`, `alerts`,
`sightings`. Each inference points at an `unresolved` row, and deleting the
parent before the child would leave a dangling evidence chain rather than a clean
absence. Wrapped in try/rollback/raise.

**What it does not do: erase.**

```python
plan["secure_erase"] = False
plan["erase_note"] = ("rows are gone from every query; SQLite keeps the freed pages"
                      " inside the database file until VACUUM, and the WAL and any"
                      " filesystem snapshot are outside this module's reach")
```

`vacuum=True` is not offered, because VACUUM takes an exclusive lock and this
database has a live API and two dashboards on it. Calling this erasure would be
a lie, so it does not.

`purge_audit_log()` is kept **separate** rather than folded in, so trimming the
record of who searched is always a deliberate act with its own audit row, and can
never happen as a side effect of routine housekeeping. It writes its own audit
row *first*, then excludes that row's id from the delete.

Measured today:

| Table | Rows | Past horizon | Deletable | Held back |
|---|---:|---:|---:|---:|
| `sightings` | 23,786 | 0 | 0 | 0 |
| `alerts` | 1,196 | 0 | 0 | 0 |
| `unresolved` | 2,721 | 0 | 0 | 0 |
| `inferences` | 0 | 0 | 0 | 0 |

Nothing is deletable — the store spans 3.3 days against a 30-day horizon, and
`posture()` says so as a caveat: *"the purge is untested against this data."*
That is the correct thing to report, and it is reported rather than left for
someone to infer from four zeroes.

### 4.6 k-anonymity

`aggregate_traffic()` is the function to hand a transport department, a
researcher or a press office.

**It has no plate parameter and cannot be given one.** A k-anonymous query with
an identity filter is a lookup wearing a hat, and offering the argument at all
would guarantee someone uses it and calls the result anonymised.

`group_by` is a **whitelist**, not a column name passed to SQL — eight
dimensions: camera, sector, road, vehicle_class, direction, hour, weekday, date.
Two reasons: an arbitrary expression is an injection, and a grouping fine enough
(plate, sighting id, second-resolution time) re-identifies individuals no matter
what k is.

**k counts distinct vehicles, not sightings.** That distinction is the whole
control: one car passing a quiet junction forty times in a day is forty rows and
one person, and a threshold on row count would publish that person's evening as
a statistic.

`k < 2` raises — *"k below 2 suppresses nothing, that is not anonymity."*

**Complementary suppression.** If exactly one group falls below k, its size is
recoverable by subtracting every published group from the total. So the next
smallest group is suppressed alongside it. That costs a real cell of data, and
it is the price of the released total meaning anything.

`representative` is false when suppression bit hard enough that the released
groups are no longer a fair picture (< 90% of sightings released) — the caller
should widen the window or coarsen the grouping rather than publish.

Measured over a 6-hour window:

| Grouping | Released | Suppressed | Suppressed share | Representative |
|---|---:|---:|---:|---|
| `camera` | 35 | 0 | 0.00% | yes |
| `sector` | 8 | 0 | 0.00% | yes |
| `camera` × `hour` | 129 | **34** | 3.01% | yes |

Coarse groupings pass everything; the fine grouping suppresses 34 cells — the
thin junction-hours where fewer than five distinct vehicles were seen. That is
k-anonymity doing visible work.

**What it does not defend against**, stated plainly: a caller who runs it
repeatedly over sliding windows and differences the results. Nothing at this
layer can — that needs query budgeting or noise, either of which would be a
bigger claim than this code can support. **The access log is the mitigation that
exists**, which is why aggregate calls should be recorded too.

### 4.7 Pseudonymisation

`pseudonym()` is HMAC-SHA256 keyed with a per-database salt (`secrets.token_hex(16)`,
stored in `meta`, generated on first use), truncated to 16 hex characters.
Digests are comparable inside one deployment and meaningless outside it.

**This is pseudonymisation and not anonymisation, and the difference is not
pedantry.** Indian registration numbers are a small enumerable space — anyone
holding the salt can hash every plausible plate in minutes and invert the whole
table. It protects a leaked export; it does not protect against someone who
leaked the database that holds the salt.

### 4.8 Posture

`posture()` is one call built to be read **by someone who does not trust the
platform**: an oversight board, a judge, a journalist. Every figure is a count
from the database rather than a policy statement.

Five caveat rules fire automatically. Two fired today:

```
- 255 registrations are held from deletion by an unacknowledged 'odd_hour'
  alert - an unworked alert queue is acting as a retention policy
- the store spans 3.3 days against a 30-day horizon, so nothing is due for
  deletion yet - the purge is untested against this data
```

The other three watch for a stale/replayed database (drift > 24 h), zero audited
lookups (*"either nobody searched, or a search path is reaching the data without
calling record()"*), and `REQUIRE_PURPOSE` being switched off (*"which makes the
audit trail decorative"*).

And a `not_claimed` block, said plainly **because the alternative is that a
screenshot of this report gets read as a certification**:

> - no conformance with CJIS, the DPDP Act or any other named standard is
>   asserted — those are findings about an organisation, not about code
> - deletion frees database pages; it does not overwrite them, and backups, the
>   WAL and filesystem snapshots are outside this module
> - the audit trail covers callers that invoke `record()`; **direct SQL against
>   the database file leaves no trace here**

There is a CLI: `python -m core.privacy --posture | --preview | --purge --actor NAME`.

---

## 5. Wiring

### 5.1 Config

52 lines added to `config.py`, 30 tunables. Every module reads them through
`getattr(config, "NAME", default)` with the *same* default, so each runs whether
or not the block is present — **but a tunable nobody can find is a tunable nobody
tunes.**

Three are environment-overridable: `GODSEYE_CONVOY_WINDOW_S`,
`GODSEYE_RETENTION_DAYS`, `GODSEYE_PSEUDONYM_SALT`.

### 5.2 Routes

**Eight** endpoints added to `api/routes.py`. (The commit message says five; the
diff adds eight — it undercounts the three `privacy/*` routes as one.)

| Route | Module | Status today |
|---|---|---|
| `GET /api/search/vehicles?q=` | `vehicle_search.search` | 200 |
| `GET /api/zones` | `geofence.load_zones` | 200 |
| `GET /api/zones/{id}/visits?hours=` | `geofence.visits` | 200 (404 on unknown zone) |
| `GET /api/plates/{plate}/convoy?hours=` | `convoy.partners` | 200 |
| `GET /api/convoys?hours=` | `convoy.groups` | 200 |
| `GET /api/privacy/posture` | `privacy.posture` | 200 |
| `GET /api/privacy/retention` | `privacy.retention_preview` | 200 |
| `GET /api/privacy/access?hours=` | `privacy.access_summary` | 200 — **was 500, fixed; see below** |

**The purge is deliberately not exposed over HTTP.** Irreversible deletion should
not be one mistyped URL away. `/api/privacy/retention` is dry-run only.

### 5.3 One defect, found while writing this — and fixed

`GET /api/privacy/access` returned **HTTP 500 on every call**:

```
TypeError: access_summary() got an unexpected keyword argument 'hours'
```

The route took `hours` and forwarded it as `hours=`; `privacy.access_summary()`
takes **`days`**.

**How it survived the commit's verification.** That commit checked the modules by
calling them directly — 33/33 calls succeeded, including the degenerate cases.
Every one of those calls used the correct keyword, because they called
`access_summary()` rather than the route that wraps it. A signature mismatch
between a route and the function it delegates to is invisible to a test that
never goes through the route, and this is exactly that class of defect. It is
also why the probe table in §5.2 exists: the routes were exercised over HTTP,
and only one of the eight failed.

Fixed in `api/routes.py` by converting at the boundary rather than changing
either side:

```python
def privacy_access(hours: float = Query(168.0, ge=1, le=8760)):
    # The route takes hours to match every other window on this API;
    # access_summary() takes days, because an audit review is asked in days.
    # Convert here rather than changing either side - the default 168 h is the
    # 7 days access_summary() would have used anyway.
    return privacy.access_summary(days=hours / 24.0)
```

The route keeps `hours` because every other window on this API is in hours, and
`access_summary()` keeps `days` because an audit review is asked in days. The
default 168 h maps to the 7.0 days the function would have used anyway, so the
fix changes no behaviour that ever worked.

Verified against the restarted API: `hours=168 → window_days 7.0`,
`hours=24 → 1.0`, `hours=1 → 0.041667`. All eight routes return 200 (404 on an
unknown zone), and `python tests.py` reports **81 passed, 0 failed**.

Nothing else was affected: `access_summary()` itself was always correct, and the
dashboard does not call this route.

---

## 6. Reproducing all of it

Every number in this document came from these calls against the live database.

```bash
python tests.py                              # 81 checks, all four modules included
```

```python
from core import vehicle_search, geofence, convoy, privacy, trajectory

# 1. search - readback, ignored terms, base rate
vehicle_search.search("red truck heading north near Salt Lake yesterday evening")
vehicle_search.explain("trucks")             # parse without running
vehicle_search.vocabulary()                  # everything the parser understands

# 2. geofence - the coverage gate first, always
geofence.observability("airport-approach", hours=6)     # usable=False
geofence.activity("salt-lake-sector-v", hours=6)
geofence.dwell_table("maidan-esplanade", hours=6)

# 3. convoy - the comparison that justifies the module
trajectory.co_travellers("TR29AK4933")       # naive: ranks mileage
idx = convoy.build_index(hours=24)
convoy.partners("TR29AK4933", hours=24, index=idx)
convoy.groups(hours=2)

# 4. privacy
privacy.holds()                              # 255 of 771 plates held
privacy.retention_preview()                  # dry run, never deletes
privacy.aggregate_traffic(group_by=("camera", "hour"), minutes=360)
privacy.posture()                            # the whole picture, with caveats
```

Two cautions when reproducing:

- Run with `GODSEYE_DEVICE=cpu` if the GPU reader is up on 8502. `convoy` imports
  `anpr.ocr.confusion_distance` lazily, and a second CUDA context on a 4 GB card
  starves the reader (see `PERFORMANCE.md`).
- Figures move as the live feed writes. The **shape** of every result above
  reproduces; the digits track the database.

---

## Where to read next

| File | What is in it |
|---|---|
| `FEATURES.md` | the same seven capabilities at a summary level, including the three not covered here |
| `PERFORMANCE.md` | how a read went from 46 s to 118 ms, and what proved it cost no accuracy |
| `README.md` | the platform, its architecture, and every measured number |
