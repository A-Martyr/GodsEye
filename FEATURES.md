# What was added, and how it works

Seven capabilities, added after a feature-by-feature comparison against
Motorola Solutions' VehicleManager LPR analytics. Everything here is running
and everything quoted below is real output from this repository's own
database, not an illustration.

| Capability | Module | Answers |
|---|---|---|
| Multi-frame fusion with tracking | `anpr/track.py`, `ANPREngine.read_sequence` | which *vehicles* went past, not what is in this photograph |
| Movement prediction | `core/forecast.py` | where will it be next, and when is it usually where |
| Convoy detection, scored | `core/convoy.py` | who is travelling **with** it, rather than near it |
| Search without a plate | `core/vehicle_search.py` | find it when nobody caught the number |
| Geographic zones | `core/geofence.py` | who entered, who left, who is still inside |
| Self-enforced privacy | `core/privacy.py` | what is held, on whose authority, and who has looked |
| Reaching an outside observer | `core/reach.py`, `core/tunnel.py` | a link that works from any network |

A single idea runs through all of them, and it is the reason they are worth
having rather than merely having: **an answer is reported together with how
much evidence is behind it.** A ranked list looks equally authoritative on two
observations or two hundred. Every function below carries a `confident`,
`reliable`, `observability` or `caveats` field, and the dashboard renders those
*before* the numbers they qualify.

---

## 1. Multi-frame fusion with tracking

**The problem.** `read_frame()` answers "what is in this photograph". A camera
never faces that question - it watches a vehicle cross a zone and gets five or
fifteen frames of the same plate. Reading the middle one throws the rest away.

**Why it works.** The frames fail *independently*. A plate lost to glare in one
is legible in the next, so agreement between frames is evidence that no amount
of work on a single frame can produce. Measured here:

```
synthetic, 10 camera conditions    1 frame  52%  ->  8 frames  88%
  fog                                       32%  ->            83%
  dusk_highiso                              35%  ->            89%
  night_glare                               41%  ->            87%
real 532-crop corpus              per frame 91%  ->  per vehicle 97%
```

**How it works.** `anpr/track.py` groups plate detections across frames into one
`Track` per vehicle by box overlap. Two details earn their place:

- `_near()` matches on centre distance when overlap is zero. A plate at
  expressway speed moves further than its own width between frames and shares
  no pixels with itself - and `high_speed` is exactly where fusion pays most
  (41% single, 80% fused), so losing the track there is the worst case.
- A size guard stops a nearby lorry's plate stealing a motorcycle's track.

`read_sequence()` then tracks, reads **every crop of every track in one batch**,
picks the best variant per frame, and votes across frames weighted by CTC score.
Batching is what makes it affordable: a ten-frame sequence is forty-odd crops,
seconds one at a time and a fraction of one batched.

**Two defects the end-to-end test caught**, both fixed and both invisible to
unit tests:

1. *13 "vehicles" reported for one scene.* The localiser loses the plate in
   precisely the blurred and underexposed frames fusion exists to survive, so
   one vehicle became two tracks, and every piece of shop signage became its
   own "vehicle". `_merge_agreeing()` now folds tracks that independently read
   the **same accepted registration**. Merging on decoded text rather than on
   geometry is what makes that safe.
2. *Sorting by track length* put a wrong read backed by 3 frames above a
   correct one at 0.962. Now: accepted first, then confidence, then evidence.

**Measured on a 7-frame sequence where only 2 frames read correctly alone:**

```
frame 0:  WB25G4XXX @ 0.962  accepted   correct
frame 1:  HR25G4XXX @ 0.960  accepted            <- one character wrong
frame 2:    KA0M631 @ 0.244  REFUSED    [blurred]
frame 4:    MH51013 @ 0.401  REFUSED    [dark]
frame 6:  WB25G4XXX @ 0.938  accepted   correct

FUSED: WB25G4XXX @ 0.962 from frames [0,1,5,6]
       votes {WB25G4XXX: 2, HR25G4XXX: 1, MH0C08C65: 1}
```

**On real traffic video** (225 frames, Hyderabad): single-frame reading claimed
**19 distinct registrations**; tracked and fused reported **14 vehicles**, at
193 ms/frame against 186 ms - fusion is essentially free. The clearest case:
`AP09DU6490` was read confidently in **5 separate frames** and would have been
written to the sightings table as a phantom vehicle. Fusion outvoted it 9-to-5
with the correct `AP09CU6490`.

**Where it is used:** *ANPR engine* tab -> "Read a sequence".

**Limits.** Tracking assumes smooth motion and non-dense plates. A grammar
limitation surfaced during video testing: `PATTERNS` requires exactly four
trailing digits, so a real 3-digit plate (`AP21 AX 369`) cannot be represented
and is forced to `AP21AX3699`.

---

## 2. Movement prediction - `core/forecast.py`

VehicleManager advertises *"determine where it may be located in the future"*
and *"day/time trend analytics ... the best time to intercept"*. This platform
had no counterpart to either.

**The structural advantage.** A roadside plate reader answers "which camera was
it seen at next", because a camera knows nothing about the road.
`data/cameras.json` carries 35 junctions and **52 links with real geometry**,
so the answer here is a junction, reached by a named road of known length, in a
compass direction, at a time taken from how fast traffic has *actually* been
moving on that link.

**`predict(plate)`** builds a transition model from consecutive sightings - not
from the `next_camera` column, which records where the simulator *intended* a
vehicle to go. What was intended and what a camera saw are different things,
and only the second is evidence.

Two rules govern it:

- **Blend, do not choose.** One observed turn must not read as a certainty, and
  a vehicle with a daily commute must not be predicted as a stranger. Each
  distribution is the vehicle's own history *shrunk towards* the junction's
  general traffic, weighted by how much history exists. `PRIOR_STRENGTH` is in
  effect the number of imaginary crowd-average journeys every vehicle starts
  with; at 4, a plate needs four observed departures before its own habit
  outweighs the crowd's.
- **Median travel time, not mean.** A vehicle that stopped for fuel mid-leg is
  in this data and would drag an average somewhere no journey actually took.

```
TR29AK4933   last seen Kalighat Crossing, 32 min ago
             confident: True  (9 own departures at this junction)
   56%  Rashbehari Crossing          0.9 km   SE   typical  1.6 min (observed)
   17%  Tollygunge Phari             2.3 km    S   typical 12.9 min (observed)
   16%  New Town Biswa Bangla Gate  21.3 km  ENE   typical 14.3 min (observed)
    8%  Park Street Crossing         5.6 km  NNE   typical  8.0 min (observed)
```

A negative `due_in_minutes` means the vehicle should already have arrived and
has not been seen - a fact about the vehicle, not a gap in the model, so it is
shown as **overdue** rather than clipped.

**`pattern_of_life(plate)`** is the other half of interception: knowing a
vehicle passes Shyambazar is worth little; knowing it passes on weekday
mornings is what lets someone be standing there. `reliable` is false when a
vehicle has too few sightings for the word "usually" to be honest.

**Speed:** 60 ms per prediction, 2 ms per pattern, over 18,763 sightings.
**API:** `GET /api/plates/{plate}/forecast`, `GET /api/plates/{plate}/pattern`.
**Dashboard:** *Track a plate* -> "Where it is likely to go next".

---

## 3. Convoy detection - `core/convoy.py`

`trajectory.co_travellers()` already answered "who else was at that camera
around then", ranked by how often. On a quiet road that is the same question as
"who is travelling with this vehicle". **On a trunk corridor at 09:00 it is
not**: a camera passing forty vehicles a minute puts hundreds of unrelated
plates inside any two-minute window, so the top of a raw count is whoever
drives the busiest route.

That distinction is not academic. Motorola sells this for trafficking
investigations, where a false convoy points an investigation at a commuter.

**Two independent tests, both reported:**

1. **Co-occurrence against chance.** Every candidate carries what that pair's
   own camera traffic would have produced by chance - `expected_passes`, the
   `lift` over it, and a tail probability.
2. **Locked hops.** Consecutive shared *movements*, weighted above repeat
   sightings at one junction. Two vehicles crossing three different junctions
   together within seconds is convoy evidence; two vehicles that both pass one
   busy junction daily at 09:00 is a shared commute, and a raw count cannot
   tell those apart.

**The improvement, on the same plate:**

```
NAIVE co_travellers()            SCORED convoy.partners()
  TN07K5264   15 shared            152 candidates -> 58 scored -> 1 convoy
  WB27YN7910  36 shared            WB50CB4531   convoy, HIGH confidence
  TR25SR3051  17 shared              40 passes vs 7.15 expected
  RJ17NW0291  12 shared              lift 5.6, log10 p = -16.8
  BR48GV2705   6 shared            WB27YN7910   "shared route", LOW,
                                                 confident: false
```

`WB50CB4531` does not appear in the naive top five at all. The naive **#2** is
demoted to a shared commute, because 36 co-occurrences is roughly what those
cameras produce by chance.

`sweep()` and `groups()` answer the question an analyst actually starts with -
"show me who is moving in groups" - across the whole network. A *group* is a
set in which **every pair independently scored**, not a set reachable through
scored pairs, which would let one strong pair drag in strangers.

**API:** `GET /api/plates/{plate}/convoy`, `GET /api/convoys`.

---

## 4. Search without a plate - `core/vehicle_search.py`

A plate-reader platform that can only be queried by registration is useless in
the one case that matters most: the witness saw the vehicle and did not get the
number. VehicleManager sells the way out as two features - "describe the
vehicle in natural language" and "find images without a license plate".

**Rule-based, deliberately.** No model. The compensating advantage is that a
rule-based parser **can be printed**: `vocabulary()` enumerates everything it
understands, which is precisely what a model in this position cannot do.

It parses vehicle class (with the synonyms a person types - bike/scooter ->
`two_wheeler`, lorry -> `truck`, cab -> `taxi`, rickshaw -> `auto`), heading,
speed, place (fuzzy against camera names, sectors and roads, plus a radius) and
time expressions.

**The important part is the negative space.** Real output for
*"red truck heading north near Salt Lake yesterday evening"*:

```
readback:  location 'salt lake' (camera) -> 3 junction(s), plus junctions
           within 2.0 km: Salt Lake Sector V, Karunamoyee Crossing,
           Ultadanga Flyover; vehicle class: truck; heading north -
           accepting N, NNW, NNE ...

ignored:   term "red", attribute "colour" - colour 'red' was NOT used to
           filter: no camera on this network reports body colour - the
           reader works on a plate crop, not the vehicle
```

Someone who asks for a red truck and is handed every truck will believe the
reds were filtered. Saying so is the feature.

**API:** `GET /api/search/vehicles?q=...`

---

## 5. Geographic zones - `core/geofence.py`

Commercial products sell a "geographic boundary" filter: draw a shape, get the
vehicles inside. **Drawing the shape is the easy half, and it is the half they
ship.** The hard half is that a zone drawn on a map is a claim about the world,
while everything this platform knows comes from 35 cameras.

Zones are circles or polygons over the real camera coordinates, persisted to
`data/zones.json` so an operator can read and edit them. Three are seeded on
first run: Maidan and Esplanade core, Salt Lake and Sector V, NSCBI Airport
approach.

**`observability()` is the module's reason for existing.** It reports whether a
zone can see what it claims to:

```
Maidan and Esplanade core
  cameras: 3   portals: 3
  CAM-20 Esplanade            portal
  CAM-21 Park Street Crossing portal
  CAM-23 Maidan Victoria Gate portal
```

A **portal** is a camera inside the zone with at least one road link to a
camera outside it - the only junctions where a crossing can be witnessed. A
zone whose boundary has no camera on it cannot observe crossings at all, and
says so rather than returning a confident empty list.

`Visit` carries `entry_confirmed` and `exit_confirmed`, which separate a
witnessed crossing from a coincidence of the query window. **"Still inside" is
an inference, not an assertion** - a vehicle can leave through a junction
nobody watches.

`screen()` returns alert-worthy zone entries shaped like `core/alerts.py`'s
output but **does not emit them**: `alerts.py` owns the alert table and its
dedup policy, and two modules writing alerts is how duplicates start.

**API:** `GET /api/zones`, `GET /api/zones/{id}/visits`.

---

## 6. Self-enforced privacy - `core/privacy.py`

**This is the one worth winning.** Every commercial plate-reader product ships
a privacy policy, a data-management FAQ and a compliance guide. Those are
documents. They describe what an operator is *supposed* to do with several
million records of where private citizens drove, and nothing in the software
makes any of it happen.

Four mechanisms that run:

**Retention.** `retention_preview()` reports what a purge would delete, per
table, with counts and date ranges. `retention_purge()` is dry-run unless told
twice, and refuses a purge larger than `PRIVACY_MAX_PURGE_FRACTION` - a purge
that large is a bug, not a policy. **The purge is deliberately not exposed over
HTTP**: irreversible deletion should not be one mistyped URL away.

**Holds.** Plates on the watchlist or attached to an open alert are excluded
from automatic deletion, and the exclusion is reported. Evidence in an open
case is not routine data.

**Audit.** One row per plate lookup - actor, plate, purpose, case reference.
`who_searched(plate)` answers the subject-access question a citizen, an
oversight board or a defence lawyer asks. `access_summary()` surfaces
`lookups_without_case_ref`, because a high count there is the shape of
browsing, not investigating.

**k-anonymity.** `aggregate_traffic()` suppresses any group of fewer than *k*
vehicles and takes no plate parameter at all - the function to hand a transport
department or a researcher without handing over a movement log.

**`posture()`** is one call built to be read by someone who does **not** trust
the platform. Real output:

```
store       21,284 sightings, 694 plates, 35 cameras, spanning 3.19 days
retention   horizon 30 days, 0 rows past it
holds       255 plates held: 7 watchlisted, 255 by open alert
            by kind: odd_hour 255, detour 68, clone 29, speeding 13
access      1 lookup in 7 days, 1 without a case reference
controls    purpose required, k=5, purge needs confirmation, safety limit 0.5,
            audit outlives data

caveats     "255 registrations are held from deletion by an unacknowledged
             'odd_hour' alert - an unworked alert queue is acting as a
             retention policy"

not_claimed "no conformance with CJIS, the DPDP Act or any other named
             standard is asserted - those are findings about an organisation,
             not about code"
            "deletion frees database pages; it does not overwrite them, and
             backups, the WAL and filesystem free space are out of scope"
```

Those last two blocks are the point. The module reports the ways it can be
wrong and refuses to claim compliance it cannot deliver.

**API:** `GET /api/privacy/posture`, `/api/privacy/retention`,
`/api/privacy/access`.

---

## 7. Reaching an outside observer - `core/reach.py`, `core/tunnel.py`

The judge console lets someone outside the team inject an incident at a moment
of their choosing. It previously only worked on the same LAN.

Three tiers, each reporting what it can honestly guarantee: a **public
Cloudflare tunnel** (works from any network - verified end to end, a real
injection from the public internet raised the alert), **same-network** links
for every adapter, and **loopback** for handing over the laptop.

Two bugs found and fixed here:

- The old reachability check connected to *our own* LAN IP from *this* machine.
  That never traverses the firewall a phone would meet, so it reported
  "reachable" whenever the socket was bound. A green tick that cannot go red is
  worse than no tick.
- `stop.bat` killed by listening port, and a quick tunnel **listens on none** -
  it would have left a public address pointing at the laptop indefinitely.

A tunnel publishes a whole **origin**, not a path, so `api/main.py` restricts
public-host requests to the judge console and gates them behind an unguessable
token. Verified: through the tunnel, `/docs` and `/api/health` return 404 while
the console returns 200. LAN requests are unaffected.

---

## Operational note: one GPU, three services

On this machine (GTX 1650, 4 GB) **multiple CUDA contexts starve each other
catastrophically**, and it is not about workload:

- The test suite alongside the API: 22 checks in 13 minutes, then **59 checks
  in 90 seconds** the moment the API stopped.
- The paddle reader's warm-up reported `1190.6s` - 20 minutes blocked against a
  normal 13-23 s, a **~50x starvation**.
- Pausing the API's live feed changed nothing. The second *context existing*
  was the problem, not the work it was doing.

**Run the demo like this:**

```bash
GODSEYE_DEVICE=cpu  uvicorn api.main:app --port 8000      # CRNN, 16 ms on CPU
GODSEYE_DEVICE=cpu  streamlit run dashboard/app.py --server.port 8501
                    dashboard_paddle.bat                  # 8502 keeps the GPU
```

Only the real-photograph reader needs the GPU - it took decodes from 1650 ms to
43 ms there. The CRNN is 16 ms on CPU against 5.5 ms on GPU, a difference
nobody perceives, in exchange for the card no longer being fought over.

**Stop the API before `python tests.py`.** Otherwise a 90-second check becomes
a 15-minute apparent hang.
