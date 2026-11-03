# Vehicle registry and Security Console

A mock VAHAN-shaped registry, and a console that checks what the cameras saw
against what the state holds on the vehicle — then routes what it finds to
whoever could act on it.

**Nothing in the existing demo depends on this.** It is a separate package, a
separate database and a separate port, and it is removable by deleting a
folder. See *Removing it* at the bottom.

---

## Run it

```bash
registry.bat                            # service on 8507 + console on 8506
```

Or by hand:

```bash
python -m registry.seed                 # build the registry (~20 s)
python -m registry.service              # lookups + the mock authority -> :8507
python -m registry.console              # the console  ->  http://localhost:8506
```

The console works with the service stopped — it reads `data/registry.db`
directly. What needs the service is **delivering** a referral, because the
whole point of that step is that it crosses a process boundary and can fail.

And from a terminal, without the UI:

```bash
python -m registry.sentinel             # sweep everything the cameras saw
python -m registry.sentinel WB24AB1234  # brief one registration
python -m registry.live WB24AB1234      # the whole chain for one plate
python -m registry.live                 # the chain over every live camera read
```

### The two demo switches

Both live in the console sidebar under **Demo controls**, and both act on the
running service rather than on the console, so what they demonstrate is the
real integration failing rather than a page pretending.

| Switch | What it shows |
|---|---|
| Registry → `UNAVAILABLE` | Every lookup answers `UNKNOWN`. The console's counts drop to zero **with a banner saying nothing was learned** — no vehicle is reported as unregistered, and no finding is raised. An outage is not a verdict. |
| Registry → `NOT_FOUND` | The opposite failure: the registry answers, and says there is no such vehicle. `404`, not `503`. |
| Authority → `OFFLINE` | Every referral fails after its retries. The incident stays on the *Authority notified* tab marked `DELIVERY_FAILED`, with **Retry failed** ready — a delivery problem, never a lost finding. |

---

## What it holds

`registry/store.py` is the seam. `lookup(plate)` returns the SHAPE a real VAHAN
integration would, so replacing that one function with a real call changes
nothing above it.

| | |
|---|---|
| Vehicle | make, model, class, colour, fuel, year, engine and chassis (masked) |
| Registration | owner, owner serial, locality, RTO office, code, state, date |
| Validity | insurance (+ insurer), PUC, fitness, permit, road tax |
| Status | active · stolen · scrapped · blacklisted · suspended, with the note behind it |
| Cases | challans, FIRs and summons — section, description, amount, station, reference |
| Audit | every lookup: who asked, about whom, and why |

**It is synthetic and says so.** Every record is invented from a hash of the
plate, so the same registration always returns the same vehicle, owner and
history — across processes and machines. No real vehicle, owner, insurer or
case is described, and `synthetic: true` rides on every response.

Two details that took a second pass:

- **A misread is not a vehicle.** Seeding straight from `sightings` registered
  strings like `6J45N1463` and gave them an invented office in a state that
  does not exist — 79 of 1,649 records. `registrable()` now requires the plate
  grammar and a real state code, so "this plate is not on the registry" became
  a usable signal instead of noise.
- **Plates are registered where their code says.** A `KA` plate came back
  registered at a Kolkata RTO, which made "registered out of state" impossible
  to tell from a generation artefact.

---

## What the sentinel flags

`registry/sentinel.py` holds the rules. The interesting ones need *both* halves
— the registry alone cannot see them, and the cameras alone cannot either:

| Finding | Fires when | Routed to |
|---|---|---|
| `stolen` | a registration reported stolen is moving | Police |
| `scrapped_but_moving` | a deregistered vehicle is photographed in traffic | Transport |
| `blacklisted` | a blacklisted registration is still in traffic | Transport |
| `open_fir` | open criminal case; **critical** only for theft/NDPS | Police |
| `uninsured` | third-party cover lapsed and the vehicle is driving | Traffic |
| `papers_lapsed` | PUC, fitness, permit or tax out of date | Transport / Pollution |
| `class_mismatch` | registry says scooter, cameras keep reading truck | Police |
| `unregistered` | a plate no registry ever issued, seen repeatedly | Police |
| `challans_outstanding` | unpaid challans over the alert threshold | Traffic |

**Every finding carries what it is not.** Where the platform's own accuracy
could explain the finding instead, the card says so *above* the recommended
action — the same discipline `core/clones.py` applies when it refuses to call a
misread a clone. These findings name an owner and route to a police station;
the reader has to see the alternative explanation before acting on one.

One tuning pass was needed and is worth recording. Drawn uniformly from the
case catalogue, **27 of 28 "critical" findings came back as open FIRs**, and
the single stolen vehicle actually moving through the network sat in the middle
of them, indistinguishable. Weighting the catalogue toward ordinary traffic
offences and grading FIRs by offence took critical from 28 to 4, with the
stolen vehicle at the top. A critical queue that is mostly noise is not a
critical queue.

---

## How it stays out of the way

| | |
|---|---|
| Imports | **nothing** in `anpr/`, `core/`, `api/`, `sim/`, `dashboard/`, `livecam/` or `phonecam/` imports this package |
| Platform database | opened `mode=ro` — a mistake here **cannot** write to `data/godseye.db`; enforced by SQLite, verified by test |
| Own storage | `data/registry.db`, its own file |
| Engine imports | **nothing** in `anpr/`, `core/`, `api/` or `sim/` imports this package |
| Dashboard imports | one file — `dashboard/registry_link.py`, a soft import that degrades to "unavailable" |
| Own ports | 8506 (console) and 8507 (service), clear of 8000/8501/8502/8503/8504/8505 |
| Own interpreter needs | none beyond the platform's — no paddle, no GPU, so it never takes a CUDA context off the cameras |
| Existing files changed | `stop.bat` (two `killport` lines) and `dashboard/app.py` (the panels below) |

### The guarantee, restated honestly

The control room **does** now import this package, so "no platform module
imports registry/" is no longer the promise — it was only ever a proxy for the
thing that matters:

> **Delete `registry/` and the dashboard still works.**

That is now an executed test rather than an assertion. `registry/selftest.py`
launches a subprocess with `registry` made unimportable, loads
`dashboard/registry_link.py`, and requires every entry point to return empty
rather than raise. The engine is still hard-isolated, and checked separately —
nothing on the path from a camera to a stored sighting may know this package
exists, which is what keeps a slow registry structurally incapable of costing a
plate read a millisecond.

The dependency still points one way: `registry/live.py` *reads* the platform's
sightings; the platform never calls into the registry. Enrichment lives here
rather than inside the camera pipeline, where it would be one `try` away from
stalling a junction.

> The isolation check used to shell out to `grep`. It was silently vacuous —
> `\s` does not survive Python's subprocess quoting to Git-for-Windows grep, so
> the pattern matched nothing and the check passed regardless of what the files
> contained. It matches in Python now, and asserts against a known positive
> first, so a broken pattern fails loudly instead of going quiet.

### In the control room

| Where | What it adds |
|---|---|
| **Track a plate** | A registry card under the trajectory: vehicle, RTO, papers, pending cases, and any findings — each with a **Refer** button and the caveat beside the claim |
| **Alerts** | A `REGISTRY: STOLEN` chip on an alert whose registration is flagged, and a quieter chip for expired papers. Silent for an ordinary vehicle, and silent during an outage |
| **Live cameras** | A `registry` column on recorded reads, and a flag on a journey card when a vehicle moving between junctions is flagged |

Every one of those renders nothing at all when the registry is absent.

### The demo watchlist entry

`BR27Z2024` is on the control-room watchlist, reasoned *"Reported stolen on the
vehicle registry — FIR 277/2026, Burrabazar PS"*. It was added so the
`REGISTRY: STOLEN` chip is actually visible to a judge: the chip only appears on
an alert whose registration is flagged, and of 1,418 registered vehicles only 17
are flagged, so the two sets did not overlap on their own.

Nothing was faked to make it work. The vehicle is genuinely `stolen` on the
registry (generated from the plate hash like every other record), it is a real
circulating simulator vehicle with 160+ sightings across 28 junctions, and the
alert is raised by `core/alerts.py` on ingest exactly as it is for every other
watchlist plate. The watchlist entry is the only thing added, and it says in its
own reason text where the claim comes from.

```bash
python -c "from core import db; db.remove_watch('BR27Z2024')"   # undo
```

To show the chain from the queue: **Alerts** → the critical `BR27Z2024` row →
**Track** → the registry card opens on the stolen banner with its findings and
a **Refer** button.

Verified: `python tests.py` → **81 passed, 0 failed** with this package present.

```bash
python -m registry.selftest          # 33 checks: isolation and the rules,
                                     # needs nothing running
python -m registry.integrationtest   # 40 checks against the LIVE service:
                                     # the four failure modes, and that the
                                     # cameras kept serving through each one
```

`selftest.py` tests the logic; `integrationtest.py` breaks things on purpose
and checks what actually comes back over HTTP. The properties that matter here
are about failure, and a unit test with a stubbed client will cheerfully report
that failure handling works.

---

## Removing it

It was built to be undone. To take it out completely:

```bash
rmdir /s /q registry
del registry.bat
del data\registry.db
```

**The dashboard keeps working after exactly that** — `registry_link.available()`
goes False and every panel renders nothing. It is the property `selftest.py`
executes rather than asserts, so it is true or the test fails.

`data/godseye.db` never had anything written to it by this package, so there is
no data to unwind. Two pre-existing files carry edits, and neither needs
reverting for the demo to run:

| File | Edit | If you also want it gone |
|---|---|---|
| `stop.bat` | two `killport` lines for 8506/8507 | delete them; leaving them just prints "nothing was listening" |
| `dashboard/app.py` | `import registry_link` + the panel calls | delete `dashboard/registry_link.py` too and remove the handful of `registry_link.` lines |

To keep the code but stop using it, simply do not start the console. Nothing
runs on a timer and nothing hooks into the platform's ingest path.

---

## Files

| File | What it does |
|---|---|
| `settings.py` | Every tunable, in its own `GODSEYE_REG_*` namespace |
| `catalogue.py` | The vocabulary: makes, models, RTOs, insurers, case types and their weights |
| `store.py` | Schema, deterministic generation, and `lookup()` — the seam |
| `seed.py` | Builds the registry from plates the cameras saw, plus invented ones |
| `sentinel.py` | The rules; `sweep()` for everything, `brief()` for one plate |
| `authority.py` | Routing and delivery — records the incident **before** notifying, so a failed referral is never a lost finding |
| `service.py` | The HTTP surface on 8507: the registry's `GET /api/v1/vehicles/{reg}`, and the simulated authority's `POST /api/v1/authority/incidents` |
| `live.py` | The chain: camera read → lookup → assessment → incident → authority. Reads the platform, is never called by it |
| `console.py` | The Streamlit console on 8506 |
| `../dashboard/registry_link.py` | The one adapter the control room reaches this package through — soft import, degrades to nothing |
| `selftest.py` | Isolation and behaviour checks, nothing running required |
| `integrationtest.py` | The four failure modes against the live service, with the cameras probed at every step |
