# From one prototype to a city

The SIH 2026 judges expected scalability problems. This is the honest answer:
what the prototype is, what was measured, what breaks first at scale, and the
path past it. Numbers marked *estimate* have not been measured.

## What the prototype is

One machine runs everything: the reader, a single SQLite database (WAL mode),
the FastAPI service and the web control room it serves, over 35 real Kolkata
junctions with simulated camera feeds plus live webcam and phone-camera inputs.

## Measured

| | Result | Source |
|---|---|---|
| Warm read of one real street photo, GTX 1650 (4 GB) | **118 ms** (easy scene) to **578 ms** (10 candidate regions) | `PERFORMANCE.md` |
| Typical warm read | ~108 ms | `RUNBOOK.md` |
| First read of a session | 0.19 s (was 8.0 s) | `PERFORMANCE.md` |
| Dashcam corpus, multi-scale escalation on | 0.21 s per read | `FIELD_TEST.md` |
| Data per stored sighting | one row: plate, camera, time, confidence, class | `core/db.py` |

## What breaks first, in order

1. **Reading compute.** At ~0.1–0.6 s per frame and 5-frame bursts, one
   GTX 1650 keeps up with roughly one or two busy junction approaches
   (*estimate*: ~1,800 vehicles/hour per approach at peak is 0.5 vehicles/s).
2. **One SQLite writer.** Fine for 35 junctions; a city's thousands of cameras
   need a server database.
3. **One process.** The control room, API and reader share one machine and one GPU
   (`RUNBOOK.md` documents the single-CUDA-context limit).

## The path past it

1. **Read at the edge, send events, not video.** Each junction gets a small
   GPU box that runs the localiser and recogniser and sends only sightings
   (plate, camera, time, confidence, crop reference) upstream. A sighting is a
   few hundred bytes, so a 4G link is enough and raw video never leaves the
   junction, which also helps privacy.
2. **Use the cameras the city already owns.** The reader is built for bad,
   low-resolution and night footage, so it sits behind existing CCTV instead of
   needing dedicated ANPR cameras.
3. **A server database, partitioned by time.** PostgreSQL with time
   partitioning, indexed on (plate, time) and (camera, time), which are the two
   queries trajectory search runs. Old partitions are dropped by the existing
   retention rules (`core/privacy.py`).
4. **A message queue between edge and centre.** Edge boxes publish sightings;
   the trajectory engine, analytics and alert rules subscribe independently, so
   each scales on its own.
5. **Shard by zone.** Trajectory search only joins sightings in neighbouring
   cameras within a time window, so the city splits into zones with overlap at
   the borders.

## Rough cost (*estimate*, to be checked against quotes)

| Item | Per junction |
|---|---|
| Edge box (Jetson-class or small GPU PC) | ₹45,000–60,000 one-time |
| 4G data for events only | ₹300–500 per month |
| Central server share | small; events, not video |

The comparison point is dedicated ANPR camera installations, which buy new
cameras and poles per junction. Reusing existing cameras removes most of that
capital cost.

## Not yet measured

- Throughput on a Jetson-class edge device.
- Database load at more than 35 junctions.
- Accuracy on Kolkata's own CCTV (the real corpus is mostly Maharashtra plates).
