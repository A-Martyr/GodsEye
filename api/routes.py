"""REST + WebSocket surface of the GodsEye platform."""
from __future__ import annotations

import base64
import json
import random
import secrets
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import (APIRouter, File, HTTPException, Query, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

import config
from anpr import imageio
from anpr import plates as plate_lib
from api.state import state
from core import alerts as alert_rules
from core import analytics, convoy, db, enrol as enrol_mod, forecast, geofence
from core import privacy, reach, trajectory, tunnel, vehicle_search

router = APIRouter(prefix="/api")
MODELS = Path(__file__).resolve().parent.parent / "models"


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN is not valid JSON)."""
    if df is None or df.empty:
        return []
    return df.replace({np.nan: None}).to_dict(orient="records")


# --- platform -----------------------------------------------------------
@router.get("/health", summary="Service health and live counters")
def health():
    return state.status()


@router.get("/network", summary="Camera network as a graph")
def get_network():
    return state.net.as_dict()


@router.get("/cameras", summary="All cameras with their current load")
def get_cameras(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440)):
    return _records(analytics.camera_density(minutes))


@router.get("/cameras/{camera_id}", summary="One camera: detail and recent reads")
def get_camera(camera_id: str, limit: int = Query(25, ge=1, le=500)):
    cam = state.net.cameras.get(camera_id)
    if not cam:
        raise HTTPException(404, f"unknown camera {camera_id}")
    return {"camera": cam,
            "neighbours": state.net.neighbours(camera_id),
            "recent": db.recent_sightings(limit, camera_id=camera_id)}


@router.get("/sightings/recent", summary="Latest reads across the network")
def recent(limit: int = Query(50, ge=1, le=1000), camera_id: str | None = None):
    return db.recent_sightings(limit, camera_id=camera_id)


# --- trajectory ---------------------------------------------------------
@router.get("/plates/search", summary="Confusion-aware plate search")
def plate_search(q: str = Query(..., min_length=2), limit: int = Query(12, ge=1, le=50),
                 hours: float = Query(48.0, ge=0.1, le=720)):
    # Windowed from the newest sighting rather than wall clock - see
    # core.trajectory.lookback(). A seeded store's present drifts from the
    # clock, and a wall-clock window empties out once it has.
    since = trajectory.lookback(hours)
    return {"query": q, "matches": trajectory.search(q, limit=limit, since=since)}


@router.get("/plates/{plate}/trajectory", summary="Reconstructed city-wide path")
def plate_trajectory(plate: str, hours: float = Query(24.0, ge=0.1, le=720)):
    since = trajectory.lookback(hours)
    traj = trajectory.reconstruct(plate, since=since)
    if not traj.sightings:
        near = trajectory.search(plate, limit=5, since=since)
        raise HTTPException(404, {"message": f"no sightings for {plate} in the last {hours}h",
                                  "did_you_mean": [m["plate"] for m in near]})
    return traj.as_dict()


@router.get("/plates/{plate}/clone-check", summary="Is this registration on two vehicles?")
def plate_clone_check(plate: str, hours: float = Query(24.0, gt=0)):
    """Query-time clone verdict, with the conflicts it ruled out and why.

    Distinct from the ingest-time clone alert in core/alerts.py, which only fires
    as a sighting arrives, is rate-limited, and so can miss a clone that is
    already sitting in the history.
    """
    from core import clones

    rep = clones.check(plate.upper().replace(" ", ""), since=trajectory.lookback(hours))
    return rep.as_dict()


@router.get("/plates/{plate}/registry", summary="Mock vehicle-registry record for a plate")
def plate_registry(plate: str):
    """Synthetic registry lookup - NOT VAHAN. See core/registry.py."""
    from core import registry

    return registry.lookup(plate.upper().replace(" ", "")).as_dict()


@router.get("/plates/resolve-partial",
           summary="Masked-plate resolution against sightings and the mock registry")
def plate_resolve_partial(mask: str = Query(..., min_length=7, max_length=11),
                          vehicle_class: str | None = None):
    """Given a masked pattern like WB01??9492, a shortlist to chase - not a verdict.
    See core/registry.py::resolve_partial()."""
    from core import registry

    return registry.resolve_partial(mask, vehicle_class=vehicle_class)


@router.get("/plates/{plate}/co-travellers", summary="Vehicles repeatedly seen alongside")
def plate_co_travellers(plate: str, window_s: float = Query(120.0, ge=10, le=900)):
    return {"plate": plate.upper(), "co_travellers": trajectory.co_travellers(plate, window_s)}


@router.get("/plates/{plate}/forecast", summary="Where this vehicle is likely to go next")
def plate_forecast(plate: str, top: int = Query(5, ge=1, le=12)):
    """Ranked next junctions with an ETA on each.

    The response carries `confident` and `observations` alongside the ranking,
    and they are not decoration: a ranked list looks like knowledge whatever is
    behind it, and a caller acting on this needs to know whether it came from
    the vehicle's own habit or from what traffic in general does there.
    """
    return forecast.predict(plate, top=top, net=state.net)


@router.get("/plates/{plate}/pattern", summary="When and where this vehicle is usually seen")
def plate_pattern(plate: str):
    """Hour-of-day and per-junction profile, plus the junction/hour pair the
    vehicle has been seen at most often. `reliable` is false when there is too
    little history for the word "usually" to be honest."""
    return forecast.pattern_of_life(plate, net=state.net)


# --- finding a vehicle without its plate --------------------------------
@router.get("/search/vehicles", summary="Find sightings by description, not plate")
def search_vehicles(q: str = Query(..., min_length=2), limit: int = Query(40, ge=1, le=200)):
    """Structured search over the attributes a witness actually remembers.

    The response carries `readback`, `ignored` and `unrecognised` next to the
    matches, and they are the point rather than decoration: someone who asks
    for a red truck and is handed every truck will believe the reds were
    filtered. The parser can enumerate exactly what it did and did not
    understand, which is the one thing a language model in this position
    cannot do.
    """
    return vehicle_search.search(q, limit=limit)


# --- geofencing ---------------------------------------------------------
@router.get("/zones", summary="Defined geographic zones")
def list_zones():
    return [z.as_dict() if hasattr(z, "as_dict") else z for z in geofence.load_zones()]


@router.get("/zones/{zone_id}/visits", summary="Vehicles that entered, left, or are still inside")
def zone_visits(zone_id: str, hours: float = Query(6.0, ge=0.1, le=720)):
    """A zone only sees what its cameras see. A vehicle can leave through a
    junction nobody watches, so "still inside" is an inference and is reported
    as one - see the observability figures on the zone itself."""
    try:
        visits = geofence.visits(zone_id, hours=hours, net=state.net)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return [v.as_dict() if hasattr(v, "as_dict") else v for v in visits]


# --- convoy -------------------------------------------------------------
@router.get("/plates/{plate}/convoy", summary="Vehicles travelling WITH this one")
def plate_convoy(plate: str, hours: float = Query(24.0, ge=0.1, le=720)):
    """Scored against chance, which is what separates this from
    /co-travellers above. On a rush-hour corridor the vehicles seen alongside
    a target most often are simply the ones on the busiest route; counting
    co-occurrences ranks those top. Here each pair is compared against how
    many co-occurrences that pair's own camera traffic would produce by
    chance, so a shared commute reads as a shared commute.
    """
    return convoy.partners(plate, hours=hours, net=state.net)


@router.get("/convoys", summary="Groups moving together across the network")
def convoys(hours: float = Query(2.0, ge=0.1, le=72)):
    return convoy.groups(hours=hours, net=state.net)


# --- privacy ------------------------------------------------------------
@router.get("/privacy/posture", summary="What this system is holding, and under what limits")
def privacy_posture():
    """One call an operator can put on screen to show what is retained, how
    long for, and who has been looking. Deliberately not a compliance claim
    against any named standard - it reports what the code does."""
    return privacy.posture()


@router.get("/privacy/retention", summary="What is past the retention horizon")
def privacy_retention():
    """Dry run only. Nothing here deletes; the purge is a separate, explicit
    call that is not exposed over HTTP, because irreversible deletion should
    not be one mistyped URL away."""
    return privacy.retention_preview()


@router.get("/privacy/access", summary="Who looked up which plate, and why")
def privacy_access(hours: float = Query(168.0, ge=1, le=8760)):
    # The route takes hours to match every other window on this API;
    # access_summary() takes days, because an audit review is asked in days.
    # Convert here rather than changing either side - the default 168 h is the
    # 7 days access_summary() would have used anyway.
    return privacy.access_summary(days=hours / 24.0)


# --- analytics ----------------------------------------------------------
@router.get("/analytics/summary", summary="City headline metrics")
def analytics_summary(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440)):
    return analytics.city_summary(minutes)


@router.get("/analytics/density", summary="Per-camera traffic density")
def analytics_density(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440)):
    return _records(analytics.camera_density(minutes))


@router.get("/analytics/links", summary="Per-link flow and speed")
def analytics_links(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440)):
    return _records(analytics.link_flows(minutes))


@router.get("/analytics/bottlenecks", summary="Worst congestion by delay caused")
def analytics_bottlenecks(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440),
                          top: int = Query(10, ge=1, le=50)):
    return _records(analytics.bottlenecks(minutes, top=top))


@router.get("/analytics/od", summary="Origin-destination matrix")
def analytics_od(minutes: float = Query(120.0, ge=1, le=1440), gateways_only: bool = False,
                 top: int = Query(40, ge=1, le=500)):
    return _records(analytics.od_matrix(minutes, gateways_only=gateways_only).head(top))


@router.get("/analytics/heatmap", summary="Weighted points for the movement heatmap")
def analytics_heatmap(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440)):
    return _records(analytics.heatmap(minutes))


@router.get("/analytics/trend", summary="Flow and speed over time")
def analytics_trend(hours: float = Query(3.0, ge=0.2, le=72),
                    bucket_min: float = Query(5.0, ge=1, le=60)):
    flow = analytics.flow_trend(hours, bucket_min)
    speed = analytics.speed_trend(hours, max(bucket_min, 10.0))
    for df in (flow, speed):
        if not df.empty:
            df["bucket"] = df["bucket"].astype(str)
    return {"flow": _records(flow), "speed": _records(speed)}


@router.get("/analytics/sectors", summary="Load by city sector")
def analytics_sectors(minutes: float = Query(config.DEFAULT_WINDOW_MIN, ge=1, le=1440)):
    return _records(analytics.sector_load(minutes))


# --- alerts and watchlist ----------------------------------------------
@router.get("/inferences", summary="Registrations inferred, not read")
def list_inferences(plate: str | None = None, hours: float = Query(6.0, gt=0),
                    limit: int = Query(100, ge=1, le=1000)):
    """Plates the repair pass recovered from neighbouring cameras.

    Deliberately a separate endpoint from `/sightings/recent`: these are
    hypotheses with an evidence chain, not observations, and an operator acting
    on one should know which they are looking at.
    """
    from core import repair as repair_mod

    since = time.time() - hours * 3600.0
    return {"inferences": db.inferences(plate=plate.upper().replace(" ", "") if plate else None,
                                        since=since, limit=limit),
            "summary": repair_mod.summary(since=since)}


@router.get("/unresolved/recent", summary="Captures the engine refused to store")
def unresolved_recent(limit: int = Query(20, ge=1, le=200)):
    """The reads below the confidence floor, newest first: what the engine
    declined to guess. Evidence blobs and crops stay server-side."""
    return db.rows("SELECT id, ts, camera_id, best_text, confidence, frames, condition"
                   " FROM unresolved ORDER BY ts DESC LIMIT ?", (limit,))


@router.post("/repair/run", summary="Reconcile refused captures against the network")
def run_repair(hours: float = Query(6.0, gt=0), limit: int = Query(200, ge=1, le=2000)):
    """Run the repair pass over the captures the engine refused.

    Trails the present by `config.REPAIR_LAG_S`: half the evidence is the
    downstream camera, which has not seen the vehicle yet at the moment a
    capture fails.
    """
    from core import repair as repair_mod

    now = time.time()
    made = repair_mod.repair(since=now - hours * 3600.0,
                             until=now - config.REPAIR_LAG_S, limit=limit)
    return {"inferred": len(made), "inferences": made[:50],
            "summary": repair_mod.summary(since=now - hours * 3600.0)}


@router.get("/alerts", summary="Alert queue")
def get_alerts(limit: int = Query(100, ge=1, le=1000), open_only: bool = False,
               hours: float = Query(24.0, ge=0.1, le=720)):
    return db.alerts(limit=limit, include_acked=not open_only, since=time.time() - hours * 3600)


@router.get("/alerts/summary", summary="Alert counts by kind and severity")
def alerts_summary(hours: float = Query(6.0, ge=0.1, le=720)):
    return alert_rules.summary(hours)


@router.post("/alerts/{alert_id}/ack", summary="Acknowledge an alert")
def ack(alert_id: int):
    db.ack_alert(alert_id)
    return {"acked": alert_id}


class WatchRequest(BaseModel):
    plate: str = Field(..., examples=["KA05MJ1234"])
    reason: str = Field("flagged by control room", max_length=200)
    severity: str = Field("high", pattern="^(low|medium|high|critical)$")


@router.get("/watchlist", summary="Watchlisted plates")
def get_watchlist():
    return db.watchlist()


@router.post("/watchlist", summary="Add a plate to the watchlist")
def add_watch(req: WatchRequest):
    plate = req.plate.upper().replace(" ", "")
    db.add_watch(plate, req.reason, req.severity)
    hits = db.sightings_for_plate(plate)
    return {"plate": plate, "reason": req.reason, "severity": req.severity,
            "historic_sightings": len(hits),
            "last_seen": hits[-1]["ts"] if hits else None}


@router.delete("/watchlist/{plate}", summary="Remove a plate from the watchlist")
def delete_watch(plate: str):
    removed = db.remove_watch(plate)
    if not removed:
        raise HTTPException(404, f"{plate} is not on the watchlist")
    return {"removed": plate.upper()}


# --- ANPR ---------------------------------------------------------------
def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _history_block(plate: str, hours: float = 24.0) -> dict:
    """The rest of the sentence after a successful read.

    A read returns a string; what an operator needs next is where that vehicle
    has been. Attached to the read itself so the join does not need a second
    round trip, and kept compact - the full path is on /plates/{plate}/trajectory.
    """
    if not plate:
        return {"known": False}
    traj = trajectory.reconstruct(plate, since=trajectory.lookback(hours))
    if not traj.sightings:
        return {"known": False, "plate": plate}
    from core import clones

    return {
        "known": True, "plate": plate, "summary": traj.summary,
        "clone": clones.check(plate, since=trajectory.lookback(hours)).as_dict(),
        "legs": [{"from": l.from_name, "to": l.to_name, "departed_ts": l.departed_ts,
                  "arrived_ts": l.arrived_ts, "minutes": round(l.minutes, 1),
                  "road_km": round(l.road_km, 2),
                  "implied_kmph": round(l.implied_kmph, 1),
                  "direction": l.direction, "plausible": l.plausible, "note": l.note}
                 for l in traj.legs],
        "sightings": [{"ts": x["ts"], "camera_id": x["camera_id"],
                       "camera_name": x.get("camera_name"), "sector": x.get("sector"),
                       "confidence": x["confidence"]} for x in traj.sightings],
    }


@router.post("/anpr/read", summary="Read a plate from an uploaded image")
async def anpr_read(file: UploadFile = File(...), history: bool = True):
    raw = await file.read()
    arr = imageio.load_bytes(raw)
    if arr is None:
        raise HTTPException(400, "could not decode that image")
    t0 = time.time()
    read = state.get_engine().read_frame(arr)
    out = {"filename": file.filename, "ms": round((time.time() - t0) * 1000, 1),
           "image_size": [int(arr.shape[1]), int(arr.shape[0])], **read.as_dict()}
    if history and read.text and read.plate_found:
        out["history"] = _history_block(read.text)
    return out


@router.get("/enrol/suggest", summary="A junction and time that would make an enrolment valid")
def enrol_suggest(plate: str, camera_id: str):
    """What the dashboard fills the enrolment form in with.

    Separate from the write so the operator can see the proposed time - and why
    it is that time - before committing to it.
    """
    net = state.net
    if camera_id not in net.cameras:
        raise HTTPException(404, f"{camera_id} is not a camera on this network")
    ts, why = enrol_mod.suggest_ts(plate, camera_id, net=net)
    verdict = enrol_mod.check(plate, camera_id, ts, net=net)
    return {"plate": plate.upper().replace(" ", ""), "camera_id": camera_id,
            "camera_name": net.name(camera_id), "ts": ts, "why": why,
            "plausible": verdict.ok, "reason": verdict.reason,
            "network_now": enrol_mod.network_now()}


@router.post("/enrol", summary="Register a vehicle at a junction from a photograph")
async def enrol_vehicle(
    file: UploadFile = File(...),
    camera_id: str = Query(..., description="Junction the photograph was taken at"),
    ts: float | None = Query(None, description="Unix seconds; omitted means derive a "
                                               "physically plausible time from the road "
                                               "distance to this vehicle's last sighting"),
    plate: str | None = Query(None, description="Correct a misread. Bounded by OCR "
                                                "confusion distance - it may fix what the "
                                                "engine got wrong, not name another vehicle."),
    operator: str | None = Query(None),
    note: str | None = Query(None),
):
    """Read the photograph here, then store it as a sighting at `camera_id`.

    The image is re-read server-side rather than trusting a plate the caller
    supplies, so the stored confidence is always a number this engine actually
    produced. `plate` may only *correct* that read, and how far it may move it
    is bounded in core/enrol.validate_correction().
    """
    raw = await file.read()
    arr = imageio.load_bytes(raw)
    if arr is None:
        raise HTTPException(400, "could not decode that image")
    read = state.get_engine().read_frame(arr)
    try:
        out = enrol_mod.enrol(read, camera_id, ts=ts, plate=plate,
                              operator=operator, note=note, net=state.net)
    except ValueError as exc:
        # A refusal here is the platform declining to fabricate, not a fault -
        # the message says which rule was hit and what to do instead.
        raise HTTPException(422, str(exc))
    out["history"] = _history_block(out["plate"])
    return out


@router.get("/enrol", summary="Operator-attested sightings")
def enrol_list(plate: str | None = None, limit: int = Query(200, ge=1, le=1000)):
    return {"enrolments": enrol_mod.enrolled(plate=plate, limit=limit)}


class DemoRequest(BaseModel):
    plate: str | None = None
    condition: str = Field("mixed", examples=plate_lib.CONDITIONS)
    severity: float | None = Field(None, ge=0.0, le=1.0)
    history: bool = Field(True, description="also return where this vehicle has been")
    frames: int = Field(1, ge=1, le=10,
                        description="Frames the node contributes for this vehicle. "
                                    "1 reads a single photograph; more reads a burst "
                                    "and votes the frames by CTC score, which is what "
                                    "a real ANPR node delivers.")


@router.post("/anpr/demo", summary="Synthesise a plate under a condition and read it back")
def anpr_demo(req: DemoRequest):
    if req.condition not in plate_lib.CONDITIONS:
        raise HTTPException(400, f"condition must be one of {plate_lib.CONDITIONS}")
    rng = random.Random()
    truth = req.plate.upper().replace(" ", "") if req.plate else None
    caps = [plate_lib.capture(truth, req.condition, rng, req.severity)]
    # keep every frame on the same plate and the same layout as the first
    for _ in range(req.frames - 1):
        caps.append(plate_lib.capture(caps[0].text, req.condition, rng, req.severity,
                                      two_row=caps[0].two_row))
    cap = caps[0]
    eng = state.get_engine()
    t0 = time.time()
    if req.frames == 1:
        read, dbg = eng.read_detailed(cap.image)
        per_frame = [read]
    else:
        per_frame = [eng.read(c.image) for c in caps]
        dbg = eng._last_debug
        read = eng.fuse_reads(list(per_frame))
    ink = dbg.get("ink")
    return {"truth": cap.text, "condition": cap.condition, "frames": req.frames,
            "ms": round((time.time() - t0) * 1000, 1),
            "correct": read.text == cap.text,
            "image_png_b64": _png_b64(cap.image),
            "ink_png_b64": _png_b64(ink) if ink is not None else "",
            "per_frame": [{"text": r.text, "confidence": round(r.confidence, 4),
                           "correct": r.text == cap.text} for r in per_frame],
            "history": _history_block(read.text) if req.history else {"known": False},
            **read.as_dict()}


@router.get("/anpr/benchmark", summary="Measured accuracy per capture condition")
def anpr_benchmark(samples: int = Query(0, ge=0, le=200),
                   refresh: bool = False):
    """Cached benchmark. `samples=0` returns the cache (or the last saved run);
    pass a sample count to measure again — that takes a few seconds per condition."""
    from anpr import benchmark as bench

    if samples and (refresh or state.benchmark_cache is None
                    or state.benchmark_cache.get("samples_per_condition") != samples):
        results, overall, elapsed = bench.run(samples, verbose=False)
        state.benchmark_cache = {
            "samples_per_condition": samples, "elapsed_s": round(elapsed, 1),
            "conditions": [r.__dict__ for r in results], "overall": overall.__dict__,
            "measured_at": time.time(),
        }
    if state.benchmark_cache is None:
        # The last saved run (python -m anpr.benchmark ... --json models/benchmark.json),
        # so the control room can show measured accuracy without re-measuring.
        path = MODELS / "benchmark.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return {"source": "saved", **json.load(fh)}
        raise HTTPException(409, "no benchmark has been run yet - call with ?samples=40")
    return state.benchmark_cache


@router.get("/anpr/real", summary="Measured accuracy on real photographs, per held-out split")
def anpr_real():
    """Exact-plate accuracy of each backend on the 532-crop real corpus, held
    out by vehicle (python -m anpr.compare_backends --real --seeds 8)."""
    path = MODELS / "backend_comparison_real.json"
    if not path.exists():
        raise HTTPException(404, "no real-photograph comparison saved")
    with open(path, encoding="utf-8") as fh:
        splits = json.load(fh)
    backends = sorted({b for s in splits.values() for b in s})
    summary = {}
    for b in backends:
        vals = [s[b] for s in splits.values() if b in s]
        summary[b] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
                      "splits": len(vals)}
    return {"corpus": "532 real plate crops, 118 vehicles, held out by vehicle",
            "splits": splits, "summary": summary}


@router.get("/anpr/model", summary="Glyph model card")
def anpr_model():
    eng = state.get_engine()
    m = eng.model
    from anpr import ocr as ocr_mod
    return {"backend": "crnn" if eng.crnn is not None else "classical",
            "classes": len(m.classes), "trained_on_glyphs": m.trained_on,
            "glyph_val_accuracy": round(m.val_accuracy, 4), "training_stages": m.stages,
            "classifier": type(m.clf).__name__,
            "grammar_patterns": ocr_mod.PATTERNS,
            "confidence_floor": (config.MIN_PLATE_CONFIDENCE_CRNN
                                 if eng.crnn is not None else config.MIN_PLATE_CONFIDENCE)}


# --- simulator control --------------------------------------------------
@router.post("/sim/live/{onoff}", summary="Pause or resume the live feed")
def sim_live(onoff: str):
    if onoff not in ("on", "off"):
        raise HTTPException(400, "use /sim/live/on or /sim/live/off")
    state.live = onoff == "on"
    return {"live": state.live}


@router.post("/sim/inject/{kind}", summary="Inject a scripted incident")
def sim_inject(kind: str):
    sim = state.get_simulator()
    if kind == "clone":
        plate = sim.inject_clone()
        return {"injected": "clone", "plate": plate}
    if kind == "loiterer":
        plate = sim.inject_loiterer()
        return {"injected": "loiterer", "plate": plate}
    if kind == "watchlist":
        plate = sim.sample_plates(1)[0]
        db.add_watch(plate, "injected demo target", "critical")
        return {"injected": "watchlist", "plate": plate}
    raise HTTPException(400, "kind must be clone, loiterer or watchlist")


# --- judge console ------------------------------------------------------
# A surface for someone outside the team to inject an incident at a moment of
# their own choosing. The point is that the control room is not told: the
# platform has to find it with the same rule that watches every capture, which
# is a claim worth demonstrating rather than asserting.
#
# Deliberately unauthenticated - a judge holding a phone should not be typing a
# password - and therefore gated on config.JUDGE_CONSOLE_ENABLED, which should
# be off outside a demonstration.
def _judge_guard(request: Request | None = None) -> None:
    """Refuse unless this caller is entitled to the console.

    Two different rules, because two different situations. A request that
    arrives on a LAN address is someone in the room and passes as it always
    has - the console is unauthenticated on purpose and nobody types anything.
    A request that arrives on a public hostname came through the tunnel, could
    be anyone on the internet, and has to carry the token from the link.

    404 rather than 403 on a bad token: a scanner sweeping trycloudflare
    hostnames learns nothing from a page that simply is not there.
    """
    if not config.JUDGE_CONSOLE_ENABLED:
        raise HTTPException(404, "judge console is disabled (GODSEYE_JUDGE=0)")
    if request is None or not config.JUDGE_TOKEN_REQUIRED_PUBLIC:
        return
    if not reach.is_public_host(request.headers.get("host", "")):
        return
    supplied = (request.query_params.get("k")
                or request.path_params.get("token")
                or request.headers.get("x-godseye-token", ""))
    if not secrets.compare_digest(str(supplied or ""), reach.token()):
        raise HTTPException(404, "not found")


@router.post("/judge/inject", summary="Inject an incident as an outside observer")
def judge_inject(request: Request,
                 kind: str = Query("clone", description="clone | loiterer | watchlist"),
                 plate: str | None = Query(None, description="Pick the vehicle, or leave "
                                                             "blank for one at random")):
    """Inject a real incident and report exactly what was written.

    Nothing is faked. For a clone this writes two ordinary sightings at two
    junctions the caller does not choose, and both go through
    core.alerts.evaluate() - the same path a camera capture takes. `detected`
    in the response is the rule's own answer, so if the detector were broken
    this would say so rather than quietly claiming success.
    """
    _judge_guard(request)
    sim = state.get_simulator()
    if kind == "clone":
        out = sim.inject_clone_detail(plate=plate, gap_minutes=config.JUDGE_CLONE_GAP_S / 60.0)
        _mark_judge("clone", out.get("plate"))
        return {"injected": "clone", **out}
    if kind == "loiterer":
        p = sim.inject_loiterer()
        _mark_judge("loiterer", p)
        return {"injected": "loiterer", "plate": p, "detected": True,
                "from_name": "a single junction", "to_name": "repeatedly"}
    if kind == "watchlist":
        p = (plate or sim.sample_plates(1)[0]).upper().replace(" ", "")
        db.add_watch(p, "flagged by an observer during the demonstration", "critical")
        _mark_judge("watchlist", p)
        return {"injected": "watchlist", "plate": p, "detected": True}
    raise HTTPException(400, "kind must be clone, loiterer or watchlist")


def _mark_judge(kind: str, plate: str | None) -> None:
    """Record that a judge released this, so the control room can be audible for it.

    The control room deliberately does NOT sound for ambient simulated alerts -
    the simulator raises them continuously and an alarm that is always going is
    an alarm nobody hears. The two events worth being audible for are a judge
    injection and a live camera read, and neither is distinguishable from
    ordinary traffic once it has become a row in `sightings`.

    phonecam carries its own provenance (it keeps the alert ids its reads
    caused). This is the equivalent for the judge console, written to the
    existing `meta` table so no schema changes and nothing else is affected.
    """
    if not plate:
        return
    try:
        rows = db.rows("SELECT value FROM meta WHERE key = 'judge_injections'")
        history = json.loads(rows[0]["value"]) if rows else []
        if not isinstance(history, list):
            history = []
    except (ValueError, KeyError, IndexError):
        history = []
    history.append({"ts": time.time(), "kind": kind,
                    "plate": plate.upper().replace(" ", "")})
    conn = db.connect()
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('judge_injections', ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (json.dumps(history[-25:]),))
    conn.commit()


JUDGE_PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GodsEye - Judge Console</title><style>
:root{--bg:#0f1216;--card:#181d24;--line:#2a323d;--ink:#eef2f7;--dim:#93a1b3;--red:#e11d48;--ok:#22c55e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:22px}
.wrap{max-width:560px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:.92rem;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:14px}
button{width:100%;padding:16px;font-size:1.03rem;font-weight:600;border:0;border-radius:11px;
color:#fff;background:var(--red);cursor:pointer;margin-bottom:10px}
button.alt{background:#334155}
button:disabled{opacity:.55;cursor:progress}
.k{color:var(--dim);font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px}
.v{font-size:1.06rem;font-weight:600;margin-bottom:12px}
.plate{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.09em;
background:#0b0e12;border:1px solid var(--line);border-radius:8px;padding:7px 11px;display:inline-block}
.ok{color:var(--ok)} .muted{color:var(--dim);font-size:.87rem}
</style></head><body><div class="wrap">
<h1>GodsEye - Judge Console</h1>
<div class="sub">Press a button whenever you like. The control room is not told what you
did or when. Their platform has to find it on its own.</div>
<div class="card">
  <button id="clone">Release a cloned number plate</button>
  <button id="loiterer" class="alt">Release a loitering vehicle</button>
  <button id="watchlist" class="alt">Flag a vehicle as wanted</button>
  <div class="muted">A cloned plate is the same registration read at two junctions no
  single vehicle could have covered in the time between them.</div>
</div>
<div id="out"></div>
<script>
const out = document.getElementById('out');
const TOKEN = '__GODSEYE_TOKEN__';
async function fire(kind, label){
  for (const b of document.querySelectorAll('button')) b.disabled = true;
  out.innerHTML = '<div class="card">Releasing ' + label + ' into the network...</div>';
  try {
    const r = await fetch('/api/judge/inject?kind=' + kind + '&k=' + TOKEN,
                          {method:'POST'});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.status);
    let h = '<div class="card"><div class="k">Registration released</div>' +
            '<div class="v"><span class="plate">' + d.plate + '</span></div>';
    if (kind === 'clone') {
      h += '<div class="k">Seen at</div><div class="v">' + d.from_name +
           '<br>' + d.to_name + '</div>' +
           '<div class="k">Which would mean</div><div class="v">' + d.road_km +
           ' km in ' + d.gap_minutes + ' min = ' + d.implied_kmph + ' km/h</div>';
    }
    h += '<div class="k">Platform response</div><div class="v ' + (d.detected ? 'ok' : '') +
         '">' + (d.detected ? 'Flagged automatically' : 'No alert raised') + '</div>' +
         '<div class="muted">Now watch the control room.</div></div>';
    out.innerHTML = h;
  } catch (e) {
    out.innerHTML = '<div class="card">Could not reach the platform: ' + e.message + '</div>';
  }
  for (const b of document.querySelectorAll('button')) b.disabled = false;
}
document.getElementById('clone').onclick = () => fire('clone', 'a cloned plate');
document.getElementById('loiterer').onclick = () => fire('loiterer', 'a loitering vehicle');
document.getElementById('watchlist').onclick = () => fire('watchlist', 'a wanted vehicle');
</script></div></body></html>"""


def judge_url(host: str | None = None, port: int | None = None) -> str:
    """The link to hand someone, on the address they can actually reach.

    Not localhost: on their phone, localhost is their phone. Prefers a running
    tunnel, because that is the only address that works for a person who is
    not on this network - which, once the link is being sent to someone rather
    than shown to someone, is the normal case. Falls back to the LAN address
    for a judge standing at the table.
    """
    if not host:
        public = tunnel.current_url()
        if public:
            return f"{public}/j/{reach.token()}"
    port = port or int(config.API_BASE.rsplit(":", 1)[-1].strip("/") or 8000)
    if not host:
        found = reach.local_addresses()
        host = found[0][0] if found else "127.0.0.1"
    return f"http://{host}:{port}/j"


@router.get("/judge/link", summary="The judge console link, and a QR for it")
def judge_link(request: Request, host: str | None = None):
    _judge_guard(request)
    return {"url": judge_url(host), "qr": "/api/judge/qr"}


@router.get("/judge/qr", include_in_schema=False)
def judge_qr(request: Request, host: str | None = None):
    """The same link as a QR code, so nobody has to type an IP address.

    Scanning is the whole point: an address like 192.168.29.204:8000 is four
    numbers and a port to get right on a phone keyboard, in a noisy room,
    under time pressure.
    """
    _judge_guard(request)
    import io as _io

    try:
        import qrcode
    except ImportError:
        raise HTTPException(501, "qrcode is not installed - use the plain URL")
    q = qrcode.QRCode(box_size=10, border=2,
                      error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(judge_url(host))
    q.make(fit=True)
    buf = _io.BytesIO()
    q.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/judge", include_in_schema=False)
def judge_console(request: Request):
    # Annotated as a plain Request, not Request | None: this is a route as well
    # as a function api/main.py calls directly, and FastAPI cannot build a
    # request field out of an optional - it tries to read it as a Pydantic type
    # and refuses at import, taking the whole API down with it.
    _judge_guard(request)
    return HTMLResponse(render_judge_page())


def render_judge_page() -> str:
    """The console with this run's token baked into its script.

    The page is only ever served to a caller that already cleared the guard,
    so handing it the token costs nothing and means the buttons keep working
    over the tunnel without the judge appending anything to a URL. On the LAN
    the token rides along and is ignored.
    """
    return JUDGE_PAGE.replace("__GODSEYE_TOKEN__", reach.token())


# --- live feed ----------------------------------------------------------
@router.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    q = state.hub.subscribe()
    try:
        await ws.send_json({"type": "hello", "data": state.status()})
        while True:
            msg = await q.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        state.hub.unsubscribe(q)
