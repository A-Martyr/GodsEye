"""REST + WebSocket surface of the GodsEye platform."""
from __future__ import annotations

import base64
import io
import random
import time

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import config
from anpr import plates as plate_lib
from api.state import state
from core import alerts as alert_rules
from core import analytics, db, trajectory

router = APIRouter(prefix="/api")


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
    since = time.time() - hours * 3600
    return {"query": q, "matches": trajectory.search(q, limit=limit, since=since)}


@router.get("/plates/{plate}/trajectory", summary="Reconstructed city-wide path")
def plate_trajectory(plate: str, hours: float = Query(24.0, ge=0.1, le=720)):
    since = time.time() - hours * 3600
    traj = trajectory.reconstruct(plate, since=since)
    if not traj.sightings:
        near = trajectory.search(plate, limit=5, since=since)
        raise HTTPException(404, {"message": f"no sightings for {plate} in the last {hours}h",
                                  "did_you_mean": [m["plate"] for m in near]})
    return traj.as_dict()


@router.get("/plates/{plate}/co-travellers", summary="Vehicles repeatedly seen alongside")
def plate_co_travellers(plate: str, window_s: float = Query(120.0, ge=10, le=900)):
    return {"plate": plate.upper(), "co_travellers": trajectory.co_travellers(plate, window_s)}


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


@router.post("/anpr/read", summary="Read a plate from an uploaded image")
async def anpr_read(file: UploadFile = File(...)):
    raw = await file.read()
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "could not decode that image")
    t0 = time.time()
    read = state.get_engine().read_frame(arr)
    return {"filename": file.filename, "ms": round((time.time() - t0) * 1000, 1),
            **read.as_dict()}


class DemoRequest(BaseModel):
    plate: str | None = None
    condition: str = Field("mixed", examples=plate_lib.CONDITIONS)
    severity: float | None = Field(None, ge=0.0, le=1.0)


@router.post("/anpr/demo", summary="Synthesise a plate under a condition and read it back")
def anpr_demo(req: DemoRequest):
    if req.condition not in plate_lib.CONDITIONS:
        raise HTTPException(400, f"condition must be one of {plate_lib.CONDITIONS}")
    rng = random.Random()
    cap = plate_lib.capture(req.plate.upper().replace(" ", "") if req.plate else None,
                            req.condition, rng, req.severity)
    t0 = time.time()
    read, dbg = state.get_engine().read_detailed(cap.image)
    ink = dbg.get("ink")
    return {"truth": cap.text, "condition": cap.condition,
            "ms": round((time.time() - t0) * 1000, 1),
            "correct": read.text == cap.text,
            "image_png_b64": _png_b64(cap.image),
            "ink_png_b64": _png_b64(ink) if ink is not None else "",
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
        raise HTTPException(409, "no benchmark has been run yet - call with ?samples=40")
    return state.benchmark_cache


@router.get("/anpr/model", summary="Glyph model card")
def anpr_model():
    m = state.get_engine().model
    from anpr import ocr as ocr_mod
    return {"classes": len(m.classes), "trained_on_glyphs": m.trained_on,
            "glyph_val_accuracy": round(m.val_accuracy, 4), "training_stages": m.stages,
            "classifier": type(m.clf).__name__,
            "grammar_patterns": ocr_mod.PATTERNS,
            "confidence_floor": config.MIN_PLATE_CONFIDENCE}


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
