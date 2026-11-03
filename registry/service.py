"""The registry's HTTP surface, and the authority that receives referrals.

    python -m registry.service        ->  http://127.0.0.1:8507

Two systems in the story, two endpoint groups here:

    GET  /api/v1/health
    GET  /api/v1/vehicles/{registration_number}     the registry
    POST /api/v1/authority/incidents                the authority

They share a process because this is a demonstration and two processes would
buy nothing. They do not share meaning: the registry answers questions about
vehicles, the authority receives referrals about incidents, and neither knows
how the other is implemented.

NOT A GOVERNMENT SYSTEM. Every record is invented by registry/store.py from a
hash of the plate. Nothing here connects to VAHAN, no VAHAN endpoint is called
or imitated, and no real authority is contacted. The /api/v1/health response
says so in its own payload so it cannot be mistaken even in isolation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Response                 # noqa: E402
from pydantic import BaseModel, Field                                # noqa: E402

from registry import settings, store                                 # noqa: E402
from registry.sentinel import AUTHORITIES                            # noqa: E402

app = FastAPI(title="GodsEye demo vehicle registry (synthetic)",
              docs_url="/docs", redoc_url=None)

_started = time.time()
_received: list = []          # referrals this run, newest last


# --- registry -----------------------------------------------------------
@app.get("/api/v1/health")
def health():
    """Liveness, and a standing disclaimer about what this is."""
    try:
        stats = store.stats()
        ok = True
    except Exception as exc:
        stats, ok = {"error": str(exc)}, False
    return {
        "status": "ok" if ok else "degraded",
        "uptime_s": round(time.time() - _started, 1),
        "synthetic": True,
        "disclaimer": ("Demonstration registry. Every record is generated from "
                       "a hash of the registration number. This is NOT VAHAN, "
                       "contains no real vehicle, owner or case data, and "
                       "connects to no government system."),
        "registry": stats,
        "force_mode": settings.FORCE_MODE or "NORMAL",
        "authority_mode": settings.AUTHORITY_MODE,
    }


@app.get("/api/v1/vehicles/{registration_number}")
def vehicle(registration_number: str, response: Response):
    """One registration.

    Three outcomes, and the HTTP status distinguishes them because a caller
    that cannot tell "no such vehicle" from "I could not look" will eventually
    treat an outage as a finding:

        200  FOUND
        404  NOT_FOUND   - asked, and there is no such registration
        503  UNKNOWN     - could not be asked; says nothing about the vehicle
    """
    reg = store.lookup(registration_number, actor="http",
                       purpose="GET /api/v1/vehicles")
    body = reg.as_dict()
    if reg.registry_status == "UNKNOWN":
        response.status_code = 503
    elif not reg.found:
        response.status_code = 404
    return body


@app.get("/api/v1/vehicles")
def vehicles(limit: int = 25, offset: int = 0):
    """A page of the registry, so the demo can show it is a real population."""
    conn = store.connect()
    try:
        rows = conn.execute(
            "SELECT plate, vehicle_class, make, model, colour, fuel, rto_code,"
            " status FROM vehicles ORDER BY plate LIMIT ? OFFSET ?",
            (max(1, min(limit, 200)), max(0, offset))).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        return {"total": total, "count": len(rows), "offset": offset,
                "synthetic": True, "vehicles": [dict(r) for r in rows]}
    finally:
        conn.close()


# --- the authority ------------------------------------------------------
class Referral(BaseModel):
    incident_id: str = Field(..., examples=["INC-0042"])
    registration_number: str
    incident_type: str
    severity: str | None = None
    camera_id: str | None = None
    location: str | None = None
    timestamp: float | None = None
    confidence: float | None = None
    registry_status: str | None = None
    summary: str | None = None
    caveat: str | None = None
    recommended_action: str | None = None
    source: str | None = None


@app.post("/api/v1/authority/incidents", status_code=201)
def receive_incident(referral: Referral):
    """A SIMULATED authority control room receiving a referral.

    Answers with its own reference, which is what makes the acknowledgement
    meaningful: the platform can then show that something outside it took
    responsibility for the incident, rather than that a POST returned 200.

    OFFLINE returns 503 rather than dropping the connection, because that is
    the honest shape of "we are up but not accepting" - and it lets the client
    distinguish a refusal it should retry from a crash.
    """
    if settings.AUTHORITY_MODE == "OFFLINE":
        raise HTTPException(503, "authority control room is offline (demo control)")

    ref = f"AUTH-{8800 + len(_received) + 1}"
    record = {"authority_incident_id": ref, "received_at": time.time(),
              "referral": referral.model_dump()}
    _received.append(record)
    del _received[:-200]
    return {
        "authority_incident_id": ref,
        "status": "ACKNOWLEDGED",
        "received_at": record["received_at"],
        "queue_position": len(_received),
        "note": ("Simulated authority endpoint for a demonstration. No real "
                 "control room has been contacted."),
    }


@app.get("/api/v1/authority/incidents")
def received(limit: int = 50):
    """What the authority has taken in - the other side of the acknowledgement."""
    return {"count": len(_received), "incidents": _received[-limit:][::-1],
            "mode": settings.AUTHORITY_MODE,
            "authorities": AUTHORITIES}


@app.post("/api/v1/authority/mode/{mode}")
def set_mode(mode: str):
    """Demo control: ONLINE or OFFLINE."""
    mode = mode.upper()
    if mode not in ("ONLINE", "OFFLINE"):
        raise HTTPException(400, "mode must be ONLINE or OFFLINE")
    settings.AUTHORITY_MODE = mode
    return {"authority_mode": mode}


@app.post("/api/v1/registry/mode/{mode}")
def set_registry_mode(mode: str):
    """Demo control: NORMAL, NOT_FOUND or UNAVAILABLE."""
    mode = mode.upper()
    if mode not in ("NORMAL", "NOT_FOUND", "UNAVAILABLE"):
        raise HTTPException(400, "mode must be NORMAL, NOT_FOUND or UNAVAILABLE")
    settings.FORCE_MODE = "" if mode == "NORMAL" else mode
    store.cache_clear()          # a cached answer would outlive the switch
    return {"registry_mode": mode}


def main() -> int:
    import uvicorn
    print(f"[registry] synthetic registry + mock authority on "
          f"{settings.SERVICE_URL}", file=sys.stderr)
    print(f"[registry] docs at {settings.SERVICE_URL}/docs", file=sys.stderr)
    uvicorn.run(app, host=settings.SERVICE_HOST, port=settings.SERVICE_PORT,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
