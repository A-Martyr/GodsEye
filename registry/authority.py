"""Referring an incident to the authority that can act on it.

Two halves, deliberately separate:

  ROUTING    which authority owns this incident. registry/sentinel.py already
             routes by the KIND of problem - a stolen vehicle to the police, a
             lapsed permit to the RTO - because that is what decides who can
             actually do something. settings.JUNCTION_AUTHORITY overrides by
             location where jurisdiction differs from that.

  DELIVERY   getting it there, and what to do when that fails.

The rule the whole module is built around: **a failed referral is a delivery
problem, never a lost incident.** The incident is written to the registry's own
store BEFORE anyone is called, and every outcome here only ever updates a
delivery column. An authority being unreachable must not be able to erase the
fact that something was found.

The endpoint on the other end is a SIMULATION - registry/service.py, running
on this machine. It is not a government system, does not pretend to be one, and
no real authority is contacted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests                                                  # noqa: E402

from registry import settings, store                             # noqa: E402
from registry.sentinel import AUTHORITIES                         # noqa: E402

# Delivery states. PENDING is the state an incident is born in, and the one it
# stays in if nobody ever tries - which is itself worth being able to see.
PENDING = "PENDING"
ACKNOWLEDGED = "ACKNOWLEDGED"
RETRYING = "RETRYING"
FAILED = "DELIVERY_FAILED"


def route(finding: dict) -> str:
    """Which authority should receive this. Returns an AUTHORITIES key.

    The finding's own authority wins. sentinel.py assigns it by the nature of
    the problem, and that is a better answer than geography: a stolen vehicle
    is a police matter at every junction, and routing it to a traffic cell
    because of where it was seen would send it to people who cannot act on it.

    Location only decides where the kind does not - which, in this demo, means
    the generic incidents that could belong to either.
    """
    by_kind = finding.get("authority")
    if by_kind in AUTHORITIES:
        return by_kind
    by_place = settings.JUNCTION_AUTHORITY.get(finding.get("camera_id") or "")
    return by_place if by_place in AUTHORITIES else "police"


def authority_name(key: str) -> str:
    return AUTHORITIES.get(key, key)


def _payload(incident: dict) -> dict:
    """What is actually sent. Deliberately small.

    Carries the caveat with the claim. These referrals name a vehicle and go to
    a police control room, and the platform's own error rate is part of the
    evidence - a receiving officer who is not told that a misread could explain
    the finding has been told something misleading.
    """
    return {
        "incident_id": incident["ref"],
        "registration_number": incident["plate"],
        "incident_type": incident["kind"],
        "severity": incident["severity"],
        "camera_id": incident.get("camera_id"),
        "location": incident.get("camera_name"),
        "timestamp": incident["ts"],
        "confidence": incident.get("confidence"),
        "registry_status": incident.get("registry_status"),
        "summary": incident.get("headline"),
        "caveat": incident.get("caveat") or "",
        "recommended_action": incident.get("recommended") or "",
        "source": "GodsEye (demonstration system - synthetic registry)",
    }


def deliver(incident: dict, *, attempts: int | None = None,
            conn=None) -> dict:
    """Refer one stored incident. Returns {status, authority_ref, error}.

    Retries a small number of times, then gives up and says so. The incident is
    left exactly as it was apart from its delivery columns.
    """
    attempts = settings.DELIVERY_ATTEMPTS if attempts is None else attempts
    url = f"{settings.SERVICE_URL}/api/v1/authority/incidents"
    payload = _payload(incident)
    last_error = ""

    for attempt in range(1, attempts + 1):
        if attempt > 1:
            store.set_delivery(incident["id"], RETRYING, error=last_error,
                               bump_attempt=False, conn=conn)
            time.sleep(settings.DELIVERY_BACKOFF_S)
        try:
            r = requests.post(url, json=payload,
                              timeout=settings.DELIVERY_TIMEOUT_S)
            if r.status_code == 503:
                # The authority is up but declaring itself unavailable - the
                # demo's OFFLINE switch. Retryable, and distinct from a crash.
                last_error = "authority reported itself offline"
                store.set_delivery(incident["id"], RETRYING, error=last_error,
                                   conn=conn)
                continue
            r.raise_for_status()
            body = r.json()
            ref = body.get("authority_incident_id")
            store.set_delivery(incident["id"], ACKNOWLEDGED, authority_ref=ref,
                               error=None, conn=conn)
            return {"status": ACKNOWLEDGED, "authority_ref": ref,
                    "attempts": attempt, "error": None}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:180]
            store.set_delivery(incident["id"], RETRYING, error=last_error,
                               conn=conn)

    store.set_delivery(incident["id"], FAILED, error=last_error,
                       bump_attempt=False, conn=conn)
    return {"status": FAILED, "authority_ref": None, "attempts": attempts,
            "error": last_error}


def refer(finding: dict, *, registry_status: str = "FOUND",
          confidence: float | None = None, notify: bool = True) -> dict:
    """Store a finding as an incident, then try to refer it.

    The order matters and is the point of the function: record first, notify
    second. Whatever happens to the notification, the incident is already safe.
    """
    routed = dict(finding)
    routed["authority"] = route(finding)
    incident = store.record_incident(routed, registry_status=registry_status,
                                     confidence=confidence)
    if not notify:
        return {"incident": incident, "delivery": {"status": PENDING}}
    outcome = deliver(incident)
    incident = store.incident_by_ref(incident["ref"]) or incident
    return {"incident": incident, "delivery": outcome}


def retry_failed(limit: int = 20) -> list[dict]:
    """Another go at everything that did not get through.

    Exists because DELIVERY_FAILED is not a terminal verdict on the incident -
    only on one round of attempts. The record is still there to try again from,
    which is the property the whole design is arranged to give.
    """
    out = []
    for inc in store.incidents(limit=limit, delivery_status=FAILED):
        out.append({"ref": inc["ref"], **deliver(inc)})
    return out


def summary() -> dict:
    """Delivery state across every incident, for the console."""
    rows = store.incidents(limit=1000)
    by = {}
    for r in rows:
        by[r["delivery_status"]] = by.get(r["delivery_status"], 0) + 1
    return {"total": len(rows), "by_delivery": by,
            "authority_mode": settings.AUTHORITY_MODE}
