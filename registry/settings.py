"""Registry tunables, in their own GODSEYE_REG_* namespace.

Separate from the platform's config.py so that nothing here can move a number
the ANPR path was measured against.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Its own file. The platform's store is never written by this package, and
# keeping the two apart is what makes "delete it and nothing changes" true.
DB_PATH = Path(os.getenv("GODSEYE_REGISTRY_DB", DATA_DIR / "registry.db"))

# The platform's store, opened READ-ONLY wherever this package reads it.
PLATFORM_DB = Path(os.getenv("GODSEYE_DB", DATA_DIR / "godseye.db"))


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- how the synthetic population is shaped -----------------------------
# Shares of the registered fleet carrying each defect. Deliberately small: a
# registry where a third of vehicles are stolen would make every rule fire and
# teach an operator nothing. These are roughly plausible orders of magnitude,
# not researched statistics, and the README says so.
SHARE_INSURANCE_EXPIRED = _float("GODSEYE_REG_SHARE_INSURANCE", 0.11)
SHARE_PUC_EXPIRED = _float("GODSEYE_REG_SHARE_PUC", 0.17)
SHARE_FITNESS_EXPIRED = _float("GODSEYE_REG_SHARE_FITNESS", 0.09)   # commercial only
SHARE_PERMIT_EXPIRED = _float("GODSEYE_REG_SHARE_PERMIT", 0.06)     # commercial only
SHARE_TAX_DUE = _float("GODSEYE_REG_SHARE_TAX", 0.08)
SHARE_STOLEN = _float("GODSEYE_REG_SHARE_STOLEN", 0.004)
SHARE_SCRAPPED = _float("GODSEYE_REG_SHARE_SCRAPPED", 0.003)
SHARE_BLACKLISTED = _float("GODSEYE_REG_SHARE_BLACKLIST", 0.005)
SHARE_HAS_CASES = _float("GODSEYE_REG_SHARE_CASES", 0.22)

# A plate whose registered class differs from what the cameras keep seeing.
# The registry's own contribution to fraud detection, and the rarest thing here.
SHARE_CLASS_MISMATCH = _float("GODSEYE_REG_SHARE_MISMATCH", 0.006)

MAX_CASES_PER_VEHICLE = _int("GODSEYE_REG_MAX_CASES", 4)

# Lookups are recorded. A registry that cannot say who asked about whom is the
# kind of system core/privacy.py exists to argue against.
AUDIT_LOOKUPS = os.getenv("GODSEYE_REG_AUDIT", "1") not in ("0", "false", "no")

# Demo control. "" is normal; "UNAVAILABLE" makes every lookup answer UNKNOWN;
# "NOT_FOUND" makes every lookup answer NOT_FOUND. Set at runtime from the
# console so the difference between "there is no such vehicle" and "we could
# not ask" can be demonstrated rather than described - they look identical on a
# slide and could not be less alike in what they justify doing.
FORCE_MODE = os.getenv("GODSEYE_REG_FORCE", "").strip().upper()

# How long a registry answer is reused before asking again. A plate crossing a
# junction is read many times a second; re-querying per frame is waste, and the
# record does not change between frames.
CACHE_TTL_S = _float("GODSEYE_REG_CACHE_TTL", 120.0)
CACHE_MAX = _int("GODSEYE_REG_CACHE_MAX", 2000)


# --- when the sentinel is allowed to speak ------------------------------
# Thresholds, not tastes. Each one exists because the platform's own error
# rate could otherwise manufacture the finding.
#
# Its own port, well clear of the platform's 8000/8501/8502 and the camera
# demos' 8503/8504/8505, so the console can be run or not run without any of
# them noticing.
CONSOLE_PORT = _int("GODSEYE_CONSOLE_PORT", 8506)

# A plate absent from the registry is usually a misread, not a forgery. One
# sighting proves nothing; the same string at several junctions is at least a
# pattern worth a human looking at the crops.
UNREGISTERED_MIN_SIGHTINGS = _int("GODSEYE_REG_MIN_UNREG", 4)

# Vehicle class disagreeing with the registry needs more than one frame -
# a single odd classification is noise.
MISMATCH_MIN_SIGHTINGS = _int("GODSEYE_REG_MIN_MISMATCH", 3)

# Unpaid challans below this are not worth an operator's attention; a console
# that lists every Rs 500 signal-jump buries the stolen vehicle underneath.
CHALLAN_ALERT_AMOUNT = _float("GODSEYE_REG_CHALLAN_ALERT", 5000.0)


# --- the registry service and the mock authority ------------------------
# One small FastAPI app carries both: GET /api/v1/vehicles/{reg} (the registry)
# and POST /api/v1/authority/incidents (the authority receiving a referral).
# They are separate systems in the story and separate endpoints here; running
# them in one process is a demo convenience, not a claim about architecture.
#
# Its own port again, clear of 8000/8501/8502/8503/8504/8505 and of the
# console's 8506.
SERVICE_PORT = _int("GODSEYE_REG_SERVICE_PORT", 8507)
SERVICE_HOST = os.getenv("GODSEYE_REG_SERVICE_HOST", "127.0.0.1")
SERVICE_URL = os.getenv("GODSEYE_REG_SERVICE_URL",
                        f"http://127.0.0.1:{SERVICE_PORT}")

# Demo control for the authority: "ONLINE" or "OFFLINE". Offline makes every
# referral fail so the retry path and DELIVERY FAILED can be shown - and, more
# importantly, so it can be shown that the incident SURVIVES the failure.
AUTHORITY_MODE = os.getenv("GODSEYE_REG_AUTHORITY", "ONLINE").strip().upper()

# Delivery. Small numbers on purpose: this is a demo, and a retry schedule that
# takes minutes cannot be shown to anyone.
DELIVERY_ATTEMPTS = _int("GODSEYE_REG_ATTEMPTS", 3)
DELIVERY_BACKOFF_S = _float("GODSEYE_REG_BACKOFF", 1.5)
DELIVERY_TIMEOUT_S = _float("GODSEYE_REG_TIMEOUT", 3.0)

# Which authority answers for which junction. Falls back to the authority the
# finding itself names (registry/sentinel.py routes by the KIND of problem),
# so this only has to carry the junctions where jurisdiction differs from that.
JUNCTION_AUTHORITY = {
    "CAM-20": "police",        # Esplanade - city police control room
    "CAM-21": "traffic",       # Park Street - traffic enforcement cell
    "CAM-28": "police",        # Rashbehari
}
