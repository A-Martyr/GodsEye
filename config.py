"""GodsEye — central configuration.

Every path and tunable the platform needs, resolved relative to this file so the
modules work regardless of the directory uvicorn/streamlit was launched from.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CAMERA_FILE = DATA_DIR / "cameras.json"
DB_PATH = Path(os.getenv("GODSEYE_DB", DATA_DIR / "godseye.db"))
GLYPH_MODEL = MODEL_DIR / "glyph_mlp.joblib"

CITY_NAME = "Kolkata"
CITY_CENTER = (22.5726, 88.3639)

# --- ANPR ---------------------------------------------------------------
# Characters that can appear on an Indian civilian plate.
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLYPH_SIZE = 32                 # classifier input is GLYPH_SIZE x GLYPH_SIZE
PLATE_NORM_HEIGHT = 64          # plate crops are normalised to this height
# Below this a read is dropped rather than stored. Confidence is the weakest
# character times the share of binarisation hypotheses that agreed, so this is
# not a probability - it is an operating point, chosen on the benchmark: it
# discards about a quarter of reads and lifts the accuracy of what is stored
# from 83% to roughly 96%.
MIN_PLATE_CONFIDENCE = 0.40
# The CRNN's confidence is the exact CTC probability of the decoded string,
# marginalised over every alignment and normalised per character (see
# anpr.crnn.ctc_score). It is a different quantity from the classical engine's
# (weakest character x agreement between binarisation hypotheses), so each
# backend is judged against its own floor, exposed as PlateRead.accepted.
#
# Measured over 800 mixed captures, this floor keeps 43.6% of reads at 92.0%
# exact-string accuracy; 0.98 keeps 38.8% at 96.5% if a site wants to be
# stricter. The previous floor of 0.70 belonged to a heuristic confidence that
# ranked correct reads *below* wrong ones (AUROC 0.37); on the exact score the
# same 90% operating point keeps three times as many reads.
MIN_PLATE_CONFIDENCE_CRNN = 0.96

# How many frames a camera contributes per vehicle. A real ANPR node is
# triggered as the vehicle enters the zone and takes a burst, not a photograph:
# the frames differ in distance, exposure and motion blur, they fail in
# different ways, and agreement between them is strong evidence. Measured over
# 250 vehicles, one frame reads 40.4% of plates and eight read 77.2% with
# identical weights. Set to 1 to go back to reading a single frame.
BURST_FRAMES = int(os.getenv("GODSEYE_BURST", "5"))

# Pairs the classifier genuinely confuses on degraded plates. Used both for
# format-aware correction and for fuzzy plate search.
CONFUSION_PAIRS = [
    ("0", "O"), ("0", "D"), ("0", "Q"), ("1", "I"), ("1", "L"), ("1", "T"),
    ("2", "Z"), ("5", "S"), ("6", "G"), ("8", "B"), ("4", "A"), ("7", "T"),
    ("9", "G"), ("U", "V"), ("M", "N"), ("C", "G"), ("K", "X"), ("3", "8"),
]

# --- Simulation ---------------------------------------------------------
FLEET_SIZE = int(os.getenv("GODSEYE_FLEET", "260"))
SIM_TICK_SECONDS = float(os.getenv("GODSEYE_TICK", "1.0"))
# Run every simulated sighting through the real OCR pipeline (renders a plate
# image, degrades it, reads it back). Truthful end-to-end, but CPU-hungry.
INLINE_OCR = os.getenv("GODSEYE_INLINE_OCR", "1") not in ("0", "false", "False")
INLINE_OCR_MAX_PER_TICK = int(os.getenv("GODSEYE_INLINE_OCR_MAX", "12"))

# --- Analytics ----------------------------------------------------------
DEFAULT_WINDOW_MIN = 30
CONGESTION_THRESHOLDS = {"free": 1.25, "moderate": 1.6, "heavy": 2.2}  # ratio vs free-flow

# --- Alerts -------------------------------------------------------------
IMPOSSIBLE_SPEED_KMPH = 160.0   # above this between two cameras => cloned plate
LOITER_REVISITS = 4             # same camera N times ...
LOITER_WINDOW_MIN = 45          # ... within this window
ODD_HOUR_RANGE = (1, 5)         # 01:00-05:00 local
API_BASE = os.getenv("GODSEYE_API", "http://127.0.0.1:8000")
