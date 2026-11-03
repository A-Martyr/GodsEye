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
FIGURE_DIR = ROOT / "charts"
ASSET_DIR = ROOT / "assets"

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

# anpr_ocr (Awiros PP-OCRv5) reports the same quantity as the CRNN - the exact
# per-character-normalised CTC probability of the decoded string - but over a
# different model, so it needs its own floor. Calibrated over the 532-crop real
# corpus: this floor keeps 89% of reads at 95.1% stored accuracy (0.90 would
# keep 86% at 96.0%, 0.50 keeps 97% at 92.3%).
MIN_PLATE_CONFIDENCE_ANPR_OCR = 0.80

# Which recogniser anpr.ocr.engine() hands to the dashboard, the API and the
# simulator. All three come through that one shared singleton, so this is the
# single switch for the whole platform.
#
#   "crnn"       CRNN + grammar-constrained CTC decode, trained on the
#                synthetic camera corpus. The default, and the only backend
#                core/repair.py can pull CTC lattices from.
#   "anpr_ocr"   Awiros PP-OCRv5, pretrained on 558,767 real Indian plates.
#                Reads 91.0% of real photographs against the CRNN's 0.9%, but
#                it is a *real-photograph* number - on the simulator's
#                synthetic captures the CRNN is the stronger reader. Needs
#                paddle, so it only loads from .venv-paddle (PADDLE_SETUP.md).
#   "classical"  segment-and-classify. No torch required.
#
# A backend that cannot load degrades to the CRNN, but says so on stderr and
# through anpr.ocr.engine_status() - never silently.
OCR_BACKEND = os.getenv("GODSEYE_OCR_BACKEND", "crnn").strip().lower()

# Which device the neural readers run on - the YOLOv8-OBB localiser and the
# CRNN both come through here. "auto" takes a CUDA GPU when torch can see one
# and stays on the CPU when it cannot, so the same tree runs on a laptop with
# a card and on a machine without one. Force it with GODSEYE_DEVICE=cpu, or
# name a particular card with cuda:1.
#
# It is worth switching on. Profiled on this project's own 1100x700 sample
# frames, an 80 ms read_frame() is 44 ms of YOLO plus ~35 ms of CRNN forward
# passes - the CRNN's two-layer bidirectional GRU is the slow half of that,
# and cuDNN fuses it. What does *not* move is the grammar-constrained beam
# search (NumPy, ~27 ms whenever the greedy decode comes out ungrammatical),
# so the ceiling on the speedup is that decode, not the network.
#
# The device is not allowed to change what the readers decide: the confidence
# floors above are calibrated on fp32 and the GPU path is fp32 for that
# reason. Nothing here turns on half precision.
DEVICE = os.getenv("GODSEYE_DEVICE", "auto").strip().lower()

_resolved_device: str | None = None


def torch_device() -> str:
    """DEVICE resolved against what is actually installed. Cached.

    DEVICE is a request; this is the answer. It never raises and never returns
    a device torch cannot use - no torch, a CPU-only torch wheel, no CUDA card,
    or a card too old for the wheel that is installed all end up on "cpu", so
    asking for a GPU can never be what stops the platform running.
    """
    global _resolved_device
    if _resolved_device is None:
        _resolved_device = _resolve_device(DEVICE or "auto")
    return _resolved_device


def _resolve_device(want: str) -> str:
    import sys

    def refuse(why: str) -> str:
        # "auto" asked for whatever is there and got the CPU, which is not a
        # problem worth printing. An explicit GODSEYE_DEVICE that could not be
        # honoured is: a demo that quietly ran ten times slower than the
        # operator expected is worse than one that said why.
        if want != "auto":
            print(f"[config] GODSEYE_DEVICE={want!r}: {why} - running on the CPU",
                  file=sys.stderr)
        return "cpu"

    if want == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        return refuse("torch is not installed")
    if not torch.cuda.is_available():
        # Far and away the usual cause is a CPU-only wheel on a machine that
        # does have a card, so name that rather than "no GPU found".
        built = getattr(torch.version, "cuda", None)
        return refuse("torch reports no usable CUDA device"
                      + ("" if built else f" (this is a CPU-only torch build, {torch.__version__})"))
    name = "cuda" if want == "auto" else want
    try:
        # A card older than anything the installed wheel was compiled for
        # loads, reports itself available, and then fails on the first kernel
        # launch. Launch one here, while there is still somewhere to fall back
        # to. (A GTX 1650 is sm_75; wheels do drop old architectures.)
        _ = (torch.zeros(8, device=name) + 1).sum().item()
    except Exception as exc:
        return refuse(f"{name} is present but unusable ({exc})")
    return name

# How many frames a camera contributes per vehicle. A real ANPR node is
# triggered as the vehicle enters the zone and takes a burst, not a photograph:
# the frames differ in distance, exposure and motion blur, they fail in
# different ways, and agreement between them is strong evidence. Measured over
# 250 vehicles, one frame reads 40.4% of plates and eight read 77.2% with
# identical weights. Set to 1 to go back to reading a single frame.
BURST_FRAMES = int(os.getenv("GODSEYE_BURST", "5"))

# Load the reader and run one throwaway inference through it at start-up, on a
# background thread, rather than leaving it to whoever clicks first.
#
# The cost is real and it is entirely front-loaded. Measured here: the first
# read of a whole frame takes 3.3 s and every one after it takes 64 ms, and
# about nine seconds separate process start from first result - CUDA context
# creation, the model loads, and ultralytics building its predictor and letting
# cuDNN autotune. None of that can be made cheaper. It can only be moved, and
# start-up is where it belongs, because demo.bat is already waiting on the port
# and nobody is watching. Turn it off with GODSEYE_WARM=0 if a cold process
# needs to start instantly - a script that reads one image and exits, say,
# where warming would be pure overhead.
WARM_READER_ON_START = os.getenv("GODSEYE_WARM", "1") not in ("0", "false", "False")

# --- anpr_ocr throughput -------------------------------------------------
# How many CPU threads paddle may use. It must be set before paddle is
# imported, which anpr/anpr_ocr_reader.py does at module scope for exactly
# this reason - paddle reads it once at init and pins its pools then.
#
# Paddle defaults this to 1, which on a 16-core machine leaves fifteen cores
# idle while a plate takes four seconds. Raising it helps, but not monotonically
# - measured here on one real 960x1280 street photograph, per 120x40 plate crop:
#
#       1 thread (paddle's default)   3909 ms
#       4 threads                     1650 ms   <- best
#       8 threads                     4208 ms   worse than one thread
#      16 threads                     5115 ms
#
# Past four it collapses, and it collapses below the single-threaded number.
# The work per crop is small enough that paddle spends longer starting and
# joining its pool than the pool saves, and on a hybrid CPU the scheduler
# spreads that across efficiency cores. Four is not a guess; anything larger
# measured worse on this machine. Re-measure before changing it.
ANPR_OCR_THREADS = int(os.getenv("GODSEYE_ANPR_OCR_THREADS", "4"))

# How many crops anpr_ocr may put through the recogniser in one forward pass.
#
# A forward pass on this model is launch-bound, not compute-bound - one crop
# costs about as much wall time as eight - so batching is close to free work.
# Measured on the GTX 1650, 28 crops of the fixed [3,48,320] input:
#
#       one at a time   470.7 ms   (16.8 ms each)
#       batch of 8      215.7 ms   ( 7.7 ms each)
#       batch of 28     167.8 ms   ( 6.0 ms each)
#
# Batching is NOT bit-identical: cuDNN picks different algorithms per batch
# size, so posteriors shift by about 1e-4. That was measured against the whole
# 532-crop real corpus rather than assumed - zero plate strings changed, zero
# accept/refuse decisions flipped, exact-match accuracy stayed at 90.98%, and
# the largest confidence change was 3.7e-05. Raising this much further trades
# a little more speed for a little more numerical drift, so re-run that
# comparison if you change it.
#
# 32 also keeps the activation footprint trivial next to a 4 GB card.
ANPR_OCR_MAX_BATCH = int(os.getenv("GODSEYE_ANPR_OCR_BATCH", "32"))

# Stop searching a photograph once a read is this confident and legal.
#
# read_frame() decodes the whole frame and then every inset of every box the
# localiser proposed - up to 48 decodes. That is the right trade for the CRNN,
# where a decode is 5 ms and the whole sweep costs a tenth of a second. It is
# the wrong trade for anpr_ocr, where a decode is over a second, and the same
# sweep is what turned one photograph into forty-six seconds. Since the boxes
# arrive in the localiser's own confidence order, the first strong legal read
# is almost always the answer, and the remaining decodes only confirm it.
#
# Applied only to the slow reader. The CRNN keeps its exhaustive sweep, which
# costs it nothing and is where its accuracy numbers come from.
FRAME_EARLY_EXIT_CONFIDENCE = float(os.getenv("GODSEYE_FRAME_EARLY_EXIT", "0.90"))

# The wall-clock ceiling on searching one photograph with the slow reader, and
# the thing that actually bounds the worst case.
#
# Early exit is an optimisation, not a guarantee - it only fires when something
# reads well, and cannot bound a case defined by nothing reading well. Measured
# on a second street photograph, no crop cleared the 0.80 floor until decode 14
# of 28, so the sweep ran to the end: 62 seconds.
#
# A time budget rather than a decode count, for two reasons. Decodes are not
# equal - measured between 1.5 s and 3.3 s here, because a square-ish crop is
# also read row-split, which is a second forward pass - so counting them does
# not bound the wait. And a count tuned to this CPU would go on rationing a GPU
# that does not need it, whereas a deadline simply never fires when decodes are
# fast.
#
# Set it to 0 for no limit, which is the right setting once paddle is running
# on a GPU. The honest cost of the default: a plate whose best read is late in
# the sweep can be missed on CPU. read_frame() says when it stopped early
# rather than reporting a bounded search as an exhaustive one.
FRAME_SEARCH_SECONDS = float(os.getenv("GODSEYE_FRAME_SEARCH_S", "20"))

# Pairs the classifier genuinely confuses on degraded plates. Used both for
# format-aware correction and for fuzzy plate search.
CONFUSION_PAIRS = [
    ("0", "O"), ("0", "D"), ("0", "Q"), ("1", "I"), ("1", "L"), ("1", "T"),
    ("2", "Z"), ("5", "S"), ("6", "G"), ("8", "B"), ("4", "A"), ("7", "T"),
    ("9", "G"), ("U", "V"), ("M", "N"), ("C", "G"), ("K", "X"), ("3", "8"),
]

# --- Network-constrained repair -----------------------------------------
# A capture below the floor is not stored as a sighting, but its CTC lattice is
# kept briefly: once the neighbouring cameras have reported, the plate can be
# recovered by ranking what *they* saw against this camera's own evidence.
# See core/repair.py for why this is a ranking problem and not a fill-in-the-
# blanks one.
REPAIR_ENABLED = os.getenv("GODSEYE_REPAIR", "1") not in ("0", "false", "False")
# How long a vehicle may take between two adjacent cameras, as multiples of the
# free-flow time. Generous on purpose: too tight a window loses the true plate,
# too wide only adds candidates, and the likelihood ranking is barely affected
# by them - a twentyfold larger set costs nine points.
REPAIR_WINDOW_MIN_FACTOR = 0.6      # faster than free flow
REPAIR_WINDOW_MAX_FACTOR = 4.0      # jammed, or stopped for a chai
REPAIR_WINDOW_SLACK_S = 90.0
# The repair runs behind the live feed: half its evidence is the *downstream*
# camera, which has not seen the vehicle yet at the moment the capture fails.
REPAIR_LAG_S = 15 * 60.0
REPAIR_EVIDENCE_TTL_S = 60 * 60.0   # then the lattices are dropped
REPAIR_CLONE_LOOKBACK_S = 6 * 3600.0
REPAIR_MISSING_FRAME_PENALTY = -30.0
# Acceptance. Both conditions together stand in for a "none of these"
# hypothesis: when the vehicle entered off a road no camera covers, the true
# plate is simply absent from the candidate set and the best wrong candidate
# still looks plausible.
#
# The margin is what actually buys the precision, and by a wide margin. Swept
# over 60 refused captures from a simulated 90 minutes: at a score floor of
# -3.0, requiring no margin accepts 59 of them at 79.7% precision, while
# requiring 0.15 accepts 46 at 100%. The floor alone is a much blunter
# instrument. The floor is kept tighter than the sweep strictly needs because
# that sweep under-represents the case the floor exists for: the simulator's
# camera coverage is dense, so the true plate was in the candidate set 93.3% of
# the time. On a network with real gaps that share is lower and the floor is
# what stops a confident-looking wrong candidate getting through.
#
# 100% on 46 samples is not 100%. The upper bound of the 95% interval is about
# 6%, so treat these as "roughly 95% precision" until measured on more data.
REPAIR_MIN_SCORE = -2.00            # per-character CTC log-likelihood
REPAIR_MIN_MARGIN = 0.15            # ... and it must beat the runner-up by this
# Wall-clock ceiling for the erasure salvage pass (anpr/salvage.py), per
# capture. A hard deadline checked between candidates, so it can be exceeded by
# at most one scoring call - about 0.74 ms. It runs only on captures the guards
# have already refused, which were about to be discarded, so this is time spent
# on work that currently returns nothing.
# How many road links out from the failed capture to gather candidates from.
# One hop assumes the ADJACENT camera saw the vehicle; it often did not, because
# this platform refuses about a fifth of its captures, so the intermediate
# camera frequently has nothing to contribute and the true plate is missing from
# the pool by construction. Measured over 800 refused captures with ground
# truth, two hops instead of one recovered 31 more of them for 1 more error -
# recall 60.2% -> 64.1%, precision 98.2% -> 98.1% - at 2.6x the candidates and
# 2.6x the scoring time. Affordable only because the repair pass runs behind the
# live feed. A third hop continues at a worse rate. Set to 1 to restore the old
# behaviour.
# Detector input size, and the sizes it falls back to when nothing decoded.
# The YOLOv8-OBB model responds to a narrow band of apparent plate sizes: on one
# real photograph the true plate scored 0.024 at 640, 0.569 at 1280 and 0.011 at
# 1920, non-monotonically (FIELD_TEST.md). There is no single right value, so a
# frame that produced no acceptable read is looked at again at other sizes.
# Escalation is gated on the READ, never on detector confidence - the detector
# gave 0.33 to a bus panel on that same photograph. Set the escalation to () to
# restore single-scale behaviour.
YOLO_IMGSZ = int(os.getenv("GODSEYE_YOLO_IMGSZ", "640"))
YOLO_ESCALATE_SCALES = (() if os.getenv("GODSEYE_YOLO_ESCALATE", "1") in ("0", "false", "False")
                        else (960, 1280, 1600))

REPAIR_CANDIDATE_HOPS = int(os.getenv("GODSEYE_REPAIR_HOPS", "2"))
REPAIR_SALVAGE_BUDGET_MS = 40.0

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

# --- Clone detection (core/clones.py) -----------------------------------
# A clone verdict accuses a real motorist, so each of these exists to rule out a
# cheaper explanation than "this registration is on two vehicles".
# Two cameras a few hundred metres apart turn ordinary clock skew into a huge
# implied speed, so a conflict has to span real distance.
CLONE_MIN_KM = 3.0
# Evidence from a read the engine barely stood behind is not evidence. Both ends
# of a conflicting pair must clear this.
CLONE_MIN_CONFIDENCE = 0.70
# One impossible leg is a lead; several independent ones are a finding. Below
# this a verdict is reported as "suspected" and stays off the alert queue.
CLONE_CONFIRM_PAIRS = 2
# How far either side of a sighting to look for the plate it might really have
# been, and how close in OCR confusion distance that plate has to be. 1.0 is
# about one substitution the engine is known to make.
CLONE_MISREAD_WINDOW_S = 15 * 60.0
CLONE_MISREAD_MAX_DISTANCE = 1.0
CLONE_ALERT_DEDUP_S = 30 * 60.0
LOITER_REVISITS = 4             # same camera N times ...
LOITER_WINDOW_MIN = 45          # ... within this window
ODD_HOUR_RANGE = (1, 5)         # 01:00-05:00 local

# --- Manual enrolment (core/enrol.py) -----------------------------------
# A real photograph read by the engine names a vehicle this network has never
# seen, so there is no trajectory to draw. Enrolment is the operator standing
# in for a camera that is not there: they supply the two facts the photograph
# does not carry - which junction, and when - and the read itself is still the
# engine's, never a typed string with an invented confidence.
#
# The row lands in `sightings` because that is what it is: an observation of a
# vehicle at a place. It is NOT an `inferences` row - nothing was inferred, the
# plate was read - but `sightings.source` records that a human, not a camera,
# attested the place and time. Everything that measures the *network* filters
# on that column; everything that tracks a *vehicle* does not.
ENROL_SOURCE = "enrolment"              # sightings.source for an uncorrected read
ENROL_SOURCE_CORRECTED = "enrolment-corrected"   # ... operator fixed the string
CAMERA_SOURCE = "camera"                # every row the simulator or a real node writes
ENROL_SOURCES = (ENROL_SOURCE, ENROL_SOURCE_CORRECTED)

# Where the default enrolment timestamp comes from. Anchoring to the newest
# camera sighting rather than the wall clock keeps an enrolment inside the
# window the rest of the platform is looking at: a seeded database is usually
# hours old by demo time, and a wall-clock enrolment would sit alone outside
# every analytics window and every "last 30 minutes" table.
ENROL_ANCHOR_TO_NETWORK_CLOCK = True

# The second enrolment of the same plate has to be far enough after the first
# that the implied speed is possible, or trajectory.reconstruct() marks the leg
# "physically impossible, same plate probably running on two vehicles" and the
# demo accuses itself of a clone. enrol.suggest_ts() derives the gap from the
# actual road distance instead of guessing: road_km / ENROL_TARGET_KMPH, floored
# at ENROL_MIN_GAP_S. The target is an ordinary city speed, comfortably under
# IMPOSSIBLE_SPEED_KMPH (160) and under the 1.6x free-flow speeding rule.
ENROL_TARGET_KMPH = 32.0
ENROL_MIN_GAP_S = 240.0                 # 4 min, clear of the alert thresholds
# Refuse to write a leg the platform would itself call impossible. Slightly
# tighter than IMPOSSIBLE_SPEED_KMPH so a rounding error cannot straddle it.
ENROL_MAX_IMPLIED_KMPH = 150.0
# How far the operator may move the string the engine read. Correcting a
# misread is supported and necessary - a monsoon capture in this repo reads one
# character wrong at 0.97, above the 0.96 floor, and without a correction the
# second enrolment never joins the first. Entering an unrelated registration is
# not supported. The bound is OCR confusion distance (anpr.ocr), so a
# correction has to be the kind of mistake this engine actually makes: 1.0 is
# one plain substitution, 0.4 one confusable pair (0/O, 8/B, 5/S).
ENROL_CORRECTION_MAX_DISTANCE = 2.0
# Re-enrolling the same plate at the same junction within this window is
# treated as a double submission rather than a second pass. A vehicle really
# can come back to a junction - that is what the loitering rule is for - so the
# window is short: it exists to absorb a double-click on the enrol button, not
# to decide what a vehicle can do.
ENROL_DUPLICATE_WINDOW_S = 90.0

# --- Live clone response (dashboard + judge console) ---------------------
# How far back the control room treats a clone alert as "live": it keeps the
# junctions glowing on the map and the banner up. Long enough that an operator
# who looked away still sees it, short enough that the map is not permanently
# lit by something from an hour ago.
CLONE_LIVE_WINDOW_S = float(os.getenv("GODSEYE_CLONE_LIVE_S", 10 * 60))
# Sound the alarm when a clone alert arrives. Off by default in code so a
# headless or recorded run is silent; the dashboard turns it on with a toggle
# the operator can see.
CLONE_ALARM_ENABLED = os.getenv("GODSEYE_CLONE_ALARM", "1") not in ("0", "false", "False")
# The sound it makes. Shipped as an asset so the control room hears the alarm
# the operators were briefed on rather than a generated tone; point
# GODSEYE_CLONE_ALARM_FILE at another file to change it. If the file is missing
# the dashboard falls back to a synthesised two-tone burst, so a checkout
# without the asset is still audible.
CLONE_ALARM_FILE = Path(os.getenv("GODSEYE_CLONE_ALARM_FILE", ASSET_DIR / "red_alert.mp3"))

# The judge console (GET /judge) lets someone outside the team inject an
# incident at a moment of their choosing. It is deliberately unauthenticated -
# a judge holding a phone should not be typing a password. On a LAN that is
# safe enough: only the room can reach it. Over the public tunnel below it is
# not, so the console is additionally gated there by JUDGE_TOKEN and by the
# public-origin restriction in api/main.py. Turn it off outside a demo.
JUDGE_CONSOLE_ENABLED = os.getenv("GODSEYE_JUDGE", "1") not in ("0", "false", "False")

# --- reaching the judge, wherever they are -------------------------------
# The LAN link only works if the judge is standing in the same room on the
# same network. A quick tunnel gives a public HTTPS URL that works from any
# network on earth - mobile data, another city - which is what you need to
# send the console to someone who is not at the table.
#
# "quick" means no Cloudflare account and no configuration: cloudflared asks
# trycloudflare.com for a throwaway hostname and prints it. The hostname is
# different every run, so the link must be re-shared after a restart.
TUNNEL_ENABLED = os.getenv("GODSEYE_TUNNEL", "0") not in ("0", "false", "False")
TUNNEL_BIN = os.getenv("GODSEYE_TUNNEL_BIN", "cloudflared")
# Where the tunnel process leaves its public URL for the other processes to
# read. The API and the dashboard are separate processes and neither owns the
# tunnel, so a file is the only thing both can see.
TUNNEL_URL_FILE = DATA_DIR / "tunnel_url.txt"
TUNNEL_LOG_FILE = DATA_DIR / "tunnel.log"
# How long to wait for cloudflared to print its hostname. It takes about five
# seconds with a working connection; the wait exists so that no connection at
# all fails quickly and says so rather than hanging the dashboard.
TUNNEL_STARTUP_S = float(os.getenv("GODSEYE_TUNNEL_WAIT", 30))

# The shared secret that makes the public link safe to send. It is carried in
# the link itself, so whoever holds the link is in and nobody types anything -
# the same "no password on a phone" principle, without leaving an open
# injector on the internet. Requests arriving on a LAN or loopback host skip
# it entirely, so nothing about the in-the-room demo changes.
#
# Stored in a file rather than generated per process: the API and the
# dashboard must agree on it, and it should survive a restart so a link you
# already sent someone keeps working.
JUDGE_TOKEN_FILE = DATA_DIR / "judge_token.txt"
JUDGE_TOKEN_REQUIRED_PUBLIC = os.getenv("GODSEYE_JUDGE_TOKEN", "1") not in ("0", "false", "False")
# Injected incidents pick their junction pair at random from links at least
# this far apart, so the team cannot know in advance where one will surface.
# Far enough that the implied speed clears IMPOSSIBLE_SPEED_KMPH by a wide
# margin at the gap below.
JUDGE_CLONE_MIN_KM = 8.0
JUDGE_CLONE_GAP_S = 180.0
API_BASE = os.getenv("GODSEYE_API", "http://127.0.0.1:8000")

# --- feature parity with commercial LPR analytics ------------------------
# Everything below backs a capability this platform did not have until the
# gap analysis against Motorola Solutions' VehicleManager. Each module reads
# these through getattr with the same default, so it runs whether or not this
# block is present - but a tunable nobody can find is a tunable nobody tunes.

# Convoy detection (core/convoy.py). The naive co-travel count in
# trajectory.co_travellers() has no notion of chance: on a rush-hour corridor
# the vehicles that "travel together" most are simply the ones on the busiest
# route. These thresholds are what separate a convoy from a shared commute.
CONVOY_WINDOW_S = float(os.getenv("GODSEYE_CONVOY_WINDOW_S", 120.0))
CONVOY_MAX_LEG_MIN = 45.0        # beyond this the pair stopped, they did not travel
CONVOY_LOCK_DRIFT_S = 15.0       # gap drift still counted as holding formation
CONVOY_MIN_LOCKED_HOPS = 3       # shared hops before "convoy" is even considered
CONVOY_MIN_PASSES = 3            # co-occurrences below this are not evidence of anything
CONVOY_P_STRONG = 1e-3           # tail probability for a confident call
CONVOY_P_WEAK = 0.05             # below this it is reported as a shared route, not a convoy
CONVOY_CONFUSABLE_DISTANCE = 1.0  # OCR-confusable plates are one vehicle, not two in convoy

# Geofencing (core/geofence.py). Zones live in a JSON file rather than the
# database because an operator should be able to read and edit them.
GEOFENCE_FILE = DATA_DIR / "zones.json"
GEOFENCE_WINDOW_HOURS = 6.0
GEOFENCE_DEDUP_MIN = 20.0            # re-entry inside this is the same visit
GEOFENCE_FALLBACK_DWELL_MIN = 20.0   # assumed dwell when no exit was ever seen
GEOFENCE_OUTBOUND_DEG = 60.0         # heading cone counted as leaving the zone
GEOFENCE_MIN_LAG_SAMPLES = 8         # below this, travel-time estimates are not used

# Attribute search (core/vehicle_search.py) - finding a vehicle when the one
# thing the witness did not get is the plate.
SEARCH_NEAR_KM = 2.0
SEARCH_RECENCY_HALFLIFE_H = 6.0
SEARCH_SPEEDING_FACTOR = 1.6         # x the link's free-flow speed to count as speeding
SEARCH_SLOW_FACTOR = 0.45
SEARCH_MAX_SCAN = 60000
SEARCH_MAX_ACTIONABLE = 40
SEARCH_MIN_CONSTRAINTS = 2           # one bare attribute matches half the city
SEARCH_BROAD_SHARE = 0.10            # a result set this large is a browse, not a search

# Vehicle registry (core/registry.py) - a mock, VAHAN-shaped lookup used as an
# independent signal inside the engine. Isolated on purpose from the registry/
# package (the control-room Security Console) - see core/registry.py's module
# docstring for why two non-integrated mock registries exist in this repo.
REGISTRY_MISMATCH_RATE = 0.03        # chance the mock disagrees with the plate's own
                                      # observed vehicle_class - this rate directly
                                      # drives clone-queue promotions in
                                      # core/clones.py, kept low on purpose
REGISTRY_NOT_FOUND_RATE = 0.03       # chance a plate is simply unregistered
REGISTRY_FORCE_MODE = os.getenv("GODSEYE_REGISTRY_MODE", "").strip().upper()  # "" | "UNKNOWN"
REGISTRY_CACHE_TTL_S = 120.0
CLONE_ATTRIBUTE_CHECK_ENABLED = os.getenv("GODSEYE_CLONE_ATTR_CHECK", "1") not in ("0", "false", "False")
REPAIR_REGISTRY_NOT_FOUND_PENALTY = -0.50   # additive to a repair candidate's CTC
                                             # score; > 3x REPAIR_MIN_MARGIN (0.15) so
                                             # it can flip a close race, well inside
                                             # REPAIR_MIN_SCORE's floor (-2.00) so it
                                             # cannot alone disqualify strong evidence.
                                             # Reasoned, not measured - flag as a
                                             # follow-up once a labelled benchmark exists.

# Partial-plate resolution (core/registry.py::resolve_partial) - masked-pattern
# enumeration ("WB01??9492") narrowed by what has actually been seen and, for
# leads never sighted, by the mock registry. CHATGPT_BRIEF.md §5 point 4.
REGISTRY_MASK_WILDCARD = "?"
REGISTRY_PARTIAL_MAX_CANDIDATES = 5000   # analytic cap, enforced before any
                                          # candidate string is generated
REGISTRY_PARTIAL_MAX_UNSEEN = 25         # unseen_registered is capped harder -
                                          # it is the weaker tier

# Privacy (core/privacy.py). The part of this comparison worth winning.
# VehicleManager answers privacy with documents - an FAQ, a policy, a
# compliance guide. Documents describe what operators should do; the settings
# below are enforced by code that runs. A plate-reading network that holds
# everything forever and logs no lookups is a different object from one that
# does not, and that difference should live in the software.
PRIVACY_RETENTION_DAYS = float(os.getenv("GODSEYE_RETENTION_DAYS", 30.0))
PRIVACY_AUDIT_RETENTION_DAYS = 365.0   # the audit trail outlives the data it describes
PRIVACY_K_ANONYMITY = 5                # groups smaller than this are suppressed, not rounded
PRIVACY_REQUIRE_PURPOSE = True         # a lookup with no stated reason is refused
PRIVACY_MAX_PURGE_FRACTION = 0.5       # a purge this large is a bug, not a policy
PRIVACY_PSEUDONYM_SALT = os.getenv("GODSEYE_PSEUDONYM_SALT", "")
