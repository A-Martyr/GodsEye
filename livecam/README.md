# Live webcam plate reader

Hold a printed, plate-format board in front of the laptop webcam; the platform's
real recogniser reads it live, votes across frames until the reading settles,
and — optionally — enrols it as a sighting so it appears on the control room's
map with a trajectory.

Entirely separate from the tuned still-photograph path. This package is
imported by nothing in `anpr/`, `core/`, `api/` or `sim/`; it only
ever reads *from* them. It runs in its own process on its own port, so nothing
here can slow down or change the numbers the rest of the platform is measured
on.

---

## Run it

```bash
python -m livecam.makeboard WB24AB1234 --pdf   # 1. print the board (A4 landscape)
livecam.bat                                    # 2. native window, lowest latency
livecam_web.bat                                # 2b. or browser, http://localhost:8503
```

Then hold the board inside the on-screen brackets. The brackets go grey → amber →
green as frames agree, and the registration locks.

Keys (native window): `q` quit · `f` fullscreen · `space` pause · `r` reset the
vote · `e` enrol · `s` save a still.

Both launchers run from `.venv-paddle`, because `anpr_ocr` is the only backend
with real-world accuracy (91.0% on real photographs against the CRNN's 0.9%) and
a printed board through a webcam is a real photograph. Without that environment
the reader silently falls back to the CRNN — so both front-ends print and
display the reason rather than letting it pass unnoticed.

**Keep the GPU to this reader:** start the API with `GODSEYE_DEVICE=cpu`. One 4 GB card, and a second CUDA
context starves both by roughly 50x (RUNBOOK.md).

---

## Measured on this machine

| | |
|---|---|
| Fast path — board inside the brackets | **~30 ms** per read, ~30 reads/s |
| Fallback — board anywhere else in frame | ~130 ms per read |
| Camera | 1280×720 at 30 fps (MSMF) |
| Start-up, including model warm | ~8 s |
| Display | 50–60 draws/s regardless of read latency |

Accuracy on a printed board through a simulated 720p webcam, over size, blur,
angle, brightness, sensor noise and JPEG quality: **every correct read scored
1.00 confidence.** It survives Gaussian blur σ21, dimming to 15% brightness,
noise σ35, JPEG quality 20, tilt to 12°, and the board shrinking to 22% of the
aim box. It fails past about 20° of tilt — hold the board flat.

---

## How it works, and why

**The display never waits for the recogniser.** Rendering runs at camera speed
and always draws the newest frame with the most recent result available;
inference runs on its own thread. A 60 ms read makes the *plate text* update at
16 Hz over video that is still 30 fps — coupling the two is the usual way a live
vision demo ends up looking broken.

**Capture keeps exactly one frame.** `cv2.VideoCapture` buffers, so a consumer
slower than the camera does not skip — it reads progressively older frames and
falls behind without bound. The grab thread overwrites a single slot and counts
what it dropped.

**The backend is chosen by measuring it.** On this laptop DSHOW negotiates 720p
and then delivers 10 fps while MSMF delivers 30. Which one wins is a property of
the driver, so `capture.py` probes for half a second and takes the faster.

**Two stages, cheapest first.** Reading just the aim-box crop is 30 ms and gets
the plate exactly right; running the detector and sweeping its crop variants is
130 ms for the same answer. In the demo's normal case the detector is 100 ms
spent confirming what one crop already knew, so it only runs when that crop does
not clear the floor. Same principle as `read_frame()`'s early exit, one level
earlier.

**Deliberately not `ANPREngine.read_frame()`.** That function is tuned for a
single uploaded photograph. In a stream two of its behaviours are actively
wrong: when nothing decodes it escalates the detector to 960/1280/1600px looking
harder — which in a video would fire on nearly every frame, since most frames
contain no board — and it sweeps up to twelve regions on a 20-second budget,
when here the next frame is 33 ms away.

**Frames vote.** The platform already establishes that frames fail
independently, so agreement between them is evidence no single frame can produce
(42.4% single-frame against 73.1% over a five-frame burst). A burst costs a
camera a deliberate trigger; a video stream produces those frames anyway. Three
agreeing frames lock the reading, which both raises accuracy and stops the
number flickering between near-misses — a reading that settles is legible, one
that redraws every 60 ms is not.

**A second confidence floor, chosen from data.** A webcam pointed at a room sees
posters, screens and whiteboards. Measured here, the line "Q3 ROADMAP 2026"
decoded to `R04DMAP202` at 0.98 and then to `OD0S1069` at 0.88 — the second one
grammatical, with a real state code, and comfortably above the engine's 0.80
floor. Since every genuine read of a printed board scores 1.00 and the room-text
decodes scored ≤0.88, `LIVE_MIN_CONFIDENCE` sits at 0.95: margin on both sides.
It also filters near-miss misreads. The engine's own calibrated floor is left
exactly as it was — this is an operating point for a webcam in a room, not a
correction to the recogniser.

**The preview is not mirrored**, which is the opposite of the usual webcam
default and deliberate: the subject here *is* text, and mirrored, the board
reads backwards on the projector while the recognised plate beside it reads
forwards — so an audience cannot check one against the other, which is the whole
reason to show the video.

---

## Enrolment — off by default

```bash
livecam.bat --enrol CAM-20
```

A locked read is then written to `data/godseye.db` as an operator-attested
sighting through `core/enrol.py` — the module built for exactly this, "an
operator standing in for a camera that is not there". Enrol the same board at a
second junction and the trajectory engine has a leg to draw, so the live read
flows into the map, the clone check and the alert rules like any camera capture.

Everything that could embarrass this live is `core/enrol.py`'s business, not
this package's: the confidence floor, the bound on correcting a misread, and the
timestamp arithmetic that stops the platform accusing its own demo vehicle of
being a cloned plate. It fires once per lock, not once per frame.

Without `--enrol` (or `GODSEYE_LIVE_ENROL=1`) the platform's database is never
opened at all.

---

## If something is wrong on the day

| Symptom | First thing to try |
|---|---|
| Reads refused, board clearly visible | Lower the floor: `set GODSEYE_LIVE_MIN_CONF=0.85`. The on-screen line already tells you what it saw and at what confidence. |
| "anpr_ocr … CRNN is being used instead" | You are not in `.venv-paddle`. See PADDLE_SETUP.md. The CRNN reads real boards at about 1%. |
| Detector picking up room text | `livecam.bat --aim-only` — skips detection entirely, reads only the brackets. Fastest and most robust. |
| Everything slow, ~1 s per read | Another CUDA context. Restart the API with `GODSEYE_DEVICE=cpu`. |
| Camera will not open | Close Teams/Zoom/the Camera app, or `--camera 1`. |
| Video choppy | Check the printed camera fps line. If it is ~10, the backend probe picked a slow one — try `--imgsz 480`. |
| Nothing detected, board too far | Fill the brackets. The board must be flat and under ~12° of tilt. |

---

## Files

| File | What it does |
|---|---|
| `settings.py` | Every tunable, in its own `GODSEYE_LIVE_*` namespace |
| `capture.py` | Webcam thread; single-slot latest frame; backend chosen by probe |
| `reader.py` | The two-stage cascade and the "is this even a plate" gate |
| `vote.py` | Rolling confidence-weighted vote and the lock |
| `engine.py` | Ties them together; the headless API both front-ends use |
| `overlay.py` | The annotated frame, shared by both front-ends |
| `window.py` | Native OpenCV window |
| `server.py` | FastAPI + MJPEG on :8503 |
| `makeboard.py` | Printable A4 board, rendered by `anpr/plates.py` itself |
