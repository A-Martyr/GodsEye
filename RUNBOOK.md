# Runbook — running the full demo

`RUN.md` gets you a control room in two minutes. This is the full demo: which
reader to run, the reasoning behind each flag, and what to do when something
will not start.

Where a number appears it was measured, not estimated.

---

## 1. What you are starting

| Service | Port | What it is for |
|---|---|---|
| **API + control room** | 8000 | the control room (`web/`, served at `/`), the simulated camera network, the incident injectors, the judge console (`/j`), the API (`/docs`) |
| Phone junction cameras (optional) | 8504 | `phonecam.bat` — phones on a real road as cameras, read by the real-plate reader |
| Security Console (optional) | 8506 / 8507 | `registry.bat` — the mock vehicle registry and its console, a separate package |

The control room is plain HTML, CSS and JS that the API serves itself, so the
core demo is **one process**.

### Which reader

The API runs one plate reader, chosen at start-up with `GODSEYE_OCR_BACKEND`:

- **`crnn`** (the default) — trained on this repo's synthetic camera model. The
  right reader for the simulated feed and the *Read a plate* camera scenarios.
- **`anpr_ocr`** — PP-OCRv5, pretrained on 558,000 real Indian plates, driven by
  this repo's decoder. The right reader for a **photograph**, and where the 91.0%
  and 91.5% figures come from. It needs the paddle virtualenv (§2).

Upload a real photo to a CRNN API and you will get a poor read; capture a
synthetic scenario on an `anpr_ocr` API and you will also get a poor read. Pick
the one that matches what you are about to show.

### Why the GPU goes to one process only

This machine has a **GTX 1650 with 4 GB**. When more than one process holds a
CUDA context on it they do not share gracefully — they starve each other:

> With the API running on the GPU, `python tests.py` took **19 minutes** instead
> of about 90 seconds, and the paddle reader's warm-up reported **1190 seconds**
> — roughly a 50× starvation.

So when `phonecam.bat` (which runs the real-plate reader) is part of the demo,
pin the API to the CPU with `GODSEYE_DEVICE=cpu` and let the phones' reader have
the card. `config.torch_device()` resolves `GODSEYE_DEVICE` against what is
installed and falls back to CPU rather than raising, so asking for a GPU can
never be the thing that stops the platform running.

---

## 2. One-time prerequisites

```bash
py -3.13 --version          # 3.11+ for the main platform
pip install -r requirements.txt
```

The real-plate reader needs its **own** virtualenv, because `paddlepaddle`
publishes no wheels for Python 3.13+ and because importing paddle into a process
that later loads a torch model corrupts torch's tensors — a binary symbol clash
between the two frameworks' compiled extensions.

```
.venv-paddle\Scripts\python.exe     # Python 3.12 + paddle + a CPU-only torch
```

`PADDLE_SETUP.md` has the build steps (Windows, Linux and macOS). Verify it in
one line:

```bash
./.venv-paddle/Scripts/python.exe -c "import paddle; print(paddle.device.get_device())"
```

`gpu:0` on a CUDA machine; `cpu` on macOS or without CUDA — the reader still
works there, about 10× slower.

And a database:

```bash
py -3.13 seed.py --hours 84
```

Roughly a minute. Only needed once, or when the alert queue has gone stale (§6).

---

## 3. Starting it — the shell matters

**On Windows the terminal is usually `cmd.exe`.** The `VAR=value command` prefix
used in these docs is POSIX syntax and does nothing in cmd — it silently runs
without the variable set.

### cmd.exe

**Synthetic demo (default reader)**
```
cd /d "<GodsEye folder>"
py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Real photographs**
```
cd /d "<GodsEye folder>"
set GODSEYE_OCR_BACKEND=anpr_ocr
.venv-paddle\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **`set` persists for the life of the window.** Start each variant in a *new*
> window, or a variable from the last run leaks into the next one.

### Git Bash, WSL, Linux, macOS

```bash
cd "<GodsEye folder>"
py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000                                   # synthetic demo
GODSEYE_OCR_BACKEND=anpr_ocr ./.venv-paddle/Scripts/python.exe -m uvicorn api.main:app --port 8000   # real photographs
```

(On Linux and macOS the virtualenv's interpreter is `.venv-paddle/bin/python`.)
Here the prefix scopes to the one command, so no window discipline is needed.

### PowerShell

```powershell
$env:GODSEYE_OCR_BACKEND = "anpr_ocr"
.venv-paddle\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Same persistence caveat as cmd — `$env:` lives for the session.

`--host 0.0.0.0` is what lets a judge's phone on the same network reach the
console; without it the API accepts connections from this machine only.

---

## 4. The .bat files

| Script | What it does |
|---|---|
| `demo.bat` | checks the environment, seeds if needed, starts the API on 8000 with `GODSEYE_DEVICE=cpu`, opens the control room |
| `api.bat` | the API (and so the control room) alone on 8000, default reader |
| `seed.bat` | builds the database |
| `check.bat` | runs `demo_check.py` — worth running on a new machine |
| `phonecam.bat` | phone junction cameras on 8504 |
| `livecam.bat` / `livecam_web.bat` | a webcam as a junction camera |
| `registry.bat` | the optional Security Console (8506) and registry service (8507) |
| `share.bat` | the public judge link through a Cloudflare tunnel (§7) |
| `stop.bat` | stops everything by port, **and `cloudflared`** |

For real photographs there is no one-click script: start the API by hand as in
§3, so the reader choice is explicit.

---

## 5. Verifying it actually came up

A port answering is not the same as a service working:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Look for `"live_feed": true`, `"cameras": 35`, `"ocr_engine_loaded": true`, and a
`simulator.captures` count that **increases** when you run it again. A zero that
stays zero means the feed is not ticking.

The reader warms in the background as the API starts, so the first read does
not land on whoever clicks first. Its log line says which reader it is and where
it runs:

```
[anpr.ocr] reader warm in 64.0s (anpr_ocr, paddle:gpu:0)
```

About a minute for `anpr_ocr`, once per process; after that a whole-photograph
read is ~108 ms warm on the GTX 1650. Start the API a few minutes before you
need it. `GODSEYE_WARM=0` turns the warm-up off.

You will also see, repeatedly:

```
Couldn't access the Hub to check for update but local file already exists.
```

That is **expected and desirable**. `HF_HUB_OFFLINE=1` is set automatically once
the weights are on disk, which stops huggingface_hub round-tripping to the
network on every load — it took reader start-up from 50 s to 6 s, and it is what
makes the platform survive venue wi-fi.

- Control room → http://localhost:8000/
- API docs → http://localhost:8000/docs
- Judge console → http://localhost:8000/j

---

## 6. Before a demo

```bash
py -3.13 tests.py
```

**101 checks.** All must pass. It points the database at a scratch temp file, so
it never touches `data/godseye.db`.

Then check the alert queue is not stale:

```bash
curl -s "http://127.0.0.1:8000/api/alerts/summary?hours=6"
```

The control room's Alerts panel shows the last 6 hours of the data. If the
newest alert is older than that the panel renders empty, which reads as a broken
platform rather than an old database. Reseed if so:

```
stop.bat
py -3.13 seed.py --hours 84
```

Then restart and test the centrepiece twice:

```bash
curl -s -X POST http://127.0.0.1:8000/api/sim/inject/clone
```

A cloned-plate alert should appear in the control room within about two
seconds, naming both junctions and the implied speed. Two runs, because it
depends on seeded data.

---

## 7. The judge link

In the control room: **Insights › Incidents › Judge console**, with a QR code
for the link. Three routes:

- **Same network** — `http://<your-lan-ip>:8000/j`, the link the panel shows.
  This is the one a judge standing at your table uses; scan the QR rather than
  typing four numbers, a port and a path on a phone keyboard.
- **Anywhere** — run `share.bat`. `cloudflared` dials out and returns an
  `https://….trycloudflare.com` address in about five seconds. No account, no
  firewall change, no port forwarding. **The address is different every run.**
- **No network** — `http://127.0.0.1:8000/j` on the laptop itself.

The LAN address changes with the network you are on, so **re-read it from the
panel on the day** rather than from any note.

A public tunnel publishes the judge console on the internet for as long as it
runs. `stop.bat` kills it.

---

## 8. Stopping

```
stop.bat
```

It goes by listening port rather than window title, and it also kills
`cloudflared` — which matters, because a quick tunnel listens on no port at all
and is structurally invisible to a port-based kill. Left running it keeps a
public address pointing at this laptop long after the demo.

By hand:

```bash
netstat -ano | findstr ":8000"
taskkill /PID <pid> /F
```

---

## 9. When something will not start

| Symptom | Cause | Fix |
|---|---|---|
| `address already in use` on 8000 | an older API is still holding it | `stop.bat`, or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |
| A real photo reads poorly | the API is running the CRNN | restart it with `GODSEYE_OCR_BACKEND=anpr_ocr` from `.venv-paddle` (§3) |
| Everything is slow, tests crawl | two processes on the 4 GB GPU | pin the API with `GODSEYE_DEVICE=cpu` (§1) |
| Alerts panel empty | the database is older than the 6-hour window | reseed (§6) |
| `'grep' is not recognized` | a POSIX command in `cmd.exe` | run it in Git Bash, or use `findstr` |
| Judge link works on the laptop, not on a phone | venue wi-fi blocks client-to-client, or the firewall | use the public tunnel, or hand over the laptop |
| First read takes a minute | paddle building the model | §5 — start the API early |

### A note on GPU clocks

If the card ever sits at **300 MHz under 100% load** with a power limit reading
`0.03 W`, that is a wedged driver state, not a setting. It is not fixable from
`nvidia-smi` — `-pl` reports "not supported" on this card because the control
genuinely does not exist. **A full reboot cleared it**, taking throughput from
14.3 to 84.7 matmuls/s and the reader from 405 ms back to 108 ms. 300 MHz at
*idle* (pstate P8) is correct and expected.

---

## Quick reference

```
# synthetic demo
py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# real photographs (cmd.exe)
set GODSEYE_OCR_BACKEND=anpr_ocr  &&  .venv-paddle\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# verify
curl -s http://127.0.0.1:8000/api/health
py -3.13 tests.py                       # expect: 101 passed, 0 failed

# stop
stop.bat
```

| | |
|---|---|
| Control room | http://localhost:8000/ |
| API docs | http://localhost:8000/docs |
| Judge console | `http://<lan-ip>:8000/j` — read it from Insights › Incidents |
