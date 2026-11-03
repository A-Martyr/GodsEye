# Runbook — starting all three services

`RUN.md` gets you a control room in two minutes. This is the full demo: the API,
the control room, and the real-photograph reader on the GPU, all at once, with
the reasoning behind each flag and what to do when one of them will not start.

Everything here has been run on this machine. Where a number appears it was
measured, not estimated.

---

## 1. What you are starting, and why it is three processes

| Service | Port | Device | What it is for |
|---|---|---|---|
| **API + live feed** | 8000 | **CPU** | the simulated camera network, incident injectors, judge console |
| **Control room** | 8501 | **CPU** | the operator dashboard — map, tracking, alerts, analytics |
| **Real-photograph reader** | 8502 | **GPU** | uploading a real photo of a real plate |

Two dashboards, not one, and that is deliberate. They run the same
`dashboard/app.py` with a different recogniser behind it:

- **8501** uses the CRNN, trained on this repo's synthetic camera model. It is
  the right reader for the ANPR-engine tab's *generated* camera scenarios.
- **8502** uses `anpr_ocr` — PP-OCRv5, pretrained on 558,000 real Indian plates.
  It is the right reader for a **photograph**, and where the 91.0% figure comes
  from.

Upload a real photo to 8501 and you will get a poor read. Capture a synthetic
scenario on 8502 and you will also get a poor read. Each is strong where the
other is weak; keeping both open is what lets you show either.

### Why the GPU goes to one process only

`GODSEYE_DEVICE=cpu` on the API and the control room is not caution, it is a
measured requirement. This machine has a **GTX 1650 with 4 GB**. When more than
one process holds a CUDA context on it they do not share gracefully — they
starve each other:

> With the API running on the GPU, `python tests.py` took **19 minutes** instead
> of about 90 seconds, and the paddle reader's warm-up reported **1190 seconds**
> — roughly a 50× starvation.

So: the two CPU services never touch the card, and 8502 gets it to itself.
`config.torch_device()` resolves `GODSEYE_DEVICE` against what is installed and
falls back to CPU rather than raising, so asking for a GPU can never be the
thing that stops the platform running.

---

## 2. One-time prerequisites

You have all of these already; this is for a fresh machine.

```bash
py -3.13 --version          # 3.11+ for the main platform
pip install -r requirements.txt
```

The GPU reader needs its **own** virtualenv, because `paddlepaddle` publishes no
wheels for Python 3.13+ and because importing paddle into a process that later
loads a torch model corrupts torch's tensors — a binary symbol clash between the
two frameworks' compiled extensions.

```
.venv-paddle\Scripts\python.exe     # Python 3.12 + paddle + a CPU-only torch
```

`PADDLE_SETUP.md` has the build steps. Verify it in one line:

```bash
./.venv-paddle/Scripts/python.exe -c "import paddle; print(paddle.device.get_device())"
```

Expect `gpu:0`. If it prints `cpu`, the reader still works — it is just ~10×
slower and you will not see the 108 ms figure.

And a database:

```bash
py -3.13 seed.py --hours 84
```

Roughly a minute. Only needed once, or when the alert queue has gone stale
(§7).

---

## 3. Starting all three — the shell matters

**Your terminal is `cmd.exe`.** The `VAR=value command` prefix used throughout
these docs is POSIX syntax and does nothing in cmd — it will silently run
without the variable set, which for 8502 means the GPU is never used and for the
API means it takes the GPU it was supposed to leave alone.

Use whichever set matches the window you are in.

### cmd.exe — one window per service

**Window 1 — API**
```
cd /d "<GodsEye folder>"
set GODSEYE_DEVICE=cpu
py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Window 2 — control room**
```
cd /d "<GodsEye folder>"
set GODSEYE_DEVICE=cpu
py -3.13 -m streamlit run dashboard/app.py --server.port 8501
```

**Window 3 — real-photograph reader, on the GPU**
```
cd /d "<GodsEye folder>"
set GODSEYE_OCR_BACKEND=anpr_ocr
.venv-paddle\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8502
```

> **`set` persists for the life of the window.** Window 3 must be a *new* one.
> Reusing window 1 or 2 carries `GODSEYE_DEVICE=cpu` into it, paddle obeys it,
> and the reader silently runs on the CPU — the single easiest way to lose the
> GPU without any error appearing.

### Git Bash / WSL

```bash
cd "<GodsEye folder>"

GODSEYE_DEVICE=cpu py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
GODSEYE_DEVICE=cpu py -3.13 -m streamlit run dashboard/app.py --server.port 8501 --server.headless true &
GODSEYE_OCR_BACKEND=anpr_ocr ./.venv-paddle/Scripts/python.exe -m streamlit run dashboard/app.py --server.port 8502 --server.headless true &
```

Here the prefix scopes to the one command, so no window discipline is needed.

### PowerShell

```powershell
$env:GODSEYE_DEVICE = "cpu"
py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Same persistence caveat as cmd — `$env:` lives for the session.

---

## 4. Why not just use the .bat files

They work, and they do not cover this.

| Script | What it does | Gap |
|---|---|---|
| `demo.bat` | checks the environment, seeds if needed, starts **8000 + 8501**, opens the browser | **never starts 8502**, and sets no `GODSEYE_DEVICE` — so both take the GPU |
| `api.bat` | API alone on 8000 | no device pin |
| `dashboard.bat` | control room alone on 8501 | no device pin |
| `dashboard_paddle.bat` | **8502 on the paddle venv, `GODSEYE_OCR_BACKEND=anpr_ocr`** | correct, use it |
| `seed.bat` | builds the database | |
| `check.bat` | runs `demo_check.py` | worth running on a new machine |
| `share.bat` | starts the Cloudflare tunnel | |
| `stop.bat` | kills 8000/8501/8502 **and `cloudflared`** | |

So the closest one-click path is:

```
demo.bat              rem API + control room, but both on the GPU
dashboard_paddle.bat  rem the real-photograph reader
```

That works for a quick look. For a demo, start them by hand with the device
pins — three processes contending for 4 GB is exactly the case §1 measured.

---

## 5. Verifying it actually came up

A port answering is not the same as a service working. Run this:

```bash
py -3.13 -c "
import time, urllib.request
t = {'API  8000':'http://127.0.0.1:8000/api/health',
     'CTRL 8501':'http://127.0.0.1:8501/_stcore/health',
     'READ 8502':'http://127.0.0.1:8502/_stcore/health'}
pend=dict(t); t0=time.time()
while pend and time.time()-t0 < 150:
    for n,u in list(pend.items()):
        try:
            with urllib.request.urlopen(u, timeout=2) as r:
                if r.status==200: print(f'  {n}  up after {time.time()-t0:5.1f}s'); pend.pop(n)
        except Exception: pass
    time.sleep(1.5)
for n in pend: print(f'  {n}  NOT UP')
"
```

Then confirm the API is actually simulating, not merely listening:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Look for `"live_feed": true`, `"cameras": 35`, and a `simulator.captures` count
that **increases** when you run it again. A zero that stays zero means the feed
is not ticking.

Confirm the reader really has the GPU:

```bash
./.venv-paddle/Scripts/python.exe -c "import paddle; print(paddle.device.get_device())"
nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader
```

Once 8502 has warmed, `nvidia-smi` should show a few hundred MB in use. **0 MiB
means nothing has claimed the card** — usually the `set` leak from §3.

- Control room → http://localhost:8501
- Real-photo reader → http://localhost:8502
- API docs → http://localhost:8000/docs

---

## 6. The paddle warm-up — the thing that looks broken and is not

**8502 shows no GPU use and does nothing until a browser opens it.**

`_warm_reader()` is a Streamlit cached resource, so it fires when a real session
connects. `curl` cannot trigger it — Streamlit runs the script over a websocket,
not the HTTP GET. Then:

```
[anpr.ocr] reader warm in 64.0s (anpr_ocr, paddle:gpu:0)
```

**About a minute, once per process.** After that a whole-photograph read is
~108 ms warm.

So: **open the 8502 tab several minutes before you need it.** The warm-up runs
in the background and by the time anyone asks, reads are instant. Discovering
this in front of an audience is a minute of silence you cannot fill.

You will also see, repeatedly, in the 8502 log:

```
Couldn't access the Hub to check for update but local file already exists.
```

That is **expected and desirable**. `HF_HUB_OFFLINE=1` is set automatically once
the weights are on disk, which is what stops huggingface_hub round-tripping to
the network on every load — it took reader startup from 50 s to 6 s and it is
what makes the platform survive venue wi-fi.

---

## 7. Before a demo

```bash
py -3.13 tests.py
```

**81 checks.** All must pass. It points the database at a scratch temp file, so
it never touches `data/godseye.db`.

Then check the alert queue is not stale:

```bash
curl -s "http://127.0.0.1:8000/api/alerts/summary?hours=24"
```

The dashboard's Alerts tab uses a 24-hour window. If the newest alert is older
than that the queue renders empty, which reads as a broken platform rather than
an old database. Reseed if so:

```
stop.bat
py -3.13 seed.py --hours 84
```

Then restart and test the centrepiece twice:

```bash
curl -s -X POST http://127.0.0.1:8000/api/sim/inject/clone
```

A critical clone alert should appear within about two seconds, naming both
junctions and the implied speed. Two runs, because it depends on seeded data.

---

## 8. The judge link

Sidebar → **Judge link**. Three routes, and the panel says which it can stand
behind.

- **Anywhere** — *Create a public link*, or `share.bat`. `cloudflared` dials out
  and returns an `https://….trycloudflare.com` address in about five seconds.
  No account, no firewall change, no port forwarding. **The address is different
  every run.**
- **Same network** — `http://<your-lan-ip>:8000/j`, with a QR code above it.
  This is the one a judge standing at your table uses; scan it rather than
  typing four numbers, a port and a path on a phone keyboard.
- **No network** — `http://127.0.0.1:8000/j` on the laptop itself.

The LAN address changes with the network you are on. It has been
`192.168.29.204` and `10.45.68.26` on this machine in the last two days, so
**re-read it from the panel on the day** rather than from any note.

A public tunnel publishes the judge console on the internet for as long as it
runs. `stop.bat` kills it; so does *Stop sharing*.

---

## 9. Stopping

```
stop.bat
```

It goes by listening port rather than window title, and it also kills
`cloudflared` — which matters, because a quick tunnel listens on no port at all
and is structurally invisible to a port-based kill. Left running it keeps a
public address pointing at this laptop long after the demo.

By hand:

```bash
netstat -ano | findstr ":8000 :8501 :8502"
taskkill /PID <pid> /F
```

---

## 10. When something will not start

| Symptom | Cause | Fix |
|---|---|---|
| `Port 8501 is not available` (exit 4) | an older Streamlit is still holding it | `stop.bat`, or `netstat -ano \| findstr :8501` then `taskkill /PID <pid> /F` |
| Reader is slow, `nvidia-smi` shows 0 MiB | `GODSEYE_DEVICE=cpu` leaked into the 8502 window | start 8502 in a **fresh** window (§3) |
| Everything is slow, tests crawl | two or three processes on the 4 GB GPU | pin the API and 8501 to CPU |
| A theme or `.streamlit/config.toml` change did nothing | **config is read at startup**; only source hot-reloads | restart that Streamlit process |
| `'grep' is not recognized` | a POSIX command in `cmd.exe` | run it in Git Bash, or use `findstr` |
| Judge link works on the laptop, not on a phone | venue wi-fi blocks client-to-client, or the firewall | use the public tunnel, or hand over the laptop |
| Dashboard stalls for a few seconds | a fragment rerun, or a first map draw | wait; or hamburger → Rerun |
| First read on 8502 takes a minute | paddle building the model | §6 — open the tab early |

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
# three windows, cmd.exe
set GODSEYE_DEVICE=cpu   &&  py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
set GODSEYE_DEVICE=cpu   &&  py -3.13 -m streamlit run dashboard/app.py --server.port 8501
set GODSEYE_OCR_BACKEND=anpr_ocr  &&  .venv-paddle\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8502

# verify
curl -s http://127.0.0.1:8000/api/health
py -3.13 tests.py                       # expect: 81 passed, 0 failed

# stop
stop.bat
```

| | |
|---|---|
| Control room | http://localhost:8501 |
| Real-photo reader | http://localhost:8502 |
| API docs | http://localhost:8000/docs |
| Judge console | `http://<lan-ip>:8000/j` — read it from the sidebar |
