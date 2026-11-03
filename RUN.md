# GodsEye — running it

City-wide ANPR intelligence platform. SIH 2026, Team Prometheus.

## Two minutes to a running control room

```bash
pip install -r requirements.txt

python seed.py                          # ~6 h of simulated city traffic into SQLite (~65 s)
streamlit run dashboard/app.py          # the control room -> http://localhost:8501
```

That is enough to demo. The dashboard reads the database directly and works
with the API stopped. For the live camera feed and the incident injectors, in a
second terminal:

```bash
uvicorn api.main:app --port 8000        # http://localhost:8000/docs
```

Python 3.11+. No GPU, no internet connection and no dataset download is needed
for any of the above.

## Letting someone else release the incident

The point of the judge console is that an outsider picks the moment. The
control room is never told, so the platform has to find the clone on its own.
Open the **Judge link** panel in the sidebar; it offers three routes, and shows
which ones it can actually stand behind.

**Anywhere in the world.** Press *Create a public link*, or run `share.bat`.
`cloudflared` dials out to Cloudflare and comes back with an `https://…
.trycloudflare.com` address about five seconds later. Send that to anyone, on
any network — a friend in another city, a phone on mobile data. Because the
connection is outbound, nothing has to be opened on the firewall and nothing
forwarded on the router, which are the two things you cannot arrange at a venue
you walked into an hour ago. It needs no Cloudflare account.

Two things to hold on to: the address is **different every run**, so the link
must be re-sent after a restart, and it stays live until you press *Stop
sharing*, close the `share.bat` window, or run `stop.bat`.

**Same network.** The `http://192.168.x.x:8000/j` links, for a judge standing
at the table. No internet needed. They only work if the other device is on this
network, and some venue Wi-Fi blocks its clients from reaching each other,
which nothing on this side can fix.

**No network at all.** `http://127.0.0.1:8000/j` on the demo laptop, handed
over. It always works, and the outsider still chooses the moment.

### Why the public link carries a token

A tunnel publishes an **origin**, not a path — everything on port 8000 would go
with it. Two things narrow that back down:

- `api/main.py` refuses every path except the judge console for requests that
  arrive on a public hostname, so `/docs`, the sightings API and the rest of the
  platform are not reachable through the tunnel.
- The console itself needs the token that is baked into the link. Nobody types
  it; it rides in the URL and in the QR code. *New link* revokes it, which kills
  every link already sent.

Requests arriving on a LAN address skip both checks, so the in-the-room demo is
exactly as it was. Set `GODSEYE_JUDGE=0` to switch the console off entirely.

If `nslookup` on this machine cannot resolve the trycloudflare address, that is
usually the local ISP resolver rather than a broken tunnel — check it against
`1.1.1.1` before assuming the link is dead. The people you sent it to are on
different resolvers and are generally unaffected.

## What to try first

- **Live network** — the Kolkata camera network, roads coloured by congestion.
- **Track a plate** — hit "Pick a busy vehicle", then retype the plate with one
  character wrong. It still finds the vehicle: search is confusion-aware and
  charges 0.4 rather than 1.0 for a substitution the OCR engine actually makes.
- **ANPR engine** — pick a camera scenario (try `monsoon` or `far_lane`) and
  press Capture & read.

## Checking it yourself

```bash
python tests.py                                  # 58-check end-to-end self-check
python -m viz.charts                             # regenerate the 13 charts in charts/
python -m anpr.benchmark --samples 100 --burst 5 # re-measure the reader (~7 min)
python -m anpr.camera_sheet                      # see what the camera model produces
python -m core.repair --hours 6                  # recover refused captures
```

`python tests.py` points the database path at a scratch temp file, so it never
touches `data/godseye.db`. Run it before any demo.

## The models are here, and that matters

This bundle carries `models/crnn.pt`, `models/glyph_mlp.joblib` and
`models/yolov8n-obb-plate.pt`. **A `git clone` does not.** `models/*.joblib` is
in `.gitignore` and the YOLO detector was never committed, so a fresh clone
silently retrains the glyph classifier on first use and falls back to the
classical morphological localiser. Working from this folder avoids both.

`torch` is an optional dependency. Without it `anpr/ocr.py` falls back to the
classical segment-and-classify reader and the platform still runs:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## What this bundle cannot do

It **cannot reproduce the 91.0% real-photograph accuracy**. That number comes
from the `anpr_ocr` backend, which needs `paddlepaddle`:

- paddle has no Python 3.14 wheels, so it needs its own environment on 3.12;
- on first use it clones PaddleOCR (~2.3 GB) and downloads the HF weights;
- importing `paddle` before torch's models are loaded corrupts torch's tensors —
  a binary symbol clash between the two frameworks' compiled extensions.

**`PADDLE_SETUP.md` is the step-by-step guide to setting all of that up** —
Python 3.12, the second virtualenv, the weights, and the load-order hazard.
The real-photo corpus needed to reproduce the 91.0% is not in this public
repository (see the licence note at the end of this file); the reader itself
runs from here once paddle is set up. The
training datasets, the YOLO run directories and both virtualenvs are all left
out of this bundle deliberately — some of the datasets are CC BY-NC-ND and
should not be redistributed.

## Where to read

| File | What is in it |
|---|---|
| `README.md` | what it does, how it is built, every measured number |
| `charts/CAPTIONS.md` | the 13 judge-facing charts and what each one argues |
| `AUGMENTATION.md` | the camera model — the 8 stages and why they are in that order |
| `PADDLE_SETUP.md` | how to set up Python 3.12 + paddle for the 91% reader |
| `RUNBOOK.md` | starting all three services — the GPU split, the shell that actually works, verification, and what to do when one will not start |
| `PERFORMANCE.md` | how a read went from 46 s to 118 ms, and what was measured to prove it cost no accuracy |
| `FEATURES.md` | the seven capabilities added after the VehicleManager comparison, how each works, and what each refuses to claim |
| `FIELD_TEST.md` | nine night phone photographs: why the localiser missed them, and the one recogniser defect it exposed |
| `PHOTOMETRIC.md` | polarity, colour and low light — the measured 21.7-point deficit on inverted plates, and the plan for it |
| `CEILING.md` | how far the reader can be pushed with no retraining — erasure decoding, burst super-resolution, RTO grammar, and where VLMs are actually worth using |
| `REPAIR_VLM_PLAN.md` | raising repair recall with a second recogniser — the measured ceiling, and why it must be a separate channel |
| `INTERNALS.md` | the four capability modules in depth — data structures, the arithmetic, every threshold, and the failure modes each one refuses to hide |
| `share.html` | a single self-contained page — open it in any browser, no install |

## Licence note on the real-photo corpus

The 91.0% real-photograph figures were measured on a corpus that is kept out of
this public repository on purpose:

- `indian-number-plates-dataset/` (Dataclusterlabspvtltd, Hugging Face) is
  **CC BY-NC-ND 4.0**, and the plate crops cut from it count as modified copies,
  which that licence does not allow to be redistributed.
- `dataset_v1_bbox/` is a Roboflow export under its own licence.
- The crops and frames show real vehicles' registrations, which is personal data.

The corpus, its hand-made labels and the scripts that use it stay in the team's
private repository. Everything else, including the 101-check self-test, runs
from this one.
