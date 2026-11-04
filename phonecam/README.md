# Phones as junction cameras

Two or three phones, each standing at a different junction of the real Kolkata
camera graph, streaming to one laptop. Carry a plate-format board past them and
the platform reads it at each junction, records a sighting, and reconstructs the
journey between them — the leg, the road, the distance, the implied speed, the
compass heading.

That last part is the point. Reading a plate is a feature; **reconstructing
where the vehicle went is the product**, and it is the thing the SIH brief
actually asks for. A single webcam can only ever demonstrate the first.

Separate from `livecam/` and from the platform: imported by nothing in `anpr/`,
`core/`, `api/` or `sim/`. It reuses `livecam`'s reader and vote,
and writes sightings only through `core/enrol.py`.

---

## Run it

```bash
python -m livecam.makeboard WB24AB1234 --pdf   # print a board
phonecam.bat                                   # start the wall + phone endpoints
```

The console prints a link per junction. Open **`/links`** on the laptop for the
same thing with QR codes, then:

1. Scan one link per phone (or type it).
2. Accept the certificate warning — once per phone.
3. Press **Start camera**, prop the phone up facing where the board will be.
4. Open the wall on the laptop and carry the board from one phone to the next.

Default junctions are `CAM-20,CAM-21,CAM-28` — Esplanade → Park Street →
Rashbehari. Chosen so one setup demonstrates everything:

| pair | km | adjacent | what it shows |
|---|---|---|---|
| CAM-20 ↔ CAM-21 | 1.4 | yes | a clean journey along a real road |
| CAM-21 ↔ CAM-28 | 4.7 | yes | clears the 3 km clone guard — `--realtime` fires the clone rule here |
| CAM-20 ↔ CAM-28 | 6.1 | **no** | the route is inferred, and the trajectory says so |

---

## The certificate warning is not optional

A browser refuses `getUserMedia` on any origin that is not a secure context,
and a LAN address over plain `http` is not one. Without TLS the phones cannot be
cameras at all — no amount of care on the page changes that.

So the laptop serves https with a certificate it generates for its own LAN
address (`phonecam/certs.py`, via the `openssl` that ships with Git for
Windows). Each phone shows one warning, accepted once and then remembered.

If a phone still refuses the camera, the page says which of the two causes it
is: `NotAllowedError` means the permission was declined, `SecurityError` means
the page is not actually on https.

---

## Timestamps: the choice that decides the story

Two phones on one table are metres apart. The junctions they represent are
kilometres apart. What gets written for the time between them decides what the
platform then says about it.

**Default — plausible.** `core/enrol.suggest_ts()` spaces the sightings by the
real road distance at an ordinary city speed, so carrying a board across a room
produces a journey the platform reads as a journey: *Esplanade → Park Street
Crossing, 1.4 km, 4.0 min, 21 km/h, heading S.*

**`--realtime` — the real clock.** Between junctions kilometres apart that
implies an impossible speed, which is exactly the point: it is how you
demonstrate the clone detector catching a registration that cannot be in two
places. Measured here, carrying the board from Park Street to Rashbehari raised:

```
CLONE - WB24AB1234 implies 10752 km/h between Park Street Crossing
        and Rashbehari Crossing - probable cloned plate
```

`--realtime` also passes `allow_implausible`, because `core/enrol.py` would
otherwise refuse to write a leg it knows is impossible — which is the correct
default and the wrong one here.

---

## How it works

**One recogniser, many junctions.** There is one GPU and one paddle context, so
there is exactly one inference thread. It cycles the junctions, reading whichever
has a frame waiting. That shares the card fairly with no locking, bounds total
load however many phones connect, and costs nothing for a junction nobody is
pointing at. Each junction keeps its own reader and its own vote — a vote is a
claim about one camera's view — while sharing the weights underneath, because
`anpr.ocr` caches its engine per backend.

**Phones pace themselves against an ack.** The page sends a frame and waits for
the server's acknowledgement before capturing the next, on top of a client-side
cap (8/s by default). A slow link therefore cannot build a backlog of frames
describing a moment that has already passed. The ack is sent on receipt rather
than after the read: gating it on inference would tie one phone's frame rate to
every other junction's share of the GPU.

**Frames vote, per junction.** Same mechanism as the single-camera demo — three
agreeing frames lock the reading. It raises accuracy and stops the number
flickering between near-misses.

**A lock is one sighting.** A board left in view relocks every few seconds;
`RELOCK_S` (25 s) keeps one pass through a junction to one row.

---

## Switching cameras off

From the control room's **Live cameras** tab, or over HTTP. Three different
things, because they answer three different questions:

| Control | What it does | HTTP |
|---|---|---|
| **Turn off** (one junction) | Stops reading it and releases its camera. Pushed frames are dropped before they are decoded; a pulled camera is closed, so the device is genuinely free. Its vote is cleared so a stale lock cannot sit on the tile looking current. | `POST /api/cameras/{id}/off` |
| **Turn all cameras off** | The same, for every junction. The GPU then does nothing. | `POST /api/cameras/off` |
| **Stop recording** | Keeps reading and showing, writes nothing to the database. For rehearsing without leaving sightings behind. | `POST /api/recording/off` |
| **Shut down service** | Stops the process. Phones lose their connection, the GPU is released, and the control room simply stops receiving their reads. | `POST /api/shutdown` |

`on` in place of `off` reverses each. A junction switched back on recovers in
about a second for a phone, and a few seconds for a pulled camera — that delay
is the physical device being reopened, which is the point of having released it.

Nothing here touches data already recorded: a live read is an ordinary
sighting, and switching the camera that produced it off does not unmake it.

## Two listeners, and why

| port | scheme | bound to | for |
|---|---|---|---|
| 8504 | https | all interfaces | the phones |
| 8505 | http | **127.0.0.1 only** | the junction tiles, viewed on this laptop |

The phones need https or the browser refuses `getUserMedia`. A page on this
laptop that embeds the tiles needs the opposite: an `<img>` pointed at an https
origin whose certificate the browser has never accepted is blocked or warned
about on every load.

So the same app also answers on loopback over plain http, serving each
junction's annotated MJPEG stream; a browser then holds one connection per tile
and receives frames as they are produced. Loopback only, deliberately: nothing
on the LAN needs this, and the phones have their own encrypted way in. The
control room itself does not embed the tiles: phone reads reach it as ordinary
sightings, through the database.

## Without phones

Every junction can be pulled from a URL instead — an IP-Webcam app's MJPEG
stream, an RTSP camera, a recorded video file, or `webcam:0` for the laptop's
own camera. This is for rehearsing the wiring when no phone is to hand; the
phones are the demo, and their sensors are better.

```bash
phonecam.bat --pull CAM-20=http://192.168.1.7:8080/video ^
             --pull CAM-21=C:\clips\junction2.mp4
```

---

## Only one phone connects, the others cannot

Almost always the network, not the software. **A phone-hosted hotspot lets the
hosting phone reach the laptop and blocks every other client from doing so** —
clients on a hotspot are isolated from each other by design on most Android and
iOS builds, and there is usually no setting to turn it off. The symptom is
exactly one phone working: the one providing the hotspot.

`GET /api/connections` settles it in one look. It lists every device that has
reached the server, with its address. **If a phone never appears there, its
packets are not arriving at all** and nothing in this code can change that. If
it appears and then fails, the problem is the certificate or the camera
permission, and the phone page says which.

In order of preference:

1. **Let the laptop host the Wi-Fi.** Windows Settings → Network & Internet →
   Mobile hotspot. Every phone then connects *to the laptop*, which makes each
   one a client of it rather than a peer — and host-to-client is never
   isolated. Full LAN speed, a read locks in well under a second, and no
   internet is needed for any of it.
2. **A normal Wi-Fi router.** Same topology, same result; home routers do not
   isolate clients by default.
3. **`phonecam.bat --tunnel`.** Publishes through a Cloudflare quick tunnel, so
   the phones reach the laptop over the internet and the local network stops
   mattering. It also removes the certificate warning, because Cloudflare
   presents a real one. The cost is bandwidth: a quick tunnel is best-effort,
   and phones are automatically dropped to 640px at 5 fps to fit it. Measured
   with three phones, that is ~3 fps each and a read locks in about **9
   seconds**, against well under one on a LAN. It works, and it is visibly
   slower.

The tunnel gates itself: every link carries a token, and on the public hostname
everything except the phone pages returns 404. The address and the token both
change when the service restarts.

## If something is wrong on the day

| Symptom | First thing to try |
|---|---|
| Phone says the camera is refused | Check the URL is `https://`. Accept the certificate. `SecurityError` = not https; `NotAllowedError` = permission declined. |
| Phone cannot reach the laptop at all | Same Wi-Fi, and the network must not be client-isolated (many guest networks are). Check the printed LAN address is the one the phone can see. |
| Certificate error that will not clear | The laptop's IP changed. Delete `phonecam/certs/` and restart — it reissues for the current address. |
| "anpr_ocr … CRNN is being used instead" | Not running from `.venv-paddle`. See PADDLE_SETUP.md. |
| Everything slow | Another CUDA context. Restart the API with `GODSEYE_DEVICE=cpu`. |
| Reads refused, board clearly visible | `set GODSEYE_LIVE_MIN_CONF=0.85`. The tile already shows what it saw and at what confidence. |
| Sightings not appearing on the map | The control room reads the same database — check you did not start with `--no-enrol`. |
| A journey is flagged as a clone unexpectedly | You are in `--realtime`. That is what it does. Drop the flag. |

---

## Files

| File | What it does |
|---|---|
| `settings.py` | Its own knobs; the reading behaviour comes from `livecam.settings` |
| `certs.py` | Self-signed certificate for this machine's LAN addresses |
| `sources.py` | One junction's video — pushed from a phone, or pulled from a URL |
| `hub.py` | The shared round-robin recogniser, per-junction votes, enrolment, journeys |
| `pages.py` | The phone capture page and the operator wall |
| `server.py` | Routing, websocket ingest, per-junction MJPEG, state API |
