"""Serving the phones and the wall.

    python -m phonecam.server --junctions CAM-20,CAM-21

Its own https app on its own port. https is not decoration: a browser refuses
getUserMedia on any origin that is not a secure context, so without TLS the
phones cannot be cameras at all. See phonecam/certs.py.
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from collections import deque
from pathlib import Path

import cv2
from fastapi import (FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse)

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import reach, tunnel                                 # noqa: E402
from livecam import overlay                                    # noqa: E402
from livecam.engine import LiveState                           # noqa: E402
from phonecam import certs, pages, settings                    # noqa: E402
from phonecam.hub import JunctionHub                           # noqa: E402

app = FastAPI(title="GodsEye — phone junction cameras", docs_url=None, redoc_url=None)
hub: JunctionHub | None = None

# Every device that has reached this server, newest first. The single most
# useful thing when a phone "will not connect": it separates the two failures
# that look identical from the outside. If the phone never appears here, its
# packets are not arriving at all - a firewall, or a hotspot isolating its
# clients from each other - and nothing in this code can help. If it appears
# and then fails, the problem is the certificate or the camera permission, and
# the phone page says which.
_seen: deque = deque(maxlen=60)


def _note(request_or_ws, kind: str, detail: str = "") -> None:
    client = getattr(request_or_ws, "client", None)
    ip = getattr(client, "host", "?") or "?"
    host = request_or_ws.headers.get("host", "")
    agent = request_or_ws.headers.get("user-agent", "")[:60]
    _seen.appendleft({"ts": time.time(), "ip": ip, "kind": kind, "detail": detail,
                      "host": host, "agent": agent})
    print(f"[phonecam] {kind} from {ip} ({host}) {detail}", file=sys.stderr)


@app.middleware("http")
async def _watch_and_gate(request, call_next):
    path = request.url.path
    # Loopback is the dashboard polling status and pulling video many times a
    # second; logging that would bury the one line that matters.
    if request.client and request.client.host not in ("127.0.0.1", "::1"):
        if not path.startswith(("/stream/", "/snapshot/", "/api/state")):
            _note(request, "page", path)

    # A tunnel publishes a whole origin, not a path, so everything this port
    # serves would otherwise be public. Only what a phone actually needs is,
    # and only with the token from its own link.
    if reach.is_public_host(request.headers.get("host", "")):
        allowed = any(path == p or path.startswith(p + "/")
                      for p in ("/cam", "/links", "/ws/cam"))
        token_ok = secrets.compare_digest(
            str(request.query_params.get("k", "")), reach.token())
        if not (allowed and token_ok):
            return JSONResponse({"detail": "not found"}, status_code=404)
    return await call_next(request)


@app.get("/api/connections")
def connections():
    """Who has reached this server, and how."""
    return {"addresses": certs.lan_addresses(), "seen": list(_seen)}


# --- pages --------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def wall():
    return HTMLResponse(pages.WALL)


@app.get("/cam/{camera_id}", response_class=HTMLResponse)
def phone(camera_id: str):
    camera_id = camera_id.upper()
    j = hub.junctions.get(camera_id)
    if j is None:
        return HTMLResponse(f"<body style='font:16px system-ui;padding:24px'>"
                            f"{camera_id} is not one of this run's junctions: "
                            f"{', '.join(hub.junctions)}</body>", status_code=404)
    return HTMLResponse(pages.phone_page(camera_id, j.name, settings.PHONE_FPS,
                                         settings.PHONE_WIDTH, settings.PHONE_JPEG_Q,
                                         settings.PHONE_MAX_INFLIGHT))


@app.get("/links", response_class=HTMLResponse)
def links(request: Request):
    """Every junction's URL, with a QR, so phones are pointed by scanning."""
    base = _base(request)
    suffix = f"?k={reach.token()}" if tunnel.current_url() else ""
    rows = []
    for cid, j in hub.junctions.items():
        url = f"{base}/cam/{cid}{suffix}"
        img = (f"<img src='/qr/{cid}{suffix}' width='190' height='190'>"
               if _qr_available()
               else "<div style='color:#93a1b3;font-size:.8rem'>install "
                    "<code>qrcode</code> for a scannable code</div>")
        rows.append(f"<div class=c><div class=n>{j.name}</div>"
                    f"<div class=i>{cid}</div>{img}<code>{url}</code></div>")
    return HTMLResponse(
        "<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<style>body{background:#0f1216;color:#eef2f7;font:15px system-ui;padding:20px}"
        ".g{display:flex;flex-wrap:wrap;gap:14px}.c{background:#181d24;border:1px solid "
        "#2a323d;border-radius:13px;padding:14px;text-align:center}.n{font-weight:600}"
        ".i{color:#93a1b3;font-size:.72rem;letter-spacing:.08em;margin-bottom:8px}"
        "code{display:block;margin-top:8px;font-size:.72rem;color:#93a1b3;word-break:break-all}"
        "h1{font-size:1.1rem}p{color:#93a1b3;font-size:.85rem;max-width:640px}</style>"
        "<h1>Point a phone at each junction</h1>"
        "<p>Open one link per phone. The browser will warn about the certificate - "
        "accept it once. Then press <b>Start camera</b> and stand the phone up "
        "facing where the plate will be.</p>"
        f"<div class=g>{''.join(rows)}</div>")


def _qr_available() -> bool:
    try:
        import qrcode  # noqa: F401
        return True
    except ImportError:
        return False


@app.get("/qr/{camera_id}")
def qr(request: Request, camera_id: str, k: str = ""):
    import io
    try:
        import qrcode
    except ImportError:
        return Response(status_code=501)
    suffix = f"?k={k}" if k else ""
    q = qrcode.QRCode(box_size=8, border=2)
    q.add_data(f"{_base(request)}/cam/{camera_id.upper()}{suffix}")
    q.make(fit=True)
    buf = io.BytesIO()
    q.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _base(request: Request | None = None) -> str:
    """The address to put in a link a phone has to reach.

    Order matters. A running tunnel wins, because it is the only address that
    works from a network this laptop is not on - and on a phone-hosted hotspot
    that is every phone except the one hosting it. Otherwise the host the
    caller actually used wins, since by definition that address reached us.
    Guessing a LAN address from the adapter list is the last resort, and the
    one that was wrong whenever the machine had more than one.
    """
    public = tunnel.current_url()
    if public:
        return public
    if request is not None:
        host = request.headers.get("host", "")
        if host:
            return f"{request.url.scheme}://{host}"
    scheme = "https" if settings.HTTPS else "http"
    return f"{scheme}://{certs.lan_addresses()[0]}:{settings.PORT}"


# --- phone ingest -------------------------------------------------------
@app.websocket("/ws/cam/{camera_id}")
async def ingest(ws: WebSocket, camera_id: str):
    camera_id = camera_id.upper()
    j = hub.junctions.get(camera_id)
    if j is None:
        await ws.close(code=4004)
        return
    # http middleware does not see websockets, so the public-origin gate is
    # repeated here rather than assumed.
    if reach.is_public_host(ws.headers.get("host", "")):
        if not secrets.compare_digest(str(ws.query_params.get("k", "")),
                                      reach.token()):
            await ws.close(code=4003)
            return
    await ws.accept()
    _note(ws, "camera connected", camera_id)
    j.source.client = ws.headers.get("user-agent", "")[:80]
    hub.log("system", camera_id, f"{j.name}: a phone connected")
    try:
        while True:
            blob = await ws.receive_bytes()
            j.source.push_jpeg(blob)
            # Acknowledged on receipt rather than after the read: the phone
            # paces itself on this, and gating it on inference would tie one
            # phone's frame rate to every other junction's share of the GPU.
            v = j.vote.state()
            await ws.send_json({
                "ok": True, "pretty": (v.text and _pretty(v.text)) or "",
                "locked": v.locked, "fps": round(j.source.fps, 1),
                "infer_ms": round(j.infer_ms, 1)})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.log("system", camera_id, f"{j.name}: the phone disconnected")


def _pretty(plate: str) -> str:
    from anpr.plates import pretty
    return pretty(plate)


# --- the wall's video ---------------------------------------------------
def _tile_frames(camera_id: str):
    j = hub.junctions[camera_id]
    period = 1.0 / max(1, settings.WALL_FPS)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.WALL_JPEG_Q]
    blank = None
    while True:
        tick = time.perf_counter()
        frame = j.source.peek()
        if frame is None:
            if blank is None:
                blank = overlay._placeholder(settings.WALL_WIDTH,
                                             int(settings.WALL_WIDTH * 9 / 16))
            img = blank
        else:
            v = j.vote.state()
            state = LiveState(vote=v, boxes=j.boxes, aim_box=j.aim_box,
                              infer_ms=j.infer_ms, infer_fps=0.0,
                              camera_fps=j.source.fps, backend=hub.backend,
                              raw_text=j.raw_text, raw_conf=j.raw_conf,
                              enrol_enabled=hub.enrol_enabled,
                              enrol_camera=j.camera_id)
            # Annotate at native size and scale afterwards, so the overlay's
            # text keeps its proportions on a small tile.
            img = overlay.draw(frame, state)
            if img.shape[1] > settings.WALL_WIDTH:
                h = int(img.shape[0] * settings.WALL_WIDTH / img.shape[1])
                img = cv2.resize(img, (settings.WALL_WIDTH, h),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, params)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(buf)).encode() + b"\r\n\r\n" + buf.tobytes() + b"\r\n")
        slack = period - (time.perf_counter() - tick)
        if slack > 0:
            time.sleep(slack)


@app.get("/snapshot/{camera_id}.jpg")
def snapshot(camera_id: str, width: int = 0):
    """One annotated frame. Built for the Streamlit control room.

    The dashboard fetches this server-side, in Python, rather than pointing an
    <img> at the MJPEG stream. That matters: the dashboard is http and this
    service is https with a certificate the browser has never seen, so an
    embedded tag would be blocked or warned about on every load. Fetched from
    the dashboard process instead, the browser never sees this origin at all.
    """
    camera_id = camera_id.upper()
    j = hub.junctions.get(camera_id)
    if j is None:
        return Response(status_code=404)
    frame = j.source.peek()
    if frame is None:
        img = overlay._placeholder(settings.WALL_WIDTH,
                                   int(settings.WALL_WIDTH * 9 / 16))
    else:
        state = LiveState(vote=j.vote.state(), boxes=j.boxes, aim_box=j.aim_box,
                          infer_ms=j.infer_ms, camera_fps=j.source.fps,
                          backend=hub.backend, raw_text=j.raw_text,
                          raw_conf=j.raw_conf, enrol_enabled=hub.enrol_enabled,
                          enrol_camera=j.camera_id)
        img = overlay.draw(frame, state)
    target = width or settings.WALL_WIDTH
    if img.shape[1] > target:
        h = int(img.shape[0] * target / img.shape[1])
        img = cv2.resize(img, (target, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), settings.WALL_JPEG_Q])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/stream/{camera_id}.mjpg")
def stream(camera_id: str):
    camera_id = camera_id.upper()
    if camera_id not in hub.junctions:
        return Response(status_code=404)
    return StreamingResponse(_tile_frames(camera_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# --- state --------------------------------------------------------------
@app.get("/api/state")
def state():
    return hub.state()


@app.get("/api/trajectory/{plate}")
def trajectory(plate: str):
    return hub.trajectory(plate.upper().replace(" ", ""))


# --- switching cameras off ----------------------------------------------
def _onoff(word: str) -> bool:
    if word not in ("on", "off"):
        raise HTTPException(400, "use /on or /off")
    return word == "on"


@app.post("/api/cameras/{onoff}")
def all_cameras(onoff: str):
    """Switch every junction on or off."""
    want = _onoff(onoff)
    changed = hub.set_all_cameras(want)
    return {"enabled": want, "changed": changed,
            "junctions": [j["camera_id"] for j in hub.state()["junctions"]]}


@app.post("/api/cameras/{camera_id}/{onoff}")
def one_camera(camera_id: str, onoff: str):
    """Switch one junction on or off."""
    want = _onoff(onoff)
    try:
        return {"camera_id": camera_id.upper(),
                "enabled": hub.set_camera(camera_id, want)}
    except KeyError:
        raise HTTPException(404, f"{camera_id} is not one of this run's junctions")


@app.post("/api/recording/{onoff}")
def recording(onoff: str):
    """Whether confirmed reads are written into the platform's database."""
    return {"recording": hub.set_recording(_onoff(onoff))}


@app.post("/api/records/clear")
def clear_records():
    """Forget this run's journeys and recent reads. In-memory only.

    The sightings themselves belong to the platform's database and are the
    control room's to delete; this exists so the wall does not go on showing
    journeys whose rows have been removed.
    """
    return hub.forget_history()


@app.post("/api/shutdown")
def shutdown():
    """Stop the live camera service entirely.

    Answers first and exits a moment later, so the caller gets a reply rather
    than a dropped connection. hub.stop() runs before the exit because that is
    what releases a pulled camera and joins the inference thread; the process
    itself holds nothing else that needs unwinding.
    """
    import threading

    def _bye() -> None:
        time.sleep(0.4)
        try:
            hub.stop()
        finally:
            os._exit(0)

    threading.Thread(target=_bye, name="phonecam-shutdown", daemon=True).start()
    return {"stopping": True}


# --- entry point --------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description="Phones as junction cameras")
    p.add_argument("--junctions", default=settings.DEFAULT_JUNCTIONS,
                   help="comma-separated camera ids from data/cameras.json")
    p.add_argument("--pull", action="append", default=[], metavar="CAM-ID=URL",
                   help="pull this junction from a URL (IP-Webcam MJPEG, RTSP, or a "
                        "video file) instead of waiting for a phone. Repeatable.")
    p.add_argument("--backend", default=None, help="anpr_ocr | crnn | classical")
    p.add_argument("--no-enrol", action="store_true",
                   help="read only; do not write sightings into the platform")
    p.add_argument("--realtime", action="store_true",
                   help="stamp sightings with the real clock. Between junctions "
                        "kilometres apart this implies an impossible speed, which "
                        "is how you demonstrate the clone detector.")
    p.add_argument("--tunnel", action="store_true",
                   help="publish through a Cloudflare quick tunnel. Use this when "
                        "the phones cannot reach this laptop directly - notably on "
                        "a phone-hosted hotspot, where only the phone hosting it "
                        "can. Also removes the certificate warning, because "
                        "Cloudflare presents a real one.")
    p.add_argument("--http", action="store_true",
                   help="serve plain http. Phone browsers will then refuse the "
                        "camera; only pull sources will work.")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-warm", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    global hub
    args = parse(argv)
    if args.port:
        settings.PORT = args.port
    if args.no_enrol:
        settings.ENROL = False
    if args.realtime:
        settings.REALTIME = True
    if args.http:
        settings.HTTPS = False

    pulls = {}
    for item in args.pull:
        if "=" not in item:
            raise SystemExit(f"--pull wants CAM-ID=URL, got {item!r}")
        cid, url = item.split("=", 1)
        pulls[cid.strip().upper()] = url.strip()

    # Check the ports BEFORE anything expensive. uvicorn only discovers a busy
    # port when it binds, which is after the reader has spent twenty-odd
    # seconds warming on the GPU and after the banner of links has been
    # printed - so the failure arrived as two lines buried under a confident
    # page of URLs, with an exit code of 0. The commonest cause is simply that
    # a previous run is still up, which is worth saying plainly.
    busy = [p for p in _ports_in_use() if p]
    if busy:
        print(f"\n[phonecam] port {', '.join(str(p) for p in busy)} is already in "
              f"use - something is still running.\n"
              f"[phonecam] Almost always a previous phonecam that was not shut "
              f"down. Close it, or run stop.bat, then try again.\n"
              f"[phonecam] To check:  netstat -ano | findstr :{busy[0]}\n",
              file=sys.stderr)
        return 1

    ids = [c.strip().upper() for c in args.junctions.split(",") if c.strip()]
    for cid in pulls:
        if cid not in ids:
            ids.append(cid)
    hub = JunctionHub([(cid, pulls.get(cid)) for cid in ids], backend=args.backend)
    hub.start(warm=not args.no_warm)

    ssl_args = {}
    if settings.HTTPS:
        pair = certs.ensure()
        if pair is None:
            print("[phonecam] no certificate - falling back to http. Phone "
                  "browsers will refuse the camera; use --pull sources instead.",
                  file=sys.stderr)
            settings.HTTPS = False
        else:
            ssl_args = {"ssl_certfile": str(pair[0]), "ssl_keyfile": str(pair[1])}

    suffix = ""
    if args.tunnel:
        # A quick tunnel is best-effort bandwidth, and three phones pushing
        # 960px frames saturate it - measured at 0.3-0.7 fps each, far too slow
        # for the vote to lock. The reader does not need that payload: measured
        # on a printed board it still reads exactly right at 480x270 and
        # quality 40, which is a third of the bytes. Only defaults are moved,
        # so an explicit GODSEYE_PHONE_* still wins.
        if "GODSEYE_PHONE_WIDTH" not in os.environ:
            settings.PHONE_WIDTH = 640
        if "GODSEYE_PHONE_JPEG_Q" not in os.environ:
            settings.PHONE_JPEG_Q = 0.55
        if "GODSEYE_PHONE_FPS" not in os.environ:
            settings.PHONE_FPS = 5
        if "GODSEYE_PHONE_INFLIGHT" not in os.environ:
            settings.PHONE_MAX_INFLIGHT = 6
        print(f"[phonecam] tunnel mode: phones send "
              f"{settings.PHONE_WIDTH}px at {settings.PHONE_FPS}/s to fit the "
              f"link", file=sys.stderr)
        if ssl_args:
            _serve_loopback_http()          # the tunnel's origin, plain http
        print("[phonecam] asking Cloudflare for a public address...",
              file=sys.stderr)
        res = tunnel.start(port=settings.LOCAL_PORT)
        if res.running:
            suffix = f"?k={reach.token()}"
            print(f"[phonecam] public address {res.url}", file=sys.stderr)
        else:
            print(f"[phonecam] no tunnel: {res.detail}", file=sys.stderr)

    base = _base()
    print("\n" + "=" * 68, file=sys.stderr)
    print(f"  Operator wall   {base}/", file=sys.stderr)
    print(f"  Point phones    {base}/links{suffix}", file=sys.stderr)
    for cid, j in hub.junctions.items():
        print(f"    {cid}  {j.name:<26} {base}/cam/{cid}{suffix}", file=sys.stderr)
    if tunnel.current_url():
        print("\n  Public link: works from any network, and the certificate is\n"
              "  Cloudflare's, so the phones will not warn. Anyone holding the\n"
              "  link can view the feed - it dies when this process does.",
              file=sys.stderr)
    else:
        print(f"\n  This laptop is on {', '.join(certs.lan_addresses())}. The phones\n"
              "  must be able to reach that address - on a phone-hosted hotspot\n"
              "  usually only the hosting phone can. If two of three will not\n"
              "  connect, restart with --tunnel.", file=sys.stderr)
        if settings.HTTPS:
            print("\n  The phones will warn about the certificate once. Accept it -\n"
                  "  without https the browser will not hand over the camera.",
                  file=sys.stderr)
    print("=" * 68 + "\n", file=sys.stderr)

    import uvicorn
    if ssl_args:
        _serve_loopback_http()
    try:
        uvicorn.run(app, host=settings.HOST, port=settings.PORT,
                    log_level="warning", **ssl_args)
    finally:
        hub.stop()
        # A quick tunnel listens on nothing, so nothing else can find it to
        # kill it. Left running it keeps a public address pointing at this
        # laptop after the demo - the one thing about a tunnel that must not
        # be left to chance (stop.bat makes the same point).
        if args.tunnel:
            tunnel.stop()
    return 0


_loopback_started = False


def _ports_in_use() -> list:
    """Which of this service's ports are already taken. Empty is the good case.

    Binds rather than asking the OS for a listener list: a port held by another
    user's process, or by a socket in a state netstat reports oddly, still
    refuses the bind - and refusing the bind is the thing that actually matters.
    """
    import socket

    wanted = [settings.PORT]
    if settings.HTTPS:
        wanted.append(settings.LOCAL_PORT)
    busy = []
    for port in wanted:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # No SO_REUSEADDR: this must fail exactly where uvicorn would.
            s.bind(("0.0.0.0" if port == settings.PORT else settings.LOCAL_HOST,
                    port))
        except OSError:
            busy.append(port)
        finally:
            s.close()
    return busy


def _serve_loopback_http() -> None:
    """The same app on 127.0.0.1 over plain http, on a daemon thread.

    Only started when the main listener is https, because that is the situation
    it exists for: the dashboard cannot embed an https stream whose certificate
    the browser has never accepted, and polling stills instead is a slideshow.
    Loopback only - nothing on the LAN needs this, and the phones have their
    own, encrypted way in.
    """
    import threading

    import uvicorn

    # Idempotent: both the tunnel and the https path want this listener, and
    # whichever runs second would otherwise fail to bind the port.
    global _loopback_started
    if _loopback_started:
        return
    _loopback_started = True

    config = uvicorn.Config(app, host=settings.LOCAL_HOST,
                            port=settings.LOCAL_PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="phonecam-loopback",
                     daemon=True).start()
    print(f"[phonecam] control-room feed on "
          f"http://{settings.LOCAL_HOST}:{settings.LOCAL_PORT}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
