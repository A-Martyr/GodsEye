"""Browser front-end: the same engine, streamed as MJPEG on its own port.

    python -m livecam.server              ->  http://localhost:8503

Its own FastAPI app on its own port, deliberately not a router bolted onto
api/main.py: the platform's API is the process the control room depends on,
and a webcam that stalls must not be able to take it with it.

The annotated frame comes from livecam/overlay.py, the same pixels the native
window shows. The page additionally polls /api/state so the registration is
selectable HTML text rather than only pixels in a video - which is what makes
it readable on a projector at the back of a room.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from livecam import overlay, settings                            # noqa: E402
from livecam.engine import LiveEngine                            # noqa: E402

app = FastAPI(title="GodsEye — live plate reader", docs_url=None, redoc_url=None)
_engine: LiveEngine | None = None


def engine() -> LiveEngine:
    global _engine
    if _engine is None:
        _engine = LiveEngine()
        if not _engine.start():
            print(f"[livecam] {_engine.error}", file=sys.stderr)
    return _engine


@app.on_event("startup")
def _startup() -> None:
    engine()


@app.on_event("shutdown")
def _shutdown() -> None:
    if _engine is not None:
        _engine.stop()


def _frames():
    """MJPEG parts, paced so the encoder does not compete with the recogniser."""
    eng = engine()
    period = 1.0 / max(1, settings.STREAM_FPS)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
    while True:
        tick = time.perf_counter()
        frame, state = eng.snapshot()
        ok, buf = cv2.imencode(".jpg", overlay.draw(frame, state), params)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(buf)).encode() + b"\r\n\r\n" + buf.tobytes() + b"\r\n")
        slack = period - (time.perf_counter() - tick)
        if slack > 0:
            time.sleep(slack)


@app.get("/stream.mjpg")
def stream():
    return StreamingResponse(_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    eng = engine()
    _, s = eng.snapshot()
    v = s.vote
    try:
        from anpr.plates import pretty
        shown = pretty(v.text) if v.text else ""
    except Exception:
        shown = v.text
    return {
        "plate": v.text, "pretty": shown, "locked": v.locked,
        "confidence": round(v.confidence, 4), "votes": v.votes,
        "progress": round(v.progress, 3),
        "runners": [{"text": t, "votes": n} for t, n in (v.runners or [])],
        "backend": s.backend, "backend_note": s.backend_note,
        "infer_ms": round(s.infer_ms, 1), "detect_ms": round(s.detect_ms, 1),
        "read_ms": round(s.read_ms, 1), "infer_fps": round(s.infer_fps, 2),
        "camera_fps": round(s.camera_fps, 2), "crops": s.crops_tried,
        "boxes": len(s.boxes or []), "paused": s.paused,
        "enrol_enabled": s.enrol_enabled, "enrol_camera": s.enrol_camera,
        "last_enrol": s.last_enrol,
    }


@app.post("/api/reset")
def reset():
    engine().reset_vote()
    return {"ok": True}


@app.post("/api/pause")
def pause():
    return {"paused": engine().toggle_pause()}


@app.post("/api/enrol")
def enrol():
    return {"message": engine().enrol()}


PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GodsEye — live plate reader</title><style>
:root{--bg:#0f1216;--card:#181d24;--line:#2a323d;--ink:#eef2f7;--dim:#93a1b3;
--ok:#22c55e;--amber:#f5a524;--red:#e11d48}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:18px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:1.15rem;margin:0;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:.82rem;margin:2px 0 16px}
.grid{display:grid;grid-template-columns:1fr 320px;gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:14px;margin-bottom:14px}
img{width:100%;border-radius:10px;display:block;background:#000}
.plate{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:2.1rem;
font-weight:700;letter-spacing:.08em;text-align:center;padding:14px 8px;
border:1px solid var(--line);border-radius:10px;background:#0b0e12}
.plate.locked{color:var(--ok);border-color:var(--ok)}
.plate.idle{color:var(--dim);font-size:1.1rem;letter-spacing:0}
.k{color:var(--dim);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
margin:12px 0 3px}
.v{font-size:.95rem}
.bar{height:8px;background:#0b0e12;border-radius:4px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--amber);width:0;transition:width .12s}
.bar>i.ok{background:var(--ok)}
button{width:100%;padding:11px;font-size:.92rem;font-weight:600;border:0;
border-radius:9px;color:#fff;background:#334155;cursor:pointer;margin-top:8px}
button.primary{background:var(--red)}
.warn{background:#2a1f3d;border-color:#4c3a6b;color:#cbb6ff;font-size:.82rem}
.muted{color:var(--dim);font-size:.8rem}
table{width:100%;border-collapse:collapse;font-size:.82rem}
td{padding:3px 0;color:var(--dim)} td.r{text-align:right;color:var(--ink)}
</style></head><body><div class="wrap">
<h1>GodsEye — live plate reader</h1>
<div class="sub">Hold a plate-format board inside the brackets. Frames are read
independently and voted, so the reading settles rather than flickering.</div>
<div id="note"></div>
<div class="grid">
  <div><div class="card" style="padding:8px"><img src="/stream.mjpg" alt="live"></div></div>
  <div>
    <div class="card">
      <div id="plate" class="plate idle">waiting…</div>
      <div class="k">Agreement</div>
      <div class="bar"><i id="bar"></i></div>
      <table style="margin-top:12px">
        <tr><td>Confidence</td><td class="r" id="conf">—</td></tr>
        <tr><td>Frames agreeing</td><td class="r" id="votes">—</td></tr>
        <tr><td>Reader</td><td class="r" id="backend">—</td></tr>
        <tr><td>Read latency</td><td class="r" id="ms">—</td></tr>
        <tr><td>Reads / sec</td><td class="r" id="rps">—</td></tr>
        <tr><td>Camera</td><td class="r" id="fps">—</td></tr>
      </table>
      <div id="also" class="muted" style="margin-top:10px"></div>
    </div>
    <div class="card">
      <button id="reset">Reset the vote</button>
      <button id="pause">Pause</button>
      <button id="enrol" class="primary">Enrol this vehicle</button>
      <div id="enrolnote" class="muted" style="margin-top:8px"></div>
    </div>
  </div>
</div></div>
<script>
const $ = id => document.getElementById(id);
async function post(p){ const r = await fetch(p,{method:'POST'}); return r.json(); }
$('reset').onclick = () => post('/api/reset');
$('pause').onclick = async () => { const d = await post('/api/pause');
  $('pause').textContent = d.paused ? 'Resume' : 'Pause'; };
$('enrol').onclick = async () => { const d = await post('/api/enrol');
  $('enrolnote').textContent = d.message || ''; };

async function tick(){
  try{
    const d = await (await fetch('/api/state')).json();
    const p = $('plate');
    if (d.pretty){ p.textContent = d.pretty;
      p.className = 'plate' + (d.locked ? ' locked' : ''); }
    else { p.textContent = 'hold the board up'; p.className = 'plate idle'; }
    $('bar').style.width = Math.round(d.progress*100) + '%';
    $('bar').className = d.locked ? 'ok' : '';
    $('conf').textContent = d.confidence ? (d.confidence*100).toFixed(0)+'%' : '—';
    $('votes').textContent = d.votes || '—';
    $('backend').textContent = d.backend || '—';
    $('ms').textContent = d.infer_ms ? d.infer_ms.toFixed(0)+' ms' : '—';
    $('rps').textContent = d.infer_fps ? d.infer_fps.toFixed(1) : '—';
    $('fps').textContent = d.camera_fps ? d.camera_fps.toFixed(1)+' fps' : '—';
    const others = (d.runners||[]).slice(1,3).map(r=>r.text+' ×'+r.votes);
    $('also').textContent = others.length ? 'also seen: '+others.join('   ') : '';
    if (d.last_enrol) $('enrolnote').textContent = d.last_enrol;
    $('note').innerHTML = d.backend_note
      ? '<div class="card warn">'+d.backend_note+'</div>' : '';
  }catch(e){}
}
setInterval(tick, 250); tick();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE)


def main() -> int:
    import uvicorn
    print(f"[livecam] http://localhost:{settings.WEB_PORT}", file=sys.stderr)
    uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
