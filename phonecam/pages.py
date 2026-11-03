"""The two pages: what a phone opens, and what the room watches.

Kept as templates here rather than inline in server.py so the routing stays
readable, and styled to match the platform's judge console so the demo looks
like one product rather than three.
"""
from __future__ import annotations

PHONE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__NAME__ — GodsEye junction camera</title><style>
:root{--bg:#0b0e12;--card:#161b22;--line:#293240;--ink:#eef2f7;--dim:#8fa0b4;
--ok:#22c55e;--amber:#f5a524;--red:#e11d48}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
body{display:flex;flex-direction:column;padding:10px;gap:8px}
.head{display:flex;align-items:baseline;gap:8px}
.cam{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
h1{font-size:1.1rem;margin:0}
#wrap{position:relative;flex:1;min-height:0;border-radius:14px;overflow:hidden;
background:#000;border:1px solid var(--line)}
video{width:100%;height:100%;object-fit:cover;display:block}
#badge{position:absolute;left:10px;top:10px;padding:5px 10px;border-radius:999px;
font-size:.74rem;font-weight:600;background:#0009;backdrop-filter:blur(6px)}
#plate{position:absolute;left:0;right:0;bottom:0;padding:12px;
background:linear-gradient(transparent,#000c);font-family:ui-monospace,Menlo,monospace;
font-size:1.6rem;font-weight:700;letter-spacing:.06em;text-align:center}
.row{display:flex;gap:8px}
button{flex:1;padding:13px;font-size:.95rem;font-weight:600;border:0;
border-radius:10px;color:#fff;background:#334155}
button.go{background:var(--red)}
.stat{font-size:.75rem;color:var(--dim);text-align:center}
.err{background:#3a1620;border:1px solid #7f1d33;color:#ffd7e0;padding:12px;
border-radius:10px;font-size:.85rem}
</style></head><body>
<div class="head"><span class="cam">__CAMID__</span><h1>__NAME__</h1></div>
<div id="wrap">
  <video id="v" playsinline autoplay muted></video>
  <div id="badge">connecting…</div>
  <div id="plate"></div>
</div>
<div id="err"></div>
<div class="row">
  <button id="start" class="go">Start camera</button>
  <button id="torch">Torch</button>
</div>
<div class="stat" id="stat">idle</div>
<script>
const CAMID = "__CAMID__", FPS = __FPS__, W = __WIDTH__, Q = __Q__;
const MAX_INFLIGHT = __INFLIGHT__;
const v = document.getElementById('v'), badge = document.getElementById('badge');
const plateEl = document.getElementById('plate'), statEl = document.getElementById('stat');
const errEl = document.getElementById('err');
let ws, stream, track, sending = false, sent = 0, inflight = 0, timer, wake;
let rtt = 0, lastSend = 0;

function say(t, colour){ badge.textContent = t; badge.style.color = colour || '#eef2f7'; }
function fail(msg){ errEl.innerHTML = '<div class="err">' + msg + '</div>'; }

async function start(){
  errEl.innerHTML = '';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    fail("This browser will not share a camera on this page. That usually means "
       + "the page was opened over http:// — reopen it over <b>https://</b> and "
       + "accept the certificate warning once.");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' },
               width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false });
  } catch (e) {
    fail("Camera refused: " + e.name + ". If this says NotAllowedError, grant the "
       + "camera permission; if SecurityError, the page is not on https.");
    return;
  }
  v.srcObject = stream;
  track = stream.getVideoTracks()[0];
  document.getElementById('start').textContent = 'Camera on';
  try { wake = await navigator.wakeLock.request('screen'); } catch(e) {}
  connect();
}

function connect(){
  // The token rides through from this page's own URL. Over a tunnel the
  // websocket is gated on it too, and a socket that opens and closes at 4003
  // looks identical to a network failure - so it is carried rather than
  // re-derived.
  const k = new URLSearchParams(location.search).get('k');
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://')
            + location.host + '/ws/cam/' + CAMID + (k ? '?k=' + k : '');
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { say('live', '#22c55e'); sending = true; inflight = 0; loop(); };
  ws.onclose = (e) => {
    // 4003 is this server refusing the token, which is a different problem
    // from a dropped connection and must not look like one.
    if (e.code === 4003){ say('link expired', '#e11d48');
      fail("This link is no longer valid. Ask for a fresh one from the "
         + "<b>/links</b> page - the token changes when the service restarts.");
      sending = false; return; }
    say('reconnecting…', '#f5a524'); sending = false;
    clearTimeout(timer); setTimeout(connect, 1200);
  };
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    inflight = Math.max(0, inflight - 1);
    if (lastSend) rtt = Math.round(performance.now() - lastSend);
    let d = {}; try { d = JSON.parse(ev.data); } catch(e){ return; }
    if (d.pretty){ plateEl.textContent = d.pretty;
      plateEl.style.color = d.locked ? '#22c55e' : '#8fa0b4'; }
    else plateEl.textContent = '';
    statEl.textContent = sent + ' sent · ' + (d.fps||0).toFixed(1)
      + ' fps · ' + (d.infer_ms||0).toFixed(0) + ' ms/read · link ' + rtt + ' ms';
  };
}

const cv = document.createElement('canvas'), cx = cv.getContext('2d');
function grab(){
  if (!v.videoWidth) return null;
  const scale = W / v.videoWidth;
  cv.width = W; cv.height = Math.round(v.videoHeight * scale);
  cx.drawImage(v, 0, 0, cv.width, cv.height);
  return new Promise(r => cv.toBlob(r, 'image/jpeg', Q));
}

async function loop(){
  if (!sending || !ws || ws.readyState !== 1) return;
  // A small window of unacknowledged frames, not one at a time. Capped in
  // time as well, so a fast link does not flood. One at a time would tie the
  // frame rate to the round trip, which is invisible on a LAN and ruinous
  // through a tunnel; a bounded window still stops a slow link building a
  // backlog of frames describing a moment that has already passed.
  if (inflight < MAX_INFLIGHT){
    const blob = await grab();
    if (blob && ws.readyState === 1){
      ws.send(await blob.arrayBuffer());
      inflight++; sent++; lastSend = performance.now();
    }
  }
  timer = setTimeout(loop, 1000 / FPS);
}

document.getElementById('start').onclick = start;
document.getElementById('torch').onclick = async () => {
  if (!track) return;
  try {
    const on = !(track.getSettings().torch);
    await track.applyConstraints({ advanced: [{ torch: on }] });
  } catch(e) { fail('This phone will not let the page control the torch.'); }
};
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && stream){
    try { wake = await navigator.wakeLock.request('screen'); } catch(e){}
  }
});
</script></body></html>"""


WALL = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GodsEye — junction wall</title><style>
:root{--bg:#0f1216;--card:#181d24;--line:#2a323d;--ink:#eef2f7;--dim:#93a1b3;
--ok:#22c55e;--amber:#f5a524;--red:#e11d48}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);padding:16px;
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto}
h1{font-size:1.15rem;margin:0}
.sub{color:var(--dim);font-size:.82rem;margin:2px 0 14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:13px;
overflow:hidden}
.tile img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:cover}
.tb{padding:9px 11px}
.tt{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.tn{font-weight:600;font-size:.95rem}
.tc{font-size:.68rem;color:var(--dim);letter-spacing:.08em;text-transform:uppercase}
.plate{font-family:ui-monospace,Menlo,monospace;font-size:1.35rem;font-weight:700;
letter-spacing:.06em;margin-top:6px;color:var(--dim)}
.plate.locked{color:var(--ok)}
.meta{font-size:.72rem;color:var(--dim);margin-top:3px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#555;
margin-right:5px;vertical-align:middle}
.dot.on{background:var(--ok)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:12px}
.k{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
margin-bottom:6px}
.ev{font-size:.82rem;padding:4px 0;border-bottom:1px solid #202832}
.ev b{font-family:ui-monospace,Menlo,monospace}
.ev.enrol{color:#9fe8b5}.ev.alert{color:#ffb4c0}.ev.journey{color:#ffd79a}
.ev.refused{color:#c9b6ff}.ev.error{color:#ff9a9a}
.t{color:var(--dim);font-size:.72rem;margin-right:6px}
.jr{padding:7px 0;border-bottom:1px solid #202832;font-size:.85rem;cursor:pointer}
.leg{font-size:.76rem;color:var(--dim);margin-left:8px}
.bad{color:#ffb4c0}
</style></head><body><div class="wrap">
<h1>GodsEye — junction wall</h1>
<div class="sub" id="sub">Each tile is a phone standing at a real junction of the
Kolkata camera network. A plate read at two of them is a journey.</div>
<div class="tiles" id="tiles"></div>
<div class="cols">
  <div class="card"><div class="k">Journeys reconstructed</div><div id="journeys">
    <span class="meta">A registration seen at two junctions will appear here.</span>
  </div></div>
  <div class="card"><div class="k">Live event log</div><div id="events"></div></div>
</div></div>
<script>
const $ = id => document.getElementById(id);
let built = false;

function tile(j){
  return `<div class="tile" id="t_${j.camera_id}">
    <img src="/stream/${j.camera_id}.mjpg" alt="${j.name}">
    <div class="tb">
      <div class="tt"><span class="tn">${j.name}</span>
        <span class="tc">${j.camera_id}${j.sector ? ' · '+j.sector : ''}</span></div>
      <div class="plate" id="p_${j.camera_id}">—</div>
      <div class="meta" id="m_${j.camera_id}"></div>
    </div></div>`;
}

function clock(ts){ return new Date(ts*1000).toLocaleTimeString(); }

async function tick(){
  let d;
  try { d = await (await fetch('/api/state')).json(); } catch(e){ return; }
  if (!built){ $('tiles').innerHTML = d.junctions.map(tile).join(''); built = true; }
  $('sub').textContent = `${d.junctions.length} junctions · reader ${d.backend}`
    + (d.enrol ? ' · writing sightings' : ' · read-only')
    + (d.realtime ? ' · real timestamps' : ' · plausible timestamps');

  for (const j of d.junctions){
    const p = $('p_'+j.camera_id), m = $('m_'+j.camera_id);
    if (!p) continue;
    p.textContent = j.pretty || '—';
    p.className = 'plate' + (j.locked ? ' locked' : '');
    m.innerHTML = `<span class="dot ${j.online?'on':''}"></span>`
      + (j.online ? `${j.fps.toFixed(1)} fps · ${j.infer_ms.toFixed(0)} ms · `
                  + `${j.locks} read${j.locks===1?'':'s'}`
                  : (j.frames ? `silent ${j.silent_for.toFixed(0)}s` : 'waiting for a phone'))
      + (j.raw_text && !j.locked ? ` · seeing ${j.raw_text} ${(j.raw_conf*100).toFixed(0)}%` : '');
  }

  $('events').innerHTML = d.events.map(e =>
    `<div class="ev ${e.kind}"><span class="t">${clock(e.ts)}</span>${e.message}</div>`
  ).join('') || '<span class="meta">nothing yet</span>';

  if (d.journeys.length){
    $('journeys').innerHTML = d.journeys.map(j =>
      `<div class="jr" onclick="showTraj('${j.plate}')"><b>${j.pretty}</b>
       <span class="leg">${j.cameras.length} junctions · ${j.hits} sightings —
       tap for the reconstruction</span><div id="tj_${j.plate}"></div></div>`).join('');
  }
}

async function showTraj(plate){
  const box = $('tj_'+plate);
  if (!box) return;
  if (box.innerHTML){ box.innerHTML = ''; return; }
  const t = await (await fetch('/api/trajectory/'+plate)).json();
  if (!t.known){ box.innerHTML = '<div class="leg">no record</div>'; return; }
  const s = t.summary;
  box.innerHTML = `<div class="leg">${s.sightings} sightings ·
    ${s.cameras_visited} cameras · ${s.distance_km.toFixed(1)} km ·
    ${s.duration_min.toFixed(0)} min</div>` + t.legs.map(l =>
    `<div class="leg ${l.plausible?'':'bad'}">${l.from} → ${l.to} ·
      ${l.km} km · ${l.minutes} min · ${l.kmph} km/h ${l.direction||''}
      ${l.plausible?'':' ⚠ '+(l.note||'implausible')}</div>`).join('');
}
setInterval(tick, 700); tick();
</script></body></html>"""


def phone_page(camera_id: str, name: str, fps: int, width: int, quality: float,
               max_inflight: int = 4) -> str:
    return (PHONE.replace("__CAMID__", camera_id).replace("__NAME__", name)
            .replace("__FPS__", str(fps)).replace("__WIDTH__", str(width))
            .replace("__Q__", str(quality))
            .replace("__INFLIGHT__", str(max_inflight)))
