/* GodsEye control room: map, Spotlight search, live reads, plate reading,
   vehicle tracking and alerts, all on the platform's own FastAPI. */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (id) => `<svg><use href="#${id}"/></svg>`;

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const d = body && body.detail;
    const msg = typeof d === 'string' ? d : (d && d.message) || `request failed (${r.status})`;
    const err = new Error(msg); err.status = r.status; err.detail = d; throw err;
  }
  return body;
}

// ---------- formatting ----------
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const hm = (ts) => new Date(ts * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
const day = (ts) => new Date(ts * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
const pct = (x) => `${Math.round(x * 100)}%`;
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${day(ts)}, ${hm(ts)}`;
}
const fmtPlate = (p) => esc(p || '');

// Where a read came from. The simulated city reads synthetic captures with the
// real OCR; enrolments are real photographs read by the real-plate reader.
function readKind(s) {
  const src = String(s.source || ''); const v = String(s.ocr_variant || '');
  if (src.startsWith('enrolment') || v.startsWith('enrol:')) return 'real';
  if (v.startsWith('engine')) return 'ocr';
  return 'mod';
}
const KIND_LABEL = { real: '<span class="pk real">real photo</span>', ocr: '<span class="pk ocr">OCR</span>', mod: '<span class="pk mod">modelled</span>' };
function provenance(s) { return KIND_LABEL[readKind(s)]; }

function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, 3200);
}

// ---------- map ----------
const LEVEL = { free: 't0', moderate: 't1', heavy: 't2', severe: 't3' };
const LEVEL_COLOR = {};
const css = getComputedStyle(document.documentElement);
for (const k of ['t0', 't1', 't2', 't3', 'blue', 'danger', 'infer', 'line-2'])
  LEVEL_COLOR[k] = css.getPropertyValue(`--${k}`).trim();

const MAP = L.map('map', { zoomControl: false, preferCanvas: false, attributionControl: true, zoomSnap: .25, zoomDelta: .5 })
  .setView([22.585, 88.39], 12);
L.control.zoom({ position: 'bottomright' }).addTo(MAP);
// Standard OSM raster tiles need no key. A CSS filter on the tile pane turns
// them into a quiet dark ground; the camera network is the information layer.
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap contributors · GodsEye prototype',
}).addTo(MAP);

// A road with no reads in the window stays a neutral grey: no data is not "free".
const QUIET_ROAD = '#56616e';
const NET = { cameras: {}, links: [], markers: {}, lines: {} };
const roadLayer = L.layerGroup().addTo(MAP);
const routeLayer = L.layerGroup().addTo(MAP);
const camLayer = L.layerGroup().addTo(MAP);
const pairKey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);

function padOpts() {
  const side = $('#side').getBoundingClientRect();
  const alerts = $('#alerts').getBoundingClientRect();
  const wide = window.innerWidth > 760;
  return wide
    ? { paddingTopLeft: [side.right + 30, 90], paddingBottomRight: [alerts.width + 40, 40] }
    : { padding: [24, 24] };
}

async function loadNetwork() {
  const net = await api('/api/network');
  for (const c of net.cameras) NET.cameras[c.id] = c;
  NET.links = net.links;
  for (const l of net.links) {
    const pts = (l.path || [[l.a_lon, l.a_lat], [l.b_lon, l.b_lat]]).map(([lon, lat]) => [lat, lon]);
    const casing = L.polyline(pts, { color: '#0c1015', weight: 8, opacity: .9, interactive: false }).addTo(roadLayer);
    const line = L.polyline(pts, { color: QUIET_ROAD, weight: 4, opacity: .95, lineCap: 'round' })
      .bindTooltip(`${esc(l.name)}<small>${l.km} km · free flow ${Math.round(l.free_kmph)} km/h</small>`, { className: 'cam-tip', sticky: true })
      .addTo(roadLayer);
    NET.lines[pairKey(l.a, l.b)] = { line, casing, link: l };
  }
  for (const c of net.cameras) {
    const ic = L.divIcon({ className: '', html: `<div class="cam${c.type === 'gateway' ? ' gw' : ''}"><span class="ping"></span><span class="dot"></span></div>`, iconSize: [14, 14], iconAnchor: [7, 7] });
    const m = L.marker([c.lat, c.lon], { icon: ic, keyboard: false })
      .bindTooltip(`${esc(c.name)}<small>${esc(c.road)} · ${esc(c.sector)}</small>`, { className: 'cam-tip', direction: 'top', offset: [0, -8] })
      .addTo(camLayer);
    m.on('click', () => openCamera(c.id));
    NET.markers[c.id] = m;
  }
  MAP.fitBounds(L.latLngBounds(net.cameras.map((c) => [c.lat, c.lon])), padOpts());
}

function camEl(id) { const m = NET.markers[id]; return m && m.getElement() && m.getElement().firstElementChild; }
function pingCam(id, real) {
  const el = camEl(id); if (!el) return;
  const cls = real ? 'hit-real' : 'hit';
  el.classList.remove('hit', 'hit-real'); void el.offsetWidth; el.classList.add(cls);
}

async function refreshCams() {
  try {
    const dens = await api(`/api/analytics/density?minutes=${WINDOW}`);
    for (const d of dens) {
      const el = camEl(d.camera_id); if (!el) continue;
      const l = d.count > 0 ? (d.load < .25 ? 't0' : d.load < .5 ? 't1' : d.load < .75 ? 't2' : 't3') : null;
      if (l) el.dataset.l = l; else delete el.dataset.l;
    }
  } catch { /* keep the last rings */ }
}
async function refreshLinks() {
  refreshCams();
  try {
    const flows = await api(`/api/analytics/links?minutes=${WINDOW}`);
    const worst = {};
    const rank = { free: 0, moderate: 1, heavy: 2, severe: 3 };
    for (const f of flows) {
      const k = pairKey(f.from_camera, f.to_camera);
      if (!worst[k] || rank[f.level] > rank[worst[k].level]) worst[k] = f;
    }
    for (const [k, rec] of Object.entries(NET.lines)) {
      const f = worst[k];
      const col = f ? LEVEL_COLOR[LEVEL[f.level] || 't3'] : QUIET_ROAD;
      rec.line.setStyle({ color: col, opacity: TRACK.plate ? .38 : .95 });
      if (f) rec.line.setTooltipContent(`${esc(rec.link.name)}<small>${Math.round(f.median_kmph)} km/h against ${Math.round(f.free_kmph)} free · ${f.level} · ${f.vehicles} vehicles</small>`);
    }
  } catch { /* the map keeps its last colours */ }
}

// ---------- status chips ----------
let WINDOW = 30;   // minutes: the analysis window for chips, road load and analytics
async function refreshSummary() {
  try {
    const s = await api(`/api/analytics/summary?minutes=${WINDOW}`);
    $('#k-vpm').textContent = s.vehicles_per_min ?? '–';
    $('#k-cams').textContent = `${s.active_cameras}/${s.total_cameras}`;
  } catch { /* leave the last values */ }
}

$('#live-chip').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  const on = btn.getAttribute('aria-pressed') !== 'true';
  try {
    await api(`/api/sim/live/${on ? 'on' : 'off'}`, { method: 'POST' });
    btn.setAttribute('aria-pressed', String(on));
    $('#live-label').textContent = on ? 'Live' : 'Paused';
  } catch (err) { toast(`Could not ${on ? 'resume' : 'pause'} the feed: ${err.message}`); }
});

// ---------- panel views ----------
let VIEW = 'feed';
let LAST_TAB = 'feed';
function show(view) {
  VIEW = view;
  if (view !== 'vehicle') LAST_TAB = view;
  for (const v of $$('#side .view')) v.hidden = v.dataset.view !== view;
  for (const b of $$('#side .seg button')) {
    const on = b.dataset.view === (view === 'vehicle' ? '' : view);
    b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on));
  }
}
for (const b of $$('#side .seg button')) b.addEventListener('click', () => { clearTrack(); show(b.dataset.view); });
$('#v-back').addEventListener('click', () => { clearTrack(); show(LAST_TAB); });

// ---------- live reads ----------
const FEED_MAX = 40;
function feedItem(s, isNew) {
  const li = document.createElement('li');
  if (isNew) li.className = 'new';
  const cam = NET.cameras[s.camera_id];
  const kind = readKind(s);
  li.innerHTML = `<span class="t">${clock(s.ts)}</span>
    <span class="pl"><span class="plate">${fmtPlate(s.plate)}</span>${provenance(s)}</span>
    <span class="conf ${kind === 'mod' ? 'mod' : ''}" title="${kind === 'mod' ? 'Modelled confidence: this capture was not run through the OCR' : 'Engine confidence'}">${pct(s.confidence)}</span>
    <span class="cam-n">${esc(cam ? cam.name : s.camera_id)}${s.speed_kmph ? ` · ${Math.round(s.speed_kmph)} km/h` : ''}${s.condition ? ` · ${esc(String(s.condition).replace(/_/g, ' '))}` : ''}</span>`;
  li.addEventListener('click', () => openVehicle(s.plate));
  return li;
}
const SEEN_REF = new Set();
function refusedItem(u) {
  const li = document.createElement('li');
  li.className = 'ref';
  const cam = NET.cameras[u.camera_id];
  li.innerHTML = `<span class="t">${clock(u.ts)}</span>
    <span class="pl"><span class="plate">${u.best_text ? fmtPlate(u.best_text) : 'unreadable'}</span><span class="pk ref">refused</span></span>
    <span class="conf" title="Below the confidence floor, so it was not stored">${pct(u.confidence)}</span>
    <span class="cam-n">${esc(cam ? cam.name : u.camera_id)}${u.condition ? ` · ${esc(String(u.condition).replace(/_/g, ' '))}` : ''} · kept for repair, not stored</span>`;
  li.title = 'Refused: the engine would not guess. Repair can recover it from neighbouring cameras.';
  return li;
}
async function pollRefused(initial) {
  if (VIEW !== 'feed' || $('#feed-all')) return;
  try {
    const rows = await api('/api/unresolved/recent?limit=8');
    const fresh = rows.filter((u) => !SEEN_REF.has(u.id)).reverse();
    for (const u of fresh) {
      SEEN_REF.add(u.id);
      const ol = $('#feed');
      // place it by time among the reads already listed
      const at = [...ol.children].find((li) => +li.dataset.ts < u.ts);
      const el = refusedItem(u); el.dataset.ts = u.ts;
      if (initial || !at) { if (at) ol.insertBefore(el, at); else ol.append(el); } else ol.insertBefore(el, at);
    }
    while ($('#feed').children.length > FEED_MAX) $('#feed').lastElementChild.remove();
  } catch { /* refusals are a bonus row type; the feed stands without them */ }
}
function pushFeed(s) {
  const ol = $('#feed');
  const el = feedItem(s, true); el.dataset.ts = s.ts;
  ol.prepend(el);
  while (ol.children.length > FEED_MAX) ol.lastElementChild.remove();
  $('#feed-meta').textContent = `Latest ${clock(s.ts)}`;
}
async function loadFeed() {
  try {
    const rows = await api('/api/sightings/recent?limit=30');
    const ol = $('#feed'); ol.innerHTML = '';
    for (const s of rows) { const el = feedItem(s, false); el.dataset.ts = s.ts; ol.append(el); }
    if (rows[0]) $('#feed-meta').textContent = `Latest ${clock(rows[0].ts)} · ${day(rows[0].ts)}`;
    SEEN_REF.clear(); pollRefused(true);
  } catch (err) { $('#feed').innerHTML = `<li class="error">${icon('i-alert')}${esc(err.message)}</li>`; }
}

// ---------- alerts ----------
const KIND = {
  clone: 'Cloned plate', watchlist: 'Watchlist hit', loitering: 'Loitering',
  speeding: 'Speeding', detour: 'Unusual detour', odd_hour: 'Odd-hour movement',
};
const ALERTS = new Map();
function parseDetail(a) { try { return typeof a.detail === 'string' ? JSON.parse(a.detail) : (a.detail || {}); } catch { return {}; } }
function alertCard(a) {
  const crit = a.severity === 'critical';
  const d = parseDetail(a);
  const el = document.createElement('article');
  el.className = `alert${crit ? ' crit' : ''}`;
  el.dataset.id = a.id;
  el.innerHTML = `<div class="alert-top">
      <span class="alert-kind">${icon(a.kind === 'clone' ? 'i-copy2' : a.kind === 'watchlist' ? 'i-flag' : 'i-alert')}${esc(KIND[a.kind] || a.kind)}</span>
      <span class="alert-time">${ago(a.ts)}</span></div>
    <p class="alert-msg">${esc(String(a.message || '').replace(/^WATCHLIST\s+/, ''))}</p>
    <div class="alert-act">
      <button class="btn btn-primary" data-act="track" type="button">${icon('i-route')}Track ${fmtPlate(a.plate)}</button>
      ${a.kind === 'clone' && d.from && d.to ? `<button class="btn btn-quiet" data-act="map" type="button">${icon('i-pin')}Show on map</button>` : ''}
      <button class="btn btn-ghost" data-act="ack" type="button" aria-label="Acknowledge">${icon('i-check')}Clear</button>
    </div>`;
  el.querySelector('[data-act="track"]').onclick = () => openVehicle(a.plate);
  const mapBtn = el.querySelector('[data-act="map"]');
  if (mapBtn) mapBtn.onclick = () => showClone(a.plate, d);
  el.querySelector('[data-act="ack"]').onclick = async () => {
    const ids = (el.dataset.ids || String(a.id)).split(',');
    try { await Promise.all(ids.map((id) => api(`/api/alerts/${id}/ack`, { method: 'POST' }))); ids.forEach((id) => ALERTS.delete(+id)); el.remove(); renderAlertMeta(); }
    catch (err) { toast(`Could not clear the alert: ${err.message}`); }
  };
  return el;
}
function addAlert(a, isNew) {
  if (a.acked || ALERTS.has(a.id)) return;
  if (a.severity !== 'critical' && !isNew) return;
  ALERTS.set(a.id, a);
  const list = $('#alerts-list');
  if (a.severity !== 'critical') { renderAlertMeta(); return; }   // counted, not stacked
  // One card per vehicle and rule: a watchlisted car passing five cameras is
  // one thing to act on, seen five times, not five alerts.
  const key = `${a.kind}|${a.plate}`;
  const prev = list.querySelector(`.alert[data-key="${CSS.escape(key)}"]`);
  const el = alertCard(a);
  el.dataset.key = key;
  el.dataset.ids = String(a.id);
  if (prev) {
    const n = (+prev.dataset.n || 1) + 1;
    el.dataset.n = n;
    el.dataset.ids = `${prev.dataset.ids || ''}${prev.dataset.ids ? ',' : ''}${a.id}`;
    el.querySelector('.alert-kind').insertAdjacentHTML('beforeend', `<span class="alert-count">×${n}</span>`);
    prev.remove();
  }
  list.prepend(el);
  // Only the newest alert carries the full blue action; the rest step back.
  $$('.alert [data-act="track"]', list).forEach((b, i) => { b.classList.toggle('btn-primary', i === 0); b.classList.toggle('btn-secondary', i !== 0); });
  while (list.querySelectorAll('.alert').length > 5) [...list.querySelectorAll('.alert')].pop().remove();
  $('.alerts-empty')?.remove();
  renderAlertMeta();
}
let ALERT_SUM = null;
function renderAlertMeta() {
  const crit = $$('#alerts-list .alert.crit').length;
  if (!crit && !$('.alerts-empty')) $('#alerts-list').innerHTML = '<p class="alerts-empty">Nothing needs action. Cloned plates and watchlist hits appear here the moment the rules catch them.</p>';
  if (!ALERT_SUM) return;
  const k = ALERT_SUM.by_kind || {};
  const PL = { speeding: ['speeding', 'speeding'], loitering: ['loitering', 'loitering'], detour: ['unusual detour', 'unusual detours'], odd_hour: ['odd-hour movement', 'odd-hour movements'] };
  const quiet = ['speeding', 'loitering', 'detour', 'odd_hour'].filter((x) => k[x]).map((x) => `${k[x]} ${PL[x][k[x] === 1 ? 0 : 1]}`);
  $('#alerts-sum').textContent = quiet.length ? `Last ${ALERT_SUM.hours} h: ${quiet.join(' · ')}` : '';
}
async function loadAlerts() {
  try {
    const [rows, sum] = await Promise.all([
      api('/api/alerts?open_only=true&hours=6&limit=60'),
      api('/api/alerts/summary?hours=6'),
    ]);
    ALERT_SUM = sum;
    const crit = rows.filter((a) => a.severity === 'critical').slice(0, 8).reverse();
    for (const a of crit) addAlert(a, false);
    renderAlertMeta();
  } catch (err) { $('#alerts-list').innerHTML = `<p class="error">${icon('i-alert')}${esc(err.message)}</p>`; }
}

// ---------- tracking a vehicle on the map ----------
const TRACK = { plate: null, timers: [] };
function clearTrack() {
  TRACK.timers.forEach(clearTimeout); TRACK.timers = []; TRACK.plate = null;
  routeLayer.clearLayers();
  for (const id of Object.keys(NET.markers)) camEl(id)?.classList.remove('sel', 'clone');
  for (const rec of Object.values(NET.lines)) rec.line.setStyle({ opacity: .95 });
}
function drawRoute(traj) {
  clearTrack();
  TRACK.plate = traj.plate;
  for (const rec of Object.values(NET.lines)) rec.line.setStyle({ opacity: .38 });
  const pts = traj.sightings.map((s) => [s.lat, s.lon]);
  const bounds = L.latLngBounds(pts);
  MAP.flyToBounds(bounds, { ...padOpts(), duration: .6, maxZoom: 14 });
  // The signature moment: the route draws itself one sighting at a time.
  let i = 0;
  const legs = traj.legs || [];
  const step = () => {
    if (TRACK.plate !== traj.plate) return;
    const s = traj.sightings[i];
    if (s) {
      camEl(s.camera_id)?.classList.add('sel');
      const mk = L.circleMarker([s.lat, s.lon], { radius: 5, color: '#0c1015', weight: 2, fillColor: LEVEL_COLOR.blue, fillOpacity: 1 }).addTo(routeLayer);
      const last = traj.sightings.length - 1;
      const sameEnds = traj.sightings[0].camera_id === traj.sightings[last].camera_id;
      if (i === 0 && !sameEnds) mk.bindTooltip('First seen', { permanent: true, className: 'stop-tag', direction: 'left', offset: [-8, 0] });
      if (i === last) mk.bindTooltip(sameEnds ? 'First and last seen' : 'Last seen', { permanent: true, className: 'stop-tag', direction: 'right', offset: [8, 0] });
    }
    const leg = legs[i];
    if (leg && leg.polyline) {
      const p = leg.polyline;
      L.polyline(p, { color: '#0c1015', weight: 9, opacity: .9, interactive: false }).addTo(routeLayer);
      L.polyline(p, { color: leg.plausible === false ? LEVEL_COLOR.danger : LEVEL_COLOR.blue, weight: 5, opacity: 1,
        dashArray: leg.plausible === false ? '2 9' : null, lineCap: 'round' }).addTo(routeLayer);
    }
    i += 1;
    if (i < traj.sightings.length) TRACK.timers.push(setTimeout(step, Math.max(60, 900 / traj.sightings.length)));
  };
  step();
}
function showClone(plate, d) {
  clearTrack();
  TRACK.plate = plate;
  const a = NET.cameras[d.from]; const b = NET.cameras[d.to];
  if (!a || !b) return;
  camEl(a.id)?.classList.add('clone'); camEl(b.id)?.classList.add('clone');
  L.polyline([[a.lat, a.lon], [b.lat, b.lon]], { color: LEVEL_COLOR.danger, weight: 3, dashArray: '4 8', lineCap: 'round' })
    .bindTooltip(`${fmtPlate(plate)} · ${(+d.km).toFixed(1)} km in ${(+d.minutes).toFixed(1)} min = ${Math.round(d.implied_kmph)} km/h`, { permanent: true, className: 'clone-tag', direction: 'center' })
    .addTo(routeLayer);
  MAP.flyToBounds(L.latLngBounds([[a.lat, a.lon], [b.lat, b.lon]]), { ...padOpts(), duration: .6, maxZoom: 14 });
}

// ---------- vehicle panel ----------
let CUR = { plate: null, query: null, distance: 0 };
async function openVehicle(plate, query, distance) {
  if (!plate) return;
  CUR = { plate, query: query || null, distance: distance || 0 };
  show('vehicle');
  closeSpot();
  await renderVehicle();
}
$('#v-hours').addEventListener('change', () => renderVehicle());
async function renderVehicle() {
  const { plate, query, distance } = CUR;
  const hours = $('#v-hours').value;
  const box = $('#vehicle');
  box.innerHTML = `<div class="veh"><div class="skel" style="height:40px;width:70%"></div>
    <div class="skel" style="height:14px;width:45%;margin-top:12px"></div><div class="skel" style="height:64px;margin-top:16px"></div></div>`;
  let traj; let clone; let inf;
  try {
    [traj, clone, inf] = await Promise.all([
      api(`/api/plates/${encodeURIComponent(plate)}/trajectory?hours=${hours}`),
      api(`/api/plates/${encodeURIComponent(plate)}/clone-check?hours=${hours}`).catch(() => null),
      api(`/api/inferences?plate=${encodeURIComponent(plate)}&hours=${hours}&limit=50`).catch(() => null),
    ]);
  } catch (err) {
    const dym = err.detail && err.detail.did_you_mean;
    box.innerHTML = `<div class="veh"><div class="veh-head"><span class="plate lg">${fmtPlate(plate)}</span></div>
      <p class="error">${icon('i-alert')}<span>No sightings of ${fmtPlate(plate)} in the last ${hours} h.${dym && dym.length ? ` Closest reads: ${dym.map((p) => `<button class="btn btn-ghost" data-p="${esc(p)}">${esc(p)}</button>`).join(' ')}` : ''}</span></p></div>`;
    $$('[data-p]', box).forEach((b) => { b.onclick = () => openVehicle(b.dataset.p, plate); });
    clearTrack();
    return;
  }
  const sm = traj.summary || {};
  const legs = traj.legs || [];
  const bad = legs.filter((l) => l.plausible === false).length;
  const nInf = inf && inf.inferences ? inf.inferences.length : 0;
  const isHit = clone && clone.verdict && clone.verdict !== 'none';
  const qn = query ? query.replace(/\s/g, '').toUpperCase() : '';
  let diffNote = '';
  if (qn && qn.length === plate.length) {
    const n = [...qn].filter((ch, i) => ch !== plate[i]).length;
    diffNote = `; ${n} character${n === 1 ? '' : 's'} differ${n === 1 ? 's' : ''}, the kind a camera misreads`;
  }
  const near = qn && qn !== plate
    ? `<p class="near">You searched <b>${esc(qn)}</b>. The closest stored read is <b>${fmtPlate(plate)}</b>${diffNote}.</p>` : '';
  box.innerHTML = `<div class="veh">
    <div class="veh-head">
      <span class="plate lg">${fmtPlate(plate)}</span>
      <div class="veh-sub">
        <span class="badge plain">${esc(sm.first_camera_name || '')} → ${esc(sm.last_camera_name || '')}</span>
        ${nInf ? `<span class="badge infer">${nInf} inferred, not read</span>` : ''}
      </div>
      ${near}
      ${clone ? `<div class="verdict ${isHit ? 'hit' : 'none'}">${icon(isHit ? 'i-copy2' : 'i-check')}<span><b>${isHit ? (clone.verdict === 'confident' ? 'Cloned plate: confident' : 'Possible cloned plate') : 'One vehicle'}</b>${esc(clone.reason || '')}</span></div>` : ''}
      <div class="stats">
        <div class="stat"><div class="v">${sm.sightings ?? '–'}</div><div class="k">sightings</div></div>
        <div class="stat"><div class="v">${sm.cameras_visited ?? '–'}</div><div class="k">junctions</div></div>
        <div class="stat"><div class="v">${sm.distance_km != null ? Math.round(sm.distance_km) : '–'}<small>km</small></div><div class="k">road distance</div></div>
        <div class="stat"><div class="v">${sm.first_seen ? hm(sm.first_seen) : '–'}</div><div class="k">first seen</div></div>
        <div class="stat"><div class="v">${sm.last_seen ? hm(sm.last_seen) : '–'}</div><div class="k">last seen</div></div>
        <div class="stat"><div class="v">${sm.mean_ocr_confidence != null ? pct(sm.mean_ocr_confidence) : '–'}</div><div class="k">read confidence</div></div>
      </div>
      <div class="actions">
        <button class="btn btn-quiet" id="v-watch" type="button">${icon('i-flag')}Add to watchlist</button>
        <button class="btn btn-quiet" id="v-reg" type="button">${icon('i-file')}Registry</button>
      </div>
      <div id="v-reg-out"></div>
    </div>
    <h3 class="list-h">Route${bad ? `, ${bad} impossible leg${bad === 1 ? '' : 's'}` : ''} · ${legs.length} legs</h3>
    <ol class="legs">${legs.slice(-30).map((l, i) => `<li class="${l.plausible === false ? 'bad' : ''}">
        <span class="n">${legs.length > 30 ? legs.length - 30 + i + 1 : i + 1}</span>
        <span class="where">${esc(l.from_name)} → ${esc(l.to_name)}</span>
        <span class="when">${hm(l.departed_ts)} to ${hm(l.arrived_ts)} · ${(+l.road_km).toFixed(1)} km</span>
        <span class="spd">${Math.round(l.implied_kmph)} km/h</span></li>`).join('')}</ol>
    <p class="fine">Readings from the simulated camera feed, read by the real OCR engine. ${nInf ? 'Violet entries were inferred from neighbouring cameras, not read.' : ''}</p>
  </div>`;
  $('#v-watch').onclick = async () => {
    try { const r = await api('/api/watchlist', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ plate, reason: 'flagged by control room', severity: 'high' }) });
      toast(`${r.plate} is on the watchlist. ${r.historic_sightings} earlier sightings on record.`); }
    catch (err) { toast(`Could not add to the watchlist: ${err.message}`); }
  };
  $('#v-reg').onclick = async () => {
    const out = $('#v-reg-out');
    out.innerHTML = '<div class="skel" style="height:80px;margin-top:12px"></div>';
    try {
      const r = await api(`/api/plates/${encodeURIComponent(plate)}/registry`);
      const rows = Object.entries(r).filter(([, v]) => v !== null && typeof v !== 'object').slice(0, 8);
      out.innerHTML = `<div class="reg"><span class="badge sim">Mock registry, not VAHAN</span>${rows.map(([k, v]) => `<div class="reg-row"><span>${esc(k.replace(/_/g, ' '))}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
    } catch (err) { out.innerHTML = `<p class="error">${icon('i-alert')}${esc(err.message)}</p>`; }
  };
  drawRoute(traj);
  vehicleExtras(plate, hours);
}

async function vehicleExtras(plate, hours) {
  const host = $('#vehicle .veh'); if (!host) return;
  const wrap = document.createElement('div');
  host.querySelector('.veh-head').after(wrap);   // above the route legs, where it is seen
  const [fc, pat, co] = await Promise.all([
    api(`/api/plates/${encodeURIComponent(plate)}/forecast?top=3`).catch(() => null),
    api(`/api/plates/${encodeURIComponent(plate)}/pattern`).catch(() => null),
    api(`/api/plates/${encodeURIComponent(plate)}/co-travellers`).catch(() => null)]);
  if (CUR.plate !== plate) return;
  let html = '';
  if (fc && fc.known && fc.candidates && fc.candidates.length) {
    html += `<h3 class="list-h">Likely next junctions${fc.confident === false ? ', from general traffic, not this vehicle\'s habit' : ''}</h3>
      <ul class="rows">${fc.candidates.map((c) => `<li><span>${esc(c.camera_name)}</span><b>${pct(c.probability)}</b><span class="sub">due in about ${Math.max(1, Math.round(c.due_in_minutes))} min · ${c.km} km ${esc(c.direction || '')}</span></li>`).join('')}</ul>`;
  }
  if (pat && pat.known && pat.hours) {
    const hrs = Array.from({ length: 24 }, (_, h) => pat.hours[h] || 0); const mx = Math.max(1, ...hrs);
    html += `<h3 class="list-h">When it is usually seen${pat.reliable ? '' : ', too little history to call it a habit'}</h3>
      <div class="mini">${hrs.map((v, h) => `<i class="${h === pat.busiest_hour ? 'on' : ''}" style="height:${Math.max(4, (v / mx) * 100)}%" title="${h}:00 · ${v}"></i>`).join('')}</div>
      <div class="mini-lbl"><span>00</span><span>06</span><span>12</span><span>18</span><span>24</span></div>
      ${pat.usual_camera_name ? `<p class="hint">Most often at <b>${esc(pat.usual_camera_name)}</b>${pat.best_intercept ? `; best chance to intercept: ${esc(pat.best_intercept.camera_name)} around ${pat.best_intercept.hour}:00` : ''}.</p>` : ''}`;
  }
  const ct = co && co.co_travellers ? co.co_travellers.slice(0, 5) : [];
  if (ct.length) {
    html += `<h3 class="list-h">Seen with</h3><ul class="rows">${ct.map((x) => `<li><button class="link" data-p="${esc(x.plate)}">${esc(x.plate)}</button><b>${x.shared}×</b><span class="sub">together at ${x.camera_count} junctions</span></li>`).join('')}</ul>
      <p class="fine">Counted co-occurrences; busy corridors produce some by chance. Insights › Analytics › Convoys scores against chance.</p>`;
  }
  wrap.innerHTML = html;
  $$('[data-p]', wrap).forEach((b) => { b.onclick = () => openVehicle(b.dataset.p); });
}

async function openCamera(id) {
  const c = NET.cameras[id];
  toast(`${c.name}: showing its latest reads`);
  show('feed');
  try {
    const rows = await api(`/api/sightings/recent?limit=30&camera_id=${encodeURIComponent(id)}`);
    const ol = $('#feed'); ol.innerHTML = '';
    for (const s of rows) ol.append(feedItem(s, false));
    $('#feed-meta').innerHTML = `${esc(c.name)} · <button class="btn btn-ghost" id="feed-all" type="button">All junctions</button>`;
    $('#feed-all').onclick = loadFeed;
  } catch (err) { toast(err.message); }
}

// ---------- Spotlight search ----------
const q = $('#q'); const list = $('#spot-list');
let SPOT = { items: [], sel: -1, seq: 0 };
function closeSpot() { list.hidden = true; q.setAttribute('aria-expanded', 'false'); SPOT.sel = -1; }
function renderSpot(items, query) {
  SPOT.items = items; SPOT.sel = items.length ? 0 : -1;
  if (!items.length) {
    list.innerHTML = `<p class="spot-empty">No stored read is close to <b>${esc(query.toUpperCase())}</b> in the last 48 hours.</p>`;
  } else {
    list.innerHTML = items.map((m, i) => `<button type="button" role="option" class="spot-item" aria-selected="${i === 0}" data-i="${i}">
      <span class="plate">${fmtPlate(m.plate)}</span>
      <span>${m.clone_verdict && m.clone_verdict !== 'none' ? '<span class="badge crit">Possible clone</span>' : ''}${m.watched ? ' <span class="badge warn">Watchlist</span>' : ''}</span>
      <span class="dist ${m.exact ? 'exact' : ''}">${m.exact ? 'Exact' : `${m.distance} apart`}</span>
      <span class="meta">${m.sightings} sightings · last at ${esc(NET.cameras[m.last_camera]?.name || m.last_camera)} · ${ago(m.last_ts)}</span>
    </button>`).join('');
    $$('.spot-item', list).forEach((b) => { b.onclick = () => pick(+b.dataset.i); });
  }
  list.hidden = false; q.setAttribute('aria-expanded', 'true');
}
function pick(i) {
  const m = SPOT.items[i]; if (!m) return;
  openVehicle(m.plate, q.value.trim(), m.exact ? 0 : m.distance);
}
let spotT;
q.addEventListener('input', () => {
  clearTimeout(spotT);
  const v = q.value.trim();
  if (v.length < 2) { closeSpot(); return; }
  spotT = setTimeout(async () => {
    const seq = ++SPOT.seq;
    try {
      const r = await api(`/api/plates/search?q=${encodeURIComponent(v)}&limit=8&hours=48`);
      if (seq === SPOT.seq) renderSpot(r.matches || [], v);
    } catch (err) { if (seq === SPOT.seq) { list.innerHTML = `<p class="spot-empty">${esc(err.message)}</p>`; list.hidden = false; } }
  }, 160);
});
q.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    if (list.hidden || !SPOT.items.length) return;
    e.preventDefault();
    SPOT.sel = (SPOT.sel + (e.key === 'ArrowDown' ? 1 : -1) + SPOT.items.length) % SPOT.items.length;
    $$('.spot-item', list).forEach((b, i) => b.setAttribute('aria-selected', String(i === SPOT.sel)));
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (SPOT.sel >= 0) pick(SPOT.sel);
  } else if (e.key === 'Escape') { closeSpot(); q.blur(); }
});
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== q && !/input|textarea|select/i.test(document.activeElement.tagName)) { e.preventDefault(); q.focus(); q.select(); }
});
document.addEventListener('click', (e) => { if (!e.target.closest('.spot')) closeSpot(); });

// ---------- read a plate ----------
const drop = $('#drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) readPhoto(f); });
$('#file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) readPhoto(f); e.target.value = ''; });

async function readPhoto(file) {
  const out = $('#read-out');
  const url = URL.createObjectURL(file);
  out.innerHTML = `<div class="read"><div class="read-img"><img src="${url}" alt="Uploaded photo"></div>
    <div class="skel" style="height:38px;width:60%;margin-top:14px"></div><div class="skel" style="height:8px;margin-top:16px"></div></div>`;
  const fd = new FormData(); fd.append('file', file);
  let r;
  try { r = await api('/api/anpr/read?history=true', { method: 'POST', body: fd }); }
  catch (err) { out.innerHTML = `<div class="read"><div class="read-img"><img src="${url}" alt="Uploaded photo"></div><p class="error">${icon('i-alert')}${esc(err.message)}</p></div>`; return; }
  renderRead(out, r, url, null, file);
}

function renderRead(out, r, url, synth, file) {
  const stored = !!r.accepted && !!r.text;
  const conf = r.confidence || 0; const floor = r.floor || 0;
  const chars = (r.text || '').split('');
  const cc = r.char_confidences || [];
  const synthBackend = r.backend && r.backend !== 'anpr_ocr';
  out.innerHTML = `<div class="read">
    <div class="read-img"><img src="${url}" alt="${synth ? 'Synthetic capture' : 'Uploaded photo'}"></div>
    ${synth ? `<p class="fine"><span class="badge sim">Synthetic capture</span> Rendered by the camera model under <b>${esc(synth.condition.replace(/_/g, ' '))}</b>; true plate ${esc(synth.truth)}, read ${synth.correct ? 'correctly' : 'wrongly'}.</p>` : ''}
    <div class="answer">
      <div class="answer-top">
        <span class="plate lg">${r.text ? esc(r.pretty || r.text) : 'No plate'}</span>
        <span class="answer-verdict ${stored ? 'stored' : 'refused'}">${stored ? 'Accepted' : 'Refused'}</span>
      </div>
      <div class="meter" title="Confidence ${pct(conf)}, floor ${pct(floor)}">
        <div class="meter-fill ${stored ? '' : 'refused'}" style="width:${Math.min(100, conf * 100)}%"></div>
        <div class="meter-floor" style="left:${floor * 100}%"></div>
      </div>
      <div class="meter-lbl"><span>Confidence <b>${pct(conf)}</b></span><span>Floor <b>${pct(floor)}</b></span></div>
      ${chars.length && cc.length === chars.length ? `<div class="chars">${chars.map((ch, i) => `<span class="${cc[i] < .9 ? 'lo' : ''}" title="${pct(cc[i])}">${esc(ch)}<i style="opacity:${Math.max(.35, cc[i])}"></i></span>`).join('')}</div>` : ''}
      ${!stored ? `<div class="reason">${icon('i-block')}<span>${esc(r.reason || (r.plate_found ? 'Below the confidence floor, so it was not stored. A wrong plate in the database is worse than none.' : 'No plate was found in this photo.'))}</span></div>` : ''}
      <div class="facts"><span>Reader <b>${esc(r.backend || '–')}</b></span><span>Time <b>${r.ms} ms</b></span>${r.image_size ? `<span>Image <b>${r.image_size.join('×')}</b></span>` : ''}${synth ? `<span>Burst <b>${synth.frames} frames</b></span>` : ''}</div>
      ${synthBackend && !synth ? '<p class="fine">This server is running the synthetic-trained reader. Real photographs read far better with the anpr_ocr reader (start the API from .venv-paddle).</p>' : ''}
      ${r.history && r.history.known ? `<div class="actions"><button class="btn btn-primary" id="r-track" type="button">${icon('i-route')}Track ${fmtPlate(r.text)} · ${r.history.summary.sightings} sightings</button></div>` : (stored ? `<p class="fine">Clears the ${pct(floor)} floor, so the platform would store it. No earlier sightings of this plate on the network.</p>` : '')}
    </div></div>`;
  const t = $('#r-track'); if (t) t.onclick = () => openVehicle(r.text);
  // A real photograph that clears the floor can be registered as a sighting at
  // the junction it was taken at; the server re-reads it, so nothing is trusted.
  if (file && stored) {
    const opts = Object.values(NET.cameras).sort((a, b) => a.name.localeCompare(b.name))
      .map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('');
    out.querySelector('.answer').insertAdjacentHTML('beforeend', `<div class="demo"><span class="demo-t">Register this read as a sighting</span>
      <div class="demo-row"><label class="win"><span class="sr">Junction</span><select id="enrol-cam">${opts}</select></label>
      <button class="btn btn-quiet" id="enrol-go" type="button">${icon('i-pin')}Enrol</button></div></div>`);
    $('#enrol-go').onclick = async () => {
      const fd = new FormData(); fd.append('file', file);
      try {
        const e = await api(`/api/enrol?camera_id=${encodeURIComponent($('#enrol-cam').value)}`, { method: 'POST', body: fd });
        toast(`${e.plate} registered at ${NET.cameras[e.camera_id]?.name || e.camera_id} as a real-photo sighting`);
        openVehicle(e.plate);
      } catch (err) { toast(err.message); }
    };
  }
}

$('#demo-go').addEventListener('click', async () => {
  const out = $('#read-out');
  const condition = $('#demo-cond').value;
  out.innerHTML = '<div class="read"><div class="skel" style="height:120px"></div><div class="skel" style="height:38px;width:60%;margin-top:14px"></div></div>';
  try {
    const r = await api('/api/anpr/demo', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ condition, frames: 5, history: true }) });
    renderRead(out, r, `data:image/png;base64,${r.image_png_b64}`, r);
  } catch (err) { out.innerHTML = `<p class="error">${icon('i-alert')}${esc(err.message)}</p>`; }
});


// ---------- insights drawer ----------
let AUDIBLE = false;
try { AUDIBLE = localStorage.getItem('godseye-audible') === '1'; } catch { /* private window */ }
const SOUND = new Audio('/static/alert.mp3');
const moreBtn = $('#more-btn'); const drawer = $('#drawer');
let TAB = 'analytics';
function setDrawer(open) {
  drawer.hidden = !open; moreBtn.setAttribute('aria-expanded', String(open));
  document.body.classList.toggle('drawer-on', open);
  if (open) loadTab(TAB);
}
moreBtn.addEventListener('click', () => setDrawer(drawer.hidden));
$('#drawer-close').addEventListener('click', () => setDrawer(false));
for (const b of $$('.drawer-head .seg button')) b.addEventListener('click', () => {
  TAB = b.dataset.tab;
  for (const x of $$('.drawer-head .seg button')) { const on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-pressed', String(on)); }
  for (const t of $$('.tab')) t.hidden = t.dataset.tab !== TAB;
  loadTab(TAB);
});
const levelBadge = (lv) => `<span class="badge ${lv === 'free' ? 'ok' : lv === 'moderate' ? 'warn' : 'crit'}">${esc(lv)}</span>`;
const loading = '<div class="skel" style="height:90px;margin-top:8px"></div><div class="skel" style="height:140px;margin-top:12px"></div>';

async function loadTab(tab) {
  const el = $(`.tab[data-tab="${tab}"]`);
  el.innerHTML = loading;
  try { await TABS[tab](el); } catch (err) { el.innerHTML = `<p class="error">${icon('i-alert')}${esc(err.message)}</p>`; }
}
const TABS = {
  async analytics(el) {
    const [sum, bn, sec, od, tr, cv] = await Promise.all([
      api(`/api/analytics/summary?minutes=${WINDOW}`), api(`/api/analytics/bottlenecks?minutes=${WINDOW}&top=6`),
      api(`/api/analytics/sectors?minutes=${WINDOW}`), api('/api/analytics/od?minutes=120&top=8'),
      api('/api/analytics/trend?hours=3&bucket_min=5'), api('/api/convoys?hours=2').catch(() => []),
    ]);
    const dens = (await api(`/api/analytics/density?minutes=${WINDOW}`).catch(() => [])).sort((a, b) => b.count - a.count);
    const secVal = (r) => r.sightings ?? r.count ?? r.vehicles ?? 0;
    const maxSec = Math.max(1, ...sec.map(secVal));
    const flow = tr.flow || [];
    const fkey = flow[0] ? Object.keys(flow[0]).find((k) => k !== 'bucket') : null;
    const maxF = Math.max(1, ...flow.map((r) => r[fkey] || 0));
    const groups = Array.isArray(cv) ? cv : (cv.groups || []);
    el.innerHTML = `
      <section class="block">
        <div class="block-head"><h2>Network</h2>
          <label class="win"><span class="sr">Analysis window</span><select id="win">
            ${[15, 30, 60, 120, 360].map((m) => `<option value="${m}" ${m === WINDOW ? 'selected' : ''}>Last ${m < 60 ? `${m} min` : `${m / 60} h`}</option>`).join('')}
          </select></label></div>
        <div class="stats">
          <div class="stat"><div class="v">${Math.round(sum.median_network_kmph ?? 0)}<small>km/h</small></div><div class="k">median speed</div></div>
          <div class="stat"><div class="v">${sum.congested_links}<small>/ ${sum.monitored_links}</small></div><div class="k">congested links</div></div>
          <div class="stat"><div class="v">${sum.unique_plates}</div><div class="k">vehicles</div></div>
        </div>
        ${sum.worst_link ? `<p class="hint">Worst corridor: <b>${esc(sum.worst_link.link)}</b>, ${Math.round(sum.worst_link.median_kmph)} km/h against ${Math.round(sum.worst_link.free_kmph)} free flow, costing ${Math.round(sum.worst_link.delay_min_total)} vehicle-minutes.</p>` : ''}
      </section>
      <section class="block"><div class="block-head"><h2>Bottlenecks</h2><span class="hint-inline">by delay caused</span></div>
        <ul class="rows">${bn.map((b) => `<li><button class="link" data-a="${esc(b.from_camera || b.from || '')}" data-b="${esc(b.to_camera || b.to || '')}">${esc(b.name || b.link)}</button>${levelBadge(b.level)}
          <span class="sub">${Math.round(b.median_kmph)} km/h against ${Math.round(b.free_kmph)} free · ${Math.round(b.delay_min_total ?? 0)} vehicle-minutes lost${b.vehicles != null ? ` · ${b.vehicles} vehicles` : ''}</span></li>`).join('') || '<li class="sub">No congested links in this window.</li>'}</ul></section>
      <section class="block"><div class="block-head"><h2>Flow</h2><span class="hint-inline">reads per 5 min, last 3 h</span></div>
        <div class="spark">${flow.map((r) => `<i style="height:${Math.max(4, ((r[fkey] || 0) / maxF) * 100)}%" title="${esc(r.bucket)}: ${r[fkey]}"></i>`).join('')}</div>
        <div class="spark-lbl"><span>${flow[0] ? esc(String(flow[0].bucket).slice(11, 16)) : ''}</span><span>now</span></div></section>
      <section class="block"><div class="block-head"><h2>Busiest junctions</h2>
          <div class="toggle" style="gap:8px"><span class="hint-inline">Heatmap</span><input type="checkbox" id="heat" ${HEAT.on ? 'checked' : ''} aria-label="Movement heatmap on the map"></div></div>
        <ul class="rows">${dens.slice(0, 6).map((d) => `<li><button class="link" data-cam="${esc(d.camera_id)}">${esc(d.name)}</button><b>${Math.round(d.per_min * 10) / 10}/min</b><span class="sub">${esc(d.road)} · ${d.unique_plates} vehicles · ${pct(d.load)} of capacity</span></li>`).join('')}</ul></section>
      <section class="block"><div class="block-head"><h2>Load by sector</h2></div>
        <div class="bars">${sec.map((r) => `<div class="bar"><span>${esc(r.sector)}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${(secVal(r) / maxSec) * 100}%"></span></span><b>${secVal(r)}</b></div>`).join('')}</div></section>
      <section class="block"><div class="block-head"><h2>Busiest journeys</h2><span class="hint-inline">origin to destination, 2 h</span></div>
        <ul class="rows">${od.map((r) => `<li><span>${esc(r.origin_name || r.origin)} → ${esc(r.destination_name || r.destination)}</span><b>${r.trips ?? r.vehicles ?? r.count ?? ''}</b></li>`).join('') || '<li class="sub">Not enough journeys yet.</li>'}</ul></section>
      <section class="block"><div class="block-head"><h2>Convoys</h2><span class="hint-inline">moving together, 2 h</span></div>
        <ul class="rows">${groups.slice(0, 5).map((g) => { const ps = g.plates || g.members || []; return `<li><span>${ps.slice(0, 4).map((p) => `<span class="plate">${esc(p)}</span>`).join(' ')}</span><b>${g.size ?? ps.length}</b>${g.note || g.summary ? `<span class="sub">${esc(g.note || g.summary)}</span>` : ''}</li>`; }).join('') || '<li class="sub">No groups moving together beyond chance.</li>'}</ul></section>`;
    $('#win').onchange = (e) => { WINDOW = +e.target.value; refreshSummary(); refreshLinks(); loadTab('analytics'); };
    $$('.rows button.link[data-a]', el).forEach((b) => { b.onclick = () => focusLink(b.dataset.a, b.dataset.b); });
    $$('.rows button.link[data-cam]', el).forEach((b) => { b.onclick = () => { const c = NET.cameras[b.dataset.cam]; MAP.flyTo([c.lat, c.lon], 15, { duration: .5 }); NET.markers[c.id].openTooltip(); }; });
    $('#heat').onchange = (e) => toggleHeat(e.target.checked);
  },

  async engine(el) {
    const [real, bench, model] = await Promise.all([
      api('/api/anpr/real').catch(() => null), api('/api/anpr/benchmark').catch(() => null), api('/api/anpr/model').catch(() => null)]);
    const names = { anpr_ocr: 'anpr_ocr, pretrained on real plates', crnn: 'CRNN, synthetic-trained', easyocr: 'EasyOCR', classical: 'Classical segmentation' };
    const sumR = real ? Object.entries(real.summary).sort((a, b) => b[1].mean - a[1].mean) : [];
    const conds = bench ? bench.conditions || [] : [];
    const ov = bench && bench.overall;
    el.innerHTML = `
      <section class="block"><div class="block-head"><h2>Real photographs</h2><span class="badge real">Measured</span></div>
        <p class="big">91.5%<small>exact plates, whole photos</small></p>
        <p class="hint">End to end, localiser plus recogniser, mean of 8 held-out splits; every split at least 90%. Small corpus, mostly Maharashtra plates (README §1, QUALITY.md).</p>
        ${real ? `<p class="list-h">Crops held out by vehicle, per reader</p><div class="bars">${sumR.map(([b, v]) => `<div class="bar"><span title="${esc(names[b] || b)}">${esc(b)}</span><span class="bar-track"><span class="bar-fill ${b === 'anpr_ocr' ? 'real' : 'sim'}" style="display:block;width:${v.mean * 100}%"></span></span><b>${pct(v.mean)}</b></div>`).join('')}</div>
        <p class="fine">${esc(real.corpus)}. Readers trained only on synthetic captures score near zero on real photographs, which is why real inputs use anpr_ocr.</p>` : ''}
      </section>
      <section class="block"><div class="block-head"><h2>Synthetic stress test</h2><span class="badge sim">Simulated</span></div>
        ${bench ? `${ov ? `<p class="hint">${pct(ov.burst_accuracy)} with 5-frame burst, ${pct(ov.accepted_accuracy)} of stored reads correct, over ${ov.samples} captures.</p>` : ''}
        <table class="tbl"><thead><tr><th>Condition</th><th>Single</th><th>Burst</th><th>Stored correct</th></tr></thead><tbody>
        ${conds.map((c) => `<tr><td>${esc(String(c.condition).replace(/_/g, ' '))}</td><td>${pct(c.plate_accuracy)}</td><td>${pct(c.burst_accuracy)}</td><td>${pct(c.accepted_accuracy)}</td></tr>`).join('')}</tbody></table>` : '<p class="hint">No saved benchmark.</p>'}
      </section>
      ${model ? `<section class="block"><div class="block-head"><h2>This server's reader</h2></div>
        <ul class="rows"><li><span>Backend</span><b>${esc(model.backend)}</b></li><li><span>Confidence floor</span><b>${pct(model.confidence_floor)}</b></li>
        <li><span>Plate grammar patterns</span><b>${(model.grammar_patterns || []).length}</b></li></ul></section>` : ''}`;
  },

  async incidents(el) {
    const [watch, link, inf] = await Promise.all([
      api('/api/watchlist').catch(() => []), api('/api/judge/link').catch(() => null), api('/api/inferences?hours=6&limit=5').catch(() => null)]);
    const host = location.hostname;
    el.innerHTML = `
      <section class="block"><div class="block-head"><h2>Release an incident</h2></div>
        <p class="hint" style="margin:0 0 10px">Writes real sightings through the same rules a camera read takes. The alert stack has to catch it on its own.</p>
        <div class="inject">
          <button class="btn btn-danger" data-k="clone" type="button">Cloned plate</button>
          <button class="btn btn-quiet" data-k="loiterer" type="button">Loiterer</button>
          <button class="btn btn-quiet" data-k="watchlist" type="button">Wanted vehicle</button>
        </div></section>
      <section class="block"><div class="block-head"><h2>Judge console</h2></div>
        ${link ? `<p class="hint" style="margin:0">Hand this to an outside observer. They pick the moment; the control room is not told.</p>
        <div class="qr"><img src="/api/judge/qr" alt="QR code for the judge console"><code>${esc(link.url)}</code></div>` : '<p class="hint">The judge console is disabled on this server.</p>'}</section>
      <section class="block"><div class="toggle"><span>Sound for critical alerts</span><input type="checkbox" id="audible" ${AUDIBLE ? 'checked' : ''} aria-label="Sound for critical alerts"></div></section>
      <section class="block"><div class="block-head"><h2>Repair</h2>${inf && inf.summary ? `<span class="hint-inline">${inf.summary.unresolved_open ?? 0} refused captures open</span>` : ''}</div>
        <p class="hint" style="margin:0 0 10px">Refused captures are matched against what neighbouring cameras saw. Results are inferences, shown in violet, never sightings.</p>
        <button class="btn btn-quiet" id="repair" type="button">${icon('i-bolt')}Run repair now</button>
        <ul class="rows">${inf && inf.inferences ? inf.inferences.map((x) => `<li><span class="plate">${esc(x.plate)}</span><span class="badge infer">inferred</span><span class="sub">${esc(NET.cameras[x.camera_id]?.name || x.camera_id || '')}${x.ts ? ` · ${ago(x.ts)}` : ''}</span></li>`).join('') : ''}</ul></section>
      <section class="block"><div class="block-head"><h2>Watchlist</h2><span class="hint-inline">${watch.length} plates</span></div>
        <ul class="rows">${watch.map((w) => `<li><button class="link" data-p="${esc(w.plate)}">${esc(w.plate)}</button><button class="btn btn-ghost" data-rm="${esc(w.plate)}" type="button">Remove</button><span class="sub">${esc(w.reason)} · ${esc(w.severity)}</span></li>`).join('')}</ul></section>
      <section class="block"><div class="block-head"><h2>Phone cameras</h2></div>
        <p class="hint" style="margin:0">Phones become junction cameras through phonecam.bat, which runs the real-plate reader. Their reads land here as <span class="badge real">Real photo</span>.</p>
        <div class="actions"><a class="btn btn-quiet" href="http://${esc(host)}:8504/links" target="_blank" rel="noopener">${icon('i-camera')}Open phone cameras</a></div></section>`;
    $$('.inject button', el).forEach((b) => { b.onclick = async () => {
      b.disabled = true;
      try { const r = await api(`/api/sim/inject/${b.dataset.k}`, { method: 'POST' }); toast(`Released: ${b.textContent.trim().toLowerCase()} ${r.plate || ''}`); }
      catch (err) { toast(err.message); } finally { b.disabled = false; } }; });
    $('#audible').onchange = (e) => { AUDIBLE = e.target.checked; try { localStorage.setItem('godseye-audible', AUDIBLE ? '1' : '0'); } catch { /* ignore */ } };
    $('#repair').onclick = async () => {
      try { const r = await api('/api/repair/run?hours=6', { method: 'POST' }); toast(`Repair inferred ${r.inferred} registration${r.inferred === 1 ? '' : 's'}`); loadTab('incidents'); }
      catch (err) { toast(err.message); } };
    $$('[data-p]', el).forEach((b) => { b.onclick = () => openVehicle(b.dataset.p); });
    $$('[data-rm]', el).forEach((b) => { b.onclick = async () => {
      try { await api(`/api/watchlist/${encodeURIComponent(b.dataset.rm)}`, { method: 'DELETE' }); loadTab('incidents'); } catch (err) { toast(err.message); } }; });
  },

  async privacy(el) {
    const [p, acc] = await Promise.all([api('/api/privacy/posture'), api('/api/privacy/access?hours=168').catch(() => null)]);
    const st = p.store || {}; const rt = p.retention || {};
    const span = st.span_days == null ? '–' : st.span_days < 1 ? `${Math.round(st.span_days * 24)}<small>h</small>` : `${Math.round(st.span_days)}<small>days</small>`;
    el.innerHTML = `
      <section class="block"><div class="block-head"><h2>What is held</h2></div>
        <div class="stats">
          <div class="stat"><div class="v">${(st.sightings ?? 0).toLocaleString('en-IN')}</div><div class="k">sightings</div></div>
          <div class="stat"><div class="v">${st.unique_plates ?? 0}</div><div class="k">plates</div></div>
          <div class="stat"><div class="v">${span}</div><div class="k">history</div></div>
        </div>
        <p class="hint">Oldest ${esc(st.oldest || '–')}, newest ${esc(st.newest || '–')}.</p></section>
      <section class="block"><div class="block-head"><h2>Retention</h2></div>
        <ul class="rows"><li><span>Horizon</span><b>${rt.horizon_days ?? '–'} days</b></li><li><span>Cut-off</span><b>${esc(rt.cutoff || '–')}</b></li>
          <li><span>Rows past the horizon</span><b>${rt.rows_past_horizon ?? 0}</b></li><li><span>Held back for open alerts</span><b>${rt.rows_held_back ?? 0}</b></li></ul>
        <p class="fine">Deletion is a separate, explicit step that is not exposed over the network.</p></section>
      <section class="block"><div class="block-head"><h2>Who looked</h2><span class="hint-inline">last 7 days</span></div>
        ${acc ? `<ul class="rows"><li><span>Plate lookups</span><b>${acc.lookups}</b></li><li><span>Distinct plates</span><b>${acc.distinct_plates}</b></li>
          <li><span>Without a case reference</span><b>${acc.no_case_ref}</b></li></ul>` : '<p class="hint">No access log.</p>'}
        ${(p.caveats || []).map((c) => `<p class="caveat">${esc(c)}</p>`).join('')}</section>`;
  },
};

const HEAT = { on: false, layer: L.layerGroup() };
async function toggleHeat(on) {
  HEAT.on = on; HEAT.layer.clearLayers();
  if (!on) { HEAT.layer.remove(); return; }
  try {
    const pts = await api(`/api/analytics/heatmap?minutes=${WINDOW}`);
    const mx = Math.max(1, ...pts.map((p) => p.weight || 0));
    for (const p of pts) {
      const w = (p.weight || 0) / mx;
      L.circle([p.lat, p.lon], { radius: 180 + 900 * w, stroke: false, fillColor: LEVEL_COLOR.blue, fillOpacity: .08 + .22 * w, interactive: false }).addTo(HEAT.layer);
    }
    HEAT.layer.addTo(MAP);
  } catch (err) { toast(err.message); }
}

function focusLink(a, b) {
  const rec = NET.lines[pairKey(a, b)];
  if (!rec) return;
  MAP.flyToBounds(rec.line.getBounds(), { ...padOpts(), duration: .5, maxZoom: 15 });
  rec.line.openTooltip();
}

// ---------- live socket ----------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/ws/live`);
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'sighting') {
      const s = msg.data;
      const real = String(s.source || '').startsWith('enrolment');
      pingCam(s.camera_id, real);
      if (VIEW === 'feed' && !$('#feed-all')) pushFeed(s);
    } else if (msg.type === 'alert') {
      const a = msg.data;
      if (ALERT_SUM && ALERT_SUM.by_kind) ALERT_SUM.by_kind[a.kind] = (ALERT_SUM.by_kind[a.kind] || 0) + 1;
      addAlert(a, true);
      if (AUDIBLE && a.severity === 'critical') { SOUND.currentTime = 0; SOUND.play().catch(() => {}); }
    } else if (msg.type === 'hello') {
      const on = msg.data && msg.data.live_feed !== false;
      $('#live-chip').setAttribute('aria-pressed', String(on));
      $('#live-label').textContent = on ? 'Live' : 'Paused';
    }
  };
  ws.onclose = () => { setTimeout(connect, 2500); };
}

// ---------- boot ----------
(async function boot() {
  try { await loadNetwork(); }
  catch (err) { toast(`Could not load the camera network: ${err.message}`); }
  await Promise.all([loadFeed(), loadAlerts(), refreshSummary(), refreshLinks()]);
  connect();
  // Deep links an operator can share: #plate=WB34ZP9421, #insights=analytics, &heat=1
  const hp = new URLSearchParams(location.hash.slice(1));
  if (hp.get('plate')) openVehicle(hp.get('plate').toUpperCase(), hp.get('q') || null, +(hp.get('d') || 0));
  if (hp.get('insights') && TABS[hp.get('insights')]) {
    TAB = hp.get('insights');
    for (const x of $$('.drawer-head .seg button')) { const on = x.dataset.tab === TAB; x.classList.toggle('on', on); x.setAttribute('aria-pressed', String(on)); }
    for (const t of $$('.tab')) t.hidden = t.dataset.tab !== TAB;
    setDrawer(true);
  }
  if (hp.get('heat') === '1') toggleHeat(true);
  setInterval(refreshSummary, 10000);
  setInterval(() => pollRefused(false), 6000);
  setInterval(refreshLinks, 15000);
  setInterval(async () => { try { ALERT_SUM = await api('/api/alerts/summary?hours=6'); renderAlertMeta(); } catch { /* keep */ } }, 30000);
})();
