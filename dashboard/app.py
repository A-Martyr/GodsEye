"""GodsEye control room — Streamlit dashboard.

    streamlit run dashboard/app.py

Reads the sighting store directly, so it works whether or not the API process is
running; the API is only needed for the live feed and the incident injectors.
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
for _p in (ROOT, str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

import config
import livefeed
import maps
import registry_link
import style
from core import alerts as alert_rules
from core import analytics, db, forecast, reach, trajectory, tunnel
from core import clones as clone_mod
from core import enrol as enrol_mod
from core.network import network

st.set_page_config(page_title="GodsEye", page_icon="👁", layout="wide",
                   initial_sidebar_state="expanded")
style.apply()

SEVERITY_COLOUR = {"critical": "#e5484d", "high": "#f76808", "medium": "#f5d90a", "low": "#8e8e8e"}
LEVEL_COLOUR = {"free": [46, 160, 67], "moderate": [212, 167, 44],
                "heavy": [232, 106, 23], "severe": [214, 40, 46]}


# --- data access --------------------------------------------------------
@st.cache_resource
def net():
    return network()


@st.cache_resource(show_spinner=False)
def _warm_reader() -> bool:
    """Start loading the plate reader the moment the page is first served.

    cache_resource rather than a bare call: Streamlit reruns this script top to
    bottom on every interaction, and a warm-up per rerun would be a thread pool
    fighting over one GPU. Cached, it happens once for the life of the process.

    It runs on a background thread and nothing waits on it, so the control room
    paints immediately. The ANPR tab is several clicks away from where an
    operator starts, and by the time they get there the reader is hot - which
    turns the first Capture & read from about nine seconds into a tenth of one.
    """
    if not config.WARM_READER_ON_START:
        return False
    try:
        from anpr.ocr import warm_async
        warm_async()
    except Exception:
        return False           # a reader that will not warm still reads on demand
    return True


_warm_reader()


@st.cache_data(ttl=5)
def summary(minutes: float):
    return analytics.city_summary(minutes)


@st.cache_data(ttl=5)
def density(minutes: float) -> pd.DataFrame:
    return analytics.camera_density(minutes)


@st.cache_data(ttl=5)
def links(minutes: float) -> pd.DataFrame:
    return analytics.link_flows(minutes)


@st.cache_data(ttl=10)
def heat(minutes: float) -> pd.DataFrame:
    return analytics.heatmap(minutes)


@st.cache_data(ttl=10)
def od(minutes: float, gateways_only: bool) -> pd.DataFrame:
    return analytics.od_matrix(minutes, gateways_only=gateways_only)


@st.cache_data(ttl=10)
def trend(hours: float, bucket: float) -> pd.DataFrame:
    return analytics.flow_trend(hours, bucket)


@st.cache_data(ttl=5)
def recent(limit: int) -> pd.DataFrame:
    return pd.DataFrame(db.recent_sightings(limit))


def api(path: str, method: str = "get", **kw):
    try:
        r = getattr(requests, method)(f"{config.API_BASE}{path}", timeout=4, **kw)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.warning(f"API call failed ({exc}). Start it with "
                   f"`uvicorn api.main:app --port 8000` for live control.")
        return None


def _fit_zoom(stops: pd.DataFrame) -> float:
    """Pick a zoom that keeps every stop on screen."""
    span = max(float(stops["lat"].max() - stops["lat"].min()),
               float(stops["lon"].max() - stops["lon"].min()), 1e-4)
    for limit, z in ((0.01, 14), (0.03, 13), (0.06, 12), (0.12, 11), (0.25, 10)):
        if span <= limit:
            return z
    return 9.0


def clock(ts) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return "-"
    return time.strftime("%d %b %H:%M:%S", time.localtime(float(ts)))


def ago(ts) -> str:
    if not ts:
        return "-"
    m = (time.time() - float(ts)) / 60.0
    if m < 1:
        return "just now"
    if m < 60:
        return f"{m:.0f} min ago"
    return f"{m/60:.1f} h ago"


# --- sidebar ------------------------------------------------------------
# The masthead, as markup rather than st.title, for one reason: a product's
# name and what it is watching are one unit, and title()+caption() renders them
# as two stacked blocks with a paragraph gap between. Set tight, with the
# wordmark letterspaced and the subtitle dimmed, they read as a masthead.
st.sidebar.markdown(
    f"""<div style="line-height:1.25;margin:0 0 1.15rem 0">
      <div style="font-size:1.32rem;font-weight:600;letter-spacing:-0.015em;
                  color:#EDF2F8">GodsEye</div>
      <div style="font-size:0.7rem;font-weight:500;letter-spacing:0.09em;
                  text-transform:uppercase;color:#5C6B7E;margin-top:0.28rem">
        {net().city} · {len(net())} cameras
      </div>
    </div>""",
    unsafe_allow_html=True)

window = st.sidebar.select_slider("Analysis window",
                                  options=[10, 15, 30, 45, 60, 120, 180, 360],
                                  value=config.DEFAULT_WINDOW_MIN,
                                  format_func=lambda m: f"{m} min" if m < 60 else f"{m//60} h")
auto = st.sidebar.checkbox("Auto-refresh (5 s)", value=False)
st.session_state["clone_alarm_on"] = st.sidebar.checkbox(
    "Audible alerts", value=config.CLONE_ALARM_ENABLED,
    help="Sounds once when a cloned plate is detected, and once when a live "
         "camera's read raises an alert. Browsers only allow a page to play "
         "audio after someone has clicked on it, so click anywhere here first.")


def _sidebar_stats() -> None:
    """Live counters. Its own fragment so auto-refresh moves these numbers
    without rerunning the page and rebuilding every map on it."""
    stats = db.stats()
    if stats["sightings"] == 0:
        st.error("No sightings yet — run `python seed.py`")
    # One row of three rather than three stacked metrics. Stacked, they occupy
    # most of the rail and push the controls below the fold; side by side they
    # read as a status line, which is what they are - nobody acts on them, they
    # just say the platform is alive and how much it is holding.
    a, b, c = st.columns(3)
    style.kpi(a, "Sightings", f"{stats['sightings']:,}", compact=True)
    style.kpi(b, "Plates", f"{stats['unique_plates']:,}", compact=True)
    style.kpi(c, "Open alerts", f"{stats['open_alerts']:,}", compact=True)
    st.caption(f"{clock(stats['first_ts'])} → {clock(stats['last_ts'])}")

    # Whether real cameras are feeding the platform, visible from every tab.
    # Silent when the service is not running - it is optional, not broken.
    cam = livefeed.status()
    if cam is not None:
        up = len([j for j in cam.get("junctions", []) if j.get("online")])
        st.caption(f"🟢 {up} live camera{'' if up == 1 else 's'} connected"
                   if up else ":grey[live service up · no camera connected]")


with st.sidebar:
    st.fragment(run_every="5s" if auto else None)(_sidebar_stats)()

with st.sidebar.expander("Live feed & incidents"):
    st.caption("Needs the API process running.")
    c1, c2 = st.columns(2)
    if c1.button("Feed on", width="stretch"):
        api("/api/sim/live/on", "post")
    if c2.button("Feed off", width="stretch"):
        api("/api/sim/live/off", "post")
    for kind, label in (("clone", "Inject cloned plate"),
                        ("loiterer", "Inject loiterer"),
                        ("watchlist", "Flag a random vehicle")):
        if st.button(label, width="stretch"):
            res = api(f"/api/sim/inject/{kind}", "post")
            if res:
                st.success(f"{res['injected']}: {res['plate']}")

def render_forecast(plate: str) -> None:
    """Where this vehicle is likely to go next, and when it is usually where.

    The two questions a sighting log cannot answer on its own, and the two
    that decide whether anyone can act on it. Everything else on this tab is
    history; this is the only part that points forward.

    Both halves lead with how much evidence is behind them. A ranked list of
    junctions looks exactly as authoritative when it rests on two observations
    as on two hundred, and an interception sent to the wrong junction costs
    more than one never sent - so `confident` and `reliable` are rendered
    before the numbers they qualify, not in a footnote under them.
    """
    # Open by default. This is the only panel on the page that points forward
    # rather than backward, and behind a collapsed expander it was a feature
    # nobody found unless they were told to click - which at a judging table
    # means nobody found it.
    with st.expander("Where it is likely to go next", expanded=True):
        c1, c2 = st.columns([3, 2])

        with c1:
            fc = forecast.predict(plate, net=net())
            if not fc.get("known"):
                st.caption(fc.get("reason", "nothing known about this registration."))
            elif not fc.get("candidates"):
                st.caption(fc.get("reason", "no onward link from where it was last seen."))
            else:
                st.caption(f"Last seen at **{fc['last_camera_name']}**, "
                           f"{fc['minutes_since_seen']:.0f} min ago.")
                if not fc["confident"]:
                    st.warning(f"**Not enough history to call this a habit.** {fc['basis']}.")
                rows = []
                for c in fc["candidates"]:
                    due = c["due_in_minutes"]
                    rows.append({
                        "Junction": c["camera_name"],
                        "Chance": f"{c['probability']:.0%}",
                        "Road km": c["km"],
                        "Dir": c["direction"],
                        "Typical": f"{c['typical_minutes']} min" if c["typical_minutes"] else "-",
                        # Negative means it should already have arrived and has
                        # not been seen - which is a fact about the vehicle, not
                        # a gap in the model, so it is shown rather than clipped.
                        "Due": ("overdue" if due is not None and due < 0
                                else (f"in {due:.0f} min" if due is not None else "-")),
                        "From": "seen" if c["from_observation"] else "road model",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Chance blends this vehicle's own turns at that junction with "
                           "what traffic in general does there - so one observed turn "
                           "cannot read as a certainty. 'Typical' is the median observed "
                           "time on that road, not its free-flow limit.")

        with c2:
            pol = forecast.pattern_of_life(plate, net=net())
            st.markdown("**When it is usually about**")
            if not pol.get("known") or not pol.get("sightings"):
                st.caption("No history for this registration.")
            else:
                s_rel = pol.get("spatial_reliable", pol.get("reliable", False))
                t_rel = pol.get("temporal_reliable", pol.get("reliable", False))
                if not s_rel and not t_rel:
                    st.caption(":orange[Not enough routine to distinguish from chance.]")
                elif s_rel and not t_rel:
                    st.caption(":orange[Uses this camera, but at unpredictable hours.]")
                elif t_rel and not s_rel:
                    st.caption(":orange[Active at these hours regularly, but on varying routes.]")
                bi = pol["best_intercept"]
                st.metric("Most often seen at", bi["camera_name"],
                          f"{bi['hour']:02d}:00 · {bi['share']:.0%} of sightings")
                hours = pol["hours"]
                st.bar_chart(pd.DataFrame({"sightings": list(hours.values())},
                                          index=[f"{h:02d}" for h in hours]),
                             height=150)


def render_sequence_reader(backend, load_engine, read_or_explain) -> None:
    """Read several frames of the same scene as one pass, fused per vehicle.

    This is the panel that shows what the platform is actually for. A camera
    does not photograph a vehicle once; it watches it cross a zone, and the
    frames fail in different ways - a plate lost to glare in one is legible in
    the next. Measured here, that is worth more than any single-frame change:
    52% to 88% across the ten synthetic camera conditions, and 91% per frame to
    97% per vehicle on the real-photograph corpus.

    Kept beside the single-image reader rather than replacing it, because the
    single-image number is the one the README quotes and the comparison is the
    point.
    """
    # Open by default, for the same reason: multi-frame fusion is the strongest
    # single result the reader has - 91% per frame to 97% per vehicle on the
    # real corpus - and it was the panel furthest down the page behind the
    # tightest-shut door.
    with st.expander("Read a sequence — several frames of the same scene", expanded=True):
        st.caption("Upload consecutive frames of one scene: a burst, or stills from a "
                   "video. Plates are tracked across the frames and each vehicle's "
                   "reads are voted, weighted by the exact CTC score. A frame that "
                   "misreads is outvoted by the ones that did not.")
        seq = st.file_uploader("Frames, in order", type=["png", "jpg", "jpeg", "bmp"],
                               accept_multiple_files=True, key="seq_upload")
        if not seq:
            return
        if len(seq) < 2:
            st.info("One frame is a photograph, not a sequence - the panel below "
                    "already reads those. Add more frames of the same scene.")
            return


        from anpr import imageio
        raws = [f.getvalue() for f in seq]

        # Cached against the frames themselves, for the reason the single-image
        # branch below is: Streamlit re-runs this script top to bottom on every
        # widget interaction ANYWHERE on the page, and it renders every tab
        # body, not just the visible one. Without this, leaving frames in the
        # uploader made a whole tracked sequence re-read on every click - a
        # tenth of a second per frame, so ten frames put one and a half seconds
        # on every unrelated interaction, including the single-image upload
        # underneath. Precisely the hazard the comment below warns about,
        # multiplied by the frame count.
        sig = f"seq:{backend}:{len(raws)}:" + hashlib.sha1(b"".join(raws), usedforsecurity=False).hexdigest()
        cached = st.session_state.get("anpr_seq_read")
        if cached and cached[0] == sig:
            found, elapsed = cached[1], cached[2]
        else:
            frames = [imageio.load_bytes(r) for r in raws]
            frames = [f for f in frames if f is not None]
            if not frames:
                st.error("Could not decode those images.")
                return
            t0 = time.perf_counter()
            eng = load_engine(backend)
            with st.spinner(f"Tracking plates across {len(frames)} frames…"):
                ok, found = read_or_explain(backend, lambda: eng.read_sequence(frames))
            if not ok:
                return
            elapsed = 1000.0 * (time.perf_counter() - t0)
            st.session_state["anpr_seq_read"] = (sig, found, elapsed)

        if not found:
            st.warning(f"No registration cleared the confidence floor in any of "
                       f"these {len(raws)} frames.")
        for track, fused, per_frame in found:
            hits = sum(1 for r in per_frame if r.text == fused.text)
            st.markdown(f"### `{fused.pretty}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{fused.confidence:.0%}")
            c2.metric("Frames seen", track.length)
            c3.metric("Frames agreeing", f"{hits}/{len(per_frame)}")
            # What every frame said, including the ones that were wrong. The
            # disagreement is the evidence: a vote nobody contested is just a
            # single read with extra steps.
            st.caption("what each frame read: " + " · ".join(
                f"`{r.text or '-'}` {r.confidence:.2f}" for r in per_frame))
        st.markdown(
            f"<div style='font-size:0.7rem;opacity:0.4'>"
            f"{elapsed:,.0f} ms · {len(raws)} frames · {backend}</div>",
            unsafe_allow_html=True)


def qr_png(link: str) -> bytes | None:
    """A QR for a link, or None if the qrcode package is missing.

    Scanning is the whole point. An address like 192.168.29.204:8000 is four
    numbers and a port to get right on a phone keyboard, in a noisy room,
    under time pressure - and the public link, with its token, is worse.
    """
    try:
        import io as _io

        import qrcode
    except ImportError:
        return None
    try:
        q = qrcode.QRCode(box_size=9, border=2,
                          error_correction=qrcode.constants.ERROR_CORRECT_M)
        q.add_data(link)
        q.make(fit=True)
        buf = _io.BytesIO()
        q.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


@st.cache_resource
def qr_available() -> bool:
    """Can this interpreter draw a QR at all?

    `qrcode` is an optional import and qr_png() swallows its absence, which is
    right - a missing package must not take the sidebar down. But swallowed
    silently it produces a panel that looks complete and simply has no QR on it,
    and the person demoing finds out at the table. Checked once per process so
    the expander can say so instead.

    Note this is per INTERPRETER, not per machine: the paddle environment that
    runs the 8502 reader is a separate virtualenv and had no `qrcode` in it even
    though the main one did.
    """
    return qr_png("https://example.invalid/probe") is not None


@st.cache_data(ttl=20, show_spinner=False)
def judge_routes() -> list[dict]:
    """Every way in, best first. Cached briefly; each call asks the network.

    Short TTL rather than none: this runs netsh and PowerShell, which is too
    much to repeat on every rerun of a page that reruns itself every few
    seconds, but a stale answer here is exactly the failure being fixed. The
    tunnel button clears this explicitly rather than waiting the TTL out,
    because a link that appears twenty seconds after the tunnel is up reads as
    a link that did not work.
    """
    live = tunnel.status()
    return [{"kind": c.kind, "url": c.url, "label": c.label, "detail": c.detail,
             "works_off_lan": c.works_off_lan, "warnings": c.warnings}
            for c in reach.candidates(live.url or None)]


if config.JUDGE_CONSOLE_ENABLED:
    with st.sidebar.expander("Judge link", expanded=False):
        st.caption("Hand this to a judge, or send it to someone who is not here. "
                   "They can release a cloned plate whenever they like - nothing "
                   "on this screen tells us they did, so the platform has to find "
                   "it on its own.")

        if not qr_available():
            st.caption(":orange[QR codes are off — `qrcode` is not installed in "
                       "this environment. The links below still work; they just "
                       "have to be typed.] `pip install qrcode[pil]`")

        _routes = judge_routes()
        _public = next((r for r in _routes if r["kind"] == "tunnel"), None)

        # --- the link that reaches someone who is not in the room ---------
        st.markdown("**Anywhere in the world**")
        if _public:
            st.success("Public link is live.")
            _png = qr_png(_public["url"])
            if _png:
                st.image(_png, caption="Point a phone camera at this", width="stretch")
            st.code(_public["url"], language=None)
            st.caption("Send this to anyone, on any network. It stops working when "
                       "the tunnel is stopped, and the address is different next time.")
            c1, c2 = st.columns(2)
            if c1.button("Stop sharing", width="stretch"):
                tunnel.stop()
                judge_routes.clear()
                st.rerun()
            if c2.button("New link", width="stretch",
                         help="Revokes the token, so any link already sent stops working."):
                reach.rotate_token()
                judge_routes.clear()
                st.rerun()
        elif not tunnel.available():
            st.info(f"No public link. `{config.TUNNEL_BIN}` is not installed - it is a "
                    "single executable from Cloudflare and needs no account. Without "
                    "it, only the same-network links below will work.")
        else:
            st.caption("Off by default: it publishes this demo on the internet for as "
                       "long as it runs.")
            if st.button("Create a public link", type="primary", width="stretch"):
                with st.spinner("Asking Cloudflare for an address (about five seconds)..."):
                    _res = tunnel.start()
                judge_routes.clear()
                if _res.running:
                    st.rerun()
                else:
                    st.error(_res.detail)

        # --- the links that need the judge to be here ---------------------
        st.markdown("**Same network only**")
        _lan = [r for r in _routes if r["kind"] == "lan"]
        if not _lan:
            st.warning("This machine has no network address - it is not on any "
                       "network right now. Only the public link, or the laptop "
                       "itself, can reach the console.")
        # The QR matters MORE here than on the public link, not less. The judge
        # this route is for is standing at the table with a phone, and
        # "192.168.29.204:8000/j" is four numbers, a port and a path to get
        # right on a phone keyboard in a noisy room. It used to be printed as
        # bare text while the tunnel link - the one you might never create - got
        # the QR.
        #
        # Only the first LAN address is given one. A laptop with Wi-Fi and
        # Ethernet up offers several, they reach the same console, and three
        # QR codes stacked in a sidebar is a puzzle rather than a shortcut.
        for _i, _r in enumerate(_lan):
            if _i == 0:
                _lan_png = qr_png(_r["url"])
                if _lan_png:
                    st.image(_lan_png, caption="Point a phone camera at this",
                             width="stretch")
            st.code(_r["url"], language=None)
            st.caption(_r["label"])
            for _w in _r["warnings"]:
                st.caption(f":orange[{_w}]")

        _fw = reach.firewall_state()
        if _fw.verdict == "unverified" and _lan:
            st.caption("To remove the doubt, run this once in an Administrator "
                       "terminal and reopen this panel:")
            st.code(reach.firewall_fix_command(), language="bat")

        _local = next((r for r in _routes if r["kind"] == "local"), None)
        if _local:
            st.markdown("**No network at all**")
            st.code(_local["url"], language=None)
            st.caption(_local["detail"])

def _live_map(window: int, auto: bool) -> None:
    """The live network map and its controls.

    Wrapped in st.fragment so toggling a layer reruns *this* block only. A full
    rerun rebuilds every chart on every tab - Streamlit renders all tab bodies,
    not just the visible one - and each rebuild remounts a deck.gl canvas.
    Chrome allows about sixteen live WebGL contexts and each map holds two, so
    remounting on every checkbox is what eventually blacks the map out.
    """
    d = density(window)
    lf = links(window)

    opt = st.columns([1.1, 1, 1, 1, 1, 1])
    basemap = opt[0].selectbox("Basemap", list(maps.BASEMAPS), index=1, key="net_basemap")
    show_roads = opt[1].checkbox("Roads", True, key="net_roads")
    show_flow = opt[2].checkbox("Flow arcs", False, key="net_arcs",
                                help="Directional volume between adjacent cameras")
    show_labels = opt[3].checkbox("Labels", True, key="net_labels")
    show_live = opt[4].checkbox("Live reads", True, key="net_live",
                                help="Reads from the last five minutes, fading with age")
    pitch = opt[5].slider("Tilt", 0, 60, 40, key="net_pitch")

    cams = maps.camera_frame(d)
    layers = []
    if show_roads:
        layers.append(maps.road_layer(maps.road_frame(net(), lf)))
    if show_flow:
        layers.append(maps.flow_layer(lf, net()))
    if show_live:
        layers.append(maps.sighting_layer(recent(400), net()))
    layers += maps.camera_layers(cams, labels=show_labels)

    # Junctions that have a real camera standing on them right now. The point
    # of the network map is where the platform's inputs are, and until now it
    # drew the simulated ones and the live ones identically.
    #
    # Only numeric layer properties here: pydeck compiles a bare string passed
    # to a layer into a data accessor, which is what silently broke the road
    # and text layers once already (README §3). Anything enum-like would need
    # pdk.types.String.
    live_ids = livefeed.live_camera_ids(livefeed.status())
    if live_ids:
        ring = pd.DataFrame([
            {"lon": net().cameras[c]["lon"], "lat": net().cameras[c]["lat"],
             "name": net().name(c)}
            for c in live_ids if c in net().cameras])
        if not ring.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=ring, get_position=["lon", "lat"],
                get_radius=340, stroked=True, filled=False,
                get_line_color=[34, 197, 94], line_width_min_pixels=3,
                pickable=False))

    # Above every other layer: a cloned plate outranks congestion colouring.
    # The phase drives the halo's breathing - deck.gl animates nothing by
    # itself, so the pulse advances once per refresh of this fragment.
    live_clones = clone_mod.live_alerts(net=net())
    layers += maps.clone_alert_layers(live_clones, phase=(time.time() % 6.0) / 6.0)

    # A stable key lets Streamlit reuse the same component instance across
    # reruns instead of tearing down the canvas and building a new one.
    st.pydeck_chart(maps.deck(layers, maps.view(net(), zoom=11.0, pitch=pitch), basemap),
                    height=560, key="live_network_map")
    st.markdown(maps.legend("network"), unsafe_allow_html=True)
    if live_clones:
        st.error("🚨 **" + ", ".join(sorted({c["plate"] for c in live_clones}))
                 + "** — the pulsing red junctions are where the same registration "
                   "was read at two places no single vehicle could have covered.")
    if live_ids:
        st.success("🟢 **Live cameras on " + ", ".join(
            sorted(net().name(c) for c in live_ids))
            + "** — ringed in green on the map. Reads from them land in the same "
              "sightings table as everything else on this page.")
    st.caption("Roads are drawn from the network's own link geometry and coloured by the "
               "worse of their two directions against that road's free-flow speed. Camera "
               "discs size with per-lane load; the halo marks gateways against junctions. "
               "Labels thin out to the busiest sites so they stay readable.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Busiest corridors right now**")
        if lf.empty:
            st.info("No vehicle has been seen at two adjacent cameras in this window yet.")
        else:
            top = lf.head(12)[["name", "direction", "vehicles", "median_kmph", "free_kmph",
                               "congestion_ratio", "level"]]
            st.dataframe(top.rename(columns={"median_kmph": "km/h", "free_kmph": "free-flow",
                                             "congestion_ratio": "x slower"}),
                         hide_index=True, width="stretch", height=430)
    with right:
        st.markdown("**Latest reads**")
        r = recent(40)
        if r.empty:
            st.info("No sightings yet.")
        else:
            r = r.assign(camera=[net().name(c) for c in r["camera_id"]],
                         time=[clock(t) for t in r["ts"]])
            st.dataframe(r[["time", "plate", "camera", "direction", "confidence"]],
                         hide_index=True, width="stretch", height=430)


@st.cache_data(ttl=10, show_spinner=False)
def clone_check(plate: str, since: float | None = None) -> dict:
    """Cached clone verdict. Re-runs on every Streamlit rerun otherwise."""
    from core import clones

    return clones.check(plate, since=since).as_dict()


def render_plate_history(plate: str, since: float | None = None, key: str = "hist",
                         compact: bool = True) -> None:
    """Route and history for one plate, for use right after a successful read.

    An ANPR read on its own is a string. What an operator actually wants the
    moment a plate resolves is the rest of the sentence: where this vehicle has
    been, when, and whether anything about that path is wrong. This renders that
    join compactly enough to sit under a read without becoming a second Track
    tab.
    """
    from anpr import plates as _plates          # tab-local elsewhere; needed here too

    traj = trajectory.reconstruct(plate, since=since)
    if not traj.sightings:
        st.info(f"`{_plates.pretty(plate)}` reads cleanly, but this network has no "
                "record of it in the window - a vehicle it has never seen, or one "
                "whose earlier passes were all refused by the confidence floor.")
        return
    summ = traj.summary

    # A read that resolves to a cloned registration is the one case where the
    # history below is actively misleading - it is two vehicles' paths spliced
    # into one line - so the warning goes above it, not after.
    rep = clone_check(plate, since)
    if rep["verdict"] == "confident":
        st.error(f"**Cloned registration — at least {rep['min_vehicles']} vehicles.** "
                 f"{rep['reason']} The route below splices their paths together.")
    elif rep["verdict"] == "suspected":
        st.warning(f"**Possible clone.** {rep['reason']}")

    m = st.columns(5)
    m[0].metric("Sightings", summ["sightings"])
    m[1].metric("Cameras", summ["cameras_visited"])
    m[2].metric("Distance", f"{summ['distance_km']:.1f} km")
    m[3].metric("Journey", f"{summ['duration_min']:.0f} min")
    m[4].metric("Last seen", ago(summ["last_seen"]))
    st.caption(f"First seen {clock(summ['first_seen'])} at {summ['first_camera_name']} · "
               f"last seen {clock(summ['last_seen'])} at {summ['last_camera_name']} · "
               f"sectors: {', '.join(summ['sectors'])}")

    layers, stops = maps.trajectory_layers(traj, net())
    if layers and not stops.empty:
        view = pdk.ViewState(latitude=float(stops["lat"].mean()),
                             longitude=float(stops["lon"].mean()),
                             zoom=_fit_zoom(stops), pitch=0)
        st.pydeck_chart(maps.deck(
            [maps.road_layer(maps.road_frame(net(), pd.DataFrame()))] + layers,
            view, list(maps.BASEMAPS)[0]), height=340, key=f"{key}_map")
    if traj.legs:
        st.dataframe(pd.DataFrame([{
            "from": l.from_name, "to": l.to_name, "left": clock(l.departed_ts),
            "arrived": clock(l.arrived_ts), "minutes": round(l.minutes, 1),
            "road km": round(l.road_km, 2), "implied km/h": round(l.implied_kmph, 1),
            "heading": l.direction, "flag": "" if l.plausible else "!",
            "note": l.note,
        } for l in traj.legs]), hide_index=True, width="stretch")
    else:
        st.caption("Only one sighting in this window - no route to draw yet.")


def render_enrol_panel(read, key: str = "enrol") -> None:
    """Register the vehicle in this photograph at a junction. -> a route.

    The dead end this exists to remove: the engine reads a real photograph
    correctly, and the platform then has nothing to say, because a car
    photographed on a phone has never passed one of these 35 cameras. The
    photograph carries the registration but not the two facts a camera
    contributes - which junction, and when - so the operator supplies those and
    the sighting becomes real. Do it again at the next junction and there is a
    trajectory to draw.

    Everything that could embarrass this live is handled in core/enrol.py rather
    than here: the confidence floor, the bound on correcting a misread, and the
    timestamp arithmetic that stops the platform accusing its own demo vehicle
    of being a cloned plate. This function is the form and the wording.
    """
    from anpr import plates as _plates          # tab-local elsewhere; needed here too

    n = net()
    plate_now = read.text.upper().replace(" ", "")
    seen = db.sightings_for_plate(plate_now)

    st.divider()
    if seen:
        st.markdown(f"#### Add another pass for `{read.pretty}`")
        st.caption(f"This network already holds {len(seen)} sighting(s) of this "
                   "registration. Enrol it at the next junction along and the route "
                   "extends.")
    else:
        st.markdown(f"#### Register `{read.pretty}` at a junction")
        st.caption("The engine read this plate, but no camera on this network has "
                   "seen the vehicle - so there is no route to draw yet. Enrolling "
                   "records that it was at a junction you name, at a time you name. "
                   "The registration and its confidence stay exactly as the engine "
                   "read them; you are supplying the camera's half of the sighting.")

    cams = sorted(n.cameras.values(), key=lambda c: c["name"])
    labels = {f"{c['name']} — {c['id']} · {c.get('sector') or '?'}": c["id"] for c in cams}

    # Deliberately not st.form. A form batches its widgets and only reruns on
    # submit, which would leave the derived timestamp below describing the
    # *previously* selected junction - the one number in this panel the
    # operator most needs to be live. Double-submission is handled where it
    # belongs instead, in core/enrol.check().
    c1, c2 = st.columns([2, 1])
    picked = c1.selectbox("Junction", list(labels), key=f"{key}_cam",
                          help="Where the photograph was taken. These are the 35 "
                               "real Kolkata junctions in data/cameras.json.")
    camera_id = labels[picked]
    corrected = c2.text_input(
        "Registration", value=plate_now, key=f"{key}_plate",
        help="Fix a misread here if the engine got a character wrong - you can "
             "see the plate, it cannot. Bounded by OCR confusion distance, so "
             "this corrects a misread rather than naming a different vehicle.")

    suggested, why = enrol_mod.suggest_ts(plate_now, camera_id, net=n)
    st.caption(f"**Proposed time** {clock(suggested)} — {why}")
    c3, c4 = st.columns([1, 2])
    adjust = c3.number_input("Adjust (minutes)", value=0, step=1, min_value=-720,
                             max_value=720, key=f"{key}_adjust",
                             help="The proposed time is derived from the road "
                                  "distance to this vehicle's last sighting, so "
                                  "the implied speed is an ordinary city speed. "
                                  "Moving it far earlier makes the leg impossible.")
    operator = c4.text_input("Recorded by", value="control-room", key=f"{key}_op")
    note = st.text_input("Note (optional)", key=f"{key}_note",
                         placeholder="e.g. photographed at the north approach")
    allow = st.checkbox(
        "Write it even if the leg is physically impossible", key=f"{key}_force",
        help="Off by default. On, this deliberately creates the signature of a "
             "cloned plate - useful to demonstrate the clone detector, "
             "misleading otherwise.")
    go = st.button("Register at this junction", type="primary", key=f"{key}_go")

    if not go:
        return

    ts = suggested + adjust * 60.0
    try:
        out = enrol_mod.enrol(read, camera_id, ts=ts, plate=corrected.strip() or None,
                              operator=operator.strip() or None,
                              note=note.strip() or None,
                              allow_implausible=allow, net=n)
    except ValueError as exc:
        # A refusal is the platform declining to fabricate, not a crash. Say
        # which rule was hit, in the operator's language.
        st.warning(f"**Not recorded.** {exc}")
        return

    # The history rendered above was drawn before this write, so redraw it
    # rather than calling st.rerun() - a rerun inside the upload branch
    # re-delivers the uploaded file and re-runs the recogniser.
    with contextlib.suppress(Exception):
        clone_check.clear()

    where = f"{out['camera_name']} ({out['camera_id']})"
    msg = f"**Recorded** `{_plates.pretty(out['plate'])}` at {where}, {clock(out['ts'])}."
    if out["corrected"]:
        msg += f" Corrected from the engine's `{out['engine_read']}`."
    st.success(msg)
    if out["alerts"]:
        for a in out["alerts"]:
            st.error(f"**{a['kind']}** — {a['message']}")
    if out["total_sightings"] < 2:
        st.info("One sighting is a point, not a path. Photograph the same vehicle at "
                "another junction and enrol it there - the route appears on the second "
                "one.")
    else:
        st.markdown(f"**Route for `{_plates.pretty(out['plate'])}` now**")
        render_plate_history(out["plate"], key=f"{key}_after")


def render_live_map(window: int, auto: bool) -> None:
    """Run the map fragment, on a timer when auto-refresh is on.

    run_every reruns only the fragment, so the live feed updates without the
    whole page - and every other map on it - being rebuilt.
    """
    st.fragment(run_every="5s" if auto else None)(_live_map)(window, auto)


# --- live cameras -------------------------------------------------------
def _camera_kpis() -> None:
    """Headline state of the live service. Its own small fragment."""
    state = livefeed.status()
    if state is None:
        return
    js = state.get("junctions", [])
    online = [j for j in js if j.get("online")]
    k = st.columns(4)
    style.kpi(k[0], "Junctions live", f"{len(online)}", "",
              f"of {len(js)} assigned")
    style.kpi(k[1], "Reader", state.get("backend", "?"))
    style.kpi(k[2], "Recording", "on" if state.get("enrol") else "off", "",
              "reads become sightings" if state.get("enrol") else "read-only")
    style.kpi(k[3], "Timestamps", "real" if state.get("realtime") else "spaced", "",
              "clone rule will fire" if state.get("realtime")
              else "by road distance")


def _camera_controls(state: dict) -> None:
    """Switching cameras and recording, and stopping the service.

    Static, like the video below it: a control inside a one-second fragment is
    rebuilt under the cursor between the press and the click landing, which is
    how a button acquires a reputation for not working.
    """
    js = state.get("junctions", [])
    any_on = any(j.get("enabled", True) for j in js)
    # Three equal buttons on their own row, with the explanation underneath
    # rather than squeezed into a fourth column beside them - at that width the
    # caption wrapped to three cramped lines and dragged the row out of square.
    c = st.columns(3)

    if any_on:
        if c[0].button("Turn all cameras off", width="stretch",
                       help="Stops reading and releases every camera. Nothing "
                            "already recorded is affected."):
            livefeed.set_all_cameras(False)
            st.rerun()
    elif js:
        if c[0].button("Turn all cameras on", type="primary", width="stretch"):
            livefeed.set_all_cameras(True)
            st.rerun()

    if state.get("enrol"):
        if c[1].button("Stop recording", width="stretch",
                       help="Keep reading and showing, but write nothing to the "
                            "database. Use this to rehearse without leaving "
                            "sightings behind."):
            livefeed.set_recording(False)
            st.rerun()
    else:
        if c[1].button("Start recording", type="primary", width="stretch",
                       help="Confirmed reads become sightings again."):
            livefeed.set_recording(True)
            st.rerun()

    if c[2].button("Shut down service", width="stretch",
                   help="Stops the live camera process entirely. The phones "
                        "will lose their connection and the GPU is released."):
        livefeed.shutdown()
        time.sleep(1.0)
        st.rerun()

    st.caption("Turning a camera off releases it and idles the GPU. Recording "
               "off keeps the reader running but stores nothing.")


def _tile_status(camera_id: str) -> None:
    """One junction's reading state, refreshed on its own.

    Per tile rather than one row for all of them, so the status sits inside the
    same bordered card as its video instead of floating in a separate row
    underneath. Only the text is in here: the video above must stay outside any
    fragment or its connection is torn down on every tick.
    """
    state = livefeed.status()
    j = next((x for x in (state or {}).get("junctions", [])
              if x["camera_id"] == camera_id), None)
    if j is None:
        return
    if not j.get("enabled", True):
        st.caption(":grey[switched off — not reading, camera released]")
        return
    if j.get("locked") and j.get("pretty"):
        # Locking and recording are different events. The tile used to show
        # only the first, so a read that was refused - or one taken while
        # recording was off - looked identical to one safely in the database,
        # and the operator was left wondering why the table below stayed empty.
        outcome = j.get("last_outcome") or ""
        note = j.get("last_outcome_note") or ""
        if outcome == "recorded":
            st.success(f"**{j['pretty']}**  ·  {j['confidence']:.0%}  ·  recorded")
        elif outcome == "refused":
            st.warning(f"**{j['pretty']}** read, but NOT recorded")
            st.caption(f":orange[{note}]")
        elif outcome == "not_recording":
            st.info(f"**{j['pretty']}** read  ·  recording is off")
        elif outcome == "error":
            st.error(f"**{j['pretty']}** read, but storing it failed")
            st.caption(f":red[{note}]")
        else:
            st.success(f"**{j['pretty']}**  ·  {j['confidence']:.0%}")
    elif j.get("pretty"):
        st.info(f"reading `{j['pretty']}` — {j['votes']} frame(s) agree")
    elif j.get("raw_text"):
        st.caption(f":orange[seeing `{j['raw_text']}` at {j['raw_conf']:.0%} "
                   f"— under the floor]")
    else:
        st.caption(":grey[nothing read yet]")
    if j.get("online"):
        st.caption(f"{j['fps']:.1f} fps · {j['infer_ms']:.0f} ms · "
                   f"{j['locks']} read(s) recorded")
    elif j.get("frames"):
        st.caption(f":orange[silent {j['silent_for']:.0f}s]")
    else:
        st.caption(":grey[waiting for a camera]")


def _camera_tiles(js: list) -> None:
    """One bordered card per junction: name, video, switch, status.

    The card is what makes this read as part of the dashboard rather than a
    row of loose images - every other panel on this page is a bordered
    container. height="stretch" keeps the three square with each other when one
    of them has a longer status than the others.

    The <img> itself is deliberately NOT inside a fragment: it points at
    phonecam's MJPEG stream, so the browser holds one connection per junction
    and frames arrive as they are produced. Re-rendering it on a timer would
    reconnect it every tick, which is what turns live video into a stutter.
    """
    cols = st.columns(min(max(len(js), 1), 3))
    for i, j in enumerate(js):
        cid = j["camera_id"]
        on = j.get("enabled", True)
        with cols[i % len(cols)]:
            with st.container(border=True, height="stretch"):
                st.markdown(
                    f"<div style='font-weight:600;font-size:0.95rem;"
                    f"line-height:1.2'>{j['name']}</div>"
                    f"<div style='font-size:0.68rem;color:#6D7C90;"
                    f"letter-spacing:0.07em;text-transform:uppercase;"
                    f"margin:2px 0 8px'>{cid} · {j.get('sector') or '—'} · "
                    f"{j['mode']}</div>", unsafe_allow_html=True)
                if on:
                    # Emitted only while the camera is on. Off, the stream
                    # would still answer with a placeholder and the browser
                    # would hold a connection open for nothing.
                    st.markdown(
                        f"<img src='{livefeed.stream_url(cid)}' "
                        f"style='width:100%;aspect-ratio:16/9;object-fit:cover;"
                        f"border-radius:8px;background:#000;display:block'>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='width:100%;aspect-ratio:16/9;"
                        "border-radius:8px;background:#0b0e12;"
                        "border:1px dashed #2a323d;display:flex;"
                        "align-items:center;justify-content:center;"
                        "color:#4E5E73;font-size:0.85rem'>camera off</div>",
                        unsafe_allow_html=True)
                st.fragment(run_every="1s")(_tile_status)(cid)
                # Pinned to the bottom, for the same reason as the tour cards
                # and more so here: a junction that has locked a plate shows a
                # tall success box while its neighbours show two lines of grey
                # caption, so without this the three switches sit at three
                # different heights and the row looks broken.
                label = "Turn off" if on else "Turn on"
                with st.container(height="stretch",
                                  vertical_alignment="bottom"):
                    if st.button(label, key=f"cam_sw_{cid}", width="stretch",
                                 type="secondary" if on else "primary"):
                        livefeed.set_camera(cid, not on)
                        st.rerun()


def _who_reached_us() -> None:
    """Which devices have actually reached the live service.

    Built for one question, asked at every demo: a phone will not connect, and
    there is no way to tell whether its packets are arriving. If the phone is
    not listed here they are not, and the cause is the network - a firewall, or
    a hotspot isolating its clients from one another, which is the usual one
    and the reason exactly the phone HOSTING a hotspot works while the others
    do not. If it is listed, the phone reached us and the problem is on it: the
    certificate was not accepted, or the camera permission was refused.
    """
    info = livefeed.connections()
    if info is None:
        return
    seen = info.get("seen") or []
    devices: dict[str, dict] = {}
    for row in seen:
        devices.setdefault(row["ip"], row)
    with st.expander(f"Devices that reached this laptop ({len(devices)})"):
        st.caption("This laptop is on " + ", ".join(info.get("addresses") or ["?"])
                   + ". A phone missing from this list never reached it — that is "
                     "the network, not the reader. A phone-hosted hotspot lets the "
                     "hosting phone through and blocks the rest.")
        if not devices:
            st.info("Nothing has reached this service yet apart from this dashboard.")
            return
        st.dataframe(pd.DataFrame([{
            "address": ip, "last seen": clock(r["ts"]), "asked for": r.get("detail") or "—",
            "device": (r.get("agent") or "")[:48],
        } for ip, r in devices.items()]), hide_index=True, width="stretch")


def _clear_records_panel() -> None:
    """Remove what the live cameras have written, after saying what will go.

    Two steps on purpose. This deletes rows from the platform's own database,
    and it sits on a page whose other buttons are harmless toggles - so it is
    kept behind an expander, away from the camera switches, and asks once.
    That is the same shape core/privacy.retention_purge() settled on for
    irreversible deletion: report first, delete on a second instruction.
    """
    # Untick the confirmation left over from a completed clear. This is the
    # only moment it can be done: Streamlit refuses to let a widget's key be
    # assigned once that widget exists, so doing it in the button handler below
    # - where the checkbox has already been created - raises
    #   StreamlitAPIException: st.session_state.clear_live_confirm cannot be
    #   modified after the widget with key clear_live_confirm is instantiated
    # The handler therefore stages the request under a key no widget owns, and
    # it is applied here on the next run. Same trick, and the same reason, as
    # track_plate()'s "pending_tab".
    if st.session_state.pop("clear_live_reset", False):
        st.session_state["clear_live_confirm"] = False

    preview = livefeed.clear_live_records(confirm=False)
    n = preview["sightings"]
    with st.expander(f"Clear live-camera records ({n})"):
        if not n:
            st.caption(":grey[Nothing recorded from the live cameras yet. "
                       "Camera sightings from the simulator are never touched "
                       "by this.]")
            return
        st.warning(
            f"**{n} sighting(s)** across {len(preview['plates'])} registration(s), "
            f"and **{preview['alerts']} alert(s)** they raised, will be deleted "
            f"from the platform's database."
            + (f" {preview['alerts_kept']} alert(s) will be KEPT, because those "
               f"registrations were also seen by the simulated cameras and the "
               f"history is not the demo's to delete."
               if preview["alerts_kept"] else ""))
        st.caption("Plates: " + ", ".join(preview["plates"][:12])
                   + ("…" if len(preview["plates"]) > 12 else ""))
        st.caption("Only rows a live camera wrote are removed — a simulated "
                   "camera's own captures cannot be caught by this.")
        ok = st.checkbox("Yes, delete these records", key="clear_live_confirm")
        if st.button("Clear now", key="clear_live_go", type="primary",
                     disabled=not ok, width="stretch"):
            done = livefeed.clear_live_records(confirm=True)
            # Staged, not assigned: the checkbox above already exists on this
            # run, so its key is read-only until the next one.
            st.session_state["clear_live_reset"] = True
            # The alarm remembers which alert ids it has already sounded; the
            # ids are gone, so the memory has to go with them or a re-run of
            # the demo stays silent.
            st.session_state.pop("live_alert_sounded", None)
            st.session_state.pop("clone_alarm_sounded", None)
            # The registry panels cache their lookups for 90 s. The plates they
            # were describing have just been deleted, so a cached card would
            # keep answering for a registration the operator has been told is
            # gone. No registry data is deleted - only the memory of it.
            registry_link.clear()
            st.success(f"Cleared {done['sightings']} sighting(s) and "
                       f"{done['alerts']} alert(s).")
            time.sleep(1.0)
            st.rerun()


def render_live_cameras(auto: bool) -> None:
    """Everything the live cameras are doing, and everything they have caused.

    The split matters. The video and status above come from the live service
    and vanish with it. Everything below comes from the database, because a
    live read is a sighting like any other - which is the whole claim this tab
    exists to make visible, and the reason it keeps working after the phones
    are put away.

    The refresh timers here are deliberately NOT tied to the sidebar's
    auto-refresh checkbox. That switch is about how often the analytics and the
    maps re-query; a camera panel that only moves when someone has ticked a box
    elsewhere is simply broken, which is what it was.
    """
    st.subheader("Cameras on the network")
    st.caption("Phones standing at real junctions. A plate read here is written "
               "as a sighting, so the trajectory engine and every alert rule act "
               "on it exactly as they do on a capture from a gantry camera.")

    state = livefeed.status()
    if state is None:
        style.callout(
            "The live camera service is not running, so there is no video and no "
            "junction status. Anything it has already recorded is still below - "
            "a live read is an ordinary sighting, and the platform does not treat "
            "it differently once it is stored.", tone="info")
        st.code(livefeed.start_command(), language="bat")
    else:
        st.fragment(run_every="3s")(_camera_kpis)()
        st.write("")
        _camera_controls(state)
        _camera_tiles(state.get("junctions", []))
        _who_reached_us()

    # Outside the branch above: what the live cameras have already recorded is
    # in the platform's database, so it can be cleared whether or not the live
    # service happens to be running now.
    _clear_records_panel()

    st.divider()
    journeys = livefeed.live_journeys()
    seen = livefeed.live_sightings(limit=40)
    fired = livefeed.alerts_from_live()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Journeys reconstructed from live reads**")
        st.caption("One registration, two junctions, in order — that is all the "
                   "trajectory engine needs. These are built by core/trajectory.py "
                   "from the stored sightings, not by the camera service.")
        if not journeys:
            st.info("No registration has been read at two different junctions yet. "
                    "Carry the board to a second camera and a route appears here.")
        for jr in journeys[:6]:
            names = " → ".join(net().name(c) for c in jr["cameras"])
            with st.container(border=True):
                a, b = st.columns([3, 1])
                a.markdown(f"### `{_plates_pretty(jr['plate'])}`")
                a.caption(f"{names}  ·  {jr['sightings']} sightings  ·  "
                          f"last seen {ago(jr['last_ts'])}")
                # A registration flagged on the registry, moving between two
                # junctions, is the case this whole chain exists for - and the
                # journey card is where it is most legible, because the route
                # is the evidence that it is moving at all.
                _jb = registry_link.badge_html(registry_link.lookup(jr["plate"]))
                if _jb:
                    a.markdown(_jb, unsafe_allow_html=True)
                b.write("")
                if b.button("Track it", key=f"lj_{jr['plate']}", type="primary",
                            help="Open this registration on Track a plate, with "
                                 "its route, clone check and co-travellers"):
                    track_plate(jr["plate"])
                traj = trajectory.reconstruct(
                    jr["plate"], since=trajectory.lookback(24.0))
                if traj.legs:
                    st.dataframe(pd.DataFrame([{
                        "from": l.from_name, "to": l.to_name,
                        "left": clock(l.departed_ts), "arrived": clock(l.arrived_ts),
                        "minutes": round(l.minutes, 1), "road km": round(l.road_km, 2),
                        "implied km/h": round(l.implied_kmph, 1),
                        "heading": l.direction,
                        "flag": "" if l.plausible else "⚠", "note": l.note,
                    } for l in traj.legs]), hide_index=True, width="stretch")

    with c2:
        st.markdown("**Alerts these reads raised**")
        st.caption("Raised by core/alerts.py on ingest — the same rules that watch "
                   "every simulated capture.")
        if not fired:
            st.caption(":grey[none yet]")
        for a in fired[:8]:
            colour = SEVERITY_COLOUR.get(a["severity"], "#888")
            st.markdown(
                f"<div style='border-left:3px solid {colour};padding-left:8px;"
                f"margin-bottom:7px'><span style='color:{colour};font-weight:700;"
                f"font-size:0.78rem'>{a['severity'].upper()} · {a['kind']}</span>"
                f"<br><span style='font-size:0.82rem'>{a['message']}</span>"
                f"<br><span style='font-size:0.7rem;color:#93a1b3'>"
                f"{clock(a['ts'])}</span></div>", unsafe_allow_html=True)

        st.markdown("**Recorded reads**")
        if not seen:
            st.caption(":grey[no live reads recorded yet]")
        else:
            # Each live read checked against the registry. "not registered" is
            # the expected answer for a board someone has written a plate on,
            # and saying so plainly is better than an empty cell: it is what a
            # fabricated plate looks like, and it is also what a misread looks
            # like, which is exactly the ambiguity an operator should carry.
            _lr = registry_link.lookup_many(tuple(s["plate"] for s in seen))
            REG_LABEL = {"FOUND": "", "NOT_FOUND": "not registered",
                         "UNKNOWN": "registry unreachable"}

            def _reg_cell(plate: str) -> str:
                r = _lr.get(plate)
                if not r:
                    return "—"
                if r.get("registry_status") != "FOUND":
                    return REG_LABEL.get(r.get("registry_status"), "—")
                v = r.get("vehicle", {})
                lab = " ".join(x for x in (v.get("make"), v.get("model")) if x)
                flag = registry_link.FLAGGED.get(r.get("status"))
                return f"⚠ {flag[1]} · {lab}" if flag else lab or "registered"

            st.dataframe(pd.DataFrame([{
                "time": clock(s["ts"]), "plate": s["plate"],
                "junction": s.get("camera_name") or net().name(s["camera_id"]),
                "conf": s["confidence"],
                "registry": _reg_cell(s["plate"]),
            } for s in seen]), hide_index=True, width="stretch", height=260)


def _plates_pretty(plate: str) -> str:
    from anpr.plates import pretty
    return pretty(plate)


# --- live clone response ------------------------------------------------
MIME_BY_SUFFIX = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
                  ".m4a": "audio/mp4", ".flac": "audio/flac"}


@st.cache_data(show_spinner=False)
def _synth_alarm_wav(rate: int = 22050) -> bytes:
    """A two-tone alarm, synthesised - the fallback when no sound file is present.

    Generated so a checkout missing config.CLONE_ALARM_FILE is still audible,
    with nothing to fetch at runtime. Each burst is enveloped at both ends: a
    raw sine switched on at full amplitude clicks, and a click is what makes an
    alert sound broken rather than urgent.
    """
    import io as _io
    import math
    import wave

    frames = bytearray()
    for _ in range(3):
        for freq, secs in ((880.0, 0.16), (620.0, 0.16)):
            n = int(rate * secs)
            for i in range(n):
                # 8 ms raised-cosine fade at each end
                edge = max(1, int(rate * 0.008))
                gain = min(1.0, i / edge, (n - i) / edge)
                v = int(15000 * gain * math.sin(2 * math.pi * freq * i / rate))
                frames += int(v).to_bytes(2, "little", signed=True)
        frames += b"\x00\x00" * int(rate * 0.06)          # gap between bursts

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _clip_seconds(raw: bytes, suffix: str, default: float = 8.0) -> float:
    """Roughly how long a clip runs, so the player can be left on the page for it.

    Exact for WAV. For MP3 it reads the bitrate out of the first frame header,
    which is right for the constant-bitrate files an alarm is normally saved as
    and close enough otherwise - the number only decides how long the element
    stays mounted, so erring long costs nothing.
    """
    with contextlib.suppress(Exception):
        if suffix == ".wav":
            import io as _io
            import wave
            with wave.open(_io.BytesIO(raw), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        if suffix == ".mp3":
            at = 0
            if raw[:3] == b"ID3":
                # ID3v2 sizes are syncsafe: seven bits per byte, big-endian.
                sz = raw[6:10]
                at = 10 + ((sz[0] & 0x7F) << 21 | (sz[1] & 0x7F) << 14
                           | (sz[2] & 0x7F) << 7 | (sz[3] & 0x7F))
            h = raw[at:at + 4]
            if len(h) == 4 and h[0] == 0xFF and h[1] & 0xE0 == 0xE0:
                kbps = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                        160, 192, 224, 256, 320, 0)[(h[2] >> 4) & 0xF]
                if kbps:
                    return (len(raw) - at) * 8 / (kbps * 1000)
    return default


@st.cache_data(show_spinner=False)
def alarm_audio() -> tuple[bytes, str, float]:
    """The alarm's bytes, MIME type and length in seconds.

    Prefers the file at config.CLONE_ALARM_FILE and falls back to the
    synthesised burst when it is missing or unreadable. Cached, so the file is
    read once per process rather than on every three-second rerun.
    """
    path = config.CLONE_ALARM_FILE
    suffix = path.suffix.lower()
    with contextlib.suppress(OSError):
        raw = path.read_bytes()
        if raw:
            return raw, MIME_BY_SUFFIX.get(suffix, "audio/mpeg"), _clip_seconds(raw, suffix)
    wav = _synth_alarm_wav()
    return wav, "audio/wav", _clip_seconds(wav, ".wav")


# Tab labels as constants. st.tabs() identifies a tab by its label string, and
# goto_tab() writes one of these into session state, so a typo would silently
# navigate nowhere rather than raising. Naming them once removes that class of
# bug and makes it obvious what the legal destinations are.
TAB_OVERVIEW = "Overview"
TAB_LIVE = "Live network"
TAB_CAMERAS = "Live cameras"
TAB_TRACK = "Track a plate"
TAB_FLOW = "City analytics"
TAB_ALERTS = "Alerts"
TAB_ANPR = "ANPR engine"


def track_plate(plate: str) -> None:
    """Open Track a plate on this registration, already searched.

    An alert names a vehicle, and the next thing anyone wants is that
    vehicle's movement. Retyping a registration off a red banner - under
    pressure, from a plate the reader itself may have read one character wrong
    - is both the slowest way to get there and the one that mistypes.

    Nothing here runs a search. Track a plate searches whatever is in "picked"
    as soon as the script reaches it, so setting the plate and the tab is
    enough: by the time the rerun paints, the trajectory, the clone check and
    the co-travellers are already on screen.

    The tab is requested through "pending_tab" rather than written straight to
    "main_tab", because Streamlit refuses to let a widget's key be assigned
    after that widget has been instantiated:

        StreamlitAPIException: st.session_state.main_tab cannot be modified
        after the widget with key main_tab is instantiated

    and the Alerts queue - one of the two places that calls this - is rendered
    inside a tab, which is necessarily after st.tabs() has run. Staging the
    request and applying it at the top of the next run, before the tabs are
    built, is what makes one code path work from both callers.

    scope="app" is load-bearing too. The clone banner lives in a fragment, and
    a bare st.rerun() inside a fragment reruns only that fragment - the tab
    would not move and the button would look broken.
    """
    st.session_state["picked"] = plate.upper().replace(" ", "")
    st.session_state["pending_tab"] = TAB_TRACK
    st.rerun(scope="app")


def goto_tab(label: str) -> None:
    """Move the operator to a tab. The general form of track_plate's second half.

    Same staging through "pending_tab" and the same scope="app", for the same
    two reasons documented above: a key cannot be assigned after its widget
    exists, and a bare rerun inside a fragment moves nothing.
    """
    st.session_state["pending_tab"] = label
    st.rerun(scope="app")


@st.cache_data(ttl=3600)
def _real_photo_accuracy() -> tuple[float, float, float] | None:
    """Mean / min / max exact-match on real photographs, across the 8-seed split.

    Read from models/backend_comparison_real.json rather than typed in, so the
    headline number on the front page is the same artefact anpr.compare_backends
    wrote and README quotes. A number nobody can trace to a file is a number a
    judge is right to distrust, and one typed into a dashboard drifts from the
    benchmark the first time the benchmark is re-run.
    """
    try:
        import json
        with open(config.MODEL_DIR / "backend_comparison_real.json", encoding="utf-8") as fh:
            per_seed = json.load(fh)
        vals = [s["anpr_ocr"] for s in per_seed.values() if "anpr_ocr" in s]
        return (sum(vals) / len(vals), min(vals), max(vals)) if vals else None
    except (OSError, ValueError, KeyError):
        # A missing or half-written benchmark file must not take the front page
        # down with it - the panel simply drops that one figure.
        return None


def render_overview(window: int) -> None:
    """The first screen. What this is, that it is running, and where to start.

    Every other tab answers a question an operator already has. This one exists
    because the first thirty seconds belong to someone who has none yet - a
    judge, a visitor, anyone who has just been handed the link - and dropping
    them into a live map with six metrics above it asks them to work out what
    they are looking at before they can be impressed by it.

    Live figures and measured figures are deliberately mixed on one row, each
    labelled with where it comes from. The two are different kinds of claim: one
    is this database right now, the other is a benchmark someone can re-run, and
    a front page that blurs them is the kind that gets picked apart.
    """
    st.markdown(
        "<div style='font-size:0.63rem;font-weight:600;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#4E5E73'>SIH 2026 &nbsp;·&nbsp; Team Prometheus</div>"
        "<div style='font-size:2.1rem;font-weight:600;letter-spacing:-0.02em;"
        "color:#EDF2F8;line-height:1.15;margin:0.1rem 0 0.25rem 0'>GodsEye</div>"
        "<div style='font-size:1.02rem;color:#93A1B4;max-width:62rem;line-height:1.55'>"
        "City-wide automatic number-plate recognition for Kolkata &mdash; a plate "
        "reader built for degraded real-world captures, on a network of 35 real "
        "junctions, with the trajectory, clone-detection and alerting layers that "
        "turn a stream of reads into something a control room can act on.</div>",
        unsafe_allow_html=True)
    st.write("")

    s = summary(window)
    acc = _real_photo_accuracy()
    st_ = db.stats()

    k = st.columns(4)
    if acc:
        style.kpi(k[0], "Real-photo accuracy", f"{acc[0]:.1%}", "",
                  f"mean of 8 vehicle splits · {acc[1]:.0%}–{acc[2]:.0%}")
    else:
        style.kpi(k[0], "Real-photo accuracy", "—", "", "benchmark file unavailable")
    style.kpi(k[1], "Read latency", "108", " ms",
              "whole photograph, GPU, warm · PERFORMANCE.md")
    style.kpi(k[2], "Camera network", f"{s['total_cameras']}", "",
              f"{s['active_cameras']} reporting now")
    style.kpi(k[3], "Sightings held", f"{st_['sightings']:,}", "",
              f"{st_['unique_plates']:,} distinct registrations")

    st.write("")
    style.section("Start here", "A three-minute tour")

    # Four cards on one row, so the titles are kept short enough to sit on a
    # single line at a quarter of the width and the captions to a similar
    # length. Uneven wrapping is what made this row look ragged when it grew
    # from three: a two-line title pushes its button below its neighbours', and
    # a row of buttons at four different heights reads as a broken layout.
    # height="stretch" then squares off whatever difference is left.
    tour = [
        ("1 &nbsp; Watch the network",
         "35 junctions, roads coloured by live congestion, and reads landing "
         "as they happen.", "Open live network", "ov_live", TAB_LIVE),
        ("2 &nbsp; Add a real camera",
         "Phones standing at real junctions. What they read becomes a sighting "
         "on this network.", "Open live cameras", "ov_cams", TAB_CAMERAS),
        ("3 &nbsp; Follow a vehicle",
         "Reconstruct a route across the camera graph. A mistyped plate still "
         "finds the vehicle.", "Open track a plate", "ov_track", TAB_TRACK),
        ("4 &nbsp; Read a photograph",
         "Upload a photo of an Indian plate, or capture one from the physical "
         "camera model.", "Open ANPR engine", "ov_anpr", TAB_ANPR),
    ]
    for col, (title, caption, label, key, dest) in zip(st.columns(4), tour):
        with col:
            with st.container(border=True, height="stretch"):
                st.markdown(f"**{title}**")
                st.caption(caption)
                # The button is pinned to the bottom of the card rather than
                # left to follow the caption. height="stretch" squares the
                # cards off, but a caption that wraps to one fewer line still
                # lifts its button above the others, and a row of buttons at
                # different heights is the thing the eye actually notices.
                with st.container(height="stretch",
                                  vertical_alignment="bottom"):
                    if st.button(label, key=key, width="stretch"):
                        goto_tab(dest)

    st.write("")
    style.callout(
        "The traffic on this network is simulated, and the accuracy figure above "
        "is not &mdash; it is measured on photographs of real vehicles. The two are "
        "kept apart deliberately throughout: every panel says which of the two it "
        "is standing on, and what it cannot support.",
        tone="info")


def _clone_watch() -> None:
    """Alarm and banner for a cloned plate, above the tabs so it cannot be missed.

    Its own fragment on a short timer, deliberately outside the tab bodies: a
    clone can be found while the operator is reading analytics, and an alert
    that only appears on the tab you are not looking at is not an alert.

    The sound fires once per alert id - a siren that never stops is worse than
    none, because the operator turns it off and then misses the next one. But
    this fragment re-runs every three seconds, and an element the run does not
    draw is removed from the page, which would cut a longer alarm off partway.
    So the player is kept mounted until the clip has had time to finish, and
    only then dropped. Re-drawing it in between does not restart playback:
    Streamlit addresses media by a hash of its bytes, so the identical clip
    keeps the identical element.
    """
    live = clone_mod.live_alerts(net=net())
    if not live:
        return

    sounded = st.session_state.setdefault("clone_alarm_sounded", set())
    # The BANNER shows for every live clone - a cloned plate is worth seeing
    # wherever it came from. The SOUND is reserved for the ones this demo
    # caused: a judge injection or a live camera read. The simulator produces
    # clones on its own, and an alarm going off for ambient traffic is one an
    # operator learns to ignore.
    audible = livefeed.audible_alert_ids()
    fresh = [a for a in live
             if a["alert_id"] not in sounded and a["alert_id"] in audible]

    worst = max(live, key=lambda a: a.get("implied_kmph") or 0.0)
    st.error(
        f"### 🚨 Cloned plate detected — `{worst['plate']}`\n"
        f"Seen at **{worst['from_name']}** and **{worst['to_name']}**, "
        f"{maps._num(worst.get('road_km'), '{:.1f}')} km apart, "
        f"{maps._num(worst.get('minutes'), '{:.1f}')} minutes apart — an implied "
        f"**{maps._num(worst.get('implied_kmph'), '{:.0f}')} km/h**. "
        f"At least {worst.get('min_vehicles', 2)} vehicles are carrying this "
        f"registration. Both junctions are lit on the live map."
        + (f"\n\n_{len(live) - 1} further clone alert(s) open._" if len(live) > 1 else ""))

    raw, mime, secs = alarm_audio()
    if fresh and st.session_state.get("clone_alarm_on", config.CLONE_ALARM_ENABLED):
        st.session_state["clone_alarm_until"] = time.time() + secs
        sounded.update(a["alert_id"] for a in fresh)
    if time.time() < st.session_state.get("clone_alarm_until", 0.0):
        st.audio(raw, format=mime, autoplay=True)

    # Track first, acknowledge second, and in that order on the row: the
    # question an operator has while a clone banner is up is "where has this
    # thing been", not "how do I make this go away". Acknowledging is the
    # destructive one of the two and does not want to be the nearest button.
    c1, c2, c3 = st.columns([1.4, 1, 4])
    if c1.button(f"Track {worst['plate']}", key="clone_track", type="primary"):
        track_plate(worst["plate"])
    if c2.button("Acknowledge", key="clone_ack"):
        for a in live:
            db.ack_alert(a["alert_id"])
        st.rerun(scope="app")
    c3.caption("Track opens this registration on the Track a plate tab with its "
               "route, clone check and co-travellers already loaded. Acknowledging "
               "clears the banner and the glow; the alert stays in the queue on "
               "the Alerts tab.")

    # Every other live clone gets its own button. With several open at once the
    # banner names only the worst, and the rest were previously unreachable
    # without retyping a registration read off a paragraph of text.
    others = [a for a in live if a["plate"] != worst["plate"]]
    if others:
        cols = st.columns(min(len(others), 4))
        for i, a in enumerate(others[:4]):
            if cols[i].button(f"Track {a['plate']}", key=f"clone_track_{a['alert_id']}"):
                track_plate(a["plate"])


def _live_alert_watch() -> None:
    """Sound and banner when a LIVE CAMERA's read raises any alert.

    Deliberately the same audio as the judge console's clone alarm - one sound
    means "the platform has found something", and teaching an operator a second
    siren for the same class of event is how both get ignored.

    Three details earn their place:

      Clones are skipped. _clone_watch above already owns them, and a live
      camera reading a cloned plate would otherwise fire two alarms at once.

      The first run is silent. The queue almost always holds older alerts on
      these plates, and blasting the siren because a page was opened is the
      fastest way to get the speakers turned off. The ids present at load are
      recorded as already heard, so only genuinely NEW ones sound.

      The player stays mounted for the clip's length. This fragment re-runs
      every few seconds and an element a run does not draw is removed, which
      would cut the alarm off partway.
    """
    # Only alerts something in THIS demo actually caused: a live camera read,
    # or a judge injection. The simulator raises alerts continuously, and
    # sounding for those made the alarm meaningless. Matching on the plate was
    # not enough - a registration the simulator also drives would sound for the
    # simulator's own alerts about it - so this matches on alert id.
    audible = livefeed.audible_alert_ids()
    # Clones stay with _clone_watch above, which owns them - otherwise a live
    # camera reading a cloned plate would sound two alarms at once.
    fired = [a for a in (livefeed.alerts_from_live(hours=2.0, limit=40) or [])
             if a["id"] in audible and a.get("kind") != "clone"]
    heard = st.session_state.get("live_alert_sounded")
    if heard is None:
        # First run on this session: adopt the backlog without sounding it.
        st.session_state["live_alert_sounded"] = {a["id"] for a in fired}
        return
    fresh = [a for a in fired if a["id"] not in heard and not a["acked"]]
    open_now = [a for a in fired if not a["acked"]]
    if not open_now:
        return

    worst = min(open_now, key=lambda a: ["critical", "high", "medium", "low"]
                .index(a["severity"]) if a["severity"] in
                ("critical", "high", "medium", "low") else 9)
    colour = SEVERITY_COLOUR.get(worst["severity"], "#888")
    st.markdown(
        f"<div style='border:1px solid {colour};border-left:5px solid {colour};"
        f"background:#1a1113;border-radius:10px;padding:10px 13px;margin-bottom:8px'>"
        f"<span style='color:{colour};font-weight:700;font-size:0.8rem'>"
        f"📷 LIVE CAMERA · {worst['severity'].upper()} · {worst['kind']}</span><br>"
        f"<span style='font-size:0.92rem'>{worst['message']}</span>"
        + (f"<br><span style='font-size:0.72rem;color:#93a1b3'>"
           f"and {len(open_now) - 1} more open on live-camera reads</span>"
           if len(open_now) > 1 else "")
        + "</div>", unsafe_allow_html=True)

    raw, mime, secs = alarm_audio()
    if fresh and st.session_state.get("clone_alarm_on", config.CLONE_ALARM_ENABLED):
        st.session_state["live_alarm_until"] = time.time() + secs
        heard.update(a["id"] for a in fresh)
    if time.time() < st.session_state.get("live_alarm_until", 0.0):
        st.audio(raw, format=mime, autoplay=True)

    if worst["plate"] and st.button(f"Track {worst['plate']}", key="live_alert_track",
                                    type="primary"):
        track_plate(worst["plate"])


st.fragment(run_every="3s")(_clone_watch)()
st.fragment(run_every="3s")(_live_alert_watch)()

# Apply a tab requested on the previous run, while the tabs widget still does
# not exist. This is the only moment it can be done: once st.tabs() below has
# been called, assigning to its key raises, and before this line there is
# nothing to assign to. pop() rather than get() so a request is honoured once
# and the operator can then move tabs freely.
_requested_tab = st.session_state.pop("pending_tab", None)
if _requested_tab:
    st.session_state["main_tab"] = _requested_tab

# key + on_change="rerun" together are what make the active tab readable and
# WRITABLE through session state - with the default on_change="ignore" the key
# is inert and st.session_state["main_tab"] cannot move the operator anywhere.
# The same switch turns on lazy tab execution, so only the open tab's body
# runs. That is a straight win here: this script renders every tab on every
# rerun otherwise, which is how an expensive panel in a tab nobody is looking
# at ends up costing a second and a half on an unrelated click.
(tab_overview, tab_live, tab_cameras, tab_track, tab_flow, tab_alerts,
 tab_anpr) = st.tabs(
    [TAB_OVERVIEW, TAB_LIVE, TAB_CAMERAS, TAB_TRACK, TAB_FLOW, TAB_ALERTS,
     TAB_ANPR], key="main_tab", on_change="rerun")


# --- overview -----------------------------------------------------------
# First in the list, so it is what an unprimed visitor lands on. Lazy tab
# execution means the live map and the analytics queries below do not run at
# all until someone opens those tabs, so leading with this costs nothing - it
# is strictly cheaper on first paint than landing on the map was.
with tab_overview:
    render_overview(window)


# --- live network -------------------------------------------------------
with tab_live:
    s = summary(window)
    # Values and units split, and the two ratios given their denominator as a
    # note rather than inside the number: "33/35" reads as a fraction to be
    # divided, "33" over "of 35 cameras" reads as a count with its total. Same
    # data, one query, no extra work - summary() was already called above.
    k = st.columns(6)
    style.kpi(k[0], "Vehicles / min", f"{s['vehicles_per_min']}")
    style.kpi(k[1], "Unique plates", f"{s['unique_plates']:,}")
    style.kpi(k[2], "Median speed", f"{s['median_network_kmph']:.0f}", " km/h")
    style.kpi(k[3], "Congested links", f"{s['congested_links']}", "",
              f"of {s['monitored_links']} monitored")
    style.kpi(k[4], "Cameras reporting", f"{s['active_cameras']}", "",
              f"of {s['total_cameras']} on the network")
    style.kpi(k[5], "Mean read confidence", f"{s['mean_ocr_confidence']:.0%}")

    if s["worst_link"]:
        w = s["worst_link"]
        style.callout(
            f"<b style='color:#C8D4E2'>Worst corridor — {w['link']}</b> &nbsp;·&nbsp; "
            f"{w['median_kmph']} km/h against a {w['free_kmph']:.0f} km/h free-flow "
            f"limit ({w['level']}), costing {w['delay_min_total']:.0f} vehicle-minutes "
            f"in the last {window} min.",
            tone="warn" if w["level"] in ("heavy", "severe") else "info")

    render_live_map(window, auto)

    st.subheader("Busiest cameras")
    # density() is cached with a short TTL, so asking again here costs nothing
    # and keeps this table independent of the map fragment above it.
    top = density(window).head(12)[["name", "sector", "road", "count", "per_min", "unique_plates",
                      "mean_confidence", "load"]]
    st.dataframe(top.style.format({"per_min": "{:.2f}", "mean_confidence": "{:.0%}",
                                   "load": "{:.2f}"}),
                 hide_index=True, width="stretch")


# --- live cameras -------------------------------------------------------
with tab_cameras:
    render_live_cameras(auto)


# --- track a plate ------------------------------------------------------
with tab_track:
    st.subheader("Reconstruct a vehicle's movement")
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("Plate number", value=st.session_state.get("picked", ""),
                          placeholder="KA 05 MJ 1234 — partial or misread is fine")
    hours = c2.number_input("Look back (hours)", 1.0, 168.0, 12.0, step=1.0)
    c3.write("")
    if c3.button("Pick a busy vehicle", width="stretch"):
        busy = db.rows("SELECT plate FROM sightings GROUP BY plate ORDER BY COUNT(*) DESC LIMIT 8")
        if busy:
            st.session_state["picked"] = busy[np.random.randint(0, len(busy))]["plate"]
            st.rerun()

    if query:
        # Measured from the newest sighting, not from wall clock. A seeded
        # database stops advancing when seeding stops, so a wall-clock window
        # sits entirely after the last sighting by the next morning and every
        # lookup returns nothing - on this database at 14.5 h old, the default
        # 12-hour window held 0 of 398 plates. It is also what keeps a manually
        # enrolled vehicle findable, since core/enrol stamps enrolments at that
        # same network clock.
        since = trajectory.lookback(hours)
        matches = trajectory.search(query, limit=8, since=since)
        st.caption(f"Searching sightings back to {clock(since)} — "
                   f"{hours:.0f} h before the network's newest read "
                   f"({clock(db.latest_ts())}).")
        if not matches:
            st.warning("No plate close to that was seen in the window.")
        else:
            CLONE_BADGE = {"confident": "  ·  🔴 CLONED — {n} vehicles",
                           "suspected": "  ·  🟠 POSSIBLE CLONE"}
            labels = [
                f"{c['plate']}  ·  {c['sightings']} sightings  ·  "
                + ("exact match" if c["exact"] else f"near match (distance {c['distance']})")
                + ("  ·  ⚠ WATCHLISTED" if c["watched"] else "")
                + CLONE_BADGE.get(c["clone_verdict"], "").format(n=c["min_vehicles"])
                for c in matches
            ]
            # Surfaced before a candidate is chosen: whatever the operator does
            # next with this plate assumes it belongs to one vehicle.
            flagged = [c for c in matches if c["clone_verdict"] != "none"]
            if flagged:
                worst = "confident" if any(c["clone_verdict"] == "confident"
                                           for c in flagged) else "suspected"
                banner = st.error if worst == "confident" else st.warning
                banner("Duplicate registration detected in these results — "
                       + "; ".join(f"**{c['plate']}** ({c['clone_verdict']})"
                                   for c in flagged))
                for c in flagged:
                    st.caption(f"{c['plate']}: {c['clone_reason']}")
            choice = st.radio("Candidates", range(len(matches)),
                              format_func=lambda i: labels[i], horizontal=False)
            plate = matches[choice]["plate"]
            traj = trajectory.reconstruct(plate, since=since)
            summ = traj.summary

            m = st.columns(6)
            m[0].metric("Sightings", summ["sightings"])
            m[1].metric("Cameras", summ["cameras_visited"])
            m[2].metric("Distance", f"{summ['distance_km']:.1f} km")
            m[3].metric("Journey time", f"{summ['duration_min']:.0f} min")
            m[4].metric("Avg speed", f"{summ['avg_kmph']:.0f} km/h")
            m[5].metric("Read confidence", f"{summ['mean_ocr_confidence']:.0%}")

            st.caption(f"First seen {clock(summ['first_seen'])} at {summ['first_camera_name']} · "
                       f"last seen {clock(summ['last_seen'])} at {summ['last_camera_name']} "
                       f"({ago(summ['last_seen'])}) · sectors: {', '.join(summ['sectors'])}")

            report = clone_check(plate, since)
            if report["verdict"] == "confident":
                st.error(f"**Cloned registration.** {report['reason']}")
            elif report["verdict"] == "suspected":
                st.warning(f"**Possible clone.** {report['reason']}")
            elif summ["implausible_legs"]:
                # Impossible legs exist but every one is better explained than by
                # a clone. Saying so is the point: at this engine's accuracy a
                # misread merges two vehicles onto one plate and looks identical.
                st.info(f"{summ['implausible_legs']} leg(s) look physically impossible, "
                        f"but this is not read as a clone — {report['reason']}")
            if report["dismissed"]:
                with st.expander(f"Conflicts ruled out ({len(report['dismissed'])})"):
                    st.dataframe(pd.DataFrame([{
                        "from": d["from_name"], "to": d["to_name"],
                        "implied km/h": round(d["implied_kmph"]),
                        "road km": round(d["road_km"], 1),
                        "ruled out as": d["explained_by_misread"],
                    } for d in report["dismissed"]]), hide_index=True, width="stretch")

            # Where this vehicle's evidence came from. An operator acting on a
            # route deserves to know which of its points a camera saw and which
            # a person attested from a photograph - the platform keeps that in
            # sightings.source precisely so it can be shown rather than assumed.
            live_rows = [s for s in traj.sightings
                         if s.get("source") in config.ENROL_SOURCES]
            if live_rows:
                st.info(f"**{len(live_rows)} of {len(traj.sightings)} sightings came "
                        f"from a live camera** — read by the engine, with the junction "
                        f"and time attested by an operator. They are ordinary "
                        f"sightings: the route, the legs and the clone check below "
                        f"treat them exactly like a gantry capture.")

            # Who the vehicle is, under where it went. The operator has just
            # reconstructed a route; the next question has always been whether
            # anything is outstanding against the registration, and until now
            # that meant leaving this application. Everything below degrades to
            # nothing if registry/ is absent - see dashboard/registry_link.py.
            with st.expander("Vehicle registry — who this registration belongs to",
                             expanded=bool(registry_link.findings(plate, hours))):
                registry_link.render_card(plate, hours=hours, key="track")

            tcol = st.columns([1.2, 1, 3])
            tmap = tcol[0].selectbox("Basemap", list(maps.BASEMAPS), index=1, key="traj_basemap")
            tpitch = tcol[1].slider("Tilt", 0, 60, 25, key="traj_pitch")
            layers, stops = maps.trajectory_layers(traj, net())
            if layers and not stops.empty:
                view = pdk.ViewState(
                    latitude=float(stops["lat"].mean()), longitude=float(stops["lon"].mean()),
                    zoom=_fit_zoom(stops), pitch=tpitch)
                st.pydeck_chart(maps.deck(
                    [maps.road_layer(maps.road_frame(net(), pd.DataFrame()))] + layers,
                    view, tmap), height=520, key="trajectory_map")
                st.markdown(maps.legend("trajectory"), unsafe_allow_html=True)
                st.caption("Numbered stops in order of sighting; arrows show the direction of "
                           "travel along each leg. The line follows the road the vehicle must "
                           "have taken, not a straight hop between cameras.")
            else:
                st.info("Only one sighting in this window — nothing to draw yet.")

            st.markdown("**Chronological path**")
            legs = pd.DataFrame([{
                "from": l.from_name, "to": l.to_name, "left": clock(l.departed_ts),
                "arrived": clock(l.arrived_ts), "minutes": round(l.minutes, 1),
                "road km": round(l.road_km, 2), "implied km/h": round(l.implied_kmph, 1),
                "free-flow km/h": l.free_flow_kmph, "heading": l.direction,
                "flag": "" if l.plausible else "⚠", "note": l.note,
            } for l in traj.legs])
            if legs.empty:
                st.info("Only one sighting in this window — no legs to reconstruct yet.")
            else:
                st.dataframe(legs, hide_index=True, width="stretch")

            with st.expander("Raw sightings"):
                st.dataframe(pd.DataFrame([{
                    "time": clock(s["ts"]), "camera": s["camera_name"], "sector": s["sector"],
                    "direction": s["direction"], "speed": s["speed_kmph"],
                    "confidence": s["confidence"], "capture": s["condition"],
                    "engine": s["ocr_variant"],
                    "attested by": ("live camera"
                                    if s.get("source") in config.ENROL_SOURCES
                                    else "camera"),
                    "true plate (sim)": s["true_plate"],
                } for s in traj.sightings]), hide_index=True, width="stretch")

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Watchlist**")
                watched = db.is_watched(plate)
                if watched:
                    st.warning(f"On the watchlist: {watched['reason']} ({watched['severity']})")
                    if st.button("Remove from watchlist"):
                        db.remove_watch(plate)
                        st.rerun()
                else:
                    reason = st.text_input("Reason", "flagged from trajectory review",
                                           key="watch_reason")
                    sev = st.selectbox("Severity", ["critical", "high", "medium", "low"], 1)
                    if st.button("Add to watchlist"):
                        db.add_watch(plate, reason, sev)
                        st.success(f"{plate} added — every future sighting raises an alert.")
                        st.rerun()
            with cc2:
                st.markdown("**Vehicles travelling with it**")
                mates = trajectory.co_travellers(plate)
                if mates:
                    st.dataframe(pd.DataFrame(mates)[["plate", "shared", "camera_count"]],
                                 hide_index=True, width="stretch")
                else:
                    st.caption("No vehicle repeatedly shares this plate's cameras.")

            render_forecast(plate)


# --- city analytics -----------------------------------------------------
with tab_flow:
    st.subheader(f"City movement — last {window} minutes")
    hm = heat(window)
    hopt = st.columns([1.2, 1, 1, 2])
    hmap = hopt[0].selectbox("Basemap", list(maps.BASEMAPS), index=1, key="heat_basemap")
    radius = hopt[1].slider("Blur", 20, 90, 46, key="heat_radius")
    intensity = hopt[2].slider("Intensity", 0.5, 3.0, 1.0, 0.1, key="heat_intensity")
    show_net = hopt[3].checkbox("Show the road network underneath", True, key="heat_net")
    c1, c2 = st.columns([3, 2])
    with c1:
        if hm.empty:
            st.info("Not enough movement in this window.")
        else:
            layers = []
            if show_net:
                layers.append(maps.road_layer(maps.road_frame(net(), pd.DataFrame())))
            layers.append(pdk.Layer(
                "HeatmapLayer", data=hm, get_position=["lon", "lat"], get_weight="weight",
                radius_pixels=radius, intensity=intensity, opacity=0.8,
                aggregation=pdk.types.String("SUM")))
            st.pydeck_chart(maps.deck(layers, maps.view(net(), zoom=10.8, pitch=0), hmap),
                            height=520, key="heatmap_map")
            st.caption("Heat follows the roads: every traversed link is sampled along its own "
                       "geometry and weighted by volume × congestion, so a busy corridor glows "
                       "along its length instead of pooling on the junctions at each end.")
    with c2:
        st.markdown("**Bottlenecks by delay caused**")
        bn = analytics.bottlenecks(window, top=10)
        if bn.empty:
            st.info("No link has enough traffic yet.")
        else:
            st.dataframe(bn[["name", "vehicles", "median_kmph", "free_kmph", "level",
                             "delay_min_per_veh", "delay_min_total"]]
                         .rename(columns={"delay_min_per_veh": "delay/veh (min)",
                                          "delay_min_total": "total delay (min)"}),
                         hide_index=True, width="stretch", height=380)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Flow trend**")
        tr = trend(3.0, 5.0)
        if tr.empty:
            st.info("Need a longer history — seed more hours.")
        else:
            chart = alt.Chart(tr).transform_fold(
                ["sightings", "unique_plates"], as_=["series", "value"]).mark_line().encode(
                x=alt.X("bucket:T", title=None),
                y=alt.Y("value:Q", title="per 5 min"),
                color=alt.Color("series:N", title=None,
                                scale=alt.Scale(range=["#4c8dff", "#f5a524"])))
            st.altair_chart(chart, width="stretch")
    with c2:
        st.markdown("**Load by sector**")
        sec = analytics.sector_load(window)
        if sec.empty:
            st.info("No data.")
        else:
            st.altair_chart(alt.Chart(sec).mark_bar(color="#4c8dff").encode(
                x=alt.X("count:Q", title="reads"),
                y=alt.Y("sector:N", sort="-x", title=None)), width="stretch")

    st.markdown("**Origin → destination demand**")
    c1, c2 = st.columns([1, 3])
    gw = c1.checkbox("Gateway cameras only", value=False,
                     help="City entry and exit points — through-traffic demand.")
    od_df = od(max(window, 60), gw)
    if od_df.empty:
        st.info("No completed origin-destination trips in this window.")
    else:
        c2.dataframe(od_df.head(15)[["origin_name", "destination_name", "trips", "median_minutes"]]
                     .rename(columns={"origin_name": "origin", "destination_name": "destination",
                                      "median_minutes": "median trip (min)"}),
                     hide_index=True, width="stretch")
        top_pairs = od_df.head(120)
        st.altair_chart(alt.Chart(top_pairs).mark_rect().encode(
            x=alt.X("destination_name:N", title="destination", sort="-color"),
            y=alt.Y("origin_name:N", title="origin", sort="-color"),
            color=alt.Color("trips:Q", scale=alt.Scale(scheme="inferno"), title="trips"),
            tooltip=["origin_name", "destination_name", "trips", "median_minutes"]
        ).properties(height=420), width="stretch")


# --- alerts -------------------------------------------------------------
with tab_alerts:
    st.subheader("Alert queue")
    summ = alert_rules.summary(hours=24)
    cols = st.columns(5)
    cols[0].metric("Last 24 h", summ["total"])
    cols[1].metric("Open", summ["open"])
    for i, sev in enumerate(["critical", "high", "medium"]):
        cols[2 + i].metric(sev.capitalize(), summ["by_severity"].get(sev, 0))

    # The Kind filter gets the wide column. It holds one chip per alert rule -
    # six of them once the network has been running a while - and in a narrow
    # column they wrap into a stack that overflows its own row and pushes the
    # queue down the page.
    c1, c2 = st.columns([1, 4])
    open_only = c1.checkbox("Open only", value=True)
    kinds = sorted(summ["by_kind"]) or ["watchlist"]
    picked = c2.multiselect("Kind", kinds, default=kinds,
                            label_visibility="collapsed",
                            placeholder="Filter by alert kind")
    rows = db.alerts(limit=250, include_acked=not open_only, since=time.time() - 24 * 3600)
    rows = [a for a in rows if a["kind"] in picked]
    # Which of these a live camera caused. The rules never knew the difference -
    # core/enrol.py runs the same evaluate() every ingest path runs - so this is
    # a join for the operator's benefit, not a second class of alert.
    from_live = set(livefeed.live_plates())
    # One bulk lookup for the whole queue rather than forty inside the loop.
    # The rows shown are capped at 40 below, so only those are asked about -
    # looking up 250 to render 40 would be paid for on every rerun.
    _regs = registry_link.lookup_many(
        tuple(a["plate"] for a in rows[:40] if a["plate"]))
    if not rows:
        st.success("Nothing outstanding.")
    else:
        for a in rows[:40]:
            colour = SEVERITY_COLOUR.get(a["severity"], "#888")
            with st.container(border=True):
                # Three columns rather than two so Track sits beside Ack. An
                # alert is a claim about a vehicle, and the queue was a dead
                # end: every one of these rows named a registration that had
                # to be copied by hand into another tab to be worth anything.
                left, mid, right = st.columns([6, 1.1, 0.9])
                badge = ("<span style='background:#14351f;color:#7ee2a8;"
                         "border-radius:999px;padding:1px 7px;font-size:0.66rem;"
                         "font-weight:600;margin-left:7px'>LIVE CAMERA</span>"
                         if a["plate"] and a["plate"] in from_live else "")
                # What the registry holds on this registration, if anything
                # remarkable. A watchlist hit on a vehicle reported stolen is a
                # different situation from the same hit on a clean one, and the
                # queue could not previously tell them apart. Silent for an
                # ordinary registration, and silent during a registry outage -
                # neither is news, and badging either would train the operator
                # to stop reading badges.
                _r = _regs.get(a["plate"] or "")
                badge += registry_link.badge_html(_r)
                badge += registry_link.papers_html(_r)
                left.markdown(
                    f"<span style='color:{colour};font-weight:700'>{a['severity'].upper()}"
                    f" · {a['kind']}</span>{badge}  &nbsp; {clock(a['ts'])} "
                    f"({ago(a['ts'])})<br>{a['message']}", unsafe_allow_html=True)
                if a["camera_id"]:
                    left.caption(f"{net().name(a['camera_id'])} · {a['camera_id']}"
                                 + (f" · plate {a['plate']}" if a["plate"] else ""))
                # Not every alert names one - a congestion or camera-health
                # alert has no registration to open, and a button that leads to
                # an empty search is worse than no button.
                if a["plate"] and mid.button("Track", key=f"track{a['id']}",
                                             help=f"Open {a['plate']} on {TAB_TRACK}"):
                    track_plate(a["plate"])
                if not a["acked"] and right.button("Ack", key=f"ack{a['id']}"):
                    db.ack_alert(a["id"])
                    st.rerun()

    st.markdown("**Watchlist**")
    wl = db.watchlist()
    if wl:
        st.dataframe(pd.DataFrame([{
            "plate": w["plate"], "reason": w["reason"], "severity": w["severity"],
            "added": clock(w["added_ts"]),
            "sightings": len(db.sightings_for_plate(w["plate"])),
        } for w in wl]), hide_index=True, width="stretch")
    else:
        st.caption("Watchlist is empty.")


# --- ANPR engine --------------------------------------------------------
with tab_anpr:
    st.subheader("Plate reader")
    st.caption("The same engine that reads every camera capture in the simulation. "
               "Captures come from a physical camera model - a pole height, a lens, an "
               "exposure and the weather - not from filters laid over a clean render. "
               "See AUGMENTATION.md for what each stage does.")

    import cv2 as _cv2

    from anpr import calibrate
    from anpr import camera as camera_mod
    from anpr import plates as plate_lib

    def cv2_gray(img):
        return _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Which recogniser reads the captures below. config.OCR_BACKEND is the
    # platform-wide default (GODSEYE_OCR_BACKEND); this overrides it per view.
    from anpr.ocr import engine as _engine
    from anpr.ocr import engine_status as _engine_why

    _BACKENDS = {
        "crnn": "CRNN — synthetic camera corpus",
        "anpr_ocr": "anpr_ocr — PP-OCRv5, real Indian plates",
        "classical": "Classical — segment & classify",
    }
    _names = list(_BACKENDS)
    backend = st.radio(
        "Recogniser", _names, horizontal=True, format_func=lambda k: _BACKENDS[k],
        index=_names.index(config.OCR_BACKEND) if config.OCR_BACKEND in _names else 0,
        help="anpr_ocr reads 91.0% of real photographs against the CRNN's 0.9%, but "
             "that is a real-photograph number: on the synthetic captures this panel "
             "generates, the CRNN is the stronger reader. anpr_ocr needs paddle, so "
             "the dashboard has to be launched from .venv-paddle - see "
             "PADDLE_SETUP.md. Upload a real photograph below to see the difference.")

    def load_engine(name):
        """Build (or reuse) the chosen reader, and say so if it degraded."""
        with st.spinner(f"Loading the {name} reader…"):
            eng_ = _engine(name)
        why = _engine_why(name)
        if why:
            st.warning(why)
        return eng_

    def read_or_explain(name, call):
        """Run a read, or explain why the reader fell over. -> (ok, value).

        An uncaught exception here paints a red Python traceback across the
        whole panel, which in front of an audience reads as a broken platform
        rather than as an optional backend that this interpreter does not
        support. anpr_ocr is the one that gets here: paddle and torch clash
        over compiled symbols inside a single process, so the reader can load,
        answer once, and then fail on the next call. The CRNN is unaffected
        and recovers immediately - switching the recogniser back is enough,
        no restart - which is what the message says to do.
        """
        try:
            return True, call()
        except Exception as exc:
            import traceback
            if name == "anpr_ocr":
                st.error(
                    "**The anpr_ocr reader failed on this capture.** paddle and "
                    "torch clash over compiled symbols inside one process, so "
                    "this backend can load, answer once and then fail. It is "
                    "only reliable when the dashboard is launched from a "
                    "paddle-only interpreter - see `PADDLE_SETUP.md`.\n\n"
                    "Pick **CRNN** in the Recogniser row above and read again. "
                    "Nothing else in the platform is affected, and no restart "
                    "is needed.")
            else:
                st.error(f"**The {name} reader failed on this capture.** "
                         f"{type(exc).__name__}: {exc}")
            with st.expander("Technical detail"):
                st.code("".join(traceback.format_exception(exc)), language="text")
            return False, None

    mode = st.radio(
        "Read mode", ["Single frame", "Burst"], horizontal=True,
        help="A single frame is one photograph, which is what most ANPR demos show. "
             "A burst is what a real node actually delivers: it is triggered as the "
             "vehicle enters the zone and takes several frames at different "
             "distances, exposures and motion blurs. They fail in different ways, so "
             "the platform reads each one and votes them by CTC score.")
    n_frames = 1
    if mode == "Burst":
        n_frames = st.slider("Frames in the burst", 2, 10, int(config.BURST_FRAMES),
                             help=f"The platform runs {config.BURST_FRAMES} "
                                  "(config.BURST_FRAMES)")

    c1, c2, c3, c4, c5 = st.columns([2, 2, 1.6, 1.2, 1])
    text = c1.text_input("Plate (blank = random)", "")
    cond = c2.selectbox("Camera scenario", plate_lib.CONDITIONS, index=0,
                        format_func=lambda k: f"{k} - {camera_mod.SCENARIOS[k]}",
                        help="A mounting, an exposure and the weather, run through "
                             "the camera model in physical order")
    fault = c3.selectbox("Plate surface", plate_lib.SURFACE_FAULTS, index=0,
                         help="What has happened to the plate itself, before any "
                              "camera sees it")
    layout = c4.radio("Layout", ["One row", "Two rows"], horizontal=False,
                      help="Motorcycles, autos and trucks carry two-row plates")
    c5.write("")
    go = c5.button("Capture & read", width="stretch", type="primary")

    if go:
        import random as _random

        rng = _random.Random()
        eng = load_engine(backend)
        truth = text.upper().replace(" ", "")
        if not truth:
            # Prefer a vehicle the network has actually seen. The point of the
            # panel below is the join between a read and that vehicle's history,
            # and a freshly invented registration has no history to join to.
            busy = db.rows("SELECT plate FROM sightings GROUP BY plate"
                           " HAVING COUNT(*) >= 3 ORDER BY RANDOM() LIMIT 1")
            truth = busy[0]["plate"] if busy else plate_lib.random_plate(rng)
        two_row = layout == "Two rows"
        # every frame is a fresh trip through the camera model: the same plate,
        # the same site, a different instant
        caps = [plate_lib.capture(truth, cond, rng, fault=fault, two_row=two_row)
                for _ in range(n_frames)]
        cap = caps[0]
        t0 = time.time()
        if n_frames == 1:
            ok, got = read_or_explain(backend, lambda: eng.read_detailed(cap.image))
            if not ok:
                st.stop()
            read, dbg = got
            per_frame = [read]
        else:
            ok, per_frame = read_or_explain(
                backend, lambda: [eng.read(c.image) for c in caps])
            if not ok:
                st.stop()
            dbg = eng._last_debug
            read = eng.fuse_reads(list(per_frame))
        ms = (time.time() - t0) * 1000

        i1, i2 = st.columns(2)
        i1.image(cap.image, caption=f"{cap.condition} · plate {cap.fault} · "
                                    f"{cap.image.shape[1]}x{cap.image.shape[0]} px"
                                    + (f" · frame 1 of {n_frames}" if n_frames > 1 else ""),
                 width="stretch")
        i1.caption(cap.detail)
        if dbg.get("ink") is not None:
            i2.image(dbg["ink"], caption=f"ink mask · winning hypothesis '{read.variant}'",
                     width="stretch")
        ok = read.text == cap.text
        st.markdown(f"### {'✅' if ok else '❌'} `{read.pretty or '—'}`"
                    f"  &nbsp;&nbsp; ground truth `{plate_lib.pretty(cap.text)}`")
        m = st.columns(4)
        # Two different quantities, deliberately side by side. The raw score is
        # an exact CTC likelihood - excellent at ORDERING reads, and on a scale
        # nobody can state the meaning of. The calibrated figure answers the
        # question an operator is actually asking, from measured outcomes on the
        # 532-crop real corpus. Below about 0.7 raw they diverge sharply, which
        # is the point: down there the score barely discriminates at all.
        _p = calibrate.p_correct(read.confidence)
        m[0].metric("Confidence", f"{read.confidence:.0%}",
                    help="Exact CTC likelihood — the reader's own score")
        if _p is not None:
            m[1].metric("P(correct)", f"{_p:.0%}",
                        help="Calibrated against measured outcomes on the real-photograph "
                             "corpus (isotonic, held-out skill +0.24). Reporting only — no "
                             "storage threshold uses it.")
        else:
            m[1].metric("Grammar pattern", read.pattern or "—")
        m[2].metric("Decode time", f"{ms:.0f} ms",
                    help=f"{n_frames} frame(s) decoded" if n_frames > 1 else None)
        if n_frames > 1:
            m[3].metric("Frames agreeing", f"{read.agreement:.0%}",
                        help="Share of frames that decoded the winning string")
        else:
            m[3].metric("State code repaired", "yes" if read.repaired else "no")

        if n_frames > 1:
            st.markdown("**What each frame saw.** The burst is the whole point: a frame "
                        "that loses the plate to glare or blur is outvoted by the ones "
                        "that did not, and the CTC score is what tells them apart.")
            solo = sum(1 for r in per_frame if r.text == cap.text)
            # the frame whose confidence the fused read reports - the single most
            # confident of the ones that agreed on the winning string
            agreeing = [i for i, r in enumerate(per_frame) if r.text == read.text]
            lead = max(agreeing, key=lambda i: per_frame[i].confidence) if agreeing else -1
            cols = st.columns(min(n_frames, 5))
            for i, (c, r) in enumerate(zip(caps, per_frame)):
                col = cols[i % len(cols)]
                hit = r.text == cap.text
                col.image(c.image, width="stretch")
                col.caption(f"{'✅' if hit else '❌'} `{r.text or '—'}` · "
                            f"{r.confidence:.0%}" + (" · **leads**" if i == lead else ""))
            st.caption(f"{solo}/{n_frames} frames read this plate correctly on their own; "
                       f"fused, the platform "
                       f"{'got it right' if ok else 'still got it wrong'}. Measured over "
                       f"1,000 vehicles across all ten conditions, one frame reads 42.4% "
                       f"of plates and five read 73.1%.")
        if read.raw and read.raw != read.text:
            st.caption(f"Unconstrained per-atom read was `{read.raw}` — the plate grammar and "
                       "the merge/skip search corrected it.")
        if read.char_confidences:
            st.altair_chart(alt.Chart(pd.DataFrame({
                "character": [f"{i+1}. {c}" for i, c in enumerate(read.text)],
                "confidence": read.char_confidences,
            })).mark_bar().encode(
                x=alt.X("character:N", sort=None), y="confidence:Q",
                color=alt.condition(alt.datum.confidence > 0.8, alt.value("#2ea043"),
                                    alt.value("#f5a524"))), width="stretch")

        if read.text:
            st.divider()
            st.markdown(f"### Where `{plate_lib.pretty(read.text)}` has been")
            st.caption("A read on its own is a string. This is the rest of the "
                       "sentence: the same registration looked up across every "
                       "camera in the network, in order.")
            render_plate_history(read.text, key="anpr_read")
            st.session_state["anpr_capture_read"] = read

    # Deliberately outside `if go:`. A Streamlit button is True for exactly one
    # script run, so anything nested under it unmounts as soon as any other
    # widget forces a rerun - and the enrolment form's own submit *is* a rerun.
    # Rendering the panel from session state instead means the form is still
    # mounted when its handler needs to run. (The upload branch below does not
    # need this: a file_uploader keeps returning its file, a button does not.)
    #
    # Offered here as well as under an uploaded photograph so the enrolment flow
    # can be rehearsed without one: type a registration this network has never
    # seen into the box above, capture it, and the same panel appears.
    _captured = st.session_state.get("anpr_capture_read")
    if _captured is not None and getattr(_captured, "plate_found", False):
        render_enrol_panel(_captured, key="anpr_read")

    render_sequence_reader(backend, load_engine, read_or_explain)

    up = st.file_uploader("…or upload a plate photograph", type=["png", "jpg", "jpeg", "bmp"])
    if up is not None:
        import cv2

        from anpr import imageio
        raw = up.getvalue()
        # Started before the decode, not just around the read: the number that
        # means anything to someone watching is how long from handing the page
        # a photograph to seeing a registration, which includes turning the
        # bytes into an array and, on the very first upload of a session,
        # whatever the engine still has to build.
        t_upload = time.perf_counter()
        arr = imageio.load_bytes(raw)
        if arr is None:
            st.error("Could not decode that image.")
        else:
            # Cache the read against the image bytes. Streamlit re-runs this
            # whole branch on every widget interaction, and the enrolment form
            # below is widgets - without this, changing the junction dropdown
            # would re-run the recogniser. That is ~40 ms on the CRNN and
            # invisible, but ~30 s on anpr_ocr, which is the backend a real
            # photograph actually needs.
            sig = f"{backend}:{hashlib.sha1(raw, usedforsecurity=False).hexdigest()}"
            cached = st.session_state.get("anpr_upload_read")
            if cached and cached[0] == sig and len(cached) == 4:
                # The elapsed time is cached with the read on purpose. Serving
                # a cached read takes microseconds, so timing the cache hit
                # would show a number that has nothing to do with reading a
                # plate - and it is the reading that is being observed.
                read, elapsed_ms, stages = cached[1], cached[2], cached[3]
            else:
                # Split into stages, because a single total is not diagnostic.
                # A read that takes fifty seconds looks identical to a slow
                # recogniser whether the time went into the model, into
                # acquiring the engine, or into decoding a very large JPEG -
                # and those have nothing to do with each other. Measured
                # out-of-process this whole path is under 200 ms, so any large
                # number here is something the dashboard does that a plain
                # script does not, and the breakdown is what will say which.
                t_eng = time.perf_counter()
                up_eng = load_engine(backend)
                ms_engine = 1000.0 * (time.perf_counter() - t_eng)

                t_read = time.perf_counter()
                with st.spinner("Locating and reading plates…"):
                    ok, read = read_or_explain(backend, lambda: up_eng.read_frame(arr))
                if not ok:
                    st.stop()
                ms_read = 1000.0 * (time.perf_counter() - t_read)

                elapsed_ms = 1000.0 * (time.perf_counter() - t_upload)
                stages = (f"decode {elapsed_ms - ms_engine - ms_read:,.0f}"
                          f" · engine {ms_engine:,.0f} · read {ms_read:,.0f}")
                st.session_state["anpr_upload_read"] = (sig, read, elapsed_ms, stages)
            shown = arr.copy()
            for (x, y, w, h) in read.candidates:
                cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 165, 255), 2)
            c1, c2 = st.columns([1, 1])
            c1.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), width="stretch",
                     caption=f"{len(read.candidates)} plate-shaped region(s) examined"
                             if read.candidates else "no plate-shaped region found")
            if read.text and read.plate_found:
                c2.markdown(f"### `{read.pretty}`")
                c2.metric("Confidence", f"{read.confidence:.0%}")
                c2.caption(f"pattern {read.pattern} · hypothesis {read.variant}")
            else:
                c2.markdown("### No registration found")
                c2.info(read.reason or "nothing plate-like in this image")
                c2.caption("The orange boxes are where it looked. The plate grammar is what "
                           "buys the accuracy on the benchmark below, and it is the same thing "
                           "that makes the engine refuse handwriting, signage and arbitrary "
                           "text.")
            # Upload-to-answer, deliberately understated. It is here to be
            # checked by someone who went looking for it, not to be part of the
            # pitch, so it sits below the result at 0.7rem and 40% opacity -
            # legible on purpose, quiet on purpose. Backend and region count
            # ride along because a bare millisecond figure is unfalsifiable:
            # 90 ms means something very different for one region than for ten.
            st.markdown(
                f"<div style='font-size:0.7rem;opacity:0.4;margin-top:-8px'>"
                f"{elapsed_ms:,.0f} ms ({stages}) · {backend} · "
                f"{len(read.candidates or [])} region(s)</div>",
                unsafe_allow_html=True)

            if read.text and read.plate_found:
                st.markdown(f"**Where `{read.pretty}` has been**")
                render_plate_history(read.text, key="anpr_upload")
                render_enrol_panel(read, key="anpr_upload")

    st.divider()
    st.subheader("Measured accuracy")
    bench_path = Path(ROOT) / "models" / "benchmark.json"
    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Samples per condition", 10, 200, 40, step=10)
    run = c1.button("Re-measure now")
    data = None
    if run:
        from anpr import benchmark as bench
        with st.spinner(f"Reading {n * len(plate_lib.CONDITIONS)} synthetic plates…"):
            results, overall, elapsed = bench.run(int(n), verbose=False)
        data = {"conditions": [r.__dict__ for r in results], "overall": overall.__dict__,
                "elapsed_s": elapsed, "samples_per_condition": int(n)}
    elif bench_path.exists():
        import json
        data = json.loads(bench_path.read_text())

    if data:
        df = pd.DataFrame(data["conditions"])
        o = data["overall"]
        c2.metric("Overall plate accuracy", f"{o['plate_accuracy']:.1%}",
                  help="Exact full-string match on a single frame, averaged over "
                       "conditions — the recogniser measured in isolation")
        if o.get("burst_accuracy"):
            c2.metric(f"Fused over {data.get('burst_frames', config.BURST_FRAMES)} frames",
                      f"{o['burst_accuracy']:.1%}",
                      help="The path the platform actually runs: a camera contributes "
                           "a burst per vehicle and the frames are voted by CTC score")
        c2.metric("Accepted-read accuracy", f"{o['accepted_accuracy']:.1%}",
                  help="Of the reads the engine was confident enough to store, "
                       "how many were exactly right")
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("plate_accuracy:Q", title="plate accuracy", axis=alt.Axis(format="%")),
            y=alt.Y("condition:N", sort="-x", title=None),
            color=alt.condition(alt.datum.plate_accuracy > 0.9, alt.value("#2ea043"),
                                alt.value("#f5a524")),
            tooltip=["condition", "plate_accuracy", "char_accuracy", "accepted_accuracy"]),
            width="stretch")
        st.dataframe(df.style.format({"plate_accuracy": "{:.1%}", "char_accuracy": "{:.1%}",
                                      "accepted_accuracy": "{:.1%}",
                                      "mean_confidence": "{:.2f}"}),
                     hide_index=True, width="stretch")
    else:
        st.caption("No benchmark on file yet — run `python -m anpr.benchmark "
                   "--samples 80 --json models/benchmark.json`.")

    with st.expander("Model card"):
        from anpr.ocr import PATTERNS, engine as _engine
        mdl = _engine().model
        st.write({
            "classifier": type(mdl.clf).__name__,
            "classes": len(mdl.classes),
            "training glyphs": mdl.trained_on,
            "training stages": mdl.stages,
            "held-out glyph accuracy": round(mdl.val_accuracy, 4),
            "grammar patterns": PATTERNS,
            "confidence floor for storage": (config.MIN_PLATE_CONFIDENCE_CRNN
                                             if _engine().crnn is not None
                                             else config.MIN_PLATE_CONFIDENCE),
        })

# No page-level refresh loop. It used to sleep five seconds, clear every cached
# query and rerun the whole script, which rebuilt all four maps and remounted
# their canvases on a timer - the flicker was the page tearing itself down. The
# live map now refreshes itself as a fragment, and the cached queries already
# carry a five-second TTL, so the data is just as fresh at a fraction of the work.
