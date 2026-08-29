"""GodsEye control room — Streamlit dashboard.

    streamlit run dashboard/app.py

Reads the sighting store directly, so it works whether or not the API process is
running; the API is only needed for the live feed and the incident injectors.
"""
from __future__ import annotations

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
import maps
from core import alerts as alert_rules
from core import analytics, db, trajectory
from core.network import network

st.set_page_config(page_title="GodsEye", page_icon="👁", layout="wide")

SEVERITY_COLOUR = {"critical": "#e5484d", "high": "#f76808", "medium": "#f5d90a", "low": "#8e8e8e"}
LEVEL_COLOUR = {"free": [46, 160, 67], "moderate": [212, 167, 44],
                "heavy": [232, 106, 23], "severe": [214, 40, 46]}


# --- data access --------------------------------------------------------
@st.cache_resource
def net():
    return network()


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
st.sidebar.title("👁 GodsEye")
st.sidebar.caption(f"City ANPR platform · {net().city} · {len(net())} cameras")

window = st.sidebar.select_slider("Analysis window",
                                  options=[10, 15, 30, 45, 60, 120, 180, 360],
                                  value=config.DEFAULT_WINDOW_MIN,
                                  format_func=lambda m: f"{m} min" if m < 60 else f"{m//60} h")
auto = st.sidebar.checkbox("Auto-refresh (5 s)", value=False)
stats = db.stats()
if stats["sightings"] == 0:
    st.sidebar.error("No sightings yet — run `python seed.py`")
st.sidebar.metric("Sightings stored", f"{stats['sightings']:,}")
st.sidebar.metric("Distinct plates", f"{stats['unique_plates']:,}")
st.sidebar.metric("Open alerts", stats["open_alerts"])
st.sidebar.caption(f"Data spans {clock(stats['first_ts'])} → {clock(stats['last_ts'])}")

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
                st.cache_data.clear()

tab_live, tab_track, tab_flow, tab_alerts, tab_anpr = st.tabs(
    ["Live network", "Track a plate", "City analytics", "Alerts", "ANPR engine"])


# --- live network -------------------------------------------------------
with tab_live:
    s = summary(window)
    k = st.columns(6)
    k[0].metric("Vehicles / min", s["vehicles_per_min"])
    k[1].metric("Unique plates", f"{s['unique_plates']:,}")
    k[2].metric("Median speed", f"{s['median_network_kmph']:.0f} km/h")
    k[3].metric("Congested links", f"{s['congested_links']}/{s['monitored_links']}")
    k[4].metric("Cameras reporting", f"{s['active_cameras']}/{s['total_cameras']}")
    k[5].metric("Mean read confidence", f"{s['mean_ocr_confidence']:.0%}")

    if s["worst_link"]:
        w = s["worst_link"]
        st.info(f"**Worst corridor right now — {w['link']}**: {w['median_kmph']} km/h against a "
                f"{w['free_kmph']:.0f} km/h free-flow limit ({w['level']}), costing "
                f"{w['delay_min_total']:.0f} vehicle-minutes in the last {window} min.")

    d = density(window)
    lf = links(window)

    opt = st.columns([1.1, 1, 1, 1, 1, 1])
    basemap = opt[0].selectbox("Basemap", list(maps.BASEMAPS), index=0, key="net_basemap")
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

    st.pydeck_chart(maps.deck(layers, maps.view(net(), zoom=11.0, pitch=pitch), basemap),
                    height=560)
    st.markdown(maps.legend("network"), unsafe_allow_html=True)
    st.caption("Roads are drawn from the network's own link geometry and coloured by the "
               "worse of their two directions against that road's free-flow speed. Camera "
               "discs size with per-lane load; the halo marks gateways against junctions.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Busiest corridors right now**")
        if lf.empty:
            st.info("No vehicle has been seen at two adjacent cameras in this window yet.")
        else:
            top = lf.head(12)[["name", "direction", "vehicles", "median_kmph", "free_kmph",
                               "congestion_ratio", "level"]]
            st.dataframe(top.rename(columns={"median_kmph": "km/h", "free_kmph": "free-flow",
                                             "congestion_ratio": "× slower"}),
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

    st.subheader("Busiest cameras")
    top = d.head(12)[["name", "sector", "road", "count", "per_min", "unique_plates",
                      "mean_confidence", "load"]]
    st.dataframe(top.style.format({"per_min": "{:.2f}", "mean_confidence": "{:.0%}",
                                   "load": "{:.2f}"}),
                 hide_index=True, width="stretch")


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
        since = time.time() - hours * 3600
        matches = trajectory.search(query, limit=8, since=since)
        if not matches:
            st.warning("No plate close to that was seen in the window.")
        else:
            labels = [
                f"{m['plate']}  ·  {m['sightings']} sightings  ·  "
                + ("exact match" if m["exact"] else f"near match (distance {m['distance']})")
                + ("  ·  ⚠ WATCHLISTED" if m["watched"] else "")
                for m in matches
            ]
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

            if summ["implausible_legs"]:
                st.error(f"{summ['implausible_legs']} leg(s) are physically impossible — "
                         "this plate is probably running on more than one vehicle.")

            tcol = st.columns([1.2, 1, 3])
            tmap = tcol[0].selectbox("Basemap", list(maps.BASEMAPS), index=0, key="traj_basemap")
            tpitch = tcol[1].slider("Tilt", 0, 60, 25, key="traj_pitch")
            layers, stops = maps.trajectory_layers(traj, net())
            if layers and not stops.empty:
                view = pdk.ViewState(
                    latitude=float(stops["lat"].mean()), longitude=float(stops["lon"].mean()),
                    zoom=_fit_zoom(stops), pitch=tpitch)
                st.pydeck_chart(maps.deck(
                    [maps.road_layer(maps.road_frame(net(), pd.DataFrame()))] + layers,
                    view, tmap), height=520)
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
                    "engine": s["ocr_variant"], "true plate (sim)": s["true_plate"],
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
                            height=520)
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

    c1, c2, c3 = st.columns([1, 1, 2])
    open_only = c1.checkbox("Open only", value=True)
    kinds = sorted(summ["by_kind"]) or ["watchlist"]
    picked = c2.multiselect("Kind", kinds, default=kinds)
    rows = db.alerts(limit=250, include_acked=not open_only, since=time.time() - 24 * 3600)
    rows = [a for a in rows if a["kind"] in picked]
    if not rows:
        st.success("Nothing outstanding.")
    else:
        for a in rows[:40]:
            colour = SEVERITY_COLOUR.get(a["severity"], "#888")
            with st.container(border=True):
                left, right = st.columns([6, 1])
                left.markdown(
                    f"<span style='color:{colour};font-weight:700'>{a['severity'].upper()}"
                    f" · {a['kind']}</span>  &nbsp; {clock(a['ts'])} ({ago(a['ts'])})<br>"
                    f"{a['message']}", unsafe_allow_html=True)
                if a["camera_id"]:
                    left.caption(f"{net().name(a['camera_id'])} · {a['camera_id']}"
                                 + (f" · plate {a['plate']}" if a["plate"] else ""))
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
               "Generate a plate under any condition, or upload a photograph.")

    from anpr import plates as plate_lib

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    text = c1.text_input("Plate (blank = random)", "")
    cond = c2.selectbox("Capture condition", plate_lib.CONDITIONS, index=len(plate_lib.CONDITIONS) - 1)
    sev = c3.slider("Severity", 0.2, 1.0, 0.75, 0.05)
    c4.write("")
    go = c4.button("Capture & read", width="stretch", type="primary")

    if go:
        import random as _random

        from anpr.ocr import engine as _engine
        cap = plate_lib.capture(text.upper().replace(" ", "") or None, cond,
                                _random.Random(), sev)
        t0 = time.time()
        read, dbg = _engine().read_detailed(cap.image)
        ms = (time.time() - t0) * 1000
        i1, i2 = st.columns(2)
        i1.image(cap.image, caption=f"camera capture · {cap.condition}", width="stretch")
        if dbg.get("ink") is not None:
            i2.image(dbg["ink"], caption=f"ink mask · winning hypothesis '{read.variant}'",
                     width="stretch")
        ok = read.text == cap.text
        st.markdown(f"### {'✅' if ok else '❌'} `{read.pretty or '—'}`"
                    f"  &nbsp;&nbsp; ground truth `{plate_lib.pretty(cap.text)}`")
        m = st.columns(4)
        m[0].metric("Confidence", f"{read.confidence:.0%}")
        m[1].metric("Grammar pattern", read.pattern or "—")
        m[2].metric("Decode time", f"{ms:.0f} ms")
        m[3].metric("State code repaired", "yes" if read.repaired else "no")
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

    up = st.file_uploader("…or upload a plate photograph", type=["png", "jpg", "jpeg", "bmp"])
    if up is not None:
        import cv2

        from anpr.ocr import engine as _engine
        arr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            st.error("Could not decode that image.")
        else:
            with st.spinner("Locating and reading plates…"):
                read = _engine().read_frame(arr)
            shown = arr.copy()
            for (x, y, w, h) in read.candidates:
                cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 165, 255), 2)
            c1, c2 = st.columns([1, 1])
            c1.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), width="stretch",
                     caption=f"{len(read.candidates)} plate-shaped region(s) examined"
                             if read.candidates else "no plate-shaped region found")
            if read.text:
                c2.markdown(f"### `{read.pretty}`")
                c2.metric("Confidence", f"{read.confidence:.0%}")
                c2.caption(f"pattern {read.pattern} · hypothesis {read.variant}")
                if read.confidence < config.MIN_PLATE_CONFIDENCE:
                    c2.warning("Below the confidence floor — the platform would discard this "
                               "read rather than store it.")
            else:
                c2.markdown("### No registration found")
                c2.info(read.reason or "nothing plate-like in this image")
                c2.caption("The orange boxes are where it looked. This is a number-plate "
                           "reader: it decodes an Indian registration — two letters, district "
                           "digits, a series and four digits — and deliberately refuses "
                           "anything that is not one. It is not a general text or handwriting "
                           "reader.")

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
                  help="Exact full-string match, averaged over conditions")
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
            "confidence floor for storage": config.MIN_PLATE_CONFIDENCE,
        })

if auto:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
