"""The security console: actionable insights, routed to who can act on them.

A separate Streamlit app on its own port, deliberately NOT a tab inside the
control room. Two reasons, and the second is the one that matters:

  It is a different job. The control room watches traffic; this watches
  vehicles against what the state holds on them. The people who act on the two
  are not the same people.

  It has to be removable. This whole package was built on the promise that the
  working demo is not disturbed, and a separate process on a separate port with
  its own database is removable by deleting a folder. A tab wired into
  dashboard/app.py would not be.

    python -m registry.console          ->  http://localhost:8506
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd                                            # noqa: E402
import streamlit as st                                         # noqa: E402

from registry import authority, live, sentinel, settings, store  # noqa: E402

st.set_page_config(page_title="GodsEye — Security Console", page_icon="🛡",
                   layout="wide", initial_sidebar_state="expanded")

SEV = {"critical": "#e5484d", "high": "#f76808",
       "medium": "#f5d90a", "low": "#8e8e8e"}
AUTH_ICON = {"police": "🚓", "traffic": "🚦", "transport": "🏛",
             "pollution": "🌫", "platform": "⚙"}


def clock(ts) -> str:
    if not ts:
        return "—"
    return time.strftime("%d %b %H:%M", time.localtime(float(ts)))


@st.cache_data(ttl=20, show_spinner=False)
def sweep(hours: float, limit: int) -> dict:
    return sentinel.sweep(hours=hours, limit=limit)


@st.cache_data(ttl=20, show_spinner=False)
def registry_stats() -> dict:
    return store.stats()


@st.cache_data(ttl=10, show_spinner=False)
def brief(plate: str, hours: float) -> dict:
    return sentinel.brief(plate, hours=hours)


# --- sidebar ------------------------------------------------------------
st.sidebar.markdown(
    """<div style="line-height:1.25;margin:0 0 1.1rem 0">
      <div style="font-size:1.3rem;font-weight:600;letter-spacing:-0.015em;
                  color:#EDF2F8">🛡 Security Console</div>
      <div style="font-size:0.68rem;font-weight:500;letter-spacing:0.09em;
                  text-transform:uppercase;color:#5C6B7E;margin-top:0.28rem">
        vehicle registry · actionable insights
      </div>
    </div>""", unsafe_allow_html=True)

hours = st.sidebar.select_slider("Look back", options=[1, 3, 6, 12, 24, 48, 168],
                                 value=24,
                                 format_func=lambda h: f"{h} h" if h < 48 else "7 days")
limit = st.sidebar.number_input("Registrations to check", 100, 5000, 1500, step=100)

rs = registry_stats()
st.sidebar.caption(f"Registry holds **{rs['vehicles']:,}** vehicles · "
                   f"{rs['cases_open']:,} open cases · "
                   f"Rs {rs['outstanding']:,.0f} outstanding")
if rs["by_status"]:
    flagged = {k: v for k, v in rs["by_status"].items() if k != "active"}
    if flagged:
        st.sidebar.caption("flagged on the registry: " + ", ".join(
            f"{v} {k}" for k, v in sorted(flagged.items())))

st.sidebar.divider()
st.sidebar.caption(
    "**This registry is synthetic.** Every record is invented by "
    "`registry/store.py` from a hash of the plate — no real vehicle, owner or "
    "case is described. It is shaped like a VAHAN response so the lookup can "
    "be swapped for a real one without changing anything above it.")
if st.sidebar.button("Refresh now", width="stretch"):
    sweep.clear()
    registry_stats.clear()
    brief.clear()
    st.rerun()


# --- demo controls ------------------------------------------------------
# The failure modes are switches rather than a paragraph, because "we handle a
# registry outage" is a claim and flipping it in front of someone is evidence.
# They act on the SERVICE, not on this process: the console asking itself to
# pretend would demonstrate nothing about the integration.
def _service(method: str, path: str, timeout: float = 2.5):
    import requests
    try:
        r = requests.request(method, f"{settings.SERVICE_URL}{path}",
                             timeout=timeout)
        return r.json() if r.content else {}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}"}


st.sidebar.divider()
with st.sidebar.expander("Demo controls", expanded=False):
    health = _service("GET", "/api/v1/health")
    if health.get("error"):
        st.caption(f"⚪ registry service not running on "
                   f"`{settings.SERVICE_URL}` — the console still works "
                   f"(it reads the store directly), but referrals cannot be "
                   f"delivered and these switches do nothing.")
    else:
        st.caption(f"🟢 service up · registry **{health.get('force_mode')}** · "
                   f"authority **{health.get('authority_mode')}**")

    st.markdown("**Registry**")
    rm = st.radio("Registry mode", ["NORMAL", "NOT_FOUND", "UNAVAILABLE"],
                  horizontal=False, label_visibility="collapsed",
                  index=["NORMAL", "NOT_FOUND", "UNAVAILABLE"].index(
                      health.get("force_mode", "NORMAL")
                      if health.get("force_mode") in
                      ("NORMAL", "NOT_FOUND", "UNAVAILABLE") else "NORMAL"),
                  key="reg_mode")
    if st.button("Apply registry mode", width="stretch"):
        _service("POST", f"/api/v1/registry/mode/{rm}")
        settings.FORCE_MODE = "" if rm == "NORMAL" else rm
        store.cache_clear()
        sweep.clear(); brief.clear(); registry_stats.clear()
        st.rerun()
    st.caption("UNAVAILABLE is the one worth showing: every lookup answers "
               "UNKNOWN, and the console stops concluding anything rather than "
               "reporting vehicles as unregistered.")

    st.markdown("**Authority**")
    am = st.radio("Authority mode", ["ONLINE", "OFFLINE"], horizontal=True,
                  label_visibility="collapsed",
                  index=0 if health.get("authority_mode") != "OFFLINE" else 1,
                  key="auth_mode")
    if st.button("Apply authority mode", width="stretch"):
        _service("POST", f"/api/v1/authority/mode/{am}")
        st.rerun()
    st.caption("OFFLINE makes every referral fail after its retries. The point "
               "is what survives: the incident is still on the queue below, "
               "with a delivery problem attached to it.")


# --- header -------------------------------------------------------------
st.markdown(
    "<div style='font-size:0.63rem;font-weight:600;letter-spacing:0.14em;"
    "text-transform:uppercase;color:#4E5E73'>GodsEye &nbsp;·&nbsp; "
    "Vehicle registry intelligence</div>"
    "<div style='font-size:1.9rem;font-weight:600;letter-spacing:-0.02em;"
    "color:#EDF2F8;line-height:1.15;margin:0.1rem 0 0.25rem 0'>"
    "Security Console</div>"
    "<div style='font-size:0.98rem;color:#93A1B4;max-width:64rem;line-height:1.5'>"
    "What the cameras saw, checked against what the state holds on the vehicle — "
    "and routed to whoever can act on it.</div>", unsafe_allow_html=True)
st.write("")

data = sweep(float(hours), int(limit))
findings = data.get("findings", [])
by_sev = data.get("by_severity", {})

# An empty queue and a blind one look the same. Say which this is, at the top,
# before any of the zeros below can be read as an all-clear.
if data.get("blind"):
    st.markdown(
        "<div style='background:#2a1f10;border:1px solid #f5a524;"
        "border-radius:10px;padding:11px 15px;margin-bottom:12px'>"
        "<b style='color:#f5a524'>The registry could not be reached.</b> "
        "<span style='color:#d5c3a5'>None of the "
        f"{data.get('checked', 0):,} registrations seen in this window could be "
        "checked, so the counts below are <i>zero because nothing was learned</i> "
        "— not because nothing is outstanding. No vehicle here is being reported "
        "as unregistered. Incidents already referred are unaffected.</span></div>",
        unsafe_allow_html=True)
elif data.get("unknown"):
    st.warning(f"{data['unknown']:,} of {data.get('checked', 0):,} registrations "
               f"could not be checked — those vehicles are absent from the "
               f"counts below rather than cleared by them.")

k = st.columns(5)
for col, label, key, colour in (
        (k[0], "Critical", "critical", SEV["critical"]),
        (k[1], "High", "high", SEV["high"]),
        (k[2], "Medium", "medium", SEV["medium"])):
    col.markdown(
        f"<div style='font-size:0.66rem;font-weight:550;letter-spacing:0.055em;"
        f"text-transform:uppercase;color:#6D7C90'>{label}</div>"
        f"<div style='font-size:2rem;font-weight:600;color:{colour};"
        f"font-variant-numeric:tabular-nums'>{by_sev.get(key, 0)}</div>",
        unsafe_allow_html=True)
k[3].markdown(
    f"<div style='font-size:0.66rem;font-weight:550;letter-spacing:0.055em;"
    f"text-transform:uppercase;color:#6D7C90'>Registrations checked</div>"
    f"<div style='font-size:2rem;font-weight:600;color:#EDF2F8;"
    f"font-variant-numeric:tabular-nums'>{data.get('checked', 0):,}</div>",
    unsafe_allow_html=True)
k[4].markdown(
    f"<div style='font-size:0.66rem;font-weight:550;letter-spacing:0.055em;"
    f"text-transform:uppercase;color:#6D7C90'>Window</div>"
    f"<div style='font-size:2rem;font-weight:600;color:#EDF2F8'>"
    f"{hours}<span style='font-size:0.9rem;color:#6D7C90'> h</span></div>",
    unsafe_allow_html=True)

st.write("")
tab_queue, tab_notified, tab_desk, tab_plate, tab_registry = st.tabs(
    ["Action queue", "Authority notified", "By authority", "Look up a plate",
     "The registry"])


def _finding_card(f: dict, key: str) -> None:
    colour = SEV.get(f["severity"], "#888")
    icon = AUTH_ICON.get(f["authority"], "•")
    with st.container(border=True):
        st.markdown(
            f"<span style='color:{colour};font-weight:700;font-size:0.78rem'>"
            f"{f['severity'].upper()}</span>"
            f"<span style='color:#6D7C90;font-size:0.72rem;margin-left:8px;"
            f"letter-spacing:0.06em;text-transform:uppercase'>{f['kind'].replace('_',' ')}"
            f"</span><br>"
            f"<span style='font-size:1.05rem;font-weight:600'>{f['headline']}</span>",
            unsafe_allow_html=True)
        st.caption(f["detail"])
        c1, c2 = st.columns([3, 2])
        with c1:
            if f.get("evidence"):
                st.markdown("".join(
                    f"<div style='font-size:0.78rem;color:#93A1B4'>• {e}</div>"
                    for e in f["evidence"]), unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='font-size:0.72rem;color:#6D7C90;text-transform:uppercase;"
                f"letter-spacing:0.06em'>Routed to</div>"
                f"<div style='font-size:0.9rem'>{icon} {f['authority_name']}</div>",
                unsafe_allow_html=True)
            if f.get("camera_name"):
                st.caption(f"last seen {f['camera_name']} · {clock(f['last_seen'])}")
        if f.get("recommended"):
            st.markdown(
                f"<div style='font-size:0.8rem;color:#9fe8b5;margin-top:4px'>"
                f"→ {f['recommended']}</div>", unsafe_allow_html=True)
        # The caveat is rendered with the finding, never in a footnote. These
        # name a vehicle and route to a police control room; the reader has to
        # see what could explain the finding other than guilt.
        if f.get("caveat"):
            st.markdown(
                f"<div style='font-size:0.78rem;color:#ffd79a;background:#2a1f10;"
                f"border-left:3px solid #f5a524;padding:6px 9px;border-radius:5px;"
                f"margin-top:6px'><b>Before acting:</b> {f['caveat']}</div>",
                unsafe_allow_html=True)
        # Referral is a button, not an automatic consequence of the finding.
        # Everything on this page is inference from a camera read; a person
        # decides that it is worth an authority's time.
        prior = store.incidents_for(f["plate"], f["kind"])
        if prior:
            p = prior[0]
            st.caption(f"referred as **{p['ref']}** · {p['delivery_status']}"
                       + (f" · {p['authority_ref']}" if p["authority_ref"] else ""))
        elif st.button(f"Refer to {f['authority_name']}", key=f"refer_{key}",
                       width="stretch"):
            res = authority.refer(f, registry_status="FOUND")
            d = res["delivery"]
            if d["status"] == authority.ACKNOWLEDGED:
                st.success(f"{res['incident']['ref']} acknowledged as "
                           f"{d['authority_ref']}")
            else:
                st.error(f"{res['incident']['ref']} recorded, but not delivered "
                         f"after {d.get('attempts')} attempts — it is on the "
                         f"Authority notified tab to retry.")
            st.rerun()


with tab_queue:
    if not findings and data.get("blind"):
        st.info("Nothing can be said about this window while the registry is "
                "unreachable. This is not an all-clear.")
    elif not findings:
        st.success("Nothing outstanding against any registration seen in this window.")
    else:
        c1, c2 = st.columns([1, 3])
        only = c1.selectbox("Severity", ["all", "critical", "high", "medium"])
        kinds = sorted({f["kind"] for f in findings})
        picked = c2.multiselect("Kind", kinds, default=kinds,
                                label_visibility="collapsed",
                                placeholder="Filter by finding type")
        shown = [f for f in findings
                 if (only == "all" or f["severity"] == only) and f["kind"] in picked]
        st.caption(f"{len(shown)} of {len(findings)} findings")
        for i, f in enumerate(shown[:60]):
            _finding_card(f, f"q{i}")
        if len(shown) > 60:
            st.caption(f"…and {len(shown) - 60} more. Narrow the filters above — "
                       "this list is capped so the page stays responsive.")


with tab_notified:
    # The whole argument of this tab is the gap between the two columns:
    # an incident is RECORDED the moment it is found, and DELIVERED separately.
    # When delivery fails the incident is still here - which is the property
    # that makes the notification path safe to depend on.
    st.caption("What has been referred to an authority, and what happened to "
               "the referral. The incident is stored before anyone is called, "
               "so a delivery failure never loses it.")

    dsum = authority.summary()
    inc = store.incidents(limit=200)
    by_d = dsum["by_delivery"]

    n = st.columns(5)
    n[0].metric("Incidents raised", dsum["total"])
    n[1].metric("Acknowledged", by_d.get(authority.ACKNOWLEDGED, 0))
    n[2].metric("Not delivered", by_d.get(authority.FAILED, 0))
    n[3].metric("Pending", by_d.get(authority.PENDING, 0)
                + by_d.get(authority.RETRYING, 0))
    _on = dsum["authority_mode"] != "OFFLINE"
    n[4].markdown(
        f"<div style='font-size:0.66rem;font-weight:550;letter-spacing:0.055em;"
        f"text-transform:uppercase;color:#6D7C90'>Authority</div>"
        f"<div style='font-size:1.15rem;font-weight:600;margin-top:0.45rem;"
        f"color:{'#46a758' if _on else '#e5484d'}'>"
        f"{'🟢 Online' if _on else '🔴 Offline'}</div>", unsafe_allow_html=True)

    cc = st.columns([1, 1, 3])
    if cc[0].button("Retry failed", width="stretch",
                    disabled=not by_d.get(authority.FAILED)):
        out = authority.retry_failed()
        ok = sum(1 for o in out if o["status"] == authority.ACKNOWLEDGED)
        st.success(f"retried {len(out)} · {ok} acknowledged")
        st.rerun()
    if cc[1].button("Sweep live cameras", width="stretch"):
        # registry/live.py runs the chain over what the phones have actually
        # read: plate -> registry -> assessment -> referral. Deduplicated, so
        # pressing it twice does not manufacture a second incident.
        s = live.sweep_live(hours=float(hours))
        st.info(f"{s['reads']} live read(s) · {s['vehicles']} vehicle(s) · "
                f"{len(s['incidents'])} referred · "
                f"{s['not_found']} not on the registry · "
                f"{s['unknown']} unknown (registry unreachable)")
        st.rerun()

    if not inc:
        st.info("Nothing has been referred yet. Refer a finding from the "
                "Action queue, or sweep the live cameras above.")
    else:
        DCOL = {authority.ACKNOWLEDGED: "#46a758", authority.FAILED: "#e5484d",
                authority.RETRYING: "#f5a524", authority.PENDING: "#6D7C90"}
        for i in inc[:40]:
            with st.container(border=True):
                colour = DCOL.get(i["delivery_status"], "#6D7C90")
                sev = SEV.get(i["severity"], "#888")
                h = st.columns([5, 2])
                h[0].markdown(
                    f"<span style='font-family:monospace;color:#6D7C90;"
                    f"font-size:0.8rem'>{i['ref']}</span> "
                    f"<span style='color:{sev};font-weight:700;font-size:0.75rem'>"
                    f"{i['severity'].upper()}</span><br>"
                    f"<span style='font-size:1rem;font-weight:600'>"
                    f"{i['headline']}</span>", unsafe_allow_html=True)
                h[1].markdown(
                    f"<div style='text-align:right'>"
                    f"<span style='color:{colour};font-weight:650;"
                    f"font-size:0.82rem'>{i['delivery_status']}</span><br>"
                    f"<span style='font-size:0.72rem;color:#6D7C90'>"
                    f"{authority.authority_name(i['authority'])}</span></div>",
                    unsafe_allow_html=True)
                meta = [clock(i["ts"])]
                if i.get("camera_name"):
                    meta.append(i["camera_name"])
                if i.get("authority_ref"):
                    meta.append(f"their ref {i['authority_ref']}")
                meta.append(f"{i['attempts']} attempt(s)")
                meta.append(f"registry {i['registry_status']}")
                st.caption(" · ".join(meta))
                if i["delivery_status"] == authority.FAILED and i.get("last_error"):
                    st.markdown(
                        f"<div style='font-size:0.76rem;color:#ffb4b4;"
                        f"background:#2a1212;border-left:3px solid #e5484d;"
                        f"padding:5px 9px;border-radius:5px'>"
                        f"<b>Not delivered:</b> {i['last_error']} — the incident "
                        f"is unaffected and can be retried.</div>",
                        unsafe_allow_html=True)
                if i.get("caveat"):
                    st.caption(f"⚠ sent with the caveat: {i['caveat']}")
        if len(inc) > 40:
            st.caption(f"…and {len(inc) - 40} older incidents.")


with tab_desk:
    st.caption("The same findings, split by who would act on them. A console "
               "that sends everything to one queue is a console nobody reads.")
    auths = data.get("by_authority", {})
    if not auths:
        st.info("Nothing to route.")
    else:
        for auth, n in sorted(auths.items(), key=lambda kv: -kv[1]):
            name = sentinel.AUTHORITIES.get(auth, auth)
            with st.expander(f"{AUTH_ICON.get(auth, '•')} {name} — {n} item(s)",
                             expanded=(auth == "police")):
                mine = [f for f in findings if f["authority"] == auth]
                st.dataframe(pd.DataFrame([{
                    "severity": f["severity"], "plate": f["plate"],
                    "finding": f["headline"],
                    "last seen": clock(f["last_seen"]),
                    "junction": f.get("camera_name") or "—",
                } for f in mine]), hide_index=True, width="stretch",
                    height=min(340, 60 + 35 * len(mine)))


with tab_plate:
    st.caption("Everything the registry holds on one registration, and what the "
               "cameras have seen of it.")
    # A form, so the lookup fires on a click as well as on Enter. A bare
    # text_input commits only on Enter, which leaves a typed plate sitting
    # there doing nothing - and at a demo the reflex is to reach for a button.
    with st.form("plate_lookup", clear_on_submit=False):
        fc = st.columns([4, 1])
        q = fc[0].text_input(
            "Registration", placeholder="WB24AB1234", label_visibility="collapsed",
            help="Any plate. Misreads will correctly come back as not "
                 "registered — that is itself a signal.")
        fc[1].form_submit_button("Look up", width="stretch", type="primary")
    if q:
        b = brief(q.upper().replace(" ", ""), float(hours))
        r = b["registration"]
        if r.get("registry_status") == "UNKNOWN":
            # Never shown as "not registered". The registry could not be
            # reached, which says nothing whatever about this vehicle.
            st.warning(f"**The registry could not be reached.** Nothing is known "
                       f"about {b['plate']} either way — this is not a finding "
                       f"about the vehicle. {r.get('status_note') or ''}")
        elif not r["found"]:
            st.error(f"**{b['plate']} is not on the registry.** Either the plate is "
                     "fabricated, or the reader misread a real one.")
        else:
            v, g, val = r["vehicle"], r["registration"], r["validity"]
            badge = {"stolen": ("#e5484d", "REPORTED STOLEN"),
                     "scrapped": ("#e5484d", "SCRAPPED / DEREGISTERED"),
                     "blacklisted": ("#f76808", "BLACKLISTED"),
                     "suspended": ("#f76808", "SUSPENDED")}.get(r["status"])
            if badge:
                st.markdown(
                    f"<div style='background:{badge[0]}22;border:1px solid {badge[0]};"
                    f"border-radius:8px;padding:9px 13px;margin-bottom:10px'>"
                    f"<b style='color:{badge[0]}'>{badge[1]}</b> — "
                    f"{r['status_note'] or ''}</div>", unsafe_allow_html=True)

            a, bb, c = st.columns(3)
            with a:
                st.markdown("**Vehicle**")
                st.write({"make": v["make"], "model": v["model"],
                          "class": v["class"], "colour": v["colour"],
                          "fuel": v["fuel"], "year": v["manufactured"],
                          "engine": v["engine_no"], "chassis": v["chassis_no"]})
            with bb:
                st.markdown("**Registered**")
                st.write({"RTO": g["rto_office"], "code": g["rto_code"],
                          "state": g["state"],
                          "registered": clock(g["registered_on"])})
            with c:
                st.markdown("**Validity**")
                now = time.time()
                rows = []
                for key, label in (("insurance_to", "Insurance"), ("puc_to", "PUC"),
                                   ("fitness_to", "Fitness"), ("permit_to", "Permit"),
                                   ("tax_to", "Road tax")):
                    if val.get(key) is None:
                        continue
                    ok = val[key] >= now
                    rows.append({"document": label, "valid to": clock(val[key]),
                                 "state": "valid" if ok else "EXPIRED"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            m = b["movement"]
            st.markdown("**Seen by the network**")
            if not m:
                st.caption("No sighting of this registration in the window.")
            else:
                mc = st.columns(4)
                mc[0].metric("Sightings", m.get("sightings", 0))
                mc[1].metric("Junctions", m.get("cameras", 0))
                mc[2].metric("Last seen", clock(m.get("last_ts")))
                mc[3].metric("Classified as",
                             max(m["classes"], key=m["classes"].get)
                             if m.get("classes") else "—")

            if r["pending_cases"]:
                st.markdown(f"**Pending cases** — Rs {r['outstanding_amount']:,.0f} "
                            f"outstanding")
                st.dataframe(pd.DataFrame([{
                    "kind": c["kind"], "section": c["section"],
                    "description": c["description"],
                    "amount": c["amount"], "raised": clock(c["raised_on"]),
                    "station": c["station"], "reference": c["reference"],
                } for c in r["pending_cases"]]), hide_index=True, width="stretch")

        if b["findings"]:
            st.markdown("**What this warrants**")
            for i, f in enumerate(b["findings"]):
                _finding_card(f, f"p{i}")
        elif r["found"]:
            st.success("Nothing outstanding against this registration.")

        who = store.who_looked_up(b["plate"], limit=8)
        with st.expander(f"Who has looked this registration up ({len(who)})"):
            st.caption("Every lookup is recorded. A registry that cannot say who "
                       "asked about a private citizen is the kind core/privacy.py "
                       "argues against.")
            if who:
                st.dataframe(pd.DataFrame([{
                    "when": clock(w["ts"]), "actor": w["actor"],
                    "purpose": w["purpose"] or "—",
                    "found": bool(w["found"]),
                } for w in who]), hide_index=True, width="stretch")


with tab_registry:
    st.caption("The mock registry itself — what it holds, and what it refuses "
               "to claim.")
    c = st.columns(4)
    c[0].metric("Vehicles", f"{rs['vehicles']:,}")
    c[1].metric("Open cases", f"{rs['cases_open']:,}")
    c[2].metric("Outstanding", f"Rs {rs['outstanding']:,.0f}")
    c[3].metric("Lookups recorded", f"{rs['lookups']:,}")

    c = st.columns(3)
    c[0].metric("Insurance expired", f"{rs['insurance_expired']:,}")
    c[1].metric("PUC expired", f"{rs['puc_expired']:,}")
    c[2].metric("Fitness expired", f"{rs['fitness_expired']:,}")

    if rs["by_status"]:
        st.markdown("**Registry status mix**")
        st.dataframe(pd.DataFrame([{"status": k, "vehicles": v}
                                   for k, v in sorted(rs["by_status"].items(),
                                                      key=lambda kv: -kv[1])]),
                     hide_index=True, width="stretch")

    st.markdown(
        "<div style='background:#161b22;border:1px solid #2a323d;border-radius:10px;"
        "padding:12px 15px;margin-top:10px'>"
        "<b>What this is not.</b> A synthetic registry, generated from a hash of "
        "each plate so the same registration always returns the same record. No "
        "real vehicle, owner, insurer or criminal case is described by it, and no "
        "connection to VAHAN or any government system exists. Its value is the "
        "<i>shape</i>: <code>registry/store.py::lookup()</code> returns what a real "
        "integration would, so swapping in a real source changes nothing above it. "
        "Vehicle class here is the simulator's label rather than a classifier "
        "reading the image, so the class-mismatch finding is only as good as that "
        "label — which is why it carries that caveat on every card."
        "</div>", unsafe_allow_html=True)


def main() -> int:
    """`python -m registry.console` -> launch streamlit on our own port."""
    import subprocess
    port = str(settings.CONSOLE_PORT)
    return subprocess.call([sys.executable, "-m", "streamlit", "run", __file__,
                            "--server.port", port, "--server.headless", "true"])


if __name__ == "__main__" and not st.runtime.exists():
    raise SystemExit(main())
