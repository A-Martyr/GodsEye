"""The one place the control room touches the demo vehicle registry.

Every other dashboard module is unchanged and unaware. If `registry/` is
deleted, this module reports `available() == False` and every panel below
renders nothing - the dashboard loses a card and keeps working. That is the
whole reason the coupling is funnelled through one file: the registry was
built to be undoable, and one soft import is a thing you can actually delete.

    dashboard/app.py  ->  dashboard/registry_link.py  ->  registry/

THE IMPORT IS SOFT ON PURPOSE and every call is wrapped. A registry that is
missing, half-seeded, mid-migration or throwing must not be able to take the
control room down with it. The platform's alert queue, trajectory engine and
camera feeds have nothing to do with vehicle registration, and none of them
should stop working because a synthetic lookup table had a bad day.

NOT VAHAN. Everything surfaced here is invented by `registry/store.py` from a
hash of the plate. The panels say so on their own face, every time, because a
card that looks like an official record is exactly the thing that gets believed
without being checked.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st

# --- the soft import ----------------------------------------------------
try:
    from registry import authority as _authority
    from registry import sentinel as _sentinel
    from registry import settings as _settings
    from registry import store as _store
    _IMPORT_ERROR = ""
except Exception as _exc:                                    # pragma: no cover
    _authority = _sentinel = _settings = _store = None
    _IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


SEV = {"critical": "#e5484d", "high": "#f76808", "medium": "#f5a524",
       "low": "#8b96a5"}

# Registry standings that change how an operator should read an alert about
# the vehicle. `active` is deliberately absent - a normal registration is not
# news, and badging every row teaches people to stop seeing the badge.
FLAGGED = {
    "stolen": ("#e5484d", "STOLEN"),
    "scrapped": ("#e5484d", "SCRAPPED"),
    "blacklisted": ("#f76808", "BLACKLISTED"),
    "suspended": ("#f76808", "SUSPENDED"),
}


def available() -> bool:
    """Is the registry importable and seeded? Everything else checks this."""
    if _store is None:
        return False
    try:
        return _settings.DB_PATH.exists()
    except Exception:
        return False


def why_unavailable() -> str:
    if _IMPORT_ERROR:
        return f"the registry package could not be imported ({_IMPORT_ERROR})"
    if _store is None:
        return "the registry package is not installed"
    try:
        if not _settings.DB_PATH.exists():
            return ("the registry has not been built yet — run "
                    "`registry.bat --seed`")
    except Exception:
        pass
    return ""


# --- reads --------------------------------------------------------------
# Short TTLs. A registration does not change minute to minute, and the alert
# queue below looks up forty plates on every rerun - without this the control
# room would pay for that on every 3-second fragment tick.
@st.cache_data(ttl=90, show_spinner=False)
def lookup(plate: str) -> dict | None:
    """One registration, or None if the registry is not usable at all.

    A dict with registry_status == "UNKNOWN" is NOT None: it means the lookup
    happened and failed, which callers must render differently from "we never
    asked". Collapsing the two is the bug this package exists to avoid.
    """
    if not available():
        return None
    try:
        return _store.cached_lookup(
            plate, actor="control room",
            purpose="operator opened this registration").as_dict()
    except Exception:
        return None


@st.cache_data(ttl=90, show_spinner=False)
def lookup_many(plates: tuple) -> dict:
    """Many registrations on one connection. {plate: dict}.

    Takes a tuple because st.cache_data hashes its arguments and a list is
    unhashable - passing a list here fails at the decorator, not in the body.
    """
    if not available() or not plates:
        return {}
    try:
        regs = _store.bulk_lookup(plates, actor="control room",
                                  purpose="alert queue enrichment")
        return {p: r.as_dict() for p, r in regs.items()}
    except Exception:
        return {}


@st.cache_data(ttl=90, show_spinner=False)
def findings(plate: str, hours: float = 24.0) -> list:
    """What the registry + the cameras together imply about this plate.

    Empty during a registry outage, by design: `sentinel.assess()` refuses to
    conclude anything from an UNKNOWN lookup.
    """
    if not available():
        return []
    try:
        return _sentinel.brief(plate, hours=hours).get("findings", [])
    except Exception:
        return []


def incidents_for(plate: str) -> list:
    """Referrals already raised for this registration. Never cached.

    Deliberately uncached: this changes the moment somebody presses Refer, and
    a stale answer here would offer to refer something twice.
    """
    if not available():
        return []
    try:
        return _store.incidents_for(plate)
    except Exception:
        return []


def refer(finding: dict, registry_status: str = "FOUND") -> dict | None:
    """Send one finding to the authority that owns it. Returns the outcome."""
    if not available():
        return None
    try:
        return _authority.refer(dict(finding), registry_status=registry_status)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def authority_name(key: str) -> str:
    try:
        return _authority.authority_name(key)
    except Exception:
        return key


def clear() -> None:
    """Drop the caches above. Called by the dashboard's refresh controls."""
    for fn in (lookup, lookup_many, findings):
        try:
            fn.clear()
        except Exception:
            pass
    try:
        _store.cache_clear()
    except Exception:
        pass


# --- badges -------------------------------------------------------------
def badge_html(reg: dict | None) -> str:
    """A compact registry standing, for a row in a list. "" when unremarkable.

    Returns nothing for an active registration and nothing for an outage. Both
    are correct: a clean vehicle is not news, and an outage is not a finding
    about the vehicle, so neither belongs as a red chip beside a plate.
    """
    if not reg or reg.get("registry_status") != "FOUND":
        return ""
    flag = FLAGGED.get(reg.get("status"))
    if not flag:
        return ""
    colour, label = flag
    return (f"<span style='background:{colour}22;color:{colour};"
            f"border:1px solid {colour}66;border-radius:999px;padding:1px 7px;"
            f"font-size:0.66rem;font-weight:700;margin-left:7px'>"
            f"REGISTRY: {label}</span>")


def papers_html(reg: dict | None) -> str:
    """Expired documents, as a quieter chip. Separate from standing on purpose.

    An expired PUC and a stolen vehicle are both "flags" and belong nowhere
    near each other in an operator's eye: one is paperwork, the other is a
    crime. Different colour, different chip, never merged into a count.
    """
    if not reg or reg.get("registry_status") != "FOUND":
        return ""
    exp = reg.get("expired") or []
    if not exp:
        return ""
    return (f"<span style='background:#2a1f10;color:#f5a524;"
            f"border:1px solid #f5a52455;border-radius:999px;padding:1px 7px;"
            f"font-size:0.66rem;font-weight:600;margin-left:7px'>"
            f"{', '.join(exp)} expired</span>")


# --- panels -------------------------------------------------------------
def _spec(pairs) -> str:
    """A label/value list. Replaces st.write(dict), which renders a JSON blob.

    The dict form was legible in a terminal and wrong here: quoted keys, braces
    and syntax colouring, in a card an operator reads at a glance. Nothing in a
    vehicle record needs to look like code.
    """
    return "".join(
        f"<div style='display:flex;justify-content:space-between;gap:10px;"
        f"font-size:0.82rem;padding:1px 0'>"
        f"<span style='color:#93a1b3'>{label}</span>"
        f"<span style='font-weight:600;text-align:right'>{value}</span></div>"
        for label, value in pairs if value not in (None, ""))


def _disclaimer() -> None:
    st.caption("Synthetic registry — every record is generated by "
               "`registry/store.py` from a hash of the plate. Not VAHAN, not a "
               "government system, no real vehicle or case.")


def render_card(plate: str, hours: float = 24.0, key: str = "reg") -> None:
    """The registry's answer for one registration, under its trajectory.

    Placed where it is because of what an operator has just done: they have
    found the vehicle and reconstructed where it went. The next question is
    always who it is and whether anything is outstanding against it - and
    until now that question had to be taken to a different application.
    """
    if not available():
        why = why_unavailable()
        if why and "not installed" not in why:
            st.caption(f"Vehicle registry unavailable — {why}")
        return

    reg = lookup(plate)
    if reg is None:
        st.caption("Vehicle registry unavailable.")
        return

    status = reg.get("registry_status")

    if status == "UNKNOWN":
        # Never rendered as "not registered". The registry could not be
        # reached, which says precisely nothing about this vehicle - and an
        # operator who reads an outage as a fabricated plate will act on it.
        st.warning("**The vehicle registry could not be reached.** Nothing is "
                   "known about this registration either way. This is not a "
                   "finding about the vehicle, and the movement above is "
                   "unaffected.")
        return

    if status == "NOT_FOUND":
        st.error(f"**{plate} is not on the vehicle registry.** Either the plate "
                 f"is fabricated, or the reader misread a real one — at this "
                 f"engine's accuracy the second is far more common, so this is "
                 f"a lead, not a conclusion.")
        _disclaimer()
        return

    v = reg.get("vehicle", {})
    g = reg.get("registration", {})
    val = reg.get("validity", {})

    flag = FLAGGED.get(reg.get("status"))
    if flag:
        colour, label = flag
        st.markdown(
            f"<div style='background:{colour}22;border:1px solid {colour};"
            f"border-radius:8px;padding:9px 13px;margin-bottom:10px'>"
            f"<b style='color:{colour}'>REGISTRY: {label}</b> — "
            f"{reg.get('status_note') or ''}</div>", unsafe_allow_html=True)

    # Observed and registered side by side, labelled as different kinds of
    # claim. The camera's reading is seconds old; the registry's record is
    # years old and was entered by somebody else. An operator comparing them
    # needs to know which is which before deciding a mismatch means anything.
    a, b, c = st.columns([1.15, 1, 1.1])
    with a:
        st.markdown("**Registered vehicle**")
        st.markdown(_spec([
            ("Make", v.get("make")), ("Model", v.get("model")),
            ("Class", str(v.get("class") or "").replace("_", " ")),
            ("Colour", v.get("colour")), ("Fuel", v.get("fuel")),
            ("Year", v.get("manufactured")),
        ]), unsafe_allow_html=True)
    with b:
        st.markdown("**Registered at**")
        st.markdown(_spec([
            ("RTO", g.get("rto_office")), ("Code", g.get("rto_code")),
            ("State", g.get("state")),
        ]), unsafe_allow_html=True)
    with c:
        st.markdown("**Papers**")
        now = time.time()
        papers = [(label, (val.get(k) or 0) >= now)
                  for k, label in (("insurance_to", "Insurance"),
                                   ("puc_to", "PUC"), ("fitness_to", "Fitness"),
                                   ("permit_to", "Permit"),
                                   ("tax_to", "Road tax"))
                  if val.get(k) is not None]
        st.markdown("".join(
            f"<div style='display:flex;justify-content:space-between;gap:10px;"
            f"font-size:0.82rem;padding:1px 0'><span style='color:#93a1b3'>"
            f"{label}</span><span style='color:"
            f"{'#7ee2a8' if ok else '#f5a524'};font-weight:600'>"
            f"{'valid' if ok else 'EXPIRED'}</span></div>"
            for label, ok in papers) or "<span style='color:#6D7C90'>—</span>",
            unsafe_allow_html=True)

    if reg.get("pending_cases"):
        with st.expander(f"Pending cases ({len(reg['pending_cases'])}) — "
                         f"Rs {reg.get('outstanding_amount', 0):,.0f} outstanding"):
            st.dataframe(pd.DataFrame([{
                "kind": x.get("kind"), "section": x.get("section"),
                "description": x.get("description"), "amount": x.get("amount"),
                "station": x.get("station"), "reference": x.get("reference"),
            } for x in reg["pending_cases"]]), hide_index=True, width="stretch")

    render_findings(plate, hours=hours, key=key)
    _disclaimer()


def render_findings(plate: str, hours: float = 24.0, key: str = "reg") -> None:
    """What the registry and the cameras together warrant, and the Refer button.

    Referral is a button and not a consequence. Everything on this page is
    inference from a camera read; a person decides it is worth an authority's
    time. The card shows the caveat with the claim for the same reason the
    referral payload carries it - whoever acts on this has to see what could
    explain it other than guilt.
    """
    found = findings(plate, hours=hours)
    if not found:
        return
    st.markdown("**What the registry warrants**")
    for i, f in enumerate(found):
        colour = SEV.get(f.get("severity"), "#888")
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{colour};font-weight:700;font-size:0.78rem'>"
                f"{f.get('severity', '').upper()}</span>"
                f"<span style='color:#6D7C90;font-size:0.72rem;margin-left:8px;"
                f"letter-spacing:0.06em;text-transform:uppercase'>"
                f"{str(f.get('kind', '')).replace('_', ' ')}</span><br>"
                f"<span style='font-size:1rem;font-weight:600'>"
                f"{f.get('headline', '')}</span>", unsafe_allow_html=True)
            if f.get("detail"):
                st.caption(f["detail"])
            if f.get("caveat"):
                st.markdown(
                    f"<div style='font-size:0.78rem;color:#ffd79a;"
                    f"background:#2a1f10;border-left:3px solid #f5a524;"
                    f"padding:6px 9px;border-radius:5px;margin-top:6px'>"
                    f"<b>Before acting:</b> {f['caveat']}</div>",
                    unsafe_allow_html=True)

            prior = [x for x in incidents_for(plate) if x["kind"] == f.get("kind")]
            if prior:
                p = prior[0]
                st.caption(f"referred as **{p['ref']}** to "
                           f"{authority_name(p['authority'])} · "
                           f"{p['delivery_status']}"
                           + (f" · their ref {p['authority_ref']}"
                              if p["authority_ref"] else ""))
            elif st.button(f"Refer to {f.get('authority_name', 'the authority')}",
                           key=f"{key}_refer_{i}", width="stretch"):
                res = refer(f)
                if not res or res.get("error"):
                    st.error(f"Could not refer this. "
                             f"{res.get('error') if res else ''}")
                else:
                    d = res["delivery"]
                    if d["status"] == "ACKNOWLEDGED":
                        st.success(f"{res['incident']['ref']} acknowledged by "
                                   f"{authority_name(res['incident']['authority'])}"
                                   f" as {d['authority_ref']}")
                    else:
                        st.warning(f"{res['incident']['ref']} is recorded, but "
                                   f"could not be delivered after "
                                   f"{d.get('attempts')} attempts. It is on the "
                                   f"Security Console to retry — the finding is "
                                   f"not lost.")
                st.rerun()
