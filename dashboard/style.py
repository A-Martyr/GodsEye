"""The small amount of styling the theme file cannot express.

Almost everything about this dashboard's appearance is declared in
.streamlit/config.toml, where Streamlit applies it before first paint and no
internal class name can be renamed out from under us. What is left here is the
handful of rules that have no config equivalent.

Kept deliberately short. Every selector below is a bet that a Streamlit
internal keeps its name, and each one is a rule that silently stops applying
when that bet loses. A stylesheet that reaches into a framework's internals is
a maintenance debt, so it is worth carrying only where the payoff is real - and
each block here says what its payoff is. Nothing here changes layout flow, so
if every rule failed at once the dashboard would look plainer and work
identically.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ---------------------------------------------------------------------
   Tabular figures. The single highest-value rule in this file.

   Proportional digits have different widths - a 1 is narrower than a 0 - so
   every number on this page changes width as it changes value. The live
   metrics refresh every five seconds and the sidebar counters every three,
   which means the whole row twitches sideways on each tick, and a reader
   watching for a change sees motion that carries no information.

   tabular-nums fixes each digit to one advance width. Numbers then update in
   place, columns of figures line up down the page, and the layout stops
   settling. It is also the cheapest thing that separates a credible
   instrument from a spreadsheet - and it costs one font feature flag.
   --------------------------------------------------------------------- */
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
.stDataFrame, .stTable, code, pre {
    font-variant-numeric: tabular-nums lining;
    font-feature-settings: "tnum" 1, "lnum" 1;
}

/* ---------------------------------------------------------------------
   Metric labels: the quiet half of the pair.

   A metric is one loud number and one soft description of it, and Streamlit
   ships them at nearly the same visual weight, which forces the eye to read
   the label before it can find the value. Dropping the label to small,
   letterspaced, dimmed caps inverts that: the row scans as six numbers with
   annotations, which is how it is actually read.

   normal white-space because the stock rule truncates - "Mean read
   confidence" was rendering as "Mean read confide...", and a metric whose
   label is cut off is a metric nobody can identify.
   --------------------------------------------------------------------- */
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.055em;
    font-size: 0.68rem !important;
    font-weight: 550;
    opacity: 0.52;
}
[data-testid="stMetricLabel"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
[data-testid="stMetricValue"] { line-height: 1.15; }

/* ---------------------------------------------------------------------
   Vertical rhythm. Streamlit reserves a large header strip and generous gaps
   between every element, which suits a document and wastes a control surface:
   the metric row and the map should be visible together without scrolling.
   Tightened, not crushed - dark interfaces need a little more air than light
   ones, not less, or the text mats together.
   --------------------------------------------------------------------- */
.block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 100%; }
[data-testid="stSidebarUserContent"] { padding-top: 1.6rem; }

/* ---------------------------------------------------------------------
   Tabs read as navigation, not as text. The stock bar is regular-weight
   labels with a coloured underline on the active one; at 15px on a dark
   ground that underline is doing all the work alone. Adding weight to the
   selected label and dimming the rest means the current section is legible
   peripherally, which is what a person glancing back at a screen needs.
   --------------------------------------------------------------------- */
[data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid #1A2331; }
[data-baseweb="tab"] { font-weight: 500; letter-spacing: 0.005em; }
[data-baseweb="tab"][aria-selected="true"] { font-weight: 600; }

/* ---------------------------------------------------------------------
   Containers. st.container(border=True) is how the alert queue draws its
   rows; the stock border is bright enough that forty of them read as a grid
   of boxes rather than a list of alerts. Sunk to just above the background so
   the rows separate without the lines becoming the pattern.
   --------------------------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
    border-color: #18202C;
}

/* Dataframes: remove the double line where a bordered table sits inside a
   bordered container, which reads as a rendering fault rather than a style. */
[data-testid="stDataFrame"] { border-radius: 6px; }

/* Buttons sized for a control surface: the stock padding is generous enough
   that a row of three actions wraps on a narrow column. */
.stButton > button { font-weight: 550; letter-spacing: 0.005em; }

/* ---------------------------------------------------------------------
   The framework's own chrome.

   A "Deploy" button in the corner of a traffic control room says, louder than
   anything else on screen, that this is a Streamlit app someone is running
   rather than a system someone operates. The running-man indicator is the
   same tell: it announces the tool, not the work.

   The hamburger menu STAYS. It carries Rerun, Clear cache and Settings, which
   are the operator's way out of a wedged session, and hiding a product's only
   escape hatch to make it look tidier is a bad trade. This removes branding,
   not function.
   --------------------------------------------------------------------- */
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
footer { visibility: hidden; height: 0; }

/* ---------------------------------------------------------------------
   Expanders. The sidebar's "Live feed & incidents" and "Judge link" sit in
   them, and the stock treatment gives each a full-strength border and a bold
   label, so two collapsed panels read as heavier than the live counters above
   them. They are drawers, and should look like it until opened.
   --------------------------------------------------------------------- */
[data-testid="stExpander"] details { border-color: #18202C; }
[data-testid="stExpander"] summary { font-weight: 550; font-size: 0.86rem; }

/* Captions carry most of this dashboard's honesty - the evidence caveats, the
   "unverified" notes, the read-back of what a search understood. Stock grey on
   near-black loses them; lifted just enough to be read without competing. */
[data-testid="stCaptionContainer"], .stCaption { color: #7F8DA0; }
</style>
"""


def kpi(container, label: str, value: str, unit: str = "", note: str = "",
        compact: bool = False) -> None:
    """One headline figure, with its unit set below the value's weight.

    st.metric renders label, value and delta at close to the same emphasis, and
    puts the unit inside the value - so "33 km/h" is one 2rem string and the
    eye spends the same effort on "km/h" as on "33". Separating them is the
    detail the research on numeric UI keeps returning to: the value is the
    information, the unit is a label for it, and setting them at one weight is
    what makes a dashboard read as a spreadsheet.

    Pure markup over values the caller already has. It issues no query and
    computes nothing, which is the constraint the whole visual pass runs
    under - the per-rerun server cost of this page must not move.
    """
    # Emitted as ONE line with no leading whitespace. st.markdown runs this
    # through a markdown parser first, and that parser treats an indented line
    # as a code block and inserts paragraph breaks at blank ones - which splits
    # the element and leaves the trailing "</div>" on the page as literal text.
    # Every helper in this module builds a single-line string for that reason.
    lab = (f'<div style="font-size:0.66rem;font-weight:550;letter-spacing:0.055em;'
           f'text-transform:uppercase;color:#6D7C90;margin-bottom:0.3rem">{label}</div>')
    un = (f'<span style="font-size:0.9rem;font-weight:500;color:#6D7C90;'
          f'margin-left:0.22rem;letter-spacing:0">{unit}</span>') if unit else ""
    # white-space:nowrap is not cosmetic. In a narrow column - the sidebar's
    # three counters are a third of a rail each - "29,406" was breaking after
    # the comma and rendering as "29,40" above "6", which reads as a different
    # number. A figure that wraps mid-digits is worse than one that is clipped,
    # because the clipped one looks wrong and the wrapped one looks fine.
    #
    # compact drops the size for those narrow places rather than letting the
    # nowrap overflow its column.
    size = "1.25rem" if compact else "1.85rem"
    val = (f'<div style="font-variant-numeric:tabular-nums lining;font-size:{size};'
           f'font-weight:550;color:#EDF2F8;letter-spacing:-0.015em;'
           f'white-space:nowrap">{value}{un}</div>')
    hint = (f'<div style="font-size:0.7rem;color:#5F6E82;margin-top:0.15rem">{note}</div>'
            if note else "")
    container.markdown(
        f'<div style="line-height:1.2;padding:0.1rem 0 0.35rem 0">{lab}{val}{hint}</div>',
        unsafe_allow_html=True)


def callout(text: str, tone: str = "info") -> None:
    """A single line of context, weighted as a line rather than a block.

    st.info paints a filled panel the width of the page. Used for a standing
    observation - the worst corridor is always present, it is not news - that
    panel competes with the alert banner directly above it, and the operator
    learns to filter the colour that also carries clone alerts.

    A rule and dimmed text says the same thing without spending attention that
    the platform needs to keep in reserve for something genuinely wrong.
    """
    edge = {"info": "#3D7EFF", "warn": "#F5A623", "bad": "#E5484D"}.get(tone, "#3D7EFF")
    st.markdown(
        f'<div style="border-left:2px solid {edge};padding:0.15rem 0 0.15rem 0.7rem;'
        f'margin:0.15rem 0 0.9rem 0;color:#93A1B4;font-size:0.84rem;'
        f'line-height:1.5">{text}</div>',
        unsafe_allow_html=True)


def section(title: str, eyebrow: str = "") -> None:
    """A section heading with an optional eyebrow above it.

    st.subheader gives every section the same weight, so a page reads as a
    stack of equal things. An eyebrow - small, caps, dim - says which part of
    the platform this is, and lets the heading itself stay quiet.
    """
    brow = (f'<div style="font-size:0.63rem;font-weight:600;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:#4E5E73;margin-bottom:0.2rem">'
            f'{eyebrow}</div>') if eyebrow else ""
    st.markdown(
        f'<div style="margin:0.35rem 0 0.55rem 0">{brow}'
        f'<div style="font-size:1.12rem;font-weight:600;color:#E7EDF5;'
        f'letter-spacing:-0.01em">{title}</div></div>',
        unsafe_allow_html=True)


def apply() -> None:
    """Inject the stylesheet. Call once, immediately after set_page_config.

    Called on every rerun rather than guarded behind a session flag on
    purpose: Streamlit rebuilds the DOM each run, so a stylesheet written only
    on the first run would vanish on the second. The cost is one markdown
    element per rerun, which is not measurable next to a single map redraw.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
