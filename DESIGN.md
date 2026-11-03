---
name: GodsEye
description: A dim control room where the city is the screen, one search bar answers any plate, and every read shows where it came from.
colors:
  ground: "#0c1015"
  surface: "#151a21"
  surface-2: "#1c222b"
  surface-3: "#242b35"
  line: "#252d37"
  line-2: "#333c48"
  text: "#e7ecf1"
  text-2: "#a7b2bf"
  text-3: "#8391a0"
  blue: "#4d84ff"
  blue-press: "#6b98ff"
  blue-soft: "rgba(77, 132, 255, .15)"
  blue-line: "rgba(77, 132, 255, .45)"
  blue-ink: "#a9c3ff"
  real: "#3cc581"
  real-soft: "rgba(60, 197, 129, .14)"
  real-ink: "#9ae6be"
  sim: "#8a97a6"
  sim-hatch-a: "#222a33"
  sim-hatch-b: "#1a2028"
  provline-hatch-a: "#1b2129"
  provline-hatch-b: "#181d24"
  provline-ink: "#c3ccd6"
  infer: "#a385ff"
  infer-soft: "rgba(163, 133, 255, .15)"
  infer-ink: "#cdbcff"
  t0: "#2fae6a"
  t1: "#e2b007"
  t2: "#ee7a2b"
  t3: "#e5484d"
  quiet-road: "#56616e"
  danger: "#f0524f"
  danger-soft: "rgba(240, 82, 79, .14)"
  danger-ink: "#ffb3b1"
  warn: "#e8a33a"
  warn-soft: "rgba(232, 163, 58, .14)"
  warn-ink: "#f3c98a"
  warn-ink-2: "#f6d6a6"
  on-blue: "#ffffff"
typography:
  display:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "34px"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "0.02em"
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  headline-figure:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "28px"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  stat:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "18px"
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: "-0.01em"
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  brand:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    letterSpacing: "-0.02em"
  search:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "15px"
    fontWeight: 500
    letterSpacing: "0.01em"
  title:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "15px"
    fontWeight: 650
    letterSpacing: "-0.01em"
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  body:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  plate:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "13.5px"
    fontWeight: 650
    letterSpacing: "0.06em"
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  body-small:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  label:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.4
    fontFeature: "\"tnum\" 1, \"cv11\" 1"
  badge:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "11.5px"
    fontWeight: 600
    lineHeight: 1
  micro:
    fontFamily: "Geist, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "10.5px"
    fontWeight: 500
    lineHeight: 1.3
  code:
    fontFamily: "ui-monospace, Cascadia Mono, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
rounded:
  tag: "5px"
  plate: "6px"
  small: "7px"
  ctl: "10px"
  icon: "11px"
  well: "12px"
  card: "14px"
  sheet: "18px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  gap: "12px"
  top-h: "44px"
  side-w: "392px"
  alerts-w: "356px"
  drawer-w: "420px"
components:
  button-primary:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.on-blue}"
    rounded: "{rounded.ctl}"
    height: "38px"
    padding: "0 14px"
  button-primary-hover:
    backgroundColor: "{colors.blue-press}"
  button-secondary:
    backgroundColor: "{colors.blue-soft}"
    textColor: "{colors.blue-ink}"
    rounded: "{rounded.ctl}"
    height: "38px"
    padding: "0 14px"
  button-quiet:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
    rounded: "{rounded.ctl}"
    height: "38px"
    padding: "0 14px"
  button-quiet-hover:
    backgroundColor: "{colors.surface-3}"
  button-danger:
    backgroundColor: "{colors.danger-soft}"
    textColor: "{colors.danger-ink}"
    rounded: "{rounded.ctl}"
    height: "38px"
    padding: "0 14px"
  button-ghost:
    textColor: "{colors.text-2}"
    rounded: "{rounded.ctl}"
    height: "38px"
    padding: "0 10px"
  button-ghost-hover:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
  chip-status:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-2}"
    typography: "{typography.body-small}"
    rounded: "{rounded.pill}"
    height: "{spacing.top-h}"
    padding: "0 14px"
  chip-button-open:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.on-blue}"
    rounded: "{rounded.pill}"
    height: "{spacing.top-h}"
  input-search:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    typography: "{typography.search}"
    rounded: "{rounded.card}"
    height: "{spacing.top-h}"
    padding: "0 8px 0 15px"
  seg-track:
    backgroundColor: "{colors.surface-2}"
    rounded: "{rounded.well}"
    padding: "3px"
  seg-option-on:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.text}"
    rounded: "9px"
    height: "34px"
  badge-real:
    backgroundColor: "{colors.real-soft}"
    textColor: "{colors.real-ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.pill}"
    height: "22px"
    padding: "0 9px"
  badge-sim:
    backgroundColor: "{colors.sim-hatch-a}"
    textColor: "{colors.text-2}"
    typography: "{typography.badge}"
    rounded: "{rounded.pill}"
    height: "22px"
    padding: "0 9px"
  badge-infer:
    backgroundColor: "{colors.infer-soft}"
    textColor: "{colors.infer-ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.pill}"
    height: "22px"
    padding: "0 9px"
  badge-crit:
    backgroundColor: "{colors.danger-soft}"
    textColor: "{colors.danger-ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.pill}"
    height: "22px"
    padding: "0 9px"
  read-label-ocr:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.text}"
    rounded: "{rounded.tag}"
    padding: "0 6px"
  read-label-modelled:
    textColor: "{colors.text-2}"
    rounded: "{rounded.tag}"
    padding: "0 6px"
  read-label-real:
    backgroundColor: "{colors.real-soft}"
    textColor: "{colors.real-ink}"
    rounded: "{rounded.tag}"
    padding: "0 6px"
  read-label-refused:
    backgroundColor: "{colors.warn-soft}"
    textColor: "{colors.warn-ink}"
    rounded: "{rounded.tag}"
    padding: "0 6px"
  plate:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.text}"
    typography: "{typography.plate}"
    rounded: "{rounded.plate}"
    height: "26px"
    padding: "0 8px"
  sheet:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.sheet}"
    padding: "8px 8px 16px"
  alert-card:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-2}"
    rounded: "{rounded.card}"
    padding: "11px 12px"
  alert-card-critical:
    backgroundColor: "{colors.danger-soft}"
    textColor: "{colors.danger-ink}"
    rounded: "{rounded.card}"
    padding: "11px 12px"
  stat-tile:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
    typography: "{typography.stat}"
    rounded: "{rounded.ctl}"
    padding: "10px 11px"
  provenance-line:
    backgroundColor: "{colors.provline-hatch-a}"
    textColor: "{colors.provline-ink}"
    rounded: "{rounded.well}"
    padding: "10px 12px"
  toast:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.text}"
    typography: "{typography.body-small}"
    rounded: "{rounded.well}"
    padding: "10px 14px"
---

# Design System: GodsEye

## Overview

**Creative North Star: "The City Is The Screen"**

GodsEye is Triffie's design language carried into a dim control room. The map is the whole viewport: a dark, desaturated Kolkata on which only two things carry colour, road load and the vehicle being tracked. Everything else floats above it as quiet charcoal sheets with soft ink shadows. One search bar sits at the top and answers any plate, including a misread or partial one. The theme is dark because operators watch it for a whole shift in a dimmed room, not because "control room" means dark. The system rejects the category default it replaced: a dark grid of glowing KPI tiles and tabs, which the owner said looked AI-generated.

Provenance is part of the visual system, not a footnote. A simulated feed is hatched slate, a real photograph is green, a registration inferred by repair is violet, and a refused capture is amber, shown as the engine declining rather than failing. Each read also carries a label saying whether its text was decoded by the OCR engine or modelled. These differences survive a washed-out projector and colour blindness because they differ in pattern and stroke as well as hue. The density is that of a working tool: 14px body, tabular figures everywhere, 12px gutters, and nothing on screen that is not information.

**Key Characteristics:**
- Full-bleed, inverted grey basemap. On the map, colour means road load, the tracked vehicle, or an alert, and nothing else.
- Charcoal sheets and pill chips floating on diffuse black shadows. No glows, no hard edges.
- One route blue for the tracked vehicle and the primary action.
- Geist throughout with tabular figures. The number plate is the headline.
- Four provenance looks (hatched simulated, green real, violet inferred, amber refused) plus OCR versus modelled read labels.
- Drawn SVG icons in a single 1.75px round stroke.
- The tracked route drawing itself sighting by sighting is the one authored motion.

## Colors

A near-black blue-grey room with a few colours, each of which has exactly one meaning.

### Primary
- **Route Blue** (blue): the tracked vehicle's route, its sighting dots and its stop tags on the map; the primary button; the brand mark; the open state of the Insights chip and checked toggles; focus rings and the text caret. **Lifted Blue** (blue-press) is hover, lighter rather than darker because the ground is dark. **Blue Wash** (blue-soft) is the secondary button and the near-match note, set in **Blue Ink** (blue-ink). **Blue Hairline** (blue-line) is the secondary button's hover border and the resting bars of spark and hourly histograms, whose current or busiest bar takes full Route Blue.

### Secondary
- **Inferred Violet** (infer) on **Violet Wash** (infer-soft) with **Violet Ink** (infer-ink): registrations inferred by repair from neighbouring cameras, never read. It appears on the "inferred" badge (a hollow ring dot), inferred rows in the feed and vehicle sheet, and inferred stop tags on the map. Never use it for anything else.

### Tertiary
- **Traffic Ramp** (t0 free, t1 moderate, t2 heavy, t3 severe): road colours and junction rings, nothing else. JS reads the same four values from the CSS custom properties, so map and key never drift.
- **Quiet Road** (quiet-road): a road with no reads in the current window. Grey means "no data", never "flowing".
- **Real Green** (real) on **Green Wash** (real-soft) with **Green Ink** (real-ink): a read from a real photograph (enrolment or phone camera), the "real photo" label, the "Measured" badge, the Live chip's pulsing dot, an accepted read's confidence meter, a "One vehicle" clone verdict, and the ping on a junction when a real read lands.
- **Simulated Slate** (sim) on the **Slate Hatch** (sim-hatch-a, sim-hatch-b at 135 degrees, 4px and 3px stripes): the "Simulated city" badge with a dashed hollow dot, the paused Live dot, and simulated bars in reader comparisons.
- **Refusal Amber** (warn) on **Amber Wash** (warn-soft) with **Amber Ink** (warn-ink; warn-ink-2 for longer reason and caveat text): refused captures in the feed, the refused label, a refused read's meter, low-confidence characters, the refused detection box, the Watchlist badge, and privacy caveats.
- **Alarm Red** (danger) on **Red Wash** (danger-soft) with **Red Ink** (danger-ink): alerts only. Critical alert cards, the clone verdict, the "Possible clone" badge, impossible legs, the clone line and its tag, clone junctions, errors, and the danger button.

### Neutral
- **Control Room Ground** (ground): the page, the map background behind the tiles, and the dark casing under every road and route line.
- **Charcoal Sheet** (surface): sheets, chips, the search field, tooltips, zoom buttons.
- **Well** (surface-2) and **Raised Well** (surface-3): stat tiles, alert cards, quiet buttons, the segmented track and its active option, plates, OCR labels, meter and bar tracks, hover fills.
- **Hairline** (line) and **Firm Hairline** (line-2): row dividers (solid in feeds and legs, dashed in Insights rows and tables), chip separators, scrollbar thumbs, the modelled label's dashed border.
- **Ink** (text), **Secondary Ink** (text-2), **Tertiary Ink** (text-3): a three-step text ramp. Tertiary Ink is the lowest used for readable text.
- **Provenance Hatch** (provline-hatch-a, provline-hatch-b) with **Provenance Ink** (provline-ink): the sheet's provenance line, a wider 6px/3px hatch of the same slate idea.
- **On Blue** (on-blue): text and icons on Route Blue, Inferred Violet and Alarm Red fills.

### Named Rules
**The Colour Means Something Rule.** The basemap is OSM's standard style inverted and greyed (`grayscale(1) invert(1) brightness(.5) contrast(1.05)` at 72% opacity). On the map, colour means road load, the tracked vehicle, or an alert. A road with no reads in the window is Quiet Road grey, never green.

**The One Route Blue Rule.** Route Blue is the vehicle being followed and the one primary action in view. In the alert stack only the newest alert's Track button is primary; every older Track steps back to the Blue Wash secondary.

**The Four Provenances Rule.** Simulated is hatched slate with a dashed dot, real is green with a solid dot, inferred is violet with a hollow ring, refused is amber. Each differs in pattern or stroke as well as hue, so none depends on colour alone.

**The Red Is For Alerts Rule.** Alarm Red appears only for alerts and what they point at: critical cards, clone verdicts, impossible legs, clone lines, errors. Severe traffic (t3) is a ramp step, not an alarm.

**The Tinted Ink Rule.** Every wash sets its text in a lighter ink of its own hue (blue near-match, green verdict, red alert, amber reason, violet inferred), never plain grey on a tint. Bold headings inside a wash step one shade lighter again.

## Typography

**Display Font:** Geist (with ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto)
**Label/Mono Font:** ui-monospace, Cascadia Mono, Consolas (12px, only for the judge-console URL beside its QR code)

**Character:** One neutral grotesque, precise and familiar, for the whole interface. Tabular figures (`tnum`) and the `cv11` alternate are always on, so confidences, counts and clocks stay aligned while the feed updates.

### Hierarchy
- **Display** (700, 34px, line-height 1.05, 0.02em): the plate of the opened vehicle or of a read result, and nothing else.
- **Headline figure** (700, 28px, -0.02em): a single headline measurement in Insights (for example the real-photo accuracy), its unit in 600, 14px, Tertiary Ink.
- **Stat** (650, 18px, -0.01em): stat-tile values, unit in 600, 12px, Tertiary Ink.
- **Brand** (700, 16px, -0.02em): the wordmark; the city name beside it is 13px Tertiary Ink behind a hairline.
- **Search** (500, 15px, 0.01em): the Spotlight field.
- **Title** (650, 15px, -0.01em): sheet and block headings.
- **Body** (400, 14px, 1.45): the default; per-character confidence cells use 650.
- **Plate** (650, 13.5px, 0.06em): every inline plate. The wide tracking gives it the reading weight of a plate rather than a costume.
- **Body Small** (400 to 600, 12.5 to 13px): chips, buttons, alert messages, hints, rows.
- **Label** (600, 12px, Tertiary Ink, sentence case): list headings and table heads. Meta lines (times, junction names) use 400, 12px, Tertiary Ink.
- **Badge** (600, 11.5px): provenance badges and read labels.
- **Micro** (500, 10.5 to 11px): map attribution and histogram axis labels. The floor; never for anything an operator acts on.

### Named Rules
**The Plate Is The Headline Rule.** Only a plate is set at display size. One headline figure per Insights block may reach 28px; everything else stays at 18px or below.

**The Self-Hosted Face Rule.** Geist is self-hosted as a variable font (100 to 900, SIL OFL; provenance in `web/fonts/README.txt`), split into latin and latin-ext `unicode-range` subsets with `font-display: swap`, latin preloaded. No third-party font or CDN requests; Leaflet is vendored too, because the room must run offline at a venue.

**The Tabular Rule.** Numbers are always tabular. Never switch `tnum` off for a figure that can change.

## Layout

The map is fixed and full-bleed. The UI floats above it with a 12px gutter (gap).

- **Top bar:** a three-column grid. Lead (brand chip plus the "Simulated city" provenance chip) sits in the side-sheet column (392px); the Spotlight search fills the middle (260 to 520px); status chips right-align in the rest (Insights button, then one chip holding Live, reads per minute and cameras). Every top-bar element is 44px tall (top-h).
- **Left side sheet** (392px, full height below the top bar): a sticky segmented control (Live reads, Read a plate), then the view. Opening a vehicle replaces the view in the same sheet with a Back button and a look-back window select.
- **Alert stack** (356px, bottom right, up to 64vh): grouped one card per vehicle and rule (a repeat adds a ×n count rather than a new card), at most five cards; non-critical alerts are counted in the header summary, not stacked.
- **Insights drawer** (420px, right, full height below the top bar): replaces the alert stack while open, with a sticky segmented header (Analytics, Engine, Incidents, Privacy) and a close button. Leaflet's zoom control shifts left by whichever right-hand sheet is showing.
- **Map key:** a translucent chip at bottom left, just right of the side sheet.

Spacing is a 4px base used in steps of 2, 3, 4, 6, 8, 10, 12, 14 and 16. Sheets pad 8px with views inset 10px; cards and wells pad 10 to 12px.

Responsive behaviour:
- **1520px and below:** the search narrows to 440px max; **1400px and below:** 400px max.
- **1360px and below:** the side sheet narrows to 360px and the alert stack to 320px.
- **1100px and below:** the status chips drop to a second top-bar row, the side sheet starts below both rows, and the map key hides.
- **760px and below:** everything stacks in document flow: top bar, map (52vh), alerts, side sheet, Insights drawer. Sheets lose their float, radius and shadow and are separated by hairlines.

## Elevation & Depth

A hybrid of tonal layering and soft ambient shadow. The map is the floor. Sheets and chips float on diffuse, near-black shadows with a negative spread. Inside a sheet, depth comes from tone (Well and Raised Well fills, hairline dividers), not from more shadow. The active segmented option is the only element raised within a sheet (`0 1px 2px rgba(0,0,0,.4)`).

### Shadow Vocabulary
- **Sheet** (`0 1px 2px rgba(0, 0, 0, .45), 0 14px 36px -8px rgba(0, 0, 0, .62)`): side sheet, alert stack, Insights drawer, Spotlight results, toast.
- **Chip** (`0 1px 2px rgba(0, 0, 0, .5), 0 6px 16px -4px rgba(0, 0, 0, .5)`): top-bar chips, search field, zoom control, map key, map tooltips.
- **Focus** (`0 0 0 3px rgba(77, 132, 255, .38)`): every focus-visible state; the search field stacks it on the Chip shadow.

### Named Rules
**The Soft Float Rule.** Shadows are diffuse and dark with a negative spread. No hard offset shadows, no coloured glows. The Live dot's pulse is a fading green ring, not a glow at rest.

**The Visible First Frame Rule.** Motion only moves content already on screen; nothing enters from opacity 0. The toast rises 6px, meters and bars grow from 20% width, a read lands as a ring that widens and fades from its junction (blue ping 0.9s; green 1.2s for a real read). Easing is `cubic-bezier(.2, .7, .2, 1)`, 150 to 180ms for state changes, up to 420ms for growth. Reduced motion collapses all animation.

**The Drawn Route Rule.** The one authored motion is the tracked route: each sighting's dot and the leg after it appear in turn (about 900ms in total, at least 60ms per step), while the road layer dims to 38% opacity so the route reads first. First seen and Last seen stop tags pin the ends.

## Shapes

Corners are generous and nest in a step-down: sheets 18px; cards, the search field, the drop zone and Spotlight results 14px; wells, the segmented track, notes and the toast 12px; icon buttons 11px; controls and stat tiles 10px; segmented options and selects 9px; character cells and small keys 7px; plates 6px; read labels 5px. Chips, badges, meters, bars and toggles are full pills. Junctions are 14px markers: a core dot on a ground-coloured gap, with a load-coloured ring around it. Map lines have round caps and run over a dark ground casing (8px under 4px roads, 9px under the 5px route). The drop zone has a 1.5px dashed border. Icons are drawn SVG at 18px (16px in buttons and segmented options) in a 1.75px round stroke.

## Components

### Buttons
Quiet and familiar from phone apps, sized for a control panel.
- **Shape:** gently rounded (10px), 38px tall, 600 weight at 13.5px, 8px icon gap. Alert-card actions shrink to 32px and 12.5px.
- **Primary:** Route Blue with white text. Hover lifts to Lifted Blue. Pressing moves it down 1px.
- **Secondary:** Blue Wash with Blue Ink. Hover adds a Blue Hairline border. Used for every Track action except the newest alert's.
- **Quiet:** Well with Ink text; hover to Raised Well.
- **Danger:** Red Wash with Red Ink; hover adds a red hairline border. Only for injecting incidents in the judge tools.
- **Ghost:** transparent, Secondary Ink, 10px side padding; hover fills with Well. Used for Clear and inline "All junctions".
- **Icon button:** 36px square, 11px radius, transparent; hover fills with Well.
- **Disabled:** 50% opacity, no press movement.

### Chips
- **Status chip:** a Charcoal Sheet 44px pill with the Chip shadow, Secondary Ink 13px text, values in bold Ink, separated by 16px Firm Hairlines.
- **Live control:** inside the status chip; Green Ink text and a pulsing green dot. Paused, the text turns Secondary Ink and the dot Simulated Slate, still.
- **Insights button:** a status chip with an icon; open, it fills Route Blue with white text.
- **Map key:** a 92% Charcoal Sheet chip of 18px by 4px swatches for the four loads, Quiet Road and the tracked vehicle.

### Provenance (signature)
- **Badges** (22px pills, 11.5px, a 7px leading dot): Simulated (Slate Hatch, dashed hollow dot), Real (Green Wash, solid dot), Inferred (Violet Wash, hollow ring), Critical (Red Wash), Watchlist (Amber Wash), Plain (Well, no dot, for route summaries).
- **Read labels** (18px tags, 5px radius, 11.5px): **OCR** on Raised Well in Ink means the engine decoded the text; **modelled** is transparent with a dashed Firm Hairline border in Secondary Ink, and its confidence is shown in 500-weight Tertiary Ink because it did not come from OCR; **real photo** in Green Wash; **refused** in Amber Wash.
- **Provenance line:** the first thing in a view's body, on the Provenance Hatch, 12px radius, 12.5px Provenance Ink, the world stated in bold Ink ("Simulated Kolkata feed.") followed by the read labels in use.

### Plate
A Raised Well tag (26px, 6px radius) in 650 weight with 0.06em tracking. Refused rows tint it Amber Wash, inferred rows Violet Wash. At display size it drops the tag entirely.

### Inputs / Fields
- **Spotlight search:** a Charcoal Sheet field (44px, 14px radius, Chip shadow) with a search icon, a 15px/500 input, and a "/" key hint on a Well with a Firm Hairline. Focus adds the blue Focus ring to the shadow. Results drop 8px below in a Sheet-shadowed list; each row shows the plate, a meta line, and a distance that turns Green Ink on an exact match.
- **Select:** a 32px Well with a 9px radius, 12.5px/600 Secondary Ink, and a drawn chevron.
- **Drop zone:** Well fill with a 1.5px dashed Firm Hairline; hover or drag-over turns the border Route Blue and the fill a blue-tinted well.
- **Toggle:** a 38px by 22px Raised Well pill with a Secondary Ink knob; checked, the track is Route Blue and the knob white.

### Navigation
- **Segmented control:** a Well track (12px radius, 3px padding) of 34px options (32px in the drawer) in 13px/600 Secondary Ink with 16px icons; the active option is Raised Well, Ink, with the only in-sheet shadow. It stays sticky at the top of its sheet.

### Feed rows
A three-column row (time, plate plus read label over the junction and speed, confidence) with hairline dividers and an 8px hover fill. Refused captures sit inline in amber, never hidden.

### Alert cards
Well cards (14px radius, 11 to 12px padding); critical cards add the Red Wash and set their kind and message in Red Ink. Top line: icon and kind (12.5px/650), a ×n repeat count, and the time. Then the message, then actions: Track (primary only on the newest card), Show on map (clone alerts), and a ghost Clear.

### Read result
The plate at display size with an Accepted (Green Ink) or Refused (Amber Ink) verdict, then an 8px **confidence meter** whose fill (green, or amber when refused) runs against a Raised Well track with a 16px Ink **floor marker** ringed in Charcoal Sheet. Below it, the **per-character strip**: one Well cell per character with a 3px bar beneath whose opacity follows that character's confidence, amber under 90%. A refusal adds an amber reason note.

### Vehicle sheet
Plate at display size, badges (route summary, inferred count), a blue near-match note when the search was corrected, the clone verdict (green "One vehicle" or red), a three-up row of stat tiles, numbered legs (22px blue circles, red for impossible legs), and a quiet registry well of label-value rows.

### Insights data
Blocks separated by hairlines, each with a title and a badge. **Rows** are label-value pairs on dashed hairlines with a sub line. **Bars** are 8px pills on Raised Well tracks (blue by default, green for real, slate for simulated, amber for warn). **Spark** and **mini** histograms are Blue Hairline bars with full Route Blue on the current or busiest bar. Tables use solid hairline heads and dashed row lines with right-aligned figures.

### Toast
A Raised Well note (12px radius, Sheet shadow, 13px Ink) centred under the top bar; it rises 6px and leaves after 3.2s.

## Do's and Don'ts

### Do:
- **Do** keep the basemap inverted and greyed, and let road load, the tracked vehicle and alerts be the only colour on the map.
- **Do** colour roads and junction rings by the traffic ramp (free, moderate, heavy, severe), and draw a road with no reads in the window in Quiet Road grey.
- **Do** draw the tracked route in Route Blue (5px) over a 9px ground casing, sighting by sighting, and dim roads to 38% while it shows.
- **Do** dash impossible legs in Alarm Red (`2 9`) and show a clone as a dashed red line (`4 8`) between its two junctions, with the implied speed in a red tag.
- **Do** keep exactly one primary Route Blue action in the alert stack: the newest alert's Track.
- **Do** mark every data view with its provenance: the hatched provenance line for the simulated city, OCR or modelled labels on reads, green for real photos, violet for inferred registrations, amber for refusals.
- **Do** show refused captures inline as the engine declining, with the floor visible on the confidence meter.
- **Do** use drawn SVG icons in the 1.75px round stroke.

### Don't:
- **Don't** use violet for anything but inferred registrations, or Alarm Red for anything but alerts and what they point at (severe traffic keeps its own ramp step, t3).
- **Don't** let any provenance depend on colour alone. Keep the hatch, the dashed dot, the hollow ring and the dashed modelled border.
- **Don't** show a modelled confidence as if it came from OCR.
- **Don't** animate any content in from opacity 0.
- **Don't** set anything but a plate at display size.
- **Don't** use hard offset shadows or coloured glows.
- **Don't** build a grid of glowing KPI tiles and tabs; the map is the screen and everything else floats over it.
