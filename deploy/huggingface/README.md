---
title: GodsEye
emoji: 👁
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: City-wide ANPR and route reconstruction, read-only demo
---

# GodsEye: read-only public demo

City-wide number-plate reading and vehicle-route reconstruction over 35 real
Kolkata junctions. Built for SIH 2026 by Team Prometheus.

**What you can do here:** watch the simulated city's live reads and alerts,
search a plate (even a misread one) and follow the route GodsEye rebuilds, and
open *Insights* for city analytics and the engine's measured accuracy.

**What is switched off:** photo uploads, the incident injectors, the watchlist
and the judge console. Those change the system, so this public copy refuses
them. Run GodsEye locally for the full platform.

The camera feeds and vehicles are simulated; the plates are generated. The
city is re-seeded every time the Space starts, and a Space that has had no
visitors for a while sleeps, so the first visit can take about a minute.

Code, documentation and measured results:
[github.com/A-Martyr/GodsEye](https://github.com/A-Martyr/GodsEye)

## Updating this Space

Upload `deploy/Dockerfile` from the repository next to this README. The Space
builds from it; it clones the GitHub repository at `GODSEYE_REF` (default
`main`). After pushing to GitHub, rebuild the Space: Settings > Factory rebuild.
Docker Spaces need a Hugging Face PRO plan.
