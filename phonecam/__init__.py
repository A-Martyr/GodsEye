"""Phones as junction cameras — a multi-camera live demo.

Two or three phones, each standing at a different junction of the real Kolkata
camera graph, streaming to one laptop. A plate read at one junction and then at
another is exactly the primitive the whole platform is built on, so this turns
the demo from "it can read a plate" into "it reconstructed where that vehicle
went", which is the thing the brief actually asks for.

Separate from livecam/ and from the platform: imported by nothing in anpr/,
core/, api/ or sim/. It reuses livecam's reader and vote (which in
turn use the platform's cached engine) and writes sightings only through
core/enrol.py, only when asked.
"""
