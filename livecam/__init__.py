"""Live webcam plate reading — a self-contained demo surface.

Nothing in this package is imported by anpr/, core/, api/, sim/ or dashboard/.
It only ever reads *from* the platform (a cached anpr.ocr engine, and
core.enrol when the operator explicitly turns enrolment on), so the tuned
still-photograph path cannot be slowed down or changed by anything in here.
"""
