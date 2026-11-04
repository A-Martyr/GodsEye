"""A mock vehicle registry — what VAHAN would answer, if it answered.

Isolated on purpose. This package owns its own SQLite file
(data/registry.db) and never opens the platform's store for writing. Nothing
outside registry/ imports it, so
it cannot change how any of them behave. Deleting this directory and
data/registry.db removes it completely.

Why a mock at all: VAHAN has no open public API. CHATGPT_BRIEF.md §B says the
design has to target a mockable interface rather than a real endpoint, so that
is what registry/store.py is - lookup(plate) returns a record whose SHAPE is
what a real integration would return, from data this file invents. Every record
is synthetic and says so.
"""
