"""Erasures, not errors: reading a plate whose damage can be located.

The problem this solves
-----------------------
core/repair.py states the blocker exactly:

    per-character CTC posteriors are peaked almost everywhere, so 99.8% of
    characters come back above 0.9 confidence including the wrong ones, and
    per-character confidence predicts per-character correctness at AUROC 0.692.
    The engine knows which *strings* it is unsure of (AUROC 0.974); it does not
    know which *character* let it down. There is no reliable mask to build, so
    there are no blanks to fill.

Every word of that is true *of the CTC posteriors*. It does not follow that no
mask exists - only that it cannot come from inside the recogniser.

Glare, motion blur, rain and occlusion are photometric and geometric events.
They are visible in the pixels with no model at all: a blown highlight is a run
of saturated columns, a motion-blurred region has a collapsed Laplacian
variance, an occlusion is a run of columns whose ink profile does not look like
the rest of the plate. This module builds the mask from the image, which is the
one place the recogniser never looks.

Why locating the damage is worth so much
----------------------------------------
This is the erasure-versus-error distinction from coding theory. A code that
corrects `t` errors - corrupt symbols at unknown positions - corrects `2t`
erasures, symbols known to be missing, because locating the damage is half the
work. Here it is the difference between "some character in this ten-character
string is wrong, find it" and "characters 5, 6 and 7 carry no information, and
the other seven are trustworthy".

Filter, do not generate
-----------------------
The obvious implementation - wildcard the damaged positions and enumerate legal
completions - does not survive contact with the clock. An exact CTC score costs
0.74 ms on a typical lattice, so three erased digits at 10^3 completions is 740
ms, and that is before the grammar admits letters.

So this never generates. It takes a candidate pool that already exists (the
plates neighbouring cameras saw, which core/repair.py already assembles at a
median of 36 per capture) and *filters* it to the entries that agree with the
characters we still trust. Filtering is string comparison and free; only the
survivors are scored. On a 36-candidate pool, matching seven known characters
usually leaves one or two, so the whole pass costs a couple of milliseconds
rather than a second.

That inversion is what makes the idea affordable, and it is why this belongs
beside repair.py rather than inside the reader: the pool is the thing that makes
it cheap, and repair is where the pool lives.

What it will not do
-------------------
It never invents a pixel and never proposes a string no other camera saw. A
capture whose damage leaves too little signal returns nothing, which is the
correct answer and the one the platform is built to prefer. In particular it
does NOT inpaint glare: synthesised strokes read confidently as characters, and
a well-formed wrong plate is the worst output this system can produce.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

# A column is called unreliable when its reliability falls below this. Tuned to
# be reluctant: marking a good column as damaged throws away real evidence and
# widens the filter, which costs precision. The failure this guards against is
# over-eager masking, not under-eager.
COLUMN_FLOOR = 0.45

# How much of a character's width must be ruined before the character itself is
# called unreadable. A third is enough: Indian plate glyphs are not redundant,
# and losing a third of the strokes of an 8 leaves something that reads equally
# well as a 6, a 9 or a B.
DAMAGED_WIDTH_SHARE = 0.33

# Characters are only ever erased in runs this long or shorter. Damage that
# spans most of a plate is not damage to work around - it is a plate that was
# not photographed, and salvaging it would be guessing with extra steps.
MAX_ERASED = 4

# A candidate must still agree on at least this many characters to be worth
# scoring. Six of ten leaves 36^4 strings consistent in principle; the pool is
# what makes it tractable, but a candidate agreeing on fewer than this is not
# being confirmed by the image in any meaningful sense.
MIN_AGREE = 5


# --- the mask -----------------------------------------------------------
def column_reliability(crop: np.ndarray) -> np.ndarray:
    """Per-column trust in a plate crop, in [0, 1].

    Three independent detectors, combined by taking the worst. They fail in
    different ways - glare saturates, blur flattens, occlusion darkens - and a
    column only has to be ruined by one of them to be useless.
    """
    if crop is None or crop.size == 0:
        return np.ones(1, dtype=np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = gray.astype(np.float32)
    h, w = gray.shape
    if w < 8:
        return np.ones(w, dtype=np.float32)

    # 1. Glare. A blown highlight is not dim detail to be recovered - the
    #    sensor clipped and the information is gone.
    saturated = (gray >= 245.0).mean(axis=0)
    glare = 1.0 - np.clip(saturated / 0.5, 0.0, 1.0)

    # 2. Motion blur / defocus. Laplacian variance per column, normalised
    #    against the plate's own median rather than an absolute threshold: a
    #    dim night capture is uniformly low and is not thereby unreadable.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    col_var = lap.var(axis=0)
    ref = float(np.median(col_var)) or 1.0
    sharp = np.clip(col_var / (ref + 1e-6), 0.0, 1.0)

    # 3. Occlusion. A column carrying neither ink nor background contrast -
    #    mud, a wiper, a tow bar - has almost no dynamic range.
    spread = gray.max(axis=0) - gray.min(axis=0)
    ref_s = float(np.median(spread)) or 1.0
    contrast = np.clip(spread / (ref_s + 1e-6), 0.0, 1.0)

    rel = np.minimum(np.minimum(glare, sharp), contrast).astype(np.float32)
    # Smooth over a few columns. A single bad column is noise; a character is
    # ruined by a run of them, and the run is what we are looking for.
    k = max(3, w // 40) | 1
    return cv2.blur(rel.reshape(1, -1), (k, 1)).ravel()


def character_cells(crop: np.ndarray, n_chars: int) -> list[tuple[int, int]]:
    """Column span of each character, best effort.

    Tries the ink profile first - plate characters are separated by gutters,
    and the valleys between them are usually visible. Falls back to equal
    division, which is a fair approximation because the plate format is
    regulated and the characters really are evenly pitched.
    """
    if n_chars <= 0 or crop is None or crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    w = gray.shape[1]
    if w < n_chars * 3:
        return []

    edges = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    ink = edges.sum(axis=0)
    if ink.max() > 0:
        ink = ink / ink.max()
        k = max(3, w // 60) | 1
        ink = cv2.blur(ink.reshape(1, -1), (k, 1)).ravel()
        # The n_chars-1 deepest valleys, kept apart so two cuts cannot land in
        # the same gutter.
        order = np.argsort(ink)
        cuts: list[int] = []
        min_gap = max(2, w // (n_chars * 3))
        for c in order:
            c = int(c)
            if c < min_gap or c > w - min_gap:
                continue
            if all(abs(c - existing) >= min_gap for existing in cuts):
                cuts.append(c)
            if len(cuts) == n_chars - 1:
                break
        if len(cuts) == n_chars - 1:
            bounds = [0, *sorted(cuts), w]
            cells = [(bounds[i], bounds[i + 1]) for i in range(n_chars)]
            # Sanity-check the segmentation before trusting it. A blown-out
            # region carries no edges at all, so it reads as one enormous
            # gutter and pulls every cut into itself - which puts the cell
            # boundaries exactly where the damage is, on the one image where
            # getting them right matters most. Uniform pitch is the safer
            # answer whenever the valleys produce cells this uneven, and the
            # plate format is regulated enough that it is a fair one.
            uniform = w / float(n_chars)
            widths = [b - a for a, b in cells]
            if all(0.5 * uniform <= x <= 2.0 * uniform for x in widths):
                return cells

    step = w / float(n_chars)
    return [(int(i * step), int((i + 1) * step)) for i in range(n_chars)]


def suspect_positions(crop: np.ndarray, n_chars: int,
                      floor: float = COLUMN_FLOOR) -> set[int]:
    """Character indices whose pixels cannot be trusted."""
    rel = column_reliability(crop)
    cells = character_cells(crop, n_chars)
    if not cells:
        return set()
    out = set()
    for i, (a, b) in enumerate(cells):
        a, b = max(0, a), min(len(rel), max(a + 1, b))
        if b <= a:
            continue
        # Fraction of the character's width that is ruined, NOT the mean
        # reliability across it. A character with a third of its strokes blown
        # away is unreadable, but averaged against its two clean thirds it
        # scores comfortably above any floor - which is exactly how a glare
        # band spanning two characters was detected on only one of them.
        ruined = float((rel[a:b] < floor).mean())
        if ruined >= DAMAGED_WIDTH_SHARE:
            out.add(i)
    return out


# --- the filter ---------------------------------------------------------
def consistent(candidate: str, observed: str, erased: set[int]) -> bool:
    """Does `candidate` agree with `observed` everywhere we still trust it?

    Length must match: a candidate of a different length is not this plate with
    some characters damaged, it is a different plate. Cheap, and it is what
    keeps the filter from quietly turning into fuzzy matching.
    """
    if len(candidate) != len(observed):
        return False
    agree = 0
    for i, (c, o) in enumerate(zip(candidate, observed)):
        if i in erased:
            continue
        if c != o:
            return False
        agree += 1
    return agree >= MIN_AGREE


def shortlist(observed: str, pool, crop: np.ndarray | None = None,
              erased: set[int] | None = None) -> tuple[list[str], set[int]]:
    """The pool entries this damaged read could actually be.

    Returns the survivors and the erasure mask that produced them, so a caller
    can record WHY a candidate was admitted rather than only that it was.
    """
    if not observed:
        return [], set()
    if erased is None:
        erased = suspect_positions(crop, len(observed)) if crop is not None else set()
    if not erased or len(erased) > MAX_ERASED:
        return [], erased
    return [p for p in pool if consistent(p, observed, erased)], erased


# --- the pass -----------------------------------------------------------
def salvage(observed: str, pool, score_fn, crop: np.ndarray | None = None,
            erased: set[int] | None = None,
            budget_ms: float = 40.0) -> dict | None:
    """Best pool entry consistent with the undamaged characters, or None.

    `score_fn(text) -> float` is the caller's exact CTC scorer, so this module
    never needs to know which backend produced the lattice - anpr_ocr and the
    CRNN both hand in their own.

    The budget is a hard wall-clock deadline, checked between candidates. It
    can be exceeded by at most one scoring call, which is 0.74 ms. That is the
    whole reason this is safe to call anywhere: it cannot become slow, it can
    only become less thorough, and a caller that gives it 0 ms gets None
    immediately rather than a stall.
    """
    survivors, mask = shortlist(observed, pool, crop, erased)
    if not survivors:
        return None

    deadline = time.perf_counter() + budget_ms / 1000.0
    scored: list[tuple[str, float]] = []
    for cand in survivors:
        if time.perf_counter() >= deadline:
            break
        scored.append((cand, float(score_fn(cand))))
    if not scored:
        return None

    scored.sort(key=lambda kv: -kv[1])
    best, best_score = scored[0]
    runner = scored[1][1] if len(scored) > 1 else None
    return {
        "plate": best,
        "score": best_score,
        "margin": (best_score - runner) if runner is not None else None,
        "erased": sorted(mask),
        "considered": len(scored),
        "survivors": len(survivors),
        "truncated": len(scored) < len(survivors),
        "observed": observed,
    }
