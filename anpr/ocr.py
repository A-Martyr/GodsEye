"""The GodsEye ANPR engine.

Pipeline
--------
    plate crop
      -> normalise (deskew, CLAHE, fixed height)      segment.normalize
      -> flatten illumination + Otsu -> ink mask       segment.binarize
      -> over-segment into atoms                       segment.segment
      -> format-constrained decode  <-- this module
      -> plate string + per-character confidence

The decoder is what lifts accuracy on bad captures. Classical ANPR pipelines
commit to one segmentation and then classify; a smeared "KA" that fused into one
blob, or a "0" that broke into two fragments, is then unrecoverable. Here the
segmenter is deliberately allowed to over-cut, and a dynamic program searches
every way of grouping 1-3 adjacent atoms into characters, scoring each grouping
against the Bharat-series plate grammar (LL DD LLL DDDD). The grammar rules out
whole families of classifier errors - a digit can never win a letter slot - and
the DP recovers merges and splits that a fixed segmentation cannot.
"""
from __future__ import annotations

import math
import re
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from anpr import crnn as crnn_mod
from anpr import model as glyph_model
from anpr import segment
from anpr import track as track_mod

# fuse_reads() appends this to a winning read's variant. Matched here so a
# re-fusion over the union of two tracks can remove the previous round's tag
# rather than stack on it and claim a provenance that never happened.
_BURST_TAG = re.compile(r":?burst\d+")


def union_box(atoms, i, j):
    """Bounding box of atoms[i:j] — one character may span several atoms."""
    x = min(atoms[k][0] for k in range(i, j))
    y = min(atoms[k][1] for k in range(i, j))
    x2 = max(atoms[k][0] + atoms[k][2] for k in range(i, j))
    y2 = max(atoms[k][1] + atoms[k][3] for k in range(i, j))
    return (x, y, x2 - x, y2 - y)


LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

# The ordinary state-series shape: 2 state letters, 1-2 district digits,
# 0-3 series letters, 4 registration digits. (The comment here used to say
# "Bharat-series", which is a different format entirely - see BH_PATTERNS.)
PATTERNS: list[str] = [
    "LL" + "D" * d + "L" * s + "DDDD"
    for d in (1, 2)
    for s in (3, 2, 1, 0)
]

# The Bharat series, and the reason it needs its own list: every shape above
# begins "LL", a state code. A BH plate begins with two DIGITS - the year of
# registration - so it cannot be expressed as a state-series pattern no matter
# how the letter and digit runs are varied:
#
#     WB01BF9492  ->  LLDDLLDDDD   an ordinary state series
#     25BH3028AB  ->  DDLLDDDDLL   Bharat: year, BH, number, series letters
#
# "BH" was already in STATE_CODES, which looks like support and is not: the
# check reads text[:2], and on a BH plate those are the year digits. Measured on
# a real photograph in FIELD_TEST.md, the correct reading was not proposable at
# all and the decode fell back to an ungrammatical string at 0.02 confidence.
#
# These vehicles are registered specifically to move between states, which is
# exactly the traffic a city ANPR network is watching for.
# Shapes only - plate_shape() renders "BH" as "LL" like any other letters, so
# the literal "BH" cannot live in a shape string and is asserted separately in
# matches_plate_grammar(). Without that second check "25XY3028AB" would pass.
BH_PATTERNS: list[str] = ["DDLLDDDD" + "L" * s for s in (1, 2)]

# Deliberately NOT added: diplomatic (##CD###) and military formats. Every
# pattern admitted here is another way for a misread to look legal, and neither
# appears in the corpus, so neither can be tested. Add them with a test case,
# not on principle.

# Real Indian state / UT codes, used to repair the first two characters.
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB", "BH",
}


def plate_shape(text: str) -> str:
    """The grammar class of a string: "MH02DW8021" -> "LLDDLLDDDD"."""
    return "".join("L" if c in LETTERS else "D" if c in DIGITS else "?" for c in text)


_PATTERN_SET = frozenset(PATTERNS)
_BH_PATTERN_SET = frozenset(BH_PATTERNS)


def is_bharat_series(text: str) -> bool:
    """A Bharat-series registration: YY BH #### XX."""
    return (len(text) >= 9 and text[2:4] == "BH"
            and plate_shape(text) in _BH_PATTERN_SET)


def matches_plate_grammar(text: str) -> bool:
    """Is `text` a well-formed registration?

    Measured against every ground-truth label in the repo's real-photo corpus:
    76/76 carry a valid state code, and 74/76 match one of PATTERNS - the two
    exceptions ("KL34F", "KL34A465") are truncated transcriptions, not plate
    formats. Tight enough to steer a decoder with; callers must still fall back
    to an ungrammatical read rather than returning nothing, or those two - and
    any format this list has not met yet - become unreadable instead of merely
    wrong.

    Two families now, checked separately rather than merged into one pattern
    list: an ordinary state series starts with a state code, a Bharat series
    starts with the registration year. Nothing that passed before fails now -
    this only ever admits more.
    """
    if not text:
        return False
    if text[:2] in STATE_CODES and plate_shape(text) in _PATTERN_SET:
        return True
    return is_bharat_series(text)


# --- district codes, as a ranking signal and never as a filter -------------
# The highest district code each state or UT actually issues. "WB99AA1234" is
# grammatically perfect and does not exist; a decoder with no notion of that
# will happily prefer it to a real plate one substitution away.
#
# These are UPPER BOUNDS, not exact valid sets, and they are used only to RANK.
# That is a deliberate pair of decisions. Exact per-state sets have gaps, are
# tedious to source correctly, and go stale as new districts are notified - and
# any error in them would make a real plate unreadable, which is far worse than
# failing to prefer it. A generous bound still rejects the obviously impossible
# while being safe to be slightly wrong about.
STATE_RTO_MAX: dict[str, int] = {
    "AN": 1, "AP": 39, "AR": 22, "AS": 34, "BR": 56, "CG": 30, "CH": 4,
    "DD": 3, "DL": 13, "DN": 9, "GA": 12, "GJ": 39, "HP": 99, "HR": 99,
    "JH": 24, "JK": 22, "KA": 71, "KL": 99, "LA": 2, "LD": 9, "MH": 50,
    "ML": 10, "MN": 7, "MP": 70, "MZ": 8, "NL": 10, "OD": 35, "OR": 35,
    "PB": 99, "PY": 5, "RJ": 58, "SK": 8, "TN": 99, "TR": 8, "TS": 38,
    "UK": 20, "UP": 96, "WB": 98,
}


def plate_plausibility(text: str) -> float:
    """How much this string looks like a registration that could exist.

    A ranking signal in [0, 1], NOT a filter. Nothing is ever rejected on this
    number: a reader that refuses to output a plate because a lookup table has
    not heard of its district has stopped being a reader. It exists to break
    ties, so that between two candidates the evidence scores almost equally the
    one that could be a real vehicle wins.

    Cheap by construction - two slices, a dict lookup and an int() - so it can
    be called on every candidate of every read without moving the clock.
    """
    if not text or not matches_plate_grammar(text):
        return 0.0
    if is_bharat_series(text):
        # Year field: BH began in 2021, and a plate stamped decades ahead is
        # not a plate. Everything else about the format is fixed.
        try:
            year = int(text[:2])
        except ValueError:
            return 0.5
        return 1.0 if 20 <= year <= 40 else 0.4

    state = text[:2]
    digits = ""
    for ch in text[2:]:
        if ch in DIGITS:
            digits += ch
        else:
            break
    if not digits:
        return 0.7
    district = int(digits)
    if district == 0:
        return 0.2                      # no state issues district 00
    ceiling = STATE_RTO_MAX.get(state)
    if ceiling is None:
        return 0.7                      # a state we have no bound for
    return 1.0 if district <= ceiling else 0.3


def repair_state_code(text: str) -> tuple[str, bool]:
    """Snap the first two characters onto a real state code when they are not
    one and the fix is a single confusable substitution.

    Module-level twin of ANPREngine._repair_state, so a backend that decodes
    without an engine instance (anpr/anpr_ocr_reader.py) can use it too. With
    39 codes in the lexicon "W8" is not ambiguous - it is WB.
    """
    if len(text) < 4 or text[:2] in STATE_CODES:
        return text, False
    best, best_cost = None, 3
    for code in STATE_CODES:
        cost = sum(a != b for a, b in zip(code, text[:2]))
        if cost < best_cost and all(
            a == b or _confusable(a, b) for a, b in zip(code, text[:2])
        ):
            best, best_cost = code, cost
    if best is None:
        return text, False
    return best + text[2:], True


# Each backend's confidence means a different thing, so each gets its own
# storage floor. Anything not listed falls back to the classical floor.
_BACKEND_FLOORS = {
    "crnn": "MIN_PLATE_CONFIDENCE_CRNN",
    "anpr_ocr": "MIN_PLATE_CONFIDENCE_ANPR_OCR",
}


@dataclass
class PlateRead:
    text: str                       # decoded plate, no spaces ("" if unreadable)
    confidence: float               # 0-1; what the floor is applied to. CRNN: the
                                    # exact CTC probability of the string, per
                                    # character. Classical: weakest character x
                                    # agreement between binarisations.
    char_confidence: float = 0.0    # the weakest character alone
    agreement: float = 0.0          # classical: share of binarisations that agreed;
                                    # burst reads: share of frames that agreed
    mean_confidence: float = 0.0    # geometric mean over the characters
    char_confidences: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    pattern: str = ""
    raw: str = ""                   # greedy per-atom read, before the decoder
    repaired: bool = False          # True if the state code was corrected
    plate_found: bool = True
    variant: str = ""               # which binarisation hypothesis won ("crnn" for the network)
    backend: str = "classical"      # which recogniser produced this read

    @property
    def floor(self) -> float:
        """The confidence floor this read is judged against."""
        name = _BACKEND_FLOORS.get(self.backend)
        return getattr(config, name) if name else config.MIN_PLATE_CONFIDENCE

    @property
    def accepted(self) -> bool:
        """Would the platform store this read? Ask the read, not a constant:
        the two backends' confidences are calibrated separately.

        A stored read must also look like a registration at its start: a real
        state code, or a Bharat-series year. The real-plate model emits an
        "UNREADABLE" token for illegible plates, and on low-resolution footage
        variants of it ("UNREADBLE", "UNREAABLE") cleared the floor at 0.86.
        Unusual formats with a valid state code are still accepted; anything
        else is returned to the caller but never stored."""
        if not (self.text and self.confidence >= self.floor):
            return False
        return self.text[:2] in STATE_CODES or is_bharat_series(self.text)
    candidates: list[tuple[int, int, int, int]] = field(default_factory=list)
    reason: str = ""                # why nothing was read, in operator language

    @property
    def pretty(self) -> str:
        from anpr.plates import pretty
        return pretty(self.text)

    def as_dict(self) -> dict:
        return {
            "text": self.text, "pretty": self.pretty, "confidence": round(self.confidence, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "pattern": self.pattern, "raw": self.raw, "repaired": self.repaired,
            "char_confidence": round(self.char_confidence, 4),
            "agreement": round(self.agreement, 3),
            "char_confidences": [round(c, 3) for c in self.char_confidences],
            "boxes": self.boxes, "plate_found": self.plate_found, "variant": self.variant,
            "backend": self.backend, "accepted": self.accepted,
            "floor": round(self.floor, 3),
            "candidates": self.candidates, "reason": self.reason,
        }


class ANPREngine:
    """Loads (or trains) the glyph model once and reads plates."""

    def __init__(self, model: glyph_model.GlyphModel | None = None, max_merge: int = 3,
                 char_bonus: float = 0.25, skip_penalty: float = 1.6,
                 crnn_reader=None, use_crnn: bool = True, easyocr_reader=None,
                 anpr_ocr_reader=None):
        self.model = model or glyph_model.load_or_train()
        # The CRNN reads the plate without segmenting it, which is what the
        # camera-realistic corpus demands. If it is not available - no torch, no
        # weights - everything still works on the classical pipeline.
        self.crnn = crnn_reader if crnn_reader is not None else (
            crnn_mod.load() if use_crnn else None)
        # Evaluation-only extra backends (anpr/easyocr_reader.py,
        # anpr/anpr_ocr_reader.py). Only used when explicitly passed in -
        # never picked up implicitly, unlike crnn/classical above - so the
        # default engine construction used by sim/api/dashboard is
        # unaffected.
        self.easyocr = easyocr_reader
        self.anpr_ocr = anpr_ocr_reader
        self.max_merge = max_merge
        # Reward for explaining one more character (an insertion bonus, as in
        # speech decoding). Without it the DP is free to drop a hard character
        # to keep its average confidence high; a plate is not allowed to be
        # partially read, so paying for coverage is the right trade.
        self.char_bonus = char_bonus
        # Cost of discarding an atom as noise. Mud specks, rivets and frame
        # fragments segment like characters; without an escape hatch the DP has
        # to fold them into a neighbouring glyph and corrupts it.
        self.skip_penalty = skip_penalty
        self._last_debug: dict = {}
        self._class_index = {c: i for i, c in enumerate(self.model.classes)}
        self._letter_idx = np.array([self._class_index[c] for c in sorted(LETTERS)
                                     if c in self._class_index])
        self._digit_idx = np.array([self._class_index[c] for c in sorted(DIGITS)
                                    if c in self._class_index])

    # --- public API ----------------------------------------------------
    def read(self, image: np.ndarray) -> PlateRead:
        """Read a plate crop (grayscale or BGR)."""
        if self.anpr_ocr is not None:
            return self.anpr_ocr.read(image)
        if self.easyocr is not None:
            return self.easyocr.read(image)
        if self.crnn is not None:
            return self._read_crnn(image)
        return self._read_classical(image)

    def _read_crnn(self, image: np.ndarray) -> PlateRead:
        """CRNN + grammar-constrained CTC decode.

        No binarisation, no segmentation: the network sees the greyscale crop
        and CTC marginalises over every alignment between its output columns and
        the string.

        Confidence is that same marginal: the exact probability CTC assigns to
        the decoded string, normalised per character. It is not the weakest
        character any more, because the weakest character was being estimated by
        a heuristic scan that ranked correct reads *below* wrong ones (AUROC
        0.37). The weakest character is still reported, as `char_confidence`,
        and still drives the state-code repair - it is now computed from the
        alignment marginal rather than guessed at.
        """
        gray = segment.to_gray(image)
        self._last_debug = {"norm": crnn_mod.prepare(gray), "ink": None, "atoms": []}
        return self._crnn_plate_read(self.crnn.read(gray))

    def _crnn_plate_read(self, decoded) -> PlateRead:
        """Turn one CRNN decode into a PlateRead."""
        text, confs, pattern, greedy, score = decoded
        if not text:
            return PlateRead(text="", confidence=0.0, plate_found=False, backend="crnn",
                             reason="the recogniser produced no registration for this crop")
        text, repaired = self._repair_state(text, confs)
        weakest = float(np.min(np.clip(confs, 1e-6, 1.0))) if confs else 0.0
        mean = float(np.exp(np.mean(np.log(np.clip(confs, 1e-6, 1.0))))) if confs else 0.0
        return PlateRead(text=text, confidence=score, char_confidence=weakest,
                         agreement=1.0, mean_confidence=mean,
                         char_confidences=list(confs), pattern=pattern,
                         raw=greedy, repaired=repaired, variant="crnn", backend="crnn")

    def read_evidence(self, images: list[np.ndarray]):
        """Read a burst and keep the CTC lattices -> (fused read, evidence).

        The evidence is what makes a failed capture repairable later: the plate
        is often still in the lattice when the decode did not find it, so a
        candidate from a neighbouring camera can be scored against the original
        image rather than against the wrong string it produced. Only the CRNN
        backend has a lattice; on the classical fallback this degrades to an
        ordinary burst read with no evidence, and those captures stay
        unrepairable.
        """
        frames = [im for im in images if im is not None and im.size]
        if self.crnn is None:
            return self.read_burst(frames), []
        reads, evidence = [], []
        for im in frames:
            decoded, lattices = self.crnn.read_with_logits(segment.to_gray(im))
            evidence.append(lattices)
            r = self._crnn_plate_read(decoded)
            if r.text:
                reads.append(r)
        return self.fuse_reads(reads), evidence

    def _read_classical(self, image: np.ndarray) -> PlateRead:
        """Segment-and-classify: kept as the fallback when torch is absent.

        Every binarisation hypothesis is decoded and the highest-scoring read
        wins, so one bad threshold no longer costs the plate.
        """
        norm = segment.normalize(image)
        self._last_debug = {"norm": norm, "ink": None, "atoms": []}
        best = None
        atoms_seen = []
        decoded_strings: list[str] = []
        for variant, ink in segment.ink_variants(norm):
            atoms = segment.segment(ink)
            atoms_seen.append(atoms)
            if len(atoms) < 4:
                continue
            units, probs = self._score_units(ink, atoms)
            decoded = self._decode(atoms, units, probs)
            if decoded is None:
                continue
            score, text, confs, pattern, spans = decoded
            decoded_strings.append(text)
            if best is None or score > best[0]:
                best = (score, text, confs, pattern, spans, atoms, ink, variant,
                        self._greedy(atoms, units, probs))
        if best is None:
            widest = max(atoms_seen, key=len) if atoms_seen else []
            self._last_debug["atoms"] = widest
            return PlateRead(
                text="", confidence=0.0, boxes=widest, plate_found=False,
                reason=(f"segmented {len(widest)} marks but no arrangement of them spells a "
                        "valid Indian registration (two letters, district digits, series "
                        "letters, four digits)"))

        _, text, confs, pattern, spans, atoms, ink, variant, raw = best
        self._last_debug = {"norm": norm, "ink": ink, "atoms": atoms, "spans": spans}
        agreement_text = text
        text, repaired = self._repair_state(text, confs)
        boxes = [self._union(atoms, a, b) for a, b in spans]
        # Confidence combines two things the engine actually knows.
        #
        # First, the weakest character: one wrong glyph makes the whole
        # registration wrong, so the minimum is the honest summary, not the mean.
        #
        # Second, and far more informative, how many of the binarisation
        # hypotheses decoded the same string. The winner is chosen as the
        # highest-scoring of eight, so its own confidence is inflated by
        # selection - measured over 500 reads it is 0.99 on correct reads and
        # 0.89 on wrong ones, which barely separates them. Agreement between
        # independent hypotheses is not selected for in the same way and scores
        # 0.80 against 0.26. Their product is what the confidence floor uses.
        clipped = np.clip(np.asarray(confs, dtype=float), 1e-6, 1.0)
        char_conf = float(clipped.min())
        agreement = (decoded_strings.count(agreement_text) / len(decoded_strings)
                     if decoded_strings else 0.0)
        return PlateRead(text=text, confidence=char_conf * agreement,
                         char_confidence=char_conf, agreement=agreement,
                         char_confidences=list(confs),
                         mean_confidence=float(np.exp(np.mean(np.log(clipped)))),
                         boxes=boxes, pattern=pattern, raw=raw, repaired=repaired,
                         variant=variant)

    def read_burst(self, images: list[np.ndarray]) -> PlateRead:
        """Fuse the frames a camera captures of one vehicle into one read.

        A real ANPR node does not get one photograph of a car. It is triggered
        by a loop or a tripwire and takes a burst as the vehicle crosses the
        zone - five to fifteen frames at different distances, exposures and
        motion blurs, of the same plate. Reading only the middle one throws
        away every other look.

        The frames are read independently and their decodes voted, weighted by
        the exact CTC score, which is what makes this work at all: a vote is
        only as good as its ability to tell a confident frame from a lucky one.
        Measured over 250 vehicles across all ten conditions, one frame reads
        40.4% of plates and eight read 77.2%, with the same weights - the frames
        fail in different ways and agreement is strong evidence.

        The reported confidence stays the best single frame's, so the storage
        floor keeps exactly the meaning it was calibrated with; how many frames
        agreed is reported separately, in `agreement`.
        """
        return self.fuse_reads([self.read(im) for im in images
                                if im is not None and im.size])

    def fuse_reads(self, reads: list[PlateRead]) -> PlateRead:
        """Combine already-decoded frames of one vehicle into a single read.

        Split out from `read_burst` so a caller that wants to show its working -
        the ANPR Lab draws every frame and what it individually read - does not
        have to decode each frame twice.
        """
        reads = [r for r in reads if r is not None and r.text]
        if not reads:
            return PlateRead(text="", confidence=0.0, plate_found=False,
                             backend="crnn" if self.crnn is not None else "classical",
                             reason="no frame in the burst held a readable registration")
        tally: dict[str, float] = {}
        for r in reads:
            tally[r.text] = tally.get(r.text, 0.0) + max(r.confidence, 1e-9)
        winner = max(tally.items(), key=lambda kv: kv[1])[0]
        agreeing = [r for r in reads if r.text == winner]
        best = max(agreeing, key=lambda r: r.confidence)
        best.agreement = len(agreeing) / len(reads)
        best.variant = f"{best.variant}:burst{len(reads)}" if best.variant else f"burst{len(reads)}"
        return best

    def read_sequence(self, frames: list[np.ndarray], max_candidates: int = 6,
                      accepted_only: bool = True):
        """Track every plate across a sequence and return one read per vehicle.

        read_frame() answers "what is in this photograph". This answers "which
        vehicles went past", which is the question a camera actually faces, and
        it is worth far more than any single-frame improvement measured here:

            synthetic, 10 conditions   1 frame 52%  ->  8 frames 88%
            real photographs           per frame 91%  ->  per vehicle 97%

        The frames fail independently - a plate lost to glare in one is legible
        in the next - so agreement between them is evidence in a way that no
        amount of work on a single frame can be.

        Returns [(track, fused_read, per_frame_reads)], best first. A vehicle
        seen once is not special-cased: it is a track of length one that fuses
        to its own read.

        accepted_only keeps just the registrations that cleared the confidence
        floor, which is what a caller almost always wants - a street scene
        produces a dozen plate-shaped pieces of signage per frame, and each one
        the tracker follows would otherwise be reported as a vehicle. Pass
        False to see everything the sweep considered.
        """
        frames = [f for f in frames if f is not None and getattr(f, "size", 0)]
        if not frames:
            return []

        per_frame_boxes = [detect_plate_candidates(f, max_candidates) for f in frames]
        tracks = track_mod.track_boxes(per_frame_boxes)

        # Every crop from every track goes through the recogniser in one batch.
        # This is what makes tracking affordable: a ten-frame sequence with two
        # vehicles is forty-odd crops, which one at a time would be seconds and
        # batched is a fraction of one. Reading per track instead would give up
        # most of that.
        jobs: list[tuple[int, int, np.ndarray]] = []      # track index, frame, crop
        for ti, track in enumerate(tracks):
            for frame_index, box in zip(track.frames, track.boxes):
                for crop in _crop_variants(frames[frame_index], box):
                    if crop.size:
                        jobs.append((ti, frame_index, crop))
        if not jobs:
            return []

        reads = self._read_many([j[2] for j in jobs])

        # Best variant per (track, frame) first - the same rule read_frame()
        # uses within one photograph - and only then vote across frames. Voting
        # over raw variants instead would let one vehicle's four crops of a
        # single frame outvote four genuinely independent looks at it.
        best_per_frame: dict[tuple[int, int], PlateRead] = {}
        for (ti, fi, _), read in zip(jobs, reads):
            key = (ti, fi)
            current = best_per_frame.get(key)
            if current is None or (read.text and read.confidence > current.confidence):
                best_per_frame[key] = read

        fused_tracks = []
        for ti, track in enumerate(tracks):
            frame_reads = [best_per_frame[(ti, fi)] for fi in track.frames
                           if (ti, fi) in best_per_frame]
            if not frame_reads:
                continue
            fused_tracks.append((track, self.fuse_reads(frame_reads), frame_reads))

        out = self._merge_agreeing(fused_tracks)

        # Accepted first, then by confidence, and only then by how many frames
        # backed it. Sorting on track length first looks reasonable and is
        # wrong: a run of signage tracked across six frames would outrank the
        # actual registration read confidently in two, which is precisely
        # backwards for an operator reading down the list.
        out.sort(key=lambda t: (t[1].accepted, t[1].confidence, t[0].length),
                 reverse=True)
        if accepted_only:
            out = [t for t in out if t[1].accepted and t[1].text]
        return out

    def _merge_agreeing(self, fused_tracks: list) -> list:
        """Fold tracks that read the same registration into one vehicle.

        The tracker follows boxes, and the localiser does not find the plate in
        every frame - it loses it to motion blur and to underexposure, which
        are the frames fusion exists to survive. A plate found in frames 0-1
        and again in 5-6 becomes two tracks separated by a gap no reasonable
        max_gap would bridge, and reporting one vehicle twice is worse than
        useless when the next thing downstream is a clone detector that counts
        distinct sightings.

        Merging on the decoded text rather than on geometry is what makes this
        safe: two tracks are only joined when the reader independently read the
        same registration off both, and only when both cleared the confidence
        floor. Anything below the floor is left as its own track, because an
        unaccepted string is not evidence of anything.
        """
        merged: dict[str, list] = {}
        loose: list = []
        for track, fused, frame_reads in fused_tracks:
            if not (fused.text and fused.accepted):
                loose.append((track, fused, frame_reads))
                continue
            if fused.text not in merged:
                merged[fused.text] = [track, frame_reads]
            else:
                held_track, held_reads = merged[fused.text]
                combined = track_mod.Track(
                    track_id=min(held_track.track_id, track.track_id),
                    frames=sorted(set(held_track.frames) | set(track.frames)),
                    boxes=held_track.boxes + track.boxes)
                merged[fused.text] = [combined, held_reads + frame_reads]

        # fuse_reads() tags its winner ":burstN" and does so by mutating the
        # read it picked, which is one of the reads passed in. Fusing a second
        # time over the union therefore stacks the tags - "burst2:burst4" - and
        # claims a provenance that never happened. Strip the previous round's
        # tag before re-fusing so the label describes the fusion actually being
        # reported.
        out = []
        for track, reads in merged.values():
            for r in reads:
                r.variant = _BURST_TAG.sub("", r.variant).strip(":")
            out.append((track, self.fuse_reads(reads), reads))
        return out + loose

    def read_detailed(self, image: np.ndarray):
        """read() plus the intermediate images, for the ANPR Lab view and for
        the self-training harvester."""
        read = self.read(image)
        return read, self._last_debug

    def _read_many(self, crops: list) -> list:
        """read() over several crops, batched where the backend can batch.

        Only anpr_ocr can: its input shape is fixed, so any set of crops is one
        tensor. The CRNN and the classical reader keep the plain loop, which
        costs them nothing - a CRNN decode is about 5 ms, and it already
        batches its own layout hypotheses internally.
        """
        if self.anpr_ocr is not None and len(crops) > 1:
            return self.anpr_ocr.read_many(crops)
        return [self.read(c) for c in crops]

    def read_frame(self, frame: np.ndarray, max_candidates: int = 12) -> PlateRead:
        """Read from a full photograph rather than a tight plate crop.

        Every candidate region the localiser proposes is decoded, plus the whole
        frame in case the image already *is* a crop, and the most confident
        legal read wins. When nothing wins, the returned read carries the
        candidate boxes and a plain-language reason, because "no plate found" on
        its own tells an operator nothing about what to try next.
        """
        cands = detect_plate_candidates(frame, max_candidates)

        # How much searching this frame is worth depends entirely on what a
        # decode costs. The CRNN decodes in about 5 ms, so trying the whole
        # frame and every inset of every box - up to 48 decodes - costs a
        # tenth of a second and is simply free accuracy. anpr_ocr decodes in
        # over a second: the identical sweep took 46 seconds on one 960x1280
        # photograph, of which 99.3% was paddle forward passes and 0.6% was
        # everything this engine actually does. Same code, same boxes, a
        # recogniser three orders of magnitude more expensive.
        thorough = self.anpr_ocr is None

        # The whole frame is read in case the caller handed us something that
        # is already a plate crop. When the localiser has found regions, it is
        # the single most expensive decode in the sweep and the least likely
        # to win, so the slow reader skips it - a 960x1280 frame costs it
        # around 3 s against 1.7 s for a crop.
        if thorough or not cands:
            best = self.read(frame)
            tried = 1
        else:
            # Only reachable with the slow reader in play, so the backend is
            # known. An empty read that every crop below is compared against.
            best = PlateRead(text="", confidence=0.0, plate_found=False,
                             backend="anpr_ocr")
            tried = 0
        best.candidates = cands

        if thorough or config.FRAME_SEARCH_SECONDS <= 0:
            deadline = math.inf
        else:
            deadline = time.perf_counter() + config.FRAME_SEARCH_SECONDS
        truncated = False

        def sweep(boxes) -> None:
            """Decode every variant of every box, keeping the best read.

            A closure rather than a second copy of the loop, because the
            escalation below has to run exactly the same sweep over a second
            set of boxes and the two must not be able to drift apart.
            """
            nonlocal best, tried, truncated
            for box in boxes:
                if time.perf_counter() >= deadline:
                    truncated = True
                    return
            # A localiser box is approximate, and a crop that carries a strip of
            # bumper into the binariser reads worse than one cut a little tight.
            # Decoding a few insets of the same region costs milliseconds and
            # recovers most of what a loose box loses.
                variants = [c for c in _crop_variants(frame, box) if c.size]
                if time.perf_counter() >= deadline:
                    truncated = True
                    return
            # One box's insets go through the recogniser together when it can
            # take them together. A forward pass is launch-bound rather than
            # compute-bound, so four crops in one batch cost barely more than
            # one - and batching per box rather than across the whole frame is
            # what keeps the early exit below meaningful. Batch everything and
            # the search would have to decode every box before it could notice
            # that the first one already answered the question.
                reads = self._read_many(variants)
                tried += len(variants)
                for r in reads:
                    if r.text and r.confidence > best.confidence:
                        r.candidates = cands
                        best = r
                # Boxes arrive in the localiser's confidence order, so a strong
                # legal read from an early one is almost certainly the answer and
                # the rest of the sweep only re-confirms it. Worth seconds each to
                # the slow reader; worth nothing to the CRNN, which does not stop.
                if (not thorough and best.text and best.accepted
                        and best.confidence >= config.FRAME_EARLY_EXIT_CONFIDENCE):
                    return

        sweep(cands)

        # --- escalation: look again at other input sizes ------------------
        # Gated on the READ, not on the detector's own confidence, and that
        # distinction is the whole design. The detector will happily award 0.33
        # to a blue panel on a bus in the background - measured, FIELD_TEST.md -
        # so "the localiser seems confident" is not evidence that a plate was
        # found. Whether anything decoded into a registration is.
        #
        # Only for the slow reader. The CRNN sweeps exhaustively and cheaply
        # already, so a second detector pass buys it nothing it will not find.
        if (not thorough and not truncated and config.YOLO_ESCALATE_SCALES
                and not (best.text and best.accepted)):
            extra = detect_plate_candidates(frame, max_candidates,
                                            scales=tuple(config.YOLO_ESCALATE_SCALES))
            fresh = [b for b in extra if all(_iou(b, c) < 0.5 for c in cands)]
            if fresh:
                cands = cands + fresh
                best.candidates = cands
                # A fresh budget for the second round. Sharing the first one
                # would mean the escalation never runs on exactly the frames
                # that needed it - the difficult ones, which are the frames
                # that spent the first budget.
                deadline = (math.inf if config.FRAME_SEARCH_SECONDS <= 0
                            else time.perf_counter() + config.FRAME_SEARCH_SECONDS)
                sweep(fresh)
        # Never let a bounded search report itself as an exhaustive one: an
        # operator told "no plate here" deserves to know the search stopped
        # early, because re-running on a tighter crop is then worth trying.
        unsearched = max(0, len(cands) - tried) if truncated else 0
        note = ("This engine reads vehicle number plates - two letters, district digits, a "
                "series and four digits - and refuses anything that is not one. It is not a "
                "general text or handwriting reader.")
        if not best.text:
            best.plate_found = False
            best.candidates = cands
            stopped = (f" Stopped after {tried} of them to keep the read "
                       f"responsive, so {unsearched} lower-ranked region(s) were "
                       f"not examined - crop to the plate and try again."
                       if truncated else "")
            best.reason = (f"examined {tried} plate-shaped region(s) and none of them held a "
                           f"readable registration.{stopped} {note}")
        elif not best.accepted:
            # Something decoded, but not well enough to act on. Saying "found"
            # here would be the dishonest answer: noise and lettering that is
            # not a plate will always produce *some* grammatical string, and the
            # confidence floor is what separates that from a real read.
            best.plate_found = False
            best.reason = (f"best candidate was {best.pretty} at {best.confidence:.0%} "
                           f"confidence, below the {best.floor:.0%} floor, so "
                           f"the platform would discard it rather than store it. {note}")
        return best

    # --- decoding ------------------------------------------------------
    def _score_units(self, ink, atoms):
        """Classify every grouping of 1..max_merge adjacent atoms."""
        units: list[tuple[int, int]] = []
        feats = []
        n = len(atoms)
        for i in range(n):
            for m in range(1, min(self.max_merge, n - i) + 1):
                box = self._union(atoms, i, i + m)
                if box[2] > 0.45 * ink.shape[1]:
                    continue
                units.append((i, i + m))
                feats.append(segment.glyph(ink, box))
        proba = self.model.clf.predict_proba(np.array(feats, np.float32))
        return units, {u: proba[k] for k, u in enumerate(units)}

    _union = staticmethod(lambda atoms, i, j: union_box(atoms, i, j))

    def _slot_best(self, p: np.ndarray, slot: str):
        idx = self._letter_idx if slot == "L" else self._digit_idx
        k = int(idx[int(np.argmax(p[idx]))])
        return self.model.classes[k], float(p[k])

    def _skip_costs(self, atoms) -> list[float]:
        """Per-atom cost of dropping it as noise: cheap for specks, dear for
        anything the size of a real character."""
        heights = sorted(a[3] for a in atoms)
        areas = sorted(a[2] * a[3] for a in atoms)
        hmed = heights[len(heights) // 2] or 1
        amed = areas[len(areas) // 2] or 1
        costs = []
        for (_, _, w, h) in atoms:
            small = (h < 0.62 * hmed) or (w * h < 0.42 * amed)
            costs.append(0.35 * self.skip_penalty if small else self.skip_penalty)
        return costs

    def _decode(self, atoms, units, probs):
        """DP over atom groupings, one pass per grammar pattern.

        dp[a][k] = best score for explaining atoms[0:a] as the first k characters
        of the pattern. Three moves are available at each step: take the next
        1..max_merge atoms as one character, or drop one atom as noise.
        """
        n = len(atoms)
        skip = self._skip_costs(atoms)
        best_overall = None
        for pattern in PATTERNS:
            K = len(pattern)
            if K > n or n > self.max_merge * K + 4:   # a pattern the atoms cannot support
                continue
            NEG = -1e9
            dp = np.full((n + 1, K + 1), NEG)
            back: dict = {}
            dp[0][0] = 0.0
            for a in range(n):
                for k in range(K + 1):
                    if dp[a][k] == NEG:
                        continue
                    drop = dp[a][k] - skip[a]
                    if drop > dp[a + 1][k]:
                        dp[a + 1][k] = drop
                        back[(a + 1, k)] = (a, k, None, None)
                    if k == K:
                        continue
                    for m in range(1, self.max_merge + 1):
                        u = (a, a + m)
                        if u not in probs:
                            continue
                        ch, p = self._slot_best(probs[u], pattern[k])
                        score = dp[a][k] + math.log(max(p, 1e-6)) + self.char_bonus
                        if score > dp[a + m][k + 1]:
                            dp[a + m][k + 1] = score
                            back[(a + m, k + 1)] = (a, k, ch, p)
            total = dp[n][K]
            if total == NEG:
                continue
            if best_overall is None or total > best_overall[0]:
                chars, confs, spans = [], [], []
                a, k = n, K
                while a > 0:
                    pa, pk, ch, p = back[(a, k)]
                    if ch is not None:
                        chars.append(ch)
                        confs.append(p)
                        spans.append((pa, a))
                    a, k = pa, pk
                best_overall = (total, "".join(reversed(chars)), list(reversed(confs)),
                                pattern, list(reversed(spans)))
        if best_overall is None:
            return None
        return best_overall   # (score, text, confs, pattern, spans)

    def _greedy(self, atoms, units, probs) -> str:
        """Unconstrained per-atom read — kept for the dashboard so an operator can
        see what the grammar actually fixed."""
        out = []
        for i in range(len(atoms)):
            p = probs.get((i, i + 1))
            if p is None:
                continue
            out.append(self.model.classes[int(np.argmax(p))])
        return "".join(out)

    def _repair_state(self, text: str, confs: list[float]) -> tuple[str, bool]:
        """Snap the first two characters to a real state code when they are not
        one and the fix is a single confusable substitution."""
        return repair_state_code(text)


# --- confusion --------------------------------------------------------
_CONF: dict[str, set[str]] = {}
for _a, _b in config.CONFUSION_PAIRS:
    _CONF.setdefault(_a, set()).add(_b)
    _CONF.setdefault(_b, set()).add(_a)


def _confusable(a: str, b: str) -> bool:
    return b in _CONF.get(a, ())


def confusion_distance(a: str, b: str) -> float:
    """Edit distance where a confusable substitution costs 0.4 instead of 1.

    Used by trajectory search: an operator typing KA05MJ1234 should still find
    the sighting a rain-blurred camera stored as KA05NJ1234.
    """
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0.0
            elif _confusable(a[i - 1], b[j - 1]):
                sub = 0.4
            else:
                sub = 1.0
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
    return prev[m]


# --- plate localisation ------------------------------------------------
import os

_yolo_model = None
def _load_yolo():
    global _yolo_model
    model_path = str(config.MODEL_DIR / "yolov8n-obb-plate.pt")
    if _yolo_model is None and os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(model_path)
        except ImportError:
            return _yolo_model
        # Move the weights once, here, rather than paying the host-to-device
        # copy on the first detection. Left to itself ultralytics picks a
        # device per predictor and would ignore GODSEYE_DEVICE=cpu on a
        # machine that has a card; every predict call below names the device
        # explicitly for the same reason.
        try:
            _yolo_model.to(config.torch_device())
        except Exception as exc:
            print(f"[anpr.ocr] YOLOv8-OBB stays on the CPU: {exc}", file=sys.stderr)
    return _yolo_model

def detect_plate_candidates(frame: np.ndarray, max_candidates: int = 12,
                            scales=None) -> list:
    """Locate plate-shaped regions in a wider photograph (YOLO or Classical).

    `scales` reaches the YOLO path only; the classical localiser runs its own
    two-scale sweep already. None means the single configured size, which is
    what every existing caller gets and what the benchmark measures.
    """
    yolo = _load_yolo()
    if yolo is not None:
        return _detect_plate_candidates_yolo(yolo, frame, max_candidates, scales)
    return _detect_plate_candidates_classical(frame, max_candidates)

# The OBB polygon behind each box from the most recent
# _detect_plate_candidates_yolo() call, keyed by the axis-aligned box tuple
# detect_plate_candidates() actually returns. detect_plate_candidates()'s own
# public contract stays a plain box list (benchmark.py, compare_backends.py
# both depend on that) - this side-channel is how _crop_variants() recovers
# the rotation that collapsing an OBB to its axis-aligned min/max throws
# away, without changing that contract. Single most-recent-call cache: safe
# because read_frame() consumes detect_plate_candidates()'s result
# synchronously, within the same call, before the next frame is processed.
_last_obb_polygons: dict[tuple, np.ndarray] = {}


def _detect_plate_candidates_yolo(yolo_model, frame: np.ndarray, max_candidates: int = 12,
                                  scales=None) -> list:
    """YOLOv8-based plate localiser (Supports OBB), optionally at several scales.

    `scales` are ultralytics `imgsz` values. The model has a narrow band of
    apparent plate sizes it responds to - measured in FIELD_TEST.md, one real
    photograph's plate scored 0.024 at imgsz=640, 0.569 at 1280 and 0.011 at
    1920, non-monotonically and swinging fiftyfold. There is no single correct
    input size, because the right one depends on how large the plate happens to
    be in the frame, which is the thing the detector is supposed to be finding
    out. Running a few sizes and pooling the boxes is the honest answer.

    Boxes from every scale go into one list and are de-duplicated by overlap,
    keeping the higher-ranked of any pair - so a plate found at two scales
    contributes once, at its best showing.
    """
    if scales is None:
        scales = (config.YOLO_IMGSZ,)
    boxes = []

    H, W = frame.shape[:2]
    # Cleared once for the whole call, not once per scale: the rectified-crop
    # side-channel is keyed by box, and clearing between scales would strip the
    # polygons off every box found by an earlier one.
    _last_obb_polygons.clear()

    for _size in scales:
        results = yolo_model(frame, verbose=False, device=config.torch_device(),
                             imgsz=int(_size))[0]
        _collect_yolo_boxes(results, boxes, W, H)

    # Highest area-weighted confidence first, then drop anything that mostly
    # repeats a box already kept. Overlap rather than exact equality because the
    # same plate found at two scales lands a pixel or two apart.
    boxes.sort(key=lambda t: -t[0])
    kept: list = []
    for score, b in boxes:
        if all(_iou(b, k) < 0.5 for k in kept):
            kept.append(b)
        if len(kept) >= max_candidates:
            break
    return kept


def _collect_yolo_boxes(results, boxes: list, W: int, H: int) -> None:
    """Append (rank, box) for one ultralytics result, and record its polygons."""
    if hasattr(results, 'obb') and results.obb is not None:
        for obb in results.obb:
            conf = float(obb.conf[0])
            coords = obb.xyxyxyxy[0].cpu().numpy()  # shape (4, 2)
            # int() each corner, not .astype(int): astype leaves numpy scalars,
            # and these coordinates travel all the way out to the API, where
            # FastAPI's JSON encoder cannot serialise a numpy.int64 and answers
            # 500. The branch below already does this with map(int, ...); the
            # OBB branch is the one YOLOv8-OBB actually takes, so it is the one
            # that showed the fault - POST /api/anpr/read failed on every
            # photograph in which a plate was found, and only on those.
            x1, y1 = (int(v) for v in np.min(coords, axis=0))
            x2, y2 = (int(v) for v in np.max(coords, axis=0))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append((conf * w * h, (x1, y1, w, h)))
                _last_obb_polygons[(x1, y1, w, h)] = coords
    else:
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append((conf * w * h, (x1, y1, w, h)))


def _detect_plate_candidates_classical(frame: np.ndarray, max_candidates: int = 12) -> list:
    """Locate plate-shaped regions in a wider photograph.

    Classical localiser, run at two scales: a plate is a horizontal band of
    tightly-spaced vertical strokes, so a morphological gradient followed by a
    wide closing joins the characters of a plate into one blob while leaving
    most of the scene apart. Candidates are filtered on aspect and fill,
    de-duplicated by overlap, and passed to the reader, which is the real
    arbiter - a region that does not decode is not a plate.

    This is the seam for a production upgrade: return boxes from a trained
    detector (YOLO, RetinaNet) here and nothing downstream changes.
    """
    gray = to_gray_frame(frame)
    if gray is None or gray.shape[0] < 30 or gray.shape[1] < 60:
        return []
    H, W = gray.shape
    scale = min(1.0, 900.0 / max(W, 1))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    inv = 1.0 / scale
    small = cv2.bilateralFilter(small, 5, 40, 40)
    h, w = small.shape

    boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    for kx, ky in ((21, 5), (35, 9), (13, 5)):
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 0.05 * w or bh < 8 or bw * bh < 700:
                continue
            ar = bw / max(bh, 1)
            if not (1.6 <= ar <= 7.5):                 # one-row and two-row plates
                continue
            fill = cv2.contourArea(c) / max(bw * bh, 1)
            if fill < 0.35:
                continue
            # edge density inside the box: characters are busy, a wall is not
            density = float((th[y:y + bh, x:x + bw] > 0).mean())
            if density < 0.12:
                continue
            pad_x, pad_y = int(0.03 * bw) + 2, int(0.12 * bh) + 2
            box = (max(0, int((x - pad_x) * inv)), max(0, int((y - pad_y) * inv)),
                   min(W, int((bw + 2 * pad_x) * inv)), min(H, int((bh + 2 * pad_y) * inv)))
            boxes.append((density * bw * bh, box))

    boxes.sort(key=lambda t: -t[0])
    kept: list[tuple[int, int, int, int]] = []
    for _, b in boxes:
        if all(_iou(b, k) < 0.35 for k in kept):
            kept.append(b)
        if len(kept) >= max_candidates:
            break
    return kept


def _crop_variants(frame: np.ndarray, box) -> list:
    """The candidate box, a tightened version, the bright plate face inside
    it, and - when the box came from a YOLOv8-OBB detection - a
    perspective-rectified crop that undoes the plate's rotation/skew."""
    x, y, w, h = box
    H, W = frame.shape[:2]
    out = [frame[y:y + h, x:x + w]]
    ix, iy = int(0.06 * w), int(0.12 * h)
    if w - 2 * ix > 24 and h - 2 * iy > 10:
        out.append(frame[y + iy:y + h - iy, x + ix:x + w - ix])
    face = _plate_face(frame, box)
    if face is not None:
        fx, fy, fw, fh = face
        if fw > 24 and fh > 10:
            out.append(frame[fy:fy + fh, fx:fx + fw])
    warped = _obb_rectified_crop(frame, box)
    if warped is not None:
        out.append(warped)
    return out


def _obb_rectified_crop(frame: np.ndarray, box):
    """Perspective-warp the OBB polygon behind `box` (if any, from the most
    recent YOLOv8-OBB detection) onto a flat rectangle, undoing rotation an
    axis-aligned crop keeps. None when the box wasn't from an OBB detection,
    or the polygon is close enough to axis-aligned that warping buys nothing.
    """
    poly = _last_obb_polygons.get(tuple(box))
    if poly is None:
        return None
    pts = np.asarray(poly, dtype=np.float32)
    # Order corners by angle around the centroid so opposite edge lengths
    # below actually measure width/height rather than a diagonal.
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    pts = pts[order]

    def edge(i, j):
        return float(np.linalg.norm(pts[i] - pts[j]))

    w = max(edge(0, 1), edge(2, 3))
    h = max(edge(1, 2), edge(3, 0))
    if w < 24 or h < 10:
        return None
    # Skip near-axis-aligned boxes: the plain crop already covers them, and
    # warping a near-rectangle just adds interpolation noise for nothing.
    axis_aligned = np.array([[pts[:, 0].min(), pts[:, 1].min()],
                             [pts[:, 0].max(), pts[:, 1].min()],
                             [pts[:, 0].max(), pts[:, 1].max()],
                             [pts[:, 0].min(), pts[:, 1].max()]], dtype=np.float32)
    if float(np.abs(pts - axis_aligned).max()) < 0.03 * max(w, h):
        return None
    if w < h:  # keep the wide side horizontal, matching a plate's aspect
        pts = np.roll(pts, 1, axis=0)
        w, h = h, w
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (int(w), int(h)))


def _plate_face(frame: np.ndarray, box):
    """Tighten a box onto the plate's own bright rectangle, if there is one."""
    x, y, w, h = box
    crop = segment.to_gray(frame[y:y + h, x:x + w])
    if crop.size == 0 or min(crop.shape) < 12:
        return None
    _, th = cv2.threshold(cv2.GaussianBlur(crop, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw < 0.45 * w or ch < 0.35 * h:
            continue
        ar = cw / max(ch, 1)
        if not (1.4 <= ar <= 8.0):
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (cx, cy, cw, ch)
    if best is None:
        return None
    cx, cy, cw, ch = best
    pad = 2
    return (max(0, x + cx - pad), max(0, y + cy - pad), cw + 2 * pad, ch + 2 * pad)


def to_gray_frame(frame: np.ndarray):
    if frame is None or frame.size == 0:
        return None
    return segment.to_gray(frame)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --- module-level singletons ------------------------------------------
# One engine per backend, built on first use and kept. The dashboard, the API
# (api/state.py) and the simulator (sim/city.py) all obtain their engine here,
# so this is where the platform-wide choice of recogniser is made.
_engines: dict[str, ANPREngine] = {}
_engine_reasons: dict[str, str] = {}
# Guards the build, not the use. warm() runs on a background thread, so a
# build can now be in flight while the API answers a request on another - and
# building twice would load the CRNN and the detector twice, which is slow
# enough to be exactly the stall the warm-up exists to remove.
_engine_lock = threading.RLock()


def engine(backend: str | None = None) -> ANPREngine:
    """The shared engine. `backend` defaults to config.OCR_BACKEND.

    Each backend is built once and cached. That caching is not an
    optimisation: building anpr_ocr loads the YOLO detector and the CRNN
    before it touches paddle, on purpose (see the load-order note in
    anpr/anpr_ocr_reader.py), so rebuilding it per call costs tens of seconds
    every time and re-runs that hazard.
    """
    name = (backend or config.OCR_BACKEND or "crnn").strip().lower()
    if name in _engines:                      # fast path, no lock
        return _engines[name]
    with _engine_lock:
        if name not in _engines:              # re-check: another thread may have won
            built, why = _build_engine(name)
            _engines[name], _engine_reasons[name] = built, why
            if why:
                # Loud on purpose. A reader that quietly degrades to a different
                # one is the most confusing failure this engine has: the platform
                # keeps working and every number silently changes.
                print(f"[anpr.ocr] {why}", file=sys.stderr)
    return _engines[name]


def warm(backend: str | None = None, verbose: bool = False) -> float:
    """Pay every first-call cost now, so nothing pays it mid-demo. Seconds taken.

    Measured on this project's GTX 1650, the first read of a whole frame takes
    3.3 seconds and the ones after it take 64 ms - fifty times faster. Nothing
    about the second read is cheaper; the first one is simply where all the
    deferred work lands. Roughly nine seconds separate process start from first
    result, and it breaks down as:

        2.7s  CUDA context creation, on the first torch call that needs a device
        1.8s  building the engine - CRNN weights, the classifier, the detector
        3.3s  ultralytics' first predict: it builds its predictor, allocates
              buffers and lets cuDNN autotune the convolutions
        0.1s  the CRNN's own first forward pass

    All of it is one-off and none of it is avoidable - it can only be moved.
    Called on a background thread at start-up it lands while the operator is
    still looking at the map, instead of on the first press of Capture & read.

    Deliberately silent about failure. This is preparation, not work: if a
    dummy read throws, the real read will throw the same way and report it
    properly, and a warm-up that could break the platform would be a poor
    trade for the seconds it saves.
    """
    started = time.perf_counter()
    eng = engine(backend)

    # Plate-crop proportions and a whole-frame shape, so the shapes the real
    # captures use are the ones cuDNN autotunes and allocates for.
    try:
        eng.read(np.full((64, 288, 3), 127, dtype=np.uint8))
    except Exception:
        pass
    try:
        eng.read_frame(np.full((704, 1120, 3), 127, dtype=np.uint8))
    except Exception:
        pass

    taken = time.perf_counter() - started
    if verbose:
        print(f"[anpr.ocr] reader warm in {taken:.1f}s "
              f"({(backend or config.OCR_BACKEND or 'crnn')}, "
              f"{_reader_device(eng)})", file=sys.stderr)
    return taken


def _reader_device(eng: ANPREngine) -> str:
    """Where this reader's arithmetic actually happens.

    Not config.torch_device() in every case, which is what this line used to
    print and what made it lie. anpr_ocr computes in paddle, and the two
    frameworks are installed separately - .venv-paddle pairs a CPU-only torch
    with a CUDA paddle, so torch's answer there is "cpu" while every plate is
    in fact being read on the GPU. A start-up line saying "cpu" on a machine
    using its GPU is worse than no line at all: it is the exact evidence
    someone would use to conclude the GPU install had failed.
    """
    if eng.anpr_ocr is not None:
        try:
            import paddle
            return f"paddle:{paddle.device.get_device()}"
        except Exception:
            return "paddle:unknown"
    return config.torch_device()


def warm_async(backend: str | None = None, verbose: bool = True) -> threading.Thread:
    """Warm the reader on a daemon thread and hand the thread back.

    A daemon so it can never hold a shutdown open: the worst case is a process
    exiting mid-warm, and a half-warmed reader is only a slow one.
    """
    thread = threading.Thread(target=warm, args=(backend, verbose),
                              name="anpr-warm", daemon=True)
    thread.start()
    return thread


def engine_status(backend: str | None = None) -> str:
    """Empty if `backend` is the reader actually running, otherwise a
    plain-language reason it is not. Built for the dashboard to surface."""
    name = (backend or config.OCR_BACKEND or "crnn").strip().lower()
    if name not in _engines:
        engine(name)
    return _engine_reasons.get(name, "")


def _build_engine(name: str) -> tuple[ANPREngine, str]:
    """(engine, reason it is not the backend that was asked for)."""
    if name == "classical":
        return ANPREngine(use_crnn=False), ""
    if name == "anpr_ocr":
        # Imported here rather than at module scope: anpr_ocr_reader imports
        # PlateRead from this module, so a top-level import would be circular.
        from anpr import anpr_ocr_reader
        # available() imports paddle, and paddle imported before torch breaks
        # torch for the rest of the process - so torch's models go in first.
        # It has to happen here rather than inside load(): by the time load()
        # runs, available() has already pulled paddle in.
        anpr_ocr_reader.preload_torch()
        if not anpr_ocr_reader.available():
            return ANPREngine(), (
                "anpr_ocr was requested but paddle is not installed in this "
                "interpreter, so the CRNN is being used instead. paddle needs "
                "Python 3.9-3.12; run from .venv-paddle. See PADDLE_SETUP.md.")
        reader = anpr_ocr_reader.load()
        if reader is None:
            return ANPREngine(), (
                "anpr_ocr was requested but failed to load - most likely the "
                "weights under models/anpr_ocr/ are missing, or the PaddleOCR "
                "clone is incomplete. The CRNN is being used instead. See "
                "PADDLE_SETUP.md.")
        # The CRNN is deliberately still loaded underneath: core/repair.py
        # needs its CTC lattices through read_evidence(), which anpr_ocr does
        # not produce.
        return ANPREngine(anpr_ocr_reader=reader), ""
    if name != "crnn":
        return ANPREngine(), (
            f"unknown OCR backend {name!r} - expected one of crnn, anpr_ocr, "
            "classical. Using the CRNN.")
    return ANPREngine(), ""
