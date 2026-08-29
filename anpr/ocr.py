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
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from anpr import model as glyph_model
from anpr import segment

def union_box(atoms, i, j):
    """Bounding box of atoms[i:j] — one character may span several atoms."""
    x = min(atoms[k][0] for k in range(i, j))
    y = min(atoms[k][1] for k in range(i, j))
    x2 = max(atoms[k][0] + atoms[k][2] for k in range(i, j))
    y2 = max(atoms[k][1] + atoms[k][3] for k in range(i, j))
    return (x, y, x2 - x, y2 - y)


LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

# Valid Bharat-series shapes: 2 state letters, 1-2 district digits,
# 0-3 series letters, 4 registration digits.
PATTERNS: list[str] = [
    "LL" + "D" * d + "L" * s + "DDDD"
    for d in (1, 2)
    for s in (3, 2, 1, 0)
]

# Real Indian state / UT codes, used to repair the first two characters.
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB", "BH",
}


@dataclass
class PlateRead:
    text: str                       # decoded plate, no spaces ("" if unreadable)
    confidence: float               # 0-1, the weakest character's confidence
    mean_confidence: float = 0.0    # geometric mean over the characters
    char_confidences: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    pattern: str = ""
    raw: str = ""                   # greedy per-atom read, before the decoder
    repaired: bool = False          # True if the state code was corrected
    plate_found: bool = True
    variant: str = ""               # which binarisation hypothesis won
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
            "char_confidences": [round(c, 3) for c in self.char_confidences],
            "boxes": self.boxes, "plate_found": self.plate_found, "variant": self.variant,
            "candidates": self.candidates, "reason": self.reason,
        }


class ANPREngine:
    """Loads (or trains) the glyph model once and reads plates."""

    def __init__(self, model: glyph_model.GlyphModel | None = None, max_merge: int = 3,
                 char_bonus: float = 0.4, skip_penalty: float = 1.6):
        self.model = model or glyph_model.load_or_train()
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
        """Read a plate crop (grayscale or BGR).

        Every binarisation hypothesis is decoded and the highest-scoring read
        wins, so one bad threshold no longer costs the plate.
        """
        norm = segment.normalize(image)
        self._last_debug = {"norm": norm, "ink": None, "atoms": []}
        best = None
        atoms_seen = []
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
        text, repaired = self._repair_state(text, confs)
        boxes = [self._union(atoms, a, b) for a, b in spans]
        # A plate is only as good as its worst character: one wrong glyph makes
        # the whole registration wrong, so the weakest character sets the
        # confidence. Measured against the mean, this separates correct from
        # incorrect reads far better (0.98 vs 0.64, against 1.00 vs 0.91) and is
        # what makes the confidence floor able to reject bad reads at all.
        clipped = np.clip(np.asarray(confs, dtype=float), 1e-6, 1.0)
        conf = float(clipped.min())
        return PlateRead(text=text, confidence=conf, char_confidences=list(confs),
                         mean_confidence=float(np.exp(np.mean(np.log(clipped)))),
                         boxes=boxes, pattern=pattern, raw=raw, repaired=repaired,
                         variant=variant)

    def read_detailed(self, image: np.ndarray):
        """read() plus the intermediate images, for the ANPR Lab view and for
        the self-training harvester."""
        read = self.read(image)
        return read, self._last_debug

    def read_frame(self, frame: np.ndarray, max_candidates: int = 12) -> PlateRead:
        """Read from a full photograph rather than a tight plate crop.

        Every candidate region the localiser proposes is decoded, plus the whole
        frame in case the image already *is* a crop, and the most confident
        legal read wins. When nothing wins, the returned read carries the
        candidate boxes and a plain-language reason, because "no plate found" on
        its own tells an operator nothing about what to try next.
        """
        cands = detect_plate_candidates(frame, max_candidates)
        best = self.read(frame)
        best.candidates = cands
        tried = 1
        for (x, y, w, h) in cands:
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            tried += 1
            r = self.read(crop)
            if r.text and r.confidence > best.confidence:
                r.candidates = cands
                best = r
        if not best.text:
            best.plate_found = False
            best.candidates = cands
            best.reason = (
                f"examined {tried} plate-shaped region(s) in this image and none of them "
                "contained a readable vehicle registration. This engine reads number "
                "plates - two letters, district digits, a series and four digits - not "
                "arbitrary text or handwriting.")
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
def detect_plate_candidates(frame: np.ndarray, max_candidates: int = 12) -> list:
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


# --- module-level singleton -------------------------------------------
_engine: ANPREngine | None = None


def engine() -> ANPREngine:
    global _engine
    if _engine is None:
        _engine = ANPREngine()
    return _engine
