"""Turn a CTC score into a probability that the read is right.

The reader's confidence is an exact CTC likelihood - the probability the image
evidence would have produced that string, marginalised over every alignment.
It is a good ORDERING: sort reads by it and the correct ones come first
(AUROC 0.974). It is not a probability of being correct, and it has never
claimed to be.

That gap shows up wherever a number is compared against a constant.
config.CLONE_MIN_CONFIDENCE is 0.70 and config.MIN_PLATE_CONFIDENCE_ANPR_OCR is
another; both are thresholds on a scale nobody can state the meaning of. "We
store reads above 0.70" invites the question "0.70 of what", and the honest
answer today is "of a quantity that is monotone in correctness".

Isotonic regression fixes that without touching the model. It fits the single
best non-decreasing map from score to observed correctness on labelled data -
two hundred-odd steps, no parameters to choose, no assumption about the shape of
the curve beyond the one thing we actually know: higher score should never mean
less likely to be right.

    raw 0.83  ->  p_correct 0.91

The threshold can then be stated in the units an operator cares about, and the
same sentence works for every backend even though their raw scores are on
incomparable scales.

Deliberately REPORTING only. Nothing in the platform changes its behaviour
because of this file - no floor moves, no read is accepted that was not accepted
before. A calibration is a claim about a corpus, and moving a storage threshold
onto it would silently retune the whole platform to the 532 crops it was fitted
on. It is here to be quoted, and to make the next threshold argument an
empirical one.
"""
from __future__ import annotations

import bisect
import json

import config

_MODEL_PATH = config.MODEL_DIR / "confidence_calibration.json"
_cache: dict | None = None


# --- fitting ------------------------------------------------------------
def _pava(ys: list[float], ws: list[float]) -> list[float]:
    """Pool-adjacent-violators: the exact isotonic fit, in one pass.

    Walks left to right maintaining blocks that are already non-decreasing.
    Whenever a new point is lower than the block before it the two are merged
    and replaced by their weighted mean, which may violate the block before
    that, so merging repeats backwards. Each point joins and leaves a block at
    most once, so it is linear despite the inner loop.
    """
    vals: list[float] = []
    wts: list[float] = []
    counts: list[int] = []
    for y, w in zip(ys, ws):
        vals.append(y)
        wts.append(w)
        counts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), counts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), counts.pop()
            w = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / w if w else 0.0)
            wts.append(w)
            counts.append(c1 + c2)
    out: list[float] = []
    for v, c in zip(vals, counts):
        out.extend([v] * c)
    return out


def fit(samples: list[tuple[float, bool]], bins: int = 40) -> dict:
    """Fit score -> P(correct) from (confidence, was_correct) pairs.

    Binned before the isotonic fit rather than run on raw points, for a reason
    worth stating: on raw points PAVA can return long flat runs at exactly 0 or
    1, and a step that says "this score means certainly wrong" is a claim no
    finite sample supports. Binning first, and smoothing each bin by Laplace,
    keeps every step strictly inside (0, 1).
    """
    pts = sorted((float(c), bool(ok)) for c, ok in samples)
    if len(pts) < 20:
        raise ValueError(f"not enough labelled reads to calibrate: {len(pts)}")

    per = max(1, len(pts) // bins)
    edges: list[float] = []
    rates: list[float] = []
    weights: list[float] = []
    for i in range(0, len(pts), per):
        chunk = pts[i:i + per]
        if not chunk:
            continue
        n = len(chunk)
        k = sum(1 for _, ok in chunk if ok)
        edges.append(chunk[0][0])
        rates.append((k + 0.5) / (n + 1.0))      # Laplace: never 0, never 1
        weights.append(float(n))

    fitted = _pava(rates, weights)
    return {
        "edges": edges,
        "p": fitted,
        "n": len(pts),
        "base_rate": sum(1 for _, ok in pts if ok) / len(pts),
        "note": "isotonic score -> P(correct); reporting only, no threshold uses it",
    }


def save(model: dict, path=None) -> None:
    p = path or _MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model, indent=1), encoding="utf-8")
    global _cache
    _cache = None


# --- applying -----------------------------------------------------------
def load(path=None) -> dict | None:
    global _cache
    if _cache is not None:
        return _cache
    p = path or _MODEL_PATH
    try:
        _cache = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _cache = None
    return _cache


def p_correct(confidence: float) -> float | None:
    """P(this read is exactly right), or None when no calibration is fitted.

    None rather than the raw score, so a caller can tell "we have not measured
    this" from "we measured it and it is 0.83". Silently returning the
    uncalibrated number would be the one outcome worse than returning nothing.
    """
    model = load()
    if not model or not model.get("edges"):
        return None
    edges = model["edges"]
    i = bisect.bisect_right(edges, float(confidence)) - 1
    if i < 0:
        return float(model["p"][0])
    return float(model["p"][min(i, len(model["p"]) - 1)])


def reliability(samples: list[tuple[float, bool]], model: dict | None = None,
                bins: int = 10) -> dict:
    """How well the calibration holds: Brier score and a reliability table.

    Brier is the mean squared error between the predicted probability and the
    outcome, so lower is better and it rewards being both confident and right.
    Quoted against the base rate, which is what a model that ignores the score
    entirely would achieve - a calibration that does not beat that is not
    telling you anything the corpus average does not.
    """
    model = model or load()
    if not model:
        return {}
    preds = []
    for conf, ok in samples:
        i = bisect.bisect_right(model["edges"], float(conf)) - 1
        p = model["p"][max(0, min(i, len(model["p"]) - 1))]
        preds.append((float(p), bool(ok)))
    brier = sum((p - ok) ** 2 for p, ok in preds) / len(preds)
    base = model.get("base_rate", sum(1 for _, ok in preds if ok) / len(preds))
    brier_base = sum((base - ok) ** 2 for _, ok in preds) / len(preds)

    preds.sort()
    table = []
    per = max(1, len(preds) // bins)
    for i in range(0, len(preds), per):
        chunk = preds[i:i + per]
        if len(chunk) < 3:
            continue
        table.append({
            "predicted": round(sum(p for p, _ in chunk) / len(chunk), 3),
            "observed": round(sum(1 for _, ok in chunk if ok) / len(chunk), 3),
            "n": len(chunk),
        })
    return {"brier": round(brier, 4), "brier_base_rate": round(brier_base, 4),
            "skill": round(1.0 - brier / brier_base, 3) if brier_base else 0.0,
            "table": table}
