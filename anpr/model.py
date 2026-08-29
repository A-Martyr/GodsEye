"""Glyph classifier — the learned half of the ANPR engine.

Trained in two stages, both on synthetic plates with known ground truth:

* **Stage 1 — segmentation-aligned.** Plates are pushed through the exact
  inference-time segmenter and kept only when it produced as many boxes as the
  plate has characters, so every crop's label is certain. Cheap, but it only
  ever sees plates the segmenter already handles well, which biases the
  classifier away from the hard conditions.

* **Stage 2 — decoder-aligned self-training.** The stage-1 engine reads a fresh
  batch of plates; where the decoded string matches ground truth, the decoder's
  own character spans are harvested as labelled crops. This recovers exactly the
  cases stage 1 threw away — merged characters, mud-covered strokes, fragments
  glued back together — and is what lifts accuracy on the dirty and damaged
  conditions.

Training takes 3-5 minutes on a laptop CPU and is cached in models/.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import joblib
import numpy as np

import config
from anpr import ocr, plates, segment


@dataclass
class GlyphModel:
    clf: object
    classes: np.ndarray
    trained_on: int
    val_accuracy: float
    stages: int = 1

    def predict(self, feats: np.ndarray):
        """-> (chars, confidences) for a stack of glyph feature vectors."""
        if len(feats) == 0:
            return [], np.zeros(0)
        proba = self.clf.predict_proba(feats)
        idx = proba.argmax(axis=1)
        return [self.classes[i] for i in idx], proba[np.arange(len(idx)), idx]


def _plate_text(rng: random.Random) -> str:
    """Mostly real Bharat-series plates, plus a slice using the letters the
    series never issues (I, O) so the classifier still knows them."""
    text = plates.random_plate(rng)
    if rng.random() < 0.08:
        head = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(2))
        text = head + text[2:]
    if rng.random() < 0.08:
        n = sum(c.isalpha() for c in text[4:-4]) or 1
        mid = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
        text = text[:4] + mid + text[-4:]
    return text


def build_stage1(n_plates: int = 3000, seed: int = 7, verbose: bool = True):
    """Segmentation-aligned crops: keep plates the segmenter cut cleanly."""
    rng = random.Random(seed)
    X, y = [], []
    kept = 0
    for i in range(n_plates):
        text = _plate_text(rng)
        cond = plates.CONDITIONS[i % len(plates.CONDITIONS)]
        cap = plates.capture(text, cond, rng)
        _, ink, boxes, feats = segment.plate_glyphs(cap.image)
        if len(boxes) != len(text):
            continue
        kept += 1
        X.extend(feats)
        y.extend(list(text))
        if verbose and (i + 1) % 750 == 0:
            print(f"  stage 1: {i+1}/{n_plates} plates, {kept} aligned, {len(y)} glyphs")
    return X, y


def build_stage2(eng, n_plates: int = 2200, seed: int = 21, verbose: bool = True):
    """Decoder-aligned crops harvested from correctly-read hard plates."""
    rng = random.Random(seed)
    X, y = [], []
    kept = 0
    # weight towards the conditions stage 1 under-samples
    pool = (["dirty"] * 4 + ["damaged"] * 3 + ["angled"] * 3 + ["mixed"] * 3 +
            ["glare"] * 2 + ["motion_blur"] * 2 + ["rain", "night", "low_res", "clean"])
    for i in range(n_plates):
        text = _plate_text(rng)
        cap = plates.capture(text, rng.choice(pool), rng)
        read, dbg = eng.read_detailed(cap.image)
        if read.text != cap.text or dbg.get("ink") is None:
            continue
        kept += 1
        ink, atoms, spans = dbg["ink"], dbg["atoms"], dbg["spans"]
        for ch, (a, b) in zip(read.text, spans):
            X.append(segment.glyph(ink, ocr.union_box(atoms, a, b)))
            y.append(ch)
        if verbose and (i + 1) % 500 == 0:
            print(f"  stage 2: {i+1}/{n_plates} plates, {kept} decoded correctly, {len(y)} glyphs")
    return X, y


def _fit(X, y, seed: int, verbose: bool):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X = np.array(X, np.float32)
    y = np.array(y)
    if len(X) < 500:
        raise RuntimeError(f"only {len(X)} glyphs available - check the segmenter")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.12, random_state=seed, stratify=y)
    if verbose:
        print(f"[glyph] training MLP on {len(Xtr)} glyphs ({len(set(y))} classes) ...")
    clf = MLPClassifier(hidden_layer_sizes=(384, 192), activation="relu", alpha=1e-4,
                        batch_size=256, learning_rate_init=2e-3, max_iter=110,
                        early_stopping=True, n_iter_no_change=8, random_state=seed)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xva, yva))
    if verbose:
        print(f"[glyph] held-out glyph accuracy {acc:.4f}")
    return clf, acc, len(Xtr)


def train(n_plates: int = 3000, seed: int = 7, self_train: bool = True,
          harvest_plates: int = 2200, verbose: bool = True) -> GlyphModel:
    from anpr.ocr import ANPREngine

    t0 = time.time()
    if verbose:
        print(f"[glyph] stage 1: synthesising {n_plates} plates across "
              f"{len(plates.CONDITIONS)} conditions ...")
    X, y = build_stage1(n_plates, seed, verbose)
    clf, acc, n = _fit(X, y, seed, verbose)
    model = GlyphModel(clf=clf, classes=np.array(clf.classes_), trained_on=n,
                       val_accuracy=acc, stages=1)
    if not self_train:
        return model

    if verbose:
        print(f"[glyph] stage 2: harvesting decoder-aligned crops from {harvest_plates} hard plates ...")
    X2, y2 = build_stage2(ANPREngine(model), harvest_plates, seed + 14, verbose)
    if len(X2) < 500:
        print("[glyph] stage 2 harvested too little - keeping the stage 1 model")
        return model
    clf, acc, n = _fit(X + X2, y + y2, seed, verbose)
    if verbose:
        print(f"[glyph] done in {time.time()-t0:.1f}s "
              f"({len(X)} stage-1 + {len(X2)} stage-2 glyphs)")
    return GlyphModel(clf=clf, classes=np.array(clf.classes_), trained_on=n,
                      val_accuracy=acc, stages=2)


def save(model: GlyphModel, path=None) -> None:
    path = path or config.GLYPH_MODEL
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_or_train(force: bool = False, **kw) -> GlyphModel:
    path = config.GLYPH_MODEL
    if path.exists() and not force:
        try:
            return joblib.load(path)
        except Exception as exc:       # corrupt or version-skewed cache
            print(f"[glyph] could not load {path} ({exc}); retraining")
    model = train(**kw)
    save(model)
    return model


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the GodsEye glyph classifier")
    ap.add_argument("--plates", type=int, default=3000)
    ap.add_argument("--harvest", type=int, default=2200)
    ap.add_argument("--no-self-train", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    m = train(args.plates, args.seed, not args.no_self_train, args.harvest)
    save(m)
    print(f"saved -> {config.GLYPH_MODEL}  (stages {m.stages}, glyph val acc {m.val_accuracy:.4f})")
