"""Awiros/anpr-ocr backend — pretrained specifically on 558,767 real Indian
license plate samples (Apache 2.0, huggingface.co/Awiros/anpr-ocr; PP-OCRv5,
PPHGNetV2_B4 backbone, 37.3M params), paired with our existing YOLOv8-OBB
plate detector. Evaluation-only, same opt-in pattern as
anpr/easyocr_reader.py — not wired into any default engine path.

Preprocessing, architecture config, and decode logic mirror the model's own
reference `test.py` (fetched from the HF repo) as closely as possible - this
is a straight port, not a reinterpretation, since the weights were trained
against that exact preprocessing.

Unlike EasyOCR, this model does no text detection of its own: it expects a
pre-cropped plate image and returns exactly one decoded string, so
read_frame()'s YOLOv8 candidate crops feed it directly with no adaptation.

`ppocr` (PaddleOCR's internal package, needed to construct this architecture)
isn't published standalone on PyPI - the model's own test.py bootstraps it by
git-cloning github.com/PaddlePaddle/PaddleOCR (Apache 2.0) into a sibling
folder, and `_ensure_paddleocr()` below does the same, cached under
models/anpr_ocr/PaddleOCR/ alongside the downloaded weights.
"""
from __future__ import annotations

import copy
import math
import os
import subprocess
import sys

import cv2
import numpy as np

import config
from anpr import crnn as crnn_mod
from anpr.ocr import (PlateRead, matches_plate_grammar, plate_plausibility,
                      plate_shape, repair_state_code)

# How close two candidates' confidences must be before plausibility is allowed
# to choose between them. Deliberately tiny: this is a tie-break, not a prior.
# A candidate the image evidence clearly prefers must win even if it names a
# district that does not exist, because the alternative is a reader that argues
# with the photograph.
PLAUSIBILITY_TIE_TOLERANCE = 0.02

# Set before paddle is imported anywhere in the process, which is why it is at
# module scope and not inside load(). Paddle reads these once, when its runtime
# initialises, and pins its thread pools then; setting them afterwards is
# silently ignored. Its own default is one thread, which leaves a 16-core
# machine reading one plate on one core - see config.ANPR_OCR_THREADS for the
# measured curve and why the answer is not "as many as possible".
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "FLAGS_paddle_num_threads"):
    os.environ.setdefault(_var, str(config.ANPR_OCR_THREADS))

# The weights are already on disk after the first run, but huggingface_hub
# still round-trips to the network to revalidate them on every load, which
# costs seconds before a single plate is read - and hangs far longer on the
# venue wi-fi this platform is otherwise built to survive. Going offline once
# the files are present keeps the cache and drops the round-trip; if they are
# missing we stay online so the first-run download still works.
if (config.MODEL_DIR / "anpr_ocr" / "model.safetensors").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

_REPO_ID = "Awiros/anpr-ocr"
_CACHE_DIR = str(config.MODEL_DIR / "anpr_ocr")
_PADDLEOCR_DIR = os.path.join(_CACHE_DIR, "PaddleOCR")

# PP-OCRv5 server rec, SVTR_HGNet. CTC head: 64 classes (63 dict chars +
# blank). NRTR head: 67 -> 68 internally (NRTRHead adds +1) to match the
# shipped weights. Copied verbatim from the model's own test.py.
CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr", "dims": 120, "depth": 2,
                            "hidden_dims": 120, "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-05},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    },
}
IMAGE_SHAPE = [3, 48, 320]


def available() -> bool:
    """Is paddle importable? The platform runs fine without it - this
    backend just isn't offered."""
    try:
        import paddle  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_paddleocr() -> None:
    """Make `ppocr` importable, cloning PaddlePaddle/PaddleOCR (Apache 2.0)
    the same way the model's own test.py bootstraps itself on first run."""
    if os.path.isfile(os.path.join(_PADDLEOCR_DIR, "ppocr", "__init__.py")):
        if _PADDLEOCR_DIR not in sys.path:
            sys.path.insert(0, _PADDLEOCR_DIR)
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"[anpr_ocr_reader] cloning PaddleOCR into {_PADDLEOCR_DIR} (first run only) ...",
          file=sys.stderr)
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/PaddlePaddle/PaddleOCR.git", _PADDLEOCR_DIR,
    ])
    sys.path.insert(0, _PADDLEOCR_DIR)


def _ensure_weights() -> tuple[str, str]:
    """Download model.safetensors + en_dict.txt from the HF repo, cached
    under models/anpr_ocr/ - untracked, like the platform's other large
    model files (see .gitignore)."""
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(_REPO_ID, "model.safetensors", local_dir=_CACHE_DIR)
    dict_path = hf_hub_download(_REPO_ID, "en_dict.txt", local_dir=_CACHE_DIR)
    return weights, dict_path


def _resize_for_rec(img_bgr: np.ndarray, target_shape) -> np.ndarray:
    _, h, w = target_shape
    img_h, img_w = img_bgr.shape[:2]
    ratio = h / img_h
    new_w = min(int(img_w * ratio), w)
    resized = cv2.resize(img_bgr, (new_w, h))
    if new_w < w:
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized
        resized = padded
    return resized


def _preprocess(img_bgr: np.ndarray, target_shape) -> np.ndarray:
    img = _resize_for_rec(img_bgr, target_shape)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose((2, 0, 1))


# --- grammar-constrained decoding --------------------------------------
#
# The shipped model reports 98.42% exact-match on its own held-out set, but
# ppocr's CTCLabelDecode is a plain argmax over 64 classes with no notion of
# what a registration looks like. On this repo's real corpus that argmax
# routinely emits the state sticker as extra characters ("MH02ER9194JH", six
# frames out of thirteen in one cluster) or drops a trailing digit
# ("MH01AX113"). Both are recoverable without touching the weights: the
# network's own posterior already ranks the correct string highly, it just
# does not win the greedy path.
#
# So: fold the 64 classes onto the 36-character plate alphabet, run a CTC
# prefix beam search, propose every grammatical reading each beam admits, and
# rank those by the *exact* CTC score of the whole string - the same
# forward-backward recursion anpr/crnn.py uses, borrowed rather than
# reimplemented (crnn.ctc_score takes the charset as a parameter for exactly
# this). An ungrammatical read is still returned when nothing grammatical is
# on the beam, so an unusual-format plate degrades to today's behaviour
# instead of vanishing.

PLATE_ALPHABET = config.ALPHABET                 # "0123456789A..Z"
PLATE_IDX = {c: i + 1 for i, c in enumerate(PLATE_ALPHABET)}   # 0 is blank
PLATE_BLANK = 0
NEG = -1e30

BEAM_WIDTH = 16
BEAM_TOP_K = 6          # classes considered per column
MAX_PREFIX = 14         # longest plate is 11; the slack is what substrings eat


def _fold_probs(probs: np.ndarray, characters: list[str]) -> np.ndarray:
    """Collapse ppocr's 64 classes onto blank + the 36 plate characters.

    `characters` is ['blank', '0'..'9', 'A'..'Z', 'a'..'z', ' ']. A plate is
    uppercase, so probability the model splits between 'B' and 'b' is the same
    evidence and belongs in one column; space carries no plate information and
    behaves exactly like a separator, so it folds into blank. Doing this before
    the search rather than upper()-ing the result afterwards is what stops a
    case-split from pushing the wrong character to the top of a column.
    """
    T = probs.shape[0]
    out = np.zeros((T, len(PLATE_ALPHABET) + 1), dtype=np.float64)
    for j, ch in enumerate(characters):
        if j >= probs.shape[1]:
            break
        col = probs[:, j]
        up = ch.upper()
        if up in PLATE_IDX:
            out[:, PLATE_IDX[up]] += col
        else:                            # 'blank', ' ', anything unexpected
            out[:, PLATE_BLANK] += col
    total = out.sum(axis=1, keepdims=True)
    return out / np.maximum(total, 1e-12)


def _greedy(folded: np.ndarray) -> str:
    """Collapsed argmax path - what CTCLabelDecode would have returned."""
    path = folded.argmax(axis=1)
    out, prev = [], -1
    for k in path:
        if k != prev and k != PLATE_BLANK:
            out.append(PLATE_ALPHABET[k - 1])
        prev = k
    return "".join(out)


def _logaddexp(a: float, b: float) -> float:
    """np.logaddexp for two plain floats, without numpy in the way.

    Identical arithmetic - both compute max + log1p(exp(min - max)) through the
    same libm - but this is a function call instead of a ufunc dispatch, and
    that is the whole point. The beam below evaluates this tens of thousands of
    times per crop on single scalars, where numpy's per-call overhead is
    several times the cost of the flop it is wrapping.

    The -700 guard is where exp underflows to zero in double precision; past it
    the smaller term cannot affect the sum, and skipping it also keeps exp()
    from being called with a large negative argument in the common case of one
    beam dominating.
    """
    if a < b:
        a, b = b, a
    if b <= NEG or b == -math.inf:
        return a
    d = b - a
    if d < -700.0:
        return a
    return a + math.log1p(math.exp(d))


def _prefix_beam(folded: np.ndarray, width: int = BEAM_WIDTH,
                 top_k: int = BEAM_TOP_K) -> list[str]:
    """CTC prefix beam search. Returns the surviving strings, best first.

    Written to stay out of numpy once the per-timestep row has been chosen.
    This search was 47% of a whole read on the GPU - larger than every paddle
    forward pass in the same read put together - not because the algorithm is
    expensive but because every scalar addition inside three nested Python
    loops was going through a numpy ufunc, and every `row[k]` was materialising
    a numpy scalar. Converting the row to plain floats once per timestep and
    using _logaddexp made it 7.1x faster and returned, on real posteriors from
    the sample photographs, the identical strings in the identical order.

    The argpartition stays: picking the top_k of 37 classes is real vector work
    and is what numpy is good at. It is the scalar arithmetic afterwards that
    had to leave.
    """
    lp = np.log(np.maximum(folded, 1e-12))
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG)}   # (log p_blank, log p_nonblank)
    for t in range(lp.shape[0]):
        row_np = lp[t]
        # One numpy call for the top_k, then out of numpy entirely: every
        # lookup below is a python float, not a 0-d array.
        cands = [int(k) for k in np.argpartition(row_np, -top_k)[-top_k:]]
        row = [float(v) for v in row_np]
        nxt: dict[tuple, tuple[float, float]] = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_tot = _logaddexp(p_b, p_nb)
            for k in cands:
                rk = row[k]
                if k == PLATE_BLANK:
                    b, nb = nxt.get(prefix, (NEG, NEG))
                    nxt[prefix] = (_logaddexp(b, p_tot + rk), nb)
                    continue
                ch = PLATE_ALPHABET[k - 1]
                if prefix and prefix[-1] == ch:
                    # a repeat only becomes a second character across a blank;
                    # without one it collapses back onto the same prefix
                    b, nb = nxt.get(prefix, (NEG, NEG))
                    nxt[prefix] = (b, _logaddexp(nb, p_nb + rk))
                    if len(prefix) < MAX_PREFIX:
                        ext = prefix + (ch,)
                        b2, nb2 = nxt.get(ext, (NEG, NEG))
                        nxt[ext] = (b2, _logaddexp(nb2, p_b + rk))
                elif len(prefix) < MAX_PREFIX:
                    ext = prefix + (ch,)
                    b2, nb2 = nxt.get(ext, (NEG, NEG))
                    nxt[ext] = (b2, _logaddexp(nb2, p_tot + rk))
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -_logaddexp(kv[1][0], kv[1][1]))[:width])
    ranked = sorted(beams.items(), key=lambda kv: -_logaddexp(kv[1][0], kv[1][1]))
    return ["".join(p) for p, _ in ranked if p]


def _grammatical_readings(text: str) -> set[str]:
    """Every legal registration this string could be hiding.

    The string itself when it is already legal; otherwise its substrings -
    which is what strips a trailing state sticker off "MH02ER9194JH" - and a
    single confusable substitution in the state-code slot, which is what turns
    "W8..." into "WB...".
    """
    out: set[str] = set()
    seen = {text}
    repaired, did = repair_state_code(text)
    if did:
        seen.add(repaired)
    for base in seen:
        if matches_plate_grammar(base):
            out.add(base)
        n = len(base)
        for i in range(n):
            for j in range(i + 7, min(n, i + 11) + 1):
                sub = base[i:j]
                if matches_plate_grammar(sub):
                    out.add(sub)
                else:
                    fixed, ok = repair_state_code(sub)
                    if ok and matches_plate_grammar(fixed):
                        out.add(fixed)
    return out


DUAL_ROW_MAX_ASPECT = 2.6      # wider than this and it is a single-row plate


def _split_rows(image: np.ndarray) -> np.ndarray | None:
    """A two-row plate, cut at its gutter and laid out as one row.

    The model card says dual-row is handled end-to-end, and on its own corpus
    it is (96.91%). On this repo's real photos it is not: every two-row plate
    inspected reads short by exactly the characters that a squashed row loses -
    MH04DW9020 -> "MH04DWM020", MH01BU1852 -> "MH01BU852", MH47AV6753 ->
    "MH47AV753", MH02EZ2599 -> "MH022599". The cause is geometric, not
    semantic: _resize_for_rec() preserves aspect at a fixed 48px height, so a
    square plate arrives 48x48 inside a 320-wide canvas and each row of
    characters is left about 20px tall.

    Cutting the plate and laying the rows side by side hands the same model a
    single-row-shaped input at full character height. This is an ADDITIONAL
    candidate, never a replacement - read() scores both and keeps whichever the
    network itself likes better, so a single-row plate that happens to be cut
    badly cannot be made worse.
    """
    h, w = image.shape[:2]
    if h < 20 or w < 20 or w / h > DUAL_ROW_MAX_ASPECT:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = ink.sum(axis=1).astype(np.float64)
    lo, hi = int(0.30 * h), int(0.70 * h)
    if hi - lo < 3:
        return None
    cut = lo + int(np.argmin(proj[lo:hi]))
    top, bottom = image[:cut], image[cut:]
    if top.shape[0] < 8 or bottom.shape[0] < 8:
        return None
    height = max(top.shape[0], bottom.shape[0])

    def to_height(part):
        ph, pw = part.shape[:2]
        return cv2.resize(part, (max(1, int(pw * height / ph)), height),
                          interpolation=cv2.INTER_CUBIC)

    return np.hstack([to_height(top), to_height(bottom)])


def _score(folded: np.ndarray, text: str) -> tuple[float, list[float]]:
    """Exact CTC log-probability of `text` and its per-character posteriors."""
    return crnn_mod.ctc_score(folded, text, idx=PLATE_IDX, blank=PLATE_BLANK,
                              is_probs=True)


def _prefer(a: PlateRead, b: PlateRead) -> bool:
    """Is read `a` better than read `b`?

    Legality first, confidence second. A well-formed registration at 0.64 is
    better evidence than an illegal string at 0.99 - the illegal one cannot be
    a plate whatever the network thinks of it. Comparing on confidence alone
    let "MH47AV753" (a dropped digit, scored 0.99 off the squashed two-row
    crop) beat the row-split read "MH47AV6753" that was actually correct.
    """
    a_ok = ":ungrammatical" not in a.variant
    b_ok = ":ungrammatical" not in b.variant
    if a_ok != b_ok:
        return a_ok
    return a.confidence > b.confidence


class AnprOcrReader:
    def __init__(self, model, post_process, paddle_mod):
        self._model = model
        self._post_process = post_process
        self._paddle = paddle_mod
        # ['blank', '0'..'9', 'A'..'Z', 'a'..'z', ' '] - the decode side of the
        # dictionary the weights were trained against. _fold_probs() needs it
        # to know which of the 64 columns is which character.
        self._characters = list(getattr(post_process, "character", []))

    def _dygraph(self):
        """paddle's dygraph guard, or a no-op on a build that lacks it.

        Never let a missing guard be the thing that stops a read: the call
        below is correct without it in a plain interpreter, and this only
        exists to make it correct everywhere else too.
        """
        try:
            return self._paddle.base.dygraph.guard()
        except Exception:
            import contextlib
            return contextlib.nullcontext()

    def probabilities(self, image: np.ndarray) -> np.ndarray:
        """One forward pass -> the CTC head's (T, 36+1) folded posterior."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Establish the dygraph context explicitly rather than trusting the
        # ambient one. Driven from Streamlit, this forward pass intermittently
        # dispatched to paddle's *static-graph* conv2d overload and died with
        #
        #   conv2d(): argument (position 2) must be Value, but got EagerParamBase
        #
        # even though in_dynamic_mode() reported True right beside it, and even
        # though the identical calls are stable in a plain interpreter across
        # hundreds of reads and many threads. guard() pins a tracer for this
        # thread for the duration of the call, which removes the ambiguity
        # instead of relying on whatever the caller's thread happened to
        # inherit. to_tensor() moved inside it deliberately: the input has to
        # be built by the same tracer that runs the layers.
        with self._dygraph(), self._paddle.no_grad():
            tensor = self._paddle.to_tensor(
                np.expand_dims(_preprocess(image, IMAGE_SHAPE), axis=0))
            preds = self._model(tensor)
            # .numpy() while the tracer is still current - outside the guard a
            # dygraph tensor has nothing to resolve itself against.
            raw = preds
            if isinstance(raw, dict):
                raw = raw.get("ctc", next(iter(raw.values())))
            elif isinstance(raw, (list, tuple)):
                raw = raw[0]
            probs = np.asarray(raw.numpy(), dtype=np.float64)[0]
        # ppocr's CTCHead softmaxes its own output at inference, so this is
        # already a distribution - do not softmax it again.
        return _fold_probs(probs, self._characters)

    def probabilities_many(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """probabilities() for several crops, stacked into one forward pass.

        The model's input shape is fixed - IMAGE_SHAPE is [3, 48, 320] and
        _preprocess() resizes everything to it - so any set of crops is one
        (N, 3, 48, 320) tensor. That matters because a forward pass here is
        launch-bound rather than compute-bound: a single crop takes about the
        same wall time as a batch of eight, so a photograph with 38 crops was
        paying 38 kernel launches and 38 device-to-host syncs to do work the
        card could have done almost all at once.

        NOT bit-identical to calling probabilities() N times, and that is worth
        stating plainly: cuDNN selects different algorithms for different batch
        sizes, so summation order changes and posteriors differ by around 1e-4.
        Measured over the whole 532-crop real corpus, that moved zero plate
        strings, flipped zero accept/refuse decisions, left exact-match
        accuracy at 90.98% either way, and the largest confidence change was
        3.7e-05 - far below the 0.80 floor's nearest approach. It is safe here
        because it was measured to be, not because it must be.
        """
        if not images:
            return []
        prepared = []
        for image in images:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            prepared.append(_preprocess(image, IMAGE_SHAPE))

        cap = max(1, int(config.ANPR_OCR_MAX_BATCH))
        folded: list[np.ndarray] = []
        # The dygraph guard is held once for the whole set rather than per
        # crop, for the reason probabilities() documents at length.
        with self._dygraph(), self._paddle.no_grad():
            for start in range(0, len(prepared), cap):
                chunk = prepared[start:start + cap]
                preds = self._model(self._paddle.to_tensor(np.stack(chunk, axis=0)))
                raw = preds
                if isinstance(raw, dict):
                    raw = raw.get("ctc", next(iter(raw.values())))
                elif isinstance(raw, (list, tuple)):
                    raw = raw[0]
                batch = np.asarray(raw.numpy(), dtype=np.float64)
                folded.extend(batch[i] for i in range(len(chunk)))
        return [_fold_probs(p, self._characters) for p in folded]

    def read_many(self, images: list[np.ndarray]) -> list[PlateRead]:
        """[read(i) for i in images], with every forward pass batched together.

        Deliberately mirrors read() rather than sharing code with it: read() is
        the reference behaviour, and the two-pass row-split rule below - decode
        the crop as-is, decode it again with its rows side by side, keep
        whichever the network prefers - has to stay identical or a dual-row
        plate reads differently depending on which entry point was used.
        """
        if not images:
            return []

        # One work list holding both the plain crop and, where _split_rows()
        # offers one, its row-split alternative - so the alternative rides in
        # the same batch instead of costing a second forward pass of its own.
        work: list[tuple[int, bool, np.ndarray]] = []
        for i, image in enumerate(images):
            im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
            work.append((i, False, im))
            stacked = _split_rows(im)
            if stacked is not None and stacked.size:
                work.append((i, True, stacked))

        posteriors = self.probabilities_many([w[2] for w in work])

        out: dict[int, PlateRead] = {}
        for (i, is_split, _), folded in zip(work, posteriors):
            read = self.decode(folded)
            if not is_split:
                out[i] = read
                continue
            read.variant += ":rowsplit"
            best = out.get(i)
            if read.text and best is not None and _prefer(read, best):
                out[i] = read
        return [out[i] for i in range(len(images))]

    def read(self, image: np.ndarray) -> PlateRead:
        """Read a pre-cropped plate image (grayscale or BGR).

        A plate too square to be a single row is read twice - as-is, and with
        its two rows laid side by side - and the network's own score picks the
        winner. See _split_rows().
        """
        best = self.decode(self.probabilities(image))
        stacked = _split_rows(image)
        if stacked is not None and stacked.size:
            alt = self.decode(self.probabilities(stacked))
            alt.variant += ":rowsplit"
            if alt.text and _prefer(alt, best):
                best = alt
        return best

    def decode(self, folded: np.ndarray) -> PlateRead:
        """Turn a folded posterior into a PlateRead.

        Split out from read() so the same posterior can be re-decoded (and so
        the decode can be tested without a paddle forward pass).
        """
        greedy = _greedy(folded)
        beams = _prefix_beam(folded)
        if greedy and greedy not in beams:
            beams.append(greedy)

        candidates: set[str] = set()
        for b in beams:
            candidates |= _grammatical_readings(b)

        scored = []
        for cand in candidates:
            total, per = _score(folded, cand)
            conf = float(np.exp(np.clip(total / len(cand), -60.0, 0.0)))
            scored.append((cand, conf, per))

        best, best_conf, best_confs = "", 0.0, []
        if scored:
            # Highest likelihood wins, as before. The only change is what
            # happens when several candidates are within a couple of percent of
            # each other - which is exactly the case where the image cannot
            # distinguish them and something else has to.
            #
            # plate_plausibility() is that something else: it knows WB98 is a
            # real district and WB99 is not. Confined to near-ties on purpose,
            # so a candidate the evidence genuinely prefers is never overruled
            # by a lookup table. The confidence returned is still the winner's
            # own likelihood, so every downstream floor keeps its meaning.
            top = max(c for _, c, _ in scored)
            close = [t for t in scored if t[1] >= top * (1.0 - PLAUSIBILITY_TIE_TOLERANCE)]
            best, best_conf, best_confs = max(
                close, key=lambda t: (plate_plausibility(t[0]), t[1]))

        grammatical = bool(best)
        if not grammatical:
            # Nothing legal survived. Return the beam's own answer anyway: a
            # format this grammar has not met (KL34A465 in the corpus today)
            # should read wrong, as it does now, not become unreadable.
            fallback = beams[0] if beams else greedy
            if not fallback:
                return PlateRead(text="", confidence=0.0, plate_found=False,
                                 backend="anpr_ocr",
                                 reason="anpr-ocr found no readable text in this crop")
            total, best_confs = _score(folded, fallback)
            best = fallback
            best_conf = float(np.exp(np.clip(total / len(fallback), -60.0, 0.0)))

        weakest = float(np.min(np.clip(best_confs, 1e-6, 1.0))) if best_confs else 0.0
        mean = (float(np.exp(np.mean(np.log(np.clip(best_confs, 1e-6, 1.0)))))
                if best_confs else 0.0)
        return PlateRead(
            text=best, confidence=best_conf, char_confidence=weakest,
            agreement=1.0, mean_confidence=mean, char_confidences=list(best_confs),
            pattern=plate_shape(best), raw=greedy, repaired=best != greedy,
            backend="anpr_ocr", plate_found=True,
            variant="anpr_ocr" if grammatical else "anpr_ocr:ungrammatical")


def preload_torch() -> None:
    """Load torch's models before paddle is imported anywhere in this process.

    The order is not commutative and the failure is not subtle. Importing
    paddle first and only then pulling torch in - the YOLOv8-OBB detector via
    anpr.ocr._load_yolo(), or anpr.crnn.load() - breaks torch for the rest of
    the process: on Windows it fails outright with

        OSError: [WinError 127] ... Error loading torch/lib/shm.dll

    and where it does load, its tensors come back corrupted with a paddle C++
    error ("PreconditionNotMet - Tensor holds no memory"). It is a binary
    symbol clash between the two frameworks' compiled extensions. Loading
    torch's models first, then paddle, avoids it.

    This has to run before *anything* imports paddle, and available() does -
    so anpr.ocr._build_engine() calls this before asking available(), rather
    than leaving it to load(), which by then is already too late.
    """
    try:
        from anpr.ocr import _load_yolo
        _load_yolo()
        from anpr import crnn as _crnn_mod
        _crnn_mod.load()
    except Exception:
        # Best effort: no torch at all is a perfectly good reason to fail
        # here, and the paddle reader does not need it.
        pass


def _paddle_device(paddle) -> str:
    """"gpu" when this paddle build can actually use one, otherwise "cpu".

    Deliberately not config.torch_device(): the wheel that decides this is
    paddle's, not torch's, and they are installed separately. The plain
    `paddlepaddle` wheel is CPU-only even on a machine with a working card and
    a CUDA torch beside it - reading plates on the GPU here needs
    `paddlepaddle-gpu` (PADDLE_SETUP.md). On macOS the answer is always "cpu":
    paddle has no Metal backend at all.

    GODSEYE_DEVICE=cpu still wins, so the one switch turns off every reader.

    That switch is read from config.DEVICE - the request - and not from
    config.torch_device() - the answer torch arrived at. The two differ in
    exactly the case this reader lives in: .venv-paddle carries a CPU-only
    torch wheel, so torch_device() reports "cpu" on a machine with a working
    card, and testing it here let a torch wheel veto a GPU that paddle was
    perfectly able to use. Reading it as an explicit operator request keeps
    GODSEYE_DEVICE=cpu working while letting the two frameworks reach their
    own conclusions, which is what the paragraph above says should happen.
    """
    if (config.DEVICE or "auto").strip().lower() == "cpu":
        return "cpu"
    try:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
        print("[anpr_ocr_reader] CPU-only paddlepaddle build - the reader stays on "
              "the CPU (PADDLE_SETUP.md covers paddlepaddle-gpu)", file=sys.stderr)
    except Exception as exc:
        print(f"[anpr_ocr_reader] paddle GPU check failed ({exc}) - using the CPU",
              file=sys.stderr)
    return "cpu"


def load(checkpoint_path: str | None = None) -> AnprOcrReader | None:
    """Construct an AnprOcrReader, or None if unavailable (no paddle, no
    network for the first-run git clone / weight download, etc.).

    `checkpoint_path`: load a fine-tuned checkpoint (a PaddleOCR .pdparams
    file, e.g. models/anpr_ocr/finetune_data/output/best_accuracy.pdparams)
    instead of the base weights from Hugging Face - the base model.safetensors
    is never touched. Used to benchmark a fine-tune against the base model
    with the exact same wrapper/preprocessing (see
    anpr/finetune_anpr_ocr_compare.py).

    Loads torch/ultralytics (if importable) before paddle, on purpose:
    loading paddle first and only later triggering a torch model load (e.g.
    the YOLOv8-OBB detector via anpr.ocr._load_yolo(), or anpr.crnn.load())
    in the same process reproducibly corrupts torch's tensors with a paddle
    C++ error ("PreconditionNotMet - Tensor holds no memory") - a binary
    symbol clash between the two frameworks' compiled extensions. Loading
    torch's model(s) first, then paddle, avoids it; the reverse order does
    not seem to (untested whether it's fully commutative beyond that).
    """
    preload_torch()
    if not available():
        return None
    try:
        _ensure_paddleocr()
        import paddle
        from ppocr.modeling.architectures import build_model as ppocr_build_model
        from ppocr.postprocess import build_post_process
        from safetensors.numpy import load_file

        paddle.set_device(_paddle_device(paddle))
        _, dict_path = _ensure_weights()  # en_dict.txt always comes from the base repo

        post_process = build_post_process({
            "name": "CTCLabelDecode", "character_dict_path": dict_path,
            "use_space_char": True,
        })
        cfg = copy.deepcopy(MODEL_CONFIG)
        model = ppocr_build_model(cfg["Architecture"])
        model.eval()
        if checkpoint_path is not None:
            state = paddle.load(checkpoint_path)
            model.set_state_dict(state)
        else:
            weights_path, _ = _ensure_weights()
            np_state = load_file(weights_path)
            model.set_state_dict({k: paddle.to_tensor(v) for k, v in np_state.items()})
        return AnprOcrReader(model, post_process, paddle)
    except Exception as exc:
        print(f"[anpr_ocr_reader] failed to load: {exc}", file=sys.stderr)
        return None
