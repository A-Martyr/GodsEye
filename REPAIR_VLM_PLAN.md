# Improving repair recall with a second recogniser

A plan for raising the recovery rate on refused captures, as a channel that runs
beside the reader rather than inside it. The 108 ms live read must not change by
a millisecond, and the 98.3% precision of what the repair pass currently accepts
must not fall.

Written 7 September 2026. **Not for this week** — the competition is Saturday and
the freeze is Thursday. This is the work after it.

---

## 1. Where the repair pass actually loses

Measured on 600 refused captures with ground truth from `data/godseye.db`
(`tools_fieldtest/`-style harness, 150 for the ranking stages):

| Outcome | Share | What would fix it |
|---|---:|---|
| Accepted by the guards, **98.3% precision** | 78.0% | nothing — this works |
| Truth in candidate set, ranked below top-1 | ~10% | better scoring or better evidence |
| **Truth absent from the candidate set** | **10.3%** | **open-vocabulary reading only** |
| Truth was top-1 but the guards refused it | 4% | looser guards, at a precision cost |

Supporting figures: candidates per capture median **36** (range 1–98); the truth
is top-1 in **89.0%** of the cases where it is present at all.

### The ceiling is the whole argument

`repair.candidate_plates()` builds its candidate set from what **neighbouring
cameras saw**. When a vehicle entered off a road no camera covers, its
registration is simply not in the set, and the best of the wrong candidates still
looks plausible — which is exactly why `REPAIR_MIN_SCORE` and `REPAIR_MIN_MARGIN`
exist, encoding a "none of these" hypothesis.

**No improvement to scoring can recover a string that is not among the
candidates.** That 10.3% is unreachable by construction, and it is the only part
of the problem a second recogniser is uniquely qualified to attack.

The other ~14% (mis-ranked plus guard-refused) is a *ranking* problem, and a
second opinion helps there too — but as a vote, not as a reader.

---

## 2. The blocking constraint: there are no pixels

```sql
CREATE TABLE IF NOT EXISTS unresolved (
    ...
    evidence    BLOB,     -- CTC lattices, float16, compressed
```

A refused capture retains **CTC lattices, not the image**. A VLM cannot read a
lattice. So nothing in this plan is possible until the plate crop is retained,
and that is a deliberate change to what the platform stores — not a detail.

### What it costs, honestly

`core/vehicle_search.py` currently tells operators, in the `NOT_RECORDED` block:

> *"nothing about the occupants is recorded — the platform stores a registration,
> a place and a time, and **deliberately stores no image of anyone**"*

Retaining crops weakens that sentence, and the sentence is load-bearing — it is
part of why the privacy story is credible. Three things keep it defensible:

1. **The plate crop only.** Not the frame, not the vehicle, not the cabin. A
   40×15 to 200×60 px region containing a registration and nothing else. A
   registration is already stored as text; the crop adds no new *class* of
   personal data, it adds a second representation of the same fact.
2. **A short, enforced TTL.** `REPAIR_EVIDENCE_TTL_S` is already 1 hour and
   `prune_evidence()` already drops lattices past it. The crop rides the same
   clock and the same pruner. It cannot outlive the lattice it belongs to.
3. **Inside the purge.** `core/privacy.py`'s `_TABLES` list must gain the column
   so `retention_preview()` and `retention_purge()` count and delete it. A stored
   artefact outside the retention machinery is exactly the failure that module
   exists to prevent.

Storage is negligible: ~2 KB JPEG per capture, 3,905 open captures ≈ **8 MB**.

**Update the `NOT_RECORDED` wording at the same commit.** The honest version is
"stores no image of any *person*; the plate crop is retained for one hour to
support the repair pass, and is purged with it."

---

## 3. Architecture: a channel, not a stage

```
LIVE PATH  (unchanged, 108 ms)
  frame -> YOLO-OBB -> anpr_ocr -> grammar CTC decode -> confidence
             |                                              |
             |                                    above floor -> sightings
             |                                    below floor -> unresolved
             |                                                     |
             |                                          (crop + lattice)
             v                                                     v
        REPAIR CHANNEL (behind REPAIR_LAG_S = 15 min, offline, batched)
             candidate_plates() ---.
             CTC rank -------------+--> fusion ladder --> inferences
             second recogniser ----'                     (never sightings)
```

Four rules, each inherited from a rule the repair pass already enforces:

| Rule | Why |
|---|---|
| Runs **behind** the live feed | `repair()` already trails by `REPAIR_LAG_S`; the downstream camera has not seen the vehicle yet |
| Writes to `inferences`, **never** `sightings` | an inference is a hypothesis with an evidence chain, not an observation |
| Inferences never feed candidate sets | the no-chaining rule; without it one mistake propagates along a corridor |
| Clone-flagged plates stay exempt | repair assumes plausible travel; the clone rule fires on implausible travel |

The second recogniser is **additive**. If it is absent, unavailable, or times
out, the pass produces exactly what it produces today. That is a hard
requirement, not a nicety — it keeps the demo path independent of a component
that may need a download, a GPU, or a network.

---

## 4. Phase 0 — the control experiment, before any VLM

**Do this first. It may deliver most of the gain for none of the risk.**

`FIELD_TEST.md` established that the localiser runs at `imgsz=640`, and that
raising it to 1600 took detections from 5 to 18 on the same images. A capture is
refused when the *recogniser* is unsure — but some of those are unsure because
the crop handed to it was poor, not because the plate is unreadable.

So before adding a model, re-run the **existing** recogniser on the retained
crop with the cheap variations already in the codebase:

- `_crop_variants()` — including the OBB perspective-rectified crop
- `_split_rows()` — the two-row path
- upscaled input (2×, 4× Lanczos) into `_resize_for_rec()`
- the whole-frame decode that `read_frame()` skips when candidates exist

**Measure the recovery rate of that alone.** It costs one afternoon, needs no new
dependency, and if it recovers 3–4 points of the 10.3% then the VLM's marginal
value drops sharply and the honest answer may be "not worth it".

This is also the control the VLM must beat. Without it you cannot attribute a
gain to the model rather than to simply looking at the crop again.

---

## 5. Phase 1 — the interface

One module, `core/second_opinion.py`, mirroring the `available()` / `load()`
shape that `anpr/crnn.py` and `anpr/anpr_ocr_reader.py` already use.

```python
@dataclass
class Opinion:
    text: str                 # what it read, uppercased, non-alphanumerics stripped
    confidence: float         # 0-1, and see the warning below
    grammatical: bool         # ocr.matches_plate_grammar(text)
    in_candidate_set: bool    # filled by the caller, not the reader
    source: str               # "moondream2" | "florence2" | "anpr_ocr_hires" | ...
    latency_ms: float
    raw: str                  # unmodified model output, for audit

def available() -> bool: ...
def read_crop(crop: np.ndarray) -> Opinion | None: ...
def read_many(crops: list[np.ndarray]) -> list[Opinion | None]: ...   # batched
```

**On `confidence`.** A VLM's token logprobs are not a calibrated plate-level
confidence, and treating them as one is the single easiest way to wreck the
98.3% precision. Two defences:

- Never compare a VLM confidence against `REPAIR_MIN_SCORE`. They are different
  quantities on different scales. The CTC score stays the only number the
  absolute floor is applied to.
- Prefer **agreement** to self-reported confidence. Run the model twice at
  different temperatures or on two crop variants; a model that returns the same
  string twice is worth more than one that returns a high logprob once.

### Model candidates for a 4 GB GTX 1650

| Model | Params | Fits 4 GB | Note |
|---|---:|---|---|
| **Florence-2-base** | 0.23 B | comfortably | has an explicit OCR task token; start here |
| Florence-2-large | 0.77 B | yes | the obvious second try |
| **Moondream2** | 1.8 B | int8, tight | built for edge, strong on small text |
| Qwen2-VL-2B | 2 B | int4 only | slow on this card |
| 7B-class / API models | — | **no** | 14 GB fp16; API means venue-network dependency |

Anything above ~2 B is out on this hardware, and API calls are out on principle:
the whole platform is built to survive a venue with bad wi-fi.

---

## 6. Phase 2 — the fusion ladder

This is the heart of it. A VLM handed an illegible crop **will return a
well-formed Indian registration** — it does not have a "nothing legal was on the
beam" state the way `_grammatical_readings()` does. So the ladder is built
around *corroboration*, not around trusting the model.

Given a refused capture with CTC ranking `ranked[]` and an `Opinion`:

| # | Condition | Action | Marked |
|---|---|---|---|
| **A** | CTC already accepted by the guards | keep CTC. Log agreement/disagreement only | `ctc` |
| **B** | VLM text **==** CTC top-1, guards had refused it | **accept** — two independent methods agree | `ctc+vlm` |
| **C** | VLM text in candidate set, ranked 2nd–5th, its CTC score within `δ` of top-1 | **promote** to that candidate | `vlm-reranked` |
| **D** | VLM text in candidate set, ranked far down | keep CTC, record the disagreement | `ctc` |
| **E** | VLM text **not** in candidate set, grammatical, agrees across two runs | record as **open-vocabulary hypothesis** — never auto-accepted | `vlm-only` |
| **F** | VLM text not in set, ungrammatical or unstable | discard | — |

**Rung B is the highest-value, lowest-risk rung.** It targets the 4% that were
correct and refused, and the agreement of two methods with independent failure
modes is genuinely strong evidence. Implement B first and measure it alone.

**Rung E is the one that attacks the 10.3%, and the one that can hurt you.** It
must be a *separate class of output*, surfaced to an operator for confirmation
and never counted in the same accuracy figure as the rest. The `inferences`
table gains a `source` column so the two can never be conflated in a query.

```sql
ALTER TABLE inferences ADD COLUMN source TEXT NOT NULL DEFAULT 'ctc';
ALTER TABLE inferences ADD COLUMN second_opinion TEXT;   -- JSON: raw, conf, agreement
```

Config, mirroring the existing repair constants:

```python
SECOND_OPINION_ENABLED   = os.getenv("GODSEYE_SECOND_OPINION", "0") not in ("0", ...)
SECOND_OPINION_MODEL     = os.getenv("GODSEYE_SO_MODEL", "florence2-base")
SECOND_OPINION_RERANK_DELTA = 0.25   # rung C: how close in CTC score to promote
SECOND_OPINION_MIN_AGREE = 2         # runs that must return the same string
SECOND_OPINION_TIMEOUT_S = 8.0       # past this the channel is skipped entirely
SECOND_OPINION_CROP_TTL_S = REPAIR_EVIDENCE_TTL_S   # crops die with the lattices
```

**Default off.** It turns on for evaluation and stays off in the demo path until
it has earned its place.

---

## 7. Phase 3 — measurement

You have **3,905 refused captures with `true_plate` ground truth**. This is
directly measurable, which is unusual and worth exploiting properly.

### Protocol

1. Freeze an evaluation split — sample 800 refused captures by `id`, store the
   list. Every configuration is scored on the same 800.
2. Report five numbers per configuration:

| Metric | Definition | Today |
|---|---|---:|
| Ceiling | truth in candidate set | 89.7% |
| Yield | captures that produce an accepted inference | 78.0% |
| **Precision** | accepted inferences that are correct | **98.3%** |
| Recall | correct inferences ÷ all refused captures | ~76.7% |
| Open-vocab recovery | rung-E hypotheses that are correct | n/a |

3. Report **rung B, C and E separately.** A single blended number hides which
   part of the ladder is working.

### Acceptance criteria

Ship the channel only if, on the frozen split:

- **Precision does not fall below 97%** on rungs A–D combined. The current 98.3%
  is the platform's most defensible number and recall is not worth trading for it.
- **Yield rises by ≥ 3 percentage points**, or rung E recovers ≥ 20% of the 10.3%
  that is otherwise unreachable.
- The live read stays within noise of **108 ms** — measured, not assumed, with
  the channel enabled and the model resident.

### Kill criteria

Stop and write up the negative result if:

- Phase 0 (crop re-read, no VLM) already recovers most of the reachable gain —
  then the model is not the contribution and should not be added.
- Rung E precision is below ~60%. An open-vocabulary hypothesis that is wrong
  more than twice in five is not worth surfacing to an operator; it converts a
  clean "we could not read this" into a lead that wastes someone's time.
- The model cannot be made to return a stable string across two runs on the same
  crop. Instability there means the confidence signal does not exist.

**A negative result here is publishable inside the project**, in the same way
The team's working notes record that fine-tuning the CRNN on 540 real crops does not work.
That entry has saved repeating the experiment more than once.

---

## 8. Order of work

| Phase | Work | Days | Blocking? |
|---|---|---:|---|
| **0a** | Add `crop BLOB` to `unresolved`, wire the TTL and the pruner | 0.5 | yes |
| **0b** | Add the column to `privacy._TABLES`; fix the `NOT_RECORDED` wording | 0.5 | yes |
| **0c** | Freeze the 800-capture evaluation split; build the scorer | 0.5 | yes |
| **0d** | **Control: re-read crops with the existing recogniser** | 1 | — |
| 1 | `core/second_opinion.py` + Florence-2-base behind it | 1 | — |
| 2 | Rung B only. Measure. | 0.5 | — |
| 3 | Rungs C and E, `source` column, operator surface | 1.5 | — |
| 4 | Full evaluation, write-up, ship-or-kill | 1 | — |

**Roughly a week**, and phase 0d may end it early — which would be a good
outcome, not a wasted one.

---

## 9. What not to do

**Do not replace the live recogniser.** PP-OCRv5 is pretrained on 558,000 real
Indian plates. The CTC score it produces is verified against `torch.nn.CTCLoss`
to 2.8e-14 and carries the refusal floor, `CLONE_MIN_CONFIDENCE`, the repair
ranking and `_prefer()`. Swapping it for a model that emits text with no
calibrated confidence removes the number five mechanisms depend on.

**Do not let a VLM answer reach `sightings`.** Ever. The `inferences` separation
is the reason a wrong guess costs nothing here, and it is what makes the whole
experiment safe to run.

**Do not report VLM-recovered plates in the headline accuracy.** They are a
different class of claim with a different error profile. `source` exists so the
two can be reported apart, and they should be.

**Do not add this before Saturday.**
