# How the reader got fast

Every number here was measured on this machine - a GTX 1650 (4 GB, Turing),
16 logical cores, Windows 11 - against two real 960x1280 Kolkata street
photographs, timed as a warm `read_frame()`. "Photo 1" holds a clear taxi
plate the localiser finds three regions in; "photo 2" is a harder scene with
ten candidate regions, most of them shop signage.

| Stage | Photo 1 | Photo 2 |
|---|---|---|
| Where it started | 46,283 ms | ~61,800 ms |
| Paddle given more than one core, and stopped phoning home | 5,046 ms | 21,246 ms * |
| `paddlepaddle-gpu` | 260 ms | 1,335 ms |
| Beam search out of numpy | 125 ms | 809 ms |
| Batched forward passes | **118 ms** | **578 ms** |

`*` truncated by a search budget, and wrong because of it - see below.

**Photo 1 is about 390x faster than where it started, and reads the same plate
at the same confidence.** The reads never changed: `WB25G4XXX` at 0.962 and
`WB04F620` at 0.878, at every stage.

Separately, the wait before the *first* read of a session went from 8.0 s to
0.19 s. That was never compute - it was model loading and cuDNN autotuning
landing on whoever clicked first.

---

## What actually turned out to be slow

Almost none of it was the model.

### 1. Paddle was using one core out of sixteen

Paddle sets `OMP_NUM_THREADS=1` itself, at import. Nothing in this repo asked
for that. It has to be overridden *before* paddle is imported, which is why
`anpr/anpr_ocr_reader.py` does it at module scope.

The scaling is not what you would guess - per 120x40 plate crop:

| Threads | Per crop |
|---|---|
| 1 (paddle's default) | 3909 ms |
| **4** | **1650 ms** |
| 8 | 4208 ms |
| 16 | 5115 ms |

Four is measured, not chosen. Sixteen is slower than one: the work per crop is
too small to pay for building and joining a large pool, and on a hybrid CPU the
scheduler spreads it onto efficiency cores. Re-measure before changing
`config.ANPR_OCR_THREADS`.

### 2. Every startup revalidated the weights over the network

`huggingface_hub` checks its cached 149 MB of weights against huggingface.co on
every load. Going offline once the files are present took reader startup from
**50 s to 6 s** - and removes a hang on venue wi-fi, which the rest of the
platform is built to survive.

### 3. The same photograph was decoded ten times

`read_frame()` decodes the whole frame, then several insets of every box the
localiser proposes - up to 48 decodes. Its own comment reads *"Decoding a few
insets costs milliseconds"*, and for the CRNN at 5 ms a decode that is exactly
right. For `anpr_ocr` at 1650 ms it was wrong by a factor of about 330.

The sweep is now cost-aware: the CRNN keeps it, `anpr_ocr` skips the
whole-frame decode when the localiser found regions, exits early on a confident
legal read, and is bounded by `config.FRAME_SEARCH_SECONDS`.

### 4. The GPU was sitting idle

`.venv-paddle` carried the CPU-only `paddlepaddle` wheel, so the 91% reader ran
entirely on the CPU while the CRNN used the card. Installing
`paddlepaddle-gpu==3.3.1` (CUDA 12.6) took a decode from 1650 ms to 43 ms.

Installing it alone would have changed nothing. `_paddle_device()` gated on
`config.torch_device()`, and this venv deliberately pairs a **CPU-only torch**
with a CUDA paddle - so it returned `"cpu"` and would have pinned paddle to the
CPU anyway. It now reads `config.DEVICE`, the operator's explicit request, so
`GODSEYE_DEVICE=cpu` still turns everything off but a torch wheel cannot veto a
GPU that paddle can use. `PADDLE_SETUP.md` has the install.

### 5. Then a numpy loop became the bottleneck

With the forwards fast, profiling put the CTC prefix beam at **47% of a whole
read** - 876 ms of 1860 ms, more than every GPU forward pass in that read put
together. The same code, unchanged all day, promoted from an invisible 0.6% to
the single largest cost simply because everything around it got faster.

The algorithm was fine. Three nested Python loops were calling `np.logaddexp`
on *single scalars* tens of thousands of times per crop, paying numpy's ufunc
dispatch for one flop each, and every `row[k]` was materialising a numpy
scalar. `_logaddexp()` does the same arithmetic in plain Python - both are
`max + log1p(exp(min - max))` through the same libm - and the row is pulled out
of numpy once per timestep.

**8.9x faster, 875.7 ms to 98.5 ms.** The `argpartition` stayed: picking the
top 6 of 37 classes is real vector work, and is what numpy is good at.

### 6. Batching what the card should have done at once

A forward pass here is launch-bound, not compute-bound - one crop costs about
as much wall time as eight - and `IMAGE_SHAPE` is fixed at `[3,48,320]`, so any
set of crops is one tensor. Photo 2 was paying 38 kernel launches and 38
device-to-host syncs.

`read_many()` batches a box's crop variants, carrying each variant's row-split
alternative in the same batch. Batching is per **box**, not per frame: batch
the whole frame and the search must decode every box before it can notice the
first one already answered, which throws the early exit away.

---

## The accuracy question, and how it was settled

Most of the above cannot change a read. Batching can: cuDNN picks different
algorithms for different batch sizes, so summation order changes and posteriors
shift by about 1e-4. That is very likely harmless, and "very likely" is not a
standard to ship accuracy on - `MIN_PLATE_CONFIDENCE_ANPR_OCR` is 0.80, and a
read sitting within 1e-4 of that floor would flip between stored and refused.

So it was measured, over the whole real corpus, using the shipped method:

```
532 crops, 118 vehicles
  plate strings changed         0 / 532
  accept/refuse flips           0 / 532
  rowsplit variant changed      0 / 532
  largest confidence change     3.67e-05
  exact match      90.98%  ->  90.98%     (484/532 both)
```

90.98% is the 91.0% this project quotes, so that was the real baseline rather
than a proxy. Re-run that comparison if `config.ANPR_OCR_MAX_BATCH` changes.

The beam rewrite needed no such argument: it returned the identical strings in
the identical order, all sixteen per beam, on every posterior tried.

---

## A bug found on the way

`POST /api/anpr/read` returned 500 on **every photograph in which a plate was
found**, and worked when none was. `detect_plate_candidates()`'s OBB branch
built boxes with `.astype(int)`, leaving `numpy.int64` scalars that travel out
through `PlateRead.candidates` to FastAPI, whose JSON encoder cannot serialise
them. The dashboard never showed it, because it calls the engine in-process
with no serialisation step.

---

## What is left

Photo 2's remaining 578 ms is roughly 50 ms of localiser, the batched forwards,
and about 90 ms of grammar and CTC scoring. The largest untouched item is the
**localiser**, which runs on the CPU-only torch in `.venv-paddle` at ~50 ms.
Moving it to the GPU means either a CUDA torch beside CUDA paddle - risking the
symbol clash `PADDLE_SETUP.md` documents, on a 4 GB card - or an ONNX export
that sidesteps torch entirely. Neither was attempted.

Two limits worth knowing, both confirmed on close-up phone photographs rather
than the road scenes the models were trained on:

- **Localisation** misses plates that fill much of the frame. Padding the image
  partially recovers detections, which is what identifies it as a relative-scale
  problem rather than anything about the plate.
- **Decorative fonts** on the digits of motorcycle plates read wrong even from a
  perfect hand crop.

In both cases the platform refused rather than storing a wrong registration,
which is the confidence floor doing its job.
