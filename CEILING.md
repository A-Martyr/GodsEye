# Pushing the reader to its ceiling without retraining

Everything here is inference-time. No new weights, no fine-tune, no labelled
data. The organising idea is in §0, and if you read only one section make it §1.

---

## 0. The lever you already have: this is a propose-and-rank system

The decode path is not a monolith. `ANPROCRReader.decode()` does this:

```python
beams      = _prefix_beam(folded)          # propose
candidates = ∪ _grammatical_readings(b)    # propose more
best       = argmax _score(folded, cand)   # rank, by EXACT CTC likelihood
```

`_score()` is `crnn.ctc_score()` — the exact forward-backward probability that
this image's evidence would have produced that string, verified against
`torch.nn.CTCLoss` to 2.8e-14.

**That means the scorer is already a general-purpose judge of any hypothesis.**
Every idea below is a new *proposer*. None of them replaces the ranking, none
touches the model weights, and every one is additive: if a proposer suggests
nonsense, the CTC score rejects it exactly as it rejects a bad beam today.

This is why the ceiling is reachable without retraining. You are not making the
model better. You are giving a very good judge more things to judge.

---

## 1. Erasures, not errors — the single biggest idea here

### The blocker, as currently stated

`core/repair.py` says, correctly:

> *per-character CTC posteriors are peaked almost everywhere, so 99.8% of
> characters come back above 0.9 confidence including the wrong ones, and
> per-character confidence predicts per-character correctness at AUROC 0.692.
> The engine knows which strings it is unsure of (AUROC 0.974); it does not know
> which character let it down. **There is no reliable mask to build, so there
> are no blanks to fill.***

Every word of that is true — **about the CTC posteriors.** It does not follow
that no mask exists. It follows that the mask cannot come from *inside* the
recogniser.

### Why this matters so much

This is the erasure-versus-error distinction from coding theory. A code that
corrects *t* errors — corrupt symbols at unknown positions — corrects **2t**
erasures, symbols known to be missing. Knowing *where* the damage is roughly
doubles what you can recover, because you no longer spend evidence locating it.

Right now every unclear character is an *error*: the decoder does not know which
of the ten positions failed, so it must consider corruption everywhere. Localise
the damage and the same evidence becomes far more powerful.

### Where the mask comes from — outside the recogniser

Your user's exact case — rain, glare, two or three unclear digits — is a
**photometric and geometric** problem, and photometric problems are visible in
the pixels without any model at all:

| Damage | Detector | Cost |
|---|---|---|
| **Glare / specular blowout** | column-wise count of saturated pixels (V > 250) in the plate crop; a blown region is a contiguous run of columns | trivial, OpenCV |
| **Motion blur** | direction and extent from the power spectrum, or a per-column variance-of-Laplacian profile | trivial |
| **Rain streaks / droplets** | local high-frequency anomaly vs the plate's own texture statistics | cheap |
| **Occlusion** (mud, bar, wiper) | column-wise darkness/edge-density dropout against the plate's median | trivial |
| **Under-resolution** | per-character width in px after rectification | free, it's geometry |

Each yields a **per-column reliability profile** across the plate. Map columns to
character cells (the plate grammar tells you how many cells to expect, and the
CTC alignment from `ctc_score` gives you where each character was emitted), and
you have exactly the mask `repair.py` says cannot be built.

### The decode that follows

```
WB01?F94?2       7 of 10 characters reliable, 3 erased
```

Now enumerate legal completions rather than re-reading the image:

1. **Grammar** — position 3–4 must be digits, 5–6 letters. Cuts the space hard.
2. **RTO table** (§2.1) — `WB01` is Kolkata; `WB99` does not exist.
3. **Network lexicon** (§2.2) — plates this network has actually seen.
4. **Neighbour candidates** — the 36-median set `candidate_plates()` already builds.

Then rank every completion with `ctc_score()` against the retained lattice. The
scorer already exists; you are handing it a much better shortlist.

**The combinatorics are strongly in your favour.** Three erased characters is at
most 36³ ≈ 46,656 blind, but constrained to legal positions it is nearer 10³ or
26²·10, and intersected with a 36-candidate neighbour set it is usually **one**.
Seven known characters out of ten pins an Indian registration almost uniquely.

### Why this is safe

It never invents pixels. It never asks a model to imagine a character. It says
"these three positions carry no information" and then asks which legal
registrations are consistent with the seven that do — arbitrated by the same
exact likelihood that governs everything else. A glare-blown plate with no legal
completion returns nothing, which is the correct answer.

**Do not inpaint the glare.** Inpainting synthesises plausible strokes, the
recogniser reads them confidently, and you get a well-formed wrong plate — the
worst output this platform can produce. Mask, don't fill.

---

## 2. New proposers, cheapest first

### 2.1 The RTO district table — pure data entry, no model

`PATTERNS` is currently generic:

```python
"LL" + "D"*d + "L"*s + "DDDD"   for d in (1,2), s in (3,2,1,0)
```

Real Indian registrations are far more constrained than that. The district code
after the state is **not** free: Delhi runs 01–13, Maharashtra 01–50, West Bengal
has 01–98 with gaps. `WB99AA1234` is grammatically legal today and does not
exist in the world.

Building `STATE_RTO_RANGES: dict[str, set[int]]` is a lookup table anyone can
type from public RTO listings. It:

- prunes candidates during the beam, not after
- makes `repair_state_code()` far stronger — a state code is only repairable to
  one whose district code is also valid
- is free at runtime (a set lookup)

**Add the formats you are currently missing** while you are in there:

| Format | Shape | Note |
|---|---|---|
| **Bharat series** | `DDLLDDDDLL` | `25BH3028AB` — [FIELD_TEST.md](FIELD_TEST.md) found this fails today |
| Diplomatic | `DDCDDDD` | `CD`/`CC`/`UN` corps codes |
| Military | `DDLDDDDDDL` | broad-arrow prefix |
| Temporary | `LLDDTRDDDD` | `TR` series |

Every one of these currently decodes to an ungrammatical string and collapses in
confidence, exactly as image 6 of the field test did at 0.02.

### 2.2 The network lexicon — the asset nobody else has

Your `sightings` table holds every plate this network has confidently seen.
1,138 distinct registrations today; on a real deployment, tens of thousands.

A vehicle seen at this junction last Tuesday is *enormously* more likely than a
random legal string. Bias the candidate ranking with a prior built by counting —
no training, just `SELECT plate, COUNT(*) FROM sightings`.

**With the anti-chaining discipline that already exists.** `repair.py` refuses to
let an inference become evidence for another inference, for exactly the right
reason. The lexicon needs the same rule:

- only plates seen at **≥2 distinct cameras** enter it
- only from **accepted** reads, never from `inferences`
- the prior is a **tie-breaker weight**, never an override — a plate absent from
  the lexicon must still be readable, or the network can never see a new vehicle

Without those three, a single misread becomes self-reinforcing and the reader
slowly hallucinates its own history.

### 2.3 Per-camera state-code priors — counting, again

A camera near the Odisha border sees more `OD` plates than one in Salt Lake.
Compute `P(state | camera)` from stored sightings and use it as a weak prior in
the beam's first two positions. Zero training. Recompute nightly.

Weak prior only, and floor it — a stolen Kerala vehicle must still be readable in
north Kolkata.

---

## 3. Better evidence: give the same model more to work with

### 3.1 Burst sub-pixel super-resolution — real information, not invented

This is the highest-value item in this section, and it is not hallucination.

You already capture `BURST_FRAMES = 5` per vehicle and fuse them **at the
decision level** — five reads voted by CTC score, 91% → 97% per vehicle. That
discards a great deal.

Five frames of a moving vehicle sample the plate at **different sub-pixel
offsets**. Align them to a common frame (ECC, or optical flow), upsample onto a
2× grid, and accumulate. This is genuine resolution recovery — the information
is physically present across the burst and simply thrown away by reading each
frame independently. It is how every phone camera's night mode works.

```
5 frames × sub-pixel offsets  →  align (ECC)  →  accumulate on 2× grid
                              →  one crop with real added detail
                              →  decode as an ADDITIONAL variant
```

Additive, like `_obb_rectified_crop`: it becomes one more candidate crop, and
`_prefer()` arbitrates. If alignment fails the variant is simply worse and loses.

**This directly attacks the far-lane and small-plate failures** — `far_lane` is
21% single-frame, 67% burst in your own benchmark, and it is bounded by pixels on
the plate, which is precisely what this recovers.

### 3.2 Per-character ROI decode — recover the resolution `_resize_for_rec` discards

```python
def _resize_for_rec(img_bgr, target_shape):
    _, h, w = target_shape          # 48 x 320
    ratio = h / img_h
    new_w = min(int(img_w * ratio), w)
```

**Every crop is squashed to 48 px high, whatever its source resolution.** A
4.5:1 single-row plate lands ~216 px wide — about **21 px per character** for a
ten-character registration. A 700 px-wide crop from a good capture is thrown away
down to that.

`_split_rows()` already exploits this insight for two-row plates: cut, lay side
by side, decode again, keep the better read. The single-row analogue:

- split the plate into **two overlapping halves**, decode each at the full 48 px
  height → ~43 px per character, double the effective resolution
- or decode per character-cell at native resolution and combine

Combine by scoring the *stitched* hypothesis with `ctc_score()` against the
whole-plate lattice, so a stitch that disagrees with the global evidence loses.
This is exactly the two-row pattern generalised, and it needs no new model.

### 3.3 Restoration: what is safe and what is not

| Technique | Pretrained, no training | Verdict |
|---|---|---|
| **Zero-DCE** low-light enhancement | yes — zero-**reference**, needs no paired data | **safe**, try first on night captures |
| Retinex / CLAHE | classical | safe, already cheap |
| **NAFNet / Restormer** deblur | pretrained checkpoints | safe *as a variant*, never as a replacement |
| Derain networks | pretrained | safe as a variant |
| **Real-ESRGAN / SwinIR** single-image SR | pretrained | **caution** — invents texture; a hallucinated serif becomes a character |
| **Glare inpainting** | — | **unsafe. Do not.** See §1 |

The rule that makes all of these safe: **restoration output is an additional
crop variant, never a substitute.** The original always stays in the pool, and
the exact CTC score picks. A restoration that helps wins on evidence; one that
invents detail produces a string the lattice does not support and loses.

### 3.4 TTA fused in lattice space, not string space

`_crop_variants()` exists. Extend it — rotations ±2°/±5°, gamma, sharpening,
scales — but fuse properly:

Instead of decoding each variant and voting on strings, collect each variant's
**folded probability lattice**, and for each candidate string sum
`ctc_score(lattice_v, s)` across variants. That is a proper likelihood ensemble
over an augmentation distribution rather than a popularity contest between
strings, and it uses machinery you already have.

---

## 4. Better arbitration

### 4.1 Calibrate the confidence — fitting two parameters, not a model

You have **3,905 refused captures with ground truth**, and more accepted ones.
The CTC score is a likelihood, not a probability, so `CLONE_MIN_CONFIDENCE = 0.70`
and `REPAIR_MIN_SCORE = -2.00` are thresholds on an uncalibrated scale.

Fit an isotonic or Platt map from score → P(correct) on that labelled set. It is
a one- or two-parameter fit, not training, and it converts every threshold in the
platform from a tuned constant into a stated probability. *"We store reads above
0.9 probability-of-correct"* is a far better sentence than *"above a CTC score of
−2.0"*, and it makes the floor tunable against an operational target.

### 4.2 Condition-aware evidence weighting — you already measured the weights

`sim/city.py` carries `CONDITION_ACCURACY`: daylight 1.00, night_ir 1.00,
monsoon 0.98, dusk 0.89, night_glare 0.87, fog 0.83, high_speed 0.80,
far_lane 0.67, cheap_cam 0.27, storm 0.00.

In burst fusion, frames are currently equal. They should not be: estimate each
frame's condition from image statistics (mean luminance, saturated fraction,
Laplacian variance, colour temperature) and weight its CTC contribution by the
measured reliability of that condition. A storm frame contributing equally with a
daylight frame is actively harmful — you measured it at 0.00.

### 4.3 Cross-camera majority vote on a trajectory

A vehicle read `WB01BF9492` at four cameras and `WB01BF9432` at a fifth: the odd
one out is a misread, and the trajectory says so. Currently each read stands
alone once accepted.

Requires identity association, which §5 gives you properly.

---

## 5. Network-level joint decoding — the most ambitious idea here

Everything above improves one capture at a time. This changes the unit.

**Today:** each capture is decoded to one string independently; `repair.py` uses
neighbour context only for captures that were *refused*.

**Instead:** keep the **top-K hypotheses with scores** for every capture. Build a
graph where nodes are `(capture, candidate)` and edges join candidates that are
consistent with plausible travel on the road network — the same
`_window()`/`free_flow_minutes()` machinery `repair.py` already has. Then solve
for the assignment maximising

```
Σ log P(candidate | image evidence)  +  λ · Σ trajectory consistency
```

This is the data-association problem from multi-target tracking, and it is
solvable with Viterbi over the graph or a min-cost flow.

**Why it is powerful here:** a wrong read that implies an impossible journey gets
demoted by evidence from a *different camera*. Your road graph — 35 real
junctions, 52 real links — stops being only an output and becomes an input to
recognition. That is a genuine research-grade contribution, and no
single-camera ANPR product can do it.

**Two dangers, both already understood in this codebase:**

- **Clones must be exempt.** Joint decoding assumes plausible travel; a clone is
  implausible travel. Run this over a cloned plate and it will smooth two real
  vehicles into one coherent journey and erase the anomaly — the exact failure
  `repair.py` guards against. Same exemption list.
- **It must not overwrite `sightings`.** Reassignments are inferences with an
  evidence chain, and the existing separation is what keeps that honest.

---

## 6. Using VLMs at their maximum

Ranked by value. Note that the best uses are **not** "read the plate".

### 6.1 VLM as erasure localiser — the highest-value use

This is the §1 mask, from a second source.

> *"Look at this number plate image. Do not read it. For each character
> position, say whether it is clearly legible, partly obscured, or unreadable,
> and say why — glare, blur, dirt, or occlusion."*

That is a **perceptual quality judgement**, not a recognition task. Hallucination
risk is far lower because you never ask for the answer, and it supplies the one
signal the CTC posteriors demonstrably cannot (AUROC 0.692). Cross-check it
against the photometric detectors in §1 and take the intersection — two
independent mask sources agreeing is strong.

### 6.2 VLM as verifier, not reader

> *"Does this image show the registration `WB01BF9492`? Answer yes or no, and
> list any character positions that differ."*

Discrimination is much easier than generation, the output space is tiny, and it
returns a per-character disagreement signal. Use it on the repair ladder's rung B
(§ REPAIR_VLM_PLAN.md) and to break ties between close CTC candidates.

### 6.3 VLM as multiple-choice ranker

> *"Which of these 36 registrations does this image show? Answer with one, or
> 'none of these'."*

The candidate set from `candidate_plates()` is a **closed answer space**, so the
model cannot hallucinate outside it — the structural fix for the biggest VLM
risk. Include "none of these" explicitly, or you force a wrong answer on the
10.3% where the truth is absent.

### 6.4 VLM as failure triage → adaptive restoration

Classify the failure mode (glare / motion blur / rain / occlusion / too small /
oblique), then route to the matching restoration in §3.3 instead of running all
of them on everything. Cheap, low risk, and it makes the restoration pipeline
adaptive rather than a fixed gauntlet.

### 6.5 VLM for vehicle attributes → an independent clone check

Two payoffs from one pass over the *vehicle* crop:

- It closes the documented `NOT_RECORDED` gap in `core/vehicle_search.py` —
  colour, make, body type — so "red truck heading north" filters on all three
  terms instead of two.
- **It gives clone detection a second modality.** `core/clones.py` currently
  rests on geometry and timing. "The vehicle carrying this registration is a
  white hatchback at camera A and a black SUV at camera B" is independent
  evidence of a clone, and it is evidence a plate-only system cannot obtain.

That second point may be worth more than the reading improvements. It attacks the
misread-versus-clone ambiguity that `clones.py` exists to manage.

### 6.6 Decomposed structured reading

If you do use a VLM to read, decompose it:

> *"Read only the two-letter state code."* → *"Read only the district digits."*
> → *"Read only the series letters."* → *"Read only the four-digit number."*

Each sub-answer has a small constrained space and can be validated against the
RTO table independently. A wrong state code is caught before it contaminates the
rest. Run each twice and require agreement (self-consistency); disagreement marks
that field as an erasure and hands it to §1.

---

## 7. What not to do

| Don't | Why |
|---|---|
| Replace the recogniser with a VLM | loses the calibrated CTC score that the floor, the clone screen, repair ranking and `_prefer()` all depend on |
| Inpaint glare or occlusion | synthesises strokes that read confidently as characters |
| Trust single-image SR as a replacement | invents texture; keep it as a variant only |
| Let inferences enter candidate sets or the lexicon | one mistake propagates along a corridor and manufactures a journey nobody made |
| Let joint decoding touch clone-flagged plates | it will smooth away the anomaly it exists to find |
| Report VLM-recovered plates in the headline accuracy | different error profile, different class of claim |

---

## 8. Ranked by expected value

Gains are estimates, not measurements, except where marked.

| # | Idea | § | Effort | Confidence | Est. gain |
|---:|---|---|---|---|---|
| 1 | **Erasure masking + constrained completion** | 1 | med | high | large on glare/rain/occlusion |
| 2 | **RTO district table + missing formats** | 2.1 | low | high | small overall, **fixes BH entirely** |
| 3 | **Burst sub-pixel SR** | 3.1 | med | high | large on far-lane / small plates |
| 4 | **Per-character ROI decode** | 3.2 | low | high | medium on long plates |
| 5 | Confidence calibration | 4.1 | low | high | no accuracy gain — makes every threshold principled |
| 6 | Network lexicon prior | 2.2 | low | med | medium, with the anti-chaining rules |
| 7 | VLM erasure localiser | 6.1 | med | med | multiplies #1 |
| 8 | Condition-aware frame weighting | 4.2 | low | med | small–medium |
| 9 | TTA fused in lattice space | 3.4 | low | med | small–medium |
| 10 | VLM multiple-choice ranker | 6.3 | med | med | targets the 10.3% ceiling |
| 11 | Zero-DCE / deblur as variants | 3.3 | low | med | medium on night |
| 12 | **Network joint decoding** | 5 | high | med | potentially largest; also the research contribution |
| 13 | VLM attributes → clone evidence | 6.5 | med | med | new capability, not accuracy |
| 14 | Per-camera state priors | 2.3 | low | low | small |

**Start with 2, 4 and 5.** All three are a day's work each, none needs a new
model, and #5 makes every subsequent experiment measurable in probability terms
rather than in arbitrary score units.

**Then 1 and 3**, which are the two large ones and both attack the failures you
have actually measured — glare/rain and small plates.

**Measure every one on the frozen 800-capture split** from
[REPAIR_VLM_PLAN.md](REPAIR_VLM_PLAN.md) §7, reporting yield and precision
separately. The 98.3% precision on accepted inferences is the number that must
not move.
