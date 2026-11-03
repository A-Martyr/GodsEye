# Reading plates the light got to first

An implementation plan for the photometric half of the reader: polarity,
colour, and low light. No retraining, no new model, and nothing on the live
path that a bright plate has to pay for.

---

## 0. The premise, corrected — and the measurement that motivates it

The starting idea was "plates are converted to grayscale before recognition, so
in dark conditions we could invert the gradient". Half right, and the half that
is wrong changes where the work should go.

**`anpr_ocr` does not grayscale.** `_preprocess()` keeps all three channels:

```python
img = _resize_for_rec(img_bgr, target_shape)
img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
return img.transpose((2, 0, 1))          # 3 x 48 x 320
```

The **CRNN** path does (`segment.to_gray`), and so do `salvage.column_reliability`
and the classical fallback. So colour is available to the 91% reader and thrown
away by everything else.

**What is genuinely missing is photometric variants at all.** `_crop_variants()`
produces four candidates and every one of them is geometric:

```
box  ·  tightened box  ·  bright plate face  ·  OBB-rectified
```

Not one changes a pixel value. The reader gets several framings of the same
exposure and never a second opinion about the exposure itself.

### The measurement

Over the 532-crop real corpus, splitting by whether the text is darker or
lighter than its background (Otsu, minority class):

| Class | n | Exact match | Mean luma |
|---|---:|---:|---:|
| dark text on light plate | 505 | **92.1%** | 151 |
| **light text on dark plate** | **27** | **70.4%** | 96 |
| | | | |
| crops with mean luma ≥ 90 | 507 | 91.9% | |
| **crops with mean luma < 90** | **25** | **72.0%** | |

**A 21.7-point deficit on inverted-polarity plates**, and a nearly identical
19.9-point deficit on dark ones. The two populations overlap almost entirely,
which is a hint that it is one phenomenon and not two.

Two honest caveats, stated before the plan leans on the number:

- **The samples are tiny.** 27 and 25 crops, so about 8 and 7 errors. On its own
  this is suggestive, not conclusive — and §5 exists because of that.
- **The aggregate upside is small.** Closing the gap entirely on 27 crops
  recovers ~6 plates, which is **1.1 points** of corpus accuracy. The corpus is
  overwhelmingly bright, daylight, dark-on-light.

The reason to do it anyway is in the next section.

### Why the small class matters more than its size

The corpus is not the deployment. Two arguments:

**The night photographs.** `FIELD_TEST.md` is nine phone images taken on a dark
street — precisely this population, and every one of them failed. A city ANPR
network runs all night.

**The failing polarity is the growing plate class.** Indian registration plates
are colour-coded, and the coding determines polarity:

| Plate | Background | Text | Polarity |
|---|---|---|---|
| Private | white | black | dark-on-light |
| Commercial / taxi | yellow | black | dark-on-light |
| **Electric vehicle** | **green** | **white** | **light-on-dark** |
| **Rental, self-drive** | **black** | **yellow** | **light-on-dark** |
| Temporary / some government | red | white | light-on-dark |

The two classes the reader is worst at are **electric vehicles and self-drive
rentals** — the two categories growing fastest on Indian roads. A reader tuned
on today's corpus is tuned on yesterday's fleet.

---

## 1. The architecture: every idea below is a proposer

`decode()` is already propose-and-rank, and `_score()` is an exact CTC
likelihood verified against `torch.nn.CTCLoss` to 2.8e-14. So:

> **A photometric transform is a new CROP VARIANT, never a replacement.**
> The original always stays in the pool. A transform that helps wins on
> evidence; one that destroys information produces a string the lattice does
> not support and loses on its own merits.

This is the same contract `_obb_rectified_crop()` already honours, and it is
what makes the whole family safe to try: the worst case of a bad variant is
wasted time, never a wrong answer that outranks a right one.

---

## 2. The trigger: spend effort where the image says to

The obvious gate is "run the extra variants when confidence is low". It is the
wrong one here, and the halves-decode experiment showed why: gating on
confidence alone spent two forward passes on 15% of reads and won nothing,
because a weak read is not evidence of *which* weakness.

We know exactly which images are affected, from statistics that cost microseconds:

```python
def photometric_profile(crop) -> dict:
    """Which photometric failure, if any, this crop is in. Two moments and an
    Otsu threshold - cheaper than a single decode by three orders of magnitude,
    so it can run on every read without appearing on the clock."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "luma": float(g.mean()),
        "inverted": bool((bw == 0).mean() > 0.5),   # text is the minority class
        "blown": float((g >= 245).mean()),
        "flat": float(g.std()),
    }
```

Route on the profile:

| Condition | Variants to add | Why |
|---|---|---|
| `inverted` | polarity inversion (§3.1) | measured 21.7-point deficit |
| `luma < 90` | CLAHE, gamma, Zero-DCE (§3.3) | measured 19.9-point deficit |
| `blown > 0.15` | highlight recovery (§3.4) | retro-reflective plates under IR |
| `std < 25` | local contrast normalisation | fog, haze, a dirty lens |
| none of these | **nothing — today's path exactly** | 92% of the corpus pays nothing |

That last row is the point. On this corpus roughly **95% of crops trigger no
extra work at all**, so the mean cost is a rounding error while the affected 5%
gets several attempts.

---

## 3. The variants, best first

### 3.1 Polarity inversion — the one with measured evidence

```python
def invert_variant(crop):
    """A light-on-dark plate, rendered the way the recogniser expects.

    Green EV and black rental plates carry light text on a dark ground. The
    model has seen overwhelmingly the opposite, and 255 - x costs nothing.
    """
    return 255 - crop
```

Add when `profile["inverted"]`. **Expected: recover part of the 21.7-point gap
on ~5% of crops; ≤1.1 points overall on this corpus, materially more at night.**

Note it must be applied to the **colour** crop, not a grayscale one — inverting
BGR preserves the channel relationships the model uses, and `_preprocess()` takes
three channels.

### 3.2 Best-channel selection — colour the greyscale throws away

**Measured, and it is the strongest evidence in this document.** Twenty-one real
Indian plates (`Downloads/Demo images`) through the shipped pipeline. Every one
was detected and read; the confidences split cleanly by plate colour:

| Plate colour | n | mean confidence | worst |
|---|---:|---:|---:|
| white | 11 | **0.994** | 0.958 |
| **yellow (commercial)** | **6** | **0.954** | **0.824** |

Five of the seven lowest-confidence reads in the set are yellow, and the only
ungrammatical output — `WB053587`, a dropped series letter — is yellow.

The cause is arithmetic, not the model. Ink-versus-plate separability
(Otsu between-class variance) per channel:

| Plate | blue | green | red | grayscale |
|---|---:|---:|---:|---:|
| white | 0.036 | 0.036 | 0.036 | 0.036 |
| **yellow** | **0.010** | 0.025 | **0.037** | 0.024 |

A white plate is achromatic, so all three channels carry identical information
and grayscale loses nothing. A yellow plate is `(B≈0, G≈255, R≈255)` with black
text at `(0,0,0)` — so in **blue, the background and the text are both near
zero** and the channel is very nearly noise. It carries 43% of the separability
it carries on a white plate, and `_preprocess()` normalises it to `[-1, 1]`
anyway, amplifying that noise to full scale before the model sees it.

Two consequences, one for each reader:

- **`anpr_ocr`** is handed three channels of which one is nearly empty. It has
  red available and still scores 4 points lower on yellow.
- **The CRNN and classical paths greyscale**, and grayscale scores **0.024
  against red's 0.037** — they discard **35% of the available contrast** on
  every commercial plate in the country.

Kolkata's taxi fleet is yellow-plated. So is every goods vehicle.

Grayscale is a fixed weighting, `0.299R + 0.587G + 0.114B`, and it is not the
best separator for a coloured plate:

| Plate | Background in BGR | Best channel for text contrast |
|---|---|---|
| Yellow, black text | (0, 255, 255) | **B** — background goes dark, text stays dark… **no**: R or G |
| Green, white text | (0, 128, 0) | **R** or **B** — background dark, text bright |
| White, black text | (255, 255, 255) | any |

Rather than reason per plate colour, pick empirically: compute Otsu separability
(between-class variance) on each of B, G, R and grayscale, and offer the winner
as a variant when it beats grayscale by a margin.

```python
def best_channel_variant(crop):
    """The channel that separates ink from plate best, as a 3-channel image."""
    best, best_sep = None, 0.0
    for ch in (cv2.split(crop) + [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)]):
        t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sep = _between_class_variance(ch, t)
        if sep > best_sep:
            best, best_sep = ch, sep
    return cv2.cvtColor(best, cv2.COLOR_GRAY2BGR)
```

This is the variant that most directly answers "we are throwing colour away" —
and it is worth noting it helps the **CRNN and classical paths most**, since
those genuinely do grayscale everything.

### 3.3 Low-light: three cheap ones and one free one

| Technique | Cost | Notes |
|---|---|---|
| **CLAHE** on L of LAB | ~0.1 ms | already used in `segment.normalize`; local, so it survives a half-lit plate |
| **Gamma** 0.5 and 0.7 | ~0.05 ms | two variants, trivial |
| **Local mean division** | ~0.3 ms | flattens a headlight gradient across the plate |
| **Zero-DCE** | ~5 ms | pretrained, **zero-reference** — needs no paired data, which is why it fits here |

Start with CLAHE + gamma. They are three lines and cover most of what Zero-DCE
would, and adding a model is a bigger commitment than adding a `cv2` call.

### 3.4 Highlight recovery — the counter-intuitive night case

The instinct at night is "the plate is too dark". Often it is the opposite.

Indian plates are **retro-reflective by regulation**. Under a camera's IR
illuminator they return light straight to the lens, so the plate face clips to
white while the rest of the scene stays black. The failure is a **blown
highlight**, and brightening makes it worse.

```python
if profile["blown"] > 0.15:
    # Compress the top of the range instead of stretching it, and pull the
    # exposure down before the recogniser normalises to [-1, 1].
    variants.append(cv2.convertScaleAbs(crop, alpha=0.6, beta=0))
    variants.append(_tone_map_highlights(crop))
```

`anpr/salvage.py` already computes the saturated-column share for its erasure
mask, so the detector exists — it just is not being used to route preprocessing.

### 3.5 What NOT to add

| | Why |
|---|---|
| Unconditional enhancement | changes the 92%-accurate majority path for the sake of 5% |
| Single-image super-resolution | invents texture; a hallucinated serif is a character |
| Glare inpainting | synthesised strokes read confidently — mask, do not fill (`CEILING.md` §1) |
| Histogram equalisation (global) | a half-lit plate has two populations; global EQ serves neither |

---

## 4. Colour as semantics, not just as contrast

The same pixels answer three questions the platform currently cannot.

### 4.1 Plate colour → vehicle class, with no VLM

`core/vehicle_search.py` reports `make`, `model` and `colour` as
recognised-but-not-recorded. **Vehicle class is a different matter** — on a real
capture it is currently unknown, and the plate itself declares it:

```
yellow background  ->  commercial (taxi, goods)
white background   ->  private
green background   ->  electric
black background   ->  rental, self-drive
```

A histogram over the plate face plus a nearest-centroid classifier in HSV. No
model, no training data, maybe forty lines. It fills a column the simulator
provides for free today and real cameras do not.

### 4.2 Plate colour → an independent clone signal

This is the one worth doing for its own sake.

`core/clones.py` decides a clone on **geometry and timing alone** — one plate,
two places, an impossible implied speed. Its hardest job is separating a real
clone from a misread, which is why it screens OCR-confusable plates.

Plate background colour is **independent evidence**. A registration that reads
yellow at Ultadanga and white at Gariahat is not a fast vehicle and not a
misread — it is two different plates. That is a second modality on the platform's
headline capability, from pixels already in hand.

```python
# in the sighting: plate_colour TEXT  ("white" | "yellow" | "green" | "black" | ...)
# in clones.py:    a colour disagreement RAISES the verdict; agreement does not
#                  lower it, because two clones can share a colour.
```

Asymmetric on purpose: disagreement is strong evidence, agreement is weak.

### 4.3 Plate colour → expected polarity

Closes the loop on §3.1. Once the colour is known, polarity is known a priori
rather than inferred from Otsu, which makes the inversion trigger more reliable
on exactly the degraded crops where Otsu is least trustworthy.

---

## 5. Measurement — and why this section is the hard part

**The corpus cannot measure this.** 27 inverted crops and 25 dark ones means any
change moves single-digit counts, and this session has twice produced a
confident wrong conclusion from exactly that: a calibration split that read
−0.093 and was really +0.236, and a repair precision "drop" that was 3 errors
against 5.

So the first task is not a variant. It is an evaluation set.

### 5.1 Build the affected-class corpus first

`anpr/camera.py` is an eight-stage physical camera model that already renders
`night_ir`, `night_glare`, `dusk_highiso` and `fog`. Point it at the **505
known-good bright crops' ground-truth labels** and render each under the dark
conditions, giving hundreds of labelled examples of the population that
currently has 25.

```bash
python -m anpr.camera_sheet          # see what each condition produces
```

Two things to check before trusting it, because a synthetic evaluation set can
flatter a fix that only works on synthetic degradation:

- does the renderer produce **light-on-dark** plates at all, or only darkened
  dark-on-light ones? If the latter, §3.1 cannot be evaluated on it and needs
  real inverted plates instead.
- does it model **retro-reflective blow-out** (§3.4), or only under-exposure?

If the answer to either is no, that is a finding about the camera model worth
fixing on its own — it means the platform has never simulated the failure mode
its own field test just demonstrated.

### 5.2 Report per-class, never in aggregate

A blanket corpus number will hide a 20-point gain on 5% of crops inside noise.
Every experiment reports:

| | inverted | dark | blown | rest | overall |
|---|---|---|---|---|---|
| baseline | 70.4% | 72.0% | — | 92.1% | 91.0% |
| + variant | | | | **must not move** | |

**The "rest" column is the guard.** These variants are gated, so they should not
touch the 92% majority at all. If that column moves, the trigger is leaking and
the change is unsafe regardless of what the affected classes did.

### 5.3 Acceptance and kill criteria

Ship a variant when, on the affected class and over **≥5 resamples**:

- the affected-class accuracy rises by **≥5 points**, and
- the unaffected class moves by **less than 0.5 points**, and
- mean added latency over the whole corpus is **under 5 ms**.

Abandon it when the affected-class gain is inside the resampling spread — which
for 27 crops is roughly ±9 points, and is the reason §5.1 comes first.

---

## 6. Cost

Per read, on this corpus's distribution:

```
profile computation      ~0.05 ms   every read
inverted     (5% of reads)  1 extra decode
dark         (5%)           2 extra decodes
blown        (?)            2 extra decodes
                            ------------------
mean added cost          ~0.15 decodes/read  ≈ +3%
worst case (one crop)    +4 decodes
```

Against a 108 ms warm whole-photograph read this is inside the noise, and it is
spent only on crops that are currently failing seven times in ten.

---

## 7. Order of work

| # | Task | Why here |
|---:|---|---|
| 1 | `photometric_profile()` + per-class reporting in `compare_backends` | everything downstream needs the split |
| 2 | **Build the affected-class evaluation set** (§5.1) | without it nothing after is measurable |
| 3 | Polarity inversion variant (§3.1) | the only one with measured evidence behind it |
| 4 | CLAHE + gamma variants (§3.3) | three lines, covers most of low light |
| 5 | Best-channel selection (§3.2) | biggest win for the CRNN/classical paths |
| 6 | Highlight recovery (§3.4) | needs the blown-crop set to exist first |
| 7 | Plate-colour classifier (§4.1) | independent of 3–6; unlocks 4.2 and 4.3 |
| 8 | Colour as clone evidence (§4.2) | the highest-value item, and the least about accuracy |

Items 1–2 are prerequisites and worth doing even if every variant is then
abandoned: a reader whose failures cannot be split by cause cannot be improved
except by luck.

## 8. What this will and will not do

**Will:** materially improve a small, growing, currently-worst class — EV green
plates, black rental plates, and anything photographed after dark. Give clone
detection a second modality. Fill `vehicle_class` from the image on real
captures.

**Will not:** move the headline 91.0% by more than about a point on this corpus,
because this corpus is bright daylight and the affected class is 5% of it. The
honest claim is *"our worst class improved by X points"*, and quoting an
aggregate would understate the engineering and overstate the corpus.
