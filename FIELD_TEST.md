# Field test: nine night photographs from a phone

Nine images shot on a phone on a Kolkata street after dark, run through the
`anpr_ocr` reader on 7 September 2026. Every one of them failed or returned
nonsense through the shipped pipeline. This is what actually went wrong, in the
order the pipeline goes wrong in.

The short version: **the recogniser is not the problem.** Given a correctly
located plate it read the only fully-legible registration in the set exactly,
twice, at 0.97 and 1.00 confidence. The pipeline fails earlier — the localiser
never finds the plate, because the photographs are downscaled 2.5× before it
sees them.

---

## What was tested

| | |
|---|---|
| Images | 9, JPEG, from WhatsApp |
| Dimensions | 1200×1600 (portrait) ×3, 1600×1200 (landscape) ×5, 772×1379 ×1 |
| Conditions | night, street lighting, phone camera |
| Subjects | parked and moving vehicles at 3–15 m, mostly oblique |
| Backend | `anpr_ocr` (PP-OCRv5), paddle on `gpu:0` |
| Localiser | `models/yolov8n-obb-plate.pt` via `detect_plate_candidates()` |

---

## Result through the shipped pipeline

| # | Detector boxes | `read_frame()` returned | Confidence |
|---|---:|---|---:|
| 1 | **0** | `KA261` | 0.14 |
| 2 | **0** | `KPH82` | 0.06 |
| 3 | **0** | `A01` | 0.11 |
| 4 | **0** | `PAKA52` | 0.11 |
| 5 | 2 | `KAM81` | 0.23 |
| 6 | **0** | `UAD22E` | 0.23 |
| 7 | 1 | `KA0EM04L` | 0.36 |
| 8 | 2 | `DL14A1` | 0.28 |
| 9 | **0** | `KA82` | 0.10 |

**Six of nine returned no detection at all.** Where the whole-frame fallback
ran, it produced strings that are not registrations, at confidences of 0.06 to
0.36 — all far below `CLONE_MIN_CONFIDENCE` (0.70), so nothing here would have
entered the sighting store as evidence. The platform was wrong, but it was not
*confidently* wrong, which is the failure mode that matters.

### The three detections were not plates

The regions the detector did return, enlarged:

| Crop | What it actually is |
|---|---|
| `05_0`, `05_1` | the edge of a blue shop awning |
| `07_0` | a pink neon sign reading `SANSKRIT…` |
| `08_0` | a painted board reading `DALIAPAR…` |
| `08_1` | a red sign reading `SPEED` |

At night the brightest high-contrast rectangles of text in a street scene are
shop signs, and that is what a plate detector starved of resolution locks onto.

---

## Stage 1 — the localiser, and the actual root cause

`_detect_plate_candidates_yolo()` calls the model with ultralytics' defaults.
The confidence default (0.25) is documented in a comment there. The one that
does the damage is not mentioned anywhere: **`imgsz=640`**.

A 1600×1200 photograph is letterboxed to 640 wide before the model sees it — a
2.5× reduction. A plate 100 px across in the original is 40 px by the time it
is being detected, and the model was fine-tuned on dashcam and Roboflow frames
where plates are several times larger.

Sweeping both knobs separates the two explanations. Total detections across all
nine images:

| `imgsz` | conf 0.25 | conf 0.10 | conf 0.05 | conf 0.01 |
|---:|---:|---:|---:|---:|
| **640** (shipped) | **5** | 12 | 20 | 40 |
| 960 | 11 | 18 | 25 | 62 |
| 1280 | 13 | 18 | 25 | 42 |
| **1600** | **18** | 25 | 28 | 67 |

Resolution, not confidence. At the *same* 0.25 threshold, raising `imgsz` to
1600 takes detections from 5 to 18 — and they are real plates, at real
confidence, not marginal ones scraped off the bottom of the threshold:

| # | boxes at 640 | boxes at 1600 | best detection confidence |
|---|---:|---:|---:|
| 1 | 0 | 14 | **0.899** |
| 2 | 0 | 10 | 0.644 |
| 3 | **0** | 1 | **0.840** |
| 4 | 0 | 7 | **0.861** |
| 5 | 2 (signage) | 9 | 0.570 |
| 6 | **0** | 1 | **0.901** |
| 7 | 1 (signage) | 8 | **0.900** |
| 8 | 2 (signage) | 16 | **0.904** |
| 9 | **0** | 1 | 0.392 |

Images 3, 6 and 9 found nothing whatsoever at 640 and are found at 0.84, 0.90
and 0.39 at 1600. Image 8's best detection goes from a `SPEED` sign to a plate
at 0.904.

---

## Stage 2 — the recogniser, given a correct crop

Cropping each plate at `imgsz=1600, conf=0.25` and handing it straight to the
recogniser:

| # | detector conf | crop size | recogniser read | conf | plate says |
|---|---:|---:|---|---:|---|
| 1 | 0.90 | 46×14 | `MA1B81` | 0.11 | illegible |
| 2 | 0.64 | 58×13 | `MNREAA4B05` | 0.59 | illegible |
| 3 | 0.84 | 104×29 | `WB05555` | 0.51 | illegible |
| 4 | 0.86 | 46×13 | `KNRE3149` | 0.42 | illegible |
| 5 | 0.57 | 49×13 | `KAN81` | 0.18 | not a plate |
| 6 | 0.90 | 70×33 | `MZ5BH3028` | **0.02** | **`25BH3028…`** |
| 7 | 0.90 | 71×22 | **`WB01BF9492`** | **0.97** | **`WB01BF9492`** |
| 8 | 0.90 | 79×23 | **`WB01BF9492`** | **1.00** | **`WB01BF9492`** |
| 9 | 0.39 | 72×47 | `UNREA5B2E` | 0.64 | illegible |

**Only three plates in this set are legible to a human.** On those, the
recogniser scored **2 of 3 exact** — reading `WB01BF9492` correctly from two
different photographs of the same car, at 0.97 and 1.00.

The other six are not recogniser failures. At 46×14 px and 58×13 px there is no
registration in the image to recover; a human staring at the enlarged crop
cannot read them either. Reporting those as reader errors would be measuring
the wrong thing.

---

## The one real recogniser defect: the Bharat (BH) series

Image 6 is a `25 BH 3028` plate — the Bharat series, issued since 2021 for
vehicles that move between states. The reader returned `MZ5BH3028` at **0.02**
confidence.

The cause is in the grammar, `anpr/ocr.py`:

```python
PATTERNS: list[str] = [
    "LL" + "D" * d + "L" * s + "DDDD"
    for d in (1, 2)
    for s in (3, 2, 1, 0)
]
```

Every one of the eight patterns begins `LL` — two letters, a state code. A
Bharat plate begins with two **digits**, the year of registration:

```
WB01BF9492  -> LLDDLLDDDD  legal = True
25BH3028AB  -> DDLLDDDDLL  legal = False
25BH3028    -> DDLLDDDD    legal = False
```

`"BH"` *is* present in `STATE_CODES`, which looks like support but is not:
the code checks the first two characters, and on a BH plate those are the year
digits, never `BH`.

So `_grammatical_readings()` cannot propose the correct string, and the decode
falls back to an ungrammatical reading — which the module is designed to allow
rather than return nothing. The 0.02 confidence is the system correctly
reporting that it does not believe its own answer.

**This is a genuine gap, not a tuning problem.** BH plates are increasingly
common on exactly the interstate traffic an ANPR network watches for. The fix
is to add the BH shape to `PATTERNS` and special-case the state-code check for
it — a bounded change, but one that needs its own test, because a grammar that
admits `DD` prefixes gives every misread number two more ways to look legal.

---

## The fix for the localiser, and what it costs

Pass `imgsz` explicitly in `_detect_plate_candidates_yolo()` instead of taking
ultralytics' 640 default. Measured on this GPU, over these nine images:

| `imgsz` | detector latency | plates found |
|---:|---:|---:|
| 640 (shipped) | 23.2 ms | 5 |
| 960 | 26.5 ms | 11 |
| 1280 | 32.3 ms | 13 |
| **1600** | **38.8 ms** | **18** |

**+15.6 ms for 3.6× the detections.** Against a warm whole-photograph read of
~108 ms (`PERFORMANCE.md`), that takes a read to roughly 124 ms — still inside
the same order, and far inside the 46 s the pipeline started at.

Two things to check before shipping it:

- **Re-run the benchmark.** `anpr.benchmark`'s `scenes` / `whole frames`
  sub-test is the only part that exercises the detector at all, so it is the
  only number that should move. The headline per-condition and burst numbers
  feed `ANPREngine.read()` a pre-made crop and never call the localiser — they
  must come back unchanged, and if they do not, something else changed.
- **More boxes is more recogniser calls.** At `imgsz=1600` image 8 returns 16
  candidate regions rather than 2. `read_frame()` is cost-aware and early-exits
  once a grammatical plate clears confidence, so the worst case is bounded — but
  the worst case is what a frame with no plate in it hits every time.

A cheaper variant worth measuring: scale `imgsz` to the image, capped, so a
1600-wide photograph is detected at its own resolution while a 640-wide camera
frame costs exactly what it costs today.

---

## What this test does and does not show

**Shows.** The localiser is the binding constraint on real photographs at this
resolution, and it is constrained by a default nobody chose. The recogniser is
in good shape: given a plate a human can read, it read it exactly, twice.

**Does not show** that accuracy on real photographs is 2/9, or 22%, or any
number of that shape. Six of these nine images contain no legible registration
at any resolution — they are photographs of a street that happen to have
vehicles in them, at distances where the plate occupies under 60 px. The
91.0% figure in `README.md` is measured over a 532-crop corpus of plates that
are actually present and readable, and this set is not a replacement for it.

The useful reading is the conditional one: **of the plates that were there to
be read, the reader read two of three — and the third failed on a plate format
the grammar does not yet describe.**

## Reproducing

```bash
# stage 1: what the shipped pipeline does
./.venv-paddle/Scripts/python.exe <scratch>/diag_batch.py "<folder>" <out>

# the imgsz / confidence sweep
py -3.13 <scratch>/sweep.py

# stage 2: recogniser on correctly-localised crops
./.venv-paddle/Scripts/python.exe <scratch>/stage2.py
```

Stop the 8502 dashboard first. It holds a CUDA context on a 4 GB card, and a
second one starves both — see the operational note at the end of `FEATURES.md`.

---

# Addendum: the Kolkata taxi, and why `imgsz` is not the whole story

A second image, tested after the nine above: a yellow Ambassador taxi,
front-on, daylight, plate `WB 04 E 7711` large and sharp in a 591x716 frame. It
failed, returning `KKAC81` at 0.156.

The pipeline found **one** box at the shipped `imgsz=640` — a 35x14 px blue
smear on a bus in the background, at 0.332 — and read that.

## Four hypotheses, three killed by measurement

**Not the recogniser.** Handed the true plate's box directly, it returns
`WB04E7711` at confidence **1.0000**. The recognition half of this platform is
not involved in the failure.

**Not yellow-plate blindness in the detector.** Over 21 real plates, detector
confidence by plate colour: yellow **0.831** mean, white 0.803. Yellow plates
are read slightly worse by the RECOGNISER (0.954 vs 0.994, see PHOTOMETRIC.md)
but they are not harder to FIND.

**Not low contrast against the bodywork.** A yellow plate on a yellow car looks
like it should vanish into it. Measured as LAB distance between the plate and
the panel immediately around it, this plate scores 78.1 against a 46.3 median
over the same 21 images - the **81st percentile**. It stands out better than
most.

**It is scale.** Confidence assigned to the true plate, and its rank among all
boxes, as the detector's input size varies:

    imgsz   true-plate conf   rank      top box
      640            0.024      13        0.332
      800        not found       -        0.282
      960            0.194       1        0.194
     1120            0.331       1        0.331
     1280            0.569       1        0.569
     1440            0.069       6        0.244
     1600            0.210       3        0.677
     1920            0.011      13        0.506

Non-monotonic, and swinging by a factor of fifty. 1920 is worse than 960. This
is not "the image is too small for the detector", which is what the nine night
photographs showed and what raising `imgsz` fixes. It is a **narrow scale band**:
the model responds to a plate whose apparent size lands where its training data
put plates, and this one lands there at roughly 190-250 px and nowhere else.

At the shipped 640 the plate is ~127 px and scores 0.024, ranked **13th of 24
boxes** - so far below `conf=0.25` that no threshold change would recover it
without admitting a dozen pieces of bus.

## The fix is multi-scale, not bigger

Running the detector at several sizes and pooling the boxes:

    single 640 (today)   ->  'KKAC81'      0.156   wrong
    multi-scale 960/1280/1600  ->  'WB04E7711'   1.0000  correct

Which also reconciles this with the nine night photographs. There, `imgsz=1600`
took detections from 5 to 18. Here, 1600 is worse than 1280 and 1920 is worse
than 960. Both are the same fact seen from different sides: **there is no single
right input size, because the right size depends on how large the plate happens
to be in the frame** - and that is exactly what the detector is being asked to
discover.

Cost is one detector pass per scale, about 25-40 ms each on this GPU. Gate it on
the first pass returning nothing convincing, and a clean photograph pays for one
pass as it does today.

## Implemented, and what it actually bought

Shipped as `config.YOLO_ESCALATE_SCALES = (960, 1280, 1600)`, escalating only
when the first pass produced no ACCEPTED read - never on detector confidence,
which awarded 0.33 to the bus. `GODSEYE_YOLO_ESCALATE=0` restores the old
behaviour.

| Population | Before | After | Latency |
|---|---|---|---|
| The taxi | `KKAC81` 0.16 | **`WB04E7711` 1.00** | 88 ms -> 702 ms |
| 21 demo photos (all already worked) | - | **21/21 identical** | 116 -> 105 ms median (noise) |
| 9 night photos (all failed) | all nonsense | **all 9 changed**, 4+ now real plates at >=0.94 | 101 -> 661 ms median |
| 532-photo `--real-frames` corpus | 88% / 90% | **88% / 90% unchanged** | 0.15 -> 0.21 s/read |

Read that last row honestly: **on the corpus this buys nothing and costs 40%
mean latency.** That corpus is dashcam frames where the detector already works,
so only its ~11% failures escalate, and escalating does not rescue them - they
fail for want of pixels, not for want of scale.

What it does rescue is the population the corpus does not contain: phone
photographs, awkward subject distances, and the plate that happens to land
outside the detector's scale band. Those are exactly the images a person points
at a car, and exactly where the platform was returning a piece of a bus.

The cost is paid only by frames that already failed, and never by one that
read cleanly - which is why the 21 photographs that worked before are
byte-identical and no slower.
