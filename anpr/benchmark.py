"""Accuracy benchmark for the ANPR engine.

Reports plate-level (exact string) and character-level accuracy for every
condition in the degradation model, plus the accuracy achieved when reads below
the confidence floor are rejected — which is how the platform actually runs, and
what an operator cares about: of the plates the system *claims* to have read,
how many are right.

    python -m anpr.benchmark --samples 120
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np

import config
from anpr import ocr, plates


@dataclass
class ConditionResult:
    condition: str
    samples: int
    plate_accuracy: float
    char_accuracy: float
    accepted: int
    accepted_accuracy: float
    mean_confidence: float


def char_accuracy(truth: str, pred: str) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for a, b in zip(truth, pred) if a == b)
    return hits / max(len(truth), len(pred))


def run(samples: int = 120, seed: int = 2024, conditions=None, verbose: bool = True):
    eng = ocr.engine()
    rng = random.Random(seed)
    conditions = conditions or plates.CONDITIONS
    results: list[ConditionResult] = []
    t0 = time.time()
    total_reads = 0

    for cond in conditions:
        exact = chars = acc_exact = accepted = 0
        confs = []
        for _ in range(samples):
            cap = plates.capture(None, cond, rng)
            read = eng.read(cap.image)
            total_reads += 1
            ok = read.text == cap.text
            exact += ok
            chars += char_accuracy(cap.text, read.text)
            confs.append(read.confidence)
            if read.confidence >= config.MIN_PLATE_CONFIDENCE:
                accepted += 1
                acc_exact += ok
        results.append(ConditionResult(
            condition=cond, samples=samples,
            plate_accuracy=exact / samples,
            char_accuracy=chars / samples,
            accepted=accepted,
            accepted_accuracy=(acc_exact / accepted) if accepted else 0.0,
            mean_confidence=float(np.mean(confs)),
        ))
        if verbose:
            r = results[-1]
            print(f"  {cond:12} plate {r.plate_accuracy:6.1%}   char {r.char_accuracy:6.1%}   "
                  f"accepted {r.accepted:4}/{samples}  of which {r.accepted_accuracy:6.1%}")

    overall = ConditionResult(
        condition="OVERALL", samples=samples * len(conditions),
        plate_accuracy=float(np.mean([r.plate_accuracy for r in results])),
        char_accuracy=float(np.mean([r.char_accuracy for r in results])),
        accepted=sum(r.accepted for r in results),
        accepted_accuracy=float(np.average(
            [r.accepted_accuracy for r in results],
            weights=[max(r.accepted, 1e-9) for r in results])),
        mean_confidence=float(np.mean([r.mean_confidence for r in results])),
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"\n  {'OVERALL':12} plate {overall.plate_accuracy:6.1%}   "
              f"char {overall.char_accuracy:6.1%}   "
              f"accepted-read accuracy {overall.accepted_accuracy:6.1%}")
        print(f"  {total_reads} reads in {elapsed:.1f}s "
              f"({total_reads/max(elapsed,1e-9):.1f} plates/s single-threaded)")
    return results, overall, elapsed


def run_scenes(samples: int = 40, seed: int = 99, verbose: bool = True):
    """Measure the whole upload path: find the plate in a frame, then read it.

    Reported separately from the crop benchmark because it measures two things
    at once - localisation and reading - and a failure of either looks the same
    to an operator dragging a photograph onto the dashboard.
    """
    from anpr import ocr as ocr_mod

    eng = ocr.engine()
    rng = random.Random(seed)
    t0 = time.time()
    located = read_ok = 0
    for i in range(samples):
        sc = plates.scene(None, plates.CONDITIONS[i % len(plates.CONDITIONS)], rng)
        cands = ocr_mod.detect_plate_candidates(sc.image)
        if any(ocr_mod._iou(c, sc.box) > 0.3 for c in cands):
            located += 1
        r = eng.read_frame(sc.image)
        read_ok += r.text == sc.text
    elapsed = time.time() - t0
    out = {"samples": samples, "localised": located / samples,
           "read_accuracy": read_ok / samples, "elapsed_s": elapsed,
           "seconds_per_frame": elapsed / max(samples, 1)}
    if verbose:
        print(f"  whole frames: plate located in {located}/{samples} "
              f"({out['localised']:.0%}), read correctly {read_ok}/{samples} "
              f"({out['read_accuracy']:.0%}), {out['seconds_per_frame']:.2f}s per frame")
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark the GodsEye ANPR engine")
    ap.add_argument("--samples", type=int, default=120, help="plates per condition")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--json", type=str, default="", help="write results to this path")
    ap.add_argument("--scenes", type=int, default=0,
                    help="also measure the whole-frame path on N composited scenes")
    args = ap.parse_args()

    print(f"GodsEye ANPR benchmark - {args.samples} plates x {len(plates.CONDITIONS)} conditions\n")
    results, overall, elapsed = run(args.samples, args.seed)
    scenes = run_scenes(args.scenes, args.seed) if args.scenes else None
    if args.json:
        payload = {"conditions": [asdict(r) for r in results], "overall": asdict(overall),
                   "elapsed_s": elapsed, "samples_per_condition": args.samples,
                   "scenes": scenes}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
