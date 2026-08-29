"""Head-to-head: the classical segment-and-classify engine against the CRNN.

    python -m anpr.compare_backends --samples 100

Both read the *same* captures, so the comparison isolates the recogniser. This
is the measurement that decides which backend the platform should ship with,
and it is the one to re-run after any change to the camera model - a corpus
change moves both numbers and only the gap between them is meaningful.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import config
from anpr import crnn as crnn_mod
from anpr import ocr, plates


def run(samples: int = 100, seed: int = 4242, verbose: bool = True) -> dict:
    classical = ocr.ANPREngine(use_crnn=False)
    reader = crnn_mod.load()
    if reader is None:
        raise SystemExit("no CRNN weights at models/crnn.pt - run "
                         "python -m anpr.train_crnn first")
    neural = ocr.ANPREngine(crnn_reader=reader)

    out: dict[str, dict] = {}
    if verbose:
        print(f"{'scenario':14} {'classical':>20}   {'CRNN':>20}")
    for cond in plates.CONDITIONS:
        rng = random.Random(seed)
        caps = [plates.capture(None, cond, rng) for _ in range(samples)]
        stats = {}
        for name, eng in (("classical", classical), ("crnn", neural)):
            t0 = time.time()
            ok = kept = kept_ok = 0
            for c in caps:
                r = eng.read(c.image)
                hit = r.text == c.text
                ok += hit
                if r.accepted:
                    kept += 1
                    kept_ok += hit
            stats[name] = {
                "plate_accuracy": ok / samples,
                "stored": kept,
                "stored_accuracy": (kept_ok / kept) if kept else 0.0,
                "ms_per_read": (time.time() - t0) / samples * 1000,
            }
        out[cond] = stats
        if verbose:
            c, n = stats["classical"], stats["crnn"]
            print(f"{cond:14} {c['plate_accuracy']:6.0%} "
                  f"(stored {c['stored']:3}, {c['stored_accuracy']:4.0%})   "
                  f"{n['plate_accuracy']:6.0%} "
                  f"(stored {n['stored']:3}, {n['stored_accuracy']:4.0%})", flush=True)

    # Snapshot the per-scenario rows first: adding OVERALL to `out` while
    # iterating over it is what made the second backend disappear.
    rows = dict(out)
    overall = {}
    for name in ("classical", "crnn"):
        acc = sum(v[name]["plate_accuracy"] for v in rows.values()) / len(rows)
        stored = sum(v[name]["stored"] for v in rows.values())
        stored_ok = sum(v[name]["stored"] * v[name]["stored_accuracy"]
                        for v in rows.values())
        overall[name] = {
            "plate_accuracy": acc,
            "stored": stored,
            "stored_accuracy": (stored_ok / stored) if stored else 0.0,
            "ms_per_read": sum(v[name]["ms_per_read"] for v in rows.values()) / len(rows),
        }
    out["OVERALL"] = overall
    if verbose:
        o = out["OVERALL"]
        print(f"\n{'OVERALL':14} {o['classical']['plate_accuracy']:6.1%} "
              f"(stored {o['classical']['stored']}, "
              f"{o['classical']['stored_accuracy']:.0%})   "
              f"{o['crnn']['plate_accuracy']:6.1%} "
              f"(stored {o['crnn']['stored']}, {o['crnn']['stored_accuracy']:.0%})")
        print(f"{'speed':14} {o['classical']['ms_per_read']:6.0f} ms/read"
              f"{'':>14}{o['crnn']['ms_per_read']:6.0f} ms/read")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the two GodsEye recognisers")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    out = run(args.samples, args.seed)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
