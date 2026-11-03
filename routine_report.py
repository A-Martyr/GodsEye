"""One comparable block of numbers about a GodsEye database.

Run it against the database before a change and after it; the two outputs are
meant to be diffed. Point it anywhere with GODSEYE_DB.

    GODSEYE_DB=...\\godseye-before-routine.db py -3.13 routine_report.py BEFORE
    GODSEYE_DB=...\\godseye.db                 py -3.13 routine_report.py AFTER

The headline is TEMPORAL RATIO. It is the mean total-variation distance between
a plate's own hour-of-day profile and the city's, divided by the same quantity
computed on counts resampled FROM the city's profile. 1.0 means a vehicle's
schedule is indistinguishable from the city's demand curve - it has no routine
of its own. Below 1.0 means the profiles are smoother than multinomial noise,
which is what continuous movement through time produces.
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from collections import defaultdict

from core import db, forecast
from core.network import network

LABEL = sys.argv[1] if len(sys.argv) > 1 else "REPORT"
RNG = random.Random(20260906)      # fixed: the null must not move between runs


def tvd(d1: dict, d2: dict) -> float:
    return 0.5 * sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in set(d1) | set(d2))


def resample_null(reference: dict, n: int) -> dict:
    keys = list(reference)
    weights = [reference[k] for k in keys]
    draw = defaultdict(int)
    for k in RNG.choices(keys, weights, k=n):
        draw[k] += 1
    return {k: v / n for k, v in draw.items()}


def main() -> None:
    net = network()
    print(f"\n{'=' * 72}\n{LABEL}   db={db.config.DB_PATH if hasattr(db, 'config') else ''}\n{'=' * 72}")

    st = db.stats()
    print(f"store: {st['sightings']} sightings · {st['unique_plates']} plates · "
          f"{st['alerts']} alerts ({st['open_alerts']} open)")
    span_h = (st["last_ts"] - st["first_ts"]) / 3600.0 if st["first_ts"] else 0.0
    print(f"span : {span_h:.1f} h")

    rows = db.rows("SELECT plate, camera_id, ts FROM sightings ORDER BY plate, ts")

    # ---- 1. temporal signal: does a plate have an hour profile of its own? ----
    by_plate_hour = defaultdict(lambda: defaultdict(int))
    global_hour = defaultdict(int)
    for r in rows:
        h = time.localtime(r["ts"]).tm_hour
        by_plate_hour[r["plate"]][h] += 1
        global_hour[h] += 1
    gtot = sum(global_hour.values()) or 1
    gdist = {h: global_hour[h] / gtot for h in range(24) if global_hour[h]}
    city_peak = max(gdist, key=gdist.get) if gdist else 0

    obs, null = [], []
    for p, hrs in by_plate_hour.items():
        n = sum(hrs.values())
        if n < 20:
            continue
        obs.append(tvd({h: k / n for h, k in hrs.items()}, gdist))
        null.append(tvd(resample_null(gdist, n), gdist))
    t_ratio = statistics.fmean(obs) / statistics.fmean(null) if obs and null else float("nan")
    print("\n-- TEMPORAL signal (hour-of-day) --")
    print(f"  city busiest hour       : {city_peak:02d}:00 ({gdist.get(city_peak, 0):.1%})")
    print(f"  plates with >=20 sightings: {len(obs)}")
    print(f"  observed mean TVD       : {statistics.fmean(obs):.4f}" if obs else "  none")
    print(f"  null     mean TVD       : {statistics.fmean(null):.4f}" if null else "")
    print(f"  TEMPORAL RATIO          : {t_ratio:.3f}   (1.0 = no personal schedule)")

    # ---- 2. spatial signal: does a plate have turns of its own? --------------
    legs_t = []
    seq = defaultdict(list)
    for r in rows:
        seq[r["plate"]].append((r["ts"], r["camera_id"]))
    for p, s in seq.items():
        for (t1, c1), (t2, c2) in zip(s, s[1:]):
            g = (t2 - t1) / 60.0
            if 0 < g <= forecast.MAX_LEG_MINUTES and c1 != c2:
                legs_t.append((t2, p, c1, c2, g))
    legs_t.sort()
    prior_all = forecast.TransitionModel([(a, b, m) for _, _, a, b, m in legs_t])
    own_all = defaultdict(list)
    for _, p, a, b, m in legs_t:
        own_all[p].append((a, b, m))

    s_obs, s_null, pairs = [], [], 0
    for p, lg in own_all.items():
        m = forecast.TransitionModel(lg)
        for a, outs in m.counts.items():
            n = sum(outs.values())
            crowd_c = prior_all.counts.get(a, {})
            tot = sum(crowd_c.values())
            if n < 6 or tot < 30:
                continue
            crowd = {c: k / tot for c, k in crowd_c.items()}
            s_obs.append(tvd({c: k / n for c, k in outs.items()}, crowd))
            s_null.append(tvd(resample_null(crowd, n), crowd))
            pairs += 1
    s_ratio = statistics.fmean(s_obs) / statistics.fmean(s_null) if s_obs else float("nan")
    print("\n-- SPATIAL signal (which turn) --")
    print(f"  legs                    : {len(legs_t)}")
    print(f"  (plate,junction) pairs >=6 departures: {pairs}")
    print(f"  SPATIAL RATIO           : {s_ratio:.3f}   (1.0 = no personal corridor)")

    # ---- 3. held-out next-junction accuracy ---------------------------------
    cut = int(0.8 * len(legs_t))
    train, test = legs_t[:cut], legs_t[cut:]
    prior = forecast.TransitionModel([(a, b, m) for _, _, a, b, m in train])
    own_legs = defaultdict(list)
    for _, p, a, b, m in train:
        own_legs[p].append((a, b, m))
    owns = {p: forecast.TransitionModel(v) for p, v in own_legs.items()}
    EMPTY = forecast.TransitionModel([])

    res = defaultdict(lambda: [0, 0, 0])
    eta_err = []
    for _, p, a, b, mins in test:
        nb = net.neighbours(a)
        if not nb:
            continue
        om = owns.get(p, EMPTY)
        variants = {
            "blended": forecast._blended(om, prior, a, net),
            "crowd":   forecast._blended(EMPTY, prior, a, net),
            "uniform": {c: 1.0 / len(nb) for c in nb},
        }
        for name, d in variants.items():
            if not d:
                continue
            top = [c for c, _ in sorted(d.items(), key=lambda kv: -kv[1])[:3]]
            s = res[name]
            s[0] += 1
            s[1] += (top[0] == b)
            s[2] += (b in top)
        pred = om.median_minutes(a, b) or prior.median_minutes(a, b) or net.free_flow_minutes(a, b)
        if pred:
            eta_err.append(abs(pred - mins))
    print(f"\n-- held-out next junction (train {len(train)} / test {len(test)}) --")
    print(f"  {'model':<10}{'n':>7}{'top-1':>9}{'top-3':>9}")
    for name in ("blended", "crowd", "uniform"):
        n, t1, t3 = res[name]
        if n:
            print(f"  {name:<10}{n:>7}{t1/n:>8.1%}{t3/n:>9.1%}")
    if eta_err:
        print(f"  ETA median abs error  : {statistics.median(eta_err):.2f} min")

    # ---- 4. what the shipped gate says --------------------------------------
    top_plates = [r["plate"] for r in db.rows(
        "SELECT plate, COUNT(*) n FROM sightings GROUP BY plate ORDER BY n DESC LIMIT 40")]
    pol = [forecast.pattern_of_life(p) for p in top_plates]
    rel = sum(1 for x in pol if x.get("reliable"))
    conf = sum(1 for p in top_plates if forecast.predict(p).get("confident"))
    print("\n-- shipped gates, on the 40 most-sighted plates --")
    print(f"  pattern_of_life reliable : {rel}/40")
    print(f"  predict confident        : {conf}/40")
    if any("routine_supported" in x for x in pol):
        sup = sum(1 for x in pol if x.get("routine_supported"))
        print(f"  routine_supported        : {sup}/40")

    # ---- 5. alert mix (a reseed must not wreck the demo) --------------------
    kinds = db.rows("SELECT kind, COUNT(*) n FROM alerts GROUP BY kind ORDER BY n DESC")
    print("\n-- alerts by kind --")
    for k in kinds:
        print(f"  {k['kind']:<12}{k['n']:>6}")


if __name__ == "__main__":
    main()
