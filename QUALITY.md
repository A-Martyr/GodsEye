# Quality evidence

The SIH 2026 judges asked for objective evidence of code quality. This file
records what was measured, when, and with what, so anyone can re-run it.
Last run: 24 October 2026, Python 3.13.3, torch 2.13, Windows 11.

## End-to-end self-check

```bash
python tests.py
```

**101 passed, 0 failed.** The suite walks the whole platform against a scratch
database: reads plates, writes sightings, reconstructs a trajectory through a
deliberately misread plate, computes analytics, fires every alert rule, checks
the privacy and enrolment rules, and injects a live clone that the ingest rule
has to catch on its own (for example `BR47NN7544`, 266 km/h implied between
Shyambazar Five Point and Kalighat Crossing).

## Accuracy, measured, not claimed

| What | Result | Re-run with |
|---|---|---|
| Real photographs, end to end (localiser + recogniser), 8 held-out-vehicle splits | **91.5% mean, 90.0% min, 94.8% max** | `python -m anpr.compare_backends --real-frames --seeds 8` |
| Real plate crops, 532 crops / 118 vehicles | 91.0% exact plate | `python -m anpr.compare_backends --real-all` |
| Synthetic stress test, 1,000 captures over 10 conditions, 5-frame burst | 73.1% exact; 96.2% of stored reads correct | `python -m anpr.benchmark --samples 100 --scenes 60 --layouts 100 --burst 5` |
| Field test, 9 real night phone photos | all 9 failed before the multi-scale fix (every wrong read below the 0.70 floor, so none stored); after it, 4+ read real plates at ≥ 0.94 | `FIELD_TEST.md` |

Limits: the real corpus is small and about 90% Maharashtra plates; it is a
first real-world result, not a Kolkata validation.

## Static analysis

| Tool | Before (24 Oct) | After fixes | Notes |
|---|---|---|---|
| ruff (pyflakes + bugbear rules) | 84 findings | 64 | Remaining are style-level: 47 × `zip()` without `strict=`, unused loop variables, `raise ... from`. Two unused `config` imports were kept on purpose in benchmark scripts. |
| bandit (security) | 2 high, 32 medium | **0 high**, 29 medium | See below. |
| radon (complexity / maintainability) | — | average cyclomatic complexity **B (5.2)** over 953 blocks; maintainability index A for 77 of 82 modules, C for `anpr/ocr.py`, `core/vehicle_search.py`, `dashboard/app.py` | The three C modules are the next refactor targets. |

Re-run with:

```bash
uvx ruff check anpr core api dashboard livecam phonecam registry sim viz *.py --select E9,F,B --statistics
uvx bandit -q -r anpr core api dashboard livecam phonecam registry sim viz config.py seed.py tests.py routine_report.py
uvx radon cc -a anpr core api dashboard livecam phonecam registry sim viz
```

### Fixed in this pass

- **A real crash:** `python -m core.repair --prune` called `prune_evidence`
  without importing it (pyflakes F821). Now `db.prune_evidence`.
- **Model loading:** every `torch.load` now passes `weights_only=True`, so a
  tampered checkpoint cannot run code on load (bandit B614).
- **Cache keys:** the two SHA-1 hashes used only as dashboard cache keys are
  marked `usedforsecurity=False` (bandit B324, the two high findings).
- 19 automatic fixes: unused imports, f-strings without placeholders, a
  duplicate stop-word.

### Remaining medium findings, reviewed

- **23 × B608 (SQL built from strings):** spot-checked in `core/convoy.py`,
  `core/privacy.py`, `core/vehicle_search.py` and `registry/live.py`. The
  interpolated parts are code-generated placeholder lists (`?, ?, ?`), table
  names and column names; every value is bound as a `?` parameter. Not an
  injection path, but a query builder would make that obvious to a scanner.
- **3 × B104 (bind to all interfaces):** the phone-camera and demo servers
  listen on the LAN on purpose, so phones can connect. Should be opt-in in a
  deployment.
- **2 × B615:** the `anpr_ocr` weights are fetched from Hugging Face without a
  pinned revision. Pinning a commit hash is the next fix.
- **1 × B314:** XML parsing of the local labelled dataset, not network input.

## Not yet done

- No unit-test coverage figure; the suite is end-to-end. `pytest-cov` is next.
- SonarQube itself was not run; ruff, bandit and radon cover the same ground
  (bugs, security hotspots, complexity) without a server.
