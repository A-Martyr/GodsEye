"""Pre-demo environment check — what the .bat launchers run before starting anything.

Two minutes before a demo is the wrong moment to discover that the interpreter
on this machine is missing fastapi, or that models/ did not survive the copy.
demo.bat calls this first and refuses to open two windows that would both fail;
check.bat runs it on its own so the answer can be had without starting servers.

The distinction that matters here is FAIL vs WARN. torch and ultralytics are
genuinely optional — without them anpr/ocr.py falls back to the classical
segment-and-classify reader and the platform still demos, just less accurately —
so their absence is a warning. fastapi, uvicorn and the database are not
optional: without them there is nothing to show, so those are failures.

Exit code is 0 when the demo can run (warnings and all) and 1 when it cannot.
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
_worst = 0


def line(status: str, label: str, detail: str = "") -> None:
    global _worst
    if status is FAIL:
        _worst = 2
    elif status is WARN and _worst < 1:
        _worst = 1
    print(f"  [{status}] {label:<26} {detail}")


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    print()
    print("  GodsEye - pre-demo check")
    print("  " + "=" * 58)
    print()

    # --- interpreter ----------------------------------------------------
    v = sys.version_info
    pyver = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 11):
        line(FAIL, "python", f"{pyver} - the platform needs 3.11 or newer")
        print()
        print("  Cannot continue on this interpreter.")
        return 1
    line(OK, "python", f"{pyver}  ({sys.executable})")

    # --- packages -------------------------------------------------------
    # (import name, what breaks without it, fatal?)
    required = [
        ("numpy", "everything", True),
        ("cv2", "image handling", True),
        ("streamlit", "the registry console only (optional)", False),
        ("fastapi", "the API", True),
        ("uvicorn", "the API", True),
        ("sklearn", "the glyph classifier", True),
        ("torch", "the CRNN reader (falls back to classical)", False),
        ("ultralytics", "the YOLOv8-OBB localiser (falls back to classical)", False),
    ]
    missing_fatal = []
    for name, why, fatal in required:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if found:
            line(OK, name)
        elif fatal:
            missing_fatal.append(name)
            line(FAIL, name, f"missing - {why} will not run")
        else:
            line(WARN, name, f"missing - {why}")

    # --- models ---------------------------------------------------------
    print()
    for fname, why, fatal in [
        ("crnn.pt", "the neural plate reader", False),
        ("glyph_mlp.joblib", "the classical reader (retrains on first use)", False),
        ("yolov8n-obb-plate.pt", "plate localisation (falls back to classical)", False),
    ]:
        path = os.path.join(ROOT, "models", fname)
        if os.path.exists(path):
            line(OK, f"models/{fname}", f"{os.path.getsize(path) / 1e6:.1f} MB")
        else:
            line(WARN, f"models/{fname}", f"missing - {why}")

    # --- device ---------------------------------------------------------
    print()
    try:
        import config
        device = config.torch_device()
        if device == "cpu":
            # Not a problem, just slower - say so plainly rather than warn.
            line(OK, "compute device", "cpu")
        else:
            import torch
            line(OK, "compute device", f"{device} - {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        line(WARN, "compute device", f"could not resolve ({exc}) - will use the cpu")

    # --- database -------------------------------------------------------
    try:
        import config
        db = config.DB_PATH
        if not os.path.exists(db):
            line(FAIL, "database", "missing - run seed.bat to build it")
        else:
            import sqlite3
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                n = con.execute("select count(*) from sightings").fetchone()[0]
            finally:
                con.close()
            if n == 0:
                line(FAIL, "database", "no sightings - run seed.bat to build it")
            else:
                line(OK, "database", f"{n:,} sightings")
    except Exception as exc:
        line(FAIL, "database", f"unreadable ({exc}) - run seed.bat")

    # --- verdict --------------------------------------------------------
    print()
    if _worst == 2:
        print("  Something the demo needs is missing - see the FAIL lines above.")
        if missing_fatal:
            print()
            print("  Install the packages with:")
            print("      " + os.path.basename(sys.executable)
                  + " -m pip install -r requirements.txt")
        print()
        return 1
    if _worst == 1:
        print("  Ready, with reduced accuracy - see the warn lines above.")
    else:
        print("  Ready.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
