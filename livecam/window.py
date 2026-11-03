"""Native-window front-end. The shortest path from sensor to screen.

    python -m livecam.window
    python -m livecam.window --aim-only --enrol CAM-20

Keys:  q/Esc quit · f fullscreen · space pause · r reset the vote
       e enrol the locked read · s save a still
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from livecam import overlay, settings                            # noqa: E402
from livecam.engine import LiveEngine                            # noqa: E402

WINDOW = "GodsEye — live plate reader"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GodsEye live webcam plate reader")
    p.add_argument("--camera", type=int, default=None, help="webcam index (default 0)")
    p.add_argument("--backend", default=None,
                   help="anpr_ocr (default, real-photo reader) | crnn | classical")
    p.add_argument("--aim-only", action="store_true",
                   help="skip the detector and read only the aim box — fastest, "
                        "and the fallback if YOLO struggles with a hand-held board")
    p.add_argument("--no-aim", action="store_true", help="detector only, no aim box")
    p.add_argument("--enrol", metavar="CAM-ID", default=None,
                   help="write confirmed reads into the platform as sightings at "
                        "this junction (off unless given)")
    p.add_argument("--mirror", action="store_true",
                   help="mirror the preview (off by default — it would show the "
                        "board's text backwards)")
    p.add_argument("--no-warm", action="store_true", help="skip the warm-up")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--imgsz", type=int, default=None, help="detector input size")
    return p


def apply(args) -> None:
    """CLI overrides land on the settings module before anything reads them."""
    if args.aim_only:
        settings.AIM_ONLY = True
    if args.no_aim:
        settings.AIM_BOX = False
    if args.mirror:
        settings.MIRROR_PREVIEW = True
    if args.imgsz:
        settings.DETECT_IMGSZ = args.imgsz
    if args.enrol:
        settings.ENROL_ENABLED = True
        settings.ENROL_CAMERA = args.enrol.upper()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    apply(args)

    eng = LiveEngine(backend=args.backend, camera_index=args.camera,
                     enrol_camera=args.enrol)
    if not eng.start(warm=not args.no_warm):
        print(f"[livecam] {eng.error}", file=sys.stderr)
        return 1

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    full = False
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        full = True

    # ASCII only - a Windows console in a non-UTF-8 code page mangles the rest.
    print(f"[livecam] running on {eng.reader.actual}. "
          f"q quit | f fullscreen | space pause | r reset | e enrol", file=sys.stderr)

    target = 1.0 / 60.0
    try:
        while True:
            tick = time.perf_counter()
            frame, state = eng.snapshot()
            cv2.imshow(WINDOW, overlay.draw(frame, state))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                full = not full
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            elif key == ord("r"):
                eng.reset_vote()
            elif key == ord(" "):
                eng.toggle_pause()
            elif key == ord("e"):
                print(f"[livecam] {eng.enrol()}", file=sys.stderr)
            elif key == ord("s"):
                name = f"livecam_{int(time.time())}.png"
                cv2.imwrite(name, overlay.draw(frame, state))
                print(f"[livecam] saved {name}", file=sys.stderr)

            # The window is closed by the X button as well as by q.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Redrawing faster than the eye resolves only burns the CPU the
            # recogniser is sharing.
            slack = target - (time.perf_counter() - tick)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
