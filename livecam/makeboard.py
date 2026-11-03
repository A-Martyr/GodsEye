"""Generate a printable, plate-format board to hold up at the camera.

    python -m livecam.makeboard WB24AB1234
    python -m livecam.makeboard --random 3 --pdf

Built on anpr/plates.py's own render_plate(), deliberately. The recogniser was
trained on real Indian plates and the platform's synthetic corpus is rendered
by that function, so a board printed from it is as close to the reader's
expectations as anything that can be produced on an office printer. A
hand-lettered board is a different distribution and reads far worse - that is
the single biggest thing that decides whether the live demo works.

Printed A4 landscape at 300dpi, the plate about 270mm wide, which is roughly
half a real plate and reads comfortably at arm's length from a laptop webcam.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anpr import plates as plate_lib                             # noqa: E402

DPI = 300
MM = DPI / 25.4
PAGE_W, PAGE_H = int(297 * MM), int(210 * MM)      # A4 landscape
PLATE_MM = 270
OUT_DIR = Path(__file__).resolve().parent / "boards"


def board(text: str, *, two_row: bool = False, commercial: bool = False,
          ind_strip: bool = False, font_path: str | None = None) -> Image.Image:
    """One A4 page carrying one plate, centred, with a cut guide."""
    plate = plate_lib.render_plate(text, width=1800, two_row=two_row,
                                   commercial=commercial, ind_strip=ind_strip,
                                   font_path=font_path, rng=random.Random(7))
    target_w = int(PLATE_MM * MM)
    scale = target_w / plate.shape[1]
    plate = cv2.resize(plate, (target_w, int(plate.shape[0] * scale)),
                       interpolation=cv2.INTER_CUBIC)

    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    img = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    x = (PAGE_W - img.width) // 2
    y = (PAGE_H - img.height) // 2
    page.paste(img, (x, y))

    draw = ImageDraw.Draw(page)
    pad = int(6 * MM)
    draw.rectangle([x - pad, y - pad, x + img.width + pad, y + img.height + pad],
                   outline=(190, 190, 190), width=2)
    caption = (f"{plate_lib.pretty(text)}    ·    GodsEye live demo    ·    "
               f"cut on the grey line and hold flat, filling the on-screen brackets")
    draw.text((x - pad, y + img.height + pad + int(4 * MM)), caption, fill=(120, 120, 120))
    return page


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Printable plate board for the live demo")
    p.add_argument("plates", nargs="*", help="registrations, e.g. WB24AB1234")
    p.add_argument("--random", type=int, default=0, metavar="N",
                   help="generate N valid random registrations instead")
    p.add_argument("--two-row", action="store_true", help="motorcycle/truck layout")
    p.add_argument("--commercial", action="store_true", help="yellow commercial plate")
    p.add_argument("--ind", action="store_true",
                   help="include the blue IND band (the reader removes it with the "
                        "frame; leave it off for the simplest board)")
    p.add_argument("--pdf", action="store_true", help="also write a combined PDF")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args(argv)

    texts = [t.upper().replace(" ", "") for t in args.plates]
    if args.random:
        rng = random.Random()
        texts += [plate_lib.random_plate(rng) for _ in range(args.random)]
    if not texts:
        texts = [plate_lib.random_plate(random.Random())]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    for text in texts:
        page = board(text, two_row=args.two_row, commercial=args.commercial,
                     ind_strip=args.ind)
        path = out / f"board_{text}.png"
        page.save(path, dpi=(DPI, DPI))
        pages.append(page)
        print(f"[makeboard] {path}")

    if args.pdf and pages:
        pdf = out / "boards.pdf"
        pages[0].save(pdf, save_all=True, append_images=pages[1:],
                      resolution=float(DPI))
        print(f"[makeboard] {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
