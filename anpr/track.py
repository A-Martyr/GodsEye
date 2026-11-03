"""Following one plate across consecutive frames.

A camera does not photograph a vehicle once. It watches it cross a zone, and
the same registration appears in five or fifteen frames at different
distances, exposures and motion blurs. Reading the middle one and discarding
the rest throws away the strongest signal the platform has: measured over 250
vehicles across all ten camera conditions, one frame reads 40.4% of plates and
eight read 77.2% with identical weights, because the frames fail in *different*
ways and agreement between them is evidence. On the real-photograph corpus the
same effect takes 92.3% per frame to 97.0% per vehicle.

None of that works without knowing which boxes in which frames are the same
vehicle, which is what this module does and nothing else. It holds no model
and imports nothing from the reader, so anpr/ocr.py can use it without a
circular import.

Deliberately not a Kalman/Hungarian tracker. Plates are small, roughly convex,
move smoothly and are rarely dense in frame - a vehicle is not a crowd - so
greedy association on overlap answers the question at a fraction of the code,
and every extra moving part here would be one more thing to be wrong during a
demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[int, int, int, int]          # x, y, w, h


@dataclass
class Track:
    """One vehicle's plate, and every frame it was seen in."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.boxes)

    @property
    def last_frame(self) -> int:
        return self.frames[-1] if self.frames else -1

    @property
    def last_box(self) -> Box | None:
        return self.boxes[-1] if self.boxes else None


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _near(a: Box, b: Box, reach: float) -> float:
    """A weak match for boxes that no longer overlap at all.

    Overlap alone loses a vehicle exactly where the extra frames are worth
    most. A plate crossing the frame at expressway speed can move further than
    its own width between captures, so consecutive detections of the same
    vehicle share no pixels and IoU is flat zero - and high_speed is one of the
    conditions where single-frame accuracy is 41% and fusing eight frames is
    80%. Distance between centres, measured in box widths, still recognises it.

    Scored below any real overlap so this can only ever pick up a track that
    IoU had already given up on.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    scale = max(aw, bw, 1)
    d = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale
    if d > reach:
        return 0.0
    # Similar size as well as position: a plate does not double in size between
    # two frames, and without this a nearby lorry's plate can steal the track.
    ratio = min(aw, bw) / max(aw, bw, 1)
    if ratio < 0.5:
        return 0.0
    return 0.05 * (1.0 - d / reach)          # always < any genuine IoU match


def track_boxes(per_frame: list[list[Box]], min_iou: float = 0.25,
                max_gap: int = 2, reach: float = 2.5) -> list[Track]:
    """Group detections across frames into one Track per vehicle.

    `per_frame[i]` is the boxes found in frame i, in the localiser's own
    confidence order. Returns tracks ordered by first appearance.

    max_gap allows a vehicle to be missed for a frame or two and still be
    recognised as the same one afterwards. That matters more than it sounds:
    a plate lost to a single frame of glare or a passing pole would otherwise
    become two short tracks, and two three-frame tracks fuse worse than one of
    six.
    """
    tracks: list[Track] = []
    active: list[Track] = []
    next_id = 0

    for frame_index, boxes in enumerate(per_frame):
        # Retire anything not seen recently, so it cannot claim a new box.
        active = [t for t in active if frame_index - t.last_frame <= max_gap]
        claimed: set[int] = set()

        # Score every (track, box) pair, then take them best-first. Greedy on a
        # sorted list rather than in detection order: otherwise a weak box that
        # happens to come first can take a track that a better one wanted.
        pairs = []
        for ti, t in enumerate(active):
            last = t.last_box
            if last is None:
                continue
            for bi, box in enumerate(boxes):
                score = iou(last, box)
                if score < min_iou:
                    score = _near(last, box, reach)
                if score > 0:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        for _score, ti, bi in pairs:
            if ti in used_tracks or bi in claimed:
                continue
            used_tracks.add(ti)
            claimed.add(bi)
            active[ti].frames.append(frame_index)
            active[ti].boxes.append(boxes[bi])

        for bi, box in enumerate(boxes):
            if bi in claimed:
                continue
            fresh = Track(track_id=next_id, frames=[frame_index], boxes=[box])
            next_id += 1
            tracks.append(fresh)
            active.append(fresh)

    return tracks
