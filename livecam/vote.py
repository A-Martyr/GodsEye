"""Fusing reads across a stream of frames.

The platform already establishes why this works: frames fail independently,
so agreement between them is evidence no amount of work on a single frame can
produce (42.4% single-frame against 73.1% over a five-frame burst). A burst
costs a camera a deliberate trigger; a video stream produces the same frames
anyway, so here the fusion is free.

Two jobs, and the second is the one an audience notices. It raises accuracy,
and it stops the number on screen flickering between near-misses - a reading
that settles and holds is legible, one that redraws every 60ms is not.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Ballot:
    ts: float
    text: str
    confidence: float
    read: object          # the PlateRead itself, kept for enrolment


@dataclass
class VoteState:
    """What the vote currently believes, and how strongly."""
    text: str = ""                 # leader, locked or not
    locked: bool = False
    confidence: float = 0.0        # best single frame that agreed
    votes: int = 0
    total: int = 0                 # accepted reads in the window, any string
    progress: float = 0.0          # 0..1 towards a lock, for the UI
    margin: float = 0.0
    best_read: object = None
    runners: list = field(default_factory=list)   # [(text, votes)] for display


class StreamVote:
    """A rolling, confidence-weighted vote over recent accepted reads."""

    def __init__(self, window_s: float, min_agree: int, min_margin: float,
                 hold_s: float):
        self.window_s = window_s
        self.min_agree = max(1, min_agree)
        self.min_margin = min_margin
        self.hold_s = hold_s
        self._ballots: deque[Ballot] = deque()
        self._locked_text = ""
        self._locked_at = 0.0
        self._locked_read = None
        self._locked_conf = 0.0
        # Set when a lock is first reached, cleared once a consumer has taken
        # it. Enrolment must fire once per lock, not once per frame.
        self._fresh_lock = False

    def reset(self) -> None:
        self._ballots.clear()
        self._locked_text = ""
        self._locked_at = 0.0
        self._locked_read = None
        self._locked_conf = 0.0
        self._fresh_lock = False

    def add(self, read, ts: float | None = None, accepted: bool | None = None) -> None:
        """Offer one frame's read. Only accepted reads become ballots.

        `accepted` defaults to the engine's own per-backend decision. The live
        demo passes its own, higher one (livecam.reader.live_accepted) so a
        read the room produced cannot vote.
        """
        ts = time.perf_counter() if ts is None else ts
        ok = getattr(read, "accepted", False) if accepted is None else accepted
        if read is not None and ok and read.text:
            self._ballots.append(Ballot(ts, read.text, float(read.confidence), read))
        # The lock is decided here, as frames arrive - not inside state().
        # Hiding a state change in a getter means it only happens if some
        # caller happens to ask, which made enrolment fire a frame late in one
        # front-end and not at all in another.
        self._evaluate(ts)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._ballots and self._ballots[0].ts < cutoff:
            self._ballots.popleft()

    def _evaluate(self, now: float) -> list:
        """Re-tally, take or release the lock, and return the ranking.

        Idempotent: `_fresh_lock` only flips when the winning string actually
        changes, so calling this twice for one frame cannot enrol twice.
        """
        self._prune(now)

        tally: dict[str, list] = {}
        for b in self._ballots:
            slot = tally.setdefault(b.text, [0.0, 0, None])
            slot[0] += b.confidence
            slot[1] += 1
            if slot[2] is None or b.confidence > slot[2].confidence:
                slot[2] = b.read

        ranked = sorted(tally.items(), key=lambda kv: -kv[1][0])
        if ranked:
            text, (score, votes, best) = ranked[0]
            runner = ranked[1][1][0] if len(ranked) > 1 else 0.0
            margin = score / runner if runner > 0 else float("inf")
            if votes >= self.min_agree and margin >= self.min_margin:
                if text != self._locked_text:
                    self._fresh_lock = True
                self._locked_text = text
                self._locked_at = now
                self._locked_read = best
                self._locked_conf = float(best.confidence)

        # A lock outlives the ballots that made it, so the answer stays on
        # screen after the board is lowered.
        if self._locked_text and now - self._locked_at > self.hold_s:
            self._locked_text = ""
            self._locked_read = None
            self._locked_conf = 0.0
        return ranked

    def state(self, now: float | None = None) -> VoteState:
        now = time.perf_counter() if now is None else now
        ranked = self._evaluate(now)
        runners = [(t, v[1]) for t, v in ranked[:3]]
        leader, votes, best = "", 0, None
        margin = 0.0
        if ranked:
            leader, (score, votes, best) = ranked[0]
            runner = ranked[1][1][0] if len(ranked) > 1 else 0.0
            margin = (score / runner) if runner > 0 else 99.0

        if self._locked_text:
            return VoteState(
                text=self._locked_text, locked=True, confidence=self._locked_conf,
                votes=max(votes if leader == self._locked_text else 0, self.min_agree),
                total=len(self._ballots), progress=1.0, margin=min(margin, 99.0),
                best_read=self._locked_read, runners=runners)

        return VoteState(
            text=leader, locked=False,
            confidence=float(best.confidence) if best else 0.0,
            votes=votes, total=len(self._ballots),
            progress=min(1.0, votes / self.min_agree) if votes else 0.0,
            margin=min(margin, 99.0), best_read=best, runners=runners)

    def take_fresh_lock(self):
        """The PlateRead for a lock not yet consumed, once. None otherwise.

        Guards the enrolment path: without it a locked read three seconds long
        would enrol on every one of the ~50 frames it survives.
        """
        if self._fresh_lock and self._locked_read is not None:
            self._fresh_lock = False
            return self._locked_read
        return None
