"""Follow faces between frames so the same face is not re-embedded every frame.

Embedding is the expensive per-face step: 64ms each, against 115ms for detection, which
runs once per frame regardless of how many people are in it (#10). So a room with three
people spends most of its time re-answering a question it already answered, about faces
that have barely moved.

The fix is to notice that the face in this frame is the same face as the one in the last
frame. Boxes that overlap heavily between consecutive frames almost certainly belong to
the same person, because a face cannot cross the room in 30ms. Each track keeps the name
it was given, and ArcFace only runs for a face that is new or whose answer has gone stale.

It also fixes a cosmetic problem that looks like a broken system: recognising every frame
independently makes the label flicker between the right name and Unknown as the face
moves, since some frames land just under the threshold (issue #9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Boxes overlapping by more than this in consecutive frames are treated as the same face.
# Lower than the 0.4 used for NMS: this compares a face against itself one frame later,
# where the overlap is large, and being strict here costs a needless re-embed.
IOU_THRESHOLD = 0.3

# Re-embed a track this often even when it is matching happily. Without it a wrong name,
# or a name assigned while someone was turned away, would stick for as long as that person
# stays in frame.
REFRESH_EVERY = 30  # frames, so about 5 seconds at this pipeline's speed

# Keep a track alive briefly after it stops being seen, so one dropped detection does not
# discard a name and force a re-embed on the very next frame.
MAX_MISSES = 5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(intersection / (area_a + area_b - intersection))


@dataclass
class Track:
    """One face followed across frames."""

    track_id: int
    bbox: np.ndarray
    name: str | None = None
    score: float = 0.0
    last_embedded: int = -10**9
    misses: int = 0

    def is_stale(self, frame_index: int, refresh_every: int) -> bool:
        return frame_index - self.last_embedded >= refresh_every


@dataclass
class Tracker:
    """Greedy IoU tracking. No motion model, no Kalman filter.

    Those would matter for fast movement or heavy occlusion. For faces at a door, at a
    frame every 180ms, overlap is enough, and a simple thing that is obviously correct
    beats a clever one whose failures are hard to see.
    """

    iou_threshold: float = IOU_THRESHOLD
    refresh_every: int = REFRESH_EVERY
    max_misses: int = MAX_MISSES
    tracks: list[Track] = field(default_factory=list)
    frame_index: int = 0
    _next_id: int = 0

    def update(self, boxes: list[np.ndarray]) -> list[Track]:
        """Associate this frame's boxes with existing tracks, one track per box."""
        self.frame_index += 1
        unmatched = list(self.tracks)
        result: list[Track] = []

        for box in boxes:
            best, best_iou = None, self.iou_threshold
            for track in unmatched:
                overlap = iou(box, track.bbox)
                if overlap >= best_iou:
                    best, best_iou = track, overlap

            if best is None:
                best = Track(track_id=self._next_id, bbox=box)
                self._next_id += 1
                self.tracks.append(best)
            else:
                unmatched.remove(best)
                best.bbox = box
                best.misses = 0
            result.append(best)

        # Tracks nobody matched this frame: keep them briefly, then forget them.
        for track in unmatched:
            track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return result

    def needs_embedding(self, track: Track) -> bool:
        """New tracks always, existing ones only when their answer has gone stale.

        Checking `last_embedded` rather than whether the track has a name: a track that
        was embedded and came back Unknown has no name either, and re-embedding a genuine
        stranger on every frame would give back exactly the cost this class exists to
        avoid.
        """
        never_embedded = track.last_embedded < 0
        return never_embedded or track.is_stale(self.frame_index, self.refresh_every)

    def record(self, track: Track, name: str | None, score: float) -> None:
        track.name = name
        track.score = score
        track.last_embedded = self.frame_index
