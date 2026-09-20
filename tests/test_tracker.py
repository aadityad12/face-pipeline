"""Tests for tracking.

The tracker's job is to skip work safely. Both halves matter: it has to actually skip
(otherwise it buys nothing) and it has to skip only when the answer is still right
(otherwise it puts the wrong name on a face, which is worse than being slow).
"""

import numpy as np

from facepipe.gallery import Gallery
from facepipe.tracker import Tracker, iou


def box(x1, y1, x2, y2) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def test_iou_of_identical_boxes():
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0


def test_iou_of_separate_boxes():
    assert iou(box(0, 0, 10, 10), box(50, 50, 60, 60)) == 0.0


def test_a_face_that_barely_moves_keeps_its_track():
    tracker = Tracker()
    first = tracker.update([box(100, 100, 200, 200)])[0]
    second = tracker.update([box(105, 103, 205, 203)])[0]
    assert first.track_id == second.track_id


def test_a_face_somewhere_else_is_a_new_track():
    tracker = Tracker()
    first = tracker.update([box(100, 100, 200, 200)])[0]
    second = tracker.update([box(400, 400, 500, 500)])[0]
    assert first.track_id != second.track_id


def test_two_people_keep_separate_tracks():
    tracker = Tracker()
    left, right = tracker.update([box(0, 0, 100, 100), box(300, 0, 400, 100)])
    left2, right2 = tracker.update([box(5, 2, 105, 102), box(305, 1, 405, 101)])

    assert left.track_id == left2.track_id
    assert right.track_id == right2.track_id
    assert left.track_id != right.track_id


def test_new_track_needs_embedding_and_then_does_not():
    tracker = Tracker()
    track = tracker.update([box(0, 0, 100, 100)])[0]
    assert tracker.needs_embedding(track)

    tracker.record(track, "Aaditya", 0.83)
    same = tracker.update([box(2, 2, 102, 102)])[0]
    assert not tracker.needs_embedding(same)
    assert same.name == "Aaditya"


def test_a_stranger_is_not_re_embedded_every_frame():
    """An Unknown track has no name, but it has been embedded and should stay skipped.

    Checking `name is None` instead of when it was last embedded would re-embed every
    stranger on every frame, which is exactly the cost the tracker exists to remove.
    """
    tracker = Tracker()
    track = tracker.update([box(0, 0, 100, 100)])[0]
    tracker.record(track, None, 0.12)

    same = tracker.update([box(1, 1, 101, 101)])[0]
    assert not tracker.needs_embedding(same)


def test_a_track_goes_stale_and_is_re_checked():
    """A name must not stick forever, or a wrong one never gets corrected."""
    tracker = Tracker(refresh_every=5)
    track = tracker.update([box(0, 0, 100, 100)])[0]
    tracker.record(track, "Aaditya", 0.83)

    for _ in range(4):
        track = tracker.update([box(0, 0, 100, 100)])[0]
        assert not tracker.needs_embedding(track)

    track = tracker.update([box(0, 0, 100, 100)])[0]
    assert tracker.needs_embedding(track)


def test_one_dropped_detection_does_not_lose_the_name():
    """Detection occasionally misses a frame; that should not throw away the identity."""
    tracker = Tracker()
    track = tracker.update([box(0, 0, 100, 100)])[0]
    tracker.record(track, "Aaditya", 0.83)

    tracker.update([])  # face missed this frame
    back = tracker.update([box(0, 0, 100, 100)])[0]

    assert back.track_id == track.track_id
    assert back.name == "Aaditya"


def test_a_face_that_leaves_is_eventually_forgotten():
    tracker = Tracker(max_misses=2)
    tracker.update([box(0, 0, 100, 100)])
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == []


def test_pipeline_reuses_the_name_instead_of_re_embedding(tmp_path):
    """The point of the whole thing: the embedder is called once, not once per frame."""
    from tests.test_backends import FakeDetector, FakeEmbedder

    class CountingEmbedder(FakeEmbedder):
        calls = 0

        def embed(self, aligned_bgr):
            CountingEmbedder.calls += 1
            return super().embed(aligned_bgr)

    from facepipe.pipeline import Pipeline

    gallery = Gallery(tmp_path)
    pipeline = Pipeline(
        detector=FakeDetector(), embedder=CountingEmbedder(), gallery=gallery
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gallery.add("Aaditya", pipeline.embed_largest_face(frame))

    CountingEmbedder.calls = 0
    for _ in range(10):
        faces = pipeline.process(frame)

    assert CountingEmbedder.calls == 1, "should embed once, then follow the track"
    assert faces[0].match.name == "Aaditya"


def test_tracking_can_be_turned_off(tmp_path):
    """Needed for still images, and for measuring what tracking is worth."""
    from tests.test_backends import FakeDetector, FakeEmbedder

    class CountingEmbedder(FakeEmbedder):
        calls = 0

        def embed(self, aligned_bgr):
            CountingEmbedder.calls += 1
            return super().embed(aligned_bgr)

    from facepipe.pipeline import Pipeline

    pipeline = Pipeline(
        detector=FakeDetector(),
        embedder=CountingEmbedder(),
        gallery=Gallery(tmp_path),
        track=False,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    CountingEmbedder.calls = 0
    for _ in range(5):
        pipeline.process(frame)

    assert CountingEmbedder.calls == 5
