"""Tests for the parts of detection that can be checked without a model or a camera.

NMS and the letterbox round trip are pure arithmetic, so they get hand-made inputs with
known answers. Everything else needs a real face and gets checked by looking at the output
image instead.
"""

import numpy as np

from facepipe.scrfd import (
    anchor_centers,
    distance2bbox,
    letterbox,
    nms,
    to_frame_coords,
    to_model_coords,
)


def test_nms_removes_a_near_duplicate():
    # Two boxes on top of each other: the same face seen by two neighbouring anchors.
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.4) == [0]  # keeps the higher scoring one


def test_nms_keeps_two_separate_faces():
    boxes = np.array([[0, 0, 100, 100], [500, 500, 600, 600]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.4)) == [0, 1]


def test_nms_keeps_two_people_standing_close():
    # Overlap here is 20x100 / (2*100x100 - 20x100) = ~0.11, below the threshold, so both
    # survive. This is the case that breaks if the threshold is set too low.
    boxes = np.array([[0, 0, 100, 100], [80, 0, 180, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.4)) == [0, 1]


def test_nms_on_nothing():
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.4) == []


def test_letterbox_does_not_stretch():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, scale, pad = letterbox(frame, (640, 640))

    assert canvas.shape == (640, 640, 3)
    assert scale == 1.0  # 640 wide already fits, so no resize
    assert pad == (0.0, 0.0)
    assert canvas[479:, :].sum() == 0  # the bottom 160 rows are padding


def test_letterbox_scales_a_bigger_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, scale, _ = letterbox(frame, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert scale == 0.5


def test_coordinate_round_trip():
    """Issue #2: a point mapped into the padded image and back must land where it started.

    This is the check that catches boxes being drawn in the wrong place.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, scale, pad = letterbox(frame, (640, 640))

    point = np.array([[600.0, 200.0]], dtype=np.float32)
    there_and_back = to_frame_coords(to_model_coords(point, scale, pad), scale, pad)
    np.testing.assert_allclose(there_and_back, point, atol=1e-4)


def test_anchor_centers_match_the_model_output_rows():
    """The row counts printed in issue #1 should fall out of the grid arithmetic."""
    assert len(anchor_centers(80, 80, 8, 2)) == 12800
    assert len(anchor_centers(40, 40, 16, 2)) == 3200
    assert len(anchor_centers(20, 20, 32, 2)) == 800


def test_anchor_centers_repeat_consecutively():
    centers = anchor_centers(2, 2, 8, 2)
    # Two anchors per point, stored next to each other.
    np.testing.assert_array_equal(centers[0], centers[1])
    np.testing.assert_array_equal(centers[0], [0, 0])
    np.testing.assert_array_equal(centers[2], [8, 0])


def test_distance2bbox_pushes_outward_from_the_centre():
    centers = np.array([[100.0, 100.0]], dtype=np.float32)
    distances = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        distance2bbox(centers, distances), [[90.0, 80.0, 130.0, 140.0]]
    )
