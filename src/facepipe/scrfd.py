"""SCRFD: find the faces.

The model does not return boxes. For each point on a grid laid over the image it returns a
score, four distances to the face's edges, and five landmark offsets. This module turns
that into one box plus five landmarks per face, in original frame coordinates.

Three things happen here and all three are easy to get silently wrong:

1. preprocessing  - make the frame look like the model's training data (issue #2)
2. decoding       - points + distances -> boxes, then NMS to kill duplicates (issue #3)
3. mapping back   - undo the letterbox so boxes land on the real frame (issue #2)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from facepipe.config import DET_INPUT_SIZE, DET_MODEL
from facepipe.types import Detection

# Confirmed from the model's own output shapes in issue #1, not guessed:
# stride 8 -> 80x80 grid = 6400 points, but the score tensor has 12800 rows, so 2 anchors
# per point. 40x40x2 = 3200 and 20x20x2 = 800 confirm it for the other two strides.
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2

# SCRFD was trained on pixels scaled to roughly [-1, 1] by (x - 127.5) / 128.
PIXEL_MEAN = 127.5
PIXEL_STD = 128.0

SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.4


# --------------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------------


def letterbox(
    frame: np.ndarray, size: tuple[int, int] = DET_INPUT_SIZE
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize to fit inside `size` without stretching, and pad the rest with black.

    Stretching a 640x480 frame to 640x640 would squash every face, and SCRFD was not
    trained on squashed faces. So scale by the same factor in both directions and fill the
    remainder.

    The padding goes on the right and bottom, which puts the image at the top left and
    leaves the offset at (0, 0). Centring it would work equally well but adds an offset to
    subtract later for no benefit.

    Returns the padded image, the scale used, and the (x, y) padding offset. The caller
    needs the last two: the model's boxes come back in padded coordinates and are useless
    until they are mapped back.
    """
    out_w, out_h = size
    h, w = frame.shape[:2]

    scale = min(out_w / w, out_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    canvas[:new_h, :new_w] = resized

    return canvas, scale, (0.0, 0.0)


def to_frame_coords(
    points: np.ndarray, scale: float, pad: tuple[float, float]
) -> np.ndarray:
    """Padded-image coordinates -> original frame coordinates."""
    return (points - np.asarray(pad, dtype=np.float32)) / scale


def to_model_coords(
    points: np.ndarray, scale: float, pad: tuple[float, float]
) -> np.ndarray:
    """Original frame coordinates -> padded-image coordinates. The inverse of the above."""
    return points * scale + np.asarray(pad, dtype=np.float32)


def preprocess(canvas: np.ndarray) -> np.ndarray:
    """Letterboxed BGR image -> the (1, 3, 640, 640) float32 blob the model wants.

    Three conversions, none of which raises if skipped; they just make detection worse:
    OpenCV stores blue first and the model wants red first, pixel values need to be scaled
    to the training range, and the channel axis moves to the front.
    """
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - PIXEL_MEAN) / PIXEL_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return chw[np.newaxis, ...]


# --------------------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------------------


def anchor_centers(height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
    """The (x, y) pixel position of every anchor on one grid.

    Grid point (col, row) sits at (col * stride, row * stride) in the padded image. Each
    point carries `num_anchors` anchors, stored consecutively, so each centre repeats.
    """
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    centers = centers.reshape(-1, 2)
    if num_anchors > 1:
        centers = np.repeat(centers, num_anchors, axis=0)
    return centers


def distance2bbox(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Anchor centre + 4 distances -> box corners.

    The model predicts how far each edge is from the anchor, so the box is the centre
    pushed outward: left and top subtract, right and bottom add.
    """
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Anchor centre + 10 offsets -> 5 landmark points. Offsets are (x, y) pairs."""
    pairs = distances.reshape(-1, 5, 2)
    return pairs + centers[:, np.newaxis, :]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns the indices worth keeping.

    Neighbouring anchors all see the same face, so one face arrives as a cluster of
    overlapping boxes. Keep the highest scoring box, drop the ones overlapping it by more
    than the threshold, repeat on what is left.

    The threshold is the interesting part. Too low and two people standing close together
    get merged into one box; too high and duplicates survive.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))

        # Intersection of the best box with every remaining box.
        xx1 = np.maximum(x1[best], x1[order[1:]])
        yy1 = np.maximum(y1[best], y1[order[1:]])
        xx2 = np.minimum(x2[best], x2[order[1:]])
        yy2 = np.minimum(y2[best], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)

        iou = inter / (areas[best] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]

    return keep


def group_outputs(outputs: list[np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    """Sort the model's 9 tensors into {stride: {score, bbox, kps}}.

    Grouped by shape rather than by position, so this does not silently break if the
    export order ever differs: last dimension 1 is a score, 4 is box distances, 10 is
    landmarks, and more rows means a finer grid, so a smaller stride.
    """
    by_kind: dict[str, list[np.ndarray]] = {"score": [], "bbox": [], "kps": []}
    for tensor in outputs:
        cols = tensor.shape[-1]
        kind = {1: "score", 4: "bbox", 10: "kps"}.get(cols)
        if kind is None:
            raise ValueError(f"unexpected output with {cols} columns")
        by_kind[kind].append(tensor)

    grouped: dict[int, dict[str, np.ndarray]] = {}
    for kind, tensors in by_kind.items():
        if len(tensors) != len(STRIDES):
            raise ValueError(f"expected {len(STRIDES)} {kind} tensors, got {len(tensors)}")
        # Most rows = finest grid = smallest stride.
        for stride, tensor in zip(STRIDES, sorted(tensors, key=len, reverse=True)):
            grouped.setdefault(stride, {})[kind] = tensor
    return grouped


class SCRFDDetector:
    """Runs SCRFD and returns faces in original frame coordinates."""

    def __init__(
        self,
        model_path: Path = DET_MODEL,
        input_size: tuple[int, int] = DET_INPUT_SIZE,
        score_threshold: float = SCORE_THRESHOLD,
        iou_threshold: float = NMS_IOU_THRESHOLD,
    ):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        canvas, scale, pad = letterbox(frame, self.input_size)
        blob = preprocess(canvas)
        outputs = self.session.run(None, {self.input_name: blob})

        boxes, scores, landmarks = self._decode(outputs)
        if len(boxes) == 0:
            return []

        keep = nms(boxes, scores, self.iou_threshold)

        height, width = frame.shape[:2]
        detections = []
        for i in keep:
            box = to_frame_coords(boxes[i].reshape(2, 2), scale, pad).reshape(4)
            box[0::2] = box[0::2].clip(0, width - 1)
            box[1::2] = box[1::2].clip(0, height - 1)
            detections.append(
                Detection(
                    bbox=box,
                    score=float(scores[i]),
                    kps=to_frame_coords(landmarks[i], scale, pad),
                )
            )
        return detections

    def _decode(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The 9 raw tensors -> boxes, scores and landmarks in padded coordinates."""
        grouped = group_outputs(outputs)
        in_w, in_h = self.input_size

        all_boxes, all_scores, all_kps = [], [], []
        for stride in STRIDES:
            tensors = grouped[stride]
            scores = tensors["score"].reshape(-1)

            # Distances come out in stride units, so scale them back to pixels.
            bbox_distances = tensors["bbox"].reshape(-1, 4) * stride
            kps_distances = tensors["kps"].reshape(-1, 10) * stride

            centers = anchor_centers(in_h // stride, in_w // stride, stride, NUM_ANCHORS)

            above = scores >= self.score_threshold
            if not above.any():
                continue

            all_boxes.append(distance2bbox(centers[above], bbox_distances[above]))
            all_kps.append(distance2kps(centers[above], kps_distances[above]))
            all_scores.append(scores[above])

        if not all_boxes:
            empty = np.zeros((0, 4), np.float32)
            return empty, np.zeros((0,), np.float32), np.zeros((0, 5, 2), np.float32)

        return (
            np.concatenate(all_boxes),
            np.concatenate(all_scores),
            np.concatenate(all_kps),
        )


def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Boxes, landmarks and scores drawn on a copy of the frame."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{det.score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        for x, y in det.kps.astype(int):
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    return out
