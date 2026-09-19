"""The whole thing, wired together.

frame -> detect -> align -> embed -> match -> annotated frame

Everything above lives in its own module; this is the only place that knows the order.
Per-stage timings are collected here too, because the stage that costs the most is the
thing worth optimising and guessing at it would be exactly the mistake this project is
supposed to avoid (issues #9, #10).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from facepipe.align import align_face
from facepipe.backends import Detection, FaceDetector, FaceEmbedder
from facepipe.gallery import MATCH_THRESHOLD, Gallery, Match


@dataclass
class RecognizedFace:
    detection: Detection
    match: Match


@dataclass
class Timings:
    """Rolling per-stage timings, in milliseconds."""

    window: int = 30
    stages: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))

    def record(self, stage: str, ms: float) -> None:
        samples = self.stages[stage]
        samples.append(ms)
        if len(samples) > self.window:
            samples.popleft()

    def averages(self) -> dict[str, float]:
        return {
            stage: sum(samples) / len(samples)
            for stage, samples in self.stages.items()
            if samples
        }

    @property
    def fps(self) -> float:
        total = sum(self.averages().values())
        return 1000.0 / total if total > 0 else 0.0


class Pipeline:
    """One frame in, recognised faces out."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
        gallery: Gallery | None = None,
        threshold: float = MATCH_THRESHOLD,
    ):
        # The ONNX backends are imported here rather than at the top of the module, so
        # that a Pipeline built with other backends never loads onnxruntime at all. That
        # is the difference between the seam being real and being a type annotation.
        if detector is None or embedder is None:
            from facepipe.arcface import ArcFaceEmbedder
            from facepipe.scrfd import SCRFDDetector

        # `x if x is not None else ...` rather than `x or ...`: Gallery defines __len__,
        # so an empty gallery is falsy, and `gallery or Gallery()` silently discarded the
        # one passed in and built a default pointing at a different directory. Nothing
        # failed; enrolments just went somewhere else.
        self.detector = detector if detector is not None else SCRFDDetector()
        self.embedder = embedder if embedder is not None else ArcFaceEmbedder()
        self.gallery = gallery if gallery is not None else Gallery()
        self.threshold = threshold
        self.timings = Timings()

    def process(self, frame: np.ndarray) -> list[RecognizedFace]:
        start = time.perf_counter()
        detections = self.detector.detect(frame)
        self.timings.record("detect", (time.perf_counter() - start) * 1000)

        results: list[RecognizedFace] = []
        align_ms = embed_ms = match_ms = 0.0

        for detection in detections:
            t0 = time.perf_counter()
            aligned = align_face(frame, detection.kps)
            t1 = time.perf_counter()
            embedding = self.embedder.embed(aligned)
            t2 = time.perf_counter()
            match = self.gallery.match(embedding, self.threshold)
            t3 = time.perf_counter()

            align_ms += (t1 - t0) * 1000
            embed_ms += (t2 - t1) * 1000
            match_ms += (t3 - t2) * 1000
            results.append(RecognizedFace(detection=detection, match=match))

        self.timings.record("align", align_ms)
        self.timings.record("embed", embed_ms)
        self.timings.record("match", match_ms)
        return results

    def embed_largest_face(self, frame: np.ndarray) -> np.ndarray | None:
        """Embedding of the biggest face in the frame, for enrolment.

        Biggest because the person enrolling is the one closest to the camera; anyone in
        the background is not who we are trying to add.
        """
        detections = self.detector.detect(frame)
        if not detections:
            return None
        largest = max(
            detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        )
        return self.embedder.embed(align_face(frame, largest.kps))


def annotate(frame: np.ndarray, faces: list[RecognizedFace]) -> np.ndarray:
    """Draw a box and a name on each recognised face.

    Known faces are green, Unknown red, so the Unknown case is visible at a glance rather
    than having to read the label.
    """
    out = frame.copy()
    for face in faces:
        x1, y1, x2, y2 = face.detection.bbox.astype(int)
        color = (0, 200, 0) if face.match.is_known else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            face.match.label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return out
