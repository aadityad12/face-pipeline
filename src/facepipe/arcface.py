"""ArcFace: turn a face into 512 numbers.

This is the step that makes recognition possible without training anything. ArcFace has
never seen me, and it is not going to learn me; it just maps any aligned face to a point in
512-dimensional space, placing two photos of the same person close together. Enrolling
someone is therefore just storing their 512 numbers once (issue #6).

Why cosine similarity and not plain distance: ArcFace is trained with an angular margin,
meaning it is pushed to separate identities by *angle*, not by magnitude. The angle is what
carries the identity, so comparing directions is the model's own measure rather than
something bolted on afterwards.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facepipe.config import REC_INPUT_SIZE, REC_MODEL
from facepipe.models import load_session

# Recognition uses a different normalization from detection: /127.5 rather than /128.
PIXEL_MEAN = 127.5
PIXEL_STD = 127.5


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length so only direction matters.

    Without this, the same face in bright and dim light can produce vectors of different
    magnitude and count as less similar for a reason that has nothing to do with identity.
    Once every embedding has length 1, cosine similarity is just a dot product.
    """
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 means identical direction, 0.0 unrelated, negative means opposite."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class ArcFaceEmbedder:
    """Runs ArcFace on an aligned 112x112 face."""

    def __init__(
        self, model_path: Path = REC_MODEL, input_size: tuple[int, int] = REC_INPUT_SIZE
    ):
        self.session = load_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Aligned BGR crop -> the (1, 3, 112, 112) float32 blob the model wants."""
        if aligned_bgr.shape[:2][::-1] != self.input_size:
            aligned_bgr = cv2.resize(aligned_bgr, self.input_size)
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - PIXEL_MEAN) / PIXEL_STD
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Aligned face -> a 512-number embedding of unit length."""
        blob = self.preprocess(aligned_bgr)
        output = self.session.run(None, {self.input_name: blob})[0]
        return l2_normalize(output.reshape(-1))
