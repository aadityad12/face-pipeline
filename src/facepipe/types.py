"""Data types shared across the seam.

Detection lives here rather than in scrfd.py because both sides need it: the ONNX
detector produces them, and the pipeline above the seam consumes them. Importing it from
scrfd.py meant that merely referring to the interface dragged in onnxruntime, which
defeated the point of having a seam at all (issue #8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One face, in original frame coordinates."""

    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    score: float
    kps: np.ndarray  # (5, 2) right eye, left eye, nose, right mouth, left mouth
