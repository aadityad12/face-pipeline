"""Line the face up before recognition.

ArcFace saw every training face in the same pose: a 112x112 crop with the eyes, nose and
mouth corners at fixed pixel positions, like passport photos. A face off a webcam is
tilted, off centre and a different size, so handing it the raw box means handing it
something it never trained on. As usual in this pipeline, nothing errors; the embedding
just gets worse (issue #5).

The fix uses the 5 landmarks the detector already gives us. We know where they are and
where they should be, so we solve for the transform between the two and apply it.
"""

from __future__ import annotations

import cv2
import numpy as np

# Where the 5 landmarks sit in ArcFace's 112x112 training crop. Published with the model;
# these are what "lined up" means for it.
# Order matches the detector's: right eye, left eye, nose, right mouth, left mouth.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best rotation + uniform scale + translation taking `src` points onto `dst`.

    Deliberately limited to those three. A full affine transform would fit the 5 points
    more exactly, but it can stretch and skew, and that changes the shape of the face,
    which is the thing recognition depends on. Rotating, resizing and shifting move the
    face without deforming it.

    With 5 points and only 4 degrees of freedom this is overdetermined, so there is no
    exact answer and we take the least-squares best fit (the Umeyama method): centre both
    sets, take the SVD of their covariance, and read the rotation off it.

    Returns a 2x3 matrix ready for cv2.warpAffine.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean

    covariance = dst_centered.T @ src_centered / n
    u, singular_values, vt = np.linalg.svd(covariance)

    # Guard against the SVD handing back a reflection, which would mirror the face.
    correction = np.ones(2)
    if np.linalg.det(covariance) < 0:
        correction[-1] = -1

    rotation = u @ np.diag(correction) @ vt
    scale = (singular_values * correction).sum() / src_centered.var(axis=0).sum()
    translation = dst_mean - scale * rotation @ src_mean

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def align_face(
    frame: np.ndarray, kps: np.ndarray, size: tuple[int, int] = (112, 112)
) -> np.ndarray:
    """Landmarks -> a 112x112 crop with the face in ArcFace's expected pose."""
    matrix = similarity_transform(kps, ARCFACE_TEMPLATE)
    return cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR)


def crop_resize(
    frame: np.ndarray, bbox: np.ndarray, size: tuple[int, int] = (112, 112)
) -> np.ndarray:
    """Just crop the box and resize, ignoring the landmarks.

    Not used by the pipeline. It exists so alignment can be measured against the naive
    alternative rather than assumed to help (issue #5's open question).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return cv2.resize(frame[y1:y2, x1:x2], size, interpolation=cv2.INTER_LINEAR)
