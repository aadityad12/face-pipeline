"""Where frames come from.

Two sources, one interface. The camera is what the demo runs on; the image file is what
makes debugging possible, because a detector cannot be debugged against input that changes
30 times a second. Same frame in, same numbers out, so a change can be judged.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# How long to wait for the camera to actually start producing an image.
WARMUP_TIMEOUT_S = 3.0


class CameraSource:
    """Frames from a webcam."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera {index}. On macOS the terminal app needs camera "
                "permission: System Settings > Privacy & Security > Camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._warm_up()

    def _warm_up(self) -> None:
        """Read and throw away frames until the sensor produces a real image.

        A macOS webcam hands back completely black frames for the first few hundred
        milliseconds after opening. Nothing errors: read() succeeds and returns an array
        of zeros. Grabbing one frame immediately after opening gets that, and then the
        detector correctly reports no faces, because there genuinely are none.

        A frame is judged real once its pixels vary at all. A black frame has a standard
        deviation of exactly 0; any real scene, however dim, has some variation.
        """
        deadline = time.monotonic() + WARMUP_TIMEOUT_S
        while time.monotonic() < deadline:
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.std() > 1.0:
                return
        raise RuntimeError(
            f"camera {self.cap.getBackendName()} only produced blank frames for "
            f"{WARMUP_TIMEOUT_S}s. Is the lens covered, or is another app using it?"
        )

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class ImageSource:
    """The same frame every time, for repeatable debugging."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        frame = cv2.imread(str(self.path))
        if frame is None:
            raise FileNotFoundError(f"could not read image: {self.path}")
        self.frame = frame

    def read(self) -> np.ndarray:
        # A copy, so a caller drawing boxes on it cannot corrupt the source frame.
        return self.frame.copy()

    def release(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def open_source(spec: str | int):
    """"0" or 0 means the webcam; anything else is treated as an image path."""
    if isinstance(spec, int) or str(spec).isdigit():
        return CameraSource(int(spec))
    return ImageSource(spec)
