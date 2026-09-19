"""Run detection on one image or one webcam frame and save the annotated result.

This is the check for issues #2 and #3: the boxes have to land on the faces and the five
landmark dots have to land on eyes, nose and mouth corners. Numbers in a terminal cannot
tell you that; looking at the image can.

Usage:
    uv run python scripts/detect_faces.py --camera
    uv run python scripts/detect_faces.py --image data/sample.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2

from facepipe.capture import open_source
from facepipe.config import DATA_DIR
from facepipe.scrfd import SCRFDDetector, draw


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", action="store_true", help="grab one webcam frame")
    group.add_argument("--image", help="path to an image file")
    parser.add_argument("--out", default=None, help="where to save the annotated image")
    parser.add_argument(
        "--save-frame",
        default=None,
        help="also save the unannotated frame, to reuse as a fixed test image",
    )
    args = parser.parse_args()

    source = open_source(0 if args.camera else args.image)
    frame = source.read()
    source.release()
    if frame is None:
        raise SystemExit("no frame came back from the camera")

    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)
        print(f"saved raw frame to {args.save_frame}")

    detector = SCRFDDetector()

    start = time.perf_counter()
    detections = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{len(detections)} face(s) in {elapsed_ms:.1f} ms  (frame {frame.shape})")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox.astype(int)
        print(f"  face {i}: score={det.score:.3f} box=({x1},{y1})-({x2},{y2})")

    out_path = args.out or str(DATA_DIR / "scratch" / "detected.jpg")
    (DATA_DIR / "scratch").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, draw(frame, detections))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
