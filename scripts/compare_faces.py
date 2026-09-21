"""Compare two face images and print how similar the embeddings are.

This is the sanity check for issue #6. Two photos of the same person should score high and
two different people should score low. If that gap is not there, something upstream is
wrong (most likely alignment) and there is no point building matching on top of it.

--no-align crops the detector's box and resizes it instead of aligning, which answers the
open question in issue #5: how much does alignment actually buy?

Usage:
    uv run python scripts/compare_faces.py a.jpg b.jpg
    uv run python scripts/compare_faces.py a.jpg b.jpg --no-align
    uv run python scripts/compare_faces.py a.jpg b.jpg --dump
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from facepipe.align import align_face, crop_resize
from facepipe.arcface import ArcFaceEmbedder, cosine_similarity
from facepipe.config import DATA_DIR
from facepipe.scrfd import SCRFDDetector


def face_crop(detector: SCRFDDetector, path: str, use_alignment: bool) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"could not read {path}")

    detections = detector.detect(frame)
    if not detections:
        raise SystemExit(f"no face found in {path}")

    # Biggest face, on the assumption that the subject is the one closest to the camera.
    best = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return align_face(frame, best.kps) if use_alignment else crop_resize(frame, best.bbox)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="crop the box and resize instead of aligning, to measure what alignment buys",
    )
    parser.add_argument(
        "--dump", action="store_true", help="save both crops so they can be looked at"
    )
    args = parser.parse_args()

    detector = SCRFDDetector()
    embedder = ArcFaceEmbedder()

    crops = [
        face_crop(detector, path, not args.no_align) for path in (args.first, args.second)
    ]
    embeddings = [embedder.embed(crop) for crop in crops]

    def label(path: str) -> str:
        # Last two parts, because every person's photos are numbered the same way and
        # "00.jpg vs 00.jpg" looks like a file compared against itself. The dumped crops
        # would collide on disk for the same reason.
        parts = Path(path).parts
        return "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]

    if args.dump:
        out_dir = DATA_DIR / "scratch"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "noalign" if args.no_align else "aligned"
        for path, crop in zip((args.first, args.second), crops):
            out = out_dir / f"{label(path).replace('/', '_').removesuffix('.jpg')}_{suffix}.png"
            cv2.imwrite(str(out), crop)
            print(f"wrote {out}")

    mode = "box crop, no alignment" if args.no_align else "aligned"
    print(f"\n{mode}")
    print(f"  {label(args.first)} vs {label(args.second)}")
    print(f"  cosine similarity = {cosine_similarity(*embeddings):.4f}")


if __name__ == "__main__":
    main()
