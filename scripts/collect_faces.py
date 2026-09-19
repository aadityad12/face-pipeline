"""Capture a set of face photos for one person, for the threshold measurement in #11.

Picking a match threshold needs pairs: photos of the same person (which should score high)
and photos of different people (which should score low). A handful of shots each, taken on
the same webcam in the same room, is a far more honest test set for this pipeline than
stock photos would be.

Each shot is triggered by hand and can be retaken, because a timed countdown gives you no
time to actually get into the pose and no way to see what the camera got.

Vary the shots deliberately: look away, tilt your head, move closer and further, take your
glasses off. A test set of near-identical photos would report an accuracy this pipeline
does not really have.

Usage:
    uv run python scripts/collect_faces.py --name "Aaditya" --shots 8
    uv run python scripts/collect_faces.py --name "Aaditya" --shots 8 --open
"""

from __future__ import annotations

import argparse
import re
import subprocess

import cv2

from facepipe.capture import CameraSource
from facepipe.config import DATA_DIR
from facepipe.scrfd import SCRFDDetector, draw

PROMPTS = [
    "look straight at the camera",
    "turn your head slightly left",
    "turn your head slightly right",
    "tilt your head",
    "move closer",
    "move further away",
    "look slightly up",
    "neutral again, different expression",
]

# OpenCV keeps a few frames buffered, so the first read after a pause can be seconds old.
# Reading a few and keeping the last one gets the frame from when Enter was actually hit.
BUFFER_DEPTH = 5


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def grab_current_frame(camera: CameraSource):
    frame = None
    for _ in range(BUFFER_DEPTH):
        latest = camera.read()
        if latest is not None:
            frame = latest
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument(
        "--open", action="store_true", help="open each shot so you can see what was caught"
    )
    args = parser.parse_args()

    out_dir = DATA_DIR / "faces" / slug(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = DATA_DIR / "scratch"
    preview_dir.mkdir(parents=True, exist_ok=True)

    detector = SCRFDDetector()
    print(f"\ncapturing {args.shots} shots for {args.name} into {out_dir}")
    print("only one person in frame, so anyone else needs to step out of shot\n")

    kept = 0
    with CameraSource() as camera:
        while kept < args.shots:
            prompt = PROMPTS[kept % len(PROMPTS)]
            input(f"[{kept + 1}/{args.shots}] {prompt}  -  press Enter to capture ")

            frame = grab_current_frame(camera)
            if frame is None:
                print("    no frame from the camera, try again\n")
                continue

            faces = detector.detect(frame)
            if not faces:
                print("    no face detected, try again\n")
                continue
            if len(faces) > 1:
                print(f"    {len(faces)} faces in frame - everyone else step out, try again\n")
                continue

            preview = preview_dir / "last_capture.jpg"
            cv2.imwrite(str(preview), draw(frame, faces))
            print(f"    got a face, score {faces[0].score:.2f}  (preview: {preview})")
            if args.open:
                subprocess.run(["open", str(preview)], check=False)

            answer = input("    Enter to keep, or r to retake: ").strip().lower()
            if answer == "r":
                print("    retaking\n")
                continue

            path = out_dir / f"{kept:02d}.jpg"
            cv2.imwrite(str(path), frame)
            kept += 1
            print(f"    saved {path.name}\n")

    print(f"kept {kept} photos for {args.name} in {out_dir}")


if __name__ == "__main__":
    main()
