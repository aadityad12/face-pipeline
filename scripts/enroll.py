"""Add people to the gallery, list who is in it, or remove someone.

Enrolling is just storing an embedding under a name, so this is fast and trains nothing.
Several shots per person is the default because pose moves an embedding more than anything
else (measured in #5), and one photo pins a person to one pose.

Usage:
    uv run python scripts/enroll.py --name "Aaditya" --camera --shots 5
    uv run python scripts/enroll.py --name "Aaditya" --images data/faces/aaditya/*.jpg
    uv run python scripts/enroll.py --list
    uv run python scripts/enroll.py --remove "Aaditya"
"""

from __future__ import annotations

import argparse
import time

import cv2

from facepipe.capture import CameraSource
from facepipe.gallery import Gallery
from facepipe.pipeline import Pipeline


def enroll_from_camera(pipeline: Pipeline, name: str, shots: int, delay: float) -> int:
    added = 0
    with CameraSource() as camera:
        for shot in range(shots):
            print(f"  shot {shot + 1}/{shots} - move your head a little...")
            time.sleep(delay)
            frame = camera.read()
            if frame is None:
                print("    no frame")
                continue
            embedding = pipeline.embed_largest_face(frame)
            if embedding is None:
                print("    no face found, skipping")
                continue
            pipeline.gallery.add(name, embedding)
            added += 1
    return added


def enroll_from_images(pipeline: Pipeline, name: str, paths: list[str]) -> int:
    added = 0
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  could not read {path}")
            continue
        embedding = pipeline.embed_largest_face(frame)
        if embedding is None:
            print(f"  no face in {path}")
            continue
        pipeline.gallery.add(name, embedding)
        added += 1
        print(f"  added {path}")
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="who to enrol")
    parser.add_argument("--camera", action="store_true", help="capture from the webcam")
    parser.add_argument("--images", nargs="+", help="enrol from image files instead")
    parser.add_argument("--shots", type=int, default=5, help="webcam shots to take")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between shots")
    parser.add_argument("--list", action="store_true", help="show who is enrolled")
    parser.add_argument("--remove", help="forget this person")
    args = parser.parse_args()

    if args.list:
        people = Gallery().people()
        if not people:
            print("nobody enrolled yet")
        for name, count in people.items():
            print(f"{name}: {count} embedding(s)")
        return

    if args.remove:
        gallery = Gallery()
        dropped = gallery.remove(args.remove)
        gallery.save()
        print(f"removed {args.remove} ({dropped} embeddings)")
        return

    if not args.name or not (args.camera or args.images):
        parser.error("--name plus either --camera or --images")

    pipeline = Pipeline()
    print(f"enrolling {args.name}")
    added = (
        enroll_from_images(pipeline, args.name, args.images)
        if args.images
        else enroll_from_camera(pipeline, args.name, args.shots, args.delay)
    )

    pipeline.gallery.save()
    print(f"\nadded {added} embedding(s) for {args.name}")
    for name, count in pipeline.gallery.people().items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
