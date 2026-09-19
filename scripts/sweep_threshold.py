"""Measure where the match threshold should go, instead of guessing it.

The threshold is the whole security property of this system. Below it a face is Unknown,
above it the face gets a name, and the number decides whether a stranger gets let in or a
real person gets turned away. Copying 0.5 from a tutorial is a guess, and the right value
depends on this alignment, this model and these photos.

Method: embed every photo, then score every possible pair.

  genuine pair   - two photos of the same person, should score high
  impostor pair  - two photos of different people, should score low

Then sweep a threshold across the range and count the two kinds of mistake at each value:

  false accept  - an impostor pair scoring above the threshold  (a stranger gets a name)
  false reject  - a genuine pair scoring below it               (a real person is refused)

Those two trade off directly: raising the threshold cuts false accepts and causes false
rejects. For a door, letting the wrong person in is much worse than making the right
person try twice, so the pick is the lowest threshold with no false accepts, plus margin,
rather than whatever value maximises raw accuracy.

Usage:
    uv run python scripts/sweep_threshold.py
    uv run python scripts/sweep_threshold.py --write-results
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from facepipe.align import align_face
from facepipe.arcface import ArcFaceEmbedder
from facepipe.config import DATA_DIR, PROJECT_ROOT
from facepipe.scrfd import SCRFDDetector


def load_people(faces_dir: Path) -> dict[str, list[Path]]:
    people = {}
    for person_dir in sorted(p for p in faces_dir.iterdir() if p.is_dir()):
        images = sorted(person_dir.glob("*.jpg"))
        if images:
            people[person_dir.name] = images
    return people


def embed_all(people: dict[str, list[Path]]) -> dict[str, list[np.ndarray]]:
    detector, embedder = SCRFDDetector(), ArcFaceEmbedder()
    embeddings: dict[str, list[np.ndarray]] = {}

    for name, paths in people.items():
        vectors = []
        for path in paths:
            frame = cv2.imread(str(path))
            detections = detector.detect(frame)
            if not detections:
                print(f"  no face in {path}, skipped")
                continue
            largest = max(
                detections,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            )
            vectors.append(embedder.embed(align_face(frame, largest.kps)))
        embeddings[name] = vectors
        print(f"  {name}: {len(vectors)} embeddings")
    return embeddings


def build_pairs(
    embeddings: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    genuine, impostor = [], []

    for vectors in embeddings.values():
        for a, b in itertools.combinations(vectors, 2):
            genuine.append(float(np.dot(a, b)))

    for first, second in itertools.combinations(embeddings.keys(), 2):
        for a in embeddings[first]:
            for b in embeddings[second]:
                impostor.append(float(np.dot(a, b)))

    return np.array(genuine), np.array(impostor)


def histogram(scores: np.ndarray, label: str, width: int = 40) -> list[str]:
    """A rough text histogram, so the separation can be seen and not just described."""
    lines = [f"{label} (n={len(scores)})"]
    edges = np.arange(-0.2, 1.01, 0.1)
    counts, _ = np.histogram(scores, bins=edges)
    peak = max(counts.max(), 1)
    for i, count in enumerate(counts):
        bar = "#" * int(width * count / peak)
        lines.append(f"  {edges[i]:+.1f} to {edges[i + 1]:+.1f}  {bar} {count}")
    return lines


def sweep(genuine: np.ndarray, impostor: np.ndarray) -> list[tuple[float, int, int]]:
    rows = []
    for threshold in np.arange(0.20, 0.85, 0.01):
        false_accepts = int((impostor >= threshold).sum())
        false_rejects = int((genuine < threshold).sum())
        rows.append((float(threshold), false_accepts, false_rejects))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces", default=str(DATA_DIR / "faces"))
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    people = load_people(Path(args.faces))
    if len(people) < 2:
        raise SystemExit("need photos of at least 2 people to form impostor pairs")

    print("embedding photos")
    embeddings = embed_all(people)
    genuine, impostor = build_pairs(embeddings)

    report: list[str] = []
    report.append(f"people: {', '.join(f'{k} ({len(v)})' for k, v in embeddings.items())}")
    report.append(f"genuine pairs: {len(genuine)}   impostor pairs: {len(impostor)}")
    report.append("")
    report.append(
        f"same person      min {genuine.min():+.3f}  mean {genuine.mean():+.3f}  max {genuine.max():+.3f}"
    )
    report.append(
        f"different people min {impostor.min():+.3f}  mean {impostor.mean():+.3f}  max {impostor.max():+.3f}"
    )
    report.append(f"gap between them: {genuine.min() - impostor.max():+.3f}")
    report.append("")
    report.extend(histogram(genuine, "same person"))
    report.append("")
    report.extend(histogram(impostor, "different people"))
    report.append("")

    rows = sweep(genuine, impostor)
    report.append("threshold   false accepts   false rejects")
    for threshold, false_accepts, false_rejects in rows:
        if round(threshold * 100) % 5 == 0:
            report.append(
                f"   {threshold:.2f}          {false_accepts:3d}             {false_rejects:3d}"
            )
    report.append("")

    perfect = [t for t, false_accepts, false_rejects in rows if not false_accepts and not false_rejects]
    if not perfect:
        report.append("no threshold in range separates the two sets cleanly")
        report.append("pick from the sweep above, trading false accepts against rejects")
        chosen = None
    else:
        low, high = min(perfect), max(perfect)
        chosen = round((low + high) / 2, 2)
        report.append(f"zero-error band: {low:.2f} to {high:.2f}")
        report.append(f"chosen: {chosen:.2f}, the middle of it")
        report.append("")
        report.append(
            "Picking the middle rather than either edge: the edges are exactly where the "
            "closest impostor pair and the furthest genuine pair sit, so a threshold there "
            "is one unseen face away from being wrong. The middle is the furthest point "
            "from both mistakes."
        )
        report.append("")
        report.append(
            f"Caveat: {len(embeddings)} people and {len(impostor)} impostor pairs is a "
            "small test set, all from one webcam in one room. The impostor maximum is "
            "almost certainly optimistic, since two people who happen to look alike would "
            "score higher than any pair here. For a door I would rather refuse a real "
            "person than admit a stranger, so if this were going into production the "
            "threshold should move up within the band, not down."
        )

    text = "\n".join(report)
    print("\n" + text)

    if args.write_results:
        out = PROJECT_ROOT / "docs" / "results.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# Measured results\n\n## Match threshold (issue #11)\n\n"
            "Generated by `scripts/sweep_threshold.py`.\n\n```\n" + text + "\n```\n"
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
