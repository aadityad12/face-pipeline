"""Quantise ArcFace to INT8 and measure what it costs.

An FPGA DPU is integer hardware, so running these models on a board means quantising them
first: storing the weights as 8-bit integers instead of 32-bit floats. That is not free.
Less precision moves the embeddings, and if they move far enough the threshold measured in
#11 is no longer the right threshold.

Doing it on the laptop is the same question in a much easier place, and it answers
something concrete before anyone touches hardware: how much does INT8 shift a face
embedding, and does the same-person / different-person separation survive it? (issue #10)

Measured here:

  speed   - median embedding time, float32 vs int8
  drift   - cosine between the float32 and int8 embedding of the same photo
            (1.0 means the quantised model agrees exactly with the original)
  effect  - whether the zero-error threshold band from #11 still holds under INT8

Usage:
    uv run python scripts/quantize.py
    uv run python scripts/quantize.py --write-results
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from facepipe.align import align_face
from facepipe.arcface import ArcFaceEmbedder
from facepipe.config import DATA_DIR, MODEL_DIR, PROJECT_ROOT, REC_MODEL
from facepipe.scrfd import SCRFDDetector

INT8_MODEL = MODEL_DIR / "w600k_r50_int8.onnx"


def quantize_model() -> Path:
    """Dynamic INT8 quantisation: weights to int8, activations measured at runtime.

    Dynamic rather than static because static needs a calibration set fed through the
    model. A real DPU flow uses static quantisation with calibration data, so this is the
    easier cousin of what the board would need, and the drift it shows is a lower bound on
    what the board would see.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if INT8_MODEL.exists():
        print(f"{INT8_MODEL.name} already exists, reusing it")
        return INT8_MODEL

    print(f"quantising {REC_MODEL.name} -> {INT8_MODEL.name}")
    quantize_dynamic(
        model_input=str(REC_MODEL),
        model_output=str(INT8_MODEL),
        weight_type=QuantType.QUInt8,
    )
    return INT8_MODEL


def load_faces(limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    """Aligned 112x112 crops from the photo set used in #11."""
    detector = SCRFDDetector()
    faces = []
    for person_dir in sorted(p for p in (DATA_DIR / "faces").iterdir() if p.is_dir()):
        for path in sorted(person_dir.glob("*.jpg"))[:limit]:
            frame = cv2.imread(str(path))
            detections = detector.detect(frame)
            if detections:
                largest = max(
                    detections,
                    key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                )
                faces.append((person_dir.name, align_face(frame, largest.kps)))
    return faces


def median_ms(embedder: ArcFaceEmbedder, face: np.ndarray, runs: int) -> float:
    embedder.embed(face)  # warm up
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        embedder.embed(face)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def pair_stats(names: list[str], vectors: list[np.ndarray]) -> tuple[float, float]:
    """Highest different-person score and lowest same-person score."""
    genuine, impostor = [], []
    for i, j in itertools.combinations(range(len(names)), 2):
        score = float(np.dot(vectors[i], vectors[j]))
        (genuine if names[i] == names[j] else impostor).append(score)
    return max(impostor), min(genuine)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    int8_path = quantize_model()
    size_fp32 = REC_MODEL.stat().st_size / 1e6
    size_int8 = int8_path.stat().st_size / 1e6

    print("loading faces")
    faces = load_faces()
    names = [name for name, _ in faces]
    print(f"  {len(faces)} faces")

    fp32 = ArcFaceEmbedder()
    int8 = ArcFaceEmbedder(model_path=int8_path)

    fp32_vectors = [fp32.embed(face) for _, face in faces]
    int8_vectors = [int8.embed(face) for _, face in faces]

    drift = [float(np.dot(a, b)) for a, b in zip(fp32_vectors, int8_vectors)]
    fp32_ms = median_ms(fp32, faces[0][1], args.runs)
    int8_ms = median_ms(int8, faces[0][1], args.runs)

    fp32_impostor_max, fp32_genuine_min = pair_stats(names, fp32_vectors)
    int8_impostor_max, int8_genuine_min = pair_stats(names, int8_vectors)

    lines = [
        f"{'':<22}{'float32':>12}{'int8':>12}",
        "-" * 46,
        f"{'model size (MB)':<22}{size_fp32:12.1f}{size_int8:12.1f}",
        f"{'embed time (ms)':<22}{fp32_ms:12.1f}{int8_ms:12.1f}",
        f"{'highest impostor':<22}{fp32_impostor_max:12.3f}{int8_impostor_max:12.3f}",
        f"{'lowest genuine':<22}{fp32_genuine_min:12.3f}{int8_genuine_min:12.3f}",
        f"{'zero-error band':<22}"
        f"{f'{fp32_impostor_max:.2f}-{fp32_genuine_min:.2f}':>12}"
        f"{f'{int8_impostor_max:.2f}-{int8_genuine_min:.2f}':>12}",
        "-" * 46,
        "",
        f"embedding drift, float32 vs int8 on the same photo:",
        f"  min {min(drift):.4f}   mean {statistics.mean(drift):.4f}   max {max(drift):.4f}",
        "",
        f"speed change: {(fp32_ms - int8_ms) / fp32_ms:+.0%}   "
        f"size change: {(size_int8 - size_fp32) / size_fp32:+.0%}",
    ]

    band_survives = int8_impostor_max < 0.33 < int8_genuine_min
    lines.append("")
    lines.append(
        f"the 0.33 threshold from #11 {'still separates' if band_survives else 'NO LONGER separates'} "
        "the two groups under int8"
    )

    text = "\n".join(lines)
    print("\n" + text)

    if args.write_results:
        out = PROJECT_ROOT / "docs" / "results.md"
        existing = out.read_text() if out.exists() else "# Measured results\n"
        marker = "## INT8 quantisation (issue #10)"
        block = (
            f"{marker}\n\nDynamic INT8 quantisation of ArcFace, measured on the same "
            f"{len(faces)} photos used for the threshold.\nGenerated by "
            f"`scripts/quantize.py`.\n\n```\n{text}\n```\n"
        )
        if marker in existing:
            existing = existing.split(marker)[0].rstrip() + "\n\n"
        out.write_text(existing.rstrip() + "\n\n" + block)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
