"""One place for every constant, so no magic numbers end up buried in the pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# InsightFace publishes these two models together in one zip (the "buffalo_l" bundle).
# We only need two files out of it: the detector and the embedder.
MODEL_BUNDLE_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)

# SCRFD: finds faces. Gives a box + 5 landmarks per face.
DET_MODEL = MODEL_DIR / "det_10g.onnx"
DET_INPUT_SIZE = (640, 640)  # width, height the detector expects

# ArcFace: turns one aligned face into a 512-number embedding.
REC_MODEL = MODEL_DIR / "w600k_r50.onnx"
REC_INPUT_SIZE = (112, 112)

# Pinned after the first download so a corrupted or swapped file fails loudly (issue #1).
EXPECTED_SHA256: dict[str, str] = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
