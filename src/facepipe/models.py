"""Loading the ONNX models, with an error a human can act on.

Without this, running anything before downloading the weights gives an onnxruntime
NO_SUCHFILE stack trace, which tells you a path is missing but not that there is a script
whose whole job is to put a file there. The models are ~190MB and deliberately not
committed, so a fresh clone always hits this once.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def load_session(model_path: Path) -> ort.InferenceSession:
    """Open an ONNX model, or explain how to get it."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path.name} is missing from {model_path.parent}.\n"
            "The model weights are ~190MB so they are not committed. Download them with:\n"
            "    uv run python scripts/fetch_models.py"
        )
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
