"""Print what each model expects as input and returns as output.

Issue #1: I can't write the code around a model until I know its shapes. Everything in
#2 (preprocessing) and #3 (decoding) is written against what this prints.

Usage:  uv run python scripts/inspect_models.py
"""

from __future__ import annotations

import onnxruntime as ort

from facepipe.config import DET_MODEL, REC_MODEL


def describe(path) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"\n=== {path.name} ===")
    print("inputs:")
    for i in sess.get_inputs():
        print(f"  {i.name:<24} {i.shape}  {i.type}")
    print(f"outputs: ({len(sess.get_outputs())} tensors)")
    for o in sess.get_outputs():
        print(f"  {o.name:<24} {o.shape}  {o.type}")


if __name__ == "__main__":
    for model in (DET_MODEL, REC_MODEL):
        describe(model)
