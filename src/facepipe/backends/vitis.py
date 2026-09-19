"""A deliberately unimplemented FPGA backend.

This file does not work and is not meant to. It exists so that "the inference backend is
swappable" is something you can check rather than something I claim in a README, and so
that the work a real port would involve is written down in the place it would happen
(issues #8 and #13).

What would actually change, and what would not:

  unchanged   the pipeline order, tracking, the gallery, cosine matching, enrolment,
              the dashboard, the 0.33 threshold decision, alignment (it is a warpAffine
              on the CPU either way)

  changed     how the model is loaded and run, and where preprocessing happens

The interesting part is that a DPU gives you raw tensors in and raw tensors out. There is
no library call that hands back faces. Letterboxing, the anchor decode, NMS and the
normalisation constants are still my code, unchanged in logic, which is the reason this
project runs the ONNX graphs directly instead of using insightface's wrapper (ADR 0001).

The parts I would expect to be genuinely hard:

  1. Quantisation. The DPU is integer hardware, so both models have to be quantised to
     INT8 with a calibration set, and that moves the embeddings. #10 measures how much
     that costs on the laptop, which is the same question in an easier place. If the
     embeddings shift, the threshold measured in #11 is no longer the right threshold and
     the sweep has to be re-run on the board.

  2. Unsupported operations. Whatever the compiler cannot map to the DPU gets left on the
     ARM cores, so a graph that looks fine can end up with a slow section in the middle.
     Worth checking before assuming any speedup.

  3. Layout. The DPU's tensors are typically NHWC and the ONNX graphs here are NCHW, so
     the transpose in preprocessing changes.
"""

from __future__ import annotations

import numpy as np

from facepipe.types import Detection

NOT_IMPLEMENTED = (
    "The FPGA backend is not implemented. This project runs on a laptop CPU; see "
    "src/facepipe/backends/vitis.py for what a real port would involve, and issue #13."
)


class VitisFaceDetector:
    """Placeholder for SCRFD running on a DPU."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError(NOT_IMPLEMENTED)


class VitisFaceEmbedder:
    """Placeholder for ArcFace running on a DPU."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError(NOT_IMPLEMENTED)
