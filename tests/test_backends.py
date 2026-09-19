"""Tests that the backend seam is real.

The claim in #8 is that the pipeline talks to an interface, not to onnxruntime. That is
easy to say and easy to quietly break, so it gets checked: the real classes satisfy the
protocols, the pipeline runs on substitutes that have never heard of ONNX, and nothing
above the seam imports onnxruntime.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from facepipe.backends import FaceDetector, FaceEmbedder
from facepipe.gallery import Gallery
from facepipe.pipeline import Pipeline
from facepipe.types import Detection


class FakeDetector:
    """Finds one face in the middle of the frame. No model involved."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        return [
            Detection(
                bbox=np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75], dtype=np.float32),
                score=0.99,
                kps=np.array(
                    [[40, 50], [70, 50], [55, 70], [45, 90], [68, 90]], dtype=np.float32
                ),
            )
        ]


class FakeEmbedder:
    """Returns a fixed unit vector."""

    def __init__(self, axis: int = 0):
        self.axis = axis

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        vector = np.zeros(512, dtype=np.float32)
        vector[self.axis] = 1.0
        return vector


def test_real_backends_satisfy_the_protocols():
    """The ONNX classes fit the interface without ever referring to it."""
    from facepipe.arcface import ArcFaceEmbedder
    from facepipe.scrfd import SCRFDDetector

    # __new__ skips loading the models; a runtime_checkable Protocol checks methods.
    assert isinstance(SCRFDDetector.__new__(SCRFDDetector), FaceDetector)
    assert isinstance(ArcFaceEmbedder.__new__(ArcFaceEmbedder), FaceEmbedder)
    assert isinstance(FakeDetector(), FaceDetector)
    assert isinstance(FakeEmbedder(), FaceEmbedder)


def test_pipeline_does_not_load_onnxruntime_for_other_backends(tmp_path):
    """The strongest form of the claim: onnxruntime is never even imported.

    Run in a subprocess so this module's other tests, which do use the real models,
    cannot have already imported it.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from facepipe.gallery import Gallery
        from facepipe.pipeline import Pipeline
        sys.path.insert(0, "tests")
        from test_backends import FakeDetector, FakeEmbedder

        Pipeline(detector=FakeDetector(), embedder=FakeEmbedder(), gallery=Gallery(sys.argv[1]))
        assert "onnxruntime" not in sys.modules, "the seam leaks: onnxruntime got imported"
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert "clean" in result.stdout, result.stderr


def test_pipeline_runs_without_any_real_model(tmp_path):
    """If the pipeline needed onnxruntime, this test could not exist."""
    pipeline = Pipeline(
        detector=FakeDetector(), embedder=FakeEmbedder(), gallery=Gallery(tmp_path)
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    pipeline.gallery.add("Aaditya", pipeline.embed_largest_face(frame))
    faces = pipeline.process(frame)

    assert len(faces) == 1
    assert faces[0].match.name == "Aaditya"


def test_swapping_the_embedder_changes_the_answer(tmp_path):
    """Recognition follows the backend, which is what swappable has to mean."""
    gallery = Gallery(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    enrolled = Pipeline(detector=FakeDetector(), embedder=FakeEmbedder(0), gallery=gallery)
    gallery.add("Aaditya", enrolled.embed_largest_face(frame))

    # A different backend produces a different embedding for the same face.
    other = Pipeline(detector=FakeDetector(), embedder=FakeEmbedder(7), gallery=gallery)
    assert not other.process(frame)[0].match.is_known


def test_fpga_backend_is_honestly_unimplemented():
    from facepipe.backends.vitis import VitisFaceDetector, VitisFaceEmbedder

    with pytest.raises(NotImplementedError):
        VitisFaceDetector()
    with pytest.raises(NotImplementedError):
        VitisFaceEmbedder()


def test_nothing_above_the_seam_imports_onnxruntime():
    """The seam is only real if the rest of the pipeline does not reach past it."""
    src = Path(__file__).resolve().parents[1] / "src" / "facepipe"
    allowed = {"scrfd.py", "arcface.py"}  # below the line: these own the runtime
    imports = re.compile(r"^\s*(?:import|from)\s+onnxruntime", re.MULTILINE)

    offenders = [
        path.name
        for path in src.rglob("*.py")
        if path.name not in allowed and imports.search(path.read_text())
    ]
    assert offenders == [], f"these reach past the backend seam: {offenders}"
