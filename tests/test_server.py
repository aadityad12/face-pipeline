"""Tests for the dashboard's API.

No camera and no models here: a fake worker supplies the "latest frame" and a fake
pipeline supplies the recognition result, so what gets tested is the wiring, including
the failure cases a browser can actually hit (no frame yet, no face in frame, no name).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from facepipe.gallery import Gallery, Match
from facepipe.pipeline import RecognizedFace
from facepipe.types import Detection
from facepipe.server import create_app


class FakeWorker:
    """Stands in for the camera loop."""

    def __init__(self, frame=None, faces=(), error=None):
        self.frame = frame
        self.faces = list(faces)
        self.error = error
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def snapshot(self):
        return (b"jpeg-bytes" if self.frame is not None else None, self.frame)


class FakePipeline:
    """Stands in for detect -> align -> embed -> match."""

    def __init__(self, gallery, embedding=None):
        self.gallery = gallery
        self.threshold = 0.5
        self.embedding = embedding
        self.timings = _FakeTimings()

    def embed_largest_face(self, frame):
        return self.embedding


class _FakeTimings:
    fps = 12.5

    def averages(self):
        return {"detect": 100.0, "embed": 40.0}


def face(name, score):
    detection = Detection(
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        score=0.9,
        kps=np.zeros((5, 2), dtype=np.float32),
    )
    return RecognizedFace(detection=detection, match=Match(name=name, score=score))


@pytest.fixture
def client_factory(tmp_path):
    def build(frame=None, faces=(), embedding=None, error=None):
        pipeline = FakePipeline(Gallery(tmp_path), embedding=embedding)
        worker = FakeWorker(frame=frame, faces=faces, error=error)
        return TestClient(create_app(pipeline=pipeline, worker=worker)), pipeline

    return build


def test_status_reports_faces_and_timings(client_factory):
    client, _ = client_factory(faces=[face("Aaditya", 0.83), face(None, 0.12)])
    body = client.get("/api/status").json()

    assert body["faces"][0] == {"name": "Aaditya", "score": 0.83, "known": True}
    assert body["faces"][1]["known"] is False
    assert body["fps"] == 12.5
    assert body["threshold"] == 0.5


def test_status_surfaces_a_camera_error(client_factory):
    client, _ = client_factory(error="could not open camera 0")
    assert "could not open camera" in client.get("/api/status").json()["error"]


def test_enroll_adds_the_face_in_frame(client_factory):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0

    client, pipeline = client_factory(frame=frame, embedding=embedding)
    response = client.post("/api/enroll", json={"name": "Aaditya"})

    assert response.status_code == 200
    assert response.json()["people"] == {"Aaditya": 1}
    assert pipeline.gallery.people() == {"Aaditya": 1}


def test_enroll_without_a_camera_frame_is_not_a_500(client_factory):
    client, _ = client_factory(frame=None)
    assert client.post("/api/enroll", json={"name": "Aaditya"}).status_code == 503


def test_enroll_with_no_face_in_frame(client_factory):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    client, _ = client_factory(frame=frame, embedding=None)

    response = client.post("/api/enroll", json={"name": "Aaditya"})
    assert response.status_code == 422
    assert "no face" in response.json()["detail"]


def test_enroll_needs_a_name(client_factory):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    client, _ = client_factory(frame=frame, embedding=np.zeros(512, dtype=np.float32))
    assert client.post("/api/enroll", json={"name": "   "}).status_code == 400


def test_gallery_lists_and_forgets(client_factory):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0

    client, _ = client_factory(frame=frame, embedding=embedding)
    client.post("/api/enroll", json={"name": "Aaditya"})
    assert client.get("/api/gallery").json()["people"] == {"Aaditya": 1}

    assert client.delete("/api/gallery/Aaditya").json()["embeddings"] == 1
    assert client.get("/api/gallery").json()["people"] == {}


def test_index_page_is_served(client_factory):
    client, _ = client_factory()
    response = client.get("/")
    assert response.status_code == 200
    assert "video_feed" in response.text


def test_importing_the_server_does_not_load_the_models():
    """A fresh clone has no model weights yet, so importing must not need them.

    server.py used to end with `app = create_app()`, which ran on import, built a
    Pipeline and loaded ~190MB of ONNX. On a clone with no models downloaded, pytest
    could not even collect this file. uvicorn is given a factory instead.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import facepipe.server
        assert "onnxruntime" not in sys.modules, "importing the server loaded a model"
        print("clean")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "clean" in result.stdout, result.stderr
