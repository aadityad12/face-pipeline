"""The dashboard: a live annotated feed, plus enrolling people from the browser.

The camera loop runs once in the background rather than inside the request handler. If it
ran per request, two open tabs would both try to open the camera and the second would
fail, and the models would run once per viewer instead of once per frame. Instead one
producer loop keeps the latest annotated frame, and every request just reads it (issue #4).

Frames go out as MJPEG: the response stays open and JPEGs are sent one after another,
which any <img> tag renders like a flipbook. About fifteen lines and no client library.
The cost is that every frame is sent whole, with no compression between frames, so it is
bandwidth-hungry. Irrelevant on localhost, and something to revisit over a network.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from facepipe.capture import CameraSource
from facepipe.pipeline import Pipeline, annotate

STATIC_DIR = Path(__file__).parent / "static"
JPEG_QUALITY = 80
BOUNDARY = "frame"


class CameraWorker:
    """Reads the camera, runs the pipeline, and keeps the latest annotated frame."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.faces: list = []
        self.running = False
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            camera = CameraSource()
        except RuntimeError as exc:
            self.error = str(exc)
            return

        try:
            while self.running:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                faces = self.pipeline.process(frame)

                start = time.perf_counter()
                ok, buffer = cv2.imencode(
                    ".jpg", annotate(frame, faces), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                self.pipeline.timings.record("encode", (time.perf_counter() - start) * 1000)

                if ok:
                    with self.lock:
                        self.latest_jpeg = buffer.tobytes()
                        self.latest_frame = frame
                        self.faces = faces
        finally:
            camera.release()

    def snapshot(self) -> tuple[bytes | None, np.ndarray | None]:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return self.latest_jpeg, frame


class EnrollRequest(BaseModel):
    name: str


def create_app(pipeline: Pipeline | None = None, worker: CameraWorker | None = None):
    """The worker is injectable so the routes can be tested without a real camera."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker.start()
        yield
        app.state.worker.stop()

    app = FastAPI(title="face-pipeline", lifespan=lifespan)
    app.state.pipeline = pipeline if pipeline is not None else Pipeline()
    app.state.worker = (
        worker if worker is not None else CameraWorker(app.state.pipeline)
    )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/video_feed")
    def video_feed() -> StreamingResponse:
        def frames():
            while True:
                jpeg, _ = app.state.worker.snapshot()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n"
                )
                # The pipeline is slower than any browser, so this just avoids spinning
                # and re-sending a frame that has not changed.
                time.sleep(0.03)

        return StreamingResponse(
            frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )

    @app.get("/api/status")
    def status() -> dict:
        worker = app.state.worker
        timings = app.state.pipeline.timings
        return {
            "error": worker.error,
            "faces": [
                {
                    "name": face.match.name,
                    "score": round(face.match.score, 3),
                    "known": face.match.is_known,
                }
                for face in worker.faces
            ],
            "fps": round(timings.fps, 1),
            "stage_ms": {k: round(v, 1) for k, v in timings.averages().items()},
            "threshold": app.state.pipeline.threshold,
        }

    @app.get("/api/gallery")
    def gallery() -> dict:
        return {"people": app.state.pipeline.gallery.people()}

    @app.post("/api/enroll")
    def enroll(request: EnrollRequest) -> dict:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="a person needs a name")

        _, frame = app.state.worker.snapshot()
        if frame is None:
            raise HTTPException(status_code=503, detail="no camera frame yet")

        embedding = app.state.pipeline.embed_largest_face(frame)
        if embedding is None:
            raise HTTPException(status_code=422, detail="no face in the current frame")

        app.state.pipeline.gallery.add(name, embedding)
        app.state.pipeline.gallery.save()
        return {"name": name, "people": app.state.pipeline.gallery.people()}

    @app.delete("/api/gallery/{name}")
    def forget(name: str) -> dict:
        dropped = app.state.pipeline.gallery.remove(name)
        app.state.pipeline.gallery.save()
        return {"removed": name, "embeddings": dropped}

    return app


# Deliberately no module-level `app = create_app()`. That ran on import, which built a
# Pipeline, which loaded ~190MB of models, so simply importing this module needed the
# weights to already be on disk. A fresh clone does not have them yet, so the tests could
# not even be collected. uvicorn takes a factory instead, and importing the module is now
# free (issue #4).
def run() -> None:
    """Entry point for `uv run serve`."""
    import uvicorn

    uvicorn.run("facepipe.server:create_app", factory=True, host="127.0.0.1", port=8000)
