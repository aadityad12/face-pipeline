# face-pipeline

Face detection and recognition built end to end on a laptop: a camera frame goes in, a name
comes out.

This is a take-home project for the SCE AI x FPGA team. The goal is not a polished app. It is
to own every stage between the camera and the answer, so I can explain what each one does and
why it is there.

> This README is written as I go. It is filled in properly once the pipeline works; right now
> it records what I am building and what I do not know yet.

## The pipeline

```
camera frame                        (480, 640, 3) BGR
  -> letterbox + normalize          (1, 3, 640, 640) float32
  -> SCRFD                          9 raw tensors
  -> decode + NMS                   one box + 5 landmarks per face
  -> align to template              (112, 112, 3) per face
  -> ArcFace                        512 numbers per face
  -> compare to enrolled people     a name, or Unknown
  -> draw + stream to the browser
```

Two separate models doing two separate jobs. **SCRFD** answers *where are the faces*.
**ArcFace** answers *whose face is this*, by turning a face into 512 numbers that are similar
for two photos of the same person. Enrolling someone is just storing their 512 numbers, so
adding a person takes seconds and trains nothing.

## Running it

```bash
uv sync
uv run python scripts/fetch_models.py
uv run python scripts/inspect_models.py
```

`uv sync` builds the environment from the lockfile, including fetching Python 3.12 itself, so
it should reproduce exactly. The models are ~190MB and are downloaded rather than committed.

## Tools, and why

| Tool | Why |
|---|---|
| `uv`, Python 3.12 | Lockfile plus a managed interpreter, so `uv sync` gives the same environment on any machine. 3.12 rather than 3.14 because packages with compiled code inside lag behind new Python releases, and I would rather not be compiling C++ on a deadline. |
| `onnxruntime` | Runs the two models. Chosen over the `insightface` wrapper deliberately: see [ADR 0001](docs/decisions/0001-onnxruntime-instead-of-insightface.md). |
| SCRFD (`det_10g.onnx`) | Face detection. Returns 5 landmarks along with each box, which is exactly what alignment needs later. |
| ArcFace (`w600k_r50.onnx`) | Face embeddings. Trained so that the angle between two embeddings is the meaningful comparison, which makes cosine similarity the model's own measure rather than something bolted on. |
| `opencv-python-headless` | Camera capture, warping and JPEG encoding. Headless because the output goes to a browser, not a desktop window. |
| FastAPI + one plain HTML page | The dashboard. Deliberately not React: the brief says looks are not being judged, so that time went into the pipeline and the measurements instead. |

## Plan and progress

The work is broken down in [the issues](../../issues), grouped into Detection, Recognition,
and Hardening and Measurement. Three issues are tagged `out-of-scope` and left open on
purpose, because they are things this project does **not** do.

## Still to be written

- Measured results: per-stage timings, and how the match threshold was chosen
- What I got wrong along the way
- Known limitations
- How I used AI while building this
- Notes on the face images used and how they are handled
