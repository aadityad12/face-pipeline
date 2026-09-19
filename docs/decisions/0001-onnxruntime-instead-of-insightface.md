# 0001 - Run the ONNX models directly instead of using the insightface package

## Context

InsightFace publishes both models I need (SCRFD for detection, ArcFace for embeddings) and
also publishes a Python package that wraps them. With that package, this whole project is
roughly three lines: create a `FaceAnalysis`, call `.get(image)`, and read the names off the
result.

## Decision

Use `onnxruntime` to run the two `.onnx` files directly, and write the preprocessing,
decoding, NMS and alignment myself.

## Why

**The practical reason.** `insightface` ships source-only on PyPI, with no prebuilt wheel for
macOS on Apple Silicon, so installing it means compiling C and Cython on my laptop. That is a
bad thing to be debugging on a short deadline. `onnxruntime` and `opencv-python-headless` both
have arm64 wheels and install in seconds.

**The reason that actually matters.** The wrapper hides every stage I am supposed to
understand. Letterboxing, decoding the raw model outputs into boxes, NMS, aligning the face to
a template, normalizing the embedding: `.get(image)` does all of it invisibly. Writing it
myself is the difference between using a face recognition library and knowing how one works,
and understanding the pipeline end to end is the entire point of this project.

## Cost

About 200 lines of preprocessing and postprocessing code that a library would have given me
for free, and several ways to be subtly wrong that a library would not have. Those are
recorded in the issues: #2 (preprocessing), #3 (decoding and NMS), #5 (alignment).

## Status

Accepted.
