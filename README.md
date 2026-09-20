# face-pipeline

Face detection and recognition built end to end on a laptop: a camera frame goes in, a name
comes out.

Take-home project for the SCE AI x FPGA team. The goal was not a polished app. It was to
own every stage between the camera and the answer, so I can explain what each one does,
why it is there, and what it costs.

Nothing here is trained. Two pretrained models do the heavy lifting and everything between
them is mine.

## The pipeline

```
camera frame                 (480, 640, 3) BGR uint8
  │
  ├─ letterbox + normalize   (1, 3, 640, 640) float32     scale + padding kept
  ├─ SCRFD                   9 raw tensors
  ├─ decode + NMS            1 box + 5 landmarks per face, back in frame coordinates
  │
  ├─ track across frames     IoU > 0.3 = same face, reuse its name and skip ahead
  ├─ align to template       (112, 112, 3) per face       eyes on fixed pixels
  ├─ ArcFace                 (512,) per face              L2-normalized
  ├─ match vs gallery        cosine >= 0.33 ? name : Unknown
  │
  └─ draw + JPEG + MJPEG     browser
```

Two models doing two different jobs. **SCRFD** answers *where are the faces* and returns a
box plus five landmarks. **ArcFace** answers *whose face is this*, by turning one aligned
face into 512 numbers that are similar for two photos of the same person. Enrolling
someone is just storing their 512 numbers, so adding a person takes a second.

## Running it

```bash
uv sync
uv run pytest tests/ -q                       # 54 tests, no model weights needed
uv run python scripts/fetch_models.py         # ~190MB, not committed
uv run serve                                  # http://127.0.0.1:8000
```

On macOS your terminal needs camera permission (System Settings → Privacy & Security →
Camera). `uv sync` fetches Python 3.12 itself and installs from the lockfile, so the
environment should reproduce exactly.

Other things to run:

```bash
uv run python scripts/detect_faces.py --image data/sample.jpg   # detection only
uv run python scripts/compare_faces.py a.jpg b.jpg              # how similar are two faces
uv run python scripts/compare_faces.py a.jpg b.jpg --no-align   # what alignment is worth
uv run python scripts/collect_faces.py --name "You" --shots 8   # build a test set
uv run python scripts/sweep_threshold.py                        # where the threshold goes
uv run python scripts/bench.py                                  # per-stage timings
uv run python scripts/quantize.py                               # INT8 experiment
```

## Tools, and why

| Tool | Why | Rejected |
|---|---|---|
| `uv`, Python 3.12 | Lockfile plus a managed interpreter, so `uv sync` reproduces the environment. 3.12 not 3.14 because packages with compiled C++ inside lag new Python releases, and I did not want to debug a build on a deadline | 3.14 (my system default), conda, bare venv |
| `onnxruntime` | Runs both models. Same graph format Vitis AI consumes, so the port story stays real | PyTorch: 300MB+, and a DPU does not eat `.pt` |
| **SCRFD** `det_10g.onnx` | Detection. Returns 5 landmarks with every box, which is exactly what alignment needs, and it is built for edge budgets | MTCNN (3 stages, slow), Haar cascades (no landmarks, poor on angles) |
| **ArcFace** `w600k_r50.onnx` | Embeddings. Trained with an angular margin, so the *angle* between two embeddings carries the identity and cosine similarity is the model's own measure rather than something bolted on | FaceNet (older), training anything myself (absurd at this scale) |
| `opencv-python-headless` | Capture, warping, JPEG. Headless because output goes to a browser, which is also how it would run on a headless board | full `opencv-python`, which drags in GUI libraries |
| FastAPI + one HTML page | The dashboard. Deliberately **not** React: the brief says looks are not judged, so that time went into the pipeline and the measurements | React/Vite, correct for the real project, wrong use of three days here |
| MJPEG | ~15 lines, works in any `<img>`, no client library. Costs bandwidth: every frame sent whole with no compression between frames | WebRTC, days of work for a demo |

**The decision the rest follows from:** run the ONNX graphs directly rather than use the
`insightface` package, whose `FaceAnalysis.get(image)` would collapse this project into
three lines. Two reasons, in
[ADR 0001](docs/decisions/0001-onnxruntime-instead-of-insightface.md). The practical one is
that it ships source-only on PyPI with no arm64 wheel, so the "easy" path means compiling C
on an M-series Mac. The real one is that on a DPU there *is* no `.get(image)` — you get raw
tensors in and out, so letterboxing, decoding, NMS and alignment are my code either way.
Writing them on the laptop is what makes a port a port instead of a rewrite.

## What each stage owns

| Stage | What it does | The part that is easy to get wrong |
|---|---|---|
| Capture | Webcam or image file | macOS returns **black frames** for the first few hundred ms and `read()` reports success throughout |
| Letterbox | Fit 640x480 into 640x640 without stretching | Stretching squashes faces; the scale and padding must be kept or every box lands in the wrong place |
| Preprocess | BGR→RGB, `(x-127.5)/128`, HWC→CHW | Feeding raw 0–255 gives **48 detections on a 6-person photo** and no error |
| Decode | Anchor centre + 4 distances → box, on 3 grids | 2 anchors per grid point, which I confirmed from the output shapes (12800 = 80×80×2) rather than guessing |
| NMS | One box per face, IoU 0.4 | Too low merges two people standing close; too high leaves duplicates |
| Align | 5 landmarks → 112×112 via rotation, scale, shift | Skipping it costs more than everything else combined (below). A full affine would fit better but deforms the face |
| Embed | ArcFace → 512 numbers, L2-normalized | Without normalizing, the dot product is not a cosine |
| Track | IoU-match boxes between frames, reuse the name | Re-embedding a *stranger* every frame would give back the whole saving, so the check is when a track was last embedded, not whether it has a name |
| Match | Best cosine vs the gallery, else Unknown | Without the Unknown case a stranger always gets the nearest name, because someone is always nearest |

## Measured results

Full numbers in [docs/results.md](docs/results.md). The five that mattered:

**Alignment is worth more than I expected.** Two real photos of me, different pose and
distance: **0.834** aligned, **0.490** with just the detector's box. Rotating one photo by
hand and comparing against the upright original:

| same face, tilted | aligned | no alignment |
|---|---|---|
| 10° | +0.993 | +0.803 |
| 20° | +0.986 | +0.581 |
| 30° | +0.985 | **+0.391** |

The highest score I measured between two genuinely different people was **+0.214**. So
without alignment, tilting my head gets me most of the way to being a stranger to my own
face. Brightness and resolution barely move the embedding; alignment is specifically
cancelling pose.

**The threshold is measured, not copied.** 16 photos of 2 people → 56 same-person pairs and
64 different-person pairs:

```
same person       min +0.430   mean +0.655   max +0.863
different people  min +0.090   mean +0.160   max +0.235
```

Nothing lands between 0.235 and 0.430, so any threshold in 0.24–0.42 is perfect on this
data. I took **0.33**, the middle, because the edges sit exactly where the nearest mistake
of each kind is. The 0.5 I had been running with beforehand refuses **4 of 56** same-person
pairs — about 7% of the time telling an enrolled person they are Unknown, silently.

**I was wrong about the bottleneck.** One frame, one face:

```
detect (model)   115.2 ms   63%
embed (model)     64.4 ms   35%
everything else    4.1 ms    2%
total            183.8 ms   5.4 fps
```

I had assumed embedding dominated and said so in an earlier commit. It only dominates with
several people in frame, because embedding is per *face* while detection is per *frame*.
The two models are 98% of the work; all the code I wrote is the other 2%.

**Tracking is a 4.3x win on a crowd.** Detection is per frame, embedding is per face, so
the cost of a room full of people is entirely in the embedder. Boxes overlapping heavily
between consecutive frames are the same person, so each track keeps its name and ArcFace
only runs for a face that is new or stale. 6 faces, 20 frames:

| | no tracking | tracking |
|---|---|---|
| static scene | 1.82 fps | **7.91 fps** |
| faces drifting a few px/frame | 1.89 fps | **8.27 fps** |

The remaining ~125ms is detection, which tracking cannot touch. With a *single* face the
win is small, because detection already dominates that frame. This is a crowd optimisation.

**INT8 is cheap here.** An FPGA DPU is integer hardware, so this is the question the board
poses, asked somewhere easier:

| | float32 | int8 |
|---|---|---|
| model size | 174 MB | 44 MB |
| embed time | 66.2 ms | 39.2 ms |
| highest impostor | 0.235 | 0.237 |
| lowest genuine | 0.430 | 0.428 |

Embeddings move by a mean cosine of **0.9935** and 0.33 still separates the two groups.
Caveat: this is *dynamic* quantisation, the easy cousin of the static, calibration-set flow
a DPU needs, so the drift here is a lower bound.

## Moving this to the Kria K26

[docs/kria-mapping.md](docs/kria-mapping.md) in full. The short version: the pipeline
order, gallery, matching, enrolment, dashboard and alignment do not change at all, and
neither does the *logic* of letterboxing, decoding and NMS — a DPU gives you raw tensors,
so that code is needed regardless. What changes is how the model is loaded and run
(`backends/base.py`, two small protocols), static INT8 quantisation of both models, tensor
layout (NHWC vs NCHW), and possibly the threshold, which would need re-measuring on the
board.

What I would budget time for: operations the compiler cannot map to the DPU get left on
the ARM cores, so a graph that compiles cleanly can still be slow in the middle. And if
inference gets much faster, the preprocessing that is 2% here does not shrink, on cores
slower than this laptop's.

## What I got wrong

Four bugs, and they are all the same bug.

1. **Black camera frames.** First webcam test reported 0 faces. I assumed my decoding was
   broken; the frame was every pixel 0, because macOS cameras return black frames while
   warming up and `read()` reports success the whole time.
2. **An empty gallery is falsy.** `Pipeline` did `gallery or Gallery()`. Because `Gallery`
   defines `__len__`, an empty one is falsy, so it discarded the gallery passed in and
   built a default pointing elsewhere. Matching still worked. The enrolments were just
   going somewhere else, and I only noticed because a reload came back empty.
3. **A decorative abstraction.** `backends/base.py` imported `Detection` from `scrfd.py`,
   so importing the interface imported onnxruntime and the whole seam was worth nothing.
   Had I written the protocol and moved on, it would have looked right in review.
4. **A 190MB import.** `server.py` ended with `app = create_app()`, so importing it loaded
   both models, and a fresh clone could not even collect the tests. Found by cloning my
   own repo into a temp directory, which I should have been doing from the start.

None of them raised an exception. That is the theme of this whole project: in a pipeline
like this, almost nothing fails loudly. Skip the RGB conversion and you get worse
detections. Skip alignment and you get worse embeddings. Guess the threshold and you refuse
7% of real people. Every one of those looks exactly like "the model isn't very good" unless
you measure it.

## Known limitations

- **No liveness detection.** Hold up a printed photo of an enrolled person and this will
  happily recognise them. First thing to fix for anything guarding a real door (issue #12).
- **The threshold rests on a small test set.** 2 people, one webcam, one room, and nobody
  in it looks alike, so 0.235 for the impostor maximum is optimistic. For a door I would
  move the threshold up within the band, since refusing a real person is much better than
  admitting a stranger.
- **5.4 fps** on this CPU with one face, and tracking does not help much there because
  detection dominates a single-face frame. Tracking takes 6 faces from 1.8 to 7.9 fps.
- **The tracker is greedy IoU with no motion model.** Fine for faces at a door at ~8 fps;
  it would lose people who move fast or cross behind each other.
- **MJPEG is bandwidth-hungry.** Irrelevant on localhost, would matter over a network.
- **One camera, localhost only** (issue #14).
- Enrolment takes the largest face in frame, so it assumes the person enrolling is closest.

## Face data

`data/faces/` has 16 photos of two people. Both of us consented to them being in a public
repo, and they exist because the threshold measurement needs real same-person and
different-person pairs from the same camera.

Enrolled embeddings (`data/gallery/`) are gitignored. They are biometric data and they are
runtime state, not source.

Worth recording: the very first frame I captured for this project had a stranger sitting at
a table behind me, and the detector found her face at 0.71 confidence in a box 30×39 pixels
wide. She had no idea. I re-shot the photo rather than commit it. That is the privacy
problem of this entire technology showing up uninvited on day one, and a real deployment in
the SCE office needs an answer to it — consent, retention, and who can query the gallery —
well before it needs better accuracy.

## How I used AI

The brief asked for it, so here is an honest account.

I used Claude heavily, mostly as a way to go faster on things I could not have written from
scratch in three days: the SCRFD decoding, the Umeyama similarity transform, and the
scaffolding around FastAPI and the scripts.

What I did not delegate was the deciding and the checking. Every measurement in this README
is one I ran, and several of them exist because I did not want to take a claim on trust —
the alignment comparison, the normalization experiment, the threshold sweep and the
detector size sweep are all there because "this matters" is not the same as knowing how
much. Two of the four bugs above were found by checking output I had been told was fine:
the empty gallery turned up because I reloaded from disk to see if persistence worked, and
the 190MB import turned up because I cloned my own repo to see what was actually in it.

The issues in this repo were written by hand, before the code, and the commit messages say
what I actually understood at the time. Where I was wrong — the bottleneck, mainly — the
history says so rather than being tidied up.

## Plan and progress

Work is broken down in [the issues](../../issues): Detection, Recognition, and Hardening
and Measurement. Three issues are tagged `out-of-scope` and left open deliberately, because
they are things this project does **not** do.
