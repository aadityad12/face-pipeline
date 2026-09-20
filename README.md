# face-pipeline

A face detection and recognition pipeline that runs end to end on my laptop. A camera frame
goes in and a name comes out.

This is my take home project for the SCE AI x FPGA team. I was not trying to build
something polished. I was trying to understand every step between the camera and the
answer, so that I can explain what each step does, why it is there, and what it costs.

I did not train anything. Two pretrained models do the actual recognition, and everything
between them is code I wrote.

## How to run it

You need [uv](https://docs.astral.sh/uv/). Then:

**1. Install everything**

```bash
uv sync
```

This also downloads Python 3.12 for you, so you do not need to have it installed already.

**2. Run the tests (optional, but it proves the install worked)**

```bash
uv run pytest tests/ -q
```

55 tests, and none of them need the models, so this works before the next step.

**3. Download the two models**

```bash
uv run python scripts/fetch_models.py
```

About 190MB. They are too big to commit so they get downloaded instead. If you skip this
step, anything that needs a model will tell you to come back and run it.

**4. Start the dashboard**

```bash
uv run serve
```

Then open **http://127.0.0.1:8000**. You will see your camera with a red box labelled
`Unknown`, because nobody is enrolled yet. Type a name, click enrol, and the box turns green
with that name on it.

On macOS your terminal needs camera permission the first time. That is in System Settings,
under Privacy & Security, then Camera.

### Other things you can run

```bash
# detect faces in an image, result gets written to data/scratch/
uv run python scripts/detect_faces.py --image data/sample.jpg

# how similar are two faces? first is the same person, second is two different people
uv run python scripts/compare_faces.py data/faces/aaditya/00.jpg data/faces/aaditya/05.jpg
uv run python scripts/compare_faces.py data/faces/aaditya/00.jpg data/faces/oak/00.jpg

# the same comparison with alignment turned off, to see what alignment is worth
uv run python scripts/compare_faces.py data/faces/aaditya/00.jpg data/faces/aaditya/05.jpg --no-align

# where should the match threshold go? this is how I picked 0.33
uv run python scripts/sweep_threshold.py

# how long does each stage take?
uv run python scripts/bench.py

# what does INT8 quantisation cost? this is the FPGA question
uv run python scripts/quantize.py

# take 8 photos of someone, for testing
uv run python scripts/collect_faces.py --name "You" --shots 8
```

## What the pipeline actually does

```
camera frame                 (480, 640, 3) BGR
  |
  |- letterbox + normalize   (1, 3, 640, 640) float32    keep the scale and padding
  |- SCRFD                   9 tensors of raw numbers
  |- decode + NMS            one box + 5 landmarks per face, back in frame coordinates
  |
  |- track across frames     if this is the same face as last frame, reuse its name
  |- align to 112x112        using the 5 landmarks
  |- ArcFace                 512 numbers per face
  |- match against gallery   cosine >= 0.33 gives a name, below that gives Unknown
  |
  |- draw, JPEG, MJPEG       browser
```

There are two models doing two completely different jobs, and keeping that separate in my
head was the thing that made the rest of this make sense.

**SCRFD** answers *where are the faces*. You give it a whole frame and it gives back a box
around each face, plus five landmark points (two eyes, nose tip, two mouth corners). It has
no idea who anyone is.

**ArcFace** answers *whose face is this*. You give it one cropped and lined up face and it
gives back 512 numbers, which is called an embedding. Two photos of the same person produce
similar numbers and two different people produce different ones. It has never seen me and it
does not need to be retrained to recognise me. Enrolling someone is just saving their 512
numbers once, so adding a person takes a second.

## Tools I used and why

| Tool | Why I picked it | What I did not pick |
|---|---|---|
| `uv` with Python 3.12 | It gives a lockfile and downloads its own Python, so `uv sync` gives you the same environment I have. 3.12 instead of 3.14 because opencv and onnxruntime have compiled C++ inside them, and new Python versions often do not have prebuilt versions yet, which means compiling them yourself. I did not want to spend my three days on that | 3.14, which is what my Mac actually runs |
| `onnxruntime` | Runs both models. It is also the same format Vitis AI uses on the FPGA, so the work does not get thrown away later | PyTorch, which is 300MB+ and is not what a board runs |
| **SCRFD** (`det_10g.onnx`) | Face detection. It gives 5 landmarks along with every box, which is exactly what I need for alignment later, and it is built to be fast | MTCNN, which is 3 models in a row and slower. Haar cascades, which give no landmarks and struggle with angles |
| **ArcFace** (`w600k_r50.onnx`) | Face embeddings. It is trained in a way that makes the angle between two embeddings the meaningful thing, so cosine similarity is the model's own measure and not something I invented | FaceNet, which is older. Training my own, which is not realistic here |
| `opencv-python-headless` | Camera, warping and JPEG. Headless because the output goes to a browser and not a desktop window, which is also how it would run on a board with no screen | Full `opencv-python`, which pulls in GUI libraries I would never use |
| FastAPI and one HTML page | The dashboard. I deliberately did not use React. The brief says looks are not being judged, so I would rather put that time into the pipeline and the measurements | React, which is the right answer for the real project but the wrong use of three days here |
| MJPEG for the video | About 15 lines and it works in a plain `<img>` tag with no extra libraries. The cost is that every frame gets sent whole with no compression between frames, so it uses a lot of bandwidth | WebRTC, which would take days for a demo |

### The decision everything else follows from

InsightFace publishes both of these models, and also publishes a Python package that wraps
them. With that package this whole project is about three lines: build a `FaceAnalysis`,
call `.get(image)`, and read the names off the result.

I ran the ONNX models directly instead. There were two reasons, written up properly in
[ADR 0001](docs/decisions/0001-onnxruntime-instead-of-insightface.md).

The practical one is that the package ships source only on PyPI with no prebuilt version for
Apple Silicon, so the easy path actually means compiling C on my Mac.

The real one is that on an FPGA there is no `.get(image)`. A DPU gives you raw tensors in and
raw tensors out and that is all. The letterboxing, the decoding, NMS and the alignment are my
code either way. So writing them now means a port to the board is a port, while using the
wrapper would have made it a rewrite. It cost me about 200 lines that a library would have
hidden, and those 200 lines are the part of this project I actually understand.

## What each stage does, and how it goes wrong

| Stage | What it does | The quiet way it breaks |
|---|---|---|
| Capture | Reads the webcam or an image file | macOS gives back completely black frames for the first few hundred ms, and `read()` says success the whole time |
| Letterbox | Fits a 640x480 frame into 640x640 without stretching | Stretching squashes faces and the model was not trained on squashed faces. Also, if you do not keep the scale and padding, every box gets drawn in the wrong place |
| Preprocess | BGR to RGB, `(x - 127.5) / 128`, then reorder the axes | Feeding raw 0 to 255 finds **48 faces in a photo of 6 people**, and throws no error at all |
| Decode | An anchor point plus 4 distances becomes a box, on 3 grids | There are 2 anchors per grid point. I got that from the model's own output shapes instead of guessing (12800 rows = 80 x 80 x 2) |
| NMS | Keeps one box per face, at IoU 0.4 | Too low and two people standing close together merge into one box. Too high and duplicates survive |
| Align | 5 landmarks to a 112x112 crop, by rotating, resizing and shifting | Skipping it costs more than everything else on this list. Allowing stretching would fit the points better, but it deforms the face, and the shape of the face is the identity |
| Track | Matches boxes between frames and reuses the name | Checking "does this track have a name" would re-embed every stranger every frame, because Unknown tracks have no name. It checks when the track was last embedded instead |
| Embed | ArcFace to 512 numbers, normalised to length 1 | If you do not normalise, the dot product is not a cosine and the threshold means nothing |
| Match | Best score in the gallery, or Unknown | Without the Unknown case every stranger gets the nearest name, because somebody is always nearest |

## What I measured

All of the numbers are in [docs/results.md](docs/results.md), and every one of them can be
reproduced with the scripts above. Five things mattered.

### 1. Alignment matters much more than I thought

Two real photos of me at a different angle and distance: **0.834** with alignment, **0.490**
with just the detector's box. I also rotated one photo by hand and compared it against the
upright original:

| same face, tilted by | aligned | no alignment |
|---|---|---|
| 10 degrees | +0.993 | +0.803 |
| 20 degrees | +0.986 | +0.581 |
| 30 degrees | +0.985 | **+0.391** |

The highest score I got between two genuinely different people was **+0.214**. So without
alignment, tilting my head gets me most of the way to being a stranger to my own face.
Brightness and resolution barely move the embedding at all, so alignment is specifically
fixing pose and nothing else.

### 2. I measured the threshold instead of copying one

16 photos of 2 people, which gives 56 same person pairs and 64 different person pairs:

```
same person        min +0.430   mean +0.655   max +0.863
different people   min +0.090   mean +0.160   max +0.235
```

Nothing at all lands between 0.235 and 0.430, so every threshold from 0.24 to 0.42 gets
every pair right. I picked **0.33** because it is the middle of that gap, and the edges are
exactly where the nearest mistake of each kind sits.

The part that surprised me is that the **0.5** I had been using before I measured refuses 4
of the 56 same person pairs. That is about 7% of the time telling a correctly enrolled
person they are Unknown, silently, with nothing in the output to say anything is wrong.

### 3. I was wrong about the bottleneck

One frame with one face:

```
detect (model)   115.2 ms   63%
embed (model)     64.4 ms   35%
everything else    4.1 ms    2%
total            183.8 ms   5.4 fps
```

I had assumed embedding was the slow part, and said so in an earlier commit. It is only the
slow part when there are several people in frame, because embedding runs per face while
detection runs per frame. With one face, detection costs nearly twice as much.

The other useful number is that the two models are 98% of the frame, so optimising any of
the code I wrote myself would have been a waste of time.

(These move around a bit on a different machine or a warm laptop. On a fresh clone I get
somewhere between 150 and 185ms.)

### 4. Tracking is a 4.3x win once there is a crowd

Since embedding runs per face, a room full of people who are barely moving spends almost all
of its time re-answering a question it already answered. Boxes that overlap a lot between
one frame and the next are the same person, so each track keeps its name, and ArcFace only
runs for a face that is new or has gone stale.

6 faces, 20 frames:

| | no tracking | tracking |
|---|---|---|
| people sitting still | 1.82 fps | **7.91 fps** |
| faces drifting a few pixels per frame | 1.89 fps | **8.27 fps** |

The 125ms that is left is detection, which tracking cannot help with. With a single face the
win is small, because detection already dominates that frame. This is a fix for crowds.

### 5. INT8 is cheap here, which is a good sign for the board

An FPGA DPU is integer hardware, so running these models on one means quantising them first.
That is the same question the board asks, so I asked it somewhere easier:

| | float32 | int8 |
|---|---|---|
| model size | 174 MB | 44 MB |
| embed time | 66.2 ms | 39.2 ms |
| highest different-person score | 0.235 | 0.237 |
| lowest same-person score | 0.430 | 0.428 |

The embeddings move by a mean cosine of **0.9935**, so barely at all, and 0.33 still
separates the two groups cleanly. It is 75% smaller and 41% faster.

Being honest about what this does not prove: this is dynamic quantisation, where only the
weights get quantised ahead of time. A real DPU needs static quantisation with a calibration
set, on both models rather than one. So this is the easier version and the drift here is a
lower bound. It answers "is this obviously going to fail", not "will this work".

## Moving this to the Kria K26

Written up properly in [docs/kria-mapping.md](docs/kria-mapping.md).

The short version is that most of it does not change. The pipeline order, the gallery,
matching, enrolment, the dashboard and alignment are all plain numpy, and none of them know
what ran the model. The letterboxing, decoding and NMS do not change either, because a DPU
hands back raw tensors and that code is needed no matter what runs the network.

What does change is how the model is loaded and run, which is the two small interfaces in
`src/facepipe/backends/`, plus static INT8 quantisation of both models, the tensor layout
(DPUs are usually NHWC and these graphs are NCHW), and possibly the threshold, which would
have to be measured again on the board if quantisation moves the embeddings.

There is a deliberately unimplemented FPGA backend in
[`src/facepipe/backends/vitis.py`](src/facepipe/backends/vitis.py). It does not work and is
not supposed to. It is there so that "the backend is swappable" is something you can check
instead of something I claim, and so the notes about what a port needs sit next to the code
they affect.

What I would expect to actually go wrong: any operation the compiler cannot map to the DPU
gets left on the ARM cores, so a model that compiles cleanly can still be slow in the middle.
And if inference gets much faster, the preprocessing that is 2% here does not shrink, on
cores that are slower than my laptop.

## Bugs I hit

There were four, and they all turned out to be the same bug.

1. **Black camera frames.** My first webcam test found 0 faces and I assumed my decoding was
   broken. The frame was every pixel 0, because macOS cameras return black frames while they
   warm up and `read()` reports success the entire time.
2. **An empty gallery is falsy.** `Pipeline` did `gallery or Gallery()`, and since `Gallery`
   defines `__len__`, an empty one counts as false. So it threw away the gallery I passed in
   and made a new one pointing somewhere else. Matching still worked fine. The enrolments
   were just going to the wrong place, and I only caught it because a reload came back empty.
3. **An abstraction that did nothing.** My backend interface imported `Detection` from
   `scrfd.py`, so importing the interface imported onnxruntime, and the whole thing was
   decorative. If I had written it and moved on it would have looked correct in a review and
   been worth nothing.
4. **An import that loaded 190MB.** `server.py` ended with `app = create_app()`, so importing
   it loaded both models, and on a fresh clone the tests could not even be collected. I found
   it by cloning my own repo into a temp folder, which I should have been doing from the
   start.

Not one of them raised an error. That is the thing this project taught me. In a pipeline like
this almost nothing fails loudly. Skip the RGB conversion and detection gets worse. Skip
alignment and recognition gets worse. Guess the threshold and you turn away 7% of real
people. All of those look exactly like "the model is not very good" unless you go and measure
them.

## What this does not do

- **No liveness detection.** Hold a printed photo of an enrolled person up to the camera and
  this will happily let them in. That is the first thing to fix for anything on a real door
  (issue #12).
- **The threshold comes from a small test set.** 2 people, one webcam, one room, and nobody
  in it looks alike, so 0.235 for the highest different-person score is optimistic. For a
  door I would move the threshold up inside the safe band, because refusing a real person is
  much better than letting in a stranger.
- **5.4 fps with one face.** Tracking does not help much there, because detection dominates.
- **The tracker is simple.** Greedy overlap matching, no motion model. Fine for faces at a
  door, but it would lose people moving fast or crossing behind each other.
- **MJPEG uses a lot of bandwidth.** Does not matter on localhost, would matter over a
  network.
- **One camera, localhost only** (issue #14).
- Enrolment uses the largest face in frame, so it assumes the person enrolling is the closest
  one to the camera.

## About the face photos

`data/faces/` has 16 photos of two people, me and a friend. We both agreed to them being in a
public repo. They are there because measuring the threshold needs real same person and
different person pairs from the same camera, and stock photos would not have given me an
honest answer.

Enrolled embeddings live in `data/gallery/` and are gitignored. They are biometric data, and
they are also just runtime state rather than source code.

One thing worth writing down. The very first frame I captured for this project had a stranger
sitting at a table behind me, and the detector found her face at 0.71 confidence in a box 30
by 39 pixels. She had no idea. I retook the photo instead of committing that one. That is the
privacy problem with this entire technology turning up uninvited on day one, and if this ends
up on a door in the SCE office then consent, how long embeddings are kept, and who is allowed
to query the gallery all need answers before accuracy does.

## How I used AI

The brief asked us to use AI, so here is an honest account of how I did.

I used Claude a lot, mostly to move faster on things I could not have written from scratch in
three days: the SCRFD decoding, the alignment maths, and the scaffolding around FastAPI and
the scripts.

What I did not hand over was the deciding and the checking. Every measurement in this README
is one I ran, and several of them exist because I did not want to just accept a claim. The
alignment comparison, the normalisation experiment, the threshold sweep and the detector size
test are all there because "this matters" and knowing how much it matters are two different
things. Two of the four bugs above turned up because I went and checked something I had been
told was fine. The gallery bug appeared when I reloaded from disk to see if saving actually
worked, and the 190MB import appeared when I cloned my own repo to see what was really in it.

The issues in this repo were written by hand before I wrote any code, and the commit messages
say what I understood at the time. Where I turned out to be wrong, mainly about the
bottleneck, the history says so instead of being cleaned up afterwards.

## How the work was planned

All of it is broken down in [the issues](../../issues), grouped into Detection, Recognition,
and Hardening and Measurement. I wrote them before starting so that I had a plan, and I
commented on them as I went whenever something surprised me.

Three issues are still open on purpose and tagged `out-of-scope`. They are the things this
project does not do, and I would rather have them written down than quietly missing.
