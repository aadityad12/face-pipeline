# What would change on the Kria K26

Notes on moving this from a laptop CPU to the board, written from what the laptop build
actually taught me rather than from the Vitis AI documentation. The code side of this is
`src/facepipe/backends/` (issue #8); the unimplemented FPGA backend there says the same
things next to the functions they affect.

## What does not change

Most of it, which was the point of building it this way.

The pipeline order, the gallery, cosine matching, enrolment, the Unknown case, the
dashboard, the threshold decision from #11, and alignment (a `warpAffine` on the CPU
either way) are all plain numpy and Python. None of them know what executed the network.

Letterboxing, the anchor decode, NMS and the normalisation constants do not change either
in *logic*, and this is the part I did not appreciate before starting. A DPU hands you raw
tensors in and raw tensors out. There is no `FaceAnalysis.get(image)` that returns faces.
All of that is still my code, which is the reason this project runs the ONNX graphs
directly instead of using the insightface wrapper
([ADR 0001](decisions/0001-onnxruntime-instead-of-insightface.md)). Had I used the wrapper,
the port would be a rewrite.

## What does change

**How the model is loaded and run.** One class each for the detector and the embedder,
satisfying the two protocols in `backends/base.py`. That is the whole surface.

**Quantisation.** The DPU is integer hardware, so both models have to be quantised to INT8
with a calibration set. #10 measured the easy version of this on the laptop: dynamic INT8
on ArcFace was 41% faster, 75% smaller, and moved the embeddings by a mean cosine of
0.9935, leaving the threshold band essentially unchanged. That is encouraging but it is a
lower bound, because a DPU needs *static* quantisation with calibration data, and both
models rather than one.

**The threshold, possibly.** If quantisation moves the embeddings far enough, the 0.33
from #11 is no longer the right number. The sweep has to be re-run on the board with
board-quantised models. The script does not care what produced the embeddings, so this is
re-running `scripts/sweep_threshold.py`, not rewriting it.

**Tensor layout.** The DPU is typically NHWC and these ONNX graphs are NCHW, so the
transpose at the end of preprocessing changes.

## What I would expect to go wrong

**Unsupported operations.** Whatever the compiler cannot map to the DPU gets left on the
ARM cores. A graph that compiles cleanly can still end up with a slow section in the
middle, so the first thing worth doing is checking the compiler's partition report before
assuming any speedup at all. This is the one I would budget the most time for.

**Preprocessing becoming the bottleneck.** On the laptop the two models are 98% of the
frame and everything I wrote is the other 2% (#10). If the DPU makes inference 10x faster,
that 2% does not shrink, and the ARM cores on a K26 are slower than this laptop's CPU. The
letterboxing and colour conversion could become the thing to optimise, which is a strange
inversion but follows directly from the measurements.

**Camera geometry deciding the detector input size.** #10 found 480x480 is 1.7x faster than
640x640 and lost nothing on my test data, while 320 silently dropped distant faces. The
right size depends on how far people stand from the door camera, which is a question about
the physical install, not about the model.

## What the laptop build cannot tell you

Whether the DPU's INT8 behaves like onnxruntime's INT8, thermals and sustained throughput
on the board, how the camera pipeline behaves without OpenCV's desktop drivers, and
anything about running detection and embedding concurrently on the fabric.
