"""The seam between "what the pipeline does" and "what runs the models".

Everything above this line - tracking, matching, the gallery, the dashboard - is plain
numpy and Python and does not care what executed the neural network. Everything below it
is specific to a piece of hardware and a runtime.

Two protocols, deliberately small:

    FaceDetector.detect(frame)    -> boxes + 5 landmarks, in frame coordinates
    FaceEmbedder.embed(aligned)   -> a 512-number unit vector

Structural typing (typing.Protocol), so a backend does not import or subclass anything
from here. It just has to have the right methods, which is what makes an alternative
implementation genuinely independent rather than a subclass of this one (issue #8).

Where the line falls, and why it is drawn here:

    letterboxing, normalising, decoding, NMS, alignment   -> below the line, per backend
    tracking, matching, enrolment, drawing, streaming     -> above it, shared

Preprocessing sits below the line because it is defined by what the model wants (the
exact normalisation constants differ between SCRFD and ArcFace already), and decoding
because a different runtime can hand back its outputs in a different layout. That is
also precisely the code an FPGA port has to rewrite, so putting it here means the port
touches one directory instead of the whole pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from facepipe.types import Detection

__all__ = ["Detection", "FaceDetector", "FaceEmbedder"]


@runtime_checkable
class FaceDetector(Protocol):
    """Finds faces in a frame."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """BGR frame of any size -> one Detection per face, in that frame's coordinates.

        The caller is owed frame coordinates, not model coordinates. Whatever resizing or
        padding a backend does internally is its own business to undo.
        """
        ...


@runtime_checkable
class FaceEmbedder(Protocol):
    """Turns one aligned face into an embedding."""

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """A 112x112 aligned BGR crop -> a 512-number vector of unit length.

        Unit length is part of the contract, not an optimisation: the matching code takes
        a dot product and calls it a cosine, which is only true if both sides are
        normalised.
        """
        ...
