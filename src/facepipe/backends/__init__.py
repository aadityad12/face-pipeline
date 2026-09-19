"""Inference backends: the only part of the project tied to specific hardware."""

from facepipe.backends.base import Detection, FaceDetector, FaceEmbedder

__all__ = ["Detection", "FaceDetector", "FaceEmbedder"]
