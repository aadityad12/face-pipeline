"""Who the system knows, and deciding who it is looking at.

Enrolling is just storing someone's embedding under a name. Nothing is trained, so adding
a person takes a second, and matching is comparing the live embedding against the stored
ones (issue #7).

Two decisions worth stating:

Several embeddings per person, not one. One photo pins a person to one pose and one
lighting, and the measurements in #5 show pose is what moves an embedding most. A live
face at a different angle can then fall below the threshold and read as Unknown.

Matching on the best single embedding, not the average of a person's embeddings.
Averaging several poses lands somewhere between them, which can end up resembling nobody
in particular. Taking the closest match asks the question that is actually being asked:
does this face look like *any* face I have on file for this person?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facepipe.config import DATA_DIR

# Measured, not guessed: see docs/results.md and issue #11.
#
# 16 photos of 2 people gave 56 same-person pairs and 64 different-person pairs. Every
# threshold from 0.24 to 0.42 separates them with no mistakes of either kind, and 0.33 is
# the middle of that band, so it sits as far as possible from both.
#
# The 0.5 this started as would have refused 4 of the 56 same-person pairs.
MATCH_THRESHOLD = 0.33


@dataclass
class Match:
    """The result of looking one face up."""

    name: str | None  # None means nobody was close enough
    score: float  # best cosine similarity found, whether or not it passed

    @property
    def is_known(self) -> bool:
        return self.name is not None

    @property
    def label(self) -> str:
        return f"{self.name} {self.score:.2f}" if self.name else f"Unknown {self.score:.2f}"


class Gallery:
    """The enrolled people, persisted to disk."""

    def __init__(self, directory: Path = DATA_DIR / "gallery"):
        self.directory = Path(directory)
        self.embeddings = np.zeros((0, 512), dtype=np.float32)
        self.names: list[str] = []
        self.load()

    # -- persistence ---------------------------------------------------------------

    @property
    def _npz_path(self) -> Path:
        return self.directory / "embeddings.npz"

    @property
    def _json_path(self) -> Path:
        return self.directory / "people.json"

    def load(self) -> None:
        if not self._npz_path.exists():
            return
        data = np.load(self._npz_path, allow_pickle=False)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.names = [str(n) for n in data["names"]]

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._npz_path,
            embeddings=self.embeddings,
            names=np.array(self.names, dtype=object).astype("U"),
        )
        # Written alongside the embeddings purely so a human can see who is enrolled
        # without loading numpy. The npz is the source of truth.
        summary = {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "people": {name: self.names.count(name) for name in sorted(set(self.names))},
        }
        self._json_path.write_text(json.dumps(summary, indent=2) + "\n")

    # -- enrolment -----------------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Store one more embedding for `name`."""
        name = name.strip()
        if not name:
            raise ValueError("a person needs a name")
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"expected {self.embeddings.shape[1]}-d embedding, got {vector.shape[1]}"
            )
        self.embeddings = np.vstack([self.embeddings, vector])
        self.names.append(name)

    def remove(self, name: str) -> int:
        """Forget a person entirely. Returns how many embeddings were dropped."""
        keep = [i for i, n in enumerate(self.names) if n != name]
        dropped = len(self.names) - len(keep)
        self.embeddings = self.embeddings[keep]
        self.names = [self.names[i] for i in keep]
        return dropped

    # -- lookup --------------------------------------------------------------------

    def match(self, embedding: np.ndarray, threshold: float = MATCH_THRESHOLD) -> Match:
        """Closest enrolled person, or Unknown if nobody is close enough.

        The threshold is the whole safety property. Without it every face returns the
        closest name, because someone is always closest, so a stranger walking past gets
        confidently labelled as whoever they happen to resemble most. A recognition system
        that cannot say "I don't know" is not usable for a door.
        """
        if len(self.names) == 0:
            return Match(name=None, score=0.0)

        # Everything is unit length already, so the dot product is the cosine.
        scores = self.embeddings @ np.asarray(embedding, dtype=np.float32).reshape(-1)
        best = int(np.argmax(scores))
        best_score = float(scores[best])

        if best_score < threshold:
            return Match(name=None, score=best_score)
        return Match(name=self.names[best], score=best_score)

    # -- introspection -------------------------------------------------------------

    def people(self) -> dict[str, int]:
        """Enrolled names and how many embeddings each has."""
        return {name: self.names.count(name) for name in sorted(set(self.names))}

    def __len__(self) -> int:
        return len(self.names)
