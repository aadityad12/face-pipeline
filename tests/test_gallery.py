"""Tests for enrolment and matching.

These use made-up embeddings rather than real faces, because what is being tested is the
decision logic: does it pick the closest person, does it refuse when nobody is close
enough, and does it survive a restart. Whether real faces actually land where they should
is a different question, answered by measurement in #11.
"""

import numpy as np
import pytest

from facepipe.gallery import Gallery


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def embedding(index: int, dims: int = 512) -> np.ndarray:
    """A unit vector pointing along one axis, so different indices are unrelated."""
    vector = np.zeros(dims, dtype=np.float32)
    vector[index] = 1.0
    return vector


def nudged(index: int, toward: int, amount: float = 0.3) -> np.ndarray:
    """Close to `index` but not identical, like a second photo of the same person."""
    vector = embedding(index) + amount * embedding(toward)
    return vector / np.linalg.norm(vector)


def at_cosine(index: int, away_from: int, cosine: float) -> np.ndarray:
    """A unit vector at exactly `cosine` from `index`, for testing threshold edges."""
    return cosine * embedding(index) + np.sqrt(1 - cosine**2) * embedding(away_from)


def test_matches_the_closest_person(tmp_path):
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))
    gallery.add("someone else", embedding(1))

    result = gallery.match(nudged(0, 5))
    assert result.name == "aaditya"
    assert result.is_known


def test_returns_unknown_when_nobody_is_close_enough(tmp_path):
    """The safety property: a stranger must not be given the nearest name."""
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))

    result = gallery.match(embedding(7))  # unrelated direction, cosine 0
    assert result.name is None
    assert not result.is_known
    assert result.score < 0.5


def test_unknown_still_reports_the_score(tmp_path):
    """Needed for #11: the near misses are what a threshold sweep is made of.

    A face at 0.20 is below the measured threshold, but the score still comes back,
    because a sweep is built out of exactly these near misses.

    This test originally used 0.35, which was a near miss against the provisional 0.5 and
    became a match once #11 measured the threshold at 0.33. Using a value that close to
    the line made the test about the threshold rather than about reporting the score, so
    it now passes the threshold in explicitly.
    """
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))

    result = gallery.match(at_cosine(0, 1, 0.20), threshold=0.33)
    assert not result.is_known
    assert result.score == pytest.approx(0.20, abs=1e-5)


def test_threshold_is_the_only_thing_deciding_known_from_unknown(tmp_path):
    """The same face is known or unknown depending purely on where the line is set.

    This is why #11 exists: that number is the whole security property, so it should be
    measured rather than picked.
    """
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))
    face = at_cosine(0, 1, 0.45)

    assert not gallery.match(face, threshold=0.5).is_known
    assert gallery.match(face, threshold=0.4).is_known


def test_empty_gallery_is_not_a_crash(tmp_path):
    result = Gallery(tmp_path).match(embedding(0))
    assert result.name is None


def test_several_embeddings_per_person(tmp_path):
    """A person enrolled from several angles should match on the closest one."""
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))
    gallery.add("aaditya", embedding(1))  # a very different pose

    assert gallery.match(nudged(1, 4)).name == "aaditya"
    assert gallery.people() == {"aaditya": 2}


def test_survives_a_restart(tmp_path):
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))
    gallery.add("friend", embedding(1))
    gallery.save()

    reloaded = Gallery(tmp_path)
    assert reloaded.people() == {"aaditya": 1, "friend": 1}
    assert reloaded.match(nudged(1, 6)).name == "friend"


def test_removing_a_person(tmp_path):
    gallery = Gallery(tmp_path)
    gallery.add("aaditya", embedding(0))
    gallery.add("friend", embedding(1))

    assert gallery.remove("friend") == 1
    assert gallery.people() == {"aaditya": 1}
    assert not gallery.match(embedding(1)).is_known


def test_rejects_a_nameless_person(tmp_path):
    with pytest.raises(ValueError):
        Gallery(tmp_path).add("   ", embedding(0))


def test_rejects_a_wrong_sized_embedding(tmp_path):
    with pytest.raises(ValueError):
        Gallery(tmp_path).add("aaditya", unit(1.0, 0.0, 0.0))


def test_pipeline_keeps_the_gallery_it_was_given(tmp_path):
    """An empty Gallery is falsy because it defines __len__.

    Pipeline used to do `gallery or Gallery()`, which quietly threw away an empty gallery
    and built a default one pointing somewhere else. Nothing raised; enrolments just went
    to the wrong directory.
    """
    from facepipe.pipeline import Pipeline

    given = Gallery(tmp_path)
    assert not given  # this is the trap

    pipeline = Pipeline(detector=object(), embedder=object(), gallery=given)
    assert pipeline.gallery is given
