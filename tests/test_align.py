"""Tests for the alignment maths.

Alignment either works or quietly degrades recognition, and the failure looks identical to
a bad threshold or a bad model. So the arithmetic gets checked against known answers here,
and the actual crops get looked at by eye (scripts/compare_faces.py --dump).
"""

import numpy as np

from facepipe.align import ARCFACE_TEMPLATE, similarity_transform
from facepipe.arcface import cosine_similarity, l2_normalize


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def rotation_matrix(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )


def test_identity_when_points_already_match():
    matrix = similarity_transform(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
    np.testing.assert_allclose(apply(matrix, ARCFACE_TEMPLATE), ARCFACE_TEMPLATE, atol=1e-4)


def test_recovers_a_known_rotation_scale_and_shift():
    """A tilted, distant, off-centre face should land back on the template.

    This is the whole job of alignment, so it is the test that matters: take the template,
    rotate it 20 degrees, halve it, shift it, and check the solver puts it back.
    """
    moved = 0.5 * (ARCFACE_TEMPLATE @ rotation_matrix(20).T) + np.array([300.0, 120.0])
    matrix = similarity_transform(moved, ARCFACE_TEMPLATE)
    np.testing.assert_allclose(apply(matrix, moved), ARCFACE_TEMPLATE, atol=1e-3)


def test_does_not_mirror_the_face():
    """A reflection would fit the points but swap left and right. It must not be used."""
    mirrored = ARCFACE_TEMPLATE.copy()
    mirrored[:, 0] = 112 - mirrored[:, 0]
    matrix = similarity_transform(mirrored, ARCFACE_TEMPLATE)
    # A pure rotation+scale has positive determinant; a reflection would be negative.
    assert np.linalg.det(matrix[:, :2]) > 0


def test_handles_imperfect_landmarks():
    """Real faces never match the template exactly, so it has to be a best fit."""
    rng = np.random.default_rng(0)
    noisy = ARCFACE_TEMPLATE + rng.normal(0, 1.5, ARCFACE_TEMPLATE.shape)
    matrix = similarity_transform(noisy, ARCFACE_TEMPLATE)
    residual = np.abs(apply(matrix, noisy) - ARCFACE_TEMPLATE).max()
    assert residual < 5.0  # close, but not exact, which is expected


def test_l2_normalize_gives_unit_length():
    vector = np.array([3.0, 4.0])
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(vector)), 1.0)


def test_cosine_ignores_magnitude():
    """Two embeddings pointing the same way are identical however long they are."""
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a * 10) == 1.0
    assert cosine_similarity(a, -a) == -1.0
