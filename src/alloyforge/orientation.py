from __future__ import annotations

import numpy as np
from itertools import permutations, product


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random SO(3) rotation using a normalized quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def validate_rotation_matrix(matrix: np.ndarray, tolerance: float = 1e-8) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("An orientation matrix must have shape (3, 3)")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=tolerance):
        raise ValueError("Orientation matrix must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=tolerance):
        raise ValueError("Orientation matrix determinant must equal +1")


def rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _cubic_symmetry() -> list[np.ndarray]:
    operations = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.5:
                operations.append(matrix)
    return operations


def _hexagonal_symmetry() -> list[np.ndarray]:
    operations = []
    for angle in range(0, 360, 60):
        radians = np.radians(angle)
        operations.append(np.array([
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ]))
        axis = np.array([np.cos(radians / 2), np.sin(radians / 2), 0.0])
        operations.append(2.0 * np.outer(axis, axis) - np.eye(3))
    return operations


CUBIC_SYMMETRY = _cubic_symmetry()
HEXAGONAL_SYMMETRY = _hexagonal_symmetry()


def minimum_misorientation_deg(
    left: np.ndarray, right: np.ndarray, structure: str
) -> float:
    """Return the disorientation angle reduced by the phase's proper rotations."""
    symmetry = (
        CUBIC_SYMMETRY if structure in {"fcc", "bcc"}
        else HEXAGONAL_SYMMETRY if structure == "hcp"
        else [np.eye(3)]
    )
    delta = np.asarray(left) @ np.asarray(right).T
    return min(rotation_angle_deg(operation @ delta) for operation in symmetry)
