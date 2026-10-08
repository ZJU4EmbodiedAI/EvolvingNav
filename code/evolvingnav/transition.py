"""Row-normalized state transitions and fixed-target identity dynamics."""

from __future__ import annotations

import numpy as np


class IdentityTransition:
    dynamic = False

    def matrix(self, states: list[int], elapsed_s: float) -> np.ndarray:
        return np.eye(len(states), dtype=float)


class MatrixTransition:
    """A supplied row-stochastic transition, useful for deterministic integration tests."""

    dynamic = True

    def __init__(self, matrix: np.ndarray) -> None:
        self.values = np.asarray(matrix, dtype=float)
        if self.values.ndim != 2 or self.values.shape[0] != self.values.shape[1]:
            raise ValueError("transition must be square")
        if np.any(self.values < 0) or not np.allclose(self.values.sum(axis=1), 1):
            raise ValueError("transition rows must sum to one")

    def matrix(self, states: list[int], elapsed_s: float) -> np.ndarray:
        if elapsed_s < 0:
            raise ValueError("time cannot go backwards")
        if len(states) != len(self.values):
            raise ValueError("transition state count mismatch")
        return np.eye(len(states)) if elapsed_s == 0 else self.values
