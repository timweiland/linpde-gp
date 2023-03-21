import probnum as pn
from probnum.typing import DTypeLike, ShapeLike
import numpy as np


class DiagonalPlusLowRankMatrix(pn.linops.LinearOperator):
    def __init__(self, diagonal: np.ndarray, U: np.ndarray):
        self._diagonal = pn.linops.Scaling(diagonal)
        self._U = U
        self._small_matrix = pn.linops.aslinop(
            np.eye(self._U.shape[1]) + self._U.T @ (self._diagonal.inv() @ self._U)
        )
        self._small_matrix.is_symmetric = True
        self._small_matrix.is_positive_definite = True
        super().__init__(self._diagonal.shape, self._diagonal.dtype)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return self._diagonal @ x + self._U @ (self._U.T @ x)

    def _solve(self, B: np.ndarray) -> np.ndarray:
        return self._diagonal.inv() @ B - self._diagonal.inv() @ (
            self._U @ self._small_matrix.solve(self._U.T @ (self._diagonal.inv() @ B))
        )
