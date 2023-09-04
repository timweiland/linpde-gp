import numpy as np
from probnum import linops
from scipy.linalg import cho_factor, cho_solve


class DenseCholeskySolverLinearOperator(linops.LinearOperator):
    def __init__(self, linop: linops.LinearOperator):
        if not linop.is_symmetric or not linop.is_positive_definite:
            raise ValueError(
                "CholeskySolverLinop can only be applied to symmetric "
                "positive definite linear operators."
            )
        self._linop = linop
        self._cho = cho_factor(linop.todense())
        super().__init__(
            shape=linop.shape,
            dtype=linop.dtype,
        )

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return cho_solve(self._cho, x)

    def _transpose(self) -> linops.LinearOperator:
        return self
