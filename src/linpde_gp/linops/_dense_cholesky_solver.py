import numpy as np
from probnum import linops
from scipy.linalg import cho_factor, cho_solve
from probnum.linops._vectorize import vectorize_matmat
import functools
import torch


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

    @functools.cached_property
    def cho_torch(self):
        return torch.linalg.cholesky(
            torch.tensor(self._linop.todense()).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    @vectorize_matmat(method=True)
    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return cho_solve(self._cho, x)

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(x, self.cho_torch)

    def _transpose(self) -> linops.LinearOperator:
        return self

    @functools.cached_property
    def raw_factor(self) -> np.ndarray:
        if self._cho[1]:
            # Lower triangular factor
            return np.tril(self._cho[0])
        return np.triu(self._cho[0]).T