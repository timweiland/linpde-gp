import numpy as np
from probnum import linops
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from probnum.linops._vectorize import vectorize_matmat
import functools
import torch

from ._outer_product import OuterProductMatrix


class InverseTriangularMatrix(linops.LinearOperator):
    def __init__(self, U: np.ndarray, lower=True):
        self._U = U
        self._U_torch = torch.from_numpy(U).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._lower = lower
        super().__init__(U.shape, U.dtype)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return solve_triangular(self._U, x, lower=self._lower)

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self._U_torch, x, upper=not self._lower)

    def _transpose(self) -> linops.LinearOperator:
        return InverseTriangularMatrix(self._U.T, lower=not self._lower)


class DenseCholeskySolverLinearOperator(OuterProductMatrix):
    def __init__(self, linop: linops.LinearOperator):
        if not linop.is_symmetric or not linop.is_positive_definite:
            raise ValueError(
                "CholeskySolverLinop can only be applied to symmetric "
                "positive definite linear operators."
            )
        self._linop = linop
        self._cho = cho_factor(linop.todense(), False)
        self._factor_linop = InverseTriangularMatrix(np.triu(self._cho[0]), lower=False)

        super().__init__(U=self._cho[0])

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

    @property
    def U(self) -> InverseTriangularMatrix:
        return self._factor_linop
