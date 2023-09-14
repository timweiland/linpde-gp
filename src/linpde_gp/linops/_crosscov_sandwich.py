import numpy as np
from probnum import linops
from scipy.linalg import solve_triangular
import torch


class CrosscovSandwichLinearOperator(linops.LinearOperator):
    def __init__(
        self, crosscov: linops.LinearOperator, sandwiched_linop: linops.LinearOperator
    ):
        self._crosscov = crosscov
        self._sandwiched_linop = sandwiched_linop
        super().__init__((crosscov.shape[0], crosscov.shape[0]), crosscov.dtype)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return (self._crosscov @ self._sandwiched_linop @ self._crosscov.T) @ x
    
    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return (self._crosscov @ self._sandwiched_linop @ self._crosscov.T) @ x

    def _transpose(self) -> linops.LinearOperator:
        return self

    def _get_column(self, idx: int) -> np.ndarray:
        from linpde_gp.linops import (
            DenseCholeskySolverLinearOperator,
            RankFactorizedMatrix,
        )

        if isinstance(self._sandwiched_linop, DenseCholeskySolverLinearOperator):
            e = np.zeros(self._sandwiched_linop.shape[1])
            e[idx] = 1.0
            return solve_triangular(self._sandwiched_linop.raw_factor.T, e, lower=False)
        elif isinstance(self._sandwiched_linop, RankFactorizedMatrix):
            return self._sandwiched_linop.U[:, idx]
        return NotImplemented

    def _diagonal(self) -> np.ndarray:
        diag = np.zeros(self._crosscov.shape[0])
        for i in range(self._sandwiched_linop.shape[1]):
            v = self._crosscov @ self._get_column(i)
            diag += v**2
        return diag
