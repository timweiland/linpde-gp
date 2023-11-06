import numpy as np
from probnum import linops
from scipy.linalg import solve_triangular
import torch
import functools


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
    
    @functools.cached_property
    def torch_raw_factor_transposed(self) -> torch.Tensor:
        return torch.tensor(self._sandwiched_linop.raw_factor.T).to("cuda" if torch.cuda.is_available() else "cpu")

    def _get_column(self, idx: int, device) -> np.ndarray:
        from linpde_gp.linops import (
            DenseCholeskySolverLinearOperator,
            RankFactorizedMatrix,
        )

        if isinstance(self._sandwiched_linop, DenseCholeskySolverLinearOperator):
            e = torch.zeros(self._sandwiched_linop.shape[1], dtype=torch.float64).to(device)
            e[idx] = 1.0
            return torch.linalg.solve_triangular(self.torch_raw_factor_transposed, e.reshape((e.size(dim=0), 1)), upper=True).reshape(-1)
        elif isinstance(self._sandwiched_linop, RankFactorizedMatrix):
            return self._sandwiched_linop.U[:, idx]
        return NotImplemented

    def _diagonal(self) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diag = torch.zeros(self._crosscov.shape[0], dtype=torch.float64).to(device)
        for i in range(self._sandwiched_linop.shape[1]):
            v = self._crosscov @ self._get_column(i, device)
            diag += v**2
        return diag.cpu().numpy()
