import numpy as np
from probnum import linops
from scipy.linalg import solve_triangular
import torch
import functools

from ._outer_product import OuterProduct


class CrosscovSandwich(linops.LinearOperator):
    def __init__(
        self, crosscov: linops.LinearOperator, sandwiched_linop: OuterProduct
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

    def _diagonal(self, compute_iterative=True) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diag = torch.zeros(self._crosscov.shape[0], dtype=torch.float64).to(device)
        if compute_iterative:
            for i in range(self._sandwiched_linop.U.shape[1]):
                e = torch.zeros(
                    self._sandwiched_linop.U.shape[1], dtype=torch.float64
                ).to(device)
                e[i] = 1.0
                if self._sandwiched_linop.is_symmetric:
                    diag += (self._crosscov @ self._sandwiched_linop.U @ e) ** 2
                else:
                    diag += (self._crosscov @ self._sandwiched_linop.U @ e) * (
                        self._crosscov @ self._sandwiched_linop.V @ e
                    )
        else:
            if self._sandwiched_linop.is_symmetric:
                cols = (
                    self._crosscov
                    @ self._sandwiched_linop.U
                    @ torch.eye(
                        self._sandwiched_linop.U.shape[1], dtype=torch.float64
                    ).to(device)
                )
                diag = torch.sum(cols**2, dim=-1)
            else:
                diag = torch.sum(
                    self._crosscov
                    @ self._sandwiched_linop.U
                    @ torch.eye(
                        self._sandwiched_linop.U.shape[1], dtype=torch.float64
                    ).to(device)
                    * (
                        self._crosscov
                        @ self._sandwiched_linop.V
                        @ torch.eye(
                            self._sandwiched_linop.V.shape[1], dtype=torch.float64
                        ).to(device)
                    ),
                    dim=-1,
                )
        return diag.cpu().numpy()
