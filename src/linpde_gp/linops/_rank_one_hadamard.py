import probnum as pn
from probnum.linops._linear_operator import LinearOperator
from probnum.typing import ArrayLike
import numpy as np
import torch


class RankOneHadamardProduct(pn.linops.LinearOperator):
    """
    A Hadamard product of a rank-one outer product and a linear operator,
    i.e. (row_factors @ col_factors.T) * linop.

    Args:
        row_factors (ArrayLike): The row factors of the Hadamard product.
        col_factors (ArrayLike): The column factors of the Hadamard product.
        linop (pn.linops.LinearOperator): The linear operator.
    """

    def __init__(
        self,
        row_factors: ArrayLike,
        col_factors: ArrayLike,
        linop: pn.linops.LinearOperator,
    ):
        super().__init__(shape=linop.shape, dtype=linop.dtype)
        self._row_factors = np.asarray(row_factors, dtype=linop.dtype).reshape(
            -1, order="C"
        )
        self._col_factors = np.asarray(col_factors, dtype=linop.dtype).reshape(
            -1, order="C"
        )

        if self._row_factors.size == 1:
            self._row_factors = self._row_factors * np.ones(linop.shape[0])
        if self._col_factors.size == 1:
            self._col_factors = self._col_factors * np.ones(linop.shape[1])

        if self._row_factors.size != linop.shape[0]:
            raise ValueError(
                f"Row factors have shape {self._row_factors.size} but should have "
                f"size {linop.shape[0]}."
            )
        if self._col_factors.size != linop.shape[1]:
            raise ValueError(
                f"Column factors have size {self._col_factors.size} but should have "
                f"size {linop.shape[1]}."
            )
        self._linop = linop

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._row_factors_torch = torch.tensor(self._row_factors).to(device)
        self._col_factors_torch = torch.tensor(self._col_factors).to(device)

    @property
    def row_factors(self) -> ArrayLike:
        return self._row_factors

    @property
    def col_factors(self) -> ArrayLike:
        return self._col_factors

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return self._linop

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return self._row_factors[:, None] * (
            self._linop @ (self._col_factors[:, None] * x)
        )

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self._row_factors_torch[:, None] * (
            self._linop @ (self._col_factors_torch[:, None] * x)
        )

    def _todense(self) -> np.ndarray:
        return np.outer(self._row_factors, self._col_factors) * self._linop.todense()

    def _diagonal(self) -> np.ndarray:
        return (self._row_factors * self._col_factors) * self._linop.diagonal()

    def _transpose(self) -> LinearOperator:
        return RankOneHadamardProduct(
            row_factors=self._col_factors,
            col_factors=self._row_factors,
            linop=self._linop.T,
        )

    def __repr__(self) -> str:
        return f"RankOneHadamardProduct(shape {self._row_factors.shape}, shape {self._col_factors.shape}, {self._linop})"
