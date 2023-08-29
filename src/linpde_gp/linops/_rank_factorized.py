import probnum as pn
from probnum.typing import DTypeLike, ShapeLike
import numpy as np


class RankFactorizedMatrix(pn.linops.LinearOperator):
    def __init__(
        self, U: np.ndarray | None, shape: ShapeLike = None, dtype: DTypeLike = None
    ):
        if U is None:
            self._U = None
            if shape is None or dtype is None:
                raise ValueError(
                    "When initializing a trivial RankOneFactorizedMatrix, you need to specify the shape and dtype."
                )
            super().__init__(shape, dtype)
        else:
            self._U = U
            super().__init__(U.shape, U.dtype)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._U is None:
            return np.zeros_like(x)
        return self._U @ (self._U.T @ x)

    def append_factor(self, factor: np.ndarray):
        if self._U is None:
            self._U = factor.reshape(-1, 1)
        else:
            self._U = np.concatenate((self._U, factor.reshape(-1, 1)), axis=-1)


class LowRankProduct(pn.linops.LinearOperator):
    def __init__(
        self,
        U: np.ndarray | None,
        V: np.ndarray | None,
        shape: ShapeLike = None,
        dtype: DTypeLike = None,
    ):
        if U is None and V is None:
            self._U = None
            self._V = None
            if shape is None or dtype is None:
                raise ValueError(
                    "When initializing a trivial RankOneFactorizedMatrix, you need to specify the shape and dtype."
                )
            super().__init__(shape, dtype)
        elif U is not None and V is not None:
            self._U = U
            self._V = V
            super().__init__((U.shape[0], V.shape[0]), U.dtype)
        else:
            raise ValueError(
                "U and V must both be either None or not None at the same time."
            )

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._U is None or self._V is None:
            return np.zeros_like(x)
        return self._U @ (self._V.T @ x)

    def append_factors(self, U_factor: np.ndarray, V_factor: np.ndarray):
        if self._U is None or self._V is None:
            self._U = U_factor.reshape(-1, 1)
            self._V = V_factor.reshape(-1, 1)
        else:
            self._U = np.concatenate((self._U, U_factor.reshape(-1, 1)), axis=-1)
            self._V = np.concatenate((self._V, V_factor.reshape(-1, 1)), axis=-1)
