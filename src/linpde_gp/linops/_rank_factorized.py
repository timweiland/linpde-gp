import probnum as pn
from probnum.typing import DTypeLike, ShapeLike
import numpy as np

class RankFactorizedMatrix(pn.linops.LinearOperator):
    def __init__(self, U: np.ndarray | None, C: np.ndarray | None, shape: ShapeLike = None, dtype: DTypeLike = None):
        if U is None and C is None:
            self._U = None
            self._C = None
            if shape is None or dtype is None:
                raise ValueError("When initializing a trivial RankOneFactorizedMatrix, you need to specify the shape and dtype.")
            super().__init__(shape, dtype)
        elif U is not None and C is not None:
            self._U = U
            self._C = C
            super().__init__(U.shape, U.dtype)
        else:
            raise ValueError("You need to specify either both U and C or neither of them.")
    
    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._U is None:
            return np.zeros_like(x)
        return self._C * self._U @ (self._U.T @ x)
    
    def append_factor(self, factor: np.ndarray, constant: float):
        if self._U is None:
            self._U = factor.reshape(-1, 1)
            self._C = np.array([constant])
        else:
            self._U = np.concatenate((self._U, factor.reshape(-1, 1)), axis=-1)
            self._C = np.concatenate((self._C, np.array([constant])), axis=-1)