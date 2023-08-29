import numpy as np
from probnum.linops import LinearOperator
from probnum.typing import DTypeLike, ShapeLike
from pykeops.numpy import LazyTensor


class KeOpsLinearOperator(LinearOperator):
    def __init__(self, lazy_tensor: LazyTensor, dense_fallback: callable = None):
        self._lazy_tensor = lazy_tensor
        self._dense_fallback = dense_fallback
        super().__init__(lazy_tensor.shape, lazy_tensor.dtype)

    @property
    def lazy_tensor(self) -> LazyTensor:
        return self._lazy_tensor

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return self._lazy_tensor @ x

    def _transpose(self) -> LinearOperator:
        return KeOpsLinearOperator(self.lazy_tensor.T)

    def _todense(self) -> np.ndarray:
        if self._dense_fallback is not None:
            return self._dense_fallback()
        return self._lazy_tensor @ np.eye(self.shape[1], dtype=self.dtype)