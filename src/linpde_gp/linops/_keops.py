import functools
import numpy as np
from probnum.linops import LinearOperator
from probnum.typing import DTypeLike, ShapeLike
from pykeops.numpy import LazyTensor
from pykeops.torch import LazyTensor as LazyTensor_Torch
import torch


class KeOpsLinearOperator(LinearOperator):
    def __init__(
        self,
        lazy_tensor: LazyTensor,
        lazy_tensor_torch: LazyTensor_Torch = None,
        dense_fallback: callable = None,
    ):
        self._lazy_tensor = lazy_tensor
        self._lazy_tensor_torch = lazy_tensor_torch
        self._dense_fallback = dense_fallback
        super().__init__(lazy_tensor.shape, lazy_tensor.dtype)

    @property
    def lazy_tensor(self) -> LazyTensor:
        return self._lazy_tensor

    @property
    def lazy_tensor_torch(self) -> LazyTensor_Torch:
        return self._lazy_tensor_torch

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x)
        if self._lazy_tensor.shape[0] == 1 or self._lazy_tensor.shape[1] == 1:
            return self.todense() @ x
        return self._lazy_tensor @ x

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self._lazy_tensor_torch is None:
            return super()._matmul_torch(x)
        x = x.contiguous()
        if (
            self._lazy_tensor_torch.shape[0] == 1
            or self._lazy_tensor_torch.shape[1] == 1
        ) or (self.shape[0] <= 128 and self.shape[1] <= 128):
            return self._todense_torch @ x
        return self._lazy_tensor_torch @ x

    def _transpose(self) -> LinearOperator:
        return KeOpsLinearOperator(
            self.lazy_tensor.T,
            lazy_tensor_torch=self.lazy_tensor_torch.T
            if self.lazy_tensor_torch is not None
            else None,
            dense_fallback=lambda: self._dense_fallback().T
            if self._dense_fallback is not None
            else None,
        )

    def _todense(self) -> np.ndarray:
        if self._dense_fallback is not None:
            return self._dense_fallback()
        return self._lazy_tensor @ np.eye(self.shape[1], dtype=self.dtype)
    
    @functools.cached_property
    def _todense_torch(self) -> torch.Tensor:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.as_tensor(self.todense()).to(device)