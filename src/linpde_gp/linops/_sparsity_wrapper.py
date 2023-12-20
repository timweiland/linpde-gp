import numpy as np
import probnum as pn
from probnum.typing import DTypeLike, ShapeLike
import torch


class SparsityWrapper(pn.linops.LinearOperator):
    """
    A wrapper class that leverages zero sparsity in the input vectors
    to accelerate matrix-vector products with an inner linear operator.

    Args:
        linop: The inner linear operator.
    """

    def __init__(self, linop: pn.linops.LinearOperator):
        self._inner_linop = linop
        self._zero_linop = pn.linops.Zero(linop.shape, linop.dtype)
        super().__init__(linop.shape, linop.dtype)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        zero_res = self._zero_linop @ x
        if not np.any(x):
            return zero_res
        nonzero_idcs = np.nonzero(x.sum(axis=-2))
        nonzero_idcs = nonzero_idcs[:-1] + (slice(None),) + (nonzero_idcs[-1],)

        if x.ndim > 2:
            nonzero_res = self._inner_linop @ x[nonzero_idcs].T
            zero_res[nonzero_idcs] = nonzero_res.T
        else:
            nonzero_res = self._inner_linop @ x[nonzero_idcs]
            zero_res[nonzero_idcs] = nonzero_res

        return zero_res

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        zero_res = self._zero_linop @ x
        if not torch.any(x):
            return zero_res
        nonzero_idcs = torch.nonzero(x.sum(axis=-2), as_tuple=True)
        nonzero_idcs = nonzero_idcs[:-1] + (slice(None),) + (nonzero_idcs[-1],)

        if x.ndim > 2:
            nonzero_res = self._inner_linop @ x[nonzero_idcs].T
            zero_res[nonzero_idcs] = nonzero_res.T
        else:
            nonzero_res = self._inner_linop @ x[nonzero_idcs]
            zero_res[nonzero_idcs] = nonzero_res

        return zero_res
