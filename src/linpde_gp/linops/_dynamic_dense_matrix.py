import probnum as pn
from probnum.typing import ShapeLike, DTypeLike
import torch
import numpy as np


class DynamicDenseMatrix(pn.linops.LinearOperator):
    """
    A matrix that can be dynamically filled with columns.

    The shape is fixed at initialization, and the columns that have not been filled
    yet are zero.

    Useful e.g. for efficient implementation of outer products X @ X.T.

    Args:
        shape: The shape of the matrix.
        dtype: The data type of the matrix.
    """

    def __init__(self, shape: ShapeLike, dtype: DTypeLike):
        self._data = None
        self._num_cols = 0
        super().__init__(shape, dtype)

    @property
    def data(self) -> torch.Tensor | None:
        return self._data[:, : self._num_cols]

    def set_data(self, new_data: torch.Tensor):
        new_data_torch = (
            new_data
            if isinstance(new_data, torch.Tensor)
            else torch.tensor(new_data, device=self._data.device)
        ).to(self._data.device)
        self._data[:, : new_data.shape[1]] = new_data_torch
        self._data[:, new_data.shape[1] :] = torch.zeros_like(
            self._data[:, new_data.shape[1] :]
        )
        self._num_cols = new_data.shape[1]

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._data is None:
            return pn.linops.Zero(self.shape, self.dtype) @ x
        return self._data.cpu().numpy() @ x

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self._data is None:
            return pn.linops.Zero(self.shape, self.dtype) @ x
        return self._data @ x

    def append_column(self, x: np.ndarray | torch.Tensor):
        if self._data is None:
            device = "cpu" if isinstance(x, np.ndarray) else x.device
            self._data = torch.zeros(self.shape, device=device, dtype=torch.float64)
        x_torch = x if isinstance(x, torch.Tensor) else torch.tensor(x, device="cpu")
        self._data[:, self._num_cols] = x_torch
        self._num_cols += 1

    def _transpose(self) -> pn.linops.LinearOperator:
        if self._num_cols == 0:
            return pn.linops.Zero((self.shape[1], self.shape[0]), self.dtype)
        return pn.linops.Matrix(self._data.T)
