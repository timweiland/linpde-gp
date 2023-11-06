import probnum as pn
from probnum.typing import DTypeLike, ShapeLike
import numpy as np
import torch

class DynamicMatrix:
    def __init__(self, row_size, initial_capacity=1430, device='cuda'):
        self.data = torch.empty((row_size, initial_capacity), device=device, dtype=torch.float64)
        self.current_size = 0
        self.capacity = initial_capacity

    def append(self, col):
        # Check if there's space left
        if self.current_size == self.capacity:
            self._resize(2 * self.capacity)
        
        self.data[:, self.current_size] = col
        self.current_size += 1

    def _resize(self, new_capacity):
        new_data = torch.empty((self.data.size(0), new_capacity), device=self.data.device, dtype=self.data.dtype)
        new_data[:, :self.current_size] = self.data
        self.data = new_data
        self.capacity = new_capacity

    def get_data(self):
        return self.data[:, :self.current_size]

class RankFactorizedMatrix(pn.linops.LinearOperator):
    def __init__(
        self, U: np.ndarray | None, shape: ShapeLike = None, dtype: DTypeLike = None, initial_capacity=1430,
    ):
        self._num_factors = 0
        self._initial_capacity = initial_capacity
        if U is None:
            self._U = None
            self._U_torch = None
            if shape is None or dtype is None:
                raise ValueError(
                    "When initializing a trivial RankOneFactorizedMatrix, you need to specify the shape and dtype."
                )
            super().__init__(shape, dtype)
        else:
            self._U = U
            self._U_torch = torch.from_numpy(U).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._num_factors = U.shape[1]
            super().__init__(U.shape, U.dtype)

    @property
    def U(self) -> np.ndarray | None:
        return self._U
    
    @property
    def num_factors(self) -> int:
        return self._num_factors

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._U is None:
            return np.zeros_like(x)
        return self._U @ (self._U.T @ x)

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self._U_torch is None:
            return torch.zeros_like(x)
        mat = self._U_torch.get_data()
        return mat @ (mat.T @ x)

    def append_factor(self, factor: np.ndarray):
        if (self._U is None and self._U_torch is None):
            if isinstance(factor, np.ndarray):
                self._U = factor.reshape(-1, 1)
            else:
                # torch.Tensor
                self._U_torch = DynamicMatrix(factor.shape[0], device=factor.device, initial_capacity=self._initial_capacity)
        else:
            if isinstance(factor, np.ndarray):
                self._U = np.concatenate((self._U, factor.reshape(-1, 1)), axis=-1)
            else:
                # torch.Tensor
                self._U_torch.append(factor)
        self._num_factors += 1
    
    def _transpose(self) -> pn.linops.LinearOperator:
        return self


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
            self._U_torch = None
            self._V_torch = None
            if shape is None or dtype is None:
                raise ValueError(
                    "When initializing a trivial RankOneFactorizedMatrix, you need to specify the shape and dtype."
                )
            super().__init__(shape, dtype)
        elif U is not None and V is not None:
            self._U = U
            self._V = V
            self._U_torch = torch.from_numpy(U).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._V_torch = torch.from_numpy(V).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            super().__init__((U.shape[0], V.shape[0]), U.dtype)
        else:
            raise ValueError(
                "U and V must both be either None or not None at the same time."
            )

    @property
    def U(self) -> np.ndarray | None:
        return self._U

    @property
    def V(self) -> np.ndarray | None:
        return self._V

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self._U is None or self._V is None:
            return np.zeros_like(x)
        return self._U @ (self._V.T @ x)

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self._U_torch is None or self._V_torch is None:
            return torch.zeros_like(x)
        return self._U_torch @ (self._V_torch.T @ x)

    def append_factors(self, U_factor: np.ndarray, V_factor: np.ndarray):
        if self._U is None or self._V is None:
            if isinstance(U_factor, np.ndarray):
                self._U = U_factor.reshape(-1, 1)
                self._V = V_factor.reshape(-1, 1)
            else:
                # torch.Tensor
                self._U_torch = U_factor.reshape(-1, 1)
                self._V_torch = V_factor.reshape(-1, 1)
        else:
            if isinstance(U_factor, np.ndarray):
                self._U = np.concatenate((self._U, U_factor.reshape(-1, 1)), axis=-1)
                self._V = np.concatenate((self._V, V_factor.reshape(-1, 1)), axis=-1)
            else:
                # torch.Tensor
                self._U_torch = torch.cat(
                    (self._U_torch, U_factor.reshape(-1, 1)), dim=-1
                )
                self._V_torch = torch.cat(
                    (self._V_torch, V_factor.reshape(-1, 1)), dim=-1
                )
    
    def _transpose(self) -> pn.linops.LinearOperator:
        return LowRankProduct(self._V, self._U)
