import probnum as pn
from probnum.linops._linear_operator import BinaryOperandType, LinearOperator
from probnum.typing import ShapeLike
import numpy as np
import torch
import functools
import operator

from ._block import BlockMatrix


class OuterProduct(pn.linops.LinearOperator):
    """
    An outer product U @ V.T.

    Args:
        U (pn.linops.LinearOperator): The linear operator U.
        V (pn.linops.LinearOperator | None, optional): The linear operator V. If None, defaults to U.
    """

    def __init__(
        self, U: pn.linops.LinearOperator, V: pn.linops.LinearOperator | None = None
    ):
        self._U = U
        self._V = V if V is not None else U
        super().__init__((U.shape[0], self._V.shape[0]), U.dtype)
        if V is None:
            self.is_symmetric = True
    
    @property
    def shape(self) -> tuple[int, int]:
        # Enable dynamic shape for dynamic outer products
        return (self._U.shape[0], self._V.shape[0])

    @property
    def U(self) -> pn.linops.LinearOperator:
        return self._U

    @property
    def V(self) -> pn.linops.LinearOperator:
        return self._V

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return self._U @ (self._V.T @ x)

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self._U @ (self._V.T @ x)

    def _transpose(self) -> pn.linops.LinearOperator:
        return OuterProduct(self._V, self._U)

    def _diagonal(self) -> np.ndarray:
        if self.is_symmetric:
            return (
                torch.sum(
                    (self._U @ torch.eye(self._U.shape[1], dtype=torch.float64)) ** 2,
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
        mat = self._U @ torch.eye(self._U.shape[1], dtype=torch.float64)
        mat *= self._V @ torch.eye(self._V.shape[1], dtype=torch.float64)
        return torch.sum(mat, dim=-1).cpu().numpy()

    def __add__(self, other: BinaryOperandType) -> LinearOperator:
        if isinstance(other, OuterProduct):
            return SumOuterProduct(self, other)
        return super().__add__(other)


class SumOuterProduct(OuterProduct):
    def __init__(self, *Ls: OuterProduct):
        self._block_U = BlockMatrix([[L.U for L in Ls]])
        self._block_V = BlockMatrix([[L.V for L in Ls]])
        self._Ls = tuple(Ls)

        for L in Ls:
            if L.shape != Ls[0].shape:
                raise ValueError("All matrices must have the same shape.")
        super().__init__(self._block_U)

    @property
    def Ls(self) -> tuple[OuterProduct]:
        return self._Ls

    @property
    def U(self) -> BlockMatrix:
        return self._block_U

    @property
    def V(self) -> BlockMatrix:
        return self._block_V

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(operator.add, (summand @ x for summand in self._Ls))

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return functools.reduce(operator.add, (summand @ x for summand in self._Ls))

    def _transpose(self) -> pn.linops.LinearOperator:
        return SumOuterProduct(*[L.T for L in self._Ls])


class ExtendedOuterProduct(OuterProduct):
    """
    An extended outer product matrix.

    Extends the shape of an outer product U @ V.T by padding both U and V
    with zero rows efficiently.

    Args:
        L (OuterProductMatrix): The input matrix.
        target_shape (tuple): The target shape of the extended matrix.

    Raises:
        ValueError: If the target shape is not square or if it is smaller than the shape of `L`.
    """

    def __init__(self, L: OuterProduct, target_shape: ShapeLike):
        self._L = L
        assert len(target_shape) == 2
        if target_shape[0] != target_shape[1]:
            raise ValueError("Target shape must be square.")
        if target_shape[0] < L.shape[0]:
            raise ValueError("Target shape must be larger than the shape of L.")
        if target_shape == L.shape:
            self._U_extended = L.U
            self._V_extended = L.V
            self._L_extended = L
        else:
            shape_diff = target_shape[0] - L.shape[0]
            L_U = L if isinstance(L, pn.linops.Identity) else L.U
            self._U_extended = BlockMatrix(
                [[L_U], [pn.linops.Zero((shape_diff, L_U.shape[1]), L_U.dtype)]]
            )
            if L.is_symmetric:
                self._V_extended = self._U_extended
            else:
                self._V_extended = BlockMatrix(
                    [[L.V], [pn.linops.Zero((shape_diff, L.V.shape[1]), L.V.dtype)]]
                )
            self._L_extended = pn.linops.BlockDiagonalMatrix(
                L, pn.linops.Zero((shape_diff, shape_diff), L.dtype)
            )
        super().__init__(self._U_extended, None if L.is_symmetric else self._V_extended)
        if L.is_symmetric:
            self.is_symmetric = True

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return self._L_extended @ x

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self._L_extended @ x
