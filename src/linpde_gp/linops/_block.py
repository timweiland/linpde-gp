import numpy as np
import probnum as pn
from probnum.typing import LinearOperatorLike


class BlockMatrix(pn.linops.LinearOperator):
    """
    A linear operator that represents a linear system of the form:

        | A  B | | x | = | u |
        | C  D | | y |   | v |

    Parameters
    ----------
        A (LinearOperatorLike): Top left block.
        B (LinearOperatorLike | None): Top right block.
        C (LinearOperatorLike | None): Bottom left block.
        D (LinearOperatorLike): Bottom right block.
        is_spd (bool, optional): If True, the operator is symmetric positive
            definite. Default is False.

    Notes
    -----
    Block diagonal: Set B = C = None.
    SPD: Set exactly one of B or C to None. This block will be inferred
        automatically from symmetry.
    Triangular: Set one of B or C to None. This block will be zero, 
        and this also defines the structure (lower or upper triangular).
    """

    def __init__(
        self,
        A: LinearOperatorLike,
        B: LinearOperatorLike | None,
        C: LinearOperatorLike | None,
        D: LinearOperatorLike,
        is_spd=False,
    ):
        self._A = pn.linops.aslinop(A)
        self._D = pn.linops.aslinop(D)

        dtype = np.promote_types(
            self._A.dtype if B is None else np.promote_types(self._A.dtype, B.dtype),
            self._D.dtype if C is None else np.promote_types(self._D.dtype, C.dtype),
        )

        super().__init__(
            shape=(A.shape[0] + D.shape[0], A.shape[1] + D.shape[1]), dtype=dtype
        )

        self.is_block_diagonal = False
        if B is None and C is None:
            # Block diagonal
            self._B = pn.linops.Zero((self._A.shape[0], self._D.shape[1]), dtype)
            self._C = pn.linops.Zero((self._D.shape[0], self._A.shape[1]), dtype)
            self.is_block_diagonal = True
            if self._A.is_symmetric and self._D.is_symmetric:
                self.is_symmetric = True
        elif is_spd:
            assert (B is None) ^ (C is None)  # Exactly one of B or C must be None
            assert self._A.is_symmetric and self._D.is_symmetric
            assert self._A.is_positive_definite
            if C is None:
                self._B = pn.linops.aslinop(B)
                self._C = self._B.T
            else:
                self._C = pn.linops.aslinop(C)
                self._B = self._C.T
            self.is_symmetric = True
            self.is_positive_definite = True
        elif self._A.is_lower_triangular and self._D.is_lower_triangular and B is None:
            # Lower triangular block matrix
            self._C = (
                pn.linops.aslinop(C)
                if C is not None
                else pn.linops.Zero((self._D.shape[0], self._A.shape[1]), dtype)
            )
            self._B = pn.linops.Zero((self._A.shape[0], self._D.shape[1]), dtype)
            self.is_lower_triangular = True
        elif self._A.is_upper_triangular and self._D.is_upper_triangular and C is None:
            # Upper triangular block matrix
            self._B = (
                pn.linops.aslinop(B)
                if B is not None
                else pn.linops.Zero((self._A.shape[0], self._D.shape[1]), dtype)
            )
            self._C = pn.linops.Zero((self._D.shape[0], self._A.shape[1]), dtype)
            self.is_upper_triangular = True
        else:
            assert B is not None and C is not None
            self._B = pn.linops.aslinop(B)
            self._C = pn.linops.aslinop(C)

        self._schur = None

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def C(self):
        return self._C

    @property
    def D(self):
        return self._D

    @property
    def schur(self):
        if self._schur is None:
            self._schur = self.D - self.C @ self.A.inv() @ self.B
            self._schur.is_symmetric = self.is_symmetric
            self._schur.is_positive_definite = self.is_positive_definite
        return self._schur

    def _split_input(self, x: np.ndarray, axis: int):
        return np.split(x, [self.A.shape[1]], axis=axis)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        x0, x1 = self._split_input(x, axis=-2)

        return np.concatenate(
            (
                self.A @ x0 + self.B @ x1,
                self.C @ x0 + self.D @ x1,
            ),
            axis=-2,
        )

    def _transpose(self) -> pn.linops.LinearOperator:
        return BlockMatrix(self.A.T, self.C.T, self.B.T, self.D.T)

    def schur_update(self, A_inv_u, v):
        if self.is_block_diagonal:
            return np.concatenate((A_inv_u, self.D.inv() @ v))
        y = self.schur.inv() @ (v - self.C @ A_inv_u)
        x = A_inv_u - self.A.inv() @ self.B @ y
        return np.concatenate((x, y))

    def _cholesky(self, lower: bool) -> pn.linops.LinearOperator:
        A_sqrt = self.A.cholesky(True)
        A_sqrt.is_lower_triangular = True

        L_A_inv_B = A_sqrt.inv() @ self._B

        # Compute the Schur complement manually using L_A_inv_B which we need anyway
        if self._schur is None:
            self._schur = self.D - L_A_inv_B.T @ L_A_inv_B
        self._schur.is_symmetric = True
        self._schur.is_positive_definite = True
        S_sqrt = self._schur.cholesky(True)
        S_sqrt.is_lower_triangular = True

        if lower:
            block_sqrt = BlockMatrix(
                A_sqrt, None, L_A_inv_B.T, S_sqrt
            )
        else:
            block_sqrt = BlockMatrix(
                A_sqrt.T, L_A_inv_B, None, S_sqrt.T
            )
        return block_sqrt

    def _solve(self, B: np.ndarray) -> np.ndarray:
        assert B.ndim == 2
        b0, b1 = self._split_input(B, axis=-2)
        if self.is_block_diagonal:
            return np.concatenate((self.A.inv() @ b0, self.D.inv() @ b1), axis=-2)
        if self.is_symmetric:
            L = self.cholesky(True)
            return L.T.inv() @ (L.inv() @ B)
        if self.is_lower_triangular:
            y0 = self.A.inv() @ b0
            y1 = self.D.inv() @ (b1 - self.C @ y0)
            return np.concatenate((y0, y1), axis=-2)
        if self.is_upper_triangular:
            y1 = self.D.inv() @ b1
            y0 = self.A.inv() @ (b0 - self.B @ y1)
            return np.concatenate((y0, y1), axis=-2)
        return super()._solve(B)

    def _todense(self) -> np.ndarray:
        return np.block(
            [
                [self.A.todense(cache=False), self.B.todense(cache=False)],
                [self.C.todense(cache=False), self.D.todense(cache=False)],
            ]
        )

    def _trace(self) -> np.number:
        return self.A.trace() + self.D.trace()

    def _det(self) -> np.inexact:
        if self.is_block_diagonal or self.is_lower_triangular or self.is_upper_triangular:
            return self.A.det() * self.D.det()
        if self.is_symmetric and self.is_positive_definite:
            # TODO: This holds generally when A is invertible, SPD is just a
            # special case.
            return self.A.det() * self.schur.det()
        return super()._det()
