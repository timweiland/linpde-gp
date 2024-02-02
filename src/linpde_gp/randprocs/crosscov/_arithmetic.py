from collections.abc import Sequence
import functools
from math import prod
import operator

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ScalarType
from linpde_gp.linops import RankOneHadamardProduct

from linpde_gp.randprocs.covfuncs import TensorProductGrid

from . import _pv_crosscov


class ScaledProcessVectorCrossCovariance(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(
        self,
        pv_crosscov: _pv_crosscov.ProcessVectorCrossCovariance,
        scalar: ArrayLike,
        scale_randvar=True,
    ):
        self._pv_crosscov = pv_crosscov
        self._scalar = np.asarray(scalar)
        self._scale_randvar = scale_randvar

        super().__init__(
            randproc_input_shape=pv_crosscov.randproc_input_shape,
            randproc_output_shape=pv_crosscov.randproc_output_shape,
            randvar_shape=pv_crosscov.randvar_shape,
            reverse=pv_crosscov.reverse,
        )

    @property
    def pv_crosscov(self) -> _pv_crosscov.ProcessVectorCrossCovariance:
        return self._pv_crosscov

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @property
    def scale_randvar(self) -> bool:
        return self._scale_randvar

    @functools.cached_property
    def _broadcasted_scalar(self) -> np.ndarray:
        if self.scale_randvar:
            return np.broadcast_to(self.scalar, self.randvar_shape)
        return np.broadcast_to(self.scalar, self.randproc_output_shape)

    def _reshape_scalar_to_res(self, res: np.ndarray) -> np.ndarray:
        if (self.scale_randvar and self.reverse) or (
            not self.scale_randvar and not self.reverse
        ):
            ndim_diff = res.ndim - self._broadcasted_scalar.ndim
            return self._broadcasted_scalar.reshape(
                self._broadcasted_scalar.shape + (1,) * ndim_diff
            )
        return self._broadcasted_scalar

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        crosscov_res = self._pv_crosscov(x)
        return self._reshape_scalar_to_res(crosscov_res) * crosscov_res

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        crosscov_res = self._pv_crosscov.jax(x)
        return self._reshape_scalar_to_res(crosscov_res) * crosscov_res

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        covop = self._pv_crosscov.evaluate_linop(x)  # pylint: disable=protected-access

        if self._scalar.ndim == 0:
            return self._scalar * covop

        scalar_flat = self._broadcasted_scalar.reshape(-1, order="C")
        if (self.scale_randvar and self.reverse) or (
            not self.scale_randvar and not self.reverse
        ):
            return pn.linops.Scaling(scalar_flat) @ covop
        return covop @ pn.linops.Scaling(scalar_flat)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._pv_crosscov}"


class FunctionScaledProcessVectorCrossCovariance(
    _pv_crosscov.ProcessVectorCrossCovariance
):
    def __init__(
        self,
        pv_crosscov: _pv_crosscov.ProcessVectorCrossCovariance,
        fn: pn.functions.Function,
    ):
        self._pv_crosscov = pv_crosscov

        if fn.input_shape != pv_crosscov.randproc_input_shape:
            raise ValueError()
        if fn.output_shape != pv_crosscov.randproc_output_shape:
            raise ValueError()

        self._fn = fn

        super().__init__(
            randproc_input_shape=pv_crosscov.randproc_input_shape,
            randproc_output_shape=pv_crosscov.randproc_output_shape,
            randvar_shape=pv_crosscov.randvar_shape,
            reverse=pv_crosscov.reverse,
        )

    @property
    def pv_crosscov(self) -> _pv_crosscov.ProcessVectorCrossCovariance:
        return self._pv_crosscov

    @property
    def fn(self) -> pn.functions.Function:
        return self._fn

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._fn(x) * self._pv_crosscov(x)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._fn(x) * self._pv_crosscov.jax(x)

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        inner_linop = self._pv_crosscov._evaluate_linop(x)  # pylint: disable=protected-access
        fn_vals = self._fn(x)
        if self.reverse:
            return RankOneHadamardProduct(1, fn_vals, inner_linop)
        return RankOneHadamardProduct(fn_vals, 1, inner_linop)

    def __repr__(self) -> str:
        return f"{self._fn} * {self._pv_crosscov}"
    
@pn.functions.Function.__mul__.register
@pn.functions.Function.__rmul__.register
def _(
    self, other: _pv_crosscov.ProcessVectorCrossCovariance, /
) -> pn.functions.Function:
    return FunctionScaledProcessVectorCrossCovariance(other, fn=self)


class SumProcessVectorCrossCovariance(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(self, *pv_crosscovs: _pv_crosscov.ProcessVectorCrossCovariance):
        self._pv_crosscovs = tuple(pv_crosscovs)

        assert all(
            pv_crosscov.randproc_input_shape == pv_crosscovs[0].randproc_input_shape
            and (
                pv_crosscov.randproc_output_shape
                == pv_crosscovs[0].randproc_output_shape
            )
            and pv_crosscov.randvar_shape == pv_crosscovs[0].randvar_shape
            and pv_crosscov.reverse == pv_crosscovs[0].reverse
            for pv_crosscov in self._pv_crosscovs
        )

        super().__init__(
            randproc_input_shape=self._pv_crosscovs[0].randproc_input_shape,
            randproc_output_shape=self._pv_crosscovs[0].randproc_output_shape,
            randvar_shape=self._pv_crosscovs[0].randvar_shape,
            reverse=self._pv_crosscovs[0].reverse,
        )

    @property
    def pv_crosscovs(self) -> Sequence[_pv_crosscov.ProcessVectorCrossCovariance]:
        return self._pv_crosscovs

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return sum(pv_crosscov(x) for pv_crosscov in self._pv_crosscovs)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return sum(pv_crosscov.jax(x) for pv_crosscov in self._pv_crosscovs)

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        return functools.reduce(
            operator.add,
            (
                pv_crosscov._evaluate_linop(x)  # pylint: disable=protected-access
                for pv_crosscov in self._pv_crosscovs
            ),
        )

    def __repr__(self) -> str:
        return " + ".join(repr(pv_crosscov) for pv_crosscov in self._pv_crosscovs)


class TensorProductProcessVectorCrossCovariance(
    _pv_crosscov.ProcessVectorCrossCovariance
):
    def __init__(
        self,
        *pv_crosscovs: _pv_crosscov.ProcessVectorCrossCovariance,
        grid_factorized=False,
    ):
        self._pv_crosscovs = tuple(pv_crosscovs)

        assert all(
            pv_crosscov.randproc_input_shape == pv_crosscovs[0].randproc_input_shape
            and (
                pv_crosscov.randproc_output_shape
                == pv_crosscovs[0].randproc_output_shape
            )
            and pv_crosscov.reverse == pv_crosscovs[0].reverse
            for pv_crosscov in self._pv_crosscovs
        )

        if grid_factorized:
            assert all(
                pv_crosscov.randvar_ndim == pv_crosscovs[0].randvar_ndim
                for pv_crosscov in self._pv_crosscovs
            )
            randvar_shape = functools.reduce(
                operator.add,
                tuple(pv_crosscov.randvar_shape for pv_crosscov in self._pv_crosscovs),
            )
        else:
            assert all(
                pv_crosscov.randvar_shape == pv_crosscovs[0].randvar_shape
                for pv_crosscov in self._pv_crosscovs
            )
            randvar_shape = self._pv_crosscovs[0].randvar_shape

        super().__init__(
            randproc_input_shape=(len(self._pv_crosscovs),),
            randproc_output_shape=self._pv_crosscovs[0].randproc_output_shape,
            randvar_shape=randvar_shape,
            reverse=self._pv_crosscovs[0].reverse,
        )

        self._is_grid_factorized = grid_factorized

    @property
    def is_grid_factorized(self) -> bool:
        return self._is_grid_factorized

    @property
    def pv_crosscovs(self) -> Sequence[_pv_crosscov.ProcessVectorCrossCovariance]:
        return self._pv_crosscovs

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.is_grid_factorized and not isinstance(x, TensorProductGrid):
            raise NotImplementedError()
        if self.is_grid_factorized and isinstance(x, TensorProductGrid):
            res = functools.reduce(
                np.kron,
                (
                    pv_crosscov(x.factors[dim])
                    for dim, pv_crosscov in enumerate(self._pv_crosscovs)
                ),
            )
            x_batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]
            if self.reverse:
                return res.reshape(
                    self.randvar_shape + x_batch_shape + self.randproc_output_shape
                )
            return res.reshape(
                x_batch_shape + self.randproc_output_shape + self.randvar_shape
            )
        return prod(
            pv_crosscov(x[..., dim])
            for dim, pv_crosscov in enumerate(self._pv_crosscovs)
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.is_grid_factorized:
            raise NotImplementedError()
        return prod(
            pv_crosscov.jax(x[..., dim])
            for dim, pv_crosscov in enumerate(self._pv_crosscovs)
        )

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        if not (self.is_grid_factorized and isinstance(x, TensorProductGrid)):
            raise NotImplementedError()
        return functools.reduce(
            pn.linops.Kronecker,
            (
                pv_crosscov.evaluate_linop(
                    x.factors[dim]
                )  # pylint: disable=protected-access
                for dim, pv_crosscov in enumerate(self._pv_crosscovs)
            ),
        )

    def __repr__(self) -> str:
        res = "TensorProductPVCrosscov [\n\t"
        res += ",\n\t".join(repr(pv_crosscov) for pv_crosscov in self._pv_crosscovs)
        res += "\n]"
        return res


class LinOpProcessVectorCrossCovariance(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(
        self,
        linop: pn.linops.LinearOperator,
        pv_crosscov: _pv_crosscov.ProcessVectorCrossCovariance,
    ):
        assert pv_crosscov.randvar_ndim == 1
        assert linop.shape[1:] == pv_crosscov.randvar_shape

        self._linop = linop
        self._pv_crosscov = pv_crosscov

        super().__init__(
            randproc_input_shape=self._pv_crosscov.randproc_input_shape,
            randproc_output_shape=self._pv_crosscov.randproc_output_shape,
            randvar_shape=linop.shape[0:1],
            reverse=self._pv_crosscov.reverse,
        )

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return self._linop

    @property
    def pv_crosscov(self) -> _pv_crosscov.ProcessVectorCrossCovariance:
        return self._pv_crosscov

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._linop(
            self._pv_crosscov(x),
            axis=0 if self.reverse else -1,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        covop = self._pv_crosscov._evaluate_linop(x)  # pylint: disable=protected-access

        return self._linop @ covop

    def __repr__(self) -> str:
        return f"{repr(self._linop)} @ {repr(self._pv_crosscov)}"
