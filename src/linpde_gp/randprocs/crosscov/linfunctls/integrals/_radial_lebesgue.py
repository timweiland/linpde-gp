import numpy as np
import probnum as pn
from jax import numpy as jnp
from linpde_gp import functions, linfunctls
from linpde_gp.linops import KeOpsLinearOperator
from linpde_gp.randvars import LinearOperatorCovariance
from probnum.randprocs import covfuncs
from pykeops.numpy import LazyTensor, Vi, Vj

from .._base import LinearFunctionalProcessVectorCrossCovariance


def _Vi(x: np.ndarray) -> LazyTensor:
    assert x.ndim == 2
    if x.shape == (1, 1):
        return x[0, 0]
    return Vi(x)


def _Vj(x: np.ndarray) -> LazyTensor:
    assert x.ndim == 2
    if x.shape == (1, 1):
        return x[0, 0]
    return Vj(x)


class UnivariateRadialCovarianceFunctionLebesgueIntegral(
    LinearFunctionalProcessVectorCrossCovariance
):
    def __init__(
        self,
        radial_covfunc: covfuncs.CovarianceFunction,
        integral: linfunctls.VectorizedLebesgueIntegral,
        radial_antideriv: functions.JaxFunction,
        reverse: bool = False,
    ):
        assert radial_covfunc.input_shape == ()

        super().__init__(
            covfunc=radial_covfunc,
            linfunctl=integral,
            reverse=reverse,
        )

        self._radial_antideriv = radial_antideriv

    @property
    def integral(self) -> linfunctls.VectorizedLebesgueIntegral:
        return self.linfunctl

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        l = self.covfunc.lengthscale
        a, b = (
            self.integral.domains.pure_array[..., 0],
            self.integral.domains.pure_array[..., 1],
        )
        if self.reverse:
            a, x = make_broadcastable(a, x)
            b = np.reshape(b, a.shape)
        else:
            x, a = make_broadcastable(x, a)
            b = np.reshape(b, a.shape)
        return l * (
            (-1) ** (b < x) * self._radial_antideriv(np.abs(b - x) / l)
            - (-1) ** (a < x) * self._radial_antideriv(np.abs(a - x) / l)
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        l = self.covfunc.lengthscale
        a, b = (
            self.integral.domains.pure_array[..., 0],
            self.integral.domains.pure_array[..., 1],
        )
        if self.reverse:
            a, x = make_broadcastable_jax(a, x)
            b = jnp.reshape(b, a.shape)
        else:
            x, a = make_broadcastable_jax(x, a)
            b = jnp.reshape(b, a.shape)

        return l * (
            (-1) ** (b < x) * self._radial_antideriv.jax(jnp.abs(b - x) / l)
            - (-1) ** (a < x) * self._radial_antideriv.jax(jnp.abs(a - x) / l)
        )

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        # Build KeOps lazy tensor
        l = self.covfunc.lengthscale
        a, b = (
            self.integral.domains.pure_array[..., 0],
            self.integral.domains.pure_array[..., 1],
        )
        a_contiguous = np.ascontiguousarray(a.reshape((-1, 1)))
        x_contiguous = np.ascontiguousarray(x.reshape((-1, 1)))
        b_contiguous = np.ascontiguousarray(b.reshape((-1, 1)))
        if self.reverse:
            a_lazy, x_lazy = Vi(a_contiguous), Vj(x_contiguous)
            b_lazy = Vi(b_contiguous)
            a, x = make_broadcastable(a, x)
            b = np.reshape(b, a.shape)
        else:
            x_lazy, a_lazy = Vi(x_contiguous), Vj(a_contiguous)
            b_lazy = Vj(b_contiguous)
            x, a = make_broadcastable(x, a)
            b = np.reshape(b, a.shape)

        l = np.asarray(l)[()]
        lazy_tensor = l * (
            (b_lazy - x_lazy).ifelse(1, -1)
            * self._radial_antideriv._evaluate_keops(
                LazyTensor.abs(b_lazy - x_lazy) / l
            )
            - (a_lazy - x_lazy).ifelse(1, -1)
            * self._radial_antideriv._evaluate_keops(
                LazyTensor.abs(a_lazy - x_lazy) / l
            )
        )

        return KeOpsLinearOperator(
            lazy_tensor,
            dense_fallback=lambda: l
            * (
                (-1) ** (b < x) * self._radial_antideriv(np.abs(b - x) / l)
                - (-1) ** (a < x) * self._radial_antideriv(np.abs(a - x) / l)
            ),
        )


def univariate_radial_covfunc_lebesgue_integral_lebesgue_integral(
    k: covfuncs.CovarianceFunction,
    integral0: linfunctls.VectorizedLebesgueIntegral,
    integral1: linfunctls.VectorizedLebesgueIntegral,
    radial_antideriv_2: functions.JaxFunction,
):
    l = k.lengthscale

    a, b = integral0.domains.pure_array[..., 0], integral0.domains.pure_array[..., 1]
    c, d = integral1.domains.pure_array[..., 0], integral1.domains.pure_array[..., 1]

    lazy_tensor = _univariate_radial_covfunc_lebesgue_integral_lebesgue_integral_keops(
        a, b, c, d, l, radial_antideriv_2
    )
    return LinearOperatorCovariance(
        KeOpsLinearOperator(
            lazy_tensor,
            dense_fallback=lambda: _univariate_radial_covfunc_lebesgue_integral_lebesgue_integral_dense(
                a, b, c, d, l, radial_antideriv_2
            ),
        ),
        shape0=integral0.output_shape,
        shape1=integral1.output_shape,
    )


def _univariate_radial_covfunc_lebesgue_integral_lebesgue_integral_dense(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    l: float,
    radial_antideriv_2: functions.JaxFunction,
):
    a, c = make_broadcastable(a, c)
    b, d = make_broadcastable(b, d)

    return l**2 * (
        radial_antideriv_2(np.abs(b - c) / l)
        - radial_antideriv_2(np.abs(a - c) / l)
        - radial_antideriv_2(np.abs(b - d) / l)
        + radial_antideriv_2(np.abs(a - d) / l)
    )


def _univariate_radial_covfunc_lebesgue_integral_lebesgue_integral_keops(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    l: float,
    radial_antideriv_2: functions.JaxFunction,
):
    a, c = Vi(a.reshape((-1, 1))), Vj(c.reshape((-1, 1)))
    b, d = Vi(b.reshape((-1, 1))), Vj(d.reshape((-1, 1)))
    l = np.asarray(l)[()]

    return l**2 * (
        radial_antideriv_2._evaluate_keops(LazyTensor.abs(b - c) / l)
        - radial_antideriv_2._evaluate_keops(LazyTensor.abs(a - c) / l)
        - radial_antideriv_2._evaluate_keops(LazyTensor.abs(b - d) / l)
        + radial_antideriv_2._evaluate_keops(LazyTensor.abs(a - d) / l)
    )


def make_broadcastable(x0: np.ndarray, x1: np.ndarray):
    x0_ndim = x0.ndim
    x0 = np.expand_dims(x0, axis=tuple(-np.arange(1, x1.ndim + 1)))
    x1 = np.expand_dims(x1, axis=tuple(np.arange(x0_ndim)))
    return x0, x1


def make_broadcastable_jax(x0: np.ndarray, x1: np.ndarray):
    x0_ndim = x0.ndim
    x0 = jnp.expand_dims(x0, axis=tuple(-jnp.arange(1, x1.ndim + 1)))
    x1 = jnp.expand_dims(x1, axis=tuple(jnp.arange(x0_ndim)))
    return x0, x1
