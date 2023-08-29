from jax import numpy as jnp
import numpy as np
import probnum as pn

from linpde_gp import functions, linfunctls
from linpde_gp.randprocs import covfuncs
from linpde_gp.randvars import Covariance

from pykeops.numpy import LazyTensor, Pm

from ._radial_lebesgue import (
    UnivariateRadialCovarianceFunctionLebesgueIntegral,
    univariate_radial_covfunc_lebesgue_integral_lebesgue_integral,
)


class HalfIntegerMaternRadialAntiderivative(functions.JaxFunction):
    def __init__(self, order_int: int) -> None:
        super().__init__(input_shape=(), output_shape=())

        self._sqrt_2nu = np.sqrt(2 * order_int + 1)
        self._neg_inv_sqrt_2nu = -(1.0 / self._sqrt_2nu)

        # Compute the polynomial part of the function
        p_i = functions.RationalPolynomial(
            covfuncs.Matern.half_integer_coefficients(order_int)
        )

        poly = p_i

        for _ in range(order_int):
            p_i = p_i.differentiate()

            poly += p_i

        self._poly = poly

        self._C_1 = -self._neg_inv_sqrt_2nu * self._poly.coefficients[0]

    def _evaluate(  # pylint: disable=arguments-renamed
        self, r: np.ndarray
    ) -> np.ndarray:
        return (
            self._neg_inv_sqrt_2nu
            * np.exp(-self._sqrt_2nu * r)
            * self._poly(self._sqrt_2nu * r)
            + self._C_1
        )

    def _evaluate_jax(  # pylint: disable=arguments-renamed
        self, r: jnp.ndarray
    ) -> jnp.ndarray:
        return (
            self._neg_inv_sqrt_2nu
            * jnp.exp(-self._sqrt_2nu * r)
            * self._poly.jax(self._sqrt_2nu * r)
            + self._C_1
        )


class HalfIntegerMaternRadialSecondAntiderivative(functions.JaxFunction):
    def __init__(self, order_int: int) -> None:
        super().__init__(input_shape=(), output_shape=())

        self._sqrt_2nu = np.sqrt(2 * order_int + 1)
        self._inv_2nu = 1.0 / (2 * order_int + 1)

        self._antideriv_1 = HalfIntegerMaternRadialAntiderivative(order_int)
        self._C_1 = self._antideriv_1._C_1

        # Compute the polynomial part of the function
        p_i = functions.RationalPolynomial(
            covfuncs.Matern.half_integer_coefficients(order_int)
        )

        poly = p_i

        for i in range(1, order_int + 1):
            p_i = p_i.differentiate()

            poly += (i + 1) * p_i

        self._poly = poly

        self._C_2 = -self._inv_2nu * self._poly.coefficients[0]

    def _evaluate(  # pylint: disable=arguments-renamed
        self, r: np.ndarray
    ) -> np.ndarray:
        return (
            self._inv_2nu * np.exp(-self._sqrt_2nu * r) * self._poly(self._sqrt_2nu * r)
            + self._C_1 * r
            + self._C_2
        )

    def _evaluate_jax(  # pylint: disable=arguments-renamed
        self, r: jnp.ndarray
    ) -> jnp.ndarray:
        return (
            self._inv_2nu
            * jnp.exp(-self._sqrt_2nu * r)
            * self._poly.jax(self._sqrt_2nu * r)
            + self._C_1 * r
            + self._C_2
        )
    
    def _evaluate_keops(
        self, r: LazyTensor
    ) -> LazyTensor:
        return (
            self._inv_2nu
            * LazyTensor.exp(-self._sqrt_2nu * r)
            * self._poly._evaluate_keops(self._sqrt_2nu * r)
            + self._C_1 * r
            + self._C_2
        )


class UnivariateHalfIntegerMaternLebesgueIntegral(
    UnivariateRadialCovarianceFunctionLebesgueIntegral
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        assert matern.input_shape == ()
        assert matern.is_half_integer

        super().__init__(
            radial_covfunc=matern,
            integral=integral,
            radial_antideriv=HalfIntegerMaternRadialAntiderivative(matern.p),
            reverse=reverse,
        )

    @property
    def matern(self) -> covfuncs.Matern:
        return self.covfunc

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"matern={self.matern!r}, "
            f"integral={self.integral!r}, "
            f"reverse={self.reverse!r})"
        )


@linfunctls.VectorizedLebesgueIntegral.__call__.register(  # pylint: disable=no-member
    UnivariateHalfIntegerMaternLebesgueIntegral
)
def _(self, kL_or_Lk: UnivariateHalfIntegerMaternLebesgueIntegral, /) -> Covariance:
    if kL_or_Lk.reverse:  # Lk
        integral0 = kL_or_Lk.integral
        integral1 = self
    else:  # kL'
        integral0 = self
        integral1 = kL_or_Lk.integral

    return univariate_radial_covfunc_lebesgue_integral_lebesgue_integral(
        kL_or_Lk.matern,
        integral0,
        integral1,
        radial_antideriv_2=HalfIntegerMaternRadialSecondAntiderivative(
            kL_or_Lk.matern.p
        ),
    )