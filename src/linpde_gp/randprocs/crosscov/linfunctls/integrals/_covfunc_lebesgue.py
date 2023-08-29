import functools
import types

import numpy as np
import probnum as pn
import scipy.integrate
from jax import numpy as jnp
from linpde_gp import linfunctls
from linpde_gp.domains import Interval
from linpde_gp.randvars import ArrayCovariance, Covariance

from .._base import LinearFunctionalProcessVectorCrossCovariance


class np_vectorize_method(np.vectorize):
    def __get__(self, obj, objtype=None):
        """https://docs.python.org/3/howto/descriptor.html#functions-and-methods"""
        if obj is None:
            return self
        return types.MethodType(self, obj)


class CovarianceFunction_Identity_LebesgueIntegral(
    LinearFunctionalProcessVectorCrossCovariance
):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        if integral.output_shape != () or integral.domain.common_type is not Interval:
            raise NotImplementedError()

        super().__init__(
            covfunc=covfunc,
            linfunctl=integral,
            reverse=reverse,
        )

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self.linfunctl

    @functools.partial(np_vectorize_method, excluded={0})
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        @np.vectorize
        def integrate(domain):
            return scipy.integrate.quad(
                lambda t: self.covfunc(x, t),
                a=domain[0],
                b=domain[1],
            )[0]

        return integrate(self.integral.domains.array)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()


@linfunctls.VectorizedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, Lk_or_kL: CovarianceFunction_Identity_LebesgueIntegral, /) -> Covariance:
    if Lk_or_kL.reverse:  # Lk
        integral0 = Lk_or_kL.integral
        integral1 = self
    else:  # kL'
        integral0 = self
        integral1 = Lk_or_kL.integral

    @np.vectorize
    def integrate(domain0, domain1):
        return scipy.integrate.dblquad(
            lambda x1, x0: Lk_or_kL.covfunc(x0, x1),
            *domain0,
            *domain1,
        )[0]

    domains0 = np.expand_dims(
        integral0.domains.array, axis=tuple(-np.arange(1, integral1.domains.ndim + 1))
    )
    domains1 = np.expand_dims(
        integral1.domains.array, axis=tuple(np.arange(integral0.domains.ndim))
    )
    res = integrate(domains0, domains1)

    return ArrayCovariance(
        res, shape0=integral0.output_shape, shape1=integral1.output_shape
    )
