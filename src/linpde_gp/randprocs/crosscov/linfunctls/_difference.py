# Difference covariance functions.
# Arise e.g. from the fundamental theorem of calculus.
import numpy as np
from probnum.randprocs.covfuncs import CovarianceFunction

from linpde_gp import domains, linfuncops, linfunctls
from linpde_gp.randprocs.covfuncs.linfuncops.diffops import (
    UnivariateHalfIntegerMatern_Derivative_Derivative,
    UnivariateHalfIntegerMatern_Identity_Derivative,
)
from linpde_gp.randprocs.crosscov.linfunctls.integrals import (
    UnivariateHalfIntegerMaternLebesgueIntegral,
)

from .. import ProcessVectorCrossCovariance, _arithmetic


class CovarianceFunction_Identity_Difference(
    _arithmetic.SumProcessVectorCrossCovariance
):
    def __init__(
        self,
        covfunc: CovarianceFunction,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        *,
        reverse: bool = False,
    ):
        self._covfunc = covfunc
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._reverse = bool(reverse)

        super().__init__(
            linfunctls._EvaluationFunctional(covfunc.input_shape, (), upper_bound)(
                covfunc,
                argnum=0 if reverse else 1,
            ),
            -linfunctls._EvaluationFunctional(covfunc.input_shape, (), lower_bound)(
                covfunc,
                argnum=0 if reverse else 1,
            ),
        )

    @property
    def covfunc(self) -> CovarianceFunction:
        return self._covfunc

    @property
    def lower_bound(self) -> np.ndarray:
        return self._lower_bound

    @property
    def upper_bound(self) -> np.ndarray:
        return self._upper_bound

    def __repr__(self) -> str:
        if self.reverse:
            return f"{self.covfunc}({self.upper_bound}, x) - {self.covfunc}({self.lower_bound}, x)"
        else:
            return f"{self.covfunc}(x, {self.upper_bound}) - {self.covfunc}(x, {self.lower_bound})"


@linfunctls.FiniteVolumeFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Difference
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Difference, /) -> float:
    if not isinstance(self.volume, domains.Interval):
        raise NotImplementedError()

    k = pv_crosscov.covfunc
    if (pv_crosscov.lower_bound, pv_crosscov.upper_bound) == (
        self.volume[0],
        self.volume[1],
    ):
        return 2 * (
            k(self.volume[1], self.volume[1]) - k(self.volume[1], self.volume[0])
        )
    if pv_crosscov.reverse:
        return (
            k(pv_crosscov.upper_bound, self.volume[1])
            - k(pv_crosscov.upper_bound, self.volume[0])
            - k(pv_crosscov.lower_bound, self.volume[1])
            + k(pv_crosscov.lower_bound, self.volumen[0])
        )
    return (
        k(self.volume[1], pv_crosscov.upper_bound)
        - k(self.volume[0], pv_crosscov.upper_bound)
        - k(self.volume[1], pv_crosscov.lower_bound)
        + k(self.volume[0], pv_crosscov.lower_bound)
    )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    UnivariateHalfIntegerMatern_Identity_Derivative
)
def _(
    self,
    covfunc: UnivariateHalfIntegerMatern_Identity_Derivative,
    /,
    *,
    argnum: int = 0,
) -> CovarianceFunction_Identity_Difference:
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    integral_reverse = argnum == 0
    if covfunc.reverse != integral_reverse:
        return -1 * CovarianceFunction_Identity_Difference(
            covfunc.matern,
            self.domain[0],
            self.domain[1],
            reverse=integral_reverse,
        )
    return CovarianceFunction_Identity_Difference(
        covfunc.matern,
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )


@linfuncops.diffops.PartialDerivative.__call__.register(
    UnivariateHalfIntegerMaternLebesgueIntegral
)
def _(
    self, pv_crosscov: UnivariateHalfIntegerMaternLebesgueIntegral, /
) -> ProcessVectorCrossCovariance:
    return -1 * CovarianceFunction_Identity_Difference(
        pv_crosscov.matern,
        pv_crosscov.integral.domain[0],
        pv_crosscov.integral.domain[1],
        reverse=pv_crosscov.reverse,
    )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    UnivariateHalfIntegerMatern_Derivative_Derivative
)
def _(
    self,
    covfunc: UnivariateHalfIntegerMatern_Derivative_Derivative,
    /,
    *,
    argnum: int = 0,
) -> CovarianceFunction_Identity_Difference:
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    integral_reverse = argnum == 0

    return CovarianceFunction_Identity_Difference(
        UnivariateHalfIntegerMatern_Identity_Derivative(
            covfunc.matern,
            reverse=not integral_reverse,
        ),
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )
