# Difference covariance functions.
# Arise e.g. from the fundamental theorem of calculus.
import numpy as np
from probnum.randprocs.covfuncs import CovarianceFunction

from linpde_gp import linfunctls

from .. import _arithmetic


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
