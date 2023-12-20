from typing import Optional
import numpy as np
import probnum as pn
from probnum.randprocs.covfuncs import CovarianceFunction


class FunctionScaledCovarianceFunction(CovarianceFunction):
    def __init__(
        self,
        covfunc: CovarianceFunction,
        /,
        fn0: pn.functions.Function = None,
        fn1: pn.functions.Function = None,
    ) -> None:
        self._covfunc = covfunc
        if fn0 is not None:
            if fn0.input_shape != covfunc.input_shape_0:
                raise ValueError()
            if fn0.output_shape != covfunc.output_shape_0:
                raise ValueError()
        if fn1 is not None:
            if fn1.input_shape != covfunc.input_shape_1:
                raise ValueError()
            if fn1.output_shape != covfunc.output_shape_1:
                raise ValueError()
        self._fn0 = fn0
        self._fn1 = fn1

        super().__init__(
            input_shape_0=covfunc.input_shape_0,
            input_shape_1=covfunc.input_shape_1,
            output_shape_0=covfunc.output_shape_0,
            output_shape_1=covfunc.output_shape_1,
        )

    @property
    def covfunc(self) -> CovarianceFunction:
        return self._covfunc

    @property
    def fn0(self) -> pn.functions.Function:
        return self._fn0

    @property
    def fn1(self) -> pn.functions.Function:
        return self._fn1

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        f0_res = self.fn0(x0) if self.fn0 is not None else 1
        if self.fn1 is None:
            f1_res = 1
        else:
            f1_res = self._fn1(x1) if x1 is not None else self._fn1(x0)

        if self.output_ndim_0 > 0 and self.output_ndim_1 > 0:
            return f0_res[..., None] * self.covfunc(x0, x1) * f1_res[..., None, :]
        return f0_res * self.covfunc(x0, x1) * f1_res

    def __repr__(self) -> str:
        return f"{self._fn0} * {self._covfunc} * {self._fn1}"
