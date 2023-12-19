import functools

import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from ._lindiffop import LinearDifferentialOperator


class ScaledLinearDifferentialOperator(LinearDifferentialOperator):
    def __init__(
        self, lindiffop: LinearDifferentialOperator, /, scalar: ScalarLike
    ) -> None:
        self._lindiffop = lindiffop

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

        super().__init__(
            coefficients=float(self._scalar) * self._lindiffop.coefficients,
            input_shapes=self._lindiffop.input_shapes,
        )

    @property
    def lindiffop(self) -> LinearDifferentialOperator:
        return self._lindiffop

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._scalar * self._lindiffop(f, **kwargs)

    def _jax_fallback(self, f, /, **kwargs):
        raise NotImplementedError()

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> LinearDifferentialOperator:
        if np.ndim(other) == 0:
            return ScaledLinearDifferentialOperator(
                lindiffop=self._lindiffop,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)

    @functools.singledispatchmethod
    def weak_form(self, test_basis, /):
        return self._scalar * self._lindiffop.weak_form(test_basis)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._lindiffop}"

class FunctionScaledLinearDifferentialOperator(LinearDifferentialOperator):
    def __init__(
        self, lindiffop: LinearDifferentialOperator, /, fn: pn.functions.Function
    ) -> None:
        self._lindiffop = lindiffop

        if fn.input_shape != lindiffop.output_domain_shape:
            raise ValueError()
        if fn.output_shape != lindiffop.output_codomain_shape:
            raise ValueError()

        self._fn = fn

        super().__init__(
            coefficients=self._fn * self._lindiffop.coefficients,
            input_shapes=self._lindiffop.input_shapes,
        )

    @property
    def lindiffop(self) -> LinearDifferentialOperator:
        return self._lindiffop

    @property
    def fn(self) -> pn.functions.Function:
        return self._fn

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._fn * self._lindiffop(f, **kwargs)

    def _jax_fallback(self, f, /, **kwargs):
        raise NotImplementedError()

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    @functools.singledispatchmethod
    def __rmul__(self, other) -> LinearDifferentialOperator:
        if isinstance(other, pn.functions.Function):
            return FunctionScaledLinearDifferentialOperator(
                lindiffop=self._lindiffop,
                fn=self._fn * other,
            )

        return super().__rmul__(other)

    @functools.singledispatchmethod
    def weak_form(self, test_basis, /):
        return self._fn * self._lindiffop.weak_form(test_basis)

    def __repr__(self) -> str:
        return f"{self._fn} * {self._lindiffop}"

@pn.functions.Function.__mul__.register
@pn.functions.Function.__rmul__.register
def _(
    self, other: FunctionScaledLinearDifferentialOperator, /
) -> pn.functions.Function:
    return FunctionScaledLinearDifferentialOperator(other.lindiffop, fn=self * other.fn)