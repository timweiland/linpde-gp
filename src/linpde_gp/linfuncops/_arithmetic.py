import functools
import operator
from typing import Generic, TypeVar

import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from ._linfuncop import LinearFunctionOperator


class ScaledLinearFunctionOperator(LinearFunctionOperator):
    def __init__(
        self, linfuncop: LinearFunctionOperator, /, scalar: ScalarLike
    ) -> None:
        self._linfuncop = linfuncop

        super().__init__(
            input_shapes=self._linfuncop.input_shapes,
            output_shapes=self._linfuncop.output_shapes,
        )

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

    @property
    def linfuncop(self) -> LinearFunctionOperator:
        return self._linfuncop

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._scalar * self._linfuncop(f, **kwargs)

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> LinearFunctionOperator:
        if np.ndim(other) == 0:
            return ScaledLinearFunctionOperator(
                self._linfuncop,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._linfuncop}"


class FunctionScaledLinearFunctionOperator(LinearFunctionOperator):
    def __init__(
        self, linfuncop: LinearFunctionOperator, /, fn: pn.functions.Function
    ) -> None:
        self._linfuncop = linfuncop

        if fn.input_shape != linfuncop.output_domain_shape:
            raise ValueError()
        if fn.output_shape != linfuncop.output_codomain_shape:
            raise ValueError()

        self._fn = fn

        super().__init__(
            input_shapes=self._linfuncop.input_shapes,
            output_shapes=self._linfuncop.output_shapes,
        )

    @property
    def linfuncop(self) -> LinearFunctionOperator:
        return self._linfuncop

    @property
    def fn(self) -> pn.functions.Function:
        return self._fn

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self.fn * self._linfuncop(f, **kwargs)

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> LinearFunctionOperator:
        if isinstance(other, pn.functions.Function):
            return FunctionScaledLinearFunctionOperator(
                linfuncop=self._linfuncop,
                fn=self.fn * other,
            )

        return super().__rmul__(other)

    def __repr__(self) -> str:
        return f"{self.fn} * {self._linfuncop}"


T = TypeVar("T", bound=LinearFunctionOperator)


class SumLinearFunctionOperator(LinearFunctionOperator, Generic[T]):
    def __init__(self, *summands: T) -> None:
        self._summands = tuple(summands)

        input_domain_shape = self._summands[0].input_domain_shape
        input_codomain_shape = self._summands[0].input_codomain_shape
        output_domain_shape = self._summands[0].output_domain_shape
        output_codomain_shape = self._summands[0].output_codomain_shape

        assert all(
            summand.input_domain_shape == input_domain_shape
            for summand in self._summands
        )
        assert all(
            summand.input_codomain_shape == input_codomain_shape
            for summand in self._summands
        )
        assert all(
            summand.output_domain_shape == output_domain_shape
            for summand in self._summands
        )
        assert all(
            summand.output_codomain_shape == output_codomain_shape
            for summand in self._summands
        )

        super().__init__(
            input_shapes=(input_domain_shape, input_codomain_shape),
            output_shapes=(output_domain_shape, output_codomain_shape),
        )

    @property
    def summands(self) -> tuple[T, ...]:
        return self._summands

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return functools.reduce(
            operator.add, (summand(f, **kwargs) for summand in self._summands)
        )

    @__call__.register
    def _(self, randproc: pn.randprocs.RandomProcess, /) -> pn.randprocs.RandomProcess:
        return super().__call__(randproc)

    def __repr__(self):
        return " + ".join(str(summand) for summand in self._summands)


class CompositeLinearFunctionOperator(LinearFunctionOperator, Generic[T]):
    def __init__(self, *linfuncops: T) -> None:
        assert all(
            L0.input_shapes == L1.output_shapes
            for L0, L1 in zip(linfuncops[:-1], linfuncops[1:])
        )

        self._linfuncops = tuple(linfuncops)

        super().__init__(
            input_shapes=self._linfuncops[-1].input_shapes,
            output_shapes=self._linfuncops[0].output_shapes,
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return functools.reduce(
            lambda h, linfuncop: linfuncop(h, **kwargs),
            reversed(self._linfuncops),
            f,
        )

    @property
    def linfuncops(self) -> tuple[T, ...]:
        return self._linfuncops

    def __repr__(self) -> str:
        return " @ ".join(repr(linfuncop) for linfuncop in self._linfuncops)


@LinearFunctionOperator.__matmul__.register  # pylint: disable=no-member
def _(self, other: SumLinearFunctionOperator) -> SumLinearFunctionOperator:
    return SumLinearFunctionOperator(*(self @ summand for summand in other.summands))


@pn.functions.Function.__mul__.register
@pn.functions.Function.__rmul__.register
def _(self, other: FunctionScaledLinearFunctionOperator, /) -> pn.functions.Function:
    return FunctionScaledLinearFunctionOperator(other.linfuncop, fn=self * other.fn)
