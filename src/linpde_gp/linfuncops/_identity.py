import functools

import probnum as pn
from probnum.typing import ShapeLike

from . import _linfuncop


class Identity(_linfuncop.LinearFunctionOperator):
    def __init__(
        self,
        domain_shape: ShapeLike,
        codomain_shape: ShapeLike,
    ) -> None:
        super().__init__(
            input_shapes=(domain_shape, codomain_shape),
            output_shapes=(domain_shape, codomain_shape),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /) -> pn.functions.Function:
        if f.input_shape != self.input_domain_shape:
            raise ValueError()

        if f.output_shape != self.input_codomain_shape:
            raise ValueError()

        return f

    @__call__.register
    def _(self, f: pn.randprocs.RandomProcess, /) -> pn.randprocs.RandomProcess:
        if f.input_shape != self.input_domain_shape:
            raise ValueError()

        if f.output_shape != self.input_codomain_shape:
            raise ValueError()

        return f
