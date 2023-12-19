import functools

import numpy as np
import probnum as pn
from linpde_gp import domains, functions
from linpde_gp.domains import VectorizedDomain
from linpde_gp.typing import DomainLike
from probnum.typing import ArrayLike, ShapeLike

from . import _linfunctl


class VectorizedLebesgueIntegral(_linfunctl.LinearFunctional):
    def __init__(
        self,
        input_domains: ArrayLike | VectorizedDomain,
        input_codomain_shape: ShapeLike = (),
    ):
        if isinstance(input_domains, VectorizedDomain):
            self._domains = input_domains
        else:
            self._domains = VectorizedDomain(input_domains)

        super().__init__(
            input_shapes=(self._domains.input_shape, input_codomain_shape),
            output_shape=self._domains.shape + input_codomain_shape,
        )

    @property
    def domains(self) -> VectorizedDomain:
        return self._domains

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /) -> np.ndarray:
        try:
            return super().__call__(f)
        except NotImplementedError as err:
            import scipy.integrate  # pylint: disable=import-outside-toplevel

            if self.input_codomain_shape != ():
                raise NotImplementedError from err

            @np.vectorize
            def integrate(domain):
                match domain:
                    case domains.Interval():
                        return scipy.integrate.quad(f, a=domain[0], b=domain[1])[0]
                    case domains.Box():
                        return scipy.integrate.nquad(
                            f,
                            ranges=[tuple(interval) for interval in domain],
                        )[0]
                raise NotImplementedError from err

            return integrate(self._domains.array)

    @__call__.register
    def _(self, f: functions.Constant, /) -> np.ndarray:
        @np.vectorize
        def integrate(domain):
            return f.value * domain.volume

        return integrate(self._domains.array)

    def __repr__(self) -> str:
        return f"∫_{{{self._domains.array.shape}}}"


class LebesgueIntegral(VectorizedLebesgueIntegral):
    def __init__(self, input_domain: DomainLike, input_codomain_shape: ShapeLike = ()):
        super().__init__(domains.asdomain(input_domain), input_codomain_shape)
