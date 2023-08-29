import functools

from linpde_gp.linfuncops.diffops import LinearDifferentialOperator
from probnum.typing import ArrayLike

from ._arithmetic import CompositeLinearFunctional
from ._integrals import VectorizedLebesgueIntegral


class FiniteVolumeFunctional(CompositeLinearFunctional):
    def __init__(
        self, volumes: ArrayLike, diffop: LinearDifferentialOperator, normalize=False
    ):
        self._volumes = volumes
        self._integral = VectorizedLebesgueIntegral(volumes)
        if normalize:
            self._integral = (1.0 / self._integral.domains.volume) * self._integral
        self._diffop = diffop

        super().__init__(
            linop=None,
            linfunctl=self._integral,
            linfuncop=self._diffop,
        )

        self._normalize = normalize

    @property
    def volumes(self) -> ArrayLike:
        return self._volumes

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def integral(self) -> VectorizedLebesgueIntegral:
        return self._integral

    @property
    def diffop(self) -> LinearDifferentialOperator:
        return self._diffop

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)
