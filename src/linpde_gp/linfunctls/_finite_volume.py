import functools

from linpde_gp.domains import Box, Interval
from linpde_gp.linfuncops.diffops import LinearDifferentialOperator

from ._arithmetic import CompositeLinearFunctional
from ._integrals import LebesgueIntegral


class FiniteVolumeFunctional(CompositeLinearFunctional):
    def __init__(self, volume: Box | Interval, diffop: LinearDifferentialOperator):
        self._volume = volume
        self._integral = LebesgueIntegral(volume)
        self._diffop = diffop

        super().__init__(
            linop=None,
            linfunctl=self._integral,
            linfuncop=self._diffop,
        )

    @property
    def volume(self) -> Box | Interval:
        return self._volume

    @property
    def diffop(self) -> LinearDifferentialOperator:
        return self._diffop

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)
