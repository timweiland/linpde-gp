import functools

import numpy as np
import probnum as pn
from linpde_gp.domains import Domain


class SolutionErrorEstimator:
    TEMPORAL_RESOLUTION_DEFAULT = 10
    SPATIAL_RESOLUTION_DEFAULT = 50

    def __init__(
        self,
        solution: callable,
        domain: Domain,
        resolution: tuple[int, ...] | None = None,
    ):
        self._solution = solution
        self._domain = domain
        if resolution is None:
            resolution = (self.TEMPORAL_RESOLUTION_DEFAULT,) + (
                self.SPATIAL_RESOLUTION_DEFAULT,
            ) * (domain.size - 1)
        self._resolution = resolution

    @property
    def solution(self):
        return self._solution

    @property
    def domain(self):
        return self._domain

    @property
    def resolution(self):
        return self._resolution

    @property
    def _X(self):
        return self.domain.uniform_grid(self.resolution)

    @functools.cached_property
    def solution_normalization_constant(self):
        return np.sqrt(np.mean(self.solution(self._X) ** 2))

    def __call__(self, gp: pn.randprocs.GaussianProcess):
        return (
            np.sqrt(np.mean((self.solution(self._X) - gp.mean(self._X)) ** 2))
            / self.solution_normalization_constant
        )
