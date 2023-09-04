import abc
from typing import Optional

import numpy as np

from ..._gp_solver import GPInferenceParams
from .._solver_state import SolverState


class ConcretePolicy(abc.ABC):
    def __init__(self, gp_params: GPInferenceParams):
        self._gp_params = gp_params

    @abc.abstractmethod
    def __call__(
        self,
        solver_state: SolverState,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class Policy(abc.ABC):
    @abc.abstractmethod
    def get_concrete_policy(
        self, gp_params: GPInferenceParams, **kwargs
    ) -> ConcretePolicy:
        raise NotImplementedError
