import abc
from typing import Any

from ..._gp_solver import GPInferenceParams
from .._solver_state import SolverState


class ConcreteStoppingCriterion(abc.ABC):
    def __init__(self, gp_params: GPInferenceParams):
        self._gp_params = gp_params

    @abc.abstractmethod
    def __call__(
        self,
        solver_state: SolverState,
    ) -> bool:
        raise NotImplementedError


class StoppingCriterion(abc.ABC):
    @abc.abstractmethod
    def get_concrete_criterion(
        self, gp_params: GPInferenceParams
    ) -> ConcreteStoppingCriterion:
        raise NotImplementedError

    def __or__(self, other: Any):
        if isinstance(other, StoppingCriterion):
            from ._arithmetic import (
                ORStoppingCriterion,
            )  # pylint: disable=import-outside-toplevel

            return ORStoppingCriterion(self, other)
        return NotImplemented

    def __and__(self, other: Any):
        if isinstance(other, StoppingCriterion):
            from ._arithmetic import (
                ANDStoppingCriterion,
            )  # pylint: disable=import-outside-toplevel

            return ANDStoppingCriterion(self, other)
        return NotImplemented
