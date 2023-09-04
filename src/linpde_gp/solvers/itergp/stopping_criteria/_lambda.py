from typing import Callable

from ..._gp_solver import GPInferenceParams
from .._solver_state import SolverState
from ._stopping_criterion import ConcreteStoppingCriterion, StoppingCriterion


class ConcreteLambdaStoppingCriterion(ConcreteStoppingCriterion):
    def __init__(
        self, gp_params: GPInferenceParams, func: Callable[[SolverState], bool]
    ):
        self._func = func
        super().__init__(gp_params)

    def __call__(self, solver_state: SolverState) -> bool:
        return self._func(solver_state)


class LambdaStoppingCriterion(StoppingCriterion):
    def __init__(self, func: Callable[[SolverState], bool]):
        self._func = func
        super().__init__()

    def get_concrete_criterion(
        self, gp_params: GPInferenceParams
    ) -> ConcreteStoppingCriterion:
        return ConcreteLambdaStoppingCriterion(gp_params, self._func)
