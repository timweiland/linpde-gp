import abc

from ..._gp_solver import GPInferenceParams
from .._solver_state import SolverState
from ._stopping_criterion import ConcreteStoppingCriterion, StoppingCriterion


class ConcreteIterationStoppingCriterion(ConcreteStoppingCriterion):
    def __init__(self, gp_params: GPInferenceParams, num_iterations: int):
        self._num_iterations = num_iterations
        super().__init__(gp_params)

    def __call__(self, solver_state: SolverState) -> bool:
        return solver_state.iteration >= self._num_iterations


class IterationStoppingCriterion(StoppingCriterion):
    def __init__(self, num_iterations: int):
        self._num_iterations = num_iterations
        super().__init__()

    def get_concrete_criterion(
        self, gp_params: GPInferenceParams
    ) -> ConcreteIterationStoppingCriterion:
        return ConcreteIterationStoppingCriterion(gp_params, self._num_iterations)
