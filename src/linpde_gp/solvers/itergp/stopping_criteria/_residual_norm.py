from ..._gp_solver import GPInferenceParams
from .._solver_state import SolverState
from ._stopping_criterion import ConcreteStoppingCriterion, StoppingCriterion


class ConcreteResidualNormStoppingCriterion(ConcreteStoppingCriterion):
    def __init__(self, gp_params: GPInferenceParams, threshold: float):
        self._threshold = threshold
        super().__init__(gp_params)

    def __call__(self, solver_state: SolverState) -> bool:
        return solver_state.relative_error < self._threshold


class ResidualNormStoppingCriterion(StoppingCriterion):
    def __init__(self, threshold: float):
        self._threshold = threshold
        super().__init__()

    def get_concrete_criterion(
        self, gp_params: GPInferenceParams
    ) -> ConcreteStoppingCriterion:
        return ConcreteResidualNormStoppingCriterion(gp_params, self._threshold)
