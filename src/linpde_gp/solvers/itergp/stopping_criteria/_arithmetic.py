from ..._gp_solver import GPInferenceParams
from .._solver_state import SolverState
from ._stopping_criterion import ConcreteStoppingCriterion, StoppingCriterion


class ConcreteORStoppingCriterion(ConcreteStoppingCriterion):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        criterion_A: ConcreteStoppingCriterion,
        criterion_B: ConcreteStoppingCriterion,
    ):
        self._criterion_A = criterion_A
        self._criterion_B = criterion_B
        super().__init__(gp_params)

    def __call__(self, solver_state: SolverState) -> bool:
        return self._criterion_A(solver_state) or self._criterion_B(solver_state)


class ORStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion_A: StoppingCriterion, criterion_B: StoppingCriterion):
        self._criterion_A = criterion_A
        self._criterion_B = criterion_B
        super().__init__()

    def get_concrete_criterion(
        self, gp_params: GPInferenceParams
    ) -> ConcreteORStoppingCriterion:
        concrete_A = self._criterion_A.get_concrete_criterion(gp_params)
        concrete_B = self._criterion_B.get_concrete_criterion(gp_params)
        return ConcreteORStoppingCriterion(gp_params, concrete_A, concrete_B)


class ConcreteANDStoppingCriterion(ConcreteStoppingCriterion):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        criterion_A: ConcreteStoppingCriterion,
        criterion_B: ConcreteStoppingCriterion,
    ):
        self._criterion_A = criterion_A
        self._criterion_B = criterion_B
        super().__init__(gp_params)

    def __call__(self, solver_state: SolverState) -> bool:
        return self._criterion_A(solver_state) and self._criterion_B(solver_state)


class ANDStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion_A: StoppingCriterion, criterion_B: StoppingCriterion):
        self._criterion_A = criterion_A
        self._criterion_B = criterion_B
        super().__init__()

    def get_concrete_criterion(
        self, gp_params: GPInferenceParams
    ) -> ConcreteANDStoppingCriterion:
        concrete_A = self._criterion_A.get_concrete_criterion(gp_params)
        concrete_B = self._criterion_B.get_concrete_criterion(gp_params)
        return ConcreteANDStoppingCriterion(gp_params, concrete_A, concrete_B)
