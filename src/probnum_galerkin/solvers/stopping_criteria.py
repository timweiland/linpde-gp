import abc

import numpy as np
import probnum as pn
from probnum.typing import FloatArgType


class StoppingCriterion(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: "probnum_galerkin.solvers.beliefs.LinearSystemBelief",
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> bool:
        pass


class MaxIterations(StoppingCriterion):
    def __init__(self, maxiter: int) -> None:
        self._maxiter = maxiter

    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: "probnum_galerkin.solvers.beliefs.LinearSystemBelief",
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> bool:
        return solver_state.iteration >= self._maxiter


class ResidualNorm(StoppingCriterion):
    def __init__(self, atol: FloatArgType = 1e-5, rtol: FloatArgType = 1e-5) -> None:
        self.atol = pn.utils.as_numpy_scalar(atol)
        self.rtol = pn.utils.as_numpy_scalar(rtol)

    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: "probnum_galerkin.solvers.beliefs.LinearSystemBelief",
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> bool:
        # Compare residual to tolerances
        b_norm = np.linalg.norm(problem.b, ord=2)

        return (
            solver_state.residual_norm <= self.atol
            or solver_state.residual_norm <= self.rtol * b_norm
        )
