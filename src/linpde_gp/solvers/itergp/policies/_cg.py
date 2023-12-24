from ._policy import ConcretePolicy, Policy
from .._solver_state import SolverState


class ConcreteCGPolicy(ConcretePolicy):
    def __init__(self, gp_params):
        self._gp_params = gp_params

    def __call__(self, solver_state: SolverState, rng=None):
        residual = solver_state.predictive_residual
        return residual


class CGPolicy(Policy):
    def get_concrete_policy(self, gp_params, **kwargs):
        return ConcreteCGPolicy(gp_params)
