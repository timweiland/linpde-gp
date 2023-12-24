from ._policy import ConcretePolicy, Policy
from .._solver_state import SolverState


class ConcreteVarianceCGPolicy(ConcretePolicy):
    def __init__(self, gp_params):
        self._gp_params = gp_params

    def __call__(self, solver_state: SolverState, rng=None):
        residual = solver_state.crosscov_residual
        return residual


class VarianceCGPolicy(Policy):
    def get_concrete_policy(self, gp_params, **kwargs):
        return ConcreteVarianceCGPolicy(gp_params)
