from ._policy import ConcretePolicy, Policy
from .._solver_state import SolverState

import torch


class ConcretePredefinedPolicy(ConcretePolicy):
    def __init__(self, gp_params, policy_vecs: torch.Tensor):
        self._gp_params = gp_params
        self._policy_vecs = policy_vecs
        self._start_iteration = None

    def __call__(self, solver_state: SolverState, rng=None):
        if self._start_iteration is None:
            self._start_iteration = solver_state.iteration
        return self._policy_vecs[solver_state.iteration - self._start_iteration]


class PredefinedPolicy(Policy):
    def __init__(self, policy_vecs: torch.Tensor):
        assert policy_vecs.ndim == 2
        self._policy_vecs = policy_vecs

    def get_concrete_policy(self, gp_params, **kwargs):
        return ConcretePredefinedPolicy(gp_params, self._policy_vecs)
