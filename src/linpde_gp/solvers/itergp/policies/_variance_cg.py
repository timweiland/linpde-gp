from ._policy import ConcretePolicy, Policy
from .._solver_state import SolverState
import numpy as np
import torch

from linpde_gp.linfunctls import (
    _EvaluationFunctional,
)


class ConcreteVarianceCGPolicy(ConcretePolicy):
    def __init__(self, gp_params, target_points):
        self._gp_params = gp_params
        self._target_points = target_points

        eval_fctl = _EvaluationFunctional(
            self._gp_params.prior.input_shape,
            self._gp_params.prior.output_shape,
            target_points,
        )
        self._eval_crosscov = eval_fctl(self._gp_params.kLas).linop

    def __call__(self, solver_state: SolverState, rng=None):
        weights = np.random.dirichlet(np.ones(self._eval_crosscov.shape[0]))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = torch.tensor(weights, dtype=torch.float64, device=device)

        target_rhs = self._eval_crosscov.T @ weights
        residual = target_rhs - (
            self._gp_params.prior_gram @ (solver_state.inverse_approx @ target_rhs)
        )
        residual_norm = residual.norm(p=2)
        print(f"Residual norm: {residual_norm}")
        return residual


class VarianceCGPolicy(Policy):
    def __init__(self, target_points):
        self._target_points = target_points

    def get_concrete_policy(self, gp_params, **kwargs):
        return ConcreteVarianceCGPolicy(gp_params, self._target_points)
