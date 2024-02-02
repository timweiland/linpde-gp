from dataclasses import dataclass

import numpy as np
from linpde_gp.linops import OuterProduct

from .._gp_solver import GPInferenceParams


@dataclass
class SolverState:
    iteration: int
    predictive_residual: np.ndarray
    representer_weights: np.ndarray
    inverse_approx: OuterProduct
    action_matrix: np.ndarray
    K_hat_inverse_approx: OuterProduct
    gp_params: GPInferenceParams
    relative_error: float
    relative_crosscov_error: float | None = None
    S_LKL_S: np.ndarray | None = None
    crosscov_residual: np.ndarray | None = None
