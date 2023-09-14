from dataclasses import dataclass

import numpy as np
from linpde_gp.linops import LowRankProduct, RankFactorizedMatrix

from .._gp_solver import GPInferenceParams


@dataclass
class SolverState:
    iteration: int
    predictive_residual: np.ndarray
    representer_weights: np.ndarray
    inverse_approx: RankFactorizedMatrix
    K_hat_inverse_approx: LowRankProduct
    gp_params: GPInferenceParams
    relative_error: float
    marginal_uncertainty: np.ndarray | None = None
