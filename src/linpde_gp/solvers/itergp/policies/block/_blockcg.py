from typing import List, Optional, Tuple

import numpy as np
from linpde_gp.linops import BlockMatrix2x2
import torch

from ...._gp_solver import GPInferenceParams
from ..._solver_state import SolverState
from .._policy import ConcretePolicy, Policy
from ._blockwise import BlockPolicy, ConcreteBlockPolicy


class ConcreteBlockCG(ConcreteBlockPolicy):
    def __init__(self, gp_params: GPInferenceParams, block_idx: int):
        super().__init__(gp_params, block_idx)

    def __call__(
        self, solver_state: SolverState, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if isinstance(solver_state.predictive_residual, np.ndarray):
            residual = np.copy(solver_state.predictive_residual)
        else:
            residual = torch.clone(solver_state.predictive_residual)
        residual[: self._block_start] = 0.0
        residual[self._block_end :] = 0.0
        return residual


class BlockCG(BlockPolicy):
    def get_concrete_policy(
        self, gp_params: GPInferenceParams, block_idx: int
    ) -> ConcretePolicy:
        return ConcreteBlockCG(gp_params, block_idx)
