from typing import Optional

import numpy as np

from ...._gp_solver import GPInferenceParams
from ..._solver_state import SolverState
from ._blockwise import BlockPolicy, ConcreteBlockPolicy


class ConcreteBlockCholesky(ConcreteBlockPolicy):
    def __init__(self, gp_params: GPInferenceParams, block_idx: int):
        super().__init__(gp_params, block_idx)
        self._num_iterations = self._block_end - self._block_start
        assert self._num_iterations > 0
        self._cur_iteration = 0

    def __call__(
        self, solver_state: SolverState, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if self._cur_iteration >= self._num_iterations:
            return None
        action = np.zeros(self._gp_params.prior_gram.shape[1])
        action[self._block_start + self._cur_iteration] = 1.0
        self._cur_iteration += 1
        return action


class BlockCholesky(BlockPolicy):
    def get_concrete_policy(
        self, gp_params: GPInferenceParams, block_idx: int
    ) -> ConcreteBlockCholesky:
        return ConcreteBlockCholesky(gp_params, block_idx)
