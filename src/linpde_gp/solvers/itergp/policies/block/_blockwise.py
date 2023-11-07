import abc
from typing import List, Optional, Tuple

import numpy as np
from linpde_gp.linops import BlockMatrix2x2

from ...._gp_solver import GPInferenceParams
from ..._solver_state import SolverState
from .._policy import ConcretePolicy, Policy


class ConcreteBlockPolicy:
    def __init__(self, gp_params: GPInferenceParams, block_idx: int):
        self._block_idx = block_idx
        self._gp_params = gp_params
        assert isinstance(gp_params.prior_gram, BlockMatrix2x2)
        block = gp_params.prior_gram.diagonal_blocks[block_idx]
        self._block_start = int(
            np.sum(
                [b.shape[1] for b in gp_params.prior_gram.diagonal_blocks[:block_idx]]
            )
        )
        self._block_end = self._block_start + block.shape[1]

    @abc.abstractmethod
    def __call__(
        self, solver_state: SolverState, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        raise NotImplementedError


class BlockPolicy:
    @abc.abstractmethod
    def get_concrete_policy(
        self, gp_params: GPInferenceParams, block_idx: int
    ) -> ConcreteBlockPolicy:
        raise NotImplementedError


class ConcreteBlockwisePolicy(ConcretePolicy):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        block_policies: List[BlockPolicy],
        num_iterations_per_block: List[int],
        cycle_schedule: List[int],
    ):
        self._cur_num_iterations = 0
        self._num_iterations_per_block = num_iterations_per_block
        self._cycle_schedule = cycle_schedule
        self._cycle_idx = 0
        self._cur_block_idx = cycle_schedule[0]
        self._total_iterations_per_block = [0] * len(block_policies)
        self._max_iterations_per_block = [
            block.shape[1] for block in gp_params.prior_gram.diagonal_blocks
        ]

        self._block_policies = block_policies
        self._cur_policy = block_policies[self._cur_block_idx].get_concrete_policy(
            gp_params, self._cur_block_idx
        )
        super().__init__(gp_params)

    def _next(self) -> bool:
        while (
            self._cur_num_iterations
            >= self._num_iterations_per_block[self._cur_block_idx]
            or self._total_iterations_per_block[self._cur_block_idx]
            >= self._max_iterations_per_block[self._cur_block_idx]
        ):
            self._cycle_idx += 1
            self._cur_block_idx = self._cycle_schedule[
                self._cycle_idx % len(self._cycle_schedule)
            ]
            self._cur_num_iterations = 0
            self._cur_policy = self._block_policies[
                self._cur_block_idx
            ].get_concrete_policy(self._gp_params, self._cur_block_idx)

        return True

    def __call__(
        self, solver_state: SolverState, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if not self._next():
            return None

        self._cur_num_iterations += 1
        self._total_iterations_per_block[self._cur_block_idx] += 1
        return self._cur_policy(solver_state, rng)


class BlockwisePolicy(Policy):
    def __init__(
        self,
        block_policies: List[BlockPolicy],
        num_iterations_per_block: List[int] | int,
        cycle_schedule: List[int],
    ):
        self._block_policies = block_policies
        if isinstance(num_iterations_per_block, int):
            num_iterations_per_block = [num_iterations_per_block] * len(block_policies)
        self._num_iterations_per_block = num_iterations_per_block
        self._cycle_schedule = cycle_schedule
        super().__init__()

    def get_concrete_policy(self, gp_params: GPInferenceParams) -> ConcretePolicy:
        return ConcreteBlockwisePolicy(
            gp_params,
            self._block_policies,
            self._num_iterations_per_block,
            self._cycle_schedule,
        )
