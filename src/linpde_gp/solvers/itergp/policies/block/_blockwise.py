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
        mode="cycle",
    ):
        self._cur_block_idx = 0
        self._cur_num_iterations = 0
        self._num_iterations_per_block = num_iterations_per_block
        self._mode = mode

        self._block_policies = block_policies
        self._cur_policy = block_policies[0].get_concrete_policy(gp_params, 0)
        super().__init__(gp_params)

    def _next(self) -> bool:
        if self._mode == "cycle":
            num_tried = 0
            while self._cur_num_iterations >= self._num_iterations_per_block[
                self._cur_block_idx
            ] and num_tried < len(self._block_policies):
                self._cur_policy = self._block_policies[
                    self._cur_block_idx
                ].get_concrete_policy(self._gp_params, self._cur_block_idx)
                self._cur_num_iterations = 0
                self._cur_block_idx = (self._cur_block_idx + 1) % len(
                    self._block_policies
                )
                num_tried += 1
            if num_tried == len(self._block_policies):
                return False
        elif self._mode == "sequential":
            while self._cur_num_iterations >= self._num_iterations_per_block[
                self._cur_block_idx
            ] and self._cur_block_idx < len(self._block_policies):
                self._cur_policy = self._block_policies[
                    self._cur_block_idx
                ].get_concrete_policy(self._gp_params, self._cur_block_idx)
                self._cur_num_iterations = 0
                self._cur_block_idx += 1
            if self._cur_block_idx == len(self._block_policies):
                return False
        else:
            raise ValueError(f"Undefined mode: {self._mode}")
        return True

    def __call__(
        self, solver_state: SolverState, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if not self._next():
            return None

        self._cur_num_iterations += 1
        return self._cur_policy(solver_state, rng)


class BlockwisePolicy(Policy):
    def __init__(
        self,
        block_policies: List[BlockPolicy],
        num_iterations_per_block: List[int] | int,
        mode="cycle",
    ):
        self._block_policies = block_policies
        if isinstance(num_iterations_per_block, int):
            num_iterations_per_block = [num_iterations_per_block] * len(block_policies)
        self._num_iterations_per_block = num_iterations_per_block
        self._mode = mode
        super().__init__()

    def get_concrete_policy(self, gp_params: GPInferenceParams) -> ConcretePolicy:
        return ConcreteBlockwisePolicy(
            gp_params, self._block_policies, self._num_iterations_per_block, self._mode
        )
