from ._itergp import IterGPSolver
from typing import Iterable
from .policies import CGPolicy
from .loggers import Logger, TQDMLogger
from .stopping_criteria import IterationStoppingCriterion, ResidualNormStoppingCriterion
import numpy as np


class IterGP_CG_Solver(IterGPSolver):
    def __init__(
        self,
        max_iterations: int = 1000,
        threshold=1e-2,
        *,
        eval_points: np.ndarray = None,
        benchmark_folder: str | None = None,
        use_torch=True,
        compute_residual_directly=False,
        preconditioner=None,
        num_actions_compressed=100,
        num_actions_explorative=10,
        loggers: Iterable[Logger] = [TQDMLogger(notebook=True)],
    ):
        policy = CGPolicy()
        stopping_criterion = IterationStoppingCriterion(
            max_iterations
        ) | ResidualNormStoppingCriterion(threshold)
        super().__init__(
            policy,
            stopping_criterion,
            eval_points=eval_points,
            benchmark_folder=benchmark_folder,
            use_torch=use_torch,
            compute_residual_directly=compute_residual_directly,
            num_actions_compressed=num_actions_compressed,
            num_actions_explorative=num_actions_explorative,
            loggers=loggers,
        )
