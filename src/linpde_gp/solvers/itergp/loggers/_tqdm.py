from ._logger import Logger
from .._solver_state import SolverState

from tqdm import tqdm
from tqdm import tqdm_notebook

import torch


class TQDMLogger(Logger):
    def __init__(self, notebook=False):
        self._notebook = notebook
        self._pbar = None

    def start(self, gp_params):
        super().start(gp_params)
        if self._notebook:
            self._pbar = tqdm_notebook(total=gp_params.prior_gram.shape[1])
        else:
            self._pbar = tqdm(total=gp_params.prior_gram.shape[1])

    def __call__(self, solver_state: SolverState):
        mem_gb = torch.cuda.memory_allocated(0) / 1e9
        memory_str = f"{mem_gb:.2f}"

        description_str = f"Rel. error: {solver_state.relative_error:.2e}. Rel. crosscov error: {solver_state.relative_crosscov_error:.2e}. Memory {memory_str} GB."
        self._pbar.set_description(description_str)
        self._pbar.update(1)

    def finish(self):
        self._pbar.close()
