from ._logger import Logger
from .._solver_state import SolverState

import torch
import pickle

from pathlib import Path


class PickleLogger(Logger):
    def __init__(self, filepath):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        self._rel_errors = []
        self._mems = []

    def start(self, gp_params):
        super().start(gp_params)

    def __call__(self, solver_state: SolverState):
        mem_gb = torch.cuda.memory_allocated(0) / 1e9
        self._mems.append(mem_gb)
        self._rel_errors.append(solver_state.relative_error.cpu().numpy())

    def finish(self):
        data_dict = {
            "rel_errors": self._rel_errors,
            "mems": self._mems
        }
        with open(self._filepath, "wb") as f:
            pickle.dump(data_dict, f)

        return
