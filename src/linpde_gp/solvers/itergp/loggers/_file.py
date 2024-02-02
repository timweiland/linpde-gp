from ._logger import Logger
from .._solver_state import SolverState

import torch

from pathlib import Path


class FileLogger(Logger):
    def __init__(self, filepath):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        # Make file
        with open(self._filepath, "w") as f:
            f.write("")

    def start(self, gp_params):
        super().start(gp_params)

    def __call__(self, solver_state: SolverState):
        mem_gb = torch.cuda.memory_allocated(0) / 1e9

        line_str = f"Iteration {solver_state.iteration + 1}: Rel. error: {solver_state.relative_error:.2e}"
        if solver_state.relative_crosscov_error is not None:
            line_str += f" | Rel. crosscov error: {solver_state.relative_crosscov_error:.2e}"
        line_str += f" | Memory {mem_gb:.2f} GB."
        with open(self._filepath, "a") as f:
            f.write(line_str + "\n")

    def finish(self):
        with open(self._filepath, "a") as f:
            f.write("Done.\n")
        return
