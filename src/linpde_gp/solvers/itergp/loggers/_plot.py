from ._logger import Logger
from .._solver_state import SolverState

import torch

from matplotlib import pyplot as plt


class PlotLogger(Logger):
    def __init__(self):
        self._rel_errors = []
        self._mems = []

    def start(self, gp_params):
        super().start(gp_params)

    def __call__(self, solver_state: SolverState):
        mem_gb = torch.cuda.memory_allocated(0) / 1e9
        self._mems.append(mem_gb)
        self._rel_errors.append(solver_state.relative_error.cpu().numpy())

    def finish(self):
        return

    def plot_error(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self._rel_errors)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative error")
        return fig, ax
    
    def plot_memory(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self._mems)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Memory (GB)")
        return fig, ax