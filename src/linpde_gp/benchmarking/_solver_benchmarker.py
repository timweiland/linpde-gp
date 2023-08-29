from timeit import default_timer as timer

import numpy as np
import probnum as pn
from tqdm import tqdm


class SolverResult:
    def __init__(self, solve_fn: callable, error_estimator: callable):
        start_time = timer()
        posterior = solve_fn()
        self._solve_time = timer() - start_time
        self._error = error_estimator(posterior)
        self._N = posterior._gram_matrix.shape[1]

    @property
    def solve_time(self):
        return self._solve_time

    @property
    def error(self):
        return self._error

    @property
    def N(self):
        return self._N
    
    def __repr__(self):
        return f"SolverResult(solve_time={self.solve_time:.2f}, error={self.error:.2E}, N={self.N})"


class SolverBenchmarker:
    def __init__(self, prior: pn.randprocs.GaussianProcess, error_estimator: callable):
        self._prior = prior
        self._error_estimator = error_estimator
        self._solver_args = {}
        self._solver_results = {}

    @property
    def error_estimator(self):
        return self._error_estimator

    @property
    def prior(self):
        return self._prior

    @property
    def solver_args(self):
        return self._solver_args

    @property
    def solver_results(self) -> dict[str, list[SolverResult]]:
        return self._solver_results

    def __call__(self, solver: callable, args_grid, name: str):
        if name not in self._solver_args:
            self._solver_args[name] = []
            self._solver_results[name] = []

        for args in tqdm(args_grid):
            self._solver_results[name].append(
                SolverResult(lambda: solver(self.prior, **args), self.error_estimator)
            )
            self._solver_args[name].append(args)

    def best_args(self, name: str):
        return self._solver_args[name][np.argmin([r.error for r in self._solver_results[name]])]
