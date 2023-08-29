from typing import Optional
from jax import numpy as jnp
from ._gp_solver import GPInferenceParams
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from linpde_gp.linops import BlockMatrix2x2
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction

from ._gp_solver import ConcreteGPSolver, GPInferenceParams, GPSolver
from .covfuncs import DowndateCovarianceFunction


class CholeskyCovarianceFunction(DowndateCovarianceFunction):
    def __init__(self, gp_params: GPInferenceParams, solve_fn: callable):
        self._gp_params = gp_params
        self._solve_fn = solve_fn
        super().__init__(gp_params.prior.cov)

    def _downdate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        kLas_x0 = self._gp_params.kLas(x0)
        kLas_x1 = self._gp_params.kLas(x1) if x1 is not None else kLas_x0

        x0_batch_ndim = x0.ndim - self._gp_params.prior.cov.input_ndim
        x1_batch_ndim = (
            x1.ndim - self._gp_params.prior.cov.input_ndim
            if x1 is not None
            else x0_batch_ndim
        )
        output_ndim_0 = self._gp_params.prior.cov.output_ndim_0
        output_ndim_1 = self._gp_params.prior.cov.output_ndim_1
        kLas_x0 = np.expand_dims(
            kLas_x0,
            axis=tuple(x0_batch_ndim + output_ndim_0 + np.arange(output_ndim_1)),
        )
        kLas_x1 = np.expand_dims(
            kLas_x1, axis=tuple(x1_batch_ndim + np.arange(output_ndim_0))
        )

        return (kLas_x0[..., None, :] @ (self._solve_fn(kLas_x1[..., None])))[..., 0, 0]

    def _downdate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        kLas_x0 = self._gp_params.kLas.jax(x0)
        kLas_x1 = self._gp_params.kLas.jax(x1) if x1 is not None else kLas_x0

        x0_batch_ndim = x0.ndim - self._gp_params.prior.cov.input_ndim
        x1_batch_ndim = (
            x1.ndim - self._gp_params.prior.cov.input_ndim
            if x1 is not None
            else x0_batch_ndim
        )
        output_ndim_0 = self._gp_params.prior.cov.output_ndim_0
        output_ndim_1 = self._gp_params.prior.cov.output_ndim_1
        kLas_x0 = jnp.expand_dims(
            kLas_x0,
            axis=tuple(x0_batch_ndim + output_ndim_0 + jnp.arange(output_ndim_1)),
        )
        kLas_x1 = jnp.expand_dims(
            kLas_x1, axis=tuple(x1_batch_ndim + jnp.arange(output_ndim_0))
        )

        return (kLas_x0[..., None, :] @ (self._solve_fn(kLas_x1[..., None])))[..., 0, 0]


class ConcreteCholeskySolver(ConcreteGPSolver):
    """Concrete solver that uses the Cholesky decomposition.

    Uses a block Cholesky decomposition if possible.
    """

    def __init__(
        self,
        gp_params: GPInferenceParams,
        load_path: str | None = None,
        save_path: str | None = None,
        dense: bool = False,
    ):
        self._dense = dense
        self._dense_cho_factor = None
        super().__init__(gp_params, load_path, save_path)

    @property
    def dense(self) -> bool:
        return self._dense

    @property
    def dense_cho_factor(self):
        if not self.dense:
            raise ValueError("Solver is not dense")
        if self._dense_cho_factor is None:
            self._dense_cho_factor = cho_factor(self._gp_params.prior_gram.todense())
        return self._dense_cho_factor

    def _compute_representer_weights(self):
        if self.dense:
            return cho_solve(self.dense_cho_factor, self._get_full_residual())
        if self._gp_params.prev_representer_weights is not None:
            # Update existing representer weights
            assert isinstance(self._gp_params.prior_gram, BlockMatrix2x2)
            new_residual = self._get_residual(
                self._gp_params.Ys[-1], self._gp_params.Ls[-1], self._gp_params.bs[-1]
            )
            return self._gp_params.prior_gram.schur_update(
                self._gp_params.prev_representer_weights, new_residual
            )
        full_residual = self._get_full_residual()
        return self._gp_params.prior_gram.solve(full_residual)

    @property
    def posterior_cov(self) -> JaxCovarianceFunction:
        solve_fn = (
            lambda x: self._gp_params.prior_gram.solve(x)
            if not self.dense
            else cho_solve(self.dense_cho_factor, x)
        )
        return CholeskyCovarianceFunction(self._gp_params, solve_fn)

    def _save_state(self) -> dict:
        # TODO: Actually save the Cholesky decomposition of the linear operator
        state = {"representer_weights": self._representer_weights}
        return state

    def _load_state(self, state: dict):
        self._representer_weights = state["representer_weights"]


class CholeskySolver(GPSolver):
    """Solver that uses the Cholesky decomposition."""
    def __init__(self, load_path: str | None = None, save_path: str | None = None, dense: bool = False):
        self._dense = dense
        super().__init__(load_path, save_path)
    
    @property
    def dense(self) -> bool:
        return self._dense

    def get_concrete_solver(
        self, gp_params: GPInferenceParams
    ) -> ConcreteCholeskySolver:
        return ConcreteCholeskySolver(gp_params, self._load_path, self._save_path, self.dense)
