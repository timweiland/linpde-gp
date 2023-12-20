import functools
from typing import Optional

import numpy as np
from jax import numpy as jnp
from linpde_gp.linfunctls import (
    CompositeLinearFunctional,
    LinearFunctional,
    _EvaluationFunctional,
)
from linpde_gp.linops import (
    BlockMatrix2x2,
    DenseCholeskySolverLinearOperator,
    CrosscovSandwich,
)
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction
from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance
from linpde_gp.randvars import LinearOperatorCovariance
from probnum import linops
from probnum.typing import ArrayLike
from probnum.linops._arithmetic_fallbacks import SumLinearOperator
from scipy.linalg import cho_factor, cho_solve

from ._gp_solver import ConcreteGPSolver, GPInferenceParams, GPSolver
from .covfuncs import DowndateCovarianceFunction


class CholeskyCovarianceFunction(DowndateCovarianceFunction):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        solve_fn: callable,
    ):
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

    def linop(
        self, x0: ArrayLike, x1: ArrayLike | None = None
    ) -> linops.LinearOperator:
        cho_linop = DenseCholeskySolverLinearOperator(self._gp_params.prior_gram)
        crosscov_x0 = self._gp_params.kLas.evaluate_linop(x0)
        crosscov_x1 = (
            self._gp_params.kLas.evaluate_linop(x1) if x1 is not None else crosscov_x0
        )
        if x1 is None:
            downdate_linop = CrosscovSandwich(crosscov_x0, cho_linop)
        else:
            downdate_linop = crosscov_x0 @ cho_linop @ crosscov_x1.T
        return SumLinearOperator(self._gp_params.prior.cov.linop(x0, x1), -downdate_linop, expand_sum=False)


class CholeskyCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        L: LinearFunctional,
        reverse=False,
    ):
        self._gp_params = gp_params
        self._cho_linop = DenseCholeskySolverLinearOperator(gp_params.prior_gram)
        self._L = L
        self._argnum = 0 if reverse else 1
        super().__init__(
            randproc_input_shape=self._gp_params.prior.cov.input_shape,
            randproc_output_shape=self._gp_params.prior.cov.output_shape_1
            if reverse
            else self._gp_params.prior.cov.output_shape_0,
            randvar_shape=L.output_shape,
            reverse=reverse,
        )

        self._L_prior = self._L(self._gp_params.prior.cov, argnum=self._argnum)
        self._L_kLas = self._L(self._gp_params.kLas).linop
        self._kLas = self._gp_params.kLas

    @property
    def cho_linop(self) -> linops.LinearOperator:
        return self._cho_linop

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.reverse:
            return (
                self._L_prior(x)
                - self._L_kLas @ self._cho_linop @ self._gp_params.kLas(x).T
            )
        return (
            self._L_prior(x)
            - self._gp_params.kLas(x) @ self._cho_linop @ self._L_kLas.T
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return None

    def _evaluate_linop(self, x: np.ndarray) -> linops.LinearOperator:
        if self.reverse:
            return (
                self._L_prior.evaluate_linop(x)
                - self._L_kLas
                @ self._cho_linop
                @ self._gp_params.kLas.evaluate_linop(x).T
            )
        return (
            self._L_prior.evaluate_linop(x)
            - self._gp_params.kLas.evaluate_linop(x) @ self._cho_linop @ self._L_kLas.T
        )


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
        super().__init__(gp_params, load_path, save_path)

    @property
    def dense(self) -> bool:
        return self._dense

    @functools.cached_property
    def dense_linop(self):
        if not self.dense:
            raise ValueError("Solver is not dense")
        return DenseCholeskySolverLinearOperator(self._gp_params.prior_gram)
    
    @property
    def inverse_approximation(self) -> linops.LinearOperator:
        return self.dense_linop

    def _compute_representer_weights(self):
        if self.dense:
            return self.dense_linop @ self._get_full_residual()
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
            else self.dense_linop @ x
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

    def __init__(
        self,
        load_path: str | None = None,
        save_path: str | None = None,
        dense: bool = False,
    ):
        self._dense = dense
        super().__init__(load_path, save_path)

    @property
    def dense(self) -> bool:
        return self._dense

    def get_concrete_solver(
        self, gp_params: GPInferenceParams
    ) -> ConcreteCholeskySolver:
        return ConcreteCholeskySolver(
            gp_params, self._load_path, self._save_path, self.dense
        )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
@_EvaluationFunctional.__call__.register
def _(
    self,
    cov: CholeskyCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    return CholeskyCrossCovariance(
        cov._gp_params,
        self,
        reverse=(argnum == 0),
    )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
@_EvaluationFunctional.__call__.register
def _(
    self,
    crosscov: CholeskyCrossCovariance,
    /,
) -> linops.LinearOperator:
    L1_L2_prior_cov = self(crosscov._L_prior).linop
    L2_kLas = self(crosscov._gp_params.kLas).linop
    L1_kLas = crosscov._L_kLas
    if crosscov.reverse:
        res = SumLinearOperator(L1_L2_prior_cov, -L1_kLas @ crosscov.cho_linop @ L2_kLas.T, expand_sum=False)
    else:
        res = SumLinearOperator(L1_L2_prior_cov, -L2_kLas @ crosscov.cho_linop @ L1_kLas.T, expand_sum=False)
    return LinearOperatorCovariance(
        res,
        shape0=crosscov.randvar_shape if crosscov.reverse else self.output_shape,
        shape1=self.output_shape if crosscov.reverse else crosscov.randvar_shape,
    )
