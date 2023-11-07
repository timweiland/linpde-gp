from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp
import numpy as np
from tqdm.notebook import tqdm
import torch

from linpde_gp.linfunctls import (
    CompositeLinearFunctional,
    LinearFunctional,
    _EvaluationFunctional,
)
from linpde_gp.linops import (
    LinearOperator,
    LowRankProduct,
    RankFactorizedMatrix,
    BlockDiagonalMatrix,
)
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction
from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance
from linpde_gp.randvars import LinearOperatorCovariance

from .._gp_solver import ConcreteGPSolver, GPInferenceParams, GPSolver
from .._solver_benchmarker import SolverBenchmarker
from ..covfuncs import DowndateCovarianceFunction
from ._solver_state import SolverState
from .policies import CGPolicy, Policy
from .stopping_criteria import (
    IterationStoppingCriterion,
    ResidualNormStoppingCriterion,
    StoppingCriterion,
)

import probnum as pn


class IterGPCovarianceFunction(DowndateCovarianceFunction):
    def __init__(self, gp_params: GPInferenceParams, solver_state: SolverState):
        self._gp_params = gp_params
        self._solver_state = solver_state
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

        if x1 is None:
            k_xX_U = kLas_x0[..., None, :] @ self._solver_state.inverse_approx._U
            # k_U_T_Xx is just k_xX_U, but with a different layout.
            target_axes = list(np.arange(len(k_xX_U.shape)))
            output_0_start = x0_batch_ndim
            output_0_stop = output_0_start + output_ndim_0
            output_1_start = output_0_stop
            output_1_stop = output_1_start + output_ndim_1
            # Keep the batch components, swap the output components for correct
            # broadcasting, and swap the final two components to get the
            # transpose
            target_axes = (
                target_axes[:output_0_start]
                + target_axes[output_1_start:output_1_stop]
                + target_axes[output_0_start:output_0_stop]
                + [target_axes[output_1_stop + 1]]
                + [target_axes[output_1_stop]]
            )
            k_U_T_Xx = np.transpose(k_xX_U, target_axes)
            return (k_xX_U @ k_U_T_Xx)[..., 0, 0]

        return (
            kLas_x0[..., None, :]
            @ (self.solver_state.inverse_approx @ kLas_x1[..., None])
        )[..., 0, 0]

    def _downdate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        pass


class IterGPCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        inverse_approx: LinearOperator,
        L: LinearFunctional,
        reverse=False,
    ):
        self._gp_params = gp_params
        self._inverse_approx = inverse_approx
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

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.reverse:
            return (
                self._L_prior(x)
                - self._L_kLas @ self._inverse_approx @ self._gp_params.kLas(x).T
            )
        return (
            self._L_prior(x)
            - self._gp_params.kLas(x) @ self._inverse_approx @ self._L_kLas.T
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return None

    def _evaluate_linop(self, x: np.ndarray) -> LinearOperator:
        if self.reverse:
            return (
                self._L_prior.evaluate_linop(x)
                - self._L_kLas
                @ self._inverse_approx
                @ self._gp_params.kLas.evaluate_linop(x).T
            )
        return (
            self._L_prior.evaluate_linop(x)
            - self._gp_params.kLas.evaluate_linop(x)
            @ self._inverse_approx
            @ self._L_kLas.T
        )


class ConcreteIterGPSolver(ConcreteGPSolver):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        policy: Policy,
        stopping_criterion: StoppingCriterion,
        *,
        eval_points: np.ndarray = None,
        benchmark_folder: str | None = None,
        use_torch=True,
        compute_residual_directly=False,
        store_K_hat_inverse_approx=False,
        preconditioner=None,
        inverse_approx_initial_capacity=1430,
    ):
        self.policy = policy.get_concrete_policy(gp_params)
        self.stopping_criterion = stopping_criterion.get_concrete_criterion(gp_params)
        self.solver_state = SolverState(0, None, None, None, None, gp_params, None)
        self.eval_points = eval_points
        self.benchmark_folder = benchmark_folder
        self.use_torch = use_torch
        self.compute_residual_directly = compute_residual_directly
        self.store_K_hat_inverse_approx = store_K_hat_inverse_approx
        self.preconditioner = preconditioner
        self.inverse_approx_initial_capacity = inverse_approx_initial_capacity
        super().__init__(gp_params)

    def _compute_representer_weights(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        residual = self._get_full_residual()
        if self.use_torch:
            residual = torch.from_numpy(residual).to(device)
        K_hat = self._gp_params.prior_gram

        self.solver_state.representer_weights = (
            np.zeros((K_hat.shape[1]))
            if not self.use_torch
            else torch.zeros((K_hat.shape[1]), dtype=torch.float64).to(device)
        )
        residual_norm = (
            np.linalg.norm(residual, ord=2)
            if not self.use_torch
            else torch.norm(residual, p=2)
        )
        self.solver_state.predictive_residual = torch.clone(residual)
        self.solver_state.relative_error = 1.0
        min_error = 1.0

        cur_inverse_approx_term = RankFactorizedMatrix(
            None, K_hat.shape, np.float64, self.inverse_approx_initial_capacity
        )

        self.solver_state.inverse_approx = cur_inverse_approx_term

        new_start = 0
        if self._gp_params.prior_inverse_approx is not None:
            desired_dim = K_hat.shape[1]
            prior_dim = self._gp_params.prior_inverse_approx.shape[1]
            new_start = prior_dim
            self.solver_state.inverse_approx += BlockDiagonalMatrix(
                self._gp_params.prior_inverse_approx,
                pn.linops.Zero((desired_dim - prior_dim, desired_dim - prior_dim)),
            )
            assert self._gp_params.prev_representer_weights is not None
            prev_representer_weights = self._gp_params.prev_representer_weights
            if self.use_torch:
                prev_representer_weights = torch.from_numpy(
                    prev_representer_weights
                ).to(device)
            self.solver_state.representer_weights[
                :prior_dim
            ] += prev_representer_weights
            self.store_K_hat_inverse_approx = False

            self.solver_state.predictive_residual = (
                residual - K_hat @ self.solver_state.representer_weights
            )

            residual_norm = (
                np.linalg.norm(self.solver_state.predictive_residual[new_start:], ord=2)
                if not self.use_torch
                else torch.norm(self.solver_state.predictive_residual[new_start:], p=2)
            )

        if self.store_K_hat_inverse_approx:
            self.solver_state.K_hat_inverse_approx = LowRankProduct(
                None, None, K_hat.shape, np.float64
            )

        benchmarker = SolverBenchmarker(self.benchmark_folder)
        benchmarker.start_benchmark()

        pbar = tqdm(total=K_hat.shape[1])

        if self.eval_points is not None:
            if self._gp_params.prior_marginal_uncertainty is None:
                self.solver_state.marginal_uncertainty = (
                    self._gp_params.prior.cov.linop(self.eval_points).diagonal().copy()
                )
            else:
                self.solver_state.marginal_uncertainty = (
                    self._gp_params.prior_marginal_uncertainty.copy()
                )

            if self.use_torch:
                self.solver_state.marginal_uncertainty = torch.from_numpy(
                    self.solver_state.marginal_uncertainty
                ).to(device)
            eval_fctl = _EvaluationFunctional(
                self._gp_params.prior.input_shape,
                self._gp_params.prior.output_shape,
                self.eval_points,
            )
            eval_crosscov = eval_fctl(self._gp_params.kLas).linop
        while (not self.stopping_criterion(self.solver_state)) and (
            self.solver_state.iteration < K_hat.shape[1]
        ):
            action = self.policy(self.solver_state)
            if action is None:  # Policy ran out of actions, quit early
                break
            if self.preconditioner is not None:
                action = self.preconditioner @ action
            alpha = (
                torch.dot(action, self.solver_state.predictive_residual)
                if self.use_torch
                else np.dot(action, self.solver_state.predictive_residual)
            )
            K_hat_action = K_hat @ action
            C_K_hat_action = self.solver_state.inverse_approx @ K_hat_action
            if self.store_K_hat_inverse_approx:
                K_hat_C_K_hat_action = (
                    self.solver_state.K_hat_inverse_approx @ K_hat_action
                )
            else:
                K_hat_C_K_hat_action = K_hat @ C_K_hat_action
            search_direction = action - C_K_hat_action
            K_hat_search_direction = K_hat_action - K_hat_C_K_hat_action
            normalization_constant = (
                torch.dot(action, K_hat_search_direction)
                if self.use_torch
                else np.dot(action, K_hat_search_direction)
            )

            action_T_action = (
                torch.dot(action, action) if self.use_torch else np.dot(action, action)
            )
            rayleigh = normalization_constant / action_T_action
            normalization_constant = (
                1.0 / normalization_constant
                if normalization_constant > 0.0
                else (torch.tensor(0.0) if self.use_torch else 0.0)
            )
            sqrt_normalization_constant = (
                np.sqrt(normalization_constant)
                if not self.use_torch
                else torch.sqrt(normalization_constant)
            )
            cur_inverse_approx_term.append_factor(
                search_direction * sqrt_normalization_constant
            )
            if self.store_K_hat_inverse_approx:
                self.solver_state.K_hat_inverse_approx.append_factors(
                    K_hat_search_direction * sqrt_normalization_constant,
                    search_direction * sqrt_normalization_constant,
                )

            self.solver_state.representer_weights += (
                alpha * normalization_constant
            ) * search_direction
            if not self.compute_residual_directly:
                self.solver_state.predictive_residual -= (
                    alpha * normalization_constant
                ) * K_hat_search_direction
            else:
                self.solver_state.predictive_residual = (
                    residual - K_hat @ self.solver_state.representer_weights
                )

            if normalization_constant < 0:
                print(f"Warning: Normalization constant < 0.")

            if self.eval_points is not None:
                self.solver_state.marginal_uncertainty -= (
                    eval_crosscov @ (sqrt_normalization_constant * search_direction)
                ) ** 2

            if self.use_torch:
                self.solver_state.relative_error = (
                    torch.norm(self.solver_state.predictive_residual[new_start:], p=2)
                    / residual_norm
                )
            else:
                self.solver_state.relative_error = (
                    np.linalg.norm(
                        self.solver_state.predictive_residual[new_start:], ord=2
                    )
                    / residual_norm
                )

            if self.solver_state.relative_error < min_error:
                min_error = self.solver_state.relative_error

            metrics = {
                "relative_error": self.solver_state.relative_error,
                "rayleigh": rayleigh,
            }
            if len(self._gp_params.Ls) > 1:
                block_errors = []
                cur_block_start = 0
                for i in range(len(self._gp_params.Ls)):
                    cur_block_end = cur_block_start + K_hat.diagonal_blocks[i].shape[1]
                    block_errors.append(
                        torch.norm(
                            self.solver_state.predictive_residual[
                                cur_block_start:cur_block_end
                            ],
                            p=2,
                        )
                        / torch.norm(residual[cur_block_start:cur_block_end], p=2)
                    )
                    cur_block_start = cur_block_end
            else:
                block_errors = [self.solver_state.relative_error]

            block_errors_str = ", ".join(
                [f"{block_err:.2f}" for block_err in block_errors]
            )

            mem_gb = torch.cuda.memory_allocated(0) / 1e9 if self.use_torch else 0.0
            memory_str = f"{mem_gb:.2f}" if self.use_torch else "N/A"

            benchmarker.log_metric(metrics)
            pbar.set_description(
                f"Relative error {self.solver_state.relative_error:.2f}, min {min_error:.2f}, Rayleigh {rayleigh:.2f}. Block errors: {block_errors_str}. Mem: {memory_str} "
            )
            pbar.update(1)
            self.solver_state.iteration += 1

        pbar.close()
        benchmarker.save_values()
        if self.use_torch:
            self.solver_state.representer_weights = (
                self.solver_state.representer_weights.cpu().numpy()
            )
        return self.solver_state.representer_weights

    def compute_posterior_cov(self, x0: np.ndarray, x1: Optional[np.ndarray]):
        self.compute_representer_weights()

        k_xx = self._gp_params.prior.cov(x0, x1)
        kLas_x0 = self._gp_params.kLas(x0)
        kLas_x1 = self._kLas(x1) if x1 is not None else kLas_x0

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

        if x1 is None:
            k_xX_U = kLas_x0[..., None, :] @ self.solver_state.inverse_approx._U
            # k_U_T_Xx is just k_xX_U, but with a different layout.
            target_axes = list(np.arange(len(k_xX_U.shape)))
            output_0_start = x0_batch_ndim
            output_0_stop = output_0_start + output_ndim_0
            output_1_start = output_0_stop
            output_1_stop = output_1_start + output_ndim_1
            # Keep the batch components, swap the output components for correct
            # broadcasting, and swap the final two components to get the
            # transpose
            target_axes = (
                target_axes[:output_0_start]
                + target_axes[output_1_start:output_1_stop]
                + target_axes[output_0_start:output_0_stop]
                + [target_axes[output_1_stop + 1]]
                + [target_axes[output_1_stop]]
            )
            k_U_T_Xx = np.transpose(k_xX_U, target_axes)
            return k_xx - (k_xX_U @ k_U_T_Xx)[..., 0, 0]

        return (
            k_xx
            - (
                kLas_x0[..., None, :]
                @ (self.solver_state.inverse_approx @ kLas_x1[..., None])
            )[..., 0, 0]
        )

    def _load_state(self, state: dict):
        pass

    def _save_state(self) -> dict:
        pass

    @property
    def posterior_cov(self) -> JaxCovarianceFunction:
        return IterGPCovarianceFunction(self._gp_params, self.solver_state)

    @property
    def inverse_approximation(self) -> RankFactorizedMatrix:
        if self.solver_state.inverse_approx is None:
            self.compute_representer_weights()
        return self.solver_state.inverse_approx


class IterGPSolver(GPSolver):
    def __init__(
        self,
        policy: Policy,
        stopping_criterion: StoppingCriterion,
        *,
        eval_points: np.ndarray = None,
        benchmark_folder: str | None = None,
        use_torch=True,
        compute_residual_directly=False,
        preconditioner=None,
        inverse_approx_initial_capacity=1430,
    ):
        self.policy = policy
        self.stopping_criterion = stopping_criterion
        self.eval_points = eval_points
        self.benchmark_folder = benchmark_folder
        self.use_torch = use_torch
        self.compute_residual_directly = compute_residual_directly
        self.preconditioner = preconditioner
        self.inverse_approx_initial_capacity = inverse_approx_initial_capacity
        super().__init__()

    def get_concrete_solver(self, gp_params: GPInferenceParams) -> ConcreteIterGPSolver:
        return ConcreteIterGPSolver(
            gp_params,
            self.policy,
            self.stopping_criterion,
            eval_points=self.eval_points,
            benchmark_folder=self.benchmark_folder,
            use_torch=self.use_torch,
            compute_residual_directly=self.compute_residual_directly,
            preconditioner=self.preconditioner,
            inverse_approx_initial_capacity=self.inverse_approx_initial_capacity,
        )


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
            preconditioner=None,
        )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
def _(
    self,
    cov: IterGPCovarianceFunction,
    /,
    *,
    argnum: int = 0,
):
    return IterGPCrossCovariance(
        cov._gp_params,
        cov._solver_state.inverse_approx,
        self,
        reverse=(argnum == 0),
    )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
def _(
    self,
    crosscov: IterGPCrossCovariance,
    /,
) -> LinearOperator:
    L1_L2_prior_cov = self(crosscov._L_prior).linop
    L2_kLas = self(crosscov._gp_params.kLas).linop
    L1_kLas = crosscov._L_kLas
    if crosscov.reverse:
        res = L1_L2_prior_cov - L1_kLas @ crosscov._inverse_approx @ L2_kLas.T
    else:
        res = L1_L2_prior_cov - L2_kLas @ crosscov._inverse_approx @ L1_kLas.T
    return LinearOperatorCovariance(
        res,
        shape0=crosscov.randvar_shape if crosscov.reverse else self.output_shape,
        shape1=self.output_shape if crosscov.reverse else crosscov.randvar_shape,
    )
