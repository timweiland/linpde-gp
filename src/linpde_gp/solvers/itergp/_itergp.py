from dataclasses import dataclass
from typing import List, Optional, Iterable

import jax.numpy as jnp
import numpy as np
from probnum import linops
from probnum.typing import ArrayLike
from tqdm.notebook import tqdm
import torch

from linpde_gp.linfunctls import (
    CompositeLinearFunctional,
    LinearFunctional,
    _EvaluationFunctional,
)
from linpde_gp.linops import (
    BlockMatrix,
    ProductBlockMatrix,
    BlockMatrix2x2,
    ExtendedOuterProduct,
    OuterProduct,
    LinearOperator,
    OuterProduct,
    DynamicDenseMatrix,
    ShapeAlignmentLinearOperator,
    CrosscovSandwich,
)
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction
from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance
from linpde_gp.randvars import LinearOperatorCovariance

from .._gp_solver import ConcreteGPSolver, GPInferenceParams, GPSolver
from .._solver_benchmarker import SolverBenchmarker
from ..covfuncs import DowndateCovarianceFunction
from ._solver_state import SolverState
from .policies import CGPolicy, Policy
from .loggers import Logger, TQDMLogger
from .stopping_criteria import (
    IterationStoppingCriterion,
    ResidualNormStoppingCriterion,
    StoppingCriterion,
)

import probnum as pn
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import svd


def hutch_plusplus(A, k=100):
    """
    Hutch++ trace estimator for symmetric matrices.
    """
    # Sample S iid with entries in {+1, -1}
    S = np.random.choice([-1, 1], size=(A.shape[0], k))
    G = np.random.choice([-1, 1], size=(A.shape[0], k))
    Q, _ = np.linalg.qr(A @ S)
    Z = G - Q @ Q.T @ G
    return np.trace(Q.T @ (A @ Q)) + np.trace(Z.T @ (A @ Z)) / k


class IterGPCovarianceFunction(DowndateCovarianceFunction):
    def __init__(
        self,
        gp_params: GPInferenceParams,
        solver_state: SolverState,
    ):
        self._gp_params = gp_params
        self._solver_state = solver_state
        self._L_block = None
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

    def linop(self, x0: ArrayLike, x1: ArrayLike | None = None) -> LinearOperator:
        crosscov_x0 = self._gp_params.kLas.evaluate_linop(x0)
        crosscov_x1 = (
            self._gp_params.kLas.evaluate_linop(x1) if x1 is not None else crosscov_x0
        )
        if x1 is None:
            L = self._gp_params.prior.cov.linop(x0) - CrosscovSandwich(
                crosscov_x0, self._solver_state.inverse_approx
            )
            return L
        L = self._gp_params.prior.cov.linop(x0, x1) - (
            crosscov_x0 @ self._solver_state.inverse_approx @ crosscov_x1.T
        )
        return L

    def sample(self, X_test: ArrayLike) -> np.ndarray:
        if self._L_block is None:
            k_XX = self._gp_params.prior.cov.linop(X_test)
            k_XX_cho = k_XX.cholesky()
            kL_X = self._gp_params.kLas.evaluate_linop(X_test)
            self._kL_X_aligned = ShapeAlignmentLinearOperator(
                self._gp_params.kLas, X_test
            )

            Si = self._solver_state.action_matrix
            print("Computing kL_X_action...")
            kL_X_action = kL_X @ Si.todense()
            print("Done.")
            Si_LkL_Si = self._solver_state.S_LKL_S
            print("Computed Si_LkL_Si")
            self._Si_LkL_Si_cho = cho_factor(
                Si_LkL_Si + 1e-9 * np.eye(Si_LkL_Si.shape[1])
            )

            cross_term = (k_XX_cho.inv() @ kL_X_action).T
            print("Cross term shape:" + str(cross_term.shape))

            S = Si_LkL_Si - cross_term @ cross_term.T
            print("Computed S")
            S_cho, _ = cho_factor(S + 1e-9 * np.eye(S.shape[1]), lower=True)
            S_cho = np.tril(S_cho)
            S_cho = linops.aslinop(S_cho)
            S_cho.is_lower_triangular = True
            print("Computed cho factor of S")

            self._L_block = BlockMatrix2x2(k_XX_cho, None, cross_term, S_cho)

        Si = self._solver_state.action_matrix
        U = np.random.normal(size=(self._L_block.shape[1],))
        XY_transformed = self._L_block @ U.flatten()
        X_transformed, Y_transformed = (
            XY_transformed[: self._L_block.A.shape[1]],
            XY_transformed[self._L_block.A.shape[1] :],
        )

        X_transformed_reshaped = X_transformed.reshape(
            self._gp_params.prior.output_shape + (-1,), order="C"
        )
        X_transformed_reshaped = np.moveaxis(X_transformed_reshaped, 0, -1)
        X_transformed_reshaped = X_transformed_reshaped.reshape(-1, order="C")

        Z = Si.T @ np.concatenate(self._gp_params.Ys, axis=-1)
        final_sample = X_transformed_reshaped + self._kL_X_aligned @ Si @ cho_solve(
            self._Si_LkL_Si_cho, Z - Y_transformed
        )
        batch_dim_size = len(X_test.shape) - self.input_ndim
        return final_sample.reshape(
            X_test.shape[:batch_dim_size] + self._gp_params.prior.output_shape
        )


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
        store_K_hat_inverse_approx=True,
        num_actions_compressed=100,
        num_actions_explorative=10,
        loggers: Iterable[Logger] = None,
    ):
        self.policy = policy.get_concrete_policy(gp_params)
        self.stopping_criterion = stopping_criterion.get_concrete_criterion(gp_params)
        self.solver_state = SolverState(
            0, None, None, None, None, None, gp_params, None, None, None, None
        )
        self.eval_points = eval_points
        self.benchmark_folder = benchmark_folder
        self.use_torch = use_torch
        self.compute_residual_directly = compute_residual_directly
        self.store_K_hat_inverse_approx = store_K_hat_inverse_approx
        self.num_actions_compressed = num_actions_compressed
        self.num_actions_explorative = num_actions_explorative
        self.loggers = loggers
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
            else torch.zeros((K_hat.shape[1]), dtype=torch.float64, device=device)
        )
        residual_norm = (
            np.linalg.norm(residual, ord=2)
            if not self.use_torch
            else torch.norm(residual, p=2)
        )
        self.solver_state.predictive_residual = torch.clone(residual)
        self.solver_state.relative_error = 1.0

        # avg_crosscov_weights = torch.zeros_like(self.solver_state.representer_weights)
        max_num_actions = self.num_actions_compressed + self.num_actions_explorative

        search_directions = DynamicDenseMatrix(
            (K_hat.shape[0], max_num_actions), np.float64
        )
        cur_inverse_approx_term = OuterProduct(search_directions)
        self.solver_state.inverse_approx = cur_inverse_approx_term

        cur_action_matrix = DynamicDenseMatrix(
            (K_hat.shape[0], max_num_actions), K_hat.dtype
        )

        if self.store_K_hat_inverse_approx:
            K_hat_search_directions = DynamicDenseMatrix(
                (K_hat.shape[0], max_num_actions), np.float64
            )
        else:
            K_hat_search_directions = None

        if self.store_K_hat_inverse_approx:
            cur_K_hat_inverse_approx_term = OuterProduct(
                K_hat_search_directions, search_directions
            )
            self.solver_state.K_hat_inverse_approx = cur_K_hat_inverse_approx_term

        new_start = 0

        avg_crosscov = None
        if self.eval_points is not None:
            eval_fctl = _EvaluationFunctional(
                self._gp_params.prior.input_shape,
                self._gp_params.prior.output_shape,
                self.eval_points,
            )
            eval_crosscov = eval_fctl(self._gp_params.kLas).linop
            N_eval_points = eval_crosscov.shape[0]
            avg_crosscov = (1 / N_eval_points) * (
                eval_crosscov.T
                @ torch.ones(N_eval_points, dtype=torch.float64, device=device)
            )
            self.solver_state.crosscov_residual = torch.clone(avg_crosscov)

        if self._gp_params.prior_inverse_approx is not None:
            prior_dim = self._gp_params.prior_inverse_approx.shape[0]
            new_start = prior_dim

            assert isinstance(self._gp_params.prior_inverse_approx, OuterProduct)
            prior_inverse_approx_extended = ExtendedOuterProduct(
                self._gp_params.prior_inverse_approx, K_hat.shape
            )
            self.solver_state.inverse_approx += prior_inverse_approx_extended
            assert self._gp_params.prev_representer_weights is not None
            prev_representer_weights = self._gp_params.prev_representer_weights
            if self.use_torch:
                prev_representer_weights = torch.from_numpy(
                    prev_representer_weights
                ).to(device)
            self.solver_state.representer_weights[
                :prior_dim
            ] += prev_representer_weights

            if self.store_K_hat_inverse_approx:
                prior_K_hat_inverse_approx_extended = ExtendedOuterProduct(
                    self._gp_params.prior_K_hat_inverse_approx, K_hat.shape
                )  # TODO: It's not that simple. Crosscov term is missing!
                self.solver_state.K_hat_inverse_approx += (
                    prior_K_hat_inverse_approx_extended
                )

            self.solver_state.predictive_residual = (
                residual - K_hat @ self.solver_state.representer_weights
            )

            residual_norm = (
                np.linalg.norm(self.solver_state.predictive_residual[new_start:], ord=2)
                if not self.use_torch
                else torch.norm(self.solver_state.predictive_residual[new_start:], p=2)
            )

            if self.eval_points is not None:
                self.solver_state.crosscov_residual = avg_crosscov - (
                    K_hat @ (self.solver_state.inverse_approx @ avg_crosscov)
                )

            assert self._gp_params.prior_action_matrix is not None
            N_obs_prior = self._gp_params.prior_action_matrix.shape[0]
            prior_action_matrix_extended = BlockMatrix(
                [
                    [self._gp_params.prior_action_matrix],
                    [
                        pn.linops.Zero(
                            (
                                K_hat.shape[0] - N_obs_prior,
                                self._gp_params.prior_action_matrix.shape[1],
                            )
                        )
                    ],
                ]
            )
            self.solver_state.action_matrix = BlockMatrix(
                [[prior_action_matrix_extended, cur_action_matrix]],
                cache_transpose=False,
            )
            # S_LKL_S = torch.zeros((N_obs_total, N_obs_total), dtype=torch.float64)
            # S_LKL_S[:N_obs_prior, :N_obs_prior] = torch.tensor(
            #     self._gp_params.prior_S_LKL_S.todense(), dtype=torch.float64
            # ).to(device)
        else:
            pass
            # self.solver_state.action_matrix = cur_action_matrix
            # S_LKL_S = torch.zeros(
            #     (cur_action_matrix.shape[1], cur_action_matrix.shape[1])
            # )

        benchmarker = SolverBenchmarker(self.benchmark_folder)
        benchmarker.start_benchmark()

        for logger in self.loggers:
            logger.start(self._gp_params)

        cur_action_idx = new_start

        self._rayleighs = []

        while not self.stopping_criterion(self.solver_state):
            action = self.policy(self.solver_state)
            if action is None:  # Policy ran out of actions, quit early
                break
            K_hat_action = K_hat @ action

            # S_K_hat_action = (self.solver_state.action_matrix.T @ K_hat_action)[:cur_action_idx]
            # S_LKL_S[cur_action_idx, :cur_action_idx] = S_K_hat_action
            # S_LKL_S[:cur_action_idx, cur_action_idx] = S_K_hat_action
            cur_action_matrix.append_column(action.cpu().numpy()) # Store on CPU

            alpha = (
                torch.dot(action, self.solver_state.predictive_residual)
                if self.use_torch
                else np.dot(action, self.solver_state.predictive_residual)
            )
            C_K_hat_action = self.solver_state.inverse_approx @ K_hat_action
            # if self.store_K_hat_inverse_approx:
            #     K_hat_C_K_hat_action = (
            #         self.solver_state.K_hat_inverse_approx @ K_hat_action
            #     )
            # else:  TODO: Re-add this later for efficiency once the crosscov term is added.
            K_hat_C_K_hat_action = K_hat @ C_K_hat_action

            search_direction = action - C_K_hat_action
            K_hat_search_direction = K_hat_action - K_hat_C_K_hat_action
            normalization_constant = (
                torch.dot(action, K_hat_search_direction)
                if self.use_torch
                else np.dot(action, K_hat_search_direction)
            )
            # S_LKL_S[cur_action_idx, cur_action_idx] = torch.dot(action, K_hat_action)
            cur_action_idx += 1

            action_T_action = (
                torch.dot(action, action) if self.use_torch else np.dot(action, action)
            )
            rayleigh = torch.dot(action, K_hat_action) / action_T_action
            self._rayleighs.append(rayleigh)

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

            search_directions.append_column(
                search_direction * sqrt_normalization_constant
            )
            if self.store_K_hat_inverse_approx:
                K_hat_search_directions.append_column(
                    K_hat_search_direction * sqrt_normalization_constant
                )

            if search_directions._num_cols >= max_num_actions:
                print("Compressing...")
                # Compress
                D = search_directions.data
                K_hat_D = K_hat_search_directions.data
                M = eval_crosscov @ D
                _, _, VT = svd(M, full_matrices=False)
                V = VT.T
                V = V[:, : self.num_actions_compressed]
                search_directions.set_data(D @ V)
                K_hat_search_directions.set_data(K_hat_D @ V)

            self.solver_state.representer_weights += (
                alpha * normalization_constant
            ) * search_direction

            if self.eval_points is not None:
                alpha_crosscov = torch.dot(search_direction, avg_crosscov)
                self.solver_state.crosscov_residual -= (
                    alpha_crosscov * normalization_constant * K_hat_search_direction
                )

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

            if self.eval_points is not None:
                self.solver_state.relative_crosscov_error = torch.norm(
                    self.solver_state.crosscov_residual, p=2
                ) / torch.norm(avg_crosscov, p=2)

            for logger in self.loggers:
                logger(self.solver_state)

            self.solver_state.iteration += 1

        # if self.solver_state.iteration < self.inverse_approx_initial_capacity:
        #     diff = (
        #         self.inverse_approx_initial_capacity - self.solver_state.iteration
        #     )  # Amount of actions we over-allocated
        #     N_actions_actual = self.solver_state.action_matrix.shape[1] - diff
        #     new_action_matrix = np.zeros(
        #         (self.solver_state.action_matrix.shape[0], N_actions_actual)
        #     )
        #     new_action_matrix = self.solver_state.action_matrix[:, :N_actions_actual]
        #     self.solver_state.action_matrix = new_action_matrix

        # self.solver_state.S_LKL_S = S_LKL_S.cpu().numpy()

        for logger in self.loggers:
            logger.finish()

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
    def inverse_approximation(self) -> OuterProduct:
        if self.solver_state.inverse_approx is None:
            self.compute_representer_weights()
        return self.solver_state.inverse_approx

    @property
    def K_hat_inverse_approximation(self) -> OuterProduct:
        if self.solver_state.K_hat_inverse_approx is None:
            self.compute_representer_weights()
        return self.solver_state.K_hat_inverse_approx

    @property
    def action_matrix(self) -> np.ndarray:
        if self.solver_state.action_matrix is None:
            self.compute_representer_weights()
        return self.solver_state.action_matrix

    @property
    def S_LKL_S(self) -> np.ndarray:
        return pn.linops.aslinop(
            np.random.rand(
                self.solver_state.action_matrix.shape[1],
                self.solver_state.action_matrix.shape[1],
            )
        )


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
        num_actions_compressed=100,
        num_actions_explorative=10,
        loggers: Iterable[Logger] = [TQDMLogger(notebook=True)],
    ):
        self.policy = policy
        self.stopping_criterion = stopping_criterion
        self.eval_points = eval_points
        self.benchmark_folder = benchmark_folder
        self.use_torch = use_torch
        self.compute_residual_directly = compute_residual_directly
        self.num_actions_compressed = num_actions_compressed
        self.num_actions_explorative = num_actions_explorative
        self.loggers = loggers
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
            num_actions_compressed=self.num_actions_compressed,
            num_actions_explorative=self.num_actions_explorative,
            loggers=self.loggers,
        )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
@_EvaluationFunctional.__call__.register
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
@_EvaluationFunctional.__call__.register
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
