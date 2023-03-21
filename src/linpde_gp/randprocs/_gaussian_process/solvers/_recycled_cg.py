from ._gp_solver import GPSolver, ConcreteGPSolver, GPInferenceParams
from typing import List
import probnum as pn
from probnum.typing import DTypeLike, ShapeLike
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinop
from scipy.sparse.linalg import cg
from linpde_gp.linops import DiagonalPlusLowRankMatrix, RankFactorizedMatrix

def reorthogonalize(v: np.ndarray, basis: List[np.ndarray], precond_inv):
    v_orth = v.copy()
    for u in basis:
        v_orth -= u * (np.dot(u, precond_inv @ v_orth) / np.dot(u, precond_inv @ u))
    return v_orth

def reorthogonalize_double(v: np.ndarray, basis: List[np.ndarray], precond_inv):
    return reorthogonalize(reorthogonalize(v, basis, precond_inv), basis, precond_inv)

class ConcreteRecycledCGSolver(ConcreteGPSolver):
    def __init__(self, gp_params: GPInferenceParams):
        self._spaced_out_indices = np.linspace(0, gp_params.prior_gram.shape[1] - 1, 20).astype(int)
        super().__init__(gp_params)

    def _compute_representer_weights(self):
        residual = self._get_full_residual()
        K_hat = self._gp_params.prior_gram
        self.inverse_approx = RankFactorizedMatrix(None, None, K_hat.shape, np.float64)

        representer_weights = np.zeros((K_hat.shape[1]))
        predictive_residual = residual
        iteration = 0
        print(f"Solving for matrix size {K_hat.shape}")
        while not self._stopping_criterion(predictive_residual):
            action = self._pick_action(iteration, predictive_residual)
            alpha = np.dot(action, predictive_residual)
            K_hat_action = K_hat @ action
            C_K_hat_action = self.inverse_approx @ K_hat_action
            search_direction = action - C_K_hat_action
            K_hat_search_direction = K_hat_action - K_hat @ C_K_hat_action
            normalization_constant = action.T @ K_hat_search_direction
            normalization_constant = 1.0 / normalization_constant if normalization_constant > 0.0 else 0.
            self.inverse_approx.append_factor(search_direction, normalization_constant)
            representer_weights += (alpha * normalization_constant) * search_direction
            predictive_residual = residual - K_hat @ representer_weights
            iteration += 1
            if iteration % 10 == 0:
                print(np.linalg.norm(predictive_residual))

        print(np.linalg.norm(predictive_residual))
        print(f"Required {iteration} iterations for matrix size {K_hat.shape}")
        return representer_weights

    def compute_posterior_cov(
        self, k_xx: np.ndarray, k_xX: np.ndarray, k_Xx: np.ndarray
    ):
        self.compute_representer_weights()
        return k_xx - (k_xX[..., None, :] @ (self.inverse_approx @ k_Xx[..., None]))[..., 0, 0]

    def _stopping_criterion(self, predictive_residual):
        val = np.linalg.norm(predictive_residual)
        #print(f"Stopping criterion: {val}")
        return val < 1e-4

    def _pick_action(self, iteration: int, predictive_residual: np.ndarray):
        CG_START = 20
        if iteration < CG_START:
            e = np.zeros_like(predictive_residual)
            e[self._spaced_out_indices[iteration]] = 1.
            if iteration == CG_START - 1:
                self._preconditioner = DiagonalPlusLowRankMatrix(1e-5 * np.ones(self._gp_params.prior_gram.shape[0]), self._gp_params.prior_gram @ (self.inverse_approx._C * self.inverse_approx._U))
            return e
        else:
            #return self._preconditioner.solve(predictive_residual)
            
            corrected_residual = predictive_residual
            if iteration == CG_START:
                self._orth_residuals = [predictive_residual]
                self._prev_action = corrected_residual
            if iteration > CG_START:
                #corrected_residual = reorthogonalize_double(predictive_residual, self._orth_residuals, self._preconditioner.inv())

                # beta = (
                #     np.dot(corrected_residual, self._preconditioner.inv() @ corrected_residual) 
                #     / np.dot(self._orth_residuals[-1], self._preconditioner.inv() @ self._orth_residuals[-1])
                # )
                # self._orth_residuals.append(corrected_residual)
                # action = self._preconditioner.inv() @ corrected_residual + beta * self._prev_action
                #action = self._preconditioner.inv() @ corrected_residual
                action = corrected_residual
                self._prev_action = action
                return action
            # if iteration > 0:
            #     beta = (
            #         np.dot(predictive_residual, predictive_residual) / np.dot(self._prev_residual, self._prev_residual)
            #     )
            #     corrected_residual += beta * self._prev_action
            # self._prev_residual = predictive_residual
            # self._prev_action = corrected_residual
            return corrected_residual
            return self._preconditioner.inv() @ corrected_residual


class RecycledCGSolver(GPSolver):
    def __init__(self):
        super().__init__()

    def get_concrete_solver(self, gp_params: GPInferenceParams) -> ConcreteRecycledCGSolver:
        return ConcreteRecycledCGSolver(gp_params)
