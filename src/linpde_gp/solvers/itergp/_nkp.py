import numpy as np
import probnum as pn
import nkp
import functools
import torch
from scipy.optimize import minimize


def F_norm_squared(A, B, C):
    return np.sum(A**2) * np.sum(B**2) * np.sum(C**2)


def F_inner_prod(A1, B1, C1, A2, B2, C2):
    return np.sum(A1 * A2) * np.sum(B1 * B2) * np.sum(C1 * C2)


def proximity(C: tuple[tuple[np.ndarray], ...], X: tuple[np.ndarray]):
    C_norm_sq = 0.0
    for i in range(len(C)):
        for j in range(len(C)):
            C_norm_sq += F_inner_prod(*C[i], *C[j])

    X_norm_sq = F_norm_squared(*X)

    inner_prod = 0.0
    for i in range(len(C)):
        inner_prod += F_inner_prod(*C[i], *X)
    
    print(f"C_norm_sq: {C_norm_sq}")
    print(f"X_norm_sq: {X_norm_sq}")
    print(f"inner_prod: {inner_prod}")
    
    return C_norm_sq + X_norm_sq - 2 * inner_prod

def score(alpha: np.ndarray, C: tuple[tuple[np.ndarray, ...]]):
    p = len(C)
    d = len(C[0])
    assert alpha.shape == (d*p,)
    alpha = alpha.reshape((d, p), order='C')
    
    X = []
    for i in range(d):
        cur_factor = C[0][i] * alpha[i, 0]

        for j in range(1, p):
            cur_factor += C[j][i] * alpha[i, j]
        X.append(cur_factor)

    X_norm_sq = F_norm_squared(*X)

    inner_prod = 0.0
    for i in range(len(C)):
        inner_prod += F_inner_prod(*C[i], *X)
    
    return X_norm_sq - 2 * inner_prod

def nkp_sum(C):
    N_sum = len(C)
    d = len(C[0])
    alpha_start = (1./d) * np.ones((d, N_sum))
    res = minimize(score, alpha_start.reshape(-1, order="C"), args=(C,), bounds=[(0, 1)]*d*N_sum)
    print(res)
    alpha_opt = res.x.reshape((d, N_sum), order="C")

    X_opt = [np.zeros_like(C[0][i]) for i in range(d)]
    for i in range(d):
        for j in range(N_sum):
            X_opt[i] = X_opt[i] + alpha_opt[i, j] * C[j][i]
    return X_opt



def kronecker_factors(K):
    if not isinstance(K, pn.linops.Kronecker):
        return [K.todense()]
    return kronecker_factors(K.A) + kronecker_factors(K.B)

class BasicInverse(pn.linops.LinearOperator):
    def __init__(self, A: np.ndarray):
        self._A = A
        super().__init__(A.shape, A.dtype)
    
    @functools.cached_property
    def _A_torch(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(self._A, dtype=torch.float64).to(device)
    
    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._A, x)
    
    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self._A_torch, x)

def gram_matrix(A):
    # G_ij = tr(A_i^T A_j)
    d = len(A)
    G = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            G[i, j] = np.trace(A[i].T @ A[j])
    return G

def proximity_3D(alpha_all, C):
    d = len(C)
    alpha_all = alpha_all.reshape((d, -1), order='C')
    alpha, beta, gamma = alpha_all[0], alpha_all[1], alpha_all[2]

    A_hat, B_hat, D_hat = gram_matrix(C[0]), gram_matrix(C[1]), gram_matrix(C[2])

    return np.sum(A_hat * B_hat * D_hat) - 2 * np.sum((A_hat @ alpha) * (B_hat @ beta) * (D_hat @ gamma)) + (alpha.T @ A_hat @ alpha) * (beta.T @ B_hat @ beta) * (gamma.T @ D_hat @ gamma)

def nkp_3D(C):
    d = len(C)
    N_sum = len(C[0])
    alpha_start = (1/N_sum) * np.ones((d, N_sum))
    res = minimize(proximity_3D, alpha_start.reshape(-1, order="C"), args=(C,))
    print(res)
    alpha_opt = res.x.reshape((d, N_sum), order="C")

    X_opt = [np.zeros_like(C[0][i]) for i in range(d)]
    for i in range(d):
        for j in range(N_sum):
            X_opt[i] = X_opt[i] + alpha_opt[i, j] * C[i][j]
    return X_opt

def kronecker_preconditioner(kronecker_sum):
    N_sum = len(kronecker_sum.summands)
    d = len(kronecker_factors(kronecker_sum.summands[0]))

    C = [[] for _ in range(d)]
    for summand in kronecker_sum.summands:
        factors = kronecker_factors(summand)
        for i in range(d):
            C[i].append(factors[i])
    
    K_facs = nkp_3D(C)
    return functools.reduce(
        lambda A, B: pn.linops.Kronecker(A, B, small_dense_factors=False),
        [BasicInverse(factor) for factor in K_facs]
    )