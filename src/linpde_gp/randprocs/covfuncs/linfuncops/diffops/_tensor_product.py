import functools
import operator
from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.utils import ArrayLike
from pykeops.numpy import LazyTensor

from linpde_gp.linfuncops import diffops

from ..._jax_arithmetic import JaxFunctionScaledCovarianceFunction, JaxSumCovarianceFunction
from ..._tensor_product import (
    TensorProduct,
    TensorProductGrid,
    split_outputs_contiguous,
)


class TensorProduct_LinDiffOp_LinDiffOp(JaxSumCovarianceFunction[TensorProduct]):
    def __init__(
        self,
        k: TensorProduct,
        *,
        L0: diffops.LinearDifferentialOperator,
        L1: diffops.LinearDifferentialOperator,
    ):
        self._k = k

        self._L0 = L0
        self._L1 = L1

        L0kL1s = np.empty(self.k.input_shape, dtype=dict)
        for idx, _ in np.ndenumerate(L0kL1s):
            L0kL1s[idx] = {}

        L0_coeffs = self._L0.coefficients[()]
        L1_coeffs = self._L1.coefficients[()]

        summands = []
        for L0_multi_index, L0_coeff in L0_coeffs.items():
            for L1_multi_index, L1_coeff in L1_coeffs.items():
                cur_tp_factors = []
                for domain_idx, _ in np.ndenumerate(L0_multi_index.array):
                    PD_0 = diffops.PartialDerivative(
                        diffops.MultiIndex.from_index(
                            domain_idx, L0_multi_index.shape, L0_multi_index[domain_idx]
                        )
                    )
                    PD_1 = diffops.PartialDerivative(
                        diffops.MultiIndex.from_index(
                            domain_idx, L1_multi_index.shape, L1_multi_index[domain_idx]
                        )
                    )

                    cur_factor = PD_0.get_factor_at_dim(domain_idx)(
                        PD_1.get_factor_at_dim(domain_idx)(
                            self.k.factors[domain_idx[0]], argnum=1
                        ),
                        argnum=0,
                    )
                    L0kL1s[domain_idx][
                        (L0_multi_index[domain_idx], L1_multi_index[domain_idx])
                    ] = cur_factor
                    cur_tp_factors.append(cur_factor)
                summands.append(self._get_summand(L0_coeff, L1_coeff, cur_tp_factors))
        self._L0kL1s = L0kL1s

        super().__init__(*summands)

    @property
    def k(self) -> TensorProduct:
        return self._k

    @property
    def L0(self) -> diffops.LinearDifferentialOperator:
        return self._L0

    @property
    def L1(self) -> diffops.LinearDifferentialOperator:
        return self._L1
    
    def _get_summand(self, L0_coeff, L1_coeff, cur_tp_factors):
        tp = TensorProduct(*cur_tp_factors)
        if isinstance(L0_coeff, float) and isinstance(L1_coeff, float):
            return L0_coeff * L1_coeff * tp
        elif isinstance(L0_coeff, pn.functions.Function) and isinstance(L1_coeff, float):
            return L1_coeff * JaxFunctionScaledCovarianceFunction(tp, fn0=L0_coeff)
        elif isinstance(L0_coeff, float) and isinstance(L1_coeff, pn.functions.Function):
            return L0_coeff * JaxFunctionScaledCovarianceFunction(tp, fn1=L1_coeff)
        elif isinstance(L0_coeff, pn.functions.Function) and isinstance(L1_coeff, pn.functions.Function):
            return JaxFunctionScaledCovarianceFunction(tp, fn0=L0_coeff, fn1=L1_coeff)
        else:
            raise ValueError("Unknown combination of L0_coeff and L1_coeff")

    def _compute_res(
        self, compute_functional, res_zero_value=0.0, reduction_operator=operator.mul
    ):
        L0kL1_evals = np.empty(self.k.input_shape, dtype=dict)
        for idx, _ in np.ndenumerate(L0kL1_evals):
            L0kL1_evals[idx] = {}

        L0_coeffs = self._L0.coefficients[()]
        L1_coeffs = self._L1.coefficients[()]
        res = res_zero_value

        for L0_multi_index, L0_coeff in L0_coeffs.items():
            for L1_multi_index, L1_coeff in L1_coeffs.items():
                factors = []
                for domain_idx, _ in np.ndenumerate(L0kL1_evals):
                    orders_idx = (
                        L0_multi_index[domain_idx],
                        L1_multi_index[domain_idx],
                    )
                    if orders_idx not in L0kL1_evals[domain_idx]:
                        L0kL1_evals[domain_idx][orders_idx] = compute_functional(
                            self._L0kL1s[domain_idx][orders_idx], domain_idx
                        )
                    factors.append(L0kL1_evals[domain_idx][orders_idx])

                res += (
                    L0_coeff * L1_coeff * functools.reduce(reduction_operator, factors)
                )
        return res

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._compute_res(
            lambda k, idx: k(
                x0[(..., *idx)], x1[(..., *idx)] if x1 is not None else None
            ),
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._compute_res(
            lambda k, idx: k.jax(
                x0[(..., *idx)], x1[(..., *idx)] if x1 is not None else None
            )
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        x0s, x1s = split_outputs_contiguous(x0, x1, len(self._k.factors))
        return self._compute_res(
            lambda k, idx: (
                k._keops_lazy_tensor(  # pylint: disable=protected-access
                    x0s[idx[0]], x1s[idx[0]]
                )
            )
        )

    def linop(
        self, x0: ArrayLike, x1: Optional[ArrayLike] = None
    ) -> pn.linops.LinearOperator:
        # Fallback to parent implementation (sum of TensorProduct)
        return super().linop(x0, x1)
