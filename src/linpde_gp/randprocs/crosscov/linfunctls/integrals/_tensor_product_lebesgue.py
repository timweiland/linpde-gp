import functools
from math import prod

import numpy as np
import probnum as pn
from linpde_gp import domains, linfuncops, linfunctls
from linpde_gp.domains import TensorProductDomain
from linpde_gp.randprocs import covfuncs
from linpde_gp.randvars import (ArrayCovariance, Covariance,
                                LinearOperatorCovariance)

from ... import _arithmetic


class TensorProduct_Identity_LebesgueIntegral(
    _arithmetic.TensorProductProcessVectorCrossCovariance
):
    def __init__(
        self,
        tensor_product: covfuncs.TensorProduct,
        integral: linfunctls.VectorizedLebesgueIntegral,
        reverse: bool = False,
    ):
        self._tensor_product = tensor_product
        self._integral = integral
        self._reverse = bool(reverse)

        assert self._tensor_product.input_shape == self._integral.input_domain_shape
        assert self._integral.domains.common_type in [domains.Interval, domains.Box]

        grid_factorized = isinstance(integral.domains, TensorProductDomain)
        argnum = 0 if reverse else 1
        if grid_factorized:
            pv_crosscovs = tuple(
                linfunctls.VectorizedLebesgueIntegral(factor_domains)(
                    factor, argnum=argnum
                )
                for factor_domains, factor in zip(
                    integral.domains.factors, tensor_product.factors
                )
            )
        else:
            pv_crosscovs = tuple(
                linfunctls.VectorizedLebesgueIntegral(factor_domains)(
                    factor, argnum=argnum
                )
                for factor_domains, factor in zip(
                    self._integral.domains.factorize(), self._tensor_product.factors
                )
            )

        super().__init__(*pv_crosscovs, grid_factorized=grid_factorized)

    @property
    def tensor_product(self) -> covfuncs.TensorProduct:
        return self._tensor_product

    @property
    def integral(self) -> linfunctls.VectorizedLebesgueIntegral:
        return self._integral


@linfunctls.VectorizedLebesgueIntegral.__call__.register(  # pylint: disable=no-member
    _arithmetic.TensorProductProcessVectorCrossCovariance
)
def _(
    self: linfunctls.VectorizedLebesgueIntegral,
    pv_crosscov: _arithmetic.TensorProductProcessVectorCrossCovariance,
    /,
) -> Covariance:
    res = prod(
        linfunctls.VectorizedLebesgueIntegral(factor_domains)(factor).array
        for factor_domains, factor in zip(
            self.domains.factorize(), pv_crosscov.pv_crosscovs
        )
    )
    shape0 = pv_crosscov.randvar_shape if pv_crosscov.reverse else self.output_shape
    shape1 = self.output_shape if pv_crosscov.reverse else pv_crosscov.randvar_shape
    return ArrayCovariance(res, shape0, shape1)


@linfunctls.VectorizedLebesgueIntegral.__call__.register(  # pylint: disable=no-member
    TensorProduct_Identity_LebesgueIntegral
)
def _(
    self: linfunctls.VectorizedLebesgueIntegral,
    pv_crosscov: TensorProduct_Identity_LebesgueIntegral,
    /,
) -> Covariance:
    shape0 = pv_crosscov.randvar_shape if pv_crosscov.reverse else self.output_shape
    shape1 = self.output_shape if pv_crosscov.reverse else pv_crosscov.randvar_shape
    if isinstance(self.domains, TensorProductDomain) and isinstance(
        pv_crosscov.integral.domains, TensorProductDomain
    ):
        int2_argnum = 0 if pv_crosscov.reverse else 1
        get_int = lambda domain: linfunctls.VectorizedLebesgueIntegral(domain)
        factor_integrals = tuple(
            get_int(factor_int1)(get_int(factor_int2)(factor_tp, argnum=int2_argnum))
            for factor_int1, factor_int2, factor_tp in zip(
                self.domains.factors,
                pv_crosscov.integral.domains.factors,
                pv_crosscov.tensor_product.factors,
            )
        )
        factor_integrals = tuple(cov.linop for cov in factor_integrals)
        res = functools.reduce(pn.linops.Kronecker, factor_integrals)
        return LinearOperatorCovariance(res, shape0, shape1)
    res = prod(
        linfunctls.VectorizedLebesgueIntegral(factor_domains)(factor).array
        for factor_domains, factor in zip(
            self.domains.factorize(), pv_crosscov.pv_crosscovs
        )
    )
    return ArrayCovariance(res, shape0, shape1)


# TODO: Compare at some point whether pulling the diffop has advantages
# over directly applying partial derivatives to TensorProduct crosscov.
# @linfuncops.diffops.LinearDifferentialOperator.__call__.register(  # pylint: disable=no-member
#     TensorProduct_Identity_LebesgueIntegral
# )
# def _(
#     self: linfuncops.diffops.LinearDifferentialOperator,
#     pv_crosscov: TensorProduct_Identity_LebesgueIntegral,
#     /,
# ):
#     integral_argnum = 0 if pv_crosscov.reverse else 1
#     return pv_crosscov.integral(
#         self(pv_crosscov.tensor_product, argnum=1 - integral_argnum),
#         argnum=integral_argnum
#     )


@linfuncops.diffops.PartialDerivative.__call__.register(  # pylint: disable=no-member
    _arithmetic.TensorProductProcessVectorCrossCovariance
)
def _(
    self: linfuncops.diffops.PartialDerivative,
    pv_crosscov: _arithmetic.TensorProductProcessVectorCrossCovariance,
    /,
):
    factors = []
    for dim_order, factor in zip(self.multi_index.array, pv_crosscov.pv_crosscovs):
        factors.append(linfuncops.diffops.Derivative(dim_order)(factor))
    return _arithmetic.TensorProductProcessVectorCrossCovariance(*factors, grid_factorized=pv_crosscov.grid_factorized)


@linfuncops.diffops.PartialDerivative.__call__.register(  # pylint: disable=no-member
    TensorProduct_Identity_LebesgueIntegral
)
def _(
    self: linfuncops.diffops.PartialDerivative,
    pv_crosscov: TensorProduct_Identity_LebesgueIntegral,
    /,
):
    argnum = 1 if pv_crosscov.reverse else 0
    return pv_crosscov.integral(
        self(pv_crosscov.tensor_product, argnum=argnum), argnum=1 - argnum
    )
