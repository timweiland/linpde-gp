from math import prod

from linpde_gp import domains, linfuncops, linfunctls
from linpde_gp.randprocs import covfuncs
from linpde_gp.randvars import ArrayCovariance, Covariance

from ... import _arithmetic


class TensorProduct_Identity_LebesgueIntegral(
    _arithmetic.TensorProductProcessVectorCrossCovariance
):
    def __init__(
        self,
        tensor_product: covfuncs.TensorProduct,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        self._tensor_product = tensor_product
        self._integral = integral
        self._reverse = bool(reverse)

        assert self._tensor_product.input_shape == self._integral.input_domain_shape
        assert isinstance(self._integral.domain, domains.CartesianProduct)

        super().__init__(
            *(
                linfunctls.LebesgueIntegral(domain, factor.input_shape)(
                    factor, argnum=0 if reverse else 1
                )
                for domain, factor in zip(
                    self._integral.domain.factors, self._tensor_product.factors
                )
            )
        )

    @property
    def tensor_product(self) -> covfuncs.TensorProduct:
        return self._tensor_product

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self._integral


def as_num(x: Covariance):
    return x.array.reshape(())


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    _arithmetic.TensorProductProcessVectorCrossCovariance
)
def _(
    self, pv_crosscov: _arithmetic.TensorProductProcessVectorCrossCovariance, /
) -> Covariance:
    res = prod(
        as_num(linfunctls.LebesgueIntegral(domain, factor.randproc_input_shape)(factor))
        for domain, factor in zip(self.domain.factors, pv_crosscov.pv_crosscovs)
    )
    return ArrayCovariance(res, (), ())


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
    return _arithmetic.TensorProductProcessVectorCrossCovariance(*factors)
