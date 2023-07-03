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
    TensorProduct_Identity_LebesgueIntegral
)
def _(self, pv_crosscov: TensorProduct_Identity_LebesgueIntegral, /) -> Covariance:
    res = prod(
        as_num(linfunctls.LebesgueIntegral(domain, factor.randproc_input_shape)(factor))
        for domain, factor in zip(self.domain.factors, pv_crosscov.pv_crosscovs)
    )
    return ArrayCovariance(res, (), ())


@linfuncops.diffops.PartialDerivative.__call__.register(  # pylint: disable=no-member
    TensorProduct_Identity_LebesgueIntegral
)
def _(
    self: linfuncops.diffops.PartialDerivative,
    pv_crosscov: TensorProduct_Identity_LebesgueIntegral,
    /,
) -> TensorProduct_Identity_LebesgueIntegral:
    return TensorProduct_Identity_LebesgueIntegral(
        self(pv_crosscov.tensor_product, argnum=1 if pv_crosscov.reverse else 0),
        pv_crosscov.integral,
        reverse=pv_crosscov.reverse,
    )
