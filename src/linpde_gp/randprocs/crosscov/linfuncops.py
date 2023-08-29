from linpde_gp import linfuncops
from linpde_gp.randprocs import crosscov

from ._arithmetic import (
    ScaledProcessVectorCrossCovariance,
    SumProcessVectorCrossCovariance,
)
from .linfunctls._difference import CovarianceFunction_Identity_Difference
from .linfunctls._dirac import (
    CovarianceFunction_Dirac_Identity,
    CovarianceFunction_Identity_Dirac,
)
from .linfunctls._evaluation import (
    CovarianceFunction_Evaluation_Identity,
    CovarianceFunction_Identity_Evaluation,
)
from .linfunctls.integrals import UnivariateHalfIntegerMaternLebesgueIntegral


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(
    self, pv_crosscov: ScaledProcessVectorCrossCovariance, /
) -> ScaledProcessVectorCrossCovariance:
    return ScaledProcessVectorCrossCovariance(
        pv_crosscov=self(pv_crosscov.pv_crosscov),
        scalar=pv_crosscov.scalar,
    )


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: SumProcessVectorCrossCovariance, /):
    return SumProcessVectorCrossCovariance(
        *(self(summand) for summand in pv_crosscov.pv_crosscovs)
    )


@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: crosscov.Zero, /):
    return crosscov.Zero(
        randproc_input_shape=self.output_domain_shape,
        randproc_output_shape=self.output_codomain_shape,
        randvar_shape=pv_crosscov.randvar_shape,
        reverse=pv_crosscov.reverse,
    )


@linfuncops.LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Dirac
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Dirac, /):
    return CovarianceFunction_Identity_Dirac(
        self(pv_crosscov.covfunc, argnum=0),
        pv_crosscov.dirac,
    )


@linfuncops.LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Dirac_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Dirac_Identity, /):
    return CovarianceFunction_Dirac_Identity(
        self(pv_crosscov, argnum=1),
        pv_crosscov.dirac,
    )


@linfuncops.LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Evaluation
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Evaluation, /):
    return CovarianceFunction_Identity_Evaluation(
        self(pv_crosscov.covfunc, argnum=0),
        pv_crosscov.evaluation_fctl,
    )


@linfuncops.LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Evaluation_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Evaluation_Identity, /):
    return CovarianceFunction_Evaluation_Identity(
        self(pv_crosscov, argnum=1),
        pv_crosscov.evaluation_fctl,
    )


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, stacked_pv_crosscov: crosscov.StackedProcessVectorCrossCovariance, /):
    return stacked_pv_crosscov.pv_crosscovs[self.idx]


@linfuncops.diffops.PartialDerivative.__call__.register(
    UnivariateHalfIntegerMaternLebesgueIntegral
)
def _(self, pv_crosscov: UnivariateHalfIntegerMaternLebesgueIntegral, /):
    if self.order == 0:
        return pv_crosscov
    lower_order_pd = linfuncops.diffops.Derivative(self.order - 1)
    return -1 * CovarianceFunction_Identity_Difference(
        lower_order_pd(pv_crosscov.matern, argnum=1 if pv_crosscov.reverse else 0),
        pv_crosscov.integral.domains.pure_array[..., 0],
        pv_crosscov.integral.domains.pure_array[..., 1],
        reverse=pv_crosscov.reverse,
    )
