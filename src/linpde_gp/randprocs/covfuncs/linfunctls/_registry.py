import functools
import operator

import numpy as np
from probnum.randprocs import covfuncs as pn_covfuncs
from probnum.randprocs.covfuncs._arithmetic_fallbacks import (
    ScaledCovarianceFunction,
    SumCovarianceFunction,
)

from linpde_gp import domains, linfunctls
from linpde_gp.linfuncops.diffops import Derivative, PartialDerivative
from linpde_gp.linfunctls._evaluation import _EvaluationFunctional
from linpde_gp.linfunctls.projections.l2 import (
    L2Projection_UnivariateLinearInterpolationBasis,
)
from linpde_gp.randprocs import covfuncs

from .._utils import validate_covfunc_transformation

########################################################################################
# General `LinearFunctional`s ##########################################################
########################################################################################


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k_scaled: ScaledCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k_scaled, argnum)

    # pylint: disable=protected-access
    return k_scaled._scalar * self(k_scaled._covfunc, argnum=argnum)


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: SumCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    # pylint: disable=protected-access
    return functools.reduce(
        operator.add, (self(summand, argnum=argnum) for summand in k._summands)
    )


@linfunctls.CompositeLinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    res = k

    if self.linfuncop is not None:
        res = self.linfuncop(res, argnum=argnum)

    res = self.linfunctl(res, argnum=argnum)

    if self.linop is not None:
        from ...crosscov import (  # pylint: disable=import-outside-toplevel
            LinOpProcessVectorCrossCovariance,
        )

        res = LinOpProcessVectorCrossCovariance(self.linop, res)

    return res


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    from ...crosscov import (  # pylint: disable=import-outside-toplevel
        StackedProcessVectorCrossCovariance,
    )

    L_covfuncs = np.copy(k.covfuncs)
    for idx, covfunc in np.ndenumerate(L_covfuncs):
        L_covfuncs[idx] = self(covfunc, argnum=argnum)

    return StackedProcessVectorCrossCovariance(L_covfuncs)


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    from ...crosscov import Zero  # pylint: disable=import-outside-toplevel

    return Zero(
        randproc_input_shape=k.input_shape_1 if argnum == 0 else k.input_shape_0,
        randproc_output_shape=k.output_shape_1 if argnum == 0 else k.output_shape_0,
        randvar_shape=self.output_shape,
        reverse=(argnum == 0),
    )


########################################################################################
# Point Evaluation #####################################################################
########################################################################################


@linfunctls.DiracFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    match argnum:
        case 0:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Dirac_Identity,
            )

            return CovarianceFunction_Dirac_Identity(
                covfunc=k,
                dirac=self,
            )
        case 1:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Identity_Dirac,
            )

            return CovarianceFunction_Identity_Dirac(
                covfunc=k,
                dirac=self,
            )

    raise ValueError("`argnum` must either be 0 or 1.")


@_EvaluationFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    match argnum:
        case 0:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Evaluation_Identity,
            )

            return CovarianceFunction_Evaluation_Identity(
                covfunc=k,
                evaluation_fctl=self,
            )
        case 1:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Identity_Evaluation,
            )

            return CovarianceFunction_Identity_Evaluation(
                covfunc=k,
                evaluation_fctl=self,
            )

    raise ValueError("`argnum` must either be 0 or 1.")


########################################################################################
# Integrals ############################################################################
########################################################################################


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def covfunc_lebesgue_integral(
    self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0
):
    validate_covfunc_transformation(self, k, argnum)

    try:
        return super(linfunctls.LebesgueIntegral, self).__call__(k, argnum=argnum)
    except NotImplementedError:
        from ...crosscov.linfunctls.integrals import (  # pylint: disable=import-outside-toplevel
            CovarianceFunction_Identity_LebesgueIntegral,
        )

        return CovarianceFunction_Identity_LebesgueIntegral(
            k, self, reverse=(argnum == 0)
        )


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.input_shape == () and k.is_half_integer:
        from ...crosscov.linfunctls.integrals import (  # pylint: disable=import-outside-toplevel
            UnivariateHalfIntegerMaternLebesgueIntegral,
        )

        return UnivariateHalfIntegerMaternLebesgueIntegral(
            matern=k,
            integral=self,
            reverse=(argnum == 0),
        )

    return covfunc_lebesgue_integral(self, k, argnum=argnum)


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.TensorProduct, /, *, argnum: int = 0):
    if argnum not in (0, 1):
        raise ValueError("`argnum` must either be 0 or 1.")

    from ...crosscov.linfunctls.integrals import (  # pylint: disable=import-outside-toplevel
        TensorProduct_Identity_LebesgueIntegral,
    )

    return TensorProduct_Identity_LebesgueIntegral(
        tensor_product=k,
        integral=self,
        reverse=(argnum == 0),
    )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    covfuncs.linfuncops.diffops.HalfIntegerMatern_Identity_DirectionalDerivative
)
def _(
    self,
    covfunc: covfuncs.linfuncops.diffops.HalfIntegerMatern_Identity_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    from ...crosscov.linfunctls import CovarianceFunction_Identity_Difference

    integral_reverse = argnum == 0
    if covfunc.reverse != integral_reverse:
        return -1 * CovarianceFunction_Identity_Difference(
            covfunc.matern,
            self.domain[0],
            self.domain[1],
            reverse=integral_reverse,
        )
    return CovarianceFunction_Identity_Difference(
        covfunc.matern,
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian
)
def _(
    self,
    covfunc: covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    from ...crosscov.linfunctls import CovarianceFunction_Identity_Difference

    integral_reverse = argnum == 0
    D = Derivative(1)
    if covfunc.reverse != integral_reverse:
        return -1 * CovarianceFunction_Identity_Difference(
            D(covfunc.matern, argnum=1 - argnum),
            self.domain[0],
            self.domain[1],
            reverse=integral_reverse,
        )
    return CovarianceFunction_Identity_Difference(
        D(covfunc.matern, argnum=argnum),
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )


def reduce_derivative_order(
    k: pn_covfuncs.CovarianceFunction,
    D1: PartialDerivative,
    D2: PartialDerivative,
    argnum: int,
):
    assert D1.input_domain_shape == () and D2.input_domain_shape == ()
    if argnum == 0:
        D1 = Derivative(D1.order - 1)
    else:
        D2 = Derivative(D2.order - 1)
    return D1(D2(k, argnum=1), argnum=0)


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
)
def _(
    self,
    covfunc: covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative,
    /,
    *,
    argnum: int = 0,
):
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    from ...crosscov.linfunctls import CovarianceFunction_Identity_Difference

    integral_reverse = argnum == 0
    D1 = Derivative(1)
    D2 = Derivative(1)
    return CovarianceFunction_Identity_Difference(
        reduce_derivative_order(covfunc.matern, D1, D2, argnum),
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian
)
def _(
    self,
    covfunc: covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    from ...crosscov.linfunctls import CovarianceFunction_Identity_Difference

    integral_reverse = argnum == 0
    if covfunc.reverse:
        D1 = Derivative(2)
        D2 = Derivative(1)
    else:
        D1 = Derivative(1)
        D2 = Derivative(2)
    return CovarianceFunction_Identity_Difference(
        reduce_derivative_order(covfunc.matern, D1, D2, argnum),
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian
)
def _(
    self,
    covfunc: covfuncs.linfuncops.diffops.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian,
    /,
    *,
    argnum: int = 0,
):
    if not isinstance(self.domain, domains.Interval):
        raise NotImplementedError()

    from ...crosscov.linfunctls import CovarianceFunction_Identity_Difference

    integral_reverse = argnum == 0
    D1 = Derivative(2)
    D2 = Derivative(2)
    return CovarianceFunction_Identity_Difference(
        reduce_derivative_order(covfunc.matern, D1, D2, argnum),
        self.domain[0],
        self.domain[1],
        reverse=integral_reverse,
    )


########################################################################################
# Projections ##########################################################################
########################################################################################


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    try:
        return super(L2Projection_UnivariateLinearInterpolationBasis, self).__call__(
            k, argnum=argnum
        )
    except NotImplementedError:
        from ...crosscov.linfunctls.projections import (  # pylint: disable=import-outside-toplevel
            CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis,
        )

        return CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis(
            covfunc=k,
            proj=self,
            reverse=(argnum == 0),
        )


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    from ...crosscov.linfunctls.projections import (  # pylint: disable=import-outside-toplevel
        CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis,
        Matern32_L2Projection_UnivariateLinearInterpolationBasis,
    )

    if k.nu == 1.5:
        return Matern32_L2Projection_UnivariateLinearInterpolationBasis(
            covfunc=k,
            proj=self,
            reverse=(argnum == 0),
        )

    return CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis(
        covfunc=k,
        proj=self,
        reverse=(argnum == 0),
    )


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.GalerkinCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.P is self:
        from ...crosscov import (  # pylint: disable=import-outside-toplevel
            ParametricProcessVectorCrossCovariance,
        )

        return ParametricProcessVectorCrossCovariance(
            crosscov=k.PkP,
            basis=k.P.basis,
            reverse=(argnum == 0),
        )

    return super(L2Projection_UnivariateLinearInterpolationBasis, self).__call__(
        k, argnum=argnum
    )
