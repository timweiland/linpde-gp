import functools
import operator
from typing import List

import numpy as np
import probnum as pn

from linpde_gp.linops import RankOneHadamardProduct
from linpde_gp.linfunctls import (
    LinearFunctional,
    ScaledLinearFunctional,
    VectorizedLebesgueIntegral,
)
from linpde_gp.randvars import (
    ArrayCovariance,
    Covariance,
    LinearOperatorCovariance,
    ScaledCovariance,
)

from .._arithmetic import (
    FunctionScaledProcessVectorCrossCovariance,
    LinOpProcessVectorCrossCovariance,
    ScaledProcessVectorCrossCovariance,
    SumProcessVectorCrossCovariance,
)
from .._zero import Zero
from ._dirac import CovarianceFunction_Dirac_Identity, CovarianceFunction_Identity_Dirac
from ._evaluation import (
    CovarianceFunction_Evaluation_Identity,
    CovarianceFunction_Identity_Evaluation,
)


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: ScaledProcessVectorCrossCovariance, /) -> Covariance:
    return ScaledCovariance(
        self(pv_crosscov.pv_crosscov),
        pv_crosscov.scalar,
        reverse=not pv_crosscov.reverse
        if pv_crosscov.scale_randvar
        else pv_crosscov.reverse,
    )


@ScaledLinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: ScaledProcessVectorCrossCovariance, /) -> Covariance:
    if pv_crosscov.reverse:
        alpha = pv_crosscov.scalar
        beta = self.scalar
    else:
        alpha = self.scalar
        beta = pv_crosscov.scalar
    cov = self.linfunctl(pv_crosscov.pv_crosscov)
    assert isinstance(cov, Covariance)
    return LinearOperatorCovariance(
        RankOneHadamardProduct(alpha, beta, cov.linop),
        cov.shape0,
        cov.shape1,
    )


@VectorizedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: FunctionScaledProcessVectorCrossCovariance, /) -> Covariance:
    centers = (self.domains.pure_array[..., 0] + self.domains.pure_array[..., 1]) / 2

    center_vals = pv_crosscov.fn(centers)

    return (center_vals * self)(pv_crosscov.pv_crosscov)


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, sum_pv_crosscov: SumProcessVectorCrossCovariance, /) -> Covariance:
    return functools.reduce(
        operator.add,
        (self(summand) for summand in sum_pv_crosscov.pv_crosscovs),
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    LinOpProcessVectorCrossCovariance
)
def _(self, pv_crosscov: LinOpProcessVectorCrossCovariance, /) -> Covariance:
    cov = self(pv_crosscov.pv_crosscov)
    axis = cov.ndim0 - 1 if pv_crosscov.reverse else cov.ndim0 + cov.ndim1 - 1

    array_res = pv_crosscov.linop(cov.array, axis=axis)
    shape0 = array_res.shape[: cov.ndim0]
    shape1 = array_res.shape[cov.ndim0 :]
    return ArrayCovariance(array_res, shape0, shape1)


@LinearFunctional.__call__.register(Zero)  # pylint: disable=no-member
def _(self: LinearFunctional, pv_crosscov: Zero, /) -> Covariance:
    if pv_crosscov.reverse:
        return LinearOperatorCovariance(
            pn.linops.Zero((pv_crosscov.randvar_size, self.output_size)),
            pv_crosscov.randvar_shape,
            self.output_shape,
        )
    return LinearOperatorCovariance(
        pn.linops.Zero((self.output_size, pv_crosscov.randvar_size)),
        self.output_shape,
        pv_crosscov.randvar_shape,
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Dirac_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Dirac_Identity, /) -> Covariance:
    res = self(pv_crosscov.covfunc, argnum=1)(pv_crosscov.dirac.X)
    return ArrayCovariance(res, pv_crosscov.dirac.output_shape, self.output_shape)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Dirac
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Dirac, /) -> Covariance:
    res = self(pv_crosscov.covfunc, argnum=0)(pv_crosscov.dirac.X)
    return ArrayCovariance(res, self.output_shape, pv_crosscov.dirac.output_shape)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Evaluation_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Evaluation_Identity, /) -> Covariance:
    kL = self(pv_crosscov.covfunc, argnum=1)
    X = pv_crosscov.evaluation_fctl.X
    res = kL.evaluate_linop(X)
    X_batch_shape = X.shape[: X.ndim - kL.randproc_input_ndim]

    return LinearOperatorCovariance(
        res, kL.randproc_output_shape + X_batch_shape, kL.randvar_shape
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Evaluation
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Evaluation, /) -> Covariance:
    # Result shape is randvar_shape + batch_shape + output_shape
    Lk = self(pv_crosscov.covfunc, argnum=0)
    X = pv_crosscov.evaluation_fctl.X
    res = Lk.evaluate_linop(X)
    X_batch_shape = X.shape[: X.ndim - Lk.randproc_input_ndim]

    return LinearOperatorCovariance(
        res, Lk.randvar_shape, Lk.randproc_output_shape + X_batch_shape
    )
