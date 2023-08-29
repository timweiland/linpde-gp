import functools
import operator
from typing import List

import numpy as np
import probnum as pn

from linpde_gp.linfunctls import LinearFunctional
from linpde_gp.randvars import (
    ArrayCovariance,
    Covariance,
    ScaledCovariance,
    LinearOperatorCovariance,
)

from .._arithmetic import (
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
        shape = pv_crosscov.randvar_shape + self.output_shape
        return ArrayCovariance(
            np.zeros(shape), pv_crosscov.randvar_shape, self.output_shape
        )
    shape = self.output_shape + pv_crosscov.randvar_shape
    return ArrayCovariance(
        np.zeros(shape), self.output_shape, pv_crosscov.randvar_shape
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
    res = self(pv_crosscov.covfunc, argnum=1)(pv_crosscov.evaluation_fctl.X)
    X_batch_shape = pv_crosscov.evaluation_fctl.X.shape[: -kL.randproc_input_ndim]

    res = _move_shape_blocks(
        res, [X_batch_shape, kL.randproc_output_shape, kL.randvar_shape], [1, 0, 2]
    )
    return ArrayCovariance(
        res, kL.randproc_output_shape + X_batch_shape, kL.randvar_shape
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Evaluation
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Evaluation, /) -> Covariance:
    # Result shape is randvar_shape + batch_shape + output_shape
    Lk = self(pv_crosscov.covfunc, argnum=0)
    res = self(pv_crosscov.covfunc, argnum=0)(pv_crosscov.evaluation_fctl.X)
    X_batch_shape = pv_crosscov.evaluation_fctl.X.shape[: -Lk.randproc_input_ndim]

    # Reshape to randvar_shape + output_shape + batch_shape (reorder)
    res = _move_shape_blocks(
        res, [Lk.randvar_shape, X_batch_shape, Lk.randproc_output_shape], [0, 2, 1]
    )
    return ArrayCovariance(
        res, Lk.randvar_shape, Lk.randproc_output_shape + X_batch_shape
    )


def _move_shape_blocks(x: np.ndarray, input_shapes, new_positions: List[int]):
    shape_indices = []
    cur_offset = 0
    for input_shape in input_shapes:
        shape_indices.append(tuple(range(cur_offset, cur_offset + len(input_shape))))
        cur_offset += len(input_shape)
    output_shape_indices = [shape_indices[i] for i in new_positions]
    return np.transpose(x, functools.reduce(operator.add, output_shape_indices))
