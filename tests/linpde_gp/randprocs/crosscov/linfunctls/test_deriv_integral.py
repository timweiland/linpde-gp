import numpy as np
import probnum as pn

import pytest
from pytest_cases import fixture

import linpde_gp
from linpde_gp.domains import Interval
from linpde_gp.linfuncops.diffops import Derivative, PartialDerivative
from linpde_gp.linfunctls import LebesgueIntegral
from linpde_gp.randprocs.covfuncs import Matern
from linpde_gp.randprocs.crosscov import (
    ScaledProcessVectorCrossCovariance,
    SumProcessVectorCrossCovariance,
)
from linpde_gp.randprocs.crosscov.linfunctls import (
    CovarianceFunction_Evaluation_Identity,
    CovarianceFunction_Identity_Difference,
    CovarianceFunction_Identity_Evaluation,
)


@fixture
def matern():
    return Matern((), nu=2.5)


@fixture
def lebesgue_integral():
    return LebesgueIntegral(Interval(0, 1))


@fixture
@pytest.mark.parametrize("order", [1, 2])
def derivative(order: int):
    return Derivative(order)


def reduce_derivative_order(
    k: pn.randprocs.covfuncs.CovarianceFunction,
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


def reduce_single_derivative_order(
    k: pn.randprocs.covfuncs.CovarianceFunction, D: PartialDerivative, argnum: int
):
    if argnum == 0:
        D1 = D
        D2 = Derivative(0)
    else:
        D1 = Derivative(0)
        D2 = D
    return reduce_derivative_order(k, D1, D2, argnum)


@pytest.mark.parametrize("argnum", [0, 1])
def test_kDI(matern, lebesgue_integral, derivative, argnum):
    kDI = lebesgue_integral(derivative(matern, argnum=argnum), argnum=argnum)
    assert isinstance(kDI, CovarianceFunction_Identity_Difference)

    k_new = reduce_single_derivative_order(matern, derivative, argnum)
    assert isinstance(kDI.covfunc, type(k_new))


@pytest.mark.parametrize("argnum", [0, 1])
def test_DkI(matern, lebesgue_integral, derivative, argnum):
    DkI = derivative(lebesgue_integral(matern, argnum=argnum))
    assert isinstance(DkI, ScaledProcessVectorCrossCovariance)
    assert DkI.scalar == -1.0
    assert isinstance(DkI.pv_crosscov, CovarianceFunction_Identity_Difference)

    k_new = reduce_single_derivative_order(matern, derivative, argnum)
    assert isinstance(DkI.pv_crosscov.covfunc, type(k_new))


@fixture
@pytest.mark.parametrize("order", [1, 2])
def D1(order: int):
    return Derivative(order)


@fixture
@pytest.mark.parametrize("order", [1, 2])
def D2(order: int):
    return Derivative(order)


@pytest.mark.parametrize("argnum", [0, 1])
def test_D1kD2_I_after(matern, lebesgue_integral, D1, D2, argnum):
    DkDI = lebesgue_integral(D1(D2(matern, argnum=1), argnum=0), argnum=argnum)
    assert isinstance(DkDI, CovarianceFunction_Identity_Difference)

    k_new = reduce_derivative_order(matern, D1, D2, argnum)
    assert isinstance(DkDI.covfunc, type(k_new))


def test_D1kD2_I_before(matern, lebesgue_integral, D1, D2):
    DkDI = D1(lebesgue_integral(D2(matern, argnum=1), argnum=1))
    assert isinstance(DkDI, SumProcessVectorCrossCovariance)
    assert len(DkDI.pv_crosscovs) == 2
    k_new = reduce_derivative_order(matern, D1, D2, 1)
    for pv_crosscov in DkDI.pv_crosscovs:
        actual_crosscov = pv_crosscov
        if isinstance(actual_crosscov, ScaledProcessVectorCrossCovariance):
            actual_crosscov = actual_crosscov.pv_crosscov
        assert isinstance(
            actual_crosscov,
            (
                CovarianceFunction_Identity_Evaluation,
                CovarianceFunction_Evaluation_Identity,
            ),
        )
        assert isinstance(actual_crosscov.covfunc, type(k_new))
