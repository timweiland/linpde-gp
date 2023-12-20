import numpy as np
import probnum as pn

import pytest

from linpde_gp.linops import RankOneHadamardProduct


@pytest.fixture(params=[(1, 1), (2, 2), (3, 3), (4, 5), (5, 4), (10, 10)])
def rand_mat(request):
    np.random.seed(4590290)
    return np.random.rand(*request.param)


@pytest.fixture
def rand_alpha(rand_mat):
    np.random.seed(23498657)
    return np.random.rand(rand_mat.shape[0])


@pytest.fixture
def rand_beta(rand_mat):
    np.random.seed(7899877)
    return np.random.rand(rand_mat.shape[1])


@pytest.fixture
def linop(rand_alpha, rand_beta, rand_mat):
    return RankOneHadamardProduct(rand_alpha, rand_beta, pn.linops.Matrix(rand_mat))


@pytest.fixture
def dense_mat(rand_alpha, rand_beta, rand_mat):
    return np.outer(rand_alpha, rand_beta) * rand_mat


def test_matmul(linop, dense_mat):
    np.random.seed(12309124)
    x = np.random.rand(linop.shape[1])
    np.testing.assert_allclose(linop @ x, dense_mat @ x)


def test_todense(linop, dense_mat):
    np.testing.assert_allclose(linop.todense(), dense_mat)


def test_transpose(linop, dense_mat):
    np.testing.assert_allclose(linop.T.todense(), dense_mat.T)


def test_diagonal(linop, dense_mat):
    if linop.shape[0] != linop.shape[1]:
        pytest.skip()
    np.testing.assert_allclose(linop.diagonal(), np.diag(dense_mat))
