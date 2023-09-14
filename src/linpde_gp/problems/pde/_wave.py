import functools
from collections.abc import Sequence

import numpy as np
import probnum as pn
from jax import numpy as jnp
from linpde_gp import domains, functions
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike
from probnum.typing import FloatLike

from ._bvp import DirichletBoundaryCondition, InitialBoundaryValueProblem
from ._linear_pde import LinearPDE


class WaveEquation(LinearPDE):
    def __init__(
        self,
        domain: DomainLike,
        rhs: pn.functions.Function | None = None,
        c: FloatLike = 1.0,
    ):
        self._c = float(c)

        super().__init__(
            domain=domain,
            diffop=diffops.WaveOperator(domain_shape=domain.shape, c=self._c),
            rhs=rhs,
        )


class WaveEquationDirichletProblem(InitialBoundaryValueProblem):
    def __init__(
        self,
        t0: FloatLike,
        spatial_domain: DomainLike,
        T: FloatLike = float("inf"),
        rhs: pn.functions.Function | None = None,
        c: FloatLike = 1.0,
        initial_values: pn.functions.Function | None = None,
        solution: pn.functions.Function | None = None,
    ):
        domain = domains.CartesianProduct(
            domains.Interval(t0, T),
            spatial_domain,
        )

        pde = WaveEquation(domain, rhs=rhs, c=c)

        # Initial condition
        if initial_values is None:
            initial_values = functions.Zero(
                input_shape=spatial_domain.shape, output_shape=()
            )

        if (
            initial_values.input_shape != spatial_domain.shape
            and initial_values.output_shape != ()
        ):
            raise ValueError()

        initial_condition = DirichletBoundaryCondition(domain[1], initial_values)

        # Spatial boundary conditions
        boundary_conditions = tuple(
            DirichletBoundaryCondition(
                domains.CartesianProduct(domain[0], boundary_part),
                np.zeros(()),
            )
            for boundary_part in domain[1].boundary
        )

        if solution is None:
            if isinstance(initial_values, functions.Zero):
                solution = functions.Zero(domain.shape, output_shape=())
            elif isinstance(domain[1], (domains.Box)):
                if (
                    isinstance(initial_values, functions.TruncatedSineSeries)
                    and initial_values.domain == domain[1]
                ):
                    solution = Solution_WaveEquation_DirichletProblem_ND_InitialTruncatedSineSeries_BoundaryZero(  # pylint: disable=line-too-long
                        t0=t0,
                        spatial_domain=spatial_domain,
                        initial_values=initial_values,
                        c=c,
                    )

        super().__init__(
            pde=pde,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )


class Solution_WaveEquation_DirichletProblem_ND_InitialTruncatedSineSeries_BoundaryZero(
    pn.functions.Function
):
    def __init__(
        self,
        t0: FloatLike,
        spatial_domain: domains.Box,
        initial_values: functions.TruncatedSineSeries,
        c: FloatLike,
    ):
        self._t0 = float(t0)
        self._spatial_domain = spatial_domain
        self._initial_values = initial_values
        self._c = float(c)

        assert isinstance(self._spatial_domain, domains.Box)
        assert self._spatial_domain == self._initial_values.domain

        super().__init__(
            input_shape=(1 + self._spatial_domain.shape[0],), output_shape=()
        )

    @functools.cached_property
    def _lambdas(self) -> np.ndarray:
        return self._c * np.linalg.norm(
            self._initial_values.half_angular_frequencies, axis=-1
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        ls = self._spatial_domain.bounds[:, 0]

        return np.sum(
            self._initial_values.coefficients
            * np.cos(self._lambdas * (x[..., 0] - self._t0)[..., None, None])
            * np.prod(
                np.sin(
                    self._initial_values.half_angular_frequencies
                    * (x[..., 1:] - ls)[..., None, None, :]
                ),
                axis=-1,
            ),
            axis=(-2, -1),
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        ls = self._spatial_domain.bounds[:, 0]

        return jnp.sum(
            self._initial_values.coefficients
            * jnp.cos(self._lambdas * (x[..., 0] - self._t0)[..., None, None])
            * jnp.prod(
                jnp.sin(
                    self._initial_values.half_angular_frequencies
                    * (x[..., 1:] - ls)[..., None, None, :]
                ),
                axis=(-2, -1),
            )
        )
