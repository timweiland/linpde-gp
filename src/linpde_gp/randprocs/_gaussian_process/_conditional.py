from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import probnum as pn
from linpde_gp import linfunctls
from linpde_gp.functions import JaxFunction
from linpde_gp.linfuncops import LinearFunctionOperator
from linpde_gp.linfunctls import CompositeLinearFunctional, LinearFunctional
from linpde_gp.linops import (
    BlockMatrix,
    BlockMatrix2x2,
    DenseCholeskySolverLinearOperator,
    ShapeAlignmentLinearOperator,
)
from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance
from linpde_gp.randvars import Covariance, LinearOperatorCovariance
from linpde_gp.solvers import (
    CholeskySolver,
    ConcreteCholeskySolver,
    ConcreteGPSolver,
    GPInferenceParams,
    GPSolver,
)
from linpde_gp.solvers.itergp import ConcreteIterGPSolver
from linpde_gp.typing import RandomVariableLike
from numpy.typing import ArrayLike

pn.config.register(
    "default_solver_linpde_gp",
    CholeskySolver(),
    "Default solver for GP conditioning in linpde-gp.",
)


class ConditionalGaussianProcess(pn.randprocs.GaussianProcess):
    @classmethod
    def from_observations(
        cls,
        prior: pn.randprocs.GaussianProcess,
        Y: ArrayLike,
        X: ArrayLike | None = None,
        *,
        L: None | LinearFunctional | LinearFunctionOperator = None,
        b: None | RandomVariableLike = None,
        noise: float = 0.0,
        solver: GPSolver = None,
    ):
        if solver is None:
            solver = pn.config.default_solver_linpde_gp
        Y, L, b, kLa, gram = cls._preprocess_observations(
            prior=prior,
            Y=Y,
            X=X,
            L=L,
            b=b,
            noise=noise,
        )

        kLas = ConditionalGaussianProcess.PriorPredictiveCrossCovariance((kLa,))

        return cls(
            prior=prior,
            Ys=(Y,),
            Ls=(L,),
            bs=(b,),
            kLas=kLas,
            gram_matrix=gram,
            solver=solver,
        )

    def __init__(
        self,
        *,
        prior: pn.randprocs.GaussianProcess,
        Ys: Sequence[np.ndarray],
        Ls: Sequence[LinearFunctional],
        bs: Sequence[pn.randvars.Normal | pn.randvars.Constant | None],
        kLas: ConditionalGaussianProcess.PriorPredictiveCrossCovariance,
        gram_matrix: pn.linops.LinearOperator,
        solver: GPSolver,
        full_representer_weights: np.ndarray | None = None,
    ):
        self._prior = prior

        self._Ys = tuple(Ys)
        self._Ls = tuple(Ls)
        self._bs = tuple(bs)

        self._kLas = kLas

        self._gram_matrix = gram_matrix

        inference_params = GPInferenceParams(
            prior, gram_matrix, Ys, Ls, bs, kLas, None, full_representer_weights
        )
        self._solver = solver.get_concrete_solver(inference_params)
        self._abstract_solver = solver

        super().__init__(
            mean=ConditionalGaussianProcess.Mean(
                prior_mean=self._prior.mean,
                kLas=self._kLas,
                solver=self._solver,
            ),
            cov=self._solver.posterior_cov,
        )

    @functools.cached_property
    def gram(self) -> pn.linops.LinearOperator:
        return self._gram_matrix

    @property
    def abstract_solver(self) -> GPSolver:
        return self._abstract_solver

    @property
    def solver(self) -> ConcreteGPSolver:
        return self._solver

    @property
    def representer_weights(self) -> np.ndarray:
        return self._solver.compute_representer_weights()

    class PriorPredictiveCrossCovariance(ProcessVectorCrossCovariance):
        def __init__(
            self,
            kLas: Sequence[ProcessVectorCrossCovariance],
        ) -> None:
            self._kLas = tuple(kLas)

            assert all(
                kLa.randproc_input_shape == self._kLas[0].randproc_input_shape
                and kLa.randproc_output_shape == self._kLas[0].randproc_output_shape
                and not kLa.reverse
                for kLa in self._kLas
            )

            super().__init__(
                randproc_input_shape=self._kLas[0].randproc_input_shape,
                randproc_output_shape=self._kLas[0].randproc_output_shape,
                randvar_shape=(sum(kLa.randvar_size for kLa in self._kLas),),
                reverse=False,
            )

        def append(
            self, kLa: ProcessVectorCrossCovariance
        ) -> ConditionalGaussianProcess.PriorPredictiveCrossCovariance:
            return ConditionalGaussianProcess.PriorPredictiveCrossCovariance(
                self._kLas + (kLa,)
            )

        def replace_last(
            self, kLa: ProcessVectorCrossCovariance
        ) -> ConditionalGaussianProcess.PriorPredictiveCrossCovariance:
            return ConditionalGaussianProcess.PriorPredictiveCrossCovariance(
                self._kLas[:-1] + (kLa,)
            )

        def _evaluate(self, x: np.ndarray) -> np.ndarray:
            batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

            return np.concatenate(
                [
                    np.reshape(
                        kLa(x),  # shape: batch_shape + u_output_shape + Lu_output_shape
                        batch_shape + self.randproc_output_shape + (-1,),
                        "C",
                    )
                    for kLa in self._kLas
                ],
                axis=-1,
            )

        def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
            batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

            return jnp.concatenate(
                [
                    jnp.reshape(
                        kLa(x),  # shape: batch_shape + u_output_shape + Lu_output_shape
                        batch_shape + self.randproc_output_shape + (-1,),
                        "C",
                    )
                    for kLa in self._kLas
                ],
                axis=-1,
            )

        def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
            return BlockMatrix([[kLa.evaluate_linop(x) for kLa in self._kLas]])

        def __iter__(self) -> Iterator[ProcessVectorCrossCovariance]:
            for kLa in self._kLas:
                yield kLa

    class Mean(JaxFunction):
        def __init__(
            self,
            prior_mean: JaxFunction,
            kLas: ConditionalGaussianProcess.PriorPredictiveCrossCovariance,
            solver: ConcreteGPSolver,
        ):
            self._prior_mean = prior_mean
            self._kLas = kLas
            self._solver = solver

            super().__init__(
                input_shape=self._prior_mean.input_shape,
                output_shape=self._prior_mean.output_shape,
            )

        def _evaluate(self, x: np.ndarray) -> np.ndarray:
            m_x = self._prior_mean(x)
            kLas_x = ShapeAlignmentLinearOperator(self._kLas, x)
            correction = kLas_x @ self._solver.compute_representer_weights()

            desired_shape = m_x.shape
            return (m_x.reshape(correction.shape, order="C") + correction).reshape(
                desired_shape, order="C"
            )

        @functools.partial(jax.jit, static_argnums=0)
        def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
            m_x = self._prior_mean.jax(x)
            kLas_x = self._kLas.jax(x)

            return m_x + kLas_x @ self._solver.compute_representer_weights()

    def condition_on_observations(
        self,
        Y: ArrayLike,
        X: ArrayLike | None = None,
        *,
        L: LinearFunctional | LinearFunctionOperator | None = None,
        b: RandomVariableLike | None = None,
        noise: float = 1e-8,
        solver: GPSolver = None,
        fresh_start=False,
    ):
        if fresh_start:
            return ConditionalGaussianProcess.from_observations(
                prior=self,
                Y=Y,
                X=X,
                L=L,
                b=b,
                noise=noise,
                solver=solver,
            )
        if solver is None:
            solver = pn.config.default_solver_linpde_gp
        Y, L, b, kLa, gram = self._preprocess_observations(
            prior=self._prior,
            Y=Y,
            X=X,
            L=L,
            b=b,
            noise=noise,
        )

        # Compute lower-left block in the new covariance matrix
        gram_L_La_prev_blocks = L(self._kLas).linop

        # Update the Cholesky decomposition of the previous covariance matrix and the
        # representer weights

        gram_matrix = BlockMatrix2x2(
            self.gram,
            gram_L_La_prev_blocks.T,
            None,
            gram,
            is_spd=True,
        )

        kLas = self._kLas.append(kLa)

        return ConditionalGaussianProcess(
            prior=self._prior,
            Ys=self._Ys + (Y,),
            Ls=self._Ls + (L,),
            bs=self._bs + (b,),
            kLas=kLas,
            gram_matrix=gram_matrix,
            solver=solver,
        )

    @classmethod
    def _preprocess_observations(
        cls,
        *,
        prior: pn.randprocs.GaussianProcess,
        Y: ArrayLike,
        X: ArrayLike | None,
        L: LinearFunctional | LinearFunctionOperator | None,
        b: RandomVariableLike | None,
        noise: float = 1e-12,
    ) -> tuple[
        np.ndarray,
        LinearFunctional,
        pn.randvars.Normal | pn.randvars.Constant | None,
        ProcessVectorCrossCovariance,
        np.ndarray,
        pn.linops.LinearOperator,
    ]:
        # TODO: Allow `RandomProcessLike` for `b` ("b = b(X)")

        # Build measurement functional `L`
        match L:
            case LinearFunctional():
                if X is not None:
                    raise TypeError(
                        "If `L` is a `LinearFunctional`, `X` must be `None`."
                    )
            case LinearFunctionOperator():
                if X is None:
                    raise ValueError(
                        "`X` must not be omitted if `L` is a `LinearFunctionOperator`."
                    )

                L = L.to_linfunctl(X)
            case None:
                if X is None:
                    raise ValueError("`X` and `L` can not be omitted at the same time.")

                L = linfunctls._EvaluationFunctional(  # pylint: disable=protected-access
                    input_domain_shape=prior.input_shape,
                    input_codomain_shape=prior.output_shape,
                    X=X,
                )
            case _:
                raise TypeError("TODO")

        assert isinstance(L, LinearFunctional)

        # Check measurement noise model
        if b is not None:
            b = pn.randvars.asrandvar(b)

            if not isinstance(b, (pn.randvars.Constant, pn.randvars.Normal)):
                raise TypeError(
                    f"`b` must be a `Normal` or a `Constant` `RandomVariable`"
                    f"({type(b)=})"
                )

            if b.shape != L.output_shape:
                raise ValueError(f"{b.shape=} must be equal to {L.output_shape}")

        # Compute the joint measure (f, L[f])
        Lf = L(prior)
        kLa = L(prior.cov, argnum=1)

        # Compute predictive covariance matrix
        gram = Lf.cov

        # Check observations
        Y = np.asarray(Y)
        if (
            isinstance(
                L,
                linfunctls._EvaluationFunctional,  # pylint: disable=protected-access
            )
            and prior.mean.output_ndim > 0
        ):
            if Y.shape[-prior.mean.output_ndim :] != prior.mean.output_shape:
                raise ValueError(
                    f"Expected Y to have shape (batch shape) + "
                    f"{prior.mean.output_shape}, got shape {Y.shape}"
                )
            Y = np.moveaxis(
                Y,
                -prior.mean.output_ndim + np.arange(prior.mean.output_ndim),
                np.arange(prior.mean.output_ndim),
            )
        if Y.shape != L.output_shape:
            raise ValueError(
                f"Expected Y to have shape {L.output_shape}, got shape {Y.shape}."
            )
        Y = Y.reshape(-1, order="C")

        assert Y.size == Lf.cov.shape[1]

        if b is not None:
            gram = gram + pn.linops.aslinop(b.cov)

        if noise > 0:
            # Small nugget
            gram = gram + noise * pn.linops.Identity(gram.shape)

        gram.is_symmetric = True
        gram.is_positive_definite = True

        return Y, L, b, kLa, gram


pn.randprocs.GaussianProcess.condition_on_observations = (
    lambda *args, **kwargs: (  # pylint: disable=unnecessary-lambda
        ConditionalGaussianProcess.from_observations(*args, **kwargs)
    )
)


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    ConditionalGaussianProcess.PriorPredictiveCrossCovariance
)
def _(
    self, crosscov: ConditionalGaussianProcess.PriorPredictiveCrossCovariance, /
) -> ConditionalGaussianProcess.PriorPredictiveCrossCovariance:
    return ConditionalGaussianProcess.PriorPredictiveCrossCovariance(
        (self(kLa) for kLa in crosscov)
    )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
def _(
    self, crosscov: ConditionalGaussianProcess.PriorPredictiveCrossCovariance, /
) -> Covariance:
    linop_res = BlockMatrix([[self(kLa_prev).linop for kLa_prev in crosscov]])
    return LinearOperatorCovariance(
        linop_res, self.output_shape, crosscov.randvar_shape
    )


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    ConditionalGaussianProcess
)
def _(
    self, conditional_gp: ConditionalGaussianProcess, /
) -> ConditionalGaussianProcess:
    # pylint: disable=protected-access

    linop_prior = self(conditional_gp._prior)

    return ConditionalGaussianProcess(
        prior=linop_prior,
        Ys=conditional_gp._Ys,
        Ls=conditional_gp._Ls,
        bs=conditional_gp._bs,
        kLas=self(conditional_gp._kLas),
        gram_matrix=conditional_gp.gram,
        solver=conditional_gp.abstract_solver,
        full_representer_weights=conditional_gp.representer_weights,
    )


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
def _(
    self, conditional_gp: ConditionalGaussianProcess, /
) -> ConditionalGaussianProcess:
    # pylint: disable=protected-access

    linfunctl_prior = self(conditional_gp._prior)
    crosscov = self(conditional_gp._kLas).linop

    mean = linfunctl_prior.mean + crosscov @ conditional_gp.representer_weights
    if isinstance(conditional_gp.solver, ConcreteCholeskySolver):
        cho_linop = DenseCholeskySolverLinearOperator(conditional_gp.gram)
        cov = linfunctl_prior.cov - crosscov @ cho_linop @ crosscov.T
    elif isinstance(conditional_gp.solver, ConcreteIterGPSolver):
        cov = (
            linfunctl_prior.cov
            - crosscov @ conditional_gp.solver.inverse_approximation @ crosscov.T
        )

    return pn.randvars.Normal(mean, cov)


@LinearFunctional.__call__.register
@CompositeLinearFunctional.__call__.register
def _(
    self, mean: ConditionalGaussianProcess.Mean, /
) -> ConditionalGaussianProcess.Mean:
    L_prior_mean = self(mean._prior_mean)
    LkL = self(mean._kLas).linop

    return L_prior_mean + (LkL @ mean._solver.compute_representer_weights()).reshape(
        L_prior_mean.shape
    )
