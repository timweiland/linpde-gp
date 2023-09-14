import functools
import operator
from typing import Generic, Optional, TypeVar

from jax import numpy as jnp
import numpy as np
from probnum import linops
from probnum.randprocs.covfuncs._arithmetic_fallbacks import (
    ScaledCovarianceFunction,
    SumCovarianceFunction,
)
from probnum.typing import ArrayLike, ScalarLike, ScalarType
from pykeops.numpy import LazyTensor
from pykeops.torch import LazyTensor as LazyTensor_Torch
import torch

from ._jax import JaxCovarianceFunction, JaxCovarianceFunctionMixin


class JaxScaledCovarianceFunction(JaxCovarianceFunctionMixin, ScaledCovarianceFunction):
    def __init__(self, covfunc: JaxCovarianceFunction, scalar: ScalarLike) -> None:
        if not isinstance(covfunc, JaxCovarianceFunctionMixin):
            raise TypeError()

        super().__init__(covfunc, scalar)

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @property
    def covfunc(self) -> JaxCovarianceFunction:
        return self._covfunc

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._scalar * self.covfunc.jax(x0, x1)

    def __rmul__(self, other: ArrayLike) -> JaxCovarianceFunction:
        if np.ndim(other) == 0:
            return JaxScaledCovarianceFunction(
                self.covfunc,
                scalar=np.asarray(other) * self.scalar,
            )

        return super().__rmul__(other)

    def _keops_lazy_tensor(self, x0: np.ndarray, x1: np.ndarray | None) -> LazyTensor:
        return self._scalar[()] * self.covfunc._keops_lazy_tensor(x0, x1)

    def _keops_lazy_tensor_torch(
        self, x0: torch.Tensor, x1: torch.Tensor | None
    ) -> LazyTensor_Torch:
        return torch.from_numpy(self._scalar[()]).to(
            x0.device
        ) * self.covfunc._keops_lazy_tensor_torch(x0, x1)

    def linop(
        self, x0: ArrayLike, x1: ArrayLike | None = None
    ) -> linops.LinearOperator:
        return self._scalar * self.covfunc.linop(x0, x1)


T = TypeVar("T", bound=JaxCovarianceFunction)


class JaxSumCovarianceFunction(
    JaxCovarianceFunctionMixin, SumCovarianceFunction, Generic[T]
):
    def __init__(self, *summands: T):
        if not all(
            isinstance(summand, JaxCovarianceFunctionMixin) for summand in summands
        ):
            raise TypeError()

        super().__init__(*summands)

    @property
    def summands(self) -> tuple[T, ...]:
        return self._summands

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return functools.reduce(
            operator.add,
            (summand.jax(x0, x1) for summand in self.summands),
        )

    def linop(
        self, x0: ArrayLike, x1: ArrayLike | None = None
    ) -> linops.LinearOperator:
        return functools.reduce(
            operator.add,
            (summand.linop(x0, x1) for summand in self.summands),
        )
