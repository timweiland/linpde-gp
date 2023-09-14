import functools

from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike

from .. import domains
from ._jax import JaxFunction


class TruncatedSineSeries(JaxFunction):
    def __init__(
        self,
        domain: domains.Interval | domains.Box,
        coefficients: ArrayLike,
    ) -> None:
        domain = domains.asdomain(domain)

        if not isinstance(domain, (domains.Interval, domains.Box)):
            raise TypeError("`domain` must be `Interval` or `Box`")

        self._domain = domain

        super().__init__(input_shape=self._domain.shape, output_shape=())

        coefficients = np.asarray(coefficients)

        input_size = int(np.prod(domain.shape))
        if coefficients.ndim != input_size:
            raise ValueError()

        self._coefficients = coefficients

    @property
    def domain(self) -> domains.Interval:
        return self._domain

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @functools.cached_property
    def half_angular_frequencies(self) -> np.ndarray:
        if isinstance(self._domain, domains.Interval):
            l, r = self._domain

            return np.pi * np.arange(1, self._coefficients.shape[-1] + 1) / (r - l)
        else:
            widths = self._domain.bounds[:, 1] - self._domain.bounds[:, 0]

            freqs_per_dim = [
                np.pi * np.arange(1, self.coefficients.shape[i] + 1) / widths[i]
                for i in range(self._coefficients.ndim)
            ]
            return np.stack(np.meshgrid(*freqs_per_dim, indexing="ij"), axis=-1)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        if isinstance(self._domain, domains.Interval):
            l, _ = self._domain

            return np.sum(
                self._coefficients
                * np.sin(self.half_angular_frequencies * (x[..., None] - l)),
                axis=-1,
            )
        else:
            ls = self._domain.bounds[:, 0]

            return np.sum(
                self._coefficients
                * np.prod(
                    np.sin(
                        self.half_angular_frequencies * (x - ls)[..., None, None, :]
                    ),
                    axis=-1,
                ),
                axis=(-2, -1),
            )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        if isinstance(self._domain, domains.Interval):
            l, _ = self._domain

            return jnp.sum(
                self._coefficients
                * jnp.sin(self.half_angular_frequencies * (x[..., None] - l)),
                axis=-1,
            )
        else:
            ls = self._domain[:, 0]

            return jnp.sum(
                self._coefficients
                * jnp.prod(
                    jnp.sin(self.half_angular_frequencies * (x - ls)[..., None, None:]),
                    axis=-1,
                ),
                axis=(-2, -1),
            )
