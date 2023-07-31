from __future__ import annotations

import functools

import numpy as np
import probnum as pn
from probnum.typing import FloatLike, ShapeLike

from ._laplacian import WeightedLaplacian


class WaveOperator(WeightedLaplacian):
    def __init__(self, domain_shape: ShapeLike, c: FloatLike = 1.0) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if len(domain_shape) != 1:
            raise ValueError(
                "The `WaveOperator` only applies to functions with `input_ndim == 1`."
            )

        self._c = float(c)

        laplacian_weights = np.zeros(domain_shape, dtype=np.double)
        laplacian_weights[0] = 1.0
        laplacian_weights[1:] = -self._c**2

        super().__init__(
            laplacian_weights,
        )

    @property
    def c(self) -> np.ndarray:
        return self._c

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)
