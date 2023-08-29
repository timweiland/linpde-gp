import functools
import operator

import numpy as np
from probnum.typing import ArrayLike

from ._box import Box
from ._interval import Interval
from ._vectorize import VectorizedDomain


class TensorProductDomain(VectorizedDomain):
    def __init__(self, *factors: ArrayLike):
        self._factors = [VectorizedDomain(factor) for factor in factors]

    @property
    def factors(self) -> tuple[VectorizedDomain]:
        return self._factors

    @functools.cached_property
    def _dense_representation(self) -> VectorizedDomain:
        starts = np.meshgrid(
            *(factor.pure_array[..., 0] for factor in self._factors), indexing="ij"
        )
        starts = np.stack(starts, axis=-1)
        ends = np.meshgrid(
            *(factor.pure_array[..., 1] for factor in self._factors), indexing="ij"
        )
        ends = np.stack(ends, axis=-1)

        domains = np.stack((starts, ends), axis=-1)
        return VectorizedDomain.from_pure_array(domains, type=Box)

    @property
    def array(self) -> np.ndarray:
        return self._dense_representation.array

    @property
    def pure_array(self) -> np.ndarray:
        return self._dense_representation.pure_array

    @property
    def shape(self):
        return functools.reduce(
            operator.add,
            (factor.shape for factor in self._factors),
        )

    @property
    def input_shape(self):
        return (len(self._factors),)

    @functools.cached_property
    def volume(self):
        return functools.reduce(np.kron, (factor.volume for factor in self._factors))

    @classmethod
    def from_endpoints(cls, *endpoints: ArrayLike):
        return cls(
            *(
                [
                    Interval(endpoint_arr[i], endpoint_arr[i + 1])
                    for i in range(len(endpoint_arr) - 1)
                ]
                for endpoint_arr in endpoints
            )
        )
    
    @property
    def common_type(self):
        return Box
