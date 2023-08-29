import functools

import numpy as np
from probnum.typing import ArrayLike

from ._box import Box
from ._domain import Domain
from ._interval import Interval


def convert_to_numpy_array(obj):
    if isinstance(obj, Domain):
        arr = np.empty((), dtype=object)
        arr[()] = obj
        return arr
    elif isinstance(obj, (list, tuple)):
        sub_arrays = [convert_to_numpy_array(item) for item in obj]
        sub_shape = sub_arrays[0].shape
        assert all(sub_shape == sub_array.shape for sub_array in sub_arrays)
        if sub_shape == ():
            sub_arrays = [sub_array[()] for sub_array in sub_arrays]
        arr = np.empty((len(sub_arrays),) + sub_shape, dtype=object)
        arr[:] = sub_arrays
        return arr
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for idx, item in np.ndenumerate(obj):
            if not isinstance(item, Domain):
                raise ValueError(f"Invalid entry at index {idx}: {item}")
        return obj
    else:
        raise ValueError("Unsupported data type encountered.")


class VectorizedDomain:
    def __init__(self, domains: ArrayLike):
        self._array = convert_to_numpy_array(domains)

        self._input_shape = None
        for idx in np.ndindex(self._array.shape):
            if self._input_shape is None:
                self._input_shape = self._array[idx].shape
            elif self._input_shape != self._array[idx].shape:
                raise ValueError(
                    f"Input shapes must be equal, got {self._input_shape} and " 
                    f"{self._array[idx].shape} at index {idx}."
                )
        self._pure_array = np.asarray(self._array.tolist())

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def pure_array(self) -> np.ndarray:
        return self._pure_array
    
    @property
    def shape(self):
        return self.array.shape
    
    @property
    def input_shape(self):
        return self._input_shape
    
    @functools.cached_property
    def volume(self):
        volume_arr = np.zeros(self.shape)
        for idx in np.ndindex(self.shape):
            volume_arr[idx] = self.array[idx].volume
        return volume_arr

    @classmethod
    def from_pure_array(cls, pure_array, type=Interval):
        if type is Interval:
            assert len(pure_array.shape) >= 1
            assert pure_array.shape[-1] == 2
            domain_arr = np.empty(pure_array.shape[:-1], dtype=object)
            for idx in np.ndindex(pure_array.shape[:-1]):
                domain_arr[idx] = Interval(*pure_array[idx])
            return cls(domain_arr)
        elif type is Box:
            assert len(pure_array.shape) >= 2
            assert pure_array.shape[-1] == 2
            domain_arr = np.empty(pure_array.shape[:-2], dtype=object)
            for idx in np.ndindex(pure_array.shape[:-2]):
                domain_arr[idx] = Box(pure_array[idx])
            return cls(domain_arr)
        raise NotImplementedError()

    @functools.cached_property
    def common_type(self):
        res = None
        for _, item in np.ndenumerate(self.array):
            if res is None:
                res = type(item)
            elif res != type(item):
                return None
        return res

    def factorize(self):
        if self.common_type is Interval:
            return self
        elif self.common_type is Box:
            assert len(self.pure_array.shape) >= 2
            assert self.pure_array.shape[-1] == 2
            return tuple(
                VectorizedDomain.from_pure_array(
                    self.pure_array[..., idx, :], type=Interval
                )
                for idx in range(self.pure_array.shape[-2])
            )
        raise ValueError(f"Cannot factorize domain type {self.common_type}.")

    def __getitem__(self, idx):
        return self.array[idx]

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)

    def __repr__(self):
        return repr(self.array)
