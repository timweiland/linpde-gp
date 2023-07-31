from collections.abc import Sequence

import numpy as np
import probnum as pn
from jax import numpy as jnp
from probnum.typing import ArrayLike

from . import _jax


class StackedFunction(_jax.JaxFunction):
    def __init__(self, fns: ArrayLike) -> None:
        self._fns = np.asarray(fns)
        fns_flat = self._fns.reshape(-1, order="C")

        input_shape = fns_flat[0].input_shape

        assert all(f.input_shape == input_shape for f in fns_flat)
        assert all(f.output_shape == fns_flat[0].output_shape for f in fns_flat)

        output_shape = self._fns.shape + fns_flat[0].output_shape
        self._single_output_shape = fns_flat[0].output_shape

        super().__init__(input_shape, output_shape)

    @property
    def fns(self) -> np.ndarray:
        return self._fns

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        batch_shape = x.shape[: x.ndim - self.input_ndim]

        evals = np.empty_like(self._fns, dtype=np.object_)
        for idx, fn in np.ndenumerate(self._fns):
            evals[idx] = fn(x)

        res = np.zeros(self._fns.shape + batch_shape + self._single_output_shape)
        for idx, eval_at_idx in np.ndenumerate(evals):
            res[idx] = eval_at_idx
        batch_start = len(self._fns.shape)
        batch_ndim = len(batch_shape)
        res = np.moveaxis(
            res, np.arange(batch_start, batch_start + batch_ndim), np.arange(batch_ndim)
        )
        return res

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_shape = x.shape[: x.ndim - self.input_ndim]

        evals = jnp.empty_like(self._fns, dtype=np.object_)
        for idx, fn in np.ndenumerate(self._fns):
            evals[idx] = fn(x)

        res = jnp.zeros(self._fns.shape + batch_shape + self._single_output_shape)
        for idx, eval_at_idx in np.ndenumerate(evals):
            res[idx] = eval_at_idx
        batch_start = len(self._fns.shape)
        batch_ndim = len(batch_shape)
        res = jnp.moveaxis(
            res, np.arange(batch_start, batch_start + batch_ndim), np.arange(batch_ndim)
        )
        return res
