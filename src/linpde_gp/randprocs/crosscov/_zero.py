import numpy as np
import probnum as pn
from jax import numpy as jnp

from ._pv_crosscov import ProcessVectorCrossCovariance


class Zero(ProcessVectorCrossCovariance):
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

        return np.zeros_like(
            x,
            shape=(
                self.randvar_shape + batch_shape + self.randproc_output_shape
                if self.reverse
                else batch_shape + self.randproc_output_shape + self.randvar_shape
            ),
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

        return jnp.zeros_like(
            x,
            shape=(
                self.randvar_shape + batch_shape + self.randproc_output_shape
                if self.reverse
                else batch_shape + self.randproc_output_shape + self.randvar_shape
            ),
        )

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]
        output_shape = batch_shape + self.randproc_output_shape
        output_size = np.prod(output_shape)

        return pn.linops.Zero(
            shape=(
                (self.randvar_size, output_size)
                if self.reverse
                else (output_size, self.randvar_size)
            ),
            dtype=x.dtype,
        )
