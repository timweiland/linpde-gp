from typing import TYPE_CHECKING

from probnum.linops import LinearOperator

if TYPE_CHECKING:
    from linpde_gp.randprocs.crosscov import ProcessVectorCrossCovariance

import numpy as np
import torch


class ShapeAlignmentLinearOperator(LinearOperator):
    def __init__(
        self,
        pv_crosscov: "ProcessVectorCrossCovariance",
        x: np.ndarray,
    ):
        self._pv_crosscov = pv_crosscov
        self._x = x
        self._inner_linop = pv_crosscov.evaluate_linop(x)

        self._x_batch_shape = x.shape[: x.ndim - pv_crosscov.randproc_input_ndim]
        self._x_output_shape = pv_crosscov.randproc_output_shape

        super().__init__(
            shape=self._inner_linop.shape,
            dtype=x.dtype,
        )

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        inner_res = self._inner_linop @ x
        original_shape = inner_res.shape
        expanded_shape = list(original_shape)

        # Expand axis -2 to self._x_output_shape + self._x_batch_shape
        if inner_res.ndim == 1:
            expanded_shape = self._x_output_shape + self._x_batch_shape
        else:
            expanded_shape[-2:-1] = self._x_output_shape + self._x_batch_shape
        inner_res = inner_res.reshape(expanded_shape)
        # Move batch shape to front
        if inner_res.ndim == 1:
            inner_res = np.moveaxis(
                inner_res,
                np.arange(
                    len(self._x_output_shape),
                    len(self._x_output_shape) + len(self._x_batch_shape),
                ),
                np.arange(len(self._x_batch_shape)),
            )
        else:
            x_shape_start = (
                inner_res.ndim
                - 1
                - len(self._x_output_shape)
                - len(self._x_batch_shape)
            )
            inner_res = np.moveaxis(
                inner_res,
                np.arange(
                    x_shape_start + len(self._x_output_shape),
                    x_shape_start
                    + len(self._x_output_shape)
                    + len(self._x_batch_shape),
                ),
                np.arange(x_shape_start, x_shape_start + len(self._x_batch_shape)),
            )
        return inner_res.reshape(original_shape)

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        inner_res = self._inner_linop @ x
        original_shape = inner_res.shape
        expanded_shape = list(original_shape)

        # Expand axis -2 to self._x_output_shape + self._x_batch_shape
        if inner_res.ndim == 1:
            expanded_shape = self._x_output_shape + self._x_batch_shape
        else:
            expanded_shape[-2:-1] = self._x_output_shape + self._x_batch_shape
        inner_res = inner_res.reshape(expanded_shape)
        # Move batch shape to front
        if inner_res.ndim == 1:
            inner_res = torch.moveaxis(
                inner_res,
                tuple(torch.arange(
                    len(self._x_output_shape),
                    len(self._x_output_shape) + len(self._x_batch_shape),
                )),
                tuple(torch.arange(len(self._x_batch_shape))),
            )
        else:
            x_shape_start = (
                inner_res.ndim
                - 1
                - len(self._x_output_shape)
                - len(self._x_batch_shape)
            )
            inner_res = torch.moveaxis(
                inner_res,
                tuple(torch.arange(
                    x_shape_start + len(self._x_output_shape),
                    x_shape_start
                    + len(self._x_output_shape)
                    + len(self._x_batch_shape),
                )),
                tuple(torch.arange(x_shape_start, x_shape_start + len(self._x_batch_shape))),
            )
        return inner_res.reshape(original_shape)