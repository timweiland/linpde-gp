import numpy as np
from linpde_gp.domains import Interval
from linpde_gp.linfunctls import LebesgueIntegral, VectorizedLebesgueIntegral

from ._constant import Constant
from ._piecewise import Piecewise
from ._polynomial import Polynomial


@VectorizedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Polynomial) -> np.ndarray:
    assert self.domains.common_type is Interval
    f_antideriv = f.integrate()

    return f_antideriv(self.domains.pure_array[..., 1]) - f_antideriv(
        self.domains.pure_array[..., 0]
    )


@VectorizedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Piecewise) -> np.ndarray:
    assert self.domains.common_type is Interval

    @np.vectorize
    def integrate(domain):
        idx_start = np.searchsorted(f.xs, domain[0], side="right") - 1
        idx_stop = np.searchsorted(f.xs, domain[1], side="right")

        if idx_start == -1:
            raise ValueError("Integral domain is larger than function domain")

        xs = (domain[0],) + tuple(f.xs[idx_start + 1 : idx_stop - 1]) + (domain[1],)
        fs = f.pieces[idx_start : idx_stop - 1]

        return sum(
            LebesgueIntegral((piece_l, piece_r))(piece)
            for piece, piece_l, piece_r in zip(fs, xs[:-1], xs[1:])
        )

    return integrate(self.domains.array)

@VectorizedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Constant) -> np.ndarray:
    if f.value == 0.:
        return np.zeros(self.domains.shape)
    return f.value * self.domains.volume