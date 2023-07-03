from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._lindiffop import LinearDifferentialOperator


class ShallowWaterOperator_1D_Mass(LinearDifferentialOperator):
    def __init__(self, mean_height: float):
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {
                    (0,): {MultiIndex((1, 0)): 1.0},
                    (1,): {MultiIndex((0, 1)): mean_height},
                },
                (2,),
                (2,),
            ),
            input_shapes=((2,), (2,)),
        )


class ShallowWaterOperator_1D_Momentum(LinearDifferentialOperator):
    def __init__(self, g: float = 9.81):
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {
                    (0,): {MultiIndex((0, 1)): g},
                    (1,): {MultiIndex((1, 0)): 1.0},
                },
                (2,),
                (2,),
            ),
            input_shapes=((2,), (2,)),
        )


def get_shallow_water_diffops_1D(mean_height: float, g: float = 9.81):
    """Returns the differential operators for the 1D shallow water equations.

    Args:
        mean_height (float): The mean height of the water.
        g (float): The gravitational constant.
    """
    return (
        ShallowWaterOperator_1D_Mass(mean_height),
        ShallowWaterOperator_1D_Momentum(g),
    )


class ShallowWaterOperator_2D_Mass(LinearDifferentialOperator):
    def __init__(self, mean_height):
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {
                    (0,): {MultiIndex((1, 0, 0)): 1.0},
                    (1,): {MultiIndex((0, 1, 0)): mean_height},
                    (2,): {MultiIndex((0, 0, 1)): mean_height},
                },
                (3,),
                (3,),
            ),
            input_shapes=((3,), (3,)),
        )


class ShallowWaterOperator_2D_Momentum_u(LinearDifferentialOperator):
    def __init__(self, g: float = 9.81, coriolis_coefficient: float = 2e-4):
        coefficient_dict = {
            (0,): {MultiIndex((0, 1, 0)): g},
            (1,): {MultiIndex((1, 0, 0)): 1.0},
        }
        if coriolis_coefficient != 0.0:
            coefficient_dict[(2,)] = {MultiIndex((0, 0, 0)): -coriolis_coefficient}
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                coefficient_dict,
                (3,),
                (3,),
            ),
            input_shapes=((3,), (3,)),
        )


class ShallowWaterOperator_2D_Momentum_v(LinearDifferentialOperator):
    def __init__(self, g: float = 9.81, coriolis_coefficient: float = 2e-4):
        coefficient_dict = {
            (0,): {MultiIndex((0, 0, 1)): g},
            (2,): {MultiIndex((1, 0, 0)): 1.0},
        }
        if coriolis_coefficient != 0.0:
            coefficient_dict[(1,)] = {MultiIndex((0, 0, 0)): coriolis_coefficient}
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                coefficient_dict,
                (3,),
                (3,),
            ),
            input_shapes=((3,), (3,)),
        )


def get_shallow_water_diffops_2D(
    mean_height: float, g: float = 9.81, coriolis_coefficient: float = 2e-4
):
    """Returns the differential operators for the 2D shallow water equations.

    Args:
        mean_height (float): The mean height of the water.
        g (float): The gravitational constant.
        coriolis_coefficient (float): The coriolis coefficient.
    """
    return (
        ShallowWaterOperator_2D_Mass(mean_height),
        ShallowWaterOperator_2D_Momentum_u(g, coriolis_coefficient),
        ShallowWaterOperator_2D_Momentum_v(g, coriolis_coefficient),
    )
