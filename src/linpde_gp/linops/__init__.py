from probnum.linops import *

from ._block import BlockMatrix, BlockMatrix2x2
from ._concatenated import ConcatenatedLinearOperator
from ._crosscov_sandwich import CrosscovSandwichLinearOperator
from ._dense_cholesky_solver import DenseCholeskySolverLinearOperator
from ._dynamic_dense_matrix import DynamicDenseMatrix
from ._keops import KeOpsLinearOperator
from ._outer_product import (
    OuterProductMatrix,
    ExtendedOuterProductMatrix,
)
from ._shape_alignment import ShapeAlignmentLinearOperator
from ._sparsity_wrapper import SparsityWrapper
